"""Compose pipelines — typed wiring of resurrected bricks.

A pipeline YAML declares ``steps``, each naming a revived tool (by registry
slug) and mapping its input/output ports. Compose loads every step's
:class:`~sanjeevini.contracts.schema.ContractSchema` from the registry and
type-checks each ``A.output → B.input`` edge with
:meth:`ContractSchema.compatible_with` **before** any container runs. A
``--dry-run`` validates and reports; a full run executes each step inside a
:class:`~sanjeevini.sandbox.docker_sandbox.DockerSandbox`, threading files
between steps through one shared working directory bind-mounted into each.

Edges are discovered from ``${steps.<step>.outputs.<port>}`` references in a
step's ``inputs``. A step whose tool has no registry contract is left
unchecked (its schema is unknown), so a pipeline of not-yet-resurrected tools
validates cleanly — the type gate only fires where both endpoints are known.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import yaml

from sanjeevini.contracts.schema import ContractSchema
from sanjeevini.registry.catalog import default_registry_dirs, load_catalog

SchemaResolver = Callable[[str], "ContractSchema | None"]
ImageResolver = Callable[[str], "str | None"]

_STEP_REF_RE = re.compile(r"\$\{steps\.([A-Za-z0-9_]+)\.outputs\.([A-Za-z0-9_]+)\}")

# Any ``${...}`` reference in a command, path, or input expression.
_VAR_RE = re.compile(r"\$\{([^}]+)\}")

# Where the shared pipeline working directory is mounted inside every step's
# container. A step writes its outputs here; the next step reads them from the
# same mount, which is how files flow between otherwise-isolated containers.
WORKDIR_MOUNT = "/work"


class StepRun(NamedTuple):
    """The result of executing one pipeline step.

    Attributes:
        returncode: Process exit code (0 == success).
        stdout: Captured standard output.
        stderr: Captured standard error.
    """

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """Whether the step's command succeeded."""
        return self.returncode == 0


class StepExecutor(Protocol):
    """Runs one resolved step command in a tool's image. Injected for testing."""

    def run(
        self,
        *,
        image: str,
        command: str,
        host_workdir: Path,
        gpus: str | None,
        docker_host: str | None,
    ) -> StepRun:
        """Execute ``command`` in ``image`` with ``host_workdir`` bind-mounted."""
        ...


@dataclass
class PipelineStep:
    """One step of a Compose pipeline.

    Attributes:
        name: Unique step name (referenced by downstream steps).
        tool: Registry slug of the revived tool this step runs.
        description: Human-readable summary.
        inputs: Port name → source expression (``${params.x}`` or
            ``${steps.a.outputs.y}``).
        outputs: Port name → destination path expression.
        command: The shell command template executed inside the sandbox.
        resources: Hardware requests (``gpus``, ``min_ram_gb``, ``min_cores``).
    """

    name: str
    tool: str
    description: str = ""
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    command: str = ""
    resources: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """The outcome of running (or attempting to run) a pipeline.

    Attributes:
        success: Whether the pipeline validated and every step succeeded.
        errors: Validation or execution errors (empty on success).
        outputs: Resolved output-manifest paths.
        steps_run: Names of steps that executed.
    """

    success: bool
    errors: list[str] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)
    steps_run: list[str] = field(default_factory=list)


def _load_pipeline_yaml(source: Path | str) -> dict[str, Any]:
    """Load pipeline YAML from a path or inline content.

    A ``str`` is treated as a filesystem path only when it names a file that
    actually exists; otherwise it is parsed as inline YAML (the MCP server passes
    YAML text, not a path). This is more robust than a newline heuristic — a
    single-line inline document like ``{name: x, steps: []}`` still loads.

    Args:
        source: A filesystem path or inline YAML string.

    Returns:
        The parsed mapping.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        looks_like_path = "\n" not in source and len(source) < 4096
        if looks_like_path and Path(source).is_file():
            text = Path(source).read_text(encoding="utf-8")
        else:
            text = source
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("pipeline YAML must be a mapping at the top level")
    return data


def _parse_step(raw: dict[str, Any]) -> PipelineStep:
    """Build a :class:`PipelineStep` from a raw YAML step mapping."""
    return PipelineStep(
        name=str(raw.get("name", "")),
        tool=str(raw.get("tool", "")),
        description=str(raw.get("description", "")),
        inputs={k: str(v) for k, v in (raw.get("inputs") or {}).items()},
        outputs={k: str(v) for k, v in (raw.get("outputs") or {}).items()},
        command=str(raw.get("command", "")),
        resources=dict(raw.get("resources") or {}),
    )


def _registry_resolvers(
    registry_dirs: list[Path] | None,
) -> tuple[SchemaResolver, ImageResolver]:
    """Return (schema, image) resolvers mapping a tool slug to its registry entry.

    Loads the catalog once and closes over both maps, so a pipeline gets a step's
    type schema (for validation) and its resurrected image (for execution) from a
    single scan.
    """
    dirs = registry_dirs if registry_dirs is not None else default_registry_dirs()
    entries = load_catalog(dirs)
    schema_by_slug = {entry.slug: entry.schema for entry in entries}
    image_by_slug = {entry.slug: entry.image for entry in entries if entry.image}

    def resolve_schema(tool: str) -> ContractSchema | None:
        return schema_by_slug.get(tool)

    def resolve_image(tool: str) -> str | None:
        return image_by_slug.get(tool)

    return resolve_schema, resolve_image


class Pipeline:
    """A parsed, type-checkable Compose pipeline."""

    def __init__(
        self,
        yaml_path: Path | str,
        *,
        registry_dirs: list[Path] | None = None,
        schema_resolver: SchemaResolver | None = None,
        image_resolver: ImageResolver | None = None,
    ) -> None:
        """Parse the pipeline and resolve each step's contract schema and image.

        Args:
            yaml_path: Path to the pipeline YAML, or inline YAML content.
            registry_dirs: Directories to load contract schemas from (defaults to
                the standard registry locations).
            schema_resolver: Injected slug → schema resolver (overrides the
                registry lookup; used for testing).
            image_resolver: Injected slug → image resolver (overrides the registry
                lookup; used for testing execution without a real catalog).
        """
        raw = _load_pipeline_yaml(yaml_path)
        self.name: str = str(raw.get("name", ""))
        self.description: str = str(raw.get("description", ""))
        self.params: dict[str, Any] = dict(raw.get("params") or {})
        self.steps: list[PipelineStep] = [_parse_step(s) for s in (raw.get("steps") or [])]
        self.output_manifest: dict[str, str] = {
            k: str(v) for k, v in (raw.get("outputs") or {}).items()
        }
        self.sanity_check: dict[str, Any] = dict(raw.get("sanity_check") or {})

        if schema_resolver is not None or image_resolver is not None:
            resolve_schema: SchemaResolver = schema_resolver or (lambda _t: None)
            resolve_image: ImageResolver = image_resolver or (lambda _t: None)
        else:
            resolve_schema, resolve_image = _registry_resolvers(registry_dirs)
        self.schemas: dict[str, ContractSchema | None] = {
            step.name: resolve_schema(step.tool) for step in self.steps
        }
        self.images: dict[str, str | None] = {
            step.name: resolve_image(step.tool) for step in self.steps
        }

    def validate(self) -> list[str]:
        """Type-check all port mappings and return the errors (empty = OK).

        For each edge ``A.output → B.input`` discovered from B's input
        references, calls ``A.schema.compatible_with(B.schema, port_map)`` when
        both schemas are known. References to undefined steps are reported.

        Returns:
            A list of human-readable error messages (empty if the pipeline is
            valid).
        """
        errors: list[str] = []
        step_names = {step.name for step in self.steps}

        for step in self.steps:
            port_maps: dict[str, dict[str, str]] = {}
            for in_name, expr in step.inputs.items():
                for match in _STEP_REF_RE.finditer(expr):
                    src_step, src_out = match.group(1), match.group(2)
                    if src_step not in step_names:
                        errors.append(
                            f"step '{step.name}' input '{in_name}' references "
                            f"unknown step '{src_step}'"
                        )
                        continue
                    port_maps.setdefault(src_step, {})[src_out] = in_name

            downstream = self.schemas.get(step.name)
            for src_step, port_map in port_maps.items():
                upstream = self.schemas.get(src_step)
                if upstream is None or downstream is None:
                    continue
                errors.extend(upstream.compatible_with(downstream, port_map))

        return errors

    def run(
        self,
        input_overrides: dict[str, str] | None = None,
        docker_host: str | None = None,
        *,
        executor: StepExecutor | None = None,
        host_workdir: Path | None = None,
    ) -> PipelineResult:
        """Validate then execute the pipeline.

        Aborts without running any container if validation fails.

        Args:
            input_overrides: ``param → value`` overrides for this run.
            docker_host: Remote Docker endpoint, if any.
            executor: Injected step executor; defaults to a real Docker-backed one.
            host_workdir: Shared working directory for step I/O; a temp directory
                is created (and cleaned up) when omitted.

        Returns:
            The :class:`PipelineResult`.
        """
        errors = self.validate()
        if errors:
            return PipelineResult(success=False, errors=errors)

        missing_images = [step.tool for step in self.steps if not self.images.get(step.name)]
        if missing_images:
            raise RuntimeError(
                "cannot run: no resurrected image for "
                f"{', '.join(sorted(set(missing_images)))} — resurrect and register "
                "them first (each needs a published image in its contract)."
            )

        executor = executor or DockerStepExecutor()
        if host_workdir is not None:
            host_workdir.mkdir(parents=True, exist_ok=True)
            return self._execute(input_overrides or {}, docker_host, executor, host_workdir)
        with tempfile.TemporaryDirectory(prefix="jeeva-compose-") as tmp:
            return self._execute(input_overrides or {}, docker_host, executor, Path(tmp))

    def _execute(
        self,
        input_overrides: dict[str, str],
        docker_host: str | None,
        executor: StepExecutor,
        host_workdir: Path,
    ) -> PipelineResult:
        """Run each step in order, threading resolved outputs between them.

        Files flow through a single host directory bind-mounted at
        :data:`WORKDIR_MOUNT` in every step's container, so ``${workdir}`` paths
        written by one step are visible to the next. Execution stops at the first
        step that exits non-zero.
        """
        params = self._resolved_params(input_overrides)
        step_outputs: dict[str, dict[str, str]] = {}
        steps_run: list[str] = []

        for step in self.steps:
            resolved_inputs = {
                port: _resolve_expr(expr, params, WORKDIR_MOUNT, {}, {}, step_outputs)
                for port, expr in step.inputs.items()
            }
            resolved_outputs = {
                port: _resolve_expr(expr, params, WORKDIR_MOUNT, {}, {}, step_outputs)
                for port, expr in step.outputs.items()
            }
            command = _resolve_expr(
                step.command, params, WORKDIR_MOUNT, resolved_inputs, resolved_outputs, step_outputs
            )
            image = self.images[step.name]
            assert image is not None  # guarded by the missing-image check above
            run = executor.run(
                image=image,
                command=command,
                host_workdir=host_workdir,
                gpus=_as_str(step.resources.get("gpus")),
                docker_host=docker_host,
            )
            steps_run.append(step.name)
            if not run.ok:
                return PipelineResult(
                    success=False,
                    errors=[
                        f"step '{step.name}' ({step.tool}) failed with exit "
                        f"{run.returncode}: {run.stderr.strip()[:400] or '(no stderr)'}"
                    ],
                    steps_run=steps_run,
                )
            step_outputs[step.name] = resolved_outputs

        outputs = {
            name: _resolve_expr(expr, params, WORKDIR_MOUNT, {}, {}, step_outputs)
            for name, expr in self.output_manifest.items()
        }
        return PipelineResult(success=True, outputs=outputs, steps_run=steps_run)

    def _resolved_params(self, input_overrides: dict[str, str]) -> dict[str, str]:
        """Merge declared param defaults with CLI overrides into a flat string map.

        A param declared ``{type, default}`` contributes its default; an override
        wins over it. A required param with neither a default nor an override is
        left absent, so an unresolved ``${params.x}`` surfaces plainly rather than
        being silently blanked.
        """
        values: dict[str, str] = {}
        for name, decl in self.params.items():
            if isinstance(decl, dict) and "default" in decl:
                values[name] = _as_str(decl["default"]) or ""
        values.update(input_overrides)
        return values


class ComposeCommand:
    """CLI handler for ``jeeva run`` (Compose)."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Store parsed CLI arguments.

        Args:
            args: The ``run`` subparser namespace.
        """
        self.args = args

    def run(self) -> None:
        """Validate (``--dry-run``) or execute the pipeline."""
        registry_dirs = (
            [Path(d) for d in self.args.registry] if getattr(self.args, "registry", None) else None
        )
        pipeline = Pipeline(self.args.pipeline, registry_dirs=registry_dirs)

        if self.args.dry_run:
            errors = pipeline.validate()
            if not errors:
                print(f"PASS — {len(pipeline.steps)} steps, all port mappings type-check.")
                return
            print(f"{len(errors)} validation error(s):")
            for i, err in enumerate(errors, 1):
                print(f"  {i}. {err}")
            sys.exit(1)

        overrides = _parse_overrides(getattr(self.args, "input", None))
        try:
            result = pipeline.run(overrides, docker_host=getattr(self.args, "docker_host", None))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            sys.exit(1)
        if not result.success:
            print(f"{len(result.errors)} validation error(s):")
            for i, err in enumerate(result.errors, 1):
                print(f"  {i}. {err}")
            sys.exit(1)
        print(f"pipeline '{pipeline.name}' completed: {len(result.steps_run)} steps")


def _parse_overrides(pairs: list[str] | None) -> dict[str, str]:
    """Parse ``KEY=VALUE`` CLI override strings into a dict."""
    overrides: dict[str, str] = {}
    for pair in pairs or []:
        key, sep, value = pair.partition("=")
        if sep:
            overrides[key.strip()] = value.strip()
    return overrides


def _as_str(value: Any) -> str | None:
    """Stringify a scalar param/resource value; ``None`` stays ``None``."""
    return None if value is None else str(value)


def _resolve_expr(
    expr: str,
    params: dict[str, str],
    workdir: str,
    inputs: dict[str, str],
    outputs: dict[str, str],
    step_outputs: dict[str, dict[str, str]],
) -> str:
    """Substitute ``${...}`` references in ``expr`` against the run context.

    Supported references: ``${workdir}``, ``${params.NAME}``, ``${inputs.PORT}``,
    ``${outputs.PORT}``, and ``${steps.STEP.outputs.PORT}``. An unknown reference
    is left verbatim so a mistake shows up in the command rather than silently
    resolving to an empty string.

    Args:
        expr: The template string (command, path, or input expression).
        params: Resolved pipeline params.
        workdir: The in-container shared working directory mount.
        inputs: This step's already-resolved input ports (empty when resolving
            inputs/outputs themselves).
        outputs: This step's already-resolved output ports.
        step_outputs: Resolved output ports of all earlier steps, by step name.

    Returns:
        ``expr`` with every recognised reference replaced.
    """

    def repl(match: re.Match[str]) -> str:
        ref = match.group(1).strip()
        if ref == "workdir":
            return workdir
        if ref.startswith("params."):
            return params.get(ref[len("params.") :], match.group(0))
        if ref.startswith("inputs."):
            return inputs.get(ref[len("inputs.") :], match.group(0))
        if ref.startswith("outputs."):
            return outputs.get(ref[len("outputs.") :], match.group(0))
        parts = ref.split(".")
        if len(parts) == 4 and parts[0] == "steps" and parts[2] == "outputs":
            return step_outputs.get(parts[1], {}).get(parts[3], match.group(0))
        return match.group(0)

    return _VAR_RE.sub(repl, expr)


class DockerStepExecutor:
    """Default :class:`StepExecutor`: runs a step in a fresh Docker sandbox.

    The shared host working directory is bind-mounted at :data:`WORKDIR_MOUNT`
    (read-write) so step outputs persist for downstream steps, and the container
    is torn down after the step regardless of outcome.
    """

    def run(
        self,
        *,
        image: str,
        command: str,
        host_workdir: Path,
        gpus: str | None,
        docker_host: str | None,
    ) -> StepRun:  # pragma: no cover - requires a Docker daemon and real images
        """Execute ``command`` in ``image`` and return its :class:`StepRun`."""
        from sanjeevini.sandbox.docker_sandbox import DockerSandbox

        with DockerSandbox(
            image,
            workdir=WORKDIR_MOUNT,
            docker_host=docker_host,
            gpus=gpus,
            extra_volumes=[(str(host_workdir), WORKDIR_MOUNT)],
        ) as box:
            result = box.exec(["bash", "-lc", command], timeout=86400)
        return StepRun(result.returncode, result.stdout, result.stderr)
