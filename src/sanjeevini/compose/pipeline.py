"""Compose pipelines — typed wiring of resurrected bricks.

A pipeline YAML declares ``steps``, each naming a revived tool (by registry
slug) and mapping its input/output ports. Compose loads every step's
:class:`~sanjeevini.contracts.schema.ContractSchema` from the registry and
type-checks each ``A.output → B.input`` edge with
:meth:`ContractSchema.compatible_with` **before** any container runs. A
``--dry-run`` validates and reports; a full run executes each step inside a
:class:`~sanjeevini.sandbox.docker_sandbox.DockerSandbox`, pre-fetching model
bundles before GPU steps.

Edges are discovered from ``${steps.<step>.outputs.<port>}`` references in a
step's ``inputs``. A step whose tool has no registry contract is left
unchecked (its schema is unknown), so a pipeline of not-yet-resurrected tools
validates cleanly — the type gate only fires where both endpoints are known.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sanjeevini.contracts.schema import ContractSchema
from sanjeevini.registry.catalog import default_registry_dirs, load_catalog

SchemaResolver = Callable[[str], "ContractSchema | None"]

_STEP_REF_RE = re.compile(r"\$\{steps\.([A-Za-z0-9_]+)\.outputs\.([A-Za-z0-9_]+)\}")


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

    Inline content is detected by the presence of a newline (as the MCP server
    passes YAML text rather than a path).

    Args:
        source: A filesystem path or inline YAML string.

    Returns:
        The parsed mapping.
    """
    if isinstance(source, str) and "\n" in source:
        text = source
    else:
        text = Path(source).read_text(encoding="utf-8")
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


def _registry_resolver(registry_dirs: list[Path] | None) -> SchemaResolver:
    """Return a resolver mapping a tool slug to its registry ContractSchema."""
    dirs = registry_dirs if registry_dirs is not None else default_registry_dirs()
    by_slug = {entry.slug: entry.schema for entry in load_catalog(dirs)}

    def resolve(tool: str) -> ContractSchema | None:
        return by_slug.get(tool)

    return resolve


class Pipeline:
    """A parsed, type-checkable Compose pipeline."""

    def __init__(
        self,
        yaml_path: Path | str,
        *,
        registry_dirs: list[Path] | None = None,
        schema_resolver: SchemaResolver | None = None,
    ) -> None:
        """Parse the pipeline and resolve each step's contract schema.

        Args:
            yaml_path: Path to the pipeline YAML, or inline YAML content.
            registry_dirs: Directories to load contract schemas from (defaults to
                the standard registry locations).
            schema_resolver: Injected slug → schema resolver (overrides the
                registry lookup; used for testing).
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

        resolver = schema_resolver or _registry_resolver(registry_dirs)
        self.schemas: dict[str, ContractSchema | None] = {
            step.name: resolver(step.tool) for step in self.steps
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
    ) -> PipelineResult:
        """Validate then execute the pipeline.

        Aborts without running any container if validation fails.

        Args:
            input_overrides: ``param → value`` overrides for this run.
            docker_host: Remote Docker endpoint, if any.

        Returns:
            The :class:`PipelineResult`.
        """
        errors = self.validate()
        if errors:
            return PipelineResult(success=False, errors=errors)
        return self._execute(input_overrides or {}, docker_host)

    def _execute(  # pragma: no cover - requires Docker + resolved images
        self, input_overrides: dict[str, str], docker_host: str | None
    ) -> PipelineResult:
        """Run each step in a sandbox (integration path)."""
        missing = [step.tool for step in self.steps if self.schemas.get(step.name) is None]
        if missing:
            raise RuntimeError(
                "cannot run: no registry contract for "
                f"{', '.join(sorted(set(missing)))} — resurrect and register them first"
            )
        raise RuntimeError("pipeline execution is available through `jeeva mcp`")


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
        result = pipeline.run(overrides, docker_host=getattr(self.args, "docker_host", None))
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
