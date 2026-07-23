"""The disposable Docker sandbox every resurrection run executes inside.

A :class:`DockerSandbox` is a long-lived container you repeatedly ``exec`` into,
copy files to and from, and ``snapshot`` after an expensive step so a later
failure never forces a re-pay of a 40-minute build. It is the foundational
"organ" of Sanjeevini: the repair loop, scouts, and pinners all run their real
work inside one.

Ported from the Lazarus sandbox, with three additions for Sanjeevini:

* **Long-read hardware profiles** — GPU passthrough via ``--gpus`` and
  high-RAM/CPU limits for HiFi assembly and basecalling.
* **Checkpoint hooks** — when ``checkpoint_dir`` is set, every :meth:`exec`
  writes a :class:`~sanjeevini.sandbox.checkpoint.TurnRecord` so the repair loop
  can persist and resume state turn by turn.
* **ONT raw-data mounts** — a helper to mount a POD5/FAST5/SLOW5/BLOW5
  directory read-only at ``/data/raw`` inside the container.

Docker is driven through its CLI via :mod:`subprocess` (no Docker SDK
dependency). Local vs. remote is Docker's own concern: pass ``docker_host`` and
the ``DOCKER_HOST`` environment variable is set for every invocation to that
instance. The subprocess call is injected (``runner=``) so the whole surface is
unit-testable without a Docker daemon.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from sanjeevini.sandbox.checkpoint import CheckpointStore, TurnRecord

if TYPE_CHECKING:
    from sanjeevini.sandbox.egress import EgressGateway

# runner(argv, *, timeout, env) -> (returncode, stdout, stderr)
# May raise subprocess.TimeoutExpired to signal a timeout.
Runner = Callable[..., "tuple[int, str, str]"]

_DEFAULT_WORKDIR = "/workspace"
_ONT_RAW_MOUNT = "/data/raw"

# Network egress policy modes for the sandbox.
#   "open"       — default Docker networking, no restriction (fastest, least safe).
#   "restricted" — enforced allowlist: the container joins an ``--internal``
#                  network whose only route out is a filtering Squid proxy
#                  (see :mod:`sanjeevini.sandbox.egress`). Fail-closed.
#   "none"       — no network at all (``docker run --network none``).
NETWORK_MODES = ("open", "restricted", "none")

# Standard proxy env vars set inside a restricted container, and the loopback
# addresses that must bypass the proxy.
_PROXY_ENV_VARS = ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")
_NO_PROXY_VALUE = "localhost,127.0.0.1,::1"

# Hosts a resurrection legitimately needs to reach: language package indexes and
# OS/scientific mirrors. This is the allowlist the restricted-mode proxy enforces.
EGRESS_ALLOWLIST: tuple[str, ...] = (
    "pypi.org",
    "files.pythonhosted.org",
    "deb.debian.org",
    "security.debian.org",
    "archive.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
    "ports.ubuntu.com",
    "repo.anaconda.com",
    "conda.anaconda.org",
    "cloud.r-project.org",
    "bioconductor.org",
    "github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
    "objects.githubusercontent.com",
)


def plan_network_args(network: str) -> list[str]:
    """Return the static ``docker run`` network args for ``open``/``none`` modes.

    Pure and unit-testable. ``restricted`` adds no static args here — its network
    is a per-run internal network created by
    :class:`~sanjeevini.sandbox.egress.EgressGateway` and attached in
    :meth:`DockerSandbox.start` — so this returns ``[]`` for it.

    Args:
        network: One of :data:`NETWORK_MODES`.

    Returns:
        The docker run flags to add for the mode (possibly empty).

    Raises:
        ValueError: If ``network`` is not a known mode.
    """
    if network not in NETWORK_MODES:
        raise ValueError(f"unknown network mode {network!r}; expected one of {NETWORK_MODES}")
    return ["--network", "none"] if network == "none" else []


def _proxy_env_args(proxy_url: str) -> list[str]:
    """Return ``-e`` flags exporting the standard proxy vars into a container."""
    args: list[str] = []
    for var in _PROXY_ENV_VARS:
        args += ["-e", f"{var}={proxy_url}"]
    for var in ("NO_PROXY", "no_proxy"):
        args += ["-e", f"{var}={_NO_PROXY_VALUE}"]
    return args


class DockerError(RuntimeError):
    """Docker is missing, unreachable, or a lifecycle command failed."""


class ExecResult(NamedTuple):
    """Outcome of one :meth:`DockerSandbox.exec` call.

    Attributes:
        returncode: Process exit code; 0 means success.
        stdout: Captured standard output.
        stderr: Captured standard error.
        duration_s: Wall-clock duration of the command, in seconds.
    """

    returncode: int
    stdout: str
    stderr: str
    duration_s: float

    @property
    def ok(self) -> bool:
        """Whether the command succeeded (exit code 0)."""
        return self.returncode == 0


def find_docker() -> str:
    """Locate the ``docker`` binary, falling back to OrbStack's shim.

    Returns:
        An absolute path to a docker executable if one is found, otherwise the
        bare string ``"docker"`` to be resolved on ``PATH`` at call time.
    """
    exe = shutil.which("docker")
    if exe:
        return exe
    cand = os.path.expanduser("~/.orbstack/bin/docker")
    return cand if os.path.exists(cand) else "docker"


def _subprocess_runner(
    argv: Sequence[str],
    *,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Default runner: shell out to the real ``docker`` binary.

    Args:
        argv: Full argument vector, including the docker binary.
        timeout: Seconds before the call is aborted, or ``None`` for no limit.
        env: Extra environment variables merged over the current environment.

    Returns:
        A ``(returncode, stdout, stderr)`` triple.

    Raises:
        DockerError: If the docker binary cannot be found.
        subprocess.TimeoutExpired: If the command exceeds ``timeout``.
    """
    merged = {**os.environ, **(env or {})}
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            timeout=timeout,
            env=merged,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DockerError(
            f"could not find executable {argv[0]!r}; is Docker/OrbStack installed?"
        ) from exc
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", "replace"),
        proc.stderr.decode("utf-8", "replace"),
    )


def _new_name(prefix: str = "sanjeevini") -> str:
    """Return a unique container name of the form ``{prefix}-{hex}``."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class DockerSandbox:
    """A disposable Docker container: exec into it, move files, snapshot, tear down.

    Use as a context manager so the container is always removed::

        with DockerSandbox("ubuntu:24.04") as box:
            result = box.exec(["echo", "hello"])
            box.snapshot("sanjeevini/scratch:ready")

    When ``checkpoint_dir`` is given, every :meth:`exec` writes a turn record and
    successful turns are snapshotted, so a later :class:`DockerSandbox` pointed at
    the same directory can resume from the last known-good state.
    """

    def __init__(
        self,
        image: str,
        workdir: str = _DEFAULT_WORKDIR,
        docker_host: str | None = None,
        gpus: str | None = None,
        extra_volumes: Sequence[tuple[str, str]] = (),
        memory_gb: float | None = None,
        cpus: int | None = None,
        checkpoint_dir: Path | None = None,
        *,
        name: str | None = None,
        platform: str = "linux/amd64",
        binary: str | None = None,
        runner: Runner = _subprocess_runner,
        network: str = "open",
        egress_allowlist: Sequence[str] = EGRESS_ALLOWLIST,
        egress_image: str | None = None,
        pids_limit: int | None = 4096,
        no_new_privileges: bool = True,
    ) -> None:
        """Configure a sandbox (no container is created until :meth:`start`).

        Args:
            image: Docker image to run.
            workdir: Working directory inside the container.
            docker_host: Remote Docker endpoint (e.g. ``ssh://user@gpu-box``);
                exported as ``DOCKER_HOST`` for every command. ``None`` uses the
                local daemon.
            gpus: GPU spec forwarded to ``docker run --gpus`` (e.g. ``"all"``).
            extra_volumes: ``(host, container)`` bind mounts, added read-write.
            memory_gb: Hard memory limit in gigabytes, or ``None`` for no limit.
            cpus: CPU count limit, or ``None`` for no limit.
            checkpoint_dir: Directory for turn-level checkpoints; enables
                per-exec recording and resume. ``None`` disables checkpointing.
            name: Container name; a unique one is generated if omitted.
            platform: Docker platform string (default ``linux/amd64``).
            binary: Path to the docker binary; auto-detected if omitted.
            runner: Injected subprocess callable, for testing without Docker.
            network: Egress policy — one of :data:`NETWORK_MODES`. ``"restricted"``
                enforces the allowlist with a per-run filtering proxy on an
                internal network (fail-closed).
            egress_allowlist: Domains the restricted-mode proxy permits. Defaults
                to :data:`EGRESS_ALLOWLIST` (package indexes and OS mirrors).
            egress_image: Squid image for restricted mode; ``None`` uses the
                egress module's default (overridable via ``$JEEVA_EGRESS_PROXY_IMAGE``).
            pids_limit: Hard cap on process count (fork-bomb guard). ``None``
                disables the cap. Defaults to 4096 — ample for parallel builds.
            no_new_privileges: Add ``--security-opt no-new-privileges`` so a
                setuid binary in the target cannot escalate. Defaults to ``True``.
        """
        if network not in NETWORK_MODES:
            raise ValueError(f"unknown network mode {network!r}; expected one of {NETWORK_MODES}")
        self.image = image
        self.workdir = workdir
        self.docker_host = docker_host
        self.gpus = gpus
        self.memory_gb = memory_gb
        self.cpus = cpus
        self.platform = platform
        self.name = name or _new_name()
        self.binary = binary or find_docker()
        self._runner = runner
        self.network = network
        self.egress_allowlist = list(egress_allowlist)
        self.egress_image = egress_image
        self.pids_limit = pids_limit
        self.no_new_privileges = no_new_privileges
        self._gateway: EgressGateway | None = None
        self.container_id: str | None = None
        self.started = False

        # host -> container mounts, each with an optional read-only flag.
        self._volumes: list[tuple[str, str, bool]] = [
            (host, container, False) for host, container in extra_volumes
        ]

        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self._store: CheckpointStore | None = (
            CheckpointStore(self.checkpoint_dir) if self.checkpoint_dir is not None else None
        )
        # Continue the turn counter from any existing checkpoint (resume).
        self._turn = self._store.next_turn() - 1 if self._store is not None else 0

    # ---- transport ---------------------------------------------------------

    @property
    def _env(self) -> dict[str, str] | None:
        """Extra env for subprocess calls (sets ``DOCKER_HOST`` when remote)."""
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    def _docker(self, args: Sequence[str], *, timeout: float | None = None) -> tuple[int, str, str]:
        """Run a raw ``docker`` subcommand and return ``(code, out, err)``."""
        argv = [self.binary, *args]
        return self._runner(argv, timeout=timeout, env=self._env)

    def _docker_checked(self, args: Sequence[str], *, timeout: float | None = None) -> str:
        """Run a docker subcommand, raising :class:`DockerError` on failure.

        Returns:
            The stripped stdout of the command.
        """
        code, out, err = self._docker(args, timeout=timeout)
        if code != 0:
            raise DockerError(f"docker {' '.join(args)} failed (exit {code}):\n{err or out}")
        return out.strip()

    # ---- ONT raw-data mount ------------------------------------------------

    def add_ont_raw_mount(self, host_dir: Path | str, container_dir: str = _ONT_RAW_MOUNT) -> None:
        """Mount an ONT raw-signal directory read-only inside the container.

        POD5/FAST5/SLOW5/BLOW5 data is large and must never be mutated by a run,
        so it is bind-mounted read-only (default target ``/data/raw``).

        Args:
            host_dir: Host directory holding the raw signal files.
            container_dir: Mount point inside the container.

        Raises:
            RuntimeError: If called after the container has started.
        """
        if self.started:
            raise RuntimeError("cannot add a volume after the sandbox has started")
        self._volumes.append((str(host_dir), container_dir, True))

    # ---- lifecycle ---------------------------------------------------------

    def start(self) -> str:
        """Create and start the detached container.

        Idempotent: calling again on a started sandbox returns the existing id.

        Returns:
            The Docker container id.

        Raises:
            DockerError: If the container fails to start.
        """
        if self.started and self.container_id is not None:
            return self.container_id

        args: list[str] = [
            "run",
            "-d",
            "--platform",
            self.platform,
            "--name",
            self.name,
            "-w",
            self.workdir,
        ]
        if self.gpus:
            args += ["--gpus", self.gpus]
        if self.memory_gb is not None:
            args += ["--memory", f"{self.memory_gb}g"]
        if self.cpus is not None:
            args += ["--cpus", str(self.cpus)]
        if self.pids_limit is not None:
            args += ["--pids-limit", str(self.pids_limit)]
        if self.no_new_privileges:
            args += ["--security-opt", "no-new-privileges"]
        if self.network == "restricted":
            # Enforced allowlist: stand up the egress gateway (fail-closed) and
            # join its internal network, whose only route out is the proxy.
            args += self._start_egress_gateway()
        else:
            args += plan_network_args(self.network)
        for host, container, read_only in self._volumes:
            spec = f"{host}:{container}:ro" if read_only else f"{host}:{container}"
            args += ["-v", spec]
        args += [self.image, "sleep", "infinity"]

        self.container_id = self._docker_checked(args)
        self.started = True

        if self.gpus:
            self._warn_on_cuda_mismatch()
        return self.container_id

    def _start_egress_gateway(self) -> list[str]:
        """Bring up the restricted-mode egress gateway and return its docker args.

        Returns the ``--network`` flag joining the sandbox to the gateway's
        internal network plus the proxy env vars. Raises (fail-closed) if the
        gateway cannot start, so a restricted run never silently gets open
        networking.
        """
        from sanjeevini.sandbox.egress import EgressGateway

        gateway = EgressGateway(
            runner=self._runner,
            allowlist=self.egress_allowlist,
            docker_host=self.docker_host,
            image=self.egress_image,
            binary=self.binary,
            name=f"jeeva-egress-{self.name}",
        )
        proxy_url, network_name = gateway.start()
        self._gateway = gateway
        return ["--network", network_name, *_proxy_env_args(proxy_url)]

    def _warn_on_cuda_mismatch(self) -> None:
        """Warn (never fail) if the image's CUDA version may exceed the host driver.

        Reads the image's ``CUDA_VERSION`` env var and compares it to the host's
        maximum supported CUDA reported by ``nvidia-smi``. Any probe failure is
        silently ignored — this is a best-effort courtesy, not a gate.
        """
        try:
            code, out, _ = self._docker(["exec", self.name, "bash", "-lc", "echo $CUDA_VERSION"])
            image_cuda = out.strip() if code == 0 else ""
            if not image_cuda:
                return
            code, out, _ = self._docker(
                ["exec", self.name, "bash", "-lc", "nvidia-smi 2>/dev/null || true"]
            )
            match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", out)
            if not match:
                return
            host_cuda = match.group(1)
            if _version_tuple(image_cuda) > _version_tuple(host_cuda):
                warnings.warn(
                    f"image CUDA {image_cuda} exceeds host driver CUDA {host_cuda}; "
                    "GPU code may fail to run. Continuing anyway.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (DockerError, OSError):
            return

    def stop(self, force: bool = False) -> None:
        """Stop and remove the container, guaranteeing no container is left running.

        Args:
            force: If ``True``, ``docker rm -f`` (kill + remove) immediately.
                Otherwise stop gracefully then remove.
        """
        try:
            if self.started:
                if force:
                    self._docker(["rm", "-f", self.name])
                else:
                    self._docker(["stop", self.name])
                    self._docker(["rm", self.name])
                self.started = False
        finally:
            # Always tear the egress gateway down — including when the container
            # itself never started (a failed run must not leak the proxy/network).
            if self._gateway is not None:
                self._gateway.stop()
                self._gateway = None

    # ---- execution ---------------------------------------------------------

    def exec(self, cmd: list[str], timeout: int = 300) -> ExecResult:
        """Run ``cmd`` inside the container and return its result.

        When ``checkpoint_dir`` is set, a turn record is written after the call;
        successful calls are additionally snapshotted and the snapshot tag is
        recorded so the repair loop can resume from it.

        Args:
            cmd: Command to run, as an argv list (never a shell string).
            timeout: Seconds before the command is aborted.

        Returns:
            The :class:`ExecResult`.

        Raises:
            RuntimeError: If the sandbox has not been started.
            TimeoutError: If the command exceeds ``timeout``.
        """
        if not self.started:
            raise RuntimeError("sandbox must be started before exec()")

        args = ["exec", "-w", self.workdir, self.name, *cmd]
        start = time.monotonic()
        try:
            code, out, err = self._docker(args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"command timed out after {timeout}s: {' '.join(cmd)}") from exc
        result = ExecResult(
            returncode=code,
            stdout=out,
            stderr=err,
            duration_s=time.monotonic() - start,
        )

        if self._store is not None:
            self._checkpoint(cmd, result)
        return result

    def _checkpoint(self, cmd: list[str], result: ExecResult) -> None:
        """Snapshot on success and persist a turn record for ``result``."""
        assert self._store is not None
        self._turn += 1
        snapshot_tag: str | None = None
        if result.ok:
            snapshot_tag = f"sanjeevini-checkpoint/{self.name}:turn-{self._turn:04d}"
            try:
                self.snapshot(snapshot_tag)
            except DockerError:
                snapshot_tag = None
        record = TurnRecord(
            turn=self._turn,
            cmd=list(cmd),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_s=result.duration_s,
            snapshot_tag=snapshot_tag,
            timestamp=_utc_now_iso(),
        )
        self._store.write(record)

    # ---- file transfer -----------------------------------------------------

    def copy_in(self, src: Path, dst: str) -> None:
        """Copy a host file or directory into the container.

        Args:
            src: Host path to copy from.
            dst: Destination path inside the container.

        Raises:
            DockerError: If the copy fails.
        """
        self._docker_checked(["cp", str(src), f"{self.name}:{dst}"])

    def copy_out(self, src: str, dst: Path) -> None:
        """Copy a file or directory out of the container onto the host.

        Args:
            src: Path inside the container to copy from.
            dst: Host destination path.

        Raises:
            DockerError: If the copy fails.
        """
        self._docker_checked(["cp", f"{self.name}:{src}", str(dst)])

    # ---- snapshotting ------------------------------------------------------

    def snapshot(self, tag: str) -> str:
        """Commit the live container to an image ``tag``.

        This banks an expensive successful step (a completed build, a resolved
        binary chain) so a later failure restarts from the checkpoint instead of
        from scratch. Only call on successful exec results.

        Args:
            tag: Image name (with optional ``:tag``) to commit to.

        Returns:
            The ``tag`` that was written.

        Raises:
            DockerError: If ``docker commit`` fails.
        """
        self._docker_checked(["commit", self.name, tag])
        return tag

    # ---- resume ------------------------------------------------------------

    @property
    def previous_turns(self) -> list[TurnRecord]:
        """Turn records from an existing checkpoint, ordered by turn.

        Returns:
            The recorded turns, or an empty list if checkpointing is disabled or
            the directory holds no records.
        """
        return self._store.read_all() if self._store is not None else []

    def last_successful_snapshot(self) -> str | None:
        """Return the resume image tag from the checkpoint, if any.

        Returns:
            The snapshot tag of the most recent successful checkpointed turn, or
            ``None`` if checkpointing is disabled or no such turn exists.
        """
        return self._store.last_successful_snapshot() if self._store is not None else None

    # ---- context manager ---------------------------------------------------

    def __enter__(self) -> DockerSandbox:
        """Start the container and return ``self``."""
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        """Tear the container down, leaving nothing running."""
        self.stop(force=True)


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse a dotted version like ``"11.8"`` into a comparable int tuple."""
    parts: list[int] = []
    for chunk in version.split("."):
        match = re.match(r"\d+", chunk)
        parts.append(int(match.group()) if match else 0)
    return tuple(parts)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
