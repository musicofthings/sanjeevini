"""Tests for sanjeevini.sandbox.docker_sandbox.

Unit tests inject a fake runner so the full command-construction, checkpoint,
timeout and resume logic is exercised with no Docker daemon. Tests that drive a
real daemon are marked ``integration`` and skipped by ``-m 'not integration'``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sanjeevini.sandbox.checkpoint import CheckpointStore
from sanjeevini.sandbox.docker_sandbox import DockerSandbox, ExecResult, plan_network_args


class FakeRunner:
    """Records every docker argv and returns scripted results.

    By default every call succeeds with empty output; ``run -d`` yields a fake
    container id. Per-subcommand overrides and a timeout trigger are supported.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.timeout_on: str | None = None  # subcommand that should time out
        self.fail_on: dict[str, tuple[int, str, str]] = {}  # subcommand -> result

    def __call__(self, argv, *, timeout=None, env=None):  # noqa: ANN001
        self.calls.append(list(argv))
        self.envs.append(env)
        sub = argv[1] if len(argv) > 1 else ""
        if self.timeout_on is not None and sub == self.timeout_on:
            raise subprocess.TimeoutExpired(cmd=list(argv), timeout=timeout or 0)
        if sub in self.fail_on:
            return self.fail_on[sub]
        if sub == "run":
            return (0, "fakecontainerid123\n", "")
        return (0, "", "")

    def last(self, sub: str) -> list[str] | None:
        for argv in reversed(self.calls):
            if len(argv) > 1 and argv[1] == sub:
                return argv
        return None


def _box(**kw) -> tuple[DockerSandbox, FakeRunner]:
    runner = FakeRunner()
    box = DockerSandbox(kw.pop("image", "ubuntu:24.04"), runner=runner, binary="docker", **kw)
    return box, runner


# --------------------------------------------------------------------------
# Unit tests (no Docker)
# --------------------------------------------------------------------------


def test_start_returns_container_id_and_builds_args() -> None:
    box, runner = _box(gpus="all", memory_gb=64, cpus=8, workdir="/w")
    cid = box.start()
    assert cid == "fakecontainerid123"
    run = runner.last("run")
    assert run[:2] == ["docker", "run"]
    assert "-d" in run
    assert "--gpus" in run and "all" in run
    assert "--memory" in run and "64g" in run
    assert "--cpus" in run and "8" in run
    assert "-w" in run and "/w" in run
    assert run[-3:] == ["ubuntu:24.04", "sleep", "infinity"]


def test_plan_network_args_pure() -> None:
    assert plan_network_args("open") == []
    assert plan_network_args("none") == ["--network", "none"]
    # restricted's network is added by the gateway in start(), not here.
    assert plan_network_args("restricted") == []


def test_plan_network_args_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown network mode"):
        plan_network_args("bogus")


def test_start_hardens_by_default() -> None:
    box, runner = _box()
    box.start()
    run = runner.last("run")
    assert "--pids-limit" in run and "4096" in run
    assert run[run.index("--security-opt") + 1] == "no-new-privileges"


def test_start_hardening_can_be_disabled() -> None:
    box, runner = _box(pids_limit=None, no_new_privileges=False)
    box.start()
    run = runner.last("run")
    assert "--pids-limit" not in run
    assert "--security-opt" not in run


def test_start_network_none_is_offline() -> None:
    box, runner = _box(network="none")
    box.start()
    run = runner.last("run")
    assert run[run.index("--network") + 1] == "none"


def test_start_restricted_enforces_via_gateway() -> None:
    box, runner = _box(network="restricted", name="sbx")
    box.start()
    run = runner.last("run")
    # The sandbox joins the gateway's internal network — its only route out —
    # and is handed the proxy URL that resolves to the proxy on that network.
    assert run[run.index("--network") + 1] == "jeeva-egress-sbx-net"
    assert "HTTP_PROXY=http://jeeva-egress-sbx:3128" in run
    assert "HTTPS_PROXY=http://jeeva-egress-sbx:3128" in run
    # The gateway itself was created: an internal network and the proxy container.
    assert ["docker", "network", "create", "--internal", "jeeva-egress-sbx-net"] in runner.calls
    assert any(c[:2] == ["docker", "create"] and "jeeva-egress-sbx" in c for c in runner.calls)


def test_start_restricted_fails_closed_when_gateway_fails() -> None:
    from sanjeevini.sandbox.docker_sandbox import DockerError

    box, runner = _box(network="restricted", name="sbx")
    runner.fail_on["network"] = (1, "", "no bridge")  # network create fails
    with pytest.raises(DockerError):
        box.start()
    # Fail-closed: the sandbox container was never launched with open networking.
    assert runner.last("run") is None


def test_unknown_network_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown network mode"):
        _box(network="firewalled")


def test_start_is_idempotent() -> None:
    box, runner = _box()
    box.start()
    box.start()
    assert sum(1 for c in runner.calls if c[1] == "run") == 1


def test_extra_volumes_mounted_rw() -> None:
    box, runner = _box(extra_volumes=[("/host/data", "/data")])
    box.start()
    run = runner.last("run")
    assert "-v" in run
    assert "/host/data:/data" in run


def test_ont_raw_mount_is_readonly() -> None:
    box, runner = _box()
    box.add_ont_raw_mount("/host/pod5")
    box.start()
    run = runner.last("run")
    assert "/host/pod5:/data/raw:ro" in run


def test_add_volume_after_start_raises() -> None:
    box, _ = _box()
    box.start()
    with pytest.raises(RuntimeError):
        box.add_ont_raw_mount("/host/pod5")


def test_docker_host_sets_env() -> None:
    box, runner = _box(docker_host="ssh://user@gpu-box")
    box.start()
    assert runner.envs[-1] == {"DOCKER_HOST": "ssh://user@gpu-box"}


def test_exec_before_start_raises() -> None:
    box, _ = _box()
    with pytest.raises(RuntimeError):
        box.exec(["echo", "hi"])


def test_exec_returns_result_and_builds_argv() -> None:
    box, runner = _box(workdir="/w")
    box.start()
    result = box.exec(["echo", "hello"])
    assert isinstance(result, ExecResult)
    assert result.ok and result.returncode == 0
    ex = runner.last("exec")
    assert ex[:2] == ["docker", "exec"]
    assert "-w" in ex and "/w" in ex
    assert ex[-2:] == ["echo", "hello"]


def test_exec_timeout_raises_timeouterror() -> None:
    box, runner = _box()
    box.start()
    runner.timeout_on = "exec"
    with pytest.raises(TimeoutError):
        box.exec(["sleep", "60"], timeout=1)


def test_checkpoint_writes_one_record_per_exec(tmp_checkpoint_dir: Path) -> None:
    box, _ = _box(checkpoint_dir=tmp_checkpoint_dir)
    box.start()
    for _ in range(3):
        box.exec(["true"])
    files = sorted(tmp_checkpoint_dir.glob("turn_*.json"))
    assert len(files) == 3
    store = CheckpointStore(tmp_checkpoint_dir)
    assert [r.turn for r in store.read_all()] == [1, 2, 3]


def test_checkpoint_snapshot_only_on_success(tmp_checkpoint_dir: Path) -> None:
    box, runner = _box(checkpoint_dir=tmp_checkpoint_dir)
    box.start()
    box.exec(["true"])  # success -> snapshot recorded
    runner.fail_on["exec"] = (1, "", "boom")
    box.exec(["false"])  # failure -> no snapshot
    store = CheckpointStore(tmp_checkpoint_dir)
    recs = store.read_all()
    assert recs[0].ok and recs[0].snapshot_tag is not None
    assert not recs[1].ok and recs[1].snapshot_tag is None
    # a commit was issued for the successful turn only
    assert any(c[1] == "commit" for c in runner.calls)


def test_no_checkpoint_dir_writes_nothing(tmp_path: Path) -> None:
    box, runner = _box()
    box.start()
    box.exec(["true"])
    assert not any(c[1] == "commit" for c in runner.calls)


def test_resume_reads_previous_checkpoint(tmp_checkpoint_dir: Path) -> None:
    box1, _ = _box(checkpoint_dir=tmp_checkpoint_dir)
    box1.start()
    box1.exec(["build"])
    box1.exec(["test"])

    # A fresh sandbox pointing at the same dir sees the prior turns and its
    # own exec continues the turn counter rather than clobbering turn 1.
    box2, _ = _box(checkpoint_dir=tmp_checkpoint_dir)
    prev = box2.previous_turns
    assert [r.turn for r in prev] == [1, 2]
    assert box2.last_successful_snapshot() == prev[-1].snapshot_tag
    box2.start()
    box2.exec(["continue"])
    assert [r.turn for r in box2.previous_turns] == [1, 2, 3]


def test_snapshot_returns_tag_and_commits() -> None:
    box, runner = _box()
    box.start()
    tag = box.snapshot("sanjeevini/scratch:ready")
    assert tag == "sanjeevini/scratch:ready"
    commit = runner.last("commit")
    assert commit[-2:] == [box.name, "sanjeevini/scratch:ready"]


def test_copy_in_and_out_build_cp_args(tmp_path: Path) -> None:
    box, runner = _box()
    box.start()
    box.copy_in(tmp_path / "a.txt", "/work/a.txt")
    cp = runner.last("cp")
    assert cp[-1] == f"{box.name}:/work/a.txt"
    box.copy_out("/work/b.txt", tmp_path / "b.txt")
    cp = runner.last("cp")
    assert cp[-2] == f"{box.name}:/work/b.txt"


def test_context_manager_removes_container() -> None:
    runner = FakeRunner()
    with DockerSandbox("ubuntu:24.04", runner=runner, binary="docker") as box:
        name = box.name
        box.exec(["true"])
    rm = runner.last("rm")
    assert rm == ["docker", "rm", "-f", name]


def test_stop_graceful_stops_then_removes() -> None:
    box, runner = _box()
    box.start()
    box.stop(force=False)
    subs = [c[1] for c in runner.calls]
    assert "stop" in subs and "rm" in subs


def test_docker_error_on_failed_start() -> None:
    from sanjeevini.sandbox.docker_sandbox import DockerError

    runner = FakeRunner()
    runner.fail_on["run"] = (125, "", "no such image")
    box = DockerSandbox("bad:image", runner=runner, binary="docker")
    with pytest.raises(DockerError):
        box.start()


def _cuda_runner(image_cuda: str, host_smi: str):
    """Build a runner scripting CUDA probe responses for the GPU warn path."""

    def runner(argv, *, timeout=None, env=None):  # noqa: ANN001
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "run":
            return (0, "cid\n", "")
        if sub == "exec":
            script = argv[-1]
            if "CUDA_VERSION" in script:
                return (0, image_cuda + "\n", "")
            if "nvidia-smi" in script:
                return (0, host_smi, "")
        return (0, "", "")

    return runner


def test_gpu_cuda_mismatch_warns() -> None:
    runner = _cuda_runner("12.4", "CUDA Version: 11.8 \n")
    box = DockerSandbox("nv:img", gpus="all", runner=runner, binary="docker")
    with pytest.warns(RuntimeWarning, match="CUDA"):
        box.start()


def test_gpu_cuda_compatible_no_warning(recwarn) -> None:
    runner = _cuda_runner("11.2", "CUDA Version: 11.8 \n")
    box = DockerSandbox("nv:img", gpus="all", runner=runner, binary="docker")
    box.start()
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


def test_gpu_no_cuda_env_no_warning(recwarn) -> None:
    runner = _cuda_runner("", "CUDA Version: 11.8 \n")  # image has no CUDA_VERSION
    box = DockerSandbox("nv:img", gpus="all", runner=runner, binary="docker")
    box.start()
    assert not [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]


def test_version_tuple_parses_dotted() -> None:
    from sanjeevini.sandbox.docker_sandbox import _version_tuple

    assert _version_tuple("11.8") == (11, 8)
    assert _version_tuple("12.4.1") == (12, 4, 1)
    assert _version_tuple("cuda") == (0,)


# --------------------------------------------------------------------------
# Integration tests (real Docker) — per PRD test spec
# --------------------------------------------------------------------------

_HAS_DOCKER = shutil.which("docker") is not None

integration = pytest.mark.integration
requires_docker = pytest.mark.skipif(not _HAS_DOCKER, reason="docker binary not found")


@integration
@requires_docker
def test_exec_simple() -> None:
    with DockerSandbox("ubuntu:24.04") as box:
        result = box.exec(["echo", "hello"])
    assert result.returncode == 0
    assert "hello" in result.stdout


@integration
@requires_docker
def test_exec_timeout() -> None:
    with DockerSandbox("ubuntu:24.04") as box, pytest.raises(TimeoutError):
        box.exec(["sleep", "60"], timeout=1)


@integration
@requires_docker
def test_copy_round_trip(tmp_path: Path) -> None:
    src = tmp_path / "in.txt"
    src.write_text("payload-42")
    out = tmp_path / "out.txt"
    with DockerSandbox("ubuntu:24.04") as box:
        box.copy_in(src, "/tmp/in.txt")
        box.copy_out("/tmp/in.txt", out)
    assert out.read_text() == "payload-42"


@integration
@requires_docker
def test_snapshot() -> None:
    with DockerSandbox("ubuntu:24.04") as box:
        assert box.exec(["true"]).ok
        image = box.snapshot("sanjeevini-test/snap:latest")
    # image is inspectable == it exists locally
    proc = subprocess.run(["docker", "image", "inspect", image], capture_output=True, check=False)
    subprocess.run(["docker", "image", "rm", "-f", image], capture_output=True, check=False)
    assert proc.returncode == 0


@integration
@requires_docker
def test_checkpoint_writes(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    with DockerSandbox("ubuntu:24.04", checkpoint_dir=ckpt) as box:
        for _ in range(3):
            box.exec(["true"])
    assert len(list(ckpt.glob("turn_*.json"))) == 3


@integration
@requires_docker
def test_resume_reads_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpt"
    with DockerSandbox("ubuntu:24.04", checkpoint_dir=ckpt) as box:
        box.exec(["true"])
    box2 = DockerSandbox("ubuntu:24.04", checkpoint_dir=ckpt)
    assert len(box2.previous_turns) == 1
    assert box2.last_successful_snapshot() is not None
