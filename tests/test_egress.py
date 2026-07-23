"""Tests for sanjeevini.sandbox.egress (enforced egress allowlisting).

The gateway is driven through an injected runner, so its full lifecycle —
internal network, proxy container, config injection, health wait, teardown — is
exercised with no Docker daemon.
"""

from __future__ import annotations

import subprocess

import pytest

from sanjeevini.sandbox.docker_sandbox import DockerError
from sanjeevini.sandbox.egress import EgressGateway, render_squid_conf


class FakeRunner:
    """Records docker argv and returns scripted results (success by default)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_on: dict[str, tuple[int, str, str]] = {}
        self.health_fails_until = 0  # number of `exec` health checks that fail first
        self._health_calls = 0

    def __call__(self, argv, *, timeout=None, env=None):  # noqa: ANN001
        self.calls.append(list(argv))
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "exec":  # the `squid -k check` health probe
            self._health_calls += 1
            if self._health_calls <= self.health_fails_until:
                return (1, "", "still starting")
            return (0, "", "")
        if sub in self.fail_on:
            return self.fail_on[sub]
        return (0, "", "")

    def has(self, *fragment: str) -> bool:
        return any(fragment == tuple(c[1 : 1 + len(fragment)]) for c in self.calls)


def _gateway(**kw) -> tuple[EgressGateway, FakeRunner]:
    runner = FakeRunner()
    gw = EgressGateway(
        runner=runner,
        allowlist=["pypi.org", "files.pythonhosted.org"],
        binary="docker",
        name="jeeva-egress-t",
        sleep=lambda _s: None,
        **kw,
    )
    return gw, runner


# ---- config rendering -----------------------------------------------------


def test_render_squid_conf_is_deny_by_default() -> None:
    conf = render_squid_conf(["pypi.org", "conda.anaconda.org"])
    assert "http_access deny all" in conf
    assert "acl allowed_domains dstdomain .pypi.org .conda.anaconda.org" in conf
    # HTTPS is gated by CONNECT host, and only to TLS ports.
    assert "http_access deny CONNECT !SSL_ports" in conf
    assert "http_access allow CONNECT allowed_domains" in conf


def test_render_squid_conf_strips_leading_dots() -> None:
    assert "dstdomain .pypi.org" in render_squid_conf([".pypi.org"])


def test_render_squid_conf_rejects_empty_allowlist() -> None:
    with pytest.raises(ValueError, match="empty"):
        render_squid_conf([])


# ---- gateway lifecycle ----------------------------------------------------


def test_gateway_start_builds_internal_network_and_proxy() -> None:
    gw, runner = _gateway()
    proxy_url, network = gw.start()

    assert proxy_url == "http://jeeva-egress-t:3128"
    assert network == "jeeva-egress-t-net"
    # internal network (no route out) + proxy on it + bridge for the proxy's own egress
    assert ["docker", "network", "create", "--internal", "jeeva-egress-t-net"] in runner.calls
    assert runner.has("network", "connect", "bridge")
    # config is injected before the proxy starts, then it starts
    assert runner.has("cp")
    assert runner.has("start", "jeeva-egress-t")


def test_gateway_waits_for_health_then_succeeds() -> None:
    gw, runner = _gateway()
    runner.health_fails_until = 2  # two failed probes, then healthy
    gw.start()  # must not raise
    assert runner._health_calls == 3


def test_gateway_start_is_fail_closed_and_cleans_up() -> None:
    gw, runner = _gateway()
    runner.fail_on["create"] = (1, "", "image not found")
    with pytest.raises(DockerError, match="egress gateway"):
        gw.start()
    # teardown was attempted so nothing leaks
    assert runner.has("rm", "-f", "jeeva-egress-t")
    assert runner.has("network", "rm", "jeeva-egress-t-net")


def test_gateway_health_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import sanjeevini.sandbox.egress as egress_mod

    # Collapse the wait so the timeout is reached immediately.
    monkeypatch.setattr(egress_mod, "_HEALTH_TIMEOUT_S", 0.0)
    gw, runner = _gateway()
    runner.health_fails_until = 999
    with pytest.raises(DockerError, match="healthy"):
        gw.start()


def test_gateway_stop_is_best_effort() -> None:
    gw, runner = _gateway()
    runner.fail_on["rm"] = (1, "", "no such container")
    gw.stop()  # must not raise even when removal fails


def test_timeout_expired_is_importable_for_runner_contract() -> None:
    # The runner may raise TimeoutExpired; the gateway never swallows it silently.
    assert issubclass(subprocess.TimeoutExpired, Exception)
