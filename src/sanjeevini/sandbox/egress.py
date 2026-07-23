"""Enforced egress allowlisting for the ``restricted`` sandbox.

Setting ``HTTP_PROXY`` in a container is only advice: untrusted build code can
ignore it and open a raw socket to anywhere. To *enforce* an allowlist you have
to remove the route, not ask nicely — so ``restricted`` mode uses the standard
egress-gateway pattern:

1. The sandbox joins a Docker ``--internal`` network, which has **no route to the
   internet at all**.
2. A filtering forward proxy (Squid) is attached to *both* that internal network
   and a normal bridge, so it is the sandbox's only path out.
3. Squid allowlists by destination domain — on plain HTTP requests and on HTTPS
   ``CONNECT`` targets alike — with **no TLS interception**: it splices the
   tunnel through once the host is approved, so TLS stays end-to-end and no CA is
   injected into the sandbox.

Because the sandbox has no other route, code that ignores the proxy and dials an
IP directly gets nothing. The control cannot be bypassed from inside. Startup is
**fail-closed**: if the gateway cannot come up, the caller gets a
:class:`~sanjeevini.sandbox.docker_sandbox.DockerError` rather than a container
that silently has open networking.

Residual risk: allowlisting the ``CONNECT`` host (not the inner TLS SNI) leaves a
narrow domain-fronting gap — reaching a disallowed host that is co-hosted on an
allowed one's IP. That is negligible for a package-mirror allowlist; SNI-peek
splicing (``ssl_bump peek`` at ``SslBump1``) is the documented next step if it
ever matters, at the cost of managing a bump certificate.

The whole surface is driven through an injected ``runner`` (the same one
:class:`DockerSandbox` uses), so gateway orchestration is unit-testable without a
Docker daemon.
"""

from __future__ import annotations

import os
import tempfile
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

# runner(argv, *, timeout, env) -> (returncode, stdout, stderr)
Runner = Callable[..., "tuple[int, str, str]"]

SQUID_PORT = 3128
_SQUID_CONF_PATH = "/etc/squid/squid.conf"

# The proxy image. Pinning is an operator's call (reproducibility vs. security
# updates); override with ``$JEEVA_EGRESS_PROXY_IMAGE``. Default names a Squid
# image tag rather than ``latest`` so a run is at least repeatable by default.
DEFAULT_PROXY_IMAGE = os.environ.get("JEEVA_EGRESS_PROXY_IMAGE", "ubuntu/squid:6.6-24.04_beta")

# Health-probe budget: how long to wait for Squid to accept its config.
_HEALTH_TIMEOUT_S = 30.0
_HEALTH_INTERVAL_S = 0.5


def render_squid_conf(allowlist: Sequence[str]) -> str:
    """Render a deny-by-default Squid config allowing only ``allowlist`` domains.

    Each domain is matched as ``.<domain>``, which in Squid covers the domain and
    all its subdomains. The policy gates both plain HTTP (by request host) and
    HTTPS (by ``CONNECT`` target host); HTTPS tunnels are spliced, never bumped,
    so TLS is never intercepted.

    Args:
        allowlist: The destination domains permitted to leave the sandbox.

    Returns:
        A complete ``squid.conf`` as text.

    Raises:
        ValueError: If ``allowlist`` is empty (a deny-all proxy is never what the
            caller means, and would fail every install silently).
    """
    if not allowlist:
        raise ValueError("egress allowlist is empty; refusing to build a deny-all proxy")
    domains = " ".join(f".{d.lstrip('.')}" for d in allowlist)
    return "\n".join(
        (
            f"http_port {SQUID_PORT}",
            "",
            "# Deny-by-default egress allowlist (domains + all their subdomains).",
            f"acl allowed_domains dstdomain {domains}",
            "acl SSL_ports port 443",
            "acl Safe_ports port 80",
            "acl Safe_ports port 443",
            "acl CONNECT method CONNECT",
            "",
            "# Never tunnel to non-TLS ports; only allow CONNECT to allowed hosts.",
            "http_access deny CONNECT !SSL_ports",
            "http_access allow CONNECT allowed_domains",
            "http_access allow allowed_domains",
            "http_access deny all",
            "",
            "# Do not leak the client or advertise the proxy.",
            "via off",
            "forwarded_for delete",
            "httpd_suppress_version_string on",
            "",
        )
    )


class EgressGateway:
    """Lifecycle for one sandbox's enforced-egress network + filtering proxy.

    Each gateway owns a uniquely named ``--internal`` network and a Squid
    container attached to it and to the default bridge. :meth:`start` brings both
    up (fail-closed) and returns the proxy URL plus the internal network name the
    sandbox must join; :meth:`stop` removes them.
    """

    def __init__(
        self,
        *,
        runner: Runner,
        allowlist: Sequence[str],
        docker_host: str | None = None,
        image: str | None = None,
        binary: str = "docker",
        name: str | None = None,
        bridge_network: str = "bridge",
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Configure the gateway (nothing is created until :meth:`start`).

        Args:
            runner: Injected ``docker`` subprocess callable (shared with the sandbox).
            allowlist: Destination domains the sandbox may reach.
            docker_host: Remote Docker endpoint, exported as ``DOCKER_HOST``.
            image: Squid image to run.
            binary: Path to the docker binary.
            name: Base name for the proxy container / network; a unique one is
                generated if omitted.
            bridge_network: The external (internet-facing) network the proxy also
                joins so it can reach the allowlisted hosts.
            sleep: Sleep function, injected for tests.
        """
        self._runner = runner
        self._allowlist = list(allowlist)
        self._docker_host = docker_host
        self._image = image or DEFAULT_PROXY_IMAGE
        self._binary = binary
        self._sleep = sleep
        self._bridge = bridge_network
        base = name or f"jeeva-egress-{uuid.uuid4().hex[:12]}"
        self.proxy_name = base
        self.network_name = f"{base}-net"
        self._started = False

    @property
    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self._docker_host} if self._docker_host else None

    def _docker(self, args: Sequence[str], *, timeout: float | None = None) -> tuple[int, str, str]:
        return self._runner([self._binary, *args], timeout=timeout, env=self._env)

    def _checked(self, args: Sequence[str], *, timeout: float | None = None) -> str:
        code, out, err = self._docker(args, timeout=timeout)
        if code != 0:
            # Import here to avoid a circular import at module load.
            from sanjeevini.sandbox.docker_sandbox import DockerError

            raise DockerError(f"egress gateway: docker {' '.join(args)} failed: {err or out}")
        return out.strip()

    @property
    def proxy_url(self) -> str:
        """The in-sandbox proxy URL (resolved by Docker DNS on the shared net)."""
        return f"http://{self.proxy_name}:{SQUID_PORT}"

    def start(self) -> tuple[str, str]:
        """Create the internal network and start the filtering proxy (fail-closed).

        Returns:
            ``(proxy_url, internal_network_name)`` — set ``HTTP(S)_PROXY`` to the
            first and join the sandbox to the second.

        Raises:
            DockerError: If any step fails; partially created resources are torn
                down first so a failure never leaks a network or container.
        """
        try:
            # 1. An internal network: attached containers get no default route out.
            self._checked(["network", "create", "--internal", self.network_name])
            # 2. Create (not yet start) the proxy on that network.
            self._checked(
                ["create", "--name", self.proxy_name, "--network", self.network_name, self._image]
            )
            # 3. Also attach it to the bridge, so the proxy itself can reach the
            #    internet while the sandbox (internal-only) cannot.
            self._checked(["network", "connect", self._bridge, self.proxy_name])
            # 4. Inject the allowlist config, then start.
            self._install_conf()
            self._checked(["start", self.proxy_name])
            self._await_healthy()
        except Exception:
            self.stop()
            raise
        self._started = True
        return self.proxy_url, self.network_name

    def _install_conf(self) -> None:
        """Write the rendered allowlist config into the proxy container.

        Uses ``docker cp`` (works with remote daemons too) rather than a bind
        mount, so no host path needs to be daemon-accessible.
        """
        conf = render_squid_conf(self._allowlist)
        with tempfile.TemporaryDirectory(prefix="jeeva-egress-conf-") as tmp:
            conf_path = Path(tmp) / "squid.conf"
            conf_path.write_text(conf, encoding="utf-8")
            self._checked(["cp", str(conf_path), f"{self.proxy_name}:{_SQUID_CONF_PATH}"])

    def _await_healthy(self) -> None:
        """Poll until Squid accepts its config, or raise on timeout (fail-closed)."""
        from sanjeevini.sandbox.docker_sandbox import DockerError

        deadline = time.monotonic() + _HEALTH_TIMEOUT_S
        last = ""
        while time.monotonic() < deadline:
            code, out, err = self._docker(
                ["exec", self.proxy_name, "squid", "-k", "check"], timeout=10
            )
            if code == 0:
                return
            last = err or out
            self._sleep(_HEALTH_INTERVAL_S)
        raise DockerError(
            f"egress proxy did not become healthy in {_HEALTH_TIMEOUT_S:.0f}s: {last}"
        )

    def stop(self) -> None:
        """Remove the proxy container and internal network. Best-effort, no raise."""
        self._docker(["rm", "-f", self.proxy_name])
        self._docker(["network", "rm", self.network_name])
        self._started = False
