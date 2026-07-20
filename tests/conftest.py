"""Shared pytest fixtures for the Sanjeevini test suite.

Fixtures for later phases (``mock_httpx``, ``sample_registry``) import their
dependencies lazily so the whole suite still collects while those modules are
unbuilt — a test only fails if it actually uses a not-yet-implemented fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Return an empty directory for a :class:`CheckpointStore` under ``tmp_path``."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def sample_contract_schema():
    """Return a :class:`ContractSchema` with one BAM input and one VCF output."""
    from sanjeevini.contracts.schema import (
        ContractSchema,
        GenomicFileType,
        IOPort,
    )

    return ContractSchema(
        inputs=[IOPort(name="bam_in", type=GenomicFileType.BAM, description="aligned reads")],
        outputs=[IOPort(name="vcf_out", type=GenomicFileType.VCF, description="variant calls")],
    )


@pytest.fixture
def mock_httpx(monkeypatch):
    """Patch ``httpx.Client.get`` to return canned responses.

    Returns a registry dict mapping URL -> (status_code, json_payload); tests
    populate it before triggering the code under test. Used by the pinner tests
    in later phases.
    """
    import httpx

    responses: dict[str, tuple[int, object]] = {}

    class _FakeResponse:
        def __init__(self, status_code: int, payload: object) -> None:
            self.status_code = status_code
            self._payload = payload

        def json(self) -> object:
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {self.status_code}", request=None, response=None)

    def _fake_get(self, url, *args, **kwargs):  # noqa: ANN001 - matches httpx signature
        status, payload = responses.get(str(url), (404, {}))
        return _FakeResponse(status, payload)

    monkeypatch.setattr(httpx.Client, "get", _fake_get, raising=True)
    return responses


@pytest.fixture
def sample_registry(tmp_path: Path) -> Path:
    """Create a minimal registry directory with two contract entries.

    Returns:
        The registry root directory holding two ``contract.yaml`` files under
        per-tool subdirectories.
    """
    import yaml

    from sanjeevini.contracts.schema import (
        ContractSchema,
        GenomicFileType,
        IOPort,
        SequencingPlatform,
    )

    root = tmp_path / "registry"

    entries = {
        "sniffles2": ContractSchema(
            inputs=[IOPort(name="bam_in", type=GenomicFileType.BAM)],
            outputs=[IOPort(name="vcf_out", type=GenomicFileType.VCF)],
            platform=SequencingPlatform.ONT,
            workflow_type="python",
        ),
        "minimap2": ContractSchema(
            inputs=[IOPort(name="fastq_in", type=GenomicFileType.FASTQ)],
            outputs=[IOPort(name="bam_out", type=GenomicFileType.BAM)],
            platform=SequencingPlatform.ANY,
            workflow_type="binary",
        ),
    }

    for slug, schema in entries.items():
        d = root / slug
        d.mkdir(parents=True)
        payload = {
            "slug": slug,
            "name": slug,
            "repo_url": f"https://github.com/example/{slug}",
            "schema": json.loads(schema.model_dump_json()),
        }
        (d / "contract.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")

    return root
