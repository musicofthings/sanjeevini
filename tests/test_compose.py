"""Tests for sanjeevini.compose.pipeline (target: 80% branch coverage)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from sanjeevini.compose.pipeline import (
    ComposeCommand,
    Pipeline,
    PipelineStep,
    _parse_overrides,
)
from sanjeevini.contracts.schema import ContractSchema, GenomicFileType, IOPort

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "pipelines"

_MISMATCH_YAML = """\
name: mismatch
steps:
  - name: a
    tool: maker
    outputs:
      out_vcf: ${workdir}/a.vcf
  - name: b
    tool: caller
    inputs:
      in_bam: ${steps.a.outputs.out_vcf}
    command: "call ${inputs.in_bam}"
"""

_UNKNOWN_STEP_YAML = """\
name: ghost_ref
steps:
  - name: b
    tool: caller
    inputs:
      in_bam: ${steps.ghost.outputs.x}
"""


def _schema(*, inputs=None, outputs=None) -> ContractSchema:
    return ContractSchema(inputs=inputs or [], outputs=outputs or [])


def _mismatch_resolver(out_type: GenomicFileType, in_type: GenomicFileType):
    schemas = {
        "maker": _schema(outputs=[IOPort(name="out_vcf", type=out_type)]),
        "caller": _schema(inputs=[IOPort(name="in_bam", type=in_type)]),
    }
    return lambda tool: schemas.get(tool)


# ---- parsing --------------------------------------------------------------


def test_parse_example_pipeline_steps() -> None:
    pipe = Pipeline(_EXAMPLES / "ont_sv_calling.yaml", registry_dirs=[])
    assert pipe.name == "ont_sv_calling"
    assert [s.name for s in pipe.steps] == ["basecall", "align", "sv_call"]
    assert isinstance(pipe.steps[0], PipelineStep)
    assert pipe.steps[0].resources.get("gpus") == "all"


def test_parse_inline_yaml() -> None:
    pipe = Pipeline(_MISMATCH_YAML, schema_resolver=lambda _t: None)
    assert pipe.name == "mismatch"
    assert pipe.steps[1].inputs["in_bam"] == "${steps.a.outputs.out_vcf}"


def test_non_mapping_yaml_raises() -> None:
    with pytest.raises(ValueError, match="mapping"):
        Pipeline("- just\n- a\n- list\n", schema_resolver=lambda _t: None)


# ---- validation -----------------------------------------------------------


def test_validate_example_ont_returns_no_errors() -> None:
    assert Pipeline(_EXAMPLES / "ont_sv_calling.yaml", registry_dirs=[]).validate() == []


def test_validate_example_hifi_returns_no_errors() -> None:
    pipe = Pipeline(_EXAMPLES / "hifi_variant_calling.yaml", registry_dirs=[])
    assert [s.name for s in pipe.steps] == ["align", "snv_call", "methylation", "sv_call"]
    assert pipe.validate() == []


def test_validate_flags_bam_vcf_mismatch() -> None:
    resolver = _mismatch_resolver(GenomicFileType.VCF, GenomicFileType.BAM)
    errors = Pipeline(_MISMATCH_YAML, schema_resolver=resolver).validate()
    assert errors
    assert any("out_vcf→in_bam" in e for e in errors)


def test_validate_passes_when_types_match() -> None:
    resolver = _mismatch_resolver(GenomicFileType.BAM, GenomicFileType.BAM)
    assert Pipeline(_MISMATCH_YAML, schema_resolver=resolver).validate() == []


def test_validate_reports_unknown_step_reference() -> None:
    errors = Pipeline(_UNKNOWN_STEP_YAML, schema_resolver=lambda _t: None).validate()
    assert any("unknown step 'ghost'" in e for e in errors)


# ---- run() aborts on validation errors ------------------------------------


def test_run_aborts_without_executing_on_error() -> None:
    resolver = _mismatch_resolver(GenomicFileType.VCF, GenomicFileType.BAM)
    result = Pipeline(_MISMATCH_YAML, schema_resolver=resolver).run()
    assert result.success is False
    assert result.errors
    assert result.steps_run == []


# ---- overrides ------------------------------------------------------------


def test_parse_overrides() -> None:
    assert _parse_overrides(["a=1", "b = two", "junk"]) == {"a": "1", "b": "two"}
    assert _parse_overrides(None) == {}


# ---- ComposeCommand -------------------------------------------------------


def test_compose_command_dry_run_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline_path = _EXAMPLES / "ont_sv_calling.yaml"
    monkeypatch.chdir(tmp_path)  # no ./contracts or ./registry here → no schemas
    args = argparse.Namespace(
        pipeline=str(pipeline_path),
        registry=None,
        input=None,
        docker_host=None,
        dry_run=True,
    )
    ComposeCommand(args).run()  # must not raise / must exit 0
    assert "PASS" in capsys.readouterr().out


def test_compose_command_dry_run_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(_UNKNOWN_STEP_YAML)
    args = argparse.Namespace(
        pipeline=str(bad), registry=[], input=None, docker_host=None, dry_run=True
    )
    with pytest.raises(SystemExit) as exc:
        ComposeCommand(args).run()
    assert exc.value.code == 1
    assert "validation error" in capsys.readouterr().out


def test_compose_command_run_reports_validation_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(_UNKNOWN_STEP_YAML)
    args = argparse.Namespace(
        pipeline=str(bad), registry=[], input=["x=1"], docker_host=None, dry_run=False
    )
    with pytest.raises(SystemExit) as exc:
        ComposeCommand(args).run()
    assert exc.value.code == 1
