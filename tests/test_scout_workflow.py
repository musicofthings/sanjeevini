"""Tests for sanjeevini.scouts.workflow_scout (target: 85% branch coverage)."""

from __future__ import annotations

from pathlib import Path

from sanjeevini.scouts.workflow_scout import (
    analyse_nextflow,
    analyse_snakemake,
    analyse_wdl,
    build_resurrection_plan,
    detect_workflow_language,
)


def test_detect_nextflow_from_main_nf(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("workflow { }")
    assert detect_workflow_language(tmp_path) == "nextflow"


def test_detect_snakemake(tmp_path: Path) -> None:
    (tmp_path / "Snakefile").write_text("rule all:\n    input: []")
    assert detect_workflow_language(tmp_path) == "snakemake"


def test_detect_wdl_by_glob(tmp_path: Path) -> None:
    (tmp_path / "pipeline.wdl").write_text('task t { runtime { docker: "x" } }')
    assert detect_workflow_language(tmp_path) == "wdl"


def test_detect_unknown(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("nothing here")
    assert detect_workflow_language(tmp_path) == "unknown"


def test_analyse_nextflow_detects_dsl1_with_warning(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=1\nprocess foo { }")
    ana = analyse_nextflow(tmp_path)
    assert ana.dsl_version == 1
    assert any("DSL1" in issue for issue in ana.issues)


def test_analyse_nextflow_detects_dsl2_and_version(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=2\nworkflow { }")
    (tmp_path / "nextflow.config").write_text(
        "manifest {\n  nextflowVersion = '23.04.0'\n}\n"
        'process { withName: FOO { container "biocontainers/foo:1.0" } }\n'
    )
    ana = analyse_nextflow(tmp_path)
    assert ana.dsl_version == 2
    assert ana.min_nextflow_version == "23.04.0"
    assert "biocontainers/foo:1.0" in ana.process_containers.values()


def test_analyse_nextflow_defaults_to_dsl2_when_unmarked(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("workflow { foo() }")
    assert analyse_nextflow(tmp_path).dsl_version == 2


def test_build_plan_nextflow_dsl1_pins_runner(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=1\nprocess foo { }")
    plan = build_resurrection_plan(tmp_path)
    assert plan.language == "nextflow"
    assert plan.runner_version_pin == "nextflow 22.10.8"
    assert any("DSL1" in issue for issue in plan.known_issues)


def test_build_plan_nextflow_nfcore(tmp_path: Path) -> None:
    (tmp_path / "main.nf").write_text("nextflow.enable.dsl=2\n// nf-core/sarek\nworkflow { }")
    plan = build_resurrection_plan(tmp_path)
    assert plan.nextflow is not None and plan.nextflow.nfcore_pipeline
    assert any("nf-co.re" in issue for issue in plan.known_issues)


def test_analyse_snakemake(tmp_path: Path) -> None:
    (tmp_path / "Snakefile").write_text(
        'from snakemake.utils import min_version\nmin_version("7.8.0")\n'
        'rule a:\n    container: "docker://biocontainers/bwa:0.7.17"\n'
    )
    ana = analyse_snakemake(tmp_path)
    assert ana.min_snakemake_version == "7.8.0"
    assert "biocontainers/bwa:0.7.17" in ana.docker_images


def test_build_plan_snakemake(tmp_path: Path) -> None:
    (tmp_path / "Snakefile").write_text('min_version("7.8.0")\nrule all:\n    input: []')
    plan = build_resurrection_plan(tmp_path)
    assert plan.language == "snakemake"
    assert plan.runner_version_pin == "snakemake==7.8.0"


def test_analyse_wdl_dedups_images(tmp_path: Path) -> None:
    (tmp_path / "w.wdl").write_text(
        'task a { runtime { docker: "ubuntu:20.04" } }\n'
        'task b { runtime { docker: "ubuntu:20.04" } }\n'
    )
    ana = analyse_wdl(tmp_path)
    assert ana.runtime_images == ["ubuntu:20.04"]


def test_build_plan_wdl(tmp_path: Path) -> None:
    (tmp_path / "w.wdl").write_text('task a { runtime { docker: "ubuntu:20.04" } }')
    plan = build_resurrection_plan(tmp_path)
    assert plan.language == "wdl"
    assert "miniwdl" in plan.entry_point


def test_build_plan_unknown_falls_back_to_cwl(tmp_path: Path) -> None:
    (tmp_path / "workflow.cwl").write_text("cwlVersion: v1.2\nclass: Workflow")
    plan = build_resurrection_plan(tmp_path)
    assert plan.language == "cwl"
    assert "cwltool" in plan.entry_point
