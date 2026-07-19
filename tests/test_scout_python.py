"""Tests for sanjeevini.scouts.python_scout (target: 75% branch coverage)."""

from __future__ import annotations

import pytest

from sanjeevini.scouts import python_scout as ps
from sanjeevini.scouts.python_scout import (
    PythonScout,
    detect_framework,
    detect_python_version,
    ensure_falsifiable,
    extract_doi,
    generate_sanity_check,
    is_falsifiable,
    select_base_image,
)
from sanjeevini.scouts.repo import RepoSnapshot


def _snap(files: dict[str, str], **kw) -> RepoSnapshot:
    return RepoSnapshot(
        url="https://github.com/acme/tool",
        owner="acme",
        name="tool",
        files=files,
        **kw,
    )


# ---- falsifiability guard -------------------------------------------------


def test_is_falsifiable_accepts_thresholds() -> None:
    assert is_falsifiable("ROC-AUC ≥ 0.70 on the benchmark")
    assert is_falsifiable("output is non-empty and ≥ 10 KB")
    assert is_falsifiable("VCF parses with bcftools and has ≥ 1 record")
    assert is_falsifiable("emits at least 100 sequences")


def test_is_falsifiable_rejects_execution_only() -> None:
    assert not is_falsifiable("the tool runs without error")
    assert not is_falsifiable("executes successfully to completion")
    assert not is_falsifiable("produces output")


def test_ensure_falsifiable_raises_on_weak_check() -> None:
    with pytest.raises(ValueError):
        ensure_falsifiable("the tool runs without error")


def test_ensure_falsifiable_returns_strong_check() -> None:
    check = "output file is non-empty"
    assert ensure_falsifiable(check) == check


# ---- framework / version / image detection --------------------------------


def test_detect_framework_tf1() -> None:
    assert detect_framework("tensorflow-gpu==1.14.0") == "tensorflow-1.x"
    assert detect_framework("tensorflow==1.15") == "tensorflow-1.x"


def test_detect_framework_tf2_and_torch_and_plain() -> None:
    assert detect_framework("tensorflow==2.11.0") == "tensorflow-2.x"
    assert detect_framework("torch>=1.13\nnumpy") == "pytorch"
    assert detect_framework("jax[cpu]") == "jax"
    assert detect_framework("numpy\npandas\nscipy") == "plain-python"


def test_detect_python_version_from_classifier() -> None:
    text = "Programming Language :: Python :: 3.8\nProgramming Language :: Python :: 3.9"
    assert detect_python_version(text, "plain-python") == "3.8"


def test_detect_python_version_default_by_framework() -> None:
    assert detect_python_version("no hints", "tensorflow-1.x") == "3.6"


def test_select_base_image_prefers_dockerfile_from() -> None:
    img = select_base_image("pytorch", "3.9", "torch", "FROM nvidia/cuda:11.8.0-base\nRUN x")
    assert img == "nvidia/cuda:11.8.0-base"


def test_select_base_image_by_framework() -> None:
    assert select_base_image("tensorflow-1.x", "3.6", "tensorflow==1.14", "").startswith(
        "tensorflow/tensorflow:1.14"
    )
    assert "cuda" in select_base_image("pytorch", "3.9", "torch", "")
    assert select_base_image("plain-python", "3.10", "", "") == "python:3.10-slim"


def test_extract_doi_and_arxiv() -> None:
    assert extract_doi("See https://doi.org/10.1038/s41586-021-03819-2 for details").startswith(
        "10.1038/"
    )
    assert extract_doi("preprint arXiv:2103.12345") == "arXiv:2103.12345"
    assert extract_doi("no identifier here") is None


# ---- sanity-check generation ----------------------------------------------


def test_generate_sanity_check_uses_benchmark() -> None:
    snap = _snap({"README.md": "Our model achieves an AUC of 0.92 on the test set."})
    check = generate_sanity_check(snap)
    assert "AUC" in check and is_falsifiable(check)


def test_generate_sanity_check_structural_for_vcf() -> None:
    snap = _snap({"README.md": "This caller writes a VCF of variants."})
    check = generate_sanity_check(snap)
    assert "bcftools" in check and is_falsifiable(check)


def test_generate_sanity_check_generic_fallback() -> None:
    snap = _snap({"README.md": "A tool that does a thing."})
    check = generate_sanity_check(snap)
    assert is_falsifiable(check)


# ---- PythonScout.plan -----------------------------------------------------


async def test_plan_builds_falsifiable_plan() -> None:
    snap = _snap(
        {
            "README.md": "DeepCaller predicts variants. Achieves F1 of 0.88.\n"
            "See doi.org/10.1000/xyz123",
            "requirements.txt": "tensorflow-gpu==1.14.0\nnumpy",
        },
        open_issues=[("Fails on TF 2.x", "import error")],
    )
    plan = await PythonScout("https://github.com/acme/tool", snapshot=snap).plan(confirm=False)
    assert plan.framework == "tensorflow-1.x"
    assert plan.python_version == "3.6"
    assert plan.base_image.startswith("tensorflow/tensorflow:1.14")
    assert is_falsifiable(plan.sanity_check)
    assert plan.paper_doi is not None
    assert "Fails on TF 2.x" in plan.known_issues
    assert plan.estimated_turns >= 25


async def test_plan_raises_when_sanity_check_not_falsifiable(monkeypatch) -> None:
    monkeypatch.setattr(ps, "generate_sanity_check", lambda snap: "the tool runs without error")
    snap = _snap({"README.md": "x"})
    with pytest.raises(ValueError):
        await PythonScout("https://github.com/acme/tool", snapshot=snap).plan(confirm=False)


async def test_plan_uses_injected_fetcher() -> None:
    snap = _snap({"README.md": "Outputs a FASTA of designed sequences."})

    async def fake_fetcher(url: str) -> RepoSnapshot:
        return snap

    plan = await PythonScout("https://github.com/acme/tool", fetcher=fake_fetcher).plan(
        confirm=False
    )
    assert plan.capability
    assert is_falsifiable(plan.sanity_check)


async def test_plan_confirm_prints_and_reads(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    snap = _snap({"README.md": "A tool. Produces a BAM with alignments."})
    await PythonScout("https://github.com/acme/tool", snapshot=snap).plan(confirm=True)
    out = capsys.readouterr().out
    assert "Resurrection Plan" in out
    assert "sanity check" in out
