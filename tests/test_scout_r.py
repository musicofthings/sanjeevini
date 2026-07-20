"""Tests for sanjeevini.scouts.r_scout (target: 75% branch coverage)."""

from __future__ import annotations

import pytest

from sanjeevini.scouts.r_scout import (
    RScout,
    parse_depends,
    parse_description,
    resolve_bioc_release,
    select_rocker_image,
)
from sanjeevini.scouts.repo import RepoSnapshot

_BIOC_DESCRIPTION = """\
Package: MyBioPkg
Title: Differential Expression Analysis
Version: 1.2.0
Description: Tools for differential expression
    from RNA-seq count data.
Depends: R (>= 4.1), methods
Imports: S4Vectors,
    IRanges
biocViews: RNASeq, DifferentialExpression
"""

_CRAN_DESCRIPTION = """\
Package: mycranpkg
Title: A CRAN Utility
Version: 0.3.0
Description: Small helpers.
Depends: R (>= 4.2)
Imports: stats
"""

_BIOC_320_DESCRIPTION = """\
Package: NewBioPkg
Title: Bleeding Edge
Version: 0.1.0
Description: Uses a very recent R.
Depends: R (>= 4.5)
biocViews: Software
"""


def _snap(description: str, **kw) -> RepoSnapshot:
    return RepoSnapshot(
        url="https://github.com/acme/rpkg",
        owner="acme",
        name="rpkg",
        files={"DESCRIPTION": description} if description else {},
        **kw,
    )


# ---- DCF parsing ----------------------------------------------------------


def test_parse_description_handles_continuations() -> None:
    fields = parse_description(_BIOC_DESCRIPTION)
    assert fields["Package"] == "MyBioPkg"
    assert "biocViews" in fields
    assert "IRanges" in fields["Imports"]


def test_parse_depends_extracts_r_version_and_excludes_r() -> None:
    fields = parse_description(_BIOC_DESCRIPTION)
    r_version, depends = parse_depends(fields)
    assert r_version == "4.1"
    assert "R" not in depends
    assert "methods" in depends and "S4Vectors" in depends and "IRanges" in depends


# ---- release / image resolution -------------------------------------------


def test_resolve_bioc_release_from_r_version() -> None:
    # R 4.1 shipped with both 3.13 and 3.14; the newest wins.
    assert resolve_bioc_release("4.1") == "3.14"


def test_resolve_bioc_release_none_for_unknown_r() -> None:
    assert resolve_bioc_release("3.9") is None
    assert resolve_bioc_release(None) is None


def test_select_rocker_image_bioconductor() -> None:
    assert select_rocker_image("3.14", "4.1") == "rocker/bioconductor:3.14"


def test_select_rocker_image_rver_for_cran_and_new_bioc() -> None:
    assert select_rocker_image(None, "4.2") == "rocker/r-ver:4.2"
    assert select_rocker_image("3.21", "4.5") == "rocker/r-ver:4.5"


# ---- RScout.plan ----------------------------------------------------------


async def test_plan_bioc_selects_correct_rocker_image() -> None:
    plan = await RScout("https://github.com/acme/rpkg", snapshot=_snap(_BIOC_DESCRIPTION)).plan(
        confirm=False
    )
    assert plan.bioc_release == "3.14"
    assert plan.r_version == "4.1"
    assert plan.base_image == "rocker/bioconductor:3.14"
    assert plan.package_name == "MyBioPkg"
    assert "methods" in plan.depends


async def test_plan_cran_uses_rver_image() -> None:
    plan = await RScout("https://github.com/acme/rpkg", snapshot=_snap(_CRAN_DESCRIPTION)).plan(
        confirm=False
    )
    assert plan.bioc_release is None
    assert plan.base_image == "rocker/r-ver:4.2"


async def test_plan_new_bioc_falls_back_to_rver() -> None:
    plan = await RScout("https://github.com/acme/rpkg", snapshot=_snap(_BIOC_320_DESCRIPTION)).plan(
        confirm=False
    )
    assert plan.bioc_release == "3.21"
    assert plan.base_image == "rocker/r-ver:4.5"


async def test_plan_raises_without_description() -> None:
    with pytest.raises(ValueError):
        await RScout("https://github.com/acme/rpkg", snapshot=_snap("")).plan(confirm=False)


async def test_plan_sanity_check_is_falsifiable() -> None:
    from sanjeevini.scouts.python_scout import is_falsifiable

    plan = await RScout("https://github.com/acme/rpkg", snapshot=_snap(_BIOC_DESCRIPTION)).plan(
        confirm=False
    )
    assert is_falsifiable(plan.sanity_check)


async def test_plan_confirm_prints(monkeypatch, capsys) -> None:
    monkeypatch.setattr("builtins.input", lambda *_a: "")
    await RScout("https://github.com/acme/rpkg", snapshot=_snap(_BIOC_DESCRIPTION)).plan(
        confirm=True
    )
    assert "R Resurrection Plan" in capsys.readouterr().out
