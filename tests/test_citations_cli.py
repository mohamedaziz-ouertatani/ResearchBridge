from __future__ import annotations

import pytest

from researchbridge.citations.cli_fetch import build_parser, validate_args


def test_accepts_a_source_id_alone() -> None:
    args = build_parser().parse_args(["abc123"])
    validate_args(args)  # must not raise


def test_accepts_all_alone() -> None:
    args = build_parser().parse_args(["--all"])
    validate_args(args)  # must not raise


def test_rejects_source_id_and_all_together() -> None:
    args = build_parser().parse_args(["abc123", "--all"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_neither_source_id_nor_all() -> None:
    args = build_parser().parse_args([])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_force_without_all() -> None:
    args = build_parser().parse_args(["abc123", "--force"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_accepts_force_with_all() -> None:
    args = build_parser().parse_args(["--all", "--force"])
    validate_args(args)  # must not raise


def test_source_defaults_to_semantic_scholar() -> None:
    args = build_parser().parse_args(["abc123"])
    assert args.source == "semantic_scholar"


def test_accepts_crossref_source() -> None:
    args = build_parser().parse_args(["10.1000/abc", "--source", "crossref"])
    validate_args(args)  # must not raise
    assert args.source == "crossref"


def test_rejects_unknown_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["abc123", "--source", "bogus"])
