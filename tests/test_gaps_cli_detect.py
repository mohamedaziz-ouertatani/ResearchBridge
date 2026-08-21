from __future__ import annotations

import pytest

from researchbridge.gaps.cli_detect import build_parser, validate_args


def test_accepts_a_source_id_alone() -> None:
    args = build_parser().parse_args(["1234.5678"])
    validate_args(args)  # must not raise


def test_accepts_all_alone() -> None:
    args = build_parser().parse_args(["--all"])
    validate_args(args)  # must not raise


def test_rejects_source_id_and_all_together() -> None:
    args = build_parser().parse_args(["1234.5678", "--all"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_neither_source_id_nor_all() -> None:
    args = build_parser().parse_args([])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_rejects_force_without_all() -> None:
    args = build_parser().parse_args(["1234.5678", "--force"])
    with pytest.raises(SystemExit):
        validate_args(args)


def test_accepts_force_with_all() -> None:
    args = build_parser().parse_args(["--all", "--force"])
    validate_args(args)  # must not raise
