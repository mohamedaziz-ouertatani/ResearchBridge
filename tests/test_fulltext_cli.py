from __future__ import annotations

from unittest.mock import Mock

from researchbridge.fulltext.cli import build_parser, main


def test_parser_accepts_limit_and_force() -> None:
    parser = build_parser()
    args = parser.parse_args(["--limit", "5", "--force"])
    assert args.limit == 5
    assert args.force is True


def test_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    assert args.limit is None
    assert args.force is False


def test_main_runs_the_pipeline_with_parsed_args(monkeypatch, capsys) -> None:
    import researchbridge.fulltext.cli as cli_module

    mock_pipeline = Mock()
    mock_pipeline.run.return_value = "run-123"
    monkeypatch.setattr(cli_module, "FullTextFetchPipeline", Mock(return_value=mock_pipeline))
    monkeypatch.setattr(cli_module, "make_engine", Mock())
    monkeypatch.setattr(cli_module, "make_session_factory", Mock())
    monkeypatch.setattr(cli_module, "load_config", Mock())
    monkeypatch.setattr("sys.argv", ["rb-fulltext-fetch", "--limit", "3"])

    main()

    mock_pipeline.run.assert_called_once_with(limit=3, force=False)
    assert "run-123" in capsys.readouterr().out
