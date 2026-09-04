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
    mock_pipeline_cls = Mock(return_value=mock_pipeline)
    monkeypatch.setattr(cli_module, "FullTextFetchPipeline", mock_pipeline_cls)
    monkeypatch.setattr(cli_module, "make_engine", Mock())
    monkeypatch.setattr(cli_module, "make_session_factory", Mock())
    monkeypatch.setattr(cli_module, "load_config", Mock())
    monkeypatch.setenv("CORE_API_KEY", "test-key")
    monkeypatch.setattr("sys.argv", ["rb-fulltext-fetch", "--limit", "3"])

    main()

    mock_pipeline.run.assert_called_once_with(limit=3, force=False)
    assert mock_pipeline_cls.call_args.kwargs["core_api_key"] == "test-key"
    assert "run-123" in capsys.readouterr().out
