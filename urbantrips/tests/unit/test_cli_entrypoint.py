def test_run_all_cli_parser_accepts_delete_and_no_dashboard_flags():
    from urbantrips import run_all_urbantrips

    parser = run_all_urbantrips.build_parser()
    args = parser.parse_args(["--borrar_corrida", "alias1", "--no_dashboard"])

    assert args.borrar_corrida == "alias1"
    assert args.no_dashboard is True


def test_run_all_main_forwards_cli_options(monkeypatch):
    from urbantrips import run_all_urbantrips

    calls = []
    monkeypatch.setattr(
        run_all_urbantrips,
        "run_all",
        lambda **kwargs: calls.append(kwargs),
    )

    run_all_urbantrips.main(borrar_corrida="all", crear_dashboard=False)

    assert calls == [{"borrar_corrida": "all", "crear_dashboard": False}]
