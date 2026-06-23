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


def test_parser_accepts_step_flag():
    from urbantrips import run_all_urbantrips

    parser = run_all_urbantrips.build_parser()
    args = parser.parse_args(["--step", "dashboard"])
    assert args.step == "dashboard"
    assert args.through is None


def test_parser_accepts_through_flag():
    from urbantrips import run_all_urbantrips

    parser = run_all_urbantrips.build_parser()
    args = parser.parse_args(["--through", "outputs"])
    assert args.through == "outputs"
    assert args.step is None


def test_parser_rejects_step_and_through_together():
    import pytest
    from urbantrips import run_all_urbantrips

    parser = run_all_urbantrips.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--step", "legs", "--through", "outputs"])


def test_validate_args_rejects_step_with_borrar_corrida():
    import argparse
    import pytest
    from urbantrips import run_all_urbantrips

    args = argparse.Namespace(step="legs", borrar_corrida="all")
    with pytest.raises(SystemExit):
        run_all_urbantrips._validate_args(args)


def test_main_dispatches_single_step(monkeypatch):
    from urbantrips import run_all_urbantrips

    calls = []
    monkeypatch.setattr(run_all_urbantrips, "run_all", lambda **kwargs: calls.append(("run_all", kwargs)))
    monkeypatch.setattr(run_all_urbantrips, "_run_step", lambda step: calls.append(("step", step)))

    run_all_urbantrips.main(step="dashboard")
    assert calls == [("step", "dashboard")]


def test_main_dispatches_through(monkeypatch):
    from urbantrips import run_all_urbantrips

    calls = []
    monkeypatch.setattr(run_all_urbantrips, "run_all", lambda **kwargs: calls.append(("run_all", kwargs)))
    monkeypatch.setattr(run_all_urbantrips, "_run_through", lambda through: calls.append(("through", through)))

    run_all_urbantrips.main(through="outputs")
    assert calls == [("through", "outputs")]


def test_main_default_calls_run_all(monkeypatch):
    from urbantrips import run_all_urbantrips

    calls = []
    monkeypatch.setattr(run_all_urbantrips, "run_all", lambda **kwargs: calls.append(kwargs))

    run_all_urbantrips.main()
    assert calls == [{"borrar_corrida": "", "crear_dashboard": True}]


def test_base_dir_flag_is_accepted():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args(["--base-dir", "/tmp/run_a"])
    assert args.base_dir == "/tmp/run_a"


def test_base_dir_short_flag():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args(["-d", "/tmp/run_b"])
    assert args.base_dir == "/tmp/run_b"


def test_base_dir_defaults_to_none():
    from urbantrips.run_all_urbantrips import build_parser
    parser = build_parser()
    args = parser.parse_args([])
    assert args.base_dir is None
