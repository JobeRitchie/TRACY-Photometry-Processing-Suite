"""Behavior-tab metric definitions: what counts as a visit, an entry, a decision.

Time, entries and decisions all read one cleaned zone sequence, so they cannot
disagree; arm entries are counted on the open/closed category; untracked frames
neither split a visit nor dilute a percentage.
"""
import numpy as np

import fp_analysis_gui as G

ZONES = {
    'center':          dict(x_min=30, x_max=46, y_min=30, y_max=46, category='center'),
    'open_proximal':   dict(x_min=30, x_max=46, y_min=46, y_max=60, category='open_proximal'),
    'open_distal':     dict(x_min=30, x_max=46, y_min=60, y_max=76, category='open_distal'),
    'closed_arm':      dict(x_min=46, x_max=76, y_min=30, y_max=46, category='closed'),
}
FPS = 10.0


def _app(maze='EPM_Complex'):
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.zones = dict(ZONES)
    app.params = {'maze_type': maze, 'maze_width_cm': 76,
                  'metrics_min_entry_duration': 0.5, 'metrics_min_refractory_sec': 1.0,
                  'velocity_outlier_threshold': 5}
    return app


def _seq(*parts):
    out = []
    for zone, n in parts:
        out += [zone] * n
    return out


def _metrics(app, zones):
    zones = app._clean_zone_sequence(zones, int(0.5 * FPS))
    n = len(zones)
    return app._calculate_session_metrics(zones, np.full(n, 38.0), np.full(n, 38.0),
                                          np.ones(n), FPS, 25, n / FPS)


def test_flicker_shorter_than_the_minimum_visit_is_absorbed():
    clean = G.FPAnalysisGUI._clean_zone_sequence(
        _seq(('closed_arm', 20), ('center', 2), ('closed_arm', 20)), 5)
    assert clean == ['closed_arm'] * 42


def test_tracking_gap_inside_one_zone_does_not_split_the_visit():
    clean = G.FPAnalysisGUI._clean_zone_sequence(
        _seq(('center', 20), ('unknown', 30), ('center', 20)), 5)
    assert clean == ['center'] * 70
    # A gap the animal came out of somewhere else stays unknown.
    kept = G.FPAnalysisGUI._clean_zone_sequence(
        _seq(('center', 20), ('unknown', 30), ('closed_arm', 20)), 5)
    assert kept.count('unknown') == 30


def test_a_step_from_proximal_to_distal_is_one_arm_entry():
    app = _app()
    m = _metrics(app, _seq(('center', 20), ('open_proximal', 20), ('open_distal', 20),
                           ('open_proximal', 20), ('center', 20), ('closed_arm', 20)))
    assert m['Open_Arm_Entries'] == 1
    assert m['Closed_Arm_Entries'] == 1


def test_flicker_into_the_center_is_not_an_avoid_decision():
    app = _app()
    m = _metrics(app, _seq(('closed_arm', 30), ('center', 2), ('closed_arm', 30),
                           ('center', 20), ('closed_arm', 30),
                           ('center', 20), ('open_proximal', 30)))
    assert m['Avoid_Decisions'] == 1
    assert m['Explore_Decisions'] == 1


def test_percent_open_is_of_tracked_time():
    app = _app()
    m = _metrics(app, _seq(('center', 25), ('open_proximal', 25), ('unknown', 50)))
    assert m['Percent_Time_Open'] == 50.0
    assert m['Percent_Tracked'] == 50.0
    assert m['Analyzed_Duration_sec'] == 10.0
