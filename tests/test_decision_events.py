"""Event-based decision outputs: one row per decision rather than per frame.

Pins the event detection (center decisions, open-arm excursions, the same-place
control), the held-signal mask that keeps unrecorded photometry out of every
decision analysis, the debounce, and the circular-shift null.
"""
import numpy as np

from tests.test_decision_probability import _app, _subject, POINTS

FROM, TO, MID = ['closed_a'], ['open_a'], ['center']


def _seq(*parts):
    out = []
    for zone, n in parts:
        out += [zone] * n
    return out


def _events(app, name='s1', **kw):
    kw.setdefault('min_depth_cm', 0.0)
    return app._dec_events_for_subject(name, 'G0', FROM, TO, MID, **kw)


def test_center_visits_are_split_into_explore_and_avoid():
    app = _app()
    seq = _seq(('closed_a', 40), ('center', 20), ('open_a', 30), ('center', 20),
               ('closed_a', 40), ('center', 20), ('closed_a', 40))
    rng = np.random.default_rng(0)
    _subject(app, 's1', seq, rng.standard_normal(len(seq)), fps=10.0)
    ev = _events(app)
    c = ev['center']
    # The open -> center visit is not a decision: it was not entered from From.
    assert list(c['entry']) == [40, 150]
    assert list(c['exit']) == [60, 170]
    assert list(c['explore']) == [True, False]


def test_signal_windows_read_the_frames_they_name():
    app = _app()
    seq = _seq(('closed_a', 40), ('center', 20), ('open_a', 40))
    sig = np.arange(len(seq), dtype=float)
    _subject(app, 's1', seq, sig, fps=10.0)
    ev = _events(app, win_s=1.0)
    v = app._dec_event_signals(ev['sig'], ev)
    assert v['pre'][0] == np.mean(sig[30:40])      # 1 s before center entry
    assert v['dwell'][0] == np.mean(sig[40:60])    # the whole visit
    assert v['last'][0] == np.mean(sig[50:60])     # last 1 s before leaving


def test_boundary_flicker_is_not_a_decision():
    app = _app()
    # Two-frame dips into the center are tracking jitter on the boundary.
    seq = _seq(('closed_a', 40), ('center', 2), ('closed_a', 40), ('center', 20),
               ('closed_a', 40))
    _subject(app, 's1', seq, np.random.default_rng(1).standard_normal(len(seq)), fps=10.0)
    assert len(_events(app, debounce_s=0.5)['center']['entry']) == 1
    assert len(_events(app, debounce_s=0.0)['center']['entry']) == 2


def test_held_tail_is_never_scored():
    """Rows after the recording stops repeat one sample; they are not data."""
    app = _app()
    seq = _seq(('closed_a', 40), ('center', 20), ('closed_a', 40),
               ('center', 20), ('open_a', 40), ('closed_a', 40))
    sig = np.random.default_rng(2).standard_normal(len(seq))
    sig[100:] = sig[100]                 # recording ended at row 100
    _subject(app, 's1', seq, sig, fps=10.0)
    ev = _events(app)
    assert list(ev['center']['entry']) == [40]         # the second visit is in the tail
    assert len(ev['exc']['peak']) == 0
    assert ev['hi'] <= 102
    r = app._compute_dec_prob_for_subject('s1', 'G0', FROM, TO, 1.0)
    assert np.sum(r['signals'] == sig[100]) <= 2


def test_excursion_reports_its_furthest_point_and_where_it_went_next():
    app = _app()
    seq = _seq(('closed_a', 30), ('center', 10), ('open_a', 40), ('center', 10), ('closed_a', 30))
    beh = _subject(app, 's1', seq, np.random.default_rng(3).standard_normal(len(seq)), fps=10.0)
    # Walk out along the open arm (y falls away from the center at y=25) and back.
    y = np.concatenate((np.linspace(9.0, 1.0, 25), np.linspace(1.0, 9.0, 15)))
    beh[40:80, 3] = y
    ev = _events(app)
    x = ev['exc']
    assert list(x['entry']) == [40] and list(x['exit']) == [80]
    assert x['peak'][0] == 40 + 24
    assert np.isclose(x['dmax'][0], 24.0)              # center (25, 25) to (25, 1)
    assert np.isclose(x['depth'][0], 8.0)
    assert x['next'][0] == 1                           # back to the From zone


def test_same_place_control_uses_a_further_going_excursion():
    app = _app()
    seq = _seq(('closed_a', 30), ('center', 10), ('open_a', 20), ('center', 10),
               ('closed_a', 30), ('center', 10), ('open_a', 40), ('center', 10), ('closed_a', 30))
    beh = _subject(app, 's1', seq, np.arange(len(seq), dtype=float), fps=10.0)
    beh[40:60, 3] = np.concatenate((np.linspace(9.5, 7.0, 10), np.linspace(7.0, 9.5, 10)))
    beh[110:150, 3] = np.concatenate((np.linspace(9.5, 0.5, 20), np.linspace(0.5, 9.5, 20)))
    ev = _events(app, ctrl_margin_cm=5.0)
    x = ev['exc']
    assert np.isclose(x['dmax'][0], 18.0) and np.isclose(x['dmax'][1], 24.5)
    # The shallow excursion is compared with the deep one passing 18 cm on the way out.
    hit = x['ctrl'][0]
    assert len(hit) == 1 and 110 <= hit[0] < 130
    assert ev['dist'][hit[0]] >= 18.0 > ev['dist'][hit[0] - 1]
    assert len(x['ctrl'][1]) == 0                      # nothing went further than the deepest


def test_auroc_is_one_when_signal_orders_the_choices():
    app = _app()
    assert app._dec_auroc([1, 2, 3, 7, 8, 9], [0, 0, 0, 1, 1, 1]) == 1.0
    assert app._dec_auroc([7, 8, 9, 1, 2, 3], [0, 0, 0, 1, 1, 1]) == 0.0
    assert np.isnan(app._dec_auroc([1, 2, 3], [0, 1, 1]))   # too few of one kind


def test_shift_null_separates_a_locked_signal_from_an_unrelated_one():
    rng = np.random.default_rng(4)
    app = _app()
    seq, explore = [], []
    for k in range(40):
        go = bool(rng.integers(0, 2))
        explore.append(go)
        seq += _seq(('closed_a', int(rng.integers(60, 120))), ('center', 20),
                    ('open_a' if go else 'closed_a', int(rng.integers(40, 80))),
                    ('center', 15))
    seq += ['closed_a'] * 100
    n = len(seq)
    noise = rng.standard_normal(n)
    _subject(app, 'noise', seq, noise, fps=10.0)
    locked = noise.copy()
    _subject(app, 'locked', seq, locked, fps=10.0)
    ev = _events(app, 'locked')
    for e, go in zip(ev['center']['entry'], ev['center']['explore']):
        locked[e - 10:e] += 3.0 if go else -3.0
    app.processed_data['locked']['beh_synced'][:, 6] = locked

    out = {}
    for name in ('locked', 'noise'):
        ev = _events(app, name)
        obs = app._dec_event_summary(ev, app._dec_event_signals(ev['sig'], ev), 'pre')['auroc']
        nul = app._dec_event_null(ev, 'pre', n_shift=200, rng=np.random.default_rng(5))['auroc']
        out[name] = app._dec_group_test([obs], [nul], 0.5)
    assert out['locked']['mean'] > 0.95 and out['locked']['p_shift'] < 0.01
    assert out['noise']['p_shift'] > 0.05
    assert abs(out['noise']['null_mean'] - 0.5) < 0.05
