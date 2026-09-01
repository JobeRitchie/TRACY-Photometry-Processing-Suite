"""Decision probability: P(enter the outcome zone within T s | current z-score).

The measure is a hazard, not an event probability: every frame the animal spends
in the source zone is one observation, scored 1 if the outcome zone appears
anywhere in the next T seconds.  These tests pin the three things that were
silently wrong or slow about it -- the look-ahead scan, the signal column, and
the sampling rate the window is converted with -- against a plain reference
implementation of the same definition.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


ZONES = {
    'closed_a': dict(x_min=0.0,  x_max=10.0, y_min=20.0, y_max=30.0, category='closed'),
    'open_a':   dict(x_min=20.0, x_max=30.0, y_min=0.0,  y_max=10.0, category='open'),
    'center':   dict(x_min=20.0, x_max=30.0, y_min=20.0, y_max=30.0, category='center'),
}
# A point inside each zone, plus one outside every zone.
POINTS = {'closed_a': (5.0, 25.0), 'open_a': (25.0, 5.0),
          'center': (25.0, 25.0), 'unknown': (95.0, 95.0)}


def _app(zones=None):
    """An FPAnalysisGUI shell with just what the decision-probability path reads."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.zones = dict(zones or ZONES)
    app.processed_data = {}
    app.params = {}
    return app


def _subject(app, name, zone_seq, signal, fps=20.0, n_channels=2,
             channel_names=('G0', 'G1'), signal2=None):
    """Register a subject whose position walks *zone_seq*, one zone per frame."""
    n = len(zone_seq)
    n_kin = len(G.FPAnalysisGUI.BEH_KIN_FIELDS)
    beh = np.full((n, 6 + n_channels + n_kin), np.nan)
    beh[:, 0] = np.arange(n)
    beh[:, 1] = np.arange(n) / fps
    for i, z in enumerate(zone_seq):
        beh[i, 2], beh[i, 3] = POINTS[z]
    beh[:, 6] = signal
    if n_channels > 1:
        beh[:, 7] = signal2 if signal2 is not None else -np.asarray(signal)
    # Kinematics columns start with elapsed time -- the column a hardcoded
    # "G1 is column 7" would read for a one-channel subject.
    beh[:, 6 + n_channels] = np.arange(n) / fps
    app.processed_data[name] = {
        'beh_synced': beh, 'has_position': True, 'photometry_fps': fps,
        'num_photometry_channels': n_channels,
        'channel_names': list(channel_names)}
    return beh


def _reference(app, subject, col, source, outcome, lookahead_frames):
    """The definition, written the slow obvious way, for the fast path to match."""
    beh = app.processed_data[subject]['beh_synced']
    n = len(beh)
    zones = []
    for i in range(n):
        x, y = beh[i, 2], beh[i, 3]
        zones.append('unknown' if (np.isnan(x) or np.isnan(y))
                     else app.classify_zone(x, y))
    src, out = set(source), set(outcome)
    sigs, outs = [], []
    for i in range(n - lookahead_frames):
        if zones[i] not in src:
            continue
        s = beh[i, col]
        if np.isnan(s):
            continue
        sigs.append(s)
        outs.append(1 if any(zones[j] in out
                             for j in range(i + 1, i + lookahead_frames + 1)) else 0)
    return np.array(sigs), np.array(outs)


def _walk(rng, n):
    """A zone sequence with realistic long dwells rather than per-frame flicker."""
    seq = []
    order = ['closed_a', 'center', 'open_a', 'center']
    k = 0
    while len(seq) < n:
        seq += [order[k % 4]] * int(rng.integers(15, 90))
        k += 1
    return seq[:n]


# -- The look-ahead scan ------------------------------------------------------

def test_matches_reference_implementation():
    rng = np.random.default_rng(7)
    n = 4000
    app = _app()
    seq = _walk(rng, n)
    sig = rng.standard_normal(n)
    _subject(app, 's1', seq, sig, fps=20.0)

    for look_s in (0.5, 2.0, 10.0):
        L = max(1, int(round(look_s * 20.0)))
        r = app._compute_dec_prob_for_subject(
            's1', 'G0', ['closed_a'], ['open_a'], look_s)
        exp_s, exp_o = _reference(app, 's1', 6, ['closed_a'], ['open_a'], L)
        assert np.array_equal(r['signals'], exp_s)
        assert np.array_equal(r['outcomes'].astype(int), exp_o)
        assert r['lookahead_frames'] == L


def test_nan_signal_frames_are_dropped_not_scored():
    app = _app()
    seq = ['closed_a'] * 20 + ['open_a'] * 10
    sig = np.arange(30, dtype=float)
    sig[3] = np.nan
    _subject(app, 's1', seq, sig, fps=10.0)
    r = app._compute_dec_prob_for_subject('s1', 'G0', ['closed_a'], ['open_a'], 1.0)
    assert not np.isnan(r['signals']).any()
    assert 3.0 not in set(r['signals'])


def test_outcome_must_fall_inside_the_window():
    app = _app()
    # 30 closed frames, then open.  At 10 fps a 1 s window is 10 frames, so only
    # the last 10 scoreable closed frames can see the entry.
    seq = ['closed_a'] * 30 + ['open_a'] * 20
    _subject(app, 's1', seq, np.zeros(50), fps=10.0)
    r = app._compute_dec_prob_for_subject('s1', 'G0', ['closed_a'], ['open_a'], 1.0)
    o = r['outcomes'].astype(int)
    assert o.sum() == 10
    assert o[:20].sum() == 0 and o[20:].sum() == 10


def test_tail_without_a_full_window_is_censored():
    app = _app()
    _subject(app, 's1', ['closed_a'] * 50, np.zeros(50), fps=10.0)
    r = app._compute_dec_prob_for_subject('s1', 'G0', ['closed_a'], ['open_a'], 1.0)
    assert len(r['signals']) == 40          # 50 - 10 look-ahead frames


# -- Sampling rate and channel resolution -------------------------------------

def test_lookahead_follows_each_subjects_own_sampling_rate():
    app = _app()
    seq = ['closed_a'] * 600
    _subject(app, 'slow', seq, np.zeros(600), fps=20.0)
    _subject(app, 'fast', seq, np.zeros(600), fps=30.0)
    slow = app._compute_dec_prob_for_subject('slow', 'G0', ['closed_a'], ['open_a'], 5.0)
    fast = app._compute_dec_prob_for_subject('fast', 'G0', ['closed_a'], ['open_a'], 5.0)
    assert slow['lookahead_frames'] == 100
    assert fast['lookahead_frames'] == 150   # same 5 s, not the same frame count


def test_single_channel_g1_subject_reads_its_real_trace():
    """A hardcoded 'G1 == column 7' reads elapsed time for a one-channel subject."""
    app = _app()
    seq = ['closed_a'] * 40 + ['open_a'] * 10
    sig = np.linspace(-2.0, 2.0, 50)
    _subject(app, 's1', seq, sig, fps=10.0, n_channels=1, channel_names=('G1',))
    r = app._compute_dec_prob_for_subject('s1', 'G1', ['closed_a'], ['open_a'], 1.0)
    assert np.allclose(r['signals'], sig[:40])   # 50 frames - 10 look-ahead
    # Not the elapsed-time column that sits where a second channel would be.
    assert not np.allclose(r['signals'], np.arange(40) / 10.0)


def test_g1_selects_the_second_channel_when_both_exist():
    app = _app()
    seq = ['closed_a'] * 40 + ['open_a'] * 10
    a = np.linspace(0.0, 1.0, 50)
    b = np.linspace(5.0, 6.0, 50)
    _subject(app, 's1', seq, a, fps=10.0, n_channels=2, signal2=b)
    r = app._compute_dec_prob_for_subject('s1', 'G1', ['closed_a'], ['open_a'], 1.0)
    assert np.allclose(r['signals'], b[:40])


def test_zone_codes_match_classify_zone():
    app = _app()
    rng = np.random.default_rng(3)
    n = 500
    beh = np.full((n, 8), np.nan)
    beh[:, 2] = rng.uniform(-5, 40, n)
    beh[:, 3] = rng.uniform(-5, 40, n)
    beh[10, 2] = np.nan            # a dropped track sample
    codes, index = app._dec_prob_zone_codes(beh)
    names = {v: k for k, v in index.items()}
    for i in range(n):
        x, y = beh[i, 2], beh[i, 3]
        expected = ('unknown' if (np.isnan(x) or np.isnan(y))
                    else app.classify_zone(x, y))
        assert names.get(int(codes[i]), 'unknown') == expected, i


# -- Binning ------------------------------------------------------------------

def _edges(lo, hi, w):
    """The same edge construction run_decision_probability uses."""
    n = max(1, int(np.ceil((hi - lo) / w - 1e-9)))
    return np.minimum(lo + w * np.arange(n + 1), hi)


def test_bin_probability_is_the_fraction_of_frames_that_entered():
    app = _app()
    sig = np.array([0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4])
    out = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    centers, mean, sem, per_sub, n_fr = app._bin_dec_prob_curves(
        [sig], [out], _edges(0.0, 2.0, 1.0), min_frames=1)
    assert np.allclose(centers, [0.5, 1.5])
    assert np.allclose(mean, [0.25, 0.75])
    assert list(n_fr) == [4, 4]


def test_signal_on_the_range_maximum_is_counted_in_the_top_bin():
    app = _app()
    sig = np.array([0.5, 2.0, 2.0])       # 2.0 sits exactly on the top edge
    out = np.array([0, 1, 1])
    _, mean, _, _, n_fr = app._bin_dec_prob_curves(
        [sig], [out], _edges(0.0, 2.0, 1.0), min_frames=1)
    assert list(n_fr) == [1, 2]
    assert mean[1] == 1.0


def test_out_of_range_frames_are_dropped_unless_edges_are_pooled():
    app = _app()
    sig = np.array([-9.0, 0.5, 9.0])
    out = np.array([1, 0, 1])
    edges = _edges(0.0, 2.0, 1.0)
    _, _, _, _, n_open = app._bin_dec_prob_curves([sig], [out], edges, 1, pool_edges=False)
    assert list(n_open) == [1, 0]
    _, mean_p, _, _, n_pool = app._bin_dec_prob_curves([sig], [out], edges, 1, pool_edges=True)
    assert list(n_pool) == [2, 1]
    assert mean_p[0] == 0.5 and mean_p[1] == 1.0


def test_pooled_edges_with_a_single_bin_keep_every_frame():
    """One bin is both the first and the last; it must not lose the high tail."""
    app = _app()
    sig = np.array([-5.0, 0.5, 5.0])
    out = np.array([1, 1, 1])
    _, mean, _, _, n_fr = app._bin_dec_prob_curves(
        [sig], [out], np.array([0.0, 1.0]), 1, pool_edges=True)
    assert list(n_fr) == [3]
    assert mean[0] == 1.0


def test_sparse_bins_are_dropped_per_subject():
    app = _app()
    sigs = [np.array([0.5, 0.6, 0.7, 1.5]), np.array([0.5, 0.6, 0.7, 1.5])]
    outs = [np.array([1, 1, 1, 0]), np.array([0, 0, 0, 1])]
    _, mean, sem, per_sub, _ = app._bin_dec_prob_curves(
        sigs, outs, _edges(0.0, 2.0, 1.0), min_frames=3)
    assert len(per_sub[0]) == 2          # 3 frames each -> both contribute
    assert per_sub[1] == []              # 1 frame each -> neither does
    assert mean[0] == 0.5 and np.isnan(mean[1])
    assert np.isnan(sem[1])


def test_sem_is_across_subjects_and_zero_for_a_lone_subject():
    app = _app()
    edges = _edges(0.0, 1.0, 1.0)
    sigs = [np.array([0.5, 0.5]), np.array([0.5, 0.5]), np.array([0.5, 0.5])]
    outs = [np.array([1, 1]), np.array([0, 0]), np.array([1, 0])]
    _, mean, sem, _, _ = app._bin_dec_prob_curves(sigs, outs, edges, min_frames=1)
    assert mean[0] == pytest.approx(0.5)
    assert sem[0] == pytest.approx(np.std([1.0, 0.0, 0.5], ddof=1) / np.sqrt(3))
    _, _, sem1, _, _ = app._bin_dec_prob_curves(sigs[:1], outs[:1], edges, 1)
    assert sem1[0] == 0.0


def test_bin_edges_do_not_drift_with_float_arithmetic():
    """np.arange(-3, 3.05, 0.1) is where the top bin used to appear or vanish."""
    for lo, hi, w in [(-3.0, 3.0, 0.1), (-3.0, 3.0, 0.5), (-2.5, 2.5, 0.3),
                      (-1.0, 1.0, 0.07)]:
        e = _edges(lo, hi, w)
        assert e[0] == lo
        assert e[-1] == pytest.approx(hi)
        assert np.all(np.diff(e) > 0)
