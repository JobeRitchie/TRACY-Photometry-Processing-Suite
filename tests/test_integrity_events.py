"""
Tests for the event-locked term of the Signal Integrity score.

Users saw channels with a clear response at every bout labelled "Bad" because
the score judged only the whole trace. The event term measures the response at
the bouts against the same onsets shifted in time, and may raise a score but
never lower it; a response the 415 trace shares is movement and earns nothing.
"""

import importlib.util
import os
import sys

import numpy as np

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_events', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI
FS = 30.0
N = 30000


def _gui():
    g = _G.__new__(_G)
    g.params = {'preboutframes': 150, 'postboutframes': 300}
    g.bout_offsets = []
    g.processed_data = {}
    g.log_message = lambda *a, **k: None
    return g


def _onsets(n=12, seed=1):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(np.arange(600, N - 600, 900), size=n, replace=False))


def _kernel():
    t = np.arange(int(4 * FS)) / FS
    return 2.0 * np.exp(-t / 1.0) * (1 - np.exp(-t / 0.2))


def _trace(onsets=(), seed=0):
    x = np.random.default_rng(seed).normal(0, 1, N)
    x = np.convolve(x, np.ones(5) / 5, mode='same') * 2   # some slow structure
    k = _kernel()
    for o in onsets:
        x[o:o + len(k)] += k
    return x


def _subject(sig, iso=None, onsets=(), store_onsets=True):
    """A minimal processed subject: one channel, beh_synced and one behaviour."""
    beh = np.full((N, 6 + 1 + 5), np.nan)
    beh[:, 0] = np.arange(N)
    beh[:, 1] = np.arange(N) * (1000.0 / FS)
    beh[:, 6] = sig
    entry = {'_prebout': 150, '_postbout': 300, '_fs': FS}
    traces = [sig[o - 150:o + 300] for o in onsets]
    entry['Ch0'] = traces
    entry['G0'] = traces
    if store_onsets:
        entry['onset_frames'] = [float(o) for o in onsets]
        entry['_kept_indices'] = {'Ch0': list(range(len(onsets))), 'G0': list(range(len(onsets)))}
    return {'beh_synced': beh, 'bouts': {'Approach': entry}, 'photometry_fps': FS}


# ---------------------------------------------------------------------------
# The test statistic
# ---------------------------------------------------------------------------

def test_a_planted_response_is_detected():
    on = _onsets()
    r = _G.integrity_event_test(_trace(on), on, 150, 300, FS)
    assert r['n'] == len(on)
    assert r['z'] > 5


def test_onsets_with_no_response_stay_near_the_null():
    on = _onsets()
    zs = [_G.integrity_event_test(_trace(seed=s), on, 150, 300, FS, seed=s)['z']
          for s in range(8)]
    assert max(zs) < 3


def test_the_null_is_reproducible():
    on = _onsets()
    x = _trace(on)
    a = _G.integrity_event_test(x, on, 150, 300, FS, seed=4)['z']
    b = _G.integrity_event_test(x, on, 150, 300, FS, seed=4)['z']
    assert a == b


def test_onsets_too_near_the_edge_are_dropped():
    on = np.array([10, 5000, 9000, 15000, N - 20])
    r = _G.integrity_event_test(_trace(), on, 150, 300, FS)
    assert r['n'] == 3


# ---------------------------------------------------------------------------
# Onsets for projects saved before they were stored
# ---------------------------------------------------------------------------

def test_onsets_are_recovered_from_stored_traces():
    on = _onsets()
    x = _trace(on)
    traces = [x[o - 150:o + 300] for o in on]
    assert list(_G._recover_bout_onsets(x, traces, 150)) == list(on)


def test_baseline_corrected_traces_are_still_found():
    on = _onsets()
    x = _trace(on)
    traces = [x[o - 150:o + 300] - x[o - 150:o].mean() for o in on]
    assert list(_G._recover_bout_onsets(x, traces, 150)) == list(on)


def test_evidence_uses_recovered_onsets_when_none_were_stored():
    on = _onsets()
    x = _trace(on)
    ev = _gui().integrity_event_evidence(_subject(x, onsets=on, store_onsets=False), 0, 'G0')
    assert ev is not None and ev['n_bouts'] == len(on) and ev['responsive']


# ---------------------------------------------------------------------------
# Evidence across a subject's behaviours
# ---------------------------------------------------------------------------

def test_a_responsive_channel_is_credited():
    on = _onsets()
    ev = _gui().integrity_event_evidence(_subject(_trace(on), onsets=on), 0, 'G0')
    assert ev['responsive'] and not ev['motion_locked']
    assert ev['behavior'] == 'Approach'


def test_a_stronger_415_response_is_movement_and_not_credited():
    on = _onsets()
    sig = _trace(on)
    iso = _trace(on, seed=9) + _trace(on, seed=9) - _trace(seed=9)   # twice the bump
    data = _subject(sig, onsets=on)
    ev = _gui().integrity_event_evidence(data, 0, 'G0', iso_flat=iso,
                                         fp_ts=data['beh_synced'][:, 1])
    assert ev['motion_locked'] and not ev['responsive']


def test_a_weaker_415_response_is_biology_and_still_credited():
    on = _onsets()
    k = _kernel()
    iso = _trace(seed=9)
    for o in on:
        iso[o:o + len(k)] += 0.6 * k      # 415 moves too, but less
    data = _subject(_trace(on), onsets=on)
    ev = _gui().integrity_event_evidence(data, 0, 'G0', iso_flat=iso,
                                         fp_ts=data['beh_synced'][:, 1], r415=0.3)
    assert ev['iso_z'] >= 3
    assert ev['responsive'] and not ev['motion_locked']


def test_on_a_tangled_channel_any_415_response_blocks_credit():
    on = _onsets()
    k = _kernel()
    iso = _trace(seed=9)
    for o in on:
        iso[o:o + len(k)] += 0.6 * k
    data = _subject(_trace(on), onsets=on)
    ev = _gui().integrity_event_evidence(data, 0, 'G0', iso_flat=iso,
                                         fp_ts=data['beh_synced'][:, 1], r415=0.97)
    assert ev['motion_locked'] and not ev['responsive']


# ---------------------------------------------------------------------------
# Sustained shifts that a behaviour causes
# ---------------------------------------------------------------------------

def test_a_shift_starting_at_a_bout_is_not_an_artifact():
    ts = np.arange(N) * 10.0
    shifts = [(3000, 4000, 1.0), (20000, 21000, 1.0)]
    rest, n = _gui()._split_event_locked_shifts(shifts, [ts[2900]], ts, FS)
    assert n == 1 and rest == [(20000, 21000, 1.0)]


def test_without_bouts_every_shift_still_counts():
    shifts = [(3000, 4000, 1.0)]
    assert _gui()._split_event_locked_shifts(shifts, None, None, FS) == (shifts, 0)


def test_a_shift_long_after_a_bout_still_counts():
    ts = np.arange(N) * 10.0
    shifts = [(3000, 4000, 1.0)]
    rest, n = _gui()._split_event_locked_shifts(shifts, [ts[3000 - int(40 * FS)]], ts, FS)
    assert n == 0 and rest == shifts


def test_a_quiet_415_does_not_block_credit():
    on = _onsets()
    data = _subject(_trace(on), onsets=on)
    ev = _gui().integrity_event_evidence(data, 0, 'G0', iso_flat=_trace(seed=9),
                                         fp_ts=data['beh_synced'][:, 1])
    assert ev['responsive'] and not ev['motion_locked']


def test_fewer_than_three_bouts_is_not_assessed():
    on = _onsets(n=2)
    assert _gui().integrity_event_evidence(_subject(_trace(on), onsets=on), 0, 'G0') is None


def test_no_bouts_is_not_assessed():
    data = _subject(_trace())
    data['bouts'] = {}
    assert _gui().integrity_event_evidence(data, 0, 'G0') is None


def test_offset_copies_of_a_behaviour_are_not_tested_again():
    on = _onsets()
    data = _subject(_trace(on), onsets=on)
    data['bouts']['Approach_late'] = dict(data['bouts']['Approach'])
    g = _gui()
    g.bout_offsets = [{'label': 'late', 'offset_frames': 30}]
    ev = g.integrity_event_evidence(data, 0, 'G0')
    assert ev['n_behaviors'] == 1


def test_more_behaviours_tested_means_a_stricter_bar():
    on = _onsets()
    data = _subject(_trace(on), onsets=on)
    one = _gui().integrity_event_evidence(data, 0, 'G0')
    for i in range(5):
        data['bouts'][f'Other{i}'] = dict(data['bouts']['Approach'],
                                          onset_frames=[float(o) for o in _onsets(seed=20 + i)])
    many = _gui().integrity_event_evidence(data, 0, 'G0')
    assert many['n_behaviors'] == 6
    assert many['z_adj'] < one['z_adj'] or one['z_adj'] >= 7


# ---------------------------------------------------------------------------
# Combining with the recording score
# ---------------------------------------------------------------------------

def test_no_evidence_leaves_the_score_alone():
    assert _G.combine_integrity_with_events(31.0, None) == (31.0, False)
    assert _G.combine_integrity_with_events(31.0, {'responsive': False, 'z_adj': 9}) == (31.0, False)


def test_a_response_at_the_bar_meets_the_default_threshold():
    score, raised = _G.combine_integrity_with_events(20.0, {'responsive': True, 'z_adj': 3.0})
    assert raised and score == 70.0


def test_a_strong_response_raises_to_ninety():
    score, raised = _G.combine_integrity_with_events(20.0, {'responsive': True, 'z_adj': 12.0})
    assert raised and score == 90.0


def test_events_never_lower_a_score():
    score, raised = _G.combine_integrity_with_events(97.0, {'responsive': True, 'z_adj': 3.5})
    assert score == 97.0 and not raised


# ---------------------------------------------------------------------------
# Rigs with no 415 nm channel
# ---------------------------------------------------------------------------

def test_a_missing_isosbestic_is_not_scored_as_a_tangle():
    t = np.arange(N) / FS
    sig = 1.0 + 0.2 * np.exp(-t / 300) + 0.01 * _trace(_onsets())
    m = _gui().compute_channel_integrity(sig, sig.copy(), t, FS, sensor=False)
    assert m['no_isosbestic'] and m['sensor_mode']
    assert m['score_components']['motion'] is None
    assert m['_iso_flat'] is None


def test_a_real_isosbestic_is_still_used():
    t = np.arange(N) / FS
    sig = 1.0 + 0.2 * np.exp(-t / 300) + 0.01 * _trace(_onsets())
    iso = 1.0 + 0.2 * np.exp(-t / 300) + 0.01 * _trace(seed=5)
    m = _gui().compute_channel_integrity(sig, iso, t, FS, sensor=False)
    assert not m['no_isosbestic'] and m['score_components']['motion'] is not None
    assert m['_iso_flat'] is not None


# ---------------------------------------------------------------------------
# Sparse fast transients (sharp GCaMP spikes back at baseline within ~1-2 s)
# ---------------------------------------------------------------------------

def _spiky(n_spikes, seed=0, amp=6.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, N)                     # white noise floor
    t = np.arange(int(1.5 * FS)) / FS
    k = amp * np.exp(-t / 0.3) * (1 - np.exp(-t / 0.05))
    for o in rng.choice(np.arange(100, N - 100, 120), size=n_spikes, replace=False):
        x[o:o + len(k)] += k
    return x


def test_noise_has_symmetric_transients():
    st = _G.integrity_series_stats(_gui(), np.random.default_rng(3).normal(0, 1, N), FS)
    # too few crossings either way to earn the transient credit
    assert st['transients_up'] < 15


def test_sparse_fast_spikes_are_seen_as_upward_transients():
    st = _G.integrity_series_stats(_gui(), _spiky(40), FS)
    assert st['transients_up'] >= 30 and st['transient_asym'] >= 10


def test_sparse_spikes_lift_dynamics_and_noise():
    g = _gui()
    noise_only = g.calculate_signal_quality_score(
        _G.integrity_series_stats(g, np.random.default_rng(3).normal(0, 1, N), FS), sensor=True)
    spiky = g.calculate_signal_quality_score(_G.integrity_series_stats(g, _spiky(40), FS), sensor=True)
    assert spiky[1]['dynamics'] >= 90 and spiky[1]['noise'] >= 90
    assert spiky[0] > 70 > noise_only[0]


def test_a_handful_of_spikes_is_not_enough():
    st = _G.integrity_series_stats(_gui(), _spiky(6), FS)
    base = _G.integrity_series_stats(_gui(), np.random.default_rng(0).normal(0, 1, N), FS)
    assert _gui().calculate_signal_quality_score(st, sensor=True)[0] < 50
    assert _gui().calculate_signal_quality_score(base, sensor=True)[0] < 50
