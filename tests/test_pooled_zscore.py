"""
Tests for pooled z-scoring: putting an animal's sessions on one common scale.

Z-scoring each session independently leaves an animal's days on different
amplitude scales, so "2 z-scores" means different things on a quiet day and a
noisy one. Pooling z-scores against the mean/SD of all an animal's sessions
together.

The claim these tests exist to hold up is that pooling can be applied as an
affine RESCALE of the z-scores already stored -- numerically identical to
recomputing from the corrected trace, and exactly reversible. That matters
because it must not be recomputed: smoothing rewrites corrected_* in place
before a project is saved, so recomputation would silently derive the moments
from smoothed data.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_pooled', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI

TRAILING = _M.SESSION_PATTERN_PRESETS['SubjectSession   (2FA, 2FB)']


def moments(x):
    return {'n': int(np.size(x)), 'mean': float(np.mean(x)),
            'sd': float(np.std(x, ddof=0))}


class _Pool:
    """Stub exposing the pooling engine; it touches only params/processed_data."""
    _session_pattern = _G._session_pattern
    get_animal_map = _G.get_animal_map
    get_channel_name = _G.get_channel_name
    get_num_channels = _G.get_num_channels
    _channel_wavelength = _G._channel_wavelength
    _combine_by_wavelength = _G._combine_by_wavelength
    compute_pooled_affines = _G.compute_pooled_affines
    apply_affine_to_result = _G.apply_affine_to_result
    _rescale_bout_store = _G._rescale_bout_store
    set_pooled_zscore = _G.set_pooled_zscore

    def __init__(self, processed_data=None, baseline_correct=False, pattern=TRAILING):
        self.params = {'session_pattern': pattern, 'animal_overrides': {},
                       'baseline_correct_bouts': baseline_correct}
        self.processed_data = processed_data or {}
        self.logs = []
        self.log_message = lambda msg='', *a, **k: self.logs.append(str(msg))


def make_subject(corrected, n_channels=1, names=None):
    """A processed-subject dict z-scored exactly the way process_fp_data does."""
    corrected = np.asarray(corrected, dtype=float)
    n = corrected.shape[0]
    arr = np.column_stack([np.arange(n, dtype=float), np.arange(n, dtype=float), corrected])
    z = arr.copy()
    stats = {}
    for ch in range(n_channels):
        col = corrected[:, ch] if corrected.ndim == 2 else corrected
        st = moments(col)
        stats[f'470_{ch}'] = st
        z[:, 2 + ch] = (col - st['mean']) / st['sd']
    return {'corrected_470': arr, 'zscore_470': z, 'zscore_stats': stats,
            'channel_names': names or [f'G{i}' for i in range(n_channels)]}


# ---------------------------------------------------------------------------
# pool_moments -- must equal the moments of the concatenation
# ---------------------------------------------------------------------------

def test_pooled_moments_equal_the_concatenations_moments():
    rng = np.random.default_rng(0)
    a = rng.normal(3.0, 2.0, 900)
    b = rng.normal(-1.0, 5.0, 1500)
    pooled = _M.pool_moments([moments(a), moments(b)])
    cat = np.concatenate([a, b])
    assert pooled['n'] == a.size + b.size
    assert pooled['mean'] == pytest.approx(np.mean(cat))
    assert pooled['sd'] == pytest.approx(np.std(cat, ddof=0))


def test_a_short_session_is_weighted_less_than_a_long_one():
    """n-weighting is the point: a 10-sample day must not move the pooled mean
    as much as a 10000-sample day."""
    short = {'n': 10, 'mean': 100.0, 'sd': 1.0}
    long = {'n': 10000, 'mean': 0.0, 'sd': 1.0}
    assert _M.pool_moments([short, long])['mean'] == pytest.approx(0.0999, abs=1e-3)


def test_degenerate_sessions_are_refused_rather_than_dividing_by_zero():
    assert _M.pool_moments([]) is None
    assert _M.pool_moments([{'n': 0, 'mean': 0.0, 'sd': 1.0}]) is None
    assert _M.pool_moments([{'n': 50, 'mean': 1.0, 'sd': 0.0}]) is None


def test_identical_sessions_pool_to_themselves():
    st = {'n': 500, 'mean': 2.0, 'sd': 3.0}
    pooled = _M.pool_moments([st, st])
    assert pooled['mean'] == pytest.approx(2.0)
    assert pooled['sd'] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# affine_for / invert_affine
# ---------------------------------------------------------------------------

def test_the_affine_maps_session_z_scores_onto_the_pooled_scale():
    rng = np.random.default_rng(1)
    x = rng.normal(4.0, 2.0, 400)
    y = rng.normal(-2.0, 6.0, 400)
    sx, sy = moments(x), moments(y)
    pooled = _M.pool_moments([sx, sy])

    a, b = _M.affine_for(sx, pooled)
    z_session = (x - sx['mean']) / sx['sd']
    np.testing.assert_allclose(z_session * a + b, (x - pooled['mean']) / pooled['sd'],
                               rtol=0, atol=1e-12)


def test_inverting_the_affine_round_trips():
    ab = _M.affine_for({'n': 10, 'mean': 1.0, 'sd': 2.0},
                       {'n': 20, 'mean': 0.5, 'sd': 3.0})
    ia = _M.invert_affine(ab)
    z = np.linspace(-4, 4, 101)
    np.testing.assert_allclose((z * ab[0] + ab[1]) * ia[0] + ia[1], z, atol=1e-12)


def test_a_flat_channel_yields_no_affine():
    assert _M.affine_for({'n': 5, 'mean': 0.0, 'sd': 0.0},
                         {'n': 10, 'mean': 0.0, 'sd': 1.0}) is None
    assert _M.affine_for({'n': 5, 'mean': 0.0, 'sd': 1.0}, None) is None


# ---------------------------------------------------------------------------
# Applying it to a subject -- the property that makes this safe
# ---------------------------------------------------------------------------

def test_rescaling_matches_recomputing_from_the_corrected_trace():
    """The whole design rests on this: rescaling the stored z-scores gives the
    same answer as z-scoring the concatenated corrected traces directly."""
    rng = np.random.default_rng(2)
    xa = rng.normal(3.0, 1.5, 700)
    xb = rng.normal(-1.0, 4.0, 900)
    app = _Pool({'2FA': make_subject(xa[:, None]), '2FB': make_subject(xb[:, None])})

    changed, _ = app.set_pooled_zscore(True)
    assert sorted(changed) == ['2FA', '2FB']

    pooled = _M.pool_moments([moments(xa), moments(xb)])
    for name, raw in (('2FA', xa), ('2FB', xb)):
        expected = (raw - pooled['mean']) / pooled['sd']
        np.testing.assert_allclose(app.processed_data[name]['zscore_470'][:, 2],
                                   expected, atol=1e-12)


def test_the_pooled_result_has_mean_zero_and_sd_one_across_sessions():
    rng = np.random.default_rng(3)
    xa, xb = rng.normal(3.0, 1.5, 700), rng.normal(-1.0, 4.0, 900)
    app = _Pool({'2FA': make_subject(xa[:, None]), '2FB': make_subject(xb[:, None])})
    app.set_pooled_zscore(True)

    both = np.concatenate([app.processed_data['2FA']['zscore_470'][:, 2],
                           app.processed_data['2FB']['zscore_470'][:, 2]])
    assert np.mean(both) == pytest.approx(0.0, abs=1e-12)
    assert np.std(both) == pytest.approx(1.0, abs=1e-12)
    # and the per-session SDs are no longer pinned to 1
    assert np.std(app.processed_data['2FA']['zscore_470'][:, 2]) != pytest.approx(1.0, abs=1e-3)


def test_apply_then_revert_round_trips_to_the_original_arrays():
    rng = np.random.default_rng(4)
    xa, xb = rng.normal(3.0, 1.5, 500), rng.normal(-1.0, 4.0, 600)
    app = _Pool({'2FA': make_subject(xa[:, None]), '2FB': make_subject(xb[:, None])})
    before = {s: d['zscore_470'].copy() for s, d in app.processed_data.items()}

    app.set_pooled_zscore(True)
    assert all(d['zscore_pooled'] for d in app.processed_data.values())
    app.set_pooled_zscore(False)

    for name, original in before.items():
        np.testing.assert_allclose(app.processed_data[name]['zscore_470'], original, atol=1e-12)
        assert not app.processed_data[name].get('zscore_pooled')
        assert 'zscore_pooling_applied' not in app.processed_data[name]


def test_applying_twice_does_not_double_scale():
    rng = np.random.default_rng(5)
    xa, xb = rng.normal(3.0, 1.5, 400), rng.normal(-1.0, 4.0, 400)
    app = _Pool({'2FA': make_subject(xa[:, None]), '2FB': make_subject(xb[:, None])})
    app.set_pooled_zscore(True)
    once = app.processed_data['2FA']['zscore_470'].copy()

    changed, _ = app.set_pooled_zscore(True)
    assert changed == []
    np.testing.assert_allclose(app.processed_data['2FA']['zscore_470'], once, atol=1e-15)


# ---------------------------------------------------------------------------
# What gets rescaled alongside the z-scores
# ---------------------------------------------------------------------------

def test_behaviour_synced_photometry_columns_follow_the_z_scores():
    rng = np.random.default_rng(6)
    xa, xb = rng.normal(2.0, 1.0, 300), rng.normal(-2.0, 3.0, 300)
    a, b = make_subject(xa[:, None]), make_subject(xb[:, None])
    # beh_synced: 6 metadata columns, then one photometry column per channel.
    for subj, raw in ((a, xa), (b, xb)):
        beh = np.zeros((300, 8))
        beh[:, 6] = subj['zscore_470'][:, 2]
        subj['beh_synced'] = beh

    app = _Pool({'2FA': a, '2FB': b})
    app.set_pooled_zscore(True)
    for subj in (a, b):
        np.testing.assert_allclose(subj['beh_synced'][:, 6], subj['zscore_470'][:, 2], atol=1e-12)


def test_baseline_corrected_bouts_take_the_scale_but_not_the_offset():
    """Baseline correction already subtracts a per-bout mean, so the affine
    offset cancels -- applying it would shift every bout off its own baseline."""
    rng = np.random.default_rng(7)
    xa, xb = rng.normal(2.0, 1.0, 300), rng.normal(-2.0, 3.0, 300)
    a, b = make_subject(xa[:, None]), make_subject(xb[:, None])
    trace = np.array([0.0, 1.0, 2.0, 1.0])
    for subj in (a, b):
        subj['bouts'] = {'sniff': {'_prebout': 30, '_postbout': 30, 'G0': [trace.copy()]}}

    app = _Pool({'2FA': a, '2FB': b}, baseline_correct=True)
    app.set_pooled_zscore(True)
    scale = app.processed_data['2FA']['zscore_pooling_applied']['470_0'][0]
    np.testing.assert_allclose(a['bouts']['sniff']['G0'][0], trace * scale, atol=1e-12)


def test_uncorrected_bouts_take_the_full_transform():
    rng = np.random.default_rng(8)
    xa, xb = rng.normal(2.0, 1.0, 300), rng.normal(-2.0, 3.0, 300)
    a, b = make_subject(xa[:, None]), make_subject(xb[:, None])
    trace = np.array([0.0, 1.0, 2.0, 1.0])
    for subj in (a, b):
        subj['bouts'] = {'sniff': {'G0': [trace.copy()]}}

    app = _Pool({'2FA': a, '2FB': b}, baseline_correct=False)
    app.set_pooled_zscore(True)
    scale, offset = app.processed_data['2FA']['zscore_pooling_applied']['470_0']
    np.testing.assert_allclose(a['bouts']['sniff']['G0'][0], trace * scale + offset, atol=1e-12)


def test_bout_window_metadata_is_not_treated_as_a_channel():
    rng = np.random.default_rng(9)
    xa, xb = rng.normal(2.0, 1.0, 300), rng.normal(-2.0, 3.0, 300)
    a, b = make_subject(xa[:, None]), make_subject(xb[:, None])
    for subj in (a, b):
        subj['bouts'] = {'sniff': {'_prebout': 90, '_postbout': 90,
                                   'G0': [np.array([1.0, 2.0])]}}
    app = _Pool({'2FA': a, '2FB': b})
    app.set_pooled_zscore(True)
    assert a['bouts']['sniff']['_prebout'] == 90
    assert a['bouts']['sniff']['_postbout'] == 90


def test_nested_entry_bouts_are_rescaled_too():
    rng = np.random.default_rng(10)
    xa, xb = rng.normal(2.0, 1.0, 300), rng.normal(-2.0, 3.0, 300)
    a, b = make_subject(xa[:, None]), make_subject(xb[:, None])
    for subj in (a, b):
        subj['entry_bouts'] = {'zone_entries': {'center': {'G0': [np.array([1.0, 2.0])]}}}

    app = _Pool({'2FA': a, '2FB': b}, baseline_correct=False)
    app.set_pooled_zscore(True)
    scale, offset = app.processed_data['2FA']['zscore_pooling_applied']['470_0']
    np.testing.assert_allclose(a['entry_bouts']['zone_entries']['center']['G0'][0],
                               np.array([1.0, 2.0]) * scale + offset, atol=1e-12)


def test_a_red_channel_is_pooled_against_its_own_wavelength():
    """Channel names decide the wavelength, so an R channel must not be rescaled
    by the 470 nm transform."""
    rng = np.random.default_rng(11)
    subjects = {}
    raw = {}
    for name, mean, sd in (('2FA', 1.0, 2.0), ('2FB', -3.0, 5.0)):
        x = rng.normal(mean, sd, 400)
        raw[name] = x
        st = moments(x)
        arr = np.column_stack([np.arange(400.0), np.arange(400.0), x])
        z = arr.copy()
        z[:, 2] = (x - st['mean']) / st['sd']
        subjects[name] = {'corrected_570': arr, 'zscore_570': z,
                          'zscore_stats': {'570_0': st}, 'channel_names': ['R0']}

    app = _Pool(subjects)
    app.set_pooled_zscore(True)
    pooled = _M.pool_moments([moments(raw['2FA']), moments(raw['2FB'])])
    np.testing.assert_allclose(subjects['2FA']['zscore_570'][:, 2],
                               (raw['2FA'] - pooled['mean']) / pooled['sd'], atol=1e-12)


# ---------------------------------------------------------------------------
# Who gets skipped, and why
# ---------------------------------------------------------------------------

def test_an_animal_with_one_session_is_left_alone():
    app = _Pool({'5MA': make_subject(np.random.default_rng(12).normal(0, 1, 200)[:, None])})
    changed, messages = app.set_pooled_zscore(True)
    assert changed == []
    assert any('only session' in m for m in messages)


def test_subjects_without_stored_moments_are_skipped_with_a_reason():
    """Projects processed before the moments were recorded cannot be pooled;
    they must say so rather than silently doing nothing."""
    rng = np.random.default_rng(13)
    a = make_subject(rng.normal(2.0, 1.0, 300)[:, None])
    b = make_subject(rng.normal(-2.0, 3.0, 300)[:, None])
    del b['zscore_stats']
    app = _Pool({'2FA': a, '2FB': b})

    changed, messages = app.set_pooled_zscore(True)
    assert changed == []
    assert any('reprocess' in m for m in messages)


def test_reverting_when_nothing_was_applied_is_a_no_op():
    rng = np.random.default_rng(14)
    app = _Pool({'2FA': make_subject(rng.normal(0, 1, 200)[:, None]),
                 '2FB': make_subject(rng.normal(0, 2, 200)[:, None])})
    before = app.processed_data['2FA']['zscore_470'].copy()
    changed, _ = app.set_pooled_zscore(False)
    assert changed == []
    np.testing.assert_array_equal(app.processed_data['2FA']['zscore_470'], before)


def test_unrelated_animals_are_untouched_while_a_pair_is_pooled():
    rng = np.random.default_rng(15)
    app = _Pool({'2FA': make_subject(rng.normal(2.0, 1.0, 300)[:, None]),
                 '2FB': make_subject(rng.normal(-2.0, 3.0, 300)[:, None]),
                 '9ZA': make_subject(rng.normal(0.0, 1.0, 300)[:, None])})
    lone_before = app.processed_data['9ZA']['zscore_470'].copy()

    changed, _ = app.set_pooled_zscore(True)
    assert sorted(changed) == ['2FA', '2FB']
    np.testing.assert_array_equal(app.processed_data['9ZA']['zscore_470'], lone_before)
