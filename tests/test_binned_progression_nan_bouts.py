"""A padded bout must not delete its whole group from the progression plot.

Bouts clipped at the recording edge (or extracted with a shorter window) come
back from realignment padded out to the full axis with NaN.  The binned
progression view scored each bout with plain ``np.max``/``np.mean``/trapz, so
one padded sample inside the analysis window made that bout NaN, the bin mean
NaN, and -- because every bin of that subject went the same way -- the subject
contributed nothing.  A cohort where one sex happened to carry such bouts drew
only the other sex's lines, while the metrics table (which reads the raw,
unpadded traces) listed both.
"""
import numpy as np
import pytest

import fp_analysis_gui as G

FPS = 10.0
PRE = POST = 10
TOTAL = PRE + POST
N_BOUTS = 4


def _app(nan_tail_for):
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': PRE, 'postboutframes': POST,
                  'preboutseconds': PRE / FPS, 'postboutseconds': POST / FPS}
    app.analysis_fps = lambda: FPS
    app.processed_data = {}
    for subject, level in (('f1', 1.0), ('f2', 1.0), ('m1', 2.0), ('m2', 2.0)):
        traces = []
        for _ in range(N_BOUTS):
            tr = np.full(TOTAL, level, dtype=float)
            if subject in nan_tail_for:
                tr[-3:] = np.nan          # what realignment leaves behind
            traces.append(tr)
        app.processed_data[subject] = {
            'photometry_fps': FPS,
            'bouts': {'Lick': {'G0': traces, '_prebout': PRE,
                               '_postbout': POST, '_fs': FPS}},
        }
    return app


SEX = {'f1': 'Female', 'f2': 'Female', 'm1': 'Male', 'm2': 'Male'}


def _progression(app):
    return G.FPAnalysisGUI._compute_binned_progression(
        app, 'Lick', 'G0', list(SEX), SEX,
        bin_size=2, max_bouts=None,
        win_start_samples=0, win_end_samples=POST,   # reaches into the padding
        fps=FPS, pre_frames=PRE, group_mode=True)


def _means(prog, group, metric):
    return [np.mean(prog['group_series'][group][b][metric])
            for b in prog['sorted_bins']]


def test_padded_group_still_has_values():
    prog = _progression(_app({'m1', 'm2'}))
    assert prog['groups_present'] == ['Female', 'Male']
    for metric in ('peak', 'avg', 'auc'):
        assert all(np.isfinite(v) for v in _means(prog, 'Male', metric)), metric
        assert all(np.isfinite(v) for v in _means(prog, 'Female', metric)), metric
    # The padded group is scored on the samples it actually has.
    assert _means(prog, 'Male', 'peak') == pytest.approx([2.0, 2.0])
    assert _means(prog, 'Male', 'avg') == pytest.approx([2.0, 2.0])


def test_clean_bouts_are_scored_exactly_as_before():
    prog = _progression(_app(set()))
    assert _means(prog, 'Female', 'auc') == pytest.approx([0.9, 0.9])
    assert _means(prog, 'Male', 'auc') == pytest.approx([1.8, 1.8])


def test_padded_auc_spans_the_samples_that_exist():
    prog = _progression(_app({'m1', 'm2'}))
    # 7 finite samples at 10 Hz -> 0.6 s of trace at level 2.0.
    assert _means(prog, 'Male', 'auc') == pytest.approx([1.2, 1.2])


def test_all_nan_window_drops_the_bout_not_the_subject():
    app = _app(set())
    traces = app.processed_data['m1']['bouts']['Lick']['G0']
    traces[0] = np.full(TOTAL, np.nan)      # one unusable bout in bin 0
    prog = _progression(app)
    assert 'Male' in prog['groups_present']
    assert all(np.isfinite(v) for v in _means(prog, 'Male', 'peak'))


def test_unplaced_subject_gets_its_own_series():
    """Never folded into whichever group sorts first."""
    partial = {k: v for k, v in SEX.items() if k != 'm2'}
    prog = G.FPAnalysisGUI._compute_binned_progression(
        _app(set()), 'Lick', 'G0', list(SEX), partial,
        bin_size=2, max_bouts=None,
        win_start_samples=0, win_end_samples=POST,
        fps=FPS, pre_frames=PRE, group_mode=True)
    assert 'Unassigned' in prog['groups_present']
    assert _means(prog, 'Female', 'peak') == pytest.approx([1.0, 1.0])
