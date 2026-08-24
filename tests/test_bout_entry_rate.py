"""A bout entry has to remember the rate its traces are spaced at.

In a mixed-rate project ``_entry_fs`` falls back to ``analysis_fps()`` -- the
FASTEST rate present -- when the entry carries no ``_fs``.  A 20 Hz subject's
5 s post-bout window is 100 samples; read as 30 Hz that is 3.3 s, so a 5 s
analysis window runs off the end of the trace into NaN padding.  ``np.max`` over
that returns NaN, and one NaN poisons the mean for that bout number across every
subject -- which emptied "Compare Across Bouts" while every per-subject plot
still looked fine.
"""
import types
import numpy as np
import pytest

import fp_analysis_gui as G

SLOW, FAST = 20.0, 30.0


def _app():
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.root = types.SimpleNamespace(update_idletasks=lambda: None)
    app.params = {'preboutseconds': 5.0, 'postboutseconds': 5.0,
                  'preboutframes': int(5 * FAST), 'postboutframes': int(5 * FAST),
                  'boutframe_processing_style': 'onset',
                  'baseline_correct_bouts': False}
    # One slow and one fast subject, so analysis_fps() is the fast rate.
    trace = np.arange(int(10 * SLOW), dtype=float)
    app.processed_data = {
        'slow': {'photometry_fps': SLOW,
                 'bouts': {'B': {'Ch0': [trace], '_prebout': int(5 * SLOW),
                                 '_postbout': int(5 * SLOW), '_fs': SLOW}}},
        'fast': {'photometry_fps': FAST},
    }
    return app


def test_analysis_fps_is_the_fastest_rate_in_the_project():
    assert _app().analysis_fps() == FAST


def test_a_stored_rate_stretches_the_slow_trace_onto_the_full_window():
    app = _app()
    entry = app.processed_data['slow']['bouts']['B']
    seg = app._bout_analysis_segment('slow', 'B', 'Ch0', 0, entry['Ch0'][0], 0.0, 5.0)
    assert seg is not None
    # Nearly 5 s at the axis rate: resampling 20 Hz onto a 30 Hz grid cannot
    # produce the sample at exactly the window edge, so one is missing.
    assert int(5 * FAST) - 2 <= len(seg) <= int(5 * FAST)


def test_the_segment_never_carries_nan_into_a_metric():
    """Every caller reduces this array, and one NaN turns the metric into NaN.

    A NaN in one subject then poisons the pooled mean for every other, which is
    what emptied the whole Bout Analysis tab for a slow-rate-only selection.
    """
    for drop_fs in (False, True):
        app = _app()
        entry = app.processed_data['slow']['bouts']['B']
        if drop_fs:
            del entry['_fs']              # the worse case: window overruns too
        seg = app._bout_analysis_segment('slow', 'B', 'Ch0', 0, entry['Ch0'][0], 0.0, 5.0)
        assert seg is not None and len(seg)
        assert not np.isnan(seg).any()
        assert np.isfinite(np.max(seg)) and np.isfinite(np.mean(seg))


def test_an_all_nan_window_returns_none_rather_than_an_empty_array():
    app = _app()
    entry = app.processed_data['slow']['bouts']['B']
    entry['Ch0'] = [np.full(int(10 * SLOW), np.nan)]
    assert app._bout_analysis_segment('slow', 'B', 'Ch0', 0, entry['Ch0'][0],
                                      0.0, 5.0) is None
