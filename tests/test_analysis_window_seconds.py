"""The Bout Analysis metric window is set in seconds, not frames.

A frame count is a different duration at every sampling rate, so a project that
mixes rigs measured a different span per subject from the same setting.  These
tests pin the seconds -> samples conversion and that each consumer converts at
the rate of the axis it is about to slice.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


class _Var:
    """Stand-in for a tk.StringVar, so these run without a display."""

    def __init__(self, value):
        self._v = str(value)

    def get(self):
        return self._v

    def set(self, value):
        self._v = str(value)


def _app(start='0', end='5', rates=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'preboutframes': 60, 'postboutframes': 60,
                  'preboutseconds': 3.0, 'postboutseconds': 3.0,
                  'baseline_seconds': 0.0,
                  'analysis_window_start_sec': 0.0,
                  'analysis_window_end_sec': 5.0,
                  'boutframe_processing_style': 'onset'}
    app.processed_data = {
        subj: {'photometry_fps': fs} for subj, fs in (rates or {}).items()}
    app._mixed_fps_warned = True
    app.window_start_sec_var = _Var(start)
    app.window_end_sec_var = _Var(end)
    return app


def test_seconds_become_samples_at_the_shared_axis_rate():
    app = _app(start='-1', end='2', rates={'A': 20.0})
    assert app.analysis_window_samples() == (-20, 40)


def test_a_subject_gets_its_own_rate():
    """The same window, measured on two rigs, is the same DURATION."""
    app = _app(start='0', end='2', rates={'A': 20.0, 'B': 30.0})
    assert app.analysis_window_samples('A') == (0, 40)
    assert app.analysis_window_samples('B') == (0, 60)


def test_the_window_is_mirrored_into_params():
    app = _app(start='0.5', end='4')
    app.analysis_window_seconds()
    assert app.params['analysis_window_start_sec'] == 0.5
    assert app.params['analysis_window_end_sec'] == 4.0


def test_unparseable_entries_are_refused():
    app = _app(start='abc', end='5')
    with pytest.raises(ValueError):
        app.analysis_window_seconds()


def test_an_empty_window_is_refused():
    app = _app(start='3', end='3')
    with pytest.raises(ValueError):
        app.analysis_window_seconds()


def test_onset_slice_spans_the_requested_seconds():
    """1 s of a 20 Hz trace is 20 samples, taken from the onset onwards."""
    app = _app(start='0', end='1', rates={'A': 20.0})
    app.processed_data['A'].update({
        'bouts': {'Lick': {'G0': [], '_prebout': 60, '_fs': 20.0}}})
    trace = np.arange(120, dtype=float)          # onset sits at index 60
    seg = app._bout_analysis_segment('A', 'Lick', 'G0', 0, trace, 0.0, 1.0)
    assert seg is not None
    assert len(seg) == 20
    assert seg[0] == pytest.approx(60.0)
