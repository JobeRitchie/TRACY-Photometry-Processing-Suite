"""
Tests for the multi-subject Signal Integrity dashboard: the two things that
made it unusable on a 62-subject selection.

1. Scoring a channel costs about a second of real work, and every re-plot paid
   for all of it again -- a minute of a frozen window.
2. The table lays its rows out in axes fractions inside a fixed-height figure,
   so past about 40 rows they overlapped into an unreadable smear.
"""

import importlib.util
import os
import sys

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_integrity', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI


class _Integrity:
    """The Tk-free slice of the integrity path."""

    signal_integrity_metrics = _G.signal_integrity_metrics
    invalidate_integrity_caches = _G.invalidate_integrity_caches
    _integrity_row_estimate = _G._integrity_row_estimate
    _auto_viz_figure_size = _G._auto_viz_figure_size
    _integrity_progress = _G._integrity_progress

    def __init__(self, subjects):
        self.processed_data = dict(subjects)
        self._integrity_metrics_cache = {}
        self._integrity_plot_cache = None
        self._integrity_busy = False
        self.root = None
        self.calls = []
        self.log_message = lambda *a, **k: None

    # Stands in for the ~1s of real scoring.
    def _extract_signal_integrity_metrics(self, data, sensor=None):
        self.calls.append((id(data), sensor))
        if not (data.get('has_470') or data.get('has_570')):
            return None
        return [{'channel': f'G{i}', 'overall_score': 90.0}
                for i in range(data.get('num_photometry_channels', 1))]


def _subject(channels=2, has_470=True):
    return {'num_photometry_channels': channels, 'has_470': has_470,
            'has_570': False}


def _project(n, channels=2):
    return {f'S{i:02d}': _subject(channels) for i in range(n)}


# ---------------------------------------------------------------------------
# Caching: the minute-long recompute
# ---------------------------------------------------------------------------

def test_a_subject_is_scored_once_however_often_it_is_plotted():
    app = _Integrity(_project(3))
    for _ in range(4):
        for sid in app.processed_data:
            app.signal_integrity_metrics(sid)
    assert len(app.calls) == 3


def test_changing_the_selection_only_scores_the_newcomer():
    """The whole-figure cache misses on any selection change; this is what makes
    that miss cheap instead of a full minute."""
    app = _Integrity(_project(4))
    first = ['S00', 'S01', 'S02']
    for sid in first:
        app.signal_integrity_metrics(sid)
    app.calls.clear()
    for sid in ['S00', 'S01', 'S03']:
        app.signal_integrity_metrics(sid)
    assert len(app.calls) == 1


def test_sensor_mode_is_part_of_the_key():
    """The two modes score differently, so one must not serve the other."""
    app = _Integrity(_project(1))
    app.signal_integrity_metrics('S00', sensor=False)
    app.signal_integrity_metrics('S00', sensor=True)
    app.signal_integrity_metrics('S00', sensor=False)
    assert [c[1] for c in app.calls] == [False, True]


def test_reprocessing_a_subject_invalidates_it():
    """Reprocessing replaces the data dict; a cache that survived that would
    report the old signal's score for the new data."""
    app = _Integrity(_project(2))
    app.signal_integrity_metrics('S00')
    app.calls.clear()
    app.processed_data['S00'] = _subject()       # a new dict, as reprocessing does
    app.signal_integrity_metrics('S00')
    assert len(app.calls) == 1


def test_a_recycled_id_cannot_serve_a_stale_score():
    """Keys carry id(data), which CPython reuses after a dict is freed. The
    pinned reference is what makes the key trustworthy."""
    app = _Integrity(_project(1))
    app.signal_integrity_metrics('S00')
    stale_key = next(iter(app._integrity_metrics_cache))
    replacement = _subject(channels=1)
    app._integrity_metrics_cache[stale_key] = (replacement, ['poisoned'])
    app.processed_data['S00'] = replacement
    # Same subject, different dict -> the key no longer matches, so it rescores.
    got = app.signal_integrity_metrics('S00', app.processed_data['S00'])
    assert got != ['poisoned']


def test_invalidate_clears_both_caches():
    app = _Integrity(_project(2))
    app.signal_integrity_metrics('S00')
    app._integrity_plot_cache = {'sig': 'x'}
    app.invalidate_integrity_caches()
    assert app._integrity_metrics_cache == {}
    assert app._integrity_plot_cache is None


def test_the_cache_does_not_grow_without_bound():
    """Entries pin their data dict, so unloaded subjects would leak."""
    app = _Integrity(_project(2))
    for _ in range(40):                       # 40 rounds of "reprocess both"
        for sid in list(app.processed_data):
            app.processed_data[sid] = _subject()
            app.signal_integrity_metrics(sid)
    assert len(app._integrity_metrics_cache) <= 4 * len(app.processed_data)


def test_a_subject_with_no_photometry_is_not_rescored_every_time():
    app = _Integrity({'S00': _subject(has_470=False)})
    assert app.signal_integrity_metrics('S00') is None
    assert app.signal_integrity_metrics('S00') is None
    assert len(app.calls) == 1


def test_an_unknown_subject_scores_nothing():
    app = _Integrity(_project(1))
    assert app.signal_integrity_metrics('nope') is None
    assert app.calls == []


# ---------------------------------------------------------------------------
# Sizing: the unreadable table
# ---------------------------------------------------------------------------

def test_rows_are_counted_per_channel_not_per_subject():
    app = _Integrity(_project(62, channels=2))
    assert app._integrity_row_estimate(list(app.processed_data)) == 124


def test_subjects_with_no_photometry_take_no_row():
    app = _Integrity({'a': _subject(), 'b': _subject(has_470=False)})
    assert app._integrity_row_estimate(['a', 'b']) == 2


def test_the_figure_grows_with_the_table():
    """The bug: 124 rows were laid out in a fixed 7.5in figure, about 3pt each,
    so 6pt text overlapped into a smear."""
    app = _Integrity(_project(62))
    subs = list(app.processed_data)
    _, tall = app._auto_viz_figure_size('Signal Integrity', len(subs), subs)
    assert tall > 20

    rows = app._integrity_row_estimate(subs)
    # Row height in points, mirroring the table's own axes-fraction layout.
    pts = (0.90 / (rows + 2)) * tall * 0.9 * 72
    assert pts > 8          # comfortably clears the 6pt text it has to hold


def test_a_small_selection_keeps_the_original_size():
    app = _Integrity(_project(4))
    subs = list(app.processed_data)
    assert app._auto_viz_figure_size('Signal Integrity', len(subs), subs)[1] == 7.5


def test_the_single_subject_dashboard_is_untouched():
    """One subject is a panel dashboard, not a table -- it must not grow."""
    app = _Integrity(_project(1))
    assert app._auto_viz_figure_size('Signal Integrity', 1, ['S00']) == (11.0, 7.5)


def test_height_is_capped_so_a_huge_project_stays_renderable():
    app = _Integrity(_project(600))
    subs = list(app.processed_data)
    assert app._auto_viz_figure_size('Signal Integrity', len(subs), subs)[1] <= 48.0


def test_sizing_without_a_subject_list_still_scales():
    """Callers that only know the count must not fall back to the old fixed height."""
    app = _Integrity({})
    assert app._auto_viz_figure_size('Signal Integrity', 62)[1] > 20


# ---------------------------------------------------------------------------
# Re-entry: the queued clicks a pumped event loop delivers
# ---------------------------------------------------------------------------

def test_the_busy_flag_is_raised_while_scoring_and_lowered_after():
    app = _Integrity(_project(10))
    seen = []
    for sid in app._integrity_progress(list(app.processed_data), 'test'):
        seen.append(app._integrity_busy)
    assert all(seen)
    assert app._integrity_busy is False


def test_the_busy_flag_is_lowered_when_scoring_raises():
    app = _Integrity(_project(10))
    try:
        for _ in app._integrity_progress(list(app.processed_data), 'test'):
            raise RuntimeError('boom')
    except RuntimeError:
        pass
    assert app._integrity_busy is False


def test_every_subject_is_yielded_once():
    app = _Integrity(_project(7))
    subs = list(app.processed_data)
    assert list(app._integrity_progress(subs, 'test')) == subs
