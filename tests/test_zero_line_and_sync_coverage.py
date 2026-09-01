"""Time-zero marker styling + photometry-coverage handling in the sync/overlay.

Two changes are pinned here:

1. Every peri-event graph draws its t=0 marker through ``draw_zero_line``, so
   colour and weight come from one setting (default: a thin grey line) instead
   of being hard-coded red/yellow at ~30 call sites.
2. ``synchronize_behavior`` no longer fills behavior frames that fall outside
   the photometry recording with the nearest edge sample. Those rows stay NaN,
   and ``_photometry_coverage_mask`` lets a graph blank the same region in a
   project processed before the fix.

The shells follow ``test_exclusion_channel_keys.py``: an ``object.__new__``
instance with only the attributes the code under test reads, so no Tk is
needed.
"""
import numpy as np
import pytest

from fp_analysis_gui import FPAnalysisGUI


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _app(**params):
    app = object.__new__(FPAnalysisGUI)
    app.params = {'zero_line_color': FPAnalysisGUI.ZERO_LINE_DEFAULT_COLOR,
                  'zero_line_width': FPAnalysisGUI.ZERO_LINE_DEFAULT_WIDTH}
    app.params.update(params)
    return app


class _Var:
    """Stand-in for a tk.StringVar."""

    def __init__(self, value=''):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


class _Ax:
    """Records what reaches matplotlib."""

    def __init__(self):
        self.vlines = []

    def axvline(self, x, **kw):
        self.vlines.append((x, kw))
        return ('line', x, kw)


# --------------------------------------------------------------------------
# get_zero_line_style
# --------------------------------------------------------------------------
def test_default_is_a_thin_grey_line():
    color, width = _app().get_zero_line_style()
    assert color == '#808080'
    assert width == pytest.approx(0.8)
    # Thin enough to sit under the data rather than dominate it -- the 2 pt red
    # line this replaced is what prompted the setting.
    assert width < 1.0


def test_params_drive_the_style_without_any_widgets():
    color, width = _app(zero_line_color='red', zero_line_width=2.5).get_zero_line_style()
    assert color == 'red'
    assert width == pytest.approx(2.5)


def test_widgets_win_over_params():
    app = _app(zero_line_color='#808080', zero_line_width=0.8)
    app.zero_line_color_var = _Var('#c1121f')
    app.zero_line_width_var = _Var('3')
    assert app.get_zero_line_style() == ('#c1121f', 3.0)


def test_reading_the_style_commits_it_to_params_for_saving():
    app = _app()
    app.zero_line_color_var = _Var('navy')
    app.zero_line_width_var = _Var('1.25')
    app.get_zero_line_style()
    assert app.params['zero_line_color'] == 'navy'
    assert app.params['zero_line_width'] == pytest.approx(1.25)


@pytest.mark.parametrize('bad', ['', '   ', 'not-a-color', 'chartreuseX'])
def test_unusable_colour_falls_back_instead_of_raising(bad):
    app = _app()
    app.zero_line_color_var = _Var(bad)
    assert app.get_zero_line_style()[0] == '#808080'


@pytest.mark.parametrize('bad', ['', 'wide', None])
def test_unusable_width_falls_back(bad):
    app = _app()
    app.zero_line_width_var = _Var(bad)
    assert app.get_zero_line_style()[1] == pytest.approx(0.8)


def test_width_is_clamped_to_a_sane_range():
    app = _app()
    app.zero_line_width_var = _Var('-4')
    assert app.get_zero_line_style()[1] == 0.0      # negative -> hidden, not flipped
    app.zero_line_width_var = _Var('999')
    assert app.get_zero_line_style()[1] == 10.0


# --------------------------------------------------------------------------
# draw_zero_line
# --------------------------------------------------------------------------
def test_draw_zero_line_uses_the_configured_style():
    app = _app(zero_line_color='#123456', zero_line_width=2.0)
    ax = _Ax()
    app.draw_zero_line(ax, label='Bout Onset')
    (x, kw), = ax.vlines
    assert x == 0.0
    assert kw['color'] == '#123456'
    assert kw['linewidth'] == pytest.approx(2.0)
    assert kw['linestyle'] == '--'
    assert kw['label'] == 'Bout Onset'


def test_draw_zero_line_omits_the_legend_entry_when_unlabelled():
    ax = _Ax()
    _app().draw_zero_line(ax)
    assert 'label' not in ax.vlines[0][1]


def test_zero_width_hides_the_marker_entirely():
    ax = _Ax()
    assert _app(zero_line_width=0).draw_zero_line(ax, label='Entry') is None
    assert ax.vlines == []          # no line, and so no legend entry either


def test_draw_zero_line_sits_above_traces_and_heatmap_images():
    ax = _Ax()
    _app().draw_zero_line(ax)
    # imshow=0, fill_between=1, Line2D=2 -- the marker must clear all three.
    assert ax.vlines[0][1]['zorder'] > 2


def test_draw_zero_line_accepts_a_non_zero_x_and_overrides():
    ax = _Ax()
    _app().draw_zero_line(ax, x=5.0, alpha=0.2)
    x, kw = ax.vlines[0]
    assert x == 5.0
    assert kw['alpha'] == pytest.approx(0.2)


#: Every Visualization-tab plot that draws a t=0 marker. Other tabs (FLMM,
#: Kinematics, the spectrograms) keep their own markers and are out of scope.
VIZ_PERI_EVENT_PLOTS = [
    'plot_extracted_bouts', 'plot_extracted_bouts_multi',
    'plot_extracted_bouts_by_group',
    'plot_zone_entry_bouts', 'plot_zone_entry_bouts_multi',
    '_plot_zone_entry_bouts_group_comparison',
    'compare_bout_channels', '_plot_length_binned_axis',
    # The three Compare Across Bouts entry points share one renderer, the way
    # the bout-length views share _plot_length_binned_axis; the marker is drawn
    # there, so that is what has to route through the helper.
    '_render_bout_comparison',
]


@pytest.mark.parametrize('name', VIZ_PERI_EVENT_PLOTS)
def test_every_peri_event_plot_routes_through_the_helper(name):
    """No hard-coded red/yellow t=0 axvline is left on the Visualization tab.

    Each of these drew its own 2 pt red (trace) or yellow (heatmap) line, which
    is why the two used to drift apart. They must all go through
    ``draw_zero_line`` or the single setting stops being single.
    """
    import inspect
    import re

    src = inspect.getsource(getattr(FPAnalysisGUI, name))
    offenders = re.findall(r'^.*\.axvline\(\s*0[,)].*$', src, re.M)
    offenders = [o for o in offenders if 'draw_zero_line' not in o]
    assert offenders == [], offenders
    assert 'self.draw_zero_line(' in src


# --------------------------------------------------------------------------
# _sync_match_tolerance
# --------------------------------------------------------------------------
def test_tolerance_is_two_periods_of_the_coarser_recording():
    fp = np.arange(0, 100, 0.05)        # 20 Hz photometry
    beh = np.arange(0, 100, 1 / 30)     # 30 fps camera
    assert FPAnalysisGUI._sync_match_tolerance(fp, beh) == pytest.approx(0.1)
    # ...and it follows the behaviour file when that is the coarse one.
    assert FPAnalysisGUI._sync_match_tolerance(fp, np.arange(0, 100, 1.0)) == pytest.approx(2.0)


def test_tolerance_is_permissive_when_it_cannot_be_measured():
    # A single sample, or a constant timestamp column, must not mask a session.
    assert FPAnalysisGUI._sync_match_tolerance([1.0], [2.0]) == float('inf')
    assert FPAnalysisGUI._sync_match_tolerance(np.zeros(50), np.zeros(50)) == float('inf')


# --------------------------------------------------------------------------
# synchronize_behavior coverage
# --------------------------------------------------------------------------
def _sync_app():
    app = object.__new__(FPAnalysisGUI)
    app.params = {}
    app.log_message = lambda *a, **k: None
    return app


def _sync_inputs(fp_start, n_beh=400, n_fp=400, dt=0.05):
    """One channel of photometry starting `fp_start` s into the behavior file."""
    beh_ts = np.arange(n_beh) * dt
    beh_raw = np.column_stack([np.arange(n_beh), beh_ts])
    fp_ts = fp_start + np.arange(n_fp) * dt
    zscore = np.column_stack([fp_ts / 60.0, fp_ts, np.arange(n_fp, dtype=float)])
    return zscore, beh_raw


def test_frames_before_the_photometry_started_stay_blank():
    # FP begins 5 s (100 frames) into the behavior timeline. The tolerance is
    # two sample periods, so the two rows straddling the boundary still snap to
    # the first real sample -- everything earlier must stay blank.
    zscore, beh_raw = _sync_inputs(fp_start=5.0)
    out = _sync_app().synchronize_behavior(zscore, beh_raw)
    ch = out[:, 6]
    assert np.all(np.isnan(ch[:98])), 'uncovered head was filled with the edge sample'
    assert np.isfinite(ch[100:]).all()
    # The covered rows still carry the right sample, not a shifted one.
    assert ch[100] == pytest.approx(0.0)
    assert ch[150] == pytest.approx(50.0)


def test_frames_after_the_photometry_stopped_stay_blank():
    zscore, beh_raw = _sync_inputs(fp_start=0.0, n_beh=400, n_fp=300)
    ch = _sync_app().synchronize_behavior(zscore, beh_raw)[:, 6]
    assert np.isfinite(ch[:300]).all()
    assert np.all(np.isnan(ch[303:]))


def test_a_fully_covered_session_is_untouched():
    zscore, beh_raw = _sync_inputs(fp_start=0.0)
    ch = _sync_app().synchronize_behavior(zscore, beh_raw)[:, 6]
    assert np.isfinite(ch).all()
    assert ch == pytest.approx(np.arange(400, dtype=float))


def test_a_faster_camera_than_photometry_is_still_fully_matched():
    """30 fps behaviour against 20 Hz photometry: every row is within tolerance."""
    beh_ts = np.arange(600) / 30.0
    beh_raw = np.column_stack([np.arange(600), beh_ts])
    fp_ts = np.arange(400) * 0.05
    zscore = np.column_stack([fp_ts / 60.0, fp_ts, np.arange(400, dtype=float)])
    ch = _sync_app().synchronize_behavior(zscore, beh_raw)[:, 6]
    assert np.isfinite(ch).all()


def test_the_uncovered_stretch_is_reported():
    zscore, beh_raw = _sync_inputs(fp_start=5.0)
    app = _sync_app()
    logged = []
    app.log_message = logged.append
    app.synchronize_behavior(zscore, beh_raw)
    reports = [m for m in logged if 'outside the photometry recording' in m]
    assert len(reports) == 1, logged
    assert 'of 400 behavior frame(s)' in reports[0]
    # ~100 frames uncovered (24.5%), so the "check your files" nudge fires too.
    assert 'no signal behind them' in reports[0]


def test_a_covered_session_reports_nothing():
    zscore, beh_raw = _sync_inputs(fp_start=0.0)
    app = _sync_app()
    logged = []
    app.log_message = logged.append
    app.synchronize_behavior(zscore, beh_raw)
    assert not [m for m in logged if 'outside the photometry recording' in m]


# --------------------------------------------------------------------------
# _contiguous_runs / _photometry_coverage_mask
# --------------------------------------------------------------------------
def test_contiguous_runs():
    f = FPAnalysisGUI._contiguous_runs
    assert f([]) == []
    assert f([False, False]) == []
    assert f([True, True, True]) == [(0, 2)]
    assert f([True, False, True, True, False, True]) == [(0, 0), (2, 3), (5, 5)]


def _coverage_app():
    app = object.__new__(FPAnalysisGUI)
    app.params = {}
    return app


def _coverage_inputs(fp_start, n_beh=400, n_fp=400, dt=0.05):
    beh = np.full((n_beh, 13), np.nan)
    beh[:, 0] = np.arange(n_beh)
    beh[:, 1] = np.arange(n_beh) * dt
    fp_ts = fp_start + np.arange(n_fp) * dt
    zscore = np.column_stack([fp_ts / 60.0, fp_ts, np.zeros(n_fp)])
    return {'zscore': zscore}, beh


def test_coverage_mask_finds_the_uncovered_head():
    data, beh = _coverage_inputs(fp_start=5.0)
    mask = _coverage_app()._photometry_coverage_mask(data, beh)
    assert mask is not None
    assert not mask[:98].any()      # two-sample-period edge tolerance, as above
    assert mask[101:].all()


def test_coverage_mask_is_none_when_everything_is_covered_is_still_all_true():
    data, beh = _coverage_inputs(fp_start=0.0)
    mask = _coverage_app()._photometry_coverage_mask(data, beh)
    assert mask is not None and mask.all()


def test_coverage_mask_declines_when_the_clocks_do_not_share_an_epoch():
    """The relative-matching sync path: comparing the two ranges is meaningless."""
    data, beh = _coverage_inputs(fp_start=0.0)
    data['zscore'][:, 1] += 1e6
    assert _coverage_app()._photometry_coverage_mask(data, beh) is None


def test_coverage_mask_declines_without_usable_timestamps():
    data, beh = _coverage_inputs(fp_start=0.0)
    beh[:, 1] = np.nan
    assert _coverage_app()._photometry_coverage_mask(data, beh) is None
    assert _coverage_app()._photometry_coverage_mask({}, beh) is None
