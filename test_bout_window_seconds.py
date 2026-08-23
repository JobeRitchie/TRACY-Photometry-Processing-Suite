"""The bout window is specified in seconds, not samples.

A sample count means a different duration at every sampling rate, so a project
holding cohorts recorded on different rigs could not be analysed together: a
200-sample window is 10.0 s at 19.94 Hz but 6.7 s at 29.99 Hz, and traces from
the faster rig were drawn on the slower rig's axis, stretching them by 1.5x.

These tests pin the two halves of the fix -- each subject's window is converted
to samples at its OWN rate when its bouts are extracted, and stored traces are
resampled onto one shared time axis when they are read back -- using the two
real NSF rates.
"""
import numpy as np

import fp_analysis_gui as G

SLOW = 19.936   # 2023 cohort, samples/s per LED channel
FAST = 29.990   # 2024 cohort


def _app(pre=5.0, post=10.0, baseline=0.0, rates=(SLOW, FAST)):
    """An FPAnalysisGUI shell with subjects at the given rates, and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'preboutseconds': pre, 'postboutseconds': post,
                  'baseline_seconds': baseline,
                  'preboutframes': 90, 'postboutframes': 90, 'baseline_frames': 45}
    app.processed_data = {f'S{i}': {'photometry_fps': r} for i, r in enumerate(rates)}
    app.log_message = lambda *a, **k: None
    return app


def test_window_is_sized_per_subject_from_seconds():
    """The same 5 s / 10 s window is a different sample count on each rig."""
    app = _app()
    slow_pre, slow_post, _ = app.bout_window_samples('S0')
    fast_pre, fast_post, _ = app.bout_window_samples('S1')

    assert (slow_pre, slow_post) == (100, 199)      # 5 s / 10 s at 19.936 Hz
    assert (fast_pre, fast_post) == (150, 300)      # 5 s / 10 s at 29.990 Hz
    # Both really are the requested duration, to within a rounded sample.
    assert abs(slow_pre / SLOW - 5.0) < 1.0 / SLOW
    assert abs(fast_post / FAST - 10.0) < 1.0 / FAST


def test_analysis_axis_is_the_highest_rate_so_no_samples_are_discarded():
    app = _app()
    assert app.analysis_fps() == FAST
    pre, post, _ = app.bout_window_samples()
    assert (pre, post) == (150, 300)


def test_baseline_defaults_to_half_the_pre_window_and_honours_seconds():
    half = _app(pre=5.0, baseline=0.0)
    assert half.bout_window_samples('S1')[2] == 150 // 2

    explicit = _app(pre=5.0, baseline=2.0)
    assert explicit.bout_window_samples('S1')[2] == int(round(2.0 * FAST))

    # A baseline longer than the pre-window is clamped, not allowed to read
    # forward across the onset.
    over = _app(pre=1.0, baseline=99.0)
    pre, _, base = over.bout_window_samples('S1')
    assert base == pre


def _peak_trace(fs, pre_s, post_s, peak_s=0.9, width_s=0.4):
    """A gaussian bump at *peak_s* after onset, sampled at *fs*."""
    n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
    t = (np.arange(n_pre + n_post) - n_pre) / fs
    return np.exp(-0.5 * ((t - peak_s) / width_s) ** 2), n_pre


def test_a_fast_trace_keeps_its_peak_time_on_the_slow_axis():
    """The bug this fixes: without resampling a 30 Hz trace read on a 20 Hz
    axis puts its peak 1.5x too late."""
    app = _app()
    trace, src_pre = _peak_trace(FAST, 5.0, 10.0)

    dst_pre = int(round(5.0 * SLOW))
    dst_total = dst_pre + int(round(10.0 * SLOW))
    out = app._realign_bout_trace(trace, src_pre, dst_pre, dst_total,
                                  src_fs=FAST, dst_fs=SLOW)

    assert len(out) == dst_total
    peak_s = (int(np.nanargmax(out)) - dst_pre) / SLOW
    assert abs(peak_s - 0.9) < 0.05

    # And the un-resampled read is what was wrong before.
    naive = app._realign_bout_trace(trace, src_pre, dst_pre, dst_total)
    naive_peak_s = (int(np.nanargmax(naive)) - dst_pre) / SLOW
    assert naive_peak_s > 1.2


def test_slow_and_fast_subjects_average_onto_the_same_peak():
    """Two rigs, one grand average: the cohorts must reinforce, not smear."""
    app = _app()
    dst_pre, dst_post, _ = app.bout_window_samples()
    dst_total = dst_pre + dst_post

    aligned = []
    for fs in (SLOW, FAST):
        trace, src_pre = _peak_trace(fs, 5.0, 10.0)
        aligned.append(app._realign_bout_trace(trace, src_pre, dst_pre, dst_total,
                                               src_fs=fs, dst_fs=app.analysis_fps()))

    mean = np.nanmean(np.vstack(aligned), axis=0)
    peak_s = (int(np.nanargmax(mean)) - dst_pre) / app.analysis_fps()
    assert abs(peak_s - 0.9) < 0.05
    # Averaging two traces of the same shape must not shrink the peak.
    assert np.nanmax(mean) > 0.95


def test_resampling_does_not_invent_data_beyond_the_stored_window():
    """A trace stored with a shorter window stays blank where it has nothing."""
    app = _app()
    trace, src_pre = _peak_trace(FAST, 1.0, 1.0)        # only +/-1 s stored
    dst_pre = int(round(5.0 * SLOW))
    dst_total = dst_pre + int(round(10.0 * SLOW))
    out = app._realign_bout_trace(trace, src_pre, dst_pre, dst_total,
                                  src_fs=FAST, dst_fs=SLOW)

    t = (np.arange(dst_total) - dst_pre) / SLOW
    assert np.all(np.isnan(out[t < -1.0]))
    assert np.all(np.isnan(out[t > 1.0]))
    assert np.isfinite(out[np.argmin(np.abs(t))])


def test_same_rate_is_left_untouched():
    """No interpolation when there is nothing to reconcile."""
    app = _app(rates=(FAST, FAST))
    app._sync_bout_window_frames()
    trace, src_pre = _peak_trace(FAST, 5.0, 10.0)
    out = app._realign_bouts([trace], src_pre, FAST)
    assert out[0] is trace


def test_a_project_saved_in_frames_keeps_its_window_duration():
    """Migration must not silently adopt this build's default window."""
    app = _app()
    app.params['preboutseconds'] = None
    app.params['postboutseconds'] = None
    app.params['preboutframes'] = 150          # 5 s at the 29.99 Hz axis
    app.params['postboutframes'] = 300         # 10 s
    app.params['baseline_frames'] = 75

    app._migrate_bout_window_to_seconds()

    assert abs(app.params['preboutseconds'] - 5.0) < 0.01
    assert abs(app.params['postboutseconds'] - 10.0) < 0.01
    assert abs(app.params['baseline_seconds'] - 2.5) < 0.01


def test_sync_publishes_the_shared_axis_into_the_frame_params():
    """The many frame-denominated readers must see the shared axis."""
    app = _app()
    pre, post, base = app._sync_bout_window_frames()
    assert (app.params['preboutframes'], app.params['postboutframes']) == (pre, post)
    assert app.params['baseline_frames'] == base
    assert (pre, post) == (150, 300)            # sized at the 29.99 Hz axis


# ---------------------------------------------------------------- mixed-window warning

def _app_with_stored(rates_by_subject, pre=5.0, post=10.0):
    """App whose subjects hold a stored bout trace of the RIGHT duration for
    their own rate, i.e. exactly what a clean re-extract produces."""
    app = _app(pre=pre, post=post, rates=())
    app.processed_data = {}
    for s, fs in rates_by_subject.items():
        n = int(round(pre * fs)) + int(round(post * fs))
        app.processed_data[s] = {
            'photometry_fps': fs,
            'bouts': {'Approach': {'G0': [np.zeros(n)], 'G1': []}},
        }
    return app


def _capture_warning(monkeypatch, app, subjects):
    seen = []
    monkeypatch.setattr(G, 'messagebox',
                        type('MB', (), {'showwarning': staticmethod(
                            lambda *a, **k: seen.append(a))})())
    app._warn_if_mixed_bout_windows(subjects)
    return seen


def test_rates_differing_in_the_third_decimal_do_not_warn(monkeypatch):
    """The real NSF rates: 19.932 / 19.936 / 19.940 / 29.990 round a 15 s window
    to 14.995-15.005 s.  That is one sample of rounding, not four windows."""
    rates = {'2-Q': 19.940, '1-51': 19.936, '1-B': 19.932, 'nsf1': 29.990}
    app = _app_with_stored(rates)
    seen = _capture_warning(monkeypatch, app, list(rates))
    assert seen == []
    assert app._mixed_window_warn_sig is None


def test_a_genuinely_stale_window_still_warns(monkeypatch):
    """A subject left at the old window must still be reported."""
    rates = {'1-51': 19.936, 'nsf1': 29.990}
    app = _app_with_stored(rates)
    stale = np.zeros(int(round(3.0 * 29.990)) + int(round(3.0 * 29.990)))  # 6 s
    app.processed_data['nsf1']['bouts']['Approach']['G0'] = [stale]

    seen = _capture_warning(monkeypatch, app, list(rates))
    assert len(seen) == 1
    body = seen[0][1]
    assert 'nsf1' in body
    assert '1-51' not in body      # the up-to-date subject is not implicated


def test_group_axis_rate_matches_the_axis_the_traces_are_on():
    """A group-level caller turns a sample index into a time, so it must get the
    rate the resampled traces actually carry -- not the most common rate, which
    labelled a 450-sample 15 s window as 22.6 s."""
    app = _app()   # 19.936 x1, 29.990 x1
    app._mixed_fps_warned = False
    assert app.get_fps() == app.analysis_fps() == FAST

    # Per-subject callers still get that subject's own rate.
    assert app.get_fps('S0') == SLOW
    assert app.get_fps('S1') == FAST

    pre, post, _ = app.bout_window_samples()
    assert abs((pre + post) / app.get_fps() - 15.0) < 0.05


def test_group_axis_rate_with_a_slow_majority():
    """The modal rate being slow must not drag the axis off the traces."""
    app = _app(rates=(SLOW, SLOW, SLOW, FAST))
    app._mixed_fps_warned = False
    assert app.get_fps() == FAST
