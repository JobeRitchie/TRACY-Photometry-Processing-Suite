"""Per-subject session start/end bounds, in video frames.

Recordings routinely start before the animal is in the apparatus and end after
it is lifted out.  Those stretches are not merely uninteresting: pixel->cm
calibration is fitted to the observed X range, so a few seconds of the animal
being carried past the camera rescales the whole session and moves every zone
boundary.  The bounds trim the raw file at read time, before the recording-length
cap, so a 600 s cap means 600 s of real session.

The tests that matter most here are the alignment ones: trimming the head moves
signal sample 0, and every video-frame->sample conversion has to move with it or
bouts land in the wrong place.
"""
import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G


VIDEO_FPS = 30.0
ROW_DT = 0.025            # 40 Hz raw rows -> 20 Hz per channel over 2 LED states
LED_CYCLE = [1, 2]


def _raw_csv(tmp_path, n_rows=48000, name='raw'):
    """An interleaved FPData CSV: 415/470 alternating at 40 raw rows/s."""
    rng = np.random.default_rng(0)
    cols = ['FrameCounter', 'SystemTimestamp', 'LedState', 'ComputerTimestamp', 'G0', 'G1']
    rows = []
    for f in range(n_rows):
        t = 1000.0 + f * ROW_DT
        rows.append([f, t, LED_CYCLE[f % len(LED_CYCLE)], t * 1000.0,
                     1.0 + 0.05 * np.sin(f / 37.0) + 0.01 * rng.standard_normal(),
                     1.2 + 0.05 * np.sin(f / 53.0) + 0.01 * rng.standard_normal()])
    path = tmp_path / f"{name}_FPData.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)


def _abel_csv(tmp_path, n_frames=30000, junk_frames=0, name='raw'):
    """An ABEL position track; the first *junk_frames* sit outside the arena.

    The arena is x/y in [100, 300] px; the junk is at x/y ~ [10, 40], which is
    where the animal-in-hand actually shows up in the real NSF recordings.
    """
    rng = np.random.default_rng(1)
    frames = np.arange(n_frames)
    x = 100.0 + 200.0 * rng.random(n_frames)
    y = 100.0 + 200.0 * rng.random(n_frames)
    if junk_frames:
        x[:junk_frames] = 10.0 + 30.0 * rng.random(junk_frames)
        y[:junk_frames] = 10.0 + 30.0 * rng.random(junk_frames)
    path = tmp_path / f"{name}_ABELposition.csv"
    pd.DataFrame({'frame': frames, 'timestamp': frames / VIDEO_FPS,
                  'X': x, 'Y': y}).to_csv(path, index=False)
    return str(path)


class _Flag:
    def __init__(self, v): self.v = v
    def get(self): return self.v


def _app(bounds=None, max_seconds=0.0, enabled=True, **extra):
    """An FPAnalysisGUI shell that can run process_fp_data without Tk."""
    import types
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.root = types.SimpleNamespace(update_idletasks=lambda: None)
    for flag in ('process_green', 'process_red', 'subtract_iso_green',
                 'subtract_iso_red', 'show_470', 'show_570'):
        setattr(app, flag, _Flag(True))
    app.boutframes_path_var = _Flag("")
    app.ttl_path_var = _Flag("")
    app.processed_data = {}
    app.bout_offsets = []
    app.per_subject_boutframe_shifts = {}
    app.detected_photometry_fps = None
    app.processing_summary = {
        'fallbacks_used': [], 'mismatched_led_samples': [], 'structure_warnings': [],
        'zero_variance_channels': [], 'no_position_data': [], 'missing_behavior': [],
    }
    app.zones = {}
    app.params = {
        'precut': 100,
        'maxlengthseconds': max_seconds,
        'processing_rolling_avg_enabled': False,
        'boutframes_video_fps': VIDEO_FPS,
        'auto_scale_boutframes': True,
        'boutframe_manual_shift': 0,
        'precut_correct_boutframes': True,
        'abel_position_enabled': True,
        'abel_position_pattern': 'ABELposition',
        'session_bounds_enabled': enabled,
        'session_bounds_frames': bounds or {},
        'maze_type': 'OFT',
        'maze_width_cm': 51.0,
        'y_calibration_method': 'normalize_y_min',
        'velocity_outlier_threshold': 5,
        'preboutseconds': 3.0, 'postboutseconds': 3.0,
        'preboutframes': 60, 'postboutframes': 60,
        'baseline_correct_bouts': False, 'baseline_seconds': 0.0, 'baseline_frames': 0,
        'oft_min_entry_duration': 0.5, 'oft_min_time_between_entries': 1.0,
        'min_open_arm_duration': 2.0, 'min_time_between_entries': 3.0,
        'outback_velocity_threshold': 2.0,
        'distal_threshold': 25, 'time_bin_size': 60, 'spatial_bin_size': 2,
        'exclude_frames_before': 0,
    }
    app.params.update(extra)
    return app


def _run(app, subject, fp, pos=None):
    res = app.process_fp_data(subject, fp, pos)
    app.processed_data[subject] = res
    return res


def _seconds(res):
    return len(res['zscore']) / float(res['photometry_fps'])


# ── Trimming ────────────────────────────────────────────────────────

def test_start_bound_drops_the_head(tmp_path):
    fp = _raw_csv(tmp_path)                     # 48000 rows = 1200 s
    full = _seconds(_run(_app(), 'S', fp))
    bounded = _seconds(_run(_app({'S': {'start': 3000, 'end': 0}}), 'S', fp))
    # 3000 video frames at 30 fps = 100 s, minus the 2.5 s the precut already cut
    assert full - bounded == pytest.approx(100.0 - 100 * ROW_DT, abs=0.2)


def test_the_length_cap_counts_from_the_start_bound(tmp_path):
    """The whole point: 600 s must mean 600 s of real session."""
    fp = _raw_csv(tmp_path)
    res = _run(_app({'S': {'start': 3000, 'end': 0}}, max_seconds=600.0), 'S', fp)
    assert _seconds(res) == pytest.approx(600.0, abs=0.2)


def test_end_bound_drops_the_tail(tmp_path):
    fp = _raw_csv(tmp_path)
    res = _run(_app({'S': {'start': 3000, 'end': 3000 + 300 * 30}}), 'S', fp)
    assert _seconds(res) == pytest.approx(300.0, abs=0.2)


def test_the_tighter_of_end_bound_and_cap_wins(tmp_path):
    fp = _raw_csv(tmp_path)
    short_end = _run(_app({'S': {'start': 3000, 'end': 3000 + 200 * 30}},
                          max_seconds=600.0), 'S', fp)
    short_cap = _run(_app({'S': {'start': 3000, 'end': 3000 + 900 * 30}},
                          max_seconds=400.0), 'S', fp)
    assert _seconds(short_end) == pytest.approx(200.0, abs=0.2)
    assert _seconds(short_cap) == pytest.approx(400.0, abs=0.2)


def test_an_end_at_or_before_the_start_is_refused(tmp_path):
    """A reversed pair would empty the recording; it must be ignored instead."""
    fp = _raw_csv(tmp_path)
    full = _seconds(_run(_app(), 'S', fp))
    bad = _seconds(_run(_app({'S': {'start': 3000, 'end': 2000}}), 'S', fp))
    assert bad == pytest.approx(full, abs=0.05)


def test_the_toggle_switches_bounds_off_without_clearing_them(tmp_path):
    fp = _raw_csv(tmp_path)
    full = _seconds(_run(_app(), 'S', fp))
    off = _seconds(_run(_app({'S': {'start': 3000, 'end': 0}}, enabled=False), 'S', fp))
    assert off == pytest.approx(full, abs=0.05)


def test_ids_are_matched_case_insensitively(tmp_path):
    """Bounds are pasted from spreadsheets that spell '1-f' as '1-F'."""
    fp = _raw_csv(tmp_path)
    lower = _seconds(_run(_app({'1-f': {'start': 3000, 'end': 0}}), '1-f', fp))
    upper = _seconds(_run(_app({'1-F': {'start': 3000, 'end': 0}}), '1-f', fp))
    assert upper == pytest.approx(lower, abs=0.05)


# ── Alignment ───────────────────────────────────────────────────────

def test_a_video_frame_lands_on_the_same_sample_before_and_after_trimming(tmp_path):
    """The invariant the whole feature rests on.

    Column 1 of the z-score array is the acquisition clock, so if frame F maps to
    the same timestamp in both runs the bout lands on the same real moment.
    """
    fp = _raw_csv(tmp_path)
    start = 3000
    a0 = _app(); r0 = _run(a0, 'S', fp)
    a1 = _app({'S': {'start': start, 'end': 0}}); r1 = _run(a1, 'S', fp)

    probes = np.linspace(start + 60, start + 8000, 30).astype(int)
    i0 = a0._transform_boutframe_values(probes, 'S')
    i1 = a1._transform_boutframe_values(probes, 'S')
    t0, t1 = r0['zscore'][:, 1], r1['zscore'][:, 1]
    ok = (i0 >= 0) & (i0 < len(t0)) & (i1 >= 0) & (i1 < len(t1))
    assert ok.sum() >= 25
    assert np.array_equal(t0[i0[ok]], t1[i1[ok]])


def test_bouts_before_the_start_bound_fall_off_the_front(tmp_path):
    fp = _raw_csv(tmp_path)
    start = 3000
    app = _app({'S': {'start': start, 'end': 0}})
    _run(app, 'S', fp)
    idx = app._transform_boutframe_values(np.array([0, start // 3, start - 30]), 'S')
    assert (idx < 0).all()


def test_the_trim_keeps_the_led_interleave_in_phase(tmp_path):
    """Cutting on an odd raw row moves one channel a sample against the other.

    A sub-sample error, but a systematic one -- it shows up as a fixed lag
    between channels that no amount of shifting boutframes can fix.
    """
    fp = _raw_csv(tmp_path)
    # 3001 video frames = 100.033 s, which lands mid-cycle in the raw stream.
    a0 = _app(); r0 = _run(a0, 'S', fp)
    a1 = _app({'S': {'start': 3001, 'end': 0}}); r1 = _run(a1, 'S', fp)
    off = len(r0['zscore']) - len(r1['zscore'])
    assert off > 0
    # The deinterleaved signal must be a clean suffix of the unbounded one --
    # same rows, same values.  (dF/F is not: its photobleaching fit is refitted
    # over the shorter window, so it moves by a hair on every sample.)
    assert np.array_equal(r0['zscore'][off:, 1], r1['zscore'][:, 1])
    assert np.allclose(r0['data_470'][off:], r1['data_470'], equal_nan=True)


def test_the_start_offset_survives_losing_the_raw_array(tmp_path):
    """A reloaded project has no `raw`, and must not fall back to the precut."""
    fp = _raw_csv(tmp_path)
    app = _app({'S': {'start': 3000, 'end': 0}})
    res = _run(app, 'S', fp)
    live = app._photometry_start_offset_samples('S')
    app.processed_data['S'] = {k: v for k, v in res.items() if k != 'raw'}
    assert app._photometry_start_offset_samples('S') == live


# ── Position ────────────────────────────────────────────────────────

def test_out_of_arena_tracking_before_the_start_is_dropped(tmp_path):
    """The head of the track must not reach pixel->cm calibration."""
    fp = _raw_csv(tmp_path)
    junk = 3000
    pos = _abel_csv(tmp_path, junk_frames=junk)

    r0 = _run(_app(), 'S', fp, pos)
    r1 = _run(_app({'S': {'start': junk, 'end': 0}}), 'S', fp, pos)

    px0, px1 = r0['position_px'], r1['position_px']
    assert np.nanmin(px0[:, 0]) < 50          # junk reached calibration
    assert np.nanmin(px1[:, 0]) >= 95         # and no longer does
    assert np.nanmin(px1[:, 1]) >= 95


def test_bounded_position_fills_the_arena_instead_of_a_sub_rectangle(tmp_path):
    """Calibration fitted to junk squashes the real session into part of the box."""
    fp = _raw_csv(tmp_path)
    junk = 3000
    pos = _abel_csv(tmp_path, junk_frames=junk)
    start_sec = junk / VIDEO_FPS

    def _real_span(res):
        beh = res['beh_synced']
        fs = float(res['photometry_fps'])
        after = np.arange(len(beh)) >= 0
        # Only look at the stretch that is real session in BOTH runs.
        elapsed = np.arange(len(beh)) / fs
        offset = res.get('start_offset_samples', 0) / fs
        after &= (elapsed + offset) >= start_sec + 5.0
        x = beh[after, 2]
        x = x[np.isfinite(x)]
        return float(np.max(x) - np.min(x))

    unbounded = _real_span(_run(_app(), 'S', fp, pos))
    bounded = _real_span(_run(_app({'S': {'start': junk, 'end': 0}}), 'S', fp, pos))
    assert bounded > unbounded + 5.0
    assert bounded == pytest.approx(51.0, abs=1.0)


def test_position_after_the_end_bound_is_dropped(tmp_path):
    fp = _raw_csv(tmp_path)
    pos = _abel_csv(tmp_path)
    end = 9000                                   # 300 s of video
    app = _app({'S': {'start': 0, 'end': end}})
    res = _run(app, 'S', fp, pos)
    covered = app._inverse_transform_boutframe_values(
        np.array([len(res['zscore']) - 1]), 'S')
    assert covered[0] <= end


# ── Spreadsheet import ──────────────────────────────────────────────

def test_a_seconds_column_is_never_read_as_frames():
    """'Real Start Time (sec)' sits next to 'Real Start Frame' in the real sheet.

    Picking it would put every start thirty times too early, which looks like a
    working import.
    """
    cols = ['cohort', 'animal-ID', 'sex', 'TX',
            'Real Start Time (sec)', 'Real Start Frame']
    id_col, start_col, end_col = G.FPAnalysisGUI._match_bounds_columns(cols)
    assert start_col == 'Real Start Frame'
    assert id_col == 'animal-ID'
    assert end_col is None


def test_start_and_end_frame_columns_are_both_found():
    cols = ['Subject', 'start_frame', 'end_frame']
    id_col, start_col, end_col = G.FPAnalysisGUI._match_bounds_columns(cols)
    assert (id_col, start_col, end_col) == ('Subject', 'start_frame', 'end_frame')


def test_a_sheet_with_title_rows_above_the_header_still_parses(tmp_path):
    """The real spreadsheet has two junk rows above its header row."""
    path = tmp_path / 'bounds.xlsx'
    rows = [[None] * 3,
            ['animal-ID', 'Real Start Time (sec)', 'Real Start Frame'],
            ['1-51', 30, 1200],
            ['1-B', 44, 1760]]
    pd.DataFrame(rows).to_excel(path, index=False, header=False)

    app = object.__new__(G.FPAnalysisGUI)
    for df in app._read_bounds_candidates(str(path)):
        id_col, start_col, end_col = app._match_bounds_columns(df.columns)
        if id_col is not None and start_col is not None:
            app.params = {'session_bounds_frames': {}}
            app.processed_data = {'1-51': {}, '1-B': {}}
            applied, unmatched, skipped = app._apply_bounds_table(
                df, id_col, start_col, end_col)
            assert sorted(applied) == ['1-51', '1-B']
            assert app.params['session_bounds_frames']['1-51'] == {'start': 1200, 'end': 0}
            return
    pytest.fail("no reading of the sheet produced usable columns")


# ── Non-ABEL position files ─────────────────────────────────────────

def _animalposition_csv(tmp_path, n_frames=30000, junk_frames=0, name='raw'):
    """A shared-clock AnimalPosition file (frame, computer timestamp, X, Y)."""
    rng = np.random.default_rng(2)
    frames = np.arange(n_frames)
    x = 100.0 + 200.0 * rng.random(n_frames)
    y = 100.0 + 200.0 * rng.random(n_frames)
    if junk_frames:
        x[:junk_frames] = 10.0 + 30.0 * rng.random(junk_frames)
        y[:junk_frames] = 10.0 + 30.0 * rng.random(junk_frames)
    path = tmp_path / f"{name}_AnimalPosition.csv"
    pd.DataFrame({'frame': frames,
                  'timestamp': 1000.0 + frames / VIDEO_FPS,
                  'X': x, 'Y': y}).to_csv(path, index=False)
    return str(path)


def test_bounds_also_trim_a_shared_clock_position_file(tmp_path):
    """The generic sync path leaves uncovered rows NaN but keeps their X/Y.

    Position analysis would still read them, and calibration still fits to
    them, so the rows have to go rather than merely lose their signal.
    """
    fp = _raw_csv(tmp_path)
    junk = 3000
    pos = _animalposition_csv(tmp_path, junk_frames=junk)

    r0 = _run(_app(), 'S', fp, pos)
    r1 = _run(_app({'S': {'start': junk, 'end': 0}}), 'S', fp, pos)

    assert np.nanmin(r0['position_px'][:, 0]) < 50
    assert np.nanmin(r1['position_px'][:, 0]) >= 95
    assert len(r1['beh']) == len(r0['beh']) - junk
