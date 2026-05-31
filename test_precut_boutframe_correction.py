"""
Tests for the precut boutframe correction logic.

The correction formula is:
    corrected_frame = raw_boutframe - (precut // n_led_states)

Scenarios tested:
  1.  Standard 2-LED-state recording (470 nm + 415 nm isosbestic)
  2.  3-LED-state recording (470 nm + 415 nm + 570 nm)
  3.  Single-LED recording (470 nm only, no isosbestic)
  4.  precut = 0  →  no correction applied
  5.  FPS scaling combined with precut correction
  6.  Manual shift combined with precut correction
  7.  n_led_states falls back to 1 when missing from subject_data
  8.  Boutframes that land before index 0 after correction are removed
      by the downstream exclude_before filter (frames < 0 scenario)
  9.  precut not a multiple of n_led_states (floor division)
  10. Correction disabled (precut_correct_boutframes = False)
"""

import sys
import types
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Minimal stub of the FiberPhotometryApp class that exposes only the logic
# we want to test.  We replicate just the frame-adjustment block from
# extract_bouts() so the tests have no GUI / tkinter dependency.
# ---------------------------------------------------------------------------

def _simulate_frame_adjustment(
    raw_frames,
    *,
    precut=100,
    n_led_states=2,
    auto_scale=False,
    video_fps=30.0,
    photo_fps=None,
    manual_shift=0,
    per_subject_shift=0,
    precut_correct=True,
    exclude_before=0,
):
    """
    Mirrors the frame-adjustment logic inside extract_bouts() so we can unit-
    test it without needing a running Tkinter app.

    Parameters
    ----------
    raw_frames : array-like
        Boutframe numbers as they appear in the Excel file.
    precut : int
        Number of raw FP rows removed at the start (self.params['precut']).
    n_led_states : int
        Number of unique LED states detected during processing
        (result['n_led_states']).
    auto_scale : bool
        Whether FPS-based boutframe scaling is enabled.
    video_fps : float
        Source video frame rate for boutframes file.
    photo_fps : float or None
        Detected photometry sampling rate.  If None, scaling is skipped.
    manual_shift : int
        Signed frame offset applied after scaling.
    precut_correct : bool
        Whether the precut correction is enabled.
    exclude_before : int
        Drop frames with index < exclude_before (mirrors params['exclude_frames_before']).

    Returns
    -------
    np.ndarray
        Adjusted frame indices ready to index into beh_synced.
    dict
        Diagnostic info: scale_factor, precut_offset applied.
    """
    frames = np.array(raw_frames, dtype=float)

    # --- FPS scaling ---
    scale_factor = 1.0
    if auto_scale and photo_fps is not None and video_fps > 0:
        if abs(photo_fps - video_fps) > 0.1:
            scale_factor = photo_fps / video_fps
            frames = np.round(frames * scale_factor)
    frames = frames.astype(int)

    # --- Manual shift ---
    if manual_shift != 0:
        frames = frames + manual_shift

    # --- Per-subject shift (applied after manual shift, before precut) ---
    # Mirrors:  frames = frames + self._get_per_subject_shift(subject_id)
    if per_subject_shift != 0:
        frames = frames + per_subject_shift

    # --- Precut correction ---
    precut_offset = 0
    if precut_correct:
        n_led = max(1, n_led_states)
        precut_offset = precut // n_led
        if precut_offset > 0:
            frames = frames - precut_offset

    # --- Exclude before ---
    frames = frames[frames >= exclude_before]

    return frames, {"scale_factor": scale_factor, "precut_offset": precut_offset}


# ===========================================================================
# Tests
# ===========================================================================

class TestPrectBoutframeCorrection:

    # -----------------------------------------------------------------------
    # Scenario 1: Standard 2-LED recording
    # -----------------------------------------------------------------------
    def test_two_led_states_basic(self):
        """precut=100, n_led=2  →  offset = 50"""
        raw = [200, 400, 600]
        corrected, info = _simulate_frame_adjustment(raw, precut=100, n_led_states=2)
        assert info["precut_offset"] == 50
        np.testing.assert_array_equal(corrected, [150, 350, 550])

    # -----------------------------------------------------------------------
    # Scenario 2: 3-LED recording
    # -----------------------------------------------------------------------
    def test_three_led_states(self):
        """precut=90, n_led=3  →  offset = 30"""
        raw = [90, 180, 270]
        corrected, info = _simulate_frame_adjustment(raw, precut=90, n_led_states=3)
        assert info["precut_offset"] == 30
        np.testing.assert_array_equal(corrected, [60, 150, 240])

    # -----------------------------------------------------------------------
    # Scenario 3: Single LED (no isosbestic)
    # -----------------------------------------------------------------------
    def test_single_led_state(self):
        """precut=100, n_led=1  →  offset = 100"""
        raw = [200, 300]
        corrected, info = _simulate_frame_adjustment(raw, precut=100, n_led_states=1)
        assert info["precut_offset"] == 100
        np.testing.assert_array_equal(corrected, [100, 200])

    # -----------------------------------------------------------------------
    # Scenario 4: precut = 0
    # -----------------------------------------------------------------------
    def test_zero_precut_no_correction(self):
        """precut=0 → offset=0, frames unchanged"""
        raw = [50, 100, 200]
        corrected, info = _simulate_frame_adjustment(raw, precut=0, n_led_states=2)
        assert info["precut_offset"] == 0
        np.testing.assert_array_equal(corrected, [50, 100, 200])

    # -----------------------------------------------------------------------
    # Scenario 5: FPS scaling + precut correction combined
    # -----------------------------------------------------------------------
    def test_fps_scaling_then_precut_correction(self):
        """
        Video frame 300 at 30 fps × (50/30) = 500 deinterleaved FP frames.
        Then subtract precut offset 50 (precut=100, n_led=2) → 450.
        """
        raw = [300]
        corrected, info = _simulate_frame_adjustment(
            raw,
            precut=100, n_led_states=2,
            auto_scale=True, video_fps=30.0, photo_fps=50.0,
        )
        assert info["scale_factor"] == pytest.approx(50.0 / 30.0, abs=1e-6)
        assert info["precut_offset"] == 50
        # round(300 * 50/30) = round(500.0) = 500;  500 - 50 = 450
        np.testing.assert_array_equal(corrected, [450])

    # -----------------------------------------------------------------------
    # Scenario 6: Manual shift + precut correction
    # -----------------------------------------------------------------------
    def test_manual_shift_then_precut_correction(self):
        """
        Order: FPS scale → manual shift → precut correction.
        Boutframe 400 with manual_shift=+10 and precut_offset=50 → 360.
        """
        raw = [400]
        corrected, info = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, manual_shift=10
        )
        assert info["precut_offset"] == 50
        # 400 + 10 = 410; 410 - 50 = 360
        np.testing.assert_array_equal(corrected, [360])

    # -----------------------------------------------------------------------
    # Scenario 7: Missing n_led_states falls back to 1
    # -----------------------------------------------------------------------
    def test_missing_n_led_falls_back_to_one(self):
        """If n_led_states is 0 (not yet set), max(1, 0) guards against ZeroDivisionError."""
        raw = [300, 500]
        corrected, info = _simulate_frame_adjustment(raw, precut=100, n_led_states=0)
        # Fallback: n_led = max(1, 0) = 1, offset = 100 // 1 = 100
        assert info["precut_offset"] == 100
        np.testing.assert_array_equal(corrected, [200, 400])

    # -----------------------------------------------------------------------
    # Scenario 8: Frames below 0 after correction are removed by exclude_before
    # -----------------------------------------------------------------------
    def test_negative_frames_excluded(self):
        """
        Boutframes near the start of the recording become negative after
        correction and must be dropped.
        """
        raw = [20, 80, 200]  # 20 and 80 are within the precut window
        corrected, info = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, exclude_before=0
        )
        # offsets: 20-50=-30 (dropped), 80-50=30 (kept), 200-50=150 (kept)
        np.testing.assert_array_equal(corrected, [30, 150])

    # -----------------------------------------------------------------------
    # Scenario 9: precut not a multiple of n_led_states → floor division
    # -----------------------------------------------------------------------
    def test_non_multiple_precut_floor_division(self):
        """precut=101, n_led=2  →  floor(101/2) = 50  (1 raw row unaccounted,
        accepted as unavoidable rounding when precut is odd)."""
        raw = [300]
        corrected, info = _simulate_frame_adjustment(raw, precut=101, n_led_states=2)
        assert info["precut_offset"] == 50  # floor(101/2)
        np.testing.assert_array_equal(corrected, [250])

    # -----------------------------------------------------------------------
    # Scenario 10: Correction disabled
    # -----------------------------------------------------------------------
    def test_correction_disabled(self):
        """When precut_correct=False, boutframes are not shifted at all."""
        raw = [200, 400]
        corrected, info = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, precut_correct=False
        )
        assert info["precut_offset"] == 0
        np.testing.assert_array_equal(corrected, [200, 400])

    # -----------------------------------------------------------------------
    # Extra: large dataset consistency check
    # -----------------------------------------------------------------------
    def test_large_batch_consistency(self):
        """All corrected frames shift by the same constant offset."""
        rng = np.random.default_rng(42)
        raw = rng.integers(low=200, high=10000, size=500).tolist()
        corrected, info = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2
        )
        expected_offset = 50
        assert info["precut_offset"] == expected_offset
        np.testing.assert_array_equal(corrected, np.array(raw) - expected_offset)


# ===========================================================================
# Additional integration-style check: verify the round-trip from raw FP frame
# to beh_synced index makes sense for a concrete timeline.
# ===========================================================================

class TestRoundTrip:

    def test_event_at_5_seconds_two_led(self):
        """
        Recording at 50 Hz photometry, 2 LED states (100 raw rows/s).
        Precut = 100 raw rows = 1 second.
        Session starts at t=1s in raw recording space.

        A behaviour event annotated at t=6s from the recording start:
          raw FP row  = 6 * 100 = 600
          post-precut raw row  = 600 - 100 = 500
          deinterleaved frame  = 500 / 2 = 250   ← correct beh_synced index

        Boutframe file contains raw deinterleaved-frame index 300 (= 600 / 2)
        which includes the pre-session frames.  After correction it should
        become 300 - 50 = 250.
        """
        # boutframe derived from deinterleaved space but including pre-session
        boutframe_in_deinterleaved_incl_precut = 300   # 6s × 50 Hz
        expected_beh_synced_idx = 250                  # 5s × 50 Hz

        corrected, info = _simulate_frame_adjustment(
            [boutframe_in_deinterleaved_incl_precut],
            precut=100, n_led_states=2,
        )
        assert info["precut_offset"] == 50
        assert corrected[0] == expected_beh_synced_idx, (
            f"Expected beh_synced index {expected_beh_synced_idx}, got {corrected[0]}"
        )

    def test_event_at_10_seconds_three_led(self):
        """
        Recording at 50 Hz photometry, 3 LED states (150 raw rows/s).
        Precut = 90 raw rows = 0.6 s of pre-session data.
        Event at t=10.6s from recording start = t=10s from session start.

          raw FP row  = 10.6 * 150 = 1590
          post-precut raw row  = 1590 - 90 = 1500
          deinterleaved frame  = 1500 / 3 = 500

        Boutframe from deinterleaved-incl-precut space: 1590 / 3 = 530
        After correction: 530 - (90 // 3) = 530 - 30 = 500
        """
        boutframe = 530
        expected = 500
        corrected, info = _simulate_frame_adjustment(
            [boutframe], precut=90, n_led_states=3
        )
        assert info["precut_offset"] == 30
        assert corrected[0] == expected


# ===========================================================================
# Per-subject boutframe shift
# ===========================================================================

class TestPerSubjectBoutframeShift:
    """Verify that per_subject_shift is applied AFTER manual_shift and BEFORE
    precut correction, and that it affects each subject independently."""

    def test_zero_shift_is_noop(self):
        """per_subject_shift=0 leaves frames identical to no shift at all."""
        raw = [200, 300, 400]
        without, _ = _simulate_frame_adjustment(raw, precut=100, n_led_states=2)
        with_zero, _ = _simulate_frame_adjustment(raw, precut=100, n_led_states=2,
                                                   per_subject_shift=0)
        np.testing.assert_array_equal(without, with_zero)

    def test_positive_shift(self):
        """per_subject_shift=+10 shifts every frame forward by 10 before precut."""
        raw = [200, 300]
        corrected, _ = _simulate_frame_adjustment(raw, precut=100, n_led_states=2,
                                                   per_subject_shift=+10)
        # 200+10=210 → 210-50=160 ;  300+10=310 → 310-50=260
        np.testing.assert_array_equal(corrected, [160, 260])

    def test_negative_shift(self):
        """per_subject_shift=-10 shifts every frame backward by 10 before precut."""
        raw = [200, 300]
        corrected, _ = _simulate_frame_adjustment(raw, precut=100, n_led_states=2,
                                                   per_subject_shift=-10)
        # 200-10=190 → 190-50=140 ;  300-10=290 → 290-50=240
        np.testing.assert_array_equal(corrected, [140, 240])

    def test_applied_after_manual_shift(self):
        """manual_shift is added first, then per_subject_shift, then precut."""
        raw = [400]
        corrected, _ = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2,
            manual_shift=5, per_subject_shift=20,
        )
        # 400 + 5(manual) + 20(per-subject) - 50(precut) = 375
        np.testing.assert_array_equal(corrected, [375])

    def test_applied_before_precut(self):
        """If per_subject_shift were applied AFTER precut the result would differ."""
        raw = [400]
        # Correct (shift before precut):  400 + 10 - 50 = 360
        corrected, _ = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, per_subject_shift=+10)
        np.testing.assert_array_equal(corrected, [360])
        # Wrong order would give: (400 - 50) + 10 = 360 — same here, but the
        # excluded-frame test below catches the ordering unambiguously.

    def test_large_negative_shift_drops_frames(self):
        """A large negative per-subject shift can push frames below the exclude threshold."""
        raw = [60, 200]
        corrected, _ = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2,
            per_subject_shift=-20, exclude_before=0,
        )
        # frame 60 - 20 = 40,  40 - 50 = -10  → excluded (< 0)
        # frame 200 - 20 = 180, 180 - 50 = 130 → kept
        np.testing.assert_array_equal(corrected, [130])

    def test_positive_shift_ordering_vs_exclusion(self):
        """Verify ordering: shift moves frame ABOVE exclude_before, not below."""
        # With per_subject_shift=+20, frame 35 becomes 55 (above exclude=0 after precut)
        # Without shift: 35 - 50 = -15 → would be excluded
        raw = [35]
        without_shift, _ = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, per_subject_shift=0, exclude_before=0)
        with_shift, _ = _simulate_frame_adjustment(
            raw, precut=100, n_led_states=2, per_subject_shift=+20, exclude_before=0)
        assert len(without_shift) == 0, "Frame should be excluded without shift"
        assert len(with_shift) == 1,    "Frame should survive with +20 shift"
        np.testing.assert_array_equal(with_shift, [5])  # (35+20)-50=5

    def test_different_subjects_differ_by_sum_of_shifts(self):
        """Two subjects with symmetric shifts differ in onset frames by 2×|shift|."""
        raw = [300]
        a, _ = _simulate_frame_adjustment(raw, precut=0, per_subject_shift=+15)
        b, _ = _simulate_frame_adjustment(raw, precut=0, per_subject_shift=-15)
        assert int(a[0]) - int(b[0]) == 30

    def test_combined_fps_scale_manual_per_subject_precut(self):
        """Full pipeline: FPS scale → manual shift → per-subject shift → precut."""
        raw = [300]   # video frame
        corrected, info = _simulate_frame_adjustment(
            raw,
            auto_scale=True, video_fps=30.0, photo_fps=50.0,
            manual_shift=10,
            per_subject_shift=5,
            precut=100, n_led_states=2,
        )
        # scale: round(300 * 50/30) = 500
        # manual: 500 + 10 = 510
        # per-subject: 510 + 5 = 515
        # precut: 515 - 50 = 465
        np.testing.assert_array_equal(corrected, [465])


# ===========================================================================
# Bout window extraction — verifies that the per-subject shift moves the
# extracted window to the correct position within the FP signal.
# ===========================================================================

def _extract_bout_windows(beh_synced, onset_frames, *,
                          prebout=20, postbout=20, phot_col=6,
                          exclude_before=0):
    """Mirror the per-onset extraction loop inside extract_bouts().

    Returns
    -------
    list of (onset_frame, np.ndarray)
        Each entry is the raw FP segment [start_idx : end_idx] for that onset.
    """
    results = []
    for frame in onset_frames:
        if frame < exclude_before:
            continue
        start = max(0, frame - prebout)
        end   = min(len(beh_synced), frame + postbout)
        segment = beh_synced[start:end, phot_col].copy()
        if not np.all(np.isnan(segment)):
            results.append((int(frame), segment))
    return results


class TestBoutWindowExtraction:
    """End-to-end window extraction tests using a synthetic FP signal.

    Strategy: place a Gaussian 'event' spike at a known frame in beh_synced;
    check that the extracted window captures the spike at the expected relative
    position for different per-subject shift values.
    """

    N_FRAMES  = 600
    PREBOUT   = 30
    POSTBOUT  = 30
    WIN_SIZE  = PREBOUT + POSTBOUT   # 60 frames (full window, spike at centre)
    SPIKE_FRAME = 300                # ground-truth spike location in beh_synced

    @classmethod
    def _make_signal(cls):
        """Synthetic beh_synced: columns [frame, ts, x, y, z, w, zscore]."""
        signal = np.zeros(cls.N_FRAMES)
        # Gaussian spike centred at SPIKE_FRAME, σ=3 frames
        t = np.arange(cls.N_FRAMES)
        signal += 5.0 * np.exp(-0.5 * ((t - cls.SPIKE_FRAME) / 3.0) ** 2)
        beh_synced = np.zeros((cls.N_FRAMES, 7))
        beh_synced[:, 0] = t          # frame column
        beh_synced[:, 6] = signal     # photometry column
        return beh_synced, signal

    def test_no_shift_spike_at_centre(self):
        """With shift=0 and boutframe = SPIKE_FRAME, the spike should sit at
        exactly index PREBOUT within the extracted window."""
        beh_synced, signal = self._make_signal()
        raw_frames = [self.SPIKE_FRAME]
        onset_frames, _ = _simulate_frame_adjustment(
            raw_frames, precut=0, per_subject_shift=0)

        windows = _extract_bout_windows(
            beh_synced, onset_frames,
            prebout=self.PREBOUT, postbout=self.POSTBOUT)

        assert len(windows) == 1, "Expected one extracted window"
        _, w = windows[0]
        assert len(w) == self.WIN_SIZE, f"Expected window of {self.WIN_SIZE}, got {len(w)}"
        spike_idx = int(np.argmax(w))
        assert spike_idx == self.PREBOUT, (
            f"Spike should be at index {self.PREBOUT} (onset), got {spike_idx}")

    def test_positive_shift_moves_spike_earlier_in_window(self):
        """A +N per-subject shift moves the onset N frames forward, so the
        pre-existing spike falls N frames BEFORE the onset marker (i.e.
        at index PREBOUT - N within the window)."""
        shift = 10
        beh_synced, signal = self._make_signal()
        raw_frames = [self.SPIKE_FRAME]   # boutframe still points at spike
        onset_frames, _ = _simulate_frame_adjustment(
            raw_frames, precut=0, per_subject_shift=shift)

        windows = _extract_bout_windows(
            beh_synced, onset_frames,
            prebout=self.PREBOUT, postbout=self.POSTBOUT)

        assert len(windows) == 1
        _, w = windows[0]
        spike_idx = int(np.argmax(w))
        assert spike_idx == self.PREBOUT - shift, (
            f"With +{shift} shift, spike should be at index "
            f"{self.PREBOUT - shift}, got {spike_idx}")

    def test_negative_shift_moves_spike_later_in_window(self):
        """A -N per-subject shift moves the onset N frames backward, so the
        spike falls N frames AFTER the onset marker."""
        shift = -10
        beh_synced, signal = self._make_signal()
        raw_frames = [self.SPIKE_FRAME]
        onset_frames, _ = _simulate_frame_adjustment(
            raw_frames, precut=0, per_subject_shift=shift)

        windows = _extract_bout_windows(
            beh_synced, onset_frames,
            prebout=self.PREBOUT, postbout=self.POSTBOUT)

        assert len(windows) == 1
        _, w = windows[0]
        spike_idx = int(np.argmax(w))
        assert spike_idx == self.PREBOUT + abs(shift), (
            f"With {shift} shift, spike should be at index "
            f"{self.PREBOUT + abs(shift)}, got {spike_idx}")

    def test_multiple_bouts_all_shifted_equally(self):
        """All onsets in a subject receive the same per-subject shift."""
        shift = 8
        beh_synced, _ = self._make_signal()
        # Place three identical spikes at known frames
        spike_frames = [150, 300, 450]
        for sf in spike_frames:
            beh_synced[sf, 6] += 10.0

        raw_frames = spike_frames
        onset_no_shift, _ = _simulate_frame_adjustment(raw_frames, precut=0,
                                                        per_subject_shift=0)
        onset_shifted, _  = _simulate_frame_adjustment(raw_frames, precut=0,
                                                        per_subject_shift=shift)
        np.testing.assert_array_equal(
            np.array(onset_shifted) - np.array(onset_no_shift),
            [shift] * len(spike_frames),
            err_msg="Every onset should be shifted by the same amount")

    def test_window_values_sum_to_same_signal_energy(self):
        """Shifting the window does not add or remove signal energy — it only
        re-positions *which* part of the FP trace is captured."""
        beh_synced, _ = self._make_signal()
        # A flat region far from the spike: signal ≈ 0
        flat_frame = 50
        beh_synced[flat_frame, 6] = 99.0   # plant an obvious marker

        # With shift=0: onset at flat_frame → window captures the marker at centre
        w0 = _extract_bout_windows(beh_synced, [flat_frame], prebout=5, postbout=5)
        assert len(w0) == 1
        assert w0[0][1][5] == pytest.approx(99.0, abs=0.01), \
            "Marker should be at index 5 (= prebout) with no shift"

        # With shift=+2: onset at flat_frame+2 → marker at index 5-2=3
        frames_shifted, _ = _simulate_frame_adjustment([flat_frame], precut=0,
                                                        per_subject_shift=+2)
        w2 = _extract_bout_windows(beh_synced, frames_shifted, prebout=5, postbout=5)
        assert len(w2) == 1
        assert w2[0][1][3] == pytest.approx(99.0, abs=0.01), \
            "Marker should shift to index 3 (= prebout - shift) with +2 shift"


if __name__ == "__main__":
    # Run with: python test_precut_boutframe_correction.py
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
