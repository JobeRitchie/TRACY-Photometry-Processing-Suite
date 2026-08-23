"""The recording-length cap is a duration, not a row count.

'maxlengthframe' counted raw rows and was doubled on the assumption of a
two-state interleave, so the same setting meant a different recording length on
every rig: 20 000 rows is 500 s at 40 Hz raw and 100 s at 200 Hz.  A project
holding cohorts from both would have been trimmed to different lengths.

These tests pin the replacement -- 'maxlengthseconds', read against the raw
timestamp column -- at two raw rates, plus the migration for projects saved with
the old frame cap.
"""
import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G


def _raw_csv(tmp_path, row_dt, n_rows, names=('G0', 'R1')):
    """An interleaved FPData CSV whose rows are *row_dt* seconds apart."""
    rng = np.random.default_rng(0)
    led_cycle = [1, 2, 4]
    cols = ['FrameCounter', 'SystemTimestamp', 'LedState', 'ComputerTimestamp'] + list(names)
    rows = []
    for f in range(n_rows):
        vals = [f, 1000.0 + f * row_dt, led_cycle[f % 3], 50000.0 + f * 50.0]
        vals += [1.0 + 0.1 * (j + 1) + 0.05 * np.sin(f / 31.0 + j) + 0.01 * rng.standard_normal()
                 for j in range(len(names))]
        rows.append(vals)
    path = tmp_path / f"raw_{row_dt}.csv"
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    return str(path)


class _Flag:
    def __init__(self, v): self.v = v
    def get(self): return self.v


def _app(max_seconds):
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
    app.detected_photometry_fps = None
    app.processing_summary = {
        'fallbacks_used': [], 'mismatched_led_samples': [],
        'zero_variance_channels': [], 'no_position_data': [],
    }
    app.params = {'precut': 100, 'maxlengthseconds': max_seconds,
                  'processing_rolling_avg_enabled': False}
    return app


def _analysed_seconds(data):
    return data['corrected_470'].shape[0] / float(data['photometry_fps'])


# 40 Hz and 200 Hz raw rows -- 300 s of recording at each.
RATES = [(0.025, 12000), (0.005, 60000)]


@pytest.mark.parametrize("row_dt,n_rows", RATES)
def test_cap_trims_the_same_duration_at_every_rate(tmp_path, row_dt, n_rows):
    path = _raw_csv(tmp_path, row_dt, n_rows)
    data = _app(20.0).process_fp_data("s", path)
    assert abs(_analysed_seconds(data) - 20.0) < 0.5


@pytest.mark.parametrize("row_dt,n_rows", RATES)
def test_zero_means_no_cap(tmp_path, row_dt, n_rows):
    path = _raw_csv(tmp_path, row_dt, n_rows)
    data = _app(0.0).process_fp_data("s", path)
    full = (n_rows - 100) * row_dt          # everything after the precut
    assert abs(_analysed_seconds(data) - full) < 0.5


@pytest.mark.parametrize("row_dt,n_rows", RATES)
def test_a_cap_longer_than_the_recording_keeps_everything(tmp_path, row_dt, n_rows):
    path = _raw_csv(tmp_path, row_dt, n_rows)
    data = _app(10000.0).process_fp_data("s", path)
    full = (n_rows - 100) * row_dt
    assert abs(_analysed_seconds(data) - full) < 0.5


def _migrating_app():
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'maxlengthseconds': 0.0}
    app.processed_data = {'S0': {'photometry_fps': 30.0}}
    app.log_message = lambda *a, **k: None
    return app


def test_the_old_default_frame_cap_migrates_to_no_cap():
    """2e7 frames was larger than any recording; it must not become a real cap."""
    app = _migrating_app()
    app._migrate_maxlength_to_seconds({'maxlengthframe': 20000000})
    assert app.params['maxlengthseconds'] == 0.0


def test_a_hand_set_frame_cap_migrates_at_the_project_rate():
    app = _migrating_app()
    app._migrate_maxlength_to_seconds({'maxlengthframe': 3000})
    assert app.params['maxlengthseconds'] == pytest.approx(100.0, abs=0.01)


def test_a_project_already_in_seconds_is_left_alone():
    app = _migrating_app()
    app.params['maxlengthseconds'] = 45.0
    app._migrate_maxlength_to_seconds({'maxlengthseconds': 45.0, 'maxlengthframe': 3000})
    assert app.params['maxlengthseconds'] == 45.0
