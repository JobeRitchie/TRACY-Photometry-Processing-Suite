"""TTL-derived bouts land on the TTL's own clock time.

A TTL frame is found by matching its ComputerTimestamp to a row of
``beh_synced``, so it is already the index extract_bouts slices.  It used to be
written to boutframes.xlsx as that row and then pushed through the boutframe
transform like a hand-scored video frame: on the photometry grid the
recording-start correction moved every TTL bout early by the precut (a fear
response 20 s after the tone onset showed up at 21.7 s), and at 20 Hz the FPS
scaling compressed them as well.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp_analysis_gui as G  # noqa: E402
from test_session_bounds import _app, _Flag  # noqa: E402

RESPONSE_S = 20.0
ONSETS_S = [60.0, 150.0, 240.0]


def _write_recording(folder, fs_channel, seconds=330.0):
    """Two-LED FPData with a G0 response RESPONSE_S after each TTL onset."""
    rng = np.random.default_rng(0)
    fs_raw = 2 * fs_channel
    n = int(seconds * fs_raw)
    t = np.arange(n) / fs_raw
    led = np.where(np.arange(n) % 2 == 0, 2, 1)
    led[0] = 7
    sys_ts = 11768.0 + t
    comp_ms = 50366140.0 + t * 1000.0
    g0 = 0.05 + 0.0005 * rng.standard_normal(n)
    g1 = 0.03 + 0.0005 * rng.standard_normal(n)
    for on in ONSETS_S:
        bump = np.exp(-0.5 * ((t - on - RESPONSE_S) / 0.15) ** 2)
        g0 = g0 + np.where(led == 2, 0.02 * bump, 0.0)
    fp = os.path.join(folder, 'S1FPData.csv')
    pd.DataFrame({'FrameCounter': np.arange(n), 'SystemTimestamp': sys_ts,
                  'LedState': led, 'ComputerTimestamp': comp_ms,
                  'G0': g0, 'G1': g1}).to_csv(fp, index=False)
    rows = []
    for on in ONSETS_S:
        rows += [(50366140.0 + on * 1000.0, 1), (50366140.0 + (on + 22) * 1000.0, 1)]
    ttl = os.path.join(folder, 'S1_TS_1.csv')
    pd.DataFrame(rows).to_csv(ttl, header=False, index=False)
    return fp, ttl


def _peak_latencies(folder, fs_channel, mode):
    fp, ttl = _write_recording(str(folder), fs_channel)
    app = _app(boutframe_start_correction=mode, boutframe_fps_scaling=mode,
               preboutseconds=5.0, postboutseconds=25.0,
               preboutframes=int(5 * fs_channel), postboutframes=int(25 * fs_channel),
               ttl_format='pairs_start', boutframes_video_fps=30.0)
    app.ttl_path_var = _Flag(ttl)
    res = app.process_fp_data('S1', fp)
    app.processed_data['S1'] = res
    fs = res['photometry_fps']
    pre = int(round(5.0 * fs))
    traces = res['bouts']['TTL']['Ch0']
    assert len(traces) == len(ONSETS_S)
    return np.array([(np.argmax(np.asarray(tr)) - pre) / fs for tr in traces])


@pytest.mark.parametrize('fs_channel', [30.0, 20.0, 60.0])
@pytest.mark.parametrize('mode', ['auto', 'on', 'off'])
def test_ttl_response_lands_at_its_clock_latency(tmp_path, fs_channel, mode):
    lat = _peak_latencies(tmp_path, fs_channel, mode)
    assert np.max(np.abs(lat - RESPONSE_S)) <= 1.5 / fs_channel, lat


def test_manual_shift_still_moves_ttl_bouts(tmp_path):
    fp, ttl = _write_recording(str(tmp_path), 30.0)
    app = _app(preboutseconds=5.0, postboutseconds=25.0, preboutframes=150,
               postboutframes=750, ttl_format='pairs_start', boutframe_manual_shift=30)
    app.ttl_path_var = _Flag(ttl)
    res = app.process_fp_data('S1', fp)
    fs = res['photometry_fps']
    lat = [(np.argmax(np.asarray(tr)) - 150) / fs for tr in res['bouts']['TTL']['Ch0']]
    # Shifting the bout 1 s later moves the response 1 s earlier in the window.
    assert np.max(np.abs(np.array(lat) - (RESPONSE_S - 1.0))) <= 0.05, lat
