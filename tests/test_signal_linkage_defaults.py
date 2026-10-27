"""Signal Linkage smoothing is set in seconds and tuned for slow sensors.

The SG window used to be 11 *frames*: 0.37 s at 30 Hz, 0.55 s at 20 Hz. On real
drinking GCaMP and GRAB-NE startle data that window missed most responses a 2 s
window found, at the same false-positive rate on random onsets. These tests pin
the three properties that came out of that tuning:

* the window means the same duration at any sampling rate;
* reliability is computed on smoothed bouts, so a slow, noisy but repeatable
  response scores clearly above random onsets;
* the reported latency of a fast rise is not dragged by the wide test window.
"""
import numpy as np

import fp_analysis_gui as G


def _app():
    return G.FPAnalysisGUI.__new__(G.FPAnalysisGUI)


def _cfg(fps, **kw):
    cfg = {'fps': fps, 'sg_win_s': 2.0, 'sg_poly': 3, 'search_s': 1.0,
           'base_s': 1.0, 'resp_s': 1.0, 'relia_s': 1.5, 'n_perm': 0, 'seed': 7}
    cfg.update(kw)
    return cfg


def _bouts(fps, response, n=60, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pre = post = int(round(5 * fps))
    t = (np.arange(pre + post) - pre) / fps
    return rng.normal(0, noise, (n, pre + post)) + response(t), pre


def _ramp(t):
    return 0.6 * np.clip(t / 2.0, 0, 1)


def _startle(t):
    return np.where(t >= 0, 3.0 * np.exp(-np.clip(t, 0, None) / 1.5), 0.0)


def test_window_is_the_same_duration_at_20_and_30_hz():
    app = _app()
    for fps in (20.0, 30.0):
        mat, pre = _bouts(fps, _ramp)
        m = app._sig_linkage_one(mat, [], pre, mat.shape[1], _cfg(fps))
        assert abs(m['win'] / fps - 2.0) < 0.1


def test_slow_response_reliability_clears_random_onsets():
    app = _app()
    fps = 30.0
    resp, pre = _bouts(fps, _ramp)
    null, _ = _bouts(fps, lambda t: 0 * t, seed=1)
    r_resp = app._sig_linkage_one(resp, [], pre, resp.shape[1], _cfg(fps))['reliability']
    r_null = app._sig_linkage_one(null, [], pre, null.shape[1], _cfg(fps))['reliability']
    assert r_resp >= 0.15 > 0.05 > abs(r_null)


def test_fast_rise_latency_is_at_onset():
    app = _app()
    fps = 20.0
    mat, pre = _bouts(fps, _startle, noise=0.3)
    m = app._sig_linkage_one(mat, [], pre, mat.shape[1], _cfg(fps))
    assert m['peak_rate'] > 0
    assert abs(m['peak_lat']) <= 0.15
