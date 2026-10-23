"""Mirror-FDR transient detection.

The MAD threshold (median + 2.5 x unscaled MAD, about 1.7 noise SD) reports
dozens of "spikes" a minute on pure noise and rises with the activity it is
meant to count. Mirror-FDR uses downward excursions as the noise null and picks
the lowest height at which they are at most q of the upward ones, measured
above a local baseline. These tests pin the properties that made it worth
adding: near-zero detections on noise, recovery of injected transients, no
spurious detections on a raised plateau, and amplitudes above the local
baseline rather than absolute peak heights.
"""
import numpy as np
import pytest

import fp_analysis_gui as G

FS = 30.0


def _app(method='mirror_fdr', q=0.05):
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'spike_method': method, 'spike_fdr_q': q,
                  'spike_threshold_mad': 2.5, 'spike_min_width': 2,
                  'spike_max_width': 90, 'spike_min_prominence': 0.5,
                  'spike_min_interval': 10}
    return app


def _noise(minutes=10, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, int(minutes * 60 * FS))


def _inject(x, times_s, amp, td=0.8, tr=0.1):
    t = np.arange(0, 5 * td, 1 / FS)
    k = np.exp(-t / td) - np.exp(-t / tr)
    k /= k.max()
    y = x.copy()
    peaks = []
    for t0 in times_s:
        i = int(t0 * FS)
        seg = k[:len(y) - i]
        y[i:i + len(seg)] += amp * seg
        peaks.append(i + int(np.argmax(k)))
    return y, np.array(peaks)


@pytest.mark.parametrize('seed', [0, 1])
def test_pure_noise_gives_almost_nothing(seed):
    x = _noise(seed=seed)
    mirror = _app().detect_spikes(x, FS)
    mad = _app('mad').detect_spikes(x, FS)
    assert mirror['method'] == 'mirror_fdr'
    assert mirror['spike_rate_per_min'] < 1.0
    # the old threshold calls pure noise a busy recording
    assert mad['spike_rate_per_min'] > 10.0


@pytest.mark.parametrize('seed', [0, 1])
def test_injected_transients_recovered_with_baseline_amplitude(seed):
    times = np.arange(10, 590, 6.0)                  # 10 per minute
    x, peaks = _inject(_noise(seed=seed), times, amp=6.0)
    x += 50.0                                         # absolute offset must not leak into amplitude
    r = _app().detect_spikes(x, FS)
    det = np.asarray(r['spike_indices'])
    hit = [np.any(np.abs(det - p) <= int(0.5 * FS)) for p in peaks]
    assert np.mean(hit) > 0.9
    assert len(det) < 1.1 * len(peaks)
    assert r['fdr'] <= 0.05
    assert 4.0 < np.median(r['spike_amplitudes']) < 8.0


def test_raised_plateau_is_not_spike_dense():
    x = _noise(seed=3)
    n = len(x)
    x[n // 3: n // 3 + int(90 * FS)] += 3.0
    inside = slice(n // 3 + int(15 * FS), n // 3 + int(75 * FS))
    det = np.asarray(_app().detect_spikes(x, FS)['spike_indices'], int)
    assert np.sum((det >= inside.start) & (det < inside.stop)) <= 1
    mad = np.asarray(_app('mad').detect_spikes(x, FS)['spike_indices'], int)
    assert np.sum((mad >= inside.start) & (mad < inside.stop)) > 20


def test_mad_method_unchanged_by_default():
    app = _app('mad')
    r = app.detect_spikes(_noise(seed=4), FS)
    assert 'method' not in r and np.isfinite(r['mad'])


def test_source_is_cited():
    assert 'Dombeck' in G.FPAnalysisGUI.SPIKE_MIRROR_FDR_SOURCE
