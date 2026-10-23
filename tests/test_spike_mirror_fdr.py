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


def test_baseline_window_is_adjustable():
    x, _ = _inject(_noise(seed=5), np.arange(10, 590, 6.0), amp=6.0)
    x += 4.0 * np.sin(np.arange(len(x)) / FS * 2 * np.pi / 60)   # 60 s drift
    short = _app(); short.params['spike_baseline_window_s'] = 5.0
    long_ = _app(); long_.params['spike_baseline_window_s'] = 120.0
    b_short = short.detect_spikes(x, FS)['baseline']
    b_long = long_.detect_spikes(x, FS)['baseline']
    # a short window follows the drift, a long one cannot
    assert np.std(b_short) > 2 * np.std(b_long)


def _decay_events(times_amps, td, seed):
    x = _noise(seed=seed)
    t = np.arange(0, 6 * td, 1 / FS)
    k = np.exp(-t / td) - np.exp(-t / 0.1); k /= k.max()
    peaks = []
    for t0, a in times_amps:
        i = int(t0 * FS); seg = k[:len(x) - i]; x[i:i + len(seg)] += a * seg
        peaks.append(i + int(np.argmax(k)))
    return x, np.array(peaks)


@pytest.mark.parametrize('seed', [0, 1])
def test_burst_aware_counts_event_cut_short_by_its_successor(seed):
    # a 6-SD event followed 0.6 s later by a 10-SD one, while it is still decaying
    starts = np.arange(10, 590, 30.0)
    pairs = [(t, 6.0) for t in starts] + [(t + 0.6, 10.0) for t in starts]
    x, peaks = _decay_events(pairs, td=2.0, seed=seed)
    first = peaks[:len(peaks) // 2]
    app = _app(); app.params['spike_burst_aware'] = True
    old = _app(); old.params['spike_burst_aware'] = False

    def found(a):
        d = np.asarray(a.detect_spikes(x, FS)['spike_indices'])
        return np.mean([np.any(np.abs(d - p) <= int(0.3 * FS)) for p in first])
    assert found(app) >= 0.7
    assert found(app) > found(old) + 0.15


@pytest.mark.parametrize('seed', [0, 1])
def test_burst_aware_does_not_split_single_slow_events(seed):
    x, peaks = _decay_events([(t, 8.0) for t in np.arange(10, 590, 10.0)], td=4.0, seed=seed)
    app = _app(); app.params['spike_burst_aware'] = True
    assert len(app.detect_spikes(x, FS)['spike_indices']) <= 1.1 * len(peaks)
