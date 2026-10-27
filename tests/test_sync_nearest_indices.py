"""synchronize_behavior's nearest-sample lookup must match the per-frame argmin.

The lookup used to search the whole photometry clock once per behavior frame,
which is quadratic in session length (a 40-minute session spent ~100 s there).
The sorted replacement has to return the very same index for every frame --
ties to the lowest index included, since a stalled clock (NSF07) repeats
timestamps and the old argmin picked the first of them.
"""

import numpy as np
import pytest

import fp_analysis_gui

_nearest = fp_analysis_gui.FPAnalysisGUI._nearest_sample_indices


def _argmin_reference(ref, targets):
    out = np.full(len(targets), -1, dtype=np.intp)
    for i, t in enumerate(targets):
        if not np.isnan(t):
            out[i] = np.argmin(np.abs(ref - t))
    return out


@pytest.mark.parametrize('seed', range(20))
def test_matches_argmin_on_messy_clocks(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 400))
    ref = np.cumsum(rng.uniform(0, 0.05, n))
    if seed % 3 == 0:
        ref[n // 3: n // 2] = ref[n // 3]          # stalled clock: repeats
    if seed % 4 == 0:
        ref = ref[rng.permutation(n)]              # out of order
    ref = np.round(ref, 2)                         # exact ties between neighbours
    targets = np.round(rng.uniform(ref.min() - 1, ref.max() + 1, 300), 3)
    targets[::17] = np.nan
    assert np.array_equal(_nearest(ref, targets), _argmin_reference(ref, targets))


def test_equidistant_neighbours_go_to_lower_index():
    ref = np.array([0.0, 1.0, 2.0])
    assert list(_nearest(ref, np.array([0.5, 1.5]))) == [0, 1]


def test_nan_in_reference_keeps_argmin_behaviour():
    ref = np.array([0.0, np.nan, 2.0])
    targets = np.array([1.9, 0.1])
    assert np.array_equal(_nearest(ref, targets), _argmin_reference(ref, targets))


def test_empty_inputs():
    assert list(_nearest(np.array([]), np.array([1.0]))) == [-1]
    assert list(_nearest(np.array([1.0]), np.array([np.nan]))) == [-1]
