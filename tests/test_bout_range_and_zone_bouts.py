"""Bout range limiter ("40-50") and zone entries exposed as bout behaviors."""
import numpy as np
import pytest

import fp_analysis_gui as G


def _app():
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    return app


@pytest.mark.parametrize("text, expect", [
    ("all", None), ("", None), ("0", None), ("10", 10), ("bad", None),
    ("1-10", 10), ("-10", 10),
])
def test_plain_limits(text, expect):
    assert _app()._get_max_bouts_limit(text) == expect


@pytest.mark.parametrize("text, start, stop", [
    ("40-50", 39, 50), ("40:50", 39, 50), ("40 \u2013 50", 39, 50),
    ("5-", 4, None), ("2 to 3", 1, 3),
])
def test_range_limits(text, start, stop):
    lim = _app()._get_max_bouts_limit(text)
    assert isinstance(lim, G._BoutRange)
    assert (lim.start, lim.stop) == (start, stop)
    assert _app()._get_max_bouts_limit("9-3") is None


def test_slice_and_offset():
    app = _app()
    seq = list(range(1, 101))
    lim = app._get_max_bouts_limit("40-50")
    assert app._slice_bouts(seq, lim) == list(range(40, 51))
    assert app._bout_limit_offset(lim) == 39
    assert app._slice_bouts(seq, 3) == [1, 2, 3]
    assert app._slice_bouts(seq, None) is seq
    assert app._bout_limit_offset(5) == 0
    assert app._slice_bouts(seq, app._get_max_bouts_limit("98-")) == [98, 99, 100]


def test_entry_channel_bouts_range():
    app = _app()
    app._realign_bouts = lambda b, *a, **k: list(b)
    app._entry_prebout = lambda e: 0
    app._entry_fs = lambda e: 1.0
    entry = {'G0': [np.zeros(4) + i for i in range(1, 21)]}
    got = app._entry_channel_bouts(entry, 'G0', app._get_max_bouts_limit("5-7"))
    assert [g[0] for g in got] == [5, 6, 7]


def test_zone_entry_bouts():
    app = _app()
    app.params = {'baseline_correct_bouts': False, 'maze_type': 'OFT'}
    app.bout_window_samples = lambda sid: (5, 5, 2)
    app.get_fps = lambda *a, **k: 10.0
    app.get_channel_name = lambda d, ch: f'G{ch}'
    app._is_valid_bout = lambda t: True
    n = 100
    beh = np.zeros((n, 8))
    beh[:, 6] = np.arange(n)
    beh[:, 7] = -np.arange(n)
    data = {'entry_frames': {'transition_to_center': [20, 50, 2, 97],
                             'transition_to_corner': [], 'transition_to_edge': [60]},
            'num_photometry_channels': 2}
    out = app._zone_entry_bout_entries('S1', data, beh)
    assert set(out) == {'Zone - Center Entry', 'Zone - Edge Entry'}
    c = out['Zone - Center Entry']
    assert len(c['G0']) == 2          # frames 2 and 97 run off the recording
    assert c['_kept_indices']['G0'] == [0, 1]
    assert c['G0'][0][5] == 20 and c['G1'][0][5] == -20
    assert c['onset_frames'] == [20, 50, 2, 97]

    app.processed_data = {'S1': dict(data, beh_synced=beh)}
    app._ensure_zone_entry_bouts('S1')
    assert 'Zone - Edge Entry' in app.processed_data['S1']['bouts']
