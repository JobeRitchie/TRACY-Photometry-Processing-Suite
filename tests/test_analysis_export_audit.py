"""Regressions from the analysis/export audit.

Each test pins a number that was wrong on real projects: rows the photometry
never recorded being averaged as data, the behavior row rate read off a frozen
clock, AUC in samples, |r| reported as r, and a subject in two groups.
"""
import numpy as np
import pytest

import fp_analysis_gui as G

N_KIN = len(G.FPAnalysisGUI.BEH_KIN_FIELDS)


def _app(zones=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'maze_type': 'EPM', 'maze_width_cm': 76, 'outback_velocity_threshold': 2.0}
    app.zones = zones or {
        'center': {'category': 'center', 'x_min': 30, 'x_max': 46, 'y_min': 30, 'y_max': 46},
        'open_up': {'category': 'open', 'x_min': 30, 'x_max': 46, 'y_min': 46, 'y_max': 76},
    }
    app.processed_data = {}
    app.exclusions = {}
    app.log_message = lambda *a, **k: None
    app._mixed_fps_warned = False
    return app


def _beh(n_rec=300, n_tail=200, n_lead=40, fps=30.0, seed=0):
    """One-channel beh_synced: a recorded stretch between a held head and tail."""
    rng = np.random.default_rng(seed)
    n = n_lead + n_rec + n_tail
    beh = np.full((n, 6 + 1 + N_KIN), np.nan)
    sig = rng.standard_normal(n_rec)
    full = np.concatenate([np.full(n_lead, sig[0]), sig, np.full(n_tail, sig[-1])])
    beh[:, 6] = full
    beh[:, 2] = 38.0
    beh[:, 3] = 38.0                       # center while recorded
    beh[n_lead + n_rec:, 3] = 60.0         # open arm only after the recording stopped
    elapsed = np.arange(n_rec) / fps / 60.0
    beh[:, 7] = np.concatenate([np.full(n_lead, elapsed[0]), elapsed,
                                np.full(n_tail, elapsed[-1])])
    beh[:, 10] = np.abs(beh[:, 2] - 38.0)
    beh[:, 11] = np.abs(beh[:, 3] - 38.0)
    return beh, sig, n_lead, n_rec


def test_recorded_rows_drops_the_held_head_and_tail_only():
    beh, _sig, n_lead, n_rec = _beh()
    rec = G.FPAnalysisGUI._recorded_rows(beh)
    assert not rec[:n_lead - 1].any()
    assert rec[n_lead:n_lead + n_rec].all()
    assert not rec[n_lead + n_rec + 1:].any()
    # A reprocessed subject (uncovered rows already NaN) is left alone.
    clean = beh.copy()
    clean[~rec, 6] = np.nan
    assert G.FPAnalysisGUI._recorded_rows(clean).all()


def test_zone_average_ignores_rows_the_photometry_never_recorded():
    app = _app()
    beh, sig, n_lead, n_rec = _beh()
    zones = [app.classify_zone(x, y) for x, y in beh[:, 2:4]]
    za = app.calculate_zone_averages(beh, zones, n_channels=1)
    # The animal only reached the open arm after the recording stopped.
    assert 'open_up' not in za
    assert za['center']['G0_mean'] == pytest.approx(np.mean(sig))
    assert n_rec <= za['center']['G0_n'] <= n_rec + 2


def test_distance_bins_keep_their_axis_but_not_the_held_rows():
    app = _app()
    beh, sig, _l, n_rec = _beh()
    _x, y_avg, _e = app.calculate_distance_based_averages(beh, n_channels=1)
    assert y_avg[0]['G0_mean'] == pytest.approx(np.mean(sig))
    assert 22 in y_avg and y_avg[22]['G0_n'] == 0      # visited only in the tail
    assert np.isnan(y_avg[22]['G0_mean'])


def test_stored_averages_are_rebuilt_once_when_they_include_held_rows():
    app = _app()
    beh, sig, _l, _n = _beh()
    data = {'beh_synced': beh, 'has_position': True, 'num_photometry_channels': 1,
            'zone_averages': {'center': {'G0_mean': 99.0, 'G0_std': 0.0, 'G0_n': 1},
                              'open_up': {'G0_mean': 99.0, 'G0_std': 0.0, 'G0_n': 1}}}
    za = app._get_or_compute_zone_averages(data)
    assert za['center']['G0_mean'] == pytest.approx(np.mean(sig))
    assert 'open_up' not in za


def test_behavior_row_rate_is_not_inflated_by_a_frozen_clock():
    app = _app()
    beh, _sig, _l, _n = _beh(fps=30.0)
    data = {'beh_synced': beh, 'num_photometry_channels': 1, 'photometry_fps': 30.0}
    assert app._beh_row_rate('s', data) == pytest.approx(30.0, rel=0.02)


def test_kinematics_signal_is_blank_where_nothing_was_recorded():
    app = _app()
    beh, _sig, n_lead, n_rec = _beh()
    beh[:, 8] = 1.0                        # velocity
    data = {'beh_synced': beh, 'has_position': True, 'num_photometry_channels': 1,
            'photometry_fps': 30.0}
    k = app._compute_kinematics(data, 0, {'win': 11, 'poly': 3})
    assert np.isnan(k['sig'][n_lead + n_rec + 5:]).all()
    assert np.isfinite(k['sig'][n_lead:n_lead + n_rec]).all()


def test_group_label_names_every_selected_group():
    join = G.FPAnalysisGUI._join_group_labels
    assert join(None, 'Male Water') == 'Male Water'
    assert join('Male Water', 'All Water') == 'Male Water + All Water'
    assert join('Male Water + All Water', 'All Water') == 'Male Water + All Water'
    assert join('N/A', 'All Water') == 'All Water'


def test_mean_rolling_correlation_keeps_its_sign():
    f = G.FPAnalysisGUI._mean_rolling_r
    assert f({'r': np.array([-0.4, -0.5, 0.1]), 'params': {'fisher_z': False}}) == pytest.approx(-0.8 / 3)
    z = np.arctanh(np.array([-0.4, -0.5]))
    assert f({'r': z, 'params': {'fisher_z': True}}) == pytest.approx(np.tanh(z.mean()))
    assert np.isnan(f({'r': np.array([np.nan])}))


def test_bout_auc_is_integrated_over_seconds():
    app = _app()
    fs = 20.0
    app.params.update(preboutframes=20, postboutframes=40, boutframe_processing_style='onset')
    trace = np.concatenate([np.zeros(20), np.ones(40)])
    app.processed_data['s1'] = {'photometry_fps': fs, 'bouts': {
        'b': {'Ch0': [trace], '_prebout': 20, '_postbout': 40, '_fs': fs}}}
    _by, rows, _cols, n = app._compute_bout_metrics_for(
        'b', 'Ch0', ['s1'], {}, 0.0, 2.0, [('auc', 'AUC (z·s)')], None, 'both')
    assert n == 1
    # 2 s at z = 1 is 2 z*s (39 sample intervals / 20 Hz), not 39.
    assert rows[0]['AUC (z·s)'] == pytest.approx(39 / fs)


def test_kinematics_channel_index_resolves_past_the_second_channel():
    app = _app()
    app.processed_data['s1'] = {'num_photometry_channels': 4,
                                'channel_names': ['G0', 'G1', 'G2', 'G3'],
                                'beh_synced': np.zeros((5, 6 + 4 + N_KIN))}
    for label, slot in (('G0', 0), ('G1', 1), ('G2', 2), ('G3', 3)):
        app.kin_channel_var = type('V', (), {'get': lambda self, _l=label: _l})()
        assert app._kin_channel_index() == slot


def _zscore_for(beh, n_lead, n_rec, fps=30.0):
    """Photometry matrix whose clock covers only the recorded rows of ``beh``."""
    ts = 1000.0 + np.arange(len(beh)) * (1000.0 / fps)
    beh[:, 1] = ts
    z = np.column_stack([np.arange(n_rec) / fps / 60.0, ts[n_lead:n_lead + n_rec],
                         beh[n_lead:n_lead + n_rec, 6]])
    return z


def test_length_cap_cuts_the_behavior_tail_and_keeps_the_head():
    app = _app()
    beh, _sig, n_lead, n_rec = _beh(n_rec=600, n_tail=400, n_lead=40)
    z = _zscore_for(beh, n_lead, n_rec)
    out, n_cut = app._trim_behavior_to_photometry_end(beh, z)
    # Bout frames index rows from the start, so the head must not move.
    assert np.array_equal(out[:n_lead + n_rec, 6], beh[:n_lead + n_rec, 6])
    assert n_lead + n_rec <= len(out) <= n_lead + n_rec + 2
    assert n_cut == len(beh) - len(out)
    assert G.FPAnalysisGUI._recorded_rows(out)[n_lead:].all()


def test_behavior_is_left_alone_when_nothing_follows_the_photometry():
    app = _app()
    beh, _sig, n_lead, n_rec = _beh(n_rec=600, n_tail=0, n_lead=40)
    z = _zscore_for(beh, n_lead, n_rec)
    out, n_cut = app._trim_behavior_to_photometry_end(beh, z)
    assert n_cut == 0 and len(out) == len(beh)
