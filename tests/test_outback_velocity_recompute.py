"""The Out/Back minimum velocity can be retuned without a full reprocess.

The threshold decides which open-arm frames count as moving away from the centre
and which count as moving back, so it is the one number a reader of an Out/Back
plot wants to sweep. It used to live only in Edit Parameters, where changing it
meant re-running the whole pipeline from the source files.

_recompute_outback_all() re-derives the stats from the stored position track.
These tests pin that it matches what processing would have produced at the new
threshold, that a bigger threshold really does keep fewer frames, and that a
maze with no open arms is reported rather than silently emptied.

The position track is the real DR22 recording in Examples/.
"""
import os

import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G

TRACK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'Examples', 'DR22AnimalPosition0.csv')

FPS = 30.0
N_CH = 2


def _app(threshold=2.0, maze_type='EPM'):
    """An FPAnalysisGUI shell with EPM zones and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {
        'maze_width_cm': 76.0,
        'maze_type': maze_type,
        'y_calibration_method': 'legacy',
        'velocity_outlier_threshold': 5,
        'outback_velocity_threshold': float(threshold),
    }
    app.zone_templates = app._create_zone_templates()
    app.zones = {k: dict(v) for k, v in app.zone_templates['EPM']['zones'].items()}
    app.processed_data = {}
    return app


def _load(app, subject='DR22'):
    """Process the track the way process_fp_data does, and record the zone
    classification the Out/Back stats are computed against."""
    if not os.path.exists(TRACK):
        pytest.skip(f'position track not available: {TRACK}')
    df = pd.read_csv(TRACK, header=None)
    x = df.iloc[:, 2].to_numpy(float)
    y = df.iloc[:, 3].to_numpy(float)

    n = len(x)
    beh = np.full((n, app._beh_width(N_CH)), np.nan)
    beh[:, 0] = np.arange(n, dtype=float)
    beh[:, 1] = np.arange(n) / FPS
    beh[:, 2] = x
    beh[:, 3] = y
    rng = np.random.default_rng(0)
    for c in range(N_CH):
        beh[:, 6 + c] = np.cumsum(rng.normal(0, 0.05, n))

    px = np.column_stack([x, y]).copy()
    beh = app.process_position_data(beh, n_channels=N_CH, fps=FPS,
                                    position_px=px, calibrate=True)
    zones = ['unknown' if (np.isnan(beh[i, 2]) or np.isnan(beh[i, 3]))
             else app.classify_zone(beh[i, 2], beh[i, 3])
             for i in range(len(beh))]
    app.processed_data[subject] = {
        'beh_synced': beh, 'position_px': px, 'has_position': True,
        'num_photometry_channels': N_CH, 'photometry_fps': FPS,
        'outback': app.calculate_outback_movements(beh, zones,
                                                   n_channels=N_CH, fps=FPS),
    }
    return app.processed_data[subject], zones


def test_recompute_matches_processing_at_the_new_threshold():
    """Re-running detection at 6 cm/s gives what processing at 6 cm/s gives."""
    app = _app(2.0)
    data, zones = _load(app)
    before = data['outback']
    assert before['out']['G0_n'] > 0, 'fixture produced no Out frames to retune'

    reference = _app(6.0)
    expected = reference.calculate_outback_movements(
        data['beh_synced'], zones, n_channels=N_CH, fps=FPS)

    app.params['outback_velocity_threshold'] = 6.0
    assert app._recompute_outback_all() == 1

    got = app.processed_data['DR22']['outback']
    for direction in ('out', 'back'):
        for key, value in expected[direction].items():
            assert np.allclose(got[direction][key], value, equal_nan=True), \
                f'{direction}/{key}'
    # ...and it really did change something.
    assert got['out']['G0_n'] != before['out']['G0_n']


def test_a_higher_threshold_keeps_fewer_frames():
    """The threshold is a floor on speed, so raising it is monotonic."""
    app = _app(1.0)
    _load(app)

    counts = []
    for threshold in (1.0, 3.0, 8.0, 20.0):
        app.params['outback_velocity_threshold'] = threshold
        app._recompute_outback_all()
        ob = app.processed_data['DR22']['outback']
        counts.append(ob['out'].get('G0_n', 0) + ob['back'].get('G0_n', 0))

    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] > counts[-1]


def test_photometry_only_subject_is_skipped():
    """No position track means nothing to re-detect, and no exception."""
    app = _app(2.0)
    _load(app)
    app.processed_data['PHOT_ONLY'] = {
        'beh_synced': None, 'has_position': False,
        'num_photometry_channels': N_CH, 'photometry_fps': FPS,
    }
    assert app._recompute_outback_all() == 1
    assert 'outback' not in app.processed_data['PHOT_ONLY']


def test_non_epm_maze_updates_nothing_and_keeps_existing_stats():
    """Out/Back is an open-arm analysis; OFT has none, so the caller is told
    zero subjects updated rather than having the stored stats blanked."""
    app = _app(2.0)
    data, _ = _load(app)
    kept = data['outback']

    app.params['maze_type'] = 'OFT'
    assert app._recompute_outback_all() == 0
    assert app.processed_data['DR22']['outback'] is kept
