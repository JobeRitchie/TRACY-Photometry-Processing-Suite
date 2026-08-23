"""Zone changes can be applied to already-processed subjects without a reprocess.

Retuning zone boundaries used to mean re-running the whole pipeline from the
source files, because calibration rewrites beh_synced's X/Y columns from pixels
to centimetres IN PLACE -- so nothing downstream could recover the numbers the
zones are really defined against.

Processing now snapshots the uncalibrated pixel track alongside the subject, and
recalculate_position_analyses() re-derives every position-dependent result from
it. These tests pin the property that makes that safe: the result is identical
to a full reprocess, and a subject with no snapshot is reported rather than
silently rescaled.

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


def _app(maze_width=76.0, y_method='legacy'):
    """An FPAnalysisGUI shell with EPM zones and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {
        'maze_width_cm': float(maze_width),
        'maze_type': 'EPM',
        'y_calibration_method': y_method,
        'velocity_outlier_threshold': 5,
        'min_open_arm_duration': 2.0,
        'min_time_between_entries': 3.0,
        'oft_min_entry_duration': 0.5,
        'oft_min_time_between_entries': 1.0,
        'baseline_correct_bouts': True,
        'preboutseconds': 3.0,
        'postboutseconds': 3.0,
        'baseline_seconds': 0.0,
    }
    app.zone_templates = app._create_zone_templates()
    app.zones = {k: dict(v) for k, v in app.zone_templates['EPM']['zones'].items()}
    app.processed_data = {}
    return app


def _pixels():
    if not os.path.exists(TRACK):
        pytest.skip(f'position track not available: {TRACK}')
    df = pd.read_csv(TRACK, header=None)
    return df.iloc[:, 2].to_numpy(float), df.iloc[:, 3].to_numpy(float)


def _load(app, subject='DR22'):
    """Process the track the way process_fp_data does: snapshot the pixels,
    then calibrate in place."""
    x, y = _pixels()
    n = len(x)
    beh = np.full((n, app._beh_width(N_CH)), np.nan)
    beh[:, 0] = np.arange(n, dtype=float)
    beh[:, 1] = np.arange(n) / FPS
    beh[:, 2] = x
    beh[:, 3] = y
    rng = np.random.default_rng(0)
    for c in range(N_CH):
        beh[:, 6 + c] = np.cumsum(rng.normal(0, 0.05, n))
    beh[:, app._beh_kin_base(N_CH)] = np.arange(n) / FPS / 60.0

    px = np.column_stack([x, y]).copy()
    beh = app.process_position_data(beh, n_channels=N_CH, fps=FPS,
                                    position_px=px, calibrate=True)
    beh, entry_frames = app.detect_zone_entries(beh, fps=FPS)
    app.processed_data[subject] = {
        'beh_synced': beh, 'position_px': px, 'has_position': True,
        'num_photometry_channels': N_CH, 'photometry_fps': FPS,
        'entry_frames': entry_frames,
    }
    return app.processed_data[subject]


def test_maze_width_change_matches_a_full_reprocess():
    """Recalculating at a new maze width gives exactly what reprocessing gives."""
    app = _app(76.0)
    before = _load(app)['beh_synced'][:, 2:4].copy()

    reference = _app(110.0)
    expected = _load(reference)['beh_synced']

    app.params['maze_width_cm'] = 110.0
    summary = app.recalculate_position_analyses()

    got = app.processed_data['DR22']['beh_synced']
    assert summary == {'updated': 1, 'recalibrated': 1, 'no_pixels': [], 'failed': []}
    assert np.allclose(got[:, 2:4], expected[:, 2:4], equal_nan=True)
    # Velocity / distance / distance-from-centre follow the new scale too.
    kin = app._beh_kin_base(N_CH)
    assert np.allclose(got[:, kin + 1:], expected[:, kin + 1:], equal_nan=True)
    # ...and it really did change something.
    assert not np.allclose(got[:, 2:4], before, equal_nan=True)


def test_calibration_method_change_reads_the_pixels_not_the_centimetres():
    """Switching Y-calibration method is the case that cannot be done by
    re-scaling the stored centimetres: the independent Y scale has already
    destroyed the original Y pixel range."""
    app = _app(76.0, y_method='independent_y_scale')
    independent_cm = _load(app)['beh_synced'][:, 2:4].copy()

    reference = _app(76.0, y_method='legacy')
    expected = _load(reference)['beh_synced'][:, 2:4].copy()

    app.params['y_calibration_method'] = 'legacy'
    app.recalculate_position_analyses()
    assert np.allclose(app.processed_data['DR22']['beh_synced'][:, 2:4], expected,
                       equal_nan=True)

    # Without the snapshot the same switch lands somewhere else entirely.
    naive_app = _app(76.0, y_method='legacy')
    naive = np.full((len(independent_cm), naive_app._beh_width(N_CH)), np.nan)
    naive[:, 2:4] = independent_cm
    naive = naive_app.process_position_data(naive, n_channels=N_CH, fps=FPS,
                                            position_px=None, calibrate=True)
    assert np.nanmax(np.abs(naive[:, 3] - expected[:, 1])) > 0.5


def test_subject_without_a_pixel_snapshot_is_reported_not_rescaled():
    """Projects saved before the snapshot existed keep their centimetres, and
    are named back to the caller as needing a real reprocess."""
    app = _app(76.0)
    _load(app)
    del app.processed_data['DR22']['position_px']        # legacy project
    kept = app.processed_data['DR22']['beh_synced'][:, 2:4].copy()

    app.params['maze_width_cm'] = 110.0
    summary = app.recalculate_position_analyses()

    assert summary['no_pixels'] == ['DR22']
    assert summary['recalibrated'] == 0
    assert summary['updated'] == 1                       # zone results still redone
    assert np.allclose(app.processed_data['DR22']['beh_synced'][:, 2:4], kept,
                       equal_nan=True)


def test_zone_geometry_change_redrives_the_zone_results():
    """Moving zone boundaries alone updates entries, bouts and averages."""
    app = _app(76.0)
    _load(app)
    app.recalculate_position_analyses()
    data = app.processed_data['DR22']
    before_entries = {k: list(v) for k, v in data['entry_frames'].items()}
    before_averages = repr(data['zone_averages'])

    # Shrink the centre square, so its neighbours' entry criteria shift.
    centre = app.zones['center']
    cx = (centre['x_min'] + centre['x_max']) / 2
    cy = (centre['y_min'] + centre['y_max']) / 2
    centre['x_min'], centre['x_max'] = cx - 4, cx + 4
    centre['y_min'], centre['y_max'] = cy - 4, cy + 4

    summary = app.recalculate_position_analyses()
    assert summary['updated'] == 1 and not summary['failed']
    assert data['entry_frames'] != before_entries
    assert repr(data['zone_averages']) != before_averages
    for key in ('entry_bouts', 'distance_averages_x', 'distance_averages_y',
                'distance_averages_euclidean'):
        assert key in data


def test_photometry_only_subjects_are_left_alone():
    """A subject with no position data is skipped, not counted or crashed on."""
    app = _app(76.0)
    _load(app)
    app.processed_data['NOPOS'] = {'beh_synced': None, 'has_position': False}
    summary = app.recalculate_position_analyses()
    assert summary['updated'] == 1
    assert not summary['failed']
