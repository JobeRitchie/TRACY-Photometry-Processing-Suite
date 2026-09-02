"""Zone entries can drive the coherence epoch analyses, with a dwell filter.

The Coherence tab is onset-driven: every epoch analysis needs a list of frame
indices to centre a pre/post window on.  Scored behaviors supplied those; zone
entries were detected during processing but had no way into the tab, so
coherence around an open-arm entry could not be asked for at all.

Both now resolve through ``_resolve_epoch_onsets``.  The property that makes
zone entries usable — and that these tests pin — is the minimum-time-in-zone
filter: an entry the animal leaves after two seconds, run with a ten second post
window, measures coherence in whatever zone it moved to next.  Detection's own
duration threshold does not cover this, because it is applied at a different
(and usually much shorter) value.

The position track is the real DR22 EPM recording in Examples/.
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
SUBJECT = 'DR22'


def _app(maze='EPM'):
    """An FPAnalysisGUI shell with EPM zones and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {
        'maze_width_cm': 76.0,
        'maze_type': maze,
        'y_calibration_method': 'normalize_y_min',
        'velocity_outlier_threshold': 5,
        'min_open_arm_duration': 2.0,
        'min_time_between_entries': 3.0,
        'oft_min_entry_duration': 0.5,
        'oft_min_time_between_entries': 1.0,
        'baseline_correct_bouts': True,
        'preboutseconds': 3.0,
        'postboutseconds': 3.0,
        'baseline_seconds': 0.0,
        'exclude_frames_before': 0,
    }
    app.zone_templates = app._create_zone_templates()
    app.zones = {k: dict(v) for k, v in app.zone_templates[maze]['zones'].items()}
    app.processed_data = {}
    return app


def _load(app, subject=SUBJECT):
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
    beh[:, app._beh_kin_base(N_CH)] = np.arange(n) / FPS / 60.0

    beh = app.process_position_data(beh, n_channels=N_CH, fps=FPS,
                                    position_px=np.column_stack([x, y]).copy(),
                                    calibrate=True)
    beh, entry_frames = app.detect_zone_entries(beh, fps=FPS)

    # A z-score matrix on the photometry grid, the way processing leaves it:
    # elapsed, timestamp, then one column per channel.
    zscore = np.column_stack([beh[:, app._beh_kin_base(N_CH)], beh[:, 1],
                              beh[:, 6], beh[:, 7]])
    app.processed_data[subject] = {
        'beh_synced': beh, 'has_position': True,
        'num_photometry_channels': N_CH, 'photometry_fps': FPS,
        'entry_frames': entry_frames, 'bouts': {},
        'zscore_470': zscore, 'has_470': True, 'has_570': False,
        'channel_names': ['G0', 'G1'],
    }
    return app.processed_data[subject]


def test_zone_entries_are_offered_as_epoch_behaviors():
    """A detected entry type shows up in the dropdown; an undetected one does not."""
    app = _app()
    _load(app)

    offered = app._available_zone_entry_epochs()
    assert 'Zone: Open Arm Entry' in offered
    # The plain EPM template defines no open_distal zones, so that entry type
    # has no detections and must not be offered.
    assert 'Zone: Open Distal Entry' not in offered

    # …and the label round-trips back to the detection key.
    assert app._zone_entry_epoch_key('Zone: Open Arm Entry') == 'transition_to_open'
    assert app._zone_entry_epoch_key('zone: open arm entry') == 'transition_to_open'
    assert app._zone_entry_epoch_key('Grooming') is None


def test_resolver_returns_the_detected_entry_frames():
    """With no dwell filter, the resolver hands back exactly what detection found."""
    app = _app()
    data = _load(app)

    got = app._resolve_epoch_onsets(SUBJECT, data, 'Zone: Open Arm Entry')
    assert got == app._as_onset_indices(data['entry_frames']['transition_to_open'])
    assert len(got) > 0


def test_min_time_in_zone_drops_entries_the_animal_left_early():
    """The filter is monotone, and a long threshold really does remove entries.

    DR22's six open-arm entries include visits shorter than 15 s, so a 15 s
    requirement must keep strictly fewer than all of them — the case that
    matters, because a 10 s post window over a 5 s visit is measuring the
    centre, not the open arm.
    """
    app = _app()
    data = _load(app)

    counts = [len(app._resolve_epoch_onsets(SUBJECT, data, 'Zone: Open Arm Entry',
                                            min_bout_sec=t))
              for t in (0.0, 2.0, 5.0, 10.0, 15.0, 600.0)]

    assert counts == sorted(counts, reverse=True), counts
    assert counts[0] == 6            # all detected entries
    assert counts[-1] == 0           # nothing survives a 10-minute requirement
    assert counts[4] < counts[0]     # 15 s is genuinely selective


def test_surviving_entries_really_stay_in_the_zone():
    """Every kept onset is followed by an unbroken in-zone run of >= threshold."""
    app = _app()
    data = _load(app)

    threshold = 10.0
    kept = app._resolve_epoch_onsets(SUBJECT, data, 'Zone: Open Arm Entry',
                                     min_bout_sec=threshold)
    assert kept, 'expected at least one entry to survive a 10 s requirement'

    labels = app._zone_labels_for(SUBJECT, data)
    targets = app._entry_target_zones('transition_to_open')
    for onset in kept:
        dwell = app._zone_entry_dwell_frames(labels, onset, targets) / FPS
        assert dwell >= threshold - 1e-9, (onset, dwell)


def test_decision_entries_measure_dwell_in_the_centre():
    """Explore / avoid decisions are stamped at a CENTRE entry, so their dwell
    is centre occupancy — reading it as the open arm would reject every epoch."""
    app = _app()
    data = _load(app)

    assert app._entry_target_zones('explore_decision') == ['center']
    assert app._entry_target_zones('avoid_decision') == ['center']
    assert app._entry_target_zones('center_to_open') == ['center']
    assert set(app._entry_target_zones('transition_to_open')) == \
        {'open_arm_up', 'open_arm_down'}

    kept = app._resolve_epoch_onsets(SUBJECT, data, 'Zone: Explore Decision',
                                     min_bout_sec=2.0)
    assert len(kept) > 0


def test_distal_entries_need_the_complex_template():
    """'Open Distal Entry' is empty on the plain EPM template and populated on
    EPM_Complex — the difference is the zone geometry, not the detector."""
    plain = _app('EPM')
    plain_data = _load(plain)
    assert plain._resolve_epoch_onsets(SUBJECT, plain_data,
                                       'Zone: Open Distal Entry') == []

    complex_app = _app('EPM_Complex')
    complex_data = _load(complex_app)
    assert len(complex_app._resolve_epoch_onsets(
        SUBJECT, complex_data, 'Zone: Open Distal Entry')) > 0


def test_scored_behaviors_still_resolve_through_the_shared_path():
    """The resolver did not break the behavior it replaced."""
    app = _app()
    data = _load(app)
    data['bouts'] = {'Grooming': {'onset_frames': [100, 250.0, float('nan'), 900]}}

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming') == [100, 250, 900]
    # Case-insensitive, as the tab's editable combobox allows.
    assert app._resolve_epoch_onsets(SUBJECT, data, 'grooming') == [100, 250, 900]
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Rearing',
                                     boutframes_file='') == []


def test_zone_label_cache_invalidates_when_zones_move():
    """Zone geometry is part of the cache key, so an edited zone is not served
    a stale classification."""
    app = _app()
    data = _load(app)

    before = app._zone_labels_for(SUBJECT, data)
    n_open_before = sum(1 for z in before if z.startswith('open_arm'))

    app.zones['open_arm_up']['y_min'] += 10.0
    after = app._zone_labels_for(SUBJECT, data)
    n_open_after = sum(1 for z in after if z.startswith('open_arm'))

    assert n_open_after < n_open_before


def test_entries_are_mapped_onto_the_photometry_grid():
    """beh_synced is not always the signal's grid.

    On the acquisition-position path it has one row per VIDEO frame, so an
    entry frame reused as a signal index lands at the wrong time — and the
    error grows with elapsed time, which is exactly the silent kind. The join
    is the elapsed value synchronize_behavior wrote into each behavior row from
    the photometry sample it matched.
    """
    app = _app()

    video_fps, photo_fps, minutes = 30.0, 20.0, 5.0
    n_beh = int(video_fps * 60 * minutes)
    n_sig = int(photo_fps * 60 * minutes)

    beh = np.full((n_beh, app._beh_width(N_CH)), np.nan)
    beh[:, 0] = np.arange(n_beh, dtype=float)
    beh[:, 1] = np.arange(n_beh) / video_fps
    kb = app._beh_kin_base(N_CH)
    # Elapsed (minutes) each behavior row was matched to, exactly as sync writes it.
    beh[:, kb] = (np.arange(n_beh) / video_fps) / 60.0

    photo_elapsed = (np.arange(n_sig) / photo_fps) / 60.0
    zscore = np.column_stack([photo_elapsed, np.arange(n_sig) / photo_fps,
                              np.zeros(n_sig), np.zeros(n_sig)])

    data = {
        'beh_synced': beh, 'num_photometry_channels': N_CH,
        'photometry_fps': photo_fps, 'zscore_470': zscore,
        'has_470': True, 'has_570': False, 'channel_names': ['G0', 'G1'],
        'entry_frames': {'transition_to_open': [0, 1800, 5400, 8999]},
        'bouts': {},
    }
    app.processed_data['MIXED'] = data

    got = app._resolve_epoch_onsets('MIXED', data, 'Zone: Open Arm Entry')

    # Video frame f is at f/30 s, which is photometry sample f/30*20.
    assert got == [0, 1200, 3600, 5999]
    # And the naive reuse would have run off the end of the signal entirely.
    assert max(data['entry_frames']['transition_to_open']) > n_sig


def test_matching_grids_are_left_alone():
    """When the two grids already agree the onsets pass through untouched."""
    app = _app()
    data = _load(app)
    detected = app._as_onset_indices(data['entry_frames']['transition_to_open'])
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Zone: Open Arm Entry') == detected
