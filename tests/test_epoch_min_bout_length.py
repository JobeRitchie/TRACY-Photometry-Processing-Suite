"""The coherence epoch minimum applies to scored bouts, not only zone entries.

A zone entry that the animal leaves after two seconds, measured with a ten
second post window, is mostly coherence in whatever zone came next; the same is
true of a two second bout of grooming.  The Coherence tab's minimum therefore
has to reach both onset sources, and it can only reach a scored behavior when
the boutframes file carries END frames — a start-only file has no length to
measure, so the filter must pass the onsets through untouched rather than
rejecting every one of them.

Lengths live in photometry samples (``_transform_boutframe_values`` converts
video frames on the way in), so the threshold is applied as seconds x the
subject's own sampling rate.
"""
import os

import pandas as pd
import pytest

import fp_analysis_gui as G

FPS = 20.0
SUBJECT = 'S1'


def _app():
    """An FPAnalysisGUI shell with one processed subject and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {
        'exclude_frames_before': 0,
        'auto_scale_boutframes': False,
        'boutframes_video_fps': FPS,
        'boutframe_manual_shift': 0,
        'precut_correct_boutframes': False,
    }
    app.per_subject_boutframe_shifts = {}
    app.processed_data = {SUBJECT: {'photometry_fps': FPS, 'bouts': {}}}
    return app


def _bouts(onsets, durations):
    return {'Grooming': {'onset_frames': list(onsets),
                         'end_frames': None if durations is None else
                         [o + d for o, d in zip(onsets, durations)],
                         'durations': durations}}


def test_short_bouts_are_dropped_at_the_threshold():
    """Only bouts at least as long as the minimum survive, and the filter is
    monotone in the threshold."""
    app = _app()
    data = app.processed_data[SUBJECT]
    # 1 s, 3 s, 5 s and 10 s bouts at 20 Hz.
    data['bouts'] = _bouts([100, 500, 900, 1400],
                           [1 * FPS, 3 * FPS, 5 * FPS, 10 * FPS])

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming',
                                     boutframes_file='') == [100, 500, 900, 1400]
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=3.0,
                                     boutframes_file='') == [500, 900, 1400]
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=6.0,
                                     boutframes_file='') == [1400]
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=60.0,
                                     boutframes_file='') == []


def test_threshold_is_seconds_not_frames():
    """A 5 s minimum at 20 Hz means 100 samples, not 5."""
    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = _bouts([10, 200], [6.0, 120.0])   # 0.3 s and 6 s

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=5.0,
                                     boutframes_file='') == [200]


def test_start_only_bouts_are_not_filtered_away():
    """Without end frames there is no length to test, so every onset survives —
    silently dropping them all would be the worse failure."""
    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = {'Grooming': {'onset_frames': [100, 500, 900],
                                  'end_frames': None, 'durations': None}}

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=30.0,
                                     boutframes_file='') == [100, 500, 900]

    # A legacy entry predating the start/end feature carries neither key.
    data['bouts'] = {'Grooming': {'onset_frames': [100, 500, 900]}}
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=30.0,
                                     boutframes_file='') == [100, 500, 900]


def test_lengths_stay_paired_with_their_own_onsets():
    """Onset coercion drops NaN rows; the durations must be dropped with them.

    Filtering after coercion would pair bout i with bout i+1's duration from the
    first missing onset onwards — a silent misattribution, not a crash.
    """
    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = {'Grooming': {
        'onset_frames': [100, float('nan'), 500, 900],
        'durations': [10 * FPS, 1 * FPS, 1 * FPS, 10 * FPS],
        'end_frames': [100 + 10 * FPS, float('nan'), 500 + FPS, 900 + 10 * FPS],
    }}

    # 100 and 900 are the long ones; a naive shift would keep 500 instead.
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=5.0,
                                     boutframes_file='') == [100, 900]


def test_bouts_with_no_scored_end_are_dropped_when_a_minimum_is_asked_for():
    """An unmeasurable bout cannot be shown to meet the threshold."""
    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = {'Grooming': {
        'onset_frames': [100, 500],
        'durations': [10 * FPS, float('nan')],
        'end_frames': [100 + 10 * FPS, float('nan')],
    }}

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=5.0,
                                     boutframes_file='') == [100]
    # …but with no minimum asked for, it is still a usable onset.
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming',
                                     boutframes_file='') == [100, 500]


def test_end_frames_alone_are_enough():
    """Projects saved before durations were stored still filter, from the ends."""
    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = {'Grooming': {'onset_frames': [100, 500],
                                  'end_frames': [100 + FPS, 500 + 10 * FPS]}}

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=5.0,
                                     boutframes_file='') == [500]


def test_boutframes_fallback_reads_start_end_columns(tmp_path):
    """A subject with no stored bouts falls back to the workbook, where the
    behavior lives under 'Name__start' / 'Name__end' — the old raw-column
    lookup matched neither, so the fallback found nothing at all."""
    path = os.path.join(str(tmp_path), 'boutframes.xlsx')
    df = pd.DataFrame({'Grooming__start': [100.0, 500.0, 900.0],
                       'Grooming__end':   [120.0, 600.0, 1100.0]})
    try:
        with pd.ExcelWriter(path) as xl:
            df.to_excel(xl, sheet_name=SUBJECT, index=False)
    except (ImportError, ValueError) as exc:      # no openpyxl in this env
        pytest.skip(f'cannot write xlsx: {exc}')

    app = _app()
    data = app.processed_data[SUBJECT]
    data['bouts'] = {}

    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming',
                                     boutframes_file=path) == [100, 500, 900]
    # 1 s, 5 s and 10 s bouts at 20 Hz.
    assert app._resolve_epoch_onsets(SUBJECT, data, 'Grooming', min_bout_sec=4.0,
                                     boutframes_file=path) == [500, 900]

