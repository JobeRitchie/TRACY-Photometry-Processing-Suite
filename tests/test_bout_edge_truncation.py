"""A bout whose pre/post window runs off the end of the recording is excluded.

Extraction used to clamp the slice to whatever fitted (`max(0, ...)` /
`min(len, ...)`), so the last bout of a recording came back short. Every
downstream plotter pads a short trace back to the full window with NaN, which
put a blank tail on that heatmap row and quietly dropped the subject out of the
group mean part-way along the x-axis. 55 bouts in the NSF_CIE_definitive project
were affected, all of them at a recording's end.

These tests pin the exclusion, and that the surviving traces stay mapped to the
right rows of onset_frames.
"""
import numpy as np
import pandas as pd

import fp_analysis_gui as G


PRE, POST = 10, 20
N_FRAMES = 200


def _app(onsets, ends=None, n_frames=N_FRAMES, nan_span=None):
    """An FPAnalysisGUI shell wired to a synthetic single-behavior recording."""
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {
        'exclude_frames_before': 0,
        'baseline_correct_bouts': False,
        'auto_scale_boutframes': False,
        'precut_correct_boutframes': False,
        'preboutframes': PRE,
        'postboutframes': POST,
    }
    app.bout_offsets = []
    app.processed_data = {'S1': {'num_photometry_channels': 2}}
    app.log_message = lambda *a, **k: None
    app.read_boutframes_sheet = lambda *a, **k: pd.DataFrame()
    app.bout_window_samples = lambda subject=None: (PRE, POST, PRE // 2)
    app.get_fps = lambda subject=None: 20.0
    app.get_channel_name = lambda data, ch: ('G0', 'G1')[ch]
    app._parse_boutframes_dataframe = lambda df: (
        [('Approach', np.asarray(onsets, dtype=int),
          None if ends is None else np.asarray(ends, dtype=float))], ends is not None)
    app._transform_boutframe_values = (
        lambda vals, subj, as_int=False: np.asarray(vals, dtype=int if as_int else float))
    app._apply_bout_exclusions = lambda f, e, *a, **k: (f, e, 0)

    # Two channels of noise: nothing here is flat, so _is_valid_bout keeps it.
    rng = np.random.default_rng(0)
    beh = np.zeros((n_frames, 6 + 2))
    beh[:, 6] = rng.normal(size=n_frames)
    beh[:, 7] = rng.normal(size=n_frames)
    if nan_span is not None:
        beh[nan_span[0]:nan_span[1], 6:8] = np.nan
    return app, beh


def _extract(app, beh):
    return app.extract_bouts('S1', beh, 'unused.xlsx')['Approach']


def test_bout_running_past_the_end_is_excluded():
    # 190 + POST(20) = 210 > 200 frames.
    app, beh = _app([50, 100, 190])
    entry = _extract(app, beh)

    assert len(entry['G0']) == 2
    assert all(len(t) == PRE + POST for t in entry['G0'])
    assert not any(np.isnan(t).any() for t in entry['G0'])
    # onset_frames keeps every bout; _kept_indices says which survived.
    assert entry['onset_frames'] == [50, 100, 190]
    assert entry['_kept_indices']['G0'] == [0, 1]


def test_bout_running_before_the_start_is_excluded():
    # 5 - PRE(10) < 0.
    app, beh = _app([5, 100])
    entry = _extract(app, beh)

    assert len(entry['G0']) == 1
    assert entry['_kept_indices']['G0'] == [1]


def test_bout_with_nan_inside_its_window_is_excluded():
    app, beh = _app([50, 100], nan_span=(95, 98))
    entry = _extract(app, beh)

    assert len(entry['G0']) == 1
    assert entry['_kept_indices']['G0'] == [0]


def test_bouts_fully_inside_the_recording_are_all_kept():
    app, beh = _app([50, 100, 150])
    entry = _extract(app, beh)

    assert len(entry['G0']) == 3
    assert len(entry['G1']) == 3
    assert entry['_kept_indices']['G1'] == [0, 1, 2]


def test_last_bout_exactly_reaching_the_final_frame_is_kept():
    # onset + POST == n_frames is the last window that fits.
    app, beh = _app([N_FRAMES - POST])
    entry = _extract(app, beh)

    assert len(entry['G0']) == 1
    assert len(entry['G0'][0]) == PRE + POST
