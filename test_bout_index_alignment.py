"""A channel's bout trace list is not parallel to onset_frames/end_frames.

Extraction rejects a bout per channel (all-NaN, or `_is_valid_bout` failing) but
keeps every bout in onset_frames/end_frames. Consumers that measure a bout over
`starts[i]..ends[i]`, where `i` came from enumerating the channel's traces, then
read a different bout's window from the first rejection onwards -- silently, and
only on recordings that contain a flat or NaN bout.

These tests pin the mapping that `_kept_indices` + `_original_bout_index`
provide, including the two channels of one subject disagreeing about which
bouts survived.
"""
import types

import numpy as np
import pytest

import fp_analysis_gui as G


def _app():
    """An FPAnalysisGUI shell with just enough state, and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.params = {'preboutframes': 2, 'postboutframes': 2,
                  'boutframe_processing_style': 'whole'}
    app.processed_data = {}
    return app


# --------------------------------------------------------------------------
# _original_bout_index
# --------------------------------------------------------------------------

def test_maps_trace_index_past_a_rejected_bout():
    app = _app()
    # Bout 1 was rejected on G0, so trace 1 is really bout 2.
    entry = {'G0': ['t0', 't2', 't3'],
             '_kept_indices': {'G0': [0, 2, 3]}}
    assert app._original_bout_index(entry, 'G0', 0, 4) == 0
    assert app._original_bout_index(entry, 'G0', 1, 4) == 2
    assert app._original_bout_index(entry, 'G0', 2, 4) == 3


def test_channels_of_one_subject_can_disagree():
    app = _app()
    entry = {'G0': ['a', 'b'], 'G1': ['a', 'b', 'c'],
             '_kept_indices': {'G0': [0, 2], 'G1': [0, 1, 2]}}
    assert app._original_bout_index(entry, 'G0', 1, 3) == 2
    assert app._original_bout_index(entry, 'G1', 1, 3) == 1


def test_out_of_range_trace_index_is_refused():
    app = _app()
    entry = {'G0': ['a'], '_kept_indices': {'G0': [0]}}
    assert app._original_bout_index(entry, 'G0', 5, 3) is None


def test_index_beyond_the_frame_arrays_is_refused():
    app = _app()
    entry = {'G0': ['a'], '_kept_indices': {'G0': [9]}}
    assert app._original_bout_index(entry, 'G0', 0, 3) is None


def test_legacy_store_without_the_map_indexes_directly_when_intact():
    """No _kept_indices and no dropped bouts: direct indexing is still sound."""
    app = _app()
    entry = {'G0': ['a', 'b', 'c']}
    assert app._original_bout_index(entry, 'G0', 1, 3) == 1


def test_legacy_store_with_a_detectable_skew_refuses_to_guess():
    """Lengths disagree and nothing recorded which bouts went: do not guess."""
    app = _app()
    entry = {'G0': ['a', 'b']}          # 2 traces, 3 frames
    assert app._original_bout_index(entry, 'G0', 1, 3) is None


# --------------------------------------------------------------------------
# _bout_analysis_segment reads the right window
# --------------------------------------------------------------------------

def _subject_with_a_rejected_bout():
    """3 bouts at frames 10/20/30; bout 1 (frame 20) rejected on G0.

    Photometry columns start at beh_synced index 6 (see _bout_channel_column),
    so G0 is column 6. It holds a ramp, so a sample's value is its frame number
    and the segment we get back names the bout it came from.
    """
    beh = np.zeros((60, 7))
    beh[:, 6] = np.arange(60, dtype=float)
    return {
        'beh_synced': beh,
        'num_photometry_channels': 1,
        'channel_names': ['G0'],
        'bouts': {
            'Sniff': {
                'G0': ['trace_for_10', 'trace_for_30'],
                '_kept_indices': {'G0': [0, 2]},
                'onset_frames': [10, 20, 30],
                'end_frames': [14, 24, 34],
                'durations': [4, 4, 4],
                '_prebout': 2, '_postbout': 2,
            }
        },
    }


def test_whole_style_reads_the_bout_the_trace_belongs_to():
    app = _app()
    app.processed_data = {'S1': _subject_with_a_rejected_bout()}
    stored = np.zeros(4)

    # Trace 1 is bout 2 -> frames 30..34, i.e. ramp values 30..33.
    seg = app._bout_analysis_segment('S1', 'Sniff', 'G0', 1, stored, 0, 4)
    assert seg is not None
    assert seg[0] == pytest.approx(30.0), (
        f"read the wrong bout's window: starts at {seg[0]}, expected 30")
    assert len(seg) == 4


def test_first_trace_is_unaffected():
    app = _app()
    app.processed_data = {'S1': _subject_with_a_rejected_bout()}
    seg = app._bout_analysis_segment('S1', 'Sniff', 'G0', 0, np.zeros(4), 0, 4)
    assert seg[0] == pytest.approx(10.0)


def test_legacy_skewed_store_falls_back_to_the_onset_slice():
    """Rather than reading a neighbouring bout's window."""
    app = _app()
    data = _subject_with_a_rejected_bout()
    del data['bouts']['Sniff']['_kept_indices']      # pre-fix project
    app.processed_data = {'S1': data}
    stored = np.arange(4, dtype=float) + 100.0       # recognisable onset trace
    seg = app._bout_analysis_segment('S1', 'Sniff', 'G0', 1, stored, 0, 2)
    # Came from the stored onset-aligned trace, not from beh_synced.
    assert seg is not None
    assert seg[0] >= 100.0


# --------------------------------------------------------------------------
# the metadata key must not be mistaken for signal
# --------------------------------------------------------------------------

def test_rescale_does_not_touch_the_index_map():
    """_rescale_bout_store recurses into dict values and scales lists that are
    keyed by channel -- the index map must be skipped, not multiplied."""
    app = _app()
    store = {
        'Sniff': {
            'G0': [np.array([1.0, 2.0])],
            '_kept_indices': {'G0': [0, 2]},
        }
    }
    app._rescale_bout_store(store, {'G0': (2.0, 0.0)}, scale_only=True)
    assert store['Sniff']['_kept_indices']['G0'] == [0, 2], (
        "the index map was scaled as if it were signal")
    assert store['Sniff']['G0'][0].tolist() == [2.0, 4.0]
