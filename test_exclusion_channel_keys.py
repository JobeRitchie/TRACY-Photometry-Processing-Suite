"""Exclusions must be looked up under each subject's own channel designation.

The Exclusions tab keys every checkbox by the subject's real channel name --
``G0``, ``G1``, ``R2``, ``R4``, ``R5`` -- but the plotting and export code works
in positional slots, because the bout/entry stores are keyed ``Ch0``/``Ch1``
with ``G0``/``G1`` kept only as positional aliases of the first two.  Those call
sites used to ask ``is_subject_channel_excluded(subject, 'G0')``, so on a
recording designated ``R4``/``R5`` the lookup could never match: ticking the box
did nothing and the excluded trace was still drawn and exported.  On a plain
``G0``/``G1`` cohort everything worked, which is why it read as "exclusions work
sometimes".

The same literal broke *mixed* cohorts even when every subject is green-named,
because ``_viz_bout_channel_slots`` reads its labels off the first subject that
has data and applies them to all of them.

These tests pin the resolver (``channel_exclusion_key`` and friends) and the two
properties that matter downstream: a red-channel cohort is filtered, and a
subject with no channel designation at all still behaves exactly as before,
under the positional ``G0``/``G1`` labels the tab falls back to.
"""
import pytest

import fp_analysis_gui as G


def _app(channel_names=None, exclusions=None, subjects=('S1',)):
    """An FPAnalysisGUI shell with processed_data + exclusions and no Tk.

    ``channel_names`` maps subject -> the list stored in ``channel_names``
    (``None`` for a subject that never got a designation).
    """
    app = object.__new__(G.FPAnalysisGUI)
    channel_names = channel_names or {}
    app.processed_data = {}
    for s in subjects:
        data = {'has_470': True}
        names = channel_names.get(s)
        if names is not None:
            data['channel_names'] = list(names)
        app.processed_data[s] = data
    app.exclusions = {k: list(v) for k, v in (exclusions or {}).items()}
    return app


# ---------------------------------------------------------------------------
# the resolver itself
# ---------------------------------------------------------------------------

def test_key_is_the_real_designation_not_a_literal():
    app = _app({'S1': ['R4', 'R5']})
    assert app.channel_exclusion_key('S1', 0) == 'R4'
    assert app.channel_exclusion_key('S1', 1) == 'R5'


def test_key_falls_back_to_positional_labels_without_designations():
    """A subject with no channel_names keeps the legacy G0/G1 keys.

    get_channel_name returns 'Ch0'/'Ch1' there, but the Exclusions tab labels
    those checkboxes 'G0'/'G1', so the lookup must too or this fix would break
    every project recorded from a header-less CSV.
    """
    app = _app({'S1': None})
    assert app.channel_exclusion_key('S1', 0) == 'G0'
    assert app.channel_exclusion_key('S1', 1) == 'G1'


def test_key_falls_back_for_a_partially_mapped_subject():
    """channel_names can carry 'Ch1' placeholders for unmapped slots."""
    app = _app({'S1': ['G0', 'Ch1']})
    assert app.channel_exclusion_key('S1', 0) == 'G0'
    assert app.channel_exclusion_key('S1', 1) == 'G1'


def test_key_for_an_unknown_subject_does_not_raise():
    app = _app()
    assert app.channel_exclusion_key('nobody', 0) == 'G0'


def test_channel_slot_index_parses_ch_keys():
    app = _app()
    assert app.channel_slot_index('Ch0') == 0
    assert app.channel_slot_index('Ch1') == 1
    assert app.channel_slot_index('Ch7') == 7      # past the old 4-entry map
    assert app.channel_slot_index(3) == 3
    assert app.channel_slot_index('G1') == 0       # not a slot key -> default
    assert app.channel_slot_index(None, default=2) == 2


# ---------------------------------------------------------------------------
# the property that was broken: a red-channel cohort is actually filtered
# ---------------------------------------------------------------------------

def test_red_channel_exclusion_is_honoured():
    """The bug: excluding R4 left slot 0 included, so the trace was still drawn."""
    app = _app({'S1': ['R4', 'R5']}, {'S1': ['R4']})
    assert app.is_channel_slot_excluded('S1', 0) is True
    assert app.is_channel_slot_excluded('S1', 1) is False


def test_red_channel_exclusion_was_invisible_to_the_old_literal():
    """Guards the premise: the literal these call sites used cannot match."""
    app = _app({'S1': ['R4', 'R5']}, {'S1': ['R4']})
    assert app.is_subject_channel_excluded('S1', 'G0') is False


def test_green_cohort_is_unchanged():
    app = _app({'S1': ['G0', 'G1']}, {'S1': ['G1']})
    assert app.is_channel_slot_excluded('S1', 0) is False
    assert app.is_channel_slot_excluded('S1', 1) is True


def test_undesignated_subject_still_filters_on_g0():
    app = _app({'S1': None}, {'S1': ['G0']})
    assert app.is_channel_slot_excluded('S1', 0) is True
    assert app.is_channel_slot_excluded('S1', 1) is False


# ---------------------------------------------------------------------------
# mixed cohorts: one shared literal cannot describe every subject
# ---------------------------------------------------------------------------

def test_mixed_cohort_filters_each_subject_by_its_own_designation():
    app = _app(
        {'green': ['G0', 'G1'], 'red': ['R4', 'R5']},
        {'green': ['G0'], 'red': ['R4']},
        subjects=('green', 'red'),
    )
    assert app.get_included_subjects_for_slot(['green', 'red'], 0) == []
    assert app.get_included_subjects_for_slot(['green', 'red'], 1) == ['green', 'red']


def test_mixed_cohort_under_a_shared_label_kept_the_excluded_subject():
    """Guards the premise for the _viz_bout_channel_slots sibling defect.

    The label is read off the first subject that has data, so filtering the
    whole pool by it silently spared every differently-designated subject.
    """
    app = _app(
        {'green': ['G0', 'G1'], 'red': ['R4', 'R5']},
        {'green': ['G0'], 'red': ['R4']},
        subjects=('green', 'red'),
    )
    assert app.get_included_subjects_for_channel(['green', 'red'], 'G0') == ['red']


def test_slot_key_filter_resolves_a_ch_key_per_subject():
    app = _app(
        {'green': ['G0', 'G1'], 'red': ['R4', 'R5']},
        {'red': ['R5']},
        subjects=('green', 'red'),
    )
    assert app.included_subjects_for_slot_key(['green', 'red'], 'Ch1') == ['green']


def test_slot_key_filter_treats_none_as_an_unused_slot():
    app = _app({'S1': ['G0', 'G1']}, {'S1': ['G0', 'G1']})
    assert app.included_subjects_for_slot_key(['S1'], None) == ['S1']


# ---------------------------------------------------------------------------
# spike exports iterated ['G0','G1'], which matched neither the stored spike
# data nor the exclusion keys on a red-channel cohort
# ---------------------------------------------------------------------------

def _spike_app(spike_data, exclusions=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.spike_data = spike_data
    app.exclusions = {k: list(v) for k, v in (exclusions or {}).items()}
    return app


def test_spike_export_channels_follow_the_stored_designations():
    app = _spike_app({'S1': {'R4': {}, 'R5': {}}})
    assert app._spike_export_channels('S1', {'R4', 'R5'}, []) == ['R4', 'R5']


def test_spike_export_channels_drop_excluded_and_unselected():
    app = _spike_app({'S1': {'R4': {}, 'R5': {}}})
    assert app._spike_export_channels('S1', {'R4', 'R5'}, ['R5']) == ['R4']
    assert app._spike_export_channels('S1', {'R5'}, []) == ['R5']


def test_spike_export_channels_sort_like_the_checkboxes():
    app = _spike_app({'S1': {'R10': {}, 'G1': {}, 'R2': {}, 'G0': {}}})
    assert app._spike_export_channels(
        'S1', {'R10', 'G1', 'R2', 'G0'}, []) == ['G0', 'G1', 'R2', 'R10']


def test_spike_export_channels_for_a_subject_with_no_spike_data():
    app = _spike_app({})
    assert app._spike_export_channels('S1', {'G0'}, []) == []
