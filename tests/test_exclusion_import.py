"""Importing exclusions from another project carries them per animal.

A channel comes across only when the source project excluded it on every day
that animal was recorded: a fiber that was bad on one day but fine on another is
a good fiber, and the bad day was the recording.  It then lands on every session
the animal has in the importing project.  Behavior exclusions describe one
session's scoring and are never carried.
"""
import fp_analysis_gui as G

SRC_PATTERN = G.DEFAULT_SESSION_PATTERN


def _source(exclusions, subjects, pattern=SRC_PATTERN):
    return {'project_name': 'Src', 'exclusions': exclusions,
            'processed_subjects': list(subjects),
            'parameters': {'session_pattern': pattern}}


def _app(subjects, channels=('G0', 'G1'), exclusions=None, pattern=SRC_PATTERN):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {'channel_names': list(channels)} for s in subjects}
    app.exclusions = {k: list(v) for k, v in (exclusions or {}).items()}
    app.params = {'session_pattern': pattern}
    app.log_message = lambda *a, **k: None
    return app


def test_excluded_on_every_day_is_carried():
    carried, partial, _ = G.animal_level_exclusions(
        {'A_D1': ['G0'], 'A_D2': ['G0', 'G1'], 'A_D3': ['G0']},
        ['A_D1', 'A_D2', 'A_D3'])
    assert carried == {'A': ['G0']}
    assert partial == {'A': {'G1': 1}}


def test_excluded_on_one_day_of_several_is_not_carried():
    carried, partial, _ = G.animal_level_exclusions(
        {'A_D2': ['G0']}, ['A_D1', 'A_D2', 'A_D3'])
    assert carried == {}
    assert partial == {'A': {'G0': 1}}


def test_unexcluded_sessions_count_even_when_absent_from_exclusions():
    """The days with no exclusions are what make the others partial."""
    carried, _, _ = G.animal_level_exclusions({'A_D1': ['G0']}, ['A_D1', 'A_D2'])
    assert carried == {}


def test_single_session_animal_is_carried():
    carried, _, _ = G.animal_level_exclusions(
        {'A_D1': ['G1'], 'B_D1': [], 'B_D2': []}, ['A_D1', 'B_D1', 'B_D2'])
    assert carried == {'A': ['G1']}


def test_behavior_is_never_carried():
    carried, partial, _ = G.animal_level_exclusions(
        {'A_D1': ['Behavior'], 'A_D2': ['Behavior']}, ['A_D1', 'A_D2'])
    assert carried == {} and partial == {}


def test_multi_day_source_onto_single_session_target():
    src = _source({'A_D1': ['G0'], 'A_D2': ['G0'], 'B_D1': ['G1']},
                  ['A_D1', 'A_D2', 'B_D1', 'B_D2'])
    app = _app(['A', 'B', 'C'])
    plan = app.plan_exclusion_import(src)
    assert plan['additions'] == {'A': ['G0']}


def test_carried_exclusion_lands_on_every_target_session():
    src = _source({'A': ['G1']}, ['A', 'B'], pattern='')
    app = _app(['A_pre', 'A_post', 'B_pre', 'B_post'])
    plan = app.plan_exclusion_import(src)
    assert plan['additions'] == {'A_post': ['G1'], 'A_pre': ['G1']}


def test_source_with_one_day_each_still_matches_by_split_id():
    """No animal repeats in the source, so its split is not adopted -- but its
    IDs still name the animal a multi-day target groups under."""
    src = _source({'A_D1': ['G0']}, ['A_D1', 'B_D1'])
    app = _app(['A_D5', 'A_D6', 'B_D5'])
    plan = app.plan_exclusion_import(src)
    assert plan['additions'] == {'A_D5': ['G0'], 'A_D6': ['G0']}


def test_existing_exclusions_are_kept_and_not_duplicated():
    src = _source({'A_D1': ['G0', 'G1'], 'A_D2': ['G0', 'G1']}, ['A_D1', 'A_D2'])
    app = _app(['A'], exclusions={'A': ['G0']})
    plan = app.plan_exclusion_import(src)
    assert plan['additions'] == {'A': ['G1']}


def test_channel_the_target_lacks_is_reported_not_added():
    src = _source({'A_D1': ['R4'], 'A_D2': ['R4']}, ['A_D1', 'A_D2'])
    app = _app(['A'])
    plan = app.plan_exclusion_import(src)
    assert plan['additions'] == {}
    assert plan['missing_channel'] == [('A', 'R4')]


def test_animals_not_in_target_are_reported():
    src = _source({'Z_D1': ['G0']}, ['Z_D1', 'A_D1', 'A_D2'])
    plan = _app(['A']).plan_exclusion_import(src)
    assert plan['unmatched'] == ['Z']
