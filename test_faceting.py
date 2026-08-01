"""
Tests for the factor layer: resolving "split by Group", "split by Session", or
both into the series a plot should draw.

Each factor's control carries one of three things -- COMBINE (pool the levels),
SPLIT (one series per level), or a literal level (filter to it) -- and the
series label is the cross-product of whatever is SPLIT. That is the whole
vocabulary, which is why Group x Session costs nothing extra.
"""

import importlib.util
import os
import re
import sys

import pytest

_GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_facet', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI
COMBINE, SPLIT, UNASSIGNED = _M.FACTOR_COMBINE, _M.FACTOR_SPLIT, _M.FACTOR_UNASSIGNED

SUBJECTS = ['2F_pre', '2F_post', '3M_pre', '3M_post']
FACTORS = {
    '2F_pre':  {'Group': 'Fentanyl', 'Session': 'pre',  'Animal': '2F'},
    '2F_post': {'Group': 'Fentanyl', 'Session': 'post', 'Animal': '2F'},
    '3M_pre':  {'Group': 'Saline',   'Session': 'pre',  'Animal': '3M'},
    '3M_post': {'Group': 'Saline',   'Session': 'post', 'Animal': '3M'},
}
ORDER = ['Group', 'Session', 'Animal']


def facet(selections, subjects=SUBJECTS, factors=FACTORS):
    return _M.facet_subjects(subjects, factors, selections, ORDER)


# ---------------------------------------------------------------------------
# The three sentinels
# ---------------------------------------------------------------------------

def test_combining_everything_reproduces_the_unfaceted_plot():
    labels, splits = facet({'Group': COMBINE, 'Session': COMBINE})
    assert splits == []
    assert set(labels) == set(SUBJECTS)
    assert set(labels.values()) == {'All'}


def test_splitting_one_factor_gives_one_series_per_level():
    labels, splits = facet({'Group': SPLIT, 'Session': COMBINE})
    assert splits == ['Group']
    assert labels['2F_pre'] == labels['2F_post'] == 'Fentanyl'
    assert labels['3M_pre'] == labels['3M_post'] == 'Saline'


def test_filtering_to_a_level_drops_the_other_subjects():
    labels, splits = facet({'Group': COMBINE, 'Session': 'post'})
    assert splits == []
    assert set(labels) == {'2F_post', '3M_post'}


def test_splitting_two_factors_gives_the_cross_product():
    """The payoff: session x group falls out of the same mechanism."""
    labels, splits = facet({'Group': SPLIT, 'Session': SPLIT})
    assert splits == ['Group', 'Session']
    assert labels == {
        '2F_pre':  'Fentanyl × pre',
        '2F_post': 'Fentanyl × post',
        '3M_pre':  'Saline × pre',
        '3M_post': 'Saline × post',
    }


def test_filtering_and_splitting_compose():
    labels, splits = facet({'Group': SPLIT, 'Session': 'pre'})
    assert splits == ['Group']
    assert labels == {'2F_pre': 'Fentanyl', '3M_pre': 'Saline'}


def test_series_label_order_follows_the_factor_order_not_the_dict():
    labels, _ = _M.facet_subjects(
        SUBJECTS, FACTORS, {'Session': SPLIT, 'Group': SPLIT}, ORDER)
    assert labels['2F_pre'] == 'Fentanyl × pre'   # Group first, per ORDER


def test_a_factor_with_no_selection_is_ignored():
    labels, splits = facet({'Group': SPLIT})
    assert splits == ['Group']
    assert set(labels) == set(SUBJECTS)


# ---------------------------------------------------------------------------
# Gaps in assignment
# ---------------------------------------------------------------------------

def test_an_unassigned_subject_is_labelled_rather_than_dropped():
    """A silently shrinking n is worse than a visible '(unassigned)' series."""
    factors = dict(FACTORS)
    factors['3M_pre'] = {'Session': 'pre'}          # no Group recorded
    labels, _ = facet({'Group': SPLIT}, factors=factors)
    assert labels['3M_pre'] == UNASSIGNED
    assert len(labels) == len(SUBJECTS)


def test_filtering_excludes_the_unassigned():
    factors = dict(FACTORS)
    factors['3M_pre'] = {'Session': 'pre'}
    labels, _ = facet({'Group': 'Saline'}, factors=factors)
    assert '3M_pre' not in labels
    assert '3M_post' in labels


def test_a_subject_absent_from_the_factor_table_still_appears():
    labels, _ = _M.facet_subjects(SUBJECTS + ['9Z_pre'], FACTORS, {'Group': SPLIT}, ORDER)
    assert labels['9Z_pre'] == UNASSIGNED


# ---------------------------------------------------------------------------
# Level display order -- it drives series order and colour assignment
# ---------------------------------------------------------------------------

def test_declared_level_order_wins_over_alphabetical():
    levels = _M.factor_levels(SUBJECTS, FACTORS, 'Session', level_order=['pre', 'post'])
    assert levels == ['pre', 'post']       # not ['post', 'pre']


def test_undeclared_levels_follow_alphabetically():
    levels = _M.factor_levels(SUBJECTS, FACTORS, 'Group', level_order=['Saline'])
    assert levels == ['Saline', 'Fentanyl']


def test_unassigned_sorts_last():
    factors = dict(FACTORS)
    factors['3M_pre'] = {'Session': 'pre'}
    levels = _M.factor_levels(SUBJECTS, factors, 'Group')
    assert levels[-1] == UNASSIGNED


# ---------------------------------------------------------------------------
# The GUI-facing model: built-in factors are derived, not stored
# ---------------------------------------------------------------------------

class _Facets:
    _session_pattern = _G._session_pattern
    get_animal_map = _G.get_animal_map
    get_subject_factors = _G.get_subject_factors
    get_factor_definitions = _G.get_factor_definitions
    facet = _G.facet

    def __init__(self, subjects, groups=None, params=None):
        self.params = {'session_pattern': _M.DEFAULT_SESSION_PATTERN,
                       'animal_overrides': {}, 'factor_definitions': [],
                       'subject_factors': {}}
        self.params.update(params or {})
        self.processed_data = {s: {} for s in subjects}
        self.groups = groups or {}
        self.log_message = lambda *a, **k: None


def test_group_and_session_are_available_without_being_configured():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post'],
                                    'Saline': ['3M_pre', '3M_post']})
    assert app.get_factor_definitions() == ['Group', 'Session', 'Animal']
    labels, _ = app.facet(SUBJECTS, {'Group': SPLIT, 'Session': SPLIT})
    assert labels['2F_post'] == 'Fentanyl × post'


def test_existing_groups_are_mirrored_so_nothing_that_reads_them_breaks():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post']})
    factors = app.get_subject_factors()
    assert factors['2F_pre']['Group'] == 'Fentanyl'
    assert 'Group' not in factors['3M_pre']       # ungrouped stays ungrouped


def test_a_user_defined_factor_joins_the_built_ins():
    app = _Facets(SUBJECTS, params={
        'factor_definitions': ['Sex'],
        'subject_factors': {'2F_pre': {'Sex': 'F'}, '2F_post': {'Sex': 'F'}}})
    assert 'Sex' in app.get_factor_definitions()
    labels, _ = app.facet(SUBJECTS, {'Sex': SPLIT})
    assert labels['2F_pre'] == 'F'
    assert labels['3M_pre'] == UNASSIGNED


def test_a_hand_assigned_level_overrides_the_derived_one():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre']},
                  params={'subject_factors': {'2F_pre': {'Group': 'Reassigned'}}})
    assert app.get_subject_factors()['2F_pre']['Group'] == 'Reassigned'


def test_an_empty_factor_still_gets_a_control():
    """A factor created but not yet assigned must appear, or it cannot be used."""
    app = _Facets(SUBJECTS, params={'factor_definitions': ['Drug']})
    assert 'Drug' in app.get_factor_definitions()


# ---------------------------------------------------------------------------
# Auto-assigning levels from the subject ID
# ---------------------------------------------------------------------------

def test_named_level_group_is_preferred():
    got = _M.levels_from_ids(SUBJECTS, r'_(?P<level>[^_]+)$')
    assert got == {'2F_pre': 'pre', '2F_post': 'post',
                   '3M_pre': 'pre', '3M_post': 'post'}


def test_first_capture_group_is_used_when_unnamed():
    assert _M.levels_from_ids(['2F_pre'], r'_(.+)$') == {'2F_pre': 'pre'}


def test_whole_match_is_used_when_nothing_is_captured():
    got = _M.levels_from_ids(SUBJECTS, r'pre|post')
    assert got['2F_pre'] == 'pre' and got['2F_post'] == 'post'


def test_unmatched_subjects_are_omitted_not_blanked():
    """A pattern for one cohort need not match another; blanking the rest would
    destroy hand assignments that the user made deliberately."""
    got = _M.levels_from_ids(SUBJECTS + ['Control7'], r'_(?P<level>[^_]+)$')
    assert 'Control7' not in got
    assert len(got) == 4


def test_an_empty_capture_is_not_an_assignment():
    assert _M.levels_from_ids(['2F_'], r'_(?P<level>.*)$') == {}


def test_a_malformed_pattern_raises_rather_than_guessing():
    with pytest.raises(re.error):
        _M.levels_from_ids(SUBJECTS, '(unclosed')


# ---------------------------------------------------------------------------
# Renaming and deleting a factor must touch all three structures at once
# ---------------------------------------------------------------------------

def _params():
    return {'factor_definitions': ['Drug'],
            'subject_factors': {'2F_pre': {'Drug': 'Fent', 'Sex': 'F'},
                                '3M_pre': {'Drug': 'Saline'}},
            'factor_level_order': {'Drug': ['Fent', 'Saline']}}


def test_rename_moves_definition_assignments_and_order_together():
    p = _M.rename_factor_in(_params(), 'Drug', 'Treatment')
    assert p['factor_definitions'] == ['Treatment']
    assert p['subject_factors']['2F_pre'] == {'Treatment': 'Fent', 'Sex': 'F'}
    assert p['factor_level_order'] == {'Treatment': ['Fent', 'Saline']}


def test_delete_leaves_nothing_that_could_resurrect_the_factor():
    p = _params()
    removed = _M.delete_factor_from(p, 'Drug')
    assert removed == 2
    assert p['factor_definitions'] == []
    assert p['factor_level_order'] == {}
    assert p['subject_factors'] == {'2F_pre': {'Sex': 'F'}, '3M_pre': {}}
    # get_factor_definitions() reads assignments too, so a stale one would
    # bring the factor back after it was deleted.
    app = _Facets(SUBJECTS, params=p)
    assert 'Drug' not in app.get_factor_definitions()


def test_delete_of_an_unknown_factor_is_a_no_op():
    p = _params()
    assert _M.delete_factor_from(p, 'Nope') == 0
    assert p['factor_definitions'] == ['Drug']
