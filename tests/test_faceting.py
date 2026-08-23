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

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


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
    # facet/facet_series derive the factor model from the whole project rather
    # than the selected pool, so the derived Session/Animal factors do not
    # vanish when a tab narrows its selection.
    _factor_universe = _G._factor_universe
    # The real property, so a test that sets groups exercises the same
    # conversion into Group-factor levels that the app performs.
    groups = _G.groups
    series_factor = _G.series_factor
    clear_subject_factors = _G.clear_subject_factors
    migrate_legacy_groups = _G.migrate_legacy_groups
    reset_factors = _G.reset_factors
    _apply_factor_payload = _G._apply_factor_payload
    _factor_payload = _G._factor_payload
    _FACTOR_DERIVED = _G._FACTOR_DERIVED

    def __init__(self, subjects, groups=None, params=None):
        self.params = {'session_pattern': _M.DEFAULT_SESSION_PATTERN,
                       'animal_overrides': {}, 'factor_definitions': [],
                       'subject_factors': {}, 'factor_level_order': {},
                       'series_factor': 'Group'}
        self.params.update(params or {})
        self.processed_data = {s: {} for s in subjects}
        self.log_message = lambda *a, **k: None
        self.root = None
        if groups:
            self.groups = groups


def test_group_and_session_are_available_without_being_configured():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post'],
                                    'Saline': ['3M_pre', '3M_post']})
    assert app.get_factor_definitions() == ['Group', 'Session', 'Animal']
    labels, _ = app.facet(SUBJECTS, {'Group': SPLIT, 'Session': SPLIT})
    assert labels['2F_post'] == 'Fentanyl × post'


def test_groups_are_stored_as_levels_of_the_group_factor():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post']})
    factors = app.get_subject_factors()
    assert factors['2F_pre']['Group'] == 'Fentanyl'
    assert 'Group' not in factors['3M_pre']       # ungrouped stays ungrouped
    # ...and that assignment is what is actually persisted.
    assert app.params['subject_factors']['2F_pre'] == {'Group': 'Fentanyl'}


def test_the_groups_view_reads_back_what_was_assigned():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']})
    assert app.groups == {'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']}


def test_groups_come_back_in_the_declared_plot_order():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']})
    app.params['factor_level_order']['Group'] = ['Saline', 'Fentanyl']
    assert list(app.groups) == ['Saline', 'Fentanyl']


def test_assigning_groups_replaces_rather_than_accumulates():
    """Loading a project's groups must not leave the previous project's behind."""
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post']})
    app.groups = {'Vehicle': ['3M_pre']}
    assert app.groups == {'Vehicle': ['3M_pre']}


def test_a_subject_in_two_groups_keeps_the_first():
    """A factor level is singular; the overlap is reported, not silently split."""
    legacy = {'Fentanyl': ['2F_pre'], 'Responders': ['2F_pre', '3M_pre']}
    app = _Facets(SUBJECTS, groups=legacy)
    assert app.get_subject_factors()['2F_pre']['Group'] == 'Fentanyl'
    assert _M.legacy_group_overlaps(legacy) == {'2F_pre': ['Fentanyl', 'Responders']}


def test_clearing_a_subject_takes_it_out_of_its_group():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post']})
    app.clear_subject_factors('2F_pre')
    assert app.groups == {'Fentanyl': ['2F_post']}


def test_a_user_defined_factor_joins_the_built_ins():
    app = _Facets(SUBJECTS, params={
        'factor_definitions': ['Sex'],
        'subject_factors': {'2F_pre': {'Sex': 'F'}, '2F_post': {'Sex': 'F'}}})
    assert 'Sex' in app.get_factor_definitions()
    labels, _ = app.facet(SUBJECTS, {'Sex': SPLIT})
    assert labels['2F_pre'] == 'F'
    assert labels['3M_pre'] == UNASSIGNED


def test_a_hand_assigned_level_overrides_the_derived_one():
    """Session is derived from the ID, but an explicit assignment still wins."""
    app = _Facets(SUBJECTS,
                  params={'subject_factors': {'2F_pre': {'Session': 'baseline'}}})
    assert app.get_subject_factors()['2F_pre']['Session'] == 'baseline'
    assert app.get_subject_factors()['2F_post']['Session'] == 'post'


# ---------------------------------------------------------------------------
# Animal is only a factor when it actually pools recordings
# ---------------------------------------------------------------------------

def test_animal_is_offered_when_subjects_share_an_animal():
    app = _Facets(SUBJECTS)
    assert 'Animal' in app.get_factor_definitions()
    assert app.get_subject_factors()['2F_pre']['Animal'] == '2F'


def test_animal_is_suppressed_when_it_is_one_level_per_subject():
    """One animal per recording makes Animal a restatement of the subject list
    rather than a way of dividing subjects -- a dead dropdown on every tab."""
    app = _Facets(['Control7', 'Control8', 'Treated3'])
    assert 'Animal' not in app.get_factor_definitions()
    assert 'Animal' not in app.get_subject_factors()['Control7']


def test_group_is_always_offered_even_before_anything_is_assigned():
    """It is the factor the graphing tabs plot, so an empty one is a prompt."""
    app = _Facets(['Control7', 'Control8'])
    assert app.get_factor_definitions()[0] == 'Group'


# ---------------------------------------------------------------------------
# Migrating a project saved before groups became a factor
# ---------------------------------------------------------------------------

def test_legacy_groups_convert_to_group_factor_assignments():
    payload = _M.legacy_groups_to_factors(
        {'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']})
    assert payload['subject_factors'] == {'2F_pre': {'Group': 'Fentanyl'},
                                          '3M_pre': {'Group': 'Saline'}}
    assert payload['factor_level_order'] == {'Group': ['Fentanyl', 'Saline']}


def test_loading_an_old_project_adopts_its_groups():
    app = _Facets(SUBJECTS)
    assert app.migrate_legacy_groups({'Fentanyl': ['2F_pre', '2F_post']}) is True
    assert app.groups == {'Fentanyl': ['2F_pre', '2F_post']}
    assert app._legacy_group_notice           # the user is told where they went


def test_migration_does_not_overwrite_assignments_the_project_already_has():
    """The legacy key is still written for backwards compatibility, so it can be
    a stale mirror of the factor assignments -- which are the newer copy."""
    app = _Facets(SUBJECTS, groups={'Vehicle': ['3M_pre']})
    assert app.migrate_legacy_groups({'Fentanyl': ['2F_pre']}) is False
    assert app.groups == {'Vehicle': ['3M_pre']}


def test_a_project_with_no_groups_at_all_is_left_alone():
    app = _Facets(SUBJECTS)
    assert app.migrate_legacy_groups({}) is False
    assert app.groups == {}


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


# ---------------------------------------------------------------------------
# Series: what a graphing tab actually draws
# ---------------------------------------------------------------------------

class _Series(_Facets):
    facet_series = _G.facet_series
    facet_series_for = _G.facet_series_for
    _series_members = _G._series_members
    selected_series = _G.selected_series
    fill_group_listbox = _G.fill_group_listbox

    def __init__(self, *a, selections=None, **kw):
        super().__init__(*a, **kw)
        self._active_facet = None
        self.facet_controls = {}
        if selections is not None:
            self.facet_controls['t'] = type(
                'Stub', (), {'selections': lambda _s: selections})()


GROUPS = {'Fentanyl': ['2F_pre', '2F_post'], 'Saline': ['3M_pre', '3M_post']}


def test_the_default_selection_is_recognised_as_the_plain_group_fan_out():
    assert _M.facet_is_inert({'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    assert not _M.facet_is_inert({'Group': SPLIT, 'Session': SPLIT})
    assert not _M.facet_is_inert({'Group': COMBINE})
    assert not _M.facet_is_inert({'Group': SPLIT, 'Session': 'pre'})


def test_an_untouched_tab_draws_one_series_per_group():
    """The default column reproduces the plain group fan-out -- now from a pool
    of subjects rather than from a list of group names."""
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    assert app.facet_series_for('t', SUBJECTS) == ['Fentanyl', 'Saline']
    assert app._series_members('Fentanyl') == ['2F_pre', '2F_post']


def test_deselecting_a_subject_shrinks_its_group_without_touching_factors():
    """The point of selecting subjects in group mode: drop one member of a group
    from the plot without editing the design on the Factors tab."""
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    pool = [s for s in SUBJECTS if s != '2F_post']
    assert app.facet_series_for('t', pool) == ['Fentanyl', 'Saline']
    assert app._series_members('Fentanyl') == ['2F_pre']
    assert app._series_members('Saline') == ['3M_pre', '3M_post']
    # The assignment itself is untouched, so the subject comes back on reselect.
    assert app.groups['Fentanyl'] == ['2F_pre', '2F_post']


def test_a_group_can_be_dropped_by_deselecting_all_its_members():
    """Selecting groups was the old unit of selection; it is still expressible."""
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    assert app.facet_series_for('t', GROUPS['Fentanyl']) == ['Fentanyl']


def test_splitting_session_regroups_the_selected_subjects():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': COMBINE, 'Session': SPLIT, 'Animal': COMBINE})
    series = app.facet_series_for('t', SUBJECTS)
    assert set(series) == {'pre', 'post'}
    assert set(app._series_members('pre')) == {'2F_pre', '3M_pre'}
    assert set(app._series_members('post')) == {'2F_post', '3M_post'}


def test_group_by_session_gives_the_cross_product():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': SPLIT, 'Animal': COMBINE})
    series = app.facet_series_for('t', SUBJECTS)
    assert len(series) == 4
    assert 'Fentanyl × pre' in series
    assert app._series_members('Fentanyl × post') == ['2F_post']


def test_filtering_to_a_level_shrinks_the_pool():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': 'post', 'Animal': COMBINE})
    series = app.facet_series_for('t', SUBJECTS)
    assert series == ['Fentanyl', 'Saline']
    assert app._series_members('Fentanyl') == ['2F_post']
    assert app._series_members('Saline') == ['3M_post']


def test_the_pool_is_the_selected_subjects_only():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': COMBINE, 'Session': SPLIT, 'Animal': COMBINE})
    app.facet_series_for('t', GROUPS['Fentanyl'])
    assert app._series_members('pre') == ['2F_pre']


def test_a_subject_listed_twice_is_pooled_once():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    app.facet_series_for('t', ['2F_pre', '2F_pre', '2F_post'])
    assert app._series_members('Fentanyl') == ['2F_pre', '2F_post']


def test_empty_series_do_not_take_a_colour():
    """A level nobody in the selection is at must not occupy a legend entry."""
    app = _Series(SUBJECTS, groups=GROUPS, params={
        'factor_definitions': ['Dose'],
        'subject_factors': {'2F_pre': {'Dose': 'high'}},
        'factor_level_order': {'Dose': ['high', 'low']}},
        selections={'Group': COMBINE, 'Session': COMBINE,
                    'Animal': COMBINE, 'Dose': SPLIT})
    series = app.facet_series_for('t', GROUPS['Fentanyl'])
    assert 'low' not in series
    assert series == ['high', UNASSIGNED]


def test_series_follow_the_declared_level_order_not_the_data():
    app = _Series(SUBJECTS, groups=GROUPS,
                  params={'factor_level_order': {'Session': ['pre', 'post']}},
                  selections={'Group': COMBINE, 'Session': SPLIT, 'Animal': COMBINE})
    assert app.facet_series_for('t', SUBJECTS) == ['pre', 'post']
    app.params['factor_level_order']['Session'] = ['post', 'pre']
    assert app.facet_series_for('t', SUBJECTS) == ['post', 'pre']


def test_a_tab_with_no_facet_control_falls_back_to_the_series_factor():
    """No column (or one not built yet) must still draw the historical plot
    rather than collapsing every subject into one series."""
    app = _Series(SUBJECTS, groups=GROUPS)
    assert app.facet_series_for('missing', SUBJECTS) == ['Fentanyl', 'Saline']
    assert app._series_members('Fentanyl') == ['2F_pre', '2F_post']


# ---------------------------------------------------------------------------
# The pool listbox: what group mode actually offers to select
# ---------------------------------------------------------------------------

class _Listbox:
    """Just enough of tk.Listbox for the pool selector."""

    def __init__(self):
        self._items, self._sel = [], set()

    def get(self, i):
        return self._items[i]

    def size(self):
        return len(self._items)

    def curselection(self):
        return tuple(sorted(self._sel))

    def delete(self, first, last=None):
        self._items, self._sel = [], set()

    def insert(self, where, value):
        self._items.append(value)

    def selection_set(self, i):
        self._sel.add(i)

    def selected(self):
        return [self._items[i] for i in sorted(self._sel)]


def test_group_mode_offers_every_subject_not_the_group_names():
    """The bug this replaced: a project whose subjects carry no Group level got
    an empty listbox and no way to plot anything."""
    app = _Series(SUBJECTS)                       # no group assignments at all
    assert app.groups == {}
    box = _Listbox()
    app.fill_group_listbox(box)
    assert box.selected() == sorted(SUBJECTS)


def test_an_untouched_pool_starts_fully_selected():
    app = _Series(SUBJECTS, groups=GROUPS)
    box = _Listbox()
    app.fill_group_listbox(box)
    assert box.selected() == sorted(SUBJECTS)


def test_refilling_keeps_a_narrowed_selection():
    """Factors are edited on another tab while a graphing tab sits there with a
    selection, so a refresh must not quietly widen the plot back out."""
    app = _Series(SUBJECTS, groups=GROUPS)
    box = _Listbox()
    app.fill_group_listbox(box)
    box._sel = {box._items.index('2F_pre')}
    app.fill_group_listbox(box)
    assert box.selected() == ['2F_pre']


def test_a_subject_that_leaves_the_project_leaves_the_pool():
    app = _Series(SUBJECTS, groups=GROUPS)
    box = _Listbox()
    app.fill_group_listbox(box)
    box._sel = {box._items.index(s) for s in ('2F_pre', '3M_pre')}
    del app.processed_data['2F_pre']
    app.fill_group_listbox(box)
    assert '2F_pre' not in box._items
    assert box.selected() == ['3M_pre']


def test_selecting_nothing_plots_everything():
    """An unused filter is inert -- an empty pool would mean the tab silently
    drew nothing the first time it was touched."""
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    box = _Listbox()
    app.fill_group_listbox(box)
    box._sel = set()
    assert app.selected_series('t', box) == ['Fentanyl', 'Saline']
    # Members follow the listbox, which lists subjects alphabetically.
    assert set(app._series_members('Fentanyl')) == {'2F_pre', '2F_post'}


def test_the_selection_reaches_the_series():
    app = _Series(SUBJECTS, groups=GROUPS,
                  selections={'Group': SPLIT, 'Session': COMBINE, 'Animal': COMBINE})
    box = _Listbox()
    app.fill_group_listbox(box)
    box._sel = {box._items.index(s) for s in ('2F_pre', '3M_post')}
    assert app.selected_series('t', box) == ['Fentanyl', 'Saline']
    assert app._series_members('Fentanyl') == ['2F_pre']
    assert app._series_members('Saline') == ['3M_post']


def test_facet_series_order_builds_the_cross_product_in_order():
    got = _M.facet_series_order(
        ['Group', 'Session'],
        {'Group': ['Fentanyl', 'Saline'], 'Session': ['pre', 'post']})
    assert got == ['Fentanyl × pre', 'Fentanyl × post',
                   'Saline × pre', 'Saline × post']


def test_nothing_split_is_a_single_series():
    assert _M.facet_series_order([], {}) == ['All']


# --------------------------------------------------------------------------
# Contextual option disclosure
# --------------------------------------------------------------------------
# An analysis missing from the table falls back to showing every cluster, which
# is the right default for an unknown plot type but wrong for a listed one --
# it would quietly undo the disclosure. A typo in one of the en-dash or "x"
# names is exactly how that happens, so pin the coverage.

def test_every_kinematics_analysis_has_a_disclosure_entry():
    missing = [a for a in _G.KIN_ANALYSES if a not in _M.KIN_OPTIONS_BY_PLOT]
    assert missing == []


def test_kinematics_disclosure_names_only_real_clusters():
    known = {'zone_kinematic', 'nbins', 'vel_zscore', 'spatial_bin', 'move_thresh'}
    for analysis, wanted in _M.KIN_OPTIONS_BY_PLOT.items():
        assert set(wanted) <= known, analysis


def test_kinematics_always_shown_settings_stay_out_of_the_table():
    # Temporal bin, smoothing and max lag are read by Summarize whatever
    # analysis is selected, so they are always visible rather than scoped.
    for wanted in _M.KIN_OPTIONS_BY_PLOT.values():
        assert 'timing' not in wanted


# ---------------------------------------------------------------------------
# Project boundaries and importing another project's assignments
# ---------------------------------------------------------------------------

def test_a_new_project_starts_with_no_factors_of_its_own():
    """params is merged key by key on load, so a project whose config predates a
    key would otherwise inherit the last project's assignments."""
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre']})
    app.params['subject_factors']['2F_pre']['Sex'] = 'F'
    app.reset_factors()
    assert app.params['subject_factors'] == {}
    assert app.params['factor_definitions'] == ['Group']
    assert app.groups == {}


def test_importing_assignments_overwrites_only_the_factors_it_carries():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre', '2F_post']})
    subjects, names = app._apply_factor_payload({
        'factor_definitions': ['Sex'],
        'subject_factors': {'2F_pre': {'Sex': 'F'}, '3M_pre': {'Sex': 'M'}},
        'factor_level_order': {'Sex': ['F', 'M']}}, merge=True)
    assert names == ['Sex'] and subjects == ['2F_pre', '3M_pre']
    factors = app.get_subject_factors()
    assert factors['2F_pre'] == {'Group': 'Fentanyl', 'Session': 'pre', 'Animal': '2F',
                                 'Sex': 'F'}
    assert app.groups == {'Fentanyl': ['2F_pre', '2F_post']}   # untouched
    assert app.params['factor_level_order']['Sex'] == ['F', 'M']


def test_a_round_trip_through_the_payload_preserves_the_assignments():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']})
    saved = app._factor_payload()
    fresh = _Facets(SUBJECTS)
    fresh._apply_factor_payload(saved, merge=False)
    assert fresh.groups == {'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']}


def test_an_import_never_creates_a_derived_factor_as_a_stored_one():
    """Session is derived; adding it to factor_definitions would leave a
    duplicate that outlives the IDs it came from."""
    app = _Facets(SUBJECTS)
    app._apply_factor_payload(
        {'subject_factors': {'2F_pre': {'Session': 'pre'}}}, merge=True)
    assert 'Session' not in app.params['factor_definitions']


# ---------------------------------------------------------------------------
# The series factor is a pointer, so "Group" is a default name, not a fixture
# ---------------------------------------------------------------------------

def test_the_series_factor_defaults_to_group():
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre']})
    assert app.series_factor == 'Group'


def test_renaming_the_series_factor_carries_the_groups_with_it():
    """The listboxes follow the pointer, so renaming Group does not empty them."""
    app = _Facets(SUBJECTS, groups={'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']})
    _M.rename_factor_in(app.params, 'Group', 'Drug')
    app.params['series_factor'] = 'Drug'
    assert app.series_factor == 'Drug'
    assert app.groups == {'Fentanyl': ['2F_pre'], 'Saline': ['3M_pre']}
    assert app.get_factor_definitions()[0] == 'Drug'


def test_nominating_another_factor_replots_by_it():
    app = _Facets(SUBJECTS, params={'subject_factors': {
        '2F_pre': {'Sex': 'F'}, '3M_pre': {'Sex': 'M'}}})
    app.params['series_factor'] = 'Sex'
    assert app.groups == {'F': ['2F_pre'], 'M': ['3M_pre']}


def test_a_declared_level_survives_having_no_members():
    """Naming the groups before filling them is how people work; a level that
    vanished the moment it was emptied would read as a bug."""
    app = _Facets(SUBJECTS)
    app.params['factor_level_order']['Group'] = ['Fentanyl', 'Saline']
    # No members yet, so nothing is plotted...
    assert app.groups == {}
    # ...but the declaration is what the level list and assign box read.
    assert app.params['factor_level_order']['Group'] == ['Fentanyl', 'Saline']
    app.params['subject_factors']['2F_pre'] = {'Group': 'Saline'}
    assert app.groups == {'Saline': ['2F_pre']}
