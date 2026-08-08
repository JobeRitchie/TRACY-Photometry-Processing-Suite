"""Factor-model hygiene: whitespace, declared levels, and refresh gaps.

Three separate defects, all in how the factor model is written and displayed:

* **Whitespace.** `assign_factor_level`, `_write_factor_cells`,
  `create_factor_level` and `rename_factor_level` all `.strip()`, but
  `levels_from_ids`, `_apply_factor_payload`, `legacy_groups_to_factors` and the
  `groups` setter did not.  Auto-assign from a subject-ID pattern, or an
  imported `.tracy`, could therefore mint `'Hi '` beside `'Hi'` -- two series
  where the user has one cohort, each with half the n.

* **Declared-but-empty levels.** `create_factor_level` writes only to
  `factor_level_order`; `factor_levels` kept a level only if some subject was at
  it.  Naming the groups before filling them -- which the `create_factor_level`
  docstring explicitly endorses -- therefore looked broken, and the level was
  missing from the very box you would assign it from.

* **Refresh gaps.** `_on_session_pattern_changed` creates and destroys the
  derived Session and Animal factors; `move_factor_level` changes series order.
  Neither told the grid or the facet dropdowns, so both self-healed only on the
  next tab switch.
"""
import pytest

import fp_analysis_gui as G


# --------------------------------------------------------------------------- #
#  whitespace                                                                  #
# --------------------------------------------------------------------------- #

def test_levels_from_ids_strips():
    """A pattern that captures a trailing space must not mint a second level."""
    out = G.levels_from_ids(['A1_pre ', 'A2_pre'], r'_(?P<level>.+)$')
    assert set(out.values()) == {'pre'}, out


def test_levels_from_ids_drops_a_whitespace_only_level():
    out = G.levels_from_ids(['A1_ '], r'_(?P<level>.+)$')
    assert out == {}


def test_legacy_groups_to_factors_strips():
    payload = G.legacy_groups_to_factors({'Hi ': ['S1'], 'Hi': ['S2']})
    levels = {lv[G.GROUP_FACTOR] for lv in payload['subject_factors'].values()}
    assert levels == {'Hi'}, 'a trailing space split one cohort into two'
    assert payload['factor_level_order'][G.GROUP_FACTOR] == ['Hi']


def test_legacy_groups_to_factors_drops_a_blank_name():
    payload = G.legacy_groups_to_factors({'  ': ['S1'], 'Real': ['S2']})
    assert payload['factor_level_order'][G.GROUP_FACTOR] == ['Real']
    assert 'S1' not in payload['subject_factors']


def _app(subjects=('S1', 'S2')):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {} for s in subjects}
    app.params = {
        'subject_factors': {},
        'factor_definitions': [],
        'factor_level_order': {},
        'animal_overrides': {},
        'session_pattern': G.DEFAULT_SESSION_PATTERN,
        'series_factor': 'Group',
    }
    app.log_message = lambda *a, **k: None
    app.root = None
    app._active_facet = None
    return app


def test_groups_setter_strips():
    app = _app()
    app.groups = {'Hi ': ['S1'], 'Hi': ['S2']}
    assert set(app.groups) == {'Hi'}
    assert sorted(app.groups['Hi']) == ['S1', 'S2']


def test_groups_setter_drops_a_blank_name():
    app = _app()
    app.groups = {'   ': ['S1'], 'Real': ['S2']}
    assert set(app.groups) == {'Real'}


def test_apply_factor_payload_strips():
    app = _app()
    app._apply_factor_payload({
        'factor_definitions': ['Drug '],
        'subject_factors': {'S1': {'Drug ': 'Fent '}, 'S2': {'Drug': 'Fent'}},
        'factor_level_order': {'Drug ': ['Fent ', 'Sal']},
    }, merge=False)

    assigned = app.params['subject_factors']
    assert assigned['S1'] == {'Drug': 'Fent'}
    assert assigned['S2'] == {'Drug': 'Fent'}
    assert 'Drug' in app.params['factor_definitions']
    assert 'Drug ' not in app.params['factor_definitions']
    assert app.params['factor_level_order']['Drug'] == ['Fent', 'Sal']


def test_apply_factor_payload_drops_blank_levels():
    app = _app()
    app._apply_factor_payload({
        'factor_definitions': ['Drug'],
        'subject_factors': {'S1': {'Drug': '   '}, 'S2': {'Drug': 'Fent'}},
        'factor_level_order': {},
    }, merge=False)
    assert app.params['subject_factors'].get('S1', {}) == {}
    assert app.params['subject_factors']['S2'] == {'Drug': 'Fent'}


# --------------------------------------------------------------------------- #
#  declared-but-empty levels                                                   #
# --------------------------------------------------------------------------- #

def test_declared_level_is_offered_when_asked_for():
    levels = G.factor_levels(
        ['S1'], {'S1': {'Drug': 'Fent'}}, 'Drug',
        level_order=['Fent', 'Saline'], include_declared=True)
    assert 'Saline' in levels, (
        'a level you just declared is missing from the box you assign it from')
    assert levels.index('Fent') < levels.index('Saline'), 'plot order lost'


def test_declared_level_stays_out_of_plotting_by_default():
    """The default must not change: an empty level would be an empty series."""
    levels = G.factor_levels(
        ['S1'], {'S1': {'Drug': 'Fent'}}, 'Drug',
        level_order=['Fent', 'Saline'])
    assert 'Saline' not in levels


def test_declared_levels_do_not_duplicate_present_ones():
    levels = G.factor_levels(
        ['S1', 'S2'], {'S1': {'Drug': 'Fent'}, 'S2': {'Drug': 'Saline'}},
        'Drug', level_order=['Fent', 'Saline'], include_declared=True)
    assert levels.count('Fent') == 1 and levels.count('Saline') == 1


def test_declared_level_does_not_create_an_empty_series():
    """facet_series drops labels with no members, so the dropdown is safe."""
    app = _app(('S1', 'S2'))
    app.params['subject_factors'] = {'S1': {'Group': 'Fent'}, 'S2': {'Group': 'Fent'}}
    app.params['factor_level_order'] = {'Group': ['Fent', 'Saline']}
    series = app.facet_series(['S1', 'S2'], {'Group': G.FACTOR_SPLIT})
    assert [name for name, _ in series] == ['Fent']


# --------------------------------------------------------------------------- #
#  refresh gaps                                                                #
# --------------------------------------------------------------------------- #

class _Recorder:
    def __init__(self, app):
        self.calls = []
        for name in ('refresh_factors_display', 'refresh_facet_controls',
                     '_refresh_group_name_lists'):
            setattr(app, name, (lambda n: lambda *a, **k: self.calls.append(n))(name))


def test_refresh_factor_displays_hits_all_three():
    app = _app()
    rec = _Recorder(app)
    app._refresh_factor_displays()
    assert set(rec.calls) == {'refresh_factors_display', 'refresh_facet_controls',
                              '_refresh_group_name_lists'}


def test_refresh_factor_displays_survives_an_unbuilt_tab():
    app = _app()
    rec = _Recorder(app)

    def _boom(*a, **k):
        raise AttributeError('facet controls not built yet')

    app.refresh_facet_controls = _boom
    app._refresh_factor_displays()          # must not raise
    assert 'refresh_factors_display' in rec.calls
    assert '_refresh_group_name_lists' in rec.calls


def test_session_pattern_change_refreshes_the_displays():
    app = _app()
    rec = _Recorder(app)
    app.session_pattern_custom_var = type('V', (), {'get': lambda s: r'_(?P<session>.+)$'})()
    app._animal_map_adopted = 'stale'
    app._refresh_identity_preview = lambda: None

    app._on_session_pattern_changed(from_entry=True)

    assert 'refresh_factors_display' in rec.calls, (
        'the Session/Animal factors changed but the grid was not told')
    assert 'refresh_facet_controls' in rec.calls


def test_move_factor_level_refreshes_the_dropdowns():
    app = _app()
    app.params['factor_level_order'] = {'Drug': ['Fent', 'Saline']}
    app._factor_level_names = ['Fent', 'Saline']
    app._selected_factor = lambda: 'Drug'
    app.factor_level_listbox = type(
        'LB', (), {'curselection': lambda s: (1,), 'selection_set': lambda s, i: None})()
    app._on_factor_selected = lambda: None
    rec = _Recorder(app)

    app.move_factor_level(-1)

    assert app.params['factor_level_order']['Drug'] == ['Saline', 'Fent']
    assert 'refresh_facet_controls' in rec.calls, (
        'series order changed but the plots and dropdowns were not told')
    assert '_refresh_group_name_lists' in rec.calls


# --------------------------------------------------------------------------- #
#  dead code                                                                   #
# --------------------------------------------------------------------------- #

def test_dead_exclusion_helpers_are_gone():
    """Both were unreferenced; `_apply_exclusions` ignored the toggle entirely,
    so wiring it up as the chokepoint would have reproduced the spike bug."""
    assert not hasattr(G.FPAnalysisGUI, 'filter_subjects_by_exclusions')
    assert not hasattr(G.FPAnalysisGUI, '_apply_exclusions')
