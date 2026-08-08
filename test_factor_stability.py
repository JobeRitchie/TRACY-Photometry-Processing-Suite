"""The factor model must not depend on which subjects a tab has selected.

Session and Animal are *derived* factors, and whether Animal exists at all
turns on some animal owning more than one recording. Deriving them from the
selected pool rather than the project therefore deleted them whenever the pool
was narrowed -- a level filter on Session matched nobody and drew an empty
plot, and a split on it collapsed to one 'All' series -- while the facet
dropdowns, built from the full project, still offered the levels.

Also pins that the series factor is read from the app rather than hard-coded to
'Group', so renaming it (or nominating another factor with "Plot as series")
does not leave every dropdown defaulting to COMBINE.
"""
import pytest

import fp_analysis_gui as G


SUBJECTS = ['2F_pre', '2F_post', '3M_pre', '3M_post']


def _app(subjects=SUBJECTS, series_factor=None):
    """An FPAnalysisGUI shell with a factor model and no Tk."""
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {} for s in subjects}
    app.params = {
        'subject_factors': {s: {'Group': 'Fent' if s.startswith('2F') else 'Sal'}
                            for s in subjects},
        'factor_definitions': [],
        'factor_level_order': {},
        'animal_overrides': {},
        'session_pattern': G.DEFAULT_SESSION_PATTERN,
        'series_factor': series_factor or 'Group',
    }
    app.log_message = lambda *a, **k: None
    app.root = None
    app._active_facet = None
    return app


def _labels(app, pool, selections):
    labels, splits = app.facet(pool, selections)
    return labels, splits


# ---------------------------------------------------------------------------
# derived factors survive a narrowed pool
# ---------------------------------------------------------------------------

def test_session_still_exists_for_a_narrowed_pool():
    """Repro B: pool one recording per animal, filter Session='pre'."""
    app = _app()
    pool = ['2F_pre', '3M_pre']
    factors = app.get_factor_definitions(pool)
    assert 'Session' in app.get_factor_definitions(SUBJECTS), 'precondition'

    labels, splits = _labels(app, pool, {'Group': G.FACTOR_SPLIT,
                                         'Session': 'pre'})
    assert labels, (
        'every subject was filtered out: the Session factor did not survive '
        'being derived from the narrowed pool')
    assert set(labels) == {'2F_pre', '3M_pre'}


def test_splitting_session_on_a_narrowed_pool_does_not_collapse():
    """Repro C: SPLIT on Session must not degrade to a single 'All' series."""
    app = _app()
    pool = ['2F_pre', '3M_pre']
    labels, splits = _labels(app, pool, {'Session': G.FACTOR_SPLIT})
    assert splits == ['Session'], f'Session was dropped from the split: {splits}'
    assert set(labels.values()) == {'pre'}, labels


def test_facet_series_keeps_the_pool_as_the_pool():
    """Narrowing the pool still narrows the data -- only the model is global."""
    app = _app()
    series = app.facet_series(['2F_pre', '3M_pre'], {'Group': G.FACTOR_SPLIT})
    members = {label: subs for label, subs in series}
    assert members == {'Fent': ['2F_pre'], 'Sal': ['3M_pre']}


def test_full_pool_is_unchanged():
    """The common case must behave exactly as before."""
    app = _app()
    series = app.facet_series(SUBJECTS, {'Group': G.FACTOR_SPLIT})
    members = {label: sorted(subs) for label, subs in series}
    assert members == {'Fent': ['2F_post', '2F_pre'],
                       'Sal': ['3M_post', '3M_pre']}


# ---------------------------------------------------------------------------
# the series factor is not hard-coded
# ---------------------------------------------------------------------------

def test_facet_is_inert_follows_the_series_factor():
    inert = {'Drug': G.FACTOR_SPLIT, 'Session': G.FACTOR_COMBINE}
    assert G.facet_is_inert(inert, series_factor='Drug')
    assert not G.facet_is_inert(inert, series_factor='Group')


def test_facet_is_inert_still_defaults_to_group():
    """Existing callers pass no series factor and must keep working."""
    assert G.facet_is_inert({'Group': G.FACTOR_SPLIT,
                             'Session': G.FACTOR_COMBINE})


def test_series_factor_reads_the_renamed_factor():
    app = _app(series_factor='Drug')
    assert app.series_factor == 'Drug'


# ---------------------------------------------------------------------------
# _active_facet must not outlive its tab
# ---------------------------------------------------------------------------

def test_series_members_falls_back_to_groups_without_a_facet():
    """Tabs with no facet column resolve real group membership."""
    app = _app()
    app._active_facet = None
    assert sorted(app._series_members('Fent')) == ['2F_post', '2F_pre']


def test_a_leftover_facet_would_narrow_another_tab():
    """Pins the mechanism the tab-switch clear exists to prevent."""
    app = _app()
    app._active_facet = {'Fent': ['2F_pre']}        # left by another tab
    assert app._series_members('Fent') == ['2F_pre']
    app._active_facet = None                        # what the tab switch does
    assert sorted(app._series_members('Fent')) == ['2F_post', '2F_pre']
