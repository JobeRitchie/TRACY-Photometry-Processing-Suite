"""Coherence resolves its series from the facet column and its members through
`_series_members`.

Two defects meet here.  P9: within one tab two functions disagreed -- the
progress *total* was computed with `_series_members` (facet-aware) while the
worker that actually iterates subjects read `self.groups` directly, so a
facet-derived series name (``'Fentanyl x pre'``, ``'(unassigned)'``) found zero
members and the tab returned nothing without saying why.  P10: the tab had no
facet column at all, so those names could never be produced in the first place
-- the lists held group *names*, which made the series factor the only one
reachable.

The lists now hold the subject pool and a facet column beside them says how it
divides, exactly as on the other graphing tabs.  These tests drive the real
entry points with a stub facet column and pin both halves: that a non-trivial
split reaches every member of every series, and that an untouched column still
reproduces the plain one-series-per-group fan-out.

The workers all emit a ``('status', "Group 'X' - subj (i/n)")`` line for each
member *before* touching any signal, so driving them with data that cannot be
analysed still reveals exactly which subjects were visited -- which is the only
thing under test here.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


# A second factor beside Group, so a two-factor split is available. The ids
# carry no underscore on purpose: that keeps the derived Session and Animal
# factors out of the model, leaving exactly the two factors under test.
FENT = ['Fent1', 'Fent2']
SAL = ['Sal1']
ALL_SUBJECTS = FENT + SAL

# What splitting both factors produces. Neither is a key in self.groups.
FACET_SERIES = 'Fentanyl × pre'
SAL_SERIES = 'Saline × pre'


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Listbox:
    """Just enough of a Tk listbox for `selected_series` to read a pool."""

    def __init__(self, entries, selected=None):
        self._entries = list(entries)
        self._selected = (tuple(range(len(self._entries))) if selected is None
                          else tuple(selected))

    def curselection(self):
        return self._selected

    def size(self):
        return len(self._entries)

    def get(self, i):
        return self._entries[i]


class _FacetControl:
    """Stands in for the tab's FacetControls column."""

    def __init__(self, selections):
        self._selections = dict(selections)

    def selections(self):
        return dict(self._selections)


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)

    def visited(self):
        """Subject names that reached a status line, in order."""
        out = []
        for kind, msg in self.items:
            if kind != 'status':
                continue
            # "Group 'X' - subj  (1/2)"  ->  subj
            head, _, tail = msg.rpartition('(')
            name = head.split('—')[-1].strip()
            if name:
                out.append(name)
        return out


def _base_app(facet_key='coherence_groups', selections=None):
    """An FPAnalysisGUI with no Tk, faceted so `self.groups` cannot answer."""
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {} for s in ALL_SUBJECTS}
    app.exclusions = {}
    app.params = {
        'subject_factors': {s: {'Group': 'Fentanyl' if s in FENT else 'Saline',
                                'Phase': 'pre'}
                            for s in ALL_SUBJECTS},
        'factor_definitions': [],
        'factor_level_order': {},
        'animal_overrides': {},
        'session_pattern': G.DEFAULT_SESSION_PATTERN,
        'series_factor': 'Group',
    }
    app.root = None
    app._active_facet = None
    # Split both factors: the series names are facet-derived, so a worker that
    # looks them up in self.groups finds nothing.
    app.facet_controls = {facet_key: _FacetControl(
        selections if selections is not None
        else {'Group': G.FACTOR_SPLIT, 'Phase': G.FACTOR_SPLIT})}
    app.conn_channel1_var = _Var('G0')
    app.conn_channel2_var = _Var('G1')
    app.use_exclusions_conn = _Var(False)
    app.conn_params = {
        'static_fmin': 0.5, 'static_fmax': 4.0, 'static_nperseg_sec': 2.0,
        'coherence_method': 'Welch', 'morlet_n_freqs': 50, 'morlet_w': 6.0,
        'freq_bands': [{'name': 'Slow', 'fmin': 0.5, 'fmax': 1.0}],
    }
    app.get_fps = lambda: 30.0
    app.log_message = lambda *a, **k: None
    # No usable signal: every subject is skipped straight after its status line.
    app._get_channel_signal = lambda data, ch: None
    return app


def _unfaceted_app(**kwargs):
    """The same app with the column left in its inert 'split Group' state."""
    return _base_app(selections={'Group': G.FACTOR_SPLIT,
                                 'Phase': G.FACTOR_COMBINE}, **kwargs)


def _capture_worker(app):
    """Swap `_run_with_progress` for a recorder and return the captured worker."""
    box = {}

    def _fake(title, worker_fn, on_done_fn=None, **k):
        box['worker'] = worker_fn

    app._run_with_progress = _fake
    return box


# --------------------------------------------------------------------------- #
#  the facet column is what produces the series                                #
# --------------------------------------------------------------------------- #

def test_grp_comp_series_come_from_the_facet_column():
    app = _base_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)

    assert app._grp_comp_series() == [FACET_SERIES, SAL_SERIES]


def test_grp_comp_series_untouched_column_is_the_old_group_fanout():
    app = _unfaceted_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)

    assert app._grp_comp_series() == ['Fentanyl', 'Saline']


def test_grp_comp_pool_narrows_the_series():
    """Deselecting a subject drops it from its series, without editing factors."""
    app = _base_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS, selected=[0, 2])

    assert app._grp_comp_series() == [FACET_SERIES, SAL_SERIES]
    assert app._series_members(FACET_SERIES) == [FENT[0]]


def test_conn_selection_group_mode_uses_the_facet_column():
    app = _base_app(facet_key='coherence')
    app.conn_listbox = _Listbox(ALL_SUBJECTS)
    app.conn_mode_var = _Var('Group')

    names, group_mode = app._conn_selection()
    assert group_mode is True
    assert names == [FACET_SERIES, SAL_SERIES]


def test_conn_selection_subject_mode_returns_subjects():
    """Subject mode plots one series per row and must not consult the column."""
    app = _base_app(facet_key='coherence')
    app.conn_listbox = _Listbox(ALL_SUBJECTS, selected=[0, 1])
    app.conn_mode_var = _Var('Subject')

    names, group_mode = app._conn_selection()
    assert group_mode is False
    assert names == FENT


# --------------------------------------------------------------------------- #
#  run_group_coherence_comparison                                              #
# --------------------------------------------------------------------------- #

def test_group_coherence_comparison_visits_facet_members():
    app = _base_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)
    box = _capture_worker(app)

    app.run_group_coherence_comparison()
    q = _Queue()
    box['worker'](q)

    assert q.visited() == ALL_SUBJECTS


def test_group_coherence_comparison_total_matches_members_visited():
    """The count in the status line and the loop must come from one source."""
    app = _base_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)
    box = _capture_worker(app)

    app.run_group_coherence_comparison()
    q = _Queue()
    box['worker'](q)

    totals = {msg.rpartition('/')[2].rstrip(')')
              for kind, msg in q.items if kind == 'status'}
    assert totals == {'3'}
    assert len(q.visited()) == 3


def test_group_coherence_comparison_unfaceted_still_uses_groups():
    """With the column untouched the series name is a plain group name."""
    app = _unfaceted_app()
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)
    box = _capture_worker(app)

    app.run_group_coherence_comparison()
    q = _Queue()
    box['worker'](q)

    assert q.visited() == ALL_SUBJECTS


# --------------------------------------------------------------------------- #
#  run_bout_epoch_group_comparison                                             #
# --------------------------------------------------------------------------- #

def _bout_epoch_app(unfaceted=False):
    app = _unfaceted_app() if unfaceted else _base_app()
    app.beg_behavior_var = _Var('Sniff')
    app.grp_comp_listbox = _Listbox(ALL_SUBJECTS)
    app.beg_groups_listbox = app.grp_comp_listbox
    app.bout_epoch_pre_var = _Var('2')
    app.bout_epoch_post_var = _Var('2')
    app.bout_epoch_max_bouts_var = _Var('0')
    app.boutframes_path_var = _Var('')
    # Reached only after the status line; empty bout stores end each subject.
    app.processed_data = {s: {'bouts': {}} for s in ALL_SUBJECTS}
    return app


def test_bout_epoch_group_comparison_visits_facet_members():
    app = _bout_epoch_app()
    box = _capture_worker(app)

    app.run_bout_epoch_group_comparison()
    q = _Queue()
    box['worker'](q)

    assert q.visited() == ALL_SUBJECTS


def test_bout_epoch_group_comparison_unfaceted_still_uses_groups():
    app = _bout_epoch_app(unfaceted=True)
    box = _capture_worker(app)

    app.run_bout_epoch_group_comparison()
    q = _Queue()
    box['worker'](q)

    assert q.visited() == ALL_SUBJECTS


# --------------------------------------------------------------------------- #
#  _run_group_connectivity                                                     #
# --------------------------------------------------------------------------- #

def test_group_connectivity_accepts_resolved_members():
    """The caller resolves membership on the main thread and passes it down."""
    app = _base_app()
    q = _Queue()

    app._run_group_connectivity(
        [FACET_SERIES], 'G0', 'G1', ['static'], {'fs': 30.0},
        progress_q=q, total=2, members={FACET_SERIES: list(FENT)})

    assert q.visited() == FENT


def test_group_connectivity_without_members_falls_back_to_series_members():
    """Called with no map it must still resolve a facet name, not skip it."""
    app = _base_app()
    app._active_facet = {FACET_SERIES: list(FENT)}
    q = _Queue()

    app._run_group_connectivity(
        [FACET_SERIES], 'G0', 'G1', ['static'], {'fs': 30.0},
        progress_q=q, total=2)

    assert q.visited() == FENT


def test_group_connectivity_skips_empty_series_without_error():
    app = _base_app()
    q = _Queue()

    app._run_group_connectivity(
        ['Nobody'], 'G0', 'G1', ['static'], {'fs': 30.0},
        progress_q=q, total=0, members={'Nobody': []})

    assert q.visited() == []
