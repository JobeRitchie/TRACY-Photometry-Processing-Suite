"""P10: Coherence, Spike Analysis and Signal Linkage gain facet controls.

Three tabs were left out of the factor refactor's thread B3.  Coherence and
Spike listed group *names* rather than a subject pool, so the series factor was
the only factor they could reach -- no other split, no level filter, and no way
to narrow the pool without editing the factor assignments themselves.  Signal
Linkage exposed no grouping at all: it pooled every selected subject into one
analysis and `sig_link_by_var` was declared and never read.

Each tab now holds the subject pool and a facet column says how it divides.
The two halves that matter are pinned here:

* a split must actually reach the analysis -- the series names, the subjects
  behind them, and the labels that reach the plots and exports;
* an untouched column must reproduce what the tab did before it existed.  That
  default is *not* the same on all three: Coherence and Spike drew one series
  per group, so their columns start on "split the series factor"; Signal
  Linkage pooled everything, so its column starts fully combined and is
  constructed with ``default_split=False``.
"""
import matplotlib
# Before fp_analysis_gui, which imports the Tk backend: the figure test below
# builds a real figure, and asking for a GUI canvas off the main thread makes
# pyplot dump a stack trace at import time.
matplotlib.use('Agg')

import tkinter as tk

import numpy as np
import pytest

import fp_analysis_gui as G


FENT = ['Fent1', 'Fent2']
SAL = ['Sal1']
ALL_SUBJECTS = FENT + SAL

SPLIT_BOTH = {'Group': G.FACTOR_SPLIT, 'Phase': G.FACTOR_SPLIT}
SPLIT_GROUP = {'Group': G.FACTOR_SPLIT, 'Phase': G.FACTOR_COMBINE}
COMBINE_ALL = {'Group': G.FACTOR_COMBINE, 'Phase': G.FACTOR_COMBINE}

FENT_SERIES = 'Fentanyl × pre'
SAL_SERIES = 'Saline × pre'


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Listbox:
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
    def __init__(self, selections):
        self._selections = dict(selections)

    def selections(self):
        return dict(self._selections)


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _factor_params(**extra):
    """The params a factor model needs on an ``object.__new__`` shell.

    Ids carry no underscore, which keeps the derived Session and Animal factors
    out of the model and leaves exactly Group and Phase under test.
    """
    params = {
        'subject_factors': {s: {'Group': 'Fentanyl' if s in FENT else 'Saline',
                                'Phase': 'pre'}
                            for s in ALL_SUBJECTS},
        'factor_definitions': [],
        'factor_level_order': {},
        'animal_overrides': {},
        'session_pattern': G.DEFAULT_SESSION_PATTERN,
        'series_factor': 'Group',
    }
    params.update(extra)
    return params


def _shell(facet_key, selections):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {} for s in ALL_SUBJECTS}
    app.exclusions = {}
    app.params = _factor_params()
    app.root = None
    app._active_facet = None
    app.log_message = lambda *a, **k: None
    app.facet_controls = {facet_key: _FacetControl(selections)}
    return app


# --------------------------------------------------------------------------- #
#  FacetControls.default_split                                                 #
# --------------------------------------------------------------------------- #

# ``tk_root`` is session-scoped and lives in conftest.py. It used to be a
# module-scoped Tk() here, which turned six real test_plot_scroll tests into
# silent skips on every full-suite run -- see the note in conftest.py.


@pytest.fixture
def built_column(tk_root):
    """A real FacetControls, rebuilt, so the defaults under test are the ones
    the widget actually starts in rather than a copy of the expression."""
    windows = []

    def _build(default_split):
        win = tk.Toplevel(tk_root)
        windows.append(win)
        app = _shell('x', {})
        app.ui_px = lambda n: n
        column = G.FacetControls(win, app, default_split=default_split)
        column.rebuild()
        return column.selections()

    yield _build
    for win in windows:
        try:
            win.destroy()
        except tk.TclError:
            pass


def test_untouched_column_splits_the_series_factor_by_default(built_column):
    """The state that reproduces one series per group, for the tabs that drew
    exactly that before the column existed."""
    assert built_column(True) == {'Group': G.FACTOR_SPLIT,
                                  'Phase': G.FACTOR_COMBINE}


def test_default_split_false_starts_fully_combined(built_column):
    """Signal Linkage pooled every selected subject into one analysis, so the
    ordinary default would silently change what it computes the first time the
    tab is opened."""
    assert built_column(False) == {'Group': G.FACTOR_COMBINE,
                                   'Phase': G.FACTOR_COMBINE}


def test_signal_linkage_column_is_built_combined_and_the_others_split():
    """Pin the construction sites, not just the parameter."""
    import inspect
    src = inspect.getsource(G.FPAnalysisGUI.create_signal_linkage_tab)
    assert "'signal_linkage'" in src and 'default_split=False' in src

    for fn in (G.FPAnalysisGUI.create_connectivity_analysis_tab,
               G.FPAnalysisGUI.create_spike_analysis_tab):
        src = inspect.getsource(fn)
        assert '_make_facet_controls' in src, fn.__name__
        assert 'default_split=False' not in src, (
            f'{fn.__name__} drew one series per group, so its column must '
            f'start split')


# --------------------------------------------------------------------------- #
#  Spike Analysis                                                              #
# --------------------------------------------------------------------------- #

def _spike_app(selections=SPLIT_BOTH, mode='Group', selected=None):
    app = _shell('spike', selections)
    app.spike_listbox = _Listbox(ALL_SUBJECTS, selected=selected)
    app.spike_mode_var = _Var(mode)
    return app


def test_spike_group_mode_series_come_from_the_facet_column():
    app = _spike_app()
    names, group_mode = app._spike_series()
    assert group_mode is True
    assert names == [FENT_SERIES, SAL_SERIES]


def test_spike_untouched_column_is_the_old_group_fanout():
    app = _spike_app(selections=SPLIT_GROUP)
    assert app._spike_series()[0] == ['Fentanyl', 'Saline']


def test_spike_subject_mode_still_lists_subjects():
    app = _spike_app(mode='Subject', selected=[0, 2])
    names, group_mode = app._spike_series()
    assert group_mode is False
    assert names == [FENT[0], SAL[0]]


def test_spike_row_labels_follow_the_facet_not_the_groups():
    """The six plot/export sites label each row with the series it came from.

    They used to build that map by walking `self.groups`, so a run split on two
    factors was plotted and exported back under plain group names.
    """
    app = _spike_app()
    assert app._spike_subject_to_series() == {
        FENT[0]: FENT_SERIES, FENT[1]: FENT_SERIES, SAL[0]: SAL_SERIES}


def test_spike_row_labels_drop_a_deselected_subject():
    app = _spike_app(selected=[0, 2])
    mapping = app._spike_subject_to_series()
    assert FENT[1] not in mapping, (
        'a subject the pool deselected was still labelled and drawn')
    assert mapping == {FENT[0]: FENT_SERIES, SAL[0]: SAL_SERIES}


def test_spike_row_labels_in_subject_mode_keep_plain_group_membership():
    """Subject mode has no series; the label there is decorative, as before."""
    app = _spike_app(mode='Subject')
    assert app._spike_subject_to_series() == {
        FENT[0]: 'Fentanyl', FENT[1]: 'Fentanyl', SAL[0]: 'Saline'}


def test_spike_selected_subjects_expands_the_facet_series():
    app = _spike_app(selected=[0, 2])
    assert app._spike_selected_subjects() == {FENT[0], SAL[0]}


# --------------------------------------------------------------------------- #
#  Coherence                                                                   #
# --------------------------------------------------------------------------- #

def _conn_app(selections=SPLIT_BOTH, mode='Group'):
    app = _shell('coherence', selections)
    app.conn_listbox = _Listbox(ALL_SUBJECTS)
    app.conn_mode_var = _Var(mode)
    return app


def test_conn_group_mode_series_come_from_the_facet_column():
    app = _conn_app()
    assert app._conn_selection() == ([FENT_SERIES, SAL_SERIES], True)


def test_conn_untouched_column_is_the_old_group_fanout():
    app = _conn_app(selections=SPLIT_GROUP)
    assert app._conn_selection() == (['Fentanyl', 'Saline'], True)


def test_conn_subject_mode_does_not_consult_the_column():
    app = _conn_app(mode='Subject')
    assert app._conn_selection() == (ALL_SUBJECTS, False)


def test_run_connectivity_group_mode_reaches_the_worker(monkeypatch):
    """The channel guard used to abort every group-mode run.

    `missing_ch2` was built by looking each *selected item* up in
    processed_data.  In Group mode those were group names, so nothing was
    found, both counts came out 0, and `len(missing) == len(present)` fired
    "Channel Not Available" before the worker ever started.
    """
    # Captured, not shown: the bug aborts through showwarning, and letting it
    # reach Tk turns a clean assertion into a TclError.
    warned = []
    monkeypatch.setattr(G.messagebox, 'showwarning',
                        lambda *a, **k: warned.append(a))

    app = _conn_app()
    app.conn_static_var = _Var(True)
    app.conn_sliding_var = _Var(False)
    app.conn_rolling_var = _Var(False)
    app.conn_channel1_var = _Var('G0')
    app.conn_channel2_var = _Var('G1')
    app.conn_preset_var = _Var('Long Session')
    app.conn_params = {
        'static_fmin': 0.5, 'static_fmax': 4.0, 'static_nperseg_sec': 20.0,
        'sliding_win_sec': 60.0, 'sliding_step_sec': 5.0,
        'sliding_nperseg_sec': 10.0, 'rolling_win_sec': 60.0,
        'rolling_step_sec': 5.0, 'coherence_method': 'Welch',
        'morlet_n_freqs': 50, 'morlet_w': 6.0,
    }
    app.get_fps = lambda: 30.0
    app.connectivity_results = {}
    app.conn_tree = type('T', (), {'delete': lambda self, *a: None,
                                   'get_children': lambda self: []})()
    # Every subject has both channels, so the guard has no reason to fire.
    app._get_channel_signal = lambda data, ch: np.zeros(100)

    captured = {}

    def _fake(title, worker_fn, on_done_fn=None, **k):
        captured['worker'] = worker_fn

    app._run_with_progress = _fake

    visited = []

    def _group_conn(items, ch1, ch2, types, params, progress_q=None,
                    total=None, members=None):
        visited.append((list(items), dict(members or {}), total))

    app._run_group_connectivity = _group_conn
    app._run_subject_connectivity = lambda *a, **k: pytest.fail(
        'group mode took the subject path')

    app.run_connectivity_analysis()
    assert 'worker' in captured, (
        f'the run aborted before starting ({warned}); the channel guard fired '
        f'on a group name it could not find in processed_data')
    captured['worker'](_Queue())

    items, members, total = visited[0]
    assert items == [FENT_SERIES, SAL_SERIES]
    assert members == {FENT_SERIES: FENT, SAL_SERIES: SAL}
    assert total == 3


# --------------------------------------------------------------------------- #
#  Signal Linkage                                                              #
# --------------------------------------------------------------------------- #

def _siglink_app(selections=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {s: {} for s in ALL_SUBJECTS}
    app.exclusions = {}
    app.params = _factor_params(preboutframes=150, postboutframes=150)
    app.root = None
    app._active_facet = None
    app.log_message = lambda *a, **k: None
    app.facet_controls = ({'signal_linkage': _FacetControl(selections)}
                          if selections is not None else {})
    return app


def test_siglink_no_factors_still_pools_into_one_series():
    """The historical behaviour: one analysis over every selected subject.

    Not the series-factor fallback the other tabs use -- that would relabel the
    lone series '(unassigned)' on a project with no factors assigned.
    """
    app = _siglink_app()
    app.params['subject_factors'] = {}
    series = app._sig_link_series(ALL_SUBJECTS)
    assert series == [('All', ALL_SUBJECTS)]


def test_siglink_untouched_column_pools_into_one_series():
    app = _siglink_app(COMBINE_ALL)
    assert app._sig_link_series(ALL_SUBJECTS) == [('All', ALL_SUBJECTS)]


def test_siglink_split_produces_one_series_per_level():
    app = _siglink_app(SPLIT_BOTH)
    assert app._sig_link_series(ALL_SUBJECTS) == [
        (FENT_SERIES, FENT), (SAL_SERIES, SAL)]


def _fake_compute(app, scores):
    """Stub `_compute_signal_linkage`, recording the pool each series got.

    ``scores`` is ``{behavior: {subject_key: linkage_index}}`` keyed by the
    first subject of the series, so each series can be given its own result.
    """
    app.seen = []

    def _compute(q, subs, channel, behaviors, cfg):
        app.seen.append(list(subs))
        results = {}
        for beh, by_series in scores.items():
            li = by_series.get(subs[0])
            if li is None:
                continue
            results[beh] = {
                'behavior': beh, 'linkage_index': li, 'verdict': 'Not linked',
                'n': 4, 'delta': 0.1, 'peak_rate': 0.2, 'peak_lat': 0.0,
                'reliability': 0.3, 'z': 1.0, 'p': 0.5,
                'mean_tr': np.zeros(300), 'sem': np.zeros(300),
            }
        order = sorted(results, key=lambda b: results[b]['linkage_index'],
                       reverse=True)
        skipped = [b for b in scores if b not in results]
        return {'results': results, 'order': order, 'subjects': list(subs),
                'skipped': skipped, 'cfg': cfg, 'channel': channel,
                'prebout': 150, 'total': 300}

    app._compute_signal_linkage = _compute


def test_siglink_computes_each_series_independently():
    """Pooling and splitting only at display time would compare each series
    against the other's permutation baseline."""
    app = _siglink_app(SPLIT_BOTH)
    _fake_compute(app, {'Sniff': {FENT[0]: 80.0, SAL[0]: 20.0}})

    res = app._compute_signal_linkage_series(
        _Queue(), [(FENT_SERIES, FENT), (SAL_SERIES, SAL)], 'G0', ['Sniff'],
        {'fps': 30.0, 'n_perm': 100})

    assert app.seen == [FENT, SAL]
    assert [label for label, _p in res['series']] == [FENT_SERIES, SAL_SERIES]
    assert res['series'][0][1]['results']['Sniff']['linkage_index'] == 80.0
    assert res['series'][1][1]['results']['Sniff']['linkage_index'] == 20.0
    assert res['subjects'] == ALL_SUBJECTS


def test_siglink_behavior_order_is_global_across_series():
    """Every series' table and every figure panel must list behaviors alike."""
    app = _siglink_app(SPLIT_BOTH)
    _fake_compute(app, {
        # Sniff wins in one series, Groom in the other; the mean decides.
        'Sniff': {FENT[0]: 90.0, SAL[0]: 10.0},
        'Groom': {FENT[0]: 20.0, SAL[0]: 30.0},
    })

    res = app._compute_signal_linkage_series(
        _Queue(), [(FENT_SERIES, FENT), (SAL_SERIES, SAL)], 'G0',
        ['Sniff', 'Groom'], {'fps': 30.0, 'n_perm': 100})

    assert res['order'] == ['Sniff', 'Groom']


def test_siglink_behavior_measured_in_one_series_is_not_reported_skipped():
    app = _siglink_app(SPLIT_BOTH)
    _fake_compute(app, {
        'Sniff': {FENT[0]: 90.0, SAL[0]: 10.0},
        'Groom': {FENT[0]: 20.0},          # no bouts in the Saline series
        'Rear': {},                        # nowhere at all
    })

    res = app._compute_signal_linkage_series(
        _Queue(), [(FENT_SERIES, FENT), (SAL_SERIES, SAL)], 'G0',
        ['Sniff', 'Groom', 'Rear'], {'fps': 30.0, 'n_perm': 100})

    assert 'Groom' in res['order']
    assert res['skipped'] == ['Rear']


def test_siglink_single_series_keeps_the_original_layout():
    app = _siglink_app(COMBINE_ALL)
    _fake_compute(app, {'Sniff': {ALL_SUBJECTS[0]: 55.0}})

    res = app._compute_signal_linkage_series(
        _Queue(), [('All', ALL_SUBJECTS)], 'G0', ['Sniff'],
        {'fps': 30.0, 'n_perm': 100})

    assert G.FPAnalysisGUI._sig_link_is_split(res) is False


class _Clipboard:
    def __init__(self):
        self.text = ''

    def clipboard_clear(self):
        self.text = ''

    def clipboard_append(self, s):
        self.text += s


def _copied(app, res):
    app._sig_linkage_results = res
    app.root = _Clipboard()
    app.sig_link_status_var = _Var('')
    app._copy_signal_linkage_table()
    return app.root.text.splitlines()


def test_siglink_copy_gains_a_series_column_when_split():
    app = _siglink_app(SPLIT_BOTH)
    _fake_compute(app, {'Sniff': {FENT[0]: 80.0, SAL[0]: 20.0}})
    res = app._compute_signal_linkage_series(
        _Queue(), [(FENT_SERIES, FENT), (SAL_SERIES, SAL)], 'G0', ['Sniff'],
        {'fps': 30.0, 'n_perm': 100})

    lines = _copied(app, res)
    assert lines[0].split('\t')[0] == 'Series'
    assert [l.split('\t')[0] for l in lines[1:]] == [FENT_SERIES, SAL_SERIES]


def test_siglink_copy_has_no_series_column_for_one_series():
    """A single-series run must copy exactly the columns it always did."""
    app = _siglink_app(COMBINE_ALL)
    _fake_compute(app, {'Sniff': {ALL_SUBJECTS[0]: 55.0}})
    res = app._compute_signal_linkage_series(
        _Queue(), [('All', ALL_SUBJECTS)], 'G0', ['Sniff'],
        {'fps': 30.0, 'n_perm': 100})

    lines = _copied(app, res)
    assert lines[0].split('\t')[0] == 'Behavior'
    assert lines[1].split('\t')[0] == 'Sniff'


def test_siglink_figure_draws_a_trace_per_series():
    app = _siglink_app(SPLIT_BOTH)
    _fake_compute(app, {'Sniff': {FENT[0]: 80.0, SAL[0]: 20.0}})
    res = app._compute_signal_linkage_series(
        _Queue(), [(FENT_SERIES, FENT), (SAL_SERIES, SAL)], 'G0', ['Sniff'],
        {'fps': 30.0, 'n_perm': 100, 'resp_s': 1.0})

    fig = app._build_signal_linkage_figure(res)
    try:
        # axes[0] is the summary bar; the behavior panel follows.
        panel = fig.axes[1]
        labels = [ln.get_label() for ln in panel.get_lines()]
        assert any(FENT_SERIES in str(l) for l in labels)
        assert any(SAL_SERIES in str(l) for l in labels)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
