"""FLMM runs every ticked behavior in one go, and finished runs stay cached.

Before this, the single-behavior analyses (signal ≠ 0, across bouts, between
groups, interaction) fit only the behavior in the dropdown, so covering a
project's behaviors meant one dialog-and-wait per behavior.  And the result
lived only on the canvas: running Calculate Metrics or any other plot replaced
it, and getting it back meant refitting.
"""
import tkinter as tk

import numpy as np
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _Listbox:
    def curselection(self):
        return (0,)


class _CancelEvent:
    def is_set(self):
        return False


class _Q:
    def __init__(self):
        self.cancel_event = _CancelEvent()

    def put(self, item):
        pass


def _app(choice, L=10):
    """Two groups x 5 animals; 'Sniff' differs by group, 'Rear' does not.
    'Rear' traces are longer so the two behaviors' windows differ in length."""
    rng = np.random.default_rng(1)
    store = {}
    members = {'KO': [], 'CTRL': []}
    for grp in ('KO', 'CTRL'):
        for a in range(5):
            s = f'{grp}{a}'
            members[grp].append(s)
            off = rng.normal(0, 0.2)
            sniff = [1.5 * (grp == 'KO') + off + rng.normal(0, 0.05, L) for _ in range(4)]
            rear = [off + rng.normal(0, 0.05, L + 4) for _ in range(4)]
            store[s] = {'bouts': {'Sniff': {'G0': sniff, 'G1': sniff},
                                  'Rear': {'G0': rear, 'G1': rear}}}
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = store
    app.params = {'preboutframes': 2, 'postboutframes': 8}
    app.get_fps = lambda *a, **k: 10.0
    app.log_message = lambda *a, **k: None
    app._realign_bouts = lambda bouts, prebout, src_fs=None: bouts
    app._entry_prebout = lambda entry: 2
    app._entry_fs = lambda entry: 10.0
    app._session_pattern = lambda: None
    app._animal_ids_for = lambda subj, log=None: [str(s) for s in subj]
    app.bout_analysis_by_var = _Var("Group")
    app.bout_analysis_group_listbox = _Listbox()
    app.selected_series = lambda *a: ['KO', 'CTRL']
    app._series_members = lambda g: members[g]
    app.bout_analysis_behavior_var = _Var('Sniff')
    app.bout_analysis_channel_var = _Var('G0')
    app.use_exclusions_bout = _Var(False)
    app.bout_max_bouts_var = _Var('')
    app._get_max_bouts_limit = lambda v: None
    app._flmm_r_available = lambda: False
    app._all_channel_names = lambda: ['G0', 'G1']
    app._flmm_reference_dialog = lambda *a, **k: dict(choice)
    app._flmm_confirm_run = lambda *a, **k: True

    def _run(title, worker, on_done_fn=None, **k):
        worker(_Q())
        on_done_fn()
    app._run_with_progress = _run
    app.shown = []
    app._show_pairwise_panels = lambda store: app.shown.append(store)
    return app


CHOICE = {'reference': 'between_groups', 'pointwise_only': False, 'engine': 'python',
          'behaviors': ['Sniff', 'Rear'], 'channels': ['G0', 'G1'],
          'ref_behavior': 'Sniff', 'comp_behavior': None, 'time_range': None}


@pytest.fixture(scope='module')
def ran():
    pytest.importorskip('statsmodels')
    app = _app(CHOICE)
    app.plot_timecourse_flmm()
    return app


def test_every_behavior_x_channel_is_fit_in_one_run(ran):
    store = ran.shown[-1]
    assert sorted(store['order']) == sorted(
        ['Sniff · G0', 'Sniff · G1', 'Rear · G0', 'Rear · G1'])
    assert store['facet_behaviors'] == ['Sniff', 'Rear']
    assert store['facet_channels'] == ['G0', 'G1']
    assert store['panels']['Sniff · G0']['sig_simultaneous'].all()
    assert not store['panels']['Rear · G0']['sig_simultaneous'].any()
    assert store['facet_key'](['Rear', 'G1']) == 'Rear · G1'


def test_summary_lists_each_panel(ran):
    lines = ran._flmm_multi_summary_lines(ran.shown[-1])
    sniff = [l for l in lines if l.startswith('Sniff')]
    rear = [l for l in lines if l.startswith('Rear')]
    assert len(sniff) == len(rear) == 2
    assert all(l.rstrip().endswith('ns') for l in rear)
    assert not any(l.rstrip().endswith('ns') for l in sniff)


def test_exports_carry_behavior_with_ragged_windows(ran):
    sheets = ran._flmm_workbook_frames(ran.shown[-1])
    mt = sheets['Model_terms']
    assert list(mt.columns[:2]) == ['Behavior', 'Channel']
    assert set(mt['Behavior']) == {'Sniff', 'Rear'}
    assert set(sheets['Input_bouts']['Behavior']) == {'Sniff', 'Rear'}


def test_single_behavior_run_keeps_old_layout():
    pytest.importorskip('statsmodels')
    app = _app({**CHOICE, 'behaviors': ['Sniff'], 'channels': ['G0']})
    app.plot_timecourse_flmm()
    store = app.shown[-1]
    assert store['order'] == ['G0'] and 'facets' not in store
    assert 'Behavior' not in app._flmm_workbook_frames(store)['Model_terms'].columns


def test_finished_runs_are_cached_and_recallable(ran):
    first = ran.shown[-1]
    assert ran.flmm_results_history[0] is first
    ran.plot_timecourse_flmm()                       # a second run
    assert len(ran.flmm_results_history) == 2
    assert ran.flmm_results_history[1] is first
    ran.show_flmm_result(first)                      # recall, no refit
    assert ran.shown[-1] is first and ran.flmm_timecourse_data is first

    root = tk.Tk()
    try:
        menu = tk.Menu(root, tearoff=0)
        ran._populate_flmm_history_menu(menu)
        labels = [menu.entrycget(i, 'label') for i in range(menu.index('end') + 1)
                  if menu.type(i) == 'command']
        assert labels[0] == "New analysis…"
        assert sum('Between groups' in l for l in labels) == 2
    finally:
        root.destroy()


def test_viewer_switches_behavior_and_remembers_panel(ran):
    store = ran.flmm_results_history[0]
    app = object.__new__(G.FPAnalysisGUI)
    root = tk.Tk()
    try:
        app.root = root
        app.bout_histogram_canvas = tk.Canvas(root)
        app.bout_histogram_frame = tk.Frame(app.bout_histogram_canvas)
        drawn, stats = [], []
        app._embed_plot_canvas = lambda fig, holder: None
        app._display_bout_stats = lambda lines: stats.append(lines)
        store = dict(store, make_fig=lambda k: drawn.append(k) or None)
        G.FPAnalysisGUI._show_pairwise_panels(app, store)
        root.update()
        root.after(50)
        root.update()
        combos = [w for w in app.bout_histogram_frame.winfo_children()[0].winfo_children()
                  if w.winfo_class() == 'TCombobox']
        assert len(combos) == 2                     # Behavior and Channel
        combos[0].set('Rear')
        combos[0].event_generate('<<ComboboxSelected>>')
        root.update()
        assert drawn[-1] == 'Rear · ' + drawn[0].split(' · ')[1]
        assert store['last_key'] == drawn[-1]
    finally:
        root.destroy()
