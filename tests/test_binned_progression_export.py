"""Binned Progression exports every behavior in one workbook.

The progression view could only ever export the behavior currently on screen,
so a project with six scored behaviors meant six plot-then-export round trips
with the bin size and max-bouts settings re-entered by hand each time.

These tests pin the replacement: one workbook per run, one sheet per
behavior x channel, computed from the same helper the plot draws from, with
the bouts-per-bin and max-bouts settings applied to every behavior.
"""
import os
import tempfile
import types

import numpy as np
import pytest

import fp_analysis_gui as G

FPS = 30.0
PRE = POST = 15          # frames; a 1 s window either side
N_BOUTS = 7
SUBJECTS = ['s1', 's2', 's3']
BEHAVIORS = ['Lick', 'Rear']
BEH_OFFSET = {'Lick': 0.0, 'Rear': 1000.0}


def _bout(subject_idx, bout_num, behavior):
    """A flat trace whose level identifies subject, bout number and behavior."""
    return np.full(PRE + POST,
                   BEH_OFFSET[behavior] + 100.0 * subject_idx + bout_num,
                   dtype=float)


def _app(**overrides):
    """A GUI shell with synthetic bouts and no Tk widgets."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': PRE, 'postboutframes': POST,
                  'preboutseconds': PRE / FPS, 'postboutseconds': POST / FPS}

    app.processed_data = {}
    for s_idx, subject in enumerate(SUBJECTS, 1):
        bouts = {}
        for behavior in BEHAVIORS:
            entry = {'_prebout': PRE, '_postbout': POST, '_fs': FPS}
            for ch_idx, ch in enumerate(('Ch0', 'Ch1')):
                entry[ch] = [_bout(s_idx + ch_idx, n, behavior)
                             for n in range(1, N_BOUTS + 1)]
            entry['G0'] = entry['Ch0']
            entry['G1'] = entry['Ch1']
            bouts[behavior] = entry
        app.processed_data[subject] = {
            'photometry_fps': FPS,
            'channel_names': ['G0', 'G1'],
            'bouts': bouts,
        }

    var = lambda v: types.SimpleNamespace(get=lambda v=v: v)
    app.bout_first_last_x_var = var(overrides.get('bin_size', '3'))
    app.bout_max_bouts_var = var(overrides.get('max_bouts', 'all'))
    app.metric_max_post = var(True)
    app.metric_avg_post = var(True)
    app.metric_auc = var(True)
    app.use_exclusions_bout = var(False)
    app.window_start_sec_var = var('0')
    app.window_end_sec_var = var('0.5')
    app.bout_analysis_by_var = var('Subject')
    app.bout_analysis_behavior_var = var('Lick')
    app.bout_analysis_channel_var = var('G0')
    app.exclusions = {}
    app.bout_analysis_subject_listbox = types.SimpleNamespace(
        curselection=lambda: tuple(range(len(SUBJECTS))),
        get=lambda i: SUBJECTS[i])
    app.bout_analysis_group_listbox = types.SimpleNamespace(curselection=lambda: ())
    return app


def _run_export(app, monkeypatch, path):
    monkeypatch.setattr(G.filedialog, 'asksaveasfilename', lambda **k: path)
    errors = []
    monkeypatch.setattr(G.messagebox, 'showerror',
                        lambda title, msg, **k: errors.append((title, msg)))
    monkeypatch.setattr(G.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(G.messagebox, 'showwarning', lambda *a, **k: None)
    app.export_binned_progression_all()
    assert not errors, errors
    return errors


@pytest.fixture
def out_path():
    fd, path = tempfile.mkstemp(suffix='.xlsx')
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


# --------------------------------------------------------------------------
# coverage: every behavior, every channel
# --------------------------------------------------------------------------

def test_workbook_holds_a_sheet_for_every_behavior_and_channel(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    _run_export(app, monkeypatch, out_path)

    with pd.ExcelFile(out_path) as xls:
        sheets = xls.sheet_names
    for behavior in BEHAVIORS:
        for channel in ('G0', 'G1'):
            assert f'{behavior}_{channel}' in sheets
    assert 'All Behaviors (long)' in sheets
    assert 'Export Settings' in sheets


def test_long_sheet_covers_every_behavior_channel_metric_bin(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    _run_export(app, monkeypatch, out_path)

    long = pd.read_excel(out_path, sheet_name='All Behaviors (long)')
    assert sorted(long['Behavior'].unique()) == sorted(BEHAVIORS)
    assert sorted(long['Channel'].unique()) == ['G0', 'G1']
    assert len(long['Metric'].unique()) == 3
    # 7 bouts binned by 3 -> bins 1-3, 4-6, 7-9
    assert sorted(long['Bin'].unique()) == ['1-3', '4-6', '7-9']
    assert (long['n'] == len(SUBJECTS)).all()


# --------------------------------------------------------------------------
# the numbers match what the plot would draw
# --------------------------------------------------------------------------

def test_exported_values_match_the_progression_helper(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    _run_export(app, monkeypatch, out_path)

    ws, we = app.analysis_window_samples()
    prog = app._compute_binned_progression(
        'Rear', 'Ch1', SUBJECTS, {}, 3, None, ws, we,
        app.get_fps(), PRE, group_mode=False)

    sheet = pd.read_excel(out_path, sheet_name='Rear_G1')
    peak = sheet[(sheet['Metric'] == 'Peak (z-score)') &
                 (sheet['Type'] == 'per-subject')].set_index('Subject')
    for subject in SUBJECTS:
        for b, label in zip(prog['sorted_bins'], prog['bin_labels']):
            expected = prog['subject_bin_data'][subject][b]['peak']
            assert peak.loc[subject, label] == pytest.approx(expected)


def test_summary_rows_are_mean_sem_n_across_subjects(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    _run_export(app, monkeypatch, out_path)

    sheet = pd.read_excel(out_path, sheet_name='Lick_G0')
    per_subject = sheet[(sheet['Metric'] == 'Average (z-score)') &
                        (sheet['Type'] == 'per-subject')]
    summary = sheet[(sheet['Metric'] == 'Average (z-score)') &
                    (sheet['Type'] == 'summary')].set_index('Subject')

    vals = per_subject['1-3'].to_numpy(dtype=float)
    assert summary.loc['mean', '1-3'] == pytest.approx(np.mean(vals))
    assert summary.loc['SEM', '1-3'] == pytest.approx(
        np.std(vals) / np.sqrt(len(vals)))
    assert summary.loc['n', '1-3'] == len(vals)


# --------------------------------------------------------------------------
# current settings are honoured for every behavior, not just the plotted one
# --------------------------------------------------------------------------

def test_max_bouts_caps_every_behavior(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app(max_bouts='4', bin_size='2')
    _run_export(app, monkeypatch, out_path)

    long = pd.read_excel(out_path, sheet_name='All Behaviors (long)')
    # 4 bouts binned by 2 -> two bins only, for every behavior/channel
    for behavior in BEHAVIORS:
        for channel in ('G0', 'G1'):
            sub = long[(long['Behavior'] == behavior) & (long['Channel'] == channel)]
            assert sorted(sub['Bin'].unique()) == ['1-2', '3-4']


def test_bin_size_applies_to_every_behavior(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app(bin_size='7')
    _run_export(app, monkeypatch, out_path)

    long = pd.read_excel(out_path, sheet_name='All Behaviors (long)')
    assert sorted(long['Bin'].unique()) == ['1-7']


def test_metric_checkboxes_gate_the_export(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    app.metric_avg_post = types.SimpleNamespace(get=lambda: False)
    app.metric_auc = types.SimpleNamespace(get=lambda: False)
    _run_export(app, monkeypatch, out_path)

    long = pd.read_excel(out_path, sheet_name='All Behaviors (long)')
    assert list(long['Metric'].unique()) == ['Peak (z-score)']


def test_exclusions_drop_the_excluded_channel(out_path, monkeypatch):
    pd = pytest.importorskip('pandas')
    app = _app()
    app.use_exclusions_bout = types.SimpleNamespace(get=lambda: True)
    app.exclusions = {'s2': ['G1']}
    _run_export(app, monkeypatch, out_path)

    long = pd.read_excel(out_path, sheet_name='All Behaviors (long)')
    g1 = long[long['Channel'] == 'G1']
    g0 = long[long['Channel'] == 'G0']
    assert (g1['n'] == len(SUBJECTS) - 1).all()
    assert (g0['n'] == len(SUBJECTS)).all()
