"""Bout-trace CSV exports write every ticked channel, not just G0/G1.

The Visualization tab's Export CSV for Extracted Bouts and Zone Entry Bouts
looped a literal ``['G0', 'G1']``, and Compare Across Bouts capped its export at
the plot's two panels, so a five-channel recording (G0-G4) exported only its
first two channels even with all five ticked.
"""
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G

FPS = 30.0
PRE = POST = 15
N_CH = 5
N_BOUTS = 3
SUBJECTS = ['s1', 's2']


def _trace(subject_idx, ch, bout_num):
    """A flat trace whose level identifies subject, channel and bout."""
    return np.full(PRE + POST, 100.0 * subject_idx + 10.0 * ch + bout_num)


def _entry(subject_idx):
    entry = {'_prebout': PRE, '_postbout': POST, '_fs': FPS}
    for ch in range(N_CH):
        entry[f'Ch{ch}'] = [_trace(subject_idx, ch, n) for n in range(1, N_BOUTS + 1)]
        entry[f'G{ch}'] = entry[f'Ch{ch}']
    return entry


def _app(plot_type, ticked=None):
    var = lambda v: types.SimpleNamespace(get=lambda v=v: v)
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': PRE, 'postboutframes': POST,
                  'preboutseconds': PRE / FPS, 'postboutseconds': POST / FPS,
                  'boutframe_processing_style': 'onset',
                  'baseline_correct_bouts': False}
    app.exclusions = {}
    app.processed_data = {}
    for s_idx, subject in enumerate(SUBJECTS, 1):
        app.processed_data[subject] = {
            'photometry_fps': FPS,
            'channel_names': [f'G{i}' for i in range(N_CH)],
            'bouts': {'Lick': _entry(s_idx)},
            'entry_bouts': {'open_arm': {'Open Arm': _entry(s_idx)}},
        }
    ticked = range(N_CH) if ticked is None else ticked
    app.viz_channel_vars = [var(1 if i in ticked else 0) for i in range(N_CH)]
    app.get_fps = lambda *a, **k: FPS
    app.show_470, app.show_570 = var(True), var(False)
    app.show_g0, app.show_g1 = var(True), var(True)
    app.use_exclusions_viz = var(True)
    app.viz_max_bouts_var = var('all')
    app.viz_average_within_subject = var(False)
    app.viz_bout_order_var = var('Subject')
    app.viz_bout_bin_var = var('2')
    app.bout_number_var = var('All')
    app.behavior_var = var('Lick')
    app.plot_by_var = var('Subject')
    app.plot_type_var = var(plot_type)
    app.zone_entry_type_var = var('Open Arm')
    app._get_zone_entry_type_map = lambda: {'Open Arm': ('open_arm', None)}
    app.viz_subject_listbox = types.SimpleNamespace(
        curselection=lambda: tuple(range(len(SUBJECTS))), get=lambda i: SUBJECTS[i])
    app._create_export_metadata = lambda kind='plot': {'Export Type': kind}
    return app


def _export(app, monkeypatch):
    msgs = []
    monkeypatch.setattr(G, 'messagebox', types.SimpleNamespace(
        showinfo=lambda *a, **k: msgs.append(('info', a)),
        showerror=lambda *a, **k: msgs.append(('error', a)),
        showwarning=lambda *a, **k: msgs.append(('warn', a))))
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(G, 'filedialog', types.SimpleNamespace(
        asksaveasfilename=lambda **k: os.path.join(tmp, 'out.csv')))
    app.export_plot_data()
    assert [m for m in msgs if m[0] != 'info'] == []
    return tmp


def _read(path):
    return pd.read_csv(path, comment='#')


def test_extracted_bouts_export_writes_all_five_channels(monkeypatch):
    tmp = _export(_app('Extracted Bouts'), monkeypatch)
    assert sorted(os.listdir(tmp)) == [f'out_Lick_G{i}.csv' for i in range(N_CH)]
    # G3's file holds G3's traces (subject 2, channel 3, bout 2 -> 232).
    df = _read(os.path.join(tmp, 'out_Lick_G3.csv'))
    assert np.allclose(df['s2_G3_bout2'], 232.0)
    assert len(df.columns) - 1 == len(SUBJECTS) * N_BOUTS


def test_extracted_bouts_export_follows_ticks_and_exclusions(monkeypatch):
    app = _app('Extracted Bouts', ticked={0, 2, 4})
    app.exclusions = {'s1': ['G4']}
    tmp = _export(app, monkeypatch)
    assert sorted(os.listdir(tmp)) == ['out_Lick_G0.csv', 'out_Lick_G2.csv', 'out_Lick_G4.csv']
    g4 = _read(os.path.join(tmp, 'out_Lick_G4.csv'))
    assert not any(c.startswith('s1_') for c in g4.columns)
    assert any(c.startswith('s2_') for c in g4.columns)


def test_zone_entry_bouts_export_writes_all_five_channels(monkeypatch):
    tmp = _export(_app('Zone Entry Bouts'), monkeypatch)
    df = _read(os.path.join(tmp, 'out.csv'))
    for ch in range(N_CH):
        assert np.allclose(df[f's1_G{ch}_bout1'], 100.0 + 10.0 * ch + 1)


def test_compare_across_bouts_export_is_not_capped_at_two_channels(monkeypatch):
    tmp = _export(_app('Compare Across Bouts'), monkeypatch)
    written = sorted(os.listdir(tmp))
    for ch in range(N_CH):
        assert f'out_Lick_G{ch}_boutbins.csv' in written
        assert f'out_Lick_G{ch}_boutbins_traces.csv' in written


def test_two_slot_plots_still_get_two_padded_slots():
    app = _app('Extracted Bouts', ticked={3})
    assert app._viz_bout_channel_slots(SUBJECTS, max_slots=2) == [('Ch3', 'G3'), (None, None)]
    assert app._viz_bout_channel_slots(SUBJECTS, max_slots=None) == [('Ch3', 'G3')]
