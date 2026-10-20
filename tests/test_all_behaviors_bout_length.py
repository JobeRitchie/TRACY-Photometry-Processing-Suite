"""All Behaviors export carries each bout's onset and scored length.

A bout whose window runs past the recording edge is dropped from the trace
list but kept in onset_frames/durations, so the two stop being parallel.  The
export must pair each metric row with its own bout's length, not the length at
the same trace index, or every bout after the dropped one is mislabelled.
"""
import os
import tempfile
import types

import numpy as np
import pytest

import fp_analysis_gui as G

FPS = 20.0
PRE = POST = 10
ONSETS = [100, 200, 300, 400, 500]
DURATIONS = [20.0, 40.0, 60.0, 80.0, 100.0]   # samples -> 1, 2, 3, 4, 5 s
KEPT = [0, 2, 3, 4]                            # bout 2 dropped at the edge


def _app(with_ends=True, average=False):
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': PRE, 'postboutframes': POST,
                  'preboutseconds': PRE / FPS, 'postboutseconds': POST / FPS}

    entry = {'_prebout': PRE, '_postbout': POST, '_fs': FPS,
             'onset_frames': list(ONSETS),
             'end_frames': ([o + d for o, d in zip(ONSETS, DURATIONS)]
                            if with_ends else None),
             'durations': list(DURATIONS) if with_ends else None}
    # Each trace's level is its original bout number, so a row can be traced
    # back to the bout it came from.
    traces = [np.full(PRE + POST, float(i + 1)) for i in KEPT]
    for ch in ('Ch0', 'Ch1', 'G0', 'G1'):
        entry[ch] = traces
    entry['_kept_indices'] = {ch: list(KEPT) for ch in ('Ch0', 'Ch1', 'G0', 'G1')}
    app.processed_data = {'s1': {'photometry_fps': FPS,
                                 'channel_names': ['G0', 'G1'],
                                 'bouts': {'Lick': entry}}}

    var = lambda v: types.SimpleNamespace(get=lambda v=v: v)
    for name in ('metric_max_post', 'metric_min_post', 'metric_first_deriv',
                 'metric_tau', 'metric_t_half', 'metric_auc'):
        setattr(app, name, var(False))
    app.metric_avg_post = var(True)
    app.metric_auc_mode = var('total')
    app.bout_average_within_subject = var(average)
    app.bout_max_bouts_var = var('all')
    app.use_exclusions_bout = var(False)
    app.window_start_sec_var = var('0')
    app.window_end_sec_var = var('0.5')
    app.bout_analysis_by_var = var('Subject')
    app.groups = {}
    app.exclusions = {}
    app.bout_analysis_subject_listbox = types.SimpleNamespace(
        curselection=lambda: (0,), get=lambda i: 's1')
    app.bout_analysis_group_listbox = types.SimpleNamespace(curselection=lambda: ())
    return app


@pytest.fixture
def out_path():
    fd, path = tempfile.mkstemp(suffix='.xlsx')
    os.close(fd)
    os.unlink(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def _export(app, monkeypatch, path):
    pd = pytest.importorskip('pandas')
    monkeypatch.setattr(G.filedialog, 'asksaveasfilename', lambda **k: path)
    errors = []
    monkeypatch.setattr(G.messagebox, 'showerror',
                        lambda title, msg, **k: errors.append((title, msg)))
    monkeypatch.setattr(G.messagebox, 'showinfo', lambda *a, **k: None)
    app.export_bout_metrics_all()
    assert not errors, errors
    return pd.read_excel(path, sheet_name='Lick_G0')


def test_each_row_carries_its_own_bouts_length(out_path, monkeypatch):
    df = _export(_app(), monkeypatch, out_path)
    # Avg Post-Bout is the original bout number (trace level).
    original = df['Avg Post-Bout'].round().astype(int).tolist()
    assert original == [1, 3, 4, 5]
    assert df['Bout Length (s)'].tolist() == [DURATIONS[b - 1] / FPS for b in original]
    assert df['Bout Length (samples)'].tolist() == [DURATIONS[b - 1] for b in original]
    assert df['Onset (s)'].tolist() == [ONSETS[b - 1] / FPS for b in original]


def test_start_only_boutframes_leave_length_blank(out_path, monkeypatch):
    df = _export(_app(with_ends=False), monkeypatch, out_path)
    assert df['Bout Length (s)'].isna().all()
    assert df['Onset (s)'].notna().all()


def test_averaged_sheet_reports_mean_length(out_path, monkeypatch):
    df = _export(_app(average=True), monkeypatch, out_path)
    expected = np.mean([DURATIONS[i] for i in KEPT]) / FPS
    assert df['Mean Bout Length (s)'].iloc[0] == pytest.approx(expected)
