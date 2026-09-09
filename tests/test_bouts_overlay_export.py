"""The Bouts Overlay exports a workbook a graphing package can read.

"Export CSV" on a Bouts Overlay wrote the plain z-scored session and dropped
every bout marker, so the one thing the plot is for -- signal with the scored
behavior on it -- had to be rebuilt by hand in Prism.

These tests pin the replacement: one workbook for the subject on screen, whose
Trace sheet carries the trace, a ribbon column per behavior and a highlight
column per channel x behavior, and whose Bouts sheet lists the adjusted frames
and their times. They also pin that the exported spans are the *same* spans the
plot shades -- both now read `_bout_overlay_spans`.
"""
import os
import tempfile
import types

import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G

FPS = 30.0
N_ROWS = 900                    # 30 s at 30 Hz
N_CH = 2
SUBJECT = 'm1'
# (behavior, [(start, end_or_None), ...]) in beh_synced rows
BOUTS = {
    'Explore': [(100.0, 160.0), (400.0, 460.0)],
    'Approach': [(600.0, None)],            # scored start-only
}


def _beh_synced():
    """A v2 beh_synced block: 6 meta cols, N_CH z-score cols, kinematics."""
    width = 6 + N_CH + len(G.FPAnalysisGUI.BEH_KIN_FIELDS)
    beh = np.zeros((N_ROWS, width), dtype=float)
    beh[:, 0] = np.arange(N_ROWS)                       # frame
    beh[:, 1] = np.arange(N_ROWS) / FPS                 # timestamp
    for ch in range(N_CH):
        beh[:, 6 + ch] = np.sin(np.arange(N_ROWS) / 50.0) + ch
    beh[:, 6 + N_CH] = np.arange(N_ROWS) / FPS / 60.0   # elapsed_min
    return beh


def _app():
    """A GUI shell with one processed subject and no Tk widgets."""
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {}
    app.exclusions = {}
    app._mixed_fps_warned = True

    bouts = {}
    for behavior, spans in BOUTS.items():
        entry = {'_prebout': 15, '_postbout': 15, '_fs': FPS,
                 'onset_frames': [s for s, _e in spans]}
        if any(e is not None for _s, e in spans):
            entry['end_frames'] = [e if e is not None else np.nan for _s, e in spans]
        bouts[behavior] = entry

    app.processed_data = {SUBJECT: {
        'photometry_fps': FPS,
        'num_photometry_channels': N_CH,
        'channel_names': ['G0', 'R4'],
        'beh_synced': _beh_synced(),
        'bouts': bouts,
    }}

    var = lambda v: types.SimpleNamespace(get=lambda v=v: v)
    app.boutframes_path_var = var('')
    app.behavior_var = var('')
    app.viz_overlay_behaviors = None
    app.viz_channel_vars = [var(1) for _ in range(N_CH)]
    app._overlay_export_subject = SUBJECT
    app.viz_subject_listbox = types.SimpleNamespace(curselection=lambda: (),
                                                    get=lambda i: SUBJECT)
    return app


def _export(app, monkeypatch, path):
    monkeypatch.setattr(G.filedialog, 'asksaveasfilename', lambda **k: path)
    monkeypatch.setattr(G.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(G.messagebox, 'showwarning', lambda *a, **k: None)
    monkeypatch.setattr(G.messagebox, 'showerror',
                        lambda *a, **k: pytest.fail(f'export error: {a}'))
    app.export_bouts_overlay_workbook()
    return pd.read_excel(path, sheet_name=None)


@pytest.fixture
def book(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        yield _export(_app(), monkeypatch, os.path.join(d, 'overlay.xlsx'))


def test_workbook_has_the_three_shapes(book):
    assert set(book) == {'README', 'Trace', 'Ribbon_steps', 'Bouts'}
    trace = book['Trace']
    assert len(trace) == N_ROWS
    # trace, ribbon and highlight columns all present
    assert {'Time_s', 'Time_min', 'Frame', 'G0_z', 'R4_z'} <= set(trace.columns)
    assert {'Explore_ribbon', 'Approach_ribbon'} <= set(trace.columns)
    assert {'G0_Explore', 'R4_Explore', 'G0_Approach'} <= set(trace.columns)


def test_trace_matches_the_source_signal(book):
    beh = _beh_synced()
    assert np.allclose(book['Trace']['G0_z'].to_numpy(), beh[:, 6])
    assert np.allclose(book['Trace']['R4_z'].to_numpy(), beh[:, 7])
    # Time comes from the elapsed clock, not from a row count assumption.
    assert np.allclose(book['Trace']['Time_s'].to_numpy(),
                       np.arange(N_ROWS) / FPS)


def test_ribbon_is_on_exactly_inside_the_bouts(book):
    ribbon = book['Trace']['Explore_ribbon'].to_numpy()
    on = np.flatnonzero(np.isfinite(ribbon))
    assert on.min() == 100 and on.max() == 460
    # Two spans, and the gap between them is blank rather than zero.
    assert not np.isfinite(ribbon[161:400]).any()
    assert np.isfinite(ribbon[100:161]).all()
    assert np.isfinite(ribbon[400:461]).all()
    # One constant level per behavior, and it sits below the data.
    levels = np.unique(ribbon[np.isfinite(ribbon)])
    assert len(levels) == 1
    assert levels[0] < np.nanmin(book['Trace']['G0_z'].to_numpy())


def test_ribbon_steps_give_each_bout_its_own_column(book):
    """Prism joins across a blank cell, so bouts must not share a column.

    On the Trace sheet a ribbon column is blank between bouts and a line
    dataset bridges that gap -- one ribbon from the first bout to the last.
    Here each bout is its own two-point column, which a line cannot bridge.
    """
    steps = book['Ribbon_steps']
    assert list(steps.columns) == ['Time_s', 'Explore_b1', 'Explore_b2', 'Approach_b1']
    # Two rows per bout with an end, one for the start-only bout.
    assert len(steps) == 5
    assert steps['Time_s'].is_monotonic_increasing

    b1 = steps[['Time_s', 'Explore_b1']].dropna()
    assert np.allclose(b1['Time_s'].to_numpy(), np.array([100, 160]) / FPS)
    assert b1['Explore_b1'].nunique() == 1          # a flat segment at one level

    # No column carries points from more than one bout.
    for col in steps.columns[1:]:
        assert steps[col].notna().sum() <= 2
    # The start-only bout is a single point; a line would draw nothing there.
    assert steps['Approach_b1'].notna().sum() == 1
    # Levels agree with the Trace sheet's ribbons.
    assert (steps['Explore_b1'].dropna().iloc[0]
            == book['Trace']['Explore_ribbon'].dropna().iloc[0])


def test_ribbons_do_not_overlap_between_behaviors(book):
    explore = book['Trace']['Explore_ribbon'].dropna().unique()
    approach = book['Trace']['Approach_ribbon'].dropna().unique()
    assert explore[0] != approach[0]


def test_start_only_bout_marks_one_row(book):
    ribbon = book['Trace']['Approach_ribbon'].to_numpy()
    on = np.flatnonzero(np.isfinite(ribbon))
    assert on.tolist() == [600]


def test_highlight_columns_carry_the_trace_inside_bouts_only(book):
    trace = book['Trace']
    hl = trace['G0_Explore'].to_numpy()
    sig = trace['G0_z'].to_numpy()
    inside = np.zeros(N_ROWS, dtype=bool)
    inside[100:161] = True
    inside[400:461] = True
    assert np.allclose(hl[inside], sig[inside])
    assert not np.isfinite(hl[~inside]).any()


def test_bouts_sheet_lists_every_bout_with_frames_and_times(book):
    bouts = book['Bouts']
    assert len(bouts) == 3
    explore = bouts[bouts['Behavior'] == 'Explore'].sort_values('Bout')
    assert explore['Onset_frame'].tolist() == [100.0, 400.0]
    assert explore['End_frame'].tolist() == [160.0, 460.0]
    assert np.allclose(explore['Onset_s'].to_numpy(), np.array([100, 400]) / FPS)
    assert np.allclose(explore['Duration_s'].to_numpy(), 60.0 / FPS)
    assert (explore['On_trace'] == 'yes').all()
    # A start-only bout leaves the end blank rather than inventing one.
    approach = bouts[bouts['Behavior'] == 'Approach'].iloc[0]
    assert not np.isfinite(approach['End_frame'])
    assert not np.isfinite(approach['Duration_s'])


def test_bout_past_the_end_is_listed_but_not_drawn(monkeypatch):
    app = _app()
    entry = app.processed_data[SUBJECT]['bouts']['Explore']
    entry['onset_frames'] = entry['onset_frames'] + [N_ROWS + 50.0]
    entry['end_frames'] = entry['end_frames'] + [N_ROWS + 90.0]
    with tempfile.TemporaryDirectory() as d:
        book = _export(app, monkeypatch, os.path.join(d, 'overlay.xlsx'))
    explore = book['Bouts'][book['Bouts']['Behavior'] == 'Explore']
    assert len(explore) == 3
    assert explore['On_trace'].tolist() == ['yes', 'yes', 'no']
    # ... and it adds nothing to the ribbon, which only spans real rows.
    ribbon = book['Trace']['Explore_ribbon'].to_numpy()
    assert np.flatnonzero(np.isfinite(ribbon)).max() == 460


def test_only_the_selected_channels_are_exported(monkeypatch):
    app = _app()
    app.viz_channel_vars = [types.SimpleNamespace(get=lambda: 0),
                            types.SimpleNamespace(get=lambda: 1)]
    with tempfile.TemporaryDirectory() as d:
        book = _export(app, monkeypatch, os.path.join(d, 'overlay.xlsx'))
    cols = set(book['Trace'].columns)
    assert 'R4_z' in cols and 'G0_z' not in cols
    assert 'R4_Explore' in cols and 'G0_Explore' not in cols


def test_behavior_selection_follows_the_overlay_picker(monkeypatch):
    app = _app()
    app.viz_overlay_behaviors = {'Approach'}
    with tempfile.TemporaryDirectory() as d:
        book = _export(app, monkeypatch, os.path.join(d, 'overlay.xlsx'))
    assert book['Bouts']['Behavior'].unique().tolist() == ['Approach']
    assert 'Explore_ribbon' not in book['Trace'].columns


def test_uncovered_rows_stay_blank(monkeypatch):
    """Rows the photometry never covered must not export as signal."""
    app = _app()
    data = app.processed_data[SUBJECT]
    # zscore's own clock starts 100 rows late, so the sync could only have
    # filled those rows from the nearest edge sample.
    z = np.zeros((N_ROWS - 100, 2 + N_CH))
    z[:, 1] = np.arange(100, N_ROWS) / FPS
    data['zscore'] = z
    data['beh_synced'][:, 1] = np.arange(N_ROWS) / FPS
    with tempfile.TemporaryDirectory() as d:
        book = _export(app, monkeypatch, os.path.join(d, 'overlay.xlsx'))
    trace = book['Trace']
    blanked = trace['Photometry_covered'].to_numpy() == 0
    # 99, not 100: the row on the boundary sits inside the sync's own matching
    # tolerance, so the recording does cover it.
    assert blanked.sum() == 99
    assert not np.isfinite(trace['G0_z'].to_numpy()[blanked]).any()
    # The X axis survives it: a blank X cell would drop the row in Prism.
    assert np.isfinite(trace['Time_s'].to_numpy()).all()


def test_export_and_plot_read_the_same_spans():
    """The workbook cannot drift from the graph: one resolver serves both."""
    app = _app()
    spans = app._bout_overlay_spans(app.processed_data[SUBJECT], SUBJECT)
    assert set(spans) == set(BOUTS)
    assert spans['Explore'] == [(100.0, 160.0), (400.0, 460.0)]
    assert spans['Approach'] == [(600.0, None)]
