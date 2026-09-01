"""The Edit Processing Parameters dialog, and the entry thresholds it groups.

Two things are pinned here.

First, the grouping is a layout hint and must never become a filter: a
parameter left out of PARAM_GROUPS still has to get an editor, or adding a
setting silently makes it uneditable from the GUI. The catch-all "Other" tab is
what guarantees that, so the test adds a parameter the table has never heard of
and checks it is still saved.

Second, "how long must the animal stay in a zone for it to count as an entry?"
is three different pairs of parameters -- EPM detection, OFT/Custom detection,
and the Behavioral Data metrics table -- and the same numbers are now editable
from four screens. The tests check that the pair on show follows the maze type
and that whichever screen wrote last is what the others read.
"""
import tkinter as tk

import pytest

import fp_analysis_gui as G


@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """No message box may open: every one of them blocks a headless run forever."""
    shown = []
    for name in ('showinfo', 'showwarning', 'showerror'):
        monkeypatch.setattr(G.messagebox, name,
                            lambda *a, _n=name, **k: shown.append((_n, a)))
    monkeypatch.setattr(G.messagebox, 'askyesno', lambda *a, **k: True)
    return shown


@pytest.fixture
def app(tk_root):
    """A GUI on a Toplevel of the session root -- a Tk interpreter may only be
    created once per process (see conftest)."""
    win = tk.Toplevel(tk_root)
    win.withdraw()
    try:
        yield G.FPAnalysisGUI(win)
    finally:
        win.destroy()


def _walk(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _walk(child)


def _open_dialog(app):
    """Open Edit Parameters and return (dialog, notebook)."""
    app.edit_parameters()
    dialog = [w for w in app.root.winfo_children()
              if isinstance(w, tk.Toplevel) and w.title() == "Edit Processing Parameters"][-1]
    dialog.update_idletasks()
    notebook = next(w for w in _walk(dialog) if isinstance(w, G.ttk.Notebook))
    return dialog, notebook


def _save(dialog):
    """Press the dialog's Save button."""
    button = next(w for w in _walk(dialog)
                  if isinstance(w, G.ttk.Button) and w.cget('text') == 'Save')
    button.invoke()


# --------------------------------------------------------------------------
# Grouping


def test_every_grouped_key_is_a_real_parameter(app):
    """A typo in PARAM_GROUPS would silently drop a parameter from the dialog."""
    for _tab, sections in G.PARAM_GROUPS:
        for _title, _hint, keys in sections:
            for key, _label, _hint2 in keys:
                assert key in app.params, f"PARAM_GROUPS names unknown parameter {key}"


def test_no_key_is_grouped_twice():
    seen = set()
    for _tab, sections in G.PARAM_GROUPS:
        for _title, _hint, keys in sections:
            for key, _label, _hint2 in keys:
                assert key not in seen, f"{key} appears in PARAM_GROUPS twice"
                seen.add(key)


def test_dialog_opens_with_one_tab_per_group(app):
    dialog, notebook = _open_dialog(app)
    try:
        titles = [notebook.tab(t, 'text') for t in notebook.tabs()]
        assert titles == [name for name, _sections in G.PARAM_GROUPS]
    finally:
        dialog.destroy()


def test_ungrouped_parameter_still_gets_an_editor(app):
    """The table is a layout hint, never a filter."""
    app.params['a_brand_new_setting'] = 7
    dialog, notebook = _open_dialog(app)
    try:
        titles = [notebook.tab(t, 'text') for t in notebook.tabs()]
        assert titles[-1] == "Other"
        entry = next(w for w in _walk(notebook.nametowidget(notebook.tabs()[-1]))
                     if isinstance(w, G.ttk.Entry))
        entry.delete(0, 'end')
        entry.insert(0, '9')
        _save(dialog)
        assert app.params['a_brand_new_setting'] == 9
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


def test_structured_and_derived_parameters_are_not_editable(app):
    """Dicts/lists have their own editors, and the frame counts are derived from
    the seconds -- an editable box here would invite an edit the next sync eats."""
    dialog, notebook = _open_dialog(app)
    try:
        shown = set()
        for tab in notebook.tabs():
            for w in _walk(notebook.nametowidget(tab)):
                try:
                    text = w.cget('text')
                except Exception:
                    continue
                shown.add(str(text).split('  -  ')[0])
        for key in ('bout_overlay_colors', 'factor_definitions',
                    'preboutframes', 'postboutframes', 'baseline_frames'):
            assert key not in shown
    finally:
        dialog.destroy()


def test_saving_round_trips_a_value(app):
    dialog, notebook = _open_dialog(app)
    try:
        target = None
        for tab in notebook.tabs():
            for w in _walk(notebook.nametowidget(tab)):
                if isinstance(w, G.ttk.Entry) and w.get() == str(app.params['precut']):
                    target = w
                    break
            if target is not None:
                break
        assert target is not None
        target.delete(0, 'end')
        target.insert(0, '250')
        _save(dialog)
        assert app.params['precut'] == 250
    finally:
        if dialog.winfo_exists():
            dialog.destroy()


# --------------------------------------------------------------------------
# Entry thresholds


def test_threshold_keys_follow_the_maze_type(app):
    app.params['maze_type'] = 'EPM'
    assert app.entry_threshold_keys() == (
        'min_open_arm_duration', 'min_time_between_entries')
    app.params['maze_type'] = 'EPM_Complex'
    assert app.entry_threshold_keys() == (
        'min_open_arm_duration', 'min_time_between_entries')
    app.params['maze_type'] = 'OFT'
    assert app.entry_threshold_keys() == (
        'oft_min_entry_duration', 'oft_min_time_between_entries')
    app.params['maze_type'] = 'Custom Arena'
    assert app.entry_threshold_keys() == (
        'oft_min_entry_duration', 'oft_min_time_between_entries')


def test_visualization_boxes_show_the_pair_in_force(app):
    app.params['maze_type'] = 'EPM'
    app.params['min_open_arm_duration'] = 2.5
    app.params['oft_min_entry_duration'] = 0.75
    app._refresh_entry_threshold_controls()
    assert app.viz_entry_dur_var.get() == '2.5'
    assert 'EPM' in app.viz_entry_maze_label.cget('text')

    app.params['maze_type'] = 'OFT'
    app._refresh_entry_threshold_controls()
    assert app.viz_entry_dur_var.get() == '0.75'
    assert 'OFT' in app.viz_entry_maze_label.cget('text')


def test_visualization_boxes_write_back_to_the_pair_in_force(app):
    app.params['maze_type'] = 'OFT'
    app._refresh_entry_threshold_controls()
    app.viz_entry_dur_var.set('1.25')
    app.viz_entry_ref_var.set('4')
    app.apply_viz_entry_thresholds()
    assert app.params['oft_min_entry_duration'] == 1.25
    assert app.params['oft_min_time_between_entries'] == 4.0
    # The EPM pair must not have moved.
    assert app.params['min_open_arm_duration'] == 2.0


def test_parameter_dialog_edit_reaches_the_tab_boxes(app):
    """Whichever screen wrote last has to win everywhere."""
    app.params['maze_type'] = 'EPM'
    dialog, notebook = _open_dialog(app)
    try:
        target = None
        for tab in notebook.tabs():
            for w in _walk(notebook.nametowidget(tab)):
                if isinstance(w, G.ttk.Entry) and w.get() == '2.0':
                    target = w
                    break
            if target is not None:
                break
        assert target is not None
        target.delete(0, 'end')
        target.insert(0, '5.0')
        _save(dialog)
    finally:
        if dialog.winfo_exists():
            dialog.destroy()
    assert app.params['min_open_arm_duration'] == 5.0
    assert app.viz_entry_dur_var.get() == '5'


def test_behavioral_tab_has_its_own_metrics_thresholds(app):
    app.params['metrics_min_entry_duration'] = 1.5
    app.params['metrics_min_refractory_sec'] = 2.5
    app._refresh_entry_threshold_controls()
    assert app.behav_entry_dur_var.get() == '1.5'
    assert app.behav_entry_ref_var.get() == '2.5'


def test_detection_reads_the_same_keys_the_controls_edit(app, monkeypatch):
    """The one thing that would make the new boxes a lie: detection reading a
    different pair from the one they write."""
    app.params['maze_type'] = 'OFT'
    app.params['oft_min_entry_duration'] = 3.0
    app.params['oft_min_time_between_entries'] = 6.0
    dur_key, gap_key = app.entry_threshold_keys()
    assert app.params[dur_key] == 3.0
    assert app.params[gap_key] == 6.0

    logged = []
    monkeypatch.setattr(app, 'log_message', logged.append)
    monkeypatch.setattr(app, 'get_fps', lambda *a, **k: 30.0)
    with pytest.raises(Exception):
        # No position array to detect on; the parameter read happens first and
        # is what this asserts on.
        app.detect_zone_entries(None)
    assert any('3.0s' in str(m) for m in logged)
    assert any('6.0s' in str(m) for m in logged)
