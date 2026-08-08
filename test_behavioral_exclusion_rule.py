"""The Behavioral Data tab must apply one definition of "excluded".

It had three.  `calculate_behavioral_metrics` excluded on the 'Behavior'
pseudo-channel; `export_behavioral_metrics` excluded any subject with *any*
non-empty exclusion list, so ticking an unrelated photometry channel silently
deleted that animal's behavioral row from the CSV; `visualize_behavioral_data`
and `_plot_behavioral_summary` filtered nothing at all and just replotted the
stored results.  Table, plot and CSV could therefore report three different n.

The rule kept is the tab's own: the 'Behavior' pseudo-channel, honoured by the
toggle, applied at every read site -- the toggle can be flipped after a
calculation, and the stored rows would otherwise keep the old answer.
"""
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _app(exclusions, use_exclusions=True, results=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.exclusions = {k: list(v) for k, v in exclusions.items()}
    app.use_exclusions_behavioral = _Var(use_exclusions)
    app.behavioral_results = results if results is not None else [
        {'Subject': 'S1', 'Open_Arms': 10.0},
        {'Subject': 'S2', 'Open_Arms': 20.0},
        {'Subject': 'S3', 'Open_Arms': 30.0},
    ]
    app.log_message = lambda *a, **k: None
    return app


def _subjects(rows):
    return [r['Subject'] for r in rows]


# --------------------------------------------------------------------------- #
#  the one rule                                                                #
# --------------------------------------------------------------------------- #

def test_behavior_pseudo_channel_excludes():
    app = _app({'S2': ['Behavior']})
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S3']


def test_photometry_channel_alone_does_not_exclude_behavior():
    """The export's old rule dropped S2 here; a green channel says nothing
    about whether the tracking is usable."""
    app = _app({'S2': ['G1']})
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S2', 'S3']


def test_photometry_and_behavior_together_still_exclude():
    app = _app({'S2': ['G1', 'Behavior']})
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S3']


def test_toggle_off_keeps_everything():
    app = _app({'S2': ['Behavior']}, use_exclusions=False)
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S2', 'S3']


def test_toggle_flipped_after_calculation_is_honoured():
    """The store is not re-computed, so the filter must live at the read."""
    app = _app({'S2': ['Behavior']}, use_exclusions=False)
    assert len(app.included_behavioral_results()) == 3
    app.use_exclusions_behavioral = _Var(True)
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S3']


def test_no_exclusions_at_all_is_a_passthrough():
    app = _app({})
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S2', 'S3']


def test_explicit_use_exclusions_overrides_the_toggle():
    app = _app({'S2': ['Behavior']}, use_exclusions=False)
    rows = app.included_behavioral_results(use_exclusions=True)
    assert _subjects(rows) == ['S1', 'S3']


def test_missing_toggle_attribute_does_not_crash():
    app = _app({'S2': ['Behavior']})
    del app.use_exclusions_behavioral
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S2', 'S3']


def test_time_binned_rows_filter_per_subject_not_per_row():
    rows = [{'Subject': 'S1', 'Time_Bin': '0-60s', 'Open_Arms': 1.0},
            {'Subject': 'S1', 'Time_Bin': '60-120s', 'Open_Arms': 2.0},
            {'Subject': 'S2', 'Time_Bin': '0-60s', 'Open_Arms': 3.0},
            {'Subject': 'S2', 'Time_Bin': '60-120s', 'Open_Arms': 4.0}]
    app = _app({'S2': ['Behavior']}, results=rows)
    assert _subjects(app.included_behavioral_results()) == ['S1', 'S1']


# --------------------------------------------------------------------------- #
#  table, plot and CSV agree                                                   #
# --------------------------------------------------------------------------- #

class _Tree:
    def __init__(self):
        self.rows = []
        self._cols = ()

    def get_children(self):
        return []

    def delete(self, *a):
        pass

    def heading(self, *a, **k):
        pass

    def column(self, *a, **k):
        pass

    def insert(self, parent, index, values=()):
        self.rows.append(tuple(values))

    def __setitem__(self, key, value):
        if key == 'columns':
            self._cols = tuple(value)

    def __getitem__(self, key):
        return self._cols if key == 'columns' else None


def _export_to(app, path, monkeypatch):
    """Drive the real export_behavioral_metrics into *path*; return its rows."""
    import pandas as pd

    monkeypatch.setattr(G.filedialog, 'asksaveasfilename', lambda **k: str(path))
    monkeypatch.setattr(G.messagebox, 'showinfo', lambda *a, **k: None)
    monkeypatch.setattr(G.messagebox, 'showwarning', lambda *a, **k: None)
    app.use_time_bins = _Var(False)
    app._create_export_metadata = lambda kind: {}
    app._add_metadata_comment_to_csv = lambda fn, md: None

    app.export_behavioral_metrics()
    return pd.read_csv(path)


def test_export_keeps_a_subject_excluded_only_on_a_photometry_channel(tmp_path, monkeypatch):
    """The old export rule dropped any subject with a non-empty exclusion list."""
    app = _app({'S2': ['G1']})
    df = _export_to(app, tmp_path / 'beh.csv', monkeypatch)
    assert sorted(df['Subject']) == ['S1', 'S2', 'S3'], (
        "S2 was dropped from the CSV because an unrelated photometry channel "
        "was excluded")


def test_export_drops_a_behavior_excluded_subject(tmp_path, monkeypatch):
    app = _app({'S2': ['Behavior']})
    df = _export_to(app, tmp_path / 'beh.csv', monkeypatch)
    assert sorted(df['Subject']) == ['S1', 'S3']


def test_export_matches_the_table(tmp_path, monkeypatch):
    app = _app({'S2': ['G1'], 'S3': ['Behavior']})
    app.behav_tree = _Tree()
    app._display_behavioral_results()
    table_subjects = sorted({r[0] for r in app.behav_tree.rows})

    df = _export_to(app, tmp_path / 'beh.csv', monkeypatch)
    assert sorted(df['Subject']) == table_subjects == ['S1', 'S2']


def test_table_export_and_plot_agree_on_n():
    """One subject with an unrelated photometry exclusion, one behavioral."""
    app = _app({'S2': ['G1'], 'S3': ['Behavior']})
    app.behav_tree = _Tree()

    app._display_behavioral_results()
    table_subjects = {r[0] for r in app.behav_tree.rows}

    # What the export and every plot site now read.
    shared = set(_subjects(app.included_behavioral_results()))

    assert table_subjects == shared == {'S1', 'S2'}, (
        'the table, the CSV and the plots disagree about which subjects count')
