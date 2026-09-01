"""Kinematics binned-signal curves must export every subject, not just the mean.

"Signal by acceleration bin" (and its velocity / distance siblings, which share
_kin_signal_by_bin) used to hand the Export button a table of group means only:
four columns, one row per bin per group. The animals behind the error bars were
gone, so the file could not be pasted into Prism next to the Bout Analysis
exports or fed to a per-animal test.

The table now follows the Bout Analysis plot-data layout -- Group | Subject |
one column per bin, one row per animal, then mean / SEM / n underneath -- and
the summary rows must be the values the error bars are actually drawn from,
not a second computation that can drift from the plot.
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pytest

import fp_analysis_gui as G


def _app():
    app = object.__new__(G.FPAnalysisGUI)
    app.kin_gfx = {}
    return app


def _records(n_per_group=4, n_samples=600, seed=0):
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_per_group * 2):
        accel = rng.normal(0, 8, n_samples)
        records.append({
            'subject': f'S{i + 1:02d}',
            'group': 'AIR' if i < n_per_group else 'CIE',
            'accel': accel,
            'vel': np.abs(accel),
            'sig': rng.normal(0, 1, n_samples),
        })
    return records


def _run(mode, nbins=5):
    app = _app()
    records = _records()
    fig, info = G.FPAnalysisGUI._kin_signal_by_bin(
        app, records, {'nbins': nbins, 'vel_zscore': False}, 'G0', mode, 'accel')
    assert fig is not None, info
    return app, records, fig


def test_group_export_has_one_row_per_subject():
    app, records, _fig = _run("Group")
    df = app.kin_last_table
    assert list(df.columns[:2]) == ['Group', 'Subject']
    assert len(df.columns) == 2 + 5          # Group, Subject, one column per bin

    subjects = df['Subject'].tolist()
    for k in records:
        assert k['subject'] in subjects, f"{k['subject']} missing from the export"
    # Every animal, plus the mean / SEM / n block per group and one spacer row.
    assert len(df) == len(records) + 1 + 2 * 3


def test_summary_rows_match_the_plotted_error_bars():
    app, _records, fig = _run("Group")
    df = app.kin_last_table
    bin_cols = list(df.columns[2:])

    ax = fig.axes[0]
    for line in ax.lines:
        label = line.get_label()
        if not label or label.startswith('_'):
            continue
        grp = label.split(' (')[0]
        rows = df[df['Group'] == grp]
        mean = rows[rows['Subject'] == 'mean'][bin_cols].to_numpy(float)[0]
        assert np.allclose(line.get_ydata(), mean, equal_nan=True)

        # The exported mean must also be the mean of the exported subjects,
        # which is what makes the block internally consistent for a reader.
        animals = rows[~rows['Subject'].isin(['mean', 'SEM', 'n'])]
        assert np.allclose(np.nanmean(animals[bin_cols].to_numpy(float), axis=0),
                           mean, equal_nan=True)


def test_subject_mode_drops_the_group_column():
    app, records, _fig = _run("Subject")
    df = app.kin_last_table
    assert list(df.columns[:1]) == ['Subject']
    assert 'Group' not in df.columns
    assert df['Subject'].tolist()[:len(records)] == [k['subject'] for k in records]
    assert set(df['Subject'].tolist()[-3:]) == {'mean', 'SEM', 'n'}


def test_bin_columns_are_distinct():
    """Two centers that round alike would silently merge into one column."""
    app, _records, _fig = _run("Group", nbins=30)
    cols = list(app.kin_last_table.columns[2:])
    assert len(cols) == 30
    assert len(set(cols)) == 30
