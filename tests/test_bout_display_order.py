"""Bout rows/columns can be blocked by bout ordinal instead of by subject.

The default layout keeps every bout of one subject together, subjects blocked
by group. The "Bout number" order interleaves them so bout 1 of every subject
sits together, then bout 2, and so on -- which is how a within-session drift is
read across a whole cohort. These tests pin both layouts for the heatmap row
builder and for the wide-format export column order, including the two cases
where the ordinal order must NOT apply: averaging within subject (no ordinal
left to block on) and subjects with unequal bout counts (short subjects simply
drop out of the later blocks rather than shifting anyone).
"""
import numpy as np
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


def _app(order=G.BOUT_ORDER_BY_SUBJECT):
    """An FPAnalysisGUI shell carrying only the bout-order option."""
    app = object.__new__(G.FPAnalysisGUI)
    app.viz_bout_order_var = _Var(order)
    return app


def _rows(app, subject_bout_rows, subject_order, average=False, subject_bouts=None):
    return app._build_group_heatmap_rows(
        'G0', ['A', 'B'], subject_order, subject_bouts or {},
        subject_bout_rows, average)


# --------------------------------------------------------------------------
# _bout_order_by_number

def test_order_defaults_to_by_subject_without_the_var():
    app = object.__new__(G.FPAnalysisGUI)
    assert app._bout_order_by_number() is False


def test_order_reads_the_var():
    assert _app(G.BOUT_ORDER_BY_NUMBER)._bout_order_by_number() is True
    assert _app(G.BOUT_ORDER_BY_SUBJECT)._bout_order_by_number() is False


# --------------------------------------------------------------------------
# Heatmap rows

def _two_group_store():
    """Two groups, two subjects each, bouts tagged <subject><ordinal>."""
    subject_order = [('A', 's1'), ('A', 's2'), ('B', 's3'), ('B', 's4')]
    rows = {}
    for group, subject in subject_order:
        rows[(group, subject, 'G0')] = [
            np.full(4, float(f'{ord(subject[1]) - 48}.{i}'))
            for i in (1, 2)
        ]
    return subject_order, rows


def test_by_subject_order_blocks_rows_per_group():
    app = _app()
    subject_order, store = _two_group_store()
    rows, ticks, labels, blocks = _rows(app, store, subject_order)

    assert len(rows) == 8
    assert labels == ['s1', 's2', 's3', 's4']
    assert blocks == [4, 4]                      # one block per group
    # s1's two bouts land first, s2's next, ...
    assert rows[0][0] == pytest.approx(1.1)
    assert rows[1][0] == pytest.approx(1.2)
    assert rows[2][0] == pytest.approx(2.1)


def test_by_number_order_blocks_rows_per_bout_ordinal():
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    subject_order, store = _two_group_store()
    rows, ticks, labels, blocks = _rows(app, store, subject_order)

    assert len(rows) == 8                        # same rows, new order
    assert labels == ['Bout 1', 'Bout 2']
    assert blocks == [4, 4]                      # one block per bout number
    # First block: bout 1 of s1, s2, s3, s4 -- group order preserved inside it.
    assert [r[0] for r in rows[:4]] == pytest.approx([1.1, 2.1, 3.1, 4.1])
    assert [r[0] for r in rows[4:]] == pytest.approx([1.2, 2.2, 3.2, 4.2])


def test_by_number_order_handles_unequal_bout_counts():
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    subject_order = [('A', 's1'), ('A', 's2')]
    store = {
        ('A', 's1', 'G0'): [np.full(3, 1.1), np.full(3, 1.2), np.full(3, 1.3)],
        ('A', 's2', 'G0'): [np.full(3, 2.1)],
    }
    rows, ticks, labels, blocks = _rows(app, store, subject_order)

    assert labels == ['Bout 1', 'Bout 2', 'Bout 3']
    assert blocks == [2, 1, 1]                   # s2 only reaches bout 1
    assert [r[0] for r in rows] == pytest.approx([1.1, 2.1, 1.2, 1.3])


def test_averaging_within_subject_ignores_the_bout_number_order():
    """One row per subject leaves no ordinal, so the group layout must stand."""
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    subject_order = [('A', 's1'), ('B', 's3')]
    averages = {('A', 's1', 'G0'): np.full(3, 1.0),
                ('B', 's3', 'G0'): np.full(3, 3.0)}
    rows, ticks, labels, blocks = app._build_group_heatmap_rows(
        'G0', ['A', 'B'], subject_order, averages, {}, True)

    assert labels == ['s1', 's3']
    assert blocks == [1, 1]
    assert [r[0] for r in rows] == pytest.approx([1.0, 3.0])


# --------------------------------------------------------------------------
# _bout_number_blocks

def test_bout_number_blocks_centres_its_tick_on_each_block():
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    records = [(1, 'a'), (1, 'b'), (1, 'c'), (2, 'd'), (2, 'e')]
    rows, ticks, labels, blocks = app._bout_number_blocks(records)

    assert rows == ['a', 'b', 'c', 'd', 'e']
    assert blocks == [3, 2]
    assert ticks == pytest.approx([1.0, 3.5])
    assert labels == ['Bout 1', 'Bout 2']


# --------------------------------------------------------------------------
# Export column order

def _export_dicts():
    """Two subjects' wide-format bout columns, plus the ordinal map."""
    dicts = [
        {'A_s1_G0_bout1': np.zeros(2), 'A_s1_G0_bout2': np.zeros(2),
         'A_s1_G0_bout3': np.zeros(2)},
        {'B_s2_G0_bout1': np.zeros(2), 'B_s2_G0_bout2': np.zeros(2)},
    ]
    ordinals = {'A_s1_G0_bout1': 1, 'A_s1_G0_bout2': 2, 'A_s1_G0_bout3': 3,
                'B_s2_G0_bout1': 1, 'B_s2_G0_bout2': 2}
    return dicts, ordinals


def test_export_columns_default_to_subject_blocks():
    app = _app()
    dicts, ordinals = _export_dicts()
    names = [n for n, _ in app._ordered_bout_columns(dicts, ordinals)]
    assert names == ['A_s1_G0_bout1', 'A_s1_G0_bout2', 'A_s1_G0_bout3',
                     'B_s2_G0_bout1', 'B_s2_G0_bout2']


def test_export_columns_can_block_by_bout_number():
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    dicts, ordinals = _export_dicts()
    names = [n for n, _ in app._ordered_bout_columns(dicts, ordinals)]
    assert names == ['A_s1_G0_bout1', 'B_s2_G0_bout1',
                     'A_s1_G0_bout2', 'B_s2_G0_bout2',
                     'A_s1_G0_bout3']


def test_export_columns_keep_subject_averages_together():
    """Averaged columns carry no ordinal; they sort ahead as one block."""
    app = _app(G.BOUT_ORDER_BY_NUMBER)
    dicts = [{'A_s1_G0_avg': np.zeros(2)}, {'B_s2_G0_avg': np.zeros(2)}]
    names = [n for n, _ in app._ordered_bout_columns(dicts, {})]
    assert names == ['A_s1_G0_avg', 'B_s2_G0_avg']


# --------------------------------------------------------------------------
# Rendered plots
#
# These drive the real plot functions on a real GUI, because the ordering has
# to survive the whole collect-pad-imshow path, not just the row builder: the
# rows are pooled across subjects before they reach the heatmap, and the axis
# label and tick labels have to follow them.

@pytest.fixture
def viz_app(tk_root):
    """A real GUI on a Toplevel, set up for a multi-subject bout plot."""
    tk = pytest.importorskip('tkinter')
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg')
    top = tk.Toplevel(tk_root)
    top.withdraw()
    app = G.FPAnalysisGUI(top)
    app.plot_type_var.set('Extracted Bouts')
    app.behavior_var.set('Grooming')
    app.plot_by_var.set('Subject')
    app.use_exclusions_viz.set(False)
    app.viz_average_within_subject.set(False)
    app.viz_max_bouts_var.set('all')
    app.show_g0.set(True)
    app.show_g1.set(False)
    yield app
    top.destroy()


def _stock_bouts(app, subjects, n_bouts=2):
    """Bout <s>.<b> for subject index s, bout ordinal b, on channel 0."""
    app.processed_data = {}
    for i, subj in enumerate(subjects, 1):
        traces = [np.full(6, float(f'{i}.{b}')) for b in range(1, n_bouts + 1)]
        app.processed_data[subj] = {
            'bouts': {'Grooming': {k: list(traces) for k in ('G0', 'Ch0')}},
            'num_photometry_channels': 1,
            'has_470': True,
        }


def _heatmap_rows(fig):
    """First column of the first heatmap's matrix, plus its axis labelling."""
    ax = next(a for a in fig.get_axes() if a.get_images())
    rows = np.asarray(ax.get_images()[0].get_array())[:, 0]
    return (list(np.round(rows, 2)), ax.get_ylabel(),
            [t.get_text() for t in ax.get_yticklabels()])


def test_pooled_bout_heatmap_keeps_subject_blocks_by_default(viz_app):
    plt = pytest.importorskip('matplotlib.pyplot')
    subjects = ['s1', 's2', 's3']
    _stock_bouts(viz_app, subjects)
    viz_app.viz_bout_order_var.set(G.BOUT_ORDER_BY_SUBJECT)

    fig = plt.figure(figsize=(10, 6))
    try:
        viz_app.plot_extracted_bouts_multi(fig, subjects)
        rows, ylabel, _ = _heatmap_rows(fig)
    finally:
        plt.close(fig)

    assert rows == pytest.approx([1.1, 1.2, 2.1, 2.2, 3.1, 3.2])
    assert ylabel == 'Bout #'


def test_pooled_bout_heatmap_can_block_by_bout_number(viz_app):
    plt = pytest.importorskip('matplotlib.pyplot')
    subjects = ['s1', 's2', 's3']
    _stock_bouts(viz_app, subjects)
    viz_app.viz_bout_order_var.set(G.BOUT_ORDER_BY_NUMBER)

    fig = plt.figure(figsize=(10, 6))
    try:
        viz_app.plot_extracted_bouts_multi(fig, subjects)
        rows, ylabel, tick_labels = _heatmap_rows(fig)
    finally:
        plt.close(fig)

    assert rows == pytest.approx([1.1, 2.1, 3.1, 1.2, 2.2, 3.2])
    assert ylabel == 'Bout number (all subjects)'
    assert tick_labels == ['Bout 1', 'Bout 2']


def test_zone_entry_heatmap_can_block_by_bout_number(viz_app):
    """The same option drives the Zone Entry Bouts heatmap."""
    plt = pytest.importorskip('matplotlib.pyplot')
    subjects = ['s1', 's2']
    viz_app.params['maze_type'] = 'EPM'
    viz_app.zone_entry_type_var.set('Open Arm Entry')
    viz_app.processed_data = {}
    for i, subj in enumerate(subjects, 1):
        traces = [np.full(6, float(f'{i}.{b}')) for b in (1, 2)]
        viz_app.processed_data[subj] = {
            'entry_bouts': {'transition_to_open': {
                'open_arm': {'G0': list(traces), 'G1': []}}},
            'num_photometry_channels': 1,
            'has_470': True,
        }
    viz_app.viz_bout_order_var.set(G.BOUT_ORDER_BY_NUMBER)

    fig = plt.figure(figsize=(10, 6))
    try:
        viz_app.plot_zone_entry_bouts_multi(fig, subjects)
        rows, ylabel, tick_labels = _heatmap_rows(fig)
    finally:
        plt.close(fig)

    assert rows == pytest.approx([1.1, 2.1, 1.2, 2.2])
    assert ylabel == 'Bout number (all subjects)'
    assert tick_labels == ['Bout 1', 'Bout 2']
