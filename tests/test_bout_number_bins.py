"""Compare Across Bouts groups bouts by their ordinal and exports the traces.

The view used to draw every bout as its own line -- with a dozen bouts across a
handful of subjects that is an unreadable pile -- and had no export path at all,
so the z-scored traces behind it could not leave the GUI.

These tests pin the replacement: bouts bin by ordinal ("every 3 bouts"), each
bin becomes one mean +/- SEM line, and the CSV export reproduces exactly the
numbers that were plotted.
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


def _bout(subject_idx, bout_num):
    """A flat trace whose level identifies the subject and the bout number."""
    return np.full(PRE + POST, 100.0 * subject_idx + bout_num, dtype=float)


def _app():
    """A GUI shell with synthetic bouts and no Tk widgets.

    The Compare Across Bouts path only reads StringVar/BooleanVar-shaped
    objects, so stand-ins keep this headless and fast.
    """
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': PRE, 'postboutframes': POST,
                  'preboutseconds': PRE / FPS, 'postboutseconds': POST / FPS,
                  'boutframe_processing_style': 'onset',
                  'baseline_correct_bouts': False}

    app.processed_data = {}
    for s_idx, subject in enumerate(SUBJECTS, 1):
        entry = {'_prebout': PRE, '_postbout': POST, '_fs': FPS}
        for ch in ('Ch0', 'Ch1'):
            entry[ch] = [_bout(s_idx, n) for n in range(1, N_BOUTS + 1)]
        app.processed_data[subject] = {
            'photometry_fps': FPS,
            'channel_names': ['G0', 'G1'],
            'bouts': {'Lick': entry},
        }

    var = lambda v: types.SimpleNamespace(get=lambda v=v: v)
    app.viz_bout_bin_var = var('3')
    app.viz_max_bouts_var = var('all')
    app.viz_average_within_subject = var(False)
    app.viz_bout_bin_individual = var(False)
    app.use_exclusions_viz = var(False)
    app.bout_number_var = var('All')
    app.behavior_var = var('Lick')
    app.viz_channel_vars = [var(1), var(1)]
    app.mean_linewidth_var = var('1.5')
    app.trace_linewidth_var = var('0.3')
    app.trace_alpha_var = var('0.4')
    app.zero_line_color_var = var('black')
    app.zero_line_width_var = var('1.0')
    return app


def _slots(app):
    return app._viz_bout_channel_slots(SUBJECTS, max_slots=2)


# --------------------------------------------------------------------------
# binning
# --------------------------------------------------------------------------

def test_bouts_land_in_bins_of_the_requested_size():
    app = _app()
    by_slot = app._bout_comparison_slot_subjects(SUBJECTS, _slots(app))
    collected = app._collect_bout_number_bins(by_slot, 'Lick', bin_size=3)

    per_bin = collected['Ch0']
    # 7 bouts at 3 per bin -> #1-3, #4-6, #7.
    assert sorted(per_bin) == [0, 1, 2]
    assert [len(per_bin[b]) for b in (0, 1, 2)] == [9, 9, 3]   # x3 subjects

    # Bin 1 holds bouts 4, 5 and 6 -- and only those.
    assert sorted({int(tag) for _, tag, _ in per_bin[1]}) == [4, 5, 6]
    assert {subj for subj, _, _ in per_bin[1]} == set(SUBJECTS)


def test_bin_size_of_one_keeps_a_bin_per_bout_number():
    app = _app()
    app.viz_bout_bin_var = types.SimpleNamespace(get=lambda: '1')
    by_slot = app._bout_comparison_slot_subjects(SUBJECTS, _slots(app))
    collected = app._collect_bout_number_bins(by_slot, 'Lick', bin_size=1)
    assert sorted(collected['Ch0']) == list(range(N_BOUTS))
    assert all(len(v) == len(SUBJECTS) for v in collected['Ch0'].values())


def test_selecting_one_bout_number_keeps_only_that_bout():
    app = _app()
    by_slot = app._bout_comparison_slot_subjects(SUBJECTS, _slots(app))
    collected = app._collect_bout_number_bins(by_slot, 'Lick', 3, selected_bout=5)
    per_bin = collected['Ch0']
    assert list(per_bin) == [1]                     # bout 5 lives in bin #4-6
    assert {tag for _, tag, _ in per_bin[1]} == {'5'}


def test_averaging_within_subject_gives_one_trace_per_subject_per_bin():
    app = _app()
    app.viz_average_within_subject = types.SimpleNamespace(get=lambda: True)
    by_slot = app._bout_comparison_slot_subjects(SUBJECTS, _slots(app))
    per_bin = app._collect_bout_number_bins(by_slot, 'Lick', 3)['Ch0']
    assert len(per_bin[0]) == len(SUBJECTS)
    # Subject 1's bouts 1-3 hold 101/102/103, so its bin mean is 102.
    by_subject = {s: t for s, _, t in per_bin[0]}
    assert np.allclose(by_subject['s1'], 102.0)


def test_bin_labels_read_as_ranges():
    app = _app()
    assert app._bout_bin_label(0, 3) == 'Bouts #1–3'
    assert app._bout_bin_label(2, 3) == 'Bouts #7–9'
    assert app._bout_bin_label(4, 1) == 'Bout #5'


def test_a_blank_or_bad_bin_size_falls_back_to_one():
    app = _app()
    for raw in ('', 'abc', '0', '-4'):
        app.viz_bout_bin_var = types.SimpleNamespace(get=lambda raw=raw: raw)
        assert app._viz_bout_bin_size() == 1


def test_mean_and_sem_over_ragged_traces():
    mean, sem, n = G.FPAnalysisGUI._bout_bin_mean_sem(
        [np.array([1.0, 2.0, 3.0]), np.array([3.0, 4.0])])
    assert np.allclose(mean, [2.0, 3.0, 3.0])
    assert np.allclose(n, [2, 2, 1])
    assert sem[2] == 0.0            # a single contributor has no SEM
    assert sem[0] > 0


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------

def test_the_plot_draws_one_mean_line_per_bin_not_one_per_bout():
    from matplotlib.figure import Figure

    app = _app()
    fig = Figure()
    app.plot_bout_comparison_multi(fig, SUBJECTS)

    axes = fig.get_axes()
    assert len(axes) == 2                       # one panel per channel slot
    for ax in axes:
        labelled = [ln for ln in ax.get_lines() if ln.get_label().startswith('Bout')]
        assert len(labelled) == 3               # 3 bins, not 21 bouts
        assert '(n=9)' in labelled[0].get_label()
        assert '(n=3)' in labelled[-1].get_label()
        # The bin mean is the mean of its bouts: subjects 1-3 over bouts 1-3
        # average to 202.0 at every sample.
        assert np.allclose(labelled[0].get_ydata(), 202.0)


def test_group_mode_draws_every_group_and_bin():
    from matplotlib.figure import Figure

    app = _app()
    app._series_members = lambda g: {'A': ['s1', 's2'], 'B': ['s3']}[g]
    fig = Figure()
    app.plot_bout_comparison_by_group(fig, ['A', 'B'])

    ax = fig.get_axes()[0]
    labels = [ln.get_label() for ln in ax.get_lines() if ' — ' in ln.get_label()]
    assert len(labels) == 6                     # 2 groups x 3 bins
    assert any(l.startswith('A — Bouts #1') for l in labels)
    assert any(l.startswith('B — Bouts #7') for l in labels)


def test_a_single_subject_panel_still_bins():
    from matplotlib.figure import Figure

    app = _app()
    fig = Figure()
    app.plot_bout_comparison_single(fig, app.processed_data['s1'], 's1')
    ax = fig.get_axes()[0]
    labelled = [ln for ln in ax.get_lines() if ln.get_label().startswith('Bout')]
    assert len(labelled) == 3
    assert np.allclose(labelled[0].get_ydata(), 102.0)   # s1 bouts 1-3


# --------------------------------------------------------------------------
# export
# --------------------------------------------------------------------------

def test_export_writes_the_binned_traces_and_their_summary():
    pd = pytest.importorskip('pandas')

    app = _app()
    app._create_export_metadata = lambda kind='plot': {'Export Type': kind}
    shown = {}
    app_msg = types.SimpleNamespace(
        showinfo=lambda *a, **k: shown.update(info=a),
        showerror=lambda *a, **k: shown.update(error=a))
    old_msg = G.messagebox
    G.messagebox = app_msg
    try:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, 'out.csv')
            app._export_bout_bin_traces(target, SUBJECTS, {})
            assert 'error' not in shown, shown
            written = sorted(os.listdir(tmp))
    finally:
        G.messagebox = old_msg

    # One summary + one trace file per channel.
    assert written == ['out_Lick_G0_boutbins.csv',
                       'out_Lick_G0_boutbins_traces.csv',
                       'out_Lick_G1_boutbins.csv',
                       'out_Lick_G1_boutbins_traces.csv']


def test_exported_numbers_match_what_is_plotted():
    pd = pytest.importorskip('pandas')

    app = _app()
    app._create_export_metadata = lambda kind='plot': {'Export Type': kind}
    G_msg = G.messagebox
    G.messagebox = types.SimpleNamespace(showinfo=lambda *a, **k: None,
                                         showerror=lambda *a, **k: None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            app._export_bout_bin_traces(os.path.join(tmp, 'out.csv'), SUBJECTS, {})
            summary = pd.read_csv(os.path.join(tmp, 'out_Lick_G0_boutbins.csv'),
                                  comment='#')
            traces = pd.read_csv(os.path.join(tmp, 'out_Lick_G0_boutbins_traces.csv'),
                                 comment='#')
    finally:
        G.messagebox = G_msg

    # Time axis is centred on bout onset.
    assert np.isclose(summary['time_from_onset_s'].iloc[PRE], 0.0)

    assert np.allclose(summary['All_Bouts_1-3_mean'].dropna(), 202.0)
    assert np.allclose(summary['All_Bouts_1-3_n'].dropna(), 9)
    assert np.allclose(summary['All_Bouts_7-9_mean'].dropna(), 207.0)

    # Every contributing bout is present, one column each, named by subject.
    bout_cols = [c for c in traces.columns if c != 'time_from_onset_s']
    assert len(bout_cols) == len(SUBJECTS) * N_BOUTS
    assert np.allclose(traces['All_Bouts_1-3_s2_bout2'].dropna(), 202.0)


def test_export_respects_the_selected_bout_number():
    pd = pytest.importorskip('pandas')

    app = _app()
    app.bout_number_var = types.SimpleNamespace(get=lambda: '5')
    app._create_export_metadata = lambda kind='plot': {'Export Type': kind}
    G_msg = G.messagebox
    G.messagebox = types.SimpleNamespace(showinfo=lambda *a, **k: None,
                                         showerror=lambda *a, **k: None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            app._export_bout_bin_traces(os.path.join(tmp, 'out.csv'), SUBJECTS, {})
            traces = pd.read_csv(os.path.join(tmp, 'out_Lick_G0_boutbins_traces.csv'),
                                 comment='#')
    finally:
        G.messagebox = G_msg

    bout_cols = [c for c in traces.columns if c != 'time_from_onset_s']
    assert len(bout_cols) == len(SUBJECTS)
    assert all(c.endswith('_bout5') for c in bout_cols)


def test_group_mode_export_labels_columns_by_group():
    pd = pytest.importorskip('pandas')

    app = _app()
    app._create_export_metadata = lambda kind='plot': {'Export Type': kind}
    G_msg = G.messagebox
    G.messagebox = types.SimpleNamespace(showinfo=lambda *a, **k: None,
                                         showerror=lambda *a, **k: None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            app._export_bout_bin_traces(
                os.path.join(tmp, 'out.csv'), SUBJECTS,
                {'s1': 'CIE', 's2': 'CIE', 's3': 'AIR'})
            summary = pd.read_csv(os.path.join(tmp, 'out_Lick_G0_boutbins.csv'),
                                  comment='#')
    finally:
        G.messagebox = G_msg

    assert 'CIE_Bouts_1-3_mean' in summary.columns
    assert 'AIR_Bouts_1-3_mean' in summary.columns
    # s1/s2 average 101.5..103.5 -> 152.0 across bouts 1-3; s3 alone gives 302.0.
    assert np.allclose(summary['CIE_Bouts_1-3_mean'].dropna(), 152.0)
    assert np.allclose(summary['AIR_Bouts_1-3_mean'].dropna(), 302.0)
    assert np.allclose(summary['AIR_Bouts_1-3_n'].dropna(), 3)
