"""Zone, distance and Out/Back analyses cover every channel, not just G0/G1.

Their averages were computed for the first two photometry channels only, so on
a five-channel rig (G0-G4) the plots and CSV exports had nothing for G2-G4.
"""
import os
import tempfile
import types

import numpy as np
import pandas as pd

import fp_analysis_gui as G

N_CH = 5
N_ROWS = 400


def _var(v):
    return types.SimpleNamespace(get=lambda v=v: v)


def _beh(level_by_ch):
    """beh_synced with a flat z-score per channel and a simple walk in X/Y."""
    n_kin = len(G.FPAnalysisGUI.BEH_KIN_FIELDS)
    beh = np.zeros((N_ROWS, 6 + N_CH + n_kin))
    t = np.arange(N_ROWS)
    beh[:, 0] = t
    beh[:, 2] = np.linspace(-20, 20, N_ROWS)          # X
    beh[:, 3] = np.linspace(-20, 20, N_ROWS)[::-1]    # Y
    for ch in range(N_CH):
        beh[:, 6 + ch] = level_by_ch[ch]
    kb = 6 + N_CH
    beh[:, kb + 3] = np.abs(beh[:, 2])   # distance from center X
    beh[:, kb + 4] = np.abs(beh[:, 3])   # distance from center Y
    return beh


def _app(ticked=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'maze_type': 'OFT', 'preboutframes': 15, 'postboutframes': 15}
    app.zones = {'center': {'category': 'center'}, 'periphery': {'category': 'periphery'}}
    app.classify_zone = lambda x, y: 'center' if abs(x) < 10 and abs(y) < 10 else 'periphery'
    app.exclusions = {}
    app.processed_data = {}
    for i, subject in enumerate(('s1', 's2')):
        beh = _beh([10.0 * ch + i for ch in range(N_CH)])
        zones = [app.classify_zone(x, y) for x, y in beh[:, 2:4]]
        data = {'beh_synced': beh, 'num_photometry_channels': N_CH,
                'channel_names': [f'G{c}' for c in range(N_CH)], 'has_position': True}
        data['zone_averages'] = app.calculate_zone_averages(beh, zones, n_channels=N_CH)
        x, y, e = app.calculate_distance_based_averages(beh, n_channels=N_CH)
        data.update(distance_averages_x=x, distance_averages_y=y, distance_averages_euclidean=e)
        app.processed_data[subject] = data
    ticked = range(N_CH) if ticked is None else ticked
    app.viz_channel_vars = [_var(1 if c in ticked else 0) for c in range(N_CH)]
    app.show_g0, app.show_g1 = _var(True), _var(True)
    app.use_exclusions_viz = _var(True)
    app.plot_by_var = _var('Subject')
    return app


def test_calculators_emit_every_channel():
    app = _app()
    za = app.processed_data['s2']['zone_averages']
    assert za['center']['G4_mean'] == 41.0
    d = app.processed_data['s1']['distance_averages_x']
    assert all(f'G{c}_mean' in row for row in d.values() for c in range(N_CH))
    md = app.calculate_max_distance_zone(app.processed_data['s1']['beh_synced'])
    assert md['G3_mean'] == 30.0


def test_stale_two_channel_stats_are_rebuilt():
    app = _app()
    data = app.processed_data['s1']
    for row in data['zone_averages'].values():
        for c in range(2, N_CH):
            for stat in ('mean', 'std', 'n'):
                row.pop(f'G{c}_{stat}')
    app._ensure_spatial_stats_channels(data)
    assert data['zone_averages']['center']['G4_mean'] == 40.0


def test_zone_average_plot_draws_a_bar_per_channel():
    from matplotlib.figure import Figure
    app = _app()
    fig = Figure()
    app.plot_zone_averages_multi(fig, ['s1', 's2'])
    labels = [c.get_label() for c in fig.axes[0].containers
              if not c.get_label().startswith('_')]
    assert labels == [f'G{c} Mean±SEM' for c in range(N_CH)]


def test_distance_group_comparison_has_a_row_per_channel():
    from matplotlib.figure import Figure
    app = _app(ticked={0, 2, 4})
    app._series_members = lambda g: {'A': ['s1'], 'B': ['s2']}[g]
    fig = Figure()
    app._plot_distance_group_comparison(fig, ['A', 'B'])
    titles = [ax.get_title() for ax in fig.axes]
    assert any(t.startswith('G4 - X Axis') for t in titles)
    assert len(fig.axes) == 3 * 2


def test_zone_average_export_writes_every_ticked_channel(monkeypatch):
    app = _app()
    app.plot_type_var = _var('Zone Averages')
    app.viz_subject_listbox = types.SimpleNamespace(
        curselection=lambda: (0, 1), get=lambda i: ['s1', 's2'][i])
    app._create_export_metadata = lambda kind='plot': {}
    app._add_metadata_comment_to_csv = lambda *a, **k: None
    monkeypatch.setattr(G, 'messagebox', types.SimpleNamespace(
        showinfo=lambda *a, **k: None, showerror=lambda *a, **k: None,
        showwarning=lambda *a, **k: None))
    tmp = tempfile.mkdtemp()
    monkeypatch.setattr(G, 'filedialog', types.SimpleNamespace(
        asksaveasfilename=lambda **k: os.path.join(tmp, 'z.csv')))
    app.show_470, app.show_570 = _var(True), _var(False)
    app.export_plot_data()
    df = pd.read_csv(os.path.join(tmp, 'z.csv'))
    assert sorted(df['channel'].unique()) == [f'G{c}' for c in range(N_CH)]
    s2_g4 = df[(df['subject'] == 's2') & (df['channel'] == 'G4')]
    assert s2_g4['center_mean'].iloc[0] == 41.0
