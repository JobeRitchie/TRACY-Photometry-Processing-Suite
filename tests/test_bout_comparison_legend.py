"""Compare Across Bouts keys many bout bins with a colour bar, not a legend.

With one bout per bin a walk session has 80+ bins; a legend entry per bin
outgrew each panel, and once one panel per channel was drawn the legends
swamped the figure.
"""
import types

import numpy as np
from matplotlib.figure import Figure

import fp_analysis_gui as G


def _app():
    app = object.__new__(G.FPAnalysisGUI)
    app.log_message = lambda *a, **k: None
    app.params = {'preboutframes': 10}
    app.draw_zero_line = lambda ax: None
    return app


def _collected(n_bins):
    trace = np.zeros(30)
    return {'Ch0': {b: [('s1', b, trace + b)] for b in range(n_bins)}}


def test_many_bins_get_a_colour_bar_instead_of_a_legend():
    app = _app()
    fig = Figure()
    app._render_bout_comparison(fig, [(None, None, _collected(40))],
                                [('Ch0', 'G0')], 'Walk', 10.0, 1, 'sub')
    panel = fig.axes[0]
    assert panel.get_legend() is None
    assert len(fig.axes) == 2   # panel + colour bar


def test_group_legend_keeps_one_entry_per_group_after_relocation():
    app = _app()
    fig = Figure()
    series = [('Group A', '#d62728', _collected(20)),
              ('Group B', '#1f77b4', _collected(20))]
    app._render_bout_comparison(fig, series, [('Ch0', 'G0')], 'Walk', 10.0, 1, 'sub')
    app._keep_legends_clear_of_data(fig)
    labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
    assert labels == ['Group A', 'Group B']


def test_few_bins_keep_the_per_bin_legend():
    app = _app()
    fig = Figure()
    app._render_bout_comparison(fig, [(None, None, _collected(4))],
                                [('Ch0', 'G0')], 'Walk', 10.0, 1, 'sub')
    assert len(fig.axes[0].get_legend().get_texts()) == 4
