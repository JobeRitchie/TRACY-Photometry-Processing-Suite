"""Aggregate visualization views must filter exclusions; inspection views must not.

The rule (confirmed by the user): a *single-subject* view is deliberately
unfiltered -- you look at an excluded trace precisely to judge whether the
exclusion was right. A view that pools subjects into a group statistic is
different: it reports an n, and `export_plot_data` already filters the same
stores, so an unfiltered graph and a filtered CSV disagreed about how many
subjects went into the bar.

`plot_outback_multi` and `_plot_outback_group_comparison` applied no filter at
all. These tests drive the real methods with a stub figure and assert on the
values that reach matplotlib.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _Axis:
    """Records the bar heights each series is drawn with, per subplot."""

    def __init__(self, sink, panel):
        self._sink = sink
        self._panel = panel

    def bar(self, x, heights, *a, **k):
        self._sink.append((self._panel, k.get('label'), list(np.atleast_1d(heights))))
        return []

    def get_ylim(self):
        # Both Out/Back plots annotate n at a fraction of the y range.
        return (0.0, 1.0)

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Fig:
    def __init__(self):
        self.bars = []
        self.messages = []
        self._panels = 0

    def add_subplot(self, *a, **k):
        self._panels += 1
        return _Axis(self.bars, self._panels)

    def text(self, x, y, s, *a, **k):
        self.messages.append(s)

    def __getattr__(self, name):
        return lambda *a, **k: None

    def series(self, panel, label):
        """Heights for one label in one subplot (panel 1 = G0, panel 2 = G1)."""
        for p, lbl, heights in self.bars:
            if p == panel and lbl == label:
                return heights
        return None


def _outback(g0_mean, g1_mean):
    return {
        'out': {'G0_n': 3, 'G0_mean': g0_mean, 'G1_n': 3, 'G1_mean': g1_mean},
        'back': {'G0_n': 3, 'G0_mean': g0_mean, 'G1_n': 3, 'G1_mean': g1_mean},
    }


def _app(subjects, exclusions=None, use_exclusions=True, groups=None):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = subjects
    app.exclusions = {k: list(v) for k, v in (exclusions or {}).items()}
    app.use_exclusions_viz = _Var(use_exclusions)
    app.show_g0 = _Var(True)
    app.show_g1 = _Var(True)
    app.plot_by_var = _Var('Subject')
    app._groups = groups or {}
    app._series_members = lambda g: list(app._groups.get(g, []))
    app._any_subject_has_g1 = lambda subs: True
    app.get_trace_styling = lambda: (2, 1, 0.3)
    app._apply_plot_style = lambda *a, **k: None
    return app


# ---------------------------------------------------------------------------
# plot_outback_multi
# ---------------------------------------------------------------------------

def test_outback_multi_drops_an_excluded_subject_from_the_mean():
    """Excluding S2's slot 0 must leave the G0 bar as S1's value alone."""
    app = _app(
        {'S1': {'channel_names': ['G0', 'G1'], 'outback': _outback(1.0, 5.0)},
         'S2': {'channel_names': ['G0', 'G1'], 'outback': _outback(3.0, 7.0)}},
        exclusions={'S2': ['G0']},
    )
    fig = _Fig()
    app.plot_outback_multi(fig, ['S1', 'S2'])

    g0 = fig.series(1, 'G0')
    assert g0 is not None, f'no G0 series drawn (messages={fig.messages}, bars={fig.bars})'
    assert g0[0] == pytest.approx(1.0), (
        'S2 was averaged in despite being excluded on G0')


def test_outback_multi_keeps_everyone_when_the_toggle_is_off():
    app = _app(
        {'S1': {'channel_names': ['G0', 'G1'], 'outback': _outback(1.0, 5.0)},
         'S2': {'channel_names': ['G0', 'G1'], 'outback': _outback(3.0, 7.0)}},
        exclusions={'S2': ['G0']},
        use_exclusions=False,
    )
    fig = _Fig()
    app.plot_outback_multi(fig, ['S1', 'S2'])

    assert fig.series(1, 'G0')[0] == pytest.approx(2.0), (
        'the toggle was off; both subjects count')


def test_outback_multi_filters_a_red_channel_cohort():
    """The P3 resolver has to carry through: the key here is R4, not G0."""
    app = _app(
        {'S1': {'channel_names': ['R4', 'R5'], 'outback': _outback(1.0, 5.0)},
         'S2': {'channel_names': ['R4', 'R5'], 'outback': _outback(3.0, 7.0)}},
        exclusions={'S2': ['R4']},
    )
    fig = _Fig()
    app.plot_outback_multi(fig, ['S1', 'S2'])

    assert fig.series(1, 'G0')[0] == pytest.approx(1.0)


def test_outback_multi_filters_each_slot_independently():
    """Excluding slot 0 must not disturb slot 1."""
    app = _app(
        {'S1': {'channel_names': ['G0', 'G1'], 'outback': _outback(1.0, 5.0)},
         'S2': {'channel_names': ['G0', 'G1'], 'outback': _outback(3.0, 7.0)}},
        exclusions={'S2': ['G0']},
    )
    fig = _Fig()
    app.plot_outback_multi(fig, ['S1', 'S2'])

    g1 = fig.series(1, 'G1')
    assert g1 is not None, 'no G1 series drawn'
    assert g1[0] == pytest.approx(6.0), 'a G0 exclusion leaked into G1'


# ---------------------------------------------------------------------------
# _plot_outback_group_comparison
# ---------------------------------------------------------------------------

def test_outback_group_comparison_drops_an_excluded_subject():
    app = _app(
        {'A1': {'channel_names': ['G0', 'G1'], 'outback': _outback(1.0, 5.0)},
         'A2': {'channel_names': ['G0', 'G1'], 'outback': _outback(3.0, 7.0)},
         'B1': {'channel_names': ['G0', 'G1'], 'outback': _outback(9.0, 9.0)}},
        exclusions={'A2': ['G0']},
        groups={'A': ['A1', 'A2'], 'B': ['B1']},
    )
    fig = _Fig()
    app._plot_outback_group_comparison(fig, ['A', 'B'])

    # Panel 1 is the G0 subplot; each group is its own bar series there.
    group_a = fig.series(1, 'A')
    assert group_a is not None, f'group A was not drawn: {fig.bars}'
    assert group_a[0] == pytest.approx(1.0), (
        'A2 was averaged into group A despite being excluded on G0')
    assert fig.series(1, 'B')[0] == pytest.approx(9.0), 'group B should be untouched'


def test_outback_group_comparison_respects_the_toggle():
    app = _app(
        {'A1': {'channel_names': ['G0', 'G1'], 'outback': _outback(1.0, 5.0)},
         'A2': {'channel_names': ['G0', 'G1'], 'outback': _outback(3.0, 7.0)}},
        exclusions={'A2': ['G0']},
        use_exclusions=False,
        groups={'A': ['A1', 'A2']},
    )
    fig = _Fig()
    app._plot_outback_group_comparison(fig, ['A'])

    assert fig.series(1, 'A')[0] == pytest.approx(2.0)
