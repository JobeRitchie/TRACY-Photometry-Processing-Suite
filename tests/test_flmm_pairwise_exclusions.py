"""FLMM Pairwise must honour the Apply-exclusions toggle.

`plot_timecourse_flmm` threads `use_exclusions` into `_run_flmm_factor` and
`_compute_flmm_timecourse_panel`, but the pairwise branch called
`_run_flmm_pairwise`, whose signature had no such parameter -- so the toggle
worked on every other FLMM view and silently did nothing here.  Neither worker
filtered: `_compute_pairwise_within` collected every subject on every channel,
and `_compute_pairwise_across_channels` pooled excluded channels into the very
factor it was testing.

The two modes need different filtering, which is why one parameter could not
just be forwarded:

* *within* channel -- the behavior is the factor and the channel is fixed per
  panel, so filter the subject list per channel, as the timecourse panel does.
* *across* channels -- the channel **is** the factor, so a subject excluded on
  one channel must lose that channel only.  Dropping the subject would remove
  it from every level and change what the contrast means.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


class _CancelEvent:
    def is_set(self):
        return False


class _Q:
    """A progress queue that never cancels."""

    def __init__(self):
        self.items = []
        self.cancel_event = _CancelEvent()

    def put(self, item):
        self.items.append(item)


def _bouts(n, value):
    """n bouts of a flat trace at *value*, long enough to survive L >= 3."""
    return [list(np.full(12, float(value))) for _ in range(n)]


def _app(exclusions, subjects=('S1', 'S2'), channels=('G0', 'G1'),
         behaviors=('Sniff', 'Rear')):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {}
    for si, s in enumerate(subjects):
        entry_by_beh = {}
        for bi, beh in enumerate(behaviors):
            entry = {'_prebout': 4}
            for ci, ch in enumerate(channels):
                entry[ch] = _bouts(3, si * 100 + bi * 10 + ci)
            entry_by_beh[beh] = entry
        app.processed_data[s] = {'bouts': entry_by_beh}
    app.exclusions = {k: list(v) for k, v in exclusions.items()}
    app.params = {'preboutframes': 4}
    app.get_fps = lambda *a, **k: 30.0
    app.log_message = lambda *a, **k: None
    app._realign_bouts = lambda bouts, prebout, src_fs=None: bouts
    app._entry_prebout = lambda entry: entry.get('_prebout', 4)
    return app


# --------------------------------------------------------------------------- #
#  across channels: the channel is the factor                                  #
# --------------------------------------------------------------------------- #

def test_across_channels_drops_only_the_excluded_channel():
    app = _app({'S2': ['G1']})
    Z, subs, factor, L = app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['G0', 'G1'], max_bouts=None, use_exclusions=True)

    pairs = set(zip(subs.tolist(), factor.tolist()))
    assert ('S2', 'G1') not in pairs, "S2's excluded G1 bouts still entered the fit"
    assert ('S2', 'G0') in pairs, 'S2 was dropped from the channel it is included on'
    assert ('S1', 'G0') in pairs and ('S1', 'G1') in pairs


def test_across_channels_keeps_both_levels_populated():
    """Filtering must not collapse the factor to a single level."""
    app = _app({'S2': ['G1']})
    _, _, factor, _ = app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['G0', 'G1'], max_bouts=None, use_exclusions=True)
    assert set(factor.tolist()) == {'G0', 'G1'}


def test_across_channels_toggle_off_pools_everything():
    app = _app({'S2': ['G1']})
    _, subs, factor, _ = app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['G0', 'G1'], max_bouts=None, use_exclusions=False)
    assert ('S2', 'G1') in set(zip(subs.tolist(), factor.tolist()))


def test_across_channels_defaults_to_unfiltered():
    """The default must not change behaviour for callers that never pass it."""
    app = _app({'S2': ['G1']})
    _, subs, factor, _ = app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['G0', 'G1'], max_bouts=None)
    assert ('S2', 'G1') in set(zip(subs.tolist(), factor.tolist()))


def test_across_channels_excluding_every_channel_yields_nothing():
    app = _app({'S1': ['G0', 'G1'], 'S2': ['G0', 'G1']})
    assert app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['G0', 'G1'], max_bouts=None,
        use_exclusions=True) is None


def test_across_channels_respects_a_red_designation():
    app = _app({'S2': ['R5']}, channels=('R4', 'R5'))
    _, subs, factor, _ = app._collect_flmm_across_channels(
        ['S1', 'S2'], 'Sniff', ['R4', 'R5'], max_bouts=None, use_exclusions=True)
    pairs = set(zip(subs.tolist(), factor.tolist()))
    assert ('S2', 'R5') not in pairs
    assert ('S2', 'R4') in pairs


# --------------------------------------------------------------------------- #
#  within channel: the behavior is the factor                                  #
# --------------------------------------------------------------------------- #

def _within_subjects(app, use_exclusions):
    """Run the real worker and report which subjects reached each channel panel."""
    seen = {}

    real_collect = app._collect_flmm_factor

    def _spy(subjects, behaviors, channel, max_bouts=None):
        seen[channel] = list(subjects)
        return real_collect(subjects, behaviors, channel, max_bouts=max_bouts)

    app._collect_flmm_factor = _spy
    # Stop after collection: the fit itself is not what is under test.
    app._fit_factor_timecourse = lambda *a, **k: None
    app._flmm_pairwise_pmatrix = lambda *a, **k: None

    app._compute_pairwise_within(
        ['S1', 'S2'], ['Sniff', 'Rear'], ['G0', 'G1'], None, _Q(),
        use_exclusions=use_exclusions)
    return seen


def test_within_channel_filters_per_channel():
    app = _app({'S2': ['G1']})
    seen = _within_subjects(app, use_exclusions=True)
    assert seen['G0'] == ['S1', 'S2'], 'S2 lost a channel it is included on'
    assert seen['G1'] == ['S1'], "S2's excluded G1 panel still included it"


def test_within_channel_toggle_off_keeps_everyone():
    app = _app({'S2': ['G1']})
    seen = _within_subjects(app, use_exclusions=False)
    assert seen['G0'] == ['S1', 'S2']
    assert seen['G1'] == ['S1', 'S2']


# --------------------------------------------------------------------------- #
#  the parameter is threaded at all                                            #
# --------------------------------------------------------------------------- #

def test_run_flmm_pairwise_accepts_use_exclusions():
    import inspect
    sig = inspect.signature(G.FPAnalysisGUI._run_flmm_pairwise)
    assert 'use_exclusions' in sig.parameters, (
        'plot_timecourse_flmm has nothing to thread the toggle into')
    assert sig.parameters['use_exclusions'].default is False


def test_both_pairwise_workers_accept_use_exclusions():
    import inspect
    for name in ('_compute_pairwise_within', '_compute_pairwise_across_channels'):
        sig = inspect.signature(getattr(G.FPAnalysisGUI, name))
        assert 'use_exclusions' in sig.parameters, name
