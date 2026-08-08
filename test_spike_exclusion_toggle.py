"""Spike analysis must honour its "Apply exclusions" toggle.

`analyze_spikes_for_subject` called `is_subject_channel_excluded`
unconditionally, so a channel ticked on the Exclusions tab vanished from the
spike table even with the toggle *off* -- unlike every other spike site, and
unlike every other tab.  It also asked with the name `get_channel_name`
returns, which is `Ch{n}` for a subject whose CSV header carried no G/R
designation, while the Exclusions tab labels that same checkbox `G{n}`; so on a
header-less project the filter silently did nothing even when it was wanted.

The whole-dataset MAD is pooled across subjects before any channel is analysed;
with exclusions on it must not be set by data that is about to be dropped.
"""
import numpy as np
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


def _app(exclusions, use_exclusions, channel_names=('G0', 'G1')):
    """An FPAnalysisGUI shell with two channels of flat, spike-free signal."""
    app = object.__new__(G.FPAnalysisGUI)
    data = {'channel_names': list(channel_names)} if channel_names else {}
    app.processed_data = {'S1': data}
    app.exclusions = {'S1': list(exclusions)}
    app.use_exclusions_spike = _Var(use_exclusions)
    app.log_message = lambda *a, **k: None
    app.get_fps = lambda *a, **k: 30.0
    app.get_num_channels = lambda d: 2
    app.get_channel_name = lambda d, i: (
        d.get('channel_names')[i] if d.get('channel_names') else f'Ch{i}')
    app.get_channel_signal = lambda d, i, kind='corrected': (
        np.zeros(100) if kind == 'corrected' else None)
    app.detect_spikes = lambda sig, fps, global_mad=None: {
        'n_spikes': 0, 'mean_width': 0.0, 'global_mad': global_mad}
    return app


# --------------------------------------------------------------------------- #
#  the toggle                                                                  #
# --------------------------------------------------------------------------- #

def test_toggle_off_keeps_an_excluded_channel():
    app = _app(exclusions=['G0'], use_exclusions=False)
    results = app.analyze_spikes_for_subject('S1')
    assert set(results) == {'G0', 'G1'}, (
        'exclusions were applied although the toggle is off')


def test_toggle_on_drops_an_excluded_channel():
    app = _app(exclusions=['G0'], use_exclusions=True)
    results = app.analyze_spikes_for_subject('S1')
    assert set(results) == {'G1'}


def test_explicit_use_exclusions_overrides_the_toggle():
    """The caller snapshots the toggle once and passes it down."""
    app = _app(exclusions=['G0'], use_exclusions=True)
    assert set(app.analyze_spikes_for_subject('S1', use_exclusions=False)) == {'G0', 'G1'}
    app = _app(exclusions=['G0'], use_exclusions=False)
    assert set(app.analyze_spikes_for_subject('S1', use_exclusions=True)) == {'G1'}


def test_no_toggle_attribute_means_no_filtering():
    """A headless caller with no Tk var must not crash, and must not filter."""
    app = _app(exclusions=['G0'], use_exclusions=True)
    del app.use_exclusions_spike
    assert set(app.analyze_spikes_for_subject('S1')) == {'G0', 'G1'}


# --------------------------------------------------------------------------- #
#  the key the filter asks with                                                #
# --------------------------------------------------------------------------- #

def test_red_designation_is_excluded_by_its_real_name():
    app = _app(exclusions=['R5'], use_exclusions=True, channel_names=('R4', 'R5'))
    assert set(app.analyze_spikes_for_subject('S1')) == {'R4'}


def test_undesignated_header_falls_back_to_the_g_key():
    """No channel_names: the tab labels the boxes G0/G1, so ask for those."""
    app = _app(exclusions=['G1'], use_exclusions=True, channel_names=None)
    results = app.analyze_spikes_for_subject('S1')
    assert set(results) == {'Ch0'}, (
        'the filter asked for Ch1, which the Exclusions tab never writes')


def test_undesignated_header_ignores_a_ch_key():
    """Nothing writes a 'Ch1' exclusion, so it must not filter anything out."""
    app = _app(exclusions=['Ch1'], use_exclusions=True, channel_names=None)
    assert set(app.analyze_spikes_for_subject('S1')) == {'Ch0', 'Ch1'}


# --------------------------------------------------------------------------- #
#  whole-dataset MAD                                                           #
# --------------------------------------------------------------------------- #

class _Listbox:
    def __init__(self, entries):
        self._entries = list(entries)

    def curselection(self):
        return tuple(range(len(self._entries)))

    def get(self, i):
        return self._entries[i]


class _Tree:
    """Records the rows run_spike_analysis writes to the results table."""

    def __init__(self):
        self.rows = []

    def get_children(self):
        return []

    def delete(self, *a):
        pass

    def insert(self, parent, index, values=()):
        self.rows.append(tuple(values))


def _mad_app(use_exclusions):
    """Two subjects; S2's G0 is excluded and carries a wildly different scale.

    Drives the real `run_spike_analysis`, so the whole-dataset MAD under test
    is the one the app actually computes, not a copy of the loop.
    """
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = {
        'S1': {'channel_names': ['G0', 'G1']},
        'S2': {'channel_names': ['G0', 'G1']},
    }
    app.exclusions = {'S2': ['G0']}
    app.use_exclusions_spike = _Var(use_exclusions)
    app.spike_listbox = _Listbox(['S1', 'S2'])
    app.spike_mode_var = _Var('Subject')
    app.spike_mad_mode_var = _Var('whole_dataset')
    app.spike_channel_vars = {'G0': _Var(True), 'G1': _Var(True)}
    app.spike_tree = _Tree()
    app.spike_results = {}
    app.spike_data = {}
    app.log_message = lambda *a, **k: None
    app.get_fps = lambda *a, **k: 30.0
    app.get_num_channels = lambda d: 2
    app.get_channel_name = lambda d, i: d['channel_names'][i]

    def _sig(d, i, kind='corrected'):
        if kind != 'corrected':
            return None
        scale = 100.0 if (d is app.processed_data['S2'] and i == 0) else 1.0
        return np.array([-scale, scale] * 50, dtype=float)

    app.get_channel_signal = _sig

    app.mads_used = []

    def _detect(sig, fps, global_mad=None):
        app.mads_used.append(global_mad)
        return {'total_spikes': 0, 'spike_rate_per_min': 0.0,
                'mean_amplitude': 0.0, 'mean_width': 0.0,
                'n_spikes': 0, 'global_mad': global_mad}

    app.detect_spikes = _detect
    return app


def test_excluded_channel_does_not_set_the_whole_dataset_mad():
    app = _mad_app(use_exclusions=True)
    app.run_spike_analysis()
    # S1/G0 is the only G0 left in the pool, so the MAD is its own.
    assert app.mads_used, 'no channel was analysed at all'
    assert min(app.mads_used) == pytest.approx(1.0)
    assert max(app.mads_used) == pytest.approx(1.0), (
        "S2's excluded G0 still moved the threshold every other subject is "
        "measured against")


def test_toggle_off_pools_every_channel_into_the_mad():
    app = _mad_app(use_exclusions=False)
    app.run_spike_analysis()
    assert max(app.mads_used) > 1.0


def test_excluded_channel_produces_no_table_row():
    app = _mad_app(use_exclusions=True)
    app.run_spike_analysis()
    # Rows are (subject, group, channel, ...).
    assert ('S2', 'G0') not in {(r[0], r[2]) for r in app.spike_tree.rows}
    assert ('S2', 'G1') in {(r[0], r[2]) for r in app.spike_tree.rows}


def test_toggle_off_produces_a_row_for_every_channel():
    app = _mad_app(use_exclusions=False)
    app.run_spike_analysis()
    assert {(r[0], r[2]) for r in app.spike_tree.rows} == {
        ('S1', 'G0'), ('S1', 'G1'), ('S2', 'G0'), ('S2', 'G1')}
