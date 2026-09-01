"""The Prism copy/paste blocks and the coherence frequency-axis setting.

Two things are pinned here.

`_coh_freq_axis_range` is what made "Freq axis min/max (Hz)" reach the
spectrogram panels: they used to pin the y-axis to the *analysis* range, so
narrowing the plot axis moved every line plot and left the heatmaps alone.  The
requested window is intersected with the frequencies actually computed -- a log
axis cannot show the default min of 0, and stretching past the data only adds
blank space.

`_coherence_prism_text` builds what Prism pastes: a Grouped table per band with
one column per (group x epoch) and replicates down the rows, and XY tables for
the spectra.  The tests read the blocks back and check the numbers land in the
cells Prism will graph, including the ragged case where groups hold different
numbers of animals.
"""
import numpy as np

import fp_analysis_gui as G


FREQS = np.array([0.05, 0.1, 0.5, 1.0, 2.0, 3.0])

BANDS = [
    {'name': 'Infra-slow', 'fmin': 0.05, 'fmax': 0.5},
    {'name': 'Delta', 'fmin': 1.0, 'fmax': 4.0},
]


class _Stub:
    """A bare object carrying the handful of methods under test."""

    _coh_freq_axis_range = G.FPAnalysisGUI._coh_freq_axis_range
    _get_active_freq_bands = G.FPAnalysisGUI._get_active_freq_bands
    _band_mean_from_spectrum = G.FPAnalysisGUI._band_mean_from_spectrum
    # staticmethod on the real class; rebinding the plain function here
    # would make `self` its first argument.
    _prism_num = staticmethod(G.FPAnalysisGUI._prism_num)
    _prism_spectrum_grid = G.FPAnalysisGUI._prism_spectrum_grid
    _coherence_prism_text = G.FPAnalysisGUI._coherence_prism_text
    _bands_overlapping = G.FPAnalysisGUI._bands_overlapping

    def __init__(self, **params):
        self.conn_params = {
            'static_fmin': 0.05, 'static_fmax': 3.0,
            'freq_xmin': 0.0, 'freq_xmax': 3.0,
            'freq_bands': list(BANDS),
        }
        self.conn_params.update(params)


def _blocks(text):
    """Split the export into ``{block title: [lines]}`` on its blank lines."""
    out, title, rows = {}, None, []
    for line in text.split('\n'):
        if not line.strip():
            if title is not None:
                out.setdefault(title, []).extend(rows)
            title, rows = None, []
            continue
        if '\t' not in line and not line.startswith('#'):
            if title is not None:
                out.setdefault(title, []).extend(rows)
            title, rows = line, []
        elif not line.startswith('#'):
            rows.append(line)
    if title is not None:
        out.setdefault(title, []).extend(rows)
    return out


# --------------------------------------------------------------------------
#  Frequency axis
# --------------------------------------------------------------------------

def test_axis_range_defaults_to_the_data_when_min_is_zero():
    """0 Hz is the shipped default and is unplottable on a log axis."""
    stub = _Stub(freq_xmin=0.0, freq_xmax=3.0)
    assert stub._coh_freq_axis_range(FREQS) == (0.05, 3.0)


def test_axis_range_narrows_the_panel_to_the_requested_window():
    stub = _Stub(freq_xmin=0.2, freq_xmax=1.5)
    lo, hi = stub._coh_freq_axis_range(FREQS)
    assert (lo, hi) == (0.2, 1.5)


def test_axis_range_never_stretches_past_the_computed_frequencies():
    """An axis wider than the analysis would only add blank space."""
    stub = _Stub(freq_xmin=0.01, freq_xmax=50.0)
    assert stub._coh_freq_axis_range(FREQS) == (0.05, 3.0)


def test_axis_range_falls_back_when_the_window_misses_the_data():
    stub = _Stub(freq_xmin=20.0, freq_xmax=40.0)
    assert stub._coh_freq_axis_range(FREQS) == (0.05, 3.0)


def test_axis_range_survives_a_non_numeric_setting():
    stub = _Stub(freq_xmin='', freq_xmax='abc')
    assert stub._coh_freq_axis_range(FREQS) == (0.05, 3.0)


def test_bands_overlapping_drops_bands_outside_the_analysis_range():
    """A band the analysis never reached would plot as a bar of zero."""
    stub = _Stub()
    stub.conn_params['freq_bands'] = BANDS + [
        {'name': 'Theta', 'fmin': 4.0, 'fmax': 8.0}]
    names = [b['name'] for b in stub._bands_overlapping(FREQS)]
    assert names == ['Infra-slow', 'Delta']


# --------------------------------------------------------------------------
#  Prism blocks
# --------------------------------------------------------------------------

def _pre_post_series():
    """Two groups of unequal n, with flat spectra so band means are exact."""
    def rec(pre, post, n):
        return {'freqs': FREQS,
                'Baseline': np.full(FREQS.shape, pre),
                'Post': np.full(FREQS.shape, post),
                'n_bouts': n}
    return {
        'Fentanyl': {'F1': rec(0.20, 0.50, 7), 'F2': rec(0.30, 0.60, 5)},
        'Saline': {'S1': rec(0.25, 0.25, 9)},
    }


def _text():
    stub = _Stub()
    return stub._coherence_prism_text(
        _pre_post_series(), ['Baseline', 'Post'],
        {'title': 'test', 'behavior': 'Rearing'})


def test_grouped_block_puts_each_group_and_epoch_in_its_own_column():
    blocks = _blocks(_text())
    key = [k for k in blocks if k.startswith('Infra-slow')][0]
    header = blocks[key][0].split('\t')
    assert header == ['Replicate',
                      'Fentanyl Baseline', 'Fentanyl Post',
                      'Saline Baseline', 'Saline Post']


def test_grouped_block_rows_carry_the_band_means():
    blocks = _blocks(_text())
    key = [k for k in blocks if k.startswith('Infra-slow')][0]
    row1 = blocks[key][1].split('\t')
    assert row1[:3] == ['1', '0.2', '0.5']


def test_grouped_block_pads_the_shorter_group_with_blanks():
    """Prism reads an unequal-n grouped table; it must not read a 0."""
    blocks = _blocks(_text())
    key = [k for k in blocks if k.startswith('Infra-slow')][0]
    row2 = blocks[key][2].split('\t')
    assert row2 == ['2', '0.3', '0.6', '', '']


def test_change_block_reports_post_minus_baseline_per_subject():
    blocks = _blocks(_text())
    key = [k for k in blocks if k.startswith('Change: Infra-slow')][0]
    rows = blocks[key]
    assert rows[0].split('\t') == ['Replicate', 'Fentanyl', 'Saline']
    assert rows[1].split('\t') == ['1', '0.3', '0']


def test_per_subject_row_block_carries_bouts_and_delta():
    blocks = _blocks(_text())
    rows = blocks['BAND MEANS — one row per subject']
    header = rows[0].split('\t')
    assert header[:3] == ['Group', 'Subject', 'N_Bouts']
    assert 'Change_Infra-slow' in header
    f1 = rows[1].split('\t')
    assert f1[:3] == ['Fentanyl', 'F1', '7']
    assert f1[header.index('Change_Infra-slow')] == '0.3'


def test_spectra_mean_sem_n_block_is_an_xy_table():
    blocks = _blocks(_text())
    rows = blocks['SPECTRA — Prism XY, Mean/SEM/N per column']
    header = rows[0].split('\t')
    assert header[0] == 'Frequency_Hz'
    assert header[1:4] == ['Fentanyl Baseline Mean',
                           'Fentanyl Baseline SEM',
                           'Fentanyl Baseline N']
    first = rows[1].split('\t')
    assert first[0] == '0.05'
    # Two subjects at 0.20 and 0.30 -> mean 0.25, SEM 0.05, n 2
    assert first[1] == '0.25'
    assert first[2] == '0.05'
    assert first[3] == '2'
    # Saline holds one animal: a 0 there would draw an error bar that was
    # never measured.
    assert first[header.index('Saline Baseline SEM')] == ''
    assert first[header.index('Saline Baseline N')] == '1'


def test_spectra_replicate_block_has_one_column_per_subject_and_epoch():
    blocks = _blocks(_text())
    rows = blocks['SPECTRA — Prism XY, one column per subject']
    header = rows[0].split('\t')
    assert header[0] == 'Frequency_Hz'
    # 3 subjects x 2 epochs
    assert len(header) == 1 + 6
    assert 'Fentanyl Baseline F1' in header
    assert 'Saline Post S1' in header
    assert len(rows) == 1 + FREQS.size


def test_single_epoch_export_drops_the_epoch_from_column_names():
    """The whole-session store has one spectrum per subject, not two."""
    stub = _Stub()
    series = {
        'Fentanyl': {'F1': {'freqs': FREQS,
                            'Coherence': np.full(FREQS.shape, 0.4),
                            'n_bouts': None}},
        'Saline': {'S1': {'freqs': FREQS,
                          'Coherence': np.full(FREQS.shape, 0.6),
                          'n_bouts': None}},
    }
    text = stub._coherence_prism_text(series, ['Coherence'], {'title': 'ws'})
    blocks = _blocks(text)
    key = [k for k in blocks if k.startswith('Infra-slow')][0]
    assert blocks[key][0].split('\t') == ['Replicate', 'Fentanyl', 'Saline']
    assert 'BAND CHANGE (post − baseline) — Prism Grouped layout' not in text


def test_subjects_on_a_different_frequency_grid_are_interpolated_not_dropped():
    """A shorter epoch yields fewer Welch bins; the column must still fill."""
    stub = _Stub()
    odd = np.array([0.05, 1.0, 3.0])
    series = {
        'A': {'A1': {'freqs': FREQS,
                     'Coherence': np.linspace(0.1, 0.6, FREQS.size),
                     'n_bouts': None},
              'A2': {'freqs': odd,
                     'Coherence': np.array([0.1, 0.35, 0.6]),
                     'n_bouts': None}},
    }
    text = stub._coherence_prism_text(series, ['Coherence'], {'title': 'x'})
    rows = _blocks(text)['SPECTRA — Prism XY, one column per subject']
    assert rows[0].split('\t')[1:] == ['A A1', 'A A2']
    for line in rows[1:]:
        cells = line.split('\t')
        assert cells[1] != '' and cells[2] != ''


def test_missing_values_export_as_blank_cells():
    """'nan' in a Prism cell is read as text and poisons the column."""
    stub = _Stub()
    assert stub._prism_num(np.nan) == ''
    assert stub._prism_num(None) == ''
    assert stub._prism_num(0.0) == '0'
    assert stub._prism_num(0.5) == '0.5'
