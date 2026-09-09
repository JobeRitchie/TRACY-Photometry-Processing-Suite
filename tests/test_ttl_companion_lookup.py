"""Folder runs must find the TTL companion even when its index differs.

Whole-folder processing derived the TTL name by copying the FPData file's own
trailing suffix, so BG21FPData.csv looked for BG21TTL.csv and silently reported
"no TTL/DigitalIOs file" for BG21TTL0.csv -- the same pair that worked in
single-file mode, where the user browses to the TTL by hand.
"""
import os
import types

import pytest

import fp_analysis_gui as G


def _gui(fpdata_pattern='fpdata', fpdata_suffix='0.csv'):
    gui = types.SimpleNamespace()
    gui.params = {'fpdata_pattern': fpdata_pattern, 'fpdata_suffix': fpdata_suffix}
    gui._find_ttl_companion = types.MethodType(
        G.FPAnalysisGUI._find_ttl_companion, gui)
    return gui


def _folder(tmp_path, names):
    for name in names:
        (tmp_path / name).write_text('')
    return str(tmp_path)


@pytest.mark.parametrize('files, fpdata, subject, expected, kind', [
    # The reported failure: no index on FPData, index 0 on the TTL.
    (['BG21FPData.csv', 'BG21TTL0.csv'], 'BG21FPData.csv', 'BG21',
     'BG21TTL0.csv', 'TTL'),
    # Legacy naming, where both carry the same suffix, still resolves.
    (['DG27FPDATA0.csv', 'DG27TTL0.csv'], 'DG27FPDATA0.csv', 'DG27',
     'DG27TTL0.csv', 'TTL'),
    # Separator style.
    (['M1_FPData.csv', 'M1_TTL.csv'], 'M1_FPData.csv', 'M1',
     'M1_TTL.csv', 'TTL'),
    # DigitalIOs is found when no TTL exists.
    (['BG21FPData.csv', 'BG21DigitalIOs0.csv'], 'BG21FPData.csv', 'BG21',
     'BG21DigitalIOs0.csv', 'DigitalIOs'),
])
def test_companion_found(tmp_path, files, fpdata, subject, expected, kind):
    folder = _folder(tmp_path, files)
    path, found_kind = _gui()._find_ttl_companion(folder, fpdata, subject)
    assert path is not None and os.path.basename(path) == expected
    assert found_kind == kind


def test_prefix_match_is_not_greedy(tmp_path):
    """BG2 must not adopt BG21's TTL file."""
    folder = _folder(tmp_path, ['BG2FPData.csv', 'BG21FPData.csv', 'BG21TTL0.csv'])
    path, _ = _gui()._find_ttl_companion(folder, 'BG2FPData.csv', 'BG2')
    assert path is None

    path, _ = _gui()._find_ttl_companion(folder, 'BG21FPData.csv', 'BG21')
    assert os.path.basename(path) == 'BG21TTL0.csv'


def test_no_companion_reports_none(tmp_path):
    folder = _folder(tmp_path, ['BG21FPData.csv'])
    assert _gui()._find_ttl_companion(folder, 'BG21FPData.csv', 'BG21') == (None, None)


def test_ttl_preferred_over_digitalios(tmp_path):
    folder = _folder(tmp_path, ['BG21FPData.csv', 'BG21TTL0.csv', 'BG21DigitalIOs0.csv'])
    path, kind = _gui()._find_ttl_companion(folder, 'BG21FPData.csv', 'BG21')
    assert os.path.basename(path) == 'BG21TTL0.csv' and kind == 'TTL'
