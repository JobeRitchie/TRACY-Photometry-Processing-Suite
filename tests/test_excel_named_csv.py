"""Raw files that are Excel workbooks under a .csv name.

Opening a Bonsai CSV in Excel and saving it as a workbook keeps the .csv name,
and pandas' CSV parser then dies on the zip bytes with a UnicodeDecodeError.
The readers sniff the file signature instead of trusting the extension.
"""
import numpy as np
import pandas as pd
import pytest

import fp_analysis_gui as G
from test_session_bounds import _app, _raw_csv

pytest.importorskip('openpyxl')


def _as_workbook_named_csv(csv_path, tmp_path, name):
    path = tmp_path / f"{name}.csv"
    pd.read_csv(csv_path).to_excel(path, index=False, engine='openpyxl')
    return str(path)


def test_signature_detection(tmp_path):
    csv_path = _raw_csv(tmp_path, n_rows=200)
    book = _as_workbook_named_csv(csv_path, tmp_path, 'book')
    assert G.excel_format_of(csv_path) is None
    assert G.excel_format_of(book) == 'xlsx'
    assert G.excel_format_of(str(tmp_path / 'missing.csv')) is None


def test_read_table_matches_csv(tmp_path):
    csv_path = _raw_csv(tmp_path, n_rows=200)
    book = _as_workbook_named_csv(csv_path, tmp_path, 'book')
    # Excel hands back whole-number floats as ints and may round the last bit
    pd.testing.assert_frame_equal(G.read_table(book), pd.read_csv(csv_path),
                                  check_dtype=False, rtol=1e-14)
    first = G.read_table(book, nrows=1, header=None)
    assert first.shape[0] == 1 and first.iloc[0, 0] == 'FrameCounter'


def test_workbook_processes_identically_to_csv(tmp_path):
    csv_path = _raw_csv(tmp_path, n_rows=8000)
    book = _as_workbook_named_csv(csv_path, tmp_path, 'book_FPData')
    ref = _app(enabled=False).process_fp_data('s', csv_path)
    got = _app(enabled=False).process_fp_data('s', book)
    arrays = [k for k in ref if isinstance(ref[k], np.ndarray)]
    assert arrays
    # The fit amplifies Excel's last-bit rounding to ~1e-4 absolute, so this
    # checks the file was read and aligned the same, not bit-identity.
    for k in arrays:
        assert got[k].shape == ref[k].shape, k
        np.testing.assert_allclose(got[k], ref[k], rtol=0, atol=1e-3,
                                   equal_nan=True, err_msg=k)
