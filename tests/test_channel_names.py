"""User-defined channel names (G0 -> BLA) are a display layer.

The designation stays the key everywhere; the name is substituted only where
text is shown (matplotlib, channel pickers) or exported (pandas writers inside
an export function). Project files must keep the designation.
"""

import io

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import fp_analysis_gui as fp


@pytest.fixture(autouse=True)
def _no_names_left_behind():
    fp.set_channel_labels({})
    yield
    fp.set_channel_labels({})


# ---- the name map ---------------------------------------------------------

def test_nothing_is_renamed_until_a_channel_is_named():
    assert fp.relabel_channels('G0 Z-score') == 'G0 Z-score'
    assert fp.channel_label('G0') == 'G0'


def test_a_designation_is_replaced_only_where_it_stands_alone():
    fp.set_channel_labels({'G0': 'BLA', 'R0': 'OFC'})
    assert fp.relabel_channels('G0 470nm vs R0') == 'BLA 470nm vs OFC'
    assert fp.relabel_channels('G0_zscore') == 'BLA_zscore'
    assert fp.relabel_channels('lick_G0_bouts') == 'lick_BLA_bouts'
    # part of a longer word or number: someone else's text
    assert fp.relabel_channels('G01 NSFG0 G0x') == 'G01 NSFG0 G0x'
    # an unnamed channel is left as it is
    assert fp.relabel_channels('G1') == 'G1'


def test_a_slot_label_takes_the_name_of_the_channel_in_that_slot():
    fp.set_channel_labels({'R4': 'NAc'}, slots={0: 'G0', 1: 'R4'})
    assert fp.relabel_channels('Ch1 Heatmap') == 'NAc Heatmap'
    assert fp.relabel_channels('Ch0 Heatmap') == 'Ch0 Heatmap'
    assert fp.channel_label('Ch1') == 'NAc'
    assert fp.channel_key('NAc') == 'R4'
    assert fp.channel_key('NAc', slots=True) == 'Ch1'


def test_subject_and_group_names_holding_a_designation_survive():
    fp.set_channel_labels({'G0': 'BLA'}, protected=['G0_pilot', 'mouse7', 'G0'])
    assert fp.relabel_channels('G0_pilot: G0 trace') == 'G0_pilot: BLA trace'


def test_renaming_twice_changes_nothing_more():
    fp.set_channel_labels({'G0': 'BLA', 'G1': 'CeA'})
    once = fp.relabel_channels('G0 vs G1')
    assert fp.relabel_channels(once) == once == 'BLA vs CeA'


@pytest.mark.parametrize('label, ok', [
    ('BLA', True), ('dorsal BNST', True), ('NAc-core_2', True),
    ('G1', False),            # reads back as another channel
    ('ch0', False),
    ('a/b', False),           # goes into file names
    ('x' * 40, False),
])
def test_name_validation(label, ok):
    assert (fp.channel_label_error('G0', label) is None) is ok


def test_two_channels_cannot_share_a_name():
    assert fp.channel_label_error('G1', 'bla', others=['BLA']) is not None
    assert fp.clean_channel_labels({'G0': 'BLA', 'G1': 'BLA'}) == {'G0': 'BLA'}


def test_a_stored_map_is_cleaned():
    stored = {'G0': ' BLA ', 'G1': '', 'R0': 'R0', 'nonsense': 'x', 'R1': 'G0'}
    assert fp.clean_channel_labels(stored) == {'G0': 'BLA'}
    assert fp.clean_channel_labels(None) == {}


# ---- plots ----------------------------------------------------------------

def test_every_kind_of_plot_text_carries_the_name():
    fp.set_channel_labels({'G0': 'BLA', 'R0': 'OFC'}, slots={0: 'G0', 1: 'R0'})
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [0, 1], label='G0 470nm')
        ax.plot([0, 1], [1, 0], label='R0')
        ax.set_title('Ch0: lick (n=3 bouts)')
        ax.set_ylabel('G0 Z-score')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['G0', 'R0'])
        legend = ax.legend()
        fig.suptitle('G0 vs R0')
        fig.canvas.draw()
        assert ax.get_title() == 'BLA: lick (n=3 bouts)'
        assert ax.get_ylabel() == 'BLA Z-score'
        assert [t.get_text() for t in legend.get_texts()] == ['BLA 470nm', 'OFC']
        assert [t.get_text() for t in ax.get_xticklabels()] == ['BLA', 'OFC']
        assert fig._suptitle.get_text() == 'BLA vs OFC'
    finally:
        plt.close(fig)


# ---- exports vs project files ---------------------------------------------

def _frame():
    return pd.DataFrame({'Subject': ['m1', 'm2'], 'Channel': ['G0', 'R0'],
                         'G0_mean': [1.0, 2.0], 'Ch1_peak': [3.0, 4.0]})


@fp.exports_channel_labels
def _export(df, **kw):
    buf = io.StringIO()
    df.to_csv(buf, index=False, **kw)
    return buf.getvalue()


def _project_save(df):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_an_export_writes_names_in_headers_and_cells():
    fp.set_channel_labels({'G0': 'BLA', 'R0': 'OFC'}, slots={0: 'G0', 1: 'R0'})
    lines = _export(_frame()).splitlines()
    assert lines[0] == 'Subject,Channel,BLA_mean,OFC_peak'
    assert lines[1] == 'm1,BLA,1.0,3.0'
    assert lines[2] == 'm2,OFC,2.0,4.0'


def test_a_project_save_keeps_the_designations():
    fp.set_channel_labels({'G0': 'BLA', 'R0': 'OFC'}, slots={0: 'G0', 1: 'R0'})
    assert _project_save(_frame()).splitlines()[0] == 'Subject,Channel,G0_mean,Ch1_peak'


def test_a_project_save_triggered_from_inside_an_export_is_still_untouched():
    fp.set_channel_labels({'G0': 'BLA'})
    inner = fp.keeps_channel_designations(_project_save)

    @fp.exports_channel_labels
    def export_then_save(df):
        return _export(df), inner(df)

    exported, saved = export_then_save(_frame())
    assert 'BLA_mean' in exported
    assert 'G0_mean' in saved and 'BLA' not in saved


def test_the_exported_frame_is_a_copy():
    fp.set_channel_labels({'G0': 'BLA'})
    df = _frame()
    _export(df)
    assert list(df.columns) == ['Subject', 'Channel', 'G0_mean', 'Ch1_peak']
    assert list(df['Channel']) == ['G0', 'R0']


def test_excel_sheet_names_carry_the_name(tmp_path):
    fp.set_channel_labels({'G0': 'BLA'})
    path = tmp_path / 'out.xlsx'

    @fp.exports_channel_labels
    def write():
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            _frame().to_excel(writer, sheet_name='lick_G0', index=False)
    write()
    book = pd.read_excel(path, sheet_name=None)
    assert list(book) == ['lick_BLA']
    assert 'BLA_mean' in book['lick_BLA'].columns


def test_openpyxl_workbooks_are_renamed_in_place():
    import openpyxl
    fp.set_channel_labels({'G0': 'BLA'})
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'G0 summary'
    ws.append(['Channel', 'G0', 3])
    fp.relabel_workbook(wb)
    assert ws.title == 'BLA summary'
    assert [c.value for c in ws[1]] == ['Channel', 'BLA', 3]


# ---- channel pickers ------------------------------------------------------

def test_a_picker_shows_the_name_and_hands_back_the_designation(tk_root):
    fp.set_channel_labels({'G0': 'BLA'})
    var = fp.ChannelVar(tk_root, value='G0')
    combo = fp.ChannelCombobox(tk_root, textvariable=var, values=['G0', 'G1'], width=8)
    try:
        assert tk_root.globalgetvar(str(var)) == 'BLA'
        assert var.get() == 'G0' and combo.get() == 'G0'
        assert list(combo.cget('values')) == ['BLA', 'G1']
        combo['values'] = ['G1', 'G0']
        assert list(combo.cget('values')) == ['G1', 'BLA']
        var.set('G1')
        assert var.get() == 'G1'
    finally:
        combo.destroy()


def test_renaming_keeps_every_picker_on_its_channel(tk_root):
    var = fp.ChannelVar(tk_root, value='G0')
    combo = fp.ChannelCombobox(tk_root, textvariable=var, values=['G0', 'G1'], width=4)
    try:
        fp.retarget_channel_widgets(
            lambda: fp.set_channel_labels({'G0': 'basolateral amygdala'}))
        assert var.get() == 'G0'
        assert tk_root.globalgetvar(str(var)) == 'basolateral amygdala'
        assert list(combo.cget('values')) == ['basolateral amygdala', 'G1']
        assert int(combo.cget('width')) >= len('basolateral amygdala')
        fp.retarget_channel_widgets(lambda: fp.set_channel_labels({}))
        assert var.get() == 'G0' and tk_root.globalgetvar(str(var)) == 'G0'
    finally:
        combo.destroy()


def test_a_slot_picker_hands_back_the_slot_key(tk_root):
    fp.set_channel_labels({'G1': 'CeA'}, slots={0: 'G0', 1: 'G1'})
    var = fp.ChannelVar(tk_root, value='Ch1', slots=True)
    assert tk_root.globalgetvar(str(var)) == 'CeA'
    assert var.get() == 'Ch1'
