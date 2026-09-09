"""The scroll region has to follow content that grows after the first layout.

Two tabs failed the same way on a large project: the Exclusions list gained a
row per subject and the Processing tab's advanced notebook swapped in a taller
page, and in both cases the canvas window item kept the height it was pinned to
when the tab was first laid out. `bbox('all')` therefore still described the old
content, so the scrollbar offered no way down and the overflow -- 445 px of the
Animals & Sessions page, 800 px of an 86-subject exclusion list -- was simply
unreachable.

These assert on the scroll region against the content's own requested height,
because a scrollbar that merely *exists* is not evidence that it reaches.
"""

import importlib.util
import os
import sys

import pytest

tk = pytest.importorskip('tkinter')
_ttk = pytest.importorskip('tkinter.ttk')

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_viewport', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI


class _Viewport:
    """The Tk-dependent slice of the viewport helper, without the whole GUI."""

    bind_scroll_viewport = _G.bind_scroll_viewport


@pytest.fixture
def root(tk_root):
    win = tk.Toplevel(tk_root)
    win.geometry('400x300')
    win.deiconify()
    win.update()
    yield win
    try:
        win.destroy()
    except tk.TclError:
        pass


def _viewport(root):
    canvas = tk.Canvas(root, highlightthickness=0)
    inner = _ttk.Frame(canvas)
    win_id = canvas.create_window((0, 0), window=inner, anchor='nw')
    canvas.pack(fill='both', expand=True)
    sync = _Viewport().bind_scroll_viewport(canvas, inner, win_id)
    root.update()
    return canvas, inner, sync


def _extent(canvas):
    return float(str(canvas.cget('scrollregion')).split()[3])


def test_scroll_region_follows_content_that_grew(root):
    canvas, inner, sync = _viewport(root)
    for _ in range(4):
        _ttk.Label(inner, text='row').pack()
    root.update()
    settled = _extent(canvas)

    for _ in range(60):
        _ttk.Label(inner, text='row').pack()
    root.update()
    sync()
    root.update()

    assert _extent(canvas) > settled
    assert _extent(canvas) >= inner.winfo_reqheight()


def test_grown_content_is_reachable_by_scrolling(root):
    """The bottom row can actually be brought into view."""
    canvas, inner, sync = _viewport(root)
    root.update()
    rows = [_ttk.Label(inner, text=f'row {i}') for i in range(80)]
    for row in rows:
        row.pack()
    root.update()
    sync()
    root.update()

    canvas.yview_moveto(1.0)
    root.update()
    bottom_of_view = canvas.canvasy(0) + canvas.winfo_height()
    assert bottom_of_view >= inner.winfo_reqheight() - 1


def test_short_content_still_fills_the_viewport(root):
    """The pin is what lets an `expand=True` child use the whole zone."""
    canvas, inner, sync = _viewport(root)
    _ttk.Label(inner, text='only row').pack()
    root.update()
    sync()
    root.update()

    item = canvas.find_withtag('all')[0]
    assert int(canvas.itemcget(item, 'height')) >= canvas.winfo_height()
