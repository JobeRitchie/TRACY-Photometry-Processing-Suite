"""
Tests for the Output zone's scroll position across a change of plot.

The multi-subject Signal Integrity table can run past 20 inches, so the zone is
genuinely scrolled a long way down while it is read.  Tk keeps a viewport's
pixel offset when the content it scrolls changes size, and on this path it also
never fires the <Configure> that would refresh the scroll region -- the canvas
leaves its window item at the old height, so the inner frame is never
reconfigured even though its requested size collapsed.  The next, shorter plot
was therefore drawn into an area a thousand pixels above the view: a correct
plot, an empty panel, no error, indistinguishable from Generate doing nothing.

These assert on real geometry -- canvasy(0) against the content's own height --
because the stale scroll region is half the bug and cannot be the yardstick for
the other half.
"""

import importlib.util
import os
import sys

import pytest

tk = pytest.importorskip('tkinter')
_ttk = pytest.importorskip('tkinter.ttk')

_GUI_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_scroll', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI


class _Scroller:
    """The Tk-dependent slice of the scrolling zone, without the whole GUI."""

    make_scrollable = _G.make_scrollable
    _scroll_plot_zone_to_top = _G._scroll_plot_zone_to_top

    colors = {'bg_light': '#ffffff'}


@pytest.fixture(scope='module')
def tk_root():
    """One interpreter for the module.

    Creating and tearing down a Tk() per test fails intermittently on Windows,
    and the resulting TclError is indistinguishable from having no display --
    so it would silently skip real failures rather than report them.
    """
    try:
        r = tk.Tk()
    except tk.TclError:                      # genuinely no display
        pytest.skip('no Tk display available')
    r.withdraw()
    yield r
    try:
        r.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def root(tk_root):
    win = tk.Toplevel(tk_root)
    win.geometry('600x400')
    win.deiconify()
    win.update()
    yield win
    try:
        win.destroy()
    except tk.TclError:
        pass


@pytest.fixture
def zone(root):
    """A scrolling Output zone with the frame plots are embedded into."""
    app = _Scroller()
    inner = app.make_scrollable(root)
    fig_frame = _ttk.Frame(inner)
    fig_frame.pack(fill='both', expand=True)
    root.update()
    return app, root, inner, fig_frame


def _draw(root, fig_frame, height):
    """Stand in for _embed_plot_canvas: replace the plot with one `height` tall.

    Only the two lines that matter here are reproduced -- swapping the content
    and resetting the zone -- so the test needs no matplotlib backend.
    """
    for w in fig_frame.winfo_children():
        w.destroy()
    plot = _ttk.Frame(fig_frame, height=height, width=200)
    plot.pack_propagate(False)
    plot.pack()
    root.update()
    _Scroller._scroll_plot_zone_to_top(_Scroller(), fig_frame)
    root.update()
    return plot


def _top_of_view(inner):
    """Which y of the content sits at the top edge of the viewport."""
    return inner.master.canvasy(0)


def _extent(inner):
    return float(inner.master.cget('scrollregion').split()[3])


# ---------------------------------------------------------------------------
# The reported failure
# ---------------------------------------------------------------------------

def test_a_short_plot_after_a_tall_one_is_visible(zone):
    """The bug: 400px of viewport looking at y=1600 of a 300px plot."""
    app, root, inner, fig_frame = zone

    _draw(root, fig_frame, 2000)             # the tall integrity table
    inner.master.yview_moveto(1.0)           # the user scrolls down to read it
    root.update()
    assert _top_of_view(inner) > 1000, 'precondition: scrolled well down'

    _draw(root, fig_frame, 300)              # a normal plot replaces it
    assert _top_of_view(inner) < 300, 'the viewport is parked past the plot'


def test_the_scroll_region_follows_the_plot_down(zone):
    """Left at the old extent, the scrollbar also offers to scroll into
    emptiness -- and the thumb misreports how much plot there is."""
    app, root, inner, fig_frame = zone

    _draw(root, fig_frame, 2000)
    tall = _extent(inner)
    _draw(root, fig_frame, 300)
    assert _extent(inner) < tall / 2


def test_a_fresh_plot_starts_at_its_top(zone):
    """A tall plot replacing a tall plot needs no clamping, but should still be
    looked at from the top rather than from wherever the last one was left."""
    app, root, inner, fig_frame = zone

    _draw(root, fig_frame, 4000)
    inner.master.yview_moveto(0.6)
    root.update()
    assert _top_of_view(inner) > 0

    _draw(root, fig_frame, 4000)
    assert _top_of_view(inner) == 0


# ---------------------------------------------------------------------------
# Finding the viewport
# ---------------------------------------------------------------------------

def test_the_viewport_is_found_through_intervening_frames(zone):
    """_embed_plot_canvas is handed a frame nested some way inside the zone."""
    app, root, inner, fig_frame = zone
    deep = fig_frame
    for _ in range(3):
        deep = _ttk.Frame(deep)
        deep.pack()

    _draw(root, fig_frame, 4000)
    inner.master.yview_moveto(0.6)
    root.update()
    app._scroll_plot_zone_to_top(deep)
    assert _top_of_view(inner) == 0


def test_a_widget_in_no_scrolling_zone_is_harmless(root):
    """Several tabs embed plots into plain frames; those must not raise."""
    app = _Scroller()
    plain = _ttk.Frame(root)
    plain.pack()
    app._scroll_plot_zone_to_top(plain)      # not raising is the assertion


def test_the_plots_own_canvas_is_never_mistaken_for_the_viewport(zone):
    """The walk is upwards only, so a FigureCanvasTkAgg -- itself a tk.Canvas,
    but always a child -- can never be scrolled instead of the zone."""
    app, root, inner, fig_frame = zone
    fig_canvas = tk.Canvas(fig_frame, width=200, height=2000)
    fig_canvas.pack()
    root.update()

    inner.master.yview_moveto(0.5)
    root.update()
    app._scroll_plot_zone_to_top(fig_frame)
    assert _top_of_view(inner) == 0
