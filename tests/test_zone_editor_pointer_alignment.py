"""A click in the zone editor selects the zone actually drawn under the pointer.

TRACY runs DPI-aware (main() calls _enable_dpi_awareness), so Tk's pixels are
physical device pixels.  matplotlib's Tk backend assumes otherwise: on <Map> it
reads `tk scaling`, decides the display is 1.5x at 150% Windows scaling, and
re-renders the figure 1.5x larger than its widget.  The arena was then cropped
at the right and bottom edges while every click was still multiplied by that
1.5, so the pointer grabbed a boundary nowhere near the one drawn beneath it --
measured on a 150% display, clicking the middle of the OFT arena selected
edge_left, and clicking a corner zone selected nothing at all.

Headless geometry tests cannot see this: they feed display coordinates straight
back in, so both sides of the mismatch cancel.  These drive a real Tk window.

Note that pytest's session Tk root is created without DPI awareness, so the
click assertions here would pass even on the old code; what pins the fix in any
process is the ratio and the figure-size-versus-widget-size invariant.
"""
import tkinter as tk

import pytest

import fp_analysis_gui as G

matplotlib = pytest.importorskip('matplotlib')
pytest.importorskip('matplotlib.pyplot')


@pytest.fixture(scope='module')
def editor(tk_root):
    """A real ZoneEditor in a real, mapped Tk window showing the OFT arena."""
    matplotlib.pyplot.close('all')      # backend switch would auto-close them
    matplotlib.use('TkAgg')

    app = object.__new__(G.FPAnalysisGUI)
    app.zone_templates = G.FPAnalysisGUI._create_zone_templates(app)
    app.params = {'maze_width_cm': 51.0, 'maze_type': 'OFT',
                  'zone_editor_sticky_edges': True}
    app.zones = {k: dict(v) for k, v in app.zone_templates['OFT']['zones'].items()}
    app.make_scrollable = lambda parent, **kwargs: tk.Frame(parent)
    app._refresh_entry_threshold_controls = lambda: None
    app.processed_data = {}

    window = tk.Toplevel(tk_root)
    window.geometry('900x760+40+40')
    instance = G.ZoneEditor(window, app)
    window.deiconify()
    for _ in range(20):
        tk_root.update()

    yield instance

    try:
        window.destroy()
    except Exception:
        pass


def _figure_pixels(editor):
    width_in, height_in = editor.fig.get_size_inches()
    dpi = editor.fig.get_dpi()
    return int(round(width_in * dpi)), int(round(height_in * dpi))


def test_canvas_renders_one_figure_pixel_per_tk_pixel(editor):
    """device_pixel_ratio > 1 is the whole bug: it scales clicks but not the
    widget the figure is drawn into."""
    assert editor.canvas.device_pixel_ratio == 1


def test_figure_is_rendered_at_the_size_of_its_widget(editor):
    widget = editor.canvas.get_tk_widget()
    assert (widget.winfo_width(), widget.winfo_height()) == _figure_pixels(editor)


def test_click_selects_the_zone_drawn_under_the_pointer(editor):
    widget = editor.canvas.get_tk_widget()
    ratio = editor.canvas.device_pixel_ratio
    figure_height = _figure_pixels(editor)[1]

    targets = [(6.0, 6.0, 'corner_bottom_left'),
               (25.5, 25.5, 'center'),
               (45.0, 25.5, 'edge_right'),
               (25.5, 45.0, 'edge_top')]

    for data_x, data_y, expected in targets:
        px, py = editor.ax.transData.transform((data_x, data_y))
        widget.event_generate('<Button-1>',
                              x=int(px / ratio),
                              y=int((figure_height - py) / ratio))
        widget.update()
        assert editor.selected_zone == expected, (
            'click at data (%.1f, %.1f) selected %r'
            % (data_x, data_y, editor.selected_zone))
        widget.event_generate('<ButtonRelease-1>')
        widget.update()
