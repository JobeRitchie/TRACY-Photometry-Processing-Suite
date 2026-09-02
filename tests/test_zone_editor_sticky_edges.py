"""Dragging a zone boundary keeps the arena tiled -- or doesn't, on request.

Zone templates are stored as a tiling: every boundary appears twice, once as
one zone's max and once as its neighbour's min.  The editor only ever moved the
edge under the pointer, so dragging the OFT centre opened a gap between it and
the surrounding edge zones -- position samples landing in that gap belong to no
zone at all.  Sticky edges move both halves of a shared boundary together; the
toggle turns that off for zones that are deliberately not a tiling.

These tests drive the geometry directly (no Tk, no canvas) against the real
templates from _create_zone_templates().
"""
import pytest

import fp_analysis_gui as G


MAZE_TOL = 1e-6


def _editor(template='OFT', sticky=True):
    """A ZoneEditor with real template zones and no Tk widgets."""
    app = object.__new__(G.FPAnalysisGUI)
    app.zone_templates = G.FPAnalysisGUI._create_zone_templates(app)
    template_data = app.zone_templates[template]

    editor = object.__new__(G.ZoneEditor)
    editor.main_app = app
    editor.zones = {k: dict(v) for k, v in template_data['zones'].items()}
    editor.maze_width = float(template_data['maze_width_cm'])
    editor.selected_zone = None
    editor.drag_corner = None
    editor.zone_patches = {}
    editor.zone_labels = {}
    editor._drag_edges = {}
    editor._drag_bounds0 = {}
    editor._drag_anchor = (0.0, 0.0)
    editor._sticky_edges = sticky
    return editor


def _arm_drag(editor, zone_name, handle):
    """Reproduce what on_mouse_press sets up, without the Tk half."""
    editor.selected_zone = zone_name
    editor.drag_corner = handle
    editor._drag_bounds0 = {n: dict(z) for n, z in editor.zones.items()}
    keys = ['x_min', 'x_max', 'y_min', 'y_max'] if handle == 'move' else [
        k for flag, k in (('left', 'x_min'), ('right', 'x_max'),
                          ('bottom', 'y_min'), ('top', 'y_max'))
        if flag in handle]
    editor._drag_edges = {}
    for key in keys:
        edges = [(zone_name, key)]
        if editor.sticky_edges_enabled():
            edges.extend(editor._sticky_partners(zone_name, key))
        editor._drag_edges[key] = edges
    return editor


def _boundary_partners(editor, zone_name, key):
    return sorted(editor._sticky_partners(zone_name, key))


def _tiling_gaps(zones, maze_width, samples=60):
    """Points inside the arena that belong to no zone (and how many)."""
    step = maze_width / samples
    missing = 0
    for i in range(samples):
        for j in range(samples):
            x = (i + 0.5) * step
            y = (j + 0.5) * step
            for zone in zones.values():
                if (zone['x_min'] <= x <= zone['x_max'] and
                        zone['y_min'] <= y <= zone['y_max']):
                    break
            else:
                missing += 1
    return missing


# -- the reported bug: OFT edges came apart -----------------------------


def test_oft_centre_drag_keeps_neighbours_attached():
    editor = _arm_drag(_editor('OFT'), 'center', 'left')
    editor._apply_edge_drag('x_min', 18.0)

    assert editor.zones['center']['x_min'] == pytest.approx(18.0)
    # The zones that shared x = 12.75 came with it.
    assert editor.zones['edge_left']['x_max'] == pytest.approx(18.0)
    assert editor.zones['edge_top']['x_min'] == pytest.approx(18.0)
    assert editor.zones['edge_bottom']['x_min'] == pytest.approx(18.0)
    assert editor.zones['corner_top_left']['x_max'] == pytest.approx(18.0)
    assert editor.zones['corner_bottom_left']['x_max'] == pytest.approx(18.0)
    # ...and the far side of the arena did not.
    assert editor.zones['edge_right']['x_min'] == pytest.approx(38.25)
    assert editor.zones['corner_top_right']['x_min'] == pytest.approx(38.25)


def test_oft_stays_a_tiling_after_a_drag():
    template = _editor('OFT')
    assert _tiling_gaps(template.zones, template.maze_width) == 0

    editor = _arm_drag(_editor('OFT'), 'center', 'left')
    editor._apply_edge_drag('x_min', 18.0)
    assert _tiling_gaps(editor.zones, editor.maze_width) == 0

    editor = _arm_drag(_editor('OFT'), 'center', 'top_right')
    editor._apply_edge_drag('x_max', 42.0)
    editor._apply_edge_drag('y_max', 44.0)
    assert _tiling_gaps(editor.zones, editor.maze_width) == 0


def test_sticky_off_moves_only_the_dragged_edge():
    editor = _arm_drag(_editor('OFT', sticky=False), 'center', 'left')
    editor._apply_edge_drag('x_min', 18.0)

    assert editor.zones['center']['x_min'] == pytest.approx(18.0)
    assert editor.zones['edge_left']['x_max'] == pytest.approx(12.75)
    assert editor.zones['edge_top']['x_min'] == pytest.approx(12.75)
    # Which is exactly the gap the toggle exists to allow.
    assert _tiling_gaps(editor.zones, editor.maze_width) > 0


def test_toggle_is_read_at_press_time_per_zone():
    on = _editor('OFT', sticky=True)
    off = _editor('OFT', sticky=False)
    assert _boundary_partners(on, 'center', 'x_min')
    assert on.sticky_edges_enabled() is True
    assert off.sticky_edges_enabled() is False


# -- partner selection --------------------------------------------------


def test_partners_must_adjoin_along_the_other_axis():
    editor = _editor('OFT')
    # corner_bottom_right's left edge also sits at x = 38.25, but it is on the
    # opposite side of the arena from corner_top_left's right edge at x = 12.75.
    partners = dict(_boundary_partners(editor, 'center', 'x_min'))
    assert 'corner_bottom_right' not in partners
    assert 'edge_right' not in partners
    assert partners['edge_left'] == 'x_max'
    assert partners['corner_top_left'] == 'x_max'


def test_epm_centre_drag_matches_the_old_hand_written_rules():
    """The generic rule reproduces what _update_connected_zones did for EPM."""
    editor = _arm_drag(_editor('EPM'), 'center', 'left')
    editor._apply_edge_drag('x_min', 32.0)

    assert editor.zones['closed_arm_left']['x_max'] == pytest.approx(32.0)
    # Arms flush with the centre on the perpendicular axis narrow with it, as
    # the EPM-specific code used to force.
    assert editor.zones['open_arm_up']['x_min'] == pytest.approx(32.0)
    assert editor.zones['open_arm_down']['x_min'] == pytest.approx(32.0)
    assert editor.zones['closed_arm_right']['x_min'] == pytest.approx(46.0)


def test_epm_complex_cascades_to_distal_zones():
    editor = _arm_drag(_editor('EPM_Complex'), 'open_proximal_up', 'top')
    editor._apply_edge_drag('y_max', 60.0)

    assert editor.zones['open_proximal_up']['y_max'] == pytest.approx(60.0)
    assert editor.zones['open_distal_up']['y_min'] == pytest.approx(60.0)
    assert editor.zones['open_distal_up']['y_max'] == pytest.approx(76.0)


# -- clamping -----------------------------------------------------------


def test_edge_cannot_be_dragged_through_its_own_zone():
    editor = _arm_drag(_editor('OFT'), 'center', 'left')
    editor._apply_edge_drag('x_min', 999.0)

    zone = editor.zones['center']
    assert zone['x_min'] < zone['x_max']
    assert zone['x_max'] - zone['x_min'] >= G.ZoneEditor.MIN_ZONE_SIZE - MAZE_TOL


def test_edge_cannot_collapse_a_stuck_neighbour():
    editor = _arm_drag(_editor('OFT'), 'center', 'left')
    editor._apply_edge_drag('x_min', -50.0)

    # edge_left would have been squashed to nothing; the drag stops first.
    left = editor.zones['edge_left']
    assert left['x_max'] - left['x_min'] >= G.ZoneEditor.MIN_ZONE_SIZE - MAZE_TOL
    assert editor.zones['center']['x_min'] >= G.ZoneEditor.MIN_ZONE_SIZE - MAZE_TOL


def test_edges_stay_inside_the_arena():
    editor = _arm_drag(_editor('OFT'), 'center', 'right')
    editor._apply_edge_drag('x_max', 900.0)
    assert editor.zones['center']['x_max'] <= editor.maze_width + MAZE_TOL


# -- moving a whole zone ------------------------------------------------


def test_moving_a_zone_keeps_its_size():
    editor = _arm_drag(_editor('OFT'), 'center', 'move')
    before = editor.zones['center'].copy()
    editor._apply_zone_move(3.0, -2.0)

    zone = editor.zones['center']
    assert zone['x_max'] - zone['x_min'] == pytest.approx(before['x_max'] - before['x_min'])
    assert zone['y_max'] - zone['y_min'] == pytest.approx(before['y_max'] - before['y_min'])
    assert zone['x_min'] == pytest.approx(before['x_min'] + 3.0)
    assert zone['y_min'] == pytest.approx(before['y_min'] - 2.0)
    # Neighbours were pushed, not left behind.
    assert editor.zones['edge_left']['x_max'] == pytest.approx(zone['x_min'])
    assert editor.zones['edge_right']['x_min'] == pytest.approx(zone['x_max'])


def test_moving_a_zone_cannot_leave_the_arena():
    editor = _arm_drag(_editor('OFT', sticky=False), 'corner_bottom_left', 'move')
    editor._apply_zone_move(-40.0, -40.0)

    zone = editor.zones['corner_bottom_left']
    assert zone['x_min'] >= -MAZE_TOL
    assert zone['y_min'] >= -MAZE_TOL


# -- hit testing --------------------------------------------------------


def _hit(editor, x, y, tol=1.0):
    editor._edge_tolerance = lambda: (tol, tol)
    return editor._hit_test(x, y)


def test_hit_test_prefers_an_edge_over_a_zone_body():
    editor = _editor('OFT')
    zone_name, handle = _hit(editor, 12.75, 25.0)
    assert handle in ('left', 'right')
    assert zone_name in ('center', 'edge_left')

    zone_name, handle = _hit(editor, 25.0, 25.0)
    assert (zone_name, handle) == ('center', 'move')


def test_hit_test_prefers_a_corner_over_an_edge():
    editor = _editor('OFT')
    zone_name, handle = _hit(editor, 12.75, 12.75)
    assert handle.count('_') == 1, handle


def test_hit_test_misses_outside_every_zone():
    editor = _editor('OFT')
    editor.zones = {'only': {'x_min': 10.0, 'x_max': 20.0,
                             'y_min': 10.0, 'y_max': 20.0}}
    assert _hit(editor, 40.0, 40.0) == (None, None)
