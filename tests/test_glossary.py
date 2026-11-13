"""The glossary covers every graph type the GUI offers, and every "?" lands on
a real entry.

A graph added to a selector without a glossary entry would leave its "?"
opening the tab explainer instead of the graph, with nothing to say so; these
tests are what notices.
"""
import re
from pathlib import Path

import pytest

import fp_analysis_gui as g
import tracy_glossary as G

GUI = g.FPAnalysisGUI
SOURCE = Path(g.__file__).read_text(encoding='utf-8')


def _menu_labels(menu_var):
    """Labels of the add_command entries on one tk.Menu in the source."""
    return re.findall(menu_var + r'\.add_command\(label="([^"]+)"', SOURCE)


@pytest.mark.parametrize('label', sorted(g.VIZ_OPTIONS_BY_PLOT))
def test_every_visualization_plot_type_has_an_entry(label):
    assert G.entry(f'viz:{label}') is not None


@pytest.mark.parametrize('label', GUI.KIN_ANALYSES)
def test_every_kinematics_analysis_has_an_entry(label):
    assert G.entry(f'kin:{label}') is not None


@pytest.mark.parametrize('label', GUI.DEC_MODES)
def test_every_decision_output_has_an_entry(label):
    assert G.entry(f'dec:{label}') is not None


@pytest.mark.parametrize('key', [k for k, _t in GUI.SPATIAL_PLOTS])
def test_every_spatial_transient_plot_has_an_entry(key):
    assert G.entry(f'spatial:{key}') is not None


def test_every_bout_analysis_plot_menu_item_has_an_entry():
    labels = [l for l in _menu_labels('plot_menu') if not l.endswith('…')]
    assert 'Bar Graphs (per metric)' in labels      # the regex still finds the menu
    for label in labels:
        assert G.entry(f'bout:{label}') is not None, label
    assert G.entry('bout:Time-Course Statistics (FLMM-style)') is not None


def test_coherence_graphs_and_pages_have_entries():
    for entry_id in list(GUI.COH_GRAPH_PAGE) + list(GUI.COH_PAGE_ENTRIES):
        assert G.entry(entry_id) is not None, entry_id
    # every plotting button on the Coherence tab is wired to one of them
    for fn in re.findall(r'self\.(visualize_\w+)\)\)', SOURCE):
        assert fn.startswith('visualize_')
    wired = set(re.findall(r"_g\('(coh:[^']+)'", SOURCE))
    assert wired == set(GUI.COH_GRAPH_PAGE)


def test_behavioral_summaries_have_entries():
    for name in ('Time in Each Zone', 'Entries into Each Zone',
                 'Time at Distance from Center', 'Single metric'):
        assert G.entry(f'behav:{name}') is not None
    assert GUI.TIME_AT_DISTANCE_PREFIX == '[Summary] Time at Distance from Center'


def test_every_tab_has_an_explainer_and_a_help_button():
    tabs = re.findall(r'_notebook\.add\(\w+, text="([^"]+)"\)', SOURCE)
    tabs = [t for t in tabs if t not in ('Editor', 'Scaled Preview',
                                         'Per-Subject Shifts', 'Bout Exclusions')]
    assert len(tabs) >= 16
    for tab in tabs:
        assert G.entry(f'tab:{tab}') is not None, tab
    # every control-column tab passes a help key
    calls = re.findall(r'self\.make_layout_zones\((.*?)\)\n', SOURCE, flags=re.S)
    assert len(calls) == 8
    assert all('help_key=' in c for c in calls)
    for tab in ('Welcome', 'Setup', 'Processing', 'Bout Frames Editor',
                'Factors', 'Exclusions'):
        assert f"place_tab_help(tab, '{tab}')" in SOURCE or \
               f"place_tab_help(tab_outer, '{tab}')" in SOURCE, tab


def test_entries_are_complete_and_linked():
    ids = [e.id for e in G.ENTRIES]
    assert len(ids) == len(set(ids))
    prefixes = {p for p, _t in G.SECTIONS}
    for e in G.ENTRIES:
        assert e.id.split(':', 1)[0] in prefixes, e.id
        assert e.term and e.where and e.shows, e.id
        if e.kind == 'graph':
            assert e.calc, f'{e.id} does not say how it is calculated'
        for other in e.see:
            assert G.entry(other) is not None, f'{e.id} links to missing {other}'
        for text in filter(None, (e.term, e.where, e.shows, e.calc, e.read)):
            assert '—' not in text, f'{e.id} uses an em dash'
            assert '  ' not in text, f'{e.id} has a wrapping artifact'
            assert not re.search(r'[a-z]- [a-z]', text), f'{e.id} has a split hyphen'


def test_search_ranks_the_named_graph_first():
    assert G.search('distal turn-back')[0].id == 'dec:Distal turn-back'
    assert G.search('Zone Averages')[0].id == 'viz:Zone Averages'
    assert G.search('FLMM')[0].id.startswith('bout:')
    assert {e.id for e in G.search('coherence')[:2]} == {'tab:Coherence', 'term:Coherence'}
    assert any(e.id == 'term:Bout metrics' for e in G.search('tau'))
    assert G.search('zzzz-no-such-word') == []
    assert len(G.search('')) == len(G.ENTRIES)
    assert G.search('   ') == G.ordered_entries()


def test_resolve_falls_back_for_unknown_selector_values():
    assert G.resolve('viz', 'Z-scored', 'tab:Visualization') == 'viz:Z-scored'
    # the Plot Type box is editable and starts on a value that is not a plot
    assert G.resolve('viz', 'corrected', 'tab:Visualization') == 'tab:Visualization'


def test_glossary_target_follows_the_selected_graph():
    class Var:
        def __init__(self, v): self.v = v
        def get(self): return self.v

    class Nb:
        def __init__(self, page): self.page = page
        def select(self): return 'p'
        def index(self, _p): return self.page

    app = GUI.__new__(GUI)
    app.plot_type_var = Var('Out/Back')
    app.kin_analysis_var = Var(GUI.KIN_ANALYSES[2])
    app.dec_mode_var = Var(GUI.DEC_MODES[1])
    assert app.glossary_target('visualization') == 'viz:Out/Back'
    assert app.glossary_target('kinematics') == f'kin:{GUI.KIN_ANALYSES[2]}'
    assert app.glossary_target('decision_probability') == f'dec:{GUI.DEC_MODES[1]}'
    assert app.glossary_target('signal_linkage') == 'link:Onset linkage'
    assert app.glossary_target('Factors') == 'tab:Factors'

    # tabs whose graphs are buttons: the explainer until something is drawn
    assert app.glossary_target('bout_analysis') == 'tab:Bout Analysis'
    ran = []
    app.remember_graph('bout_analysis', 'bout:Binned Progression',
                       lambda: ran.append(1))()
    assert ran == [1]
    assert app.glossary_target('bout_analysis') == 'bout:Binned Progression'

    # Coherence: a graph drawn on one page is not offered on another
    app.conn_mode_nb = Nb(0)
    assert app.glossary_target('coherence') == 'coh:Whole Session'
    app.remember_graph('coherence', 'coh:Power Spectra', lambda: None)()
    assert app.glossary_target('coherence') == 'coh:Power Spectra'
    app.conn_mode_nb = Nb(2)
    assert app.glossary_target('coherence') == 'coh:By Group'

    # Behavioral Data: nothing visualized yet, then each kind of view
    assert app.glossary_target('behavioral') == 'tab:Behavioral Data'
    app._behav_metric_var = Var('[Summary] Time at Distance from Center (X)')
    assert app.glossary_target('behavioral') == 'behav:Time at Distance from Center'
    app._behav_metric_var = Var('[Summary] Entries into Each Zone')
    assert app.glossary_target('behavioral') == 'behav:Entries into Each Zone'
    app._behav_metric_var = Var('Total_Distance_cm')
    assert app.glossary_target('behavioral') == 'behav:Single metric'
