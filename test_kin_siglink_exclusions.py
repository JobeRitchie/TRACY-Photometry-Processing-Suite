"""Kinematics and Signal Linkage must support exclusions.

Neither tab had a toggle or consulted `self.exclusions` at all -- an animal you
had marked unusable still went into every kinematics summary and every linkage
permutation test, and there was no way to say otherwise short of deselecting it
by hand every run.

The two tabs need different keys, which is why one shared filter would be wrong:

* Kinematics is computed from the tracked *position*, not from a photometry
  channel, so it reads the Exclusions tab's Behavioral Data checkbox -- the
  same key the Behavioral Data tab uses.
* Signal Linkage analyses one named photometry channel, so it is a straight
  per-channel filter on the subject list.

Both toggles must also be in EXCLUSION_TOGGLES, which is what makes them
persist with the project.
"""
import pytest

import fp_analysis_gui as G


class _Var:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value

    def set(self, value):
        self._value = value


class _Listbox:
    def __init__(self, entries, selection=None):
        self._entries = list(entries)
        self._sel = tuple(range(len(self._entries))) if selection is None else tuple(selection)

    def curselection(self):
        return self._sel

    def get(self, i):
        return self._entries[i]


# --------------------------------------------------------------------------- #
#  persistence                                                                 #
# --------------------------------------------------------------------------- #

def _registered_toggles():
    """The real EXCLUSION_TOGGLES tuple, read from __init__.

    It is an *instance* attribute set in __init__, which needs Tk, so it cannot
    be read off the class. Stubbing it in the test instead would make the
    persistence assertions describe the stub rather than the app.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(G.FPAnalysisGUI.__init__))
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (isinstance(target, ast.Attribute)
                    and target.attr == 'EXCLUSION_TOGGLES'):
                return tuple(ast.literal_eval(node.value))
    raise AssertionError('EXCLUSION_TOGGLES is not assigned in __init__')


def test_both_toggles_are_registered_for_persistence():
    """Registration is what makes the checkboxes persist with the project."""
    registered = _registered_toggles()
    assert 'kin' in registered
    assert 'siglink' in registered


def test_registered_toggles_all_have_a_variable():
    """A name in the tuple with no matching var would silently never persist."""
    import inspect

    src = inspect.getsource(G.FPAnalysisGUI.__init__)
    for name in _registered_toggles():
        assert f'self.use_exclusions_{name} = tk.BooleanVar' in src, name


def test_toggle_state_round_trips_the_new_names():
    app = object.__new__(G.FPAnalysisGUI)
    app.EXCLUSION_TOGGLES = _registered_toggles()
    for name in app.EXCLUSION_TOGGLES:
        setattr(app, f'use_exclusions_{name}', _Var(name in ('kin', 'siglink')))

    state = app.exclusion_toggle_state()
    assert state['kin'] is True and state['siglink'] is True

    app.apply_exclusion_toggle_state({'kin': True, 'siglink': False})
    assert app.use_exclusions_kin.get() is True
    assert app.use_exclusions_siglink.get() is False

    app.reset_exclusion_toggles()
    assert app.use_exclusions_kin.get() is False
    assert app.use_exclusions_siglink.get() is False


# --------------------------------------------------------------------------- #
#  Kinematics                                                                  #
# --------------------------------------------------------------------------- #

def _kin_app(exclusions, use_exclusions, mode='Subject'):
    app = object.__new__(G.FPAnalysisGUI)
    subjects = ['S1', 'S2', 'S3']
    app.processed_data = {
        s: {'has_position': True, 'group': 'Fentanyl'} for s in subjects}
    app.exclusions = {k: list(v) for k, v in exclusions.items()}
    app.use_exclusions_kin = _Var(use_exclusions)
    app.kin_by_var = _Var(mode)
    app.kin_subject_listbox = _Listbox(subjects)
    app.kin_group_listbox = _Listbox(['Fentanyl'])
    app.selected_series = lambda tab, lb: ['Fentanyl']
    app._series_members = lambda g: list(subjects)
    app.log_message = lambda *a, **k: None
    return app


def test_kinematics_excludes_on_the_behavior_key():
    app = _kin_app({'S2': ['Behavior']}, use_exclusions=True)
    mode, pairs = app._kin_selected_subjects()
    assert [s for s, _ in pairs] == ['S1', 'S3']


def test_kinematics_ignores_a_photometry_channel_exclusion():
    """A bad green fibre says nothing about whether the tracking is usable."""
    app = _kin_app({'S2': ['G1']}, use_exclusions=True)
    mode, pairs = app._kin_selected_subjects()
    assert [s for s, _ in pairs] == ['S1', 'S2', 'S3']


def test_kinematics_toggle_off_keeps_everyone():
    app = _kin_app({'S2': ['Behavior']}, use_exclusions=False)
    mode, pairs = app._kin_selected_subjects()
    assert [s for s, _ in pairs] == ['S1', 'S2', 'S3']


def test_kinematics_group_mode_filters_too():
    app = _kin_app({'S2': ['Behavior']}, use_exclusions=True, mode='Group')
    mode, pairs = app._kin_selected_subjects()
    assert mode == 'Group'
    assert [s for s, _ in pairs] == ['S1', 'S3']


def test_kinematics_group_mode_toggle_off():
    app = _kin_app({'S2': ['Behavior']}, use_exclusions=False, mode='Group')
    mode, pairs = app._kin_selected_subjects()
    assert [s for s, _ in pairs] == ['S1', 'S2', 'S3']


def test_kinematics_still_drops_subjects_without_position():
    """The exclusion filter must not replace the has_position gate."""
    app = _kin_app({}, use_exclusions=True, mode='Group')
    app.processed_data['S3']['has_position'] = False
    mode, pairs = app._kin_selected_subjects()
    assert [s for s, _ in pairs] == ['S1', 'S2']


# --------------------------------------------------------------------------- #
#  Signal Linkage                                                              #
# --------------------------------------------------------------------------- #

class _ProgressQueue:
    """Swallows the worker's progress messages."""

    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _siglink_app(exclusions, use_exclusions, channel='G0'):
    app = object.__new__(G.FPAnalysisGUI)
    subjects = ['S1', 'S2', 'S3']
    app.processed_data = {s: {} for s in subjects}
    app.exclusions = {k: list(v) for k, v in exclusions.items()}
    # The pool now passes through the facet column before the worker sees it,
    # so the shell needs the factor model the column is derived from. With no
    # facet_controls registered the column reports nothing, which is the inert
    # "one pooled series" this tab defaults to.
    app.params = {
        'subject_factors': {},
        'factor_definitions': [],
        'factor_level_order': {},
        'animal_overrides': {},
        'session_pattern': G.DEFAULT_SESSION_PATTERN,
        'series_factor': 'Group',
        'preboutframes': 150,
        'postboutframes': 150,
    }
    app.root = None
    app._active_facet = None
    app.use_exclusions_siglink = _Var(use_exclusions)
    app.sig_link_subject_listbox = _Listbox(subjects)
    app.sig_link_behavior_listbox = _Listbox(['Sniff'])
    app.sig_link_channel_var = _Var(channel)
    app.sig_link_sg_win_var = _Var('11')
    app.sig_link_sg_poly_var = _Var('3')
    app.sig_link_search_var = _Var('1.0')
    app.sig_link_base_var = _Var('1.0')
    app.sig_link_resp_var = _Var('1.0')
    app.sig_link_relia_var = _Var('1.5')
    app.sig_link_nperm_var = _Var('1000')
    app.sig_link_status_var = _Var('')
    app.get_fps = lambda *a, **k: 30.0
    app.log_message = lambda *a, **k: None

    app.captured = {}
    app.warned = []

    def _run_with_progress(title, worker_fn, on_done_fn=None, **k):
        app.captured['worker'] = worker_fn

    app._run_with_progress = _run_with_progress

    def _compute(q, subs, channel, behs, cfg):
        app.captured['subs'] = list(subs)
        return {'results': {}, 'order': [], 'subjects': list(subs),
                'skipped': []}

    app._compute_signal_linkage = _compute
    app._render_signal_linkage_results = lambda *a, **k: None
    return app


def _run_siglink(app, monkeypatch):
    monkeypatch.setattr(G.messagebox, 'showwarning',
                        lambda *a, **k: app.warned.append(a))
    app.run_signal_linkage_analysis()
    if 'worker' in app.captured:
        # A real worker always gets a progress queue; the per-series pass
        # reports which series it is on through it.
        app.captured['worker'](_ProgressQueue())
    return app.captured.get('subs')


def test_signal_linkage_filters_on_the_analysed_channel(monkeypatch):
    app = _siglink_app({'S2': ['G0']}, use_exclusions=True, channel='G0')
    assert _run_siglink(app, monkeypatch) == ['S1', 'S3']


def test_signal_linkage_ignores_an_exclusion_on_another_channel(monkeypatch):
    app = _siglink_app({'S2': ['G1']}, use_exclusions=True, channel='G0')
    assert _run_siglink(app, monkeypatch) == ['S1', 'S2', 'S3']


def test_signal_linkage_toggle_off_keeps_everyone(monkeypatch):
    app = _siglink_app({'S2': ['G0']}, use_exclusions=False, channel='G0')
    assert _run_siglink(app, monkeypatch) == ['S1', 'S2', 'S3']


def test_signal_linkage_respects_a_red_designation(monkeypatch):
    app = _siglink_app({'S2': ['R5']}, use_exclusions=True, channel='R5')
    assert _run_siglink(app, monkeypatch) == ['S1', 'S3']


def test_signal_linkage_refuses_to_run_with_nothing_left(monkeypatch):
    app = _siglink_app({s: ['G0'] for s in ('S1', 'S2', 'S3')},
                       use_exclusions=True, channel='G0')
    assert _run_siglink(app, monkeypatch) is None
    assert app.warned, 'the user was given no reason for the empty run'
