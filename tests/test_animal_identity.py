"""
Tests for animal/session identity: splitting a subject ID into the animal that
was recorded and the session it was recorded in, and using that as the mixed
model's random-intercept grouping factor.

Why this exists: a subject ID is one *recording*. When an animal is recorded on
two days the IDs encode it (`2F_pre`, `2F_post`), but nothing in the pipeline
knew, so MixedLM grouped on the subject string and treated an animal's repeated
measures as independent -- which understates the standard errors.

The functions under test are module level and Tk-free; ``cluster_codes`` is
bound to a minimal stub the same way test_decay_metrics.py does it.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

_GUI_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fp_analysis_gui.py')


def _load_gui_module():
    spec = importlib.util.spec_from_file_location('fp_analysis_gui_identity', _GUI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)   # safe: the GUI only starts under __main__
    return module


_M = _load_gui_module()
_G = _M.FPAnalysisGUI

UNDERSCORE = _M.SESSION_PATTERN_PRESETS['Subject_Session  (2F_pre, 2F_post)']
HYPHEN = _M.SESSION_PATTERN_PRESETS['Subject-Session  (2F-pre, 2F-post)']
TRAILING = _M.SESSION_PATTERN_PRESETS['SubjectSession   (2FA, 2FB)']
NONE = _M.SESSION_PATTERN_PRESETS['No sessions      (each ID is one animal)']


class _Ident:
    """Stub exposing the identity helpers; they touch only params/log_message."""
    _session_pattern = _G._session_pattern
    cluster_codes = _G.cluster_codes
    _animal_ids_for = _G._animal_ids_for
    get_animal_map = _G.get_animal_map
    get_animal_id = _G.get_animal_id
    get_session_label = _G.get_session_label

    def __init__(self, pattern=UNDERSCORE, overrides=None, subjects=()):
        self.params = {'session_pattern': pattern,
                       'animal_overrides': overrides or {}}
        self.processed_data = {s: {} for s in subjects}
        self.logs = []
        self.log_message = lambda msg='', *a, **k: self.logs.append(str(msg))


# ---------------------------------------------------------------------------
# split_subject_id -- the per-ID regex
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sid,pattern,expected', [
    ('2F_post',    UNDERSCORE, ('2F', 'post')),
    ('2F_pre',     UNDERSCORE, ('2F', 'pre')),
    ('Mouse_1_pre', UNDERSCORE, ('Mouse_1', 'pre')),   # splits on the LAST separator
    ('2F-post',    HYPHEN,     ('2F', 'post')),
    ('2FA',        TRAILING,   ('2F', 'A')),
    ('2FB',        TRAILING,   ('2F', 'B')),
])
def test_ids_split_into_animal_and_session(sid, pattern, expected):
    assert _M.split_subject_id(sid, pattern) == expected


@pytest.mark.parametrize('sid,pattern', [
    ('2FA', UNDERSCORE),      # no separator present
    ('2F_post', HYPHEN),      # wrong separator
    ('2F_post', NONE),        # session parsing disabled
    ('_post', UNDERSCORE),    # empty animal
    ('2F_post', '(('),        # malformed pattern must not raise
    ('2F_post', r'^(?P<x>.+)$'),   # pattern without an 'animal' group
])
def test_an_id_that_does_not_split_is_its_own_animal(sid, pattern):
    assert _M.split_subject_id(sid, pattern) == (sid, '')


# ---------------------------------------------------------------------------
# derive_animal_map -- the project-level adoption rule
# ---------------------------------------------------------------------------

def test_a_split_that_groups_recordings_is_adopted():
    mapping, adopted = _M.derive_animal_map(
        ['2F_pre', '2F_post', '3M_pre', '3M_post'], UNDERSCORE)
    assert adopted
    assert {a for a, _ in mapping.values()} == {'2F', '3M'}


def test_a_split_that_groups_nothing_is_refused():
    """The guard that stops `Control`/`Treated` becoming `Contro`/`Treate`."""
    mapping, adopted = _M.derive_animal_map(['Control', 'Treated'], TRAILING)
    assert not adopted
    assert mapping == {'Control': ('Control', ''), 'Treated': ('Treated', '')}


def test_a_single_session_animal_survives_alongside_paired_ones():
    mapping, adopted = _M.derive_animal_map(
        ['2F_pre', '2F_post', '5M_pre'], UNDERSCORE)
    assert adopted
    assert mapping['5M_pre'] == ('5M', 'pre')
    assert mapping['2F_pre'][0] == mapping['2F_post'][0] == '2F'


def test_overrides_win_and_can_force_adoption_on_their_own():
    """Hand-linking a pair must work even when the pattern finds nothing."""
    mapping, adopted = _M.derive_animal_map(
        ['aaa', 'bbb'], NONE, overrides={'aaa': 'rat7', 'bbb': 'rat7'})
    assert adopted
    assert mapping['aaa'][0] == mapping['bbb'][0] == 'rat7'


def test_unlisted_subjects_are_unaffected_by_an_override():
    mapping, _ = _M.derive_animal_map(
        ['2F_pre', '2F_post', 'odd'], UNDERSCORE, overrides={'odd': '2F'})
    assert mapping['odd'][0] == '2F'
    assert mapping['2F_pre'][0] == '2F'


# ---------------------------------------------------------------------------
# cluster_codes -- what the mixed model actually groups on
# ---------------------------------------------------------------------------

def test_repeated_sessions_of_one_animal_share_a_random_intercept():
    """The A4 correctness fix: 4 recordings, 2 animals, so 2 grouping levels."""
    ident = _Ident()
    subjects = np.array(['2F_pre', '2F_post', '3M_pre', '3M_post'])
    codes, n_clusters, clustered = ident.cluster_codes(subjects)
    assert clustered
    assert n_clusters == 2
    assert codes[0] == codes[1]      # 2F_pre and 2F_post are one animal
    assert codes[2] == codes[3]
    assert codes[0] != codes[2]


def test_grouping_is_unchanged_when_ids_encode_no_repeats():
    """Projects that never had multi-session subjects must fit exactly as before."""
    ident = _Ident()
    subjects = np.array(['Control', 'Treated', 'Sham'])
    codes, n_clusters, clustered = ident.cluster_codes(subjects)
    assert not clustered
    assert n_clusters == 3
    assert len(set(codes.tolist())) == 3


def test_codes_are_positional_so_repeated_rows_stay_aligned():
    """One row per bout means a subject appears many times; every row of an
    animal must carry that animal's code, in order."""
    ident = _Ident()
    subjects = np.array(['2F_pre', '3M_post', '2F_post', '3M_pre', '2F_pre'])
    codes, _, _ = ident.cluster_codes(subjects)
    assert codes[0] == codes[2] == codes[4]        # all animal 2F
    assert codes[1] == codes[3]                    # all animal 3M
    assert codes[0] != codes[1]


def test_clustering_reduces_the_independent_unit_count_it_reports():
    """n_clusters feeds use_mixed and the sup-t bootstrap gate, so it has to be
    the number of animals, not recordings."""
    ident = _Ident()
    subjects = np.array(['2F_pre', '2F_post'])
    _, n_clusters, _ = ident.cluster_codes(subjects)
    assert n_clusters == 1          # one animal -> random effect must drop out


def test_the_r_engine_groups_on_the_same_animal_ids():
    """Both engines must key (1 | id) identically or they disagree on the fit."""
    ident = _Ident()
    subjects = np.array(['2F_pre', '2F_post', '3M_pre'])
    ids = ident._animal_ids_for(subjects)
    codes, _, _ = ident.cluster_codes(subjects)
    assert ids == ['2F', '2F', '3M']
    # same partition as the Python engine's integer codes
    assert (ids[0] == ids[1]) == (codes[0] == codes[1])
    assert (ids[0] == ids[2]) == (codes[0] == codes[2])


def test_clustering_is_announced_once_it_actually_merges():
    ident = _Ident()
    ident.cluster_codes(np.array(['2F_pre', '2F_post']))
    assert any('grouped by animal' in line for line in ident.logs)


def test_no_message_when_nothing_merges():
    ident = _Ident()
    ident.cluster_codes(np.array(['Control', 'Treated']))
    assert not any('grouped by animal' in line for line in ident.logs)


# ---------------------------------------------------------------------------
# The GUI-facing accessors
# ---------------------------------------------------------------------------

def test_accessors_read_the_current_project():
    ident = _Ident(subjects=['2F_pre', '2F_post'])
    assert ident.get_animal_id('2F_pre') == '2F'
    assert ident.get_session_label('2F_post') == 'post'


def test_an_unknown_subject_falls_back_to_itself():
    ident = _Ident(subjects=['2F_pre', '2F_post'])
    assert ident.get_animal_id('nobody') == 'nobody'
    assert ident.get_session_label('nobody') == ''
