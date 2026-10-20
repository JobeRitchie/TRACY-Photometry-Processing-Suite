"""Bout-order FLMM: the covariate is the real bout number, and the model's
terms (each group's slope, their difference, the mean signal) are readable.

Before this, ``_collect_flmm_traces`` numbered bouts by their position in the
stored list.  A bout dropped at extraction -- on the NSF project, five animals
whose first approach came within the 5 s pre-window of the recording start --
shifted every later bout down by one, so the second approach was modeled as
bout 1.  The interaction model also reported only the slope *difference*,
never either group's own slope.
"""
import numpy as np

import fp_analysis_gui as G


def _app(store):
    app = object.__new__(G.FPAnalysisGUI)
    app.processed_data = store
    app.params = {'preboutframes': 2}
    app.get_fps = lambda *a, **k: 10.0
    app.log_message = lambda *a, **k: None
    app._realign_bouts = lambda bouts, prebout, src_fs=None: bouts
    app._entry_prebout = lambda entry: 2
    app._entry_fs = lambda entry: 10.0
    app._session_pattern = lambda: None
    return app


def test_dropped_first_bout_keeps_true_numbers():
    entry = {'G0': [np.ones(6), np.ones(6) * 2, np.ones(6) * 3],
             '_kept_indices': {'G0': [1, 2, 4]}}
    app = _app({'S1': {'bouts': {'Approach': entry}}})
    _, _, _, orders, _ = app._collect_flmm_traces(['S1'], 'Approach', 'G0', {},
                                                  max_bouts=None)
    assert orders.tolist() == [2.0, 3.0, 5.0]


def test_legacy_store_without_kept_indices_uses_position():
    entry = {'G0': [np.ones(6), np.ones(6)]}
    app = _app({'S1': {'bouts': {'Approach': entry}}})
    _, _, _, orders, _ = app._collect_flmm_traces(['S1'], 'Approach', 'G0', {},
                                                  max_bouts=None)
    assert orders.tolist() == [1.0, 2.0]


def _interaction_data(seed=0):
    """CTRL: no bout-order effect.  KO: response falls 0.2 z per bout."""
    rng = np.random.default_rng(seed)
    store, s2g = {}, {}
    L = 8
    for gi, grp in enumerate(('KO', 'CTRL')):
        for a in range(6):
            subj = f'{grp}{a}'
            s2g[subj] = grp
            off = rng.normal(0, 0.3)
            bouts = []
            for k in range(1, 9):
                slope = -0.2 if grp == 'KO' else 0.0
                bouts.append(1.0 + off + slope * (k - 4.5)
                             + rng.normal(0, 0.05, L))
            store[subj] = {'bouts': {'Approach': {'G0': bouts}}}
    return store, s2g


def test_interaction_terms_recover_each_group_slope():
    store, s2g = _interaction_data()
    app = _app(store)
    Z, S, Gr, O, L = app._collect_flmm_traces(list(store), 'Approach', 'G0', s2g,
                                              max_bouts=None)
    fit = app._fit_timecourse(Z, S, Gr, O, 'interaction', ref_level='CTRL',
                              log=lambda m: None)
    terms = {t['key']: t for t in app._flmm_model_terms(
        fit, 'interaction', ['CTRL', 'KO'], pointwise_only=False)}
    assert np.allclose(terms['slope_ref']['beta'], 0.0, atol=0.02)      # CTRL
    assert np.allclose(terms['slope_oth']['beta'], -0.2, atol=0.02)     # KO
    assert np.allclose(terms['slope_diff']['beta'], -0.2, atol=0.02)
    assert np.allclose(terms['slope_pooled']['beta'], -0.1, atol=0.02)
    assert terms['slope_diff']['tested'] and terms['slope_diff']['sig_simultaneous'].all()
    assert not terms['slope_ref']['sig_pointwise'].any()
    # The tested term is the same coefficient the main β(t) reports.
    b, _ = app._flmm_smooth(fit['beta'], fit['se'])
    assert np.allclose(terms['slope_diff']['beta'], b)


def test_workbook_has_bout_data_and_predictions():
    store, s2g = _interaction_data(1)
    app = _app(store)
    Z, S, Gr, O, L = app._collect_flmm_traces(list(store), 'Approach', 'G0', s2g,
                                              max_bouts=None)
    fit = app._fit_timecourse(Z, S, Gr, O, 'interaction', ref_level='CTRL',
                              log=lambda m: None)
    terms = app._flmm_model_terms(fit, 'interaction', ['CTRL', 'KO'], True)
    tested = next(t for t in terms if t['tested'])
    panel = {'channel': 'G0', 'time': (np.arange(L) - 2) / 10.0, 'Z': Z,
             'groups': Gr, 'groups_present': ['CTRL', 'KO'], 'subjects': S,
             'orders': O, 'reference': 'interaction', 'behavior': 'Approach',
             'pointwise_only': True, 'backend': 'Python FUI', 'n_subjects': 12,
             'terms': terms, 'order_center': fit['order_center'],
             **{k: tested[k] for k in ('beta', 'se', 'ci_lo', 'ci_hi', 'sci_lo',
                                       'sci_hi', 'sig_pointwise', 'sig_simultaneous')}}
    sheets = app._flmm_workbook_frames(
        {'kind': 'timecourse', 'order': ['G0'], 'panels': {'G0': panel}})
    assert set(sheets) == {'README', 'Model_terms', 'Input_bouts', 'Observed_mean_per_bout',
                           'Model_fit_per_bout'}
    bt = sheets['Input_bouts']
    assert len(bt) == Z.shape[0] and bt['Bout_number'].max() == 8
    assert np.allclose(bt.iloc[:, 5:].to_numpy(float), Z)
    # The model's KO prediction at bout 1 vs bout 8 falls by ~7 × 0.2 z.
    pr = sheets['Model_fit_per_bout'].set_index(['Group', 'Bout_number'])
    drop = pr.loc[('KO', 1)].iloc[1:].astype(float) - pr.loc[('KO', 8)].iloc[1:].astype(float)
    assert np.allclose(drop, 1.4, atol=0.15)
    ob = sheets['Observed_mean_per_bout']
    assert set(ob['n_animals']) == {6}
