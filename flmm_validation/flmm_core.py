"""Python FUI implementation candidates, compared against fastFMM ground truth.

Fits a per-timepoint mixed model (statsmodels MixedLM, REML) and computes the
joint multiplier qn several ways, to find the coefficient-covariance estimator
that best matches R fastFMM::fui.
"""
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

N_SIM = 10000
SEED = 0


def qn_from_cov(cov, n_sim=N_SIM, seed=SEED):
    """fastFMM's joint multiplier: standardize cov->corr, simulate MVN, 95th pct of max|z|."""
    d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    S = np.diag(1.0 / d)
    corr = S @ cov @ S
    corr = (corr + corr.T) / 2
    # eigen-trim to PSD
    w, V = np.linalg.eigh(corr)
    w = np.clip(w, 1e-8, None)
    corr = V @ np.diag(w) @ V.T
    rng = np.random.default_rng(seed)
    sim = rng.multivariate_normal(np.zeros(len(cov)), corr, size=n_sim, method='cholesky')
    return float(np.percentile(np.max(np.abs(sim), axis=1), 95))


def fit_timecourse(Y, X, subj_codes, coef_idx):
    """Per-timepoint MixedLM (random intercept). Returns betaTilde, se_model,
    residual matrix eta (n x L), and per-timepoint variance components."""
    n, L = Y.shape
    p = X.shape[1]
    beta = np.full(L, np.nan)
    se_model = np.full(L, np.nan)
    eta = np.full((n, L), np.nan)
    tau2 = np.full(L, np.nan)
    sig2 = np.full(L, np.nan)
    full_params = np.full((L, p), np.nan)
    for t in range(L):
        y = Y[:, t]
        mdf = None
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for meth in ('lbfgs', 'bfgs', 'cg', 'powell'):
                try:
                    mdf = sm.MixedLM(y, X, groups=subj_codes).fit(
                        reml=True, method=meth, maxiter=200, disp=False)
                    if np.all(np.isfinite(np.asarray(mdf.bse_fe))):
                        break
                except Exception:
                    mdf = None
        if mdf is None:
            continue
        fp = np.asarray(mdf.fe_params, float)
        beta[t] = fp[coef_idx]
        se_model[t] = float(mdf.bse_fe[coef_idx])
        tau2[t] = max(float(np.asarray(mdf.cov_re)[0, 0]), 0.0)
        sig2[t] = float(mdf.scale)
        full_params[t] = fp
        eta[:, t] = y - X @ fp
    return beta, se_model, eta, tau2, sig2, full_params


def gls_sandwich_cov(X, subj_codes, eta, tau2, sig2, coef_idx):
    """GLS subject-clustered (CR0) covariance of the coefficient function.

    a_i(t) = [ B(t) X_i' V_i(t)^-1 eta_i(t) ]_k ;  Cov = A' A.
    V_i = sig2 I + tau2 11'  (random intercept), inverted via Sherman-Morrison.
    """
    n, L = eta.shape
    subs = np.unique(subj_codes)
    A = np.zeros((len(subs), L))
    for t in range(L):
        s2, t2 = sig2[t], tau2[t]
        # Build X'V^-1X and per-subject pieces
        XtViX = np.zeros((X.shape[1], X.shape[1]))
        pieces = []  # (Xi_Vinv  as  X_i' V_i^-1 , for each subject)
        for si in subs:
            m = subj_codes == si
            Xi = X[m]
            ni = Xi.shape[0]
            # V_i^-1 = (1/s2)(I - (t2/(s2+ni*t2)) 11')
            c = t2 / (s2 + ni * t2)
            # X_i' V_i^-1 = (1/s2)(X_i' - c * (X_i'1)(1'))
            ones = np.ones(ni)
            XiT = Xi.T
            Xi_Vinv = (XiT - c * np.outer(XiT @ ones, ones)) / s2  # p x ni
            XtViX += Xi_Vinv @ Xi
            pieces.append((m, Xi_Vinv))
        B = np.linalg.pinv(XtViX)
        for idx, (m, Xi_Vinv) in enumerate(pieces):
            g_i = Xi_Vinv @ eta[m, t]          # p-vector
            A[idx, t] = (B @ g_i)[coef_idx]
    return A.T @ A


def analytic_cov(X, subj_codes, Y, tau2, sig2, se_model, full_params, coef_idx, smooth=True):
    """fastFMM-style analytic coefficient covariance for a random-intercept model.

    Off-diagonal Cov(beta_k(s), beta_k(t)) = g(s,t) * <w_k(s), w_k(t)>, where
    g(s,t) is the cross-location covariance of the random intercept and
    w_k(s) = [ inv(X'V(s)^-1 X) (X_j' V_j(s)^-1 1) ]_k per subject j. Diagonal is
    the model-based variance (se_model^2). g(s,t) is estimated from per-location
    random-effect BLUPs and lightly smoothed (proxy for fastFMM's fbps surface).
    """
    n, L = Y.shape
    p = X.shape[1]
    subs = np.unique(subj_codes)
    nsub = len(subs)
    # per-location subject weight vectors w_k(s)  ->  (L x nsub)
    wk = np.zeros((L, nsub))
    blups = np.zeros((nsub, L))
    for t in range(L):
        s2, t2 = sig2[t], tau2[t]
        if not (np.isfinite(s2) and s2 > 0):
            continue
        XtViX = np.zeros((p, p))
        subj_vec = []   # X_j' V_j^-1 1   per subject
        for j, si in enumerate(subs):
            m = subj_codes == si
            Xi = X[m]; ni = Xi.shape[0]
            c = t2 / (s2 + ni * t2)
            ones = np.ones(ni)
            Xi_Vinv = (Xi.T - c * np.outer(Xi.T @ ones, ones)) / s2  # p x ni
            XtViX += Xi_Vinv @ Xi
            subj_vec.append(Xi_Vinv @ ones)                          # p-vector
        B = np.linalg.pinv(XtViX)
        beta_t = full_params[t]
        for j, si in enumerate(subs):
            wk[t, j] = (B @ subj_vec[j])[coef_idx]
            m = subj_codes == si
            ni = m.sum()
            # random-intercept BLUP: shrinkage of mean residual
            resid = Y[m, t] - X[m] @ beta_t
            blups[j, t] = (ni * t2) / (s2 + ni * t2) * resid.mean()
    # g(s,t): cross-location covariance of BLUPs across subjects
    Bc = blups - blups.mean(0, keepdims=True)
    g = (Bc.T @ Bc) / max(1, nsub - 1)
    if smooth:
        # light separable smoothing of the g surface (proxy for fbps)
        from scipy.ndimage import gaussian_filter
        g = gaussian_filter(g, sigma=2.0, mode='nearest')
    # assemble coefficient covariance
    W = wk  # L x nsub ; <w(s),w(t)> = (W W^T)[s,t]
    cov = g * (W @ W.T)
    # diagonal = model-based total variance
    di = np.isfinite(se_model)
    cov[np.diag_indices(L)] = np.where(di, se_model**2, np.diag(cov))
    return cov


def main():
    dat = pd.read_csv('flmm_validation/flmm_data.csv')
    ycols = [c for c in dat.columns if c.startswith('Y.')]
    Y = dat[ycols].to_numpy(float)
    L = len(ycols)
    # Design: intercept + groupB dummy (ref = A); coef_idx=1 is groupB
    gd = (dat['group'].to_numpy() == 'B').astype(float)
    X = np.column_stack([np.ones(len(dat)), gd])
    coef_idx = 1
    subj_codes = pd.factorize(dat['id'])[0]

    beta, se_model, eta, tau2, sig2, full_params = fit_timecourse(Y, X, subj_codes, coef_idx)

    # Candidate covariances ----------------------------------------------------
    cov_sand = gls_sandwich_cov(X, subj_codes, eta, tau2, sig2, coef_idx)
    cov_anl = analytic_cov(X, subj_codes, Y, tau2, sig2, se_model, full_params, coef_idx, smooth=True)
    cov_anl_ns = analytic_cov(X, subj_codes, Y, tau2, sig2, se_model, full_params, coef_idx, smooth=False)
    # raw-data proxy (current TRACY approach): within-group centered data corr
    Zc = Y.copy()
    for g in np.unique(gd):
        mm = gd == g
        Zc[mm] -= Zc[mm].mean(0, keepdims=True)
    cov_raw = np.corrcoef(Zc, rowvar=False)

    qn_sand = qn_from_cov(cov_sand)
    qn_raw = qn_from_cov(cov_raw)
    qn_anl = qn_from_cov(cov_anl)
    qn_anl_ns = qn_from_cov(cov_anl_ns)

    # fastFMM ground truth -----------------------------------------------------
    r = pd.read_csv('flmm_validation/r_results.csv')
    r_qn = pd.read_csv('flmm_validation/r_qn.csv')['qn'].to_numpy()[coef_idx]
    r_cov = pd.read_csv('flmm_validation/r_cov_group.csv').to_numpy()
    qn_rcov = qn_from_cov(r_cov)  # reproduce fastFMM qn from its own cov (sanity)

    print("=" * 64)
    print(f"{'':20s}{'fastFMM':>14s}{'python':>14s}{'absdiff':>12s}")
    # raw beta agreement (python raw vs R smoothed betaHat — expect close shape)
    bdiff = np.nanmax(np.abs(beta - r['beta_group'].to_numpy()))
    sdiff = np.nanmax(np.abs(se_model - r['se_group'].to_numpy()))
    print(f"{'max|beta diff|':20s}{'':>14s}{'':>14s}{bdiff:>12.4f}")
    print(f"{'max|SE diff|':20s}{'':>14s}{'':>14s}{sdiff:>12.4f}")
    print(f"{'SE (median)':20s}{np.nanmedian(r['se_group']):>14.4f}"
          f"{np.nanmedian(se_model):>14.4f}")
    print("-" * 64)
    print(f"{'qn group':20s}{r_qn:>14.4f}{qn_anl:>14.4f}{abs(r_qn-qn_anl):>12.4f}"
          f"   (analytic G(s,t), smoothed)")
    print(f"{'qn group':20s}{r_qn:>14.4f}{qn_anl_ns:>14.4f}{abs(r_qn-qn_anl_ns):>12.4f}"
          f"   (analytic G(s,t), raw)")
    print(f"{'qn group':20s}{r_qn:>14.4f}{qn_sand:>14.4f}{abs(r_qn-qn_sand):>12.4f}"
          f"   (GLS sandwich)")
    print(f"{'qn group':20s}{r_qn:>14.4f}{qn_raw:>14.4f}{abs(r_qn-qn_raw):>12.4f}"
          f"   (raw-data proxy)")
    print(f"{'qn from R-cov':20s}{r_qn:>14.4f}{qn_rcov:>14.4f}{abs(r_qn-qn_rcov):>12.4f}"
          f"   (sanity: our sim on R cov)")
    print("=" * 64)
    print(f"pointwise mult = 1.96   |   fastFMM joint qn = {r_qn:.4f}")


if __name__ == '__main__':
    main()
