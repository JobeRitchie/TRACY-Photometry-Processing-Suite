"""Generate a synthetic functional-mixed-model dataset for validating TRACY's
FLMM-style time-course against the real R `fastFMM::fui`.

Writes flmm_data.csv in long-ish format: one row per (subject, bout), columns:
    id, group, bout_order, Y.1 ... Y.L
The functional outcome Y is a peri-event z-score-like trace with a temporally
correlated noise structure (so the coefficient covariance has real off-diagonal
structure) plus a per-subject random intercept.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(20240603)

L = 50                      # timepoints
n_per_group = 6             # subjects per group
groups = ['A', 'B']
bouts_per_subject = 8
tau = 1.2                   # random-intercept SD (per subject)
sigma = 1.0                 # residual SD
ar = 0.7                    # AR(1) coeff -> temporal correlation in residuals

t = np.arange(L)
# Mean response: small baseline then a bump after onset (onset at index 15)
onset = 15
mean_curve = np.zeros(L)
mean_curve[onset:] = 2.0 * np.exp(-(t[onset:] - onset) / 12.0)
# Group-B adds an extra transient (the true between-group effect over time)
group_effect = np.zeros(L)
group_effect[onset:] = 1.0 * np.exp(-(t[onset:] - onset) / 8.0)


def ar1_noise(n_rows, L, ar, sigma):
    e = np.zeros((n_rows, L))
    e[:, 0] = rng.normal(0, sigma, n_rows)
    for k in range(1, L):
        e[:, k] = ar * e[:, k - 1] + rng.normal(0, sigma * np.sqrt(1 - ar**2), n_rows)
    return e


rows = []
sid = 0
for g in groups:
    for _ in range(n_per_group):
        sid += 1
        subj = f"S{sid:02d}"
        b_i = rng.normal(0, tau)                      # random intercept
        noise = ar1_noise(bouts_per_subject, L, ar, sigma)
        for j in range(bouts_per_subject):
            y = mean_curve + (group_effect if g == 'B' else 0.0) + b_i + noise[j]
            row = {'id': subj, 'group': g, 'bout_order': j + 1}
            for k in range(L):
                row[f'Y.{k+1}'] = y[k]
            rows.append(row)

df = pd.DataFrame(rows)
df.to_csv('flmm_validation/flmm_data.csv', index=False)
print(f"wrote flmm_validation/flmm_data.csv  shape={df.shape}  L={L} "
      f"subjects={sid} groups={groups}")
