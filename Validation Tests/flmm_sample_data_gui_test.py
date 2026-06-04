"""
TRACY FLMM — Standalone GUI validation on the Sample Data
=========================================================

Loads `Sample Data/FLMM_test_data_for_jobe.csv` (the Loewinger et al. PV-cell
consumption dataset, already in bout-trace long format) and runs TRACY's
*exact* pure-Python FLMM time-course analysis on it, in a small GUI window.

Why this is a legitimate demonstration
--------------------------------------
The analysis is NOT re-implemented here. This script imports the real
``FPAnalysisGUI`` class from ``fp_analysis_gui.py`` and calls the very same
methods the TRACY GUI calls — via a lightweight subclass that skips the heavy
Tk setup:

    _fit_timecourse      -> per-timepoint random-intercept mixed model (FUI step 1)
    _flmm_smooth         -> Savitzky-Golay smoothing of beta(t), SE(t)
    _flmm_python_qn      -> GLS subject-clustered coefficient covariance -> joint qn
    _flmm_assemble_bands -> pointwise + simultaneous bands & significance masks
    _build_flmm_figure   -> the identical two-panel figure TRACY draws
    _flmm_stats_lines    -> the identical text summary TRACY shows

These methods were validated against the R package ``fastFMM`` 1.0.1:
beta(t) matches lme4/fastFMM to ~1e-14 and the simultaneous-band multiplier is
within ~5-10% (see ../flmm_validation/RESULTS.md).

Run:  python "Validation Tests/flmm_sample_data_gui_test.py"
(needs: pandas, numpy, scipy, matplotlib, statsmodels — same as TRACY)
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- make the repo root importable, then import the REAL TRACY class ----------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from fp_analysis_gui import FPAnalysisGUI  # noqa: E402

SAMPLE_CSV = os.path.join(REPO, "Sample Data", "FLMM_test_data_for_jobe.csv")


class TracyFLMMEngine(FPAnalysisGUI):
    """The real TRACY analysis methods, with the heavy GUI __init__ skipped.

    Subclassing guarantees we run the *identical* code paths as the app; we only
    override __init__ (so no Tk app is built) and log_message (print instead)."""

    def __init__(self):
        pass

    def log_message(self, message):
        print(f"[TRACY] {message}")


# Reference choices -> human labels
REFERENCES = [
    ("zero", "Signal ≠ 0 over time  (mean response vs baseline)"),
    ("across_bouts", "Across bouts  (effect of bout order within subject)"),
    ("between_groups", "Between groups  (group difference)"),
    ("interaction", "Across-bouts × group  (interaction)"),
    ("level_contrast", "Condition contrast  (one level = 0-point vs another/others)"),
    ("factor", "Factor: ALL levels vs a reference (one model — matches R fastFMM)"),
]
# Metadata columns usable as the grouping/condition factor.
GROUP_VARS = ["session", "sex", "side", "cell_type"]
ALL_OTHERS = "‹all others›"


class SampleDataTestApp:
    def __init__(self, root):
        self.root = root
        self.engine = TracyFLMMEngine()
        root.title("TRACY FLMM — Sample Data Validation (Python engine)")
        try:
            root.state("zoomed")
        except Exception:
            root.geometry("1100x800")

        self.df = pd.read_csv(SAMPLE_CSV)
        self.ycols = [c for c in self.df.columns if c.startswith("Y.")]
        self.L = len(self.ycols)

        self._build_controls()
        self._build_display()
        self._canvas = None

    # ---------------------------------------------------------------- UI
    def _build_controls(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Label(top, text="TRACY FLMM time-course — Sample Data",
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=8,
                                                       sticky="w")
        n_mice = self.df["mouse"].nunique()
        ttk.Label(top, text=(f"{len(self.df)} bouts · {n_mice} mice · {self.L} timepoints · "
                             f"file: {os.path.basename(SAMPLE_CSV)}"),
                  foreground="gray").grid(row=1, column=0, columnspan=8, sticky="w",
                                          pady=(0, 6))

        ttk.Label(top, text="Reference:").grid(row=2, column=0, sticky="e")
        self.ref_var = tk.StringVar(value="between_groups")
        self.ref_combo = ttk.Combobox(
            top, width=42, state="readonly",
            values=[lbl for _, lbl in REFERENCES],
            textvariable=tk.StringVar())
        self.ref_combo.current(2)
        self.ref_combo.grid(row=2, column=1, sticky="w", padx=(4, 12))
        self.ref_combo.bind("<<ComboboxSelected>>", self._on_ref_change)

        ttk.Label(top, text="Grouping variable:").grid(row=2, column=2, sticky="e")
        self.group_combo = ttk.Combobox(top, width=12, state="readonly",
                                         values=GROUP_VARS)
        self.group_combo.current(0)
        self.group_combo.grid(row=2, column=3, sticky="w", padx=(4, 12))
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_change)

        self.pw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Pointwise only (skip simultaneous band)",
                        variable=self.pw_var).grid(row=2, column=4, sticky="w", padx=(0, 12))

        # Condition-contrast pickers (used by the 'level_contrast' reference)
        ttk.Label(top, text="Reference level (0-point):").grid(row=4, column=0, sticky="e")
        self.reflevel_combo = ttk.Combobox(top, width=14, state="readonly")
        self.reflevel_combo.grid(row=4, column=1, sticky="w", padx=(4, 12))
        self.reflevel_combo.bind("<<ComboboxSelected>>", lambda *_: self._refresh_compto())
        ttk.Label(top, text="Compare to:").grid(row=4, column=2, sticky="e")
        self.compto_combo = ttk.Combobox(top, width=14, state="readonly")
        self.compto_combo.grid(row=4, column=3, sticky="w", padx=(4, 12))

        ttk.Label(top, text="fps:").grid(row=3, column=0, sticky="e")
        self.fps_var = tk.StringVar(value="30")
        ttk.Entry(top, width=6, textvariable=self.fps_var).grid(row=3, column=1, sticky="w")
        ttk.Label(top, text="samples before onset:").grid(row=3, column=2, sticky="e")
        # Data is 3 s pre + 3 s post -> onset at the trace midpoint.
        self.pre_var = tk.StringVar(value=str(self.L // 2))
        ttk.Entry(top, width=6, textvariable=self.pre_var).grid(row=3, column=3, sticky="w")
        ttk.Label(top, text=f"(onset at midpoint = {self.L // 2}; x-axis display only)",
                  foreground="gray").grid(row=3, column=4, columnspan=3, sticky="w")

        self.run_btn = ttk.Button(top, text="Run analysis", command=self._run)
        self.run_btn.grid(row=2, column=5, rowspan=2, padx=8)
        self.status = ttk.Label(top, text="Ready.", foreground="#1565C0")
        self.status.grid(row=2, column=6, rowspan=2, sticky="w")
        self._populate_levels()
        self._on_ref_change()

    def _build_display(self):
        body = ttk.Frame(self.root, padding=8)
        body.pack(fill="both", expand=True)
        self.plot_frame = ttk.Frame(body)
        self.plot_frame.pack(side="left", fill="both", expand=True)
        right = ttk.LabelFrame(body, text="Analysis summary", padding=4)
        right.pack(side="right", fill="y")
        self.stats_text = tk.Text(right, width=58, height=34, wrap="word",
                                  font=("Consolas", 9))
        self.stats_text.pack(fill="both", expand=True)

    def _on_ref_change(self, *_):
        ref = self._reference()
        uses_group = ref in ("between_groups", "interaction", "level_contrast", "factor")
        self.group_combo.configure(state="readonly" if uses_group else "disabled")
        # reference level used by level_contrast AND factor; compare-to only by level_contrast
        self.reflevel_combo.configure(
            state="readonly" if ref in ("level_contrast", "factor") else "disabled")
        self.compto_combo.configure(
            state="readonly" if ref == "level_contrast" else "disabled")

    def _on_group_change(self, *_):
        self._populate_levels()

    def _populate_levels(self):
        """Refresh the reference-level choices from the grouping var."""
        gvar = self.group_combo.get()
        levels = sorted(self.df[gvar].astype(str).unique().tolist())
        self.reflevel_combo.configure(values=levels)
        if self.reflevel_combo.get() not in levels:
            self.reflevel_combo.set(levels[0] if levels else "")
        self._refresh_compto()

    def _refresh_compto(self):
        """Rebuild 'Compare to' to exclude the CURRENT reference level."""
        gvar = self.group_combo.get()
        levels = sorted(self.df[gvar].astype(str).unique().tolist())
        ref_lvl = self.reflevel_combo.get()
        opts = [ALL_OTHERS] + [l for l in levels if l != ref_lvl]
        self.compto_combo.configure(values=opts)
        if self.compto_combo.get() not in opts:
            self.compto_combo.set(ALL_OTHERS)

    def _reference(self):
        return REFERENCES[self.ref_combo.current()][0]

    # ----------------------------------------------------------- analysis
    def _run(self):
        ref = self._reference()
        gvar = self.group_combo.get()
        pointwise_only = bool(self.pw_var.get())
        try:
            fps = float(self.fps_var.get())
            prebout = int(float(self.pre_var.get()))
        except ValueError:
            messagebox.showerror("Bad input", "fps and samples-before-onset must be numbers.")
            return

        # Each row of the sample CSV is already one bout trace.
        full_Z = self.df[self.ycols].to_numpy(float)
        subjects = self.df["mouse"].to_numpy(object)
        orders = self.df["bout_num"].to_numpy(float)
        ref_level = None
        model_ref = ref          # the reference key passed to TRACY's engine
        behavior_label = "consumption bout"

        if ref == "factor":
            # ONE model Y ~ gvar + (1|mouse) with ref_level as baseline; every
            # other level becomes a 'reference vs level' panel. Matches fastFMM.
            ref_level = self.reflevel_combo.get()
            factor = self.df[gvar].astype(str).to_numpy(object)
            levels = sorted(set(factor.tolist()))
            if not ref_level or ref_level not in levels:
                ref_level = levels[0]
            self.run_btn.configure(state="disabled")
            self.status.configure(
                text=f"Fitting factor model over {len(levels)} levels (~30 s)…")
            self.root.update_idletasks()
            threading.Thread(
                target=self._compute_factor,
                args=(full_Z, subjects, factor, ref_level, pointwise_only,
                      fps, prebout, gvar),
                daemon=True).start()
            return

        if ref == "level_contrast":
            # One level of `gvar` becomes the 0-point; compare vs a chosen
            # level or all others pooled.  Same model as between_groups, with an
            # explicit reference level → β(t) = (comparison) − (reference).
            ref_level = self.reflevel_combo.get()
            comp_sel = self.compto_combo.get()
            col = self.df[gvar].astype(str).to_numpy(object)
            if not ref_level:
                messagebox.showerror("Pick a level", "Choose a reference level.")
                return
            if comp_sel == ALL_OTHERS:
                comp_label = "others"
                groups = np.array([ref_level if v == ref_level else comp_label
                                   for v in col], dtype=object)
                keep = np.ones(len(col), dtype=bool)
            else:
                if comp_sel == ref_level or not comp_sel:
                    messagebox.showerror("Pick a level",
                                         "Reference and comparison levels must differ.")
                    return
                comp_label = comp_sel
                keep = np.isin(col, [ref_level, comp_sel])
                groups = col.copy()
            Z = full_Z[keep]
            subjects = subjects[keep]
            orders = orders[keep]
            groups = groups[keep]
            present = set(groups.tolist())
            if ref_level not in present or comp_label not in present:
                messagebox.showerror("Need both conditions",
                                     f"Need bouts for both '{ref_level}' and "
                                     f"'{comp_label}'.")
                return
            groups_present = [ref_level, comp_label]   # reference first
            model_ref = "between_groups"
            behavior_label = f"{comp_label} vs {ref_level}  ({gvar})"
        elif ref in ("between_groups", "interaction"):
            Z = full_Z
            groups = self.df[gvar].astype(str).to_numpy(object)
            groups_present = sorted(set(groups.tolist()))
            if len(groups_present) != 2:
                messagebox.showerror(
                    "Need exactly 2 groups",
                    f"'{gvar}' has {len(groups_present)} levels {groups_present}. "
                    "Between-groups / interaction needs exactly two — use "
                    "'Condition contrast' to pool the rest.")
                return
            behavior_label = f"consumption bout — by {gvar}"
        else:
            Z = full_Z
            groups = np.array([None] * len(self.df), dtype=object)
            groups_present = []

        self.run_btn.configure(state="disabled")
        self.status.configure(text="Running per-timepoint mixed models (~15–30 s)…")
        self.root.update_idletasks()

        args = (Z, subjects, groups, orders, model_ref, ref_level, pointwise_only,
                groups_present, fps, prebout, behavior_label)
        threading.Thread(target=self._compute, args=args, daemon=True).start()

    def _compute(self, Z, subjects, groups, orders, model_ref, ref_level,
                 pointwise_only, groups_present, fps, prebout, behavior_label):
        """Heavy lifting off the UI thread — calls TRACY's real methods."""
        try:
            e = self.engine
            fit = e._fit_timecourse(Z, subjects, groups, orders, model_ref,
                                    ref_level=ref_level)
            beta_s, se_s = e._flmm_smooth(fit["beta"], fit["se"])
            qn, qn_method = e._flmm_python_qn(fit, Z, groups, model_ref, pointwise_only)
            c_star = qn
            bands = e._flmm_assemble_bands(beta_s, se_s, c_star)
            result = dict(Z=Z, groups=groups, ref=model_ref, pointwise_only=pointwise_only,
                          groups_present=groups_present, beta_s=beta_s, se_s=se_s,
                          c_star=c_star, qn_method=qn_method, bands=bands,
                          n_subjects=len(set(subjects.tolist())), fps=fps,
                          prebout=prebout, behavior_label=behavior_label)
            self.root.after(0, lambda: self._display(result))
        except Exception as exc:  # surface errors on the UI thread
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: self._fail(exc, tb))

    def _compute_factor(self, Z, subjects, factor, ref_level, pointwise_only,
                        fps, prebout, gvar):
        """One-model factor analysis off the UI thread — TRACY's real methods."""
        try:
            e = self.engine
            fitf = e._fit_factor_timecourse(Z, subjects, factor, ref_level=ref_level)
            coef_labels = fitf["coef_labels"]
            qns = e._flmm_factor_qn(fitf, pointwise_only)
            betas_s, ses_s = [], []
            for j in range(len(coef_labels)):
                bj, sj = e._flmm_smooth(fitf["betas"][j], fitf["ses"][j])
                betas_s.append(bj); ses_s.append(sj)
            c_stars = [1.96 if pointwise_only else qns[j] for j in range(len(coef_labels))]
            result = dict(kind="factor", Z=Z, coef_labels=coef_labels, ref_level=ref_level,
                          betas_s=betas_s, ses_s=ses_s, c_stars=c_stars,
                          pointwise_only=pointwise_only,
                          n_subjects=len(set(subjects.tolist())), fps=fps,
                          prebout=prebout, gvar=gvar)
            self.root.after(0, lambda: self._display(result))
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            self.root.after(0, lambda: self._fail(exc, tb))

    def _fail(self, exc, tb):
        self.run_btn.configure(state="normal")
        self.status.configure(text="Error.", foreground="red")
        print(tb)
        messagebox.showerror("Analysis failed", f"{exc}\n\nSee console for details.")

    def _display(self, r):
        channel = "PV ΔF/F"
        L = r["Z"].shape[1]
        time_axis = (np.arange(L) - r["prebout"]) / r["fps"]

        if r.get("kind") == "factor":
            behavior = f"all {r['gvar']} (ref = {r['ref_level']})"
            backend_label = "Python FUI (GLS coefficient covariance)"
            fig = self.engine._build_flmm_factor_figure(
                time_axis, r["coef_labels"], r["ref_level"], r["betas_s"],
                r["ses_s"], r["c_stars"], r["pointwise_only"], channel, behavior)
            lines = self.engine._flmm_factor_stats_lines(
                r["coef_labels"], r["ref_level"], behavior, channel, r["Z"],
                r["n_subjects"], time_axis, r["betas_s"], r["ses_s"],
                r["c_stars"], r["pointwise_only"], backend_label)
        else:
            ci_lo, ci_hi, sci_lo, sci_hi, sig_pt, sig_sim = r["bands"]
            behavior = r["behavior_label"]
            backend_label = f"Python FUI ({r['qn_method']})"
            fig = self.engine._build_flmm_figure(
                r["Z"], r["groups"], time_axis, r["ref"], r["beta_s"],
                ci_lo, ci_hi, sci_lo, sci_hi, sig_sim, r["pointwise_only"],
                behavior, channel, r["groups_present"])
            lines = self.engine._flmm_stats_lines(
                r["ref"], behavior, channel, r["Z"], r["n_subjects"],
                r["groups_present"], time_axis, sig_sim, sig_pt, r["c_star"],
                r["pointwise_only"], backend_label)

        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, self.plot_frame).update()

        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", "\n".join(lines))

        self.run_btn.configure(state="normal")
        self.status.configure(text="Done.", foreground="#2E7D32")


def main():
    if not os.path.exists(SAMPLE_CSV):
        print(f"Sample data not found: {SAMPLE_CSV}")
        sys.exit(1)
    root = tk.Tk()
    SampleDataTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
