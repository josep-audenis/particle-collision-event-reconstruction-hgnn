"""Calibrate every suite run on the SAME candidate set as the runner full-chain.

Candidate set = SigMatch==1 & AllParticles==1 (== the runner's "Full metric rows",
the fully-reconstructed signal candidates). This makes the calibrated numbers
directly comparable to the raw flat/full-chain numbers already reported.

Per run we compute:
  flat_tp     = (1-2w)^2                         tagger-level, epsilon=100% (== runner)
  cal_tp      = <(1-2*omega_cal)^2>              tagger-level, per-event calibrated mistag
  fc_flat     = flat_tp * n_cand/events          full-chain, uncalibrated (== runner full_chain)
  fc_cal      = cal_tp  * n_cand/events          full-chain, calibrated

Calibration (isotonic) is fit on a disjoint EventNumber-parity half and evaluated
on the other half.
Emits TSV: exp channel seed ift_ver n_cand w_flat flat_tp cal_tp cal_err fc_flat fc_cal auc events
"""
from __future__ import annotations
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/tfg/hgnn")
TSV = os.path.join(ROOT, "outputs/suite/results.tsv")


def eps_eff(eta):
    return float(np.mean((1.0 - 2.0 * eta) ** 2))


def boot_err(eta, n=500):
    if n <= 0 or eta.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    return float(np.std([eps_eff(eta[rng.integers(0, eta.size, eta.size)]) for _ in range(n)]))


def fit_iso(eta, wrong):
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=0.5)
        ir.fit(eta, wrong)
        return lambda x: np.clip(ir.predict(x), 0.0, 0.5)
    except Exception:
        p = np.polyfit(eta, wrong, 1)
        return lambda x: np.clip(np.polyval(p, x), 0.0, 0.5)


def find_csv(v, exp, ch, seed):
    vdir = os.path.join(ROOT, f"LHCb_logs/IFT/version_{v}")
    for p in [f"{vdir}/signal_reco_df_*__{exp}_{ch}_s{seed}_test_final.csv",
              f"{vdir}/signal_reco_df_*_{ch}_s{seed}_test_final.csv",
              f"{vdir}/signal_reco_df_*_s{seed}_test_final.csv",
              f"{vdir}/signal_reco_df_*_test_final.csv"]:
        hits = sorted(glob.glob(p))
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            return f"AMBIG:{len(hits)}"
    return None


def run(csv, events):
    d = pd.read_csv(csv)
    if {"B_id", "ft_b_score", "ft_bbar_score"} - set(d.columns):
        return None, "missing cols"
    if "SigMatch" in d.columns:
        d = d[d["SigMatch"] == 1]
    if "AllParticles" in d.columns:
        d = d[d["AllParticles"] == 1]
    d = d[np.sign(d["B_id"]) != 0].dropna(subset=["ft_b_score", "ft_bbar_score"])
    n = len(d)
    if n == 0:
        return None, "no candidates"
    truth = (np.sign(d["B_id"].to_numpy()) == 1).astype(int)
    pred = (d["ft_bbar_score"].to_numpy() >= d["ft_b_score"].to_numpy()).astype(int)
    eta = 1.0 - np.maximum(d["ft_b_score"].to_numpy(), d["ft_bbar_score"].to_numpy())
    wrong = (pred != truth).astype(float)

    w = wrong.mean()
    flat_tp = (1 - 2 * w) ** 2

    key = d["EventNumber"].to_numpy() if "EventNumber" in d.columns else np.arange(n)
    cm = (key % 2 == 0); em = ~cm
    if cm.sum() < 20 or em.sum() < 20:
        cm = np.arange(n) % 2 == 0; em = ~cm
    cal = fit_iso(eta[cm], wrong[cm])
    eta_cal = cal(eta[em])
    cal_tp = eps_eff(eta_cal)
    cal_err = boot_err(eta_cal)

    eff = n / events if events else float("nan")
    return dict(n=n, w=w * 100, flat_tp=flat_tp * 100, cal_tp=cal_tp * 100,
                cal_err=cal_err * 100, fc_flat=flat_tp * eff * 100,
                fc_cal=cal_tp * eff * 100), None


def main():
    rows = pd.read_csv(TSV, sep="\t")
    print("\t".join(["exp", "channel", "seed", "ift_ver", "n_cand", "w_flat", "flat_tp",
                      "cal_tp", "cal_err", "fc_flat", "fc_cal", "auc", "events"]))
    errs = []
    for _, r in rows.iterrows():
        csv = find_csv(int(r["ift_ver"]), r["exp"], r["channel"], int(r["seed"]))
        if not csv or str(csv).startswith("AMBIG"):
            errs.append(f"{r['exp']} {r['channel']} s{r['seed']} v{r['ift_ver']} CSV={csv}")
            continue
        res, err = run(csv, int(r["events"]))
        if err:
            errs.append(f"{r['exp']} {r['channel']} s{r['seed']} v{r['ift_ver']} {err}")
            continue
        print("\t".join(str(x) for x in [
            r["exp"], r["channel"], int(r["seed"]), int(r["ift_ver"]), res["n"],
            f"{res['w']:.2f}", f"{res['flat_tp']:.3f}", f"{res['cal_tp']:.3f}",
            f"{res['cal_err']:.3f}", f"{res['fc_flat']:.3f}", f"{res['fc_cal']:.3f}",
            f"{float(r['auc']):.3f}", int(r["events"])]))
    if errs:
        sys.stderr.write("=== ERRORS ===\n" + "\n".join(errs) + "\n")


if __name__ == "__main__":
    main()
