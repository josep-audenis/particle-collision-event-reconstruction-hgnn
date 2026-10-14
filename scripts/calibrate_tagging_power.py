"""Calibrated effective tagging power + benchmark-grade reporting for an IFT reco CSV.

Implements the spec in wiki/benchmarking-and-calibration.md:
  - per-event mistag calibration omega_true(eta_pred) fit on a calibration split (linear + isotonic)
  - effective tagging power eps_eff = <(1 - 2*eta_cal)^2>  (per-event, calibrated)
  - dual denominator: tagger-level (per signal-matched reco candidate) and full-chain
    (per generated event, folding reconstruction+matching efficiency)
  - reliability diagram data + a standard results-table row

Usage:
  python scripts/calibrate_tagging_power.py \
    --csv LHCb_logs/IFT/version_<N>/signal_reco_df_<sample>__<tag>.csv \
    --n-generated 10111 --label "exclude_signal_tracks" --bootstrap 500
The reco CSV must have columns: B_id, ft_b_score, ft_bbar_score, and a signal-match flag
(SigMatch) and EventNumber. Calibration split is taken by EventNumber parity (disjoint from eval).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def effective_power(eta: np.ndarray) -> float:
    """Per-event effective tagging power with eps_tag=1 over the given rows."""
    return float(np.mean((1.0 - 2.0 * eta) ** 2))


def bootstrap_err(vals_fn, eta: np.ndarray, n: int) -> float:
    if n <= 0 or eta.size == 0:
        return float("nan")
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        idx = rng.integers(0, eta.size, eta.size)
        out.append(vals_fn(eta[idx]))
    return float(np.std(out))


def fit_isotonic(eta: np.ndarray, wrong: np.ndarray):
    """Monotonic calibration omega_true(eta) via pool-adjacent-violators (sklearn if available)."""
    try:
        from sklearn.isotonic import IsotonicRegression
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=0.5)
        ir.fit(eta, wrong)
        return lambda x: np.clip(ir.predict(x), 0.0, 0.5)
    except Exception:
        # linear fallback
        p = np.polyfit(eta, wrong, 1)
        return lambda x: np.clip(np.polyval(p, x), 0.0, 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--n-generated", type=int, default=None,
                    help="number of generated signal events, for the full-chain denominator")
    ap.add_argument("--label", default="run")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--match-col", default="SigMatch")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    need = {"B_id", "ft_b_score", "ft_bbar_score"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"CSV missing columns: {sorted(miss)}")

    # signal-matched, flavour-defined candidates
    d = df.copy()
    if args.match_col in d.columns:
        d = d[d[args.match_col] == 1]
    d = d[np.sign(d["B_id"]) != 0].dropna(subset=["ft_b_score", "ft_bbar_score"])
    n_cand = len(d)
    if n_cand == 0:
        raise SystemExit("no signal-matched flavour-defined candidates")

    truth = (np.sign(d["B_id"].to_numpy()) == 1).astype(int)        # 1 = b, 0 = bbar
    pred = (d["ft_bbar_score"].to_numpy() >= d["ft_b_score"].to_numpy()).astype(int)
    eta_raw = 1.0 - np.maximum(d["ft_b_score"].to_numpy(), d["ft_bbar_score"].to_numpy())
    wrong = (pred != truth).astype(float)

    # raw (uncalibrated) numbers
    w_flat = wrong.mean()
    eps_raw = effective_power(eta_raw)

    # calibration split: disjoint by EventNumber parity (fallback: row parity)
    key = d["EventNumber"].to_numpy() if "EventNumber" in d.columns else np.arange(n_cand)
    calib_mask = (key % 2 == 0)
    eval_mask = ~calib_mask
    if calib_mask.sum() < 20 or eval_mask.sum() < 20:
        calib_mask = np.arange(n_cand) % 2 == 0
        eval_mask = ~calib_mask

    cal = fit_isotonic(eta_raw[calib_mask], wrong[calib_mask])
    eta_cal_eval = cal(eta_raw[eval_mask])
    eps_cal = effective_power(eta_cal_eval)
    eps_cal_err = bootstrap_err(effective_power, eta_cal_eval, args.bootstrap)

    # reliability (eval split): predicted vs actual mistag in eta bins
    bins = np.linspace(0, 0.5, 6)
    reli = []
    bidx = np.digitize(eta_cal_eval, bins) - 1
    for b in range(len(bins) - 1):
        m = bidx == b
        if m.sum() >= 5:
            reli.append((0.5 * (bins[b] + bins[b + 1]),
                         float(eta_cal_eval[m].mean()),
                         float(wrong[eval_mask][m].mean()), int(m.sum())))

    # denominators
    eps_tagger = eps_cal                      # per signal-matched reco candidate
    full = None
    if args.n_generated:
        eff = n_cand / args.n_generated       # reco+matching efficiency proxy
        full = eps_tagger * eff

    print("=" * 60)
    print(f"CALIBRATED TAGGING POWER  [{args.label}]")
    print(f"csv: {args.csv}")
    print(f"signal-matched candidates: {n_cand}")
    print("-" * 60)
    print(f"raw flat mistag w           : {w_flat*100:.2f}%")
    print(f"raw eps_eff (uncalibrated)  : {eps_raw*100:.2f}%")
    print(f"CALIBRATED eps_eff (eval)   : {eps_cal*100:.2f} +/- {eps_cal_err*100:.2f}%   <-- tagger-level headline")
    if full is not None:
        print(f"reco+match efficiency       : {100*n_cand/args.n_generated:.2f}%  ({n_cand}/{args.n_generated})")
        print(f"FULL-CHAIN eps_eff          : {full*100:.3f}%   (per generated event)")
    print("-" * 60)
    print("reliability (eta_bin_center | mean_eta_cal | measured_wrong | n):")
    for c, e, w, n in reli:
        print(f"  {c:.2f} | {e:.3f} | {w:.3f} | {n}")
    print("=" * 60)
    print("RESULTS-TABLE ROW (label | eps_tag% | <D2>=eps_eff_raw% | eps_eff_cal% | full-chain%):")
    fc = f"{full*100:.3f}" if full is not None else "n/a"
    print(f"{args.label} | 100.0 | {eps_raw*100:.2f} | {eps_cal*100:.2f} | {fc}")


if __name__ == "__main__":
    main()
