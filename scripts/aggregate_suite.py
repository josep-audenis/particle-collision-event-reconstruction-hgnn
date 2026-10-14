"""Aggregate the experiment-suite results TSV into mean +/- std per (exp, channel).

Reads the append-only results.tsv written by suite_record.py and prints a summary
table: for each (exp, channel), the seed-averaged flat TP, full-chain TP, and AUC
with sample std, plus delta of full-chain vs the 'baseline' exp for the same channel.

Usage: python scripts/aggregate_suite.py outputs/suite/results.tsv
"""
import sys, csv
from collections import defaultdict
from statistics import mean, stdev


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs/suite/results.tsv"
    rows = list(csv.DictReader(open(path), delimiter="\t"))
    if not rows:
        print("no results yet"); return

    groups = defaultdict(list)
    for r in rows:
        groups[(r["exp"], r["channel"])].append(r)

    def agg(vals):
        vals = [v for v in vals if v == v]
        if not vals:
            return float("nan"), float("nan")
        return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0)

    # baseline full-chain per channel for delta
    base = {}
    for (exp, ch), rs in groups.items():
        if exp == "baseline":
            base[ch] = agg([float(r["full_chain"]) for r in rs])[0]

    print(f"{'exp':<16}{'ch':<4}{'n':<3}{'flat_TP':>16}{'full_chain':>18}{'AUC':>14}{'d_fc_vs_base':>14}")
    print("-" * 85)
    for (exp, ch) in sorted(groups.keys()):
        rs = groups[(exp, ch)]
        fm, fs = agg([float(r["flat_tp"]) for r in rs])
        cm, cs = agg([float(r["full_chain"]) for r in rs])
        am, as_ = agg([float(r["auc"]) for r in rs])
        seeds = ",".join(sorted(r["seed"] for r in rs))
        dfc = cm - base[ch] if ch in base else float("nan")
        print(f"{exp:<16}{ch:<4}{len(rs):<3}{fm:>7.2f}+/-{fs:<5.2f}{cm:>9.2f}+/-{cs:<5.2f}"
              f"{am:>8.3f}+/-{as_:<4.3f}{dfc:>+13.2f}")
    print("-" * 85)
    print("flat_TP, full_chain in %, per seed then averaged. d_fc = full-chain minus this channel's baseline.")


if __name__ == "__main__":
    main()
