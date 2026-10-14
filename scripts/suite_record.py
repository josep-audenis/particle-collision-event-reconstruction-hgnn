"""Parse one finished experiment-suite run and append a result row to the TSV.

Given the run's stdout log (to find the trained IFT version + selected thresholds)
and the (exp, channel, seed) identity, locate the final-test FT file, parse the
flat tagging power / rows / events / AUC, compute full-chain TP = flat * rows/events,
and append a tab-separated row. Idempotent-ish: caller gates on a DONE marker.

Usage:
  python scripts/suite_record.py --results outputs/suite/results.tsv \
      --exp baseline --channel bd --seed 42 --log outputs/suite/baseline_bd_s42.log
  # or point directly at an FT file (for offline validation):
  python scripts/suite_record.py --results /tmp/r.tsv --exp x --channel bs --seed 42 \
      --ft LHCb_logs/IFT/version_25/info_mc24_Bs_Jpsiphi__nopid_bd_test_final_FT.txt --ift-ver 25
"""
import argparse, os, re, sys

SAMPLE = {"bd": "mc24_Bd_JpsiKst", "bs": "mc24_Bs_Jpsiphi"}
HEADER = ["exp", "channel", "seed", "ift_ver", "edge", "node",
          "flat_tp", "full_chain", "auc", "rows", "events"]


def parse_log(logpath):
    """Return (ift_ver, edge, node) from a runner stdout log."""
    ift_ver, edge, node = None, None, None
    if logpath and os.path.exists(logpath):
        t = open(logpath, errors="ignore").read()
        m = re.findall(r"IFT_VERSION=(\d+)", t)
        if m:
            ift_ver = m[-1]
        s = re.findall(r"Selected thresholds:\s*edge=([0-9.]+)\s*node=([0-9.]+)", t)
        if s:
            edge, node = s[-1]
    return ift_ver, edge, node


def parse_ft(ftpath):
    """Return (flat_tp, rows, events, auc) from a final-test FT file."""
    t = open(ftpath, errors="ignore").read()
    flat = float(re.search(r"Tagging power\s+:\s+\(([-0-9.]+)", t).group(1))
    rows = int(re.search(r"Full metric rows:\s*([0-9]+)", t).group(1))
    events = int(re.search(r"Input events\s*:\s*([0-9]+)", t).group(1))
    auc_m = re.search(r"AUC\s*:\s*([0-9.]+)", t)
    auc = float(auc_m.group(1)) if auc_m else float("nan")
    return flat, rows, events, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--channel", required=True, choices=["bd", "bs"])
    ap.add_argument("--seed", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--ft", default=None, help="override FT path (else constructed from log's IFT version)")
    ap.add_argument("--ift-ver", default=None, help="override IFT version (else from log)")
    args = ap.parse_args()

    ift_ver, edge, node = parse_log(args.log)
    if args.ift_ver:
        ift_ver = args.ift_ver

    ft = args.ft
    if ft is None:
        if ift_ver is None:
            sys.exit("ERROR: no IFT version (log missing IFT_VERSION=) and no --ft given")
        tag = f"{args.exp}_{args.channel}_s{args.seed}"
        ft = f"LHCb_logs/IFT/version_{ift_ver}/info_{SAMPLE[args.channel]}__{tag}_test_final_FT.txt"
    if not os.path.exists(ft):
        sys.exit(f"ERROR: FT file not found: {ft}")

    flat, rows, events, auc = parse_ft(ft)
    full_chain = flat * rows / events if events else float("nan")

    if not os.path.exists(args.results):
        with open(args.results, "w") as f:
            f.write("\t".join(HEADER) + "\n")
    row = [args.exp, args.channel, args.seed, str(ift_ver), str(edge), str(node),
           f"{flat:.4f}", f"{full_chain:.4f}", f"{auc:.5f}", str(rows), str(events)]
    with open(args.results, "a") as f:
        f.write("\t".join(row) + "\n")
    print("RECORDED:", "\t".join(row))


if __name__ == "__main__":
    main()
