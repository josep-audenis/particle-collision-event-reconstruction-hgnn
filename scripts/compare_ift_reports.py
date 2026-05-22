from __future__ import annotations

import re
from pathlib import Path


def parse_ft(path: Path) -> dict[str, str]:
    text = path.read_text(errors="ignore")
    out = {}
    for key, pattern in {
        "candidates": r"Tagging performance for .*\((\d+)\):",
        "tagging_power": r"Tagging power\s+:\s+\(([^)]+)\)%",
        "wrong_fraction": r"Wrong fraction\s+:\s+\(([^)]+)\)%",
        "epsilon": r"Epsilon\s+:\s+\(([^)]+)\)%",
        "dsquared": r"Dsquared\s+:\s+\(([^)]+)\)%",
        "rows": r"Rows\s+:\s+(\d+)",
        "auc": r"AUC\s+:\s+([0-9.]+|nan)",
        "loss": r"Loss\s+:\s+([0-9.]+|nan)",
        "input_rows": r"Input rows\s+:\s+(\d+)",
        "processed_rows": r"Processed rows\s+:\s+(\d+)",
        "full_metric_rows": r"Full metric rows:\s+(\d+)",
    }.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else ""
    out["nan_count"] = str(text.lower().count("nan"))
    return out


def version_number(path: Path) -> int:
    match = re.search(r"version_(\d+)", str(path))
    return int(match.group(1)) if match else -1


root = Path("LHCb_logs/IFT")
paths = sorted(root.glob("version_*/info_*_FT.txt"), key=lambda p: (version_number(p), p.name))

print(
    "version,file,candidates,tagging_power,wrong_fraction,epsilon,auc,loss,"
    "input_rows,processed_rows,full_metric_rows,nan_count"
)
for path in paths:
    ft = parse_ft(path)
    print(
        f"{version_number(path)},{path.name},{ft['candidates']},{ft['tagging_power']},"
        f"{ft['wrong_fraction']},{ft['epsilon']},{ft['auc']},{ft['loss']},"
        f"{ft['input_rows']},{ft['processed_rows']},{ft['full_metric_rows']},{ft['nan_count']}"
    )
