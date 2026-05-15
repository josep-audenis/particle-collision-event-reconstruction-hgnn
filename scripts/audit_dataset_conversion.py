#!/usr/bin/env python
"""Audit cached Monte Carlo graph conversion and reconstruction outputs.

This script checks invariants that should hold after ROOT -> relation ROOT ->
PyG cache conversion. It is intentionally read-only.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import zstandard as zstd


TT_KEYS = [
    ("tracks", "to", "tracks"),
    ("tracks", "tracks"),
]
TPV_KEYS = [
    ("tracks", "to", "pvs"),
    ("tracks", "pvs"),
]


def as_list(x: Any) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def load_cache_file(path: Path) -> list:
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        with path.open("rb") as f:
            with dctx.stream_reader(f) as reader:
                payload = reader.read()
        data = torch.load(io.BytesIO(payload), weights_only=False, map_location="cpu")
    else:
        data = torch.load(path, weights_only=False, map_location="cpu")
    return as_list(data)


def first_existing_store(graph: Any, keys: list[tuple[str, ...]]) -> Any | None:
    for key in keys:
        try:
            return graph[key]
        except Exception:
            continue
    return None


def tensor_shape(x: Any) -> list[int] | None:
    if x is None:
        return None
    if hasattr(x, "shape"):
        return list(x.shape)
    return None


def scalar_int(x: Any) -> int | None:
    try:
        if torch.is_tensor(x):
            return int(x.flatten()[0].item())
        return int(x)
    except Exception:
        return None


def graph_keys(graph: Any) -> list[str]:
    out = []
    try:
        out.extend(str(k) for k in graph.keys())
    except Exception:
        pass
    try:
        out.extend(str(k) for k in graph.edge_types)
    except Exception:
        pass
    try:
        out.extend(str(k) for k in graph.node_types)
    except Exception:
        pass
    return sorted(set(out))


def audit_graph(graph: Any, source: str, index: int) -> tuple[list[dict], dict]:
    issues: list[dict] = []
    stats: dict[str, Any] = {"source": source, "index": index}

    tracks = graph["tracks"] if "tracks" in graph.node_types else None
    pvs = graph["pvs"] if "pvs" in graph.node_types else None
    tt = first_existing_store(graph, TT_KEYS)
    tpv = first_existing_store(graph, TPV_KEYS)

    n_tracks = int(tracks.x.shape[0]) if tracks is not None and hasattr(tracks, "x") else None
    n_pvs = int(pvs.x.shape[0]) if pvs is not None and hasattr(pvs, "x") else None
    stats["n_tracks"] = n_tracks
    stats["n_pvs"] = n_pvs
    stats["keys"] = graph_keys(graph)

    event_number = scalar_int(getattr(graph, "EVENTNUMBER", None) if hasattr(graph, "EVENTNUMBER") else graph.get("EVENTNUMBER", None))
    run_number = scalar_int(getattr(graph, "RUNNUMBER", None) if hasattr(graph, "RUNNUMBER") else graph.get("RUNNUMBER", None))
    stats["EVENTNUMBER"] = event_number
    stats["RUNNUMBER"] = run_number
    if event_number is None or run_number is None:
        issues.append({"severity": "warning", "check": "event_identity_present", "detail": "Missing EVENTNUMBER/RUNNUMBER on graph; log-to-truth joins and duplicate checks cannot be verified."})

    if n_tracks is None or n_tracks <= 0:
        issues.append({"severity": "error", "check": "tracks_present", "detail": "No track nodes found."})
    if n_pvs is None or n_pvs <= 0:
        issues.append({"severity": "error", "check": "pvs_present", "detail": "No PV nodes found."})

    if tracks is not None and hasattr(tracks, "ft"):
        ft = tracks.ft.detach().cpu().to(torch.int64)
        counts = torch.bincount(ft, minlength=max(3, int(ft.max().item()) + 1 if ft.numel() else 3))
        stats["ft_counts"] = counts.tolist()
        bad = ft[(ft < 0) | (ft > 2)]
        if bad.numel():
            issues.append({"severity": "error", "check": "ft_range", "detail": f"FT labels outside 0..2: {bad.unique().tolist()}"})
        if counts.numel() >= 3 and int(counts[0] + counts[2]) < 2:
            issues.append({"severity": "warning", "check": "signal_tracks", "detail": "Fewer than two signal tracks in cached graph."})
    else:
        issues.append({"severity": "error", "check": "ft_present", "detail": "Missing tracks.ft labels."})

    if tt is not None:
        edge_index = getattr(tt, "edge_index", None)
        y = getattr(tt, "y", None)
        stats["tt_edge_index_shape"] = tensor_shape(edge_index)
        stats["tt_y_shape"] = tensor_shape(y)
        if edge_index is None or edge_index.numel() == 0:
            issues.append({"severity": "error", "check": "tt_edges_present", "detail": "No track-track edges."})
        elif n_tracks is not None:
            if int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= n_tracks:
                issues.append({"severity": "error", "check": "tt_edge_index_bounds", "detail": "Track-track edge index outside track range."})
        if y is None:
            issues.append({"severity": "error", "check": "tt_y_present", "detail": "Missing LCA labels on track-track edges."})
        else:
            y64 = y.detach().cpu().to(torch.int64).flatten()
            stats["lca_counts"] = torch.bincount(y64, minlength=max(4, int(y64.max().item()) + 1 if y64.numel() else 4)).tolist()
            bad = y64[(y64 < 0) | (y64 > 3)]
            if bad.numel():
                issues.append({"severity": "error", "check": "lca_range", "detail": f"LCA labels outside 0..3: {bad.unique().tolist()}"})
            if edge_index is not None and y64.numel() != edge_index.shape[1]:
                issues.append({"severity": "error", "check": "tt_label_count", "detail": "LCA label count differs from track-track edge count."})
    else:
        issues.append({"severity": "error", "check": "tt_store_present", "detail": "Missing track-track edge store."})

    if tpv is not None:
        edge_index = getattr(tpv, "edge_index", None)
        y = getattr(tpv, "y", None)
        filt = getattr(tpv, "filter", None)
        stats["tpv_edge_index_shape"] = tensor_shape(edge_index)
        stats["tpv_y_shape"] = tensor_shape(y)
        stats["tpv_filter_shape"] = tensor_shape(filt)
        if edge_index is None or edge_index.numel() == 0:
            issues.append({"severity": "error", "check": "tpv_edges_present", "detail": "No track-PV edges."})
        elif n_tracks is not None and n_pvs is not None:
            expected = n_tracks * n_pvs
            if edge_index.shape[1] != expected:
                issues.append({"severity": "error", "check": "tpv_complete_bipartite", "detail": f"Expected {expected} track-PV edges, got {edge_index.shape[1]}."})
            if int(edge_index[0].min().item()) < 0 or int(edge_index[0].max().item()) >= n_tracks:
                issues.append({"severity": "error", "check": "tpv_track_bounds", "detail": "Track-PV track index outside range."})
            if int(edge_index[1].min().item()) < 0 or int(edge_index[1].max().item()) >= n_pvs:
                issues.append({"severity": "error", "check": "tpv_pv_bounds", "detail": "Track-PV PV index outside range."})
        if y is None:
            issues.append({"severity": "error", "check": "tpv_y_present", "detail": "Missing PV association labels."})
        elif n_tracks is not None and n_pvs is not None:
            y2 = y.detach().cpu().view(n_tracks, n_pvs)
            per_track_sum = y2.sum(dim=1)
            stats["pv_positive_tracks"] = int((per_track_sum == 1).sum().item())
            stats["pv_unmatched_tracks"] = int((per_track_sum == 0).sum().item())
            stats["pv_multi_positive_tracks"] = int((per_track_sum > 1).sum().item())
            if (per_track_sum > 1).any():
                issues.append({"severity": "error", "check": "pv_one_hot", "detail": "Some tracks have multiple true PV labels."})
            if (per_track_sum == 1).sum().item() == 0:
                issues.append({"severity": "error", "check": "pv_any_labels", "detail": "No tracks have a true PV label."})
        if filt is not None and y is not None:
            f = filt.detach().cpu().flatten()
            yf = y.detach().cpu().flatten()
            if f.numel() != yf.numel():
                issues.append({"severity": "error", "check": "pv_filter_count", "detail": "PV filter count differs from PV label count."})
    else:
        issues.append({"severity": "error", "check": "tpv_store_present", "detail": "Missing track-PV edge store."})

    for key in [
        "final_keys",
        "truth_part_keys",
        "part_ids",
        "ids",
        "init_part_ids",
        "init_ids",
        "truth_part_ids",
        "truth_ids",
        "truth_y",
        "truth_senders",
        "truth_receivers",
    ]:
        value = graph.get(key, None)
        if value is not None:
            stats[f"{key}_shape"] = tensor_shape(value)

    required_reco_keys = ["final_keys", "truth_part_keys", "part_ids", "truth_part_ids", "truth_y", "truth_senders", "truth_receivers"]
    missing_reco_keys = [key for key in required_reco_keys if graph.get(key, None) is None]
    if missing_reco_keys:
        issues.append({
            "severity": "error",
            "check": "reconstruction_runtime_keys_present",
            "detail": f"Missing keys used by EventReconstruction: {missing_reco_keys}.",
        })

    paper_eval_keys = ["ids", "init_part_ids", "init_ids", "truth_ids"]
    missing_paper_eval_keys = [key for key in paper_eval_keys if graph.get(key, None) is None]
    if missing_paper_eval_keys:
        stats["missing_optional_paper_eval_keys"] = missing_paper_eval_keys

    if graph.get("truth_senders", None) is not None and graph.get("truth_receivers", None) is not None:
        senders = graph["truth_senders"].detach().cpu()
        receivers = graph["truth_receivers"].detach().cpu()
        if senders.numel() != receivers.numel():
            issues.append({"severity": "error", "check": "truth_edge_count", "detail": "truth_senders and truth_receivers lengths differ."})
        if n_tracks is not None and senders.numel():
            if int(senders.min().item()) < 0 or int(receivers.min().item()) < 0 or int(max(senders.max().item(), receivers.max().item())) >= n_tracks:
                issues.append({"severity": "error", "check": "truth_edge_bounds", "detail": "Truth edge index outside track range."})

    return issues, stats


def audit_cache(cache_dir: Path, sample: str, max_files: int, max_graphs: int, split: str) -> dict:
    sample_dir = cache_dir / sample
    report = {"cache_dir": str(cache_dir), "sample": sample, "split": split, "files": [], "issues": [], "stats": []}
    if not sample_dir.exists():
        report["issues"].append({"severity": "error", "check": "cache_path_exists", "detail": f"Missing {sample_dir}"})
        return report

    prefixes = ["trn", "val", "tst"] if split == "all" else [split]
    files = []
    for prefix in prefixes:
        split_files = sorted(sample_dir.glob(f"{prefix}_data_*"))
        split_files = [p for p in split_files if p.suffix in {".zst", ".pt"} or p.name.endswith(".pt.zst")]
        files.extend(split_files[:max_files])
    report["files"] = [str(p) for p in files]
    if not files:
        report["issues"].append({"severity": "error", "check": "cache_files_exist", "detail": f"No trn/val/tst data files found in {sample_dir}"})
        return report

    seen_events = Counter()
    n_graphs = 0
    for path in files[:max_files]:
        try:
            graphs = load_cache_file(path)
        except Exception as exc:
            report["issues"].append({"severity": "error", "check": "load_cache_file", "detail": f"{path}: {exc}"})
            continue
        for i, graph in enumerate(graphs):
            if n_graphs >= max_graphs:
                break
            issues, stats = audit_graph(graph, str(path), i)
            report["issues"].extend(dict(issue, source=str(path), graph_index=i) for issue in issues)
            report["stats"].append(stats)
            if stats.get("EVENTNUMBER") is not None and stats.get("RUNNUMBER") is not None:
                seen_events[(stats["RUNNUMBER"], stats["EVENTNUMBER"])] += 1
            n_graphs += 1
        if n_graphs >= max_graphs:
            break

    duplicates = [f"{r}:{e} x{n}" for (r, e), n in seen_events.items() if n > 1]
    if duplicates:
        report["issues"].append({"severity": "error", "check": "duplicate_event_ids", "detail": duplicates[:20]})
    optional_missing = Counter(
        tuple(stats.get("missing_optional_paper_eval_keys", []))
        for stats in report["stats"]
        if stats.get("missing_optional_paper_eval_keys")
    )
    for keys, count in optional_missing.items():
        report["issues"].append({
            "severity": "warning",
            "check": "paper_eval_truth_keys_present",
            "detail": f"{count} graphs missing optional paper-style truth keys: {list(keys)}.",
        })
    report["num_graphs_checked"] = n_graphs
    return report


def audit_result_csvs(log_dir: Path) -> dict:
    if not log_dir.exists():
        case_variants = [
            Path("LHCb_logs"),
            Path("LHCB_logs"),
            Path("lhcb_logs"),
        ]
        for candidate in case_variants:
            if candidate.exists():
                log_dir = candidate
                break

    report = {"log_dir": str(log_dir), "files": {}, "issues": []}
    if not log_dir.exists():
        report["issues"].append({"severity": "warning", "check": "log_dir_exists", "detail": f"Missing {log_dir}"})
        return report

    for path in sorted(log_dir.rglob("*.csv")):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            report["issues"].append({"severity": "error", "check": "read_csv", "detail": f"{path}: {exc}"})
            continue
        rel = str(path)
        file_report: dict[str, Any] = {"rows": int(len(df)), "columns": list(df.columns)}
        report["files"][rel] = file_report

        for col in ["NumBackgroundParticlesInEvent", "NumBkgParticles_noniso"]:
            if col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce")
                file_report[col] = {
                    "min": float(numeric.min()),
                    "mean": float(numeric.mean()),
                    "negative_rows": int((numeric < 0).sum()),
                    "sentinel_minus_999_rows": int((numeric == -999).sum()),
                }
                bad = numeric[(numeric < 0) & (numeric != -999)]
                if len(bad):
                    report["issues"].append({"severity": "error", "check": f"{col}_nonnegative", "detail": f"{path}: {len(bad)} rows negative excluding -999 sentinel."})

        for col in ["PerfectReco", "AllParticles", "NoneIso", "PartReco", "NotFound", "SigMatch"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                file_report[col] = {"mean": float(vals.mean()), "sum": float(vals.sum()), "nan_rows": int(vals.isna().sum())}

        if {"EVENTNUMBER", "RUNNUMBER"}.issubset(df.columns):
            keys = df[["RUNNUMBER", "EVENTNUMBER"]].astype(str).agg(":".join, axis=1)
            file_report["unique_event_keys"] = int(keys.nunique())
            file_report["duplicate_event_key_rows"] = int(keys.duplicated().sum())

        if {"ft_b_score", "ft_bbar_score"}.issubset(df.columns):
            b = pd.to_numeric(df["ft_b_score"], errors="coerce")
            bbar = pd.to_numeric(df["ft_bbar_score"], errors="coerce")
            file_report["ft_score_summary"] = {
                "b_min": float(b.min()),
                "b_max": float(b.max()),
                "bbar_min": float(bbar.min()),
                "bbar_max": float(bbar.max()),
                "classified_rows": int(((b > 0.5) | (bbar > 0.5)).sum()),
            }
            if not (((b > 0.5) | (bbar > 0.5)).any()):
                report["issues"].append({"severity": "error", "check": "ft_classified_rows", "detail": f"{path}: no rows pass >0.5 FT decision."})

        if "part_ids" in df.columns:
            empty = df["part_ids"].isna() | (df["part_ids"].astype(str).str.len() == 0)
            file_report["part_ids_empty_rows"] = int(empty.sum())
            if empty.all():
                report["issues"].append({"severity": "error", "check": "part_ids_present", "detail": f"{path}: part_ids empty for every row."})

    for info_path in sorted(log_dir.rglob("info_*FT.txt")):
        text = info_path.read_text(errors="replace")
        zero_sections = text.count("(0):") + text.count("Events: 0")
        nan_count = text.lower().count("nan")
        report["files"][str(info_path)] = {"nan_count": nan_count, "zero_count_markers": zero_sections}
        if nan_count:
            report["issues"].append({"severity": "error", "check": "ft_info_nan", "detail": f"{info_path}: {nan_count} nan markers in tagging report."})
    return report


def audit_paper_dataset(paper_dir: Path, max_events: int) -> dict:
    report = {"paper_dir": str(paper_dir), "files": [], "issues": [], "stats": []}
    if not paper_dir.exists():
        report["issues"].append({"severity": "warning", "check": "paper_dir_exists", "detail": f"Missing {paper_dir}"})
        return report

    inputs = sorted(paper_dir.glob("input_*.npy"))[:max_events]
    if not inputs:
        report["issues"].append({"severity": "warning", "check": "paper_inputs_exist", "detail": f"No input_*.npy files in {paper_dir}"})
        return report

    for inp in inputs:
        idx = inp.stem.split("_", 1)[1]
        target = inp.with_name(f"target_{idx}.npy")
        try:
            data = np.load(inp, allow_pickle=True).item()
            tgt = np.load(target, allow_pickle=True).item() if target.exists() else None
        except Exception as exc:
            report["issues"].append({"severity": "error", "check": "paper_load", "detail": f"{inp}: {exc}"})
            continue
        stat = {"input": str(inp), "target": str(target) if target.exists() else None}
        for key, value in data.items():
            if hasattr(value, "shape"):
                stat[f"input_{key}_shape"] = list(value.shape)
        if tgt is not None:
            for key, value in tgt.items():
                if hasattr(value, "shape"):
                    stat[f"target_{key}_shape"] = list(value.shape)
            if "edges" in data and "edges" in tgt and data["edges"].shape[0] != tgt["edges"].shape[0]:
                report["issues"].append({"severity": "error", "check": "paper_edge_target_count", "detail": f"{inp}: input edges and target edges differ."})
        if "nodes" in data and data["nodes"].shape[1] < 13:
            report["issues"].append({"severity": "error", "check": "paper_node_feature_count", "detail": f"{inp}: expected >=13 node features."})
        report["stats"].append(stat)
        report["files"].append(str(inp))
    return report


def summarize(report: dict) -> str:
    lines = ["# Dataset Conversion Audit", ""]
    for section_name in ["cache", "logs", "paper"]:
        section = report.get(section_name, {})
        issues = section.get("issues", [])
        counts = Counter(issue["severity"] for issue in issues)
        lines.append(f"## {section_name}")
        lines.append(f"- errors: {counts.get('error', 0)}")
        lines.append(f"- warnings: {counts.get('warning', 0)}")
        if section_name == "cache":
            lines.append(f"- graphs checked: {section.get('num_graphs_checked', 0)}")
        elif section_name == "logs":
            lines.append(f"- files checked: {len(section.get('files', {}))}")
        elif section_name == "paper":
            lines.append(f"- events checked: {len(section.get('stats', []))}")
        for issue in issues[:30]:
            lines.append(f"- [{issue['severity']}] {issue['check']}: {issue['detail']}")
        if len(issues) > 30:
            lines.append(f"- ... {len(issues) - 30} more issues")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/montecarlo/cache_50000/DFEI_LHCb_LHCbcollision_normed_pt_data"))
    parser.add_argument("--sample", default="00353851_inclusive")
    parser.add_argument("--log-dir", type=Path, default=Path("LHCB_logs"))
    parser.add_argument("--paper-dir", type=Path, default=Path("data/paper/inclusive_test_dataset/truth_inclusive_10k_49193/test_dataset"))
    parser.add_argument("--max-files", type=int, default=2)
    parser.add_argument("--max-graphs", type=int, default=200)
    parser.add_argument("--max-paper-events", type=int, default=20)
    parser.add_argument("--split", choices=["all", "trn", "val", "tst"], default="all")
    parser.add_argument("--out-json", type=Path, default=Path("outputs/dataset_conversion_audit.json"))
    parser.add_argument("--out-md", type=Path, default=Path("outputs/dataset_conversion_audit.md"))
    args = parser.parse_args()

    report = {
        "cache": audit_cache(args.cache_dir, args.sample, args.max_files, args.max_graphs, args.split),
        "logs": audit_result_csvs(args.log_dir),
        "paper": audit_paper_dataset(args.paper_dir, args.max_paper_events),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(summarize(report), encoding="utf-8")
    print(summarize(report))
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 1 if any(i["severity"] == "error" for s in report.values() for i in s.get("issues", [])) else 0


if __name__ == "__main__":
    raise SystemExit(main())
