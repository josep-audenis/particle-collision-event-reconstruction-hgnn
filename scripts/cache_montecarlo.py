"""Create model cache files from LHCb ``*.mc.root`` Monte Carlo tuples.

Pipeline:
1. Convert raw ``EvtTuple/Tracks`` tuples into generated-particle ROOT files.
2. Build particle/relation ROOT files with topology labels.
3. Cache PyG ``HeteroData`` graphs as ``.pt.zst`` files consumed by loaders.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import uproot
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_DIR = Path(__file__).resolve().parent / "montecarlo_cache_support"
FULL_SIM_DIR = SUPPORT_DIR / "full_sim_processing"
RELATIONS_DIR = SUPPORT_DIR / "create_particles_relations"
CACHE_DIR = SUPPORT_DIR / "cacheing"
REQUIRED_IMPORTS = (
    "awkward",
    "matplotlib",
    "mplhep",
    "numpy",
    "pandas",
    "scipy",
    "seaborn",
    "torch",
    "uproot",
    "yaml",
    "zstandard",
)


def check_dependencies() -> None:
    missing = []
    for module_name in REQUIRED_IMPORTS:
        try:
            __import__(module_name)
        except ImportError as exc:
            missing.append(f"{module_name}: {exc}")

    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise ImportError(
            "Missing Python dependencies for Monte Carlo cache pipeline:\n"
            f"{details}\n"
            "Install with: python -m pip install -r requirements.txt"
        )

    if importlib.util.find_spec("torch_geometric") is None:
        raise ImportError(
            "Missing Python dependency for Monte Carlo cache pipeline:\n"
            "  - torch_geometric\n"
            "Install with: python -m pip install -r requirements.txt"
        )


def _load_full_sim_helpers():
    helper_path = FULL_SIM_DIR / "helper.py"
    if not helper_path.exists():
        raise FileNotFoundError(
            f"Monte Carlo cache helper not found: {helper_path}. "
            "Keep scripts/montecarlo_cache_support with scripts/cache_montecarlo.py."
        )

    sys.path.insert(0, str(FULL_SIM_DIR))
    try:
        spec = importlib.util.spec_from_file_location("dfei_full_sim_helper", helper_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load helper spec from {helper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    return {
        "add_num_vis_daugh": module.add_num_vis_daugh,
        "add_primary_hh_index": module.add_primary_hh_index,
        "adjust_df": module.adjust_df,
        "get_all_heads": module.get_all_heads,
        "get_evt_df": module.get_evt_df,
        "get_graph_df": module.get_graph_df,
        "save_root_helper": module.save_root_helper,
    }


def infer_identifier(path: Path) -> str:
    return path.name.split("_", 1)[0]


def convert_mc_root(
    mc_file: Path,
    work_dir: Path,
    sample: str,
    identifier: str,
    chunk_size: int,
    event_limit: int | None,
    start_offset: int,
    overwrite: bool,
) -> tuple[list[Path], int]:
    helpers = _load_full_sim_helpers()
    out_dir = work_dir / "DFEI_LHCb_generator" / f"{identifier}_{sample}"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"GeneratedParticles_{sample}_*.root"))
    if existing and not overwrite:
        print(f"Using existing generated particle files in {out_dir}")
        return existing, 0

    arrays_iter = uproot.iterate(f"{mc_file}:EvtTuple/Tracks", library="ak", step_size=1000)
    result = []
    written: list[Path] = []
    evt_counter = 0
    chunk_start = start_offset
    track_types = ["long"]

    def flush(end_evt: int) -> None:
        nonlocal result, chunk_start
        if not result:
            chunk_start = end_evt + 1
            return
        tree_data, branch_types = helpers["save_root_helper"](result)
        outfile = out_dir / f"GeneratedParticles_{sample}_{chunk_start:08d}_{end_evt:08d}.root"
        with uproot.recreate(outfile) as f:
            f.mktree("Particles", branch_types, title="Particles Tree")
            f["Particles"].extend(tree_data)
        written.append(outfile)
        result = []
        chunk_start = end_evt + 1

    with tqdm(desc=f"Converting {mc_file.name}", total=event_limit, unit="evt") as bar:
        for arrays in arrays_iter:
            for event_idx in range(len(arrays)):
                if event_limit is not None and evt_counter >= event_limit:
                    flush(start_offset + evt_counter - 1)
                    return written, evt_counter

                event = arrays[event_idx]
                particle_df, mother_df, ok = helpers["get_evt_df"](event, track_types)
                if ok:
                    mother_df, _ = helpers["get_all_heads"](mother_df)
                    heavy_df = mother_df[mother_df["MotherIsbhadron"] == 1]
                    heavy_heads = heavy_df[heavy_df["isHead"] == True]["MotherIndex"].to_numpy(int)
                    graph_df = helpers["get_graph_df"](particle_df, mother_df)
                    particle_df, mother_df = helpers["add_primary_hh_index"](
                        particle_df, mother_df, graph_df, heavy_heads
                    )
                    particle_df, mother_df = helpers["add_num_vis_daugh"](
                        particle_df, mother_df, graph_df, heavy_heads
                    )
                    res_df = helpers["adjust_df"](particle_df, mother_df)
                    res_df["RUNNUMBER"] = event["RUNNUMBER"]
                    res_df["EVENTNUMBER"] = event["EVENTNUMBER"]
                    res_df["EventNumber"] = start_offset + evt_counter
                    result.append(res_df)

                evt_counter += 1
                bar.update(1)
                if evt_counter % chunk_size == 0:
                    flush(start_offset + evt_counter - 1)

    flush(start_offset + evt_counter - 1)
    return written, evt_counter


def run_support_script(script_dir: Path, script: str, args: list[str]) -> None:
    cmd = [sys.executable, script, *args]
    subprocess.run(cmd, cwd=script_dir, check=True)


def generated_sample_name(generated: Path) -> str:
    stem = generated.stem
    prefix = "GeneratedParticles_"
    if not stem.startswith(prefix):
        raise ValueError(f"Unexpected generated particle filename: {generated.name}")
    sample_and_range = stem[len(prefix):]
    return sample_and_range.rsplit("_", 2)[0]


def build_relations(generated_files: list[Path], work_dir: Path, identifier: str, overwrite: bool) -> list[Path]:
    relation_files: list[Path] = []
    for generated in generated_files:
        sample = generated_sample_name(generated)
        out_dir = work_dir / "DFEI_LHCb_particles_relations" / f"{identifier}_{sample}"
        stem = generated.stem
        start, end = stem.rsplit("_", 2)[1:]
        relation_file = out_dir / f"Event_data_relations_{sample}_LHCbcollision_{start}_to_{end}.root"
        if relation_file.exists() and not overwrite:
            relation_files.append(relation_file)
            continue

        run_support_script(
            RELATIONS_DIR,
            "main.py",
            [
                "--infile",
                generated.name,
                "--dir",
                str(work_dir),
                "--sim",
                "LHCb",
                "--identifier",
                identifier,
            ],
        )
        if not relation_file.exists():
            raise FileNotFoundError(f"Expected relation file missing after relation stage: {relation_file}")
        relation_files.append(relation_file)
    return sorted(set(relation_files))


def cache_graphs(
    relation_files: list[Path],
    work_dir: Path,
    cache_out: Path,
    identifier: str,
    normed: bool,
    mode: str,
) -> None:
    for relation_file in relation_files:
        particle_file = relation_file.name.replace("relations", "particles")
        args = [
            "--indir",
            str(work_dir / "DFEI_LHCb_particles_relations"),
            "--outdir",
            str(cache_out) + os.sep,
            "--file",
            particle_file,
            "--identifier",
            identifier,
            "--mode",
            mode,
        ]
        if normed:
            args.append("--normed")
        run_support_script(CACHE_DIR, "main.py", args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO_ROOT / "data" / "montecarlo")
    parser.add_argument("--input-glob", default="*.mc.root", help="Input ROOT glob relative to input-dir.")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "data" / "montecarlo_work")
    parser.add_argument("--cache-out", type=Path, default=REPO_ROOT / "data" / "cached_montecarlo")
    parser.add_argument("--sample", default="inclusive")
    parser.add_argument("--identifier", default=None, help="Override identifier; default from filename prefix")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum raw events to process across all input files.",
    )
    parser.add_argument(
        "--max-events-per-file",
        type=int,
        default=None,
        help="Maximum raw events to process from each input file before moving to next file.",
    )
    parser.add_argument("--mode", choices=["default", "trn_only", "tst_only"], default="default")
    parser.add_argument("--no-norm", action="store_true", help="Disable graph normalization")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    check_dependencies()
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.cache_out = args.cache_out.resolve()

    mc_files = sorted(args.input_dir.glob(args.input_glob))
    if not mc_files:
        raise FileNotFoundError(f"No files matching {args.input_glob!r} found in {args.input_dir}")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.cache_out.mkdir(parents=True, exist_ok=True)

    next_start_by_sample: dict[tuple[str, str], int] = {}
    total_processed_events = 0
    for mc_file in mc_files:
        if args.max_events is not None and total_processed_events >= args.max_events:
            break

        identifier = args.identifier or infer_identifier(mc_file)
        key = (identifier, args.sample)
        start_offset = next_start_by_sample.get(key, 0)
        event_limit = args.max_events_per_file
        if args.max_events is not None:
            remaining = args.max_events - total_processed_events
            event_limit = remaining if event_limit is None else min(event_limit, remaining)

        generated, processed_events = convert_mc_root(
            mc_file=mc_file,
            work_dir=args.work_dir,
            sample=args.sample,
            identifier=identifier,
            chunk_size=args.chunk_size,
            event_limit=event_limit,
            start_offset=start_offset,
            overwrite=args.overwrite,
        )
        next_start_by_sample[key] = start_offset + processed_events
        total_processed_events += processed_events
        relations = build_relations(generated, args.work_dir, identifier, args.overwrite)
        cache_graphs(
            relations,
            work_dir=args.work_dir,
            cache_out=args.cache_out,
            identifier=identifier,
            normed=not args.no_norm,
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
