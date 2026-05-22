from pathlib import Path

import uproot


for path in sorted(Path("data/montecarlo/root").glob("mc24_*.root")):
    with uproot.open(path) as root_file:
        tree = root_file["EvtTuple/Tracks"]
        print(f"{path.name},{tree.num_entries},{path.stat().st_size}")
