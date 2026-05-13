import uproot

import argparse
from tqdm import tqdm

from helper import *

if __name__ == "__main__":
    # python main.py --file /eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_LHCb_davinci/DFEI_tuple_00258329_inclusive.root --outdir /eos/lhcb/user/y/yukaiz/DFEI_data
    parser = argparse.ArgumentParser(description='Process LHCb simulation to adapt to PYTHIA file')
    parser.add_argument('--file', type=str, default=None,
                        help="Input file holding the LHCb tupled file")
    parser.add_argument('--outdir', type=str, default=None,
                        help="Output directory holding the adapted LHCb tupled to pythia file")
    parser.add_argument('--sample', type=str, default="None",
                        help="Decay channel of interest")
    parser.add_argument('--identifier', type=str, default="0000000",
                        help="production number of the decay channel")
    parser.add_argument('--start', type=int, default=0, )  # start file name
    parser.add_argument("--custom_start", type=int, default=0)  # skipping some evts
    args = parser.parse_args()

    outdir = args.outdir + "/DFEI_LHCb_generator/" + args.identifier + "_" + args.sample + "/"
    os.makedirs(outdir, exist_ok=True)

    # If we are looking at exclusive decays require them to pass trigger selection
    tree = uproot.iterate(f"{args.file}:EvtTuple/Tracks", library="ak", step_size=1000)
    track_types = ["long"]

    result = []
    evt_counter = 0
    file_counter = 0
    entries_per_file = 1000

    while True:
        try:
            arrays = next(tree)
            print(file_counter)
            if evt_counter < args.custom_start:
                file_counter += 1
                evt_counter += 1000
                continue
        except StopIteration:
            break

        for event_idx in tqdm(range(len(arrays)), desc="Processing events", leave=False):

            event = arrays[event_idx]
            # Getting the mother and particle df
            particle_df, mother_df, flag = get_evt_df(event, track_types)
            if flag:
                # Obtaining the head of the decay
                mother_df, _ = get_all_heads(mother_df)
                heavy_df = mother_df[mother_df["MotherIsbhadron"] == 1]
                heavy_hadron_heads = heavy_df[heavy_df["isHead"] == True]["MotherIndex"].to_numpy(int)

                # Creating a merged DataFrame of reconstructed and MC truth mothers
                graph_df = get_graph_df(particle_df, mother_df)

                # Associate Primary heavy hadron index
                particle_df, mother_df = add_primary_hh_index(particle_df, mother_df, graph_df, heavy_hadron_heads)

                # Now we determine the NumVisDaugh quantity
                particle_df, mother_df = add_num_vis_daugh(particle_df, mother_df, graph_df, heavy_hadron_heads)

                res_df = adjust_df(particle_df, mother_df)

                # Adding identifiers
                res_df["RUNNUMBER"] = event["RUNNUMBER"]
                res_df["EVENTNUMBER"] = event["EVENTNUMBER"]
                res_df["EventNumber"] = evt_counter

                result.append(res_df)
            evt_counter += 1

            # Check if we've accumulated enough entries to save
            if evt_counter % entries_per_file == 0:
                start_evt = (file_counter * entries_per_file) + args.start
                end_evt = evt_counter - 1 + args.start

                # Save the file
                tree_data, branch_types = save_root_helper(result)
                outfile = f"GeneratedParticles_{args.sample}_{str(start_evt).zfill(8)}_{str(end_evt).zfill(8)}.root"
                with uproot.recreate(f"{outdir}/{outfile}") as file_particles:
                    file_particles.mktree("Particles", branch_types, title="Particles Tree")
                    file_particles["Particles"].extend(tree_data)

                # Reset for next file
                result = []
                file_counter += 1

# inclusive
# python new_main.py --file /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00353851/0000/00353851_00000001_1.mc.root --outdir /eos/lhcb/user/y/yukaiz/DFEI_data --sample inclusive --identifier 00342442
# python new_main.py --file /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2024/MC.ROOT/00353855/0000/00353855_00000005_1.mc.root --outdir /eos/lhcb/user/y/yukaiz/DFEI_data --sample inclusive --identifier 00342451

# Bu->Jpsi K
# python new_main.py --file /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2025/MC.ROOT/00361939/0000/00361939_00000001_1.mc.root --outdir /eos/lhcb/user/y/yukaiz/DFEI_data --sample Bu_JpsiK --identifier 00338678
# python new_main.py --file /eos/lhcb/grid/prod/lhcb/anaprod/lhcb/MC/2025/MC.ROOT/00361941/0000/00361941_00000001_1.mc.root --outdir /eos/lhcb/user/y/yukaiz/DFEI_data --sample Bu_JpsiK --identifier 00338660
