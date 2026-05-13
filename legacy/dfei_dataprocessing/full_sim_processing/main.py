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
    parser.add_argument('--bunch', type=int, default=1000,
                        help="Number of events stored per bunch")
    args = parser.parse_args()

    start = 0
    end = 999999
    identifier, sample = get_file_information(args.file)
    bunches = [(i, min(i + args.bunch, end + 1)) for i in range(start, end + 1, args.bunch)]
    outdir = args.outdir + "/DFEI_LHCb_generator/" + identifier + "_" + sample + "/"
    os.makedirs(outdir, exist_ok=True)

    rerun_bunches = None  # [(20000, 21000), (38000, 39000), (66000, 67000)]
    save_flag = False
    is_exclusive = "inclusive" not in args.file

    # If we are looking at exclusive decays require them to pass trigger selection
    tree = uproot.iterate(f"{args.file}:EvtTuple/Tracks", library="ak", step_size=1)
    track_types = ["long"]

    if not is_exclusive:
        for bunch in tqdm(bunches, total=len(bunches), desc="Processing bunches"):
            result = []
            if rerun_bunches is not None:
                save_flag = bunch in rerun_bunches

            for evt_counter in tqdm(range(*bunch), total=1000, desc="Processing evt counter"):
                arrays = next(tree)
                if rerun_bunches is not None:
                    if not save_flag:  # default false -> continue
                        continue

                # Getting the mother and particle df
                particle_df, mother_df = get_evt_df(arrays, track_types)

                # Obtaining the head of the decay
                mother_df, _ = get_all_heads(mother_df)  # gives all heads if we do not only care about b decay
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
                res_df["RUNNUMBER"] = arrays[0]["RUNNUMBER"]
                res_df["EVENTNUMBER"] = arrays[0]["EVENTNUMBER"]
                res_df["EventNumber"] = evt_counter

                result.append(res_df)

            # saving the output
            if rerun_bunches is not None:
                if not save_flag:
                    continue

            tree_data, branch_types = save_root_helper(result)
            outfile = f"GeneratedParticles_{sample}_{str(bunch[0]).zfill(8)}_{str(bunch[1] - 1).zfill(8)}.root"
            with uproot.recreate(f"{outdir}/{outfile}") as file_particles:
                file_particles.mktree("Particles", branch_types, title="Particles Tree")
                file_particles["Particles"].extend(tree_data)


    elif is_exclusive:
        with uproot.open(f"{args.file}") as file:
            particles = file["EvtTupleMC/DecayTree"]
            df = particles.arrays(["EVENTNUMBER", "RUNNUMBER"], library="pd")
        identifier_map = {(a, b) for a in df["EVENTNUMBER"] for b in df["RUNNUMBER"]}

        # Tracking var
        evt_counter = 0
        bunch_counter = 0
        result = []

        for arrays in tqdm(tree, desc="Processing events"):
            if (arrays["EVENTNUMBER"][0], arrays["RUNNUMBER"][0]) not in identifier_map:
                continue
            evt_counter += 1

            if rerun_bunches is not None:
                if not bunches[bunch_counter] in rerun_bunches:
                    continue

            # Getting the mother and particle df
            particle_df, mother_df = get_evt_df(arrays, track_types)

            # Obtaining the head of the decay
            mother_df, _ = get_all_heads(mother_df)  # gives all heads if we do not only care about b decay
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
            res_df["RUNNUMBER"] = arrays[0]["RUNNUMBER"]
            res_df["EVENTNUMBER"] = arrays[0]["EVENTNUMBER"]
            res_df["EventNumber"] = evt_counter

            result.append(res_df)

            if evt_counter % args.bunch == 0:
                bunch = bunches[bunch_counter]
                bunch_counter += 1

                if rerun_bunches is not None:
                    if not bunch in rerun_bunches:
                        continue

                tree_data, branch_types = save_root_helper(result)
                outfile = f"GeneratedParticles_{sample}_{str(bunch[0]).zfill(8)}_{str(bunch[1] - 1).zfill(8)}.root"
                with uproot.recreate(f"{outdir}/{outfile}") as file_particles:
                    file_particles.mktree("Particles", branch_types, title="Particles Tree")
                    file_particles["Particles"].extend(tree_data)

                result = []
