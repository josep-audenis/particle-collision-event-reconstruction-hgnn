from optparse import OptionParser

import pandas as pd

from root_files_helper import *
from obtain_evt import obtain_evts
from transform_df import DfCreator

if __name__ == "__main__":
    # python main.py --file /eos/lhcb/user/s/sicapell/DFEI/publictemp/workshop/dfei_tuple_magup.root --evt_max 99
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--file", type=str, default=None,
                      dest="FILE", help="File to be processed")
    parser.add_option("", "--nsteps", type=int, default=100,
                      dest="NSTEPS", help="Number of steps per file")
    parser.add_option("", "--evt_max", type=int, default=-1,
                      dest="EVT_MAX", help="Max number of events")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))
    nsteps = (option.EVT_MAX // option.NSTEPS) + 1
    print(f"Loading trees in {nsteps} chunks of {option.NSTEPS} events")

    # this is hardcoded rn
    only_track_types = ['long', 'up', 'down']
    print("using only track types:", only_track_types)

    tuples_names, extra_names, trees = initialize_trees(option.FILE, option.NSTEPS)
    Track_types = {
        "long": 0,
        "down": 1,
        "up": 2,
        # "merpi0": 3,
        # "photons": 4,
        # "respi0": 5,
    }

    # number of processed events
    event_counter = 0
    for step in range(nsteps): #tdqm
        if event_counter >= option.EVT_MAX and option.EVT_MAX != -1:
            print(f"Reached {event_counter} events, stopping")
            break

        events = obtain_evts(trees, tuples_names, extra_names, Track_types, only_track_types, event_counter)
        print(event_counter)
        import pdb; pdb.set_trace()
        df_creator = DfCreator(events, event_counter)
        res = []
        for evt_number in range(len(events)):
            df = df_creator.get_df(1)
            res.append(df)

        output = pd.concat(res, ignore_index=True, sort=True)
        new_branches_particles = list(output.columns)
        tree_data = {
            name: output[name].values
            for name in new_branches_particles
        }
        branch_types = {
            name: output[name].dtype
            for name in new_branches_particles
        }
        sample = "inclusive"
        outdir = "/eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_LHCb_generator/inclusive"
        start = event_counter
        end = event_counter + len(events) - 1
        outfile = f"{outdir}/GeneratedParticles_{sample}_{str(start).zfill(8)}_{str(end).zfill(8)}.root"
        with uproot.recreate(outfile) as file_particles:
            file_particles.mktree("Particles", branch_types, title="Particles Tree")
            file_particles["Particles"].extend(tree_data)

            event_counter += len(events)
        print(f"Processed {event_counter} events")
