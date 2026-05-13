import glob
import os

from optparse import OptionParser

from DecayChainReconstruction import *
from create_evts import create_evts

if __name__ == "__main__":
    # python main.py --outdir test --sample Bs_Jpsiphi --start 0 --end 999 --mode nu --norm --ft
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--outdir", type=str, default=None,
                      dest="OUTDIR", help="Output directory where pt files are cached")
    parser.add_option("", "--sample", type=str, default=None,
                      dest="SAMPLE", help="decay channel")
    parser.add_option("", "--start", type=str, default=None,
                      dest="START", help="start value of file")
    parser.add_option("", "--end", type=str, default=None,
                      dest="END", help="end value of file")
    parser.add_option("", "--mode", type=str, default=None,
                      dest="MODE", help="singlepp, nu, missasso")
    parser.add_option("--norm", action="store_true", dest="NORM", default=False)
    parser.add_option("--ft", action="store_true", dest="FT", default=False)
    parser.add_option("--pid", type=str, default="onehot",
                      dest="PID", help="onehot, realistic")
    parser.add_option("", "--sim", type=str, default="pythia",
                      dest="SIM", help="Simulation type")
    parser.add_option("", "--save", type=str, default=None,
                      dest="SAVE", help="if passing trn, tst, val only those are save")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    indir = "/eos/lhcb/user/y/yukaiz/DFEI_data/DFEI_pythia_particles_relations"
    start = option.START
    end = option.END
    if option.MODE == "singlepp":
        suffix = "_singlecollision_"
    elif option.MODE == "nu":
        suffix = "_nu7p6_"
    elif option.MODE == "missasso":
        suffix = "_missassosinglecollision_"
    else:
        print("Undefined mode")
        exit()

    infile = glob.glob(f"{indir}/{option.SAMPLE}/*particles*{suffix}*{start.zfill(8)}_to_{end.zfill(8)}*")[0]
    suffix = suffix + option.PID + "_"
    real_pid = True if option.PID == "realistic" else False
    if option.NORM:
        suffix = suffix + "normed_"
    outdir = f"{option.OUTDIR}/DFEI_{option.SIM}{suffix}pt_data/{option.SAMPLE}"
    os.makedirs(outdir, exist_ok=True)
    threshold = round((int(end) - int(start)) * 0.8)

    print("PLEASE NOTE PID FEATURE SEPARATION IS HARDCODED")
    reco_env = DecayChainReconstructionEnv(use_reco_vars=1, sim_file_name_=infile, ft_bool=option.FT)

    create_evts(reco_env, int(start), threshold, outdir, norm_graph=option.NORM, ft_bool=option.FT, real_pid=real_pid,
                save=option.SAVE)
