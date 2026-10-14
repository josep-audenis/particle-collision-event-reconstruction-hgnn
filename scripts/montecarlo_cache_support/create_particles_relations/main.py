import os

from optparse import OptionParser

from helper import *
from create_collision_maps import *
from create_events import *

if __name__ == '__main__':



    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--infile", type=str, default=None,
                      dest="INFILE", help="Input file of pythia sig file or LHCb sim file")
    parser.add_option("", "--dir", type=str, default=None,
                      dest="DIR", help="Input and Output directory")
    parser.add_option("", "--sim", type=str, default=None,
                      dest="SIM", help="Simulation type: LHCb, pythia")

    parser.add_option("", "--bkg", type=str, default=None,
                      dest="BKG", help="BKG pv collision file")
    parser.add_option("", "--bunch", type=int, default=1000,
                      dest="BUNCH", help="Number of events per bunch for pythia only")
    parser.add_option("", "--nu", type=float, default=0,
                      dest="NU", help="Simulated nu, 0 for signal collision only")
    parser.add_option("", "--identifier", type=str, default=None,
                      dest="IDENTIFIER", help="for LHCb sim only 00258329")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    print("Start creating particles and relations")
    input_var = [
        "EventNumber", "ParticleIndex", "PrimaryHeavyHadronIndex", "MotherIndex", "id", "charge",
        "gammactau", "NumVisDaugh",
        "px", "py", "pz", "px_reco", "py_reco", "pz_reco",
        "xProd", "yProd", "zProd", "xProd_reco", "yProd_reco", "zProd_reco",
        "xPV", "yPV", "zPV", "xPV_reco", "yPV_reco", "zPV_reco",
        "isFinal", "isCharged", "isbhadron", "ischadron", "inGeomAcc",
    ]
    pythia_addons = [
        "ip", "ip_reco",
        "is_sig", "nFinals", "isFrag", "fragType", "fragPartnerIndex"
    ]

    LHCb_addons = [
        "Prob_pi", "Prob_k", "Prob_p", "Prob_e", "Prob_mu", "Prob_ghost",
        "EVENTNUMBER", "RUNNUMBER"
    ]

    if option.SIM == "pythia":
        print("=" * 30)
        print("PYTHIA8 processing")
        print("Signal file      :", option.INFILE)
        print("Background file  :", option.BKG)
        sig_mode, start, end = get_signal(option.INFILE)


        np.random.seed(get_rndm_seed(option.INFILE))

        nu_suffix = "singlecollision"
        if option.NU > 0:
            nu_suffix = "nu" + str(option.NU).replace(".", "p")

        outdir = f"{option.DIR}/DFEI_{option.SIM}_particles_relations/collision_maps/"
        os.makedirs(outdir, exist_ok=True)
        mapping_file = f"Collision_map_{sig_mode}_{nu_suffix}_{str(start).zfill(8)}_{str(end).zfill(8)}_events.yaml"
        mapping = get_collision_map(outdir + mapping_file, option.INFILE, option.BKG, option.DIR, option.NU)

        outfile_suffix = f"{sig_mode}_{nu_suffix}"

        print("=" * 30)
        print("Start creating events")
        start = np.sort(list(mapping.keys()))[0]

        bunches = [range(i, i + option.BUNCH) for i in range(start, start + len(mapping), option.BUNCH)]
        import pdb; pdb.set_trace()
        for bunch in bunches:
            create_multiple_events(
                infile=option.INFILE,
                vars=input_var + pythia_addons,
                bunch=bunch,
                outfile_suffix=outfile_suffix,
                dir=option.DIR,
                sim=option.SIM,
                mapping=mapping,
                bkg_file=option.BKG
            )
    elif option.SIM == "LHCb":
        print("=" * 30)
        print("LHCb processing")
        print("Input file       :", option.INFILE)
        sig_mode, start, end = get_signal(option.INFILE)
        outfile_suffix = f"{sig_mode}_LHCbcollision"
        create_multiple_events(
            infile=option.INFILE,
            vars=input_var + LHCb_addons,
            bunch=range(start, end),
            outfile_suffix=outfile_suffix,
            dir=option.DIR,
            sim=option.SIM,
            identifier=option.IDENTIFIER
        )
