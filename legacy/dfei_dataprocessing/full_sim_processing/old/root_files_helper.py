import uproot

def initialize_trees(input_file, steps):
    tuples_names = ["event",
                    "B_Ancestor", "B_Descendants", "B_Intermediate",
                    "C_Ancestor", "C_Descendants", "C_Intermediate"]
    extra_names = ['event']
    with uproot.open(f"{input_file}:EvtTuple3/ExtraParticles", library="ak", max_num_elements=1) as tree:
        for ptc_type in [a.split("_")[1] for a in tree.keys() if "EXTRA" in a]:
            if ptc_type not in extra_names:
                extra_names.append('EXTRAPARTICLES_' + ptc_type)

    tree1 = uproot.iterate(f"{input_file}:EvtTuple1/QuarkChildren", library="ak", step_size=steps)
    tree2 = uproot.iterate(f"{input_file}:EvtTuple2/ExcitedHadrons", library="ak", step_size=steps)
    tree3 = uproot.iterate(f"{input_file}:EvtTuple3/ExtraParticles", library="ak", step_size=steps)
    trees = [tree1, tree2, tree3]

    return tuples_names, extra_names, trees



