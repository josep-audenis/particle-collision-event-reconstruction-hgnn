from PyConf.reading import get_mc_header, get_pvs
from RecoConf.event_filters import require_pvs
import Functors as F
from FunTuple import FunctorCollection
import FunTuple.functorcollections as FC

from FunTuple import FunTuple_Event, FunTuple_Particles, FunTuple_MCParticles
from DaVinci import Options, make_config

from PyConf.reading import get_particles, get_mc_particles, get_mc_track_info
from DaVinciMCTools import MCTruthAndBkgCat, MCReconstructible, MCReconstructed
from RecoConf.reconstruction_objects import make_tracks

from RecoConf.standard_particles import (
    make_long_pions,
    make_up_pions,
    make_down_pions,
    make_resolved_pi0s,
    make_merged_pi0s,
    make_photons,
)

IS_H = F.IS_HADRON @ F.PARTICLE_ID_OBJ
IS_H_BOTTOM  = F.HAS_BOTTOM & F.IS_HADRON @ F.PARTICLE_ID_OBJ
IS_H_CHARM   = F.HAS_CHARM  & F.IS_HADRON  @ F.PARTICLE_ID_OBJ



# The famous is final definition which breaks the code yey
stable_particles = ["pi+","K+","p+","e-","mu-"] # [211, 321, 2212, 11, 13]
abs_ids_stable = [f"F.IS_ABS_ID('{ptc}')" for ptc in stable_particles]
IS_STABLE = eval("|".join(abs_ids_stable))
MC_isBasic = (F.SIZE_OF @ F.GET_GENERATION(1)) == 0
IS_FINAL = MC_isBasic | IS_STABLE

# Number of decay prongs -> this does not work since we need to access children of MC particle.
# I really hate children...
INSIDE_ACCEPTANCE = (F.ETA > 2) & (F.ETA < 5)
filter_inacc = F.FILTER(INSIDE_ACCEPTANCE)
filter_charged_inacc = F.FILTER((F.CHARGE != 0) & INSIDE_ACCEPTANCE)
filter_charged = F.FILTER(F.CHARGE != 0)
filter_notcharged = F.FILTER(F.CHARGE == 0)

can_be_reconstructed = F.SIZE_OF @ filter_charged_inacc @ F.GET_ALL_DESCENDANTS > 0

num_decay_prongs = F.VALUE_OR(0)@ (
        (F.SIZE_OF @ F.FILTER(can_be_reconstructed) @ filter_notcharged) + (F.SIZE_OF @ filter_charged)
)@ filter_inacc@ F.GET_CHILDREN



FUNCTORS = {
    "particles": {
        # 4 vector
        "energy"        : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.ENERGY)           @ F.TES(particles),
        "px_reco"       : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PX)               @ F.TES(particles),
        "py_reco"       : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PY)               @ F.TES(particles),
        "pz_reco"       : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PZ)               @ F.TES(particles),
        # PV
        "xPV_reco"      : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.BPVX(get_pvs()))  @ F.TES(particles),
        "yPV_reco"      : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.BPVY(get_pvs()))  @ F.TES(particles),
        "zPV_reco"      : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.BPVZ(get_pvs()))  @ F.TES(particles),
        # Prod
        "xProd_reco"    : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.X_COORDINATE @ F.POSITION @ F.STATE_AT("FirstMeasurement") @ F.TRACK) @ F.TES(particles),
        "yProd_reco"    : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.Y_COORDINATE @ F.POSITION @ F.STATE_AT("FirstMeasurement") @ F.TRACK) @ F.TES(particles),
        "zProd_reco"    : lambda particles: F.MAP(F.VALUE_OR(0.0)   @ F.Z_COORDINATE @ F.POSITION @ F.STATE_AT("FirstMeasurement") @ F.TRACK) @ F.TES(particles),
        # Prob
        "Prob_mu"       : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_MU)        @ F.TES(particles),
        "Prob_k"        : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_K)         @ F.TES(particles),
        "Prob_pi"       : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_PI)        @ F.TES(particles),
        "Prob_p"        : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_P)         @ F.TES(particles),
        "Prob_e"        : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_E)         @ F.TES(particles),
        "Prob_ghost"    : lambda particles: F.MAP(F.VALUE_OR(-1.0)  @ F.PROBNN_GHOST)     @ F.TES(particles),

        "Charge"        : lambda particles: F.MAP(F.VALUE_OR(-999)  @ F.CHARGE)           @ F.TES(particles),
        "isCharged"     : lambda particles: F.MAP(F.VALUE_OR(False) @ F.CHARGE != 0)      @ F.TES(particles),
    },
    "mc_particles": {
        "ParticleIndex" : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(-1)   @ MCTRUTH(F.OBJECT_KEY))    @ F.TES(particles),
        "id"            : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0)    @ MCTRUTH(F.PARTICLE_ID))   @ F.TES(particles),
        # 4 vector
        "px"            : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(-1.0) @ MCTRUTH(F.PX))            @ F.TES(particles),
        "py"            : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(-1.0) @ MCTRUTH(F.PY))            @ F.TES(particles),
        "pz"            : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(-1.0) @ MCTRUTH(F.PZ))            @ F.TES(particles),
        # PV
        "xPV"           : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.MC_PV_VX))     @ F.TES(particles),
        "yPV"           : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.MC_PV_VY))     @ F.TES(particles),
        "zPV"           : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.MC_PV_VZ))     @ F.TES(particles),
        # Prod
        "xProd"         : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.ORIGIN_VX))    @ F.TES(particles),
        "yProd"         : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.ORIGIN_VY))    @ F.TES(particles),
        "zProd"         : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(0.0)   @ MCTRUTH(F.ORIGIN_VZ))    @ F.TES(particles),
        # Flags YEEEEEEY isFinal is causing an crash as usual hip hip hura
        #"isFinal"       : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(False) @ MCTRUTH(IS_FINAL))       @ F.TES(particles),
        # inGeomAcc is auto set to true since otherwise they wouldnt be able to be recoable
        #"inGeomAcc"     : lambda particles, MCTRUTH: F.MAP(F.VALUE_OR(False) @ MCTRUTH(INSIDE_ACCEPTANCE)) @ F.TES(particles),
    },
    "mc_mothers":{
        "MotherID"              : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0)        @ MCTRUTH(F.MC_MOTHER(generation, F.PARTICLE_ID)))  @ F.TES(particles),
        "MotherIndex"           : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(-1)       @ MCTRUTH(F.MC_MOTHER(generation, F.OBJECT_KEY)))   @ F.TES(particles),
        "MotherMotherIndex"     : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(-1)       @ MCTRUTH(F.MC_MOTHER(generation+1, F.OBJECT_KEY))) @ F.TES(particles),
        "MotherIshadron"        : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(False)    @ MCTRUTH(F.MC_MOTHER(generation, IS_H)))           @ F.TES(particles),
        "MotherIsbhadron"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(False)    @ MCTRUTH(F.MC_MOTHER(generation, IS_H_BOTTOM)))    @ F.TES(particles),
        "MotherIschadron"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(False)    @ MCTRUTH(F.MC_MOTHER(generation, IS_H_CHARM)))     @ F.TES(particles),
        # this num vis daugh
        #"MotherNumVisDaugh"     : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(False)    @ MCTRUTH(F.MC_MOTHER(generation, num_decay_prongs)))     @ F.TES(particles),
        # Construct gammactau from originvertex of itself to endvertex coressponding to originvertex of daughter



    },
    # this function only works for mothers of generation 2
    "mc_mothers_2":{ # can i calculate the magnitude directly?
        "MotherFD_X"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VX) - F.MC_MOTHER(generation-1, F.ORIGIN_VX)))     @ F.TES(particles),
        "MotherFD_Y"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VY) - F.MC_MOTHER(generation-1, F.ORIGIN_VY)))     @ F.TES(particles),
        "MotherFD_Z"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VZ) - F.MC_MOTHER(generation-1, F.ORIGIN_VZ)))     @ F.TES(particles),
    },
    "mc_mothers_1":{
        "MotherFD_X"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VX) - F.ORIGIN_VX))     @ F.TES(particles),
        "MotherFD_Y"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VY) - F.ORIGIN_VY))     @ F.TES(particles),
        "MotherFD_Z"       : lambda particles, generation, MCTRUTH: F.MAP(F.VALUE_OR(0.0)     @ MCTRUTH(F.MC_MOTHER(generation, F.ORIGIN_VZ) - F.ORIGIN_VZ))     @ F.TES(particles),
    }
    # TODO make this also work for mc_mother_1 which takes instead of generation - 1 the MC originvertex of the current particle
}


def main(options: Options):
    extra_tracks = {
        "long": make_long_pions(),
        "up": make_up_pions(),
        "down": make_down_pions(),
    }

    mc_header = get_mc_header()
    evt_vars = FunctorCollection()
    evt_vars['ALLPV_Index[apv_indx]'] = F.MAP(F.VALUE_OR(-1) @ F.OBJECT_KEY) @ F.MC_ALLPVS(mc_header=mc_header)

    extra_vars = {}
    ngenerations = 7
    for ttype, tracks in extra_tracks.items():
        MCTRUTH_extra_parts = MCTruthAndBkgCat(tracks, name=f'MCTRUTH_extra_{ttype}')
        extra_vars[ttype] = {}
        # Reconstructed particles
        for name, functor in FUNCTORS["particles"].items():
            extra_vars[ttype][f'{ttype}_{name}[{ttype}_indx]'] = functor(tracks)

        # Matched MC true particles
        for name, functor in FUNCTORS["mc_particles"].items():
            extra_vars[ttype][f'{ttype}_{name}[{ttype}_indx]'] = functor(tracks, MCTRUTH_extra_parts)
        # Matched MC true particles mothersssssss
        for gen in range(1,ngenerations):
            for name, functor in FUNCTORS["mc_mothers"].items():
                extra_vars[ttype][f'{ttype}_{gen}{name}[{ttype}_indx]'] = functor(tracks, gen, MCTRUTH_extra_parts)
            if gen == 1: # Calculate the FD by looking at the originvertex of daughter
                for name, functor in FUNCTORS["mc_mothers_1"].items():
                    extra_vars[ttype][f'{ttype}_{gen}{name}[{ttype}_indx]'] = functor(tracks, gen, MCTRUTH_extra_parts)
            if gen >= 2:
                for name, functor in FUNCTORS["mc_mothers_2"].items():
                    extra_vars[ttype][f'{ttype}_{gen}{name}[{ttype}_indx]'] = functor(tracks, gen, MCTRUTH_extra_parts)


        evt_vars.update(extra_vars[ttype])
    evt_tuple = FunTuple_Event(
        name="EvtTuple", tuple_name='Tracks', variables=evt_vars
    )
    return make_config(options, [require_pvs(get_pvs()), evt_tuple])
