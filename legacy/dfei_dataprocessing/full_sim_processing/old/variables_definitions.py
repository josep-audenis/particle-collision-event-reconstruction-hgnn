part_keys = ["B_Ancestor", "B_Descendants", "B_Intermediate", "EXTRAPARTICLES"]

# B_ancestor implies everything from B
# keys of saved root file/df, B related, extra tracks
df_keys = [
    ["ParticleIndex", "ParticleIndex", "ParticleIndexTRUE"],
    ["MotherIndex", "MotherIndex", "MotherIndexTRUE"],
    ["MotherId", "MotherID", "MotherIDTRUE"],
    ["id", "id", "idTRUE"],  # also normal id need to  check for extraparticles
    ["px", "px", "pxTRUE"],
    ["py", "py", "pyTRUE"],
    ["pz", "pz", "pzTRUE"],
    ["px_reco", "px", "px_reco"],  # B_ancestor does not have px reco
    ["py_reco", "py", "py_reco"],  # B_ancestor does not have py reco
    ["pz_reco", "pz", "pz_reco"],  # B_ancestor does not have pz reco
    ["xProd", "OV_X", "xProdTRUE"],
    ["yProd", "OV_Y", "yProdTRUE"],
    ["zProd", "OV_Z", "zProdTRUE"],
    ["xProd_reco", "OV_X", "xProd_reco"],  # B_ancestor does not have xProd reco
    ["yProd_reco", "OV_Y", "yProd_reco"],  # B_ancestor does not have yProd reco
    ["zProd_reco", "OV_Z", "zProd_reco"],  # B_ancestor does not have zProd reco,
    ["xPV", "PV_X", "xPVTRUE"],
    ["yPV", "PV_Y", "yPVTRUE"],
    ["zPV", "PV_Z", "zPVTRUE"],
    ["xPV_reco", "PV_X", "xPV_reco"],  # B_ancestor does not have xPV reco -> grab from reco matching
    ["yPV_reco", "PV_Y", "yPV_reco"],  # B_ancestor does not have y PV reco -> grab from reco matching
    ["zPV_reco", "PV_Z", "zPV_reco"],  # B_ancestor does not have zPV reco -> grab from reco matching
    ["isFinal", "isFinal", "isFinalTRUE"],  # this will be overwritten probably
    ["isCharged", "isCharged", "isChargedTRUE"],
    ["inGeomAcc", "inGeomAcc", "inGeomAccTRUE"],
    ["isbhadron", "isbhadron", ""],  # For extra track this will be default set to 0 since extra tracks
    ["ischadron", "ischadron", ""],  # For extra track this will be default set to 0 since extra tracks
    ["charge", "Charge", "ChargeTRUE"],
    ["gammactau", "gammactau", ""],
    ["NumVisDaugh", "NumVisDaugh", ""]
]

# need EventNumber, PrimaryHeavyHadronIndex
# ip quantity does not make sense here
