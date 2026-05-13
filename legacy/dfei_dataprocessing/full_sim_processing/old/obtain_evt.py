import pandas as pd
import numpy as np

def unpack_evt(df_mc, name="B_Ancestor", variable="EVENTNUMBER"):
    field_to_index = {
        "B_Ancestor": 'ba_indx',
        "B_Descendants": 'bd_indx',
        "B_Intermediate": 'bi_indx',
        "C_Ancestor": 'ca_indx',
        "C_Descendants": 'cd_indx',
        "C_Intermediate": 'ci_indx',
        "EXTRAPARTICLES_down": 'down_indx',
        "EXTRAPARTICLES_long": 'long_indx',
        "EXTRAPARTICLES_up": 'up_indx',
    }

    evt_arr = np.concatenate([
        np.full(df_mc.loc[i, field_to_index[name]], df_mc.loc[i, variable])
        for i in df_mc.index], dtype=int)
    return evt_arr


def obtain_evts(trees, tuples_names, extra_names, Track_types, only_track_types, event_counter):
    # obtain array of df where each df is an evt
    tuples1, tuples2, extras = {}, {}, {}
    tuples = {'event': [], 'pvs': [], 'EXTRAPARTICLES': []}
    for name in tuples_names[1:]:
        tuples[name] = []

    arrays1, arrays2, arrays3 = next(trees[0]), next(trees[1]), next(trees[2])

    # retrieve event data
    pvs_evts = pd.DataFrame([pv_k.to_list() for pv_k in arrays1['ALLPV_Index']])  # .fillna(0).astype(int)
    tuples['pvs'].append(pvs_evts)

    selected_evt1 = [f for f in arrays1.fields if 'indx' in f] + ['EVENTNUMBER', 'RUNNUMBER']
    mc1_evts = pd.concat([pd.Series(d) for d in arrays1[selected_evt1].to_list()], axis=1).T
    selected_evt2 = [f for f in arrays2.fields if 'indx' in f] + ['EVENTNUMBER', 'RUNNUMBER']
    mc2_evts = pd.concat([pd.Series(d) for d in arrays2[selected_evt2].to_list()], axis=1).T
    selected_ext = [f for f in arrays3.fields if 'indx' in f] + ['EVENTNUMBER', 'RUNNUMBER']
    extra_evts = pd.concat([pd.Series(d) for d in arrays3[selected_ext].to_list()], axis=1).T

    for name in tuples_names[1:]:
        selected_v1 = [f for f in arrays1.fields if name in f]
        tuples1[name] = pd.concat([pd.DataFrame(d) for d in arrays1[selected_v1].to_list()], ignore_index=True)
        tuples1[name]["EVENTNUMBER"] = unpack_evt(mc1_evts, name=name, variable="EVENTNUMBER")
        tuples1[name]["RUNNUMBER"] = unpack_evt(mc1_evts, name=name, variable="RUNNUMBER")

        selected_v2 = [f for f in arrays2.fields if name in f]
        tuples2[name] = pd.concat([pd.DataFrame(d) for d in arrays2[selected_v2].to_list()], ignore_index=True)
        tuples2[name]["EVENTNUMBER"] = unpack_evt(mc2_evts, name=name, variable="EVENTNUMBER")
        tuples2[name]["RUNNUMBER"] = unpack_evt(mc2_evts, name=name, variable="RUNNUMBER")

    for name in extra_names[1:]:
        selected_ext = [f for f in arrays3.fields if name in f] + ['EVENTNUMBER', 'RUNNUMBER']
        extra_new_cols = [col.replace(name, "EXTRAPARTICLES") for col in selected_ext]
        extras[name] = pd.concat([
            pd.DataFrame(d).rename(columns=dict(zip(selected_ext, extra_new_cols))) for d in
            arrays3[selected_ext].to_list()
        ], ignore_index=True)
        extras[name]['TrackTypes'] = Track_types[name.replace("EXTRAPARTICLES_", "")]

    tuples['event'].append(
        pd.concat([(mc1_evts[selected_evt1[1:-2]] + mc2_evts[selected_evt2[1:-2]]), extra_evts], axis=1))

    for name in tuples_names[1:]:
        try:
            tuples[name].append(pd.concat(
                [tuples1[name], tuples2[name]], ignore_index=False).sort_values(by='EVENTNUMBER', kind='stable'))
        except KeyError:
            tuples[name].append(tuples1[name])
    tuples['EXTRAPARTICLES'].append(
        pd.concat([extras[f"EXTRAPARTICLES_{xtr}"] for xtr in only_track_types], axis=0))


    for k in tuples.keys():
        tuples[k] = pd.concat(tuples[k], axis=0)
    groups = ["B_Ancestor", "B_Descendants", "B_Intermediate", "EXTRAPARTICLES"]
    grouped = {g: tuples[g].sort_values(by=['RUNNUMBER', 'EVENTNUMBER']).set_index(['RUNNUMBER', 'EVENTNUMBER']) for
               g in groups}
    events = []
    for e, (R, E) in tuples['event'][['RUNNUMBER', 'EVENTNUMBER']].iterrows():
        events.append({g: grouped[g].loc[R, E] for g in groups})


    return events