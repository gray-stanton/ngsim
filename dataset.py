import pandas as pd
import numpy as np
import math
import sys
def make_point_dataset(fname="./ngsim_us101.csv", location='us-101'):
    """
    Returns a DataFrame containing observations of individual vehicles.
    """
    ngsim_point = pd.read_csv(fname)
    ngsim_point.drop(["v_Class", "O_Zone", "D_Zone", 
                      "Int_ID", "Section_ID", "Direction",
                      "Movement", "Preceding", "Following"], axis=1, inplace=True)
    qstring= "Location == '" + location + "'"
    ngsim_point = ngsim_point.query(qstring)
    ts = ngsim_point.Global_Time/100
    ts = ts - ts.min()
    ngsim_point = ngsim_point.assign(timestep= ts)
    ngsim_point.sort_values(by=["timestep", "Global_X", "Global_Y"], inplace=True)
    unique_vehicles = ngsim_point.loc[:,["Vehicle_ID", "Total_Frames"]].drop_duplicates(
        keep="first", inplace=False, ignore_index=True)
    unique_vehicles = unique_vehicles.assign(Unique_Veh_ID = pd.Series(np.arange(1, len(unique_vehicles)+1, 1), dtype="Int32"))
    ngsim_point = ngsim_point.merge(unique_vehicles,how="left", left_on=["Vehicle_ID", "Total_Frames"], 
                                   right_on=["Vehicle_ID", "Total_Frames"])
    return ngsim_point






def make_triangle_dataset(point_dataset, create_dist, end_dist, extra_vars, take_timesteps=None):
    """
    Returns a DataFrame containing observations for all triangles of vehicles.
        
        Parameters:
            point_dataset (DataFrame): A DataFrame constructed with make_point_dataset
            create_dist (float): Required max sidelength for a trio of vehicles to ever form a valid triangle
            end_dist (float): Maximum sidelength for a triangle to be valid in a particualar timestep
            extra_vars (list of strings): List of names of vehicle-level variables in point_dataset other than Global_X, Y 
                to includein the final dataset.
            take_timesteps (int): Number of timesteps, starting from zero, to include.
        Returns:
            out_df (DataFrame): a Dataframe containing the variables:
                    timestep (int): time, in 0.1 s intervals, from begining of dataset.
                    tri_id: Id for specific group of 3 vehicles
                    rec_tri_id: tri_id, but broken up into temporally consistent pieces
                    ida: Unique_Veh_ID for first vehicle
                    idb: same for second
                    idc: same for third
                    Other Variables ending in 'a','b','c' are observations corresponding with that particular 
                    vehicle in the triangle, controlled by extra_vars.
    """
    timestep_splits = point_dataset.groupby("timestep").groups # Dict mapping ts number to row indices
    # Preallocating rows to reduce overhead
    PREALLOC_ROWS = 300000000
    valid_quads = np.zeros((PREALLOC_ROWS, 4), dtype=np.int32) # Set of valid (timestep, id1, id2, id3) quads
    ids_of_triangles = set()
    row = 0
    for ts in timestep_splits.keys():
        print(ts)
        if row >= PREALLOC_ROWS:
            raise MemoryError("Number of observed triangles exceeds preallocation")
        if take_timesteps is not None and ts >= take_timesteps:
            break
        cur_df = point_dataset.loc[timestep_splits[ts], :]
        pos = list(zip(list(cur_df.Global_X), list(cur_df.Global_Y), list(cur_df.Unique_Veh_ID)))
        for i in range(0, len(pos)):
            for j in range(i + 1, len(pos)):
                if (pos[j][1] - pos[i][1]) > end_dist:
                    # sorted by X, so if this car more than end_dist away in X direction, 
                    # all other cars also too far away
                    break
                dist_ij = math.sqrt((pos[j][0] - pos[i][0])**2 + 
                                  (pos[j][1] - pos[i][1])**2) 
                if dist_ij > end_dist :
                    # Technically possible that further cars could be within distance
                    continue
                else:
                    # both i, j within end_dist of each other, check more cars
                    for k in range(j + 1, len(pos)):
                        if (pos[k][0] - pos[i][0]) > end_dist:
                            break
                        dist_ik = math.sqrt((pos[k][0] - pos[i][0])**2 +  
                                             (pos[k][1] - pos[i][1])**2)
                        dist_jk = math.sqrt((pos[k][0] - pos[j][0])**2 +  
                                             (pos[k][1] - pos[j][1])**2)
                        # sort ids in ascending order, to have a consistent ordering to join on later.
                        sorted_ids = sorted([pos[i][2], pos[j][2], pos[k][2]])
                        if dist_ik <= end_dist and dist_jk <= end_dist:
                            sorted_ids.insert(0, ts)
                            valid_quads[row] = np.array(sorted_ids, dtype=np.int32)
                            row += 1
                        if dist_ik <= create_dist and dist_jk <= create_dist and dist_ij <= create_dist:
                            ids_of_triangles.add(tuple(sorted_ids[1:]))

    triangle_df = pd.DataFrame(valid_quads[0:row,:],
                               columns=["timestep", "id1", "id2", "id3"])
    triangle_id_df = pd.DataFrame(ids_of_triangles, columns = ['id1', 'id2', 'id3'])
    triangle_id_df = triangle_id_df.assign(tri_id = np.arange(1, len(triangle_id_df) + 1, dtype=np.int32))
    print("Triangle_df size: {}".format(len(triangle_df)))
    print("Number of distinct triangles: {}".format(len(triangle_id_df)))
    del valid_quads
    print("merging triangle_id")
    triangle_df = triangle_df.merge(triangle_id_df, on=['id1', 'id2', 'id3'],
                                    how="left", indicator=True)
    print("drop non-valid triangles")
    triangle_df = triangle_df.query("_merge != 'left_only'")
    position_df = point_dataset.loc[:, ["timestep", "Unique_Veh_ID", "Global_X", "Global_Y"]]
    # Merge position data for all 3 vehicles
    print("merging triangle positions")
    triangle_df = triangle_df.merge(position_df.rename(columns={
        "timestep":"timestep",
        "Unique_Veh_ID":"id1",
        "Global_X":"Global_X1",
        "Global_Y":"Global_Y1"
    }),
        how="left", on = ["timestep", "id1"]) 
    triangle_df = triangle_df.merge(position_df.rename(columns={
        "timestep":"timestep",
        "Unique_Veh_ID":"id2",
        "Global_X":"Global_X2",
        "Global_Y":"Global_Y2"
    }),
        how="left", on=["timestep", "id2"]) 
    triangle_df = triangle_df.merge(position_df.rename(columns={
        "timestep":"timestep",
        "Unique_Veh_ID":"id3",
        "Global_X":"Global_X3",
        "Global_Y":"Global_Y3"
    }),
        how="left", on=["timestep", "id3"])
    
    ## Label vertices consistently across triangles and timesteps,
    ## by assigning labels "a", "b", "c" counterclockwise based on vector
    ## from centroid to vertex, starting at negative x axis. Vertex labeling
    ## is done at triangle initial timestep and kept consistent for lifetime
    print("Making centroid")
    triangle_df = triangle_df.assign(centroid_X=(triangle_df['Global_X1'] +
                                    triangle_df['Global_X2'] + 
                                    triangle_df['Global_X3'])/3,
                       centroid_Y=(triangle_df['Global_Y1'] +
                                    triangle_df['Global_Y2'] + 
                                    triangle_df['Global_Y3'])/3
                      )
    triangle_df = triangle_df.assign(
        intangle1 = np.arctan2(triangle_df['Global_Y1'] - triangle_df['centroid_Y'],
                               triangle_df['Global_X1'] - triangle_df['centroid_X']),
        intangle2 = np.arctan2(triangle_df['Global_Y2'] - triangle_df['centroid_Y'],
                               triangle_df['Global_X2'] - triangle_df['centroid_X']),
        intangle3 = np.arctan2(triangle_df['Global_Y3'] - triangle_df['centroid_Y'],
                               triangle_df['Global_X3'] - triangle_df['centroid_X'])
                                    )
    print("Merging timestep info")
    init_timesteps = triangle_df.groupby("tri_id").apply(
        lambda x : x.timestep.min()).to_frame("min_timestep")
    triangle_df = triangle_df.merge(init_timesteps, on="tri_id")
    init_df = triangle_df.query("timestep == min_timestep")
    def _reorder(df):
        # argsort is analog of "order" in R
        # reorder indices based on intangle order
        colorder = list(np.argsort([df.intangle1, df.intangle2, df.intangle3]) + 1)
        keys = [("id"+str(i)) for i in colorder]
        return list(df[keys])
    reordering = dict(zip(list(init_df.tri_id), list(init_df.apply(_reorder, axis=1))))
    def _apply_reorder(currow):
        # cur row is (tri_id, id1, id2, id3, Global_X1, Global_Y1, Global_X2, Global_Y2, 
        # Global_X3, Global_Y3).
        timestep, tri_id, id1, id2, id3, Global_X1, Global_Y1, Global_X2, Global_Y2, Global_X3, Global_Y3 = currow
        # for each row, get the data from Global_X1, etc and reorder it to be
        # associated with the new ordering
        order = reordering[tri_id]
        curorder = [id1, id2, id3]
        match = [curorder.index(order[0]), 
                 curorder.index(order[1]),
                 curorder.index(order[2])]
        Xs = [Global_X1, Global_X2, Global_X3]
        Ys = [Global_Y1, Global_Y2, Global_Y3]
        Global_Xa = Xs[match[0]]
        Global_Ya = Ys[match[0]]
        Global_Xb = Xs[match[1]]
        Global_Yb = Ys[match[1]]
        Global_Xc = Xs[match[2]]
        Global_Yc = Ys[match[2]]
        newrow = (timestep, tri_id, order[0], order[1], order[2],
                 Global_Xa, Global_Ya,
                 Global_Xb, Global_Yb, 
                 Global_Xc, Global_Yc
              )
        return newrow
    print("starting reorder")
    newrows = [_apply_reorder(cr) for cr in zip(triangle_df["timestep"],
                                                triangle_df["tri_id"],
                                                triangle_df["id1"],
                                                triangle_df["id2"],
                                                triangle_df["id3"],
                                                triangle_df["Global_X1"],
                                                triangle_df["Global_Y1"],
                                                triangle_df["Global_X2"],
                                                triangle_df["Global_Y2"],
                                                triangle_df["Global_X3"],
                                                triangle_df["Global_Y3"])]
    del triangle_df
    reordered_df = pd.DataFrame(newrows, columns=["timestep",
                                                   "tri_id",
                                                   "ida",
                                                   "idb",
                                                   "idc",
                                                   "Global_Xa",
                                                   "Global_Ya",
                                                   "Global_Xb", 
                                                   "Global_Yb",
                                                   "Global_Xc",
                                                   "Global_Yc"])
    del newrows
    print("break up nonconseq triangles")
    reordered_df = reordered_df.assign(prev_timestep= reordered_df.groupby("tri_id").timestep.shift(1))
    reordered_df = reordered_df.assign(is_jump = reordered_df.prev_timestep +1 != reordered_df.timestep)
    reordered_df = reordered_df.assign(piece = reordered_df[['tri_id', 'is_jump']].groupby('tri_id').cumsum())
    reordered_df = reordered_df.assign(rec_tri_id = reordered_df.tri_id + (reordered_df.piece - 1) * 1000000)



    out_df = reordered_df[['timestep', 'tri_id', 'rec_tri_id', 'ida', 'idb', 'idc', 'Global_Xa',
                           'Global_Ya', 'Global_Xb', 'Global_Yb', 'Global_Xc', 'Global_Yc']]
    if len(extra_vars) > 0:
        print("merging extra variables")
        merge_vars = extra_vars.copy()
        merge_vars.extend(["Unique_Veh_ID", "timestep"])
        remap_a = {var : var + "a" for var in extra_vars}
        remap_a["Unique_Veh_ID"] = "ida"
        remap_a["timestep"] = "timestep"
        out_df = out_df.merge(
            point_dataset[merge_vars].rename(columns=remap_a),
            how="left", on = ["timestep", "ida"])
        
        remap_b = {var : var + "b" for var in extra_vars}
        remap_b["Unique_Veh_ID"] = "idb"
        remap_b["timestep"] = "timestep"
        out_df = out_df.merge(
            point_dataset[merge_vars].rename(columns=remap_b),
            how="left", on = ["timestep", "idb"])

        remap_c = {var : var + "c" for var in extra_vars}
        remap_c["Unique_Veh_ID"] = "idc"
        remap_c["timestep"] = "timestep"
        out_df = out_df.merge(
            point_dataset[merge_vars].rename(columns=remap_c),
            how="left", on = ["timestep", "idc"])
        
    return out_df











def tri_sidelengths(df, id1, id2, id3):
    a1 = df.loc[df.Unique_Veh_ID == id1, ["Global_X"]].squeeze()
    a2 = df.loc[df.Unique_Veh_ID == id1, ["Global_Y"]].squeeze()
    b1 = df.loc[df.Unique_Veh_ID == id2, ["Global_X"]].squeeze()
    b2 = df.loc[df.Unique_Veh_ID == id2, ["Global_Y"]].squeeze()
    c1 = df.loc[df.Unique_Veh_ID == id3, ["Global_X"]].squeeze()
    c2 = df.loc[df.Unique_Veh_ID == id3, ["Global_Y"]].squeeze()
    dist1 = math.sqrt((a1 - b1)** 2 + (a2 - b2)**2)
    dist2 = math.sqrt((a1 - c1)** 2 + (a2 - c2)**2)
    dist3 = math.sqrt((c1 - b1)** 2 + (c2 - b2)**2)
    return (dist1, dist2, dist3)







