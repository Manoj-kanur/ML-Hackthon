"""
Candidate Generation and Feature Extraction Pipeline for PathSnap
Extracts geometric, kinematic, topological, and temporal features for all candidates.
"""

import os
import glob
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from src.data.geo_utils import (
    compute_metric_scales,
    pt_to_segs_dists_vectorized,
    angle_diff_degrees,
    extract_monotonic_route_labels
)

FEATURE_COLUMNS = [
    'dist_perp',
    'proj_ratio',
    'heading_diff',
    'speed',
    'arc_length',
    'u_degree',
    'v_degree',
    'dist_rank',
    'dist_gap_to_best',
    'dt',
    'accel',
    'is_connected_to_prev_closest'
]

def process_single_trip(trip_dir, trip_id, has_ground_truth=True, k_nodes=20, top_cands=6):
    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    # Load data
    track = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat']).values
    arcs = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v']).values
    
    num_obs = len(track)
    if num_obs == 0:
        return pd.DataFrame()

    # Metric coordinates
    lat_mean = track['lat'].mean()
    scale_x, scale_y = compute_metric_scales(lat_mean)
    t_xy = track[['lon', 'lat']].values * np.array([scale_x, scale_y])
    nodes_xy = nodes * np.array([scale_x, scale_y])

    # Arc vectors, lengths, bearings
    u_coords = nodes_xy[arcs[:, 0]]
    v_coords = nodes_xy[arcs[:, 1]]
    arc_vecs = v_coords - u_coords
    arc_lens = np.maximum(np.linalg.norm(arc_vecs, axis=1), 1e-4)
    arc_angles = np.arctan2(arc_vecs[:, 1], arc_vecs[:, 0])

    # Degree calculations
    u_degrees = np.bincount(arcs[:, 0], minlength=len(nodes))
    v_degrees = np.bincount(arcs[:, 1], minlength=len(nodes))

    # Node adjacency mapping
    node_to_arcs = defaultdict(list)
    for a_id, (u, v) in enumerate(arcs):
        node_to_arcs[u].append(a_id)
        node_to_arcs[v].append(a_id)

    node_tree = KDTree(nodes_xy)

    # GPS kinematics: speed, heading, acceleration
    dt_arr = np.diff(track['time'].values, prepend=track['time'].iloc[0] + 1.0)
    dt_arr = np.maximum(dt_arr, 1e-3)
    
    # Trajectory displacement vectors
    d_xy = np.diff(t_xy, axis=0, prepend=t_xy[0:1])
    step_dists = np.linalg.norm(d_xy, axis=1)
    speeds = step_dists / dt_arr
    gps_angles = np.arctan2(d_xy[:, 1], d_xy[:, 0])
    accel = np.diff(speeds, prepend=speeds[0]) / dt_arr

    # Ground truth labels if available
    true_arcs = None
    if has_ground_truth and os.path.exists(route_file):
        route = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values
        if len(route) > 0:
            true_arcs = extract_monotonic_route_labels(t_xy, route, arcs, nodes_xy)

    rows = []
    prev_closest_arc = -1

    for i in range(num_obs):
        p = t_xy[i]
        curr_dt = dt_arr[i]
        curr_speed = speeds[i]
        curr_accel = accel[i]
        curr_gps_angle = gps_angles[i]
        target_arc = true_arcs[i] if true_arcs is not None else -1

        # Query KDTree for nearest nodes
        _, n_idxs = node_tree.query(p, k=min(k_nodes, len(nodes)))
        cand_arcs_set = set()
        for nid in n_idxs:
            cand_arcs_set.update(node_to_arcs[nid])

        cand_list = list(cand_arcs_set)
        if len(cand_list) == 0:
            continue

        cu = nodes_xy[arcs[cand_list, 0]]
        cv = nodes_xy[arcs[cand_list, 1]]
        cvec = cv - cu
        clen_sq = np.sum(cvec**2, axis=1)

        dists, t_proj = pt_to_segs_dists_vectorized(p, cu, cv, cvec, clen_sq)

        # Sort by perpendicular distance
        order = np.argsort(dists)
        selected_indices = order[:min(top_cands, len(order))]

        best_dist = dists[selected_indices[0]]
        closest_arc_this_step = cand_list[selected_indices[0]]

        for rank, idx in enumerate(selected_indices):
            a_id = cand_list[idx]
            d_perp = dists[idx]
            pr = t_proj[idx]
            
            # Heading error: if stopped, angle diff is set to 0.0
            if curr_speed < 1.0:
                h_diff = 0.0
            else:
                h_diff = angle_diff_degrees(curr_gps_angle, arc_angles[a_id])

            a_len = arc_lens[a_id]
            u_deg = u_degrees[arcs[a_id, 0]]
            v_deg = v_degrees[arcs[a_id, 1]]
            gap = d_perp - best_dist

            # Graph adjacency to previous closest arc
            is_conn = 0
            if prev_closest_arc != -1:
                if a_id == prev_closest_arc or arcs[prev_closest_arc, 1] == arcs[a_id, 0] or arcs[a_id, 1] == arcs[prev_closest_arc, 0]:
                    is_conn = 1

            is_match = 1 if (target_arc != -1 and a_id == target_arc) else 0

            rows.append({
                'trip_id': trip_id,
                'obs_idx': i,
                'arc_id': a_id,
                'true_arc': target_arc,
                'dist_perp': d_perp,
                'proj_ratio': pr,
                'heading_diff': h_diff,
                'speed': curr_speed,
                'arc_length': a_len,
                'u_degree': u_deg,
                'v_degree': v_deg,
                'dist_rank': rank + 1,
                'dist_gap_to_best': gap,
                'dt': curr_dt,
                'accel': curr_accel,
                'is_connected_to_prev_closest': is_conn,
                'is_match': is_match
            })

        prev_closest_arc = closest_arc_this_step

    return pd.DataFrame(rows)

def build_dataset(partition='train', max_trips=None):
    base_dir = os.path.join('data', partition)
    trip_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if max_trips:
        trip_dirs = trip_dirs[:max_trips]

    print(f"Building features for '{partition}' ({len(trip_dirs)} trips)...")
    t0 = time.time()
    dfs = []
    
    for idx, tid in enumerate(trip_dirs):
        tdir = os.path.join(base_dir, tid)
        sub_df = process_single_trip(tdir, tid, has_ground_truth=True)
        if not sub_df.empty:
            dfs.append(sub_df)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(trip_dirs):
            print(f"  Processed {idx + 1}/{len(trip_dirs)} trips in {time.time() - t0:.1f}s")

    full_df = pd.concat(dfs, ignore_index=True)
    out_file = os.path.join('artifacts', f"{partition}_features.parquet")
    full_df.to_parquet(out_file, index=False)
    print(f"Saved {partition} features ({len(full_df)} rows) to: {out_file}")
    return full_df

if __name__ == '__main__':
    # Build both train and student_test feature tables
    train_df = build_dataset('train')
    test_df = build_dataset('student_test')
