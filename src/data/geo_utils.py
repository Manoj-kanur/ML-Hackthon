"""
Geospatial and Road Network Mathematical Utilities for PathSnap
"""

import numpy as np

def compute_metric_scales(lat_mean):
    """
    Computes meters per degree for latitude and longitude at a given mean latitude.
    """
    scale_y = 111280.0
    scale_x = 111280.0 * np.cos(np.radians(lat_mean))
    return scale_x, scale_y

def pt_to_segs_dists_vectorized(p, u, v, vec, len_sq):
    """
    Computes perpendicular distance and projection ratio t for point p against multiple segments (u -> v).
    p: (2,)
    u, v, vec: (N, 2)
    len_sq: (N,)
    Returns: dists (N,), t (N,)
    """
    ap = p - u
    dot = np.sum(ap * vec, axis=1)
    t = np.clip(dot / np.maximum(len_sq, 1e-9), 0.0, 1.0)
    proj = u + t[:, None] * vec
    dist = np.linalg.norm(p - proj, axis=1)
    return dist, t

def angle_diff_degrees(angle1_rad, angle2_rad):
    """
    Computes absolute angular difference between two angles in degrees [0, 180].
    """
    diff = np.abs(angle1_rad - angle2_rad) % (2.0 * np.pi)
    diff = np.where(diff > np.pi, 2.0 * np.pi - diff, diff)
    return np.degrees(diff)

def extract_monotonic_route_labels(t_xy, route, arcs, nodes_xy):
    """
    Extracts the ground truth matching arc for each GPS observation by
    tracing monotonic progression along the reference route segments.
    """
    route_u = nodes_xy[arcs[route, 0]]
    route_v = nodes_xy[arcs[route, 1]]
    route_vec = route_v - route_u
    route_len_sq = np.sum(route_vec**2, axis=1)

    true_arcs = []
    cur_idx = 0
    num_route = len(route)

    for i, p in enumerate(t_xy):
        # Windowed search around cur_idx (allows small reversals and forward moves)
        w_start = max(0, cur_idx - 6)
        w_end = min(num_route, cur_idx + 30)
        
        dists, _ = pt_to_segs_dists_vectorized(
            p,
            route_u[w_start:w_end],
            route_v[w_start:w_end],
            route_vec[w_start:w_end],
            route_len_sq[w_start:w_end]
        )
        best_local = np.argmin(dists)
        cur_idx = w_start + best_local
        true_arcs.append(route[cur_idx])

    return np.array(true_arcs, dtype=np.int64)
