"""
Hidden Markov Model (HMM) and Viterbi Route Reconstruction for PathSnap
Implements:
- Hidden State Representation: Candidate road segments at step t.
- Emission Probability: Calibrated ML classifier probability + Gaussian distance likelihood.
- Transition Probability: Newson & Krumm exponential formulation on shortest path vs Euclidean distance difference.
- Log-Domain Viterbi Recurrence with Backtracking.
- Robust Handling of GPS Gaps and Network Disconnections.
- Direct Comparison of Classifier-Only vs Classifier + HMM.
"""

import os
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from src.features.builder import FEATURE_COLUMNS
from src.data.geo_utils import compute_metric_scales, pt_to_segs_dists_vectorized

class MapMatchingHMM:
    def __init__(self, sigma_z=8.0, beta=6.0, gamma=0.6, disconnected_penalty=25.0):
        """
        sigma_z: Standard deviation for GPS measurement noise (meters).
        beta: Scale parameter for transition probability exponential distribution (meters).
        gamma: Weight between classifier probability and spatial measurement likelihood.
        disconnected_penalty: Log-space penalty for topologically disconnected candidate pairs.
        """
        self.sigma_z = sigma_z
        self.beta = beta
        self.gamma = gamma
        self.disconnected_penalty = disconnected_penalty

    def build_network_graph(self, arcs, nodes_xy):
        """Builds directed NetworkX graph with physical edge lengths."""
        G = nx.DiGraph()
        u_coords = nodes_xy[arcs[:, 0]]
        v_coords = nodes_xy[arcs[:, 1]]
        lens = np.maximum(np.linalg.norm(v_coords - u_coords, axis=1), 0.1)

        for a_id, (u, v) in enumerate(arcs):
            G.add_edge(u, v, weight=lens[a_id], arc_id=a_id)
        return G

    def decode_trip(self, trip_cands_df, arcs, nodes_xy, G=None):
        """
        Runs Viterbi decoding over a single trip's candidate sequence.
        trip_cands_df must contain: obs_idx, arc_id, dist_perp, pred_prob, proj_ratio, dt
        """
        if G is None:
            G = self.build_network_graph(arcs, nodes_xy)

        # Group candidates by observation step
        obs_groups = [group for _, group in trip_cands_df.groupby('obs_idx')]
        T = len(obs_groups)
        if T == 0:
            return []

        # Precompute arc lengths
        u_coords = nodes_xy[arcs[:, 0]]
        v_coords = nodes_xy[arcs[:, 1]]
        arc_lens = np.maximum(np.linalg.norm(v_coords - u_coords, axis=1), 0.1)

        # Precompute GPS coords
        # Viterbi tables
        # V[t] will be a 1D array of log probabilities for each candidate at step t
        backpointers = []
        
        # Step 0: Initialize
        init_cands = obs_groups[0]
        K_0 = len(init_cands)
        v_prev = np.zeros(K_0)

        d_perp_0 = init_cands['dist_perp'].values
        p_ml_0 = np.clip(init_cands['pred_prob'].values, 1e-6, 1.0)
        # Log emission: - (d^2)/(2*sigma^2) + gamma * log(P_ml)
        log_emission_0 = - (d_perp_0**2) / (2.0 * self.sigma_z**2) + self.gamma * np.log(p_ml_0)
        v_prev = log_emission_0

        # Dynamic programming trellis
        # Cache shortest path distances between nodes
        sp_cache = {}

        def get_shortest_path_dist(u, v, max_cutoff=250.0):
            pair = (u, v)
            if pair in sp_cache:
                return sp_cache[pair]
            if u == v:
                sp_cache[pair] = 0.0
                return 0.0
            try:
                # Use cutoff to keep search instantaneous
                d = nx.dijkstra_path_length(G, u, v, weight='weight')
                sp_cache[pair] = d
                return d
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                sp_cache[pair] = None
                return None

        for t in range(1, T):
            prev_cands = obs_groups[t - 1]
            curr_cands = obs_groups[t]
            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            dt = curr_cands['dt'].iloc[0]
            # Effective beta scales if large GPS gap occurs
            beta_eff = self.beta * max(1.0, np.sqrt(dt))

            curr_d_perp = curr_cands['dist_perp'].values
            curr_p_ml = np.clip(curr_cands['pred_prob'].values, 1e-6, 1.0)
            log_emission = - (curr_d_perp**2) / (2.0 * self.sigma_z**2) + self.gamma * np.log(curr_p_ml)

            # Trajectory step distance
            step_speed = curr_cands['speed'].iloc[0]
            d_euclid = step_speed * dt

            v_curr = np.zeros(K_curr)
            bp_curr = np.zeros(K_curr, dtype=int)

            prev_arcs = prev_cands['arc_id'].values
            prev_proj = prev_cands['proj_ratio'].values
            curr_arcs = curr_cands['arc_id'].values
            curr_proj = curr_cands['proj_ratio'].values

            for j in range(K_curr):
                a_curr = curr_arcs[j]
                pr_curr = curr_proj[j]
                len_curr = arc_lens[a_curr]
                u_curr = arcs[a_curr, 0]

                # Compute transition from each candidate k in step t-1
                trans_scores = np.zeros(K_prev)
                for k in range(K_prev):
                    a_prev = prev_arcs[k]
                    pr_prev = prev_proj[k]
                    len_prev = arc_lens[a_prev]
                    v_prev_node = arcs[a_prev, 1]

                    if a_prev == a_curr:
                        # Same segment
                        delta_prog = max(0.0, (pr_curr - pr_prev) * len_curr)
                        d_net = delta_prog
                        delta_d = abs(d_net - d_euclid)
                        log_trans = - (delta_d / beta_eff)
                    else:
                        # Network path from v_prev_node to u_curr
                        d_path = get_shortest_path_dist(v_prev_node, u_curr)
                        if d_path is not None:
                            d_net = (1.0 - pr_prev) * len_prev + d_path + pr_curr * len_curr
                            delta_d = abs(d_net - d_euclid)
                            log_trans = - (delta_d / beta_eff)
                        else:
                            # Graph disconnection penalty
                            log_trans = - self.disconnected_penalty

                    trans_scores[k] = v_prev[k] + log_trans

                best_prev_idx = np.argmax(trans_scores)
                v_curr[j] = trans_scores[best_prev_idx] + log_emission[j]
                bp_curr[j] = best_prev_idx

            backpointers.append(bp_curr)
            v_prev = v_curr

        # Termination & Backtracking
        best_last_idx = np.argmax(v_prev)
        opt_path_indices = [best_last_idx]

        for t in range(T - 2, -1, -1):
            prev_idx = backpointers[t][opt_path_indices[-1]]
            opt_path_indices.append(prev_idx)

        opt_path_indices.reverse()

        # Extract predicted arc IDs
        decoded_arcs = []
        for t in range(T):
            cands_t = obs_groups[t]
            idx = opt_path_indices[t]
            decoded_arcs.append(cands_t['arc_id'].iloc[idx])

        return decoded_arcs

def evaluate_reconstruction_pipeline():
    """
    Evaluates Tuned Classifier alone vs Tuned Classifier + HMM Viterbi
    across all 10 student-test trips.
    """
    print("Loading test features and tuned champion model...")
    test_df = pd.read_parquet('artifacts/student_test_features.parquet')
    
    with open('artifacts/models/tuned_lightgbm.pkl', 'rb') as f:
        model = pickle.load(f)

    # Compute classifier probabilities
    X_test = test_df[FEATURE_COLUMNS].values
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

    hmm = MapMatchingHMM(sigma_z=8.0, beta=6.0, gamma=0.65, disconnected_penalty=20.0)

    test_dir = os.path.join('data', 'student_test')
    trip_ids = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    trip_results = []
    all_classifier_preds = []
    all_hmm_preds = []
    all_ground_truth = []

    print(f"\nRunning sequential route reconstruction across {len(trip_ids)} student-test trips...\n")

    for tid in trip_ids:
        tdir = os.path.join(test_dir, tid)
        track_df = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat']).values
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['u', 'v']).values
        route = pd.read_csv(os.path.join(tdir, f"{tid}.route"), sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values

        lat_mean = track_df['lat'].mean()
        scale_x, scale_y = compute_metric_scales(lat_mean)
        nodes_xy = nodes * np.array([scale_x, scale_y])

        trip_cands = test_df[test_df['trip_id'] == tid].copy()
        
        # 1. Classifier-Only Top-1 Prediction
        clf_top1 = trip_cands.loc[trip_cands.groupby('obs_idx')['pred_prob'].idxmax()]
        clf_pred_arcs = clf_top1['arc_id'].values
        true_arcs = clf_top1['true_arc'].values

        clf_acc = np.mean(clf_pred_arcs == true_arcs) * 100.0

        # Classifier continuity
        clf_conn = 0
        for i in range(len(clf_pred_arcs) - 1):
            a1, a2 = clf_pred_arcs[i], clf_pred_arcs[i + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0] or arcs[a1, 1] == arcs[a2, 1] or arcs[a1, 0] == arcs[a2, 0]:
                clf_conn += 1
        clf_continuity = (clf_conn / max(len(clf_pred_arcs) - 1, 1)) * 100.0

        # 2. Classifier + HMM Viterbi Route Reconstruction
        t0 = time.time()
        hmm_pred_arcs = hmm.decode_trip(trip_cands, arcs, nodes_xy)
        decode_time = time.time() - t0

        hmm_pred_arcs = np.array(hmm_pred_arcs)
        hmm_acc = np.mean(hmm_pred_arcs == true_arcs) * 100.0

        hmm_conn = 0
        for i in range(len(hmm_pred_arcs) - 1):
            a1, a2 = hmm_pred_arcs[i], hmm_pred_arcs[i + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0] or arcs[a1, 1] == arcs[a2, 1] or arcs[a1, 0] == arcs[a2, 0]:
                hmm_conn += 1
        hmm_continuity = (hmm_conn / max(len(hmm_pred_arcs) - 1, 1)) * 100.0

        print(f"Trip {tid} ({len(track_df)} pts, {len(route)} route arcs, decoded in {decode_time:.2f}s):")
        print(f"  Classifier-Only -> Top-1 Acc: {clf_acc:.2f}%, Continuity: {clf_continuity:.2f}%")
        print(f"  Classifier + HMM -> Top-1 Acc: {hmm_acc:.2f}%, Continuity: {hmm_continuity:.2f}% (Improvement: {hmm_continuity - clf_continuity:+.2f}%)\n")

        trip_results.append({
            'trip_id': tid,
            'points': len(track_df),
            'route_arcs': len(route),
            'clf_top1_acc': clf_acc,
            'hmm_top1_acc': hmm_acc,
            'clf_continuity': clf_continuity,
            'hmm_continuity': hmm_continuity,
            'decode_time_s': decode_time
        })

    summary_df = pd.DataFrame(trip_results)
    csv_out = os.path.join('artifacts', 'metrics', 'hmm_reconstruction_results.csv')
    summary_df.to_csv(csv_out, index=False)

    print("=== Consolidated HMM Route Reconstruction Evaluation ===")
    print(summary_df.to_string(index=False))

    mean_clf_acc = summary_df['clf_top1_acc'].mean()
    mean_hmm_acc = summary_df['hmm_top1_acc'].mean()
    mean_clf_conn = summary_df['clf_continuity'].mean()
    mean_hmm_conn = summary_df['hmm_continuity'].mean()

    print("\n=== Overall Student-Testing Averages ===")
    print(f"Classifier-Only Average Top-1 Accuracy:  {mean_clf_acc:.2f}%")
    print(f"Classifier + HMM Average Top-1 Accuracy: {mean_hmm_acc:.2f}%")
    print(f"Classifier-Only Average Route Continuity: {mean_clf_conn:.2f}%")
    print(f"Classifier + HMM Average Route Continuity: {mean_hmm_conn:.2f}%")
    print(f"Route Continuity Improvement:            {mean_hmm_conn - mean_clf_conn:+.2f}%\n")

    return summary_df

if __name__ == '__main__':
    evaluate_reconstruction_pipeline()
