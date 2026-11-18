"""
Evaluation on Evaluator-Test Partition (10 Reserved Evaluator Trips)
"""

import os
import time
import pickle
import pandas as pd
import numpy as np
from src.features.builder import FEATURE_COLUMNS
from src.data.geo_utils import compute_metric_scales
from src.reconstruction.hmm_decoder import MapMatchingHMM

def evaluate_evaluator_partition():
    print("Loading evaluator test features and tuned champion model...")
    eval_df = pd.read_parquet('artifacts/evaluator_test_features.parquet')

    with open('artifacts/models/tuned_lightgbm.pkl', 'rb') as f:
        model = pickle.load(f)

    # Compute classifier probabilities
    X_eval = eval_df[FEATURE_COLUMNS].values
    eval_df['pred_prob'] = model.predict_proba(X_eval)[:, 1]

    hmm = MapMatchingHMM(sigma_z=8.0, beta=6.0, gamma=0.65, disconnected_penalty=20.0)

    eval_dir = os.path.join('data', 'evaluator_test')
    trip_ids = sorted(os.listdir(eval_dir))

    trip_results = []
    print(f"Running sequential route reconstruction across {len(trip_ids)} evaluator-test trips...\n")

    for tid in trip_ids:
        tdir = os.path.join(eval_dir, tid)
        track_df = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat']).values
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['u', 'v']).values
        route = pd.read_csv(os.path.join(tdir, f"{tid}.route"), sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values

        lat_mean = track_df['lat'].mean()
        scale_x, scale_y = compute_metric_scales(lat_mean)
        nodes_xy = nodes * np.array([scale_x, scale_y])

        trip_cands = eval_df[eval_df['trip_id'] == tid].copy()
        
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

        has_match = (trip_cands.groupby('obs_idx')['is_match'].max().values > 0)
        miss_rate = float((1.0 - np.mean(has_match)) * 100.0)

        print(f"Trip {tid} ({len(track_df)} pts, {len(route)} route arcs, decoded in {decode_time:.2f}s):")
        print(f"  Classifier-Only -> Top-1 Acc: {clf_acc:.2f}%, Continuity: {clf_continuity:.2f}%")
        print(f"  Classifier + HMM -> Top-1 Acc: {hmm_acc:.2f}%, Continuity: {hmm_continuity:.2f}% (Gain: {hmm_continuity - clf_continuity:+.2f}%)")
        print(f"  Candidate Miss Rate: {miss_rate:.2f}%\n")

        trip_results.append({
            'trip_id': tid,
            'points': len(track_df),
            'route_arcs': len(route),
            'clf_top1_acc': clf_acc,
            'hmm_top1_acc': hmm_acc,
            'clf_continuity': clf_continuity,
            'hmm_continuity': hmm_continuity,
            'candidate_miss_rate': miss_rate,
            'decode_time_s': decode_time
        })

    res_df = pd.DataFrame(trip_results)
    csv_out = os.path.join('artifacts', 'metrics', 'evaluator_test_results.csv')
    res_df.to_csv(csv_out, index=False)

    print("=== Consolidated Evaluator-Test Performance Table ===")
    print(res_df.to_string(index=False))

    mean_clf_acc = res_df['clf_top1_acc'].mean()
    mean_hmm_acc = res_df['hmm_top1_acc'].mean()
    mean_clf_conn = res_df['clf_continuity'].mean()
    mean_hmm_conn = res_df['hmm_continuity'].mean()

    print("\n=== Overall Evaluator-Testing Averages ===")
    print(f"Classifier-Only Average Top-1 Accuracy:  {mean_clf_acc:.2f}%")
    print(f"Classifier + HMM Average Top-1 Accuracy: {mean_hmm_acc:.2f}%")
    print(f"Classifier-Only Average Route Continuity: {mean_clf_conn:.2f}%")
    print(f"Classifier + HMM Average Route Continuity: {mean_hmm_conn:.2f}%")
    print(f"Route Continuity Improvement:            {mean_hmm_conn - mean_clf_conn:+.2f}%\n")

if __name__ == '__main__':
    evaluate_evaluator_partition()
