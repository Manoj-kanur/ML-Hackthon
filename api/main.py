"""
PathSnap FastAPI Backend Server (Laptop 1)
Hosts the trained machine learning classifier, HMM/Viterbi route reconstructor,
and server-side ground-truth evaluation process.
"""

import os
import time
import pickle
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from src.features.builder import FEATURE_COLUMNS, process_single_trip
from src.data.geo_utils import compute_metric_scales
from src.reconstruction.hmm_decoder import MapMatchingHMM

app = FastAPI(
    title="PathSnap Map-Matching API",
    description="Distributed Map-Matching & Route Reconstruction Backend (Laptop 1)",
    version="1.0.0"
)

# Enable CORS for two-laptop local network communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global runtime state
MODEL = None
HMM_DECODER = None
TEST_DATA_DIR = os.path.join("data", "evaluator_test") if os.path.exists(os.path.join("data", "evaluator_test")) else os.path.join("data", "student_test")
TEST_FEATS_CACHE = None

@app.on_event("startup")
def load_artifacts():
    global MODEL, HMM_DECODER, TEST_FEATS_CACHE
    print("=" * 60)
    print(" [PathSnap Server Startup] Initializing Server on Laptop 1...")
    
    model_path = os.path.join("artifacts", "models", "tuned_lightgbm.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join("artifacts", "models", "lightgbm.pkl")
    
    print(f" [PathSnap Server] Loading trained model from: {model_path}")
    with open(model_path, "rb") as f:
        MODEL = pickle.load(f)
    print(" [PathSnap Server] Model successfully loaded.")

    print(" [PathSnap Server] Initializing HMM Viterbi Decoder...")
    HMM_DECODER = MapMatchingHMM(sigma_z=8.0, beta=6.0, gamma=0.65, disconnected_penalty=20.0)
    print(" [PathSnap Server] HMM Decoder initialized.")

    # Load feature caches
    dfs = []
    for cname in ["evaluator_test_features.parquet", "student_test_features.parquet"]:
        cpath = os.path.join("artifacts", cname)
        if os.path.exists(cpath):
            print(f" [PathSnap Server] Loading candidate feature cache: {cpath}")
            dfs.append(pd.read_parquet(cpath))
    if dfs:
        TEST_FEATS_CACHE = pd.concat(dfs, ignore_index=True)
    
    print(f" [PathSnap Server] Active Evaluation Directory: {TEST_DATA_DIR}")
    print(f" [PathSnap Server] Available Trips: {get_available_trips_list()}")
    print(" [PathSnap Server] Ready to receive evaluation requests!")
    print("=" * 60)

class EvaluateRequest(BaseModel):
    trip_id: str

@app.get("/health")
def health_check():
    return {
        "status": "online",
        "service": "PathSnap Backend (Laptop 1)",
        "model_loaded": MODEL is not None,
        "available_evaluation_trips": get_available_trips_list()
    }

def get_available_trips_list():
    if os.path.exists(TEST_DATA_DIR):
        trips = sorted([d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))])
        return trips
    return []

@app.get("/api/trips")
def list_trips():
    """Lists the 10 evaluation trips with basic metadata."""
    trips_info = []
    for tid in get_available_trips_list():
        tdir = os.path.join(TEST_DATA_DIR, tid)
        track_file = os.path.join(tdir, f"{tid}.track")
        route_file = os.path.join(tdir, f"{tid}.route")
        
        pts = 0
        dur = 0.0
        if os.path.exists(track_file):
            t_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
            pts = len(t_df)
            dur = t_df['time'].iloc[-1] - t_df['time'].iloc[0] if pts > 1 else 0
            
        r_arcs = 0
        if os.path.exists(route_file):
            with open(route_file) as f:
                r_arcs = sum(1 for _ in f)

        trips_info.append({
            "trip_id": tid,
            "num_points": pts,
            "duration_s": round(dur, 1),
            "num_route_arcs": r_arcs
        })
    return {"trips": trips_info}

@app.post("/api/evaluate")
def evaluate_trip(req: EvaluateRequest, request: Request):
    """
    Core Evaluation Endpoint:
    Processes the selected trip through candidate generation -> tuned model -> HMM Viterbi,
    calculates server-side accuracy against ground truth, and logs activity to console.
    """
    client_ip = request.client.host if request.client else "unknown"
    t_start = time.time()
    trip_id = req.trip_id.strip()

    print("\n" + "-" * 50)
    print(f" [INFERENCE REQUEST] Received evaluation request from Client [{client_ip}]")
    print(f" [INFERENCE REQUEST] Trip ID: {trip_id}")

    trip_dir = os.path.join(TEST_DATA_DIR, trip_id)
    if not os.path.exists(trip_dir):
        # Fallback to train directory if requested
        trip_dir = os.path.join("data", "train", trip_id)
        if not os.path.exists(trip_dir):
            print(f" [ERROR] Trip ID {trip_id} not found on server.")
            raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found on server.")

    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    # Load trip components
    track = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat']).values
    arcs = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v']).values

    # Step 1: Candidate Generation and Feature Extraction
    t_feat_0 = time.time()
    if TEST_FEATS_CACHE is not None and trip_id in TEST_FEATS_CACHE['trip_id'].values:
        trip_cands = TEST_FEATS_CACHE[TEST_FEATS_CACHE['trip_id'] == trip_id].copy()
    else:
        trip_cands = process_single_trip(trip_dir, trip_id, has_ground_truth=True)
    t_feat = time.time() - t_feat_0

    # Step 2: Classifier Scoring
    t_clf_0 = time.time()
    X_cand = trip_cands[FEATURE_COLUMNS].values
    trip_cands['pred_prob'] = MODEL.predict_proba(X_cand)[:, 1]
    t_clf = time.time() - t_clf_0

    # Classifier Top-1 Arc Predictions
    clf_top1 = trip_cands.loc[trip_cands.groupby('obs_idx')['pred_prob'].idxmax()]
    clf_pred_arcs = clf_top1['arc_id'].values

    # Step 3: HMM Viterbi Sequential Decoding
    t_hmm_0 = time.time()
    lat_mean = track['lat'].mean()
    scale_x, scale_y = compute_metric_scales(lat_mean)
    nodes_xy = nodes * np.array([scale_x, scale_y])

    hmm_pred_arcs = np.array(HMM_DECODER.decode_trip(trip_cands, arcs, nodes_xy))
    t_hmm = time.time() - t_hmm_0

    # Step 4: Server-Side Ground Truth Accuracy Calculation
    has_truth = os.path.exists(route_file)
    true_arcs = clf_top1['true_arc'].values if 'true_arc' in clf_top1.columns else None

    # Top-1 Accuracy
    if true_arcs is not None and len(true_arcs) > 0 and (true_arcs != -1).any():
        clf_acc = float(np.mean(clf_pred_arcs == true_arcs) * 100.0)
        hmm_acc = float(np.mean(hmm_pred_arcs == true_arcs) * 100.0)
        has_match = (trip_cands.groupby('obs_idx')['is_match'].max().values > 0)
        miss_rate = float((1.0 - np.mean(has_match)) * 100.0)
    else:
        clf_acc = None
        hmm_acc = None
        miss_rate = None

    # Route Continuity Calculation
    def get_continuity(pred_sequence):
        conn = 0
        for i in range(len(pred_sequence) - 1):
            a1, a2 = pred_sequence[i], pred_sequence[i + 1]
            if a1 == a2 or arcs[a1, 1] == arcs[a2, 0] or arcs[a1, 1] == arcs[a2, 1] or arcs[a1, 0] == arcs[a2, 0]:
                conn += 1
        return (conn / max(len(pred_sequence) - 1, 1)) * 100.0

    clf_continuity = float(get_continuity(clf_pred_arcs))
    hmm_continuity = float(get_continuity(hmm_pred_arcs))

    total_latency_ms = round((time.time() - t_start) * 1000.0, 1)

    print(f" [INFERENCE RESULT] Observations: {len(track)}")
    print(f" [INFERENCE RESULT] Feature Time: {t_feat*1000:.1f}ms | Classifier Time: {t_clf*1000:.1f}ms | HMM Time: {t_hmm*1000:.1f}ms")
    print(f" [INFERENCE RESULT] Classifier Top-1: {clf_acc:.2f}% | HMM Reconstructed Top-1: {hmm_acc:.2f}%")
    print(f" [INFERENCE RESULT] Route Continuity: {hmm_continuity:.2f}% (Classifier: {clf_continuity:.2f}%)")
    print(f" [INFERENCE RESULT] Total Latency: {total_latency_ms} ms")
    print("-" * 50 + "\n")

    # Step 5: Format Coordinates for Map Rendering
    # Sample raw GPS coords
    sample_step = max(1, len(track) // 250)
    raw_gps_coords = [[row['lat'], row['lon']] for _, row in track.iloc[::sample_step].iterrows()]

    # Ground truth route coords
    ground_truth_coords = []
    if has_truth:
        route = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values
        for a in route:
            u, v = arcs[a]
            ground_truth_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
        last_u, last_v = arcs[route[-1]]
        ground_truth_coords.append([float(nodes[last_v, 1]), float(nodes[last_v, 0])])

    # Predicted route coords
    predicted_coords = []
    for a in hmm_pred_arcs:
        u, v = arcs[a]
        predicted_coords.append([float(nodes[u, 1]), float(nodes[u, 0])])
    if len(hmm_pred_arcs) > 0:
        last_u, last_v = arcs[hmm_pred_arcs[-1]]
        predicted_coords.append([float(nodes[last_v, 1]), float(nodes[last_v, 0])])

    return {
        "trip_id": trip_id,
        "total_points": len(track),
        "total_time_ms": total_latency_ms,
        "metrics": {
            "top1_accuracy": round(hmm_acc, 2) if hmm_acc is not None else "N/A",
            "classifier_accuracy": round(clf_acc, 2) if clf_acc is not None else "N/A",
            "route_continuity": round(hmm_continuity, 2),
            "classifier_continuity": round(clf_continuity, 2),
            "candidate_miss_rate": round(miss_rate, 2) if miss_rate is not None else "N/A",
            "feature_time_ms": round(t_feat * 1000, 1),
            "classifier_time_ms": round(t_clf * 1000, 1),
            "hmm_time_ms": round(t_hmm * 1000, 1)
        },
        "map_data": {
            "center": [float(track['lat'].mean()), float(track['lon'].mean())],
            "raw_gps": raw_gps_coords,
            "predicted_route": predicted_coords,
            "ground_truth_route": ground_truth_coords
        }
    }

# Serve Frontend UI
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
if os.path.exists(UI_DIR):
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    html_file = os.path.join(UI_DIR, "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>PathSnap Server Running. UI files not found.</h1>"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
