"""
Interactive Geographical Visualizations using Folium for PathSnap
Generates multi-layer interactive HTML maps displaying:
- Raw Noisy GPS points (Blue markers with kinematics tooltips)
- Ground Truth Reference Route (Solid Green polyline)
- PathSnap Reconstructed Route (Red dashed polyline)
- Road Network Graph Topology (Subtle Gray lines)
- Origin and Destination Markers
- Dynamic Folium LayerControl and Floating Legend
"""

import os
import numpy as np
import pandas as pd
import folium
from folium import plugins
from src.features.builder import FEATURE_COLUMNS
from src.data.geo_utils import compute_metric_scales

def generate_interactive_map(trip_id, predicted_arcs=None, partition='student_test', output_html=None):
    base_dir = os.path.join('data', partition, trip_id)
    if not os.path.exists(base_dir):
        base_dir = os.path.join('data', 'train', trip_id)
    if not os.path.exists(base_dir):
        base_dir = trip_id

    track_file = os.path.join(base_dir, f"{trip_id}.track")
    nodes_file = os.path.join(base_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(base_dir, f"{trip_id}.arcs")
    route_file = os.path.join(base_dir, f"{trip_id}.route")

    track = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(nodes_file, sep=r'\s+', header=None, names=['lon', 'lat']).values
    arcs = pd.read_csv(arcs_file, sep=r'\s+', header=None, names=['u', 'v']).values

    center_lat = track['lat'].mean()
    center_lon = track['lon'].mean()

    # Base Folium Map using OpenStreetMap default tiles
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='OpenStreetMap')

    # Bounding box
    pad = 0.008
    min_lon, max_lon = track['lon'].min() - pad, track['lon'].max() + pad
    min_lat, max_lat = track['lat'].min() - pad, track['lat'].max() + pad

    # 1. Road Network Layer (Background)
    fg_network = folium.FeatureGroup(name='1. Road Network Graph (Background)', show=False)
    arc_count = 0
    for u, v in arcs:
        n1 = nodes[u]
        n2 = nodes[v]
        if (min_lon <= n1[0] <= max_lon and min_lat <= n1[1] <= max_lat) or \
           (min_lon <= n2[0] <= max_lon and min_lat <= n2[1] <= max_lat):
            folium.PolyLine(
                locations=[[n1[1], n1[0]], [n2[1], n2[0]]],
                color='#888888', weight=1.2, opacity=0.45
            ).add_to(fg_network)
            arc_count += 1
            if arc_count > 4000:
                break
    fg_network.add_to(m)

    # 2. Ground Truth Reference Route
    if os.path.exists(route_file):
        route = pd.read_csv(route_file, sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values
        fg_truth = folium.FeatureGroup(name=f'2. Ground Truth Reference Route ({len(route)} arcs)', show=True)
        route_coords = []
        for a in route:
            u, v = arcs[a]
            route_coords.append([nodes[u][1], nodes[u][0]])
        last_u, last_v = arcs[route[-1]]
        route_coords.append([nodes[last_v][1], nodes[last_v][0]])

        folium.PolyLine(
            locations=route_coords,
            color='#28a745', weight=5.5, opacity=0.85,
            tooltip=f"Ground Truth Reference Path ({len(route)} arcs)"
        ).add_to(fg_truth)
        fg_truth.add_to(m)

    # 3. PathSnap Reconstructed Route
    if predicted_arcs is not None and len(predicted_arcs) > 0:
        fg_pred = folium.FeatureGroup(name=f'3. PathSnap Reconstructed Route ({len(predicted_arcs)} arcs)', show=True)
        pred_coords = []
        for a in predicted_arcs:
            u, v = arcs[a]
            pred_coords.append([nodes[u][1], nodes[u][0]])
        last_u, last_v = arcs[predicted_arcs[-1]]
        pred_coords.append([nodes[last_v][1], nodes[last_v][0]])

        folium.PolyLine(
            locations=pred_coords,
            color='#dc3545', weight=4.0, opacity=0.9, dash_array='7, 7',
            tooltip=f"PathSnap HMM Reconstructed Path ({len(predicted_arcs)} observations)"
        ).add_to(fg_pred)
        fg_pred.add_to(m)

    # 4. Raw Noisy GPS Observations Layer
    fg_gps = folium.FeatureGroup(name=f'4. Raw GPS Observations ({len(track)} points)', show=True)
    step = max(1, len(track) // 400)
    for idx in range(0, len(track), step):
        row = track.iloc[idx]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3.5,
            color='#0056b3',
            fill=True,
            fill_color='#007bff',
            fill_opacity=0.75,
            weight=1,
            tooltip=f"Obs #{idx} | Time: {row['time']}s | Lat: {row['lat']:.5f}, Lon: {row['lon']:.5f}"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # 5. Start and End Points
    folium.Marker(
        location=[track.iloc[0]['lat'], track.iloc[0]['lon']],
        popup=f"Origin: Step 0 ({track.iloc[0]['time']}s)",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    folium.Marker(
        location=[track.iloc[-1]['lat'], track.iloc[-1]['lon']],
        popup=f"Destination: Step {len(track)-1} ({track.iloc[-1]['time']}s)",
        icon=folium.Icon(color="red", icon="flag")
    ).add_to(m)

    # Add Legend HTML
    legend_html = '''
    <div style="position: fixed; 
                bottom: 30px; left: 30px; width: 260px; height: 160px; 
                background-color: white; border:2px solid #333; z-index:9999; 
                font-size:12px; font-family:sans-serif; padding: 12px; border-radius: 8px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">
        <b style="font-size:13px; color:#111;">PathSnap Map Legend</b><br>
        <div style="margin-top:6px;">
            <span style="display:inline-block; width:18px; height:5px; background:#007bff; margin-right:8px; border-radius:3px;"></span>
            <b>Blue Dots:</b> Raw Noisy GPS<br>
            <span style="display:inline-block; width:18px; height:5px; background:#28a745; margin-right:8px; border-radius:3px;"></span>
            <b>Solid Green:</b> Ground Truth Route<br>
            <span style="display:inline-block; width:18px; height:4px; border-bottom:3px dashed #dc3545; margin-right:8px;"></span>
            <b>Dashed Red:</b> PathSnap Reconstructed<br>
            <span style="display:inline-block; width:18px; height:3px; background:#888888; margin-right:8px;"></span>
            <b>Gray:</b> Road Network Segments
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)

    if output_html is None:
        output_html = os.path.join('visualizations', 'maps', f'route_map_{trip_id}.html')
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"Interactive Folium map saved to: {output_html}")
    return output_html

def generate_sample_maps():
    """Generates sample maps for key demonstration trips."""
    import pickle
    from src.reconstruction.hmm_decoder import MapMatchingHMM

    print("Generating interactive Folium maps for demonstration trips...")
    with open('artifacts/models/tuned_lightgbm.pkl', 'rb') as f:
        model = pickle.load(f)

    test_df = pd.read_parquet('artifacts/student_test_features.parquet')
    X_test = test_df[FEATURE_COLUMNS].values
    test_df['pred_prob'] = model.predict_proba(X_test)[:, 1]

    hmm = MapMatchingHMM()

    sample_trips = ['00000000', '00000007', '00000040']
    for tid in sample_trips:
        tdir = os.path.join('data', 'student_test', tid)
        arcs = pd.read_csv(os.path.join(tdir, f"{tid}.arcs"), sep=r'\s+', header=None, names=['u', 'v']).values
        nodes = pd.read_csv(os.path.join(tdir, f"{tid}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat']).values
        track = pd.read_csv(os.path.join(tdir, f"{tid}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])

        lat_mean = track['lat'].mean()
        scale_x, scale_y = compute_metric_scales(lat_mean)
        nodes_xy = nodes * np.array([scale_x, scale_y])

        trip_cands = test_df[test_df['trip_id'] == tid]
        pred_arcs = hmm.decode_trip(trip_cands, arcs, nodes_xy)

        out_path = os.path.join('visualizations', 'maps', f'map_{tid}.html')
        generate_interactive_map(tid, pred_arcs, partition='student_test', output_html=out_path)

if __name__ == '__main__':
    generate_sample_maps()
