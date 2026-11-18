"""
Exploratory Data Analysis (EDA) for PathSnap
Analyzes all 80 training trips and 10 student-testing trips.
Computes dataset inventory, sampling statistics, road network properties,
and produces high-resolution figures and interactive Folium maps.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium

def analyze_trip(trip_dir, trip_id, partition):
    track_file = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_file = os.path.join(trip_dir, f"{trip_id}.route")

    # Read track
    track_df = pd.read_csv(track_file, sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    num_points = len(track_df)
    
    # Missing values check
    missing_vals = track_df.isna().sum().sum()

    # Time sampling diffs
    time_diffs = np.diff(track_df['time'].values)
    pct_1hz = np.mean(time_diffs == 1.0) * 100 if len(time_diffs) > 0 else 0
    max_gap = np.max(time_diffs) if len(time_diffs) > 0 else 0
    total_duration = track_df['time'].iloc[-1] - track_df['time'].iloc[0] if num_points > 1 else 0

    # Metric coordinates and distance
    lat_mean = track_df['lat'].mean()
    scale_y = 111280.0
    scale_x = 111280.0 * np.cos(np.radians(lat_mean))
    dx = np.diff(track_df['lon'].values) * scale_x
    dy = np.diff(track_df['lat'].values) * scale_y
    step_dists = np.sqrt(dx**2 + dy**2)
    traj_length_km = np.sum(step_dists) / 1000.0
    speeds = step_dists / np.maximum(time_diffs, 1e-3)
    avg_speed_kmh = np.mean(speeds) * 3.6 if len(speeds) > 0 else 0
    max_speed_kmh = np.max(speeds) * 3.6 if len(speeds) > 0 else 0

    # Read nodes count
    with open(nodes_file, 'r') as f:
        num_nodes = sum(1 for _ in f)

    # Read arcs count
    with open(arcs_file, 'r') as f:
        num_arcs = sum(1 for _ in f)

    # Read route count
    num_route_arcs = 0
    if os.path.exists(route_file):
        with open(route_file, 'r') as f:
            num_route_arcs = sum(1 for _ in f)

    return {
        'trip_id': trip_id,
        'partition': partition,
        'num_points': num_points,
        'duration_s': total_duration,
        'distance_km': traj_length_km,
        'avg_speed_kmh': avg_speed_kmh,
        'max_speed_kmh': max_speed_kmh,
        'pct_1hz': pct_1hz,
        'max_time_gap_s': max_gap,
        'num_nodes': num_nodes,
        'num_arcs': num_arcs,
        'num_route_arcs': num_route_arcs,
        'missing_values': missing_vals,
        'min_lon': track_df['lon'].min(),
        'max_lon': track_df['lon'].max(),
        'min_lat': track_df['lat'].min(),
        'max_lat': track_df['lat'].max(),
        'lat_mean': lat_mean
    }

def run_eda():
    print("Starting comprehensive EDA on all 80 train + 10 student_test trips...")
    records = []

    for partition in ['train', 'student_test']:
        p_dir = os.path.join('data', partition)
        if not os.path.exists(p_dir):
            continue
        trip_ids = sorted([d for d in os.listdir(p_dir) if os.path.isdir(os.path.join(p_dir, d))])
        print(f"Processing {len(trip_ids)} trips in partition '{partition}'...")
        for tid in trip_ids:
            tdir = os.path.join(p_dir, tid)
            rec = analyze_trip(tdir, tid, partition)
            records.append(rec)

    df = pd.DataFrame(records)
    csv_out = os.path.join('artifacts', 'metrics', 'dataset_inventory.csv')
    df.to_csv(csv_out, index=False)
    print(f"Dataset inventory saved to: {csv_out}")

    # Summary table per partition
    summary = df.groupby('partition').agg({
        'trip_id': 'count',
        'num_points': ['sum', 'mean', 'min', 'max'],
        'distance_km': ['mean', 'min', 'max'],
        'pct_1hz': 'mean',
        'max_time_gap_s': 'max',
        'num_nodes': ['mean', 'min', 'max'],
        'num_arcs': ['mean', 'min', 'max'],
        'num_route_arcs': ['mean', 'min', 'max']
    }).round(2)
    print("\n=== Dataset Summary Table ===")
    print(summary)

    # 1. Generate EDA Figures
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Points per trip
    sns.histplot(df['num_points'], kde=True, ax=axes[0, 0], color='#1f77b4', bins=20)
    axes[0, 0].set_title('Distribution of GPS Observations per Trip', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Number of GPS Observations')
    axes[0, 0].set_ylabel('Frequency')

    # Distance per trip
    sns.histplot(df['distance_km'], kde=True, ax=axes[0, 1], color='#2ca02c', bins=20)
    axes[0, 1].set_title('Distribution of Trajectory Length (km)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Trajectory Distance (km)')
    axes[0, 1].set_ylabel('Frequency')

    # Average Speed
    sns.histplot(df['avg_speed_kmh'], kde=True, ax=axes[1, 0], color='#ff7f0e', bins=20)
    axes[1, 0].set_title('Distribution of Average Speed (km/h)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Average Speed (km/h)')
    axes[1, 0].set_ylabel('Frequency')

    # Road network size
    sns.scatterplot(data=df, x='num_nodes', y='num_arcs', hue='partition', palette='deep', ax=axes[1, 1], alpha=0.85, s=70)
    axes[1, 1].set_title('Road Network Graph Complexity (Nodes vs Arcs)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Number of Nodes')
    axes[1, 1].set_ylabel('Number of Directed Arcs')

    plt.tight_layout()
    plot_path = os.path.join('visualizations', 'plots', 'eda_dataset_distributions.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"EDA distribution plots saved to: {plot_path}")

    # 2. Interactive Folium Map for Trip 00000031
    create_sample_folium_map('00000031')

def create_sample_folium_map(trip_id='00000031'):
    trip_dir = os.path.join('data', 'train', trip_id)
    if not os.path.exists(trip_dir):
        trip_dir = trip_id

    track = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.track"), sep=r'\s+', header=None, names=['lon', 'lat', 'time'])
    nodes = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.nodes"), sep=r'\s+', header=None, names=['lon', 'lat']).values
    arcs = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.arcs"), sep=r'\s+', header=None, names=['u', 'v']).values
    route = pd.read_csv(os.path.join(trip_dir, f"{trip_id}.route"), sep=r'\s+', header=None, names=['arc_id'])['arc_id'].values

    center_lat = track['lat'].mean()
    center_lon = track['lon'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='CartoDB positron')

    # Subsample road network around trip bbox
    min_lon, max_lon = track['lon'].min() - 0.01, track['lon'].max() + 0.01
    min_lat, max_lat = track['lat'].min() - 0.01, track['lat'].max() + 0.01

    fg_network = folium.FeatureGroup(name='Road Network (Background)', show=False)
    # Sample subset of network arcs in bbox for responsiveness
    arc_count = 0
    for u, v in arcs:
        n1 = nodes[u]
        n2 = nodes[v]
        if (min_lon <= n1[0] <= max_lon and min_lat <= n1[1] <= max_lat) or \
           (min_lon <= n2[0] <= max_lon and min_lat <= n2[1] <= max_lat):
            folium.PolyLine(
                locations=[[n1[1], n1[0]], [n2[1], n2[0]]],
                color='#aaaaaa', weight=1.2, opacity=0.5
            ).add_to(fg_network)
            arc_count += 1
            if arc_count > 3000:
                break
    fg_network.add_to(m)

    # Ground Truth Route
    fg_route = folium.FeatureGroup(name='Ground Truth Route (Reference)', show=True)
    route_coords = []
    for a in route:
        u, v = arcs[a]
        route_coords.append([nodes[u][1], nodes[u][0]])
    route_coords.append([nodes[arcs[route[-1]][1]][1], nodes[arcs[route[-1]][1]][0]])
    folium.PolyLine(
        locations=route_coords,
        color='#28a745', weight=4.5, opacity=0.85,
        tooltip=f"Ground Truth Reference Route ({len(route)} arcs)"
    ).add_to(fg_route)
    fg_route.add_to(m)

    # Raw GPS points (subsampled every 5th point for crisp visualization)
    fg_gps = folium.FeatureGroup(name='Raw Noisy GPS Observations', show=True)
    for idx in range(0, len(track), 4):
        row = track.iloc[idx]
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=3.5,
            color='#0066cc',
            fill=True,
            fill_color='#3399ff',
            fill_opacity=0.7,
            weight=1,
            tooltip=f"Step {idx}: Time {row['time']}s"
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # Start & End Markers
    folium.Marker(
        location=[track.iloc[0]['lat'], track.iloc[0]['lon']],
        popup="Trip Origin",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    folium.Marker(
        location=[track.iloc[-1]['lat'], track.iloc[-1]['lon']],
        popup="Trip Destination",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    map_out = os.path.join('visualizations', 'maps', f'eda_trip_{trip_id}.html')
    m.save(map_out)
    print(f"Interactive Folium map for Trip {trip_id} saved to: {map_out}")

if __name__ == '__main__':
    run_eda()
