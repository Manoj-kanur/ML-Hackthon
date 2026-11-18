"""
Standalone Python Evaluation Client for Laptop 2
Can be run on Laptop 2 from terminal to query Laptop 1's API and display results.
Usage:
  python ui/client_app.py --host http://192.168.1.15:8000 --trip 00000000
"""

import argparse
import requests
import json
import time

def main():
    parser = argparse.ArgumentParser(description="PathSnap Laptop 2 Evaluator Client")
    parser.add_argument("--host", default="http://localhost:8000", help="URL of Laptop 1 API")
    parser.add_argument("--trip", default="00000000", help="Trip ID to evaluate (e.g. 00000000)")
    args = parser.parse_args()

    api_url = f"{args.host.rstrip('/')}/api/evaluate"
    print("=" * 60)
    print(f" [PathSnap Laptop 2 Client] Connecting to API: {api_url}")
    print(f" [PathSnap Laptop 2 Client] Evaluating Trip:   {args.trip}")
    print("=" * 60)

    t0 = time.time()
    try:
        response = requests.post(api_url, json={"trip_id": args.trip}, timeout=15)
        response.raise_for_status()
        data = response.json()
        round_trip_ms = round((time.time() - t0) * 1000.0, 1)

        print("\n [SUCCESS] Evaluation Results Received:")
        print(f"  • Trip ID:                 {data['trip_id']}")
        print(f"  • Total GPS Observations:  {data['total_points']}")
        print(f"  • Top-1 Accuracy:          {data['metrics']['top1_accuracy']}% (Classifier baseline: {data['metrics']['classifier_accuracy']}%)")
        print(f"  • Route Continuity:        {data['metrics']['route_continuity']}% (Classifier baseline: {data['metrics']['classifier_continuity']}%)")
        print(f"  • Candidate Miss Rate:     {data['metrics']['candidate_miss_rate']}%")
        print(f"  • Server Processing Time:  {data['total_time_ms']} ms")
        print(f"  • Total Network Roundtrip: {round_trip_ms} ms")
        print(f"  • Predicted Route Points:  {len(data['map_data']['predicted_route'])}")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f" [ERROR] Could not connect to Laptop 1 API: {e}")

if __name__ == "__main__":
    main()
