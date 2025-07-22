from flask import Flask, request, jsonify
import os, json
from datetime import datetime
from collections import defaultdict

app = Flask(__name__)
logs_by_round = defaultdict(lambda: {"joins": [], "leaves": [], "crashes": []})

@app.route('/heartbeat/join', methods=['POST'])
def join():
    data = request.json
    logs_by_round[data["round"]]["joins"].append(data)
    return jsonify({"status": "join recorded"})

@app.route('/heartbeat/leave', methods=['POST'])
def leave():
    data = request.json
    logs_by_round[data["round"]]["leaves"].append(data)
    return jsonify({"status": "leave recorded"})

@app.route('/heartbeat/crash', methods=['POST'])
def crash():
    data = request.json
    logs_by_round[data["round"]]["crashes"].append(data)
    return jsonify({"status": "crash recorded"})


@app.route('/save_logs', methods=['POST'])
def save_logs():
    for round_num, log_data in logs_by_round.items():
        filename = f"client_logs_round_{round_num}.json"
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2, default=str)
    return jsonify({"status": "all logs saved"})


if __name__ == "__main__":
    print("[Flask] Monitoring server running on port 5000...")
    app.run(host="0.0.0.0", port=5000)


"""
# Inputs:
# C: Set of all clients
# A: Availability history {client_id: [True/False, ...]}
# F: Failure history {client_id: [round indices]}
# J: Participation session metadata {client_id: [(join_time, leave_time), ...]}
# r: Current round index
# n: Number of clients to select in this round
# m: Memory length (how many past rounds to consider)
# T_min: Minimum required online duration per round (e.g., 60 seconds)
# Clusters: Dictionary {cluster_id: [client_id1, client_id2, ...]}
# S: Final selected client set

def CRACS_MDA(C, A, F, J, r, n, m, T_min, Clusters):
    update_histories(A, F, J)  # Updates availability and failure records
    S = []

    # Iterate through each domain-specific cluster
    for cluster_id, clients_in_cluster in Clusters.items():
        Candidates = get_available_clients(clients_in_cluster, A, J, r, T_min)
        W = {}

        for c in Candidates:
            # --- Availability Weight ---
            if len(A[c]) >= m:
                total_time = 0
                available_time = 0
                for i in range(len(A[c]) - m, len(A[c])):
                    if A[c][i-1] and A[c][i]:
                        t1, _ = J[c][i-1]
                        _, t2 = J[c][i]
                        elapsed = (t2 - t1).total_seconds()
                        total_time += elapsed
                        available_time += elapsed
                    else:
                        # Client not consistently available
                        t1, _ = J[c][i-1]
                        _, t2 = J[c][i]
                        total_time += (t2 - t1).total_seconds()
                availability_score = available_time / total_time if total_time > 0 else 0.5
            else:
                availability_score = 0.5  # Default weight if history too short

            # --- Failure Penalty ---
            pen = 0
            max_pen = 0
            for i in range(0, r):
                p = 1 / (r - i + 1)
                max_pen += p
                if i in F[c]:
                    pen += p
            penalty_score = (1 - (pen / max_pen)) if max_pen > 0 else 1

            # Combine scores
            W[c] = availability_score * penalty_score

        # Normalize and select per cluster
        W = normalize_probabilities(W)
        k = max(1, int(n / len(Clusters)))  # Number of clients per cluster
        S += weighted_random_selection(W, k)

    return S
"""
