# ---------- client_monitoring.py ----------

import json
from datetime import datetime
from collections import defaultdict
import random

def normalize_probabilities(W):
    total = sum(W.values())
    if total == 0:
        return {k: 1 / len(W) for k in W}
    return {k: v / total for k, v in W.items()}

def weighted_random_selection(W, k):
    return random.choices(list(W.keys()), weights=W.values(), k=k)

def get_available_clients(clients, A, J, r, T_min):
    available = []
    for c in clients:
        if r < len(A[c]) and A[c][r] and r < len(J[c]):
            join_time, leave_time = J[c][r]
            duration = (leave_time - join_time).total_seconds()
            if duration >= T_min:
                available.append(c)
    return available

def update_histories(A, F, J, log_data):
    for entry in log_data.get("joins", []):
        cid = entry["client_id"]
        round_idx = entry["round"]
        timestamp = datetime.fromisoformat(entry["timestamp"])
        if cid not in A:
            A[cid] = []
        while len(A[cid]) <= round_idx:
            A[cid].append(False)
        A[cid][round_idx] = True
        #==================== can you explain what does the program do here =============
        if cid not in J:
            J[cid] = []
        while len(J[cid]) <= round_idx:
            J[cid].append((timestamp, timestamp))
        J[cid][round_idx] = (timestamp, timestamp)

    for entry in log_data.get("leaves", []):
        cid = entry["client_id"]
        round_idx = entry["round"]
        timestamp = datetime.fromisoformat(entry["timestamp"])
        if cid in J and round_idx < len(J[cid]):
            join_time, _ = J[cid][round_idx]
            J[cid][round_idx] = (join_time, timestamp)

    for entry in log_data.get("crashes", []):
        cid = entry["client_id"]
        round_idx = entry["round"]
        if cid not in F:
            F[cid] = []
        F[cid].append(round_idx)

def load_log_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def CRACS_MDA(C, A, F, J, r, n, m, T_min, Clusters, log_data):
    update_histories(A, F, J, log_data)
    S = []

    for cluster_id, clients_in_cluster in Clusters.items():
        Candidates = get_available_clients(clients_in_cluster, A, J, r, T_min)
        W = {}

        for c in Candidates:
            if len(A[c]) >= m:
                total_time = 0
                available_time = 0
                for i in range(len(A[c]) - m, len(A[c])):
                    if A[c][i - 1] and A[c][i]:
                        t1, _ = J[c][i - 1]
                        _, t2 = J[c][i]
                        elapsed = (t2 - t1).total_seconds()
                        total_time += elapsed
                        available_time += elapsed
                    else:
                        t1, _ = J[c][i - 1]
                        _, t2 = J[c][i]
                        total_time += (t2 - t1).total_seconds()
                availability_score = available_time / total_time if total_time > 0 else 0.5
            else:
                availability_score = 0.5

            pen = 0
            max_pen = 0
            for i in range(0, r):
                p = 1 / (r - i + 1)
                max_pen += p
                if i in F[c]:
                    pen += p
            penalty_score = (1 - (pen / max_pen)) if max_pen > 0 else 1

            W[c] = availability_score * penalty_score

        W = normalize_probabilities(W)
        k = max(1, int(n / len(Clusters)))
        S += weighted_random_selection(W, k)

    return S
