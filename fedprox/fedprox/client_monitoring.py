# ---------- client_monitoring.py ----------

import json
from datetime import datetime
from collections import defaultdict
import random


DEFAULT_TMAX = 600.0  # seconds
EMA_ALPHA = 0.3
T_hat = defaultdict(lambda: DEFAULT_TMAX)  # per-client smoothed estimate

def update_adaptive_tmax(client_id, J):
    latest = len(J[client_id]) - 1
    join_time, leave_time = J[client_id][latest]
    duration = (leave_time - join_time).total_seconds()
    prev = T_hat[client_id]
    T_hat[client_id] = EMA_ALPHA * duration + (1 - EMA_ALPHA) * prev

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
        r = entry["round"]
        ts = datetime.fromisoformat(entry["timestamp"])
        A.setdefault(cid, []).extend([False] * (r + 1 - len(A[cid])))
        A[cid][r] = True
        J.setdefault(cid, []).extend([(ts, ts)] * (r + 1 - len(J[cid])))
        J[cid][r] = (ts, ts)

    for entry in log_data.get("leaves", []):
        cid = entry["client_id"]
        r = entry["round"]
        ts = datetime.fromisoformat(entry["timestamp"])
        if cid in J and r < len(J[cid]):
            join, _ = J[cid][r]
            J[cid][r] = (join, ts)

    for entry in log_data.get("crashes", []):
        cid = entry["client_id"]
        r = entry["round"]
        F.setdefault(cid, []).append(r)

    for cid in J:
        if J[cid]:
            update_adaptive_tmax(cid, J)
            

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
