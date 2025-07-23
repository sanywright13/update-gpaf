from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading
from collections import defaultdict
import os, json
app = Flask(__name__)
logs_by_round = defaultdict(lambda: {"joins": [], "leaves": [], "crashes": []})

@app.route("/heartbeat/join", methods=["POST"])
def join():
    data = request.json
    logs_by_round[data["round"]]["joins"].append(data)
    print(f"[JOIN] {data}")
    return jsonify({"status": "received join"})

@app.route("/heartbeat/leave", methods=["POST"])
def leave():
    data = request.json

    print(f"[LEAVE] {data}")
    logs_by_round[data["round"]]["leaves"].append(data)

    return jsonify({"status": "received leave"})

@app.route("/heartbeat/ping", methods=["POST"])
def ping():
    data = request.json
    print(f"[PING] {data}")
    logs_by_round[data["round"]]["ping"].append(data)
    return jsonify({"status": "received ping"})

@app.route("/heartbeat/crash", methods=["POST"])
def crash():
    data = request.json
    print(f"[CRASH] {data}")
    logs_by_round[data["round"]]["crashes"].append(data)
    return jsonify({"status": "received crash"})

@app.route('/heartbeat/save_logs', methods=['POST'])
def save_logs():
    print("[Flask] save_logs called")
    if not logs_by_round:
        print("[Flask] logs_by_round is empty!")
        return jsonify({"status": "no logs to save"})

    for round_num, log_data in logs_by_round.items():
        filename = f"client_logs_round_{round_num}.json"
        print(f"[Flask] Saving logs for round {round_num} to {filename}")
        with open(filename, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

    return jsonify({"status": "all logs saved"})
# Start the Flask app with ngrok tunnel
public_url = ngrok.connect(5000)
print("🔥 Public ngrok URL:", public_url)

app.run(port=5000)
