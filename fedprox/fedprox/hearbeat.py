from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)
heartbeat_log = {}
crash_log = {}

@app.route("/heartbeat/ping", methods=["POST"])
def receive_ping():
    data = request.get_json()
    cid = data["client_id"]
    round_ = data["round"]
    timestamp = data["timestamp"]

    heartbeat_log.setdefault(cid, []).append((round_, timestamp))
    print(f"[Ping] Client {cid} Round {round_} @ {timestamp}")
    return jsonify({"status": "received"}), 200

@app.route("/heartbeat/join", methods=["POST"])
def join():
    data = request.get_json()
    cid = data["client_id"]
    round_ = data["round"]
    timestamp = data["timestamp"]
    print(f"[Join] Client {cid} started round {round_} at {timestamp}")
    return jsonify({"status": "join recorded"}), 200

@app.route("/heartbeat/leave", methods=["POST"])
def leave():
    data = request.get_json()
    cid = data["client_id"]
    round_ = data["round"]
    timestamp = data["timestamp"]
    print(f"[Leave] Client {cid} finished round {round_} at {timestamp}")
    return jsonify({"status": "leave recorded"}), 200

@app.route("/heartbeat/crash", methods=["POST"])
def crash():
    data = request.get_json()
    cid = data["client_id"]
    round_ = data["round"]
    timestamp = data["timestamp"]
    error = data.get("error", "Unknown")
    trace = data.get("trace", "")
    crash_log.setdefault(cid, []).append((round_, timestamp, error))
    print(f"[Crash] Client {cid} crashed in round {round_} @ {timestamp}\n{error}\n{trace}")
    return jsonify({"status": "crash recorded"}), 200

@app.route("/heartbeat/summary", methods=["GET"])
def summary():
    return jsonify({
        "active_clients": list(heartbeat_log.keys()),
        "heartbeat_log": heartbeat_log,
        "crashes": crash_log
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
