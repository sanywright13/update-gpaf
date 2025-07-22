from flask import Flask, request, jsonify
from pyngrok import ngrok
import threading

app = Flask(__name__)

@app.route("/heartbeat/join", methods=["POST"])
def join():
    data = request.json
    print(f"[JOIN] {data}")
    return jsonify({"status": "received join"})

@app.route("/heartbeat/leave", methods=["POST"])
def leave():
    data = request.json
    print(f"[LEAVE] {data}")
    return jsonify({"status": "received leave"})

@app.route("/heartbeat/ping", methods=["POST"])
def ping():
    data = request.json
    print(f"[PING] {data}")
    return jsonify({"status": "received ping"})

@app.route("/heartbeat/crash", methods=["POST"])
def crash():
    data = request.json
    print(f"[CRASH] {data}")
    return jsonify({"status": "received crash"})

# Start the Flask app with ngrok tunnel
public_url = ngrok.connect(5000)
print("🔥 Public ngrok URL:", public_url)

app.run(port=5000)
