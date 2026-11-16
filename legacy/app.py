from flask import Flask, render_template, jsonify, request
import subprocess
import os
import signal
import sys

app = Flask(__name__)

training_process = None
tensorboard_process = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_process
    if training_process is None or training_process.poll() is not None:
        # Start training
        # Use the same python interpreter
        python_exe = sys.executable
        training_process = subprocess.Popen([python_exe, "train.py"], cwd=os.getcwd())
        return jsonify({"status": "started", "pid": training_process.pid})
    else:
        return jsonify({"status": "already_running", "pid": training_process.pid})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global training_process
    if training_process and training_process.poll() is None:
        training_process.terminate()
        training_process = None
        return jsonify({"status": "stopped"})
    else:
        return jsonify({"status": "not_running"})

@app.route('/status')
def status():
    global training_process
    is_running = training_process is not None and training_process.poll() is None
    return jsonify({"running": is_running})

def start_tensorboard():
    global tensorboard_process
    # Start TensorBoard on port 6006
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", "logs", "--host", "0.0.0.0", "--port", "6006"])

if __name__ == '__main__':
    try:
        start_tensorboard()
        app.run(debug=True, port=5000)
    finally:
        if tensorboard_process:
            tensorboard_process.terminate()
        if training_process:
            training_process.terminate()
