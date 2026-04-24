# scripts/auto_collect.py (Linux)
import csv
import time
import threading
import json
import subprocess
import os
import socket
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# --- Home Assistant config ---
HA_URL   = "http://169.254.204.2:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YmZmNjJkNDA3ZDg0OGMzYWNlOWQwZDkwNDlmYmU3OCIsImlhdCI6MTc3NjgwMjkxNiwiZXhwIjoyMDkyMTYyOTE2fQ.deCyR50AtJKF2n1_rt_Dx0mcx4wtYoe5d5wLO_neTxM"

# --- DAQ server config ---
DAQ_HOST    = "169.254.204.78"
DAQ_PORT    = 9999
SAMPLE_RATE = 10_000
BUFFER_SIZE = 1000

# --- Appliance config ---
APPLIANCES = {
    "kettle":       ("switch.lumi_lumi_plug_maus01", 5,  5,  10),
    "fridge":       ("switch.maus02_lumi_lumi_plug",  30, 30, 10),
    "space_heater": ("switch.maus03_lumi_lumi_plug",  5,  5,  10),
    "computer":     ("switch.maus04_lumi_lumi_plug",  5,  5,  10),
}

CYCLE_MODE = "round_robin"

# shared state
sample_index    = 0
events_writer   = None
events_file     = None
collection_done = False

# ---------------------------------------------------------------
# Home Assistant control
# ---------------------------------------------------------------

def ha_call(method, endpoint, data=None):
    cmd = [
        "curl", "-s",
        "-X", method,
        "-H", f"Authorization: Bearer {HA_TOKEN}",
        "-H", "Content-Type: application/json",
    ]
    if data:
        cmd += ["-d", json.dumps(data)]
    cmd.append(f"{HA_URL}{endpoint}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.stdout:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"  WARNING: ha_call failed — {e}")
    return None

def plug_on(entity_id):
    ha_call("POST", "/api/services/switch/turn_on", {"entity_id": entity_id})

def plug_off(entity_id):
    ha_call("POST", "/api/services/switch/turn_off", {"entity_id": entity_id})

# ---------------------------------------------------------------
# DAQ client
# ---------------------------------------------------------------

class DAQClient:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        print(f"Connected to DAQ server at {host}:{port}")

    def read(self, n_samples):
        raw_len = self._recv_exactly(4)
        n_bytes = int.from_bytes(raw_len, "big")
        raw     = self._recv_exactly(n_bytes)
        return np.frombuffer(raw, dtype=np.float32)

    def _recv_exactly(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("DAQ server disconnected")
            buf += chunk
        return buf

    def close(self):
        self.sock.close()

# ---------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------

def log_event(label):
    t = sample_index / SAMPLE_RATE
    events_writer.writerow([sample_index, f"{t:.4f}", label])
    events_file.flush()
    print(f"  Logged '{label}' at {t:.4f}s")

# ---------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------

def run_all():
    global collection_done

    time.sleep(3)
    appliance_list = list(APPLIANCES.items())
    total = sum(c * (on + off) for _, (_, on, off, c) in appliance_list) / 60
    print(f"Starting automated collection in '{CYCLE_MODE}' mode...")
    print(f"Total estimated time: {total:.1f} minutes")

    for _, (entity_id, _, _, _) in appliance_list:
        plug_off(entity_id)
    time.sleep(2)

    if CYCLE_MODE == "sequential":
        for appliance_name, (entity_id, on_duration, off_duration, cycles) in appliance_list:
            print(f"\n{'='*50}")
            print(f"Starting {appliance_name} — {cycles} cycles")
            print(f"  ON: {on_duration}s  OFF: {off_duration}s")
            print(f"{'='*50}")

            for cycle in range(cycles):
                print(f"  [{appliance_name}] Cycle {cycle + 1}/{cycles}")
                plug_on(entity_id)
                time.sleep(0.3)
                log_event(f"{appliance_name}_on")
                time.sleep(on_duration)
                plug_off(entity_id)
                time.sleep(0.3)
                log_event(f"{appliance_name}_off")
                time.sleep(off_duration)

            print(f"  Finished {appliance_name}")

    elif CYCLE_MODE == "round_robin":
        max_cycles = max(cycles for _, (_, _, _, cycles) in appliance_list)
        print(f"Running {max_cycles} rounds across {len(appliance_list)} appliances")

        for cycle in range(max_cycles):
            print(f"\n--- Round {cycle + 1}/{max_cycles} ---")
            for appliance_name, (entity_id, on_duration, off_duration, cycles) in appliance_list:
                if cycle >= cycles:
                    print(f"  [{appliance_name}] skipping — done")
                    continue

                print(f"  [{appliance_name}] cycle {cycle + 1}/{cycles}")
                plug_on(entity_id)
                time.sleep(0.3)
                log_event(f"{appliance_name}_on")
                time.sleep(on_duration)
                plug_off(entity_id)
                time.sleep(0.3)
                log_event(f"{appliance_name}_off")
                time.sleep(off_duration)

    print("\nTurning all plugs off...")
    for _, (entity_id, _, _, _) in appliance_list:
        plug_off(entity_id)

    print("\nAll done! All plugs off.")
    collection_done = True

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

lightbulb_type = input("What lightbulb are you using? ").strip()
timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
DATA_FILE      = os.path.join(DATA_DIR, f"light_{timestamp}_{lightbulb_type}.csv")
EVENTS_FILE    = os.path.join(DATA_DIR, f"events_{timestamp}_{lightbulb_type}.csv")

print(f"Data file: {DATA_FILE}")
print(f"Events file: {EVENTS_FILE}")
print("Starting in 3 seconds — Ctrl+C to abort...")
time.sleep(3)

daq = DAQClient(DAQ_HOST, DAQ_PORT)

with open(DATA_FILE, "w", newline="") as df, \
     open(EVENTS_FILE, "w", newline="") as ef:

    writer        = csv.writer(df)
    events_writer = csv.writer(ef)
    writer.writerow(["sample_index", "elapsed_s", "voltage_V"])
    events_writer.writerow(["sample_index", "elapsed_s", "label"])
    events_file = ef

    experiment_thread = threading.Thread(target=run_all, daemon=True)
    experiment_thread.start()

    try:
        while not collection_done:
            samples = daq.read(BUFFER_SIZE)
            rows = [
                [sample_index + i,
                 f"{(sample_index + i)/SAMPLE_RATE:.6f}",
                 f"{float(v):.5f}"]
                for i, v in enumerate(samples)
            ]
            writer.writerows(rows)
            sample_index += len(samples)
            df.flush()

    except KeyboardInterrupt:
        print(f"\nStopped early. Turning all plugs off...")
        for _, (entity_id, _, _, _) in APPLIANCES.items():
            plug_off(entity_id)
    finally:
        daq.close()

    print(f"Data saved to {DATA_FILE}")
    print(f"Events saved to {EVENTS_FILE}")
    print("Launching visualizer...")
    subprocess.Popen([
        "uv", "run", "python",
        os.path.join(SCRIPT_DIR, "visualize.py"),
        DATA_FILE
    ])