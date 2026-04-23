# scripts/auto_collect.py
import csv
import time
import threading
import requests
from datetime import datetime
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import os
import subprocess

# --- Home Assistant config ---
HA_URL   = "http://homeassistant.local:8123"
HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YmZmNjJkNDA3ZDg0OGMzYWNlOWQwZDkwNDlmYmU3OCIsImlhdCI6MTc3NjgwMjkxNiwiZXhwIjoyMDkyMTYyOTE2fQ.deCyR50AtJKF2n1_rt_Dx0mcx4wtYoe5d5wLO_neTxM"

HA_HEADERS = {
    "Authorization": f"Bearer {HA_TOKEN}",
    "Content-Type": "application/json",
}

# --- Appliance config ---
# Each appliance: (entity_id, on_duration_s, off_duration_s, cycles)
APPLIANCES = {
    "kettle":       ("switch.lumi_lumi_plug_maus01", 5,  5, 10),
    "fridge":       ("switch.maus02_lumi_lumi_plug",  30, 30, 10),
    "space_heater": ("switch.maus03_lumi_lumi_plug",  5,  5, 10),
    "computer":     ("switch.maus04_lumi_lumi_plug",  5,  5, 10),
}

# --- DAQ config ---
CHANNEL     = "myDAQ1/ai0"
SAMPLE_RATE = 10_000
BUFFER_SIZE = 1000
DATA_DIR    = "data"

os.makedirs(DATA_DIR, exist_ok=True)

# --- Shared state ---
sample_index    = 0
events_writer   = None
events_file     = None
collection_done = False

def plug_on(entity_id):
    r = requests.post(
        f"{HA_URL}/api/services/switch/turn_on",
        headers=HA_HEADERS,
        json={"entity_id": entity_id},
    )
    return r.status_code

def plug_off(entity_id):
    r = requests.post(
        f"{HA_URL}/api/services/switch/turn_off",
        headers=HA_HEADERS,
        json={"entity_id": entity_id},
    )
    return r.status_code

def log_event(label):
    t = sample_index / SAMPLE_RATE
    events_writer.writerow([sample_index, f"{t:.4f}", label])
    events_file.flush()
    print(f"  Logged '{label}' at {t:.4f}s")

CYCLE_MODE = "round_robin"  # "sequential" or "round_robin"

def run_all():
    global collection_done

    time.sleep(3)
    appliance_list = list(APPLIANCES.items())
    total = sum(c * (on + off) for _, (_, on, off, c) in appliance_list) / 60
    print(f"Starting automated collection in '{CYCLE_MODE}' mode...")
    print(f"Total estimated time: {total:.1f} minutes")

    # ensure all plugs start off
    for _, (entity_id, _, _, _) in appliance_list:
        plug_off(entity_id)
    time.sleep(2)

    if CYCLE_MODE == "sequential":
        for appliance_name, (entity_id, on_duration, off_duration, cycles) in appliance_list:
            print(f"\n{'='*50}")
            print(f"Starting {appliance_name} — {cycles} cycles")
            print(f"  ON: {on_duration}s  OFF: {off_duration}s")
            print(f"  Estimated time: {cycles * (on_duration + off_duration) / 60:.1f} minutes")
            print(f"{'='*50}")

            for cycle in range(cycles):
                print(f"  [{appliance_name}] Cycle {cycle + 1}/{cycles}")

                status = plug_on(entity_id)
                if status != 200:
                    print(f"  WARNING: turn_on returned {status}")
                time.sleep(0.3)
                log_event(f"{appliance_name}_on")
                time.sleep(on_duration)

                status = plug_off(entity_id)
                if status != 200:
                    print(f"  WARNING: turn_off returned {status}")
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

                status = plug_on(entity_id)
                if status != 200:
                    print(f"  WARNING: turn_on returned {status}")
                time.sleep(0.3)
                log_event(f"{appliance_name}_on")
                time.sleep(on_duration)

                status = plug_off(entity_id)
                if status != 200:
                    print(f"  WARNING: turn_off returned {status}")
                time.sleep(0.3)
                log_event(f"{appliance_name}_off")
                time.sleep(off_duration)

    print("\nTurning all plugs off...")
    for _, (entity_id, _, _, _) in appliance_list:
        plug_off(entity_id)

    print("\nAll done! All plugs off.")
    collection_done = True

# --- Main ---
timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
lightbulb_type = raw("What lightbulb are you using? ")
DATA_FILE   = os.path.join(DATA_DIR, f"light_{timestamp}_{lightbulb_type}.csv")
EVENTS_FILE = os.path.join(DATA_DIR, f"events_{timestamp}_{lightbulb_type}.csv")

total = sum(c * (on + off) for _, on, off, c in APPLIANCES.values()) / 60
print(f"Estimated total run time: {total:.1f} minutes")
print(f"Data file: {DATA_FILE}")
print(f"Events file: {EVENTS_FILE}")
print("Starting in 3 seconds — Ctrl+C to abort...")
time.sleep(3)

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=0.0, max_val=5.0,
    )
    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=BUFFER_SIZE * 10,
    )

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
                samples = task.read(
                    number_of_samples_per_channel=BUFFER_SIZE
                )
                rows = [
                    [sample_index + i,
                     f"{(sample_index + i)/SAMPLE_RATE:.6f}",
                     f"{v:.5f}"]
                    for i, v in enumerate(samples)
                ]
                writer.writerows(rows)
                sample_index += len(samples)
                df.flush()
        except KeyboardInterrupt:
            print(f"\nStopped early. Turning all plugs off...")
            for entity_id, _, _, _ in APPLIANCES.values():
                plug_off(entity_id)

        print(f"Data saved to {DATA_FILE}")
        print(f"Events saved to {EVENTS_FILE}")
        print("Launching visualizer...")
        subprocess.Popen([
            "uv", "run", "python", "scripts/visualize.py", DATA_FILE
        ])