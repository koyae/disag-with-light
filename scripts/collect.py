# scripts/collect.py
import argparse
import csv, threading, os
from datetime import datetime
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

parser = argparse.ArgumentParser(description="Collect light sensor data with event logging.")
parser.add_argument("sample_rate", nargs='?', type=int, default=10_000, help="Sampling rate in Hz")
parser.add_argument("--data-dir", default="data", help="Directory to save data files")
parser.add_argument("--visualize-args", nargs=argparse.REMAINDER, default=[], help="Additional arguments to pass to the visualization script")

args = parser.parse_args()
CHANNEL     = "myDAQ1/ai0"
SAMPLE_RATE = args.sample_rate
BUFFER_SIZE = 1000
DATA_DIR    = "data"

os.makedirs(DATA_DIR, exist_ok=True)

timestamp   = datetime.now().strftime('%Y%m%d_%H%M%S')
DATA_FILE   = os.path.join(DATA_DIR, f"light_{timestamp}.csv")
EVENTS_FILE = os.path.join(DATA_DIR, f"events_{timestamp}.csv")

sample_index = 0

def listen_for_events(events_writer, events_file):
    print("Press Enter to log an event, then type a label.")
    while True:
        input()
        t = sample_index / SAMPLE_RATE
        label = input("Label (e.g. 'microwave_on'): ").strip()
        events_writer.writerow([sample_index, f"{t:.4f}", label])
        events_file.flush()
        print(f"  Logged '{label}' at {t:.4f}s")

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


    try:
        with open(DATA_FILE, "w", newline="") as df, \
            open(EVENTS_FILE, "w", newline="") as ef:

            writer        = csv.writer(df)
            events_writer = csv.writer(ef)
            writer.writerow(["sample_index", "elapsed_s", "voltage_V"])
            events_writer.writerow(["sample_index", "elapsed_s", "label"])

            t = threading.Thread(target=listen_for_events, args=(events_writer, ef), daemon=True)
            t.start()

            print(f"Recording to {DATA_FILE} — Ctrl+C to stop.")

            while True:
                samples = task.read(number_of_samples_per_channel=BUFFER_SIZE)
                rows = [
                    [sample_index + i, f"{(sample_index + i)/SAMPLE_RATE:.6f}", f"{v:.5f}"]
                    for i, v in enumerate(samples)
                ]
                writer.writerows(rows)
                sample_index += len(samples)
                df.flush()
    except KeyboardInterrupt:
        import os
        df.close()
        print(os.getcwd())
        import subprocess
        subprocess.run(
            ['uv', 'run', 'python', 'scripts/visualize.py', DATA_FILE, '0', '.1'] + (args.visualize_args)
        )
        print(f"\nDone. Data saved to {DATA_FILE}")

