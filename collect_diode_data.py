import time
import csv
from datetime import datetime
import nidaqmx
from nidaqmx.constants import TerminalConfiguration

CHANNEL = "myDAQ1/ai0"
SAMPLE_INTERVAL = 0.1  # seconds
OUTPUT_FILE = f"light_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=0.0,
        max_val=5.0,
    )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "elapsed_s", "voltage_V"])

        print(f"Saving to {OUTPUT_FILE} — press Ctrl+C to stop.")
        start = time.time()

        try:
            while True:
                value = task.read()
                elapsed = time.time() - start
                timestamp = datetime.now().isoformat()
                writer.writerow([timestamp, f"{elapsed:.3f}", f"{value:.4f}"])
                f.flush()
                print(f"{timestamp}  {elapsed:.1f}s  {value:.3f} V")
                time.sleep(SAMPLE_INTERVAL)
        except KeyboardInterrupt:
            print(f"\nDone. Data saved to {OUTPUT_FILE}")