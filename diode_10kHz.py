# collect.py
import csv
from datetime import datetime
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

CHANNEL = "myDAQ1/ai0"
SAMPLE_RATE = 10_000      # 10 kHz
BUFFER_SIZE = 1000        # read 1000 samples at a time (0.1s chunks)
OUTPUT_FILE = f"light_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan(
        CHANNEL,
        terminal_config=TerminalConfiguration.DIFF,
        min_val=0.0,
        max_val=5.0,
    )

    task.timing.cfg_samp_clk_timing(
        rate=SAMPLE_RATE,
        sample_mode=AcquisitionType.CONTINUOUS,
        samps_per_chan=BUFFER_SIZE * 10,
    )

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "elapsed_s", "voltage_V"])

        print(f"Sampling at {SAMPLE_RATE} Hz — saving to {OUTPUT_FILE}")
        print("Press Ctrl+C to stop.")

        sample_index = 0
        try:
            while True:
                samples = task.read(number_of_samples_per_channel=BUFFER_SIZE)
                rows = [
                    [sample_index + i, f"{(sample_index + i) / SAMPLE_RATE:.6f}", f"{v:.5f}"]
                    for i, v in enumerate(samples)
                ]
                writer.writerows(rows)
                sample_index += len(samples)
                f.flush()
                print(f"{sample_index} samples ({sample_index / SAMPLE_RATE:.1f}s)")
        except KeyboardInterrupt:
            print(f"\nDone. {sample_index} samples saved to {OUTPUT_FILE}")