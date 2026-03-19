import PyDAQmx
import numpy as np
import pandas as pd
import time
from datetime import datetime

SAMPLE_RATE = 1000
NUM_SAMPLES = 500
VCC = 5.0
R_FIXED = 10_000

def voltage_to_resistance(v_out):
    if v_out <= 0:
        return float('inf')
    return R_FIXED * (VCC - v_out) / v_out

def resistance_to_lux(r_ldr, gamma=0.7, r10=10_000):
    if r_ldr <= 0:
        return float('inf')
    return 10 ** ((np.log10(r10) - np.log10(r_ldr)) / gamma)

# --- Setup ---
task = PyDAQmx.Task()

task.CreateAIVoltageChan(
    "myDAQ1/ai0", "",
    PyDAQmx.DAQmx_Val_Diff,
    0.0, 5.0,
    PyDAQmx.DAQmx_Val_Volts, None
)

task.CfgSampClkTiming(
    "", SAMPLE_RATE,
    PyDAQmx.DAQmx_Val_Rising,
    PyDAQmx.DAQmx_Val_ContSamps,
    NUM_SAMPLES
)

task.StartTask()

records = [] 

print("Readibg sensor... Press Ctrl+C to stop and save.")

try:
    while True:
        buffer = np.zeros(NUM_SAMPLES, dtype=np.float64)
        read = PyDAQmx.int32()

        task.ReadAnalogF64(
            NUM_SAMPLES, 10.0,
            PyDAQmx.DAQmx_Val_GroupByChannel,
            buffer, NUM_SAMPLES,
            PyDAQmx.byref(read), None
        )

        timestamp = datetime.now()
        mean_v = buffer.mean()
        min_v  = buffer.min()
        max_v  = buffer.max()
        std_v  = buffer.std()
        r_ldr  = voltage_to_resistance(mean_v)
        lux    = resistance_to_lux(r_ldr)

        records.append({
            "timestamp":       timestamp,
            "mean_voltage_V":  round(mean_v, 6),
            "min_voltage_V":   round(min_v,  6),
            "max_voltage_V":   round(max_v,  6),
            "std_voltage_V":   round(std_v,  6),
            "ldr_resistance_ohm": round(r_ldr, 2),
            "approx_lux":      round(lux, 2),
        })

        print(
            f"[{timestamp.strftime('%H:%M:%S')}] "
            f"Mean: {mean_v:.4f} V | "
            f"R_LDR: {r_ldr:.0f} Ω | "
            f"~{lux:.1f} lux"
        )

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping acquisition...")

finally:
    task.StopTask()
    task.ClearTask()

    if records:
        df = pd.DataFrame(records)
        df.set_index("timestamp", inplace=True)

        filename = f"photoresistor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename)

        print(f"\nSaved {len(df)} records to '{filename}'")
        print(df.describe())
    else:
        print("No data recorded.")