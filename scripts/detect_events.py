# scripts/detect_events.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SAMPLE_RATE    = 10_000
WINDOW_SIZE    = int(SAMPLE_RATE * 0.5)
STEP_SIZE      = int(SAMPLE_RATE * 0.05)
MEAN_THRESHOLD = 0.005
STD_THRESHOLD  = 0.002

if len(sys.argv) < 2:
    print("Usage: python scripts/detect_events.py <data/csv_file>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
t  = df["elapsed_s"].values
v  = df["voltage_V"].values

print(f"Loaded {len(df)} rows, {t[-1]:.1f}s of data")
print("Computing rolling statistics...")

series    = pd.Series(v)
roll_mean = series.rolling(window=WINDOW_SIZE, center=False).mean()
roll_std  = series.rolling(window=WINDOW_SIZE, center=False).std()

mean_deltas_full = roll_mean.diff(WINDOW_SIZE).values
std_deltas_full  = roll_std.diff(WINDOW_SIZE).values

indices     = np.arange(WINDOW_SIZE * 2, len(v), STEP_SIZE)
mean_times  = t[indices]
std_times   = t[indices]
mean_deltas = mean_deltas_full[indices]
std_deltas  = std_deltas_full[indices]

mask        = ~(np.isnan(mean_deltas) | np.isnan(std_deltas))
mean_times  = mean_times[mask]
std_times   = std_times[mask]
mean_deltas = mean_deltas[mask]
std_deltas  = std_deltas[mask]

print(f"Mean delta range: {mean_deltas.min()*1000:+.2f}mV to {mean_deltas.max()*1000:+.2f}mV")
print(f"Std  delta range: {std_deltas.min()*1000:+.2f}mV to {std_deltas.max()*1000:+.2f}mV")
print(f"Current thresholds: mean=±{MEAN_THRESHOLD*1000:.1f}mV  std=±{STD_THRESHOLD*1000:.1f}mV")
print("Finding events...")

def find_events(times, deltas, threshold):
    events = []
    i = 0
    while i < len(deltas):
        if abs(deltas[i]) > threshold:
            j = i
            while j < len(deltas) and abs(deltas[j]) > threshold:
                j += 1
            peak_idx = i + np.argmax(np.abs(deltas[i:j]))
            events.append({
                "time_s":    times[peak_idx],
                "delta":     deltas[peak_idx],
                "direction": "on" if deltas[peak_idx] < 0 else "off",
            })
            i = j
        else:
            i += 1
    return events

mean_events = find_events(mean_times, mean_deltas, MEAN_THRESHOLD)
std_events  = find_events(std_times,  std_deltas,  STD_THRESHOLD)

def classify(mean_evs, std_evs, tol=0.5):
    classified = []
    used_std = set()
    for me in mean_evs:
        match = None
        for si, se in enumerate(std_evs):
            if si not in used_std and abs(se["time_s"] - me["time_s"]) < tol:
                match = (si, se)
                break
        if match:
            used_std.add(match[0])
            classified.append({**me, "type": "mean+std", "color": "purple"})
        else:
            classified.append({**me, "type": "mean only", "color": "red"})
    for si, se in enumerate(std_evs):
        if si not in used_std:
            classified.append({**se, "type": "std only", "color": "orange"})
    classified.sort(key=lambda x: x["time_s"])
    return classified

all_events = classify(mean_events, std_events)

print(f"\nDetected {len(all_events)} events:")
for e in all_events:
    print(f"  t={e['time_s']:.3f}s  delta={e['delta']*1000:+.2f}mV  "
          f"type={e['type']:12s}  direction={e['direction']}")

# derive events filepath from data filepath
data_dir     = os.path.dirname(csv_path)
basename     = os.path.basename(csv_path)
events_file  = os.path.join(data_dir, basename.replace("light_", "events_"))
ground_truth = None
if os.path.exists(events_file):
    ground_truth = pd.read_csv(events_file)
    print(f"\nGround truth:")
    for _, row in ground_truth.iterrows():
        print(f"  t={row['elapsed_s']:.3f}s  {row['label']}")

fig, axes = plt.subplots(3, 1, figsize=(14, 11))
fig.suptitle(basename, fontsize=11)

def draw_ground_truth(ax, ground_truth, ymin, ymax):
    if ground_truth is None:
        return
    for _, row in ground_truth.iterrows():
        ax.axvline(row["elapsed_s"], color="black", lw=1, linestyle=":", alpha=0.5)
        ax.text(row["elapsed_s"], ymin, row["label"],
                rotation=90, fontsize=7, va="bottom", color="black", alpha=0.6)

ax1 = axes[0]
ax1.plot(t, v, lw=0.3, color="steelblue")
for e in all_events:
    ax1.axvline(e["time_s"], color=e["color"], lw=1.2, linestyle="--", alpha=0.9)
    ax1.text(e["time_s"], v.max(), f"{e['type']}\n{e['delta']*1000:+.1f}mV",
             rotation=90, fontsize=6.5, va="top", color=e["color"])
draw_ground_truth(ax1, ground_truth, v.min(), v.max())
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Classified events — red=mean only, orange=std only, purple=both")
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.plot(mean_times, mean_deltas * 1000, lw=0.6, color="red", alpha=0.8)
ax2.axhline( MEAN_THRESHOLD * 1000, color="red", lw=0.8, linestyle="--",
             alpha=0.5, label=f"±{MEAN_THRESHOLD*1000:.1f}mV threshold")
ax2.axhline(-MEAN_THRESHOLD * 1000, color="red", lw=0.8, linestyle="--", alpha=0.5)
for e in mean_events:
    ax2.axvline(e["time_s"], color="red", lw=0.8, linestyle="--", alpha=0.4)
draw_ground_truth(ax2, ground_truth, mean_deltas.min()*1000, mean_deltas.max()*1000)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Δ mean (mV)")
ax2.set_title("Mean channel")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
ax3.plot(std_times, std_deltas * 1000, lw=0.6, color="darkorange", alpha=0.8)
ax3.axhline( STD_THRESHOLD * 1000, color="darkorange", lw=0.8, linestyle="--",
             alpha=0.5, label=f"±{STD_THRESHOLD*1000:.1f}mV threshold")
ax3.axhline(-STD_THRESHOLD * 1000, color="darkorange", lw=0.8, linestyle="--", alpha=0.5)
for e in std_events:
    ax3.axvline(e["time_s"], color="darkorange", lw=0.8, linestyle="--", alpha=0.4)
draw_ground_truth(ax3, ground_truth, std_deltas.min()*1000, std_deltas.max()*1000)
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Δ std (mV)")
ax3.set_title("Std channel")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()