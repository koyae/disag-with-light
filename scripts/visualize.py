# scripts/visualize.py
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python scripts/visualize.py <data/csv_file> [zoom_start_s] [zoom_end_s]")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
t = df["elapsed_s"].values
v = df["voltage_V"].values
sample_rate = 1 / (t[1] - t[0])

zoom_start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
zoom_end   = float(sys.argv[3]) if len(sys.argv) > 3 else zoom_start + 0.1

# derive events filepath from data filepath
data_dir    = os.path.dirname(csv_path)
basename    = os.path.basename(csv_path)
events_file = os.path.join(data_dir, basename.replace("light_", "events_"))

events = None
if os.path.exists(events_file):
    events = pd.read_csv(events_file)
    print(f"Loaded {len(events)} events from {events_file}")
else:
    print("No events file found.")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(basename, fontsize=11)

def draw_events(ax, events, ymin, ymax):
    if events is None:
        return
    for _, row in events.iterrows():
        #color = "green" if row["label"].endswith("_on") else "red"
        ax.axvline(row["elapsed_s"], color="green", lw=1, linestyle="--", alpha=0.8)
        ax.text(row["elapsed_s"], ymax, row["label"],
                rotation=90, fontsize=7, va="top", color="red", alpha=0.8)

ax1 = axes[0]
ax1.plot(t, v, lw=0.4, color="steelblue")
ax1.axvspan(zoom_start, zoom_end, color="orange", alpha=0.25, label="zoom region")
draw_events(ax1, events, v.min(), v.max())
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Full recording")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
mask   = (t >= zoom_start) & (t <= zoom_end)
v_zoom = v[mask]
t_zoom = t[mask]
ax2.plot(t_zoom, v_zoom, lw=0.8, color="steelblue")
draw_events(ax2, events, v_zoom.min(), v_zoom.max())
ax2.set_xlim(zoom_start, zoom_end)
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (V)")
ax2.set_title(f"Zoom: {zoom_start}s – {zoom_end}s")
ax2.grid(True, alpha=0.3)

ax3 = axes[2]
n        = len(v_zoom)
fft_vals = np.abs(np.fft.rfft(v_zoom - v_zoom.mean())) / n
freqs    = np.fft.rfftfreq(n, d=1/sample_rate)
ax3.semilogy(freqs, fft_vals, lw=0.5, color="darkorange")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Amplitude (log scale)")
ax3.set_title("Frequency spectrum of zoom window")
ax3.set_xlim(0, 500)
ax3.axvline(60,  color="red", lw=0.8, linestyle="--", label="60 Hz")
ax3.axvline(120, color="red", lw=0.8, linestyle="--", alpha=0.5, label="120 Hz")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()