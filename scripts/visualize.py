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
sample_rate = round(1 / (t[1] - t[0]))

zoom_start = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
zoom_end   = float(sys.argv[3]) if len(sys.argv) > 3 else zoom_start + 0.1

data_dir    = os.path.dirname(csv_path)
basename    = os.path.basename(csv_path)
events_file = os.path.join(data_dir, basename.replace("light_", "events_"))

events = None
if os.path.exists(events_file):
    events = pd.read_csv(events_file)
    print(f"Loaded {len(events)} events from {events_file}")
else:
    print("No events file found.")

# --- compute spectrogram ---
WINDOW_S    = 0.1
OVERLAP     = 0.5
window_size = int(WINDOW_S * sample_rate)
step_size   = int(window_size * (1 - OVERLAP))

windows      = range(0, len(v) - window_size, step_size)
n_freqs      = window_size // 2 + 1
freqs        = np.fft.rfftfreq(window_size, d=1/sample_rate)
spectrogram  = np.zeros((n_freqs, len(windows)))
window_times = np.zeros(len(windows))

for i, start in enumerate(windows):
    segment           = v[start:start + window_size]
    segment           = segment - segment.mean()
    segment           = segment * np.hanning(len(segment))
    fft_vals          = np.abs(np.fft.rfft(segment)) / window_size
    spectrogram[:, i] = fft_vals
    window_times[i]   = t[start + window_size // 2]

spect_db = 20 * np.log10(spectrogram + 1e-12)

# clip to 500 Hz
freq_mask  = freqs <= 500
freqs_plot = freqs[freq_mask]
spect_plot = spect_db[freq_mask, :]

# --- plot ---
fig, axes = plt.subplots(3, 1, figsize=(14, 14),
                         gridspec_kw={"height_ratios": [1.5, 1.5, 2]})
fig.suptitle(basename, fontsize=11, y=0.95)

def draw_events(ax, events, ymin, ymax):
    if events is None:
        return
    for _, row in events.iterrows():
        color = "green" if row["label"].endswith("_on") else "red"
        ax.axvline(row["elapsed_s"], color=color, lw=1, linestyle="--", alpha=0.8)
        ax.text(row["elapsed_s"], ymax, row["label"],
                rotation=90, fontsize=7, va="top", color=color, alpha=0.8)

# panel 1: full time series
ax1 = axes[0]
ax1.plot(t, v, lw=0.4, color="steelblue")
ax1.axvspan(zoom_start, zoom_end, color="orange", alpha=0.25, label="zoom region")
draw_events(ax1, events, v.min(), v.max())
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Full recording")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# panel 2: zoomed window
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

# panel 3: FFT of zoom window
"""ax3 = axes[2]
n        = len(v_zoom)
fft_vals = np.abs(np.fft.rfft(v_zoom - v_zoom.mean())) / n
freqs_z  = np.fft.rfftfreq(n, d=1/sample_rate)
ax3.semilogy(freqs_z, fft_vals, lw=0.5, color="darkorange")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Amplitude (log scale)")
ax3.set_title("Frequency spectrum of zoom window")
ax3.set_xlim(0, 500)
ax3.axvline(60,  color="red", lw=0.8, linestyle="--", label="60 Hz")
ax3.axvline(120, color="red", lw=0.8, linestyle="--", alpha=0.5, label="120 Hz")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)"""

# panel 4: spectrogram
ax4 = axes[2]
img = ax4.imshow(
    spect_plot,
    aspect="auto",
    origin="lower",
    extent=[window_times[0], window_times[-1], freqs_plot[0], freqs_plot[-1]],
    cmap="inferno",
    interpolation="nearest",
)
plt.colorbar(img, ax=ax4, label="Amplitude (dB)")
if events is not None:
    for _, row in events.iterrows():
        color = "green" if row["label"].endswith("_on") else "red"
        ax4.axvline(row["elapsed_s"], color=color, lw=1, linestyle="--", alpha=0.8)
        ax4.text(row["elapsed_s"], 480, row["label"],
                 rotation=90, fontsize=7, va="top", color=color, alpha=0.8)
ax4.axhline(60,  color="cyan", lw=0.8, linestyle="--", alpha=0.5, label="60 Hz")
ax4.axhline(120, color="cyan", lw=0.8, linestyle=":",  alpha=0.5, label="120 Hz")
ax4.legend(fontsize=8, loc="upper right")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Frequency (Hz)")
ax4.set_title("Spectrogram — brighter = more energy")
ax4.set_xlim(window_times[0], window_times[-1])

plt.subplots_adjust(hspace=0.5, top=0.88, bottom=0.05, left=0.08, right=0.92)
plt.show()