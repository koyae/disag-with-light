# scripts/animate_spectrogram.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

if len(sys.argv) < 2:
    print("Usage: python scripts/animate_spectrogram.py <data/csv_file> [max_freq]")
    print("Example: python scripts/animate_spectrogram.py data/light.csv 500")
    sys.exit(1)

csv_path = sys.argv[1]
max_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 500

df = pd.read_csv(csv_path)
t  = df["elapsed_s"].values
v  = df["voltage_V"].values
sr = round(1 / (t[1] - t[0]))

print(f"Loaded {len(df)} rows, {t[-1]:.1f}s of data at {sr} Hz")

# load events
data_dir    = os.path.dirname(csv_path)
basename    = os.path.basename(csv_path)
events_file = os.path.join(data_dir, basename.replace("light_", "events_"))
events      = None
if os.path.exists(events_file):
    events = pd.read_csv(events_file)
    print(f"Loaded {len(events)} events from {events_file}")

# --- compute full spectrogram upfront ---
WINDOW_S    = 0.1
OVERLAP     = 0.5
window_size = int(WINDOW_S * sr)
step_size   = int(window_size * (1 - OVERLAP))

all_windows  = list(range(0, len(v) - window_size, step_size))
n_freqs      = window_size // 2 + 1
freqs        = np.fft.rfftfreq(window_size, d=1/sr)
spectrogram  = np.zeros((n_freqs, len(all_windows)))
window_times = np.zeros(len(all_windows))

print("Computing spectrogram...")
for i, start in enumerate(all_windows):
    segment           = v[start:start + window_size]
    segment           = segment - segment.mean()
    segment           = segment * np.hanning(len(segment))
    fft_vals          = np.abs(np.fft.rfft(segment)) / window_size
    spectrogram[:, i] = fft_vals
    window_times[i]   = t[start + window_size // 2]

spect_db = 20 * np.log10(spectrogram + 1e-12)

freq_mask  = freqs <= max_freq
freqs_plot = freqs[freq_mask]
spect_plot = spect_db[freq_mask, :]

# how many spectrogram columns fit in the display window
DISPLAY_S   = 5.0   # seconds of history to show
cols_per_s  = 1 / (step_size / sr)
display_cols = int(DISPLAY_S * cols_per_s)

# animation frame rate — skip columns to control speed
# 1 = real time, 2 = 2x speed, etc.
PLAYBACK_SPEED = 2
FRAME_STEP     = max(1, int(PLAYBACK_SPEED * step_size / sr * 30))  # ~30fps target

print(f"Rendering animation at ~{PLAYBACK_SPEED}x speed...")

# --- set up figure ---
fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [1, 2]})
fig.suptitle(basename, fontsize=11)
plt.subplots_adjust(hspace=0.4, top=0.92, bottom=0.08, left=0.08, right=0.95)

ax1, ax2 = axes

# panel 1: raw signal (full) with moving cursor
ax1.plot(t, v, lw=0.3, color="steelblue", zorder=1)
if events is not None:
    for _, row in events.iterrows():
        color = "green" if row["label"].endswith("_on") else "red"
        ax1.axvline(row["elapsed_s"], color=color, lw=1,
                    linestyle="--", alpha=0.8, zorder=2)
        ax1.text(row["elapsed_s"], v.max(), row["label"],
                 rotation=90, fontsize=7, va="top", color=color, alpha=0.8)
cursor_line = ax1.axvline(0, color="yellow", lw=1.5, zorder=3)
ax1.set_xlim(t[0], t[-1])
ax1.set_ylabel("Voltage (V)")
ax1.set_xlabel("Time (s)")
ax1.set_title("Full recording")
ax1.grid(True, alpha=0.3)

# panel 2: scrolling spectrogram
# initialize with empty data
init_data = np.full((len(freqs_plot), display_cols), spect_plot.min())
img = ax2.imshow(
    init_data,
    aspect="auto",
    origin="lower",
    extent=[0, DISPLAY_S, freqs_plot[0], freqs_plot[-1]],
    cmap="inferno",
    interpolation="nearest",
    vmin=np.percentile(spect_plot, 5),
    vmax=np.percentile(spect_plot, 99),
)
plt.colorbar(img, ax=ax2, label="Amplitude (dB)")
ax2.axhline(60,  color="cyan", lw=0.8, linestyle="--", alpha=0.6, label="60 Hz")
ax2.axhline(120, color="cyan", lw=0.8, linestyle=":",  alpha=0.6, label="120 Hz")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time (s) — scrolling window")
ax2.set_title(f"Scrolling spectrogram ({DISPLAY_S}s window)")

# time label
time_text = ax1.text(0.01, 0.95, "", transform=ax1.transAxes,
                     fontsize=9, va="top", color="white",
                     bbox=dict(boxstyle="round", facecolor="steelblue", alpha=0.6))

# rolling buffer for spectrogram display
buffer = np.full((len(freqs_plot), display_cols), spect_plot.min())

def update(frame_idx):
    col = frame_idx * FRAME_STEP
    if col >= spect_plot.shape[1]:
        return img, cursor_line, time_text

    # update rolling buffer
    new_col = spect_plot[:, col:col + FRAME_STEP]
    cols_to_add = new_col.shape[1]
    buffer[:, :-cols_to_add] = buffer[:, cols_to_add:]
    buffer[:, -cols_to_add:] = new_col

    img.set_data(buffer)

    # update cursor on raw signal
    current_time = window_times[col]
    cursor_line.set_xdata([current_time, current_time])

    # update time label
    time_text.set_text(f"t = {current_time:.2f}s")

    return img, cursor_line, time_text

n_frames = spect_plot.shape[1] // FRAME_STEP
interval = 1000 / 30  # ms per frame for ~30fps

anim = animation.FuncAnimation(
    fig, update,
    frames=n_frames,
    interval=interval,
    blit=True,
)

plt.show()