# scripts/spectrogram.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

if len(sys.argv) < 2:
    print("Usage: python scripts/spectrogram.py <data/csv_file> [max_freq]")
    print("Example: python scripts/spectrogram.py data/light.csv 500")
    sys.exit(1)

csv_path = sys.argv[1]
max_freq = float(sys.argv[2]) if len(sys.argv) > 2 else 500

df = pd.read_csv(csv_path)
t  = df["elapsed_s"].values
v  = df["voltage_V"].values
sr = round(1 / (t[1] - t[0]))

print(f"Loaded {len(df)} rows, {t[-1]:.1f}s of data at {sr} Hz")

# window size: 0.1s gives good time resolution while still resolving 10Hz bins
WINDOW_S    = 0.1
OVERLAP     = 0.5                        # 50% overlap between windows
window_size = int(WINDOW_S * sr)
step_size   = int(window_size * (1 - OVERLAP))

print(f"Window: {WINDOW_S}s ({window_size} samples), step: {step_size} samples")
print("Computing spectrogram...")

# build spectrogram manually so we control everything
windows      = range(0, len(v) - window_size, step_size)
n_windows    = len(windows)
n_freqs      = window_size // 2 + 1
freqs        = np.fft.rfftfreq(window_size, d=1/sr)
spectrogram  = np.zeros((n_freqs, n_windows))
window_times = np.zeros(n_windows)

for i, start in enumerate(windows):
    segment          = v[start:start + window_size]
    segment          = segment - segment.mean()              # remove DC
    segment          = segment * np.hanning(len(segment))   # apply window fnapply window fn
    fft_vals         = np.abs(np.fft.rfft(segment)) / window_size
    spectrogram[:,i] = fft_vals
    window_times[i]  = t[start + window_size // 2]

# load events file if it exists
data_dir    = os.path.dirname(csv_path)
basename    = os.path.basename(csv_path)
events_file = os.path.join(data_dir, basename.replace("light_", "events_"))
events      = None
if os.path.exists(events_file):
    events = pd.read_csv(events_file)
    print(f"Loaded {len(events)} events from {events_file}")

# clip to max_freq
freq_mask   = freqs <= max_freq
freqs_plot  = freqs[freq_mask]
spect_plot  = spectrogram[freq_mask, :]

print("Plotting...")

fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                         gridspec_kw={"height_ratios": [1, 3]})
fig.suptitle(basename, fontsize=11)

# panel 1: raw signal for reference
ax1 = axes[0]
ax1.plot(t, v, lw=0.3, color="steelblue")
if events is not None:
    for _, row in events.iterrows():
        color = "green" if row["label"].endswith("_on") else "red"
        ax1.axvline(row["elapsed_s"], color=color, lw=1, linestyle="--", alpha=0.8)
        ax1.text(row["elapsed_s"], v.max(), row["label"],
                 rotation=90, fontsize=7, va="top", color=color, alpha=0.8)
ax1.set_ylabel("Voltage (V)")
ax1.set_xlabel("Time (s)")
ax1.set_title("Raw signal")
ax1.set_xlim(window_times[0], window_times[-1])
ax1.grid(True, alpha=0.3)

# panel 2: spectrogram
ax2 = axes[1]
# log scale amplitude for better visibility
spect_db = 20 * np.log10(spect_plot + 1e-12)

img = ax2.imshow(
    spect_db,
    aspect="auto",
    origin="lower",
    extent=[window_times[0], window_times[-1], freqs_plot[0], freqs_plot[-1]],
    cmap="inferno",
    interpolation="nearest",
)
plt.colorbar(img, ax=ax2, label="Amplitude (dB)")

# mark event lines on spectrogram too
if events is not None:
    for _, row in events.iterrows():
        color = "green" if row["label"].endswith("_on") else "red"
        ax2.axvline(row["elapsed_s"], color=color, lw=1, linestyle="--", alpha=0.8)
        ax2.text(row["elapsed_s"], max_freq * 0.95, row["label"],
                 rotation=90, fontsize=7, va="top", color=color, alpha=0.8)

# mark 60 and 120 Hz reference lines
ax2.axhline(60,  color="cyan", lw=0.8, linestyle="--", alpha=0.5, label="60 Hz")
ax2.axhline(120, color="cyan", lw=0.8, linestyle=":",  alpha=0.5, label="120 Hz")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_xlabel("Time (s)")
ax2.set_title("Spectrogram — brighter = more energy")
ax2.set_xlim(window_times[0], window_times[-1])

plt.tight_layout()
plt.show()