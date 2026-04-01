# visualize.py
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python visualize.py <csv_file>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
t = df["elapsed_s"].values
v = df["voltage_V"].values
sample_rate = 1 / (t[1] - t[0])

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(sys.argv[1], fontsize=11)

# --- Plot 1: full time series ---
ax1 = axes[0]
ax1.plot(t, v, lw=0.4, color="steelblue")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Full recording")
ax1.grid(True, alpha=0.3)

# --- Plot 2: zoomed to first 0.1s to see 60 Hz waveform ---
ax2 = axes[1]
mask = t < 0.1
ax2.plot(t[mask], v[mask], lw=0.8, color="steelblue")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Voltage (V)")
ax2.set_title("First 0.1s (shows 60 Hz AC cycles)")
ax2.grid(True, alpha=0.3)

# --- Plot 3: FFT frequency spectrum ---
ax3 = axes[2]
n = len(v)
fft_vals = np.abs(np.fft.rfft(v - v.mean())) / n
freqs = np.fft.rfftfreq(n, d=1/sample_rate)
ax3.semilogy(freqs, fft_vals, lw=0.5, color="darkorange")
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Amplitude (log scale)")
ax3.set_title("Frequency spectrum")
ax3.set_xlim(0, 500)   # focus on 0–500 Hz where interesting stuff lives
ax3.axvline(60, color="red", lw=0.8, linestyle="--", label="60 Hz")
ax3.axvline(120, color="red", lw=0.8, linestyle="--", alpha=0.5, label="120 Hz")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()