# scripts/compare_spectra.py
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 6:
    print("Usage: python scripts/compare_spectra.py <data/csv_file> <on_start> <on_end> <off_start> <off_end>")
    print("Example: python scripts/compare_spectra.py data/light.csv 5 6 25 26")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)
t  = df["elapsed_s"].values
v  = df["voltage_V"].values
sr = round(1 / (t[1] - t[0]))

on_start,  on_end  = float(sys.argv[2]), float(sys.argv[3])
off_start, off_end = float(sys.argv[4]), float(sys.argv[5])

def get_spectrum(t, v, start, end, sr):
    mask  = (t >= start) & (t <= end)
    v_win = v[mask]
    n     = len(v_win)
    freqs = np.fft.rfftfreq(n, d=1/sr)
    amps  = np.abs(np.fft.rfft(v_win - v_win.mean())) / n
    return freqs, amps

freqs_on,  amps_on  = get_spectrum(t, v, on_start,  on_end,  sr)
freqs_off, amps_off = get_spectrum(t, v, off_start, off_end, sr)

basename = os.path.basename(csv_path)
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle(basename, fontsize=11)

ax1 = axes[0]
ax1.plot(t, v, lw=0.3, color="steelblue")
ax1.axvspan(on_start,  on_end,  color="red",   alpha=0.25, label=f"on  ({on_start}–{on_end}s)")
ax1.axvspan(off_start, off_end, color="green", alpha=0.25, label=f"off ({off_start}–{off_end}s)")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("Full recording — shaded regions used for spectra")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(freqs_on,  amps_on,  lw=0.8, color="red",   alpha=0.8, label=f"appliance ON  ({on_start}–{on_end}s)")
ax2.semilogy(freqs_off, amps_off, lw=0.8, color="green", alpha=0.8, label=f"appliance OFF ({off_start}–{off_end}s)")
ax2.axvline(60,  color="gray", lw=0.8, linestyle="--", alpha=0.5, label="60 Hz")
ax2.axvline(120, color="gray", lw=0.8, linestyle=":",  alpha=0.5, label="120 Hz")
ax2.set_xlim(0, 500)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Amplitude (log scale)")
ax2.set_title("Spectra: on vs off — same scale")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

ratio        = (amps_on + 1e-12) / (amps_off + 1e-12)
common_freqs = freqs_off
ratio_interp = np.interp(common_freqs, freqs_on, ratio)

ax3 = axes[2]
ax3.semilogy(common_freqs, ratio_interp, lw=0.8, color="purple", alpha=0.8)
ax3.axhline(1.0, color="gray", lw=0.8, linestyle="--", label="no change")
ax3.axvline(60,  color="gray", lw=0.8, linestyle="--", alpha=0.5)
ax3.axvline(120, color="gray", lw=0.8, linestyle=":",  alpha=0.5)
ax3.set_xlim(0, 500)
ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("ON / OFF ratio (log scale)")
ax3.set_title("Ratio on/off — above 1.0 means appliance adds energy at that frequency")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()