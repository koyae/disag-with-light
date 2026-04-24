# scripts/live_detect.py
import sys
import os
import time
import threading
import numpy as np
import pandas as pd
import glob
import joblib
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType

# --- config ---
CHANNEL      = "myDAQ1/ai0"
SAMPLE_RATE  = 1_000
BUFFER_SIZE  = 1000
WINDOW_S     = 1.0
WINDOW_SIZE  = int(WINDOW_S * SAMPLE_RATE)
DATA_DIR     = "data"
MODEL_DIR    = "models"

MEAN_THRESHOLD  = 0.005
STD_THRESHOLD   = 0.002
BUFFER_SAMPLES  = WINDOW_SIZE * 4

FFT_MAX_FREQ = 500
FFT_N_BINS   = 64
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------

def extract_features(before, after, sr):
    features = {}

    features["mean_before"]  = before.mean()
    features["mean_after"]   = after.mean()
    features["mean_delta"]   = after.mean() - before.mean()
    features["std_before"]   = before.std()
    features["std_after"]    = after.std()
    features["std_delta"]    = after.std() - before.std()
    features["range_before"] = before.max() - before.min()
    features["range_after"]  = after.max() - after.min()
    features["range_delta"]  = features["range_after"] - features["range_before"]

    freqs_b, psd_b = welch(before - before.mean(), fs=sr,
                           nperseg=min(len(before), 1024))
    freqs_a, psd_a = welch(after  - after.mean(),  fs=sr,
                           nperseg=min(len(after),  1024))

    bin_edges    = np.linspace(0, FFT_MAX_FREQ, FFT_N_BINS + 1)
    bin_centers  = (bin_edges[:-1] + bin_edges[1:]) / 2
    psd_b_binned = np.interp(bin_centers, freqs_b, psd_b)
    psd_a_binned = np.interp(bin_centers, freqs_a, psd_a)

    for i, f in enumerate(bin_centers):
        features[f"fft_before_{f:.1f}hz"] = psd_b_binned[i]
        features[f"fft_after_{f:.1f}hz"]  = psd_a_binned[i]
        features[f"fft_delta_{f:.1f}hz"]  = psd_a_binned[i] - psd_b_binned[i]

    return features

def features_to_vector(features, feature_cols):
    return np.array([[features[c] for c in feature_cols]])

# ---------------------------------------------------------------
# Load or train model
# ---------------------------------------------------------------

def load_or_train_model():
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    meta_path  = os.path.join(MODEL_DIR, "model_meta.joblib")

    if os.path.exists(model_path) and os.path.exists(meta_path):
        print("Loading saved model...")
        model = joblib.load(model_path)
        meta  = joblib.load(meta_path)
        return model, meta["scaler"], meta["feature_cols"], meta["label_cols"]

    print("No saved model found — training from features_multilabel.csv...")
    features_path = os.path.join(DATA_DIR, "features_multilabel.csv")
    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run train_model_multilabel.py first.")
        sys.exit(1)

    df = pd.read_csv(features_path)

    label_cols   = [c for c in df.columns if c.startswith("label_")]
    feature_cols = [c for c in df.columns
                    if c not in label_cols + ["file", "event",
                                              "appliance", "action", "event_t"]]

    X = df[feature_cols].values
    Y = df[label_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    )
    model.fit(X_scaled, Y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump({
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "label_cols":   label_cols,
    }, meta_path)
    print(f"Model trained and saved to {MODEL_DIR}/")

    return model, scaler, feature_cols, label_cols

# ---------------------------------------------------------------
# Event detector
# ---------------------------------------------------------------

class EventDetector:
    def __init__(self, window_size, mean_threshold, std_threshold):
        self.window_size      = window_size
        self.mean_threshold   = mean_threshold
        self.std_threshold    = std_threshold
        self.last_event_idx   = -window_size * 2
        self.cooldown_samples = window_size * 2

    def check(self, buffer, current_idx):
        if current_idx - self.last_event_idx < self.cooldown_samples:
            return False, None, None
        if len(buffer) < self.window_size * 2:
            return False, None, None

        before = buffer[-self.window_size * 2:-self.window_size]
        after  = buffer[-self.window_size:]

        mean_delta = after.mean() - before.mean()
        std_delta  = after.std()  - before.std()

        triggered = (abs(mean_delta) > self.mean_threshold or
                     abs(std_delta)  > self.std_threshold)

        if triggered:
            self.last_event_idx = current_idx
            return True, before, after

        return False, None, None

# ---------------------------------------------------------------
# Predict and format
# ---------------------------------------------------------------

def predict_state(before, after, model, scaler, feature_cols, label_cols):
    features = extract_features(before, after, SAMPLE_RATE)
    X        = features_to_vector(features, feature_cols)
    X_scaled = scaler.transform(X)
    Y_pred   = model.predict(X_scaled)[0]

    try:
        probs = np.array([
            est.predict_proba(X_scaled)[0][1]
            for est in model.estimators_
        ])
    except Exception:
        probs = Y_pred.astype(float)

    state = {}
    for i, lc in enumerate(label_cols):
        appliance        = lc.replace("label_", "")
        state[appliance] = {
            "on":   bool(Y_pred[i]),
            "prob": float(probs[i]),
        }

    return state

def format_prediction(state, elapsed):
    lines = [f"\n{'='*50}",
             f"Event detected at t={elapsed:.2f}s",
             f"{'='*50}"]

    no_change = state.pop("no_change", None)

    sorted_state = sorted(state.items(),
                          key=lambda x: x[1]["prob"], reverse=True)

    for appliance, info in sorted_state:
        bar    = "█" * int(info["prob"] * 20)
        status = "ON " if info["on"] else "off"
        lines.append(
            f"  {appliance:<15} {status}  {bar:<20} {info['prob']:.2f}"
        )

    if no_change:
        lines.append(
            f"\n  no_change: {'yes' if no_change['on'] else 'no'}"
            f"  ({no_change['prob']:.2f})"
        )

    lines.append("=" * 50)
    return "\n".join(lines)

# ---------------------------------------------------------------
# Interactive mode
# ---------------------------------------------------------------

def interactive_mode(model, scaler, feature_cols, label_cols):
    print(f"Interactive mode — press Enter to trigger a random appliance, Ctrl+C to stop.")
    print(f"Window size: {WINDOW_S}s\n")

    # --- Home Assistant config ---
    HA_URL   = "http://homeassistant.local:8123"
    HA_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiI4YmZmNjJkNDA3ZDg0OGMzYWNlOWQwZDkwNDlmYmU3OCIsImlhdCI6MTc3NjgwMjkxNiwiZXhwIjoyMDkyMTYyOTE2fQ.deCyR50AtJKF2n1_rt_Dx0mcx4wtYoe5d5wLO_neTxM"
    HA_HEADERS = {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }
    PLUGS = {
    "computer":     "switch.lumi_lumi_plug_maus01",
    "space_heater": "switch.maus02_lumi_lumi_plug",
    "kettle":       "switch.maus03_lumi_lumi_plug",
    "fridge":       "switch.maus04_lumi_lumi_plug",
    }

    import requests
    import random

    def plug_on(entity_id):
        requests.post(
            f"{HA_URL}/api/services/switch/turn_on",
            headers=HA_HEADERS,
            json={"entity_id": entity_id},
        )

    def plug_off(entity_id):
        requests.post(
            f"{HA_URL}/api/services/switch/turn_off",
            headers=HA_HEADERS,
            json={"entity_id": entity_id},
        )

    # make sure all plugs start off
    print("Turning all plugs off...")
    for entity_id in PLUGS.values():
        plug_off(entity_id)
    time.sleep(2)

    buffer           = np.array([])
    n_samples        = 0
    collecting_after = False
    before           = None
    waiting_since    = 0
    chosen_appliance = None

    event_flag = threading.Event()

    def listen():
        while True:
            input()
            event_flag.set()

    t = threading.Thread(target=listen, daemon=True)
    t.start()

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            CHANNEL,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=0.0, max_val=5.0,
        )
        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=BUFFER_SIZE * 10,
        )

        start_time = time.time()
        try:
            while True:
                samples    = np.array(task.read(
                    number_of_samples_per_channel=BUFFER_SIZE))
                buffer     = np.concatenate([buffer, samples])
                buffer     = buffer[-BUFFER_SAMPLES:]
                n_samples += len(samples)

                if event_flag.is_set():
                    event_flag.clear()
                    time.sleep(0.05)

                    if len(buffer) < WINDOW_SIZE * 2:
                        print("  Not enough data yet...")
                        continue

                    # snapshot before
                    before        = buffer[-WINDOW_SIZE:].copy()
                    waiting_since = n_samples

                    # randomly pick and turn on an appliance
                    chosen_appliance, chosen_entity = random.choice(list(PLUGS.items()))
                    plug_on(chosen_entity)
                    print(f"  Turned on: [HIDDEN] — collecting post-event data...")
                    collecting_after = True

                if collecting_after and (n_samples - waiting_since) >= WINDOW_SIZE:
                    after            = buffer[-WINDOW_SIZE:].copy()
                    collecting_after = False

                    elapsed    = time.time() - start_time
                    mean_delta = after.mean() - before.mean()
                    std_delta  = after.std()  - before.std()

                    print(f"  before mean: {before.mean():.4f}V  "
                          f"after mean: {after.mean():.4f}V")
                    print(f"  mean_delta: {mean_delta*1000:+.2f}mV")
                    print(f"  std_delta:  {std_delta*1000:+.2f}mV")

                    if abs(mean_delta) < 0.005 and abs(std_delta) < 0.002:
                        print("  Delta too small — no significant event detected")
                    else:
                        state = predict_state(
                            before, after, model, scaler, feature_cols, label_cols
                        )
                        print(format_prediction(state, elapsed))

                    # reveal ground truth
                    print(f"\n  *** Ground truth: {chosen_appliance} ***\n")

                    # turn it back off after 5 seconds
                    time.sleep(5)
                    plug_off(chosen_entity)
                    print(f"  Turned off {chosen_appliance}. Press Enter for next trial.")

        except KeyboardInterrupt:
            print("\nTurning all plugs off...")
            for entity_id in PLUGS.values():
                plug_off(entity_id)
            print("Stopped.")
# ---------------------------------------------------------------
# Auto mode
# ---------------------------------------------------------------

def auto_mode(model, scaler, feature_cols, label_cols):
    print(f"Auto mode — event detection running")
    print(f"Mean threshold: ±{MEAN_THRESHOLD*1000:.1f}mV")
    print(f"Std threshold:  ±{STD_THRESHOLD*1000:.1f}mV")
    print("Press Ctrl+C to stop.\n")

    detector   = EventDetector(WINDOW_SIZE, MEAN_THRESHOLD, STD_THRESHOLD)
    buffer     = np.array([])
    n_samples  = 0
    start_time = time.time()

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            CHANNEL,
            terminal_config=TerminalConfiguration.DIFF,
            min_val=0.0, max_val=5.0,
        )
        task.timing.cfg_samp_clk_timing(
            rate=SAMPLE_RATE,
            sample_mode=AcquisitionType.CONTINUOUS,
            samps_per_chan=BUFFER_SIZE * 10,
        )

        try:
            while True:
                samples    = np.array(task.read(
                    number_of_samples_per_channel=BUFFER_SIZE))
                buffer     = np.concatenate([buffer, samples])
                buffer     = buffer[-BUFFER_SAMPLES:]
                n_samples += len(samples)

                elapsed = time.time() - start_time

                triggered, before, after = detector.check(buffer, n_samples)
                if triggered:
                    state = predict_state(
                        before, after, model, scaler, feature_cols, label_cols
                    )
                    print(format_prediction(state, elapsed))

        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\nStopped after {elapsed:.1f}s")

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    model, scaler, feature_cols, label_cols = load_or_train_model()

    appliance_names = [lc.replace("label_", "") for lc in label_cols
                       if lc != "label_no_change"]
    print(f"\nLoaded model — tracking: {appliance_names}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if mode == "interactive":
        interactive_mode(model, scaler, feature_cols, label_cols)
    else:
        auto_mode(model, scaler, feature_cols, label_cols)

if __name__ == "__main__":
    main()