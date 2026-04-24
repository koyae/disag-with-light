# scripts/train_model_multilabel.py
import os
import sys
import glob
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, hamming_loss
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

DATA_DIR     = "data"
SAMPLE_RATE  = 10_000
WINDOW_S     = 0.5
WINDOW_SIZE  = int(WINDOW_S * SAMPLE_RATE)
TEST_SIZE    = 0.2
RANDOM_STATE = 42
FFT_MAX_FREQ = 500
FFT_N_BINS   = 64
MODEL_DIR = "models"

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

    bin_edges   = np.linspace(0, FFT_MAX_FREQ, FFT_N_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    psd_b_binned = np.interp(bin_centers, freqs_b, psd_b)
    psd_a_binned = np.interp(bin_centers, freqs_a, psd_a)

    for i, f in enumerate(bin_centers):
        features[f"fft_before_{f:.1f}hz"] = psd_b_binned[i]
        features[f"fft_after_{f:.1f}hz"]  = psd_a_binned[i]
        features[f"fft_delta_{f:.1f}hz"]  = psd_a_binned[i] - psd_b_binned[i]

    return features

# ---------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------

def load_dataset():
    rows = []

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR   = os.path.join(SCRIPT_DIR, "..", "data")
    MODEL_DIR  = os.path.join(SCRIPT_DIR, "..", "models")

    light_files = glob.glob(os.path.join(DATA_DIR, "light_*.csv"))
    exclude_path = os.path.join(DATA_DIR, "exclude.txt")
    excluded = set()
    if os.path.exists(exclude_path):
        with open(exclude_path) as f:
            excluded = {line.strip() for line in f if line.strip()}
        print(f"Excluding {len(excluded)} files")

    light_files = [f for f in light_files
                   if os.path.basename(f) not in excluded]
    print(f"Processing {len(light_files)} files...")

    for light_path in light_files:
        basename    = os.path.basename(light_path)
        events_path = os.path.join(DATA_DIR, basename.replace("light_", "events_"))
        if not os.path.exists(events_path):
            continue

        print(f"  {basename}")
        df     = pd.read_csv(light_path)
        events = pd.read_csv(events_path)
        t      = df["elapsed_s"].values
        v      = df["voltage_V"].values
        sr     = round(1 / (t[1] - t[0]))

        appliances    = set(e.rsplit("_", 1)[0] for e in events["label"].values)
        current_state = {a: 0 for a in appliances}

        # event windows
        for _, event in events.iterrows():
            label     = event["label"]
            event_t   = float(event["elapsed_s"])
            appliance = label.rsplit("_", 1)[0]
            action    = label.rsplit("_", 1)[1]

            idx = np.searchsorted(t, event_t)
            if idx < WINDOW_SIZE or idx + WINDOW_SIZE >= len(v):
                continue

            before = v[idx - WINDOW_SIZE:idx]
            after  = v[idx:idx + WINDOW_SIZE]
            current_state[appliance] = 1 if action == "on" else 0

            features = extract_features(before, after, sr)
            for a in appliances:
                features[f"label_{a}"] = current_state[a]
            features["label_no_change"] = 0
            features["file"]            = basename
            features["event"]           = label
            features["appliance"]       = appliance
            features["action"]          = action
            features["event_t"]         = event_t
            rows.append(features)

        # no-change windows between events
        event_times = events["elapsed_s"].values
        for i in range(len(event_times) - 1):
            mid_t = (event_times[i] + event_times[i + 1]) / 2
            idx   = np.searchsorted(t, mid_t)
            if idx < WINDOW_SIZE or idx + WINDOW_SIZE >= len(v):
                continue

            state_at_mid = {a: 0 for a in appliances}
            for _, event in events.iterrows():
                if float(event["elapsed_s"]) < mid_t:
                    a      = event["label"].rsplit("_", 1)[0]
                    action = event["label"].rsplit("_", 1)[1]
                    state_at_mid[a] = 1 if action == "on" else 0

            before = v[idx - WINDOW_SIZE:idx]
            after  = v[idx:idx + WINDOW_SIZE]

            features = extract_features(before, after, sr)
            for a in appliances:
                features[f"label_{a}"] = state_at_mid[a]
            features["label_no_change"] = 1
            features["file"]            = basename
            features["event"]           = "no_change"
            features["appliance"]       = "none"
            features["action"]          = "none"
            features["event_t"]         = mid_t
            rows.append(features)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------

CLASSIFIERS = {
    "Logistic Regression": MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    ),
    "Random Forest": MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    ),
    "SVM": MultiOutputClassifier(
        SVC(kernel="rbf", random_state=RANDOM_STATE, probability=True)
    ),
    "k-NN (k=5)": MultiOutputClassifier(
        KNeighborsClassifier(n_neighbors=5)
    ),
}

# ---------------------------------------------------------------
# Train and evaluate
# ---------------------------------------------------------------

def train_and_evaluate(df):
    label_cols   = [c for c in df.columns if c.startswith("label_")]
    feature_cols = [c for c in df.columns
                    if c not in label_cols + ["file", "event", "appliance",
                                              "action", "event_t"]]

    print(f"\nDataset: {len(df)} windows, {len(label_cols)} labels")
    print(f"Feature columns: {len(feature_cols)}")
    print("\nLabel distribution:")
    for lc in label_cols:
        print(f"  {lc}: {df[lc].sum()} positive / {len(df)} total")

    X = df[feature_cols].values
    Y = df[label_cols].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # keep test indices aligned with df for timeline plot
    idx_all = np.arange(len(df))
    idx_train, idx_test = train_test_split(
        idx_all, test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    X_train, X_test = X_scaled[idx_train], X_scaled[idx_test]
    Y_train, Y_test = Y[idx_train],         Y[idx_test]
    df_test         = df.iloc[idx_test].reset_index(drop=True)

    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    results = {}
    for name, clf in CLASSIFIERS.items():
        print(f"\n{'='*50}\n{name}")
        clf.fit(X_train, Y_train)
        Y_pred      = clf.predict(X_test)
        hl          = hamming_loss(Y_test, Y_pred)
        exact_match = (Y_pred == Y_test).all(axis=1).mean()
        print(f"  Hamming loss:      {hl:.3f}")
        print(f"  Exact match ratio: {exact_match:.3f}")
        for i, lc in enumerate(label_cols):
            print(f"\n  {lc}:")
            print(classification_report(Y_test[:, i], Y_pred[:, i],
                                        target_names=["off", "on"],
                                        zero_division=0))
        results[name] = {
            "clf":         clf,
            "Y_pred":      Y_pred,
            "hl":          hl,
            "exact_match": exact_match,
        }
    print(f"DEBUG: about to save model to {MODEL_DIR}")

    import joblib

    os.makedirs(MODEL_DIR, exist_ok=True)

    # save the best model by exact match score
    best_name = max(results, key=lambda n: results[n]["exact_match"])
    best_clf  = results[best_name]["clf"]

    print(f"\nSaving best model ({best_name}) to {MODEL_DIR}/...")

    joblib.dump(best_clf, os.path.join(MODEL_DIR, "model.joblib"))
    joblib.dump({
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "label_cols":   label_cols,
        "model_name":   best_name,
    }, os.path.join(MODEL_DIR, "model_meta.joblib"))

    print(f"  model.joblib")
    print(f"  model_meta.joblib")
    print(f"  best model: {best_name} (exact match: {results[best_name]['exact_match']:.3f})")

    return results, Y_test, label_cols, feature_cols, scaler, df_test

# ---------------------------------------------------------------
# Aggregated confusion matrix
# ---------------------------------------------------------------

def plot_aggregated_confusion(results, Y_test, label_cols):
    """
    Flatten all labels and predictions into one big binary confusion matrix.
    Rows = actual, Cols = predicted, aggregated across all appliances.
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(5 * (n_models + 1), 5))
    fig.suptitle("Aggregated confusion matrices (all labels combined)", fontsize=11)

    all_true_flat = Y_test.flatten()

    for i, (name, res) in enumerate(results.items()):
        Y_pred    = res["Y_pred"]
        all_pred_flat = Y_pred.flatten()
        cm = confusion_matrix(all_true_flat, all_pred_flat)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["off", "on"],
                    yticklabels=["off", "on"],
                    ax=axes[i], cbar=False)
        axes[i].set_title(f"{name}\nHL: {res['hl']:.3f} EM: {res['exact_match']:.3f}",
                          fontsize=9)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    # overall comparison bar chart in last panel
    ax_bar = axes[-1]
    names  = list(results.keys())
    x      = np.arange(len(names))
    w      = 0.35
    ax_bar.bar(x - w/2, [results[n]["exact_match"] for n in names], w,
               label="Exact match", color="steelblue", alpha=0.8)
    ax_bar.bar(x + w/2, [1 - results[n]["hl"] for n in names], w,
               label="1 - Hamming loss", color="darkorange", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=20, fontsize=8)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_title("Overall comparison")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Timeline visualization
# ---------------------------------------------------------------

def plot_timelines(results, Y_test, label_cols, df_test):
    """
    For each model, show a timeline of predicted vs ground truth
    for each appliance label across the test set windows,
    sorted by event time.
    """
    appliance_labels = [lc.replace("label_", "") for lc in label_cols]
    n_labels         = len(label_cols)
    n_models         = len(results)

    # sort test windows by event time for a sensible x-axis
    sort_idx  = df_test["event_t"].argsort().values
    df_sorted = df_test.iloc[sort_idx].reset_index(drop=True)
    Y_sorted  = Y_test[sort_idx]
    x_times   = df_sorted["event_t"].values

    colors = {
        "true_on":   "#2ecc71",
        "true_off":  "#ecf0f1",
        "pred_on":   "#3498db",
        "pred_off":  "#ecf0f1",
        "mismatch":  "#e74c3c",
    }

    for model_name, res in results.items():
        Y_pred_sorted = res["Y_pred"][sort_idx]

        fig, axes = plt.subplots(n_labels, 1,
                                 figsize=(16, 2.5 * n_labels),
                                 sharex=True)
        if n_labels == 1:
            axes = [axes]
        fig.suptitle(f"Timeline — {model_name}", fontsize=11)

        for j, (ax, lc) in enumerate(zip(axes, label_cols)):
            appliance = lc.replace("label_", "")
            y_true    = Y_sorted[:, j]
            y_pred    = Y_pred_sorted[:, j]

            # draw ground truth as filled background
            for k in range(len(x_times)):
                x0 = x_times[k]
                x1 = x_times[k + 1] if k + 1 < len(x_times) else x0 + 1.0

                true_val = y_true[k]
                pred_val = y_pred[k]
                match    = true_val == pred_val

                # ground truth band (top half)
                ax.barh(1.5, x1 - x0, left=x0, height=0.8,
                        color=colors["true_on"] if true_val else colors["true_off"],
                        edgecolor="none", alpha=0.8)

                # prediction band (bottom half)
                bar_color = (colors["pred_on"] if pred_val else colors["pred_off"])
                if not match:
                    bar_color = colors["mismatch"]
                ax.barh(0.5, x1 - x0, left=x0, height=0.8,
                        color=bar_color,
                        edgecolor="none", alpha=0.8)

            ax.set_yticks([0.5, 1.5])
            ax.set_yticklabels(["predicted", "ground truth"], fontsize=8)
            ax.set_ylabel(appliance, fontsize=9, rotation=0,
                          labelpad=60, va="center")
            ax.set_ylim(0, 2.3)
            ax.grid(axis="x", alpha=0.3)

        axes[-1].set_xlabel("Event time (s)")

        # legend
        legend_patches = [
            mpatches.Patch(color=colors["true_on"],  label="On (ground truth)"),
            mpatches.Patch(color=colors["true_off"], label="Off (ground truth)"),
            mpatches.Patch(color=colors["pred_on"],  label="On (predicted)"),
            mpatches.Patch(color=colors["mismatch"], label="Mismatch"),
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.1))

        plt.tight_layout()
        plt.show()

def plot_per_device_confusion(results, Y_test, label_cols):
    """
    One confusion matrix per device per model.
    Layout: rows = models, columns = devices.
    """
    n_models  = len(results)
    n_devices = len(label_cols)
    appliance_labels = [lc.replace("label_", "") for lc in label_cols]

    fig, axes = plt.subplots(
        n_models, n_devices,
        figsize=(3.5 * n_devices, 3.5 * n_models),
        squeeze=False,
    )
    fig.suptitle("Per-device confusion matrices", fontsize=12)

    for i, (model_name, res) in enumerate(results.items()):
        Y_pred = res["Y_pred"]
        for j, (lc, appliance) in enumerate(zip(label_cols, appliance_labels)):
            ax     = axes[i][j]
            y_true = Y_test[:, j]
            y_pred = Y_pred[:, j]
            cm     = confusion_matrix(y_true, y_pred)

            # per-device accuracy for subtitle
            acc = (y_true == y_pred).mean()

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["off", "on"],
                yticklabels=["off", "on"],
                ax=ax,
                cbar=False,
            )
            ax.set_title(f"{appliance}\nacc: {acc:.2f}", fontsize=9)
            ax.set_xlabel("Predicted", fontsize=8)

            # only label y-axis on leftmost column
            if j == 0:
                ax.set_ylabel(f"{model_name}\nActual", fontsize=8)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()

    if len(df) == 0:
        print("No labeled events found.")
        sys.exit(1)

    features_path = os.path.join(DATA_DIR, "features_multilabel.csv")
    df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")

    results, Y_test, label_cols, feature_cols, scaler, df_test = \
        train_and_evaluate(df)

    #plot_aggregated_confusion(results, Y_test, label_cols)
    #plot_per_device_confusion(results, Y_test, label_cols)
    plot_timelines(results, Y_test, label_cols, df_test)