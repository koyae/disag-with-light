# scripts/train_model.py

"""
Performance:
    745MB of info -> ~18s runtime
    One file is fully loaded into memory at a time
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR    = "data_e"
SAMPLE_RATE = 10_000
WINDOW_S    = 0.5       # seconds before/after event to analyze
                        # window-size as a sample-count is computed
                        # dynamically from this value.
TEST_SIZE   = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------

def band_power(v, sr, low, high):
    """ voltage-info, sampling_rate, freq band bounds -> average power in band
   """
    freqs, psd = welch(v, fs=sr, nperseg=min(len(v), 1024))
    mask = (freqs >= low) & (freqs <= high)
    return psd[mask].mean() if mask.any() else 0.0

def extract_features(before, after, sr):
    features = {}


    # time domain
    features["mean_before"]  = before.mean()
    features["mean_after"]   = after.mean()
    features["mean_delta"]   = after.mean() - before.mean()
    features["std_before"]   = before.std()
    features["std_after"]    = after.std()
    features["std_delta"]    = after.std() - before.std()
    features["range_before"] = before.max() - before.min()
    features["range_after"]  = after.max() - after.min()
    features["range_delta"]  = features["range_after"] - features["range_before"]

    # frequency domain
    for label, low, high in [
        ("60hz",   55,  65),
        ("120hz",  115, 125),
        ("240hz",  235, 245),
        ("300hz",  295, 305),
        ("broad",  200, 500),
    ]:
        pb = band_power(before, sr, low, high)
        pa = band_power(after,  sr, low, high)
        features[f"power_{label}_before"] = pb
        features[f"power_{label}_after"]  = pa
        features[f"power_{label}_delta"]  = pa - pb

    # 120hz peak width
    freqs_b, psd_b = welch(before, fs=sr, nperseg=min(len(before), 1024))
    freqs_a, psd_a = welch(after,  fs=sr, nperseg=min(len(after),  1024))
    mask_120 = (freqs_b >= 100) & (freqs_b <= 140)
    features["peak_width_before"] = np.sum(psd_b[mask_120] > psd_b[mask_120].max() * 0.5)
    features["peak_width_after"]  = np.sum(psd_a[mask_120] > psd_a[mask_120].max() * 0.5)
    features["peak_width_delta"]  = features["peak_width_after"] - features["peak_width_before"]

    return features

# ---------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------

def load_dataset():
    rows = []

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
            print(f"Warning: no events file found for {basename}, skipping.",
                  file=sys.stderr
            )
            continue

        print(f"Now processing {basename}...")

        df     = pd.read_csv(light_path)
        events = pd.read_csv(events_path)
        t      = df["elapsed_s"].values
        v      = df["voltage_V"].values
        sr     = round(1 / (t[1] - t[0]))

        # Recompute window size based on actual sampling rate:
        window_size = int(WINDOW_S * sr)

        for _, event in events.iterrows():
            label   = event["label"]
            event_t = float(event["elapsed_s"])
            idx     = np.searchsorted(t, event_t)

            if idx < window_size or idx + window_size >= len(v):
                continue

            before = v[idx - window_size:idx]
            after  = v[idx:idx + window_size]

            features              = extract_features(before, after, sr, filename=light_path)
            features["label"]     = label
            features["appliance"] = label.rsplit("_", 1)[0]
            features["event"]     = label.rsplit("_", 1)[1]
            features["file"]      = basename
            rows.append(features)

    return pd.DataFrame(rows)

# ---------------------------------------------------------------
# Train and evaluate multiple classifiers
# ---------------------------------------------------------------

CLASSIFIERS = { # models and parameters
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "SVM":                 SVC(kernel="rbf", random_state=RANDOM_STATE),
    "k-NN (k=5)":          KNeighborsClassifier(n_neighbors=5),
    # "k-NN (k=5)":          KNeighborsClassifier(n_neighbors=3),
    # "k-NN (k=5)":          KNeighborsClassifier(n_neighbors=2),
}

def train_and_evaluate(df, event_type="on"):
    print(f"\nDataset: {len(df)} events across {df['appliance'].nunique()} appliances")
    print(df["appliance"].value_counts().to_string())

    # Filter by event type
    if event_type == "both":
        df_filtered = df.copy()
        event_label = "all events"
    else:
        df_filtered = df[df["event"] == event_type].copy()
        event_label = f"'{event_type}' events"

    print(f"\nUsing {len(df_filtered)} {event_label} for training/testing")

    feature_cols = [c for c in df_filtered.columns
                    if c not in ["label", "appliance", "event", "file"]]

    X = df_filtered[feature_cols].values
    y = df_filtered["appliance"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train: {len(X_train)} samples  Test: {len(X_test)} samples")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    for name, clf in CLASSIFIERS.items():
        # cross-validation on training set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")

        # fit and evaluate on held-out test set
        clf.fit(X_train, y_train)
        y_pred    = clf.predict(X_test)
        test_acc  = (y_pred == y_test).mean()

        results[name] = {
            "clf":      clf,
            "y_pred":   y_pred,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "test_acc": test_acc,
        }

        print(f"\n{'='*50}")
        print(f"{name}")
        print(f"  CV accuracy:   {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Test accuracy: {test_acc:.3f}")
        print(classification_report(y_test, y_pred))

    return results, y_test, feature_cols, scaler, X_test, df_filtered

# ---------------------------------------------------------------
# User interaction
# ---------------------------------------------------------------

def prompt_yes_no(question):
    """Ask user a yes/no question and return True for yes, False for no."""
    while True:
        response = input(f"{question} (y/n): ").strip().lower()
        if response in ["yes", "y"]:
            return True
        elif response in ["no", "n"]:
            return False
        else:
            print("Please answer 'yes' or 'no'.")

def parse_options():
    """Parse command-line arguments for features and event type options.

    Returns:
        Tuple of (use_existing, event_type)
        - use_existing: True to use cached, False to regenerate, None to prompt
        - event_type: 'on', 'off', or 'both'
    """
    parser = argparse.ArgumentParser(description="Train and evaluate appliance classifiers")

    # Features cache options
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use-cached", action="store_true",
                       help="Use cached features.csv if available")
    group.add_argument("--regenerate", action="store_true",
                       help="Regenerate features from raw data")

    # Event type option
    parser.add_argument("--event-type", choices=["on", "off", "both"], default="on",
                        help="Event type to train on (default: on)")

    args = parser.parse_args()

    # Determine use_existing value
    if args.use_cached:
        use_existing = True
    elif args.regenerate:
        use_existing = False
    else:
        use_existing = None  # Prompt the user

    return use_existing, args.event_type

def generate_and_save_features(features_path):
    """Load dataset and save features to disk."""
    print("Generating new features...")
    df = load_dataset()
    if len(df) == 0:
        print("No labeled events found.")
        sys.exit(1)
    df.to_csv(features_path, index=False)
    print(f"Features saved to {features_path}")
    return df

# ---------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------

def plot_results(results, y_test, feature_cols, df_on, scaler, event_type="on"):
    if event_type not in ["on", "off", "both"]:
        raise ValueError("Invalid event_type for plotting. Must be 'on', 'off', or 'both'.")

    event_label = f"'{event_type}'" if event_type != "both" else "'on' and 'off'"

    n_classifiers = len(results)
    classes       = sorted(set(y_test))

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Appliance classification — model comparison ({event_label} events)", fontsize=12)

    # --- row 1: confusion matrices ---
    for i, (name, res) in enumerate(results.items()):
        ax = fig.add_subplot(3, n_classifiers, i + 1)
        cm = confusion_matrix(y_test, res["y_pred"], labels=classes)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    cbar=False)
        ax.set_title(f"{name}\nCV: {res['cv_mean']:.3f} Test: {res['test_acc']:.3f}",
                     fontsize=9)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Actual", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    # --- row 2: CV accuracy comparison bar chart ---
    ax_bar = fig.add_subplot(3, 1, 2)
    names     = list(results.keys())
    cv_means  = [results[n]["cv_mean"]  for n in names]
    cv_stds   = [results[n]["cv_std"]   for n in names]
    test_accs = [results[n]["test_acc"] for n in names]
    x = np.arange(len(names))
    w = 0.35
    ax_bar.bar(x - w/2, cv_means,  w, yerr=cv_stds, label="CV accuracy",
               color="steelblue", alpha=0.8, capsize=4)
    ax_bar.bar(x + w/2, test_accs, w, label="Test accuracy",
               color="darkorange", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, fontsize=9)
    ax_bar.set_ylabel("Accuracy")
    ax_bar.set_title("Model comparison — CV vs test accuracy")
    ax_bar.set_ylim(0, 1.1)
    ax_bar.axhline(1/len(classes), color="red", lw=0.8,
                   linestyle="--", label=f"Random baseline ({1/len(classes):.2f})")
    ax_bar.legend(fontsize=8)
    ax_bar.grid(True, alpha=0.3)

    # --- row 3: logistic regression feature importance ---
    ax_feat = fig.add_subplot(3, 1, 3)
    lr = results["Logistic Regression"]["clf"]
    coef_df = pd.DataFrame(lr.coef_, columns=feature_cols, index=lr.classes_)
    top_features = coef_df.abs().max().nlargest(10).index
    coef_df[top_features].T.plot(kind="bar", ax=ax_feat)
    ax_feat.set_title("Top 10 logistic regression feature coefficients")
    ax_feat.set_xlabel("Feature")
    ax_feat.set_ylabel("Coefficient")
    ax_feat.tick_params(axis="x", rotation=45)
    ax_feat.legend(fontsize=8)
    ax_feat.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

if __name__ == "__main__":
    features_path = os.path.join(DATA_DIR, "features.csv")

    # Parse command-line arguments
    use_existing, event_type = parse_options()

    # Check if features already exist
    if os.path.exists(features_path) and use_existing is not False:
        # Prompt only if no command-line option provided
        if use_existing is None:
            use_existing = prompt_yes_no("Features file found. Use existing features.csv?")

        if use_existing:
            print(f"Loading features from {features_path}")
            df = pd.read_csv(features_path)
        else:
            df = generate_and_save_features(features_path)
    else:
        df = generate_and_save_features(features_path)

    results, y_test, feature_cols, scaler, X_test, df_filtered = train_and_evaluate(df, event_type=event_type)
    plot_results(results, y_test, feature_cols, df_filtered, scaler, event_type=event_type)
