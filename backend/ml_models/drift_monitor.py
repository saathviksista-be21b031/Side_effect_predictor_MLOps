# backend/ml_models/drift_monitor.py
import pickle
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import mean_squared_error  # kept for structure (unused)
from prometheus_client import Counter

drift_alert_counter = Counter('data_drift_alerts', 'Counts the number of data drift alerts detected')

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_mmd(X, Y, kernel="rbf"):
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions
    using the given kernel.
    """
    XX = pairwise_kernels(X, X, metric=kernel)
    YY = pairwise_kernels(Y, Y, metric=kernel)
    XY = pairwise_kernels(X, Y, metric=kernel)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def detect_drift(baseline_data, live_data, mmd_threshold=0.1):
    baseline_data = np.asarray(baseline_data)
    live_data = np.asarray(live_data)

    if baseline_data.shape[1] != live_data.shape[1]:
        raise ValueError(f"[Drift Monitor] Feature mismatch: baseline has {baseline_data.shape[1]}, live has {live_data.shape[1]}")

    drifted_columns = 0
    total_columns = baseline_data.shape[1]

    for i in range(total_columns):
        base_col = baseline_data[:, i].reshape(-1, 1)
        live_col = live_data[:, i].reshape(-1, 1)

        # Remove NaNs
        base_col = base_col[~np.isnan(base_col).squeeze()]
        live_col = live_col[~np.isnan(live_col).squeeze()]

        # Reshape again after filtering
        base_col = base_col.reshape(-1, 1)
        live_col = live_col.reshape(-1, 1)

        if len(base_col) < 2 or len(live_col) < 2:
            continue

        mmd = compute_mmd(base_col, live_col)
        if mmd > mmd_threshold:
            drifted_columns += 1

    drift_ratio = drifted_columns / total_columns
    print(f"[Drift Monitor] {drifted_columns}/{total_columns} columns drifted (ratio = {drift_ratio:.2f})")

    return drift_ratio > 0.2  # same threshold as before

def monitor_and_react(live_data, baseline_data):
    drift = detect_drift(baseline_data, live_data)
    if drift:
        print("[Drift Monitor] Drift detected! Alerting Prometheus...")
        drift_alert_counter.inc()
    else:
        print("[Drift Monitor] No significant drift detected.")
    return drift

if __name__ == "__main__":
    BASELINE_PATH = "backend/target/temp_targets/X_train_master.pkl"
    LIVE_PATH = "backend/target/inputted_features_2.pkl"
    STATUS_PATH = "backend/target/drift_status.txt"

    baseline = load_pickle(BASELINE_PATH)
    live = load_pickle(LIVE_PATH)

    if hasattr(baseline, "values"):
        baseline = baseline.values
    if hasattr(live, "values"):
        live = live.values

    drift_detected = monitor_and_react(live, baseline)

    with open(STATUS_PATH, "w") as f:
        f.write("drift_detected\n" if drift_detected else "no_drift\n")
