import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.config_loader import load_class_config
from src.data_pipeline import run_data_pipeline


def apply_paa(X, n_segments=200):
    """Simple PAA compression."""
    n_samples, n_timepoints = X.shape
    segment_size = n_timepoints // n_segments
    X_reshaped = X[:, : segment_size * n_segments].reshape(
        n_samples, n_segments, segment_size
    )
    return X_reshaped.mean(axis=2)


def test_binary_per_class():
    """Test binary classification: Normal vs. each fault class with 5 classifiers."""

    print("=" * 70)
    print("Binary Classification Test: Normal vs. Each Fault")
    print("Testing 5 sklearn classifiers")
    print("=" * 70)

    # Define 5 classifiers
    classifiers = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": SVC(
            kernel="rbf",
            random_state=42,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
        ),
    }

    # Load data WITHOUT upsampling
    print("\n[1/3] Loading data...")
    data = run_data_pipeline(
        force_reload=False,
        target_ok_ratio=0.5,
        classes_to_keep=load_class_config("all"),
    )

    X = np.array(data["torque_values"])
    y = np.array(data["labels"])
    label_mapping = data["label_mapping"]

    # Reverse mapping: int → string
    int_to_label = {v: k for k, v in label_mapping.items()}

    # Apply PAA
    print("\n[2/3] Applying PAA...")
    X_paa = apply_paa(X, n_segments=200)
    print(f"Data shape: {X_paa.shape}")
    print(f"Total samples: {len(y)}")

    # Get normal samples
    normal_mask = y == 0
    X_normal = X_paa[normal_mask]
    y_normal = np.zeros(len(X_normal))

    print(f"Normal samples: {len(X_normal)}")

    # Test each fault class
    print("\n[3/3] Testing each fault class with 5 classifiers...")
    print("=" * 70)

    results = []

    for fault_label in sorted(set(y)):
        if fault_label == 0:
            continue  # Skip normal class

        fault_name = int_to_label[fault_label]

        # Get this fault class
        fault_mask = y == fault_label
        X_fault = X_paa[fault_mask]
        y_fault = np.ones(len(X_fault))

        n_fault = len(X_fault)

        if n_fault < 10:
            print(f"\n{fault_name}: SKIPPED (only {n_fault} samples)")
            continue

        # Combine normal + this fault
        X_binary = np.vstack([X_normal, X_fault])
        y_binary = np.hstack([y_normal, y_fault])

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
        )

        print(f"\n{fault_name} ({n_fault} samples)")
        print("-" * 50)

        # Test all 5 classifiers
        fault_results = {
            "fault": fault_name,
            "n_samples": n_fault,
        }

        for clf_name, clf in classifiers.items():
            # Train
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)

            # Metrics
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred)

            fault_results[f"{clf_name}_f1"] = f1
            fault_results[f"{clf_name}_precision"] = precision
            fault_results[f"{clf_name}_recall"] = recall

            print(
                f"  {clf_name:18s} | F1: {f1:.3f} | P: {precision:.3f} | R: {recall:.3f}"
            )

        # Calculate average across all classifiers
        avg_f1 = np.mean([fault_results[f"{clf}_f1"] for clf in classifiers.keys()])
        fault_results["avg_f1"] = avg_f1

        print(f"  {'AVERAGE':18s} | F1: {avg_f1:.3f}")

        results.append(fault_results)

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Detectability Ranking (by Average F1)")
    print("=" * 70)

    results_sorted = sorted(results, key=lambda x: x["avg_f1"], reverse=True)

    print(f"\n{'Fault Class':<35} | Samples | Avg F1 | Detectable?")
    print("-" * 70)

    for r in results_sorted:
        detectable = (
            "YES" if r["avg_f1"] > 0.7 else "KIND OF" if r["avg_f1"] > 0.4 else "NO"
        )
        print(
            f"{r['fault']:<35} | {r['n_samples']:4d}    | {r['avg_f1']:.3f}  | {detectable}"
        )

    # Overall stats
    print("\n" + "=" * 70)
    print("Classifier Performance Summary")
    print("=" * 70)

    for clf_name in classifiers.keys():
        clf_avg_f1 = np.mean([r[f"{clf_name}_f1"] for r in results])
        print(f"{clf_name:18s} | Average F1: {clf_avg_f1:.3f}")

    overall_avg = np.mean([r["avg_f1"] for r in results])
    print(f"\n{'OVERALL AVERAGE':18s} | F1: {overall_avg:.3f}")


if __name__ == "__main__":
    test_binary_per_class()
