from src.benchmark import test_anomaly_detection, test_binary_per_class
from src.data import run_data_pipeline
from src.methodology import apply


def test_pipeline():
    # Simple test for the pipeline with defaults
    print("Testing data pipeline...")

    data = run_data_pipeline(
        force_reload=False,
        keep_exceptions=False,
        # classes_to_keep=[],  # Should use CLASSES_TO_KEEP default
        target_ok_ratio=0.99,
    )

    print("\n" + "=" * 60)
    print("Pipeline test complete!")
    print("=" * 60)


if __name__ == "__main__":

    # Temporary tests during development
    apply()
    test_pipeline()
    test_binary_per_class()
    test_anomaly_detection()
