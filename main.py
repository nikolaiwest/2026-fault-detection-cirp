# from src.anomaly_detection import test_anomaly_detection
from src.binary_classification import test_binary_per_class
from src.data_pipeline import run_data_pipeline


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

    # test_pipeline()
    test_binary_per_class()
    # test_anomaly_detection()
