from src.data_pipeline import run_data_pipeline

if __name__ == "__main__":

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
