from src.data_pipeline import run_data_pipeline


def test_pipeline_runs_successfully():
    """Test that pipeline completes and returns expected structure."""

    # Run pipeline
    data = run_data_pipeline(
        force_reload=False,
        keep_exceptions=False,
        target_ok_ratio=0.9,
    )

    # Basic assertions
    assert "torque_values" in data
    assert "labels" in data
    assert "label_mapping" in data

    # Check we have data
    assert len(data["torque_values"]) > 0
    assert len(data["labels"]) > 0

    # Check label 0 exists (normal class)
    assert 0 in data["labels"]

    print(f"Pipeline test passed: {len(data['labels'])} samples")
