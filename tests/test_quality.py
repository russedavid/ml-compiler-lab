import numpy as np
from compiler_lab.quality import evaluate


def test_slice_report_retains_classification_regression():
    reference = lambda x: x
    candidate = lambda x: -x
    inputs = np.array([[2, 1], [1, 2]], np.float32)
    result = evaluate(
        reference, candidate, inputs, labels=[0, 1], groups=["light", "dark"]
    )
    assert result["summary"]["reference_accuracy"] == 1
    assert result["summary"]["candidate_accuracy"] == 0
    assert result["slices"]["light"]["numerical_passes"] == 0


def test_nonfinite_candidate_fails_without_hiding_it():
    result = evaluate(lambda x: x, lambda x: np.array([np.nan]), [[1]])
    assert result["summary"]["numerical_passes"] == 0
    assert result["cases"][0]["numerical"]["reason"] == "nonfinite output"
