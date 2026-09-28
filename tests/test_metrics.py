import numpy as np
import pytest
from compiler_lab.metrics import compare, summarize


def test_compare_rejects_nonfinite_and_shape_mismatch():
    assert not compare([np.nan], [0.0])["passed"]
    assert not compare([np.inf], [np.inf])["passed"]
    assert not compare([[1.0]], [1.0])["passed"]


def test_tolerance_is_elementwise_and_rejects_local_error():
    assert compare([0, 1.00001], [0, 1])["passed"]
    assert not compare([1e-3, 1e6], [0, 1e6])["passed"]


def test_timing_summary_preserves_units():
    assert summarize([1, 2, 3])["p50_ms"] == 2
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize([float("nan")])
