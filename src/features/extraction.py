from typing import Union

import numpy as np
from numpy.typing import NDArray


def apply_paa(
    X: NDArray[Union[np.float64, np.int64]],
    n_segments: int = 200,
) -> NDArray[np.float64]:
    """
    Apply Piecewise Aggregate Approximation (PAA).

    Args:
        X: Array of shape (n_samples, n_timepoints)
        n_segments: Number of segments to compress to

    Returns:
        Array of shape (n_samples, n_segments)
    """
    n_samples, n_timepoints = X.shape
    segment_size = n_timepoints // n_segments
    X_reshaped = X[:, : segment_size * n_segments].reshape(
        n_samples, n_segments, segment_size
    )
    return X_reshaped.mean(axis=2)
