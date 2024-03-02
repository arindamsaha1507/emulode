"""List of quality of interest (QoI) functions."""

import numpy as np


class QoI:

    @staticmethod
    def max_value(results: np.ndarray) -> float:
        """Return the maximum value."""

        return np.max(results)

    @staticmethod
    def min_value(results: np.ndarray) -> float:
        """Return the minimum value."""

        return np.min(results)

    @staticmethod
    def mean_value(results: np.ndarray) -> float:
        """Return the mean value."""

        return np.mean(results)

    @staticmethod
    def std_dev(results: np.ndarray) -> float:
        """Return the standard deviation."""

        return np.std(results)
