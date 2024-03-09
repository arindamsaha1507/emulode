"""Module for testing the utils class."""

import numpy as np
import pytest
from redundant.ode import DimensionError, ParameterError, Utils


def test_check_parameters_passing():
    """Test the check_parameters method with passing parameters."""

    params = {"param1": 1.0, "param2": 2.0}
    required_params = ["param1", "param2"]
    Utils.check_parameters(params, required_params)


def test_check_parameters_failing():
    """Test the check_parameters method with failing parameters."""

    params = {"param1": 1.0}
    required_params = ["param1", "param2"]
    # Should raise a MissingParameterError
    with pytest.raises(ParameterError):
        Utils.check_parameters(params, required_params)


def test_check_dimension_passing():
    """Test the check_dimension method with passing dimension."""

    y = np.array([1.0, 2.0, 3.0])
    dim = 3
    Utils.check_dimension(y, dim)


def test_check_dimension_failing():
    """Test the check_dimension method with failing dimension."""

    y = np.array([1.0, 2.0, 3.0])
    dim = 2
    # Should raise a DimensionError
    with pytest.raises(DimensionError):
        Utils.check_dimension(y, dim)
