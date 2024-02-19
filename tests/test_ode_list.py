"""Module for testing the ODEList class."""

import numpy as np

from emulode.ode import ODEList


def test_lorenz_passing():
    """Test the lorenz method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0])
    params = {"sigma": 10.0, "rho": 28.0, "beta": 8 / 3}
    ODEList.lorenz(t, y, params)


def test_rossler_passing():
    """Test the roessler method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0])
    params = {"a": 0.2, "b": 0.2, "c": 5.7}
    ODEList.rossler(t, y, params)


def test_seir_freq_passing():
    """Test the seir_freq method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0, 4.0])
    params = {
        "PI": 0.1,
        "mu": 0.2,
        "beta": 0.2,
        "sigma": 0.2,
        "gamma": 0.2,
        "epsilon": 0.2,
        "alpha": 0.5,
    }
    ODEList.seir_freq(t, y, params)


def test_seir_density_passing():
    """Test the seir_density method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0, 4.0])
    params = {
        "PI": 0.1,
        "mu": 0.2,
        "beta": 0.2,
        "sigma": 0.2,
        "gamma": 0.2,
        "epsilon": 0.2,
        "alpha": 0.5,
    }
    ODEList.seir_dens(t, y, params)


def test_sir_si_vb_passing():
    """Test the sir_si_vb method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    params = {
        "PIB": 0.1,
        "PIM": 0.2,
        "muB": 0.2,
        "muM": 0.2,
        "beta": 0.2,
        "sigma": 0.2,
        "gamma": 0.2,
        "epsilon": 0.2,
    }
    ODEList.sir_si_vb(t, y, params)


def test_sier_dens_vacc_not_implemented():
    """Test the seir_dens_vacc method with passing parameters."""

    t = 0.0
    y = np.array([1.0, 2.0, 3.0, 4.0])
    params = {
        "PI": 0.1,
        "mu": 0.2,
        "beta": 0.2,
        "sigma": 0.2,
        "gamma": 0.2,
        "epsilon": 0.2,
        "alpha": 0.5,
    }
    # Should raise a NotImplementedError
    with np.testing.assert_raises(NotImplementedError):
        ODEList.seir_dens_vacc(t, y, params)
