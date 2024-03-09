"""Module listing various ODE systems."""

from dataclasses import dataclass
import numpy as np

from emulode.config import Configs


class ParameterError(Exception):
    """Exception raised when a parameter is missing."""

    def __init__(self, param: str) -> None:
        self.param = param
        super().__init__(f"Missing parameter '{param}'")


class DimensionError(Exception):
    """Exception raised when a dimension is incorrect."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        super().__init__(f"y must be of dimension {dim}")


class Utils:
    """Class listing various utility methods."""

    @staticmethod
    def check_parameters(params: dict[str, float], required_params: list[str]) -> None:
        """Check that the given parameters are valid."""

        for param in required_params:
            if param not in params:
                raise ParameterError(param)

    @staticmethod
    def check_dimension(y: np.ndarray, dim: int) -> None:
        """Check that the given dimension is correct."""

        if y.shape != (dim,):
            raise DimensionError(dim)


class ODEList:
    """Class listing various ODE systems."""

    @staticmethod
    def lorenz(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Lorenz system."""

        # pylint: disable=unused-argument

        Utils.check_parameters(params, ["sigma", "rho", "beta"])
        Utils.check_dimension(y, 3)

        return np.array(
            [
                params["sigma"] * (y[1] - y[0]),
                y[0] * (params["rho"] - y[2]) - y[1],
                y[0] * y[1] - params["beta"] * y[2],
            ]
        )

    @staticmethod
    def rossler(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """Rossler system."""

        # pylint: disable=unused-argument

        Utils.check_parameters(params, ["a", "b", "c"])
        Utils.check_dimension(y, 3)

        return np.array(
            [
                -y[1] - y[2],
                y[0] + params["a"] * y[1],
                params["b"] + y[2] * (y[0] - params["c"]),
            ]
        )

    @staticmethod
    def seir_freq(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """
        SEIR system with frequency depdendent force of infection,
        demography, disease induced death and loss of immunity.
        """

        # pylint: disable=unused-argument

        # raise NotImplementedError("This is a work in progress")
        # print("I will come back to this code later")

        Utils.check_parameters(
            params, ["PI", "mu", "beta", "sigma", "gamma", "epsilon", "alpha"]
        )
        Utils.check_dimension(y, 4)
        total_population = y[0] + y[1] + y[2] + y[3]  # total population
        return np.array(
            [
                params["PI"]
                - params["beta"] * y[1] * y[0] / total_population
                - params["mu"] * y[0]
                + params["alpha"] * y[3],
                params["beta"] * y[1] * y[0] / total_population
                - -params["sigma"] * y[1]
                - params["mu"] * y[1],
                params["sigma"] * y[1] - params["gamma"] * y[2] - params["mu"] * y[2],
                params["gamma"] * (1 - params["epsilon"]) * y[2]
                - params["mu"] * y[3]
                - params["alpha"] * y[3],
            ]
        )

    @staticmethod
    def seir_dens(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """
        SEIR system with density depdendent force of infection,
        demography, disease induced death and loss of immunity.
        """

        # pylint: disable=unused-argument

        # raise NotImplementedError("This is a work in progress")
        # print("I will come back to this code later")

        Utils.check_parameters(
            params, ["PI", "mu", "beta", "sigma", "gamma", "epsilon", "alpha"]
        )
        Utils.check_dimension(y, 4)
        return np.array(
            [
                params["PI"]
                - params["beta"] * y[1] * y[0]
                - params["mu"] * y[0]
                + params["alpha"] * y[3],
                params["beta"] * y[1] * y[0]
                - -params["sigma"] * y[1]
                - params["mu"] * y[1],
                params["sigma"] * y[1] - params["gamma"] * y[2] - params["mu"] * y[2],
                params["gamma"] * (1 - params["epsilon"]) * y[2]
                - params["mu"] * y[3]
                - params["alpha"] * y[3],
            ]
        )

    @staticmethod
    def sir_si_vb(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """
        SIR-SI system with frequency depdendent force of infection, demography,
        disease induced death. This is minimalist model for west-nile virus spread among
        birds via mosquito bites
        """

        # pylint: disable=unused-argument

        Utils.check_parameters(
            params, ["PIB", "PIM", "muB", "muM", "beta", "gamma", "epsilon"]
        )
        Utils.check_dimension(y, 5)
        total_population = y[0] + y[1] + y[2]
        return np.array(
            [
                params["PIB"]
                - params["beta"] * y[4] * y[0] / total_population
                - params["muB"] * y[0],
                params["beta"] * y[4] * y[0] / total_population
                - params["gamma"] * y[1]
                - params["muB"] * y[1],
                params["gamma"] * (1 - params["epsilon"]) * y[2] - params["muB"] * y[2],
                params["PIM"]
                - params["beta"] * y[1] * y[3] / total_population
                - params["muM"] * y[3],
                params["beta"] * y[1] * y[3] / total_population - params["muM"] * y[1],
            ]
        )

    @staticmethod
    def seir_dens_vacc(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """SEIR system with vaccination, demography, disease induced death and loss of immunity."""
        raise NotImplementedError("This is a work in progress")


@dataclass
class ODE:
    """Factory class for the ODE function."""

    function: callable
    parameters: dict[str, float]


class ODEFactory:
    """Factory class for the ODE function."""

    @staticmethod
    def create_from_config(configs: Configs) -> ODE:
        """Create an ODE object from the given configs."""

        function = ODEFactory.ode_function_chooser(configs.simulation.name)
        parameters = configs.simulation.parameters

        return ODE(function, parameters)

    @staticmethod
    def ode_function_chooser(chosen_ode: str) -> callable:
        """Choose the ODE function based on the given string."""
        if chosen_ode == "Rossler":
            return ODEList.rossler
        if chosen_ode == "Lorenz":
            return ODEList.lorenz
        raise ValueError(f"ODE '{chosen_ode}' not found")


if __name__ == "__main__":
    # print(ODE.lorenz(0, np.array([1, 2, 3]), {"sigma": 10, "rho": 28, "beta": 8 / 3}))
    print(
        ODEList.seir_dens_vacc(
            0, np.array([1, 2, 3]), {"sigma": 10, "rho": 28, "beta": 8 / 3}
        )
    )
