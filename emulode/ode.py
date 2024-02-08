"""Module listing various ODE systems."""

import numpy as np


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


class ODE:
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
        # raise NotImplementedError("This is a work in progress")
        # print("I will come back to this code later")

        Utils.check_parameters(
            params, ["PI", "mu", "beta", "sigma", "gamma", "epsilon", "alpha"]
        )
        Utils.check_dimension(y, 4)
        N = y[0] + y[1] + y[2] + y[3]  # total population
        return np.array(
            [
                params["PI"]
                - params["beta"] * y[1] * y[0] / N
                - params["mu"] * y[0]
                + params["alpha"] * y[3],
                params["beta"] * y[1] * y[0] / N
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
        """SEIR system with density depdendent force of infection, demography, disease induced death and loss of immunity."""
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
    def SIR_SI_VB(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """SIR-SI system with frequency depdendent force of infection, demography, disease induced death. This is minimalist model for west-nile virus spread among birds via mosquito bites"""
        # raise NotImplementedError("This is a work in progress")
        # print("I will come back to this code later")

        Utils.check_parameters(
            params, ["PIB", "PIM", "muB", "muM", "beta", "gamma", "epsilon"]
        )
        Utils.check_dimension(y, 5)
        N = y[0] + y[1] + y[2]  # total population
        return np.array(
            [
                params["PIB"] - params["beta"] * y[4] * y[0] / N - params["muB"] * y[0],
                params["beta"] * y[4] * y[0] / N
                - params["gamma"] * y[1]
                - params["muB"] * y[1],
                params["gamma"] * (1 - params["epsilon"]) * y[2] - params["muB"] * y[2],
                params["PIM"] - params["beta"] * y[1] * y[3] / N - params["muM"] * y[3],
                params["beta"] * y[1] * y[3] / N - params["muM"] * y[1],
            ]
        )

    @staticmethod
    def seir_dens_vacc(t: float, y: np.ndarray, params: dict[str, float]) -> np.ndarray:
        """SEIR system with vaccination, demography, disease induced death and loss of immunity."""
        raise NotImplementedError("This is a work in progress")
        print("I will come back to this code later")

        # Utils.check_parameters(params, ["PI", "beta", "sigma", "gamma", "epsilon", "alpha"])
        # Utils.check_dimension(y, 4)
        # return np.array(
        #     [
        #         params["PI"]-params["beta"] * y[1]*y[0]- params["mu"]*y[0]+params["alpha"]*y[3],
        #         params["beta"] * y[1]*y[0] - -params["sigma"]*y[1]-params["mu"]*y[1],
        #         params["sigma"]*y[1]-params["gamma"]*y[2]-params["mu"]*y[2],
        #         params["gamma"]*(1-params["epsilon"])*y[2] - params["mu"] * y[3]-params["alpha"]*y[3],
        #     ]
        # )


if __name__ == "__main__":
    # print(ODE.lorenz(0, np.array([1, 2, 3]), {"sigma": 10, "rho": 28, "beta": 8 / 3}))
    print(
        ODE.seir_dens_vacc(
            0, np.array([1, 2, 3]), {"sigma": 10, "rho": 28, "beta": 8 / 3}
        )
    )
