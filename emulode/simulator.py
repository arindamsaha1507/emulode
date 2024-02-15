"""Module for the simulator class."""

import argparse
import time

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from emulode.config import Configs

from emulode.solver import Solver


@dataclass
class Simulator:
    """Class for the simulator."""

    # pylint: disable=too-many-instance-attributes

    solver: Solver
    varying_parameter: str
    parameter_start: float
    parameter_end: float
    num_points: int
    function_of_interest: Callable[[np.ndarray], float] = field(default=None)

    xdata: np.ndarray = field(init=False, repr=False)
    ydata: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check that the given parameters are valid."""

        if self.parameter_start >= self.parameter_end:
            raise ValueError("parameter_start must be less than parameter_end")

        if self.num_points <= 0:
            raise ValueError("num_points must be positive")

        if (
            self.varying_parameter not in self.solver.params
            and self.varying_parameter != "t"
        ):
            raise ValueError("varying_parameter must be a parameter of the ODE or 't'")

        self.solver.set_varying_settings(
            self.varying_parameter, self.function_of_interest
        )

        self.xdata, self.ydata = self.create_data()

        self.xdata = self.xdata[:, None]
        self.ydata = self.ydata.reshape(-1, 1)

    def create_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Create data for the given solver."""

        time_start = time.time()

        x = np.linspace(self.parameter_start, self.parameter_end, self.num_points)
        y = np.array([self.solver.evaluate_at_point(xi) for xi in x])

        time_end = time.time()

        print(f"Time taken: {time_end - time_start:.2f} seconds")

        return x, y


class SimulatorFactory:
    """Factory class for the simulator."""

    @staticmethod
    def create_from_config(
        solver: Solver, configs: Configs, ideal: bool = False
    ) -> Simulator:
        """Create a simulator from the given configuration."""

        parameter_of_interest = configs.simulator.parameter_of_interest
        range_of_interest = tuple(configs.simulator.range)

        if ideal:
            n_points = configs.emulator.n_prediction_points
        else:
            n_points = configs.simulator.n_simulation_points

        return Simulator(
            solver,
            parameter_of_interest,
            range_of_interest[0],
            range_of_interest[1],
            n_points,
            lambda x: max(x[0, :]),
        )

    @staticmethod
    def create_from_commandline_arguments(
        solver: Solver, args: argparse.Namespace
    ) -> Simulator:
        """Create a simulator from the given command line arguments."""

        raise NotImplementedError("Command line arguments not supported yet")
