"""Module for the simulator class."""

import time

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from emulode.solver import Solver


@dataclass
class Simulator:
    """Class for the simulator."""

    solver: Solver
    varying_parameter: str
    parameter_start: float
    parameter_end: float
    num_points: int
    function_of_interest: Callable[[np.ndarray], float]

    xdata: np.ndarray = field(init=False)
    ydata: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Check that the given parameters are valid."""

        if self.parameter_start >= self.parameter_end:
            raise ValueError("parameter_start must be less than parameter_end")

        if self.num_points <= 0:
            raise ValueError("num_points must be positive")

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