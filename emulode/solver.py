"""Module for Solving ODEs."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from emulode.plotter import Plotter


@dataclass
class Solver:
    """Class for solving ODEs."""

    # pylint: disable=too-many-instance-attributes

    ode: Callable[[float, np.ndarray, dict[str, float]], np.ndarray]
    params: dict[str, float]
    initial_conditions: np.ndarray
    t_span: tuple[float, float]
    t_steps: int
    transience: int | float

    results: np.ndarray = field(init=False, repr=False)

    parameter_of_interest: str = field(init=False, repr=False)
    component_of_interest: int = field(init=False, repr=False)
    quantity_of_interest: Callable[[np.ndarray], float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check that the given parameters are valid."""

        if self.transience < 0:
            raise ValueError("Transience must be non-negative")

        if self.t_steps <= 0:
            raise ValueError("t_steps must be positive")

        if self.transience < 1:
            self.transience = int(self.transience * self.t_steps)

        if self.transience >= self.t_steps:
            raise ValueError("Transience must be less than t_steps")

        if len(self.t_span) != 2:
            raise ValueError("t_span must be a tuple of length 2")

        if self.t_span[0] >= self.t_span[1]:
            raise ValueError("t_span must be increasing")

    @property
    def t_initial(self) -> float:
        """Return the initial time."""
        return self.t_span[0]

    @property
    def t_final(self) -> float:
        """Return the final time."""
        return self.t_span[1]

    def solve(self) -> None:
        """Solve the ODE."""

        sol = solve_ivp(
            self.ode,
            self.t_span,
            self.initial_conditions,
            t_eval=np.linspace(self.t_initial, self.t_final, self.t_steps),
            args=(self.params,),
        )

        self.results = sol.y[:, self.transience :]

    def set_varying_settings(self, parameter: str, qoi: Callable = None) -> None:
        """Set the parameter and quantity of interest."""

        if parameter not in self.params:
            raise ValueError(f"Parameter '{parameter}' not found")

        else:
            self.parameter_of_interest = parameter
            self.quantity_of_interest = qoi

    def evaluate_at_point(self, parameter: float) -> float:
        """Evaluate the quantity of interest for the given parameter."""

        if self.parameter_of_interest is None or self.quantity_of_interest is None:
            raise ValueError("Parameter and quantity of interest not set")

        self.params[self.parameter_of_interest] = parameter

        self.solve()
        return self.quantity_of_interest(self.results)

    def phase_plot(
        self, components: tuple[int, int], filename: str = "plots/phase.png"
    ) -> None:
        """Plot the results."""
        Plotter.create_basic_plot(
            self.results[components[0], :],
            self.results[components[1], :],
            filename=filename,
        )

    def timeseries_plot(
        self, component: int, filename: str = "plots/timeseries.png"
    ) -> None:
        """Plot the results."""
        time = np.linspace(self.t_initial, self.t_final, self.t_steps)
        time = time[: len(self.results[component, :])]
        Plotter.create_basic_plot(
            time,
            self.results[component, :],
            filename=filename,
        )
