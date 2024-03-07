"""Module for Solving ODEs."""

from abc import ABC, abstractmethod
import argparse
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from emulode.ode import ODE
from emulode.config import Configs

# from emulode.plotter import Plotter


@dataclass
class Solver(ABC):
    """Base class for the solver."""

    params: dict[str, float]

    @abstractmethod
    def solve(self) -> None:
        """Abstract method for solving the system at one point."""

    @abstractmethod
    def set_varying_settings(
        self, parameter: str, qoi: Callable = None, result_dim: int = None
    ) -> None:
        """Abstract method for setting the parameter and quantity of interest."""

    @abstractmethod
    def evaluate_at_point(self, parameter: float) -> float:
        """Abstract method for evaluating the quantity of interest for the given parameter."""


@dataclass
class ODESolver(Solver):
    """Class for solving ODEs."""

    # pylint: disable=too-many-instance-attributes

    ode: Callable[[float, np.ndarray, dict[str, float]], np.ndarray]
    initial_conditions: np.ndarray
    t_span: tuple[float, float]
    t_steps: int
    transience: int | float

    results: np.ndarray = field(init=False, repr=False)

    parameter_of_interest: str = field(init=False, repr=False)
    result_dimension: int = field(init=False, repr=False)
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

    def set_varying_settings(
        self, parameter: str, qoi: Callable = None, result_dim: int = None
    ) -> None:
        """Set the parameter and quantity of interest."""

        if parameter not in self.params:
            raise ValueError(f"Parameter '{parameter}' not found")

        self.parameter_of_interest = parameter
        self.quantity_of_interest = qoi
        self.result_dimension = result_dim

    def evaluate_at_point(self, parameter: float) -> float:
        """Evaluate the quantity of interest for the given parameter."""

        if self.parameter_of_interest is None or self.quantity_of_interest is None:
            raise ValueError("Parameter and quantity of interest not set")

        self.params[self.parameter_of_interest] = parameter

        self.solve()
        return self.quantity_of_interest(self.results[self.result_dimension, :])


class SolverFactory:
    """Factory class for the solver."""

    @staticmethod
    def create_ode_solver_from_config(ode_factory: ODE, configs: Configs) -> ODESolver:
        """Create a solver from the given configuration."""

        initial_conditions = configs.solver.initial_conditions
        t_range = tuple(configs.solver.t_range)
        n_steps = configs.solver.n_steps
        transience = configs.solver.transience

        return ODESolver(
            ode_factory.parameters,
            ode_factory.function,
            initial_conditions,
            t_range,
            n_steps,
            transience,
        )

    @staticmethod
    def create_from_commandline_arguments(
        ode_factory: ODE, args: argparse.Namespace
    ) -> Solver:
        """Create a solver from the given command line arguments."""

        raise NotImplementedError("Command line arguments not supported yet")
