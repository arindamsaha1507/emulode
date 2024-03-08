"""Module for Solving ODEs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from emulode.ode import ODE
from emulode.config import Configs
from emulode.qoi import QoI

# from emulode.plotter import Plotter


@dataclass
class Solver(ABC):
    """Base class for the solver."""

    params: dict[str, float]

    parameter_of_interest: str
    result_dimension: int
    quantity_of_interest: Callable[[np.ndarray], float]

    results: np.ndarray = field(init=False, repr=False)

    @abstractmethod
    def solve(self) -> None:
        """Abstract method for solving the system at one point."""

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

    def evaluate_at_point(self, parameter: float) -> float:
        """Evaluate the quantity of interest for the given parameter."""

        self.params[self.parameter_of_interest] = parameter
        self.solve()
        return self.quantity_of_interest(self.results[self.result_dimension, :])


@dataclass
class CommandlineSolver(Solver):
    """Class for solving ODEs from command line arguments."""

    run_command: str
    replacement_prefix: str
    results_file: os.PathLike

    prefix_commands: list[str] = field(default_factory=list)

    def solve(self) -> None:
        """Simulate the system at one point."""

        command = self.run_command
        for param, value in self.params.items():
            command = command.replace(f"{self.replacement_prefix}{param}", str(value))

        for prefix_command in self.prefix_commands[::-1]:
            command = f"{prefix_command}; {command}"

        command = f"bash -c '{command}'"

        print(f"Running command: {command}")
        os.system(command)

        with open(self.results_file, "r", encoding="utf-8") as file:
            self.results = np.array([float(line) for line in file.readlines()])

    def evaluate_at_point(self, parameter: float) -> float:
        """Evaluate the quantity of interest for the given parameter."""

        self.params[self.parameter_of_interest] = parameter
        self.solve()
        return self.quantity_of_interest(self.results[self.result_dimension])


class SolverFactory:
    """Factory class for the solver."""

    @staticmethod
    def create_ode_solver_from_config(ode: ODE, configs: Configs) -> ODESolver:
        """Create a solver from the given configuration."""

        return ODESolver(
            ode.parameters,
            configs.simulator.parameter_of_interest,
            configs.simulator.result_dimension,
            QoI.max_value,
            ode.function,
            configs.solver.initial_conditions,
            configs.solver.t_range,
            configs.solver.n_steps,
            configs.solver.transience,
        )

    @staticmethod
    def create_from_commandline_arguments(configs: Configs) -> CommandlineSolver:
        """Create a solver from the given command line arguments."""

        rum_command = configs.solver.run_command
        params = configs.simulation.parameters
        replacement_prefix = configs.solver.replacement_prefix
        results_file = configs.solver.results_file
        prefix_commands = configs.solver.prefix_commands
        parameter_of_interest = configs.simulator.parameter_of_interest
        result_dimension = configs.simulator.result_dimension

        return CommandlineSolver(
            params,
            parameter_of_interest,
            result_dimension,
            QoI.max_value,
            rum_command,
            replacement_prefix,
            results_file,
            prefix_commands,
        )


def testing() -> None:
    """Test the solver."""

    run_command = "python moving_agents.py --mode $mode --movement_scale_factor $msf --num_runs 100"
    params = {"mode": 1, "msf": 2}
    replacement_prefix = "$"
    results_file = "/home/arindam/moving_agents/result.txt"
    parameter_of_interest = "msf"

    prefix_commands = ["cd ~/moving_agents", "source .venv/bin/activate"]

    solver = CommandlineSolver(
        params,
        parameter_of_interest,
        0,
        QoI.max_value,
        run_command,
        replacement_prefix,
        results_file,
        prefix_commands,
    )

    # solver.solve()
    # print(solver.results)

    print(solver.evaluate_at_point(2.5))

    # solver = CommandlineSolver(
    #     params, run_command, replacement_prefix, results_file, prefix_commands
    # )
    # solver.solve()


if __name__ == "__main__":
    testing()
