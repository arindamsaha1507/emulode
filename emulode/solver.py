"""Module for Solving ODEs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import Callable

import numpy as np

from emulode.config import Configs
from emulode.qoi import QoI


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
    def create_from_commandline_arguments(configs: Configs) -> CommandlineSolver:
        """Create a solver from the given command line arguments."""

        rum_command = configs.solver.run_command
        params = configs.solver.parameters
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

    @staticmethod
    def create_for_hpc_simulation() -> Solver:
        """Create a solver for HPC simulation."""

        raise NotImplementedError("Solver for HPC simulation is not implemented yet.")
