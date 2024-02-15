"""Module for the components of the project"""

import argparse
from dataclasses import dataclass, field

from emulode.config import Configs
from emulode.emulator import Emulator

from emulode.ode import ODE
from emulode.simulator import Simulator
from emulode.solver import Solver


@dataclass
class ODEFactory:
    """Factory class for the ODE function."""

    config: Configs

    function: callable = field(init=False)
    parameters: dict = field(init=False)

    def __post_init__(self) -> None:

        self.function = self.ode_function_chooser(self.config.ode.chosen_ode)
        self.parameters = self.config.ode.parameters

    @staticmethod
    def ode_function_chooser(chosen_ode: str) -> callable:
        """Choose the ODE function based on the given string."""
        if chosen_ode == "Rossler":
            return ODE.rossler
        if chosen_ode == "Lorenz":
            return ODE.lorenz
        raise ValueError(f"ODE '{chosen_ode}' not found")


class SolverFactory:
    """Factory class for the solver."""

    @staticmethod
    def create_from_config(ode_factory: ODEFactory, configs: Configs) -> Solver:
        """Create a solver from the given configuration."""

        initial_conditions = configs.solver.initial_conditions
        t_range = tuple(configs.solver.t_range)
        n_steps = configs.solver.n_steps
        transience = configs.solver.transience

        return Solver(
            ode_factory.function,
            ode_factory.parameters,
            initial_conditions,
            t_range,
            n_steps,
            transience,
        )

    @staticmethod
    def create_from_commandline_arguments(
        ode_factory: ODEFactory, args: argparse.Namespace
    ) -> Solver:
        """Create a solver from the given command line arguments."""

        raise NotImplementedError("Command line arguments not supported yet")


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


class EmulatorFactory:
    """Factory class for the emulator."""

    simulator: Simulator
    configs: Configs

    @staticmethod
    def create_from_config(simulator: Simulator, configs: Configs) -> Emulator:
        """Create an emulator from the given configuration."""

        n_layers = configs.emulator.n_layers
        n_predict = configs.emulator.n_prediction_points
        n_iterations = configs.emulator.n_iterations

        return Emulator(
            simulator.xdata, simulator.ydata, n_layers, n_predict, n_iterations
        )

    @staticmethod
    def create_from_commandline_arguments(
        simulator: Simulator, args: argparse.Namespace
    ) -> Emulator:
        """Create an emulator from the given command line arguments."""

        raise NotImplementedError("Command line arguments not supported yet")
