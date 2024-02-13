"""Module for the components of the project"""

from dataclasses import dataclass, field

import numpy as np
from emulode.config import Configs
from emulode.emulator import Emulator

from emulode.ode import ODE
from emulode.plotter import Plotter
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


@dataclass
class SolverFactory:
    """Factory class for the solver."""

    ode_factory: ODEFactory
    configs: Configs

    initial_conditions: list[float] = field(init=False)
    t_range: tuple[float, float] = field(init=False)
    n_steps: int = field(init=False)
    transience: int = field(init=False)

    solver: Solver = field(init=False)

    def __post_init__(self) -> None:

        self.initial_conditions = self.configs.solver.initial_conditions
        self.t_range = tuple(self.configs.solver.t_range)
        self.n_steps = self.configs.solver.n_steps
        self.transience = self.configs.solver.transience

        self.solver = Solver(
            self.ode_factory.function,
            self.ode_factory.parameters,
            self.initial_conditions,
            self.t_range,
            self.n_steps,
            self.transience,
        )


@dataclass
class SimulatorFactory:
    """Factory class for the simulator."""

    solver_factory: SolverFactory
    configs: Configs

    parameter_of_interest: str = field(init=False)
    range_of_interest: tuple[float, float] = field(init=False)
    n_points: int = field(init=False)
    component_of_interest: int = field(init=False)

    simulator: Simulator = field(init=False)

    def __post_init__(self) -> None:

        self.parameter_of_interest = self.configs.simulator.parameter_of_interest
        self.range_of_interest = tuple(self.configs.simulator.range)
        self.n_points = self.configs.simulator.n_simulation_points
        self.component_of_interest = self.configs.simulator.component_of_interest

        self.simulator = Simulator(
            self.solver_factory.solver,
            self.parameter_of_interest,
            self.range_of_interest[0],
            self.range_of_interest[1],
            self.n_points,
            component_of_interest=self.component_of_interest,
        )


@dataclass
class EmulatorFactory:
    """Factory class for the emulator."""

    simulator_factory: SimulatorFactory
    configs: Configs

    n_layers: int = field(init=False)
    n_predict: int = field(init=False)
    n_iterations: int = field(init=False)

    emulator: Emulator = field(init=False)

    def __post_init__(self) -> None:

        self.n_layers = self.configs.emulator.n_layers
        self.n_predict = self.configs.emulator.n_prediction_points
        self.n_iterations = self.configs.emulator.n_iterations

        self.emulator = Emulator(
            self.simulator_factory.simulator.xdata,
            self.simulator_factory.simulator.ydata,
            self.n_layers,
            self.n_predict,
            self.n_iterations,
        )


@dataclass
class PlotterFactory:
    """Factory class for the plotter."""

    # pylint: disable=too-many-instance-attributes

    configs: Configs
    solver: Solver
    simulator: Simulator
    emulator: Emulator

    directory: str = field(init=False)
    filename: str = field(init=False)
    xlabel: str = field(init=False)
    ylabel: str = field(init=False)
    scale: float = field(init=False)
    x_ideal: np.ndarray = field(init=False)
    y_ideal: np.ndarray = field(init=False)

    plotter: Plotter = field(init=False)

    def __post_init__(self) -> None:

        self.directory = self.configs.plotter.directory
        self.filename = self.configs.plotter.filename
        self.xlabel = self.configs.plotter.x_label
        self.ylabel = self.configs.plotter.y_label

        self.scale = (self.configs.solver.t_final - self.configs.solver.t_initial) * (
            1 - self.configs.solver.transience
        )

        self.x_ideal = np.linspace(0, self.scale, len(self.solver.results[0, :]))
        self.y_ideal = self.solver.results[
            self.configs.simulator.component_of_interest, :
        ]

        Plotter.create_combined_plot(
            f"{self.directory}/{self.filename}",
            self.xlabel,
            self.ylabel,
            self.simulator.xdata,
            self.simulator.ydata,
            self.emulator.x_predict,
            self.emulator.y_predict,
            self.emulator.y_var,
            scale=self.scale,
            x_ideal=self.x_ideal,
            y_ideal=self.y_ideal,
        )
