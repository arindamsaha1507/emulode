"""Module for the components of the project"""

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

    # pylint: disable=too-many-instance-attributes

    solver_factory: SolverFactory
    configs: Configs
    ideal: bool = False

    parameter_of_interest: str = field(init=False)
    range_of_interest: tuple[float, float] = field(init=False)
    n_points: int = field(init=False)
    component_of_interest: int = field(init=False)

    simulator: Simulator = field(init=False)

    def __post_init__(self) -> None:

        self.parameter_of_interest = self.configs.simulator.parameter_of_interest
        self.range_of_interest = tuple(self.configs.simulator.range)

        if self.ideal:
            self.n_points = self.configs.emulator.n_prediction_points
        else:
            self.n_points = self.configs.simulator.n_simulation_points

        self.component_of_interest = self.configs.simulator.component_of_interest

        self.simulator = Simulator(
            self.solver_factory.solver,
            self.parameter_of_interest,
            self.range_of_interest[0],
            self.range_of_interest[1],
            self.n_points,
            lambda x: max(x[0, :]),
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
