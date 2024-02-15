"""High Level Emulation API"""

from dataclasses import dataclass, field
import os
from emulode.components import (
    EmulatorFactory,
    ODEFactory,
    SimulatorFactory,
    SolverFactory,
)

from emulode.config import Configs
from emulode.solver import Solver
from emulode.simulator import Simulator
from emulode.emulator import Emulator
from emulode.plotter import Plotter


@dataclass
class Emulation:
    """Class for the emulation."""

    configs: Configs
    solver: Solver
    simulator: Simulator
    emulator: Emulator
    ideal: Simulator = field(default=None)

    def plot(self) -> None:
        """Create a combined plot of the emulator and the simulator."""

        Plotter.create_combined_plot(
            self.configs, self.emulator, self.simulator, self.ideal, save=True
        )


class EmulationFactory:
    """Factory class for the Emulation."""

    @staticmethod
    def create(config_file: os.PathLike) -> Emulation:
        """Create an Emulation object."""

        configs = Configs(config_file)

        ode = ODEFactory(configs)
        solver = SolverFactory(ode, configs)
        simulator = SimulatorFactory(solver, configs)
        emulator = EmulatorFactory(simulator, configs)
        ideal = SimulatorFactory(solver, configs, ideal=True)

        return Emulation(
            configs,
            solver.solver,
            simulator.simulator,
            emulator.emulator,
            ideal.simulator,
        )

    @staticmethod
    def create_without_ideal(config_file: os.PathLike) -> Emulation:
        """Create an Emulation object without ideal data."""

        configs = Configs(config_file)

        ode = ODEFactory(configs)
        solver = SolverFactory(ode, configs)
        simulator = SimulatorFactory(solver, configs)
        emulator = EmulatorFactory(simulator, configs)

        return Emulation(
            configs,
            solver.solver,
            simulator.simulator,
            emulator.emulator,
        )
