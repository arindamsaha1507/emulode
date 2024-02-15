"""High Level Emulation API"""

from dataclasses import dataclass, field
import os
from typing import Optional

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
    """
    A top level class to store all information about an emulation run.

    Note that since the `ideal` parameter is useful when comparing the
    emulator results with the simulator for all points. Providing such
    results may not be always possible. Therefore, the `ideal` parameter
    is optional.

    Args:
        configs: The configuration yml file
        solver: The solver
        simulator: The simulator containing experimental data
        emulator: The emulator
        ideal: The simulator containing ideal data (for all emulated points)

    """

    configs: Configs
    solver: Solver
    simulator: Simulator
    emulator: Emulator
    ideal: Optional[Simulator] = field(default=None)

    def plot(self) -> None:
        """
        Create a combined plot containing the `simulator`, `emulator` and
        `ideal` data (if provided) and saves it to the file specified in the
        `configs` parameter.
        """

        Plotter.create_combined_plot(
            self.configs, self.emulator, self.simulator, self.ideal, save=True
        )


class EmulationFactory:
    """
    Factory class for the creating the Emulator object.

    Currrntly, only yml configuration files are supported. Future versions
    may support JSON and other configuration files.
    """

    @staticmethod
    def create_from_yml_file(config_file: os.PathLike, ideal_run: bool) -> Emulation:
        """
        Create an `Emulation` object from a yml configuration file.

        Args:
            config_file: The configuration yml file
            ideal_run: Whether to include ideal results

        """

        configs = Configs(config_file)

        ode = ODEFactory(configs)
        solver = SolverFactory(ode, configs)
        simulator = SimulatorFactory(solver, configs)
        emulator = EmulatorFactory(simulator, configs)

        if ideal_run:
            ideal = SimulatorFactory(solver, configs, ideal=True)
        else:
            ideal = None

        return Emulation(
            configs,
            solver.solver,
            simulator.simulator,
            emulator.emulator,
            ideal.simulator,
        )

    @staticmethod
    def create_from_json_file(config_file: os.PathLike) -> Emulation:
        """
        Create an `Emulation` object from a json configuration file
        (in future).
        """

        raise NotImplementedError("JSON configuration files not supported yet")
