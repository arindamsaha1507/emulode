"""Main module for the application."""

import os

# import numpy as np

from emulode.config import Configs
from emulode.components import (
    ODEFactory,
    SolverFactory,
    SimulatorFactory,
    EmulatorFactory,
    PlotterFactory,
)


def trial() -> None:
    """Trial function for the application."""

    print(Configs(os.path.join(os.getcwd(), "config.yml")))


def main() -> None:
    """Main function for the application."""

    # pylint: disable=too-many-locals

    configs = Configs(os.path.join(os.getcwd(), "config.yml"))

    ode = ODEFactory(configs)
    solver = SolverFactory(ode, configs)
    simulator = SimulatorFactory(solver, configs)
    emulator = EmulatorFactory(simulator, configs)
    PlotterFactory(
        configs,
        solver.solver,
        simulator.simulator,
        emulator.emulator,
    )


if __name__ == "__main__":
    main()
