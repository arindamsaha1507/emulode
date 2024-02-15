"""Main module for the application."""

import os

# import numpy as np

from emulode.config import Configs
from emulode.components import (
    ODEFactory,
    SolverFactory,
    SimulatorFactory,
    EmulatorFactory,
)

from emulode.plotter import Plotter


def main() -> None:
    """Main function for the application."""

    # pylint: disable=too-many-locals

    configs = Configs(os.path.join(os.getcwd(), "config.yml"))

    ode = ODEFactory(configs)
    solver = SolverFactory(ode, configs)
    simulator = SimulatorFactory(solver, configs)
    emulator = EmulatorFactory(simulator, configs)

    ideal = SimulatorFactory(solver, configs, ideal=True)

    Plotter.create_combined_plot(
        configs, emulator.emulator, simulator.simulator, ideal.simulator, save=True
    )


if __name__ == "__main__":
    main()
