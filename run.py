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

    ideal = SimulatorFactory(solver, configs, ideal=True)

    Plotter.create_combined_plot(
        f"{configs.plotter.directory}/{configs.plotter.filename}",
        configs.plotter.x_label,
        configs.plotter.y_label,
        simulator.simulator.xdata,
        simulator.simulator.ydata,
        emulator.emulator.x_predict,
        emulator.emulator.y_predict,
        emulator.emulator.y_var,
        ideal.simulator.xdata,
        ideal.simulator.ydata,
    )


if __name__ == "__main__":
    main()
