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

# import matplotlib.pyplot as plt


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
        configs.plotter.filename,
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

    # plt.plot(emulator.emulator.x_predict, emulator.emulator.y_predict)
    # xx = emulator.emulator.x_predict.flatten()
    # yy = emulator.emulator.y_predict.flatten()
    # yv = emulator.emulator.y_var.flatten()
    # plt.fill_between(xx, yy - yv, yy + yv, color="gray", alpha=0.5)

    # plt.plot(simulator.simulator.xdata, simulator.simulator.ydata, "kx")

    # ideal = SimulatorFactory(solver, configs, ideal=True)

    # plt.plot(ideal.simulator.xdata, ideal.simulator.ydata, "r-")

    # plt.savefig("plots/emulator.png")

    # PlotterFactory(
    #     configs,
    #     solver.solver,
    #     simulator.simulator,
    #     emulator.emulator,
    # )


if __name__ == "__main__":
    main()
