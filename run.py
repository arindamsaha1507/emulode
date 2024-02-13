"""Main module for the application."""

import os

import yaml

import numpy as np

from emulode.solver import Solver
from emulode.emulator import Emulator
from emulode.ode import ODE
from emulode.plotter import Plotter
from emulode.simulator import Simulator
from emulode.config import Configs


def trial() -> None:
    """Trial function for the application."""

    print(Configs(os.path.join(os.getcwd(), "config.yml")))


def main() -> None:
    """Main function for the application."""

    # pylint: disable=too-many-locals

    with open("config.yml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    chosen_ode = config["ode"]["chosen_ode"]

    if chosen_ode == "Rossler":
        ode_func = ODE.rossler
    elif chosen_ode == "Lorenz":
        ode_func = ODE.lorenz
    else:
        raise ValueError(f"ODE '{chosen_ode}' not found")

    params = config["ode"]["parameters"][chosen_ode]
    init_conds = np.array(config["solver"]["initial_conditions"])
    t_span = tuple(config["solver"]["t_range"])
    t_steps = config["solver"]["n_steps"]
    transience = config["solver"]["transience"]

    solver = Solver(ode_func, params, init_conds, t_span, t_steps, transience)
    print(solver)

    parameter_of_interest = config["simulator"]["parameter of interest"]
    component_of_interest = config["simulator"]["component of interest"]
    range_of_interest = tuple(config["simulator"]["range"])
    n_points = config["simulator"]["n_simulation_points"]

    simulator = Simulator(
        solver,
        parameter_of_interest,
        range_of_interest[0],
        range_of_interest[1],
        n_points,
        component_of_interest=component_of_interest,
    )

    # simulator = Simulator(solver, "c", 5, 10, 10, lambda x: np.max(x[0, :]))

    print(simulator)

    n_layers = config["emulator"]["n_layers"]
    n_predict = config["emulator"]["n_prediction_points"]
    n_iterations = config["emulator"]["n_iterations"]

    emulator = Emulator(
        simulator.xdata, simulator.ydata, n_layers, n_predict, n_iterations
    )

    print(emulator)

    directory = config["plotter"]["directory"]
    filename = config["plotter"]["filename"]
    xlabel = config["plotter"]["x_label"]
    ylabel = config["plotter"]["y_label"]

    scale = (t_span[1] - t_span[0]) * (1 - transience)

    Plotter.create_combined_plot(
        f"{directory}/{filename}",
        xlabel,
        ylabel,
        simulator.xdata,
        simulator.ydata,
        emulator.x_predict,
        emulator.y_predict,
        emulator.y_var,
        scale=scale,
        x_ideal=np.linspace(0, scale, len(solver.results[0, :])),
        y_ideal=solver.results[component_of_interest, :],
    )


if __name__ == "__main__":
    # main()
    trial()
