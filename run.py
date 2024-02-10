"""Main module for the application."""

import yaml

import numpy as np

from emulode.solver import Solver
from emulode.emulator import Emulator
from emulode.ode import ODE
from emulode.plotter import Plotter
from emulode.simulator import Simulator


def run() -> None:
    """Run the application."""

    solver = Solver(
        ODE.rossler,
        {"a": 0.2, "b": 0.2, "c": 5.7},
        np.array([1, 1, 1]),
        (0, 100),
        1000,
        0.8,
    )

    # solver = Solver(
    #     ODE.lorenz,
    #     {"sigma": 10, "rho": 28, "beta": 8 / 3},
    #     np.array([1, 2, 3]),
    #     (0, 100),
    #     1000,
    #     0.1,
    # )

    solver.solve()
    solver.phase_plot(components=(0, 1))
    solver.timeseries_plot(component=0)

    simulator = Simulator(solver, "t", 0, 1, 20, component_of_interest=0)
    # simulator = Simulator(solver, "c", 5, 10, 10, lambda x: np.max(x[0, :]))

    emulator = Emulator(simulator.xdata, simulator.ydata, 3, 1000, 500)

    Plotter.create_combined_plot(
        "plots/emulator.png",
        "time",
        "x",
        simulator.xdata,
        simulator.ydata,
        emulator.x_predict,
        emulator.y_predict,
        emulator.y_var,
        scale=20,
        x_ideal=np.linspace(0, 20, len(solver.results[0, :])),
        y_ideal=solver.results[0, :],
    )


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
    main()
    # run()
