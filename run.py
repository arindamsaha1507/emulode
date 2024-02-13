"""Main module for the application."""

import os

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import yaml

import numpy as np

from emulode.solver import Solver
from emulode.emulator import Emulator
from emulode.ode import ODE
from emulode.plotter import Plotter
from emulode.simulator import Simulator


def check_keys(config, required_keys: list[str]) -> None:
    """Check that the given keys are present in the config file."""

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Key '{key}' not found in config file")


class Config(ABC):
    """Abstract class for the configuration file."""

    def __init__(self, config_dict: dict, required_keys: list[str]) -> None:

        # self.config = cofig_dict
        check_keys(config_dict, required_keys)

        for key, value in config_dict.items():
            if key in required_keys:
                setattr(self, key, value)

    @abstractmethod
    def validate(self) -> None:
        """Validate the data."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the configuration file."""


class ODEConfig(Config):
    """Class for the ODE configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = ["chosen_ode", "parameters"]
        self.chosen_ode = None

        super().__init__(config_dict, required_keys)

        if self.chosen_ode is None:
            raise ValueError("Chosen ODE not found")

        self.parameters = config_dict["parameters"][self.chosen_ode]

    def validate(self) -> None:
        """Validate the data."""

        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")

        for key, value in self.parameters.items():
            if not isinstance(value, (int, float)):
                raise ValueError(f"Parameter {key} must be a number")

    def __repr__(self) -> str:
        """Return a string representation of the configuration file."""

        return f"Chosen ODE: {self.chosen_ode}\nParameters: {self.parameters}"


def create_ode_config(config_dict: dict) -> Config:
    """Create a configuration object from the given dictionary."""

    return ODEConfig(config_dict["ode"])


def load_config(file_path: str) -> Config:
    """Load the configuration file from the given path."""

    with open(file_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return create_ode_config(config)


def trial() -> None:

    ode_config = load_config("config.yml")
    print(ode_config)


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
