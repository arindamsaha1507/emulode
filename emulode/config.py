"""Module for configuration of the application."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import yaml


class Config(ABC):
    """Abstract class for the configuration file."""

    def __init__(self, config_dict: dict, required_keys: list[str]) -> None:

        # self.config = cofig_dict
        self.check_keys(config_dict, required_keys)

        for key, value in config_dict.items():
            if key in required_keys:
                setattr(self, key, value)

    @abstractmethod
    def validate(self) -> None:
        """Validate the data."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the configuration file."""

    @staticmethod
    def check_keys(config, required_keys: list[str]) -> None:
        """Check that the given keys are present in the config file."""

        for key in required_keys:
            if key not in config:
                raise KeyError(f"Key '{key}' not found in config file")


class ODEConfig(Config):
    """Class for the ODE configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = ["chosen_ode", "parameters"]
        self.chosen_ode: str = None

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

        return (
            f"ODE System\n===========\n"
            f"Chosen ODE: {self.chosen_ode}\n"
            f"Parameters: {self.parameters}\n"
        )


class SolverConfig(Config):
    """Class for the solver configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = ["initial_conditions", "t_range", "n_steps", "transience"]
        self.initial_conditions: list[float] = None
        self.t_range: list[float] = None
        self.n_steps: int = None
        self.transience: float = None

        super().__init__(config_dict, required_keys)

    @property
    def t_initial(self) -> float:
        """Return the initial time."""

        return self.t_range[0]

    @property
    def t_final(self) -> float:
        """Return the final time."""

        return self.t_range[1]

    def validate(self) -> None:

        if not isinstance(self.initial_conditions, list):
            raise ValueError("Initial conditions must be a list")

        if not isinstance(self.t_range, list):
            raise ValueError("Time range must be a tuple")

        if not isinstance(self.n_steps, int):
            raise ValueError("Number of steps must be an integer")

        if not isinstance(self.transience, float):
            raise ValueError("Transience must be a float")

        if len(self.t_range) != 2:
            raise ValueError("Time range must have two elements")

        if self.t_initial >= self.t_final:
            raise ValueError("Initial time must be less than final time")

        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive")

        if self.transience < 0 or self.transience > 1:
            raise ValueError("Transience must be between 0 and 1")

    def __repr__(self) -> str:

        return (
            f"Solver\n======\n"
            f"Initial conditions: {self.initial_conditions}\n"
            f"Time range: {self.t_range}\n"
            f"Number of steps: {self.n_steps}\n"
            f"Transience: {self.transience}\n"
        )


class SimulatorConfig(Config):
    """Class for the simulator configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = [
            "parameter_of_interest",
            "component_of_interest",
            "range",
            "n_simulation_points",
        ]
        self.parameter_of_interest: str = None
        self.component_of_interest: int = None
        self.range: tuple[float] = None
        self.n_simulation_points: int = None

        super().__init__(config_dict, required_keys)

    def validate(self) -> None:

        if not isinstance(self.parameter_of_interest, str):
            print(self.parameter_of_interest)
            raise ValueError("Parameter of interest must be a string")

        if not isinstance(self.component_of_interest, int):
            raise ValueError("Component of interest must be an integer")

        if not isinstance(self.range, list):
            raise ValueError("Range must be a tuple")

        if not isinstance(self.n_simulation_points, int):
            raise ValueError("Number of simulation points must be an integer")

        if len(self.range) != 2:
            raise ValueError("Range must have two elements")

        if self.n_simulation_points <= 0:
            raise ValueError("Number of simulation points must be positive")

    def __repr__(self) -> str:

        return (
            f"Simulator\n=========\n"
            f"Parameter of interest: {self.parameter_of_interest}\n"
            f"Component of interest: {self.component_of_interest}\n"
            f"Range: {self.range}\n"
            f"Number of simulation points: {self.n_simulation_points}\n"
        )


class EmulatorConfig(Config):
    """Class for the emulator configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = ["n_layers", "n_prediction_points", "n_iterations"]
        self.n_layers: int = None
        self.n_prediction_points: int = None
        self.n_iterations: int = None

        super().__init__(config_dict, required_keys)

    def validate(self) -> None:

        if not isinstance(self.n_layers, int):
            raise ValueError("Number of layers must be an integer")

        if not isinstance(self.n_prediction_points, int):
            raise ValueError("Number of prediction points must be an integer")

        if not isinstance(self.n_iterations, int):
            raise ValueError("Number of iterations must be an integer")

        if self.n_layers <= 0:
            raise ValueError("Number of layers must be positive")

        if self.n_prediction_points <= 0:
            raise ValueError("Number of prediction points must be positive")

        if self.n_iterations <= 0:
            raise ValueError("Number of iterations must be positive")

    def __repr__(self) -> str:

        return (
            f"Emulator\n========\n"
            f"Number of layers: {self.n_layers}\n"
            f"Number of prediction points: {self.n_prediction_points}\n"
            f"Number of iterations: {self.n_iterations}\n"
        )


class PlotterConfig(Config):
    """Class for the plotter configuration file."""

    def __init__(self, config_dict: dict) -> None:

        required_keys = ["directory", "filename", "x_label", "y_label"]
        self.directory: str = None
        self.filename: str = None
        self.x_label: str = None
        self.y_label: str = None

        super().__init__(config_dict, required_keys)

    def validate(self) -> None:

        if not isinstance(self.directory, str):
            raise ValueError("Directory must be a string")

        if not isinstance(self.filename, str):
            raise ValueError("Filename must be a string")

        if not isinstance(self.x_label, str):
            raise ValueError("X label must be a string")

        if not isinstance(self.y_label, str):
            raise ValueError("Y label must be a string")

    def __repr__(self) -> str:

        return (
            f"Plotter\n=======\n"
            f"Directory: {self.directory}\n"
            f"Filename: {self.filename}\n"
            f"X label: {self.x_label}\n"
            f"Y label: {self.y_label}\n"
        )


@dataclass
class Configs:
    """Class for the configuration file."""

    config_file: os.PathLike

    ode: ODEConfig = field(init=False)
    solver: SolverConfig = field(init=False)
    simulator: SimulatorConfig = field(init=False)
    emulator: EmulatorConfig = field(init=False)
    plotter: PlotterConfig = field(init=False)

    def __post_init__(self) -> None:

        self.load_config()
        self.individual_validate()

    def load_config(self) -> None:
        """Load the configuration file."""

        with open(self.config_file, "r", encoding="utf-8") as file:
            config_dict = yaml.safe_load(file)

        self.ode = ODEConfig(config_dict["ode"])
        self.solver = SolverConfig(config_dict["solver"])
        self.simulator = SimulatorConfig(config_dict["simulator"])
        self.emulator = EmulatorConfig(config_dict["emulator"])
        self.plotter = PlotterConfig(config_dict["plotter"])

    def individual_validate(self) -> None:
        """Validate the data."""

        self.ode.validate()
        self.solver.validate()
        self.simulator.validate()
        self.emulator.validate()
        self.plotter.validate()

    def __repr__(self) -> str:
        """Return a string representation of the configuration file."""

        return (
            f"{self.ode}\n"
            f"{self.solver}\n"
            f"{self.simulator}\n"
            f"{self.emulator}\n"
            f"{self.plotter}\n"
        )
