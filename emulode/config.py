"""Module for configuration of the application."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
import yaml

from emulode.globals import KernelFunction, Sampler


class Config(ABC):
    """
    Abstract class for the configuration file. This class should be inherited
    by other configuration classes.

    This defines the basic structure of all the congiguration classes. It
    contains the `config_dict` which is the dictionary representation of the
    configuration file and the `required_keys` which is a list of keys that
    must be present in the configuration file. These checks are performed in
    the `__init__` method.

    Every configuration class should have a `validate` method which validates
    the data in the configuration class and a `__repr__` method which returns
    a string representation of the configuration class.

    Args:
        config_dict: The dictionary representation of the configuration file
        required_keys: A list of keys that must be present in the configuration
            file
    """

    def __init__(self, config_dict: dict, required_keys: list[str]) -> None:

        self.check_keys(config_dict, required_keys)

        for key, value in config_dict.items():
            if key in required_keys:
                setattr(self, key, value)

    @abstractmethod
    def validate(self) -> None:
        """Validate the data in the configuration file."""

    @abstractmethod
    def __repr__(self) -> str:
        """Return a string representation of the configuration class."""

    def check_keys(self, config, required_keys: list[str]) -> None:
        """
        Checks that the given keys are present in the config file. If not, raises
        a KeyError. This method is used to check the required keys in the
        initialization of the configuration base class.
        """

        for key in required_keys:

            if getattr(self, key) is not None:
                continue

            if key not in config:
                raise KeyError(f"Key '{key}' not found in config file")


class SolverConfig(Config):
    """
    Class for the solver configuration file.

    This class contains parameters related to the solver. The `required_keys` for
    this class are "initial_conditions", "t_range", "n_steps" and "transience".

    Args:
        config_dict: The dictionary representation of the 'solver' part of the configuration file

        initial_conditions: The initial conditions for the ODE
        t_range: The time range for the simulation in the form [t_initial, t_final]
        n_steps: The number of steps for the simulation
        transience: The transience for the simulation (as a fraction between 0 and 1)
    """

    def __init__(
        self,
        config_dict: dict,
        run_command: str = None,
        parameters: dict = None,
        replacement_prefix: str = None,
        results_file: os.PathLike = None,
        prefix_commands: list[str] = None,
        # initial_conditions: list[float] = None,
        # t_range: list[float] = None,
        # n_steps: int = None,
        # transience: float = None,
    ) -> None:

        # pylint: disable=too-many-arguments

        required_keys = [
            "run_command",
            "parameters",
            "replacement_prefix",
            "results_file",
            "prefix_commands",
        ]

        self.run_command: str = run_command
        self.parameters: dict = parameters
        self.replacement_prefix: str = replacement_prefix
        self.results_file: os.PathLike = results_file
        self.prefix_commands: list[str] = prefix_commands

        super().__init__(config_dict, required_keys)

    def validate(self) -> None:

        pass

    def __repr__(self) -> str:

        return (
            f"Solver\n======\n"
            f"Run command: {self.run_command}\n"
            f"Parameters: {self.parameters}\n"
            f"Replacement prefix: {self.replacement_prefix}\n"
            f"Results file: {self.results_file}\n"
            f"Prefix commands: {self.prefix_commands}\n"
        )


class SimulatorConfig(Config):
    """
    Class for the simulator configuration file.

    This class contains parameters related to the simulator. The `required_keys` for
    this class are "parameter_of_interest", "range" and
    "n_simulation_points".

    Args:
        config_dict: The dictionary representation of the 'simulator' part of the configuration file

        parameter_of_interest: The parameter of interest (Must be one of parameters in ODE)
        parameter_range: The range for the simulation in the form (min, max)
        n_simulation_points: The number of simulation points
    """

    def __init__(
        self,
        config_dict: dict,
        parameter_of_interest: str = None,
        result_dimension: int = None,
        parameter_range: tuple[float] = None,
        n_simulation_points: int = None,
        sampling_method: Sampler = None,
    ) -> None:

        # pylint: disable=too-many-arguments

        required_keys = [
            "parameter_of_interest",
            "result_dimension",
            "range",
            "n_simulation_points",
            "sampling_method",
        ]
        self.parameter_of_interest = parameter_of_interest
        self.result_dimension = result_dimension
        self.range = parameter_range
        self.n_simulation_points = n_simulation_points
        self.sampling_method = sampling_method

        super().__init__(config_dict, required_keys)

    def validate(self) -> None:

        if not isinstance(self.parameter_of_interest, str):
            print(self.parameter_of_interest)
            raise ValueError("Parameter of interest must be a string")

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
            f"Range: {self.range}\n"
            f"Number of simulation points: {self.n_simulation_points}\n"
        )


class EmulatorConfig(Config):
    """
    Class for the emulator configuration file.

    This class contains parameters related to the emulator. The `required_keys` for
    this class are "n_layers", "n_prediction_points" and "n_iterations".

    Args:
        config_dict: The dictionary representation of the 'emulator' part of the configuration file

        n_layers: The number of layers in the emulator
        n_prediction_points: The number of prediction points
        n_iterations: The number of iterations for the emulator
    """

    def __init__(
        self,
        config_dict: dict,
        n_layers: int = None,
        n_prediction_points: int = None,
        n_iterations: int = None,
        kernel_function: KernelFunction = None,
    ) -> None:

        # pylint: disable=too-many-arguments

        required_keys = [
            "n_layers",
            "n_prediction_points",
            "n_iterations",
            "kernel_function",
        ]
        self.n_layers: int = n_layers
        self.n_prediction_points: int = n_prediction_points
        self.n_iterations: int = n_iterations
        self.kernel_function: KernelFunction = kernel_function

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
    """
    Class for the plotter configuration file.

    This class contains parameters related to the plotter. The `required_keys` for
    this class are "directory", "filename", "x_label" and "y_label".

    Args:
        config_dict: The dictionary representation of the 'plotter' part of the configuration file

        directory: The directory to save the plot
        filename: The filename for the plot
        x_label: The x label for the plot
        y_label: The y label for the plot
    """

    def __init__(
        self,
        config_dict: dict,
        directory: str = None,
        filename: str = None,
        x_label: str = None,
        y_label: str = None,
    ) -> None:

        # pylint: disable=too-many-arguments

        required_keys = ["directory", "filename", "x_label", "y_label"]
        self.directory: str = directory
        self.filename: str = filename
        self.x_label: str = x_label
        self.y_label: str = y_label

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
    """
    Class for storing all configurations given in the configuration file.

    This class organises the configuration parameters into those releated to
    ODE, solver, simulator, emulator and plotter.

    Args:
        config_file: The path to the configuration file

        ode: The ODE configurations
        solver: The solver configurations
        simulator: The simulator configurations
        emulator: The emulator configurations
        plotter: The plotter configurations

    Each of these configurations are instances of the respective configuration
    classes.
    """

    config_file: os.PathLike

    # simulation: SimulationConfig = field(init=False)
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

        # self.simulation = SimulationConfig(config_dict["simulation"])
        self.solver = SolverConfig(config_dict["solver"])
        self.simulator = SimulatorConfig(config_dict["simulator"])
        self.emulator = EmulatorConfig(config_dict["emulator"])
        self.plotter = PlotterConfig(config_dict["plotter"])

    def individual_validate(self) -> None:
        """Validate the data."""

        # self.simulation.validate()
        self.solver.validate()
        self.simulator.validate()
        self.emulator.validate()
        self.plotter.validate()

    def __repr__(self) -> str:
        """Return a string representation of the configuration file."""

        return (
            # f"{self.simulation}\n"
            f"{self.solver}\n"
            f"{self.simulator}\n"
            f"{self.emulator}\n"
            f"{self.plotter}\n"
        )
