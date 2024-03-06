"""Module for the emulator class"""

import argparse
from dataclasses import dataclass, field

import numpy as np
import dgpsi

from emulode.simulator import Simulator
from emulode.config import Configs
from emulode.globals import KernelFunction


@dataclass
class Emulator:
    """
    Class for the emulator.

    Args:
        x_train: The training input data
        y_train: The training output data
        num_layers: The number of layers in the emulator
        num_predict: The number of points to predict
        num_training_iterations: The number of iterations to train the emulator
        model: The emulator model
        x_predict: The input data to predict
        y_predict: The predicted output data
        y_var: The variance of the predicted output data
    """

    # pylint: disable=too-many-instance-attributes

    x_train: np.ndarray = field(repr=False)
    y_train: np.ndarray = field(repr=False)

    num_layers: int
    num_predict: int
    num_training_iterations: int

    model: dgpsi.dgp = field(init=False)
    x_predict: np.ndarray = field(init=False, repr=False)
    y_predict: np.ndarray = field(init=False, repr=False)
    y_var: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Check that the given parameters are valid."""

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_predict <= 0:
            raise ValueError("num_predict must be positive")

        if self.num_training_iterations <= 0:
            raise ValueError("num_training_iterations must be positive")

        self.create_model()
        self.predict()

    def create_layer(
        self,
        scale_est: bool = False,
        kernal_function: KernelFunction = KernelFunction.MATERN,
    ) -> list[dgpsi.kernel]:
        """Create single layer of the emulator.

        Args:
            scale_est: Whether to estimate the scale

        Returns:
            The layer of the emulator
        """

        name = kernal_function.value

        return [dgpsi.kernel(length=np.array([1.0]), scale_est=scale_est, name=name)]

    def create_all_layers(self) -> list:
        """Create all layers of the emulator."""

        layers = []

        for idx in range(self.num_layers):
            if idx == self.num_layers - 1:
                layers.append(self.create_layer(scale_est=True))
            else:
                layers.append(self.create_layer())

        return dgpsi.combine(*layers)

    def create_model(self) -> None:
        """Create the emulator model."""

        layers = self.create_all_layers()

        self.model = dgpsi.dgp(self.x_train, [self.y_train], layers)
        self.model.train(self.num_training_iterations)

    def predict(self) -> None:
        """Predict the emulator output."""

        emul = dgpsi.emulator(self.model.estimate())

        self.x_predict = np.linspace(
            self.x_train.min(), self.x_train.max(), self.num_predict
        )[:, None].reshape(-1, 1)
        self.y_predict, self.y_var = emul.predict(self.x_predict)


class EmulatorFactory:
    """Factory class for the emulator."""

    simulator: Simulator
    configs: Configs

    @staticmethod
    def create_from_config(simulator: Simulator, configs: Configs) -> Emulator:
        """Create an emulator from the given configuration."""

        n_layers = configs.emulator.n_layers
        n_predict = configs.emulator.n_prediction_points
        n_iterations = configs.emulator.n_iterations

        return Emulator(
            simulator.xdata, simulator.ydata, n_layers, n_predict, n_iterations
        )

    @staticmethod
    def create_from_commandline_arguments(
        simulator: Simulator, args: argparse.Namespace
    ) -> Emulator:
        """Create an emulator from the given command line arguments."""

        raise NotImplementedError("Command line arguments not supported yet")
