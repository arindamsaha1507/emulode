"""Module for the emulator class"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import dgpsi

from ode import ODE
from solver import Solver
from utils import create_data, plotter


@dataclass
class Emulator:

    x_train: np.ndarray
    y_train: np.ndarray

    num_layers: int
    num_predict: int
    num_training_iterations: int

    model: dgpsi.dgp = field(init=False)
    x_predict: np.ndarray = field(init=False)
    y_predict: np.ndarray = field(init=False)
    y_var: np.ndarray = field(init=False)

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

    def create_layer(self, scale_est: bool = False) -> list[dgpsi.kernel]:
        """Create single layer of the emulator."""

        return [dgpsi.kernel(length=np.array([1.0]), scale_est=scale_est)]

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

        emulator = dgpsi.emulator(self.model.estimate())

        self.x_predict = np.linspace(
            self.x_train.min(), self.x_train.max(), self.num_predict
        )[:, None].reshape(-1, 1)
        self.y_predict, self.y_var = emulator.predict(self.x_predict)


if __name__ == "__main__":

    solver = Solver(
        ODE.lorenz,
        {"sigma": 10, "rho": 28, "beta": 8 / 3},
        np.array([1, 2, 3]),
        (0, 100),
        1000,
        0.1,
    )

    max_func = lambda x: np.max(x[0, :])

    solver.set_varying_settings("rho", max_func)

    _, ax = plt.subplots()
    # Step.plot(ax)

    xdata, ydata = create_data((0, 30), 10, solver)

    xdata = xdata[:, None]
    ydata = ydata.reshape(-1, 1)

    plotter(ax, xdata, ydata, color="r", style="o")

    emulator = Emulator(xdata, ydata, 3, 1000, 500)

    plotter(
        ax,
        emulator.x_predict,
        emulator.y_predict,
        emulator.y_var,
        color="b",
        style="-",
    )

    plt.savefig("emulator.png")
