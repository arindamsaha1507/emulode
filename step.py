"""Demonstrating Gaussian Process Emulator for a 1D step function."""

import random

import dgpsi
import numpy as np
import matplotlib.pyplot as plt


class Step:
    """Class for the step function and related methods."""

    @staticmethod
    def step(x: float, noise=0) -> float:
        """Step function."""

        if x < 0 or x > 1:
            raise ValueError("x must be between 0 and 1")

        return (-1 if x < 0.5 else 1) + random.gauss(0, noise)

    @staticmethod
    def plot(
        ax: plt.Axes, num_points: int = 100, noise: float = 0.0, dots: bool = False
    ) -> None:
        """Plot the step function."""

        x, y = Step.create_data(num_points, noise)
        plotter(ax, x, y, color="b", style="-")

    @staticmethod
    def create_data(n: int, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Create data for the step function."""

        x = np.linspace(0, 1, n)
        y = np.array([Step.step(xi, noise) for xi in x])

        return x, y


def plotter(
    ax: plt.Axes,
    xdata: np.ndarray,
    ydata: np.ndarray,
    yvar: np.ndarray = None,
    color: str = "k",
    style: str = "-",
) -> None:
    """Plot the given data."""

    xdata = xdata.flatten()
    ydata = ydata.flatten()

    if yvar is not None:
        yvar = yvar.flatten()
        ax.errorbar(xdata, ydata, yerr=yvar, fmt=color + style)
    else:
        ax.plot(xdata, ydata, color + style)


def main() -> None:
    """Main function."""

    _, ax = plt.subplots()
    Step.plot(ax)

    xdata, ydata = Step.create_data(10, 0.01)

    xdata = xdata[:, None]
    ydata = ydata.reshape(-1, 1)

    plotter(ax, xdata, ydata, color="r", style="o")

    layer1 = [dgpsi.kernel(length=np.array([1.0]), name="sexp")]
    layer2 = [dgpsi.kernel(length=np.array([1.0]), name="sexp")]
    layer3 = [dgpsi.kernel(length=np.array([1.0]), name="sexp", scale_est=True)]

    all_layer = dgpsi.combine(layer1, layer2, layer3)

    model = dgpsi.dgp(xdata, [ydata], all_layer)
    model.train()

    emulator = dgpsi.emulator(model.estimate())
    xpredict = np.linspace(0, 1, 300)[:, None].reshape(-1, 1)
    ypredict, yvar = emulator.predict(xpredict)

    plotter(ax, xpredict, ypredict, yvar, color="g", style="-")
    plt.show()


if __name__ == "__main__":
    main()
