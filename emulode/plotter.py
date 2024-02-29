"""Module for plotting data"""

import os
import numpy as np
import matplotlib.pyplot as plt

from emulode.config import Configs
from emulode.emulator import Emulator
from emulode.simulator import Simulator


class Plotter:
    """Class for plotting data"""

    @staticmethod
    def plotter(
        ax: plt.Axes,
        xdata: np.ndarray,
        ydata: np.ndarray,
        yvar: np.ndarray = None,
        style: str = None,
        legend: str = None,
    ) -> None:
        """Plot the given data."""

        # pylint: disable=too-many-arguments

        xdata = xdata.flatten()
        ydata = ydata.flatten()

        if yvar is not None:
            yvar = yvar.flatten()
            ax.plot(xdata, ydata, style, label=legend)
            ax.fill_between(xdata, ydata - yvar, ydata + yvar, color="gray", alpha=0.3)
        elif legend == "ideal":
            ax.plot(xdata, ydata, style, label=legend)
        else:
            ax.scatter(xdata, ydata, color=style, label=legend)

    @staticmethod
    def savefig(fig: plt.Figure, filename: str) -> None:
        """Save the figure to a file."""
        fig.savefig(filename)

    @staticmethod
    def create_combined_plot(
        configs: Configs,
        emulator: Emulator,
        simulator: Simulator,
        ideal: Simulator = None,
        save: bool = True,
    ) -> None:
        # file: str,
        # xlabel: str,
        # ylabel: str,
        # xdata: np.ndarray,
        # ydata: np.ndarray,
        # x_predict: np.ndarray,
        # y_predict: np.ndarray,
        # y_var: np.ndarray,
        # x_ideal: np.ndarray = None,
        # y_ideal: np.ndarray = None,
        # ) -> plt.Figure:
        """Create a combined plot of the given data."""

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        fig, ax = plt.subplots()

        # if x_ideal is not None and y_ideal is not None:

        if ideal is not None:
            Plotter.plotter(ax, ideal.xdata, ideal.ydata, style="g-", legend="ideal")

        Plotter.plotter(
            ax,
            emulator.x_predict,
            emulator.y_predict,
            emulator.y_var,
            style="b-",
            legend="emulated",
        )

        Plotter.plotter(
            ax, simulator.xdata, simulator.ydata, style="r", legend="simulated"
        )

        ax.set_xlabel(configs.plotter.x_label)
        ax.set_ylabel(configs.plotter.y_label)

        ax.legend()

        os.makedirs(configs.plotter.directory, exist_ok=True)

        filename = os.path.join(configs.plotter.directory, configs.plotter.filename)

        if save:
            Plotter.savefig(fig, filename)

    @staticmethod
    def create_basic_plot(
        xdata: np.ndarray, ydata: np.ndarray, filename: str = "plots/basic.png"
    ) -> plt.Figure:
        """Create a basic plot of the given data."""

        fig, ax = plt.subplots()

        Plotter.plotter(ax, xdata, ydata, style="-")
        Plotter.savefig(fig, filename=filename)
