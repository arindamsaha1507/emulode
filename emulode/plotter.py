"""Module for plotting data"""

import os
import numpy as np
import matplotlib.pyplot as plt


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
        file: str,
        xlabel: str,
        ylabel: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        x_predict: np.ndarray,
        y_predict: np.ndarray,
        y_var: np.ndarray,
        x_ideal: np.ndarray = None,
        y_ideal: np.ndarray = None,
    ) -> plt.Figure:
        """Create a combined plot of the given data."""

        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-locals

        fig, ax = plt.subplots()

        if x_ideal is not None and y_ideal is not None:
            Plotter.plotter(ax, x_ideal, y_ideal, style="g-", legend="ideal")

        Plotter.plotter(ax, x_predict, y_predict, y_var, "b-", legend="emulated")

        Plotter.plotter(ax, xdata, ydata, style="r", legend="simulated")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()

        directory = file.split("/")[:-1]
        directory = "/".join(directory)

        if directory:
            os.makedirs(directory, exist_ok=True)

        Plotter.savefig(fig, file)

    @staticmethod
    def create_basic_plot(
        xdata: np.ndarray, ydata: np.ndarray, filename: str = "plots/basic.png"
    ) -> plt.Figure:
        """Create a basic plot of the given data."""

        fig, ax = plt.subplots()

        Plotter.plotter(ax, xdata, ydata, style="-")
        Plotter.savefig(fig, filename=filename)
