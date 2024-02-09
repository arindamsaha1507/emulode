"""Module for plotting data"""

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
        color: str = None,
        style: str = None,
        legend: str = None,
    ) -> None:
        """Plot the given data."""

        # pylint: disable=too-many-arguments

        xdata = xdata.flatten()
        ydata = ydata.flatten()

        if yvar is not None:
            yvar = yvar.flatten()
            ax.plot(xdata, ydata, color + style, label=legend)
            ax.fill_between(xdata, ydata - yvar, ydata + yvar, color=color, alpha=0.3)
        elif style == "-":
            ax.plot(xdata, ydata, color=color, linestyle=style, label=legend)
        else:
            ax.scatter(xdata, ydata, color=color, marker=style, label=legend)

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
        data_color: str = "r",
        predict_color: str = "b",
        data_style: str = "o",
        predict_style: str = "-",
        scale: float = 1.0,
        x_ideal: np.ndarray = None,
        y_ideal: np.ndarray = None,
    ) -> plt.Figure:
        """Create a combined plot of the given data."""

        # pylint: disable=too-many-arguments

        fig, ax = plt.subplots()

        xdata *= scale
        x_predict *= scale

        if x_ideal is not None and y_ideal is not None:
            Plotter.plotter(ax, x_ideal, y_ideal, color="g", style="-", legend="ideal")

        Plotter.plotter(
            ax,
            x_predict,
            y_predict,
            y_var,
            color=predict_color,
            style=predict_style,
            legend="emulated",
        )

        Plotter.plotter(
            ax, xdata, ydata, color=data_color, style=data_style, legend="simulated"
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.legend()

        Plotter.savefig(fig, file)

    @staticmethod
    def create_basic_plot(
        xdata: np.ndarray, ydata: np.ndarray, filename: str = "plots/basic.png"
    ) -> plt.Figure:
        """Create a basic plot of the given data."""

        fig, ax = plt.subplots()

        Plotter.plotter(ax, xdata, ydata, style="-")
        Plotter.savefig(fig, filename=filename)
