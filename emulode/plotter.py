"""Module for plotting data"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Plotter:
    """Class for plotting data"""

    fig: plt.Figure = field(init=False)
    ax: plt.Axes = field(init=False)

    def __post_init__(self) -> None:
        """Create a new figure and axis."""
        self.fig, self.ax = plt.subplots()

    def plotter(
        self,
        xdata: np.ndarray,
        ydata: np.ndarray,
        yvar: np.ndarray = None,
        color: str = "k",
        style: str = "-",
    ) -> None:
        """Plot the given data."""

        # pylint: disable=too-many-arguments

        xdata = xdata.flatten()
        ydata = ydata.flatten()

        if yvar is not None:
            yvar = yvar.flatten()
            self.ax.plot(xdata, ydata, color + style)
            self.ax.fill_between(
                xdata, ydata - yvar, ydata + yvar, color=color, alpha=0.3
            )
        else:
            self.ax.scatter(xdata, ydata, color=color, marker=style)

    def savefig(self, filename: str) -> None:
        """Save the figure to a file."""
        self.fig.savefig(filename)
