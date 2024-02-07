"""Module for utility functions"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from solver import Solver


def plotter(
    ax: plt.Axes,
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
        ax.plot(xdata, ydata, color + style)
        ax.fill_between(xdata, ydata - yvar, ydata + yvar, color=color, alpha=0.3)
    else:
        ax.scatter(xdata, ydata, color=color, marker=style)


def create_data(
    param_range: tuple[float, float], num_points: int, solver: Solver
) -> tuple[np.ndarray, np.ndarray]:
    """Create data for the given solver."""

    time_start = time.time()

    x = np.linspace(param_range[0], param_range[1], num_points)
    y = np.array([solver.evaluate_at_point(xi) for xi in x])

    time_end = time.time()

    print(f"Time taken: {time_end - time_start:.2f} seconds")

    return x, y
