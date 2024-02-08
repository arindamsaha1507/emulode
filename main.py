"""Main module for the application."""

import numpy as np
import matplotlib.pyplot as plt

from emulode.solver import Solver
from emulode.utils import create_data, plotter
from emulode.emulator import Emulator
from emulode.ode import ODE

solver = Solver(
    ODE.lorenz,
    {"sigma": 10, "rho": 28, "beta": 8 / 3},
    np.array([1, 2, 3]),
    (0, 100),
    1000,
    0.1,
)

solver.set_varying_settings("rho", lambda x: np.max(x[0, :]))

_, ax = plt.subplots()

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

plt.savefig("plots/emulator.png")
