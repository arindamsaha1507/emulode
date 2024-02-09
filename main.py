"""Main module for the application."""

import numpy as np

from emulode.solver import Solver
from emulode.utils import create_data
from emulode.emulator import Emulator
from emulode.ode import ODE
from emulode.plotter import Plotter

solver = Solver(
    ODE.lorenz,
    {"sigma": 10, "rho": 28, "beta": 8 / 3},
    np.array([1, 2, 3]),
    (0, 100),
    1000,
    0.1,
)

solver.set_varying_settings("rho", lambda x: np.max(x[0, :]))

xdata, ydata = create_data((0, 30), 10, solver)

xdata = xdata[:, None]
ydata = ydata.reshape(-1, 1)

emulator = Emulator(xdata, ydata, 3, 1000, 500)

plot = Plotter()

plot.plotter(xdata, ydata, color="r", style="o")
plot.plotter(
    emulator.x_predict, emulator.y_predict, emulator.y_var, color="b", style="-"
)
plot.savefig("plots/emulator.png")
