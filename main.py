"""Main module for the application."""

import numpy as np

from emulode.solver import Solver
from emulode.emulator import Emulator
from emulode.ode import ODE
from emulode.plotter import Plotter
from emulode.simulator import Simulator

solver = Solver(
    ODE.lorenz,
    {"sigma": 10, "rho": 28, "beta": 8 / 3},
    np.array([1, 2, 3]),
    (0, 100),
    1000,
    0.1,
)

solver.solve()
solver.phase_plot(components=(0, 1))
solver.timeseries_plot(component=0)

simulator = Simulator(solver, "rho", 0, 30, 10, lambda x: np.max(x[0, :]))

emulator = Emulator(simulator.xdata, simulator.ydata, 3, 1000, 500)

Plotter.create_combined_plot(
    "plots/emulator.png",
    "rho",
    "Max(x_1)",
    simulator.xdata,
    simulator.ydata,
    emulator.x_predict,
    emulator.y_predict,
    emulator.y_var,
)
