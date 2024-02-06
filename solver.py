"""Module for Solving ODEs."""

from typing import Callable

import time
import dgpsi
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from ode import ODE

from matplotlib import pyplot as plt


@dataclass
class Solver:
    """Class for solving ODEs."""

    ode: Callable[[float, np.ndarray, dict[str, float]], np.ndarray]
    params: dict[str, float]
    initial_conditions: np.ndarray
    t_span: tuple[float, float]
    t_steps: int
    transience: int | float

    results: np.ndarray = field(init=False)

    parameter_of_interest: str = field(init=False)
    quantity_of_interest: Callable[[np.ndarray], float] = field(init=False)

    def __post_init__(self) -> None:
        """Check that the given parameters are valid."""

        if self.transience < 0:
            raise ValueError("Transience must be non-negative")

        if self.t_steps <= 0:
            raise ValueError("t_steps must be positive")

        if self.transience < 1:
            self.transience = int(self.transience * self.t_steps)

        if self.transience >= self.t_steps:
            raise ValueError("Transience must be less than t_steps")

        if len(self.t_span) != 2:
            raise ValueError("t_span must be a tuple of length 2")

        if self.t_span[0] >= self.t_span[1]:
            raise ValueError("t_span must be increasing")

    @property
    def t_initial(self) -> float:
        """Return the initial time."""
        return self.t_span[0]

    @property
    def t_final(self) -> float:
        """Return the final time."""
        return self.t_span[1]

    def solve(self) -> None:
        """Solve the ODE."""

        sol = solve_ivp(
            self.ode,
            self.t_span,
            self.initial_conditions,
            t_eval=np.linspace(self.t_initial, self.t_final, self.t_steps),
            args=(self.params,),
        )

        self.results = sol.y[:, self.transience :]

    def set_varying_settings(self, parameter: str, qoi: Callable) -> None:
        """Set the parameter and quantity of interest."""

        if parameter not in self.params:
            raise ValueError(f"Parameter '{parameter}' not found")

        self.parameter_of_interest = parameter
        self.quantity_of_interest = qoi

    def evaluate_at_point(self, parameter: float) -> float:
        """Evaluate the quantity of interest for the given parameter."""

        if self.parameter_of_interest is None or self.quantity_of_interest is None:
            raise ValueError("Parameter and quantity of interest not set")

        self.params[self.parameter_of_interest] = parameter

        self.solve()
        return self.quantity_of_interest(self.results)


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

    xdata, ydata = create_data((0, 30), 3, solver)

    xdata = xdata[:, None]
    ydata = ydata.reshape(-1, 1)

    plotter(ax, xdata, ydata, color="r", style="o")

    layer1 = [dgpsi.kernel(length=np.array([1.0]), name="sexp")]
    layer2 = [dgpsi.kernel(length=np.array([1.0]), name="sexp")]
    layer3 = [dgpsi.kernel(length=np.array([1.0]), name="sexp", scale_est=True)]

    all_layer = dgpsi.combine(layer1, layer2, layer3)

    model = dgpsi.dgp(xdata, [ydata], all_layer)
    model.train(500)#(1000)

    emulator = dgpsi.emulator(model.estimate())
    xpredict = np.linspace(0, 30, 300)[:, None].reshape(-1, 1)
    ypredict, yvar = emulator.predict(xpredict)

    plotter(ax, xpredict, ypredict, yvar, color="g", style="-")
    plt.savefig("lorenz.png")
    plt.show()


# if __name__ == "__main__":
#     main()
