"""Module to solve a network of Kuramoto oscillators."""

import argparse
from attr import dataclass

import numpy as np

from scipy.integrate import solve_ivp


def kuramoto_ode(time: float, theta: list[float], params: dict) -> list[float]:
    """Return the derivative of the Kuramoto ODE."""

    # pylint: disable=unused-argument

    coupling: np.ndarray = params["coupling"]
    natural_frequencies: np.ndarray = params["natural_frequencies"]
    num_oscillators: int = params["num_oscillators"]

    dtheta: np.ndarray = np.zeros(num_oscillators)

    for i in range(num_oscillators):
        dtheta[i] = natural_frequencies[i]
        for j in range(num_oscillators):
            dtheta[i] += coupling[i, j] * np.sin(theta[j] - theta[i])
            dtheta[i] = dtheta[i] % (2 * np.pi)

    return dtheta


def solve_kuramoto(params: dict) -> np.ndarray:
    """Solve the Kuramoto ODE."""

    initial_conditions: np.ndarray = params["initial_conditions"]
    time_span: tuple[float, float] = params["time_span"]
    # num_oscillators: int = params["num_oscillators"]

    result = solve_ivp(
        kuramoto_ode,
        time_span,
        initial_conditions,
        args=(params,),
        method="RK45",
        t_eval=np.linspace(time_span[0], time_span[1], 1000),
    )

    return result.y[:, -1]


@dataclass
class KuramotoParams:
    """Dataclass for the parameters of the Kuramoto model."""

    coupling: list[float]
    natural_frequencies: list[float]
    initial_conditions: list[float]
    time_span: list[float]
    num_oscillators: int


def parse_args() -> KuramotoParams:
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser(description="Solve the Kuramoto ODE.")
    parser.add_argument(
        "--coupling",
        type=float,
        nargs="+",
        help="The coupling matrix for the Kuramoto ODE.",
    )
    parser.add_argument(
        "--natural_frequencies",
        type=float,
        nargs="+",
        help="The natural frequencies for the Kuramoto ODE.",
    )
    parser.add_argument(
        "--initial_conditions",
        type=float,
        nargs="+",
        help="The initial conditions for the Kuramoto ODE.",
    )
    parser.add_argument(
        "--time_span", type=float, nargs=2, help="The time span for the Kuramoto ODE."
    )
    parser.add_argument(
        "--num_oscillators",
        type=int,
        help="The number of oscillators in the Kuramoto ODE.",
    )

    args = parser.parse_args()

    return KuramotoParams(
        coupling=args.coupling,
        natural_frequencies=args.natural_frequencies,
        initial_conditions=args.initial_conditions,
        time_span=args.time_span,
        num_oscillators=args.num_oscillators,
    )


def main():
    """Run the main function."""

    params = parse_args()

    solver_params = {
        "coupling": np.array(params.coupling).reshape(
            params.num_oscillators, params.num_oscillators
        ),
        "natural_frequencies": np.array(params.natural_frequencies),
        "initial_conditions": np.array(params.initial_conditions),
        "time_span": tuple(params.time_span),
        "num_oscillators": params.num_oscillators,
    }

    result = solve_kuramoto(solver_params)
    r_value = np.mean(np.exp(1j * result))
    r_value = np.abs(r_value)

    with open("kuramoto_output.txt", "w", encoding="utf-8") as file:
        file.write(str(r_value))


if __name__ == "__main__":
    main()
