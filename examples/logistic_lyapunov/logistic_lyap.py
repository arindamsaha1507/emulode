"""Module for computing the Lyapunov exponent of the logistic map."""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def logistic_map_lyapunov_exponent(r, x0, n_iterations=10000):
    """
    Compute the Lyapunov exponent for the logistic map.

    Parameters:
    r (float): The parameter of the logistic map.
    x0 (float): Initial value for the logistic map.
    n_iterations (int): Number of iterations to compute.

    Returns:
    float: The Lyapunov exponent for the logistic map.
    """
    x = x0
    sum_log_deriv = 0

    for _ in range(n_iterations):
        # Logistic map function
        x = r * x * (1 - x)

        # Derivative of logistic map
        deriv = abs(r - 2 * r * x)

        # Avoid log(0) by ensuring the derivative is never zero
        if deriv == 0:
            return float("-inf")

        sum_log_deriv += np.log(deriv)

    # Calculating the Lyapunov exponent
    lyapunov_exponent = sum_log_deriv / n_iterations

    return lyapunov_exponent


def plotter():
    """Plot the Lyapunov exponent of the logistic map."""

    # Parameters
    n_iterations = 1000
    n_points = 1000
    r_values = np.linspace(0, 4, n_points)
    x0 = 0.5

    # Compute the Lyapunov exponent for each value of r
    lyapunov_exponents = np.zeros(n_points)
    for i, r in enumerate(r_values):
        lyapunov_exponents[i] = logistic_map_lyapunov_exponent(r, x0, n_iterations)

    # Plot the Lyapunov exponent
    plt.plot(r_values, lyapunov_exponents)
    plt.xlabel("r")
    plt.ylabel("Lyapunov exponent")
    plt.title("Lyapunov exponent of the logistic map")
    plt.grid()
    plt.savefig("lyapunov_exponent.png")
    # plt.show()


def main():

    args = argparse.ArgumentParser(
        description="Compute the Lyapunov exponent of the logistic map."
    )
    args.add_argument(
        "--r", type=float, default=3.8, help="Parameter of the logistic map."
    )

    args = args.parse_args()

    with open("lyapunov_exponent.txt", "w", encoding="utf-8") as f:
        f.write(str(logistic_map_lyapunov_exponent(args.r, 0.5)))


if __name__ == "__main__":
    main()
