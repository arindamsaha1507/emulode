import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lorenz_system(t, X, sigma, beta, rho):
    """
    Lorenz system of differential equations.
    """
    x, y, z = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def jacobian_lorenz(X, sigma, beta, rho):
    """
    Jacobian of the Lorenz system.
    """
    x, y, z = X
    return np.array([[-sigma, sigma, 0], [rho - z, -1, -x], [y, x, -beta]])


def compute_lyapunov_exponents(
    initial_condition, t_max, dt, sigma, beta, rho, n_exponents=3
):
    """
    Compute the Lyapunov exponents for the Lorenz system.
    """

    def equations(t, Y):
        n = len(Y) // (n_exponents + 1)
        X = Y[:n]
        Q = Y[n:].reshape((n, n))

        dXdt = lorenz_system(t, X, sigma, beta, rho)
        dQdt = jacobian_lorenz(X, sigma, beta, rho) @ Q

        return np.concatenate([dXdt, dQdt.ravel()])

    Y0 = np.concatenate([initial_condition, np.eye(n_exponents).ravel()])
    solution = solve_ivp(
        equations, [0, t_max], Y0, method="RK45", t_eval=np.arange(0, t_max, dt)
    )

    X = solution.y[:n_exponents, :]
    Q = solution.y[n_exponents:, :].reshape((n_exponents, n_exponents, -1))

    # Compute the QR decompositions
    lyapunov_exponents = np.zeros((n_exponents, len(solution.t)))
    for i in range(len(solution.t)):
        Q[:, :, i], R = np.linalg.qr(Q[:, :, i])
        lyapunov_exponents[:, i] = np.log(np.abs(np.diag(R)))

    return np.mean(lyapunov_exponents, axis=1)


def main():
    parser = argparse.ArgumentParser(
        description="Compute the Lyapunov exponents of the Lorenz system."
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=10.0,
        help="Sigma parameter for the Lorenz system.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=8 / 3,
        help="Beta parameter for the Lorenz system.",
    )
    parser.add_argument(
        "--rho", type=float, default=28.0, help="Rho parameter for the Lorenz system."
    )
    parser.add_argument("--t_max", type=float, default=100.0, help="Integration time.")
    parser.add_argument("--dt", type=float, default=0.01, help="Integration timestep.")
    args = parser.parse_args()

    # Initial conditions
    initial_condition = np.array([1.0, 1.0, 1.0])

    lyapunov_exponents = compute_lyapunov_exponents(
        initial_condition, args.t_max, args.dt, args.sigma, args.beta, args.rho
    )

    # print("Lyapunov exponents:", lyapunov_exponents)

    with open("lyapunov_exponent.txt", "w", encoding="utf-8") as f:
        f.write(f"{lyapunov_exponents[0]}\n")


if __name__ == "__main__":
    main()
