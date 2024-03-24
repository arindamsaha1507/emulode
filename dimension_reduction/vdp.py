import multiprocessing

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


# Define the Van der Pol oscillator system
def van_der_pol_coupled_variable_k(t, state, N, b, Q, ks):
    # Unpack the state vector (x1, y1, x2, y2, ..., xN, yN)
    x = state[::2]  # x = [x1, x2, ..., xN]
    y = state[1::2]  # y = [y1, y2, ..., yN]

    # Compute the mean field
    x_mean = np.mean(x)

    # Initialize derivatives
    dxdt = np.zeros(N)
    dydt = np.zeros(N)

    # Compute the derivatives
    for i in range(N):
        dxdt[i] = y[i] + ks[i] * (Q * x_mean - x[i])
        dydt[i] = b * (1 - x[i] ** 2) * y[i] - x[i]

    # Repack the derivatives into a flat array
    derivatives = np.empty(2 * N)
    derivatives[::2] = dxdt
    derivatives[1::2] = dydt

    return derivatives


def single_run(factor):

    # Parameters

    N = 50  # Number of oscillators; this can be changed as needed
    b = 1.0
    Q = 0.5

    if isinstance(factor, float):
        ks = [
            0.4 * factor,
            0.8 * factor,
            1.2 * factor,
            1.6 * factor,
            2.0 * factor,
        ] * int(N / 5)

    elif isinstance(factor, np.ndarray):
        ks = factor

    else:
        print(type(factor))
        raise ValueError("Please provide a factor or a list of k values")

    # ks = np.ones(N) * factor
    # ks = np.sin(np.linspace(0, np.pi, N)) * factor
    # ks = list(x * factor for x in range(N))
    # Coupling strengths; we choose random values for this example

    # Initial conditions: we choose random initial conditions for this example
    initial_state = np.ones(2 * N)  # For x_i and y_i
    # initial_state = np.random.randn(2 * N)  # For x_i and y_i
    # initial_state = np.zeros(2 * N)

    # Time vector over which we'll integrate the system
    t_span = (0, 10000)  # From t=0 to t=50
    t_eval = np.linspace(
        *t_span, 100000
    )  # Generate time points where the solution is evaluated

    # Solve the system of ODEs
    sol = solve_ivp(
        van_der_pol_coupled_variable_k,
        t_span,
        initial_state,
        args=(N, b, Q, ks),
        t_eval=t_eval,
    )

    # Check if the solver was successful
    if not sol.success:
        raise Exception("ODE solver did not converge")

    # Now let's plot the results
    # plt.figure(figsize=(12, 8))
    # Plot x_i vs time for each oscillator
    # for i in range(N):
    #     plt.plot(sol.t, sol.y[2 * i], label=f"x{i+1}")

    amplitude = np.sqrt(sol.y[::2] ** 2 + sol.y[1::2] ** 2)
    amplitude_mean = np.mean(amplitude, axis=0)

    # plt.plot(sol.t[-1000:], amplitude_mean[-1000:], label="Distance from origin")

    # result = max(amplitude_mean[-1000:]) - min(amplitude_mean[-1000:])
    result = np.mean(amplitude_mean[-100:])

    # print(f"Parameter: {factor} \t Result: {result}")

    return result

    # plt.title("Dynamics of Coupled Van der Pol Oscillators")
    # plt.xlabel("Time")
    # plt.ylabel("State variables x_i")
    # plt.legend()
    # plt.grid()
    # plt.savefig("vdp.png")
    # plt.show()


def multiple_runs(factors_list):
    """Run the Van der Pol oscillator system for multiple values of the coupling factor."""

    with multiprocessing.Pool() as pool:
        results = pool.starmap(single_run, [(factor,) for factor in factors_list])

    return results


def main():
    """Main function to run the Van der Pol oscillator system."""

    factors = np.linspace(0.0, 5.0, 101)

    # factors = []
    # for i in range(101):
    #     factors.append(np.random.random(50))

    # print(factors[0])

    results = multiple_runs(factors)

    plt.plot(factors, results)
    plt.savefig("vdp.png")


if __name__ == "__main__":
    main()
