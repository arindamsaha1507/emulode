import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def kuramoto_ode(t, theta, coupling, natural_frequencies):
    num_oscillators = len(natural_frequencies)
    dtheta = np.zeros(num_oscillators)
    for i in range(num_oscillators):
        dtheta[i] = natural_frequencies[i]
        for j in range(num_oscillators):
            dtheta[i] += coupling[i, j] * np.sin(theta[j] - theta[i])
    return dtheta


def main():
    parser = argparse.ArgumentParser(description="Solve the Kuramoto model.")
    parser.add_argument(
        "--time_span",
        type=float,
        nargs=2,
        required=True,
        help="Start and end of the time interval.",
    )
    parser.add_argument(
        "--num_oscillators", type=int, required=True, help="Number of oscillators."
    )
    parser.add_argument(
        "--natural_frequencies",
        type=float,
        nargs="+",
        required=True,
        help="Natural frequencies of oscillators.",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        nargs="+",
        required=True,
        help="Coupling strengths between oscillators.",
    )
    parser.add_argument(
        "--initial_conditions",
        type=float,
        nargs="+",
        help="Initial phases of oscillators.",
        default=None,
    )

    args = parser.parse_args()

    if args.initial_conditions is None:
        args.initial_conditions = np.random.uniform(0, 2 * np.pi, args.num_oscillators)

    if len(args.coupling) != args.num_oscillators**2:
        raise ValueError("The number of coupling values must be num_oscillators^2")

    coupling_matrix = np.array(args.coupling).reshape(
        args.num_oscillators, args.num_oscillators
    )
    natural_frequencies = np.array(args.natural_frequencies)
    initial_conditions = np.array(args.initial_conditions)

    result = solve_ivp(
        kuramoto_ode,
        args.time_span,
        initial_conditions,
        args=(coupling_matrix, natural_frequencies),
        t_eval=np.linspace(args.time_span[0], args.time_span[1], 1000),
    )

    res = result.y[:, -1]
    # print(res)
    r_value = np.abs(np.mean(np.exp(1j * res)))
    # print(r_value)

    with open("kuramoto_output.txt", "w", encoding="utf-8") as file:
        file.write(str(r_value))

    # Plot the result
    # plt.figure()
    # for i in range(args.num_oscillators):
    #     plt.plot(result.t, np.sin(result.y[i]), label=f"Oscillator {i+1}")
    # plt.xlabel("Time")
    # plt.ylabel("sin(θ)")
    # plt.title("Time Evolution of sin(θ)")
    # plt.legend()
    # plt.grid()
    # plt.savefig("kuramoto_output.png")
    # plt.show()


if __name__ == "__main__":
    main()
