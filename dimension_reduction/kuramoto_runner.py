"""Module for running the Kuramoto model from the command line."""

import os

import numpy as np


def kuramoto_wrapper(arguments: list[float]) -> None:
    """Create a command to run the Kuramoto model from the given arguments."""

    arguments = [str(arg) for arg in arguments]

    command = "python dimension_reduction/kuramoto.py"

    time_start = 0
    time_end = 100

    num_oscillators = np.sqrt(1 + len(arguments)) - 1

    if int(num_oscillators) != num_oscillators:
        raise ValueError("Invalid number of arguments.")

    num_oscillators = int(num_oscillators)

    natural_frequencies = arguments[:num_oscillators]
    coupling = arguments[num_oscillators : num_oscillators + num_oscillators**2]
    initial_conditions = arguments[num_oscillators + num_oscillators**2 :]

    command += f" --time_span {time_start} {time_end}"
    command += f" --num_oscillators {num_oscillators}"
    command += f" --natural_frequencies {' '.join(natural_frequencies)}"
    command += f" --coupling {' '.join(coupling)}"
    command += f" --initial_conditions {' '.join(initial_conditions)}"
    # print(command)

    return command


def run(arguments) -> float:
    """Run the Kuramoto model from the command line."""

    # num_oscillators = 3
    # length = num_oscillators + num_oscillators**2 + num_oscillators

    # length = len(arguments)

    # arguments = np.random.uniform(0, 12, length)

    command = kuramoto_wrapper(arguments)
    os.system(command)

    with open("kuramoto_output.txt", "r", encoding="utf-8") as file:
        return float(file.read())


def main():
    """Run the main function."""

    num_oscillators = 30
    length = num_oscillators + num_oscillators**2 + num_oscillators

    arguments = np.random.uniform(0, 12, length)
    run(arguments)


if __name__ == "__main__":
    print("Running main function.")
    main()
