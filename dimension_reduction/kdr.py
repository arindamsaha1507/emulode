"""Module to reduce the dimension of a network of Kuramoto oscillators."""

import time
import mogp_emulator
import numpy as np
from scipy.stats import qmc

from dimension_reduction.kuramoto_runner import run


def rescale_parameters(parameters, scale_frequency, scale_coupling, scale_phase):
    """Rescale the parameters to the correct range.

    Args:
        parameters (np.array): The parameters to rescale.

    Returns:
        np.array: The rescaled parameters.
    """

    num_oscillators = np.sqrt(1 + len(parameters)) - 1
    if int(num_oscillators) != num_oscillators:
        print(len(parameters))
        raise ValueError("Invalid number of arguments.")
    num_oscillators = int(num_oscillators)

    for index, parameter in enumerate(parameters):
        if index < num_oscillators:
            parameters[index] = parameter * scale_frequency
        elif index < num_oscillators + num_oscillators**2:
            parameters[index] = parameter * scale_coupling
        else:
            parameters[index] = parameter * scale_phase

    return parameters


def main():
    """Run the main function."""

    print("Dimension reduction")
    print("===================")

    num_oscillators = 3
    length = num_oscillators + num_oscillators**2 + num_oscillators

    time_start = time.time()

    # ed = mogp_emulator.LatinHypercubeDesign(length)

    sampler = qmc.LatinHypercube(d=length)
    sample = sampler.random(n=100)
    inputs = qmc.scale(sample, 0, 1)

    print(len(inputs))

    inputs = [rescale_parameters(p, 10, 1, 2 * np.pi) for p in inputs]

    # for input in inputs:

    #     for index, parameter in enumerate(input):
    #         if index < num_oscillators:
    #             input[index] = parameter * 10
    #         elif index < num_oscillators + num_oscillators**2:
    #             input[index] = parameter
    #         else:
    #             input[index] = parameter * 2 * np.pi

    # exit()

    targets = np.array([run(p) for p in inputs])

    print(f"Time to generate sampling_points: {time.time() - time_start}")

    time_start = time.time()

    dr_tuned, loss = mogp_emulator.gKDR.tune_parameters(
        inputs,
        targets,
        mogp_emulator.fit_GP_MAP,
        cXs=[3.0],
        cYs=[3.0],
    )

    print(f"Number of inferred dimensions is {dr_tuned.K}")
    print(f"Loss is {loss}")

    print(f"Time to tune parameters: {time.time() - time_start}")

    time_start = time.time()

    gp_tuned = mogp_emulator.fit_GP_MAP(dr_tuned(inputs), targets)

    print(f"Time to fit GP: {time.time() - time_start}")

    time_start = time.time()

    predict_points = qmc.scale(sampler.random(n=10), 0, 1)

    predict_points = [rescale_parameters(p, 10, 1, 2 * np.pi) for p in predict_points]

    # for input in predict_points:

    #     for index, parameter in enumerate(input):
    #         if index < num_oscillators:
    #             input[index] = parameter * 10
    #         elif index < num_oscillators + num_oscillators**2:
    #             input[index] = parameter
    #         else:
    #             input[index] = parameter * 2 * np.pi

    predict_actual = np.array([run(p) for p in predict_points])

    print(f"Time to generate predict_points: {time.time() - time_start}")

    means = gp_tuned(dr_tuned(predict_points))

    for pp, m, a in zip(predict_points, means, predict_actual):
        # print("Target point: {} Predicted mean: {} Actual mean: {}".format(pp, m, a))
        print("Predicted mean: {} Actual mean: {}".format(m, a))

    print(dr_tuned(predict_points))


if __name__ == "__main__":
    main()
