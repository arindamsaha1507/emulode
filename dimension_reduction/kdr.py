"""Module to reduce the dimension of a network of Kuramoto oscillators."""

import time
import mogp_emulator
import numpy as np
from scipy.stats import qmc

from dimension_reduction.kuramoto_runner import run


def rescale_parameters(
    parameters,
    shift_frequency,
    shift_coupling,
    shift_phase,
    scale_frequency,
    scale_coupling,
    scale_phase,
):
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
            parameters[index] = parameter * scale_frequency + shift_frequency
        elif index < num_oscillators + num_oscillators**2:
            parameters[index] = parameter * scale_coupling + shift_coupling
        else:
            parameters[index] = parameter * scale_phase + shift_phase

    return parameters


def main():
    """Run the main function."""

    print("Dimension reduction")
    print("===================")

    num_oscillators = 3
    length = num_oscillators + num_oscillators**2 + num_oscillators

    shift_frequency = 0
    shift_coupling = 0
    shift_phase = 0

    rescale_frequency = 1
    rescale_coupling = 1
    rescale_phase = 2 * np.pi

    time_start = time.time()

    # ed = mogp_emulator.LatinHypercubeDesign(length)

    sampler = qmc.LatinHypercube(d=length)
    sample = sampler.random(n=200)
    inputs = qmc.scale(sample, 0, 1)

    print(len(inputs))

    inputs = [
        rescale_parameters(
            p,
            shift_frequency,
            shift_coupling,
            shift_phase,
            rescale_frequency,
            rescale_coupling,
            rescale_phase,
        )
        for p in inputs
    ]

    targets = np.array([run(p) for p in inputs])

    print(f"Time to generate sampling_points: {time.time() - time_start}")

    time_start = time.time()

    dr_tuned, loss = mogp_emulator.gKDR.tune_parameters(
        inputs,
        targets,
        mogp_emulator.fit_GP_MAP,
        cXs=[1.0, 0.5, 3.0],
        cYs=[1.0, 0.5, 3.0],
    )

    print(f"Number of inferred dimensions is {dr_tuned.K}")
    print(f"Loss is {loss}")

    print(f"Time to tune parameters: {time.time() - time_start}")

    time_start = time.time()

    gp_tuned = mogp_emulator.fit_GP_MAP(dr_tuned(inputs), targets)

    print(f"Time to fit GP: {time.time() - time_start}")

    time_start = time.time()

    predict_points = qmc.scale(sampler.random(n=1000), 0, 1)

    predict_points = [
        rescale_parameters(
            p,
            shift_frequency,
            shift_coupling,
            shift_phase,
            rescale_frequency,
            rescale_coupling,
            rescale_phase,
        )
        for p in predict_points
    ]

    predict_actual = np.array([run(p) for p in predict_points])

    print(f"Time to generate predict_points: {time.time() - time_start}")

    means = gp_tuned(dr_tuned(predict_points))

    for m, a in zip(means, predict_actual):
        print(
            f"Predicted mean: {m} Actual mean: {a} Error: {abs(m - a)} Percent error: {abs(m - a) / a * 100}"
        )


if __name__ == "__main__":
    main()
