"""Module to reduce the dimension of a network of Kuramoto oscillators."""

import time
import mogp_emulator
import numpy as np
from scipy.stats import qmc

from dimension_reduction.vdp import multiple_runs


def main():
    """Run the main function."""

    print("Dimension reduction")
    print("===================")

    num_oscillators = 50

    time_start = time.time()

    # ed = mogp_emulator.LatinHypercubeDesign(length)

    sampler = qmc.LatinHypercube(d=num_oscillators)
    sample = sampler.random(n=500)
    inputs = qmc.scale(sample, 0, 1)

    print(inputs)

    # exit()

    print(len(inputs))

    targets = multiple_runs(inputs)

    print(targets)
    # exit()

    print(f"Time to generate sampling_points: {time.time() - time_start}")

    time_start = time.time()

    dr_tuned = mogp_emulator.gKDR(inputs, targets, K=4)

    # dr_tuned, loss = mogp_emulator.gKDR.tune_parameters(
    #     inputs,
    #     targets,
    #     mogp_emulator.fit_GP_MAP,
    #     cXs=[1.0, 0.5, 3.0],
    #     cYs=[1.0, 0.5, 3.0],
    # )

    print(f"Number of inferred dimensions is {dr_tuned.K}")
    # print(f"Loss is {loss}")

    print(f"Time to tune parameters: {time.time() - time_start}")

    time_start = time.time()

    gp_tuned = mogp_emulator.fit_GP_MAP(dr_tuned(inputs), targets)

    print(f"Time to fit GP: {time.time() - time_start}")

    time_start = time.time()

    predict_points = qmc.scale(sampler.random(n=30), 0, 1)

    predict_actual = multiple_runs(predict_points)

    print(f"Time to generate predict_points: {time.time() - time_start}")

    means = gp_tuned(dr_tuned(predict_points))

    mean_error = np.mean(np.abs(means - predict_actual))

    for m, a in zip(means, predict_actual):
        print(f"Predicted mean: {m} Actual mean: {a} Error: {abs(m - a)}")

    print(f"Mean error: {mean_error}")


if __name__ == "__main__":
    main()
