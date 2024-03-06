"""Module for testing the MOGP model."""

from matplotlib import pyplot as plt
import mogp_emulator
import numpy as np
from scipy.stats import qmc


def simulator(x):
    """Simulator function."""
    return np.sin(2 * np.pi * x)


sampler = qmc.LatinHypercube(d=1)
sample = sampler.random(n=10)
inputs = qmc.scale(sample, -1, 1).flatten()
inputs.sort()

print(inputs)
targets = [simulator(x) for x in inputs]
print(targets)

# gp = mogp_emulator.GaussianProcess(inputs, targets)
gp = mogp_emulator.fit_GP_MAP(inputs, targets, kernel="Matern52")

predict_points = np.linspace(-1, 1, 100)
means, variances, derivs = gp.predict(predict_points)

print(means)

plt.plot(predict_points, simulator(predict_points), label="True")
plt.plot(predict_points, means, label="Mean")
plt.fill_between(
    predict_points,
    means - 1.96 * np.sqrt(variances),
    means + 1.96 * np.sqrt(variances),
    alpha=0.2,
    color="C1",
)
plt.plot(inputs, targets, "o", color="r", label="Data")

plt.legend()
plt.savefig("mogp_test.png")

print(derivs)
