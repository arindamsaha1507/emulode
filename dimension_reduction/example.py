import mogp_emulator
import numpy as np

# simple Dimension Reduction examples

# simulator function -- returns a single "important" dimension from
# at least 4 inputs


def f(x):
    return (x[0] - x[1] + 2.0 * x[3]) / 3.0


# Experimental design -- create a design with 5 input parameters
# all uniformly distributed over [0,1].

ed = mogp_emulator.LatinHypercubeDesign(5)

# sample space

inputs = ed.sample(100)

# run simulation

targets = np.array([f(p) for p in inputs])

###################################################################################

# First example -- dimension reduction given a specified number of dimensions
# (note that in real life, we do not know that the underlying simulation only
# has a single dimension)

print("Example 1: Basic Dimension Reduction")

# create DR object with a single reduced dimension (K = 1)

dr = mogp_emulator.gKDR(inputs, targets, K=3)

print(dr)

# use it to create GP

gp = mogp_emulator.fit_GP_MAP(dr(inputs), targets)

# create 5 target points to predict

predict_points = ed.sample(5)
predict_actual = np.array([f(p) for p in predict_points])

means = gp(dr(predict_points))

for pp, m, a in zip(predict_points, means, predict_actual):
    print("Target point: {} Predicted mean: {} Actual mean: {}".format(pp, m, a))
