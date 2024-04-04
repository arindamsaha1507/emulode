import numpy as np
import matplotlib.pyplot as plt
import nolds
import warnings

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Henon map
def henon_map(x, y, a, b):
    return 1 - a * x**2 + y, b * x

# Parameters
a_values = np.linspace(0.9, 1.4, 500)  # Range of a values
b = 0.3  # Fixed value of b

# Store Lyapunov exponents
lyap_exponents = []

# Iterate over a values
for a in a_values:
    x, y = 0.1, 0.1  # Initial conditions
    n_iterations = 5000  # Total number of iterations

    # Generate time series
    time_series = []
    for _ in range(n_iterations):
        x, y = henon_map(x, y, a, b)
        time_series.append(x)

    # Calculate the largest Lyapunov exponent using Rosenstein's algorithm
    lyap_exp = nolds.lyap_r(np.array(time_series), emb_dim=2, lag=1, min_tsep=10, tau=1, min_neighbors=20, trajectory_len=20, fit='RANSAC')
    lyap_exponents.append(lyap_exp)

# Plot the Lyapunov exponent as a function of a
plt.plot(a_values, lyap_exponents)
plt.xlabel('a')
plt.ylabel('Largest Lyapunov Exponent')
plt.title('Largest Lyapunov Exponent vs a (Henon Map)')
plt.grid(True)
plt.show()
