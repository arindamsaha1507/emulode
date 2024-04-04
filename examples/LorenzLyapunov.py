import numpy as np
from scipy.integrate import solve_ivp
import nolds#library for calculating Lyapunov exponents

import matplotlib.pyplot as plt

# Lorenz system
def lorenz(t, X, sigma, rho, beta):
    x, y, z = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Parameters
sigma, rho, beta = 10, 28, 8/3

# Initial conditions
X0 = [1, 1, 1]

# Integrate the Lorenz equations
t_span = (0, 100)
t_eval = np.linspace(*t_span, 10000)
sol = solve_ivp(lorenz, t_span, X0, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')

# Extract the solution
x, y, z = sol.y

# Plot the solution
# plt.plot(sol.t, x)
# plt.xlabel('Time')
# plt.ylabel('X')
# plt.title('Lorenz Oscillator')
# plt.show()

# Calculate Lyapunov exponents (using Rosenstein's algorithm or other methods)
# This part is more complex and requires a separate implementation.
# You can use existing libraries like nolds (https://github.com/CSchoel/nolds) for this purpose.

# Calculate the largest Lyapunov exponent from the x-component of the Lorenz system
lyap_exp = nolds.lyap_r(x, emb_dim=3)
print('Largest Lyapunov exponent (Lorenz):', lyap_exp)

