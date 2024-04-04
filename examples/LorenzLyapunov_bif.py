import numpy as np
from scipy.integrate import solve_ivp
import nolds
import matplotlib.pyplot as plt
import warnings

# Suppress specific runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nolds.measures")

# Lorenz system
def lorenz(t, X, sigma, rho, beta):
    x, y, z = X
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

# Parameters
sigma, beta = 10, 8/3
rho_values = np.linspace(5, 40, 100)  # Range of rho values

# Initial conditions
X0 = [1, 1, 1]

# Store Lyapunov exponents
lyap_exponents = []

# Iterate over rho values
for rho in rho_values:
    # Integrate the Lorenz equations
    t_span = (0, 100)
    t_eval = np.linspace(*t_span, 10000)
    sol = solve_ivp(lorenz, t_span, X0, args=(sigma, rho, beta), t_eval=t_eval, method='RK45')

    # Extract the solution
    x, y, z = sol.y

    # Calculate the largest Lyapunov exponent
    lyap_exp = nolds.lyap_r(x, emb_dim=3)
    lyap_exponents.append(lyap_exp)

# Plot the Lyapunov exponent as a function of rho
plt.plot(rho_values, lyap_exponents)
plt.xlabel('Rho')
plt.ylabel('Largest Lyapunov Exponent')
plt.title('Largest Lyapunov Exponent vs Rho (Lorenz System)')
plt.grid(True)
plt.show()
