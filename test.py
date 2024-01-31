from dgpsi import dgp, kernel, combine, lgp, path, emulator
import numpy as np
import matplotlib.pyplot as plt

# Draw some data points
n = 10
X = np.linspace(0, 1.0, n)[:, None]
f = lambda x: -1.0 if x < 0.5 else 1.0
Y = np.array([f(x) for x in X]).reshape(-1, 1)
Xt = np.linspace(0, 1.0, 200)[:, None]
Yt = np.array([f(x) for x in Xt]).flatten()
plt.plot(Xt, Yt)
plt.scatter(X, Y, color="r")

# Construct a three-layered DGP structure
layer1 = [kernel(length=np.array([1.0]), name="sexp")]
layer2 = [kernel(length=np.array([1.0]), name="sexp")]
layer3 = [kernel(length=np.array([1.0]), name="sexp", scale_est=True)]
all_layer = combine(layer1, layer2, layer3)
m = dgp(X, [Y], all_layer)
