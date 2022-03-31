# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.InterpolatedUnivariateSpline.html

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline

rng = np.random.default_rng()

x = np.linspace(-3, 3, 10)

y = np.exp(-x**2) + 0.1 * rng.standard_normal(10)

spl = InterpolatedUnivariateSpline(x, y)

plt.plot(x, y, 'ro', ms=5)

xs = np.linspace(-3, 3, 1000)

plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)

plt.show()