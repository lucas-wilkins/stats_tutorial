import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
from collections import defaultdict

plt.figure("Poisson Fractions")

def poisson_distribution(lam, max_i):
    values = [np.exp(-lam)]

    for i in range(1, max_i+1):
        factor = lam / i
        values.append(values[-1]*factor)

    return values

max_i = 100

x = poisson_distribution(10, max_i)
y = poisson_distribution(20, max_i)

data = defaultdict(float)

for i, pi in enumerate(x):
    for j_minus_1, pj in enumerate(y[1:]):
        j = j_minus_1 + 1
        fraction = Fraction(i, j)

        data[fraction] += pi*pj

for x in data:
    p = data[x]

    plt.vlines(x, 0, p, color='k')

plt.xlim([0, 2.5])
plt.ylim([0, 0.06])

plt.xlabel("x/y")
plt.ylabel("p(x/y)")

plt.show()