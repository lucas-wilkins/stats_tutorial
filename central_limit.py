import numpy as np
import matplotlib.pyplot as plt

n = 1000

# Simulate random numbers between 0 and 1
distribution = np.random.rand(n)

# Show histogram
plt.hist(distribution)
plt.title("Distribution of values")
plt.show()

# Simulate sampling distribution for mean
means = []
for i in range(10_000):
    sample = np.random.rand(n)
    means.append(np.mean(sample))

plt.hist(means, bins=30)
plt.title("Distribution of estimated means")
plt.show()