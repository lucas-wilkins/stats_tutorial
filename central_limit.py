import numpy as np
import matplotlib.pyplot as plt

plt.figure("Uniform Distribution")

plt.plot([-1,0,0,1,1,2], [0,0,1,1,0,0])
plt.xlabel("x")
plt.ylabel("Probability Density")


# Simulate random numbers between 0 and 1


# Show histogram
plt.figure("Uniform Distribution Sample")
distribution = np.random.rand(100_000)
plt.hist(distribution, bins=np.linspace(-1, 2, 301))

plt.xlabel("x")
plt.ylabel("Frequency Density")


plt.figure("Uniform Mean Sampling Distribution")

# Simulate sampling distribution for mean
means = []
for i in range(10_000):
    sample = np.random.rand(1000)
    means.append(np.mean(sample))

plt.xlabel(r"$\hat\mu$")
plt.ylabel("Probability Density")

print("Variance of means:", np.var(means))

plt.hist(means, bins=30)
plt.show()