import numpy as np
import matplotlib.pyplot as plt

error_data = {}

for biased in [True, False]:

    title_prefix = "Efficient" if biased else "Unbiased"

    plt.figure(f"{title_prefix} - Sampling Distribution")

    n_values = np.array([5, 10, 15, 20, 25]) # Sample Size
    bin_edges = np.linspace(0, 2, 31)

    mean_variances = []
    errors = []

    for n in n_values:
        variances = []
        for i in range(100_000):
            sample = np.random.randn(n) # Mean of 0, variance of 1
            mean = np.mean(sample)
            if biased:
                variance = np.sum((sample - mean)**2)/n
            else:
                variance = np.sum((sample - mean)**2)/(n-1)
            variances.append(variance)

        mean_variances.append(np.mean(variances))
        errors.append(np.sum((np.array(variances) - 1)**2)/n)

        hist_data, bins = np.histogram(variances, bins=bin_edges)
        bin_centres = 0.5*(bins[1:] + bins[:-1])

        plt.plot(bin_centres, hist_data)

    plt.legend([str(n) for n in n_values])

    # Show estimator mean against n
    plt.figure(f"{title_prefix} - Estimator Variance")
    plt.plot(n_values, mean_variances)

    if biased:
        plt.plot(n_values, (n_values-1)/n_values)
        plt.legend(["Mean of Variance Estimator", "(n-1)/n"])
    else:
        plt.ylim([0, 2])

    plt.xlabel("n")
    plt.ylabel("Mean of Variance Estimate")
    plt.xticks(n_values)

    error_data[biased] = errors


plt.figure("Error Comparison")
plt.plot(n_values, error_data[True])
plt.plot(n_values, error_data[False])
plt.xlabel("n")
plt.ylabel("Mean Squared Error")
plt.xticks(n_values)

plt.legend(["Efficient Estimator", "Unbiased Estimator"])

plt.show()
