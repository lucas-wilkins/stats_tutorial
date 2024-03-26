import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import loggamma

k_true = 4

x = np.linspace(0, np.pi, 21)
base_rate_parameters = np.sin(x)

rate_parameters = k_true * base_rate_parameters

plt.figure("Toy Model - Different k values")
for k in np.linspace(0.5*k_true, 1.5*k_true, 11):

    color = 'r' if k == k_true else 'k'
    plt.plot(x, k*base_rate_parameters, color=color)

plt.xlabel(r"$\theta_i$")
plt.ylabel("$x_i$")

plt.figure("Toy Model - Sample")
sample = np.array([np.random.poisson(rate) for rate in rate_parameters])
plt.scatter(x, sample)
plt.plot(x, rate_parameters)
plt.xlabel(r"$\theta_i$")
plt.ylabel("$x_i$")

def poisson_log_likilihood(lam, x):
    """ Negative log of the likelihood"""
    return lam + loggamma(x+1) - np.log(lam)*x

for i in range(50):

    sample = np.array([np.random.poisson(rate) for rate in rate_parameters])
    def correct_variance(k):
        model = k*np.sin(x)
        non_zero = model != 0
        return np.sum(((model[non_zero] - sample[non_zero])**2)/model[non_zero])

    def incorrect_variance(k):
        model = k*np.sin(x)
        non_zero = sample != 0
        return np.sum(((model[non_zero] - sample[non_zero])**2)/sample[non_zero])

    def log_liklihood_objective(k):
        model = k * np.sin(x)
        non_zero = model != 0
        log_liklihoods = [poisson_log_likilihood(m, s) for (m, s) in zip(model[non_zero], sample[non_zero])]
        return np.sum(log_liklihoods)


    correct = minimize(correct_variance, (1.0, )).x
    incorrect = minimize(incorrect_variance, (1.0, )).x
    like = minimize(log_liklihood_objective, (1.0,)).x

    print(correct, incorrect, like)

    plt.figure("Toy Model - Fitting - Variance Based")

    ax_correct = plt.plot(x, base_rate_parameters*correct, color='g', alpha=0.5)[0]
    ax_incorrect = plt.plot(x, base_rate_parameters*incorrect, color='r', alpha=0.5)[0]

    plt.figure("Toy Model - Fitting - Likilihood Based")

    ax_like = plt.plot(x, base_rate_parameters*like, color='b', alpha=0.5)[0]


plt.figure("Toy Model - Fitting - Variance Based")
ax_gt = plt.plot(x, rate_parameters, color='k')[0]
plt.xlabel(r"$\theta_i$")
plt.ylabel("$x_i$")
plt.legend([ax_correct, ax_incorrect, ax_gt], ["Model Variance", "Sample Variance", "Ground Truth"])

plt.figure("Toy Model - Fitting - Likilihood Based")
ax_gt = plt.plot(x, rate_parameters, color='k')[0]
plt.xlabel(r"$\theta_i$")
plt.ylabel("$x_i$")
plt.legend([ax_like, ax_gt], ["Liklihood Based Estimates", "Ground Truth"])

plt.show()