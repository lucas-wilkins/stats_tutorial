import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import loggamma

k_true = 0.01

theta = np.linspace(0, np.pi, 21)
base_reflectance_parameters = np.sin(theta)

rate_in_true = 500 * (1 + 3*theta / np.pi)
rate_out_true = k_true * base_reflectance_parameters * rate_in_true

plt.figure("Quotient Toy Model - Lambdas")
plt.subplot(1,2,1)

plt.plot(theta, rate_in_true)
plt.plot(theta, rate_out_true)
plt.title("Incoming/Sample Rates")
plt.legend(["In", "Out"])

plt.xlabel(r"$\theta_i$")
plt.ylabel("$\lambda_i$")

plt.subplot(1,2,2)
plt.plot(theta, rate_out_true/rate_in_true, color='k')
plt.title("Reflectance")

plt.xlabel(r"$\theta_i$")
plt.ylabel("$r_i$")

plt.figure("Quotient Toy Model - Sample Example")

in_sample = np.array([np.random.poisson(rate) for rate in rate_in_true])
out_sample = np.array([np.random.poisson(rate) for rate in rate_out_true])

plt.scatter(theta, in_sample)
plt.scatter(theta, out_sample)


def poisson_log_likilihood(lam, x):
    """ Negative log of the likelihood"""
    return lam + loggamma(x+1) - np.log(lam)*x

for i in range(15):

    in_sample = np.array([np.random.poisson(rate) for rate in rate_in_true])
    out_sample = np.array([np.random.poisson(rate) for rate in rate_out_true])

    quotients = out_sample / in_sample

    # (A/B)**2 ((varA / A)**2 + (varB / B)**2)
    # varA = A, so this reduces to (A/B)**2
    quotient_variance = (quotients ** 2) * 2

    def variance_objective(k):
        model = k*np.sin(theta)
        non_zero = quotients != 0
        return np.sum(((model[non_zero] - quotients[non_zero])**2) / quotient_variance[non_zero])

    def log_liklihood_objective(parameters):


        k = parameters[0]

        # Likelihoods of incoming particles
        in_model = np.array(parameters[1:])
        non_zero = in_model != 0
        log_likelihood_in = [poisson_log_likilihood(m, s) for (m, s) in zip(in_model[non_zero], in_sample[non_zero])]

        r_model = k * np.sin(theta)
        out_model = in_model * r_model
        non_zero = out_model != 0

        log_likelihoods_out = [poisson_log_likilihood(m, s) for (m, s) in zip(out_model[non_zero], out_sample[non_zero])]

        return np.sum(log_likelihoods_out) + np.sum(log_likelihood_in)


    correct = minimize(
        variance_objective,
        x0 = np.array([1.0, ])
    ).x

    bounds = [(0,1)] + [(0,None) for _ in in_sample]

    like = minimize(
        log_liklihood_objective,
        x0 = np.array([1.0] + list(in_sample)),
        bounds=bounds).x

    # like = minimize(log_liklihood_objective, x0 = np.array([1.0] + [3.0 for _ in rate_in_true])).x
    # like = minimize(log_liklihood_objective, x0 = np.array([1.0] + list(rate_in_true))).x

    # print(correct, like)

    plt.figure("Quotient Toy Model - Fitting - Variance Based")

    ax_correct = plt.plot(theta, base_reflectance_parameters * correct, color='b', alpha=0.5)[0]

    plt.figure("Quotient Toy Model - Fitting - Likilihood Based")

    plt.subplot(1,2,1)
    ax_like_main = plt.plot(theta, like[1:], color='b', alpha=0.5)[0]

    plt.subplot(1,2,2)
    ax_like = plt.plot(theta, base_reflectance_parameters * like[0], color='b', alpha=0.5)[0]



plt.figure("Quotient Toy Model - Fitting - Variance Based")
ax_gt = plt.plot(theta, k_true * base_reflectance_parameters, color='k')[0]
plt.xlabel(r"$\theta_i$")
plt.ylabel("$x_i$")

plt.figure("Quotient Toy Model - Fitting - Likilihood Based")

plt.subplot(1,2,1)
plt.title("Incomming")
plt.plot(theta, rate_in_true, color='k')
plt.xlabel(r"$\theta_i$")
plt.ylabel("$\lambda_i$")

plt.subplot(1, 2, 2)
plt.title("Reflectance")
plt.plot(theta, k_true * base_reflectance_parameters, color='k')
plt.xlabel(r"$\theta_i$")
plt.ylabel("$r_i$")

plt.show()