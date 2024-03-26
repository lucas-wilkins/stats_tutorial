import numpy as np

import matplotlib.pyplot as plt

for condition in ["Low Rate", "High Rate"]:

    if condition == "Low Rate":
        min_rate = 0
        max_rate = 10
        error_scale = 1

    elif condition == "High Rate":
        min_rate = 10
        max_rate = 100
        error_scale = 10


    rate_parameter = np.linspace(min_rate, max_rate, 1001)

    plt.figure(f"{condition} Poisson Variance Error Bars Percentiles")

    error_bottom = rate_parameter - error_scale/np.sqrt(rate_parameter)
    error_top = rate_parameter + error_scale/np.sqrt(rate_parameter)

    plt.plot(rate_parameter, rate_parameter)
    plt.fill_between(rate_parameter, error_bottom, error_top, alpha=0.5)

    plt.xlim([min_rate, max_rate])
    plt.ylim([min_rate, max_rate])

    plt.xlabel("Rate parameter")
    plt.ylabel("Mean rate estimate")

    plt.figure(f"{condition} Poisson Proxy Error Bars Percentiles")

    rate_parameter = np.linspace(0,max_rate, 11)

    estimates = np.array([np.random.poisson(lam=rate) for rate in rate_parameter])

    error_bottom = estimates - error_scale/np.sqrt(estimates)
    error_top = estimates + error_scale/np.sqrt(estimates)

    zeros = estimates == 0
    error_bottom[zeros] = 0.0
    error_top[zeros] = 1000000000

    plt.plot(rate_parameter, estimates)
    plt.fill_between(rate_parameter, error_bottom, error_top, alpha=0.5)

    plt.xlim([min_rate, max_rate])
    plt.ylim([min_rate, max_rate])

    plt.xlabel("Rate parameter")
    plt.ylabel("Mean rate estimate")

plt.show()