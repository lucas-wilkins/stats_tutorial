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
        error_scale = 1


    rate_parameter = np.linspace(min_rate, max_rate, 1001)

    plt.figure(f"{condition} Poisson Variance Error Bars")

    error_bottom = rate_parameter - error_scale*np.sqrt(rate_parameter)
    error_top = rate_parameter + error_scale*np.sqrt(rate_parameter)

    plt.plot(rate_parameter, rate_parameter)
    plt.fill_between(rate_parameter, error_bottom, error_top, alpha=0.5)

    plt.xlim([min_rate, max_rate])
    plt.ylim([min_rate, max_rate])

    plt.xlabel("Rate parameter")
    plt.ylabel("Mean rate estimate")

    # Plot on next figure too
    plt.figure(f"{condition} Poisson Proxy Error Bars")

    plt.plot(rate_parameter, rate_parameter)
    plt.fill_between(rate_parameter, error_bottom, error_top, alpha=0.5)



    rate_parameter = np.linspace(0,max_rate, 21)

    estimates = np.array([np.random.poisson(lam=rate) for rate in rate_parameter])

    error_bottom = estimates - error_scale*np.sqrt(estimates)
    error_top = estimates + error_scale*np.sqrt(estimates)


    plt.plot(rate_parameter, estimates)
    plt.fill_between(rate_parameter, error_bottom, error_top, alpha=0.5)

    plt.xlim([min_rate, max_rate])
    plt.ylim([min_rate, max_rate])

    plt.xlabel("Rate parameter")
    plt.ylabel("Mean rate estimate")

plt.show()