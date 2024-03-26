import numpy as np
import matplotlib.pyplot as plt

plt.figure("Normal vs Poisson")
def poisson_distribution(lam, max_i):
    values = [np.exp(-lam)]

    for i in range(1, max_i+1):
        factor = lam / i
        values.append(values[-1]*factor)

    return values

normal_support = np.linspace(-10, 100, 501)

def normal_distribution(mu, sigma):
    return np.exp(-0.5*((normal_support-mu)/sigma)**2) / (np.sqrt(2*np.pi)*sigma)

rate_parameters = [0.125, 0.25, 0.5, 1, 2, 4, 8,16,32]

for i, rate in enumerate(rate_parameters):
    plt.subplot(3,3,i+1)
    # plt.plot(normal_support, normal_distribution(rate, rate))
    plt.fill_between(normal_support, 0*normal_support, normal_distribution(rate, np.sqrt(rate)), alpha=0.5)

    max_n = int(2*rate + 5)

    max_p = np.max([
            np.max(poisson_distribution(rate, max_n - 1)),
            1/np.sqrt(2*np.pi*rate)
        ]) * 1.05

    plt.scatter(np.arange(max_n*2), poisson_distribution(rate, max_n*2-1), s=4, color='k')
    for x, y in zip(np.arange(max_n*2), poisson_distribution(rate, max_n*2-1)):
        plt.vlines(x, 0, y, 'k')


    plt.xlim([-4, max_n])
    plt.ylim([0, max_p])

    plt.xticks([])
    plt.yticks([])

    plt.title(r"$\lambda = %g$"%rate)

plt.show()