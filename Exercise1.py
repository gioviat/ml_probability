# Imports

import argparse
import math
import random
import numpy as np
import time
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rand_gaussian(mu, sigma, n = 0):

	def draw_sample():

		# TODO: Draw a random number from the Gaussian distribution.

		return 0

	if n > 0:
		return [draw_sample() for _ in range(n)]

	return draw_sample()


def normal_pdf(x, mu, sigma):

	# TODO: Implement the Gaussian PDF N(mu, sigma ** 2)

	return 0


def sample_mean(x):

	# TODO: Implement the sample mean

	return 0


def sample_std(sigma, n):

	# TODO: Implement the sample standard deviation

	return 0


def inv_cdf(area):

	# TODO: Implement an approximation of the cumulative distribution function to get the inverse

	return 0


def posterior_parametrization(x, sigma_pop, mu_prior, sigma_prior):

	# TODO: Implement the closed-form parametrization of the posterior N(mean_post, sigma_post ** 2)

	mean_post = 0
	sigma_post = 0

	return mean_post, sigma_post


def CI(mean, err, z):

	# TODO: Return the upper and lower bounds of the confidence / credible interval

	lower_bound = 0
	upper_bound = 0

	return [lower_bound, upper_bound]


def init():

	# Variables

	res = args.plot_resolution

	# Inverse cumulative density function

	z = inv_cdf(args.alpha)

	# The true distribution of the population: N(mu_pop, sigma_pop ** 2)
	# Assume that sigma is known and mu is unknown.

	mu_pop = args.mu_pop
	sigma_pop = args.sigma_pop

	# Prior distribution of mu ~ N(mu_prior, sigma_prior ** 2)

	mu_prior = args.mu_prior
	sigma_prior = args.sigma_prior

	# Sampled observations

	samples_pop = []

	# Init plot

	ax = plt.gca()

	# Sampling loop

	for i in range(500):
		# Clear previous plot

		plt.cla()

		# Add a new sample to the observations

		samples_pop.append(rand_gaussian(mu_pop, sigma_pop))

		# Compute sample mean and standard deviation

		s_mean = sample_mean(samples_pop)
		s_std = sample_std(sigma_pop, len(samples_pop))

		# Compute the frequentist interval

		ci = CI(s_mean, s_std, z)

		# Compute posterior parametrization

		mu_posterior, sigma_posterior = posterior_parametrization(samples_pop, sigma_pop, mu_prior, sigma_prior)

		# Compute the credible interval

		posterior_ci = CI(mu_posterior, math.sqrt(sigma_posterior), z)

		# Plot samples from the population

		ax.hist(samples_pop, density = True, label = 'Samples from population')

		# Plot the true mean

		ax.axvline(x = mu_pop, label = 'pPopulation mu', color = 'r')

		# Plot frequentist interval

		ax.axvline(x = ci[0], label = 'Lower confidence interval', color = 'c')
		ax.axvline(x = ci[1], label = 'Upper confidence interval', color = 'c')

		# Plot prior

		min_x = mu_prior - sigma_prior * 2
		max_x = mu_prior + sigma_prior * 2
		x = [min_x + i * (max_x - min_x) / res for i in range(res)]
		y = [normal_pdf(xi, mu_prior, sigma_prior) for xi in x]
		ax.plot(x, y, label = 'Prior f(mu)')

		# Plot posterior

		min_x = mu_posterior - 1 * 4
		max_x = mu_posterior + 1 * 4
		x = [min_x + i * (max_x - min_x) / res for i in range(res)]
		y = [normal_pdf(xi, mu_posterior, sigma_posterior) for xi in x]
		ax.plot(x, y, label = 'Posterior f(x|mu)')

		# Redraw

		ax.legend(loc='upper right')
		plt.axis(args.plot_boundaries)
		plt.title(args.title)
		plt.draw()
		plt.pause(1e-17)
		time.sleep(0.001)

	plt.show()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	# Input Arguments

	parser.add_argument('--title',
						default = 'Ex1: Confidence and Credible Intervals',
						required = False)

	parser.add_argument('--alpha',
						default = 0.99,
						required = False)

	parser.add_argument('--mu_pop',
						default = 10,
						required = False)

	parser.add_argument('--sigma_pop',
						default = 3,
						required = False)

	parser.add_argument('--mu_prior',
						default = 14,
						required = False)

	parser.add_argument('--sigma_prior',
						default = 10,
						required = False)

	parser.add_argument('--plot-resolution',
						default = 100,
						required = False)

	parser.add_argument('--plot-boundaries',
						default = [4, 16, 0, 0.8],  # min_x, max_x, min_y, max_y
						required = False)

	parser.add_argument('--font-size',
						default = 10,
						required = False)

	args = parser.parse_args()

	init()

