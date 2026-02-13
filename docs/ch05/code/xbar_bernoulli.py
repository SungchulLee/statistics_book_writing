import sampling_distribution as sd
import scipy.stats as stats

population = stats.binom(n=1, p=0.4).rvs(100_000)
sample_size = 1_000
n_samples = 10_000

coin_flip_population = sd.SamplingDistributionXBar(population, sample_size, n_samples)
coin_flip_population.plot()