import sampling_distribution as sd
import scipy.stats as stats

population = stats.norm().rvs(100_000)
sample_size = 10
n_samples = 10_000

normal_population = sd.SamplingDistributionXBar(population, sample_size, n_samples)
normal_population.plot()