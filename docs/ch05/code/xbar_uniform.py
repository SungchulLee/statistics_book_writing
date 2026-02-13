import matplotlib.pyplot as plt
import numpy as np

# Set a random seed for reproducibility
np.random.seed(1)

# Define parameters for the population, sample size, and number of samples for the simulation
sample_size = 5        # Size of a single random sample
n_samples = 10_000     # Number of samples to draw for the sampling distribution
n_population = 10_000 # Size of the population to simulate

def plot_distributions():
    """
    Generates a plot showing the population distribution, sample distribution, and sampling distribution.

    - Population Distribution: A histogram of the entire population.
    - Sample Distribution: A scatter plot showing a single sample drawn from the population.
    - Sampling Distribution: A histogram of the means of multiple random samples drawn from the population.
    """
    # Generate a large population from a uniform distribution
    population = np.random.uniform(size=(n_population,))

    # Generate a single random sample from the population
    single_sample = np.random.choice(population, size=sample_size, replace=False)

    # Generate multiple samples from the population and compute their means to form the sampling distribution
    sample_means = [np.mean(np.random.choice(population, size=sample_size, replace=False)) for _ in range(n_samples)]

    # Create a 3-row subplot to compare the population, sample, and sampling distributions
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # Plot the population distribution
    ax0.hist(population, bins=np.linspace(0, 1, 100))
    ax0.set_title('Population Distribution', fontsize=20)

    # Plot the sample distribution (scatter plot)
    ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
    ax1.set_title(f'Sample Distribution of {sample_size} Samples', fontsize=20)

    # Plot the sampling distribution (histogram of sample means)
    ax2.hist(sample_means, bins=np.linspace(0, 1, 100))
    ax2.set_title('Sampling Distribution of $\\bar{X}$', fontsize=20)

    # Adjust the aesthetics of the plots
    for ax in (ax0, ax1, ax2):
        ax.spines['left'].set_visible(False)  # Hide the left spine
        ax.spines['right'].set_visible(False) # Hide the right spine
        ax.spines['top'].set_visible(False)   # Hide the top spine
        ax.spines['bottom'].set_position('zero') # Position the bottom spine at the zero level
        ax.set_yticks([])  # Remove the y-axis ticks for a cleaner look

    # Show the plot with all subplots aligned
    plt.tight_layout()
    plt.show()

# Call the function to plot the distributions when the script is run
if __name__ == "__main__":
    plot_distributions()