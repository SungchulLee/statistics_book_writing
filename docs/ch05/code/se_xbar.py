import matplotlib.pyplot as plt
import numpy as np

# Fix the random seed for reproducibility
np.random.seed(0)

def main():
    # -------------------------------------------------------------------------
    # Step 1: Generate the sampling distribution of the sample mean (X_bar)
    # -------------------------------------------------------------------------
    X_bar = []
    for _ in range(10_000):
        # Draw a random sample of size 5 from Uniform(0,1)
        x = np.random.uniform(size=(5,))
        # Compute its sample mean
        x_bar = x.mean()
        # Collect the result
        X_bar.append(x_bar)

    # -------------------------------------------------------------------------
    # Step 2: Compute the estimated mean and standard error
    # -------------------------------------------------------------------------
    average = np.array(X_bar).mean()   # This approximates the true population mean μ = 0.5
    standard_error = np.array(X_bar).std()  # Empirical standard error of the sample mean

    print(f'(Estimated) Mean of X_bar : {average:.4}')
    print(f'Standard Error   of X_bar : {standard_error:.4}')

    # -------------------------------------------------------------------------
    # Step 3: Visualize the sampling distribution
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_title("Sampling Distribution of X_bar", fontsize=20)

    # Histogram of simulated sample means
    ax.hist(X_bar, bins=100, density=True, alpha=0.3)

    # Vertical line at the estimated mean
    ax.vlines(average, ymin=0, ymax=5, alpha=1.0, color='k', ls='-', lw=5)

    # Vertical dashed lines at mean ± standard error
    ax.vlines(average + standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')

    # -------------------------------------------------------------------------
    # Step 4: Annotate the standard error visually with a double-headed arrow
    # -------------------------------------------------------------------------
    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(
        text='',
        xy=(average, 5),
        xytext=(average + standard_error, 5),
        arrowprops=arrowprops
    )
    ax.annotate(
        text='Standard Error',
        xy=(average, 5.5),
        xytext=(average, 5.5),
        fontsize=15
    )

    # -------------------------------------------------------------------------
    # Step 5: Adjust plot aesthetics
    # -------------------------------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 6)

    # Remove top and right spines for a cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Show the plot
    plt.show()

if __name__ == "__main__":
    main()