import matplotlib.pyplot as plt
from scipy import special

def prob(n, c, r, t):
    """
    Calculate the probability of capturing 't' tagged birds in a recapture
    sample of size 'r', given that there are 'n' birds in total.

    Parameters:
    - n: Total number of birds in the population
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - Probability of observing 't' tagged birds in the recapture sample.
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)

def capture_recapture(c=10, r=10, t=3):
    """
    Calculate the probability distribution over possible total population sizes
    and determine the MLE (Maximum Likelihood Estimate) for the population size.

    Parameters:
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    """
    prob_list = []

    # Calculate probability for each possible population size n
    for n in range(c + r - t, 10 * (c + r - t)):
        prob_list.append(prob(n, c, r, t))

    # Determine the MLE for the population size
    prob_max = max(prob_list)
    idx = prob_list.index(prob_max)
    mle_n = idx + (c + r - t)
    print(f'MLE n: {mle_n}')

    return prob_list, mle_n

def draw(prob_list, mle_n, c=10, r=10, t=3):
    """
    Plot the probability distribution of the total population size and highlight the MLE.

    Parameters:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    - c, r, t: Parameters for the capture-recapture model
    """
    idx = mle_n - (c + r - t)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(range(c + r - t, 10 * (c + r - t)), prob_list, label='Probability')
    ax.plot([mle_n, mle_n], [0, prob_list[idx]], 'o--r', label=f'MLE: {mle_n}')

    # Customize plot
    ax.set_xlabel('Total Population Size (n)')
    ax.set_ylabel('Probability')
    ax.set_title('Capture-Recapture MLE for Population Size')
    ax.legend()
    plt.show()

# Parameters for capture-recapture model
c = 5  # Birds captured and tagged in the first stage
r = 6  # Birds recaptured in the second stage
t = 2  # Tagged birds in the recapture stage

# Calculate probabilities and MLE
prob_list, mle_n = capture_recapture(c, r, t)

# Plot the probability distribution and highlight the MLE
draw(prob_list, mle_n, c, r, t)