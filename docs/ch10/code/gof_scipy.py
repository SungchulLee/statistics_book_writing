#!/usr/bin/env python3
# ======================================================================
# 16_gof_02_scipy_equal_expected_3cats.py
# ======================================================================
# Chi-square GOF using scipy.stats.chisquare with equal expected counts.
# ======================================================================

from scipy import stats

# Observed frequencies for each outcome: Win, Loss, Tie
observed_frequencies = [4, 13, 7]

# Expected frequencies assuming an even distribution
total_games = sum(observed_frequencies)
expected_frequencies = [total_games / 3] * 3

# Perform the chi-square goodness-of-fit test
chi_square_statistic, p_value = stats.chisquare(f_obs=observed_frequencies, f_exp=expected_frequencies)

# Output results
print(f"{chi_square_statistic = }")
print(f"{p_value = }")
