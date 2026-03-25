# Financial Applications of ANOVA

## Overview

Portfolio managers, risk analysts, and quantitative researchers routinely need to determine whether observed differences in returns, volatilities, or risk metrics across groups are statistically significant or merely due to sampling variability. The ANOVA framework developed in the preceding sections provides exactly this capability: it tests whether the mean of a continuous financial variable differs across two or more categories while controlling the Type I error rate. This section applies one-way and two-way ANOVA to four common financial analysis tasks and illustrates the complete workflow with a worked example.

## Comparing Portfolio Returns

A fundamental question in portfolio management is whether different investment strategies produce meaningfully different average returns. Suppose an analyst manages $k$ portfolios, each following a distinct strategy (e.g., value, momentum, low-volatility), and observes monthly returns over $T$ months. The one-way ANOVA tests

$$
H_0: \mu_1 = \mu_2 = \cdots = \mu_k \quad \text{vs.} \quad H_1: \text{at least one } \mu_i \text{ differs}
$$

where $\mu_i$ denotes the population mean monthly return of strategy $i$. If each strategy's returns are approximately normally distributed with equal variance across strategies, the F-statistic

$$
F = \frac{\text{MSB}}{\text{MSW}} = \frac{\text{SSB}/(k-1)}{\text{SSW}/(N-k)}
$$

follows an $F(k-1, \, N-k)$ distribution under $H_0$, where $N = kT$ is the total number of return observations.

!!! warning "Heteroscedasticity in financial returns"
    Financial return series frequently exhibit unequal variances across strategies or time periods (volatility clustering). When Levene's test rejects the equal-variance assumption, Welch's ANOVA should be used instead of the classical F-test. See [Welch's One-Way ANOVA](../anova_welch/welch_one_way.md) for the adjusted procedure.

## Sector Analysis

Equity analysts often ask whether mean returns differ across industry sectors (e.g., technology, healthcare, energy, financials). Here the groups are defined by sector classification, and each observation is the return of a stock belonging to that sector over a specified period. The one-way ANOVA hypothesis is

$$
H_0: \mu_{\text{tech}} = \mu_{\text{health}} = \mu_{\text{energy}} = \cdots \quad \text{vs.} \quad H_1: \text{at least one sector mean differs}
$$

A significant F-test indicates that sector membership explains a meaningful portion of return variability. Post-hoc pairwise comparisons (e.g., Tukey HSD) then identify which specific sector pairs exhibit significantly different mean returns. The effect size $\eta^2 = \text{SSB}/\text{SST}$ quantifies the proportion of total return variability attributable to sector differences.

## Factor Model Testing

In the Fama-French framework, stocks are sorted into portfolios based on characteristics such as size (market capitalization) and value (book-to-market ratio). To test whether size explains cross-sectional variation in returns, an analyst forms $k$ size-sorted portfolios and applies one-way ANOVA to their mean returns. The null hypothesis is that all size groups earn the same average return:

$$
H_0: \mu_{\text{small}} = \mu_{\text{mid}} = \mu_{\text{large}}
$$

Rejection of $H_0$ provides evidence for a size premium. When testing two characteristics simultaneously (e.g., size and value), two-way ANOVA is appropriate, with the model

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

where $\alpha_i$ represents the size effect, $\beta_j$ represents the value effect, and $(\alpha\beta)_{ij}$ captures any interaction between size and value. A significant interaction term indicates that the value premium depends on firm size (or vice versa).

## Trading Strategy Evaluation Across Market Regimes

A trader may want to evaluate whether a strategy's performance varies across market regimes (e.g., bull, bear, and sideways markets). This naturally calls for a two-way ANOVA with Factor A = trading strategy and Factor B = market regime. The model is

$$
Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

where $Y_{ijk}$ is the return of strategy $i$ during regime $j$ in period $k$, $\alpha_i$ is the main effect of strategy $i$, $\beta_j$ is the main effect of regime $j$, and $(\alpha\beta)_{ij}$ captures the strategy-regime interaction. The three hypothesis tests of interest are:

- **Strategy main effect**: $H_0: \alpha_1 = \alpha_2 = \cdots = 0$ (strategies perform equally on average)
- **Regime main effect**: $H_0: \beta_1 = \beta_2 = \cdots = 0$ (regimes do not affect average returns)
- **Interaction**: $H_0: (\alpha\beta)_{ij} = 0$ for all $i, j$ (strategy performance does not depend on regime)

A significant interaction is particularly informative: it reveals that some strategies thrive in specific regimes while underperforming in others, which has direct implications for regime-aware portfolio allocation.

## Worked Example: Comparing Three Fund Returns

!!! example "One-way ANOVA for fund comparison"
    An analyst collects 12 monthly returns (in percent) for three mutual funds:

    | Fund A | Fund B | Fund C |
    |--------|--------|--------|
    | 1.2 | 0.8 | 2.1 |
    | 0.9 | 1.1 | 1.8 |
    | 1.5 | 0.6 | 2.4 |
    | 1.1 | 0.9 | 1.9 |

    **Step 1: Compute group means and the overall mean.**

    $\bar{Y}_A = 1.175$, $\bar{Y}_B = 0.850$, $\bar{Y}_C = 2.050$, $\bar{Y} = 1.358$

    **Step 2: Compute sums of squares.**

    $$
    \text{SSB} = \sum_{i=1}^{3} n_i (\bar{Y}_i - \bar{Y})^2 = 4[(1.175 - 1.358)^2 + (0.850 - 1.358)^2 + (2.050 - 1.358)^2] \approx 2.977
    $$

    $$
    \text{SSW} = \sum_{i=1}^{3} \sum_{j=1}^{4} (Y_{ij} - \bar{Y}_i)^2 \approx 0.345
    $$

    **Step 3: Compute mean squares and F-statistic.**

    $$
    \text{MSB} = \frac{2.977}{3 - 1} = 1.489, \quad \text{MSW} = \frac{0.345}{12 - 3} = 0.0383
    $$

    $$
    F = \frac{1.489}{0.0383} \approx 38.8
    $$

    **Step 4: Decision.** With $k - 1 = 2$ and $N - k = 9$ degrees of freedom, the critical value at $\alpha = 0.05$ is $F_{0.05, 2, 9} \approx 4.26$. Since $38.8 \gg 4.26$, we reject $H_0$ and conclude that the three funds have significantly different mean returns.

    **Step 5: Post-hoc analysis.** A Tukey HSD test would then identify which specific fund pairs differ. Given the group means, Fund C's substantially higher average return is likely the primary driver of the significant F-test. See [Tukey HSD](../post_hoc/tukey.md) for the pairwise comparison procedure.
