# Survivorship Bias

Survivorship bias occurs when analysis focuses only on subjects that "survived" a selection process, ignoring those that did not. This systematically distorts conclusions by making success appear more common or more predictable than it actually is.

## Definition

**Survivorship bias** is a form of selection bias where the sample consists only of entities that passed through some filter (survived, succeeded, returned), while those that were filtered out (failed, died, were lost) are invisible to the analyst. Conclusions drawn from the surviving sample do not generalize to the full population.

## Explanation

The canonical example is Abraham Wald's analysis of WWII bomber damage. The military proposed adding armor where returning planes had the most bullet holes (wings and fuselage). Wald recognized that planes hit in the cockpit or engine never returned -- the surviving sample was missing exactly the fatal damage. The correct strategy was to armor the areas with few holes on survivors.

The same logic applies broadly:

- **Financial markets**: Studying only stocks in today's index ignores delisted failures, overstating average returns.
- **Startups**: Analyzing traits of successful companies (charismatic leaders, high R&D) while ignoring failed companies with the same traits leads to overconfidence in those factors.
- **Clinical research**: Reporting outcomes only for patients who completed a trial ignores dropouts who may have fared worse.

To avoid survivorship bias: include all cases (both successes and failures), consider base rates (if 90% of startups fail, shared traits among survivors may be coincidental), and examine the selection mechanism explicitly.

## Examples

```python
import numpy as np

np.random.seed(42)
n_funds = 1000
years = 10

# Simulate annual returns for hedge funds (many fail and close)
returns = np.random.normal(0.05, 0.15, (n_funds, years))
cumulative = np.cumprod(1 + returns, axis=1)

# Fund "dies" if cumulative return falls below 0.5 (50% drawdown)
survived = np.all(cumulative > 0.5, axis=1)

all_final = cumulative[:, -1]
survivor_final = all_final[survived]

print(f"Total funds: {n_funds}")
print(f"Survivors: {survived.sum()}")
print(f"Mean final value (all funds):      {all_final.mean():.3f}")
print(f"Mean final value (survivors only): {survivor_final.mean():.3f}")
print(f"Survivorship bias: {survivor_final.mean() - all_final.mean():+.3f}")
```

## Exercises

**Exercise 1.**
A financial advisor shows that their recommended mutual funds have averaged 12% annual returns over the past 10 years. Explain how survivorship bias could inflate this figure.

??? success "Solution to Exercise 1"
    Mutual funds that performed poorly over the past 10 years were likely closed, merged, or delisted. The advisor's current list of recommended funds only includes those that survived -- precisely because they performed well. Funds that lost money and were shut down are excluded from the average.

    If the closed funds averaged, say, 2% before being shut down, the true average across all funds (surviving and closed) would be substantially lower than 12%. This is survivorship bias: by conditioning on survival (continued existence), we overestimate the typical performance.

---

**Exercise 2.**
During World War II, Abraham Wald studied bullet holes on returning aircraft to decide where to add armor. Explain why reinforcing the areas with the most bullet holes would be a mistake due to survivorship bias.

??? success "Solution to Exercise 2"
    The aircraft examined were those that **returned safely** despite being hit. Areas with many bullet holes on returning planes are areas where damage is survivable -- the plane can fly home even when hit there.

    The planes that were shot in other areas (engines, fuel tanks, cockpit) did not return -- they were shot down. Those critical areas show **few** bullet holes on surviving planes precisely because hits there are fatal.

    Wald correctly reasoned that armor should be added to the areas with **few** bullet holes on returning planes, because those are the areas where hits cause the plane to be lost. Reinforcing the heavily-hit areas would protect parts of the plane that can already tolerate damage.

---

**Exercise 3.**
A study of successful entrepreneurs finds that 70% dropped out of college. Should we conclude that dropping out increases the probability of entrepreneurial success? Why or why not?

??? success "Solution to Exercise 3"
    No. This study examines only successful entrepreneurs (survivors), ignoring the vastly larger number of college dropouts who did not become successful entrepreneurs. The relevant comparison requires the success rate among all dropouts versus all graduates:

    $$
    P(\text{success} \mid \text{dropout}) \quad \text{vs.} \quad P(\text{success} \mid \text{graduate})
    $$

    The study instead computed $P(\text{dropout} \mid \text{success})$, which is the inverse conditional probability. By Bayes' theorem, these are different quantities. If dropouts far outnumber graduates in the general population, a high $P(\text{dropout} \mid \text{success})$ is consistent with $P(\text{success} \mid \text{dropout})$ being very low.

---

**Exercise 4.**
Propose a concrete strategy to mitigate survivorship bias in a longitudinal study tracking startup companies over 5 years.

??? success "Solution to Exercise 4"
    Key strategies include:

    1. **Intent-to-treat enrollment:** Register all startups at the beginning and track them regardless of whether they survive. Record the outcome (success, failure, pivot, acquisition) for every enrolled company.
    2. **Track exits:** When a startup fails or is acquired, record its final status and the date of exit rather than simply removing it from the dataset.
    3. **Include all data:** In the analysis, include the partial trajectories of companies that failed. For example, compute average revenue using all company-years, not just surviving companies.
    4. **Report attrition:** Explicitly report the number of startups that exited the study at each time point and compare the characteristics of survivors to non-survivors.
    5. **Sensitivity analysis:** Compute results both with and without failed startups to quantify the magnitude of survivorship bias.

---

**Exercise 5.**
A textbook on long-term investing reports that "stocks return 10% per year on average" based on the U.S. S&P 500 over 1928–2023. Discuss two distinct ways this figure may be inflated by survivorship bias at the *index* level and the *country* level.

??? success "Solution to Exercise 5"
    **Index-level survivorship:** The S&P 500 is reconstituted regularly. Companies that decline are removed from the index, and growing companies are added. The index's reported return reflects the survivors at each rebalance — not a buy-and-hold portfolio of the original constituents. Studies comparing the live index to a frozen 1928 constituent list find the gap is on the order of 0.5–1% per year.

    **Country-level survivorship:** The U.S. is one of the only major markets without a 20th-century catastrophic break (no expropriation, no war on home soil destroying the market). Russian (1917), Chinese (1949), and German (1923, 1945) investors lost most or all of their equity wealth. The 10% U.S. figure averages over a survivor; the global equity premium is closer to 5–7%. Dimson, Marsh, and Staunton's *Triumph of the Optimists* is the canonical reference on this point.

    Both biases pull historical estimates upward and have direct implications for retirement and pension-fund planning.

---

**Exercise 6.**
A medical journal publishes only studies with $p < 0.05$. Explain how this **publication bias** is a form of survivorship bias at the *study* level. What is one consequence for meta-analyses, and one common adjustment?

??? success "Solution to Exercise 6"
    Each completed study is a "case." Studies that find significant results are more likely to "survive" through to publication; studies with $p > 0.05$ are often filed away unpublished (the **file-drawer problem**). The published literature is therefore a non-random sample of all studies conducted, biased toward larger effects.

    **Consequence for meta-analyses:** pooled effect estimates that combine published studies overstate the true effect, because the smaller, null, and contradictory studies are missing. The meta-analytic effect estimate is biased upward, and the confidence interval understates uncertainty.

    **Common adjustment:** the **funnel plot** displays each study's effect size against its standard error. Under no publication bias, the plot is a symmetric funnel widening for smaller studies; asymmetry (small studies with small effects missing from the lower-left) is diagnostic of publication bias. Quantitative corrections include Egger's regression test, trim-and-fill (which imputes missing studies), and selection models that explicitly parameterize the probability of publication as a function of $p$-value. The most reliable remedy, however, is **preregistration**: committing to a study and an analysis plan before data are collected.
