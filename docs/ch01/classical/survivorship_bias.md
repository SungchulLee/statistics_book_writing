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
