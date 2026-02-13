#!/usr/bin/env python3
# ===================================================================================
# 15_welch2_02_robust_ols_HC3_wald.py
# ===================================================================================
# Heteroskedasticity-robust factorial ANOVA via OLS + HC3 covariance + Wald F-tests.
#
# Procedure:
#   1) Fit OLS with categorical factors and interaction: y ~ C(A) * C(B)
#   2) Get robust covariance (HC3)
#   3) Use Wald F-tests to test the joint null for each term's coefficients:
#        H0: all level-contrast coefficients for that term = 0
#
# This provides a pragmatic analog to "Welch-style" robustness for multi-factor
# designs when a direct Welch–James implementation is unavailable.
# ===================================================================================

import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

def term_param_constraints(param_names, term_prefix):
    # Build comma-separated string like "C(Temperature)[T.High] = 0, C(Temperature)[T.Medium] = 0"
    targets = [p for p in param_names if p.startswith(term_prefix)]
    return ", ".join([f"{t} = 0" for t in targets])

def main():
    data = {
        "Temperature": ["High", "High", "High", "Low", "Low", "Low", "Medium", "Medium", "Medium"],
        "Fertilizer":  ["A",    "B",    "C",    "A",  "B",  "C",  "A",      "B",      "C"     ],
        "Growth":      [12,     15,     14,     10,   13,   11,   14,       16,       15      ],
    }
    df = pd.DataFrame(data)

    # OLS with interaction
    model = ols("Growth ~ C(Temperature) * C(Fertilizer)", data=df).fit()
    # HC3-robust covariance
    rob = model.get_robustcov_results(cov_type="HC3")

    # Build constraints for main effects & interaction using parameter names
    pnames = model.params.index.tolist()

    c_temp = term_param_constraints(pnames, "C(Temperature)[")
    c_fert = term_param_constraints(pnames, "C(Fertilizer)[")
    c_inter = term_param_constraints(pnames, "C(Temperature)[T")

    # Wald F-tests
    print("=== Robust HC3 Wald tests (multi-parameter) ===")
    if c_temp:
        print("\nMain effect: Temperature")
        print(rob.f_test(c_temp))
    else:
        print("\nMain effect: Temperature — no extra levels (nothing to test).")

    if c_fert:
        print("\nMain effect: Fertilizer")
        print(rob.f_test(c_fert))
    else:
        print("\nMain effect: Fertilizer — no extra levels (nothing to test).")

    if c_inter:
        # Interaction terms have names like 'C(Temperature)[T.High]:C(Fertilizer)[T.B]'
        inter_targets = [p for p in pnames if ":" in p]
        c_inter = ", ".join([f"{t} = 0" for t in inter_targets])
        print("\nInteraction: Temperature × Fertilizer")
        print(rob.f_test(c_inter))
    else:
        print("\nInteraction: none detected.")

    # Show standard ANOVA (homoskedastic) for reference
    print("\n(Reference OLS ANOVA—non-robust):")
    print(sm.stats.anova_lm(model, typ=2))

if __name__ == "__main__":
    main()
