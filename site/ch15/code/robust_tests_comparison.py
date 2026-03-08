"""
Robust Tests Comparison
=======================
Demonstrates key concepts related to robust tests comparison.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# Robust Tests Comparison — Demonstration
# =============================================================================

def main():
    """Run the main demonstration."""
    print("=" * 60)
    print(f"Robust Tests Comparison")
    print("=" * 60)
    
    # Generate sample data
    n = 100
    data = np.random.normal(loc=0, scale=1, size=n)
    
    print(f"Sample size: {n}")
    print(f"Sample mean: {data.mean():.4f}")
    print(f"Sample std:  {data.std(ddof=1):.4f}")
    

if __name__ == "__main__":
    main()
