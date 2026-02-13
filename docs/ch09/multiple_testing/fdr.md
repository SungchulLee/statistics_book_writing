# False Discovery Rate (Benjamini–Hochberg)

## Overview

The **False Discovery Rate (FDR)** controls the expected proportion of false discoveries among all rejected hypotheses.

## Benjamini–Hochberg Procedure

1. Order p-values: $p_{(1)} \leq \cdots \leq p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m}\alpha$
3. Reject $H_{(1)}, \ldots, H_{(k)}$

FDR control is less conservative than FWER control and is widely used in genomics and other high-dimensional settings.
