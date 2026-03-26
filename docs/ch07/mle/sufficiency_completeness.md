# Sufficiency and Completeness

## Why These Concepts Matter

When estimating parameters of a distribution, we face a key question: can we reduce the data to a simpler summary without losing any information about the parameter? **Sufficiency** answers this question. A related concept, **completeness**, ensures that an unbiased estimator based on a sufficient statistic is the unique best (in the sense of minimum variance). Together, sufficiency and completeness connect to the Lehmann-Scheffe theorem, which provides a constructive method for finding the uniformly minimum variance unbiased estimator (UMVUE).

## Sufficient Statistics for the Normal Model

A statistic $T(\mathbf{X})$ is **sufficient** for a parameter $\theta$ if the conditional distribution of the data $\mathbf{X}$ given $T(\mathbf{X})$ does not depend on $\theta$. Intuitively, once we know the value of a sufficient statistic, the remaining data carry no additional information about $\theta$.

The **factorization theorem** (Fisher-Neyman) provides a practical criterion: $T(\mathbf{X})$ is sufficient for $\theta$ if and only if the joint density can be written as

$$
f(\mathbf{x}; \theta) = g(T(\mathbf{x}), \theta) \cdot h(\mathbf{x})
$$

where $g$ depends on the data only through $T$ and $h$ does not depend on $\theta$.

For the normal model $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$, the joint density is

$$
f(\mathbf{x}; \mu, \sigma^2) = \left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2\right)
$$

Expanding the exponent using $\sum(x_i - \mu)^2 = \sum(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2$:

$$
f(\mathbf{x}; \mu, \sigma^2) = \underbrace{\left(\frac{1}{2\pi\sigma^2}\right)^{n/2} \exp\left(-\frac{\sum(x_i - \bar{x})^2 + n(\bar{x} - \mu)^2}{2\sigma^2}\right)}_{g\left((\bar{x},\, \sum(x_i - \bar{x})^2),\; \mu,\, \sigma^2\right)} \cdot \underbrace{1}_{h(\mathbf{x})}
$$

The density depends on the data only through $\bar{x}$ and $\sum(x_i - \bar{x})^2$. By the factorization theorem, $(\bar{X}, S^2)$ — or equivalently $(\bar{X}, \sum(X_i - \bar{X})^2)$ — is jointly sufficient for $(\mu, \sigma^2)$.

!!! tip "What Sufficiency Means in Practice"

    Once we compute $\bar{X}$ and $S^2$, we have captured all the information the data contain about $(\mu, \sigma^2)$. The individual observations $X_1, \ldots, X_n$ carry no additional value for estimating these parameters. This is why statistical summaries often report just the sample mean and sample variance.

## Completeness

Sufficiency tells us that a statistic captures all the information. But different unbiased estimators based on a sufficient statistic might still exist. **Completeness** rules this out by ensuring that any unbiased function of the statistic is unique.

A sufficient statistic $T$ is **complete** if for every measurable function $g$:

$$
E_\theta[g(T)] = 0 \quad \text{for all } \theta \in \Theta \quad \Longrightarrow \quad P_\theta(g(T) = 0) = 1 \quad \text{for all } \theta \in \Theta
$$

In words: the only unbiased estimator of zero based on $T$ is the function that is identically zero. This means there is no "wasted" information in $T$ — we cannot extract a nontrivial signal that is zero on average for every parameter value.

For the normal model, the statistic $(\bar{X}, S^2)$ is not only sufficient but also complete. This follows from the fact that the normal family is a full-rank exponential family, and complete sufficient statistics exist for all full-rank exponential families.

## The Lehmann-Scheffe Theorem

The payoff of combining sufficiency and completeness is the Lehmann-Scheffe theorem, which identifies the unique best unbiased estimator.

!!! info "Theorem (Lehmann-Scheffe)"

    Let $T$ be a complete and sufficient statistic for $\theta$. If $h(T)$ is any unbiased estimator of a function $\tau(\theta)$ — that is, $E_\theta[h(T)] = \tau(\theta)$ for all $\theta$ — then $h(T)$ is the unique **uniformly minimum variance unbiased estimator (UMVUE)** of $\tau(\theta)$.

The proof relies on two facts. First, by the Rao-Blackwell theorem, conditioning any unbiased estimator on a sufficient statistic cannot increase its variance. Second, completeness guarantees that only one unbiased function of $T$ exists for each target $\tau(\theta)$, so the Rao-Blackwellized estimator is unique.

## Application to the Normal Model

Since $(\bar{X}, S^2)$ is complete and sufficient for $(\mu, \sigma^2)$ in the normal model, any unbiased function of $(\bar{X}, S^2)$ is automatically the UMVUE.

**UMVUE of $\mu$:** The sample mean $\bar{X}$ is a function of the sufficient statistic and satisfies $E[\bar{X}] = \mu$, so it is the unique UMVUE of $\mu$.

**UMVUE of $\sigma^2$:** The sample variance $S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$ is a function of the sufficient statistic and satisfies $E[S^2] = \sigma^2$, so it is the unique UMVUE of $\sigma^2$.

!!! warning "MLE vs UMVUE"

    The MLE of $\sigma^2$ is $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(X_i - \bar{X})^2$, which divides by $n$ instead of $n-1$. This is biased, so it is **not** the UMVUE despite being a function of the sufficient statistic. The Lehmann-Scheffe theorem requires unbiasedness — a biased function of a complete sufficient statistic is not the UMVUE.

!!! example "Verifying UMVUE Properties"

    For $n = 20$ observations from $N(\mu, \sigma^2)$:

    - $\bar{X}$ is the UMVUE of $\mu$ with variance $\sigma^2/20$. No other unbiased estimator of $\mu$ can have variance smaller than $\sigma^2/20$.
    - $S^2$ is the UMVUE of $\sigma^2$ with variance $2\sigma^4/19$. The MLE $\hat{\sigma}^2_{\text{MLE}} = (19/20)S^2$ has lower MSE but is biased, so it does not qualify as a UMVUE.
