# Sampling Distributions for General OLS Estimators

## Two Orthogonal Projections of Gaussian Errors

!!! info "Theorem"
    $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ is independent of $\mathbf{H}\boldsymbol{\varepsilon}$ for Gaussian errors.

??? note "Proof"
    **Key Properties of Gaussian Random Vectors.** In the case of Gaussian random vectors, orthogonality implies independence. If two linear transformations of a multivariate Gaussian random vector are uncorrelated (orthogonal), they are also independent. The projection matrices $\mathbf{H}$ and $\mathbf{I} - \mathbf{H}$ satisfy: $\mathbf{H}^2 = \mathbf{H}$ (idempotent), $(\mathbf{I} - \mathbf{H})^2 = \mathbf{I} - \mathbf{H}$ (idempotent), and $\mathbf{H}(\mathbf{I} - \mathbf{H}) = 0$ (orthogonality).

    **Gaussian Error Assumption.** With $\boldsymbol{\varepsilon} \sim N(0, \sigma^2I)$, the transformed vectors $\mathbf{H}\boldsymbol{\varepsilon}$ and $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ are linear combinations of the components of $\boldsymbol{\varepsilon}$, and they remain Gaussian because any linear transformation of a Gaussian random vector is still Gaussian.

    **Means:**

    - $\mathbb{E}[\mathbf{H}\boldsymbol{\varepsilon}] = \mathbf{H}\mathbb{E}[\boldsymbol{\varepsilon}] = 0$
    - $\mathbb{E}[(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}] = (\mathbf{I} - \mathbf{H})\mathbb{E}[\boldsymbol{\varepsilon}] = 0$

    **Covariances:**

    - $\text{Cov}(\mathbf{H}\boldsymbol{\varepsilon}, \mathbf{H}\boldsymbol{\varepsilon}) = \sigma^2\mathbf{H}$
    - $\text{Cov}((\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}, (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}) = \sigma^2(\mathbf{I} - \mathbf{H})$
    - $\text{Cov}(\mathbf{H}\boldsymbol{\varepsilon}, (\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}) = \sigma^2\mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{0}$

    Since $\mathbf{H}\boldsymbol{\varepsilon}$ and $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ are uncorrelated and jointly Gaussian, they are independent.

## Orthogonal Decomposition of Gaussian Errors

!!! info "Theorem"
    $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ is independent of $(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top\boldsymbol{\varepsilon}$ for Gaussian errors.

??? note "Proof"
    **Key Definitions.** The projection matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top$ projects onto the column space of $\mathbf{X}$, and $(\mathbf{I} - \mathbf{H})$ projects onto the orthogonal complement. The quantity $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ is the component of the error vector in the orthogonal complement, while $(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \boldsymbol{\varepsilon}$ is a transformation that lies in the column space of $\mathbf{X}$.

    **Gaussianity.** Both $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ and $(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \boldsymbol{\varepsilon}$ are linear transformations of $\boldsymbol{\varepsilon}$, so they are Gaussian. If they are orthogonal, they are independent.

    **Covariance Analysis.**

    $$
    \begin{array}{lll}
    \text{Cov}((\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon},\; (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \boldsymbol{\varepsilon})
    &=&
    \mathbb{E}[(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}\boldsymbol{\varepsilon}^\top\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}]\\
    &=&
    \sigma^2(\mathbf{I} - \mathbf{H})\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\\
    &=&
    \sigma^2\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}
    -\sigma^2\mathbf{H}\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\\
    &=&
    \sigma^2\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}
    -\sigma^2\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\\
    &=&
    \sigma^2\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}
    -\sigma^2\mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}=\mathbf{0}\\
    \end{array}
    $$

    Since $\boldsymbol{\varepsilon}$ is Gaussian, uncorrelated components are also independent. Therefore, $(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$ is **independent** of $(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \boldsymbol{\varepsilon}$.
