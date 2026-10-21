# Bayes 분산 검정

## 개요

이 페이지는 두 집단의 분산을 비교하는 Bayes 접근을 제시한다. $p$값 하나를 계산하는 대신, Bayes 틀은 각 집단의 분산과 분산비에 대한 완전한 사후분포를 만들어 낸다. 켤레 정규-역감마 모형을 쓰고 몬테카를로로 사후표본을 뽑은 뒤 신용구간과 사후확률로 결과를 요약한다.

!!! note "사전분포가 처음이라면"

    이 쪽은 사전분포·사후분포·켤레 사전분포를 **이미 아는 것으로 보고** 분산 하나에 적용한다.
    그 틀 자체는 [6.3 베이즈 추정 ★](../../ch06/bayesian/prior_posterior.md)이 세운다 —
    모수에 분포를 주고 자료로 갱신한다는 발상, 켤레가 왜 편한지, MAP 가 무엇인지가 거기 있다.

---

## 1. 정규-역감마 모형

각 집단 $i$에 대해 자료를 $x_{ij} \mid \mu_i, \sigma_i^2 \sim \mathcal{N}(\mu_i, \sigma_i^2)$로 모형화하고 켤레 사전분포를 둔다.

$$
\mu_i \mid \sigma_i^2 \sim \mathcal{N}\!\left(m_0,\; \frac{\sigma_i^2}{\kappa_0}\right), \qquad \sigma_i^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

여기서 $m_0$, $\kappa_0$, $\alpha_0$, $\beta_0$은 초모수이다. 모호한 사전분포($\kappa_0 \approx 0$, $\alpha_0 \approx 0$, $\beta_0 \approx 0$)에서는 사후분포를 자료가 지배한다.

---

## 2. 사후 갱신

표본평균 $\bar{x}$와 편차제곱합 $S = \sum_{j=1}^{n}(x_j - \bar{x})^2$을 갖는 $n$개 관측값이 주어지면 사후 모수는

$$
\kappa_n = \kappa_0 + n, \qquad m_n = \frac{\kappa_0 m_0 + n \bar{x}}{\kappa_n}
$$

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \qquad \beta_n = \beta_0 + \frac{1}{2}\left(S + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - m_0)^2\right)
$$

분산의 주변 사후분포는

$$
\sigma_i^2 \mid \mathbf{x}_i \sim \text{Inv-Gamma}(\alpha_n,\, \beta_n)
$$

!!! note "$n/2$인가 $(n-1)/2$인가"
    여기서 $\alpha_n = \alpha_0 + n/2$인 것이 15.6절 [Bayes 분산 검정](./bayesian_variance.md)의 $\alpha_n = \alpha_0 + (n-1)/2$와 다르다. **두 모형이 다르기 때문이다.**

    - **15.6절**은 $\mu$에 사전분포를 두지 않고 $\bar{x}$로 대체한 뒤 자유도 $n-1$을 쓴다. 이 경우 무정보 극한에서 빈도주의 결과와 정확히 일치한다.
    - **여기**는 $\mu \mid \sigma^2 \sim \mathcal{N}(m_0, \sigma^2/\kappa_0)$이라는 결합 켤레 사전분포를 둔다. $\mu$를 적분해 내면 $\alpha_n = \alpha_0 + n/2$가 된다.

    $\kappa_0 \to 0$인 극한은 사전분포 $p(\mu, \sigma^2) \propto (\sigma^2)^{-\alpha_0 - 3/2}$에 해당하며, 표준적인 Jeffreys 사전분포 $p(\mu,\sigma^2) \propto 1/\sigma^2$과 다르다. 그래서 **신용구간이 빈도주의 신뢰구간보다 약간 좁게 나온다.**

    아래 보기에서 $\sigma_1^2$의 95% 신용구간은 $(1.133, 9.085)$인데 빈도주의 신뢰구간은 $(1.241, 11.761)$이다. 로그 척도 폭으로 $2.082$ 대 $2.249$로 **7% 좁다**. $n = 8$처럼 작은 표본에서 이 차이가 눈에 띈다.

    빈도주의 결과와 일치시키려면 $\alpha_0 = -1/2$로 두어 $\alpha_n = (n-1)/2$가 되게 하면 된다(부적절 사전분포이지만 사후분포는 적절하다).

---

다음 코드는 사후 모수를 계산하고 $\sigma^2$의 사후분포에서 표본을 뽑는다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 사후분포에서 분산 뽑는 함수. 위의 갱신식을 그대로 옮기고, 기본 초모수를 $\kappa_0 = 10^{-6}$, $\alpha_0 = \beta_0 = 10^{-2}$ 로 두어 거의 정보를 주지 않는 사전분포를 쓴다.

**(1)** 이 기본값에서 사후분포가 **거의** $\text{Inv-Gamma}(n/2,\, S/2)$ 임을 보이고, 그 극한에서 추축량이 $S/\sigma^2 \sim \chi^2_n$ 이 됨을 보이시오. 빈도주의는 같은 자리에 $\chi^2_{n-1}$ 을 쓴다. 신용구간이 신뢰구간보다 **좁은** 까닭이 이것임을 밝히시오.

**(2)** `x1` 에 함수를 적용해 사전분포의 기여가 실제로 얼마나 작은지 수로 보이고, 신용구간과 신뢰구간의 로그 폭을 견주시오. $\alpha_0 = -1/2$ 로 두면 둘이 정확히 같아지는지 확인하시오.

</div>

??? success "풀이"

    **(1) 기본값에서 사전분포는 거의 사라진다.** 사후 초모수는

    $$
    \alpha_n = \alpha_0 + \frac n2, \qquad
    \beta_n = \beta_0 + \frac12\left(S + \frac{\kappa_0 n}{\kappa_0 + n}(\bar x - m_0)^2\right)
    $$

    이다. $\kappa_0 = 10^{-6}$ 이므로 $\dfrac{\kappa_0 n}{\kappa_0 + n} \approx \kappa_0 = 10^{-6}$ 이고, 평균 항은 $\tfrac12 \cdot 10^{-6}(\bar x - m_0)^2$ 로 $\bar x$ 가 10 남짓일 때 $10^{-4}$ 수준이다. $\alpha_0 = \beta_0 = 10^{-2}$ 도 작다. 그러므로

    $$
    \alpha_n \approx \frac n2, \qquad \beta_n \approx \frac S2
    $$

    이고, 사후분포는 **거의** $\text{Inv-Gamma}(n/2,\, S/2)$ 다.

    **추축량을 꺼내면 자유도가 드러난다.** $V \sim \text{Inv-Gamma}(a, b)$ 이면 $1/V \sim \text{Gamma}(a, \text{rate } b)$ 이고 척도를 두 배로 바꾸면 $2b/V \sim \text{Gamma}(a, \text{rate } \tfrac12) = \chi^2_{2a}$ 이므로

    $$
    \frac{2\beta_n}{\sigma^2} \;\sim\; \chi^2_{2\alpha_n}
    $$

    이다. 무정보 극한 $\alpha_n = n/2$, $\beta_n = S/2$ 를 넣으면

    $$
    \frac{S}{\sigma^2} \;\sim\; \chi^2_{n}
    $$

    이 된다. **빈도주의는 같은 $S$ 를 놓고 $S/\sigma^2 \sim \chi^2_{n-1}$ 을 쓴다.** 15.2절이 유도한 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$ 이 그것이고, $S = (n-1)s^2$ 이니 같은 양이다. **같은 자료에 자유도만 하나 다른 두 분포를 들이대는 것**이다.

    자유도가 하나 큰 쪽이 상대적으로 좁다. 구간의 꼴이 양쪽 모두

    $$
    \left(\frac{S}{\chi^2_{0.975,\,d}},\; \frac{S}{\chi^2_{0.025,\,d}}\right)
    $$

    이므로 **상한과 하한의 비가 $\chi^2_{0.975,d}/\chi^2_{0.025,d}$ 로 자료와 무관**하고($S$ 가 약분된다 — 8장이 다룬 사실이다), 이 비는 $d$ 가 커질수록 작아진다. 그래서 $d = n$ 쪽이 $d = n-1$ 쪽보다 좁다. 로그 척도의 폭이 바로 이 비의 로그다.

    까닭은 모형이 다른 데 있다. 이 쪽의 사전분포는 $\mu \mid \sigma^2 \sim \mathcal N(m_0, \sigma^2/\kappa_0)$ 라는 **결합** 켤레 사전분포이고, $\kappa_0 \to 0$ 극한은 $p(\mu,\sigma^2) \propto (\sigma^2)^{-\alpha_0-3/2}$ 에 해당해 표준 제프리스 사전분포 $1/\sigma^2$ 와 다르다. $(\sigma^2)^{-1/2}$ 하나가 더 붙어 있고, 그것이 자유도 하나로 나타난다. **빈도주의와 맞추려면** $\alpha_n = (n-1)/2$ 가 되게 $\alpha_0 = -1/2$ 로 두면 된다(부적절 사전분포이지만 사후분포는 적절하다).

    **(2) 수치적으로.** `x1` 에 함수를 적용한다.

    ```python
    import numpy as np
    from scipy.stats import invgamma

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        """정규-역감마 켤레모형의 사후 모수를 구한다.

        기본값은 거의 정보를 주지 않는 사전분포다(k0, a0, b0 가 모두 작다).
        이러면 사후분포가 자료에 거의 전적으로 맡겨진다.
        """
        x = np.asarray(x, dtype=float)
        n = x.size
        xbar = x.mean()
        S = np.sum((x - xbar)**2)
        k_n = k0 + n
        m_n = (k0 * m0 + n * xbar) / k_n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return m_n, k_n, a_n, b_n

    def draw_posterior_sigma2(x, n_draws=10000, rng=None):
        """분산의 사후분포에서 표본을 뽑는다.

        평균을 적분해 없앤 sigma^2 의 주변 사후분포가 역감마가 된다.
        그래서 MCMC 없이 바로 뽑을 수 있다.
        """
        if rng is None:
            rng = np.random.default_rng()
        m_n, k_n, a_n, b_n = posterior_params(x)
        sig2 = invgamma(a=a_n, scale=b_n).rvs(size=n_draws, random_state=rng)
        return sig2

    # === 사전분포의 기여가 얼마나 작은가 ===
    from scipy.stats import chi2

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    n = x1.size
    S = np.sum((x1 - x1.mean()) ** 2)
    m_n, k_n, a_n, b_n = posterior_params(x1)

    print(f"n = {n},  xbar = {x1.mean():.4f},  S = {S:.4f},  s^2 = {x1.var(ddof=1):.4f}")
    print(f"a_n = {a_n:.4f}   (무정보 극한 n/2 = {n / 2:.4f})")
    print(f"b_n = {b_n:.6f}   (무정보 극한 S/2 = {S / 2:.6f})")
    print(f"  b0 가 더한 것        = {1e-2:.6f}")
    print(f"  평균 항이 더한 것    = {0.5 * (1e-6 * n / k_n) * x1.mean() ** 2:.3e}")
    print(f"  둘을 합쳐 b_n 의 {100 * (b_n - S / 2) / b_n:.4f} %")

    # === 추축량: 2 b_n / sigma^2 ~ chi2(2 a_n) 인가 ===
    lo, hi = invgamma(a=a_n, scale=b_n).ppf([0.025, 0.975])
    piv = 2 * b_n / chi2(2 * a_n).ppf([0.975, 0.025])
    print(f"\n역감마 분위수로  : ({lo:.6f}, {hi:.6f})")
    print(f"2b_n/chi2(2a_n) 로: ({piv[0]:.6f}, {piv[1]:.6f})")

    # === 신용구간 대 빈도주의 신뢰구간 ===
    f_lo, f_hi = S / chi2(n - 1).ppf([0.975, 0.025])
    print(f"\n95% 신용구간      = ({lo:.3f}, {hi:.3f})      상한/하한 = {hi / lo:.4f}")
    print(f"95% 신뢰구간      = ({f_lo:.3f}, {f_hi:.3f})     상한/하한 = {f_hi / f_lo:.4f}")
    print(f"로그 폭           = {np.log(hi / lo):.4f} 대 {np.log(f_hi / f_lo):.4f}"
          f"   -> {100 * (1 - np.log(hi / lo) / np.log(f_hi / f_lo)):.1f} % 좁다")
    print(f"자유도 비교: 2a_n = {2 * a_n:.2f}  대  n-1 = {n - 1}")
    print(f"chi2 분위수 비 (자료와 무관): {chi2(2 * a_n).ppf(0.975) / chi2(2 * a_n).ppf(0.025):.4f}"
          f"  대  {chi2(n - 1).ppf(0.975) / chi2(n - 1).ppf(0.025):.4f}")

    # === a0 = -1/2 로 두면 빈도주의와 정확히 같아진다 ===
    g_lo, g_hi = invgamma(a=(n - 1) / 2, scale=S / 2).ppf([0.025, 0.975])
    print(f"\na0 = -1/2 (a_n = (n-1)/2) 신용구간 = ({g_lo:.6f}, {g_hi:.6f})")
    print(f"빈도주의 신뢰구간                  = ({f_lo:.6f}, {f_hi:.6f})")
    print(f"차 = {abs(g_lo - f_lo):.3e}, {abs(g_hi - f_hi):.3e}")

    # === 함수가 제대로 도는지 ===
    rng = np.random.default_rng(0)
    d = draw_posterior_sigma2(x1, n_draws=200_000, rng=rng)
    print(f"\n사후표본 20 만 개:  평균 {d.mean():.4f} (닫힌 꼴 {b_n / (a_n - 1):.4f}),  "
          f"중앙값 {np.median(d):.4f} (닫힌 꼴 {invgamma(a=a_n, scale=b_n).ppf(0.5):.4f})")
    ```

    출력:

    ```text
    n = 8,  xbar = 12.6250,  S = 19.8750,  s^2 = 2.8393
    a_n = 4.0100   (무정보 극한 n/2 = 4.0000)
    b_n = 9.947580   (무정보 극한 S/2 = 9.937500)
      b0 가 더한 것        = 0.010000
      평균 항이 더한 것    = 7.970e-05
      둘을 합쳐 b_n 의 0.1013 %

    역감마 분위수로  : (1.132684, 9.085126)
    2b_n/chi2(2a_n) 로: (1.132684, 9.085126)

    95% 신용구간      = (1.133, 9.085)      상한/하한 = 8.0209
    95% 신뢰구간      = (1.241, 11.761)     상한/하한 = 9.4757
    로그 폭           = 2.0820 대 2.2487   -> 7.4 % 좁다
    자유도 비교: 2a_n = 8.02  대  n-1 = 7
    chi2 분위수 비 (자료와 무관): 8.0209  대  9.4757

    a0 = -1/2 (a_n = (n-1)/2) 신용구간 = (1.241197, 11.761265)
    빈도주의 신뢰구간                  = (1.241197, 11.761265)
    차 = 0.000e+00, 0.000e+00

    사후표본 20 만 개:  평균 3.2993 (닫힌 꼴 3.3048),  중앙값 2.6953 (닫힌 꼴 2.7016)
    ```

    **사전분포의 기여는 $0.1\%$ 다.** $\beta_n = 9.947580$ 가운데 $S/2 = 9.937500$ 이 자료 몫이고, $\beta_0 = 0.01$ 과 평균 항 $7.97\times10^{-5}$ 를 합친 $0.0101$ 이 사전분포 몫이다. $\alpha_n$ 도 $4.0100$ 으로 $n/2 = 4$ 에 거의 같다. **"무정보"라는 말이 수로 확인된다.**

    **추축량이 정확히 맞는다.** 역감마 분위수로 잡은 구간 $(1.132684,\, 9.085126)$ 과 $2\beta_n/\chi^2_{2\alpha_n}$ 로 잡은 구간이 소수점 여섯째 자리까지 같다. (1)의 $2\beta_n/\sigma^2 \sim \chi^2_{2\alpha_n}$ 이 확인된 것이다.

    **자유도 하나가 $7.4\%$ 의 폭 차이를 만든다.** 신용구간의 상한/하한 비가 $8.0209$, 신뢰구간은 $9.4757$ 이고 둘 다 $\chi^2$ 분위수의 비라 **자료와 무관**하다. 로그 폭이 $2.0820$ 대 $2.2487$ 로 신용구간이 $7.4\%$ 좁다. 자유도가 $8.02$ 대 $7$ 이니 $15\%$ 더 많은 "정보"를 쓴 셈이고, 그 정보는 자료에서 온 것이 아니라 $\kappa_0 \to 0$ 극한이 숨겨 들여온 $(\sigma^2)^{-1/2}$ 에서 왔다.

    **$\alpha_0 = -1/2$ 가 그 어긋남을 정확히 지운다.** $\alpha_n = (n-1)/2 = 3.5$ 로 두면 신용구간이 $(1.241197,\, 11.761265)$ 로 빈도주의 신뢰구간과 **차가 정확히 $0$** 이다. 자유도 하나가 두 접근을 가르는 전부였다는 뜻이다.

    마지막 줄은 함수가 제대로 도는지 본 것이다. 사후표본 20 만 개의 평균 $3.2993$ 과 중앙값 $2.6953$ 이 닫힌 꼴 $\beta_n/(\alpha_n-1) = 3.3048$ 과 $2.7016$ 에 맞는다.

---

## 3. 두 집단 비교

$\sigma_1^2$과 $\sigma_2^2$을 비교하려면 각 사후분포에서 독립적으로 뽑아 비를 만든다.

$$
\rho = \frac{\sigma_1^2}{\sigma_2^2}
$$

$\rho$의 95% 신용구간이 1을 제외하면 분산이 다르다는 증거가 된다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 두 분산비의 사후분포. 집단당 $n = 8$ 인 두 표본의 사후분포에서 각각 2 만 개를 뽑아 비 $\rho = \sigma_1^2/\sigma_2^2$ 를 만든다.

**(1)** 두 집단의 $\alpha_n$ 이 같으면 $\rho$ 의 사후분포가 $\dfrac{\beta_{n,1}}{\beta_{n,2}} F(2\alpha_n, 2\alpha_n)$ 임을 보이시오. 그것으로 사후중앙값·사후평균·신용구간·$P(\rho>1)$ 을 닫힌 꼴로 적고, **사후중앙값이 표본분산의 비와 거의 같은** 까닭을 밝히시오.

**(2)** 모의실험이 그 닫힌 꼴을 재현하는지 확인하고, $P(\rho>1)$ 을 $F$ 검정의 단측 $p$ 값과 견주시오. 사후평균과 사후중앙값이 어긋나는 크기는 얼마인가.

</div>

??? success "풀이"

    **(1) 비의 사후분포는 척도를 바꾼 F 분포다.** 보기 1 에서 $2\beta_{n,i}/\sigma_i^2 \sim \chi^2_{2\alpha_{n,i}}$ 임을 보았다. 두 집단이 독립이고 $\alpha_{n,1} = \alpha_{n,2} = \alpha_n$ 이면 $\sigma_i^2 = 2\beta_{n,i}/W_i$, $W_i \sim \chi^2_{2\alpha_n}$ 이므로

    $$
    \rho = \frac{\sigma_1^2}{\sigma_2^2}
    = \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{W_2}{W_1}
    = \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{W_2/(2\alpha_n)}{W_1/(2\alpha_n)}
    \;\sim\; \frac{\beta_{n,1}}{\beta_{n,2}}\, F(2\alpha_n,\, 2\alpha_n)
    $$

    이다. 두 자유도가 같은 F 분포는 역수에 대해 닫혀 있어 중앙값이 정확히 1 이므로

    $$
    \operatorname{median}(\rho) = \frac{\beta_{n,1}}{\beta_{n,2}},
    \qquad
    \text{신용구간} = \frac{\beta_{n,1}}{\beta_{n,2}}\Bigl[F_{0.025},\; F_{0.975}\Bigr],
    \qquad
    P(\rho > 1) = P\!\left(F > \frac{\beta_{n,2}}{\beta_{n,1}}\right)
    $$

    이다. 사후평균은 독립성과 $E[1/V] = \alpha/\beta$ (역감마의 역수가 감마이므로)에서

    $$
    E[\rho] = E[\sigma_1^2]\,E\!\left[\frac{1}{\sigma_2^2}\right]
    = \frac{\beta_{n,1}}{\alpha_n - 1}\cdot\frac{\alpha_n}{\beta_{n,2}}
    = \frac{\alpha_n}{\alpha_n - 1}\cdot\frac{\beta_{n,1}}{\beta_{n,2}}
    $$

    가 된다. **평균이 중앙값의 $\dfrac{\alpha_n}{\alpha_n-1}$ 배**라는 깔끔한 꼴이다. $\alpha_n = 4.01$ 이면 $4.01/3.01 = 1.3322$ 배다. 이 배율이 1 이 아닌 것이 곧 사후분포의 치우침이며, $n$ 이 작을수록 커진다.

    **중앙값이 표본분산의 비와 거의 같은 까닭.** 보기 1 에서 본 대로 무정보 기본값에서 $\beta_{n,i} \approx S_i/2$ 이므로

    $$
    \operatorname{median}(\rho) = \frac{\beta_{n,1}}{\beta_{n,2}} \approx \frac{S_1/2}{S_2/2}
    = \frac{S_1}{S_2} = \frac{(n-1)s_1^2}{(n-1)s_2^2} = \frac{s_1^2}{s_2^2}
    $$

    다. 두 집단의 $n$ 이 같아 $n-1$ 이 약분되고, 사전분포의 $\beta_0$ 는 분자와 분모에 똑같이 더해지므로 비에 거의 영향을 주지 않는다. **사후중앙값은 자료가 주는 비를 거의 그대로 돌려준다.** 보고해야 할 것이 "비가 얼마인가"라면 평균이 아니라 중앙값이다.

    **(2) 수치적으로.** 아래 코드는 보기 1 의 `posterior_params` 와 `draw_posterior_sigma2` 를 그대로 이어받는다.

    ```python
    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    # 두 집단의 사후표본을 각각 뽑아 나눈다. 이 비의 분포가 곧 분산비의
    # 사후분포다. 빈도주의 F 검정이 p-값 하나를 주는 자리에서, 베이즈 쪽은
    # "비가 1 보다 클 확률"을 그대로 셈할 수 있다.
    rng = np.random.default_rng(0)
    s1 = draw_posterior_sigma2(x1, n_draws=20000, rng=rng)
    s2 = draw_posterior_sigma2(x2, n_draws=20000, rng=rng)
    ratio = s1 / s2

    print(f"Sample variances:           {np.var(x1, ddof=1):.4f}, "
          f"{np.var(x2, ddof=1):.4f}")
    print(f"Posterior mean of sigma1^2: {s1.mean():.3f}")
    print(f"Posterior mean of sigma2^2: {s2.mean():.3f}")
    print(f"Posterior mean of ratio:    {ratio.mean():.3f}")
    print(f"95% credible interval:      ({np.percentile(ratio, 2.5):.3f}, "
          f"{np.percentile(ratio, 97.5):.3f})")
    print(f"P(sigma1^2 > sigma2^2):     {np.mean(ratio > 1.0):.4f}")

    # === 닫힌 꼴과 맞춰 본다 ===
    from scipy.stats import f as f_dist

    _, _, a1, b1 = posterior_params(x1)
    _, _, a2, b2 = posterior_params(x2)
    assert a1 == a2, "두 집단의 a_n 이 같아야 F 로 적힌다"
    Fd = f_dist(2 * a1, 2 * a2)
    sc = b1 / b2

    print(f"\na_n = {a1:.4f} (두 집단 공통),  b_n = {b1:.6f}, {b2:.6f}")
    print(f"rho ~ (b1/b2) x F(2a,2a) = {sc:.6f} x F({2 * a1:.2f},{2 * a2:.2f})")
    print("                닫힌 꼴    모의 2만개")
    for name, closed, mc in (
            ("median    ", sc * Fd.ppf(0.5), np.median(ratio)),
            ("mean      ", (b1 / (a1 - 1)) * (a2 / b2), ratio.mean()),
            ("q(0.025)  ", sc * Fd.ppf(0.025), np.percentile(ratio, 2.5)),
            ("q(0.975)  ", sc * Fd.ppf(0.975), np.percentile(ratio, 97.5)),
            ("P(rho>1)  ", Fd.sf(b2 / b1), np.mean(ratio > 1))):
        print(f"  {name}  {closed:10.6f}  {mc:12.6f}")

    print(f"\n표본분산의 비 s1^2/s2^2 = {np.var(x1, ddof=1) / np.var(x2, ddof=1):.6f}")
    print(f"사후중앙값 b1/b2        = {sc:.6f}   (차 {abs(sc - np.var(x1, ddof=1) / np.var(x2, ddof=1)):.2e})")
    print(f"평균/중앙값 = {((b1 / (a1 - 1)) * (a2 / b2)) / sc:.4f} 배")

    # === 빈도주의 F 검정과 견준다 ===
    n = x1.size
    Fs = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    p_two = 2 * min(f_dist(n - 1, n - 1).cdf(Fs), f_dist(n - 1, n - 1).sf(Fs))
    fl, fh = Fs / f_dist(n - 1, n - 1).ppf([0.975, 0.025])
    print(f"\nF 검정: F = {Fs:.4f}, 양측 p = {p_two:.4f}, 단측(하단) p = {p_two / 2:.4f}")
    print(f"  P(rho > 1) = {Fd.sf(b2 / b1):.4f}  <- 단측 p 와 견주라")
    print(f"빈도주의 신뢰구간 = ({fl:.6f}, {fh:.6f})   로그 폭 {np.log(fh / fl):.4f}")
    print(f"베이즈 신용구간   = ({sc * Fd.ppf(0.025):.6f}, {sc * Fd.ppf(0.975):.6f})"
          f"   로그 폭 {np.log(Fd.ppf(0.975) / Fd.ppf(0.025)):.4f}")
    print(f"  -> {100 * (1 - np.log(Fd.ppf(0.975) / Fd.ppf(0.025)) / np.log(fh / fl)):.1f} % 좁다")
    ```

    출력:

    ```text
    Sample variances:           2.8393, 6.0000
    Posterior mean of sigma1^2: 3.317
    Posterior mean of sigma2^2: 6.956
    Posterior mean of ratio:    0.634
    95% credible interval:      (0.106, 2.074)
    P(sigma1^2 > sigma2^2):     0.1573

    a_n = 4.0100 (두 집단 공통),  b_n = 9.947580, 21.010231
    rho ~ (b1/b2) x F(2a,2a) = 0.473464 x F(8.02,8.02)
                    닫힌 꼴    모의 2만개
      median        0.473464      0.477285
      mean          0.630760      0.633695
      q(0.025)      0.107025      0.106483
      q(0.975)      2.094531      2.073990
      P(rho>1)      0.155034      0.157350

    표본분산의 비 s1^2/s2^2 = 0.473214
    사후중앙값 b1/b2        = 0.473464   (차 2.49e-04)
    평균/중앙값 = 1.3322 배

    F 검정: F = 0.4732, 양측 p = 0.3448, 단측(하단) p = 0.1724
      P(rho > 1) = 0.1550  <- 단측 p 와 견주라
    빈도주의 신뢰구간 = (0.094739, 2.363662)   로그 폭 3.2168
    베이즈 신용구간   = (0.107025, 2.094531)   로그 폭 2.9740
      -> 7.5 % 좁다
    ```

    **닫힌 꼴과 모의실험이 맞는다.** 중앙값 $0.473464$ 대 $0.477285$, 평균 $0.630760$ 대 $0.633695$, $P(\rho>1)$ $0.155034$ 대 $0.157350$ 이다. 꼬리 분위수는 $0.107025$ 대 $0.106483$ 과 $2.094531$ 대 $2.073990$ 인데, 2 만 개로 $97.5\%$ 분위수를 재면 표준오차가 $\sqrt{0.975 \cdot 0.025/20000}\,/\,f(x_{0.975}) = 0.032$ 이므로 $0.021$ 차이는 그 안이다. 아래쪽 분위수의 표준오차는 $0.0017$ 이고 차는 $0.0005$ 다. $P(\rho>1)$ 의 표준오차는 $0.0026$ 이고 차는 $0.0023$ 이다. **모두 몬테카를로 오차로 설명된다.**

    **중앙값이 표본분산의 비와 사실상 같다.** $s_1^2/s_2^2 = 0.473214$, 사후중앙값 $\beta_{n,1}/\beta_{n,2} = 0.473464$ 로 차가 $2.5\times10^{-4}$ 다. (1)에서 $\beta_0$ 가 분자와 분모에 똑같이 더해져 약분된다고 한 것이 확인되었다.

    **평균과 중앙값은 $1.3322$ 배 어긋난다.** (1)이 예측한 $\alpha_n/(\alpha_n-1) = 4.01/3.01 = 1.3322$ 와 소수점 넷째 자리까지 같다. 쪽에서 보고한 "사후평균 $0.634$"는 $\rho$ 의 대표값이 아니라 꼬리까지 포함한 무게중심이며, 자료가 말하는 비는 $0.473$ 이다. **$n = 8$ 에서 어느 요약값을 쓰느냐가 $33\%$ 를 가른다.**

    **판정은 같다.** 신용구간 $(0.107,\, 2.095)$ 가 1 을 담으므로 등분산과 일관되고, $F$ 검정도 양측 $p = 0.3448$ 로 기각하지 못한다. $P(\rho>1) = 0.1550$ 이 $F$ 검정의 단측 $p$ 값 $0.1724$ 와 가까운 것도 우연이 아니다(연습문제 3 과 같은 대응이다). 둘의 차 $0.017$ 은 자유도 $8.02$ 대 $7$ 에서 온다. 신용구간이 신뢰구간보다 로그 폭으로 $7.5\%$ 좁은 것도 같은 까닭이며, 보기 1 에서 한 집단에 대해 본 $7.4\%$ 와 같은 현상이다.

    $P(\sigma_1^2 > \sigma_2^2 \mid \text{자료}) = 0.155$ 이니 집단 1 의 분산이 더 작을 사후확률이 $0.845$ 다. 방향은 뚜렷하지만 확정하기에는 부족하다.

!!! warning "사후평균이 표본분산보다 크다"
    표본분산은 $2.839$와 $6.000$인데 사후평균은 $3.317$과 $6.956$이다. 각각 17%와 16% 크다.

    사전분포의 영향이 아니라 **역감마분포가 오른쪽으로 치우쳐 있어서** 평균이 최빈값보다 크기 때문이다. 사후 최빈값은 $\beta_n/(\alpha_n+1) = 9.948/5.01 = 1.986$, 중앙값은 $2.702$로 오히려 표본분산보다 작다.

    **분산의 사후분포를 요약할 때는 평균보다 중앙값이나 최빈값이 나을 때가 많다.** 특히 $n$이 작으면 사후평균이 크게 위로 치우친다.

![역감마 사후분포의 최빈값·중앙값·평균과 표본분산](./img/posterior_summary_skew.png)

왼쪽이 집단 1의 사후분포 $\text{Inv-Gamma}(4.01,\ 9.947)$이다. 네 개의 세로선이 같은 분포를 요약하는 네 가지 숫자인데 **순서가 최빈값 $1.986$ < 중앙값 $2.702$ < 표본분산 $2.839$ < 사후평균 $3.305$**이다. 어느 것을 "그 분산"이라고 부르느냐에 따라 보고하는 값이 1.7배까지 달라진다.

원인은 오른쪽 꼬리다. 역감마분포는 $\sigma^2$이 클 가능성을 완전히 배제하지 않으므로 꼬리가 길게 늘어지고, 평균은 그 꼬리에 끌려 올라간다. $n = 8$에서 $\alpha_n = 4.01$이니 꼬리가 특히 두껍다. **분포가 치우쳐 있을 때 평균은 "대표값"이 아니라 "꼬리까지 포함한 무게중심"이다.**

오른쪽이 이 현상의 크기를 표본크기의 함수로 보여 준다. 사후평균/표본분산 비가 $n = 8$에서 $1.16$, 곧 16% 크다. $n = 20$에서 $1.05$, $n = 60$에서 $1.02$로 빠르게 1에 다가간다. 반대로 최빈값은 아래쪽에서 1에 접근한다. **$n$이 30을 넘으면 어느 요약값을 쓰든 실질적 차이가 없지만, $n$이 10 남짓이면 선택이 결과를 바꾼다.**

실무 권고는 분명하다. 작은 표본에서 분산의 사후분포를 한 숫자로 보고해야 한다면 **중앙값**을 쓰라. 평균처럼 꼬리에 끌리지 않고 최빈값처럼 사전분포의 형태에 민감하지도 않다. 더 나은 선택은 물론 한 숫자로 줄이지 않고 신용구간을 함께 보고하는 것이다.

---

## 4. 해석

- 각 $\sigma_i^2$의 사후분포가 자료와 사전분포가 주어졌을 때 집단분산에 대한 모든 정보를 요약한다.
- 사후확률 $P(\sigma_1^2 > \sigma_2^2 \mid \text{자료})$는 고정된 유의수준 없이 관심 질문에 직접 답한다.
- 모호한 사전분포에서 Bayes 신용구간이 빈도주의 신뢰구간을 근사하지만 해석이 다르다. 신용구간은 "참 모수가 이 구간에 있을 확률이 95%"라고 말한다(모형과 사전분포가 주어졌을 때).

!!! danger "이 방법도 정규성을 가정한다"
    "고전적 $F$ 검정과 달리 이 접근은 검정 절차 자체에서 정규성을 가정하지 않는다"는 서술을 볼 수 있으나 **오해를 부른다.**

    가능도가 $x_{ij} \sim \mathcal{N}(\mu_i, \sigma_i^2)$이므로 **정규성 가정이 모형의 핵심에 들어 있다.** 자료가 비정규이면 사후분포 자체가 잘못 설정되며, 그 취약성은 Bartlett 검정이나 $F$ 검정과 본질적으로 같다.

    Bayes 접근이 얻는 것은 (1) 확률적 해석과 (2) 사전정보의 반영이지, 분포 가정으로부터의 자유가 아니다.

    비정규 자료에 로버스트한 Bayes 분석을 하려면 가능도를 $t$ 분포로 바꾸거나 비모수 사전분포(디리클레 과정 등)를 써야 하며, 그러면 켤레성이 깨져 MCMC가 필요하다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 역감마 사전분포 $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)$과 정규 가능도에서 출발하여 위의 사후 모수 $\alpha_n$과 $\beta_n$을 유도하라. 켤레 갱신의 각 단계를 보여라.

</div>

??? success "풀이"

    $\mu$가 알려진 $\mathcal{N}(\mu, \sigma^2)$에서 나온 $n$개 관측값의 가능도는

    $$
    L(\sigma^2) \propto (\sigma^2)^{-n/2} \exp\!\left(-\frac{S}{2\sigma^2}\right)
    $$

    여기서 $S = \sum(x_j - \mu)^2$이다. 역감마 사전분포의 밀도는

    $$
    p(\sigma^2) \propto (\sigma^2)^{-\alpha_0 - 1} \exp\!\left(-\frac{\beta_0}{\sigma^2}\right)
    $$

    곱하면

    $$
    p(\sigma^2 \mid \mathbf{x}) \propto (\sigma^2)^{-(\alpha_0 + n/2) - 1} \exp\!\left(-\frac{\beta_0 + S/2}{\sigma^2}\right)
    $$

    이는 $\text{Inv-Gamma}(\alpha_0 + n/2,\; \beta_0 + S/2)$ 밀도의 핵이다.

    **$\mu$를 모를 때.** 결합 정규-역감마 사전분포를 쓰면 $\mu$를 적분해 낼 때 추가 항이 나타난다.

    $$
    \int \exp\!\left(-\frac{n(\bar x - \mu)^2 + \kappa_0(\mu - m_0)^2}{2\sigma^2}\right) d\mu
    $$

    지수부의 이차식을 $\mu$에 대해 완전제곱으로 정리하면

    $$
    n(\bar x - \mu)^2 + \kappa_0(\mu - m_0)^2 = \kappa_n(\mu - m_n)^2 + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - m_0)^2
    $$

    이고, $\mu$에 대해 적분하면 첫 항이 $\sqrt{2\pi\sigma^2/\kappa_n}$을 낳고(이것이 $\sigma^2$의 지수를 $1/2$만큼 바꾼다) 두 번째 항이 $\beta_n$에 더해진다. 그 결과

    $$
    \beta_n = \beta_0 + \frac{1}{2}\left(S + \frac{\kappa_0 n}{\kappa_n}(\bar{x} - m_0)^2\right)
    $$

    이고 $\alpha_n = \alpha_0 + n/2$이다. 여기서 $S$가 $\bar{x}$에 대한 편차제곱합으로 바뀐다는 점에 주의하라. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 정보 사전분포를 쓰도록 코드를 수정하라. $\alpha_0 = 3$, $\beta_0 = 10$으로 두면 사전분포가 $\sigma^2 = 5$ 근처를 중심으로 한다. 모호한 사전분포로 얻은 신용구간과 비교하라. 표본크기가 작을 때 정보 사전분포가 결과에 어떤 영향을 주는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import invgamma

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        x = np.asarray(x, dtype=float)
        n = x.size
        xbar = x.mean()
        S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    rng = np.random.default_rng(0)

    a_v, b_v = posterior_params(x1, a0=1e-2, b0=1e-2)      # vague
    draws_v = invgamma(a=a_v, scale=b_v).rvs(20000, random_state=rng)

    a_i, b_i = posterior_params(x1, a0=3, b0=10)           # informative
    draws_i = invgamma(a=a_i, scale=b_i).rvs(20000, random_state=rng)

    print(f"Vague:       a={a_v:.2f}, b={b_v:.2f}, mean={draws_v.mean():.2f}, "
          f"CI=({np.percentile(draws_v,2.5):.2f}, {np.percentile(draws_v,97.5):.2f})")
    print(f"Informative: a={a_i:.2f}, b={b_i:.2f}, mean={draws_i.mean():.2f}, "
          f"CI=({np.percentile(draws_i,2.5):.2f}, {np.percentile(draws_i,97.5):.2f})")
    ```

    출력:

    ```text
    Vague:       a=4.01, b=9.95, mean=3.32, CI=(1.13, 9.11)
    Informative: a=7.00, b=19.94, mean=3.32, CI=(1.53, 7.09)
    ```

    !!! note "평균은 거의 그대로이고 구간만 좁아졌다"
        정보 사전분포가 사후평균을 $\sigma^2 = 5$ 쪽으로 끌어당길 것으로 예상되지만, 두 사후평균이 모두 $3.32$로 사실상 같다.

        우연이 아니라 역감마분포의 구조 때문이다. 사후평균은 $\beta_n/(\alpha_n-1)$인데, 정보 사전분포가 $\alpha_n$을 $4.01 \to 7.00$으로, $\beta_n$을 $9.95 \to 19.94$로 **비슷한 비율로** 키웠다. $19.94/6 = 3.32$와 $9.95/3.01 = 3.31$이 우연히 일치한 것이다.

        분명하게 바뀐 것은 **구간의 폭**이다. $(1.13, 9.11)$에서 $(1.53, 7.09)$로, 로그 척도 폭이 $2.087$에서 $1.533$으로 **27% 좁아졌다**. 정보 사전분포가 추가한 정보가 불확실성을 줄인 것이다.

    $\alpha_0 = 3$은 관측값 약 $2\alpha_0 = 6$개어치의 정보에 해당하므로, $n = 8$인 자료와 대등한 영향력을 갖는다. 그래서 구간이 눈에 띄게 좁아진다.

    $n$이 커지면 사전분포의 영향이 줄어들고 두 사후분포가 수렴한다. $n = 100$이면 $\alpha_n$이 $50.01$ 대 $53$으로 6% 차이에 불과하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\mathcal{N}(0, 4)$에서 $n_1 = 30$개, $\mathcal{N}(0, 9)$에서 $n_2 = 30$개를 생성하라. 분산비 $\rho = \sigma_1^2 / \sigma_2^2$의 사후분포를 계산하고 $P(\rho < 1 \mid \text{자료})$를 구하라. 고전적 $F$ 검정의 $p$값과 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import invgamma, f as f_dist

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        n = x.size
        xbar = x.mean()
        S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    rng = np.random.default_rng(42)
    x1 = rng.normal(0, 2, 30)     # sd = 2, so variance = 4
    x2 = rng.normal(0, 3, 30)     # sd = 3, so variance = 9

    print(f"sample variances: {np.var(x1, ddof=1):.3f}, "
          f"{np.var(x2, ddof=1):.3f} (true: 4, 9)")

    a1, b1 = posterior_params(x1)
    a2, b2 = posterior_params(x2)
    s1 = invgamma(a=a1, scale=b1).rvs(50000, random_state=rng)
    s2 = invgamma(a=a2, scale=b2).rvs(50000, random_state=rng)
    ratio = s1 / s2

    print(f"P(rho < 1 | data) = {np.mean(ratio < 1):.4f}")

    F_stat = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    p_val = 2 * min(f_dist.cdf(F_stat, 29, 29), f_dist.sf(F_stat, 29, 29))
    print(f"F-test: F = {F_stat:.4f}, p = {p_val:.4f}")
    ```

    출력:

    ```text
    sample variances: 2.413, 5.828 (true: 4, 9)
    P(rho < 1 | data) = 0.9908
    F-test: F = 0.4141, p = 0.0205
    ```

    두 접근이 같은 방향의 결론을 준다. 사후확률 $P(\rho < 1) = 0.991$이 높고 $F$ 검정의 $p$값 $0.021$이 작다.

    **두 수치의 관계.** 양측 $p$값 $0.0205$의 절반인 단측 $p$값이 $0.0103$이고, $1 - 0.9908 = 0.0092$와 매우 가깝다. **우연이 아니다.** 무정보 사전분포에서 Bayes 사후확률과 빈도주의 단측 $p$값이 수치적으로 일치하는 것이 위치-척도 모수에 대한 일반적 성질이다.

    작은 차이($0.0092$ 대 $0.0103$)는 앞서 논한 $n/2$ 대 $(n-1)/2$ 자유도 차이에서 온다.

    **해석은 다르다.** $p = 0.021$은 "등분산이 참이라면 이만큼 극단적인 자료가 나올 확률이 2.1%"이고, $P(\rho < 1) = 0.991$은 "이 자료를 보았을 때 $\sigma_1^2 < \sigma_2^2$일 확률이 99.1%"이다. 후자가 대체로 사람들이 알고 싶어 하는 것이며, $p$값을 이렇게 해석하는 흔한 오류의 근원이기도 하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> Bayes 접근이 분산비 자체보다 그 로그($\log(\sigma_1^2/\sigma_2^2)$)로 요약하는 이유를 설명하라. 사후표본 50,000개를 뽑아 $\rho$와 $\log(\rho)$의 히스토그램을 그려라. 어느 쪽이 대칭에 가까운가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import invgamma, skew
    import matplotlib.pyplot as plt

    def posterior_params(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        n = x.size; xbar = x.mean(); S = np.sum((x - xbar)**2)
        k_n = k0 + n
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / k_n) * (xbar - m0)**2)
        return a_n, b_n

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    rng = np.random.default_rng(0)

    a1, b1 = posterior_params(x1)
    a2, b2 = posterior_params(x2)
    s1 = invgamma(a=a1, scale=b1).rvs(50000, random_state=rng)
    s2 = invgamma(a=a2, scale=b2).rvs(50000, random_state=rng)
    ratio = s1 / s2

    print(f"skewness of ratio:     {skew(ratio):.3f}")
    print(f"skewness of log ratio: {skew(np.log(ratio)):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(ratio, bins=60, edgecolor='k', alpha=0.7)
    axes[0].set_title("Posterior of ratio")
    axes[1].hist(np.log(ratio), bins=60, edgecolor='k', alpha=0.7)
    axes[1].set_title("Posterior of log(ratio)")
    plt.tight_layout()
    plt.show()
    ```

    출력:

    ```text
    skewness of ratio:     9.518
    skewness of log ratio: -0.017
    ```

    ![분산비와 로그 분산비의 사후분포](./img/bayesian_var_test_308.png)

    **차이가 극적이다.** 비의 왜도가 $9.52$인데 로그를 취하면 $-0.017$로 사실상 완벽히 대칭이 된다.

    비 $\rho = \sigma_1^2/\sigma_2^2$은 아래로 0에 의해 유계이고 오른쪽 꼬리가 무겁다. 로그 변환은 $(0, \infty)$를 $(-\infty, \infty)$로 보내 훨씬 대칭에 가까운 분포를 만든다.

    **왜 정확히 대칭이 되는가.** $\sigma_1^2$과 $\sigma_2^2$이 독립인 역감마이므로 $\rho$는 척도가 조정된 $F$ 분포를 따르고, $\log \rho$는 Fisher의 $z$ 분포를 따른다. $\alpha_1 = \alpha_2$(여기서 둘 다 $4.01$)이면 $\log\rho$의 분포가 그 중심을 기준으로 **정확히 대칭**이다. 15.3절 [F 분포와 자유도](../f_test/f_distribution_details.md) 연습문제 4에서 확인한 역수 성질과 같은 사실이다.

    대칭 분포는 평균과 신용구간으로 요약하기 쉽고, 로그 척도의 백분위 구간은 상대적 관점에서 더 균형 잡힌 구간으로 되돌아간다.

    (다만 15.6절 [붓스트랩 분산 검정](bootstrap_var_test.md)에서 논했듯, **백분위 구간 자체는 단조변환에 불변**이므로 신용구간의 끝점은 어느 척도에서 계산하든 같다. 로그가 도움이 되는 것은 시각화, 정규근사, 사후 요약통계량의 해석에서다.) $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> Bayes 분산 검정과 고전적 $F$ 검정이 실질적으로 다른 결론을 낼 상황을 기술하라. 표본크기, 사전분포의 정보량, 정규성 이탈의 역할을 고려하라.

</div>

??? success "풀이"

    두 접근이 가장 크게 갈리는 상황은 다음과 같다.

    **1. 작은 표본 + 정보 사전분포.** $n$이 매우 작고(집단당 5 정도) 연구자가 등분산을 지지하는 강한 사전정보를 갖고 있으면, 표본분산이 중간 정도로 달라도 $\rho$의 사후분포가 1 근처에 집중된다. $F$ 검정은 사전정보를 무시하고 잡음이 섞인 표본추정값만으로 기각 여부를 정한다.

    연습문제 2에서 $n = 8$, $\alpha_0 = 3$인 사전분포가 신용구간을 27% 좁혔음을 보았다. $n = 5$이면 효과가 더 크다.

    **2. 비정규 자료.** $F$ 검정은 정규성 이탈에 매우 민감하다. 꼬리가 두꺼우면 표본분산이 부풀려져 제1종 오류가 커진다.

    **그러나 여기 제시한 Bayes 모형도 가능도에서 정규성을 가정하므로 같은 오설정을 겪는다.** 위 경고 상자에서 지적한 대로이다. 이것이 두 방법이 **다르게** 반응하는 경우가 아니라 **똑같이** 실패하는 경우이다.

    Bayes 틀이 유리해지는 지점은 로버스트 가능도(스튜던트 $t$ 등)로 확장할 수 있다는 것이며, 그러면 이상점을 우아하게 처리한다. 하지만 그것은 다른 모형이지 위 코드가 아니다.

    **3. 단측 질문.** Bayes 접근은 $P(\sigma_1^2 > \sigma_2^2)$에 자연스럽게 답하지만, $F$ 검정은 보통 양측으로 설정되며 단측 대립가설에는 명시적 수정이 필요하다. 질문이 방향성을 가질 때 사후확률이 더 직접적이고 해석하기 쉬운 답을 준다.

    연습문제 3에서 $P(\rho < 1) = 0.991$이 단측 $p$값 $0.010$과 수치적으로 대응했음을 보았다. **무정보 사전분포에서는 두 수치가 사실상 같으므로 이 항목의 차이는 계산이 아니라 해석에 있다.**

    **4. 결정이론적 활용.** Bayes 사후분포는 손실함수와 결합하여 최적 결정을 내리는 데 쓸 수 있다. 예컨대 "$\rho > 1.5$이면 공정을 조정한다"는 결정 규칙에 대해 $P(\rho > 1.5 \mid \text{자료})$를 직접 계산할 수 있다. $F$ 검정은 $H_0: \rho = 1$이라는 점가설만 다루므로 이런 질문에 직접 답하지 못한다.

    **요약.** 무정보 사전분포와 정규 자료에서는 두 방법이 수치적으로 거의 같다. 차이가 나는 것은 (1) 사전정보가 있을 때, (2) 관심 질문이 점가설이 아닐 때이며, (3) 비정규성은 **두 방법 모두의 문제**이지 한쪽의 장점이 아니다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span> 경고 상자는 사후평균이 표본분산보다 "각각 17%와 16% 크다"고 하고, 그림은 네 요약값의 순서가 **최빈값 $1.986$ < 중앙값 $2.702$ < 표본분산 $2.839$ < 사후평균 $3.305$** 라고 한다. 무정보 극한($\alpha_n = n/2$, $\beta_n = S/2$)에서 이 네 수를 모두 $S$ 와 $n$ 만으로 적고 다음에 답하라.

**(1)** 사후평균과 사후최빈값을 $s^2$ 으로 나눈 값을 $n$ 만의 함수로 적고 $n = 8$ 에서 값을 구하시오. 그림 오른쪽의 "$n = 8$ 에서 $1.16$, $n = 20$ 에서 $1.05$, $n = 60$ 에서 $1.02$"가 맞는지 확인하시오. 두 집단의 $n$ 이 똑같이 8 인데 경고 상자는 왜 "17%와 16%"라고 다르게 적었는가.

**(2)** $\chi^2_n$ 의 중앙값이 $n - \tfrac23$ 에 가깝다는 사실을 써서, 그림이 보인 네 수의 순서가 **모든 $n$ 에서** 성립함을 보이시오.

**(3)** 사후평균이 $s^2$ 의 $1\%$ 안에 들어오려면 $n$ 이 얼마여야 하는가. 최빈값은 어떤가. 쪽은 "$n$ 이 30 을 넘으면 어느 요약값을 쓰든 실질적 차이가 없다"고 하는데, 그 말이 세 요약값 모두에 대해 옳은지 판정하시오.

**(4)** $n \le 2$ 이면 무엇이 무너지는가.

</div>

??? success "풀이"

    무정보 극한에서 $\sigma^2 \mid \mathbf x \sim \text{Inv-Gamma}(n/2,\, S/2)$ 이므로

    $$
    \text{최빈값} = \frac{\beta_n}{\alpha_n + 1} = \frac{S}{n+2}, \qquad
    \text{사후평균} = \frac{\beta_n}{\alpha_n - 1} = \frac{S}{n-2}
    $$

    이고, 보기 1 의 추축량 $S/\sigma^2 \sim \chi^2_n$ 에서 $\sigma^2 = S/W$ 가 $W$ 의 감소함수이므로 중앙값은 중앙값으로 옮겨 간다.

    $$
    \text{중앙값} = \frac{S}{\operatorname{med}(\chi^2_n)}, \qquad
    s^2 = \frac{S}{n-1}
    $$

    **네 수가 모두 $S$ 를 분자로 갖고 분모만 다르다.** $n = 8$, $S = 19.875$ 를 넣으면 $19.875/10 = 1.9875$, $19.875/7.3441 = 2.7062$, $19.875/7 = 2.8393$, $19.875/6 = 3.3125$ 이고 그림의 $1.986$, $2.702$, $2.839$, $3.305$ 와 맞는다(차이는 $\beta_0 = 0.01$ 몫이다).

    **(1) 비는 $S$ 가 약분되어 $n$ 만의 함수다.**

    $$
    \frac{\text{사후평균}}{s^2} = \frac{n-1}{n-2}, \qquad
    \frac{\text{최빈값}}{s^2} = \frac{n-1}{n+2}
    $$

    $n = 8$ 이면 $7/6 = 1.1667$ 과 $7/10 = 0.7000$ 이다. 평균은 $16.7\%$ 위, 최빈값은 $30\%$ 아래다.

    쪽의 기본 사전분포($\alpha_0 = 10^{-2}$)까지 넣으면 분모가 $2\alpha_n - 2 = n - 1.98$ 이 되어

    $$
    \frac{\text{사후평균}}{s^2} \approx \frac{n-1}{n-1.98}
    $$

    이고, $n = 8, 20, 60$ 에서 $7/6.02 = 1.1628$, $19/18.02 = 1.0544$, $59/58.02 = 1.0169$ 다. **그림의 $1.16$, $1.05$, $1.02$ 가 그대로 나온다.** ($n=20$ 을 무정보 극한값 $1.0556$ 으로 읽으면 $1.06$ 이 되므로, 그림은 사전분포를 넣은 쪽이다.)

    **"17%와 16%"는 몬테카를로 오차다.** 두 집단 모두 $n = 8$ 이므로 비의 참값은 같아야 한다. 실제로 닫힌 꼴은 집단 1 이 $3.304844/2.839286 = 1.1640$, 집단 2 가 $6.980143/6 = 1.1634$ 로 **둘 다 $16.4\%$** 다. 경고 상자의 $3.317$ 과 $6.956$ 은 2 만 개 사후표본의 평균이고, 그것으로 재면 $1.1683$ 과 $1.1593$ 이 되어 $17\%$ 와 $16\%$ 로 갈린다. **없는 차이를 본 것이다.**

    **(2) 분모의 크기 순서가 전부다.** 네 수가 $S/(\text{분모})$ 꼴이므로 분모가 작을수록 값이 크다. 분모는

    $$
    n - 2 \;<\; n-1 \;<\; \operatorname{med}(\chi^2_n) \;<\; n+2
    $$

    이다. 가운데 부등식이 핵심인데, 윌슨–힐퍼티 근사 $\operatorname{med}(\chi^2_n) \approx n\left(1 - \tfrac{2}{9n}\right)^3 = n - \tfrac23 + O(n^{-1})$ 에서 $\operatorname{med}(\chi^2_n) \approx n - \tfrac23 > n-1$ 임을 읽는다($n = 8$ 에서 참값 $7.3441$, 근사 $7.3333$ 으로 둘 다 $7$ 보다 크다). 또 $\chi^2_n$ 의 중앙값은 평균 $n$ 보다 작으므로 $\operatorname{med}(\chi^2_n) < n < n+2$ 다. 역수를 취해 순서를 뒤집으면

    $$
    \frac{S}{n+2} \;<\; \frac{S}{\operatorname{med}(\chi^2_n)} \;<\; \frac{S}{n-1} \;<\; \frac{S}{n-2}
    $$

    곧 **최빈값 < 중앙값 < 표본분산 < 사후평균**이 모든 $n$ 에서 성립한다. 그림의 순서는 $n = 8$ 의 우연이 아니다. $\square$

    **(3) 평균은 $n \ge 102$, 최빈값은 $n \ge 298$.** $\dfrac{n-1}{n-2} \le 1.01$ 은 $n - 1 \le 1.01n - 2.02$, 곧 $n \ge 102$ 와 같고, $\dfrac{n-1}{n+2} \ge 0.99$ 는 $0.01n \ge 2.98$, 곧 $n \ge 298$ 과 같다. **최빈값이 세 배 느리게 수렴한다.**

    $n = 30$ 에서 세 요약값의 편차를 재면

    | 요약값 | $\text{값}/s^2$ | 편차 |
    |---|---|---|
    | 사후평균 | $29/28 = 1.0357$ | $+3.6\%$ |
    | 중앙값 | $29/29.336 = 0.9885$ | $-1.1\%$ |
    | 최빈값 | $29/32 = 0.9063$ | $-9.4\%$ |

    이다. **쪽의 말은 평균과 중앙값에만 옳다.** 최빈값은 $n = 30$ 에서도 아직 $9.4\%$ 낮다. 쪽이 권한 "중앙값을 쓰라"는 결론은 오히려 강화된다. 중앙값은 $n = 8$ 에서도 $-4.7\%$ 로 세 요약값 가운데 가장 작게 틀린다.

    **(4) $n = 2$ 면 사후평균이 존재하지 않는다.** $\alpha_n = n/2 = 1$ 이라 $\beta_n/(\alpha_n - 1)$ 의 분모가 0 이 되고, 역감마분포의 평균은 $\alpha > 1$ 에서만 유한하다. $n = 1$ 이면 $S = 0$ 이어서 $\beta_n = \beta_0$ 만 남고 사후분포가 사전분포로 되돌아간다. 중앙값과 최빈값은 두 경우 모두 유한하다.

    **세 요약값 가운데 존재 조건을 갖는 것은 평균뿐이다.** 치우친 분포에서 평균을 대표값으로 쓰지 말라는 쪽의 권고가, 극단으로 보내면 "그 값이 아예 없다"는 데까지 간다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span> 보기 2 는 $P(\rho > 1) = 0.1550$ 이 $F$ 검정의 단측 $p$ 값 $0.1724$ 와 "가깝다"고 하고, 연습문제 3 은 $1 - P(\rho < 1) = 0.0092$ 가 단측 $p$ 값 $0.0103$ 과 "매우 가깝다"고 한다. **두 번 다 베이즈 쪽이 더 작다.** 이것이 우연이 아님을 보이자. $n_1 = n_2 = n$ 이고 무정보 극한이라 한다.

**(1)** $P(\rho > 1 \mid \mathbf x) = P\bigl(F(n,n) < F_{\text{obs}}\bigr)$ 임을 보이시오. 여기서 $F_{\text{obs}} = s_1^2/s_2^2$ 이다.

**(2)** 빈도주의 단측 $p$ 값이 **같은 식에서 자유도만 $n-1$ 인 것**임을 적고, 두 수가 모든 자료에서 정확히 같아지는 사전분포를 말하시오.

**(3)** 왜 베이즈 쪽이 항상 더 작은가. 자유도를 키우면 $F(d,d)$ 가 어디로 가는지로 설명하고, 보기 2 와 연습문제 3 의 수로 확인하시오.

</div>

??? success "풀이"

    **(1) 비의 사후분포를 꼬리에서 읽는다.** 무정보 극한에서 $2\alpha_n = n$, $2\beta_{n,i} = S_i$ 이므로 보기 1 의 추축량이 $\sigma_i^2 = S_i/W_i$, $W_i \sim \chi^2_n$ (독립)을 준다. 따라서

    $$
    \rho = \frac{\sigma_1^2}{\sigma_2^2} = \frac{S_1}{S_2}\cdot\frac{W_2}{W_1},
    \qquad \frac{W_2}{W_1} = \frac{W_2/n}{W_1/n} \sim F(n,n)
    $$

    이고 $P(\rho > 1) = P\bigl(F > S_2/S_1\bigr)$, $F \sim F(n,n)$ 이다. 여기서 $W_2/W_1$ 과 $W_1/W_2$ 가 **같은 분포**를 가지므로($W_1, W_2$ 가 동일분포 독립이다) 임의의 $c > 0$ 에 대해 $P(F > c) = P(1/F > c) = P(F < 1/c)$ 다. 그러므로

    $$
    P(\rho > 1 \mid \mathbf x) = P\!\left(F(n,n) < \frac{S_1}{S_2}\right) = P\bigl(F(n,n) < F_{\text{obs}}\bigr)
    $$

    이다. 마지막 등식은 $n_1 = n_2$ 라서 $S_1/S_2 = (n-1)s_1^2/\{(n-1)s_2^2\} = F_{\text{obs}}$ 이기 때문이다. $\square$

    **(2) 빈도주의는 자유도만 하나 적다.** $F_{\text{obs}} < 1$ 쪽의 단측 $p$ 값은 정의상

    $$
    p_{\text{단측}} = P\bigl(F(n-1,\,n-1) \le F_{\text{obs}}\bigr)
    $$

    이다. (1)의 결과와 **글자 하나 차이 없이 같은 함수**이고 자유도만 $n$ 대 $n-1$ 이다. 보기 1 이 본 대로 $2\alpha_n = n - 1$ 이 되게 하려면 $\alpha_0 = -1/2$ 로 두면 되고($\beta_0 = 0$, $\kappa_0 = 0$ 과 함께), 그러면 두 수가 **자료와 무관하게 항등적으로 같아진다**. 연습문제 3 이 "일반적 성질"이라고 부른 일치가 이 항등식이다.

    **(3) 자유도가 커지면 $F(d,d)$ 가 1 로 모인다.** $W_1/d, W_2/d \to 1$ (확률수렴)이므로 $F(d,d) \to 1$ 이고, 분포가 1 쪽으로 좁아지면 1 에서 떨어진 고정점 $F_{\text{obs}}$ 아래의 확률은 **$d$ 가 커질수록 작아진다.** 자유도가 $n-1$ 에서 $n$ 으로 하나 늘어난 베이즈 쪽이 더 작은 꼬리확률을 내놓는 것이 그래서 필연이다.

    | 자료 | $F_{\text{obs}}$ | $P(F(n,n) < F_{\text{obs}})$ | 단측 $p = P(F(n{-}1,n{-}1) < F_{\text{obs}})$ | 비 |
    |---|---|---|---|---|
    | 보기 2 ($n=8$) | $0.473214$ | $0.155188$ | $0.172386$ | $0.900$ |
    | 연습문제 3 ($n=30$) | $0.414089$ | $0.009195$ | $0.010241$ | $0.898$ |

    가운데 열의 $0.155188$ 은 쪽이 보고한 사후확률 $0.155034$ 와 $1.5\times10^{-4}$ 만 다르고(차이는 $\beta_0 = 0.01$ 몫이다), $0.009195$ 는 연습문제 3 의 $1 - 0.990814 = 0.009186$ 과 맞는다. **두 자료 모두 베이즈 쪽이 약 $10\%$ 작다.**

    기본 사전분포는 자료가 말하는 것보다 **언제나 조금 더 센 증거**를 보고한다. 보기 1 이 본 "신용구간이 $7.4\%$ 좁다"와 같은 자유도 하나이고, 연습문제 9 가 그 대가를 포함률로 잰다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 노트 상자는 세 가지를 주장한다. ① $\kappa_0 \to 0$ 극한이 사전분포 $p(\mu, \sigma^2) \propto (\sigma^2)^{-\alpha_0 - 3/2}$ 에 해당한다. ② 그것은 제프리스 사전분포 $1/\sigma^2$ 과 다르다. ③ $\alpha_0 = -1/2$ 로 두면 빈도주의와 맞는다. **셋이 하나의 계산임을 보이자.**

**(1)** 결합 사전분포를 $\mu$ 와 $\sigma^2$ 의 함수로 적고 $\kappa_0 \to 0$, $\beta_0 \to 0$ 극한을 취해 ①을 확인하시오. 지수부가 1 로 가는데도 $(\sigma^2)^{-1/2}$ 가 남는 까닭을 밝히시오.

**(2)** $\alpha_0 = -1/2$ 가 정확히 $p(\mu,\sigma^2) \propto 1/\sigma^2$ 임을 확인하고, **그 사전분포에서 출발해** $\mu$ 를 적분해 내어 $\sigma^2 \mid \mathbf x \sim \text{Inv-Gamma}\bigl(\tfrac{n-1}{2},\, \tfrac S2\bigr)$, 곧 $S/\sigma^2 \sim \chi^2_{n-1}$ 을 유도하시오.

**(3)** 사전분포에 붙은 $(\sigma^2)^{-1/2}$ 하나가 왜 자유도 하나와 같은가. "편차제곱합에 0 을 더하는 관측값 한 개"라는 말로 설명하고, 보기 1 의 "$15\%$ 더 많은 정보"가 그것임을 밝히시오.

</div>

??? success "풀이"

    **(1) 조건부 사전분포의 정규화상수가 범인이다.** $\mu \mid \sigma^2 \sim \mathcal N(m_0, \sigma^2/\kappa_0)$ 과 $\sigma^2 \sim \text{Inv-Gamma}(\alpha_0,\beta_0)$ 을 곱하면

    $$
    p(\mu,\sigma^2) = \sqrt{\frac{\kappa_0}{2\pi\sigma^2}}\,
    \exp\!\left(-\frac{\kappa_0(\mu-m_0)^2}{2\sigma^2}\right)
    \cdot \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)}(\sigma^2)^{-\alpha_0-1}
    \exp\!\left(-\frac{\beta_0}{\sigma^2}\right)
    $$

    이다. $\sigma^2$ 의 거듭제곱을 모으면

    $$
    p(\mu,\sigma^2) \;\propto\; \sqrt{\kappa_0}\;(\sigma^2)^{-\alpha_0 - 3/2}
    \exp\!\left(-\frac{\kappa_0(\mu-m_0)^2 + 2\beta_0}{2\sigma^2}\right)
    $$

    이고, $\kappa_0 \to 0$ 과 $\beta_0 \to 0$ 에서 지수부의 분자가 0 으로 가 지수함수가 1 이 된다. $\sqrt{\kappa_0}$ 는 $(\mu,\sigma^2)$ 에 의존하지 않는 상수라 비례관계에서 지워진다(그 대가로 사전분포가 부적절해진다). 남는 것은

    $$
    p(\mu,\sigma^2) \;\propto\; (\sigma^2)^{-\alpha_0 - 3/2}
    $$

    로 노트 상자의 ①과 같다.

    **$(\sigma^2)^{-1/2}$ 가 남는 까닭.** 그것은 지수부에서 온 것이 아니라 $\mathcal N(m_0, \sigma^2/\kappa_0)$ 의 **정규화상수** $\sqrt{\kappa_0/(2\pi\sigma^2)}$ 에서 왔다. 켤레성을 얻으려고 $\mu$ 의 사전분산을 $\sigma^2/\kappa_0$ 으로 **$\sigma^2$ 에 묶어** 둔 탓에 정규화상수가 $\sigma^2$ 에 의존한다. $\kappa_0 \to 0$ 은 $\mu$ 의 사전분포를 평평하게 만들 뿐이고, 묶어 둔 흔적인 $(\sigma^2)^{-1/2}$ 는 지우지 못한다. 보기 1 의 풀이가 "$\kappa_0 \to 0$ 극한이 숨겨 들여온 $(\sigma^2)^{-1/2}$"라고 부른 것이 이 인수다.

    **(2) $\alpha_0 = -1/2$ 가 바로 제프리스다.** $-\alpha_0 - 3/2 = \tfrac12 - \tfrac32 = -1$ 이므로

    $$
    p(\mu,\sigma^2) \;\propto\; (\sigma^2)^{-1} = \frac{1}{\sigma^2}
    $$

    이고, 이것이 정규 위치–척도 모형의 표준 제프리스 사전분포($(\mu, \log\sigma)$ 에 평평한 분포)다. **②와 ③이 같은 말이었다.** 이 사전분포로 갱신하면

    $$
    p(\mu,\sigma^2\mid\mathbf x) \;\propto\; \frac{1}{\sigma^2}\,(\sigma^2)^{-n/2}
    \exp\!\left(-\frac{S + n(\bar x - \mu)^2}{2\sigma^2}\right)
    $$

    이고($\sum(x_j-\mu)^2 = S + n(\bar x - \mu)^2$ 을 썼다), $\mu$ 에 대해 적분하면

    $$
    \int_{-\infty}^{\infty}\exp\!\left(-\frac{n(\bar x-\mu)^2}{2\sigma^2}\right)d\mu
    = \sqrt{\frac{2\pi\sigma^2}{n}} \;\propto\; (\sigma^2)^{1/2}
    $$

    이므로

    $$
    p(\sigma^2\mid\mathbf x) \;\propto\; (\sigma^2)^{-n/2 - 1 + 1/2}
    \exp\!\left(-\frac{S}{2\sigma^2}\right)
    = (\sigma^2)^{-\frac{n-1}{2}-1}\exp\!\left(-\frac{S}{2\sigma^2}\right)
    $$

    곧 $\sigma^2 \mid \mathbf x \sim \text{Inv-Gamma}\bigl(\tfrac{n-1}2, \tfrac S2\bigr)$ 다. 보기 1 의 추축량 $2\beta_n/\sigma^2 \sim \chi^2_{2\alpha_n}$ 에 $2\alpha_n = n-1$, $2\beta_n = S$ 를 넣으면

    $$
    \frac{S}{\sigma^2} \sim \chi^2_{n-1}
    $$

    으로 15.2절의 빈도주의 추축량과 **완전히 같다.** $\square$

    **(3) $(\sigma^2)^{-1/2}$ 는 제곱합을 늘리지 않는 관측값 한 개다.** 가능도에서 관측값 하나가 하는 일은

    $$
    (\sigma^2)^{-1/2}\exp\!\left(-\frac{(x_j-\mu)^2}{2\sigma^2}\right)
    $$

    두 몫이다. $(\sigma^2)^{-1/2}$ 는 $\alpha_n$ 을 $\tfrac12$ 올리고(자유도 $2\alpha_n$ 을 1 올리고), 지수부는 $\beta_n$ 에 $\tfrac12(x_j - \mu)^2$ 을 더한다. 그런데 (1)에서 남은 사전분포의 $(\sigma^2)^{-1/2}$ 에는 **지수부가 딸려 있지 않다.** 그러므로 그것은 $x_j = \bar x$ 에 정확히 놓인 가짜 관측값 하나와 대수적으로 같다. 자유도는 1 늘지만 편차제곱합 $S$ 는 조금도 늘지 않는다.

    **분모는 그대로인데 자유도만 커지므로 사후분포가 자료가 허락한 것보다 좁아진다.** 보기 1 에서 자유도가 $8.02$ 대 $7$ 로 $8.02/7 = 1.146$, 곧 $15\%$ 많았던 것이 이 가짜 관측값 하나다. 그 결과가 로그 폭 $7.4\%$ 의 축소이고, 연습문제 9 가 그것을 포함률의 손실로 환산한다. 켤레성이라는 편리함에는 자유도 하나의 값이 매겨져 있다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 경고 상자는 "그 취약성은 Bartlett 검정이나 $F$ 검정과 본질적으로 같다"고 말만 한다. **수로 재자.** $n = 8$ 이고 참 분산이 모두 1 인 모집단에서 자료를 뽑아 쪽의 기본 사전분포로 $95\%$ 신용구간을 만들고, 그 구간이 참 분산을 담는 비율(빈도주의 포함률)을 센다.

**(1)** 정규자료에서도 포함률이 명목 $95\%$ 에 못 미칠 것을 **닫힌 꼴로** 예측하시오. (구간은 자유도 $2\alpha_n$ 으로 만들었지만 $S/\sigma^2$ 의 실제 분포는 $\chi^2_{n-1}$ 이다.)

**(2)** 첨도 $\beta_2$ 가 $1.8$(균일), $3$(정규), $9$(지수)인 세 모집단에서 5 만 번 모의해 포함률과 양쪽으로 빠지는 비율을 재시오.

**(3)** $\alpha_0 = -1/2$ 로 바꾸면 어느 줄이 고쳐지고 어느 줄이 안 고쳐지는가. 그 답이 경고 상자의 주장과 어떻게 맞물리는지 쓰시오.

</div>

??? success "풀이"

    **(1) 자유도 둘을 맞바꾼 확률이다.** 신용구간은 보기 1 의 추축량으로

    $$
    \left(\frac{2\beta_n}{\chi^2_{0.975,\,2\alpha_n}},\; \frac{2\beta_n}{\chi^2_{0.025,\,2\alpha_n}}\right)
    $$

    이고 무정보 극한에서 $2\beta_n \approx S$ 다. 이 구간이 참값 $\sigma^2$ 을 담는다는 것은

    $$
    \chi^2_{0.025,\,2\alpha_n} \;<\; \frac{S}{\sigma^2} \;<\; \chi^2_{0.975,\,2\alpha_n}
    $$

    과 같다. 그런데 정규자료에서 $S/\sigma^2$ 의 **실제** 분포는 $\chi^2_{n-1}$ 이다. 그러므로

    $$
    \text{포함률} = F_{n-1}\bigl(\chi^2_{0.975,\,2\alpha_n}\bigr) - F_{n-1}\bigl(\chi^2_{0.025,\,2\alpha_n}\bigr)
    $$

    이고, $2\alpha_n = n-1$ 이면 정의상 정확히 $0.95$, $2\alpha_n = n$ 이면 그보다 작다. **구간을 만들 때 쓴 자유도와 자료가 실제로 갖는 자유도가 어긋난 만큼이 손실이다.**

    **(2)–(3) 수로.**

    ```python
    import numpy as np
    from scipy.stats import chi2

    def coverage(x, sig2, a0, b0=1e-2, k0=1e-6, m0=0.0):
        """95% 신용구간이 참 분산을 담는 비율과 양쪽으로 빠지는 비율.

        구간은 보기 1 의 추축량 2 b_n / sigma^2 ~ chi2(2 a_n) 으로 만든다.
        자료를 B x n 행렬로 받아 한 번에 B 개의 구간을 만든다.
        """
        n = x.shape[1]
        xbar = x.mean(axis=1)
        S = ((x - xbar[:, None]) ** 2).sum(axis=1)
        a_n = a0 + n / 2.0
        b_n = b0 + 0.5 * (S + (k0 * n / (k0 + n)) * (xbar - m0) ** 2)
        lo = 2 * b_n / chi2(2 * a_n).ppf(0.975)
        hi = 2 * b_n / chi2(2 * a_n).ppf(0.025)
        return np.mean((lo < sig2) & (sig2 < hi)), np.mean(sig2 <= lo), np.mean(sig2 >= hi)

    rng = np.random.default_rng(0)
    n, B = 8, 50_000

    # 참 분산이 모두 1 이고 첨도만 다른 세 모집단
    cases = [
        ("정규  (b2=3.0)", rng.normal(5.0, 1.0, (B, n)), 1.0),
        ("균일  (b2=1.8)", rng.uniform(0.0, np.sqrt(12.0), (B, n)), 1.0),
        ("지수  (b2=9.0)", rng.exponential(1.0, (B, n)), 1.0),
    ]

    print(f"n = {n}, 반복 {B:,} 회, 명목 95% 신용구간의 실제 포함률")
    for tag, a0 in (("a0 = 1e-2 (쪽의 기본값)", 1e-2), ("a0 = -1/2 (제프리스)", -0.5)):
        print(f"\n[{tag}]   자유도 2a_n = {2 * (a0 + n / 2.0):.2f}")
        print("                   포함률   참값<하한   참값>상한")
        for name, x, s2 in cases:
            c, d, u = coverage(x, s2, a0=a0)
            print(f"  {name}    {c:.4f}     {d:.4f}      {u:.4f}")

    # 정규자료면 포함률이 닫힌 꼴로도 나온다: S/sigma^2 ~ chi2(n-1) 이므로
    print()
    for a0 in (1e-2, -0.5):
        d = 2 * (a0 + n / 2.0)
        exact = chi2(n - 1).cdf(chi2(d).ppf(0.975)) - chi2(n - 1).cdf(chi2(d).ppf(0.025))
        print(f"정규 닫힌 꼴: 자유도 {d:.2f} 로 만든 구간을 chi2({n - 1}) 로 재면 {exact:.4f}")
    ```

    출력:

    ```text
    n = 8, 반복 50,000 회, 명목 95% 신용구간의 실제 포함률

    [a0 = 1e-2 (쪽의 기본값)]   자유도 2a_n = 8.02
                       포함률   참값<하한   참값>상한
      정규  (b2=3.0)    0.9361     0.0146      0.0493
      균일  (b2=1.8)    0.9794     0.0000      0.0205
      지수  (b2=9.0)    0.7365     0.0694      0.1941

    [a0 = -1/2 (제프리스)]   자유도 2a_n = 7.00
                       포함률   참값<하한   참값>상한
      정규  (b2=3.0)    0.9507     0.0256      0.0237
      균일  (b2=1.8)    0.9909     0.0005      0.0086
      지수  (b2=9.0)    0.7858     0.0857      0.1285

    정규 닫힌 꼴: 자유도 8.02 로 만든 구간을 chi2(7) 로 재면 0.9345
    정규 닫힌 꼴: 자유도 7.00 로 만든 구간을 chi2(7) 로 재면 0.9500
    ```

    **정규자료: 닫힌 꼴이 맞는다.** 모의값 $0.9361$ 이 예측값 $0.9345$ 와 맞고(5 만 번의 표준오차는 $0.0011$ 이다), $\alpha_0 = -1/2$ 로 바꾸면 $0.9507$ 로 명목값 $0.9500$ 을 정확히 회복한다. 자유도 하나가 포함률 $1.6$ 퍼센트포인트다. 빠지는 방향도 한쪽으로 쏠려 있다($0.0146$ 대 $0.0493$). 자유도를 키우면 $\chi^2$ 분위수가 둘 다 커지고 구간의 양 끝이 둘 다 **내려가므로**, 구간이 참값 아래에 놓이는 일이 더 흔해지기 때문이다. 제프리스로 고치면 $0.0256$ 대 $0.0237$ 로 균형이 맞는다.

    **첨도가 방향을 정한다.** 균일분포($\beta_2 = 1.8 < 3$)에서는 포함률이 $0.9794$ 로 명목값을 **넘는다.** $\operatorname{Var}(S^2)$ 이 4 차 적률에 달려 있으므로 첨도가 3 보다 작으면 $S$ 가 $\chi^2_{n-1}$ 이론보다 덜 흔들리고, 그만큼 구간이 필요 이상으로 넓어진다. 보수적이지만 낭비다.

    지수분포($\beta_2 = 9$)에서는 반대로 무너진다. **$95\%$ 라고 적힌 구간이 실제로는 $73.7\%$ 만 담는다.** 빠지는 확률 $0.2635$ 가 명목 $0.05$ 의 $5.3$ 배이고, 그 가운데 $0.1941$ 은 참 분산이 **상한보다 위**에 있는 경우다. 치우친 자료에서는 대부분의 표본이 작은 $S$ 를 주고 드물게 큰 $S$ 를 주므로, 구간이 보통 너무 낮게 놓인다.

    **자유도 수정은 정규성을 사 주지 못한다.** $\alpha_0 = -1/2$ 는 정규자료의 $0.9345$ 를 $0.9500$ 으로 **정확히** 고치지만, 지수자료는 $0.7365$ 에서 $0.7858$ 로 올려 놓을 뿐 여전히 $16$ 퍼센트포인트가 모자란다. 자유도 하나는 **모형 안의 산수**이고 첨도가 틀어 놓는 것은 **모형 자체**다. 경고 상자가 "Bayes 접근이 얻는 것은 확률적 해석과 사전정보의 반영이지 분포 가정으로부터의 자유가 아니다"라고 한 말이 이 표의 마지막 줄이다. 로버스트하게 하려면 가능도를 바꾸어야 하고, 그러면 켤레성이 깨진다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 보기 2 의 코드에는 `assert a1 == a2, "두 집단의 a_n 이 같아야 F 로 적힌다"` 가 있다. **그 가정을 깨 보자.** $n_1 = 8$, $n_2 = 40$ 으로 두고 $d_i = 2\alpha_{n,i}$ 라 쓴다.

**(1)** $\alpha_{n,1} \ne \alpha_{n,2}$ 일 때 $\rho$ 의 사후분포가 $\dfrac{\beta_{n,1}}{\beta_{n,2}}\cdot\dfrac{d_2}{d_1}\,F(d_2, d_1)$ 임을 보이고, 보기 2 의 꼴이 그 특수한 경우임을 확인하시오.

**(2)** 보기 2 의 "사후중앙값 $=$ 표본분산의 비"가 깨짐을 보이시오. 제프리스 사전분포($\alpha_0 = -1/2$, $\beta_0 = 0$, $\kappa_0 = 0$)에서는 보정인수가 정확히 $F(n_2-1,\, n_1-1)$ 의 중앙값임을 보이고 그 값을 구하시오.

**(3)** 모의로 (1)의 닫힌 꼴을 확인하고, 제프리스 신용구간이 빈도주의 신뢰구간과 정확히 같은지 확인하시오.

**(4)** 두 집단의 번호를 맞바꾸면 보정인수가 어떻게 되는가. $n_1 = n_2$ 일 때만 문제가 사라지는 까닭을 밝히시오.

</div>

??? success "풀이"

    **(1) 자유도가 다르면 비례상수가 하나 더 붙는다.** 보기 1 의 추축량에서 $\sigma_i^2 = 2\beta_{n,i}/W_i$, $W_i \sim \chi^2_{d_i}$ 이고 두 집단이 독립이므로

    $$
    \rho = \frac{\sigma_1^2}{\sigma_2^2}
    = \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{W_2}{W_1}
    = \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{d_2}{d_1}\cdot\frac{W_2/d_2}{W_1/d_1}
    \;\sim\; \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{d_2}{d_1}\,F(d_2,\,d_1)
    $$

    이다. **분자 자유도가 $d_2$ 이고 분모 자유도가 $d_1$ 로 뒤바뀌어 들어가는 것**에 주의한다. $d_1 = d_2$ 면 $d_2/d_1 = 1$ 이라 보기 2 의 $\dfrac{\beta_{n,1}}{\beta_{n,2}}F(2\alpha_n, 2\alpha_n)$ 로 돌아간다. 보기 2 의 `assert` 는 이 비례상수를 잊지 않으려는 안전장치였다.

    **(2) 제프리스에서는 보정인수가 F 의 중앙값 그 자체다.** $\alpha_0 = -1/2$, $\beta_0 = 0$, $\kappa_0 = 0$ 이면 $d_i = n_i - 1$ 이고 $\beta_{n,i} = S_i/2 = (n_i-1)s_i^2/2$ 이므로

    $$
    \frac{\beta_{n,1}}{\beta_{n,2}}\cdot\frac{d_2}{d_1}
    = \frac{(n_1-1)s_1^2}{(n_2-1)s_2^2}\cdot\frac{n_2-1}{n_1-1}
    = \frac{s_1^2}{s_2^2}
    $$

    로 깔끔하게 약분되어

    $$
    \rho \mid \mathbf x \;\sim\; \frac{s_1^2}{s_2^2}\,F(n_2-1,\, n_1-1),
    \qquad
    \operatorname{median}(\rho) = \frac{s_1^2}{s_2^2}\cdot\operatorname{med}F(n_2-1,\,n_1-1)
    $$

    가 된다. $n_1 = 8$, $n_2 = 40$ 이면 $\operatorname{med}F(39,7) = 1.084166$ 이므로 **사후중앙값이 표본분산의 비를 $8.4\%$ 과대평가한다.** 보기 2 에서 둘이 같았던 것은 $d_1 = d_2$ 라서 $F(d,d)$ 가 역수에 닫혀 있고 그 중앙값이 정확히 1 이었기 때문이다.

    쪽의 기본 사전분포($\alpha_0 = 10^{-2}$)에서는 $d_i \approx n_i$ 라서 보정인수가

    $$
    \frac{n_1-1}{n_2-1}\cdot\frac{n_2}{n_1}\cdot\operatorname{med}F(n_2,\,n_1)
    \approx \frac{7}{39}\cdot\frac{40.02}{8.02}\cdot 1.070881 = 0.9591
    $$

    로 이번에는 $4\%$ **과소**평가가 된다. $\beta_{n,1}/\beta_{n,2}$ 의 $(n_i-1)$ 과 $d_2/d_1$ 의 $n_i$ 가 더는 약분되지 않기 때문이다.

    **(3)–(4) 수로.**

    ```python
    import numpy as np
    from scipy.stats import invgamma, f as f_dist

    def post(x, m0=0.0, k0=1e-6, a0=1e-2, b0=1e-2):
        n = x.size
        xbar = x.mean()
        S = ((x - xbar) ** 2).sum()
        return a0 + n / 2.0, b0 + 0.5 * (S + (k0 * n / (k0 + n)) * (xbar - m0) ** 2)

    rng = np.random.default_rng(0)
    n1, n2 = 8, 40
    y1 = rng.normal(10.0, 2.0, n1)
    y2 = rng.normal(10.0, 2.0, n2)          # 참 분산이 같다 -> rho = 1
    v1, v2 = y1.var(ddof=1), y2.var(ddof=1)
    Fobs = v1 / v2
    print(f"n1 = {n1}, n2 = {n2}   s1^2 = {v1:.6f}, s2^2 = {v2:.6f}, s1^2/s2^2 = {Fobs:.6f}")

    for tag, kw in (("기본 (a0=1e-2)", dict(a0=1e-2)),
                    ("제프리스 (a0=-1/2, b0=0, k0=0)", dict(a0=-0.5, b0=0.0, k0=0.0))):
        a1, b1 = post(y1, **kw)
        a2, b2 = post(y2, **kw)
        d1, d2 = 2 * a1, 2 * a2
        F = f_dist(d2, d1)
        sc = (b1 / b2) * (d2 / d1)
        g = np.random.default_rng(7)
        r = invgamma(a=a1, scale=b1).rvs(1_000_000, random_state=g) / \
            invgamma(a=a2, scale=b2).rvs(1_000_000, random_state=g)
        print(f"\n[{tag}]  d1 = {d1:.2f}, d2 = {d2:.2f}")
        print(f"  rho ~ {sc:.6f} x F({d2:.2f}, {d1:.2f}),   F 의 중앙값 = {F.ppf(0.5):.6f}")
        print("            닫힌 꼴      모의 100만개")
        for nm, c, m in (("중앙값  ", sc * F.ppf(0.5), np.median(r)),
                         ("q(0.025)", sc * F.ppf(0.025), np.percentile(r, 2.5)),
                         ("q(0.975)", sc * F.ppf(0.975), np.percentile(r, 97.5))):
            print(f"  {nm}  {c:10.6f}  {m:14.6f}")
        print(f"  사후중앙값 / (s1^2/s2^2) = {sc * F.ppf(0.5) / Fobs:.6f}")

    # 제프리스 신용구간과 빈도주의 신뢰구간
    fl = Fobs / f_dist(n1 - 1, n2 - 1).ppf(0.975)
    fh = Fobs / f_dist(n1 - 1, n2 - 1).ppf(0.025)
    jl = Fobs * f_dist(n2 - 1, n1 - 1).ppf(0.025)
    jh = Fobs * f_dist(n2 - 1, n1 - 1).ppf(0.975)
    print(f"\n빈도주의 신뢰구간 = ({fl:.6f}, {fh:.6f})")
    print(f"제프리스 신용구간 = ({jl:.6f}, {jh:.6f})   차 = {abs(jl - fl):.2e}, {abs(jh - fh):.2e}")

    # 번호를 맞바꾸면 보정인수가 뒤집힌다
    print(f"\n보정인수: 1->2 일 때 med F({n2-1},{n1-1}) = {f_dist(n2-1, n1-1).ppf(0.5):.6f}, "
          f"2->1 일 때 med F({n1-1},{n2-1}) = {f_dist(n1-1, n2-1).ppf(0.5):.6f}")
    print(f"n1 = n2 = 8 이면 med F(7,7) = {f_dist(7, 7).ppf(0.5):.6f}")
    ```

    출력:

    ```text
    n1 = 8, n2 = 40   s1^2 = 1.416158, s2^2 = 3.475742, s1^2/s2^2 = 0.407440

    [기본 (a0=1e-2)]  d1 = 8.02, d2 = 40.02
      rho ~ 0.365608 x F(40.02, 8.02),   F 의 중앙값 = 1.070881
                닫힌 꼴      모의 100만개
      중앙값      0.391523        0.391514
      q(0.025)    0.144681        0.144940
      q(0.975)    1.400980        1.395410
      사후중앙값 / (s1^2/s2^2) = 0.960933

    [제프리스 (a0=-1/2, b0=0, k0=0)]  d1 = 7.00, d2 = 39.00
      rho ~ 0.407440 x F(39.00, 7.00),   F 의 중앙값 = 1.084166
                닫힌 꼴      모의 100만개
      중앙값      0.441733        0.441729
      q(0.025)    0.154744        0.155064
      q(0.975)    1.757301        1.748523
      사후중앙값 / (s1^2/s2^2) = 1.084166

    빈도주의 신뢰구간 = (0.154744, 1.757301)
    제프리스 신용구간 = (0.154744, 1.757301)   차 = 0.00e+00, 4.44e-16

    보정인수: 1->2 일 때 med F(39,7) = 1.084166, 2->1 일 때 med F(7,39) = 0.922368
    n1 = n2 = 8 이면 med F(7,7) = 1.000000
    ```

    **닫힌 꼴이 맞는다.** 중앙값은 두 사전분포 모두 소수 다섯째 자리까지 모의값과 같고($0.391523$ 대 $0.391514$, $0.441733$ 대 $0.441729$), 아래 꼬리 분위수도 넷째 자리에서 맞는다. 위 꼬리만 $1.757301$ 대 $1.748523$ 으로 상대오차 $0.5\%$ 가 나는데, 분모 자유도가 7 인 $F$ 의 오른쪽 꼬리는 밀도가 $0.04$ 밖에 안 되어(표본분위수의 표준오차가 $0.004$) 몬테카를로가 가장 느리게 수렴하는 자리다.

    **"사후중앙값 = 표본분산의 비"가 양쪽으로 깨진다.** 기본 사전분포에서 $0.960933$, 제프리스에서 $1.084166$ 이다. (2)에서 손으로 계산한 $0.9591$ 과 $1.084166$ 이 그대로 나왔다(앞의 작은 차는 $\beta_0 = 0.01$ 몫이다). **집단 크기가 다르면 사후중앙값은 자료가 주는 비를 그대로 돌려주지 않는다.**

    **제프리스 신용구간은 빈도주의 신뢰구간과 정확히 같다.** $(0.154744,\, 1.757301)$ 이 차 $0$ 과 $4\times10^{-16}$ 으로 일치한다. $1/F_{0.975}(n_1-1, n_2-1) = F_{0.025}(n_2-1, n_1-1)$ 이라는 $F$ 분포의 역수 성질 때문이고, **집단 크기가 달라도 성립한다.** 보기 2 가 본 "$7.5\%$ 좁다"는 어긋남은 크기가 달라서가 아니라 $\alpha_0$ 가 $-1/2$ 가 아니어서 생긴 것이다.

    **(4) 번호를 맞바꾸면 보정인수가 뒤집힌다.** 같은 공식을 $1/\rho$ 에 쓰면 보정인수가 $\operatorname{med}F(n_1-1, n_2-1) = \operatorname{med}F(7,39) = 0.922368 = 1/1.084166$ 이 된다. 즉 **어느 집단을 "1 번"이라고 부르느냐에 따라 사후중앙값이 비를 $8.4\%$ 과대평가하거나 $7.8\%$ 과소평가한다.** 중앙값은 단조변환에 옮겨 가므로 모순은 없지만, 라벨이 보고값을 바꾼다는 것은 그 요약값이 이 상황에서 좋지 않다는 뜻이다. $n_1 = n_2$ 일 때만 문제가 사라지는데, 그때는 $d_1 = d_2$ 라서 $F(d,d)$ 가 역수에 닫혀 있고 중앙값이 정확히 $\operatorname{med}F(7,7) = 1$ 이기 때문이다. 자기역수성이 라벨 대칭성을 보장한다.

    **쪽의 두 결론은 크기가 같다는 우연에 기대고 있었다.** 집단 크기가 다르면 보고할 것은 한 숫자가 아니라 신용구간 전체이고, 사전분포는 $\alpha_0 = -1/2$ 로 두어야 빈도주의와 어긋나지 않는다.

---

## 정리하며

베이즈 분산 비교의 **구현**이다.

- **정규–역감마 켤레 모형을 쓴다.** 사후분포가 닫힌 형태로 나오므로 MCMC 없이 직접 표본을 뽑을 수 있다.
- **몬테카를로로 사후표본을 만든다.** 각 집단의 $\sigma_i^2$ 을 사후분포에서 뽑고 비를 계산하면 분산비의 사후표본이 된다.
- **결과가 세 가지 형태로 요약된다.** 사후평균, 신용구간, 그리고 $P(\sigma_1^2/\sigma_2^2>1\mid\text{자료})$ 같은 직접 확률.
- **신용구간이 $1$ 을 포함하는지가 판정에 해당하지만**, 베이즈에서는 그 확률 자체를 보고하는 편이 자연스럽다.
- **사전분포를 바꿔 가며 민감도를 본다.** 결론이 크게 흔들리면 자료가 부족하다는 신호다.

다음 절 **응용**으로 15장을 마무리한다.
