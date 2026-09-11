# Bayes 분산 검정

빈도주의 분산 검정은 표집분포와 $p$값에 기댄다. Bayes 접근은 근본적으로 다른 관점을 취한다. 모분산 $\sigma^2$을 사전분포를 갖는 확률변수로 취급하고, 관측 자료로 이 분포를 갱신한 뒤, 그 결과인 사후분포를 통해 추론한다. 이 틀은 $\sigma^2$에 대한 불확실성을 수량화하고 집단 간 분산을 비교하는 자연스러운 방법을 제공한다.

## 분산에 대한 켤레 사전분포

자료가 알려진 평균 $\mu$를 갖는 정규분포를 따를 때 분산 $\sigma^2$의 켤레 사전분포는 **역감마**분포이다.

$$
\sigma^2 \sim \text{Inv-Gamma}(\alpha_0, \beta_0)
$$

밀도는

$$
p(\sigma^2) = \frac{\beta_0^{\alpha_0}}{\Gamma(\alpha_0)} (\sigma^2)^{-\alpha_0 - 1} \exp\!\left(-\frac{\beta_0}{\sigma^2}\right), \quad \sigma^2 > 0
$$

초모수 $\alpha_0 > 0$과 $\beta_0 > 0$이 사전 믿음을 부호화한다.

- $\alpha_0$은 사전분포의 강도를 조절한다($\alpha_0$이 클수록 사전 믿음이 강하다)
- $\beta_0 / \alpha_0$은 $\sigma^2$의 사전 기댓값을 근사한다($\alpha_0$이 클 때)
- 사전평균은 $\alpha_0 > 1$일 때 $E[\sigma^2] = \beta_0 / (\alpha_0 - 1)$이다

흔히 쓰는 약정보 선택은 $\alpha_0 = \beta_0 = 0.01$(또는 $\alpha_0 = \beta_0 = 0.001$)이며, 자료가 지배하도록 하는 확산 사전분포를 만든다.

!!! note "$\alpha_0$을 "가상 관측값 수"로 읽기"
    사후 갱신식 $\alpha_n = \alpha_0 + (n-1)/2$을 보면 $\alpha_0$이 자유도의 절반 단위로 더해진다. 곧 **사전분포는 약 $2\alpha_0$개의 관측값에 해당하는 정보를 담는다.**

    $\alpha_0 = 0.01$이면 관측값 0.02개어치이므로 사실상 무정보이다. $\alpha_0 = 50$이면 관측값 100개어치이므로 $n = 100$인 자료와 대등한 영향력을 갖는다(연습문제 3 참조).

## 사후분포

$\mu$가 알려진 관측값 $X_1, \ldots, X_n \sim N(\mu, \sigma^2)$이 주어지면 $\sigma^2$의 사후분포도 역감마이다.

$$
\sigma^2 \mid X_1, \ldots, X_n \sim \text{Inv-Gamma}(\alpha_n, \beta_n)
$$

여기서

$$
\alpha_n = \alpha_0 + \frac{n}{2}, \qquad \beta_n = \beta_0 + \frac{1}{2}\sum_{i=1}^{n}(X_i - \mu)^2
$$

$\mu$를 모르는 (보통의) 경우에는 $\mu$를 $\bar{X}$로 바꾸고 자유도 $n - 1$을 쓴다.

$$
\alpha_n = \alpha_0 + \frac{n - 1}{2}, \qquad \beta_n = \beta_0 + \frac{(n-1)S^2}{2}
$$

!!! warning "두 공식을 섞지 말라"
    $\mu$를 아는 경우는 $\alpha_n = \alpha_0 + n/2$, 모르는 경우는 $\alpha_n = \alpha_0 + (n-1)/2$이다. $\beta_n$의 제곱합도 각각 $\sum(X_i-\mu)^2$과 $\sum(X_i-\bar X)^2 = (n-1)S^2$으로 다르다.

    실무에서는 거의 항상 $\mu$를 모르므로 두 번째 공식을 쓴다. $n$과 $n-1$을 혼동하면 $n$이 작을 때 결과가 눈에 띄게 달라진다.

이 켤레성 덕분에 사후분포가 닫힌 형태를 가지므로 계산이 간단하다.

## Bayes 신용구간

$\sigma^2$의 $100(1-\alpha)\%$ 신용구간은 다음을 만족하는 구간 $[L, U]$이다.

$$
P(L \le \sigma^2 \le U \mid \text{자료}) = 1 - \alpha
$$

역감마 사후분포에서는 $\text{Inv-Gamma}(\alpha_n, \beta_n)$의 분위수로 신용구간을 얻는다.

!!! note "신용구간과 신뢰구간"
    Bayes 신용구간은 직접적인 확률 해석을 갖는다. 자료와 사전분포가 주어졌을 때 $\sigma^2$이 그 구간 안에 있을 확률이 $(1-\alpha)$이다. 빈도주의 신뢰구간은 그런 주장을 하지 않는다. 반복된 표본의 $(1-\alpha)$에서 절차가 $\sigma^2$을 포함한다는 뜻이다. 확산 사전분포에서는 두 구간이 수치적으로 비슷하다.

    연습문제 1에서 확산 사전분포를 쓰면 두 구간이 소수 셋째 자리까지 **일치**함을 확인한다.

## Bayes 가설검정

$H_0\colon \sigma^2 = \sigma_0^2$을 $H_1\colon \sigma^2 \neq \sigma_0^2$에 대해 검정하려면 Bayes 접근은 $H_0$의 사후확률을 계산하거나 **Bayes 인자**를 쓴다.

$$
\text{BF}_{01} = \frac{p(\text{자료} \mid H_0)}{p(\text{자료} \mid H_1)}
$$

여기서 $p(\text{자료} \mid H_0)$은 $H_0$ 아래의 주변가능도($\sigma^2$이 $\sigma_0^2$으로 고정)이고 $p(\text{자료} \mid H_1)$은 $\sigma^2$의 사전분포에 대해 가능도를 적분한 것이다.

두 분산을 비교하는 더 간단한 접근은 분산비의 사후분포를 쓰는 것이다. $\sigma_1^2$과 $\sigma_2^2$의 독립인 사후분포가 주어지면

$$
R = \frac{\sigma_1^2}{\sigma_2^2}
$$

를 계산한다. $R$의 $95\%$ 신용구간이 1을 포함하면 자료가 등분산과 일관된다.

## 두 분산 비교

역감마 사후분포를 갖는 독립인 두 집단에 대해

$$
\sigma_1^2 \mid \text{자료}_1 \sim \text{Inv-Gamma}(\alpha_{n_1}, \beta_{n_1})
$$

$$
\sigma_2^2 \mid \text{자료}_2 \sim \text{Inv-Gamma}(\alpha_{n_2}, \beta_{n_2})
$$

비 $R = \sigma_1^2 / \sigma_2^2$은 단순한 닫힌 형태의 분포를 갖지 않지만 몬테카를로 표집으로 추정할 수 있다. 첫 사후분포에서 $\sigma_1^{2(b)}$을, 둘째에서 $\sigma_2^{2(b)}$을 뽑고 $b = 1, \ldots, B$에 대해 $R^{(b)} = \sigma_1^{2(b)} / \sigma_2^{2(b)}$을 계산한다.

$\sigma_1^2 > \sigma_2^2$일 사후확률은

$$
P(\sigma_1^2 > \sigma_2^2 \mid \text{자료}) \approx \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}(R^{(b)} > 1)
$$

## 예제

관측값 $n = 20$개의 표본분산이 $S^2 = 15.3$이다. 약정보 사전분포 $\sigma^2 \sim \text{Inv-Gamma}(0.01, 0.01)$을 쓰면

$$
\alpha_n = 0.01 + \frac{19}{2} = 9.51
$$

$$
\beta_n = 0.01 + \frac{19 \times 15.3}{2} = 145.36
$$

사후분포는 $\sigma^2 \mid \text{자료} \sim \text{Inv-Gamma}(9.51, 145.36)$이다.

사후평균은 $\beta_n / (\alpha_n - 1) = 145.36 / 8.51 = 17.08$로 표본분산 15.3보다 크다. 이 차이는 사전분포의 영향이 아니라(사전분포는 거의 무정보이다) **역감마분포가 오른쪽으로 치우쳐 있어서 평균이 최빈값보다 크기** 때문이다. 최빈값은 $\beta_n/(\alpha_n+1) = 145.36/10.51 = 13.83$이고 중앙값은 $15.84$이다.

## Python 구현

```python
import numpy as np
from scipy import stats

# Data
n = 20
s_squared = 15.3

# Weakly informative prior
alpha_0, beta_0 = 0.01, 0.01

# Posterior parameters (mu unknown)
alpha_n = alpha_0 + (n - 1) / 2
beta_n = beta_0 + (n - 1) * s_squared / 2

# Posterior summaries
print(f"Posterior: Inv-Gamma({alpha_n}, {beta_n:.2f})")
print(f"Posterior mean:   {beta_n / (alpha_n - 1):.4f}")
print(f"Posterior mode:   {beta_n / (alpha_n + 1):.4f}")
print(f"Posterior median: {stats.invgamma.median(a=alpha_n, scale=beta_n):.4f}")

# 95% credible interval using inverse-gamma quantiles
ci_lower = stats.invgamma.ppf(0.025, a=alpha_n, scale=beta_n)
ci_upper = stats.invgamma.ppf(0.975, a=alpha_n, scale=beta_n)
print(f"95% credible interval: ({ci_lower:.4f}, {ci_upper:.4f})")

# Frequentist interval for comparison
fl = (n - 1) * s_squared / stats.chi2.ppf(0.975, n - 1)
fu = (n - 1) * s_squared / stats.chi2.ppf(0.025, n - 1)
print(f"95% confidence interval: ({fl:.4f}, {fu:.4f})")

# Monte Carlo comparison of two variances
rng = np.random.default_rng(42)
alpha_n1, beta_n1 = 9.51, 145.36   # Group 1 posterior
alpha_n2, beta_n2 = 12.01, 120.10  # Group 2 posterior

sigma1_samples = stats.invgamma.rvs(a=alpha_n1, scale=beta_n1,
                                    size=10000, random_state=rng)
sigma2_samples = stats.invgamma.rvs(a=alpha_n2, scale=beta_n2,
                                    size=10000, random_state=rng)

ratio = sigma1_samples / sigma2_samples
print(f"P(sigma1^2 > sigma2^2 | data) = {np.mean(ratio > 1):.3f}")
print(f"95% credible interval for ratio: "
      f"({np.percentile(ratio, 2.5):.3f}, {np.percentile(ratio, 97.5):.3f})")
```

출력:

```text
Posterior: Inv-Gamma(9.51, 145.36)
Posterior mean:   17.0811
Posterior mode:   13.8306
Posterior median: 15.8365
95% credible interval: (8.8422, 32.5915)
95% confidence interval: (8.8487, 32.6390)
P(sigma1^2 > sigma2^2 | data) = 0.835
95% credible interval for ratio: (0.644, 3.674)
```

두 구간이 사실상 일치한다($8.842$ 대 $8.849$, $32.59$ 대 $32.64$). 확산 사전분포에서 Bayes 신용구간과 빈도주의 신뢰구간이 수렴한다는 사실을 보여준다.

분산비에 대해서는 $P(\sigma_1^2 > \sigma_2^2 \mid \text{자료}) = 0.835$이고 95% 신용구간 $(0.644, 3.674)$가 1을 포함한다. 곧 집단 1의 분산이 더 클 가능성이 높지만($83.5\%$) 결론을 내릴 만큼 확실하지는 않다.

## 장점과 한계

**장점:**

- $\sigma^2$에 대한 직접적인 확률 진술(신용구간이 직관적으로 해석된다)
- 분야 지식에서 오는 사전정보를 반영할 수 있다
- 점근근사에 의존하지 않는다
- 사후표집으로 여러 분산을 비교하는 자연스러운 틀

**한계:**

- 사전분포를 지정해야 하며 그 선택이 논쟁적일 수 있다
- 켤레 분석이 정규성을 가정한다. 비정규 자료에는 더 복잡한 모형이 필요하다
- 모형이 복잡해지면 계산 비용이 늘어난다(MCMC로 다룰 수 있기는 하다)

**정규성 가정에 대한 유의.** 마지막 한계가 특히 중요하다. 켤레 역감마 분석은 정규 가능도에 기반하므로 **비정규성에 대한 취약성이 Bartlett 검정과 본질적으로 같다.** "Bayes 방법이므로 가정에서 자유롭다"는 오해를 경계해야 한다. 자료가 두꺼운 꼬리를 갖는다면 $t$ 가능도를 쓰는 모형으로 확장해야 하며, 그러면 켤레성이 깨져 MCMC가 필요하다.


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
무정보 사전분포 $\sigma^2 \sim \text{Inv-Gamma}(0.001, 0.001)$과 $\sum(x_i - \bar{x})^2 = 180$인 $n = 20$ 표본에 대해 사후분포와 $\sigma^2$의 95% 신용구간을 구하라.

</div>

??? success "풀이"
    $\mu$를 모르므로 $\alpha_n = \alpha_0 + (n-1)/2$를 쓴다.

    $$
    \alpha_n = 0.001 + \frac{19}{2} = 9.501, \qquad \beta_n = 0.001 + \frac{180}{2} = 90.001.
    $$

    사후분포는 $\text{Inv-Gamma}(9.501, 90.001)$이다.

    !!! warning "$n/2$가 아니라 $(n-1)/2$이다"
        평균을 자료에서 추정했으므로 자유도가 $n-1 = 19$이고 $\alpha_0$에 더해지는 값은 $9.5$이다. $n/2 = 10$을 쓰면 $\alpha_n = 10.001$이 되어 아래의 모든 결과가 달라진다(신용구간이 $(5.27, 18.77)$이 되어 빈도주의 구간과 어긋난다).

    사후평균은 $\beta_n/(\alpha_n - 1) = 90.001/8.501 = 10.587$이다. 표본분산 $s^2 = 180/19 = 9.474$보다 큰데, 이는 역감마분포의 오른쪽 치우침 때문이다.

    ```python
    from scipy import stats

    alpha_n, beta_n = 0.001 + 19 / 2, 0.001 + 180 / 2
    print(f"Posterior: Inv-Gamma({alpha_n}, {beta_n})")
    print(f"mean = {beta_n / (alpha_n - 1):.4f}, s^2 = {180 / 19:.4f}")
    print(f"95% credible: ({stats.invgamma.ppf(0.025, a=alpha_n, scale=beta_n):.4f}, "
          f"{stats.invgamma.ppf(0.975, a=alpha_n, scale=beta_n):.4f})")
    print(f"95% confidence: ({180 / stats.chi2.ppf(0.975, 19):.4f}, "
          f"{180 / stats.chi2.ppf(0.025, 19):.4f})")
    ```

    출력:

    ```text
    Posterior: Inv-Gamma(9.501, 90.001)
    mean = 10.5871, s^2 = 9.4737
    95% credible: (5.4787, 20.2071)
    95% confidence: (5.4791, 20.2099)
    ```

    **95% 신용구간은 $(5.479, 20.207)$이다.**

    빈도주의 신뢰구간 $(5.479, 20.210)$과 소수 셋째 자리까지 일치한다. 우연이 아니다. $\alpha_0, \beta_0 \to 0$일 때 역감마 사후분포가 $2\beta_n/\sigma^2 \sim \chi^2_{2\alpha_n}$을 만족하는데, $2\alpha_n \to n-1 = 19$이고 $2\beta_n \to (n-1)s^2 = 180$이므로 정확히 빈도주의 추축량 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$과 같아진다.

    **곧 무정보 사전분포를 쓴 Bayes 분석은 빈도주의 결과와 수치적으로 동일하다.** 차이는 오직 해석에 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
분산에 대한 Bayes 신용구간과 빈도주의 신뢰구간을 비교하라. 핵심적인 철학적 차이는 무엇인가?

</div>

??? success "풀이"
    **빈도주의 신뢰구간:** $P(L < \sigma^2 < U) = 0.95$는 실험을 여러 번 반복하면 계산된 구간의 95%가 참 $\sigma^2$을 포함한다는 뜻이다. 모수는 고정이고 구간이 확률변수이다.

    **Bayes 신용구간:** $P(L < \sigma^2 < U \mid \text{자료}) = 0.95$는 관측 자료와 사전분포가 주어졌을 때 $\sigma^2$이 이 구간에 있을 사후확률이 95%라는 뜻이다. (자료가 주어지면) 구간이 고정이고 모수가 확률변수로 취급된다.

    무정보 사전분포와 큰 $n$에서 두 구간은 수치적으로 비슷하며, 연습문제 1에서 보았듯 아예 일치할 수도 있다.

    **핵심 차이는 "확률"이라는 말의 대상이다.** 빈도주의에서 확률은 절차의 장기적 성질이고, Bayes에서는 모수에 대한 믿음의 정도이다.

    **왜 이 구분이 실무적으로 중요한가.** 흔한 오해는 빈도주의 95% CI를 두고 "$\sigma^2$이 이 구간에 있을 확률이 95%"라고 말하는 것이다. 빈도주의 틀에서 $\sigma^2$은 상수이므로 이 진술은 무의미하다(확률이 0 또는 1이다). Bayes 신용구간에서만 그 진술이 정당하다.

    구간이 수치적으로 같더라도 이 해석의 차이는 남는다. 연습문제 1처럼 두 구간이 완전히 일치하는 경우, 그 구간에 어느 해석을 붙일지는 사전분포를 인정하느냐에 달려 있다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
작은 표본에서 사전분포의 선택이 Bayes 분산 검정에 어떤 영향을 주는가? 강한 정보 사전분포로 예시하라.

</div>

??? success "풀이"
    강한 정보 사전분포 $\sigma^2 \sim \text{Inv-Gamma}(50, 500)$을 생각하자. 사전평균은 $500/49 = 10.20$이고 10 근처에 촘촘하다.

    ```python
    from scipy import stats

    alpha_0, beta_0 = 50, 500
    print(f"Prior mean: {beta_0 / (alpha_0 - 1):.3f}")
    for n, s2 in [(5, 25), (20, 25), (100, 25), (1000, 25)]:
        an = alpha_0 + (n - 1) / 2
        bn = beta_0 + (n - 1) * s2 / 2
        lo = stats.invgamma.ppf(0.025, a=an, scale=bn)
        hi = stats.invgamma.ppf(0.975, a=an, scale=bn)
        print(f"n = {n:>4}: posterior mean = {bn / (an - 1):6.3f}, "
              f"95% CrI = ({lo:6.3f}, {hi:6.3f})")
    ```

    출력:

    ```text
    Prior mean: 10.204
    n =    5: posterior mean = 10.784, 95% CrI = ( 8.202, 14.162)
    n =   20: posterior mean = 12.607, 95% CrI = ( 9.763, 16.262)
    n =  100: posterior mean = 17.640, 95% CrI = (14.482, 21.474)
    n = 1000: posterior mean = 23.678, 95% CrI = (21.777, 25.743)
    ```

    자료는 모두 $s^2 = 25$를 시사하는데도 사후평균이 다음과 같이 이동한다.

    | $n$ | 사후평균 | 사전평균(10.2)과 자료(25) 사이의 위치 |
    |---|---|---|
    | 5 | 10.78 | 사실상 사전분포 |
    | 20 | 12.61 | 사전분포 쪽 16% |
    | 100 | 17.64 | 중간 (50%) |
    | 1000 | 23.68 | 자료 쪽 91% |

    !!! warning "$n = 100$으로도 사전분포를 이기지 못한다"
        "$n = 100$이면 자료가 사전분포를 압도한다"는 서술은 이 사전분포에 대해 **틀리다**. 사후평균 $17.64$는 사전평균 $10.2$와 자료 $25$의 거의 정확한 중간이다.

        이유는 본문의 "가상 관측값" 해석으로 설명된다. $\alpha_0 = 50$은 자유도 $2\alpha_0 = 100$, 곧 **관측값 약 101개어치**의 정보를 담는다. $n = 100$인 자료의 자유도는 99이므로 사전분포와 자료가 대등하다.

        자료가 사전분포를 압도하려면 $n - 1 \gg 2\alpha_0 = 100$, 곧 $n \gg 100$이어야 한다. 표에서 $n = 1000$이 되어야 자료 쪽으로 91% 이동한다.

    **실무 지침.** 작은 표본에서는 사전분포 민감도 분석이 필수적이다. 여러 사전분포(정보, 약정보, 무정보) 아래에서 사후분포를 계산하여 결론이 바뀌는지 확인하라. 그리고 사전분포를 고를 때는 $\alpha_0$이 몇 개의 가상 관측값에 해당하는지 늘 확인하여, 자신이 실제로 가진 사전 지식의 양에 맞추어야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
Bayes 인자로 분산에 대한 두 가설 $H_0: \sigma^2 = \sigma_0^2$과 $H_1: \sigma^2 \neq \sigma_0^2$을 비교하는 방법을 기술하라.

</div>

??? success "풀이"
    Bayes 인자 $BF_{01}$은 주변가능도의 비이다.

    $$
    BF_{01} = \frac{P(\text{자료} \mid H_0)}{P(\text{자료} \mid H_1)}
    $$

    $H_0$ 아래에서 $\sigma^2$이 $\sigma_0^2$으로 고정되므로 주변가능도는 $\sigma_0^2$에서 평가한 가능도이다. $H_1$ 아래에서는 $\sigma^2$의 사전분포에 대해 가능도를 적분한다.

    $$
    P(\text{자료} \mid H_1) = \int_0^\infty L(\sigma^2)\, p(\sigma^2)\, d\sigma^2.
    $$

    정규 가능도와 역감마 사전분포에서는 이 적분이 닫힌 형태로 계산된다.

    $BF_{01} > 1$이면 $H_0$에 유리하고 $BF_{01} < 1$이면 $H_1$에 유리하다. 흔한 해석 기준은 $BF > 10$이면 강한 증거, $BF > 100$이면 결정적 증거이다. $p$값과 달리 Bayes 인자는 귀무가설에 **반대하는** 증거뿐 아니라 **찬성하는** 증거도 제공할 수 있다.

    !!! warning "Bayes 인자는 사전분포에 민감하다"
        $H_1$ 아래의 주변가능도가 사전분포 $p(\sigma^2)$에 대한 적분이므로 **사전분포의 선택이 $BF$에 직접 영향을 준다.** 특히 사전분포를 넓게(확산되게) 만들수록 $H_1$의 주변가능도가 작아져 $BF_{01}$이 커진다. 곧 무정보 사전분포를 쓰려는 시도가 자동으로 $H_0$에 유리하게 작용한다.

        이를 **Lindley 역설** 또는 Jeffreys-Lindley 역설이라 한다. 극단적으로 $\beta_0 \to 0$, $\alpha_0 \to 0$인 부적절 사전분포에서는 $BF$가 정의되지 않는다.

        신용구간이 사전분포에 둔감한 것과 대조적이다. 연습문제 1에서 무정보 사전분포의 신용구간이 빈도주의 구간과 일치했지만, 같은 사전분포로 Bayes 인자를 계산하면 무의미한 값이 나온다.

        **실무 지침.** Bayes 인자를 쓸 때는 반드시 (1) 적절한(proper) 사전분포를 쓰고, (2) 그 사전분포를 정당화하며, (3) 사전분포를 바꿔 가며 $BF$의 안정성을 확인해야 한다. 분산 비교에서는 Bayes 인자보다 **분산비의 사후분포와 신용구간**을 보고하는 편이 사전분포 의존성이 훨씬 낮아 안전하다. $\square$

---

## 정리하며

베이즈는 $\sigma^2$ 을 **확률변수로** 다룬다.

- **역감마가 켤레 사전분포다.** 정규 가능도와 짝을 이루어 사후분포도 역감마가 되며, 6장에서 본 켤레족의 성질 그대로다.
- **$p$ 값 대신 사후분포 전체를 얻는다.** 신용구간과 $P(\sigma_1^2>\sigma_2^2\mid\text{자료})$ 같은 직접적인 확률 진술이 가능하다. **빈도주의로는 할 수 없는 말이다.**
- **분산비의 사후분포를 그릴 수 있다.** 두 집단을 비교할 때 비의 분포를 직접 얻어 $1$ 이 어디쯤 놓이는지 본다.
- **사전분포가 소표본에서 결과를 좌우한다.** 무정보 사전분포를 쓰더라도 그 선택이 결론에 미치는 영향을 민감도 분석으로 확인해야 한다.
- **여전히 정규 가능도를 가정한다.** 베이즈라고 분포 가정이 없는 것이 아니며, 이 점을 오해하기 쉽다.

다음 절 **가능도비 검정**으로 넘어간다.
