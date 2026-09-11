# Lilliefors 검정

## 개요

Lilliefors 검정은 귀무가설의 모수(평균과 분산)를 미리 지정하지 않고 자료에서 추정하는 흔한 상황을 위해 고안된 Kolmogorov-Smirnov 검정의 변형이다. 모수적 붓스트랩으로 KS 통계량의 귀무분포를 보정함으로써, 복합(composite) 정규성을 검정할 때 타당한 $p$값을 제공한다.

## 표준 KS 검정의 문제

일표본 KS 통계량은

$$
D_n = \sup_x |F_n(x) - F_0(x)|.
$$

$F_0 = \mathcal{N}(\cdot\,; \mu, \sigma)$이고 $\mu$, $\sigma$를 $\hat{\mu} = \bar{X}$, $\hat{\sigma} = S$로 추정하면, 적합된 CDF $\hat{F}_0$이 미리 지정된 $F_0$보다 $F_n$에 체계적으로 더 가까워진다. 이는 $D_n$을 줄이고 $p$값을 부풀려 표준 KS 검정을 보수적으로(과소기각하게) 만든다.

## 모수적 붓스트랩 알고리즘

Lilliefors 검정은 모수 추정을 포함한 상태에서 $D_n$의 귀무분포를 모의생성하여 이를 바로잡는다.

1. **관측 통계량을 계산한다.** 자료에서 $\hat{\mu}, \hat{\sigma}$를 적합하고

    $$
    D_{\text{obs}} = \sup_x |F_n(x) - \mathcal{N}(x;\, \hat{\mu}, \hat{\sigma})|
    $$

    를 계산한다.

2. **붓스트랩 반복.** $b = 1, \ldots, B$에 대해
    - $X_1^*, \ldots, X_n^* \overset{\text{iid}}{\sim} \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$를 모의생성한다.
    - $\hat{\mu}^* = \bar{X}^*$, $\hat{\sigma}^* = S^*$를 **다시 추정한다**.
    - $D_b^* = \sup_x |F_n^*(x) - \mathcal{N}(x;\, \hat{\mu}^*, \hat{\sigma}^*)|$를 계산한다.

3. **$p$값을 계산한다.**

    $$
    p \approx \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}(D_b^* \geq D_{\text{obs}}).
    $$

### 코드

```python
import numpy as np
from scipy import stats

def ks_stat_fitted_normal(x):
    x = np.asarray(x, dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    D, _ = stats.kstest(x, 'norm', args=(mu, sd))
    return float(D), float(mu), float(sd)

def lilliefors_normal_bootstrap(x, B=2000, seed=0):
    x = np.asarray(x, dtype=float)
    n = x.size
    D_obs, mu, sd = ks_stat_fitted_normal(x)

    rng = np.random.default_rng(seed)
    D_star = np.empty(B)
    for b in range(B):
        xb = rng.normal(mu, sd, size=n)
        D_star[b], _, _ = ks_stat_fitted_normal(xb)

    p_boot = float(np.mean(D_star >= D_obs))
    return D_obs, p_boot, mu, sd

# Example: skewed data that should be rejected
rng = np.random.default_rng(1)
x = rng.lognormal(0.0, 0.6, size=300)

D, p, mu, sd = lilliefors_normal_bootstrap(x, B=1500, seed=7)
print(f"Fitted Normal: mu = {mu:.4f}, sd = {sd:.4f}")
print(f"Lilliefors KS D = {D:.4f}, bootstrap p = {p:.4f}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

출력:

```text
Fitted Normal: mu = 1.1025, sd = 0.7348
Lilliefors KS D = 0.1391, bootstrap p = 0.0000
=> Reject normality at alpha = 0.05.
```

붓스트랩 $p$값이 정확히 0이라는 것은 1,500번의 붓스트랩 표본 중 $D_{\text{obs}} = 0.1391$ 이상을 낸 것이 하나도 없었다는 뜻이다. 붓스트랩 귀무분포의 95백분위수는 $0.0509$로, 관측값이 그 세 배에 가깝다. 정확한 $p$값을 알 수는 없고 $p < 1/1500$이라고만 말할 수 있다.

## 붓스트랩이 작동하는 이유

붓스트랩은 원자료에 적용한 것과 *같은* 절차로 표본을 생성한다. 정규분포에서 뽑고, 모수를 다시 추정하고, KS 통계량을 계산한다. 이렇게 하면 편향의 원천(모수 추정이 $D$를 줄이는 것)이 기준분포에도 그대로 복제되어 올바르게 보정된 임계값이 만들어진다.

핵심은 2단계에서 모수를 **다시 추정**하는 것이다. 이 단계를 빠뜨리고 원래의 $\hat{\mu}, \hat{\sigma}$를 고정한 채 쓰면 표준 KS 검정으로 되돌아가 버린다.

## 해석

대수정규 예에서 Lilliefors 검정은 정규성을 기각하여($p \approx 0$) 오른쪽 치우침을 올바르게 식별한다.

다만 이 예에서는 이탈이 워낙 커서 모수를 추정한 순진한 KS 검정도 $p = 1.6 \times 10^{-5}$로 기각한다. 곧 **Lilliefors 보정이 결론을 바꾸는 것은 경계선 근처에서다.** 신호가 압도적이면 어느 쪽을 쓰든 기각한다. 보정이 결정적으로 중요한 이유는 결론이 뒤집히는 사례가 있어서라기보다, 순진한 검정의 제1종 오류율이 명목값과 전혀 다르기 때문이다(연습문제 2 참조).

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 표준정규 관측값 $n = 200$개를 생성하라. 순진한 KS 검정($\hat{\mu}, \hat{\sigma}$ 추정)과 Lilliefors 붓스트랩 검정을 모두 수행하라. 두 $p$값을 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    mu_hat, sd_hat = x.mean(), x.std(ddof=1)
    D_naive, p_naive = stats.kstest(x, 'norm', args=(mu_hat, sd_hat))

    # Bootstrap Lilliefors
    B = 2000
    D_obs = D_naive
    D_star = np.empty(B)
    for b in range(B):
        xb = rng.normal(mu_hat, sd_hat, size=200)
        mu_b, sd_b = xb.mean(), xb.std(ddof=1)
        D_star[b], _ = stats.kstest(xb, 'norm', args=(mu_b, sd_b))
    p_boot = np.mean(D_star >= D_obs)

    print(f"D = {D_obs:.4f}")
    print(f"Naive KS p-value:       {p_naive:.4f}")
    print(f"Lilliefors bootstrap p: {p_boot:.4f}")
    ```

    출력:

    ```text
    D = 0.0306
    Naive KS p-value:       0.9892
    Lilliefors bootstrap p: 0.9345
    ```

    같은 $D = 0.0306$에 대해 순진한 $p$값이 $0.989$로 붓스트랩 $p$값 $0.935$보다 크다. 표준 KS 임계값이 모수 추정을 고려하지 않기 때문이다.

    이 표본에서는 둘 다 0.05를 크게 넘어 결론이 같다. 편향의 방향을 보여줄 뿐 실질적 차이는 없다. 편향의 실제 규모는 $p$값이 아니라 검정의 크기를 보아야 드러난다(연습문제 2). $\square$

<div class="drillbox" markdown>

**연습문제 2.** $n = 100$에 대해 $\alpha = 0.05$에서 Lilliefors 붓스트랩 검정($B = 500$)의 경험적 크기를 추정하는 몬테카를로 실험을 5,000회 반복으로 수행하라. 순진한 KS 검정과 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha, B = 100, 5000, 0.05, 500

    rej_naive, rej_boot = 0, 0
    for _ in range(reps):
        x = rng.normal(0, 1, size=n)
        mu, sd = x.mean(), x.std(ddof=1)
        D, p_naive = stats.kstest(x, 'norm', args=(mu, sd))
        if p_naive < alpha:
            rej_naive += 1

        D_star = np.empty(B)
        for b in range(B):
            xb = rng.normal(mu, sd, size=n)
            mu_b, sd_b = xb.mean(), xb.std(ddof=1)
            D_star[b], _ = stats.kstest(xb, 'norm', args=(mu_b, sd_b))
        if np.mean(D_star >= D) < alpha:
            rej_boot += 1

    print(f"Naive KS size:   {rej_naive / reps:.4f}")
    print(f"Lilliefors size: {rej_boot / reps:.4f}")
    ```

    출력:

    ```text
    Naive KS size:   0.0000
    Lilliefors size: 0.0484
    ```

    (이 실험은 5,000 × 500 = 250만 번의 KS 계산을 수행하므로 시간이 꽤 걸린다.)

    결과가 놀랍다. 순진한 KS 검정의 경험적 크기가 **정확히 0이다**. 5,000번의 반복 중 단 한 번도 기각하지 않았다. 명목값 0.05라면 약 250번을 기각했어야 한다.

    이는 "0.01~0.02 정도로 보수적"인 수준을 훨씬 넘어선다. 모수를 추정한 KS 검정은 사실상 **절대 기각하지 않는다**. 이유는 임계값의 차이가 크기 때문이다. $n = 100$에서 표준 KS의 5% 임계값은 $1.358/\sqrt{100} = 0.136$인 반면, 모수를 추정했을 때 $D$의 참 95백분위수는 $0.0892$에 불과하다. 임계값이 참값의 1.5배이니 기각이 일어날 리 없다.

    Lilliefors 붓스트랩의 크기 $0.0484$는 명목값 $0.05$와 잘 맞는다(몬테카를로 표준오차 $0.0031$). 보정이 올바르게 작동함을 확인해 준다.

    실무적 결론: **모수를 추정한 뒤 표준 KS 검정을 쓰는 것은 검정을 하지 않는 것과 거의 같다.** 반드시 Lilliefors 보정을 쓰거나 Shapiro-Wilk / Anderson-Darling 같은 다른 검정을 쓰라. $\square$

<div class="drillbox" markdown>

**연습문제 3.** Lilliefors $p$값의 해상도가 $1/B$인 이유를 설명하라. $p = 0.04$와 $p = 0.06$을 안정적으로 구별하려면 $B$가 얼마나 커야 하는가?

</div>

??? success "풀이"

    붓스트랩 $p$값은 $\hat{p} = \frac{1}{B}\sum_{b=1}^B \mathbf{1}(D_b^* \geq D_{\text{obs}})$이므로 $\{0, 1/B, 2/B, \ldots, 1\}$의 값만 가질 수 있다. 지시함수의 합이 정수이기 때문이다.

    귀무가설 아래에서 표준오차는 $\sqrt{\hat{p}(1-\hat{p})/B}$이다. $p = 0.04$와 $p = 0.06$을 구별하려면 표준오차가 $0.01$보다 훨씬 작아야 한다. $p \approx 0.05$에서 $\text{SE} = \sqrt{0.05 \times 0.95/B}$이므로 $\text{SE} < 0.005$를 요구하면

    $$
    \frac{0.0475}{B} < 0.000025 \quad \Longrightarrow \quad B > 1900.
    $$

    실무에서는 5% 문턱 근처의 신뢰할 만한 추론을 위해 $B \geq 2000$이 합리적인 최솟값이다. 문턱에서 더 멀리 떨어진 결정만 필요하다면 $B = 500$으로도 충분하다(연습문제 2에서 그렇게 썼다). $\square$

<div class="drillbox" markdown>

**연습문제 4.** 정규성 대신 지수성을 검정하도록 Lilliefors 붓스트랩을 수정하라. 알고리즘에 필요한 변경을 개략적으로 서술하라.

</div>

??? success "풀이"

    변경 사항은 다음과 같다.

    1. **모수 추정:** $\hat{\mu}, \hat{\sigma}$를 지수 비율의 최대가능도추정량 $\hat{\lambda} = 1/\bar{X}$로 바꾼다.
    2. **관측 통계량:** $D_{\text{obs}} = \sup_x |F_n(x) - (1 - e^{-\hat{\lambda} x})|$를 계산한다.
    3. **붓스트랩 반복:** $X_b^* \sim \text{Exp}(\hat{\lambda})$를 모의생성하고 $\hat{\lambda}_b^* = 1/\bar{X}_b^*$로 다시 추정한 뒤, 다시 적합된 지수 CDF에 대해 $D_b^*$를 계산한다.
    4. **$p$값:** 이전과 같이 $\hat{p} = \frac{1}{B}\sum \mathbf{1}(D_b^* \geq D_{\text{obs}})$.

    핵심 원리는 동일하다. 붓스트랩이 모수 추정 단계를 복제하므로 $D$의 기준분포가 올바르게 보정된다.

    한 가지 편리한 점이 있다. 정규의 경우와 마찬가지로 지수의 경우에도 $D$의 귀무분포가 **모수에 의존하지 않는다**. 지수족이 척도족이고 $\hat{\lambda}$가 척도동변 추정량이기 때문이다. 따라서 $\hat{\lambda} = 1$로 두고 임계값 표를 한 번만 만들어 두면 모든 자료에 재사용할 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 고정된 자료에 대해 $B \to \infty$일 때 붓스트랩 $p$값 $\hat{p}_B$가 참 $p$값으로 수렴함을 증명하라.

</div>

??? success "풀이"

    자료가 고정되어 있으면 $D_{\text{obs}}$는 상수이다. 각 붓스트랩 추출은 $D_b^*$를 만들어 내고, $\mathbf{1}(D_b^* \geq D_{\text{obs}})$는 성공확률

    $$
    p^* = P^*(D^* \geq D_{\text{obs}})
    $$

    를 갖는 Bernoulli 확률변수이다. 여기서 $P^*$는 붓스트랩 분포($\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$에서의 표집)를 나타낸다. 붓스트랩 $p$값은

    $$
    \hat{p}_B = \frac{1}{B}\sum_{b=1}^{B} \mathbf{1}(D_b^* \geq D_{\text{obs}}).
    $$

    자료에 조건부로 $D_b^*$들이 i.i.d.이므로 강대수의 법칙에 의해 $B \to \infty$일 때 $\hat{p}_B \xrightarrow{\text{a.s.}} p^*$이다. 여기서 $p^*$는 적합된 귀무분포 아래의 정확한 Lilliefors $p$값이다.

    중심극한정리에 의해 수렴속도는 $O(1/\sqrt{B})$이고 $\hat{p}_B$의 표준오차는 $\sqrt{p^*(1-p^*)/B}$이다.

    한 가지 주의할 점을 덧붙인다. 이 수렴은 **자료를 고정한 조건부 수렴**이다. $B \to \infty$로 보내도 $\hat{\mu}, \hat{\sigma}$가 참값이 아니라는 데서 오는 오차는 사라지지 않는다. 그러나 정규 위치-척도족에서 $D$의 귀무분포가 $\mu, \sigma$에 의존하지 않으므로(척도동변성), 이 오차는 실제로 0이다. 그래서 Lilliefors 검정이 유한표본에서도 정확한 크기를 갖는다. $\square$
