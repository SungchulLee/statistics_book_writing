# 왜도 검정

## 개요

D'Agostino 왜도 검정은 자료의 왜도가 정규분포에서 기대되는 값인 0과 유의하게 다른지 평가한다. 표본왜도를 귀무가설 아래에서 근사적으로 표준정규를 따르는 $Z$ 통계량으로 변환하므로, 비대칭을 겨냥한 표적 검정이 된다. 이 페이지는 핵심 공식을 유도하고, Python에서 검정을 시연하며, 실무적 사용을 논의한다.

## 표본왜도

표본 $X_1, \ldots, X_n$에 대한 Fisher-Pearson 왜도 계수(편향 보정판)는

$$
g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{X_i - \bar{X}}{S}\right)^3,
$$

여기서 $\bar{X}$는 표본평균, $S$는 (Bessel 보정을 한) 표본표준편차이다. 정규성 아래에서 $\mathbb{E}[g_1] = 0$이고

$$
\text{Var}(g_1) \approx \frac{6(n-2)}{(n+1)(n+3)}.
$$

## D'Agostino 변환

$H_0$ 아래에서도 $g_1$의 분포가 정확히 정규가 아니므로, D'Agostino와 Pearson(1973)은 $g_1$을 $\mathcal{N}(0,1)$에 훨씬 가까운 통계량 $Z_1$으로 보내는 비선형 변환을 제안했다. 변환은 먼저

$$
Y = g_1 \sqrt{\frac{(n+1)(n+3)}{6(n-2)}},
$$

를 계산한 뒤 고차 누적률에 대한 추가 조정을 거친다. 최종 $Z_1$은 $H_0: \text{왜도} = 0$ 아래에서 근사적으로 표준정규이다.

## 가설

$$
H_0: \gamma_1 = 0 \quad (\text{모집단 왜도가 0이다}), \qquad H_1: \gamma_1 \neq 0.
$$

양측 $p$값은 $p = 2\,\Phi(-|Z_1|)$이며 $\Phi$는 표준정규 CDF이다.

<div class="codebox" markdown>

### 예제 1. 로그정규 자료의 왜도 검정 { .eg }

```python
import numpy as np
from scipy import stats

# 로그정규는 오른쪽으로 길게 늘어진 대표적인 분포다.
rng = np.random.default_rng(0)
x = rng.lognormal(mean=0.0, sigma=0.6, size=300)

# 표본왜도를 그대로 쓰지 않고 Z 로 바꾼다. 왜도의 표집분포가 정규에서
# 멀어, 값 자체로는 얼마나 큰 것인지 판단할 수 없기 때문이다.
# 이 변환은 n>=8 부터 쓸 수 있다.
g1 = stats.skew(x, bias=False)
z, p = stats.skewtest(x)

print(f"Sample size n = {x.size}")
print(f"Sample skewness (Fisher's g1) = {g1:.4f}")
print(f"D'Agostino skewness test: Z = {z:.4f}, p-value = {p:.4g}")
if p < 0.05:
    print("=> Evidence of non-zero skewness (departing from normality).")
else:
    print("=> No strong evidence of non-zero skewness.")
```

출력:

```text
Sample size n = 300
Sample skewness (Fisher's g1) = 2.2452
D'Agostino skewness test: Z = 10.4038, p-value = 2.382e-25
=> Evidence of non-zero skewness (departing from normality).
```

</div>

## 해석

예제의 대수정규 자료에서 $g_1 = 2.245$로 크게 양수이고(오른쪽 치우침), $p$값이 $2.4 \times 10^{-25}$로 왜도가 0이라는 가설을 압도적으로 기각한다. 이론적 왜도는 $2.261$(연습문제 3 참조)이므로 표본값이 잘 맞는다.

왜도 검정은 비대칭이 의심되지만 다른 이탈은 꼭 의심되지 않을 때 특히 유용하다. 왜도 검정과 첨도 검정을 결합한 D'Agostino $K^2$ 옴니버스 검정의 한 구성요소이기도 하다.

**최소 표본크기.** SciPy의 `skewtest`는 $n \geq 8$을 요구한다. 근사는 $n$이 클수록 좋아지며, $n < 20$이면 $p$값을 조심스럽게 해석해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 표준정규분포에서 관측값 $n = 300$개를 생성하라. $g_1$과 왜도 검정 $p$값을 계산하라. $\alpha = 0.05$에서 기각할 것으로 기대하는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=300)

    g1 = stats.skew(x, bias=False)
    z, p = stats.skewtest(x)

    print(f"g1 = {g1:.4f}")
    print(f"Z = {z:.4f}, p = {p:.4g}")
    ```

    출력:

    ```text
    g1 = 0.2933
    Z = 2.0725, p = 0.03822
    ```

    !!! warning "이 표본은 제1종 오류를 보여준다"
        자료를 정확히 표준정규에서 생성했는데도 $p = 0.038 < 0.05$이므로 **검정이 기각한다**. 이것이 바로 제1종 오류이다.

    기대와 실제를 구분해야 한다. 자료가 참으로 정규이므로 우리는 기각하지 *않기를* 기대하고, 실제로 표본의 95%에서는 기각하지 않는다. 그러나 나머지 5%에서는 기각한다. 이 표본이 그 5%에 들어갔을 뿐이다.

    $g_1 = 0.2933$은 절댓값으로는 작아 보이지만, $n = 300$에서 $g_1$의 표준오차가

    $$
    \sqrt{\frac{6(n-2)}{(n+1)(n+3)}} = \sqrt{\frac{6 \times 298}{301 \times 303}} = 0.1400
    $$

    에 불과하므로 $0.2933$은 2 표준오차가 넘는다. 표본이 커지면 작은 왜도도 통계적으로 유의해진다는 점을 잘 보여준다.

    실무적 교훈: **$p$값 하나로 판정하지 말라.** 여기서 실질적으로 중요한 것은 $g_1 = 0.29$가 어떤 응용에서든 무시할 만한 크기라는 사실이다. 효과 크기를 함께 보고해야 하는 이유이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $\text{Uniform}(0, 1)$ 분포에서 뽑은 관측값 $n = 200$개에 대해 표본왜도를 계산하고 왜도 검정을 수행하라. 균등분포는 대칭이지만 정규가 아니다. 검정이 기각하는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=200)

    g1 = stats.skew(x, bias=False)
    z, p = stats.skewtest(x)
    print(f"g1 = {g1:.4f}, Z = {z:.4f}, p = {p:.4g}")
    ```

    출력:

    ```text
    g1 = -0.1735, Z = -1.0217, p = 0.3069
    ```

    균등분포의 모집단 왜도는 $\gamma_1 = 0$이므로 왜도 검정은 기각하지 *않는다*($p = 0.307$).

    이는 한계를 보여준다. 왜도 검정은 우연히 대칭인 비정규 분포를 놓칠 수 있다. $\text{Uniform}(0,1)$은 대칭이지만 초과첨도가 $-1.2$인 저첨분포이다. 첨도 검정이나 옴니버스 검정이었다면 이 이탈을 탐지했을 것이다.

    확인해 보면 같은 자료에서 `stats.kurtosistest(x)`는 $Z = -9.78$, $p = 1.4 \times 10^{-22}$로 압도적으로 기각한다. 이탈의 유형에 맞는 검정을 골라야 한다는 교훈이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $\text{Lognormal}(\mu, \sigma^2)$ 분포의 모집단 왜도가 $(e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}$임을 보여라. $\sigma = 0.6$에 대한 값을 계산하라.

</div>

??? success "풀이"

    대수정규의 적률 성질에서 $\mathbb{E}[X^k] = e^{k\mu + k^2\sigma^2/2}$이다. 처음 세 중심적률을 계산하면 왜도가 다음으로 정리된다.

    $$
    \gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1}.
    $$

    왜도는 척도불변이므로 $\mu$에 의존하지 않고 $\sigma$에만 의존한다는 점에 주목하라.

    $\sigma = 0.6$이면 $e^{0.36} = 1.4333$이므로 $e^{\sigma^2} - 1 = 0.4333$이고 $\sqrt{0.4333} = 0.6583$이다. 따라서

    $$
    \gamma_1 = (1.4333 + 2)(0.6583) = 3.4333 \times 0.6583 \approx 2.261.
    $$

    이 큰 양의 왜도 때문에 대수정규 표본에서 왜도 검정이 단호하게 기각한다. 실제로 본문의 시연에서 표본값 $g_1 = 2.245$가 이 이론값에 가깝게 나왔다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $\text{Lognormal}(0, 0.4)$에서 뽑은 관측값 $n = 100$개에 대해 $\alpha = 0.05$에서 왜도 검정의 검정력을 추정하는 모의실험을 5,000회 반복으로 수행하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 100, 5000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.lognormal(0, 0.4, size=n)
        _, p = stats.skewtest(x)
        if p < alpha:
            rejections += 1

    power = rejections / reps
    print(f"Empirical power: {power:.4f}")
    ```

    출력:

    ```text
    Empirical power: 0.9770
    ```

    검정력이 $0.977$로 매우 높다. $\sigma = 0.4$인 대수정규의 왜도는

    $$
    \gamma_1 = (e^{0.16} + 2)\sqrt{e^{0.16} - 1} = 3.1735 \times \sqrt{0.1735} = 1.322
    $$

    로 중간 정도인데, $n = 100$에서 $g_1$의 표준오차가 약 $\sqrt{6/100} = 0.245$이므로 신호 대 잡음비가 $1.322/0.245 \approx 5.4$에 이른다. 이만한 비율이면 사실상 언제나 탐지된다.

    (몬테카를로 오차는 $\sqrt{0.977 \times 0.023 / 5000} = 0.0021$이다.) $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 정확한 공식 $\text{Var}(g_1) = 6(n-2)/[(n+1)(n+3)]$에서 출발하여 큰 $n$에 대해 $\text{Var}(g_1) \approx 6/n$임을 증명하라.

</div>

??? success "풀이"

    다음에서 출발한다.

    $$
    \text{Var}(g_1) = \frac{6(n-2)}{(n+1)(n+3)}.
    $$

    분자와 분모를 각각 $n$과 $n^2$으로 나누면

    $$
    \text{Var}(g_1) = \frac{6\,(1 - 2/n)}{n\,(1 + 1/n)(1 + 3/n)}.
    $$

    $n \to \infty$일 때 $(1 - 2/n) \to 1$, $(1 + 1/n) \to 1$, $(1 + 3/n) \to 1$이므로

    $$
    \text{Var}(g_1) \to \frac{6}{n}.
    $$

    따라서 $g_1$의 표준오차는 $\sqrt{6/n}$ 정도, 곧 $1/\sqrt{n}$ 차수이며, 표본이 클수록 0이 아닌 왜도를 탐지하기 쉬워진다.

    수렴 속도를 보자. $n = 100$에서 정확한 값은 $\sqrt{6 \times 98/(101 \times 103)} = 0.2377$이고 근사값은 $\sqrt{6/100} = 0.2449$로 3% 차이이다. $n = 1000$에서는 각각 $0.07723$과 $0.07746$으로 0.3% 차이이다. $\square$

---

## 정리하며

왜도 검정의 **구성과 구현**을 살폈다.

- **피셔–피어슨 편향 보정 왜도**를 쓴다. 표본 왜도는 편향되어 있으므로 $\sqrt{n(n-1)}/(n-2)$ 를 곱해 보정한다.
- **$Z$ 로 변환하는 과정이 정교하다.** 표본 왜도의 분포가 정규와 거리가 멀어, 존슨 변환류의 보정을 거쳐야 근사적으로 표준정규가 된다. **단순히 표준오차로 나누는 것이 아니다.**
- **$n\ge8$ 이 요건이다.** 그보다 작으면 `scipy` 가 오류를 낸다.
- **치우침만 겨냥한다.** 대칭이면서 꼬리가 두꺼운 분포($t$ 분포 등)는 이 검정으로 잡히지 않는다.
- **방향을 알 수 있다는 것이 장점이다.** $Z$ 의 부호가 어느 쪽으로 치우쳤는지 말해 주며, 변환의 방향을 정하는 데 쓴다.

다음 절 **첨도 검정**으로 넘어간다.
