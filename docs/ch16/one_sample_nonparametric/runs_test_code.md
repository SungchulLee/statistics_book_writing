# 런 검정

## 개요

**Wald--Wolfowitz 런 검정**은 이진 관측값의 수열이 무작위로 생성되었는지 판정하는 비모수 절차이다.
*런*(같은 기호가 연달아 이어지는 최대 부분수열)의 개수를 세어, 독립이라는 귀무가설 아래에서
기대되는 분포와 비교한다. 런이 너무 적으면 뭉침을, 너무 많으면 체계적 교대를 시사한다.

## 런 통계량

길이 $N$인 이진 수열에 한 종류의 기호가 $N_+$개, 다른 종류가 $N_- = N - N_+$개 있다고 하자.
**런**은 같은 기호가 연달아 이어지는 최대 덩어리이다.

수열이 독립이고 동일하게 분포한다는 귀무가설 $H_0$ 아래에서 런의 개수 $R$은
다음 평균과 표준편차를 갖는다.

$$
\mu_R = \frac{2\,N_+\,N_-}{N} + 1, \qquad
\sigma_R = \sqrt{\frac{(\mu_R - 1)(\mu_R - 2)}{N - 1}}.
$$

$N$이 중간 이상이면 표준화된 통계량

$$
Z = \frac{R - \mu_R}{\sigma_R}
$$

이 근사적으로 표준정규를 따르므로 양측 $p$값은

$$
p = 2\,\Phi\!\bigl(-|Z|\bigr)
$$

이며 $\Phi$는 표준정규 누적분포함수이다.

## 런을 효율적으로 세기

$\pm 1$로 부호화된 수열 $x_1, x_2, \dots, x_N$에서 곱 $x_i\,x_{i+1}$은 연속된 두 원소가
같으면 $+1$, 다르면 $-1$이다. 부호가 바뀔 때마다 새 런이 시작되므로

$$
R = \frac{N_+ + N_- + 1 - \displaystyle\sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}.
$$

## 구현

다음 파이썬 함수는 정규근사를 이용한 런 검정을 구현한다.

```python
import numpy as np
import scipy.stats as stats


def runs_test(data):
    """Wald-Wolfowitz runs test for a +1/-1 sequence."""
    data = np.asarray(data)
    N = data.shape[0]
    N_plus = (data == 1).sum()
    N_minus = N - N_plus

    mu = 2 * N_plus * N_minus / N + 1
    sigma = np.sqrt((mu - 1) * (mu - 2) / (N - 1))
    R = (N_plus + N_minus + 1 - np.sum(data[1:] * data[:-1])) / 2

    statistic = (R - mu) / sigma
    p_value = 2 * stats.norm.cdf(-abs(statistic))
    return statistic, p_value
```

### 예제: 뭉친 수열

뭉침이 심한 수열은 런이 매우 적다.

```python
data = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
z, p = runs_test(data * 2 - 1)
# R = 2, mu = 8.76, sigma = 1.81
# z = -3.7335, p = 0.0002 → 무작위성 기각
```

### 예제: 지나치게 교대하는 수열

빈번한 교대 역시 무작위성으로부터의 이탈이다.

```python
data = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
z, p = runs_test(data * 2 - 1)
# R = 14, mu = 9.47, sigma = 1.99
# z = +2.2775, p = 0.0228 → 무작위성 기각
```

!!! warning "'잘 섞여 보임'은 무작위성의 증거가 아니다"
    두 번째 수열은 언뜻 잘 섞인 듯 보이지만 $17$개 원소에서 런이 $14$개로, 기댓값 $9.47$을
    크게 넘는다. 최장 런의 길이가 $2$에 불과한데, 무작위 수열이라면 길이 $17$에서 평균
    $4.4$ 정도의 런이 나타나야 한다.

    사람이 무작위를 흉내 낼 때 같은 값이 이어지는 것을 피하려다 정확히 이런 패턴을 만든다.
    런 검정은 뭉침과 과잉 교대를 **양쪽 모두** 잡아낸다.

## 해석

| 결과 | 의미 |
|---|---|
| $Z \ll 0$ (런이 적음) | 관측값이 **뭉쳐** 있다 --- 연속된 값이 같은 경향이 있다. |
| $Z \gg 0$ (런이 많음) | 관측값이 우연보다 자주 **교대**한다. |
| $\lvert Z \rvert$가 작음 | 선택한 유의수준에서 무작위성에 반하는 증거가 없다. |

이 검정은 기본적으로 양측이다. 런이 비정상적으로 적은 경우와 많은 경우 모두 독립성에
반하는 증거이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 동전을 20번 던져 수열
HHHHTTTTHHHHTTTTTTHH를 얻었다. 각 H를 $+1$, 각 T를 $-1$로 부호화하고 런의 개수
$R$을 센 뒤 $\mu_R$과 $\sigma_R$을 손으로 계산하라.

</div>

??? success "풀이"

    수열은 HHHH TTTT HHHH TTTTTT HH이므로 $R = 5$개의 런이 있다.
    $N = 20$, $N_+ = 10$, $N_- = 10$이다.

    $$
    \mu_R = \frac{2 \cdot 10 \cdot 10}{20} + 1 = 11.
    $$

    $$
    \sigma_R = \sqrt{\frac{(11 - 1)(11 - 2)}{20 - 1}}
             = \sqrt{\frac{90}{19}}
             \approx 2.176.
    $$

    따라서 $Z = (5 - 11)/2.176 \approx -2.757$이고 $p = 0.0058$로 뭉침의 강한 증거가 된다.
    $\square$

<div class="drillbox" markdown>

**연습문제 2.** 각 $x_i \in \{-1, +1\}$일 때
$R = \dfrac{N_+ + N_- + 1 - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}$임을 증명하라.

</div>

??? success "풀이"

    $i = 1,\dots,N-1$에 대해 지시함수 $d_i = \mathbf{1}[x_i \neq x_{i+1}]$을 정의한다.
    부호가 바뀔 때마다 새 런이 시작되므로 $R = 1 + \sum_{i=1}^{N-1} d_i$이다.

    $x_i \in \{-1,+1\}$이므로 $x_i\,x_{i+1} = 1 - 2\,d_i$이고 따라서
    $d_i = (1 - x_i\,x_{i+1})/2$이다. 합하면

    $$
    R = 1 + \sum_{i=1}^{N-1} \frac{1 - x_i\,x_{i+1}}{2}
      = 1 + \frac{(N-1) - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}
      = \frac{N + 1 - \sum_{i=1}^{N-1} x_i\,x_{i+1}}{2}.
    $$

    $N = N_+ + N_-$이므로 결과가 따라 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 위 "지나치게 교대하는 수열" 예제의 자료로 런 검정 통계량과 $p$값을
파이썬에서 계산하라. $\alpha = 0.05$에서 귀무가설이 기각되는지 확인하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    data = np.array([1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0])
    seq = data * 2 - 1  # convert to +1/-1

    N = len(seq)
    N_plus = (seq == 1).sum()   # 9
    N_minus = N - N_plus        # 8

    mu = 2 * N_plus * N_minus / N + 1               # 9.4706
    sigma = np.sqrt((mu - 1) * (mu - 2) / (N - 1))  # 1.9887
    R = (N - np.sum(seq[1:] * seq[:-1]) + 1) / 2    # 14

    z = (R - mu) / sigma
    p = 2 * stats.norm.cdf(-abs(z))
    print(f"R = {R}, Z = {z:.4f}, p = {p:.4f}")
    # R = 14.0, Z = 2.2775, p = 0.0228  →  H0 기각
    ```

    출력:

    ```
    R = 14.0, Z = 2.2775, p = 0.0228
    ```

    $p = 0.0228 < 0.05$이므로 $\alpha = 0.05$에서 **귀무가설을 기각한다**. 런이 $14$개로
    기댓값 $9.47$보다 유의하게 많아 과잉 교대의 증거가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 수열이 이진이 아니면 왜 런 검정이 부적절한지 설명하라. 연속 수열을
검정에 적합한 이진 수열로 바꾸는 흔한 방법 하나를 기술하라.

</div>

??? success "풀이"

    $\mu_R$과 $\sigma_R$의 유도는 기호가 정확히 두 종류이고 개수가 $N_+$, $N_-$로
    고정되어 있다고 가정한다. 범주가 셋 이상이면 조합론이 달라지고 정규근사가
    더 이상 성립하지 않는다.

    표준적인 해법은 연속 수열을 표본**중앙값을 기준으로 이분**하는 것이다. 중앙값보다
    큰 값을 $+1$, 작은 값을 $-1$로 부호화한다(중앙값과 같은 값은 관례에 따라 버리거나
    한쪽에 배정한다). 이렇게 얻은 이진 수열에 Wald--Wolfowitz 절차를 적용한다.
    $\square$

<div class="drillbox" markdown>

**연습문제 5.** 수열의 모든 순열에 걸친 세기 논증으로
$\operatorname{E}[R] = \mu_R = \dfrac{2\,N_+\,N_-}{N} + 1$임을 보여라.

</div>

??? success "풀이"

    $H_0$ 아래에서 수열의 $\binom{N}{N_+}$가지 배열이 모두 동등하게 가능하다.
    $d_i = \mathbf{1}[x_i \neq x_{i+1}]$일 때 $R = 1 + \sum_{i=1}^{N-1} d_i$이므로
    기댓값의 선형성에 의해

    $$
    \operatorname{E}[R] = 1 + \sum_{i=1}^{N-1} P(x_i \neq x_{i+1}).
    $$

    인접한 한 쌍의 위치 $(i, i+1)$에 대해, 균등 무작위 배열에서 그 두 자리에 놓이는
    기호의 조합은 $N$개 중 순서 있게 2개를 뽑는 $N(N-1)$가지가 모두 동등하게 가능하다.
    이 중 앞이 $+$이고 뒤가 $-$인 경우가 $N_+ N_-$가지, 그 반대가 $N_+ N_-$가지이므로

    $$
    P(x_i \neq x_{i+1}) = \frac{2\,N_+\,N_-}{N(N-1)}.
    $$

    이 확률은 $i$에 의존하지 않는다. 인접 쌍이 $N-1$개이므로

    $$
    \operatorname{E}[R] = 1 + (N-1) \cdot \frac{2\,N_+\,N_-}{N(N-1)}
                        = 1 + \frac{2\,N_+\,N_-}{N}
                        = \mu_R.
    $$

    $\square$

    !!! note "분산은 왜 이 방법으로 안 되는가"
        기댓값은 $d_i$들이 서로 종속이어도 선형성 덕에 쉽게 나온다. 그러나 분산은
        $\text{Cov}(d_i, d_{i+1}) \ne 0$을 다루어야 한다. 인접한 두 지시함수가
        관측값 $x_{i+1}$을 공유하기 때문이다. 이 공분산을 모두 더하면 본문의
        $\sigma_R^2 = (\mu_R-1)(\mu_R-2)/(N-1)$이 나온다.

<div class="drillbox" markdown>

**연습문제 6.** 런 검정의 실제 제1종 오류율을 확인하라. $N$이 작을 때 정규근사가
얼마나 정확한가?

</div>

??? success "풀이"

    $R$의 정확 귀무분포는 조합식으로 바로 쓸 수 있다. $N_+$개의 $+$가 $k$개의 런으로,
    $N_-$개의 $-$가 $k$개의 런으로 나뉘는 경우를 세면

    $$
    P(R = 2k) = \frac{2\binom{N_+-1}{k-1}\binom{N_--1}{k-1}}{\binom{N}{N_+}}, \qquad
    P(R = 2k+1) = \frac{\binom{N_+-1}{k-1}\binom{N_--1}{k} + \binom{N_+-1}{k}\binom{N_--1}{k-1}}{\binom{N}{N_+}}
    $$

    이다. 이 분포로 $|Z| > 1.96$이라는 기각역의 실제 확률을 계산한다.

    ```python
    import numpy as np
    from math import comb
    import scipy.stats as stats

    def runs_pmf(N_plus, N_minus):
        N = N_plus + N_minus
        tot = comb(N, N_plus)
        pmf = {}
        for k in range(1, min(N_plus, N_minus) + 1):
            pmf[2 * k] = 2 * comb(N_plus - 1, k - 1) * comb(N_minus - 1, k - 1) / tot
            pmf[2 * k + 1] = (comb(N_plus - 1, k - 1) * comb(N_minus - 1, k)
                              + comb(N_plus - 1, k) * comb(N_minus - 1, k - 1)) / tot
        return pmf

    crit = stats.norm.isf(0.025)
    for n in (5, 10, 15, 20, 50):
        N = 2 * n
        mu = 2 * n * n / N + 1
        sd = np.sqrt((mu - 1) * (mu - 2) / (N - 1))
        pmf = runs_pmf(n, n)
        size = sum(p for R, p in pmf.items() if abs((R - mu) / sd) > crit)
        print(N, round(size, 4))
    ```

    출력:

    ```
    10 0.0794
    20 0.037
    30 0.0398
    40 0.0363
    100 0.0555
    ```

    | $N$ ($N_+ = N_- = N/2$) | 실제 크기 |
    |---:|---:|
    | 10 | **0.0794** |
    | 20 | 0.0370 |
    | 30 | 0.0398 |
    | 40 | 0.0363 |
    | 100 | 0.0555 |

    $N = 10$에서 실제 크기가 $0.079$로 명목값의 **1.6배**이다. 근사가 비보수적이다.
    $R$이 가질 수 있는 값이 $2$부터 $10$까지 9가지뿐이라 이산성이 극심하기 때문이다.

    더 흥미로운 것은 크기가 $N$에 따라 단조 수렴하지 않고 $0.036$과 $0.079$ 사이를
    **요동한다**는 점이다. $N = 100$에서도 $0.0555$로 명목값을 넘는다. 계단함수의
    계단 위치가 임계값 $\pm 1.96\sigma_R$과 어떻게 맞물리느냐에 따라 기각역이
    한 계단 더 포함되거나 덜 포함되기 때문이다.

    **권고:** 각 기호가 최소 10개씩 있어도 근사 오차가 $\pm 0.01$ 수준으로 남는다.
    정확한 판정이 필요하면 위 `runs_pmf`로 정확 $p$값을 직접 계산한다. 관측된 $R$에
    대해 $p = \sum_{r : |r - \mu_R| \ge |R - \mu_R|} P(R = r)$을 쓰면 된다.
