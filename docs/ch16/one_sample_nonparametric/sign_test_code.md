# 부호검정

## 개요

**부호검정**은 대응자료에 대한 가장 단순한 비모수 검정 중 하나이다. 쌍별 차이의
*부호*만 살펴서 두 관련 측정값의 중앙값 차이가 0인지 판정한다. 크기 정보를 완전히
버리므로 가정이 매우 적다 --- 차이들이 독립이고 연속분포에서 나왔다는 것뿐이다.
그 덕에 매우 로버스트하지만 Wilcoxon 부호순위검정 같은 대안보다 검정력이 낮다.

## 가설과 검정통계량

$n$개의 대응 관측값 $(X_i, Y_i)$에서 차이 $D_i = X_i - Y_i$를 정의한다.
양의 차이 개수를 $n_+$, 음의 차이 개수를 $n_-$라 하자. 동점($D_i = 0$)은 제외하며
유효 표본크기는 $n = n_+ + n_-$이다.

$H_0{:}\; \text{median}(D) = 0$ 아래에서 0이 아닌 각 차이는 양수일 확률과 음수일 확률이
같으므로 $n_+$는 $\text{Binomial}(n, 1/2)$를 따른다. 양의 부호의 표본비율은

$$
\hat{p} = \frac{n_+}{n}
$$

이다. 정규근사의 표준화 통계량은

$$
Z = \frac{\hat{p} - 0.5}{\sqrt{0.5 \cdot 0.5 \,/\, n}}
  = \frac{\hat{p} - 0.5}{\,1 / (2\sqrt{n}\,)}
$$

이다. $p$값은 대립가설에 따라 달라진다.

| 대립가설 | $p$값 |
|---|---|
| $H_1{:}\; \text{median}(D) \neq 0$ | $2\,\Phi(-\lvert Z \rvert)$ |
| $H_1{:}\; \text{median}(D) > 0$ | $1 - \Phi(Z)$ |
| $H_1{:}\; \text{median}(D) < 0$ | $\Phi(Z)$ |

## 구현

```python
import numpy as np
import scipy.stats as stats


def sign_test(paired_data, test_type="two-sided"):
    """
    Sign test for paired observations.

    Parameters
    ----------
    paired_data : ndarray of shape (n, 2)
        Column 0 is post-treatment, column 1 is pre-treatment.
    test_type : str
        One of "less", "two-sided", "greater".

    Returns
    -------
    z : float
        The Z test statistic.
    p_value : float
    """
    p_0, q_0 = 0.5, 0.5

    n_plus = np.sum(paired_data[:, 0] > paired_data[:, 1])
    n_minus = np.sum(paired_data[:, 0] < paired_data[:, 1])
    n = n_plus + n_minus  # ties excluded
    p_hat = n_plus / n

    z = (p_hat - p_0) / np.sqrt(p_0 * q_0 / n)

    if test_type == "less":
        p_value = stats.norm.cdf(z)
    elif test_type == "two-sided":
        p_value = 2 * stats.norm.cdf(-abs(z))
    elif test_type == "greater":
        p_value = stats.norm.sf(z)

    return z, p_value
```

<div class="exbox" markdown>

### 보기 1. 학생의 처치 전후 점수 { .ex }

학생 15명을 처치 프로그램 전후에 측정했다.

</div>

```python
paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

diffs = paired_data[:, 0] - paired_data[:, 1]
print(diffs)
# [17 -2  6 -3 14  0  6 10 12  8  9  2  0  3  0]

z, p_value = sign_test(paired_data, test_type="two-sided")
print(f"Z = {z:.4f}, p = {p_value:.4f}")
# Z = 2.3094, p = 0.0209
```

출력:

```
[17 -2  6 -3 14  0  6 10 12  8  9  2  0  3  0]
Z = 2.3094, p = 0.0209
```

15쌍 중 **세 쌍**이 동점($D_i = 0$)이라 제외되어 $n = 12$가 남는다. 그중
$n_+ = 10$, $n_- = 2$이므로 $\hat{p} = 10/12 \approx 0.833$이고
$Z \approx 2.309$, 양측 $p$값은 약 $0.021$이다.

!!! warning "동점의 개수를 반드시 확인하라"
    이 자료에서 동점은 학생 6번(54 대 54), 13번(78 대 78), 15번(76 대 76)의
    **세 개**이다. 동점을 하나 놓치면 $n$이 $12$가 아니라 $13$이 되어 $\hat{p}$,
    $Z$, $p$값이 모두 달라진다. `sign_test`를 호출하기 전에 항상 `diffs`를 출력하여
    확인하는 습관을 들이는 것이 좋다.

## 해석

- 부호검정은 대응 비교를 Bernoulli 시행의 수열로 바꾼다. 본질적으로 "이 동전은
  공정한가?"를 묻는 것이며, 앞면은 "후 > 전"을 뜻한다.
- 부호 정보만 쓰므로 **분포무관**이다. 차이에 대해 정규성이나 대칭성 가정이 필요 없다.
- 대가는 낮은 **검정력**이다. 차이가 얼마나 큰지를 무시한다. 차이가 대략 대칭이면
  Wilcoxon 부호순위검정이 더 강력하다.
- 표본이 매우 작으면($n < 20$) 정규근사 대신 정확 이항분포를 써야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 피험자 8명을 식단 전후에 측정했다. 체중 변화(후 $-$ 전, kg)는
$-3, +1, -2, 0, -4, -1, +2, -5$이다. 정확 이항분포로 $\alpha = 0.05$에서
부호검정을 수행하라.

</div>

??? success "풀이"

    동점($D = 0$)을 버리면 $n = 7$이 남는다. 0이 아닌 차이 중
    $n_+ = 2$, $n_- = 5$이다.

    $H_0$ 아래에서 $n_+ \sim \text{Binomial}(7, 0.5)$이므로 양측 $p$값은

    $$
    p = 2 \cdot P(n_+ \leq 2) = 2 \sum_{k=0}^{2} \binom{7}{k} (0.5)^7
      = 2 \cdot \frac{1 + 7 + 21}{128} = \frac{58}{128} \approx 0.453.
    $$

    $p = 0.453 > 0.05$이므로 $H_0$을 기각하지 못한다. 중앙값 체중 변화가 0과
    다르다는 유의한 증거가 없다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 학생 자료 예제에서 $H_1{:}\; \text{median}(D) > 0$(처치가 점수를
높인다)인 단측검정으로 부호검정을 다시 수행하라. $\alpha = 0.05$에서 결론을 밝혀라.

</div>

??? success "풀이"

    예제에서 $n_+ = 10$, $n = 12$, $\hat{p} = 10/12 \approx 0.833$,
    $Z \approx 2.309$였다.

    단측(greater) 대립가설의 $p$값은

    $$
    p = 1 - \Phi(Z) = 1 - \Phi(2.309) \approx 0.0105.
    $$

    정확 이항 단측 $p$값은 $P(X \ge 10) = 79/4096 = 0.0193$이다.

    ```python
    from scipy import stats
    print(stats.binomtest(10, 12, alternative='greater').pvalue)   # 0.019287
    ```

    출력:

    ```
    0.019287109375
    ```

    두 값 모두 $0.05$보다 작으므로 $H_0$을 기각하고, 처치가 점수를 유의하게
    높인다고 결론짓는다. 다만 정규근사($0.0105$)가 정확값($0.0193$)의 절반에
    가깝다는 점에 유의하라. $n = 12$는 정규근사에 작다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** 부호검정 통계량 $Z$를 다음과 같이 쓸 수 있음을 보여라.

$$
Z = \frac{2\,n_+ - n}{\sqrt{n}}.
$$

</div>

??? success "풀이"

    정의에서 출발하면

    $$
    Z = \frac{\hat{p} - 0.5}{\sqrt{0.25/n}}
      = \frac{n_+/n - 0.5}{\,1/(2\sqrt{n}\,)}
      = \frac{(n_+ - n/2)/n}{\,1/(2\sqrt{n}\,)}
      = \frac{2\sqrt{n}\,(n_+ - n/2)}{n}
      = \frac{2\,n_+ - n}{\sqrt{n}}.
    $$

    $\square$

    이 형태가 유용한 것은 $2n_+ - n = n_+ - n_-$이기 때문이다. 즉

    $$
    Z = \frac{n_+ - n_-}{\sqrt{n_+ + n_-}}
    $$

    로도 쓸 수 있으며, 이는 "양의 부호가 음의 부호보다 얼마나 많은가"를
    표준화한 것이라는 직관을 그대로 보여 준다. 학생 자료에서
    $Z = (10 - 2)/\sqrt{12} = 8/3.464 = 2.309$로 앞의 계산과 일치한다.

<div class="drillbox" markdown>

**연습문제 4.** 차이의 분포가 심하게 치우쳐 있어도 부호검정이 타당한 이유를
설명하고, Wilcoxon 부호순위검정은 부적절하지만 부호검정은 적절한 상황을 하나 들어라.

</div>

??? success "풀이"

    부호검정은 $H_0$ 아래에서 $P(D_i > 0) = P(D_i < 0) = 0.5$라는 사실에만
    의존하며, 이는 $D_i$가 중앙값 0인 연속분포이기만 하면 성립한다. 대칭성이나
    적률의 유한성 같은 가정이 전혀 필요 없다.

    Wilcoxon 부호순위검정은 여기에 더해 $D_i$의 분포가 0을 중심으로 **대칭**임을
    요구한다. 예를 들어 차이가 중앙값 0이 되도록 이동한 지수분포에서 나온다면
    (즉 크기가 심하게 오른쪽으로 치우쳐 있다면) 부호순위검정의 대칭성 가정이
    깨지고 귀무분포가 더 이상 옳지 않다. 이 경우에도 부호검정은 타당하다.

    [Wilcoxon 부호순위검정](./wilcoxon_signed_rank.md)
    연습문제 2에서 이를 정량적으로 확인했다. 중앙값 이동 지수분포에서
    $n = 100$일 때 Wilcoxon의 제1종 오류율이 $0.376$까지 치솟는 반면
    부호검정은 $0.035$를 유지했다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 정규근사가 아니라 정확 이항분포를 쓰는 부호검정 함수를 작성하고
학생 자료로 검정하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import binom

    def sign_test_exact(paired_data):
        """Exact two-sided sign test for paired observations."""
        diffs = paired_data[:, 0] - paired_data[:, 1]
        nonzero = diffs[diffs != 0]
        n = len(nonzero)
        n_plus = (nonzero > 0).sum()

        # Two-sided p-value: 2 * min(P(X <= n_+), P(X >= n_+))
        p_left = binom.cdf(n_plus, n, 0.5)
        p_right = binom.sf(n_plus - 1, n, 0.5)  # P(X >= n_+)
        p_value = 2 * min(p_left, p_right)
        p_value = min(p_value, 1.0)  # cap at 1

        return n_plus, n, p_value

    paired_data = np.array([
        [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
        [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
        [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
    ])

    n_plus, n, p = sign_test_exact(paired_data)
    print(f"n_+ = {n_plus}, n = {n}, exact p = {p:.4f}")
    # n_+ = 10, n = 12, exact p = 0.0386
    ```

    출력:

    ```
    n_+ = 10, n = 12, exact p = 0.0386
    ```

    정확 양측 $p$값은 $0.0386$으로 정규근사값 $0.0209$의 **1.8배**이다.
    두 값 모두 $\alpha = 0.05$에서 기각하므로 결론은 같지만, 근사가
    유의성을 과장하고 있다.

    `min(p_value, 1.0)`으로 1을 자르는 처리가 필요한 이유는
    $n_+$가 $n/2$에 가까울 때 $2\min(\cdot)$이 1을 넘을 수 있기 때문이다.
    $n = 12$, $n_+ = 6$이면 $2 \times P(X \le 6) = 2 \times 0.6128 = 1.226$이 된다.

    `scipy.stats.binomtest`를 쓰면 이 처리가 자동으로 되며, 같은 값 $0.0386$을
    반환한다.

    ```python
    from scipy.stats import binomtest
    print(binomtest(10, 12).pvalue)   # 0.03857421875
    ```

    출력:

    ```
    0.03857421875
    ```

    $\square$

<div class="drillbox" markdown>

**연습문제 6.** 부호검정과 Wilcoxon 부호순위검정, 대응 $t$ 검정을 학생 자료에
모두 적용하여 $p$값을 비교하고, 순서가 왜 그렇게 나오는지 설명하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    d = np.array([17, -2, 6, -3, 14, 0, 6, 10, 12, 8, 9, 2, 0, 3, 0])

    print(stats.binomtest(10, 12).pvalue)                        # 0.038574
    print(stats.wilcoxon(d, zero_method='wilcox',
                         method='exact').pvalue)                 # 0.007579
    print(stats.ttest_1samp(d, 0).pvalue)                        # 0.003816
    ```

    출력:

    ```
    0.03857421875
    0.007579201614253502
    0.003815602209766083
    ```

    | 검정 | 양측 $p$값 | 사용 정보 |
    |:---|---:|:---|
    | 부호검정 (정확) | $0.0386$ | 부호만 |
    | Wilcoxon 부호순위 (정확) | $0.0076$ | 부호 + 순위 |
    | 대응 $t$ | $0.0038$ | 원값 |

    사용하는 정보량이 늘어날수록 $p$값이 작아진다. 이 자료에서는 그 서열이
    깨끗하게 나타난다.

    **왜 이 자료가 부호검정에 특히 불리한가.** 음의 차이 두 개가 $-2$와 $-3$으로
    **절댓값이 가장 작은 축**이다. 양의 차이는 $2$부터 $17$까지 퍼져 있다.

    - 부호검정에게 $-3$은 $-300$과 똑같은 한 개의 음수이다.
    - Wilcoxon은 $-2$와 $-3$이 절댓값 순위 $1.5$와 $3.5$를 받는다는 것을 안다.
      $W^- = 5$에 불과하고 $W^+ = 73$이다.
    - $t$ 검정은 여기에 더해 $17$과 $14$가 얼마나 큰지까지 반영한다.

    반대로 음의 차이가 $-17$과 $-14$였다면 서열이 뒤집혔을 것이다. 검정의
    상대적 성능은 자료의 구조에 달려 있으며, "$t$ 검정이 언제나 가장 작은
    $p$값을 준다"는 규칙은 존재하지 않는다.

---

## 정리하며

부호검정은 **가장 단순한** 비모수 검정이다.

- **부호만 세면 끝난다.** 양의 차이 개수가 $\text{Binomial}(n,0.5)$ 를 따르는지 보는 것이며, 이항검정 그 자체다.
- **가정이 극도로 적다.** 차이가 독립이고 연속이면 된다. **대칭성도 필요 없다**는 점이 윌콕슨과의 결정적 차이다.
- **$0$ 인 차이는 버린다.** 그만큼 유효 표본이 줄어들며, 그 수를 보고해야 한다.
- **검정력이 낮다.** 크기 정보를 버리므로 윌콕슨이나 $t$ 검정보다 약하며, 정규 자료에서 점근효율이 $2/\pi\approx0.64$ 다.
- **중앙값에 대한 검정임을 기억한다.** 평균이 아니라 중앙값이 $0$ 인지를 묻는다.

다음 절 **Wilcoxon 검정 (코드)** 로 넘어간다.
