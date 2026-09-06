# 분산 동일성에 대한 F 검정

## 개요

분산 동일성에 대한 F 검정은 정규분포를 따르는 두 모집단의 분산을 비교하는 이표본 검정이다. 검정통계량은 두 표본분산의 비이며 귀무가설 아래에서 F 분포를 따른다. 정확한 정규성 아래에서는 우아하고 최적이지만, F 검정은 정규성 이탈에 극도로 민감하기로 악명 높다. 그래서 실무에서는 Levene 검정 같은 로버스트 대안이 대체로 낫다.

## 검정 설정

$X_1, \ldots, X_{n_1} \overset{\text{iid}}{\sim} N(\mu_1, \sigma_1^2)$과 $Y_1, \ldots, Y_{n_2} \overset{\text{iid}}{\sim} N(\mu_2, \sigma_2^2)$이 독립 표본이라 하자. 가설은

$$
H_0 : \sigma_1^2 = \sigma_2^2 \quad \text{대} \quad H_1 : \sigma_1^2 \neq \sigma_2^2.
$$

## 검정통계량

F 통계량은 두 표본분산의 비로 정의된다.

$$
F = \frac{S_1^2}{S_2^2},
$$

여기서 $S_i^2 = \frac{1}{n_i - 1}\sum_{j=1}^{n_i}(X_{ij} - \bar{X}_i)^2$이다. $H_0$ 아래에서

$$
F \sim F(n_1 - 1,\; n_2 - 1).
$$

## 판정규칙

수준 $\alpha$의 양측검정에서 $p$값은

$$
p = 2\min\!\bigl(F_{F\text{-dist}}(F_{\text{obs}}),\; 1 - F_{F\text{-dist}}(F_{\text{obs}})\bigr),
$$

여기서 $F_{F\text{-dist}}$는 $F(n_1-1, n_2-1)$의 CDF이다. $p < \alpha$이면 $H_0$을 기각한다.

## 코드

```python
import numpy as np
import scipy.stats as stats


def f_test(data_0, data_1):
    """
    Two-sample F-test for equality of variances.

    H0: sigma_1^2 = sigma_2^2
    H1: sigma_1^2 != sigma_2^2
    """
    statistic = data_0.var(ddof=1) / data_1.var(ddof=1)
    df1 = data_0.shape[0] - 1
    df2 = data_1.shape[0] - 1
    p_value = 2 * min(
        stats.f(df1, df2).cdf(statistic),
        stats.f(df1, df2).sf(statistic),
    )
    return statistic, p_value
```

다음 예제는 두 번째 표본의 표준편차를 점점 키우면서 F 검정을 적용한다.

```python
size, seed = 100, 1
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = f_test(x, y)
    print(f"sigma_y={scale:.2f}  F={stat:.4f}  p={pval:.3f}")
```

출력:

```text
sigma_y=1.00  F=1.0000  p=1.000
sigma_y=1.05  F=0.9070  p=0.628
sigma_y=1.10  F=0.8264  p=0.345
sigma_y=1.15  F=0.7561  p=0.166
sigma_y=1.20  F=0.6944  p=0.071
```

!!! warning "이 예제는 표집변동을 완전히 제거한다"
    `x`와 `y`가 **같은 `random_state=seed`**를 쓰므로 두 표본이 독립이 아니다. $y = 1 + \text{scale} \times z$이고 $x = z$인 동일한 표준정규 추출값 $z$를 공유한다.

    그 결과 F 통계량이 **정확히** $1/\text{scale}^2$이 된다.

    | scale | $1/\text{scale}^2$ | 출력된 $F$ |
    |---|---|---|
    | 1.00 | 1.00000 | 1.0000 |
    | 1.05 | 0.90703 | 0.9070 |
    | 1.10 | 0.82645 | 0.8264 |
    | 1.15 | 0.75614 | 0.7561 |
    | 1.20 | 0.69444 | 0.6944 |

    `scale=1.00`에서 $F$가 정확히 1이고 $p$가 정확히 1인 것이 결정적 증거이다. 독립 표본이라면 결코 일어나지 않는다.

    이 설정은 **잡음 없는 순수한 신호**를 보여주므로 교육적으로 유용하다. 다만 이 표를 "검정력"으로 읽어서는 안 된다. 실제로는 $F$가 $1/\text{scale}^2$ 주위로 흩어지므로, 같은 $\sigma$ 비율에서도 $p$값이 표본마다 크게 달라진다.

    독립 표본으로 바꾸려면 `y`에 다른 seed를 주면 된다(예: `random_state=seed + 1`).

## 해석

- 두 모집단의 분산이 같으면($\sigma_1 = \sigma_2$) F 통계량이 1에 가깝고 $p$값이 크다.
- 분산비가 1에서 멀어질수록 F 통계량이 1에서 멀어지고 $p$값이 작아진다.
- F 검정은 비정규성에 극도로 민감하다. 중간 정도의 치우침이나 두꺼운 꼬리만으로도 실제 제1종 오류율이 $\alpha$를 크게 넘을 수 있다. 정규성이 의심스러우면 Levene이나 Brown-Forsythe 검정이 낫다.

표에서 $n_1 = n_2 = 100$이고 표준편차 비가 1.2(분산비 1.44)인데도 $p = 0.071$로 5% 수준을 넘지 못한다는 점에 주목하라. **표집변동이 전혀 없는 이상적 상황에서도** 그렇다. 두 표본을 비교하는 F 검정의 검정력이 얼마나 낮은지 보여준다.

## 연습문제

**연습문제 1.** 두 실험실이 어떤 화합물의 농도를 측정한다. 실험실 A는 $n_1 = 15$회 측정에서 $S_1^2 = 3.2$를, 실험실 B는 $n_2 = 20$회 측정에서 $S_2^2 = 1.8$을 보고했다. F 통계량을 계산하고 자유도를 밝힌 뒤 Python으로 양측 $p$값을 구하라.

??? success "연습문제 1 풀이"

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{3.2}{1.8} = 1.7778, \quad df_1 = 14,\; df_2 = 19.
    $$

    ```python
    import scipy.stats as stats

    F = 3.2 / 1.8
    df1, df2 = 14, 19
    p = 2 * min(stats.f(df1, df2).cdf(F), stats.f(df1, df2).sf(F))
    print(f"F = {F:.4f}, p = {p:.4f}")
    ```

    출력:

    ```text
    F = 1.7778, p = 0.2413
    ```

    $p = 0.241$이므로 $\alpha = 0.05$에서 $H_0$을 기각하지 못한다.

    **해석.** 표본분산의 비가 $1.78$배로 꽤 커 보이지만 유의하지 않다. 15.3절 연습문제 3의 표에 따르면 $n \approx 15$에서 F 검정이 탐지하려면 분산비가 $3$배 이상이어야 한다.

    실험실 A의 측정 정밀도가 실제로 나쁠 가능성이 높지만, 이 자료만으로는 확정할 수 없다. **표본을 늘리거나** 여러 차례 반복 측정한 결과를 축적해야 한다. $\square$

---

**연습문제 2.** $H_0$ 아래에서 $F = S_1^2/S_2^2 \sim F(d_1, d_2)$이면 $1/F = S_2^2/S_1^2 \sim F(d_2, d_1)$임을 보여라. 어느 표본분산을 분자에 두든 양측 $p$값이 같은 이유를 설명하라.

??? success "연습문제 2 풀이"

    $H_0$ 아래에서 $(n_1-1)S_1^2/\sigma^2 \sim \chi^2(d_1)$과 $(n_2-1)S_2^2/\sigma^2 \sim \chi^2(d_2)$가 독립이다. F 분포의 정의에 의해

    $$
    F = \frac{S_1^2}{S_2^2} = \frac{\chi^2(d_1)/d_1}{\chi^2(d_2)/d_2} \sim F(d_1, d_2).
    $$

    역수를 취하면 분자와 분모가 뒤바뀌므로

    $$
    \frac{1}{F} = \frac{\chi^2(d_2)/d_2}{\chi^2(d_1)/d_1} \sim F(d_2, d_1).
    $$

    **양측 $p$값의 불변성.** $F_{\text{obs}}$에 대한 양측 $p$값은

    $$
    p = 2\min\bigl(P(F(d_1,d_2) \le F_{\text{obs}}),\; P(F(d_1,d_2) \ge F_{\text{obs}})\bigr).
    $$

    분자와 분모를 바꾸면 관측값이 $1/F_{\text{obs}}$가 되고 참조분포가 $F(d_2,d_1)$이 된다. 역수 성질에서

    $$
    P(F(d_1,d_2) \ge F_{\text{obs}}) = P\!\left(\frac{1}{F(d_1,d_2)} \le \frac{1}{F_{\text{obs}}}\right) = P\!\left(F(d_2,d_1) \le \frac{1}{F_{\text{obs}}}\right)
    $$

    이므로 두 꼬리확률이 정확히 맞바뀐다. $\min$을 취하면 같은 값이 나온다.

    **수치 확인.** 연습문제 1에서 $F = 1.7778$, $df = (14,19)$일 때 $p = 0.2413$이다. 뒤집으면 $F = 0.5625$, $df = (19,14)$이고 역시 $p = 0.2413$이다.

    ```python
    import scipy.stats as stats
    print(2 * min(stats.f(14, 19).cdf(3.2/1.8), stats.f(14, 19).sf(3.2/1.8)))
    print(2 * min(stats.f(19, 14).cdf(1.8/3.2), stats.f(19, 14).sf(1.8/3.2)))
    ```

    **단측검정에서는 다르다.** 대립가설이 $\sigma_1^2 > \sigma_2^2$인지 $\sigma_1^2 < \sigma_2^2$인지에 따라 어느 꼬리를 보는지가 달라지므로, 순서를 바꾸면 대립가설도 함께 바꿔야 한다. $\square$

---

**연습문제 3.** 10,000회 반복의 몬테카를로 모의실험을 작성하라. 각 반복에서 $N(0,1)$에서 $n_1 = n_2 = 20$을 뽑아 $\alpha = 0.05$로 F 검정을 적용한다. 경험적 제1종 오류가 0.05에 가까운지 확인하라. 그런 다음 $t(3)$ 자료로 반복하고 결과를 논평하라.

??? success "연습문제 3 풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(0)
    n, alpha, n_sims = 20, 0.05, 10000

    for dist_name in ["Normal", "t(3)"]:
        rej = 0
        for _ in range(n_sims):
            if dist_name == "Normal":
                x = rng.normal(0, 1, n)
                y = rng.normal(0, 1, n)
            else:
                x = stats.t(df=3).rvs(n, random_state=rng)
                y = stats.t(df=3).rvs(n, random_state=rng)
            F = np.var(x, ddof=1) / np.var(y, ddof=1)
            p = 2 * min(stats.f(n-1, n-1).cdf(F), stats.f(n-1, n-1).sf(F))
            if p < alpha:
                rej += 1
        print(f"{dist_name}: rejection rate = {rej/n_sims:.4f}")
    ```

    출력:

    ```text
    Normal: rejection rate = 0.0499
    t(3): rejection rate = 0.2917
    ```

    정규성 아래에서 $0.0499$로 정확하다. $t(3)$에서는 **$0.2917$로 명목값의 거의 여섯 배**이다.

    등분산인 자료의 **29%에서 "분산이 다르다"고 판정한다.** 이는 15.3절 표의 $t_5$ 결과(0.156)보다도 훨씬 나쁘다. $t_3$은 네 번째 적률조차 존재하지 않아 $S^2$의 분산이 무한대이기 때문이다.

    **실무적 함의.** 명목 5% F 검정을 $t_3$ 수준의 두꺼운 꼬리 자료에 적용하면, 유의한 결과의 대부분이 거짓 양성이다. $p = 0.03$을 보고 "분산이 다르다"고 결론지어서는 안 된다.

    금융 수익률의 첨도가 흔히 3~10임을 떠올리면(15.7절), 이 상황이 실무에서 드물지 않다는 점이 우려스럽다. $\square$

---

**연습문제 4.** 단측검정 $H_0: \sigma_1^2 \le \sigma_2^2$ 대 $H_1: \sigma_1^2 > \sigma_2^2$을 하고 싶다고 하자. 기각규칙을 서술하고 Python 코드를 단측 $p$값을 반환하도록 수정하라.

??? success "연습문제 4 풀이"

    $H_1: \sigma_1^2 > \sigma_2^2$에서는 $F = S_1^2/S_2^2$의 큰 값이 $H_0$에 반하는 증거이다. 다음이면 기각한다.

    $$
    F > F_{1-\alpha}(n_1-1, n_2-1).
    $$

    단측 $p$값은 $p = P\bigl(F(n_1-1,n_2-1) \ge F_{\text{obs}}\bigr)$이다.

    ```python
    import numpy as np
    import scipy.stats as stats

    def f_test_variances(data_0, data_1, alternative='two-sided'):
        """F-test for equality of variances.

        alternative: 'two-sided', 'greater' (sigma_0 > sigma_1), or 'less'
        """
        F = np.var(data_0, ddof=1) / np.var(data_1, ddof=1)
        df1, df2 = len(data_0) - 1, len(data_1) - 1
        dist = stats.f(df1, df2)
        if alternative == 'greater':
            p = dist.sf(F)
        elif alternative == 'less':
            p = dist.cdf(F)
        elif alternative == 'two-sided':
            p = 2 * min(dist.cdf(F), dist.sf(F))
        else:
            raise ValueError(f"unknown alternative: {alternative}")
        return F, p
    ```

    **연습문제 1에 적용하면.** 실험실 A의 정밀도가 더 나쁜지($\sigma_A^2 > \sigma_B^2$)만 관심이라면 단측 $p$값은 $0.2413/2 = 0.1206$이다. 여전히 기각하지 못하지만 양측보다 절반이다.

    **주의.** 단측검정은 방향을 **자료를 보기 전에** 정했을 때만 정당하다. 관측된 $F > 1$을 보고 나서 단측으로 바꾸면 실제 유의수준이 $2\alpha$가 된다. $\square$

---

**연습문제 5.** $F \sim F(d_1, d_2)$일 때 $d_2 > 2$에 대해 $E[F] = \frac{d_2}{d_2 - 2}$임을 증명하고, $E[F]$가 $d_1$에 의존하지 않는 이유를 설명하라.

??? success "연습문제 5 풀이"

    $F = (U/d_1)/(V/d_2)$로 쓰고 $U \sim \chi^2(d_1)$, $V \sim \chi^2(d_2)$가 독립이라 하자. 그러면

    $$
    E[F] = \frac{d_2}{d_1} \cdot E[U] \cdot E[1/V].
    $$

    $E[U] = d_1$이고 $d_2 > 2$인 $V \sim \chi^2(d_2)$에 대해 $E[1/V] = 1/(d_2 - 2)$(역카이제곱 적률)이므로

    $$
    E[F] = \frac{d_2}{d_1} \cdot d_1 \cdot \frac{1}{d_2 - 2} = \frac{d_2}{d_2 - 2}.
    $$

    $d_1$ 항이 소거되는 것은 어떤 $d_1$에 대해서도 $E[U/d_1] = 1$이기 때문이다. 곧 분자의 자유도는 $F$의 평균이 아니라 **분산**에만 영향을 준다.

    **$E[F] > 1$이라는 사실의 실무적 의미.** $H_0$이 참이어도 $F$의 기댓값이 정확히 1이 아니다.

    | $d_2$ | 5 | 10 | 20 | 50 | 100 |
    |---|---|---|---|---|---|
    | $E[F]$ | 1.667 | 1.250 | 1.111 | 1.042 | 1.020 |

    분모 자유도가 작으면 등분산인데도 $F$가 평균적으로 1보다 훨씬 크게 나온다. Jensen 부등식 때문이다. $E[1/W] > 1/E[W]$이고 $W = V/d_2$의 평균이 1이므로 $E[1/W] > 1$이다.

    **그래서 $F$ 값 자체를 1과 비교하여 눈으로 판단해서는 안 된다.** 반드시 올바른 자유도의 F 분포와 비교해야 한다. $d_2 = 5$에서 $F = 1.6$은 오히려 평균보다 작은 값이다. $\square$
