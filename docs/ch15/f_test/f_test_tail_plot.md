# F 검정 꼬리 영역 시각화

## 개요

F 분포의 꼬리 영역을 시각화하는 것은 분산 동일성에 대한 F 검정이 어떻게 판정에 이르는지 이해하는 데 필수적이다. 관측된 F 통계량을 밀도 곡선 위에 표시하면 음영으로 칠한 꼬리 면적이 $p$값에 대응한다. 이 페이지는 단측과 양측 대립가설에 대해 그런 그림을 만드는 법을 시연하여 F 분포를 이용한 가설검정의 기하학적 직관을 세운다.

## F 분포와 꼬리 면적

정규 모집단에서 나온 크기 $n_1$, $n_2$인 독립 표본 둘에 대해 F 통계량

$$
F_{\text{obs}} = \frac{S_1^2}{S_2^2}
$$

은 $H_0: \sigma_1^2 = \sigma_2^2$ 아래에서 $F(d_1, d_2)$ 분포를 따른다. 여기서 $d_1 = n_1 - 1$, $d_2 = n_2 - 1$이다.

$p$값은 대립가설에 따라 달라진다.

- **오른쪽 꼬리** ($H_1: \sigma_1^2 > \sigma_2^2$): $p = P(F \ge F_{\text{obs}})$.
- **왼쪽 꼬리** ($H_1: \sigma_1^2 < \sigma_2^2$): $p = P(F \le F_{\text{obs}})$.
- **양측** ($H_1: \sigma_1^2 \neq \sigma_2^2$): $p = 2\min\!\bigl(P(F \le F_{\text{obs}}),\; P(F \ge F_{\text{obs}})\bigr)$.

## 코드

다음 코드는 두 표본에서 F 통계량을 계산하고 $F(d_1, d_2)$ 밀도와 양쪽 꼬리 영역을 그린다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f

sample1 = [12, 15, 14, 10, 13, 14, 12, 11]
sample2 = [22, 25, 20, 18, 24, 23, 19, 21]

x1 = np.asarray(sample1, dtype=float)
x2 = np.asarray(sample2, dtype=float)
df1, df2 = x1.size - 1, x2.size - 1

F_obs = x1.var(ddof=1) / x2.var(ddof=1)

xs = np.linspace(0.01, max(6, F_obs + 2), 400)
pdf = f(df1, df2).pdf(xs)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(xs, pdf, linewidth=2, label="F({},{}) PDF".format(df1, df2))

# Left-tail shading
mask_left = xs <= F_obs
ax.fill_between(xs[mask_left], pdf[mask_left], 0, alpha=0.15, label="Left tail")

# Right-tail shading
mask_right = xs >= F_obs
ax.fill_between(xs[mask_right], pdf[mask_right], 0, alpha=0.15, label="Right tail")

ax.axvline(F_obs, linestyle="--", color="black", label=f"F_obs = {F_obs:.3f}")
ax.set_xlabel("F")
ax.set_ylabel("Density")
ax.set_title(f"F({df1}, {df2}) with observed F = {F_obs:.3f}")
ax.legend()
plt.tight_layout()
plt.show()
```

$p$값을 명시적으로 계산하려면

```python
p_left = f(df1, df2).cdf(F_obs)
p_right = f(df1, df2).sf(F_obs)
p_two = 2 * min(p_left, p_right)

print(f"Sample variances: {x1.var(ddof=1):.4f}, {x2.var(ddof=1):.4f}")
print(f"F_obs = {F_obs:.4f}")
print(f"Left-tail  p-value: {p_left:.4f}")
print(f"Right-tail p-value: {p_right:.4f}")
print(f"Two-sided  p-value: {p_two:.4f}")
```

출력:

```text
Sample variances: 2.8393, 6.0000
F_obs = 0.4732
Left-tail  p-value: 0.1724
Right-tail p-value: 0.8276
Two-sided  p-value: 0.3448
```

## 해석

- $F_{\text{obs}}$까지의 왼쪽 꼬리 면적은 $H_0$ 아래에서 관측된 것만큼 작거나 더 작은 분산비를 얻을 확률이다.
- $F_{\text{obs}}$부터의 오른쪽 꼬리 면적은 그만큼 크거나 더 큰 비를 얻을 확률이다.
- 양측검정에서는 더 작은 쪽 꼬리 면적을 두 배 한다. $F_{\text{obs}}$가 1에 가까우면($d_2$가 클 때 $H_0$ 아래의 기댓값) 양쪽 꼬리가 모두 커서 $p$값이 1에 가까워진다.
- F 분포는 특히 자유도가 작을 때 오른쪽으로 치우쳐 있다. 이 비대칭 때문에 주어진 $F_{\text{obs}}$에 대해 왼쪽과 오른쪽 꼬리 $p$값이 일반적으로 같지 않다.

여기서 두 꼬리 확률이 $0.172$와 $0.828$로 합이 정확히 1이다(연속분포이므로 당연하다). 양측 $p$값 $0.345$는 작은 쪽을 두 배 한 값이다.

## 연습문제

**연습문제 1.** 위 코드의 표본에 대해 $F_{\text{obs}}$를 손으로 계산하고 코드 출력과 대조하라. 자유도를 서술하라.

??? success "풀이"

    표본 1: $\{12, 15, 14, 10, 13, 14, 12, 11\}$, $n_1 = 8$, $\bar{x}_1 = 12.625$.

    $$
    S_1^2 = \frac{1}{7}\sum(x_i - 12.625)^2
    $$

    $$
    = \frac{1}{7}(0.390625 + 5.640625 + 1.890625 + 6.890625 + 0.140625 + 1.890625 + 0.390625 + 2.640625) = \frac{19.875}{7} = 2.8393.
    $$

    표본 2: $\{22, 25, 20, 18, 24, 23, 19, 21\}$, $n_2 = 8$, $\bar{x}_2 = 21.5$.

    $$
    S_2^2 = \frac{1}{7}(0.25 + 12.25 + 2.25 + 12.25 + 6.25 + 2.25 + 6.25 + 0.25) = \frac{42}{7} = 6.0.
    $$

    $$
    F_{\text{obs}} = \frac{2.8393}{6.0} = 0.4732, \quad d_1 = 7, \; d_2 = 7.
    $$

    코드 출력과 정확히 일치한다.

    **자유도가 왜 $n-1$인가.** 각 표본분산이 자기 집단의 평균을 추정하는 데 자유도 하나를 썼기 때문이다. $S_i^2$의 분자 $\sum (x_{ij} - \bar{x}_i)^2$은 제약 $\sum_j (x_{ij} - \bar{x}_i) = 0$을 만족하므로 자유로운 성분이 $n_i - 1$개이다. $\square$

---

**연습문제 2.** 단측검정 $H_1: \sigma_1^2 > \sigma_2^2$에 맞게 오른쪽 꼬리만 음영으로 칠하도록 코드를 수정하라. 오른쪽 꼬리 $p$값은 얼마인가?

??? success "풀이"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
    df1, df2 = 7, 7
    F_obs = x1.var(ddof=1) / x2.var(ddof=1)

    xs = np.linspace(0.01, 6, 400)
    pdf = f(df1, df2).pdf(xs)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, pdf, lw=2)
    mask = xs >= F_obs
    ax.fill_between(xs[mask], pdf[mask], 0, alpha=0.3, color="red",
                    label="Right tail")
    ax.axvline(F_obs, ls="--", color="black")
    ax.set_title(f"Right-tail test: p = {f(df1,df2).sf(F_obs):.4f}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    ```

    $F_{\text{obs}} = 0.4732 < 1$이므로 오른쪽 꼬리 면적이 매우 크다. $p = 0.8276$으로 $\sigma_1^2 > \sigma_2^2$의 증거가 전혀 없다.

    **당연한 결과이다.** 표본분산이 $2.84 < 6.00$으로 오히려 집단 1이 작으므로, "집단 1의 분산이 더 크다"는 대립가설의 방향과 자료가 정반대이다.

    !!! warning "$p > 0.5$는 대립가설의 방향이 틀렸다는 신호이다"
        단측검정에서 $p$값이 $0.5$를 크게 넘으면, 자료가 대립가설과 **반대 방향**을 가리키고 있다는 뜻이다. 이때 "$H_0$을 기각하지 못한다"고만 보고하면 정보를 잃는다.

        올바른 서술은 "자료는 $\sigma_1^2 > \sigma_2^2$을 지지하지 않으며, 오히려 반대 방향($\sigma_1^2 < \sigma_2^2$, 왼쪽 꼬리 $p = 0.172$)을 시사한다"이다.

        다만 **자료를 보고 대립가설의 방향을 바꾸어서는 안 된다.** 그렇게 하면 실제 유의수준이 $2\alpha$가 된다. 방향을 미리 확신할 수 없다면 처음부터 양측검정을 계획해야 한다. $\square$

---

**연습문제 3.** $F(d_1, d_2)$ 밀도가 1을 중심으로 대칭이 아닌 이유를 설명하라. 이 비대칭이 양측 $p$값 계산에 어떤 영향을 주는가?

??? success "풀이"

    F 분포는 척도가 조정된 두 카이제곱 변수의 비로 정의되며, 둘 다 음이 아니고 오른쪽으로 치우쳐 있다. 밀도의 정의역이 $(0, \infty)$이고 $d_1 > 2$에 대해 최빈값이

    $$
    \frac{d_1-2}{d_1} \cdot \frac{d_2}{d_2+2} < 1
    $$

    이다. 이 고유한 오른쪽 치우침 때문에 일반적으로 $P(F > c) \neq P(F < 1/c)$이다.

    양측 $p$값에서는 정규분포처럼 대칭인 분포에서 하듯 한쪽 꼬리 면적을 단순히 두 배 할 수 없다. 대신 $p = 2\min(P(F \le F_{\text{obs}}), P(F \ge F_{\text{obs}}))$를 쓴다. 작은 쪽 꼬리를 골라 두 배 하는 것이다. 이렇게 하면 검정이 타당해지지만, 기각역이 $F$ 척도에서 1을 중심으로 대칭이 아니게 된다.

    **로그 척도에서는 대칭이 회복된다.** $d_1 = d_2 = d$일 때 역수 성질 $1/F \sim F(d,d)$에 의해 $\ln F$의 분포가 0을 중심으로 **정확히 대칭**이다. 본문 예제에서 $d_1 = d_2 = 7$이므로

    $$
    P(F \le 0.4732) = P(F \ge 1/0.4732) = P(F \ge 2.1132) = 0.1724
    $$

    가 성립한다. 로그 척도의 대칭성이 이 관계를 만든다.

    $d_1 \neq d_2$이면 이 대칭도 깨진다. 그때는 $\min$을 두 배 하는 규칙이 정확히 $\alpha$ 크기를 주지 않고 다소 보수적일 수 있다(합쳐서 $\alpha$보다 작을 수 있다). $\square$

---

**연습문제 4.** $(d_1, d_2) \in \{(5,5), (10,10), (30,30)\}$에 대한 $F(d_1, d_2)$ 밀도를 세 개의 부분그림으로 그려라. 자유도가 커지면서 모양이 어떻게 변하는지 논하라.

??? success "풀이"

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (d1, d2) in zip(axes, [(5, 5), (10, 10), (30, 30)]):
        xs = np.linspace(0.01, 4, 300)
        ax.plot(xs, f(d1, d2).pdf(xs), lw=2)
        ax.axvline(1, ls=":", color="gray")
        ax.set_title(f"F({d1}, {d2})")
        ax.set_xlabel("F")
    plt.tight_layout()
    plt.show()

    for d1, d2 in [(5, 5), (10, 10), (30, 30)]:
        mode = (d1 - 2) / d1 * d2 / (d2 + 2)
        mean = d2 / (d2 - 2)
        print(f"F({d1},{d2}): mode={mode:.4f}, mean={mean:.4f}")
    ```

    출력:

    ```text
    F(5,5):   mode=0.4286, mean=1.6667
    F(10,10): mode=0.6667, mean=1.2500
    F(30,30): mode=0.8750, mean=1.0714
    ```

    | $(d_1,d_2)$ | 최빈값 | 평균 | 왜도 |
    |---|---|---|---|
    | $(5,5)$ | 0.429 | 1.667 | 정의되지 않음 |
    | $(10,10)$ | 0.667 | 1.250 | 3.615 |
    | $(30,30)$ | 0.875 | 1.071 | 1.268 |

    자유도가 커지면 F 밀도가 1 주위로 더 집중되고(기댓값이 $d_2/(d_2-2) \to 1$) 더 대칭이 된다. $d_1, d_2$가 크면 $\ln F$가 근사적으로 정규이고 분포가 1 근처를 중심으로 하는 정규분포와 비슷해진다.

    !!! note "$F(5,5)$의 왜도는 존재하지 않는다"
        F 분포의 왜도 공식은 $d_2 > 6$을 요구한다. $d_2 \leq 6$이면 3차 적률이 존재하지 않는다.

        마찬가지로 평균은 $d_2 > 2$, 분산은 $d_2 > 4$가 필요하다. **분모 자유도가 작으면 F 분포의 적률이 차례로 사라진다.**

        실무적 함의: 두 표본이 각각 $n \leq 7$이면($d_2 \leq 6$) F 통계량의 왜도조차 정의되지 않을 만큼 분포가 극단적이다. 이런 자료에서 F 검정의 $p$값은 형식적으로는 계산되지만, 검정통계량이 매우 불안정하므로 결과를 신뢰하기 어렵다. $\square$

---

**연습문제 5.** $F \sim F(d_1, d_2)$이면 $1/F \sim F(d_2, d_1)$임을 증명하라. 이를 이용해 $F(d_1, d_2)$ 아래에서 $F_{\text{obs}}$의 왼쪽 꼬리 $p$값이 $F(d_2, d_1)$ 아래에서 $1/F_{\text{obs}}$의 오른쪽 꼬리 $p$값과 같음을 보여라.

??? success "풀이"

    정의에 의해 $F = (U/d_1)/(V/d_2)$이고 $U \sim \chi^2(d_1)$, $V \sim \chi^2(d_2)$가 독립이다. 그러면

    $$
    \frac{1}{F} = \frac{V/d_2}{U/d_1} \sim F(d_2, d_1)
    $$

    자유도를 뒤바꾼 F 분포의 정의 그대로이다.

    이제 $P(F \le F_{\text{obs}}) = P(1/F \ge 1/F_{\text{obs}})$이다. $1/F \sim F(d_2, d_1)$이므로 이는 오른쪽 꼬리확률 $P(F(d_2, d_1) \ge 1/F_{\text{obs}})$이며, 정확히 $1/F_{\text{obs}}$에서 평가한 $F(d_2, d_1)$의 생존함수이다.

    **수치 확인.**

    ```python
    from scipy.stats import f

    F_obs, d1, d2 = 0.4732, 7, 7
    print(f"left tail  of F({d1},{d2}) at {F_obs}:      "
          f"{f(d1, d2).cdf(F_obs):.6f}")
    print(f"right tail of F({d2},{d1}) at {1/F_obs:.4f}: "
          f"{f(d2, d1).sf(1 / F_obs):.6f}")
    ```

    두 값이 소수 여섯째 자리까지 같다.

    **실무적 의미.** 이 성질 덕분에 F 분포표에 상단 분위수만 실어도 충분하다. 하단 임계값이 필요하면 자유도를 뒤바꾼 상단 임계값의 역수를 취하면 된다.

    또한 이는 "어느 집단을 분자에 둘지"가 검정 결과에 영향을 주지 않는다는 것을 보장한다. 양측검정에서는 $\min$을 취하므로 두 배열이 정확히 같은 $p$값을 낸다(15.8절 [F 검정](f_test_variances.md) 연습문제 2 참조). $\square$
