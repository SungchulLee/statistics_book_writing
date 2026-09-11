# 형식적 정규성 검정 모음

## 개요

형식적 정규성 검정은 자료가 정규분포에서 왔다는 가설에 찬성하거나 반대하는 객관적이고 정량적인 증거를 제공한다. 시각적 확인과 달리 검정통계량과 $p$값을 내놓아 선택한 유의수준에서 원칙 있는 판정을 가능하게 한다. 이 페이지는 가장 널리 쓰이는 검정들의 귀무가설, 강점, 표본크기 고려사항을 살펴본다.

## 가설검정의 틀

모든 정규성 검정은 공통 구조를 갖는다. 귀무가설과 대립가설은

$$
H_0: X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2) \quad (\text{어떤 } \mu, \sigma^2 \text{에 대해}), \qquad H_1: \text{자료가 정규분포를 따르지 않는다}.
$$

표본에서 검정통계량 $T$를 계산한다. $H_0$ 아래에서 $T$는 알려진(또는 표로 정리된) 분포를 갖는다. $p$값은

$$
p = P(T \geq T_{\text{obs}} \mid H_0),
$$

부등호의 방향은 검정마다 다르다. $p < \alpha$이면 $H_0$을 기각한다.

## 흔한 검정 개관

| 검정 | 민감한 대상 | 표본크기 지침 | SciPy 함수 |
|---|---|---|---|
| Shapiro-Wilk | 일반적 이탈 | $n \leq 5000$에서 최선 | `stats.shapiro` |
| D'Agostino $K^2$ | 왜도와 첨도 | $n \geq 20$ | `stats.normaltest` |
| Jarque-Bera | 왜도와 첨도 | 큰 $n$(점근적) | `stats.jarque_bera` |
| Kolmogorov-Smirnov | 분포의 모양 | 모든 $n$. 모수를 알아야 함 | `stats.kstest` |
| Anderson-Darling | 꼬리 | 표로 정리된 임계값 | `stats.anderson` |
| Lilliefors | 모양(추정된 모수) | 모수 추정 시 KS를 보정 | 붓스트랩 또는 `lilliefors` |

### 코드

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
data = rng.normal(0, 1, size=100)

# Shapiro-Wilk
W, p_sw = stats.shapiro(data)
print(f"Shapiro-Wilk:     W = {W:.4f}, p = {p_sw:.4g}")

# D'Agostino K^2
K2, p_k2 = stats.normaltest(data)
print(f"D'Agostino K^2:   K2 = {K2:.4f}, p = {p_k2:.4g}")

# Jarque-Bera
JB, p_jb = stats.jarque_bera(data)
print(f"Jarque-Bera:      JB = {JB:.4f}, p = {p_jb:.4g}")

# Kolmogorov-Smirnov (fully specified N(0,1))
D, p_ks = stats.kstest(data, 'norm', args=(0, 1))
print(f"KS (vs N(0,1)):   D = {D:.4f}, p = {p_ks:.4g}")

# Anderson-Darling
ad = stats.anderson(data, dist="norm")
print(f"Anderson-Darling: A^2 = {ad.statistic:.4f}")
for cv, sl in zip(ad.critical_values, ad.significance_level):
    print(f"  {sl:.0f}% critical value: {cv:.4f}")
```

출력:

```text
Shapiro-Wilk:     W = 0.9819, p = 0.1873
D'Agostino K^2:   K2 = 1.9679, p = 0.3738
Jarque-Bera:      JB = 1.4236, p = 0.4908
KS (vs N(0,1)):   D = 0.0723, p = 0.6465
Anderson-Darling: A^2 = 0.4494
  15% critical value: 0.5550
  10% critical value: 0.6320
  5% critical value: 0.7590
  2% critical value: 0.8850
  1% critical value: 1.0530
```

자료를 실제로 $N(0,1)$에서 생성했으므로 다섯 검정 모두 정규성을 기각하지 않는다. Anderson-Darling 통계량 $0.449$도 가장 느슨한 15% 임계값 $0.555$보다 작다.

## 검정 고르기

일률적으로 최선인 검정은 없다. 일반적인 지침은 다음과 같다.

- **작은 표본($n < 50$)**: Shapiro-Wilk가 폭넓은 대립가설에 대해 검정력이 가장 좋다.
- **중간 표본($50 \leq n \leq 5000$)**: Shapiro-Wilk나 Anderson-Darling이 선호된다. 치우침이나 첨도 문제가 의심되면 D'Agostino $K^2$가 좋은 전방위 선택이다.
- **큰 표본($n > 5000$)**: 어떤 검정이든 아주 작은 이탈에도 기각한다. 검정을 효과 크기 측도(표본왜도, 초과첨도)와 시각적 확인으로 보완하라.

## 해석

유의한 결과(작은 $p$값)는 자료가 정규분포에서 왔을 가능성이 낮다는 뜻이지만, *어떻게* 벗어나는지는 알려주지 않는다. 형식적 검정은 언제나 시각적 진단과 함께 쓰라. 반대로 유의하지 않은 결과가 정규성을 증명하지도 않는다. 검정이 참된 대립가설에 대해 검정력이 부족했을 뿐일 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 표준정규 관측값 $n = 200$개를 생성하고 Shapiro-Wilk, D'Agostino $K^2$, Jarque-Bera 검정을 수행하라. $p$값을 보고하라. $\alpha = 0.05$에서 기각하는 검정이 있는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=200)

    W, p_sw = stats.shapiro(data)
    K2, p_k2 = stats.normaltest(data)
    JB, p_jb = stats.jarque_bera(data)

    print(f"Shapiro-Wilk:   p = {p_sw:.4g}")
    print(f"D'Agostino K^2: p = {p_k2:.4g}")
    print(f"Jarque-Bera:    p = {p_jb:.4g}")
    ```

    출력:

    ```text
    Shapiro-Wilk:   p = 0.4893
    D'Agostino K^2: p = 0.301
    Jarque-Bera:    p = 0.3226
    ```

    자료가 실제로 정규분포에서 왔으므로 세 $p$값이 모두 0.05를 훨씬 넘는다. 어느 검정도 기각하지 않는다. 정의상 각 검정은 $H_0$ 아래에서 5%의 확률로만 잘못 기각한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 연습문제 1을 $\text{Lognormal}(0, 0.5)$ 분포에서 뽑아 반복하라. $p$값을 비교하고 오른쪽 치우침에 가장 민감한 검정이 무엇인지 설명하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    data = rng.lognormal(0, 0.5, size=200)

    W, p_sw = stats.shapiro(data)
    K2, p_k2 = stats.normaltest(data)
    JB, p_jb = stats.jarque_bera(data)

    print(f"Shapiro-Wilk:   p = {p_sw:.4g}")
    print(f"D'Agostino K^2: p = {p_k2:.4g}")
    print(f"Jarque-Bera:    p = {p_jb:.4g}")
    ```

    출력:

    ```text
    Shapiro-Wilk:   p = 1.153e-12
    D'Agostino K^2: p = 1.664e-22
    Jarque-Bera:    p = 3.108e-107
    ```

    세 검정 모두 압도적으로 기각한다($p \ll 0.05$). 다만 여기서 **가장 작은 $p$값을 내는 것은 Shapiro-Wilk가 아니라 Jarque-Bera**($3.1 \times 10^{-107}$)이며, D'Agostino $K^2$가 그다음이다.

    이유는 대수정규분포의 이탈이 정확히 왜도와 첨도라는 두 적률에 집중되어 있고, 적률 기반 검정이 그것을 직접 겨냥하기 때문이다. $n = 200$이면 적률 추정이 충분히 안정적이므로 이 두 검정의 검정력이 매우 커진다.

    "Shapiro-Wilk가 언제나 가장 강력하다"는 통념은 **작은 표본**과 **이탈의 유형을 모를 때**에 해당한다. 이 예처럼 이탈이 왜도와 첨도로 뚜렷하고 표본이 충분하면 적률 기반 검정이 더 큰 증거를 낸다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** Kolmogorov-Smirnov 검정이 귀무가설의 모수를 완전히 지정하도록 요구하는 이유를 설명하라. $\mu$와 $\sigma$를 자료에서 추정해 꽂아 넣으면 $p$값에 무슨 일이 일어나는가?

</div>

??? success "풀이"

    KS 검정은 경험적 CDF $F_n(x)$를 완전히 지정된 이론적 CDF $F_0(x)$와 비교한다. 그 임계값과 $p$값은 $F_0$이 자료를 보기 *전에* 고정되어 있다는 가정 아래에서 유도되었다. $\mu$와 $\sigma$를 같은 자료에서 추정하면 적합된 CDF $\hat{F}(x)$가 일반적인 $F_0$보다 구성상 $F_n$에 더 가까워진다. 그래서 KS 거리 $D_n$이 체계적으로 작아지고 $p$값이 부풀려지며 검정력이 떨어진다. 올바른 절차는 모수 추정을 반영하기 위해 모의실험이나 전용 표를 쓰는 Lilliefors 검정이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 어떤 동료가 Shapiro-Wilk 검정이 기각하지 못했으니 자료가 "정규임이 증명되었다"고 주장한다. 제2종 오류와 검정력의 개념을 써서 짧게 반박하라.

</div>

??? success "풀이"

    $H_0$을 기각하지 못한 것은 $H_0$의 증명이 아니다. 유의하지 않은 $p$값은 자료가 정규성과 *양립 가능*하다는 뜻이지만, 검정이 구별할 검정력을 갖지 못한 여러 비정규 분포와도 양립 가능하다. 제2종 오류의 확률 $\beta$는 표본크기 $n$, 유의수준 $\alpha$, 참 대립분포에 의존한다. $n$이 작으면 검정력 $1 - \beta$가 상당히 낮을 수 있으므로 기각하지 못한 것에 담긴 정보가 거의 없다. 올바른 추론에는 검정력 분석이나 보조 증거(시각적 확인, 분야 지식)가 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** $\alpha = 0.05$에서 Shapiro-Wilk 검정의 경험적 크기를 추정하는 몬테카를로 실험을 설계하라. $\mathcal{N}(0,1)$에서 크기 $n = 50$인 표본 10,000개를 뽑아 검정을 적용하고 기각률을 보고하라. 0.05에 얼마나 가까운가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 50, 10000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.normal(0, 1, size=n)
        _, p = stats.shapiro(x)
        if p < alpha:
            rejections += 1

    empirical_size = rejections / reps
    print(f"Empirical size: {empirical_size:.4f}")
    ```

    출력:

    ```text
    Empirical size: 0.0487
    ```

    경험적 기각률 $0.0487$이 $0.05$에 매우 가깝다. Shapiro-Wilk 검정의 크기가 올바르게 조정되어 있음을 확인해 준다. $H_0$ 아래에서 약 $\alpha \times 100\%$의 비율로 기각한다는 뜻이다. 0.05에서 벗어난 부분은 몬테카를로 표집오차 때문이며, 그 크기는 $\sqrt{\alpha(1-\alpha)/\text{reps}} = \sqrt{0.05 \times 0.95/10000} \approx 0.0022$이다. 관측된 편차 $0.0013$은 이 오차 범위 안에 있다. $\square$
