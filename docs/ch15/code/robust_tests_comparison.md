# 로버스트 분산 검정 비교

## 개요

집단 간 분산을 비교할 때 검정의 선택은 바탕 자료의 분포에 따라 크게 중요해진다. Bartlett 검정 같은 고전적 검정은 정규성 아래에서 최적이지만 비정규성에서 무너진다. 이 페이지는 분산 동일성에 대한 여러 검정(Bartlett, 평균 중심과 중앙값 중심 Levene, Fligner-Killeen)을 서로 다른 분포 설정에서 제1종 오류 조절과 검정력 측면으로 비교한다.

## 비교하는 검정들

분산 동질성에 대한 네 가지 주요 검정은 다음과 같다.

| 검정 | 가정 | 로버스트성 |
|---|---|---|
| Bartlett | 정규성 필요 | 비정규성에 로버스트하지 않음 |
| Levene (평균) | 없음 | 중간 정도로 로버스트 |
| Brown-Forsythe (중앙값 Levene) | 없음 | 치우침에 로버스트 |
| Fligner-Killeen | 없음 | 매우 로버스트 (비모수) |

정규성과 귀무가설 $H_0: \sigma_1^2 = \cdots = \sigma_k^2$ 아래에서 네 검정 모두 대략 $\alpha$의 비율로 기각해야 한다. 비정규성 아래에서는 Bartlett 검정이 제1종 오류를 부풀리는 반면 나머지는 조절을 유지한다.

## 수학적 틀

네 검정은 공통의 관점으로 볼 수 있다. 집단 중심으로부터의 편차에 기반한 변환값 $Z_{ij}$를 정의하고 $Z_{ij}$의 집단평균이 다른지 검정하는 것이다.

$$
Z_{ij} = |X_{ij} - c_i|,
$$

여기서 $c_i$는 집단평균(Levene), 중앙값(Brown-Forsythe), 또는 순위 기반 중심(Fligner-Killeen)이다. Bartlett 검정은 대신 로그가능도비에 직접 작동한다.

## 코드

다음 모의실험은 집단분산이 모두 같은 치우친(대수정규) 분포에서 네 검정의 거짓 양성률을 비교한다.

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
n_sims, n, alpha = 2000, 30, 0.05
results = {"Bartlett": 0, "Levene (mean)": 0,
           "Brown-Forsythe": 0, "Fligner-Killeen": 0}

for _ in range(n_sims):
    g1 = rng.lognormal(0, 1, n)
    g2 = rng.lognormal(0, 1, n)
    g3 = rng.lognormal(0, 1, n)

    _, p = stats.bartlett(g1, g2, g3)
    if p < alpha:
        results["Bartlett"] += 1

    _, p = stats.levene(g1, g2, g3, center='mean')
    if p < alpha:
        results["Levene (mean)"] += 1

    _, p = stats.levene(g1, g2, g3, center='median')
    if p < alpha:
        results["Brown-Forsythe"] += 1

    _, p = stats.fligner(g1, g2, g3)
    if p < alpha:
        results["Fligner-Killeen"] += 1

for name, count in results.items():
    print(f"{name:20s}: false-positive rate = {count/n_sims:.4f}")
```

출력:

```text
Bartlett            : false-positive rate = 0.7320
Levene (mean)       : false-positive rate = 0.2605
Brown-Forsythe      : false-positive rate = 0.0295
Fligner-Killeen     : false-positive rate = 0.1140
```

분산이 실제로 다른 정규 자료에서의 검정력 비교는 다음과 같다.

```python
results_power = {"Bartlett": 0, "Levene (mean)": 0,
                 "Brown-Forsythe": 0, "Fligner-Killeen": 0}

for _ in range(n_sims):
    g1 = rng.normal(0, 1.0, n)
    g2 = rng.normal(0, 1.5, n)
    g3 = rng.normal(0, 2.0, n)

    _, p = stats.bartlett(g1, g2, g3)
    if p < alpha:
        results_power["Bartlett"] += 1

    _, p = stats.levene(g1, g2, g3, center='mean')
    if p < alpha:
        results_power["Levene (mean)"] += 1

    _, p = stats.levene(g1, g2, g3, center='median')
    if p < alpha:
        results_power["Brown-Forsythe"] += 1

    _, p = stats.fligner(g1, g2, g3)
    if p < alpha:
        results_power["Fligner-Killeen"] += 1

for name, count in results_power.items():
    print(f"{name:20s}: power = {count/n_sims:.4f}")
```

출력:

```text
Bartlett            : power = 0.9215
Levene (mean)       : power = 0.8490
Brown-Forsythe      : power = 0.8155
Fligner-Killeen     : power = 0.7810
```

## 해석

| | 대수정규 크기 ($H_0$) | 정규 검정력 ($H_1$) |
|---|---|---|
| Bartlett | **0.732** | 0.922 |
| Levene (평균) | **0.261** | 0.849 |
| Brown-Forsythe | 0.030 | 0.816 |
| Fligner-Killeen | **0.114** | 0.781 |

- **정규성과 등분산 아래**: 네 검정 모두 대략 $\alpha = 0.05$로 기각한다.
- **비정규성(대수정규)과 등분산 아래**: Bartlett의 거짓 양성률이 $0.732$로 심하게 부풀려지고, Levene(평균)도 $0.261$로 문제가 있다.
- **정규성과 이분산 아래(검정력)**: Bartlett의 검정력이 가장 높고(정규성 아래에서 최적이므로) Levene(평균), Brown-Forsythe, Fligner-Killeen 순이다.

!!! warning "Fligner-Killeen이 언제나 가장 로버스트한 것은 아니다"
    표에서 Fligner-Killeen의 대수정규 크기가 **0.114**로 명목값의 두 배가 넘는다. Brown-Forsythe의 $0.030$보다 나쁘다.

    "Fligner-Killeen이 가장 로버스트하다"는 서술은 **대칭인 두꺼운 꼬리**에 대해서는 맞다. 15.8절 [Fligner-Killeen 검정](fligner_killeen.md) 연습문제 2에서 Cauchy 자료에 대해 FK가 $0.044$로 BF의 $0.022$보다 명목값에 가까웠다.

    그러나 **강한 치우침**에서는 반대이다. $\text{Lognormal}(0,1)$은 왜도 $6.18$, 초과첨도 $110.9$로 극단적으로 치우쳐 있는데, 이때 FK의 순위 변환이 오히려 부족하다. 편차 $|x - \tilde{x}|$ 자체가 강하게 치우쳐 있어 정규점수의 집단평균이 표본마다 크게 흔들리기 때문이다.

    | 자료 | Brown-Forsythe | Fligner-Killeen |
    |---|---|---|
    | Cauchy (대칭, 극단 꼬리) | 0.022 | **0.044** |
    | $\chi^2_4$ (치우침 중간) | 0.041 | 0.055 |
    | Exponential (치우침 강함) | 0.048 | 0.087 |
    | Lognormal(0,1) (치우침 극단) | **0.030** | 0.114 |

    **결론: 치우침이 주된 문제이면 Brown-Forsythe, 대칭인 두꺼운 꼬리가 주된 문제이면 Fligner-Killeen을 쓰라.** 어느 쪽인지 모른다면 Brown-Forsythe가 안전한 기본값이다. 이 표에서 유일하게 모든 행에서 통제되는 검정이기 때문이다.

절충이 명확하다. Bartlett은 정규성이 성립할 때 검정력에서 이기지만 그렇지 않으면 파국적으로 실패한다. 범용으로는 Brown-Forsythe가 로버스트성과 검정력의 균형이 가장 좋다.

## 연습문제

**연습문제 1.** 위 거짓 양성 모의실험을 $n = 30$ 대신 $n = 100$으로 실행하라. 표본크기를 늘리면 대수정규 자료에서 Bartlett 검정의 제1종 오류 조절이 개선되는가? 설명하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 100, 0.05
    rej = 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        g3 = rng.lognormal(0, 1, n)
        _, p = stats.bartlett(g1, g2, g3)
        if p < alpha:
            rej += 1

    print(f"Bartlett false-positive rate (n=100): {rej/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett false-positive rate (n=100): 0.8134
    ```

    $n$을 늘려도 Bartlett의 제1종 오류가 개선되지 **않는다**. 오히려 $n = 30$의 $0.732$에서 $n = 100$의 $0.813$으로 **악화**된다.

    | $n$ | 20 | 30 | 100 |
    |---|---|---|---|
    | Bartlett 크기 | 0.675 | 0.732 | 0.813 |

    ($n = 20$ 값은 15.8절 [Bartlett 검정](bartlett_test.md) 연습문제 4에서 얻은 것이다.)

    **이유.** 15.3절 연습문제 4에서 유도했듯, 명목 분포와 실제 분포의 산포 비율이

    $$
    \frac{\operatorname{Var}(\ln S^2)_{\text{실제}}}{\operatorname{Var}(\ln S^2)_{\text{명목}}} = \frac{\gamma_2 + 2}{2}
    $$

    로 $n$에 의존하지 않는다. $n$이 커지면 명목 $\chi^2$ 분포가 좁아지는데 실제 분포는 같은 비율로 넓은 상태를 유지하므로, 임계값 밖으로 나가는 질량의 비율이 커진다.

    이는 **모형 가정 위반**이지 유한표본 인공물이 아니다. 자료를 아무리 모아도 잘못된 기준분포는 옳아지지 않으며, 오히려 잘못된 결론을 더 확신 있게 내리게 된다. $\square$

---

**연습문제 2.** $k = 2$개 집단인 경우 F 검정(이표본판)을 비교에 추가하라. 대수정규 자료에서 그 제1종 오류율이 Bartlett과 어떻게 비교되는가?

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 30, 0.05
    rej_f, rej_b = 0, 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        F = np.var(g1, ddof=1) / np.var(g2, ddof=1)
        p_f = 2 * min(stats.f(n-1, n-1).cdf(F), stats.f(n-1, n-1).sf(F))
        _, p_b = stats.bartlett(g1, g2)
        if p_f < alpha:
            rej_f += 1
        if p_b < alpha:
            rej_b += 1

    print(f"F-test:   {rej_f/n_sims:.4f}")
    print(f"Bartlett: {rej_b/n_sims:.4f}")
    ```

    출력:

    ```text
    F-test:   0.5108
    Bartlett: 0.5108
    ```

    **두 값이 완전히 동일하다.** 소수점 이하까지 같다.

    우연이 아니다. 15.8절 [Bartlett 검정](bartlett_test.md) 연습문제 5에서 증명했듯, $k = 2$이고 $n_1 = n_2$이면 Bartlett 통계량이 F 통계량의 엄격히 단조인 함수이다. 따라서 두 검정의 기각역이 정확히 일치하고, 모든 표본에서 같은 결론을 낸다.

    **두 검정 모두 크기가 $0.511$이다.** 등분산인 자료의 절반 이상에서 기각한다. 대수정규 자료에 분산 검정을 적용할 때 정규성 기반 방법을 쓰면 안 되는 이유를 보여준다. $\square$

---

**연습문제 3.** 집단 표준편차가 $\sigma_1 = 1$, $\sigma_2 = 1.5$, $\sigma_3 = 2$이고 자료가 $t(5)$ 분포에서 올 때 각 검정의 검정력을 추정하는 모의실험을 설계하라. 전체적으로 어느 검정이 가장 좋은지 논하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 30, 0.05
    power = {"Bartlett": 0, "Levene": 0, "BF": 0, "FK": 0}

    for _ in range(n_sims):
        g1 = stats.t(df=5).rvs(n, random_state=rng) * 1.0
        g2 = stats.t(df=5).rvs(n, random_state=rng) * 1.5
        g3 = stats.t(df=5).rvs(n, random_state=rng) * 2.0
        _, p = stats.bartlett(g1, g2, g3)
        if p < alpha:
            power["Bartlett"] += 1
        _, p = stats.levene(g1, g2, g3, center='mean')
        if p < alpha:
            power["Levene"] += 1
        _, p = stats.levene(g1, g2, g3, center='median')
        if p < alpha:
            power["BF"] += 1
        _, p = stats.fligner(g1, g2, g3)
        if p < alpha:
            power["FK"] += 1

    for name, count in power.items():
        print(f"{name:10s}: {count/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett  : 0.8578
    Levene    : 0.7096
    BF        : 0.6584
    FK        : 0.6616
    ```

    | 검정 | 기각률 ($H_1$) | $t_5$ 아래 크기 ($H_0$) |
    |---|---|---|
    | Bartlett | 0.858 | 0.267 |
    | Levene (평균) | 0.710 | 0.054 |
    | Brown-Forsythe | 0.658 | 0.042 |
    | Fligner-Killeen | 0.662 | 0.040 |

    (크기는 15.5절 비교 페이지의 $t_5$ 행에서 가져왔다.)

    Bartlett이 겉보기 "검정력"이 가장 높아 보이지만 **오도적이다.** 크기가 이미 $0.267$로 부풀려져 있기 때문이다. 등분산인 자료의 27%를 기각하던 검정이 이분산에서 86%를 기각한 것은 대단한 성취가 아니다.

    크기가 통제된 세 검정만 비교하면 **Levene(평균)이 $0.710$으로 가장 강력하다.** $t_5$는 대칭이므로 평균 중심화가 무너지지 않고, 중앙값·순위 변환이 버리는 정보를 활용하기 때문이다.

    Brown-Forsythe($0.658$)와 Fligner-Killeen($0.662$)은 사실상 같다.

    **전체적 권고.** 자료가 **대칭이면서 꼬리만 두껍다**고 확신할 수 있으면 Levene(평균)이 최선이다. 치우침 가능성이 있으면 Brown-Forsythe로 가야 한다. 본문 표에서 대수정규일 때 Levene(평균)의 크기가 $0.261$까지 치솟았음을 기억하라. $\square$

---

**연습문제 4.** Brown-Forsythe 검정이 평균 중심 Levene 검정보다 치우침에 로버스트한 이유를 수학적으로 설명하라.

??? success "풀이"

    치우친 분포에서는 평균이 꼬리 쪽으로 끌려가므로 절대편차 $Z_{ij} = |X_{ij} - \bar{X}_i|$가 그 비대칭을 물려받는다. 긴 꼬리 쪽의 관측값들이 체계적으로 더 큰 $Z_{ij}$ 값을 만들어 낸다. 이는 $Z_{ij}$의 집단내 변동을 부풀리고 분산분석 F 통계량에 영향을 준다.

    중앙값은 50백분위수이므로 극단값의 **크기**에 영향받지 않는다. 중앙값으로 중심화하면 원래의 $X_{ij}$가 치우쳐 있어도 $Z_{ij}$가 더 대칭적으로 분포한다.

    형식적으로 $X$가 분포 $F$를 가지면

    $$
    E[|X - \text{median}|] \le E[|X - \mu|]
    $$

    이다(중앙값이 기대절대편차를 최소화하므로). 따라서 중앙값 중심 편차의 분산이 더 작고 꼬리에 덜 민감하다. 이것이 $H_0$ 아래에서 검정통계량의 더 나은 보정으로 이어진다.

    **더 근본적인 이유: 표집 안정성.** 위 부등식은 모집단 수준의 진술이다. 실무에서 더 중요한 것은 **추정된 중심의 표집변동**이다.

    - 치우친 분포에서 $\bar{X}_i$는 표본마다 크게 흔들린다. 특히 극단값 하나가 포함되었는지 여부가 평균을 크게 바꾼다.
    - 중앙값은 붕괴점이 50%이므로 극단값 몇 개에 거의 영향받지 않는다.

    $\bar{X}_i$가 흔들리면 그 집단의 **모든** $Z_{ij}$가 함께 이동한다. 이 공통 이동이 $\bar{Z}_i$들을 실제보다 흩어지게 만들어 F 통계량의 분자를 부풀린다. 15.5절 비교 페이지 연습문제 1에서 논한 $\bar{X}$와 $S^2$의 상관이 바로 이 현상이다.

    본문 표의 수치가 이를 확인해 준다. 대수정규에서 Levene(평균) $0.261$ 대 Brown-Forsythe $0.030$으로 아홉 배 차이이다. $\square$

---

**연습문제 5.** 결과가 오른쪽으로 치우쳐 있다고 알려진 임상시험 자료(예: 입원 기간)를 분석한다고 하자. 후속 분석을 결정하기 전에 세 처치군의 등분산을 확인해야 한다. 어떤 검정을 권하며 그 이유는 무엇인가? 대안을 최소 두 가지 논하라.

??? success "풀이"

    입원 기간처럼 오른쪽으로 치우친 임상 자료에는 **Brown-Forsythe 검정**(`scipy.stats.levene`에 `center='median'`)을 권한다. 치우침 아래에서 제1종 오류율을 잘 조절하면서 참된 분산 차이를 탐지할 합리적인 검정력을 유지한다.

    본문 표가 이를 뒷받침한다. 극단적으로 치우친 대수정규 자료에서 Brown-Forsythe만이 $0.030$으로 통제되고, Fligner-Killeen($0.114$), Levene 평균($0.261$), Bartlett($0.732$)은 모두 부풀려진다.

    **대안 1: Fligner-Killeen 검정.** 중앙값으로부터의 절대편차의 순위에 기반한 비모수 검정이다. 이상점이 있을 수 있을 때 적절하다. 다만 위에서 보았듯 **극단적 치우침에서는 오히려 Brown-Forsythe보다 크기 조절이 나쁘다.** "가장 로버스트하다"는 통념에 기대어 무조건 선택해서는 안 된다.

    **대안 2: 로그 변환 후 Bartlett.** 치우침이 대수정규 같은 기제에서 온다면 로그 변환이 자료를 대칭화하여 더 강력한 Bartlett 검정을 타당하게 쓸 수 있다. 위험은 변환이 자료를 완전히 정규화하지 못할 수 있다는 점과, 추론이 로그 척도에서 이루어진다는 점이다.

    입원 기간의 경우 로그 변환이 특히 자연스럽다. 대수정규 모형이 생존시간·대기시간 자료에 잘 맞는 것으로 알려져 있기 때문이다. 다만 변환 후에도 반드시 정규성을 확인해야 하며(14장), 0값이 있으면 $\log(x+c)$ 형태의 조정이 필요하다.

    **대안 3: 붓스트랩 검정.** 15.6절의 붓스트랩 분산 검정은 분포 가정을 하지 않는다. 다만 강한 치우침에서 크기가 다소 부풀려질 수 있으므로(15.6절 연습문제 3에서 지수분포 0.082) 만능은 아니다.

    !!! tip "가장 나은 대응은 이 질문을 피하는 것이다"
        "등분산을 확인한 뒤 후속 분석을 결정한다"는 두 단계 절차 자체가 15.7절 [분산분석 사전검정](../applications/anova_pretest.md)에서 논한 문제를 안고 있다.

        임상시험이라면 분석계획서를 사전에 확정해야 하므로 더욱 그렇다. **처음부터 Welch 분산분석(또는 로그 변환 후 표준 분산분석)을 쓰기로 명시**하면 이 판정이 필요 없어진다.

        등분산 검정 결과는 방법 선택의 근거가 아니라 **자료의 특성을 서술하는 보조 정보**로 보고하는 것이 옳다. 집단별 표준편차와 상자그림을 함께 제시하면 독자가 스스로 판단할 수 있다.

    Bartlett 검정과 F 검정은 치우친 자료에 직접 적용해서는 안 된다. 신뢰할 수 없는 $p$값을 낳기 때문이다. $\square$
