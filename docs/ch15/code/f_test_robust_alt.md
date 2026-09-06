# F 검정의 정규성 민감도와 로버스트 대안

## 개요

분산 동일성에 대한 F 검정은 두 모집단이 모두 정규분포를 따른다고 가정한다. 이 가정이 위배되면 검정이 오도하는 결과를 낼 수 있다. 이 페이지는 Shapiro-Wilk 검정으로 정규성 가정을 확인하는 방법을 시연하고, 비정규성 아래에서 타당한 제1종 오류율을 유지하는 로버스트 대안(Levene 검정, Brown-Forsythe 검정, Fligner-Killeen 검정)을 제시한다.

## 정규성 확인

F 검정을 적용하기 전에 정규성 가정이 성립하는지 평가하는 것이 신중하다. Shapiro-Wilk 검정은 정규성에 대해 가장 강력한 검정 가운데 하나이다.

$$
H_0: \text{자료가 정규분포에서 온다} \quad \text{대} \quad H_1: \text{자료가 정규가 아니다}.
$$

$p$값이 작으면 비정규성을 시사하며, 그 경우 F 검정의 로버스트 대안을 써야 한다.

## 로버스트 대안

| 검정 | 구현 | 핵심 성질 |
|---|---|---|
| Levene (평균) | `levene(x1, x2, center='mean')` | 평균으로부터의 절대편차에 분산분석 |
| Brown-Forsythe | `levene(x1, x2, center='median')` | 중앙값으로부터의 절대편차에 분산분석 |
| Fligner-Killeen | `fligner(x1, x2)` | 비모수, 절대편차의 순위에 기반 |

Brown-Forsythe 검정은 집단평균을 집단중앙값으로 바꾸어 치우침에 더 로버스트하다. Fligner-Killeen 검정은 순위 기반 접근을 써서 대칭인 두꺼운 꼬리에 가장 로버스트하다.

## 코드

```python
import numpy as np
from scipy.stats import levene, fligner, shapiro

x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

# Step 1: Check normality with Shapiro-Wilk
W1, p1 = shapiro(x1)
W2, p2 = shapiro(x2)
print(f"Shapiro-Wilk x1: W={W1:.4f}, p={p1:.4f}")
print(f"Shapiro-Wilk x2: W={W2:.4f}, p={p2:.4f}")

# Step 2: Apply robust alternatives
Wm, pm = levene(x1, x2, center='mean')
print(f"Levene (mean-centered):           W={Wm:.4f}, p={pm:.6f}")

Wmed, pmed = levene(x1, x2, center='median')
print(f"Brown-Forsythe (median-centered): W={Wmed:.4f}, p={pmed:.6f}")

X2, pF = fligner(x1, x2)
print(f"Fligner-Killeen:                  X2={X2:.4f}, p={pF:.6f}")
```

출력:

```text
Shapiro-Wilk x1: W=0.9657, p=0.8619
Shapiro-Wilk x2: W=0.9749, p=0.9332
Levene (mean-centered):           W=1.4831, p=0.243425
Brown-Forsythe (median-centered): W=1.4706, p=0.245320
Fligner-Killeen:                  X2=1.6063, p=0.205015
```

## 해석

- Shapiro-Wilk $p$값이 크면(예: $> 0.05$) 정규성 가정이 그럴듯하고 F 검정을 쓸 수 있다.
- 어느 한 표본이라도 정규성이 기각되면 F 검정을 신뢰해서는 안 된다. Levene, Brown-Forsythe, Fligner-Killeen을 대신 쓰라.
- 위 예제에서는 로버스트 검정들이 모두 큰 $p$값을 내므로 두 집단의 분산에 유의한 차이가 없다고 본다.
- 실무에서 많은 분석가는 F 검정을 아예 건너뛰고 Levene이나 Brown-Forsythe를 기본값으로 쓴다. 정규성 아래에서도 타당하며 검정력 손실이 크지 않기 때문이다.

!!! warning "이 절차 자체가 두 단계 문제를 안고 있다"
    "Shapiro-Wilk로 정규성을 확인한 뒤 검정을 고른다"는 흐름은 15.7절 [분산분석 사전검정](../applications/anova_pretest.md)에서 논한 두 단계 절차와 같은 구조적 문제를 갖는다.

    특히 $n = 8$인 이 예제에서 Shapiro-Wilk의 검정력은 사실상 없다. $p = 0.86$과 $p = 0.93$은 "정규성이 확인되었다"가 아니라 "이 자료로는 어떤 이탈도 탐지할 수 없다"는 뜻이다.

    본문 마지막 항목의 권고("아예 로버스트 검정을 기본값으로")가 실제로 가장 나은 대응이다.

## 연습문제

**연습문제 1.** 위 코드의 두 표본에 Shapiro-Wilk 검정을 적용하라. 그 결과에 근거할 때 F 검정을 쓰는 것이 적절한가? 답을 정당화하라.

??? success "연습문제 1 풀이"

    ```python
    import numpy as np
    from scipy.stats import shapiro

    x1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
    x2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)

    W1, p1 = shapiro(x1)
    W2, p2 = shapiro(x2)
    print(f"x1: W={W1:.4f}, p={p1:.4f}")
    print(f"x2: W={W2:.4f}, p={p2:.4f}")
    ```

    출력:

    ```text
    x1: W=0.9657, p=0.8619
    x2: W=0.9749, p=0.9332
    ```

    두 $p$값이 모두 0.05를 훨씬 넘으므로 어느 표본에서도 정규성을 기각하지 못한다. 형식적으로는 F 검정이 적절해 보인다.

    **그러나 신중해야 한다.** 집단당 관측값이 $n = 8$뿐이므로 Shapiro-Wilk의 비정규성 탐지 검정력이 매우 낮다.

    14장 [Shapiro-Wilk 검정력 모의실험](../../ch14/code/shapiro_power_sim.md)에서 보았듯 $n = 20$인 $t_5$ 자료에서도 검정력이 0.2 수준이다. $n = 8$이면 그보다 훨씬 낮아 사실상 어떤 이탈도 탐지하지 못한다.

    **올바른 결론.** "정규성이 확인되었다"가 아니라 **"이 자료로는 정규성 여부를 판정할 수 없다"**이다. 그러므로 정규성에 의존하지 않는 검정을 쓰는 편이 안전하다. 위 출력에서 세 로버스트 검정의 $p$값이 $0.205$~$0.245$로 서로 비슷하므로, 어느 것을 보고해도 결론이 같다. $\square$

---

**연습문제 2.** $\chi^2(3)$ 분포에서 관측값 50개, $\chi^2(5)$ 분포에서 50개를 생성하라. F 검정, Levene 검정, Fligner-Killeen 검정을 적용하고 결론을 비교하라.

??? success "연습문제 2 풀이"

    ```python
    import numpy as np
    from scipy.stats import levene, fligner, f as fdist

    rng = np.random.default_rng(42)
    x1 = rng.chisquare(df=3, size=50)
    x2 = rng.chisquare(df=5, size=50)

    print(f"sample variances: {np.var(x1, ddof=1):.3f}, "
          f"{np.var(x2, ddof=1):.3f} (true: 6, 10)")

    # F-test
    F = np.var(x1, ddof=1) / np.var(x2, ddof=1)
    p_f = 2 * min(fdist(49, 49).cdf(F), fdist(49, 49).sf(F))
    print(f"F-test:          F={F:.3f}, p={p_f:.4g}")

    # Levene / Brown-Forsythe / Fligner-Killeen
    W, p_l = levene(x1, x2, center='mean')
    print(f"Levene (mean):   W={W:.3f}, p={p_l:.4g}")
    W, p_bf = levene(x1, x2, center='median')
    print(f"Brown-Forsythe:  W={W:.3f}, p={p_bf:.4g}")
    X2, p_fk = fligner(x1, x2)
    print(f"Fligner-Killeen: X2={X2:.3f}, p={p_fk:.4g}")
    ```

    출력:

    ```text
    sample variances: 2.207, 10.648 (true: 6, 10)
    F-test:          F=0.207, p=1.673e-07
    Levene (mean):   W=17.140, p=7.358e-05
    Brown-Forsythe:  W=12.596, p=0.0005955
    Fligner-Killeen: X2=11.175, p=0.0008291
    ```

    참 분산은 $2 \times 3 = 6$과 $2 \times 5 = 10$이므로($\operatorname{Var}(\chi^2(d)) = 2d$) 실제로 다르다. 네 검정 모두 올바르게 기각한다.

    !!! warning "표본분산 2.207은 참값 6에서 크게 벗어나 있다"
        $\chi^2(3)$ 표본의 분산이 $2.207$로 참값 $6$의 **37%에 불과하다.** $\chi^2(3)$의 초과첨도가 $12/3 = 4$로 크므로 $S^2$의 변동이 정규 대비 세 배이다. $n = 50$에서 $S^2$의 상대 표준오차가 $\sqrt{(4+2)/50} = 34.6\%$이니 $-2$ 표준오차 정도의 편차이다.

        그 결과 관측된 분산비 $0.207$이 참 비율 $0.6$보다 훨씬 극단적이고, F 검정의 $p$값 $1.7 \times 10^{-7}$이 로버스트 검정들의 $10^{-4}$ 수준보다 훨씬 작다.

    **$p$값의 순서에 주목하라.** F($1.7\times10^{-7}$) < Levene($7.4\times10^{-5}$) < Brown-Forsythe($6.0\times10^{-4}$) ≈ Fligner-Killeen($8.3\times10^{-4}$). 로버스트할수록 보수적이다.

    **여기서는 결론이 같지만 F 검정의 $p$값은 신뢰할 수 없다.** $\chi^2(3)$은 왜도 $1.63$, 초과첨도 $4$로 강하게 비정규이다. 15.3절 표에 따르면 이런 자료에서 F 검정의 실제 크기는 0.15~0.25이므로, 명목 $p = 1.7\times10^{-7}$이 실제로 나타내는 증거는 훨씬 약하다.

    이탈이 워낙 커서 어느 검정을 쓰든 기각하지만, **보고할 $p$값은 로버스트 검정의 것이어야 한다.** $\square$

---

**연습문제 3.** Levene 검정의 `center='mean'`과 `center='median'`의 차이를 설명하라. 어떤 조건에서 이 선택이 중요해지는가?

??? success "연습문제 3 풀이"

    `center='mean'`이면 절대편차를 $Z_{ij} = |X_{ij} - \bar{X}_i|$로 계산한다. `center='median'`이면 $Z_{ij} = |X_{ij} - \tilde{X}_i|$이며 $\tilde{X}_i$는 집단중앙값이다.

    이 선택은 주로 자료가 **치우쳐 있을 때** 중요하다. 오른쪽으로 치우친 분포에서는 평균이 오른쪽 꼬리 쪽으로 끌려가므로, 꼬리 쪽 관측값의 $Z_{ij}$가 커지고 평균 아래 관측값의 $Z_{ij}$가 작아진다. 이 비대칭이 $Z_{ij}$의 집단내 변동을 부풀려 검정의 보정에 영향을 준다. 중앙값은 치우침에 저항하므로 중앙값 중심 편차가 더 대칭적이고 $H_0$ 아래에서 검정통계량이 더 잘 보정된다.

    대칭 분포(정규 포함)에서는 평균과 중앙값이 가까워 두 판이 비슷하게 작동한다. 정규성 아래에서는 평균 중심 판의 검정력이 약간 높을 수 있다.

    **수치로 본 차이.** 15.8절 [로버스트 분산 검정 비교](robust_tests_comparison.md)의 결과를 정리하면($k=3$, $n=30$, $\alpha=0.05$)

    | 자료 | Levene (평균) | Brown-Forsythe (중앙값) |
    |---|---|---|
    | $\mathcal{N}(0,1)$ | 0.058 | 0.039 |
    | $t_5$ (대칭, 두꺼운 꼬리) | 0.054 | 0.042 |
    | Exponential (치우침) | **0.187** | 0.047 |
    | Lognormal(0,1) (극단 치우침) | **0.261** | 0.030 |

    대칭 분포에서는 두 판의 차이가 미미하지만, 치우친 분포에서는 평균 중심 판이 네 배에서 다섯 배까지 부풀려진다.

    **실무 지침.** 자료가 대칭임을 확신할 수 있으면 평균 중심 판이 검정력에서 조금 유리하다. 확신할 수 없으면 중앙값 중심(SciPy의 기본값)을 쓰라. $\square$

---

**연습문제 4.** Fligner-Killeen 검정은 절대편차에 정규점수 변환을 쓴다. 이 절차를 단계별로 기술하고 왜 로버스트성을 얻는지 설명하라.

??? success "연습문제 4 풀이"

    Fligner-Killeen 절차는

    1. 집단중앙값으로부터의 절대편차를 계산한다. $Z_{ij} = |X_{ij} - \tilde{X}_i|$.
    2. 모든 집단의 $Z_{ij}$ 값에 순위를 매겨 $R_{ij}$를 얻는다.
    3. 순위를 정규점수로 변환한다. $a_{ij} = \Phi^{-1}\!\left(\frac{1 + R_{ij}/(N+1)}{2}\right)$, 여기서 $\Phi^{-1}$은 표준정규 분위수함수이다.
    4. $a_{ij}$ 점수에 분산분석 형태의 검정통계량을 계산한다.

    로버스트성은 순위 변환이 편차의 **크기** 정보를 버리고 상대적 **순서**만 보존하는 데서 나온다. 극단적 이상점은 큰 순위를 받지만 불균형하게 큰 점수를 받지는 않는다. 정규점수 변환이 유계이기 때문이다.

    **정량적으로.** $N$개 편차 중 최대 편차가 받는 점수는

    $$
    a_{\max} = \Phi^{-1}\!\left(\frac{1 + N/(N+1)}{2}\right)
    $$

    이며 $N = 100$이면 $2.58$, $N = 10000$이면 $3.89$이다. **원자료의 편차가 $10^6$이든 $10^{12}$이든 점수는 이 값을 넘지 못한다.**

    반면 Bartlett 검정에서는 이상점 하나가 $S_i^2$을 통해 무제한으로 통계량을 키운다. 유계와 무계의 차이가 로버스트성의 본질이다.

    **다만 만능은 아니다.** 15.8절 비교 페이지에서 보았듯 극단적으로 치우친 자료($\text{Lognormal}(0,1)$)에서는 Fligner-Killeen의 크기가 $0.114$까지 올라가 Brown-Forsythe의 $0.030$보다 나쁘다. 순위 변환이 **대칭인** 두꺼운 꼬리에는 잘 대응하지만, 편차 자체의 강한 치우침은 완전히 제거하지 못하기 때문이다. $\square$

---

**연습문제 5.** 분산분석을 수행하기 전에 등분산을 확인해야 하는 분석가를 위한 작업 흐름을 설계하라. 정규성 확인과 검정 선택의 판정 지점을 포함하라. 이 흐름을 Python 함수로 구현하라.

??? success "연습문제 5 풀이"

    ```python
    import numpy as np
    from scipy.stats import shapiro, levene, fligner

    def check_equal_variances(*groups, alpha=0.05):
        """
        Workflow for checking equality of variances.
        Returns a dict with test results and recommendation.
        """
        # Step 1: Check normality of each group
        normality_ok = True
        shapiro_results = []
        for g in groups:
            W, p = shapiro(g)
            shapiro_results.append((round(W, 4), round(p, 4)))
            if p < alpha:
                normality_ok = False

        # Step 2: Select appropriate test
        if normality_ok:
            test_name = "Levene (mean-centered)"
            stat, pval = levene(*groups, center='mean')
        else:
            test_name = "Brown-Forsythe (median-centered)"
            stat, pval = levene(*groups, center='median')

        # Step 3: Also run Fligner-Killeen as backup
        fk_stat, fk_p = fligner(*groups)

        return {
            "normality_assumed": normality_ok,
            "shapiro_results": shapiro_results,
            "primary_test": test_name,
            "statistic": round(stat, 4),
            "p_value": pval,
            "fligner_killeen_p": fk_p,
            "equal_variances": pval >= alpha,
        }

    # Example usage
    rng = np.random.default_rng(0)
    g1 = rng.normal(0, 1, 30)
    g2 = rng.normal(0, 1.5, 30)
    g3 = rng.normal(0, 2, 30)
    result = check_equal_variances(g1, g2, g3)
    for k, v in result.items():
        print(f"{k}: {v}")
    ```

    출력:

    ```text
    normality_assumed: True
    shapiro_results: [(0.9752, 0.6879), (0.9578, 0.2725), (0.9695, 0.5253)]
    primary_test: Levene (mean-centered)
    statistic: 14.4774
    p_value: 3.736985145892101e-06
    fligner_killeen_p: 1.5741483418264177e-05
    equal_variances: False
    ```

    자료를 표준편차 $1, 1.5, 2$로 생성했으므로 등분산이 아니고, 함수가 이를 올바르게 탐지한다($p = 3.7 \times 10^{-6}$). 세 집단 모두 정규성 검정을 통과하여 평균 중심 Levene이 선택되었다. Fligner-Killeen도 $p = 1.6 \times 10^{-5}$로 같은 결론이다.

    !!! warning "이 흐름의 근본적 한계"
        위 함수는 요청받은 대로 구현되었지만, 몇 가지 문제를 안고 있다.

        1. **두 단계 결정.** 정규성 검정 결과로 분산 검정을 고르므로 결합된 절차의 크기가 통제되지 않는다.
        2. **다중 정규성 검정.** 집단이 $k$개면 Shapiro-Wilk를 $k$번 수행한다. $k = 3$, $\alpha = 0.05$이면 정규 자료에서도 적어도 하나가 기각될 확률이 $1 - 0.95^3 = 14\%$이다. 14장에서 논한 다중검정 문제이다.
        3. **`equal_variances` 필드가 오도한다.** $p \geq \alpha$를 "분산이 같다"로 반환하는데, 이는 기각 실패를 증명으로 오독하는 것이다. 특히 검정력이 낮은 작은 표본에서 위험하다.

        **개선안.**

        ```python
        def describe_variances(*groups):
            """Report, don't decide."""
            import numpy as np
            from scipy.stats import levene, fligner
            return {
                "n": [len(g) for g in groups],
                "sd": [round(float(np.std(g, ddof=1)), 4) for g in groups],
                "sd_ratio_max_min": round(
                    max(np.std(g, ddof=1) for g in groups) /
                    min(np.std(g, ddof=1) for g in groups), 3),
                "brown_forsythe_p": levene(*groups, center='median')[1],
                "fligner_killeen_p": fligner(*groups)[1],
            }
        ```

        이 판은 (1) 검정을 고르지 않고 Brown-Forsythe를 고정으로 쓰며, (2) 이분법적 판정을 반환하지 않고 집단별 표준편차와 그 비율을 함께 제시하여 독자가 판단하게 한다.

        후속 분석은 이 결과와 무관하게 Welch 분산분석을 쓰면 되므로, 애초에 판정이 필요 없다. $\square$
