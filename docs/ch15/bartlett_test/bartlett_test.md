# 분산의 동일성에 대한 Bartlett 검정


Bartlett 검정은 여러 집단에 걸친 분산의 동일성을 평가한다. 정규성 이탈에 매우 민감하므로 자료가 정규분포를 따를 때 가장 적절하다. 정규성 가정 아래에서는 강력하지만 자료가 비정규이면 잘못된 결론으로 이어질 수 있다.

## 가설

**귀무가설 ($H_0$):** 모든 집단의 분산이 같다.

$$
H_0: \sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2
$$

**대립가설 ($H_1$):** 적어도 한 집단의 분산이 다른 집단과 다르다.

$$
H_1: \sigma_i^2 \neq \sigma_j^2 \quad \text{(적어도 한 쌍의)} \quad i \neq j
$$

## 가정

1. **정규성:** Bartlett 검정은 각 집단의 자료가 정규분포를 따른다고 가정한다. 검정이 정규성 이탈에 매우 민감하므로 결정적이다.
2. **독립성:** 관측값이 집단 안에서도 집단 사이에서도 독립이어야 한다.
3. **확률표집:** 자료가 확률표본에서 나와야 한다.

## 검정통계량

Bartlett 검정의 검정통계량은 합동분산과 개별 표본분산의 비에서 유도된다.

$$
T = \frac{(N - k) \ln(S_p^2) - \sum_{i=1}^k (n_i - 1) \ln(S_i^2)}{1 + \frac{1}{3(k - 1)} \left( \sum_{i=1}^k \frac{1}{n_i - 1} - \frac{1}{N - k} \right)} \sim \chi^2_{k-1}
$$

여기서

- $N$은 전체 관측값의 수,
- $k$는 집단의 수,
- $S_p^2$은 합동분산이다.

$$
S_p^2 = \frac{\sum_{i=1}^k (n_i - 1) S_i^2}{N - k}
$$

- $S_i^2$은 집단 $i$의 표본분산이다.

검정통계량 $T$는 자유도 $k - 1$인 카이제곱분포를 따른다.

분자는 항상 음이 아니라는 점에 주목하라. 산술평균-기하평균 부등식에 의해 합동분산(가중 산술평균)이 개별 분산의 가중 기하평균보다 크거나 같기 때문이다. 등호는 모든 $S_i^2$이 같을 때만 성립하며, 그때 $T = 0$이다.

## 판정규칙

$$
T > \chi^2_{\text{critical}} \quad \Rightarrow \quad H_0 \text{ 기각}
$$

$k = 3$이고 $\alpha = 0.05$이면 $\chi^2_{0.95, 2} = 5.991$이다.

## 한계

Bartlett 검정은 정규성 가정의 위반에 로버스트하지 않다. 자료가 정규분포를 따르지 않으면 Levene 검정이나 Brown-Forsythe 검정 같은 대안을 고려해야 한다. 얼마나 심각한지는 15.4절의 [비정규성 아래의 한계](limitations.md)에서 수치로 다룬다.

## Python 구현

### SciPy 이용

```python
import numpy as np
from scipy.stats import bartlett

# Example data
group1 = [12, 15, 14, 10, 13, 14, 12, 11]
group2 = [22, 25, 20, 18, 24, 23, 19, 21]
group3 = [32, 35, 34, 30, 33, 34, 32, 31]

# Perform Bartlett's test
statistic, p_value = bartlett(group1, group2, group3)

# Output the results
print(f"Bartlett's test statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
alpha = 0.05
if p_value < alpha:
    print("Reject H0: variances are significantly different.")
else:
    print("Fail to reject H0: no significant difference in variances.")
```

출력:

```text
Bartlett's test statistic: 1.3070
P-value: 0.5202
Fail to reject H0: no significant difference in variances.
```

세 집단의 표본분산은 $2.839$, $6.000$, $2.839$이다. 최대·최소 비가 $2.1$로 꽤 크지만 각 집단 $n = 8$로는 유의하지 않다.

### 직접 계산

```python
import numpy as np
from scipy import stats

# Example data
group1 = np.array([12, 15, 14, 10, 13, 14, 12, 11])
group2 = np.array([22, 25, 20, 18, 24, 23, 19, 21])
group3 = np.array([32, 35, 34, 30, 33, 34, 32, 31])

# Step 1: Calculate variances for each group
variance_group1 = group1.var(ddof=1)
variance_group2 = group2.var(ddof=1)
variance_group3 = group3.var(ddof=1)

# Step 2: Calculate sample sizes
sample_size1 = group1.size
sample_size2 = group2.size
sample_size3 = group3.size

# Step 3: Calculate pooled variance
total_sample_size = sample_size1 + sample_size2 + sample_size3
number_of_groups = 3
degrees_of_freedom_pooled = total_sample_size - number_of_groups

pooled_variance = (
    ((sample_size1 - 1) * variance_group1) +
    ((sample_size2 - 1) * variance_group2) +
    ((sample_size3 - 1) * variance_group3)
) / degrees_of_freedom_pooled

# Step 4: Calculate the numerator (logarithmic terms)
numerator = (
    (total_sample_size - number_of_groups) * np.log(pooled_variance) -
    ((sample_size1 - 1) * np.log(variance_group1) +
     (sample_size2 - 1) * np.log(variance_group2) +
     (sample_size3 - 1) * np.log(variance_group3))
)

# Step 5: Calculate the denominator (correction term)
correction_term = (
    (1 / (sample_size1 - 1)) +
    (1 / (sample_size2 - 1)) +
    (1 / (sample_size3 - 1)) -
    (1 / degrees_of_freedom_pooled)
)
denominator = 1 + correction_term / (3 * (number_of_groups - 1))

# Step 6: Calculate Bartlett's test statistic
bartlett_statistic = numerator / denominator

# Step 7: Calculate p-value using chi-square distribution
degrees_of_freedom = number_of_groups - 1
p_value = stats.chi2.sf(bartlett_statistic, degrees_of_freedom)

# Display results
print(f"Pooled variance: {pooled_variance:.4f}")
print(f"Numerator: {numerator:.4f}, Correction denominator: {denominator:.4f}")
print(f"Bartlett's Test Statistic (T): {bartlett_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
```

출력:

```text
Pooled variance: 3.8929
Numerator: 1.3900, Correction denominator: 1.0635
Bartlett's Test Statistic (T): 1.3070
P-value: 0.5202
```

SciPy 결과와 정확히 일치한다. 보정인자가 $1.0635$로 통계량을 약 6% 줄인다는 점도 확인할 수 있다.


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
Bartlett 통계량의 분자가 항상 음이 아님을 산술평균-기하평균 부등식으로 증명하라. 등호 조건은 무엇인가?

</div>

??? success "풀이"
    분자를 $w_i = (n_i-1)/(N-k)$로 정규화된 가중치로 다시 쓰자. $\sum_i w_i = 1$이고

    $$
    \text{분자} = (N-k)\left[\ln S_p^2 - \sum_i w_i \ln S_i^2\right].
    $$

    한편 합동분산은 가중 **산술**평균이다.

    $$
    S_p^2 = \sum_i w_i S_i^2.
    $$

    그리고 $\sum_i w_i \ln S_i^2 = \ln \prod_i (S_i^2)^{w_i}$는 가중 **기하**평균의 로그이다.

    가중 산술평균-기하평균 부등식에 의해

    $$
    \sum_i w_i S_i^2 \geq \prod_i (S_i^2)^{w_i},
    $$

    양변에 로그를 취하면(로그가 증가함수이므로)

    $$
    \ln S_p^2 \geq \sum_i w_i \ln S_i^2 \implies \text{분자} \geq 0.
    $$

    **등호 조건.** 가중 산술평균-기하평균 부등식의 등호는 모든 항이 같을 때, 곧 $S_1^2 = S_2^2 = \cdots = S_k^2$일 때만 성립한다. 이때 $T = 0$이다.

    **개념적 의미.** Bartlett 통계량은 표본분산들의 **산술평균과 기하평균의 로그 차이**를 잰다. 두 평균의 차이는 자료의 산포가 클수록 커지므로, 이 통계량이 분산들의 "흩어짐"에 대한 자연스러운 척도가 된다. 값이 클수록 분산이 서로 다르다는 증거이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
보정인자 $C$가 항상 1보다 큼을 보이고, 집단 크기가 커질수록 1에 가까워짐을 확인하라. 이 보정이 필요한 이유를 설명하라.

</div>

??? success "풀이"
    보정인자는

    $$
    C = 1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^k \frac{1}{n_i-1} - \frac{1}{N-k}\right).
    $$

    **$C > 1$의 증명.** $N - k = \sum_i (n_i - 1)$이므로, $m_i = n_i - 1$이라 쓰면 괄호 안은

    $$
    \sum_i \frac{1}{m_i} - \frac{1}{\sum_i m_i}.
    $$

    $k \geq 2$이면 $\sum_i m_i > m_j$가 모든 $j$에 대해 성립하므로 $\frac{1}{\sum_i m_i} < \frac{1}{m_1} \leq \sum_i \frac{1}{m_i}$이다. 따라서 괄호 안이 양수이고 $C > 1$이다.

    **극한.** 모든 $n_i = n$이면

    $$
    C = 1 + \frac{1}{3(k-1)}\left(\frac{k}{n-1} - \frac{1}{k(n-1)}\right) = 1 + \frac{k^2-1}{3k(k-1)(n-1)} = 1 + \frac{k+1}{3k(n-1)}.
    $$

    $k = 3$일 때 $C = 1 + \frac{4}{9(n-1)}$이다.

    | $n$ | 8 | 20 | 50 | 200 |
    |---|---|---|---|---|
    | $C$ | 1.0635 | 1.0234 | 1.0091 | 1.0022 |

    본문 예제는 $k = 3$, $n = 8$이므로 $C = 1 + \frac{4}{9 \times 7} = 1.0635$이며, 직접 계산 결과와 일치한다.

    **왜 필요한가.** 보정하지 않은 통계량 $-2\ln\Lambda$는 **점근적으로만** $\chi^2_{k-1}$을 따른다. 유한표본에서는 그 평균이 $k-1$보다 크다. Bartlett은 평균이 정확히 $k-1$이 되도록 $C$로 나누는 보정을 찾아냈다.

    $C > 1$이므로 보정은 항상 통계량을 **줄인다**. 곧 보정하지 않으면 검정이 지나치게 자주 기각한다. $n = 8$에서 6% 과다기각을 바로잡는 것이다.

    이 아이디어는 "Bartlett 보정"이라는 이름으로 가능도비 검정 일반에 확장되었으며, 유한표본 근사를 개선하는 표준 기법이 되었다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
본문 예제의 자료에 Bartlett, Levene, Brown-Forsythe, Fligner-Killeen 검정을 모두 적용하고 결과를 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11])
    g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21])
    g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31])

    print("variances:", [round(g.var(ddof=1), 4) for g in (g1, g2, g3)])
    print(f"Bartlett:        stat = {stats.bartlett(g1, g2, g3)[0]:.4f}, "
          f"p = {stats.bartlett(g1, g2, g3)[1]:.4f}")
    lev = stats.levene(g1, g2, g3, center='mean')
    bf = stats.levene(g1, g2, g3, center='median')
    fk = stats.fligner(g1, g2, g3)
    print(f"Levene (mean):   stat = {lev[0]:.4f}, p = {lev[1]:.4f}")
    print(f"Brown-Forsythe:  stat = {bf[0]:.4f}, p = {bf[1]:.4f}")
    print(f"Fligner-Killeen: stat = {fk[0]:.4f}, p = {fk[1]:.4f}")
    ```

    출력:

    ```text
    variances: [2.8393, 6.0, 2.8393]
    Bartlett:        stat = 1.3070, p = 0.5202
    Levene (mean):   stat = 1.1218, p = 0.3444
    Brown-Forsythe:  stat = 1.1076, p = 0.3489
    Fligner-Killeen: stat = 2.5507, p = 0.2793
    ```

    네 검정 모두 기각하지 않는다. 결론이 일치한다.

    다만 검정통계량이 서로 다른 분포를 참조한다는 점에 유의하라. Bartlett과 Fligner-Killeen은 $\chi^2_2$를, Levene과 Brown-Forsythe는 $F_{2,21}$을 쓴다. 그래서 통계량 값을 직접 비교하는 것은 의미가 없고 $p$값만 비교해야 한다.

    $p$값이 $0.28$에서 $0.52$까지 흩어져 있지만 모두 0.05보다 훨씬 크므로 실무적으로는 같은 결론이다. **검정들의 결론이 갈릴 때만 어느 것을 신뢰할지 고민하면 된다.** 그때는 자료의 정규성 여부가 판단 기준이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
정규 자료에서 Bartlett 검정이 로버스트 검정들보다 검정력이 높은지 모의실험으로 확인하라. 세 집단, 각 $n = 20$, 표준편차 조합을 바꿔가며 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5)
    R, alpha, n = 5000, 0.05, 20

    print(f"{'sds':>16} {'Bartlett':>9} {'Levene':>8} {'BF':>8} {'FK':>8}")
    for sds in [(1, 1, 1), (1, 1, 1.5), (1, 1, 2), (1, 1.5, 2)]:
        cb = cl = cbf = cf = 0
        for _ in range(R):
            g = [rng.normal(0, s, n) for s in sds]
            cb += stats.bartlett(*g)[1] < alpha
            cl += stats.levene(*g, center='mean')[1] < alpha
            cbf += stats.levene(*g, center='median')[1] < alpha
            cf += stats.fligner(*g)[1] < alpha
        print(f"{str(sds):>16} {cb/R:>9.3f} {cl/R:>8.3f} "
              f"{cbf/R:>8.3f} {cf/R:>8.3f}")
    ```

    출력:

    ```text
                 sds  Bartlett   Levene       BF       FK
           (1, 1, 1)     0.046    0.056    0.037    0.033
         (1, 1, 1.5)     0.435    0.388    0.326    0.293
           (1, 1, 2)     0.875    0.816    0.766    0.700
         (1, 1.5, 2)     0.759    0.661    0.581    0.537
    ```

    **첫 줄(크기).** 모두 0.05 근처이다. 정규 자료에서는 네 검정 모두 크기가 올바르다. Brown-Forsythe(0.037)와 Fligner-Killeen(0.033)이 다소 보수적이다.

    **나머지 줄(검정력).** 정규 자료에서 **Bartlett이 언제나 가장 강력하다**. 순서는 항상 Bartlett > Levene > Brown-Forsythe > Fligner-Killeen이다.

    | 설정 | Bartlett | BF | 검정력 손실 |
    |---|---|---|---|
    | $(1,1,1.5)$ | 0.435 | 0.326 | $-25\%$ |
    | $(1,1,2)$ | 0.875 | 0.766 | $-12\%$ |
    | $(1,1.5,2)$ | 0.759 | 0.581 | $-23\%$ |

    Brown-Forsythe를 쓰면 정규 자료에서 검정력의 12~25%를 잃는다. 적지 않은 대가이다.

    **그러나 이것이 Bartlett을 권하는 근거가 되지는 않는다.** 15.1절에서 보았듯 대수정규 자료에서 Bartlett의 크기는 0.62이다. 정규성이 확실할 때 얻는 20%의 검정력 이득과, 정규성이 깨졌을 때 겪는 12배의 크기 팽창을 견주면 후자가 압도적으로 크다.

    **Bartlett 검정이 정당한 경우는 정규성이 이론적으로 보장되는 상황(예: 측정오차가 정규임이 확립된 계측 자료)뿐이다.** 그 외에는 로버스트 검정의 검정력 손실을 보험료로 받아들이는 편이 낫다. $\square$

---

## 정리하며

바틀렛 검정은 **$k$ 개 집단**의 등분산을 한 번에 검정한다.

- **$F$ 검정의 다집단 확장이다.** $k=2$ 면 $F$ 검정과 비슷한 역할을 한다.
- **정규성 아래에서 가장 강력하다.** 가능도비에서 유도되며, 자료가 정말 정규라면 이 장의 어떤 검정보다 예민하다.
- **그 대가가 취약성이다.** **$F$ 검정보다도 비정규성에 민감하며**, 이 장에서 가장 취약한 검정이다.
- **합동분산과 각 집단 분산의 로그 차이**를 재는 구조이며, 다음 절에서 유도한다.
- **표본크기가 다르면 가중이 들어간다.** 각 집단의 자유도로 가중하므로 큰 집단의 분산이 더 큰 영향을 준다.

다음 절 **유도**로 넘어간다.
