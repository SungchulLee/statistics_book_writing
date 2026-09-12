# 이표본 비모수 검정


## 1. Mann-Whitney U 검정 (Wilcoxon 순위합검정)

**Mann-Whitney U 검정**은 **Wilcoxon 순위합검정**이라고도 하며, 독립인 두 집단의 분포를 비교하는 비모수 통계 검정이다. 독립 이표본 t 검정 같은 모수적 검정의 가정이 충족되지 않을 때(예: 비정규성이나 순서형 자료) 특히 유용하다.

### 주요 특징

- **목적**: 독립인 두 집단의 분포가 같은지, 또는 한 집단이 다른 집단보다 큰(작은) 값을 갖는 경향이 있는지 검정한다.
- **가정**:
    - 두 집단이 서로 독립이다.
    - 관측값이 순서형, 구간형, 비율형이다(정규성은 필요 없다).
    - 두 표본이 무작위로 추출되었다.
- **귀무가설 ($H_0$)**: 두 집단이 같은 분포를 갖는다.
- **대립가설 ($H_1$)**: 분포가 다르거나 한 집단이 더 높은 값을 갖는 경향이 있다.

### 검정 원리

**1단계:** 두 집단의 자료를 모두 합쳐 값에 **순위**를 배정한다. 가장 작은 값이 순위 1, 두 번째가 순위 2 등이다. 동점이 있으면 동점 순위들의 평균을 배정한다.

**2단계:** 각 집단의 순위합을 계산한다.

- $R_1$: 집단 1의 순위합
- $R_2$: 집단 2의 순위합

**3단계:** 각 집단의 **U 통계량**을 계산한다.

$$U_1 = R_1 - \frac{n_1 (n_1 + 1)}{2}$$

$$U_2 = R_2 - \frac{n_2 (n_2 + 1)}{2}$$

여기서 $n_1$과 $n_2$는 집단 1과 2의 표본크기이다.

!!! warning "$U$의 정의는 교과서마다 다르다"
    어떤 문헌은 $U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$로 정의한다. 이는 위 정의의 $U_2$와 같다($n_1 n_2 - U_1 = U_2$). 두 관례 모두 $U = \min(U_1, U_2)$가 같으므로 **검정 결과에는 영향이 없다**.

    그러나 효과크기를 보고할 때는 차이가 난다. 이 책에서는 일관되게

    $$
    U_1 = R_1 - \frac{n_1(n_1+1)}{2} = \#\{(i,j) : X_i > Y_j\}
    $$

    를 쓰며, 따라서 $U_1/(n_1 n_2)$가 $P(X > Y)$의 추정값이다. 반대 관례를 쓰면 $P(X < Y)$가 된다. 어느 쪽을 썼는지 밝히지 않으면 방향이 뒤집힌 결론이 나온다.

**4단계:** Mann-Whitney U 통계량은

$$U = \min(U_1, U_2)$$

**5단계:** 유의성을 판정한다.

- 표본이 크면($n_1, n_2 > 20$) **정규근사**로 Z 점수를 쓴다.
- 표본이 작으면 $U$의 정확분포를 쓴다.

### 해석

- $p$값이 작으면(예: $p < 0.05$): $H_0$을 기각한다. 두 집단의 분포가 유의하게 다르다.
- $p$값이 크면: $H_0$을 기각하지 못한다. 차이의 증거가 없다.

### H_0이 기각되면 어느 집단이 더 큰가

각 집단의 **평균순위**를 비교한다. 평균순위가 높은 집단의 값이 큰 경향이 있다.

### 파이썬 구현

<div class="codebox" markdown>

**예제 1.** Mann-Whitney U 검정

```python
import numpy as np
from scipy.stats import mannwhitneyu, rankdata

# 두 교수법의 시험 점수를 견준다.
group_a = np.array([85, 78, 92, 88, 76, 95, 89, 82, 91, 87])
group_b = np.array([72, 68, 81, 75, 70, 77, 74, 69, 73, 71])

# Mann-Whitney U 는 이표본 t 검정의 비모수 대응이다. 귀무가설이 "두 평균이
# 같다"가 아니라 "무작위로 뽑은 A 가 B 보다 클 확률이 1/2"이라는 점에 주의한다.
stat, p_value = mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"U statistic: {stat}")        # 97.0
print(f"P-value: {p_value:.6f}")     # 0.000440  (asymptotic)

# 표본이 작고 동점이 없으면 정확분포를 쓸 수 있다. 근사값과 꽤 갈린다.
print(mannwhitneyu(group_a, group_b, method='exact').pvalue)   # 7.58e-05

# 유의하다는 결론이 났을 때 어느 쪽이 큰지는 평균순위로 말한다.
# 동점에 중간순위를 주려면 argsort 가 아니라 rankdata 를 써야 한다.
combined = np.concatenate([group_a, group_b])
ranks = rankdata(combined)
mean_rank_a = ranks[:len(group_a)].mean()
mean_rank_b = ranks[len(group_a):].mean()
print(f"Mean rank Group A: {mean_rank_a:.1f}")   # 15.2
print(f"Mean rank Group B: {mean_rank_b:.1f}")   # 5.8

alpha = 0.05
if p_value < alpha:
    print("Reject H0: Significant difference between groups.")
    if mean_rank_a > mean_rank_b:
        print("Group A tends to have larger values.")
    else:
        print("Group B tends to have larger values.")
else:
    print("Fail to reject H0.")
```

출력:

```
U statistic: 97.0
P-value: 0.000440
7.577561757128321e-05
Mean rank Group A: 15.2
Mean rank Group B: 5.8
Reject H0: Significant difference between groups.
Group A tends to have larger values.
```

</div>

!!! warning "`argsort(argsort(x))`는 동점을 처리하지 못한다"
    순위를 구할 때 `np.argsort(np.argsort(x)) + 1`을 쓰는 코드를 흔히 본다. 동점이 없으면 맞지만, 동점이 있으면 **중간순위 대신 임의의 순서**를 배정한다.

    ```python
    import numpy as np
    from scipy.stats import rankdata
    x = np.array([3, 1, 3, 2])
    print(np.argsort(np.argsort(x)) + 1)   # [3 1 4 2]  ← 두 3에 3과 4를 배정
    print(rankdata(x))                     # [3.5 1.  3.5 2. ]  ← 올바른 중간순위
    ```

    출력:

    ```
    [3 1 4 2]
    [3.5 1.  3.5 2. ]
    ```

    항상 `scipy.stats.rankdata`를 쓴다.

### 동치성에 관한 참고

Mann-Whitney U 검정과 Wilcoxon 순위합검정은 **통계적으로 동치**이다. 용어의 선택은 소프트웨어에 따라 다른 경우가 많다.

- **SPSS**: "Mann-Whitney U test"
- **R**: "Wilcoxon rank-sum test" (`wilcox.test`)
- **Python scipy**: `mannwhitneyu` 또는 `ranksums`

---

## 2. Mood 중앙값검정

**Mood 중앙값검정**은 둘 이상 집단의 중앙값을 비교하는 비모수 가설검정이다. 평균이나 분산이 아니라 오직 중앙값에 집중하므로 자료가 비정규이거나 순서형이거나 이상치를 포함할 때 특히 유용하다.

### 주요 특징

- **비모수**: 분포에 대한 가정이 필요 없다.
- **이상치에 로버스트**: 중앙값에 집중한다.
- **둘 이상의 집단에 적용 가능**.
- **귀무가설 ($H_0$)**: 모든 집단의 중앙값이 같다.
- **대립가설 ($H_1$)**: 적어도 한 집단의 중앙값이 다르다.

### 가정

1. 표본들이 독립이다.
2. 자료가 연속형 또는 순서형이다.
3. 각 집단이 모집단에서 뽑은 확률표본이다.

### 작동 방식

**1단계:** 합친 자료 전체의 중앙값을 계산한다.

**2단계:** 각 집단에서 자료점을 전체 중앙값보다 "위" 또는 "아래"로 분류한다.

**3단계:** 각 집단의 중앙값 위·아래 도수를 담은 분할표를 만든다.

**4단계:** 분할표에 카이제곱 독립성 검정을 적용한다.

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

여기서 $O$는 관측도수, $E$는 귀무가설 아래의 기대도수이다.

**5단계:** $p$값으로 유의성을 판정한다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 두 집단 중앙값 비교. **상황**: 두 학생 집단의 중앙값 시험점수를 비교한다.

- 집단 A: $[50, 55, 60, 65, 70]$
- 집단 B: $[45, 50, 55, 60, 65]$

</div>

### 파이썬 구현

<div class="codebox" markdown>

**예제 2.** Mood의 중앙값 검정

```python
import numpy as np
from scipy.stats import chi2_contingency

def moods_median_test(*groups, correction=True):
    """Mood 의 중앙값 검정. 여러 집단의 중앙값을 견준다.

    자료 전체의 중앙값을 구한 뒤, 집단마다 그보다 큰 값과 작은 값의 개수를
    세어 분할표를 만들고 카이제곱 검정을 돌린다. 값을 "위/아래" 둘로만
    나누므로 순위검정보다도 정보를 더 버리고, 그만큼 검정력이 낮다.
    대신 가정이 거의 없어 아주 거친 자료에도 쓸 수 있다.

    매개변수
    --------
    groups : 집단을 나타내는 배열 여럿
    correction : Yates 연속성 보정 여부 (2x2 표에서만 뜻이 있다)

    돌려주는 값
    ----------
    chi2_stat, p_value, contingency_table
    """
    # 전체 중앙값을 기준선으로 삼는다.
    combined_data = np.concatenate(groups)
    overall_median = np.median(combined_data)

    contingency_table = []
    for group in groups:
        # 중앙값과 정확히 같은 값은 어느 쪽에도 넣지 않는다.
        above_median = np.sum(group > overall_median)
        below_median = np.sum(group < overall_median)
        contingency_table.append([above_median, below_median])

    contingency_table = np.array(contingency_table).T
    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table,
                                                correction=correction)

    return chi2_stat, p_value, contingency_table

# 예시 자료
group_a = np.array([50, 55, 60, 65, 70])
group_b = np.array([45, 50, 55, 60, 65])

# 보정 여부에 따라 p-값이 꽤 갈린다. 2x2 표에서 표본이 작을 때 그렇다.
for corr in (True, False):
    chi2_stat, p_value, table = moods_median_test(group_a, group_b,
                                                  correction=corr)
    print(f"correction={corr}: chi2={chi2_stat:.4f}, p={p_value:.4f}")
print(f"Contingency Table:\n{table}")
```

출력:

```
correction=True: chi2=0.0000, p=1.0000
correction=False: chi2=0.4000, p=0.5271
Contingency Table:
[[3 2]
 [2 3]]
```

</div>

**출력:**

```
correction=True:  chi2=0.0000, p=1.0000
correction=False: chi2=0.4000, p=0.5271
Contingency Table:
[[3 2]
 [2 3]]
```

!!! warning "SciPy는 $2\times2$ 표에 Yates 보정을 기본으로 적용한다"
    `chi2_contingency`의 `correction` 인자는 $2 \times 2$ 표에서 **기본값이 `True`**이다. 위 자료에서 Yates 보정을 적용하면 $\chi^2 = 0$, $p = 1$이 되고, 적용하지 않으면 $\chi^2 = 0.4$, $p = 0.527$이 된다.

    보정이 이렇게 극단적으로 작동하는 이유는 관측도수가 $(3, 2, 2, 3)$이고 기대도수가 모두 $2.5$여서, 각 칸의 편차 $|O - E| = 0.5$가 Yates 보정량 $0.5$와 정확히 같기 때문이다. 보정 후 분자가 $0$이 된다.

    `scipy.stats.median_test`도 기본적으로 보정을 적용하여 $p = 1.0$을 반환한다. 어느 쪽이든 이 자료에서는 기각하지 않으므로 결론은 같다.

**해석**: 두 경우 모두 $p$값이 $0.05$보다 크므로 귀무가설을 기각하지 못한다. 중앙값에 유의한 차이가 없다.

### 장점과 한계

| 장점 | 한계 |
|---|---|
| 이상치에 로버스트하다 | 집단 내 변동을 무시한다 |
| 비모수적이다 | 모수적 검정에 비해 검정력이 낮다 |
| 구현이 간단하다 | 독립성을 요구한다 |
| 여러 집단에 쓸 수 있다 | 중앙값과 같은 값을 버린다 |

---

## 3. Kruskal-Wallis 검정

**Kruskal-Wallis 검정**은 일원분산분석의 비모수 확장으로, 독립인 세 집단 이상의 분포를 비교하는 데 쓰인다.

### 주요 특징

- **목적**: 셋 이상 집단의 중앙값(또는 분포)이 다른지 검정한다.
- **귀무가설**: 모든 집단이 같은 분포를 갖는다.
- **대립가설**: 적어도 한 집단이 다르다.
- **관계**: 집단이 둘뿐이면 Kruskal-Wallis 검정은 Mann-Whitney U 검정과 동치이다.

### 검정통계량

$$H = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)$$

여기서 $N$은 전체 관측값 개수, $k$는 집단 수, $n_i$는 집단 $i$의 크기, $R_i$는 집단 $i$의 순위합이다.

$H_0$ 아래에서 $H$는 근사적으로 자유도 $k - 1$인 $\chi^2$ 분포를 따른다.

### 파이썬 구현

<div class="codebox" markdown>

**예제 3.** Kruskal-Wallis 검정

```python
from scipy.stats import kruskal

group_a = [85, 78, 92, 88, 76]
group_b = [72, 68, 81, 75, 70]
group_c = [90, 95, 88, 92, 87]

# Kruskal-Wallis 는 일원배치 분산분석의 비모수 대응이다. 집단이 셋 이상일
# 때 쓰며, Mann-Whitney U 를 여러 집단으로 넓힌 것이라고 보면 된다.
stat, p_value = kruskal(group_a, group_b, group_c)

print(f"H statistic: {stat:.4f}")   # 9.4136
print(f"P-value: {p_value:.4f}")    # 0.0090

alpha = 0.05
if p_value < alpha:
    print("Reject H0: At least one group differs significantly.")
else:
    print("Fail to reject H0.")
```

출력:

```
H statistic: 9.4136
P-value: 0.0090
Reject H0: At least one group differs significantly.
```

</div>

### 사후검정

Kruskal-Wallis 검정이 유의하면, 쌍별 Mann-Whitney U 검정에 다중비교 보정(예: Bonferroni 보정)을 적용하여 구체적으로 어느 집단이 다른지 판정한다.

---

## 비교: 이표본 모수적 검정 대 비모수 검정

| 특징 | 이표본 t 검정 | Mann-Whitney U | Mood 중앙값 |
|---|---|---|---|
| **가정** | 정규성, 등분산 | 독립성, 연속성 | 독립성 |
| **검정 대상** | 평균 | 분포 / 확률적 순서 | 중앙값 |
| **검정력** | 최고 (정규일 때) | 높음 (ARE $\approx 0.955$) | 낮음 |
| **로버스트성** | 이상치에 민감 | 로버스트 | 매우 로버스트 |
| **다집단** | 분산분석 | Kruskal-Wallis | 그대로 확장 가능 |

### 선택 지침

- **정규자료, 등분산**: **이표본 t 검정**(분산이 다르면 Welch t 검정).
- **비정규 연속자료**: **Mann-Whitney U 검정**.
- **극단적인 이상치가 있는 자료**: **Mood 중앙값검정**.
- **셋 이상의 집단**: **Kruskal-Wallis**(비모수) 또는 **분산분석**(모수적).


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Mood 중앙값검정과 Mann-Whitney U 검정을 같은 자료에 적용하면 결과가 어떻게 다른가? 본문의 집단 A $= [50, 55, 60, 65, 70]$, 집단 B $= [45, 50, 55, 60, 65]$로 확인하고, 두 검정이 버리는 정보를 각각 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy.stats import mannwhitneyu, median_test
    a = np.array([50, 55, 60, 65, 70]); b = np.array([45, 50, 55, 60, 65])
    print(mannwhitneyu(a, b, method='exact'))    # U=17.0, p=0.4206
    print(median_test(a, b))                     # stat=0.0, p=1.0
    ```

    출력:

    ```
    MannwhitneyuResult(statistic=17.0, pvalue=0.42063492063492064)
    MedianTestResult(statistic=0.0, pvalue=1.0, median=57.5, table=array([[3, 2],
           [2, 3]]))
    ```

    | 검정 | 통계량 | $p$값 |
    |:---|---:|---:|
    | Mann-Whitney U | $U_1 = 17$ | $0.421$ |
    | Mood 중앙값 (Yates 보정) | $\chi^2 = 0$ | $1.000$ |
    | Mood 중앙값 (보정 없음) | $\chi^2 = 0.4$ | $0.527$ |

    두 검정 모두 기각하지 않지만 $p$값 차이가 크다.

    **Mood 중앙값검정이 버리는 것.** 각 관측값을 "전체 중앙값 위/아래"라는 이진값으로 뭉갠다. 집단 A의 $70$과 $60$이 똑같이 "위"로 처리된다. $25$개 자료점의 순서 정보 대부분이 사라지고 $2 \times 2$ 표만 남는다.

    **Mann-Whitney가 버리는 것.** 값 사이의 **거리**를 버린다. A의 $70$이 B의 $65$보다 5만큼 큰지 $500$만큼 큰지 구별하지 않는다.

    정보를 더 많이 버리는 쪽이 Mood 중앙값검정이며, 그만큼 검정력이 낮다. 실제로 Mood 중앙값검정의 정규분포 아래 ARE는 부호검정과 같은 $2/\pi \approx 0.637$로, Mann-Whitney의 $0.955$보다 훨씬 낮다.

    **그렇다면 Mood 중앙값검정을 언제 쓰는가?** 두 경우이다. (1) 분포의 **모양이 크게 다를** 때. Mann-Whitney는 모양이 다르면 중앙값이 같아도 기각할 수 있지만, Mood 검정은 중앙값만 본다. (2) 자료가 심하게 절단·중도절단되어 순위조차 신뢰할 수 없을 때.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 Mood 중앙값검정 구현은 전체 중앙값과 **정확히 같은** 관측값을 버린다. 이것이 문제가 되는 상황을 만들고, SciPy `median_test`의 `ties` 인자와 비교하라.

</div>

??? success "풀이"
    관측값이 이산적이면 중앙값과 같은 값이 여러 개 생긴다.

    ```python
    import numpy as np
    from scipy.stats import median_test
    a = np.array([3, 3, 3, 4, 5, 5])
    b = np.array([1, 2, 3, 3, 3, 3])
    print(np.median(np.concatenate([a, b])))    # 3.0

    for ties in ("below", "above", "ignore"):
        r = median_test(a, b, ties=ties)
        print(ties, r.statistic.round(4), r.pvalue.round(4))
        print(r.table)
    ```

    출력:

    ```
    3.0
    below 1.7778 0.1824
    [[3 0]
     [3 6]]
    above 0.6 0.4386
    [[6 4]
     [0 2]]
    ignore 1.7014 0.1921
    [[3 0]
     [0 2]]
    ```

    | `ties` | 처리 방식 | 분할표 | $\chi^2$ | $p$ |
    |:---|:---|:---|---:|---:|
    | `"below"` (기본값) | 중앙값과 같은 값을 "아래"로 | $\begin{smallmatrix}3&0\\3&6\end{smallmatrix}$ | 1.778 | 0.182 |
    | `"above"` | "위"로 | $\begin{smallmatrix}6&4\\0&2\end{smallmatrix}$ | 0.600 | 0.439 |
    | `"ignore"` | 버린다 | $\begin{smallmatrix}3&0\\0&2\end{smallmatrix}$ | 1.701 | 0.192 |

    같은 자료에서 $\chi^2$이 $0.6$에서 $1.78$까지, $p$값이 $0.182$에서 $0.439$까지 요동친다. $12$개 관측값 중 $7$개가 중앙값 $3$과 같기 때문이다.

    본문 구현은 `"ignore"`에 해당한다. 이 자료에서는 표본크기를 $12$에서 $5$로 줄여 검정력을 크게 떨어뜨린다.

    **권고:** 동점이 많은 이산자료에는 Mood 중앙값검정을 쓰지 말라. 결론이 임의의 처리 규칙에 좌우된다. 이런 자료에는 순위 기반 검정(중간순위로 동점을 원칙적으로 처리한다)이나 정확 순열검정이 낫다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
Kruskal-Wallis 검정이 집단 둘일 때 Mann-Whitney U 검정과 동치임을 확인하라. $H$와 $Z$ 사이에 어떤 관계가 있는가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy.stats import kruskal, mannwhitneyu, rankdata
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 15); y = rng.normal(0.7, 1, 18)

    H, pH = kruskal(x, y)
    r = mannwhitneyu(x, y, method='asymptotic', use_continuity=False)
    n1, n2 = len(x), len(y)
    mu = n1 * n2 / 2
    sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    Z = (r.statistic - mu) / sd
    print(H, Z**2, H - Z**2)        # 6.4052  6.4052  ~0
    print(pH, r.pvalue)             # 0.01138  0.01138
    ```

    출력:

    ```
    6.405228758169926 6.405228758169934 -7.993605777301127e-15
    0.011378476531193307 0.01137847653119324
    ```

    **$H = Z^2$이고 두 $p$값이 정확히 같다.**

    이유는 자유도 1인 카이제곱분포가 표준정규의 제곱이기 때문이다. $Z \sim \mathcal{N}(0,1)$이면 $Z^2 \sim \chi^2_1$이고, 집단이 둘이면 Kruskal-Wallis의 자유도가 $k - 1 = 1$이다.

    대수적으로도 확인할 수 있다. $k = 2$일 때

    $$
    H = \frac{12}{N(N+1)}\left(\frac{R_1^2}{n_1} + \frac{R_2^2}{n_2}\right) - 3(N+1)
    $$

    에 $R_2 = N(N+1)/2 - R_1$을 대입하고 정리하면

    $$
    H = \frac{\left(R_1 - \frac{n_1(N+1)}{2}\right)^2}{\frac{n_1 n_2 (N+1)}{12}} = Z^2
    $$

    가 된다.

    다만 두 검정이 완전히 같은 것은 아니다. Kruskal-Wallis는 언제나 **양측**이며 $\chi^2$ 근사만 쓴다. Mann-Whitney는 단측검정과 정확 $p$값을 지원한다. 집단이 둘이면 Mann-Whitney를 쓰는 편이 낫다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
본문의 Mann-Whitney 예제에서 SciPy의 기본 $p$값($0.000440$)과 정확 $p$값($7.58 \times 10^{-5}$)이 6배 가까이 차이 난다. 왜 이렇게 큰가?

</div>

??? success "풀이"
    자료를 다시 보자.

    - 집단 A: 76, 78, 82, 85, 87, 88, 89, 91, 92, 95
    - 집단 B: 68, 69, 70, 71, 72, 73, 74, 75, 77, 81

    B의 최댓값 $81$이 A의 최솟값 $76$과 두 번째 $78$보다 크므로 완전 분리는 아니지만 거의 그렇다. $U_1 = 97$로 최댓값 $100$에 가깝다.

    이것이 핵심이다. **관측값이 귀무분포의 극단 꼬리에 있다.** 정규근사는 분포의 중앙부에서 가장 정확하고 꼬리로 갈수록 상대오차가 커진다. $p$값이 $10^{-4}$ 수준이면 상대오차가 몇 배로 벌어지는 것이 정상이다.

    ```python
    import numpy as np, itertools
    from scipy.stats import mannwhitneyu
    ga = [85, 78, 92, 88, 76, 95, 89, 82, 91, 87]
    gb = [72, 68, 81, 75, 70, 77, 74, 69, 73, 71]
    print(mannwhitneyu(ga, gb, method='exact').pvalue)        # 7.5776e-05
    print(mannwhitneyu(ga, gb, method='asymptotic',
                       use_continuity=True).pvalue)           # 4.3964e-04
    print(mannwhitneyu(ga, gb, method='asymptotic',
                       use_continuity=False).pvalue)          # 3.8106e-04
    ```

    출력:

    ```
    7.577561757128321e-05
    0.00043963875262656465
    0.00038105845205068555
    ```

    | 방법 | $p$값 |
    |:---|---:|
    | 정확 | $7.58 \times 10^{-5}$ |
    | 정규근사, 보정 없음 | $3.81 \times 10^{-4}$ |
    | 정규근사, 연속성 보정 | $4.40 \times 10^{-4}$ |

    **절대오차는 $0.0004$로 작다.** 어느 방법을 써도 $\alpha = 0.05$나 $\alpha = 0.001$에서 같은 결론이 나온다. 배수로 보면 6배지만 결정에는 아무 영향이 없다.

    **교훈:** 아주 작은 $p$값의 정확한 자릿수에 집착할 필요는 없다. "$p < 0.001$"이 결론이고, 그 안에서 $7.6 \times 10^{-5}$인지 $4.4 \times 10^{-4}$인지는 실무적으로 같은 이야기이다. 반대로 $p$가 $0.04$와 $0.06$ 사이에 있다면 근사 오차가 결정을 뒤집을 수 있으므로 정확검정을 써야 한다.

---

## 정리하며

만–휘트니 $U$ 가 **이표본 $t$ 검정의 대안**이다.

- **두 표본을 합쳐 순위를 매긴다.** 한 집단의 순위합이 통계량이 되며, 윌콕슨 순위합검정과 같은 검정이다.
- **"분포가 같은가"를 검정한다.** 중앙값 비교로 읽으려면 **두 분포의 모양이 같다**는 가정이 필요하며, 이 단서를 빠뜨리면 결론이 과장된다.
- **위치 이동 모형에서 가장 자연스럽다.** 한 분포가 다른 것을 평행이동한 형태라면 중앙값 차이에 대한 검정이 된다.
- **순서형 자료에 쓸 수 있다.** 순위만 쓰므로 값의 간격이 의미 없어도 된다.
- **이상치에 강건하다.** 극단값도 순위 하나만 차지하므로 영향이 제한된다.

다음 절 **이표본 및 다집단 검정 (코드)** 로 넘어간다.
