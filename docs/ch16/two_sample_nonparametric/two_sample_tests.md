# 이표본 및 다집단 검정

## 개요

정규성을 가정하지 않고 두 집단 이상을 비교할 때 순위 기반 및 중앙값 기반 비모수 절차를
여럿 쓸 수 있다. 이 페이지에서는 **Wilcoxon 순위합검정**, **Wilcoxon 부호순위검정**,
**Mann--Whitney $U$ 검정**, **Kruskal--Wallis $H$ 검정**, **Mood 중앙값검정**을 다룬다.
각 검정은 원 관측값을 순위나 전체 중앙값 기준의 이진 지시값으로 바꾸어, 이상치와 비정규
자료에 로버스트한 분포무관 추론을 제공한다.

## Wilcoxon 순위합검정

순위합검정은 **독립인** 두 표본을 비교한다. $N = m + n$개 관측값을 모두 합쳐 $1$부터 $N$까지
순위를 매기고 한 표본의 순위를 합한다.

$H_0$(분포가 동일) 아래에서 표본 1의 순위합 $W$는

$$
\operatorname{E}[W] = \frac{m(N+1)}{2}, \qquad
\operatorname{Var}(W) = \frac{m\,n\,(N+1)}{12}
$$

을 만족한다. 표준화된 통계량

$$
Z = \frac{W - \operatorname{E}[W]}{\sqrt{\operatorname{Var}(W)}}
$$

을 표준정규분포와 비교한다.

```python
from scipy import stats

# 독립인 두 집단
group_a = [12, 15, 18, 22, 25]
group_b = [8, 10, 14, 19, 21, 24]

stat, p = stats.ranksums(group_a, group_b, alternative="two-sided")
print(f"Z = {stat:.4f}, p = {p:.2%}")
# Z = 0.7303, p = 46.52%
```

출력:

```
Z = 0.7303, p = 46.52%
```

## Wilcoxon 부호순위검정

**대응**자료에서는 부호순위검정이 절대차이 $|D_i|$에 순위를 매기고 원래 부호를 붙인 뒤
양의 부호순위를 합해 $W^+$를 얻는다.

차이가 대칭인 $H_0{:}\;\text{median}(D) = 0$ 아래에서

$$
\operatorname{E}[W^+] = \frac{n'(n'+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n'(n'+1)(2n'+1)}{24}
$$

이며 $n'$은 0이 아닌 차이의 개수이다.

```python
from scipy import stats
import numpy as np

paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

stat, p = stats.wilcoxon(
    paired_data[:, 0], paired_data[:, 1],
    alternative="two-sided", method="approx", zero_method="pratt"
)
print(f"W = {stat}, p = {p:.4f}")
# W = 11.0, p = 0.0086
```

출력:

```
W = 11.0, p = 0.0086
```

!!! danger "같은 자료에 두 검정을 섞어 쓰지 말 것"
    위 대응자료를 두 열로 쪼개어 `ranksums`에 넣으면 $Z = 1.472$, $p = 0.141$이 나온다.
    부호순위검정의 $p = 0.0086$과 16배 차이가 난다.

    자료의 **구조**가 어느 검정을 쓸지 결정한다. 같은 피험자를 두 번 측정했다면
    대응자료이고, 서로 다른 피험자 집단이라면 독립표본이다. 두 검정을 모두 돌려 보고
    작은 $p$값을 고르는 것은 명백한 오용이다.

## Mann--Whitney U 검정

Mann--Whitney $U$ 통계량은 한쪽 관측값이 다른 쪽을 넘어서는 쌍의 개수를 센다.

$$
U = \sum_{i=1}^{m} \sum_{j=1}^{n} \mathbf{1}[X_i > Y_j].
$$

순위합과는 $U = W - m(m+1)/2$로 연결되므로 두 검정은 동치이다.
SciPy 구현은 **동점**을 동점 보정 분산과 연속성 보정 정규근사로 처리한다.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]

stat, p = stats.mannwhitneyu(data0, data1)
print(f"U = {stat}, p = {p:.2%}")
# U = 49.0, p = 0.53%
```

출력:

```
U = 49.0, p = 0.53%
```

## Kruskal--Wallis H 검정

Kruskal--Wallis 검정은 순위합의 발상을 $k \geq 2$개의 독립집단으로 확장한다.
$N$개 관측값을 모두 합쳐 순위를 매기고

$$
H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)
$$

을 계산한다. $R_j$는 집단 $j$의 순위합, $n_j$는 집단 크기이다. $H_0$(모든 집단이 같은
모집단에서 왔다) 아래에서 $H$는 근사적으로 $\chi^2_{k-1}$을 따른다.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
data2 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]

stat, p = stats.kruskal(data0, data1, data2)
print(f"H = {stat:.4f}, p = {p:.2%}")
# H = 7.6480, p = 2.18%
```

출력:

```
H = 7.6480, p = 2.18%
```

## Mood 중앙값검정

Mood 중앙값검정은 Kruskal--Wallis의 더 간단한 대안이다. 모든 관측값의 **전체 중앙값**을
계산하고, 각 관측값을 그 중앙값보다 위인지 아래인지로 분류하여 $2 \times k$ 분할표를
만든다. 이 표에 카이제곱 검정을 적용하여 집단들의 중앙값이 같은지 판정한다.

```python
from scipy import stats

data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]
data2 = [0, 3, 9, 22, 23, 25, 25, 33, 34, 34, 40, 45, 46, 48, 62, 67, 84]

result = stats.median_test(data0, data1, data2)
print(f"Grand median = {result.median}")
print(f"Contingency table:\n{result.table}")
print(f"p-value = {result.pvalue:.4f}")
# Grand median = 34.0
# Contingency table:
# [[ 5 10  7]
#  [11  5 10]]
# p-value = 0.1261
```

출력:

```
Grand median = 34.0
Contingency table:
[[ 5 10  7]
 [11  5 10]]
p-value = 0.1261
```

!!! note "같은 자료에서 두 검정의 결론이 갈린다"
    Kruskal--Wallis는 $p = 0.0218$로 기각하지만 Mood 중앙값검정은 $p = 0.1261$로
    기각하지 못한다. Mood 검정이 각 관측값을 한 비트로 뭉개면서 정보를 크게 잃기
    때문이다.

    구체적으로 `data1`은 값이 $28$부터 $99$까지 전체적으로 높지만, Mood 검정에게는
    "중앙값 $34$보다 위인 것이 $15$개 중 $10$개"라는 정보만 남는다. Kruskal--Wallis는
    `data1`의 값들이 순위 상위권에 몰려 있다는 사실을 그대로 반영한다.

## 해석

| 검정 | 용도 | 귀무가설 | 통계량의 분포 |
|---|---|---|---|
| 순위합 | 독립인 두 표본 | 분포가 동일 | 정규 (대표본) |
| 부호순위 | 대응표본 | 중앙값 차이가 0 | 정규 (대표본) |
| Mann--Whitney $U$ | 독립인 두 표본 (동점 포함) | $P(X > Y) = 0.5$ | 정규 (대표본) |
| Kruskal--Wallis | $k$개 독립집단 | 모든 집단이 같은 모집단 | $\chi^2_{k-1}$ |
| Mood 중앙값 | $k$개 독립집단 | 모든 집단의 중앙값이 같음 | $\chi^2_{k-1}$ |

**검정의 선택**:

- 독립인 두 표본에서 순위합검정과 Mann--Whitney 검정은 동치이다. `mannwhitneyu`가
  정확검정과 효과크기 $U/(mn)$을 함께 제공하므로 대체로 더 편하다.
- 대응자료에는 부호순위검정(차이의 대칭성 필요)이나 부호검정(대칭성 불필요)을 쓴다.
- 셋 이상의 집단에는 Kruskal--Wallis가 Mood 중앙값검정보다 대체로 강력하지만,
  Mood 검정이 더 단순하고 이상치에 로버스트하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 두 학생 집단이 서로 다른 시험지를 풀었다. 집단 A의 점수는
$(72, 78, 81, 85, 90)$, 집단 B의 점수는 $(68, 74, 77, 83, 88, 92)$이다.
Wilcoxon 순위합검정을 손으로 수행하라. 합치고 순위를 매긴 뒤 집단 A의 $W$와
$Z$ 통계량을 구하라.

</div>

??? success "풀이"

    $N = 11$개 값을 합쳐 순위를 매기면

    | 값 | 68 | 72 | 74 | 77 | 78 | 81 | 83 | 85 | 88 | 90 | 92 |
    |---|---|---|---|---|---|---|---|---|---|---|---|
    | 순위 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    | 집단 | B | A | B | B | A | A | B | A | B | A | B |

    집단 A($m = 5$)의 순위합: $W = 2 + 5 + 6 + 8 + 10 = 31$.

    $$
    \operatorname{E}[W] = \frac{5 \cdot 12}{2} = 30, \qquad
    \operatorname{Var}(W) = \frac{5 \cdot 6 \cdot 12}{12} = 30.
    $$

    $$
    Z = \frac{31 - 30}{\sqrt{30}} \approx 0.183.
    $$

    양측 $p$값은 $2\Phi(-0.183) \approx 0.855$이다. $H_0$을 기각하지 못한다.
    두 집단 사이에 유의한 차이가 없다.

    ```python
    from scipy import stats
    a = [72, 78, 81, 85, 90]; b = [68, 74, 77, 83, 88, 92]
    print(stats.ranksums(a, b))
    # RanksumsResult(statistic=0.18257, pvalue=0.85513)
    print(stats.mannwhitneyu(a, b, method='exact'))
    # MannwhitneyuResult(statistic=16.0, pvalue=0.93074)
    ```

    출력:

    ```
    RanksumsResult(statistic=0.18257418583505536, pvalue=0.8551321405847059)
    MannwhitneyuResult(statistic=16.0, pvalue=0.9307359307359306)
    ```

    $U = W - m(m+1)/2 = 31 - 15 = 16$으로 확인된다. 두 집단의 값이 거의 완벽하게
    번갈아 나타나므로 $U$가 $mn/2 = 15$에 매우 가깝다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> 세 처리를 독립인 집단에 적용하여 다음 결과를 얻었다.

- 처리 1: $14, 18, 22, 25$
- 처리 2: $19, 23, 27, 30, 35$
- 처리 3: $10, 15, 20$

Kruskal--Wallis $H$ 통계량을 손으로 계산하라.

</div>

??? success "풀이"

    $N = 12$개 값을 합쳐 순위를 매기면

    | 값 | 10 | 14 | 15 | 18 | 19 | 20 | 22 | 23 | 25 | 27 | 30 | 35 |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|
    | 순위 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
    | 집단 | 3 | 1 | 3 | 1 | 2 | 3 | 1 | 2 | 1 | 2 | 2 | 2 |

    순위합: $R_1 = 2+4+7+9 = 22$, $R_2 = 5+8+10+11+12 = 46$, $R_3 = 1+3+6 = 10$.

    검산: $22 + 46 + 10 = 78 = 12 \cdot 13 / 2$. $\checkmark$

    $$
    H = \frac{12}{12 \cdot 13}\left(\frac{22^2}{4} + \frac{46^2}{5} + \frac{10^2}{3}\right) - 3 \cdot 13.
    $$

    $$
    H = \frac{12}{156}\left(121 + 423.2 + 33.333\right) - 39
      = 0.076923 \times 577.533 - 39
      = 44.426 - 39
      = 5.426.
    $$

    자유도 $k - 1 = 2$에서 $P(\chi^2_2 > 5.426) = 0.0664$이다.
    $\alpha = 0.05$에서 기각하지 못하지만 시사적인 결과이다.

    ```python
    from scipy import stats
    t1 = [14, 18, 22, 25]; t2 = [19, 23, 27, 30, 35]; t3 = [10, 15, 20]
    print(stats.kruskal(t1, t2, t3))
    # KruskalResult(statistic=5.42564, pvalue=0.06637)
    ```

    출력:

    ```
    KruskalResult(statistic=5.425641025641035, pvalue=0.06634940320987275)
    ```

    집단 크기가 $4, 5, 3$으로 매우 작아 $\chi^2$ 근사가 신뢰할 만하지 않다는 점에
    유의하라. [Kruskal-Wallis](../multi_group_nonparametric/kruskal_wallis.md)
    연습문제 2에서 보았듯 이런 크기에서 $\chi^2$ 근사는 $p$값을 크게 과대평가할
    수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> Mood 중앙값검정은 각 관측값을 이진 지시값(전체 중앙값 위/아래)으로
바꾼다. 이 접근이 Kruskal--Wallis보다 검정력이 낮지만 극단 이상치에 더 로버스트한
이유를 설명하라.

</div>

??? success "풀이"

    Mood 중앙값검정은 각 관측값을 정보 한 비트 --- 전체 중앙값을 넘는지 여부 ---
    로 줄인다. 각 관측값이 중앙값에서 *얼마나 멀리* 있는지를 무시하므로, 순위가
    보존하는 순서 정보를 버린다. Kruskal--Wallis는 완전한 순위 정보를 쓰므로
    분포 이동을 더 잘 포착하고 검정력이 더 높다.

    그러나 바로 그 정보 축소가 Mood 검정을 더 로버스트하게 만든다. 중앙값이 $30$인
    자료에 $10{,}000$이라는 극단 이상치가 있어도 $35$라는 온건한 값과 똑같이
    "위"라는 이진 부호를 받는다. Kruskal--Wallis에서는 이상치의 극단적 순위가 그
    집단의 순위합을 부풀려 결과를 왜곡할 수 있다.

    본문 예제가 이 대비를 보여 준다. Kruskal--Wallis는 $p = 0.0218$로 기각하고
    Mood 검정은 $p = 0.1261$로 기각하지 못한다.

    !!! warning "로버스트성의 한계"
        Mood 검정의 로버스트성은 과장되기 쉽다. 순위 자체도 이미 이상치에 매우
        로버스트하기 때문이다. 관측값이 $10{,}000$이든 $35$든 Kruskal--Wallis에서
        받는 최대 순위는 $N$으로 같다. 이상치 하나가 순위합을 움직일 수 있는 폭은
        $N$ 이하로 제한된다.

        [Mood 중앙값검정](../multi_group_nonparametric/median_test.md) 연습문제 4에서
        보았듯, 등분산성이 크게 깨지면 두 검정 모두 명목수준을 넘는다. Mood 검정이
        만능 보험은 아니다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span> 위 코드 예제의 Mann--Whitney 자료로 $U = W - m(m+1)/2$를 손으로
확인하라. $W$는 `data0`의 순위합이고 $m = 16$이다.

</div>

??? success "풀이"

    합친 표본은 $N = 16 + 15 = 31$개이다. 합쳐 순위를 매긴 뒤 `data0`의 순위를
    더하면 $W$가 나온다.

    ```python
    import numpy as np
    from scipy import stats

    data0 = [10, 14, 14, 18, 20, 22, 24, 25, 31, 31, 32, 39, 43, 43, 48, 49]
    data1 = [28, 30, 31, 33, 34, 35, 36, 40, 44, 55, 57, 61, 91, 92, 99]

    combined = np.array(data0 + data1)
    ranks = stats.rankdata(combined)
    W = ranks[:16].sum()
    print(f"W = {W}")                        # W = 185.0
    print(f"W - m(m+1)/2 = {W - 16*17/2}")   # 49.0

    U, p = stats.mannwhitneyu(data0, data1)
    print(f"U = {U}")                        # U = 49.0
    ```

    출력:

    ```
    W = 185.0
    W - m(m+1)/2 = 49.0
    U = 49.0
    ```

    $W = 185$이고 $W - 16 \cdot 17/2 = 185 - 136 = 49 = U$로 정확히 일치한다.

    이 자료에는 동점이 여럿 있음에 유의하라. `data0`의 $14$가 두 개, $31$이 두 개,
    $43$이 두 개이고, `data0`의 $31$과 `data1`의 $31$은 **집단을 넘어서** 동점이다.
    `rankdata`가 중간순위를 배정하므로 $W$가 정수가 아닐 수도 있지만, 여기서는
    집단 간 동점이 하나뿐이라 우연히 정수로 떨어졌다.

    $U = 49$를 $mn = 240$으로 나누면 $\hat{P}(X > Y) = 0.204$이다. `data0`의 값이
    `data1`보다 큰 쌍이 20%에 불과하다는 뜻이며, $p = 0.0053$이라는 유의한 결과와
    일관된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> $k = 2$일 때 Kruskal--Wallis $H$ 통계량이 순위합 $Z$ 통계량의
제곱으로 환원됨을 증명하라.

</div>

??? success "풀이"

    크기가 $m$과 $n = N - m$인 두 집단에서 순위합은 $R_1 + R_2 = N(N+1)/2$를
    만족한다. Kruskal--Wallis 통계량은

    $$
    H = \frac{12}{N(N+1)}\left(\frac{R_1^2}{m} + \frac{R_2^2}{n}\right) - 3(N+1)
    $$

    이다. $W = R_1$, $S = N(N+1)/2$라 두고 $R_2 = S - W$를 대입하면

    $$
    \frac{R_1^2}{m} + \frac{R_2^2}{n}
    = \frac{n W^2 + m(S - W)^2}{mn}
    = \frac{(m+n)W^2 - 2mSW + mS^2}{mn}
    = \frac{N W^2 - 2mSW + mS^2}{mn}.
    $$

    한편 목표하는 형태는 $\bigl(W - mS/N\bigr)^2$을 포함한다. 실제로

    $$
    \frac{N}{mn}\left(W - \frac{mS}{N}\right)^2
    = \frac{N W^2 - 2mSW + m^2S^2/N}{mn}
    $$

    이므로 두 식의 차이는

    $$
    \frac{mS^2 - m^2S^2/N}{mn} = \frac{S^2(N - m)}{Nn} = \frac{S^2}{N}
    $$

    이다. 따라서

    $$
    H = \frac{12}{N(N+1)}\left[\frac{N}{mn}\left(W - \frac{mS}{N}\right)^2 + \frac{S^2}{N}\right] - 3(N+1).
    $$

    $S = N(N+1)/2$이므로 $\frac{12}{N(N+1)} \cdot \frac{S^2}{N} = \frac{12}{N(N+1)} \cdot \frac{N(N+1)^2}{4} = 3(N+1)$이 되어
    상수항이 정확히 소거된다. 또 $mS/N = m(N+1)/2 = \operatorname{E}[W]$이므로

    $$
    H = \frac{12}{N(N+1)} \cdot \frac{N}{mn}\left(W - \operatorname{E}[W]\right)^2
      = \frac{\bigl(W - \operatorname{E}[W]\bigr)^2}{mn(N+1)/12}
      = Z^2. \quad \square
    $$

    이것이 $H$의 $\chi^2_1$ 분포가 $Z$의 표준정규분포의 제곱임을 확인해 준다.

    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 12); y = rng.normal(0.5, 1, 15)
    H = stats.kruskal(x, y).statistic
    Z = stats.ranksums(x, y).statistic
    print(H, Z ** 2, abs(H - Z ** 2) < 1e-10)
    # True
    ```

    출력:

    ```
    0.7714285714285722 0.7714285714285715 True
    ```

---

## 정리하며

다섯 검정의 **구현과 선택 기준**을 모았다.

| 검정 | 쓰는 곳 |
|---|---|
| 윌콕슨 순위합 · 만–휘트니 $U$ | 독립 2집단 (같은 검정) |
| 윌콕슨 부호순위 | 대응 2측정 |
| 크루스칼–월리스 $H$ | 독립 $k$집단 |
| 무드 중앙값검정 | 독립 $k$집단, 중앙값만 |

- **만–휘트니와 윌콕슨 순위합은 같은 검정이다.** 통계량의 정의만 다르고 $p$ 값이 같으며, 이름이 둘인 것이 혼동의 원인이다.
- **크루스칼–월리스는 분산분석의 순위판이다.** 기각하면 사후비교가 필요하며, 던 검정에 다중검정 보정을 쓴다.
- **무드 중앙값검정은 가정이 더 적고 검정력도 더 낮다.** 각 집단에서 전체 중앙값을 넘는 관측 수를 세어 카이제곱으로 검정한다.
- **`scipy` 의 기본값을 확인한다.** 연속성 보정, 동점 처리, 정확·근사 전환이 함수마다 다르다.
- **어느 검정이든 결론의 문장을 정확히 쓴다.** "중앙값이 다르다"인지 "분포가 다르다"인지가 가정에 달려 있다.

**이것으로 16장이 끝난다.** 분포 가정을 최소화한 검정들을 일표본·대응·이표본·다집단으로 훑었다.

다음 장 **재표집 방법**으로 넘어간다. 순위로 바꾸는 대신 **자료 자체를 다시 뽑아** 분포를 만드는 접근이다.
