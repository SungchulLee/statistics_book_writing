# 카플란-마이어 생존곡선과 로그순위 검정

## 개요

카플란-마이어 추정량은 절단자료에서 생존함수를 추정하는 비모수적 방법의 초석이다. 로그순위
검정과 결합하면 생존 양상을 시각화하고 집단 간 생존분포를 비교하는 완전한 도구가 된다. 이
절에서는 NumPy와 SciPy만으로 두 기법을 구현하고 계산의 각 단계를 설명한다.

## 카플란-마이어 추정량

### 수학적 기초

관측 시간과 절단 지시자를 갖는 대상 $n$명이 주어지면, 카플란-마이어 추정량은 서로 다른 각
사건시간 $t_{(j)}$에서의 조건부 생존확률의 곱으로 생존함수를 계산한다.

$$
\hat{S}(t) = \prod_{j:\, t_{(j)} \leq t} \left(1 - \frac{d_j}{n_j}\right)
$$

여기서 $d_j$는 시점 $t_{(j)}$의 사건 수이고 $n_j$는 $t_{(j)}$ 직전에 위험에 있는 대상 수다.

### 구현

다음 함수가 원자료에서 카플란-마이어 생존곡선을 계산한다.

```python
import numpy as np

def kaplan_meier(times, censored):
    """
    Compute the Kaplan-Meier survival function estimate.

    Parameters
    ----------
    times    : 1-d array   Observed times (event or censoring).
    censored : 1-d array   1 = censored (no event), 0 = event observed.

    Returns
    -------
    t_plot : array   Time points for step-plot.
    s_plot : array   Survival probabilities matching t_plot.
    """
    order = np.argsort(times)
    times = times[order]
    censored = censored[order]

    event_times = times[censored == 0]
    unique_events = np.unique(event_times)

    s = 1.0
    t_list = [0.0]
    s_list = [1.0]

    for t_j in unique_events:
        n_at_risk = np.sum(times >= t_j)
        d_j = np.sum((times == t_j) & (censored == 0))
        s *= (n_at_risk - d_j) / n_at_risk
        t_list.append(t_j)
        s_list.append(s)

    t_list.append(times.max())
    s_list.append(s_list[-1])

    return np.array(t_list), np.array(s_list)
```

!!! warning "이 코드의 `censored`는 $\delta$와 부호가 반대다"
    이 페이지의 코드는 `censored == 1`이 절단, `censored == 0`이 사건을 뜻한다. 이 장의 본문
    표기 $\delta_i$는 반대로 $\delta_i = 1$이 사건이다. 두 관례가 모두 쓰이므로 코드를 옮겨
    쓸 때 반드시 확인하라. `lifelines`는 `event_observed`(사건이 1), R의 `Surv()`는
    `event`(사건이 1)를 쓴다. 이 페이지의 관례가 오히려 소수파다.

**알고리즘의 핵심 단계:**

1. 관측치를 시간순으로 **정렬**한다.
2. 서로 다른 사건시간(`censored == 0`인 시점)을 **식별**한다.
3. **각 사건시간** $t_j$에서 위험집합 크기 $n_j$($t_i \geq t_j$인 대상 수)와 사건 수 $d_j$를
   계산한다.
4. 생존확률을 **갱신**한다. $\hat{S}(t_j) = \hat{S}(t_{j-1}) \times (1 - d_j / n_j)$.
5. 곡선을 관측된 최대 시점까지 **연장**한다.

!!! note "절단된 관측치"

    절단된 대상(`censored == 1`)은 생존곡선의 하강을 일으키지 않지만 이후 사건시간의 위험집합을
    줄인다. 불완전한 관측이 담은 부분적 정보를 이렇게 반영한다.

## 로그순위 검정

### 가설

로그순위 검정은 두 생존곡선을 비교한다.

$$
H_0 : S_1(t) = S_2(t) \quad \text{for all } t \geq 0
$$

$$
H_1 : S_1(t) \neq S_2(t) \quad \text{for some } t \geq 0
$$

### 검정통계량

두 집단을 합친 서로 다른 각 사건시간 $t_{(j)}$에서 $r_{1j}$와 $r_{2j}$를 위험집합 크기,
$d_{1j}$와 $d_{2j}$를 사건 수라 하고 $r_j = r_{1j} + r_{2j}$, $d_j = d_{1j} + d_{2j}$라 하자.
영가설 아래에서 집단 1의 기대 사건 수는

$$
e_{1j} = d_j \cdot \frac{r_{1j}}{r_j}
$$

이고, $t_{(j)}$에서의 분산 기여는

$$
v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
$$

이다. 검정통계량은

$$
\chi^2_{\text{LR}} = \frac{(O_1 - E_1)^2}{V_1} \;\xrightarrow{d}\; \chi^2_1
$$

이며 $O_1 = \sum d_{1j}$, $E_1 = \sum e_{1j}$, $V_1 = \sum v_j$이다.

### 구현

```python
from scipy import stats

def logrank_test(times_1, censored_1, times_2, censored_2):
    """
    Two-sample log-rank test.

    Returns
    -------
    chi2    : float   Test statistic (chi-square with 1 df).
    p_value : float   p-value from chi-square(1).
    """
    event_1 = times_1[censored_1 == 0]
    event_2 = times_2[censored_2 == 0]
    all_event_times = np.unique(np.concatenate([event_1, event_2]))

    O1, E1, V = 0.0, 0.0, 0.0

    for t_j in all_event_times:
        r1 = np.sum(times_1 >= t_j)
        r2 = np.sum(times_2 >= t_j)
        r  = r1 + r2

        d1 = np.sum(event_1 == t_j)
        d2 = np.sum(event_2 == t_j)
        d  = d1 + d2

        e1 = r1 * d / r if r > 0 else 0
        v  = r1 * r2 * d * (r - d) / (r**2 * (r - 1)) if r > 1 else 0

        O1 += d1
        E1 += e1
        V  += v

    chi2 = (O1 - E1)**2 / V if V > 0 else 0
    p_value = stats.chi2(1).sf(chi2)
    return chi2, p_value
```

이 구현은 합쳐진 각 사건시간을 순회하며 집단 1의 관측 사건 수, 기대 사건 수, 분산을 누적한 뒤
카이제곱 통계량을 계산한다.

## 모의실험과 시각화

다음 코드는 사건율이 다른 두 집단의 생존자료를 모의로 생성하고 카플란-마이어 곡선을 그린다.

```python
import matplotlib.pyplot as plt

np.random.seed(0)

# Group 1: slower event rate (scale = 20)
n1 = 40
times_1 = np.random.exponential(scale=20, size=n1)
censored_1 = (np.random.rand(n1) < 0.2).astype(int)

# Group 2: faster event rate (scale = 12)
n2 = 40
times_2 = np.random.exponential(scale=12, size=n2)
censored_2 = (np.random.rand(n2) < 0.2).astype(int)

# Compute Kaplan-Meier curves
t1, s1 = kaplan_meier(times_1, censored_1)
t2, s2 = kaplan_meier(times_2, censored_2)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.step(t1, s1, where="post", linewidth=2, label="Group 1 (slow)")
ax.step(t2, s2, where="post", linewidth=2, label="Group 2 (fast)")
ax.set_xlabel("Time")
ax.set_ylabel("Survival Probability")
ax.set_title("Kaplan-Meier Survival Curves")
ax.set_ylim(-0.02, 1.05)
ax.legend()
plt.tight_layout()
plt.show()
```

![두 집단의 카플란-마이어 생존곡선](./img/kaplan_meier_code_173.png)

집단 1은 평균 20인 지수분포에서(사건이 느림), 집단 2는 평균 12에서(사건이 빠름) 뽑았다. 각
집단의 약 20%가 무작위로 절단되었다.

!!! note "이 코드의 절단은 사건시간과 독립이 아니다"
    `censored_1 = (np.random.rand(n1) < 0.2)`는 관측된 시간과 **무관하게** 20%를 절단으로
    표시한다. 이는 실제 절단 기제(대상마다 절단시간 $C_i$가 있고 $t = \min(T, C)$)와 다르다.
    여기서는 관측 시간을 그대로 두고 이름표만 바꾸므로, 절단된 대상의 기록된 시간이 참
    사건시간이다. 교육용으로는 무해하지만, 절단이 관측 시간을 **줄인다**는 현실의 핵심
    성질이 빠져 있다. 21.1절의 모의실험 코드가 올바른 방식을 보여준다.

## 해석

- 카플란-마이어 곡선은 관측된 각 사건시간에서 떨어지는 **계단함수**다. 평평한 구간은 사건이
  없는 구간에 해당하며, 위험집합을 줄이는 절단된 관측치가 그 안에 있을 수 있다.
- 곡선이 더 오래 높이 머무를수록 그 집단의 **생존이 낫다**.
- **로그순위 검정**은 두 곡선의 시각적 분리가 통계적으로 유의한지를 형식적으로 평가한다. p-값이
  작으면(예: $p < 0.05$) 생존분포가 같다는 영가설을 기각한다.
- 로그순위 검정은 **비례위험** 가정이 성립할 때(집단 간 위험비가 시간에 걸쳐 대략 일정할 때)
  검정력이 가장 높다. 생존곡선이 교차하면 차이를 탐지하지 못할 수 있다.

!!! tip "로그순위 검정을 언제 쓰는가"

    공변량 보정이 필요 없을 때 두 개 이상의 집단을 비교하는 데 적절하다. 다변량 분석에는
    콕스 비례위험 모형을 쓰라.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
카플란-마이어 계산

환자 6명의 생존자료가 다음과 같다.

| 대상 | 시간 | 상태 (0 = 사건, 1 = 절단) |
|:-------:|:----:|:--------------------------------:|
| A | 2 | 0 |
| B | 3 | 1 |
| C | 5 | 0 |
| D | 5 | 0 |
| E | 8 | 1 |
| F | 10 | 0 |

카플란-마이어 추정량으로 각 사건시간의 $\hat{S}(t)$를 계산하라.

</div>

??? success "풀이"

    서로 다른 사건시간은 $t_{(1)} = 2$, $t_{(2)} = 5$, $t_{(3)} = 10$이다.

    | $t_{(j)}$ | $n_j$ | $d_j$ | $1 - d_j/n_j$ | $\hat{S}(t_{(j)})$ |
    |:----------:|:-----:|:-----:|:--------------:|:-------------------:|
    | 2 | 6 | 1 | 5/6 = 0.833 | 0.833 |
    | 5 | 4 | 2 | 2/4 = 0.500 | 0.833 $\times$ 0.500 = 0.417 |
    | 10 | 1 | 1 | 0/1 = 0.000 | 0.417 $\times$ 0.000 = 0.000 |

    $t_{(2)} = 5$에서는 대상 A가 $t = 2$에 사건을 겪었고 대상 B가 $t = 3$에 절단되어
    $n_2 = 4$가 위험에 남는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
로그순위 검정의 설정

두 집단을 관측했다.

**집단 1:** 3, 6+, 9, 15 (+ = 절단).

**집단 2:** 1, 4, 8+, 12.

**(a)** 두 집단의 서로 다른 사건시간을 모두 나열하라.

**(b)** $t = 1$에서 집단 1의 기대 사건 수 $e_{11}$과 분산 기여 $v_1$을 계산하라.

</div>

??? success "풀이"

    **(a)** 사건시간(절단 제외)은 집단 1에서 3, 9, 15이고 집단 2에서 1, 4, 12이다. 합친 서로
    다른 사건시간은 1, 3, 4, 9, 12, 15다.

    **(b)** $t = 1$에서 $r_1 = 4$, $r_2 = 4$, $r = 8$, $d_1 = 0$, $d_2 = 1$, $d = 1$이므로

    $$
    e_{11} = d \cdot \frac{r_1}{r} = 1 \cdot \frac{4}{8} = 0.5
    $$

    $$
    v_1 = \frac{r_1 \cdot r_2 \cdot d \cdot (r - d)}{r^2 (r - 1)} = \frac{4 \cdot 4 \cdot 1 \cdot 7}{64 \cdot 7} = \frac{112}{448} = 0.25
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
절단의 효과

대상 10명이 모두 비율 $\lambda = 0.1$인 같은 지수분포를 따른다고 하자. 시나리오 A에서는 아무도
절단되지 않는다. 시나리오 B에서는 5명이 시점 $t = 5$에 절단된다.

**(a)** 두 시나리오에서 카플란-마이어 곡선이 어떻게 다를지 정성적으로 설명하라.

**(b)** 후반 시점에서 어느 시나리오가 더 넓은 신뢰띠를 만드는가? 이유는?

</div>

??? success "풀이"

    **(a)** 시나리오 A에서는 10명 모두가 사건을 기여하므로 카플란-마이어 곡선이 완전한 정보에
    근거하며 꾸준히 0까지 내려간다. 시나리오 B에서는 $t = 5$에 절단된 5명이 그 시점에
    위험집합에서 빠진다. 곡선은 $t = 5$까지는 시나리오 A와 같지만, $t = 5$ 이후에는 위험집합이
    작아져(절단되지 않은 5명만 남는다) 각 사건이 생존 추정치를 더 크게 떨어뜨린다.

    **(b)** 시나리오 B가 $t = 5$ 이후에 더 넓은 신뢰띠를 갖는다. 그린우드 공식에 따른
    카플란-마이어 추정량의 분산은 위험집합이 작아질수록 커진다. 원래 10명 대신 5명만 위험에
    있으면 각 사건이 추정치에 더 큰 불확실성을 기여한다.

    !!! note "곡선의 기댓값은 두 시나리오에서 같다"
        절단이 편향을 만들지 않는다는 점을 놓치지 말라. 시나리오 B의 곡선은 A보다 **덜 정확할**
        뿐 체계적으로 위나 아래로 치우치지 않는다. 절단이 $t = 5$라는 고정 시점에 일어나고
        사건시간과 무관하므로 독립 절단 가정이 성립하기 때문이다. 절단이 정보를 담고 있을 때에만
        (예: 상태가 나쁜 대상이 먼저 절단될 때) 편향이 생긴다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
로그순위 검정의 해석

처리군과 대조군을 비교한 로그순위 검정이 $\chi^2_{\text{LR}} = 5.23$, $p = 0.022$를 냈다.

**(a)** 유의수준 $\alpha = 0.05$에서의 결론을 서술하라.

**(b)** 처리군의 관측 사건 수가 $O_1 = 15$이고 기대 사건 수가 $E_1 = 21.3$이었다면 효과의
방향을 해석하라.

**(c)** 로그순위 검정의 어떤 가정이 위배되면 이 결과가 오도할 수 있는가?

</div>

??? success "풀이"

    **(a)** $p = 0.022 < 0.05$이므로 5% 수준에서 $H_0$을 기각하고, 처리군과 대조군의 생존분포가
    유의하게 다르다고 결론짓는다.

    **(b)** $O_1 = 15 < E_1 = 21.3$이므로 처리군의 사건이 영가설 아래의 기대보다 적었다. 처리군의
    생존이 대조군보다 낫다(위험이 낮다)는 뜻이다.

    대략적인 위험비를 $O_1/E_1 = 15/21.3 = 0.70$으로 어림할 수도 있다. 이를 **관측/기대
    위험비**라 하며, 콕스 모형을 적합하지 않고도 효과크기를 가늠하는 데 쓰인다. 다만 어디까지나
    근사이며, 정확한 위험비와 그 신뢰구간은 콕스 모형에서 얻어야 한다.

    **(c)** 로그순위 검정은 **비례위험**을 가정한다. 집단 간 위험비가 시간에 걸쳐 일정하다는
    것이다. 생존곡선이 교차하면(예: 처리가 초기에는 이롭지만 후반에는 해로우면) 로그순위
    검정이 차이를 탐지하지 못하거나 오도하는 요약을 줄 수 있다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
로그순위 통계량의 분산 유도

영가설 아래에서 $d_{1j}$(시점 $t_{(j)}$의 집단 1 사건 수)가 초기하분포를 따름을 보이고 다음
식을 유도하라.

$$
v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
$$

</div>

??? success "풀이"

    $H_0$ 아래에서 시점 $t_{(j)}$에 $r_j$명이 위험에 있고 그중 $r_{1j}$명이 집단 1에 속한다.
    이 $r_j$명 중에서 $d_j$개의 사건이 일어난다. 집단 1의 사건 수 $d_{1j}$는 초기하분포를
    따른다. 집단 1 소속 $r_{1j}$명을 포함한 $r_j$명의 풀에서 "사건" $d_j$개를 뽑는 것이다.

    초기하 확률변수 $X \sim \text{Hyper}(N, K, n)$의 분산은

    $$
    \text{Var}(X) = n \cdot \frac{K}{N} \cdot \frac{N - K}{N} \cdot \frac{N - n}{N - 1}
    $$

    이다. $N = r_j$, $K = r_{1j}$, $n = d_j$를 대입하면

    $$
    v_j = d_j \cdot \frac{r_{1j}}{r_j} \cdot \frac{r_j - r_{1j}}{r_j} \cdot \frac{r_j - d_j}{r_j - 1}
    $$

    이고 $r_j - r_{1j} = r_{2j}$이므로

    $$
    v_j = \frac{r_{1j} \, r_{2j} \, d_j \, (r_j - d_j)}{r_j^2 \, (r_j - 1)}
    $$

    $\square$

---

## 정리하며

카플란–마이어와 로그순위를 **함께 구현**했다.

- **자료 형식이 두 열이다.** 관측시간과 사건 지시자(1=사건, 0=절단)이며, 이 형식이 생존분석의 표준 입력이다.
- **`lifelines` 가 주력 도구다.** `KaplanMeierFitter` 로 곡선을, `logrank_test` 로 비교를 수행한다.
- **위험집합 표를 함께 그린다.** 각 시점에 남은 대상 수를 곡선 아래 표시하는 것이 관례이며, **꼬리의 신뢰도를 독자가 판단할 수 있게** 해 준다.
- **중앙생존시간을 보고한다.** $\hat S(t)=0.5$ 가 되는 시점이며, 곡선이 $0.5$ 아래로 내려가지 않으면 "도달하지 않음"으로 적는다.
- **곡선과 검정을 함께 제시한다.** $p$ 값만으로는 차이의 크기와 시점을 알 수 없다.

다음 절부터 **모수적 생존 모형**으로 넘어간다.
