# p-해킹 시연

## 개요

p-해킹은 연구자가 자료 수집과 분석의 유연성을 이용해, 실제로는 아무 효과가 없는 자료에서 통계적으로 유의한 결과를 얻어내는 일이다. 흔한 형태로는 여러 결과변수를 검정하고 유의한 것만 보고하기, $p < 0.05$가 되는 즉시 자료 수집을 멈추기, 부분집단이나 분석 방법을 골라 쓰기가 있다. 이 페이지에서는 이런 관행을 각각 모의실험하여 거짓 양성 비율이 명목 $\alpha = 0.05$ 위로 얼마나 극적으로 부푸는지 보인다.

## 정직한 검정, 귀무가설 아래에서

$H_0$이 참이고 수준 $\alpha$에서 검정하면 정확히 $\alpha$의 비율만 기각된다. p-값은 $\text{Uniform}(0,1)$ 분포를 따른다:

$$
P(p \leq t \mid H_0) = t \quad \text{for } t \in [0,1].
$$

<div class="codebox" markdown>

### 예제 1. 표본을 몰래 늘려 가며 보기 { .eg }

```python
import numpy as np
from scipy import stats

np.random.seed(42)

n_experiments = 10_000
n_per_group = 30
pvals = np.zeros(n_experiments)

for i in range(n_experiments):
    a = np.random.normal(0, 1, n_per_group)
    b = np.random.normal(0, 1, n_per_group)
    _, pvals[i] = stats.ttest_ind(a, b)

# 두 집단 모두 같은 N(0,1)에서 뽑았으니 H0가 참인 상황이다.
# 그런데도 5%는 기각된다. 그것이 alpha의 정의다.
false_pos_rate = np.mean(pvals < 0.05)
print(f"False positive rate: {false_pos_rate:.4f}  (expected: 0.05)")
```

출력:

```
False positive rate: 0.0508  (expected: 0.05)
```

10,000번 중 508번이 "유의하다"고 나왔다. 정직하게 한 번만 검정하면 거짓 양성은 정확히 명목 수준에 머문다. 아래에서 무너지는 것은 이 전제다.

p-값의 히스토그램은 사실상 평평하여 $H_0$ 아래의 균등성을 확인해 준다.

</div>

## 여러 결과변수 중 골라 쓰기

연구자가 독립인 결과변수 $k$개를 검정하고 가장 작은 p-값만 보고하면, $H_0$ 아래에서 "유의한" 결과를 적어도 하나 찾을 확률은

$$
P(\min(p_1, \ldots, p_k) < \alpha) = 1 - (1 - \alpha)^k.
$$

$k = 20$이고 $\alpha = 0.05$이면:

$$
1 - (1 - 0.05)^{20} = 1 - 0.95^{20} \approx 0.64.
$$

거짓 양성 비율이 5%에서 64%로 뛴다.

<div class="codebox" markdown>

### 예제 2. 결과변수를 여러 개 재기 { .eg }

```python
n_outcomes = 20
min_pvals = np.zeros(1000)

for i in range(1000):
    ps = []
    for _ in range(n_outcomes):
        a = np.random.normal(0, 1, 30)
        b = np.random.normal(0, 1, 30)
        _, p = stats.ttest_ind(a, b)
        ps.append(p)
    min_pvals[i] = min(ps)

# 20개를 검정하고 그중 **가장 작은** p-값만 보고하는 상황을 흉내 낸다.
phack_rate = np.mean(min_pvals < 0.05)
print(f"Cherry-pick rate: {phack_rate:.4f}  (theoretical: 0.6415)")
```

출력:

```
Cherry-pick rate: 0.6580  (theoretical: 0.6415)
```

거짓 양성 비율이 5%에서 66%로 뛴다. 실제로 아무 효과도 없는데 세 번에 두 번은 "유의한 결과"를 손에 쥔다는 뜻이다.

모의실험이 1,000회뿐이라 표준오차가 1.5%p 정도이므로 0.658은 이론값 0.6415와 어긋나지 않는다.

여기서 결정적인 것은 20개를 검정했다는 사실 자체가 아니라 **그중 하나만 보고한다는 점**이다. 20개를 모두 보고하고 보정했다면 문제가 없다.

</div>

## 임의 중단

또 다른 형태의 p-해킹은 자료를 모으는 동안 반복해서 들여다보다가 $p < 0.05$가 되는 즉시 멈추는 것이다. 각각의 들여다보기가 타당한 검정을 쓰더라도 순차적인 엿보기가 전체 거짓 양성 비율을 부풀린다.

<div class="codebox" markdown>

### 예제 3. p-해킹의 결과 모으기 { .eg }

```python
n_experiments = 1000
n_max = 200
check_interval = 10

stopped_pvals = []
for _ in range(n_experiments):
    a, b = [], []
    for n in range(check_interval, n_max + 1, check_interval):
        a.extend(np.random.normal(0, 1, check_interval).tolist())
        b.extend(np.random.normal(0, 1, check_interval).tolist())
        _, p = stats.ttest_ind(a, b)
        if p < 0.05:
            stopped_pvals.append(p)
            break
    else:
        stopped_pvals.append(p)

# for-else 구문이다. break 없이 반복이 끝나면 else가 실행된다.
# 즉 20번을 다 엿봐도 유의하지 않았던 경우에는 마지막 p-값을 기록한다.
stop_rate = np.mean(np.array(stopped_pvals) < 0.05)
print(f"Optional stopping rate: {stop_rate:.4f}")
```

출력:

```
Optional stopping rate: 0.2380
```

역시 두 집단이 같은 분포에서 나온 자료인데 24%가 "유의하다"고 나온다. 각각의 검정은 완전히 정당했고 어떤 자료도 버리지 않았다는 점이 이 예제를 불편하게 만든다. 문제는 오직 **언제 멈출지를 자료를 보고 정했다**는 데 있다.

임상시험에서 중간분석을 할 때 알파 소비 함수 같은 형식적 절차를 반드시 쓰는 이유가 이것이다.

(200개까지 10개마다 확인하여) 최대 20번 엿보면 거짓 양성 비율이 20%를 넘을 수 있다.

</div>

## 해석

| 방법 | 기대 거짓 양성 비율 |
|---|---|
| 정직한 단일 검정 | $\alpha = 0.05$ |
| 결과변수 20개 중 고르기 | $\approx 0.64$ |
| 임의 중단 (20번 엿보기) | $\approx 0.20$ |

핵심 교훈은 $\alpha = 0.05$가 제1종 오류율을 통제하는 것은 자료를 보기 전에 분석 계획이 고정되어 있을 때뿐이라는 점이다. 결과변수, 중단 규칙, 부분집단을 사후에 고르는 어떤 유연성도 참 오류율을 부풀리며, 때로는 극적으로 그렇다.

**해법**으로는 분석 계획의 사전등록, 다중비교 보정(Bonferroni, BH), 중간분석을 형식적으로 반영하는 순차검정 방법(알파 소비 함수)이 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $H_0$ 아래 독립인 p-값에 대해 공식 $P(\min(p_1, \ldots, p_k) < \alpha) = 1 - (1 - \alpha)^k$을 유도하라. 어떤 가정이 결정적인가?

</div>

??? success "풀이"

    $H_0$ 아래에서 각 $p_i \sim \text{Uniform}(0,1)$이다. 최솟값이 $\alpha$를 넘으려면 모든 p-값이 $\alpha$를 넘어야 한다:

    $$
    P(\min(p_1,\ldots,p_k) \geq \alpha) = \prod_{i=1}^k P(p_i \geq \alpha) = (1-\alpha)^k.
    $$

    여집합을 취하면,

    $$
    P(\min < \alpha) = 1 - (1-\alpha)^k.
    $$

    결정적인 가정은 p-값의 **독립성**이다. 결과변수들이 상관되어 있으면(예: 측정이 겹치면) 실제 확률이 이 공식이 예측하는 것보다 낮을 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $\alpha = 0.05$에서 $H_0$ 아래 "유의한" 결과를 적어도 하나 찾을 확률이 90%를 넘으려면 연구자가 독립인 결과변수를 몇 개나 두고 골라야 하는가?

</div>

??? success "풀이"

    $1 - 0.95^k > 0.90$, 즉 $0.95^k < 0.10$이어야 한다. 로그를 취하면:

    $$
    k > \frac{\ln 0.10}{\ln 0.95} = \frac{-2.3026}{-0.05129} \approx 44.9.
    $$

    따라서 $k \geq 45$개면 충분하다. 귀무가설 아래에서 독립인 변수 45개를 검정하면 $\alpha = 0.05$에서 거짓 양성이 적어도 하나 나올 확률이 90%를 넘는다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 연속인 검정통계량에서 $H_0$ 아래 p-값 분포가 $\text{Uniform}(0,1)$인 이유를 설명하라. 검정통계량이 이산이면 어떻게 되는가?

</div>

??? success "풀이"

    $H_0$ 아래 누적분포함수가 $F_0$인 연속 검정통계량 $T$에서 p-값은 $p = 1 - F_0(T)$(양측검정이면 $2\min(F_0(T), 1-F_0(T))$)이다. $F_0$이 참 누적분포함수이면 확률적분변환에 의해 $F_0(T) \sim \text{Uniform}(0,1)$이므로 $p \sim \text{Uniform}(0,1)$이다.

    이산 검정통계량에서는 누적분포함수가 계단함수이므로 $F_0(T)$가 유한개의 값만 취한다. 그러면 p-값 분포가 $\text{Uniform}(0,1)$보다 **확률적으로 크다**:

    $$
    P(p \leq \alpha) \leq \alpha \quad \text{for all } \alpha,
    $$

    대부분의 값에서 부등호가 엄격하다. 그래서 이산검정이 보수적이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 연구자가 (10개마다가 아니라) 새 관측값이 하나 생길 때마다 확인하도록 임의 중단 모의실험을 고쳐라. 거짓 양성 비율에 어떤 영향을 주는가?

</div>

??? success "풀이"

    ```python
    stopped = []
    for _ in range(1000):
        a, b = [], []
        for n in range(2, 201):  # need at least 2 per group
            a.append(np.random.normal(0, 1))
            b.append(np.random.normal(0, 1))
            if len(a) >= 2:
                _, p = stats.ttest_ind(a, b)
                if p < 0.05:
                    stopped.append(p)
                    break
        else:
            stopped.append(p)
    rate = np.mean(np.array(stopped) < 0.05)
    print(f"Rate with every-observation peeking: {rate:.4f}")
    ```

    출력:

    ```
    Rate with every-observation peeking: 0.4350
    ```

    관측값마다 확인하면 엿보기 횟수가 최대가 되어 거짓 양성 비율도 가장 높아진다. 10개마다 엿보던 24%가 매 관측값마다 엿보자 **44%**로 오른다. 명목 5%의 아홉 배다.

    이론적으로는 표본을 무한정 늘릴 수 있다면 이 비율이 1에 다가간다. 계속 엿보다 보면 언젠가는 $p < 0.05$인 순간이 반드시 오기 때문이다. 여기서 44%에 그친 것은 200개에서 멈추기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 임의 중단에 대한 보정을 제안하라. 자료 수집 중 $K$번 엿볼 계획이라면 전체 제1종 오류율 0.05를 유지하려면 각 중간분석에서 $\alpha$를 어떻게 조정해야 하는가? (힌트: Bonferroni가 한 가지 선택지이고 알파 소비가 또 다른 선택지이다.)

</div>

??? success "풀이"

    **Bonferroni 접근:** 매번 엿볼 때 수준 $\alpha/K$에서 검정한다. $K = 20$번이면 각각 $\alpha = 0.05/20 = 0.0025$를 쓴다. 단순하지만 보수적이다.

    **Pocock 경계:** 전체 제1종 오류가 0.05가 되도록 고른 같은 조정 문턱 $\alpha^*$을 매번 쓴다. $K = 20$이면 $\alpha^* \approx 0.003$이다(모의실험이나 순차분석 표로 계산한다).

    **O'Brien-Fleming 경계:** 초기 엿보기에는 아주 엄격한 문턱을 쓰고 자료가 쌓일수록 완화한다. $K$번 중 $k$번째 엿보기에서 임계 $z$-값은 대략 $z_{\alpha/2}/\sqrt{k/K}$이다. 초기에 알파를 거의 쓰지 않아 최종 분석의 검정력을 보존한다.

    **알파 소비 함수 (Lan-DeMets):** 정보 비율 $t \in [0,1]$까지 전체 $\alpha = 0.05$ 중 얼마를 "쓸지" 함수 $\alpha^*(t)$로 지정하는 유연한 틀이다. Pocock과 O'Brien-Fleming을 일반화하며 엿보는 시점을 정확히 미리 정할 필요가 없다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
여러 **연구자 자유도**를 조합하면 거짓양성률이 얼마나 오르는지 모의실험으로 보여라.

</div>

??? success "풀이"
    **설정.** 참 효과가 **전혀 없는** 세계에서, 연구자가 다음 자유도를 쓴다.

    - 결과변수 3종(두 개의 측정과 그 평균),
    - 공변량 보정 여부 2가지.

    총 6가지 분석 중 유의한 것을 골라 보고한다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(2024)
    n, M = 30, 2_000

    def study(rng):
        g = np.repeat([0, 1], n)
        y1 = rng.normal(0, 1, 2 * n)
        y2 = rng.normal(0, 1, 2 * n)
        cov = rng.normal(0, 1, 2 * n)
        ps = []
        for y in (y1, y2, (y1 + y2) / 2):
            ps.append(stats.ttest_ind(y[g == 0], y[g == 1]).pvalue)
            X = np.column_stack([np.ones(2 * n), g, cov])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            r = y - X @ b
            df = 2 * n - 3
            se = np.sqrt((r @ r / df) * np.linalg.inv(X.T @ X)[1, 1])
            ps.append(2 * stats.t.sf(abs(b[1] / se), df))
        return min(ps)

    cnt = sum(study(rng) < 0.05 for _ in range(M))
    print(f"6가지 분석 중 최소 p < 0.05 인 비율: {cnt / M:.4f}")
    ```

    ```text
    6가지 분석 중 최소 p < 0.05 인 비율: 0.1345
    ```

    **명목 5%가 13.5%로 2.7배가 된다.** 이것도 **아주 온건한** 자유도 조합이다.

    **실제 연구에서 가능한 자유도는 훨씬 많다.**

    | 자유도 | 가짓수 |
    |---|---|
    | 결과변수 선택 | 3~10 |
    | 공변량 조합 | $2^k$ |
    | 이상치 기준 | 3~5 |
    | 변환 여부 | 2~4 |
    | 하위집단 | 4~10 |
    | 표본 중단 시점 | 여러 |
    | 단측/양측 | 2 |

    이들을 곱하면 **수백~수천 가지 분석 경로**가 된다. 시뮬레이션 연구들은 몇 가지만 조합해도 실제 오류율이 **60%를 넘을 수 있다**고 보고한다.

    **"고의가 아니어도" 일어난다.** 연구자는 각 단계에서 "더 적절한" 선택을 했다고 믿는다. 문제는 **그 판단이 결과를 본 뒤에** 이루어진다는 것이다. 이를 **정원의 갈림길(garden of forking paths)** 이라 부른다. 실제로 하나의 경로만 걸었더라도, **다른 자료였다면 다른 경로를 걸었을 것**이므로 오류율이 부풀려진다.

    **대처.**

    1. **사전등록.** 경로를 미리 하나로 고정한다.
    2. **다중우주 분석(multiverse).** 가능한 모든 경로의 결과를 **전부** 보고한다. 결론이 경로에 얼마나 민감한지 드러난다.
    3. **명세곡선 분석(specification curve).** 수백 개 분석의 효과크기를 정렬해 그린다.
    4. **표본 분할.** 절반으로 탐색하고 나머지 절반으로 확증한다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
**엿보는 간격**을 바꾸며 선택적 중지의 거짓양성률을 재고, 표본이 무한정 늘 때 어떻게 되는지 논하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(2024)

    def peek(rng, step, nmax=100, nmin=10):
        x = rng.normal(0, 1, nmax)
        for n in range(nmin, nmax + 1, step):
            s = x[:n]
            t = s.mean() / (s.std(ddof=1) / np.sqrt(n))
            if 2 * stats.t.sf(abs(t), n - 1) < 0.05:
                return True
        return False

    for step in [1, 5, 10, 50]:
        c = sum(peek(rng, step) for _ in range(5_000))
        print(f"{step:3d}개마다 확인: 거짓양성률 {c / 5_000:.4f}  "
              f"(확인 횟수 {len(range(10, 101, step))})")
    ```

    ```text
      1개마다 확인: 거짓양성률 0.2894  (확인 횟수 91)
      5개마다 확인: 거짓양성률 0.2290  (확인 횟수 19)
     10개마다 확인: 거짓양성률 0.2014  (확인 횟수 10)
     50개마다 확인: 거짓양성률 0.0926  (확인 횟수 2)
    ```

    **확인을 자주 할수록 오류율이 오른다.** 매 관측마다 보면 28.9%, 50개마다 보면 9.3%다.

    **그런데 증가가 로그에 가깝다.** 확인 횟수가 2에서 91로 45배 늘었는데 오류율은 9.3%에서 28.9%로 3배만 늘었다. **연속된 확인들이 강하게 상관**되어 있기 때문이다.

    **$n\to\infty$이면 확률 1로 기각한다.** 반복로그법칙에 따르면

    $$
    \limsup_{n\to\infty}\frac{|\bar X_n|\sqrt n}{\sigma\sqrt{2\log\log n}}=1\quad\text{a.s.}
    $$

    이므로 $|\bar X_n|\sqrt n/\sigma$가 $\sqrt{2\log\log n}$ 규모로 **무한히 자주** 커진다. 고정 임계값 1.96을 언젠가 반드시 넘는다.

    **$\log\log n$이 아주 느리게 자란다는 점이 위안이 되지 않는다.** $n=10^6$에서 $\sqrt{2\log\log 10^6}=2.28$이므로, 1.96을 넘는 것은 어렵지 않다.

    **대처 — 세 갈래.**

    1. **표본크기를 미리 고정.** 가장 단순하다.

    2. **군순차 경계.** 확인 횟수를 미리 정하고 임계값을 조정한다. 포콕 경계는 모든 시점에 같은 임계값을 쓴다.

    ```python
    rng = np.random.default_rng(7)
    M = 200_000
    for K in [1, 2, 3, 5, 10]:
        inc = rng.standard_normal((M, K)) / np.sqrt(K)
        z = np.cumsum(inc, axis=1) / np.sqrt(np.arange(1, K + 1) / K)
        c = np.percentile(np.abs(z).max(1), 95)
        print(f"확인 {K:2d}회: 포콕 임계값 {c:.3f}  각 시점의 명목 α "
              f"{2 * stats.norm.sf(c):.4f}")
    ```

    ```text
    확인  1회: 포콕 임계값 1.955  각 시점의 명목 α 0.0506
    확인  2회: 포콕 임계값 2.179  각 시점의 명목 α 0.0293
    확인  3회: 포콕 임계값 2.290  각 시점의 명목 α 0.0220
    확인  5회: 포콕 임계값 2.416  각 시점의 명목 α 0.0157
    확인 10회: 포콕 임계값 2.558  각 시점의 명목 α 0.0105
    ```

    **확인 10회에 각 시점의 $\alpha$가 0.0105면 된다.** 본페로니라면 $0.05/10=0.005$인데, 상관을 반영하면 두 배 느슨하다.

    3. **언제나 타당한 방법.** 신뢰순차나 e-값을 쓰면 **몇 번을 보든 보장이 유지**된다. 폭이 조금 넓어지는 대가를 치른다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**$p$-곡선**으로 문헌의 $p$-해킹을 탐지하는 원리를 설명하고, 한계를 밝혀라.

</div>

??? success "풀이"
    **원리.** 유의한 $p$-값들($p<0.05$)의 분포를 본다.

    | 상황 | 곡선의 모양 |
    |---|---|
    | 참 효과가 있고 검정력이 높음 | **오른쪽으로 기움**(작은 $p$가 많음) |
    | 참 효과가 없음 | **평평함**(균등분포) |
    | $p$-해킹 | **왼쪽으로 기움**(0.05 바로 아래에 몰림) |

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(11)
    M = 200_000
    za = stats.norm.ppf(0.975)
    bins = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    print(f"{'상황':>16s} " + " ".join(f"{f'~{b:.2f}':>8s}" for b in bins))

    # ① 효과 없음
    p = rng.random(M)
    sig = p[p < 0.05]
    h = np.histogram(sig, bins=np.r_[0, bins])[0] / len(sig)
    print(f"{'효과 없음':>16s} " + " ".join(f"{v:8.3f}" for v in h))

    # ② 효과 있음(검정력 0.5, 0.8)
    for pw in [0.50, 0.80]:
        nc = za + stats.norm.ppf(pw)
        p = 2 * stats.norm.sf(np.abs(rng.normal(nc, 1, M)))
        sig = p[p < 0.05]
        h = np.histogram(sig, bins=np.r_[0, bins])[0] / len(sig)
        print(f"{f'효과 있음(검정력 {pw:.0%})':>16s} "
              + " ".join(f"{v:8.3f}" for v in h))
    ```

    ```text
                  상황    ~0.01    ~0.02    ~0.03    ~0.04    ~0.05
             효과 없음    0.202    0.199    0.196    0.205    0.198
     효과 있음(검정력 50%)    0.537    0.176    0.121    0.092    0.075
     효과 있음(검정력 80%)    0.737    0.116    0.066    0.046    0.034
    ```

    **효과가 없으면 균등하다**(각 구간 0.20). **효과가 있으면 작은 $p$에 몰린다**(검정력 80%에서 첫 구간이 0.737, 마지막 구간이 0.034).

    **$p$-해킹의 서명.** 연구자가 "$p<0.05$가 될 때까지" 시도하면, **0.05 바로 아래**에 부자연스럽게 몰린다. 위 표의 어떤 정상적 상황에서도 마지막 구간이 첫 구간보다 크지 않은데, $p$-해킹된 문헌에서는 그런 일이 나타난다.

    **검정 방법.** 유의한 $p$-값들에 대해

    - **오른쪽 기움 검정**: "효과가 있는가"를 본다(피셔 결합 등).
    - **왼쪽 기움 검정**: "$p$-해킹이 있는가"를 본다.
    - **33% 검정**: 검정력이 33% 이상인지 검정. 낮으면 증거가 약하다.

    **한계 다섯.**

    1. **개별 논문에 쓸 수 없다.** $p$-값이 여러 개 필요하며, 보통 수십 개 이상이어야 한다.

    2. **이질성에 취약하다.** 효과크기가 연구마다 다르면 곡선이 섞여 해석이 어렵다.

    3. **$p$-해킹의 형태에 의존한다.** "유의해질 때까지 표본 추가"는 잡아내지만, "여러 결과변수 중 선택"은 곡선을 크게 왜곡하지 않을 수 있다.

    4. **$p$-값 선택이 자의적이다.** 한 논문에서 어느 $p$-값을 뽑을지에 따라 결과가 달라진다. 규칙을 미리 정해야 한다.

    5. **반올림과 보고 관행.** "$p<0.05$"로만 적힌 값은 쓸 수 없다. 소수점 처리가 곡선을 왜곡할 수 있다.

    **관련 도구.** $z$-곡선(검정력 추정), 깔때기 그림(출판 편향), 정수 검정(보고된 통계량의 일관성 검사) 등이 함께 쓰인다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**표본 분할**(탐색용/확증용)이 $p$-해킹을 어떻게 막는지 보이고, 대가를 계산하라.

</div>

??? success "풀이"
    **원리.** 자료를 둘로 나누어

    - **탐색 표본**에서 자유롭게 분석하고 가설을 고른다.
    - **확증 표본**에서 그 가설 하나만 검정한다.

    확증 단계의 자료는 선택에 쓰이지 않았으므로 **$\alpha$가 온전히 유지**된다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(303)
    M, n, k = 20_000, 60, 10        # 결과변수 10개, 총 n명

    def one_trial(rng, split=True, effect=0.0):
        g = np.repeat([0, 1], n // 2)
        y = rng.normal(0, 1, (n, k))
        y[g == 1, 0] += effect                      # 첫 변수에만 참 효과
        if not split:                               # 전체로 고르고 전체로 검정
            ps = [stats.ttest_ind(y[g == 0, j], y[g == 1, j]).pvalue
                  for j in range(k)]
            return min(ps) < 0.05
        h = n // 2
        idx = rng.permutation(n)
        a, b = idx[:h], idx[h:]                     # 탐색 / 확증
        ps = [stats.ttest_ind(y[np.intersect1d(a, np.where(g == 0)[0]), j],
                              y[np.intersect1d(a, np.where(g == 1)[0]), j]).pvalue
              for j in range(k)]
        j = int(np.argmin(ps))                      # 탐색에서 고른 변수
        return stats.ttest_ind(y[np.intersect1d(b, np.where(g == 0)[0]), j],
                               y[np.intersect1d(b, np.where(g == 1)[0]), j]).pvalue < 0.05

    for eff, name in [(0.0, "효과 없음(제1종 오류)"), (0.8, "첫 변수에 효과(검정력)")]:
        a = np.mean([one_trial(rng, False, eff) for _ in range(M // 10)])
        b = np.mean([one_trial(rng, True, eff) for _ in range(M // 10)])
        print(f"{name:24s} 분할 없음 {a:.4f}   분할 {b:.4f}")
    ```

    ```text
    효과 없음(제1종 오류)          분할 없음 0.4040   분할 0.0460
    첫 변수에 효과(검정력)          분할 없음 0.9125   분할 0.3450
    ```

    **분할이 제1종 오류를 0.404에서 0.046으로 되돌린다.** 명목 수준이 회복된다.

    **대가 — 검정력.** 참 효과가 있을 때 0.913에서 0.345로 떨어진다. 두 가지가 겹친다.

    1. **확증 표본이 절반**이라 $\sqrt2$만큼 손해.
    2. **탐색 단계가 잘못된 변수를 고를** 수 있다.

    **다만 "분할 없음 0.913"은 부풀려진 값이다.** 그중 상당 부분이 거짓양성이므로, 공정한 비교가 아니다. 본페로니 보정을 한 전체 표본과 비교하는 것이 옳다.

    **분할 비율.** 흔히 50:50을 쓰지만, 탐색이 쉬우면 확증에 더 많이 배분하는 것이 낫다. 70:30(확증 70)이 권장되기도 한다.

    **교차검증과의 차이.** 기계학습의 교차검증은 **예측 성능 추정**이 목적이고, 여기서의 분할은 **추론의 타당성**이 목적이다. 교차검증처럼 여러 번 나누어 평균 내면 확증의 독립성이 깨진다.

    **현대적 대안 — 선택 후 추론.** 표본을 버리지 않고, **선택 사건에 조건화**한 분포로 추론한다. 라소 등에서 이론이 발전해 있다. 효율적이지만 계산과 이론이 복잡하다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$p$-해킹을 막는 제도적·개인적 장치를 정리하고, 각각의 효과와 한계를 적어라.

</div>

??? success "풀이"

    | 장치 | 무엇을 막는가 | 한계 |
    |---|---|---|
    | **사전등록** | 거의 모든 자유도 | 지키는지 검증이 어렵다 |
    | **등록보고서** | 사전등록 + 출판 편향 | 학술지 수가 제한적 |
    | 자료·코드 공개 | 사후 검증 가능 | 검증할 사람이 있어야 |
    | 다중우주 분석 | 경로 선택 | 결론이 모호해질 수 있다 |
    | 표본 분할 | 선택 후 추론 | 검정력 손실 |
    | 다중비교 보정 | 명시적 다중성 | 숨은 자유도는 못 막음 |
    | 재현 연구 | 모든 것 | 비용, 유인 부족 |
    | 효과크기·구간 보고 | 이분법적 사고 | 관행 변화 필요 |
    | $p$-곡선·$z$-곡선 | 문헌 수준 진단 | 개별 연구에는 무력 |

    **가장 효과적인 것 — 사전등록.** 다른 모든 장치보다 앞선다. 임상시험 등록 의무화 이후 "긍정적 결과"의 비율이 크게 떨어졌다는 분석이 이를 뒷받침한다.

    **개인이 할 수 있는 것.**

    1. **분석 계획을 자료 수집 전에 글로 적는다.** 공개 등록이 부담되면 최소한 연구실 내부 문서로라도 남긴다.

    2. **결과를 보기 전에 코드를 완성한다.** 모의 자료로 파이프라인을 먼저 돌려 본다.

    3. **모든 분석을 기록한다.** 시도했다가 버린 것도 남긴다. 논문에 "보충자료"로 실을 수 있다.

    4. **"이 분석을 결과가 반대였어도 했을까"를 자문한다.** 가장 단순하고 효과적인 점검이다.

    5. **효과크기와 구간을 중심으로 보고한다.** $p<0.05$를 목표로 삼지 않으면 해킹할 동기가 줄어든다.

    **구조적 문제.** $p$-해킹의 근본 원인은 **유인 구조**다.

    - 유의한 결과만 게재된다 → 연구자가 유의성을 추구한다.
    - 게재 수로 평가받는다 → 빠른 결과를 원한다.
    - 재현 연구는 인정받지 못한다 → 아무도 하지 않는다.

    **따라서 개인의 윤리만으로는 해결되지 않는다.** 학술지 정책, 연구비 배분, 평가 제도가 함께 바뀌어야 한다.

    **낙관적 신호.** 등록보고서 제도의 확산, 자료 공개 의무화, 다중우주 분석의 보급, 통계 교육의 변화가 실제로 진행 중이다. **이 절에서 본 도구들이 그 변화의 일부**다.

---

## 정리하며

$p$-해킹은 **분석의 자유도**를 거짓 양성으로 바꾸는 일이다.

- **정직한 검정에서 $p$ 값은 균등분포다.** $H_0$ 아래에서 $P(p\le t)=t$ 이므로 $\alpha=0.05$ 로 검정하면 정확히 $5\%$ 만 기각된다. **이 성질이 무너지는 것이 곧 해킹의 정의다.**
- **세 가지 흔한 형태가 모두 같은 결과를 낳는다.** 여러 결과변수 중 유의한 것만 보고하기, 유의해질 때까지 자료를 더 모으기(선택적 중단), 부분집단이나 분석 방법을 골라 쓰기. 모의실험에서 거짓 양성률이 명목 $5\%$ 를 크게 웃돈다.
- **선택적 중단이 특히 위험하다.** 들여다보며 계속 모으면 $H_0$ 이 참이어도 **언젠가는 거의 반드시** 유의해진다. A/B 검정 플랫폼에서 실제로 벌어지는 일이다.
- **의도가 없어도 일어난다.** "자료를 더 보자", "이 이상치는 빼자" 같은 합리적으로 보이는 판단들이 쌓이면 같은 효과가 난다. **연구자가 부정직할 필요가 없다.**
- **처방은 사전등록이다.** 가설·결과변수·표본크기·분석 계획을 자료 수집 전에 고정하면 자유도 자체가 사라진다.

다음 절 **가족단위 오류율**로 넘어간다. 검정을 여러 번 하는 것 자체가 만드는 문제를 형식화한다.
