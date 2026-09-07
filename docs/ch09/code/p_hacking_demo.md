# p-해킹 시연

## 개요

p-해킹은 연구자가 자료 수집과 분석의 유연성을 이용해, 실제로는 아무 효과가 없는 자료에서 통계적으로 유의한 결과를 얻어내는 일이다. 흔한 형태로는 여러 결과변수를 검정하고 유의한 것만 보고하기, $p < 0.05$가 되는 즉시 자료 수집을 멈추기, 부분집단이나 분석 방법을 골라 쓰기가 있다. 이 페이지에서는 이런 관행을 각각 모의실험하여 거짓 양성 비율이 명목 $\alpha = 0.05$ 위로 얼마나 극적으로 부푸는지 보인다.

## 정직한 검정, 귀무가설 아래에서

$H_0$이 참이고 수준 $\alpha$에서 검정하면 정확히 $\alpha$의 비율만 기각된다. p-값은 $\text{Uniform}(0,1)$ 분포를 따른다:

$$
P(p \leq t \mid H_0) = t \quad \text{for } t \in [0,1].
$$

### 코드

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

false_pos_rate = np.mean(pvals < 0.05)
print(f"False positive rate: {false_pos_rate:.4f}  (expected: 0.05)")
```

p-값의 히스토그램은 사실상 평평하여 $H_0$ 아래의 균등성을 확인해 준다.

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

### 코드

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

phack_rate = np.mean(min_pvals < 0.05)
print(f"Cherry-pick rate: {phack_rate:.4f}  (theoretical: 0.6415)")
```

## 임의 중단

또 다른 형태의 p-해킹은 자료를 모으는 동안 반복해서 들여다보다가 $p < 0.05$가 되는 즉시 멈추는 것이다. 각각의 들여다보기가 타당한 검정을 쓰더라도 순차적인 엿보기가 전체 거짓 양성 비율을 부풀린다.

### 코드

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

stop_rate = np.mean(np.array(stopped_pvals) < 0.05)
print(f"Optional stopping rate: {stop_rate:.4f}")
```

(200개까지 10개마다 확인하여) 최대 20번 엿보면 거짓 양성 비율이 20%를 넘을 수 있다.

## 해석

| 방법 | 기대 거짓 양성 비율 |
|---|---|
| 정직한 단일 검정 | $\alpha = 0.05$ |
| 결과변수 20개 중 고르기 | $\approx 0.64$ |
| 임의 중단 (20번 엿보기) | $\approx 0.20$ |

핵심 교훈은 $\alpha = 0.05$가 제1종 오류율을 통제하는 것은 자료를 보기 전에 분석 계획이 고정되어 있을 때뿐이라는 점이다. 결과변수, 중단 규칙, 부분집단을 사후에 고르는 어떤 유연성도 참 오류율을 부풀리며, 때로는 극적으로 그렇다.

**해법**으로는 분석 계획의 사전등록, 다중비교 보정(Bonferroni, BH), 중간분석을 형식적으로 반영하는 순차검정 방법(알파 소비 함수)이 있다.

## 연습문제

**연습문제 1.** $H_0$ 아래 독립인 p-값에 대해 공식 $P(\min(p_1, \ldots, p_k) < \alpha) = 1 - (1 - \alpha)^k$을 유도하라. 어떤 가정이 결정적인가?

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

---

**연습문제 2.** $\alpha = 0.05$에서 $H_0$ 아래 "유의한" 결과를 적어도 하나 찾을 확률이 90%를 넘으려면 연구자가 독립인 결과변수를 몇 개나 두고 골라야 하는가?

??? success "풀이"

    $1 - 0.95^k > 0.90$, 즉 $0.95^k < 0.10$이어야 한다. 로그를 취하면:

    $$
    k > \frac{\ln 0.10}{\ln 0.95} = \frac{-2.3026}{-0.05129} \approx 44.9.
    $$

    따라서 $k \geq 45$개면 충분하다. 귀무가설 아래에서 독립인 변수 45개를 검정하면 $\alpha = 0.05$에서 거짓 양성이 적어도 하나 나올 확률이 90%를 넘는다. $\square$

---

**연습문제 3.** 연속인 검정통계량에서 $H_0$ 아래 p-값 분포가 $\text{Uniform}(0,1)$인 이유를 설명하라. 검정통계량이 이산이면 어떻게 되는가?

??? success "풀이"

    $H_0$ 아래 누적분포함수가 $F_0$인 연속 검정통계량 $T$에서 p-값은 $p = 1 - F_0(T)$(양측검정이면 $2\min(F_0(T), 1-F_0(T))$)이다. $F_0$이 참 누적분포함수이면 확률적분변환에 의해 $F_0(T) \sim \text{Uniform}(0,1)$이므로 $p \sim \text{Uniform}(0,1)$이다.

    이산 검정통계량에서는 누적분포함수가 계단함수이므로 $F_0(T)$가 유한개의 값만 취한다. 그러면 p-값 분포가 $\text{Uniform}(0,1)$보다 **확률적으로 크다**:

    $$
    P(p \leq \alpha) \leq \alpha \quad \text{for all } \alpha,
    $$

    대부분의 값에서 부등호가 엄격하다. 그래서 이산검정이 보수적이 된다. $\square$

---

**연습문제 4.** 연구자가 (10개마다가 아니라) 새 관측값이 하나 생길 때마다 확인하도록 임의 중단 모의실험을 고쳐라. 거짓 양성 비율에 어떤 영향을 주는가?

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

    관측값마다 확인하면 엿보는 횟수가 최대가 되어 거짓 양성 비율도 가장 높아진다. 비율이 30% 이상까지 갈 수 있어 명목 5%를 크게 웃돈다. 임의 중단의 최악의 경우이다. $\square$

---

**연습문제 5.** 임의 중단에 대한 보정을 제안하라. 자료 수집 중 $K$번 엿볼 계획이라면 전체 제1종 오류율 0.05를 유지하려면 각 중간분석에서 $\alpha$를 어떻게 조정해야 하는가? (힌트: Bonferroni가 한 가지 선택지이고 알파 소비가 또 다른 선택지이다.)

??? success "풀이"

    **Bonferroni 접근:** 매번 엿볼 때 수준 $\alpha/K$에서 검정한다. $K = 20$번이면 각각 $\alpha = 0.05/20 = 0.0025$를 쓴다. 단순하지만 보수적이다.

    **Pocock 경계:** 전체 제1종 오류가 0.05가 되도록 고른 같은 조정 문턱 $\alpha^*$을 매번 쓴다. $K = 20$이면 $\alpha^* \approx 0.003$이다(모의실험이나 순차분석 표로 계산한다).

    **O'Brien-Fleming 경계:** 초기 엿보기에는 아주 엄격한 문턱을 쓰고 자료가 쌓일수록 완화한다. $K$번 중 $k$번째 엿보기에서 임계 $z$-값은 대략 $z_{\alpha/2}/\sqrt{k/K}$이다. 초기에 알파를 거의 쓰지 않아 최종 분석의 검정력을 보존한다.

    **알파 소비 함수 (Lan-DeMets):** 정보 비율 $t \in [0,1]$까지 전체 $\alpha = 0.05$ 중 얼마를 "쓸지" 함수 $\alpha^*(t)$로 지정하는 유연한 틀이다. Pocock과 O'Brien-Fleming을 일반화하며 엿보는 시점을 정확히 미리 정할 필요가 없다. $\square$
