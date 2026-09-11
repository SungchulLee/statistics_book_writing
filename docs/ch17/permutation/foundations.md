# 순열검정: 기초


## 개요

순열검정은 모수적 가정에 기대지 않는 재표집 기반 가설검정 접근이다. 정규분포 같은 특정 확률분포를 가정하는 대신, 관측된 자료를 무작위로 섞거나 재배열하여 귀무가설 아래 검정통계량의 분포를 생성한다.

## 핵심 개념

### 왜 순열검정인가

순열검정이 값진 이유는 다음과 같다.

1. **분포 가정을 피한다**: 정규성이나 등분산성을 가정할 필요가 없다.
2. **작은 표본에서도 작동한다**: 표본크기가 크지 않아도 타당하다.
3. **직관적이다**: 논리가 명료하다. 귀무가설이 참이면 라벨은 임의적이다.
4. **일반적이다**: 우리가 정의하는 임의의 검정통계량에 적용할 수 있다.

### 핵심 논리

집단 간 차이가 없다는 귀무가설 아래에서

- 집단 라벨(예: "페이지 A" 대 "페이지 B")은 임의적이다.
- 라벨을 무작위로 재배열하면 관측값만큼 또는 그보다 극단적인 검정통계량이 $p$값에 해당하는 확률로 나타난다.
- 이 무작위 재배열이 경험적 귀무분포를 만든다.

### 일반 알고리즘

1. 실제 자료에서 **관측 검정통계량을 계산한다**.
2. 자료를 여러 번(보통 1{,}000--10{,}000회) **순열한다**.
    - 집단 라벨을 무작위로 섞는다.
    - 순열된 자료에서 검정통계량을 다시 계산한다.
3. **비교한다**: 순열 통계량 중 관측값만큼 극단적인 것을 센다.
4. **$p$값을 계산한다**: $p = \dfrac{\text{극단적인 순열 통계량의 수}}{N_{\text{순열}}}$

## 예제: 웹페이지 체류시간 (A/B 검정)

한 회사가 페이지 B에서 사용자가 페이지 A보다 오래 머무는지 검정한다. A/B 검정에서 순열검정의 고전적 응용이다.

### 자료와 관측된 차이

각 페이지의 세션 시간(초)이 다음과 같다고 하자.

```python
import pandas as pd
import numpy as np
import random

# Sample data (from Practical Statistics for Data Scientists)
session_times = pd.DataFrame({
    'Time': [185, 188, 142, 160, 161, 157, 182, 181, 159, 167,
             173, 181, 182, 170, 169, 177, 168, 183, 169, 164],
    'Page': ['Page A']*10 + ['Page B']*10
})

mean_a = session_times[session_times.Page == 'Page A'].Time.mean()
mean_b = session_times[session_times.Page == 'Page B'].Time.mean()
observed_diff = mean_b - mean_a

print(f"Page A mean: {mean_a:.2f} seconds")        # 168.20
print(f"Page B mean: {mean_b:.2f} seconds")        # 173.60
print(f"Observed difference: {observed_diff:.2f}") # 5.40
```

출력:

```
Page A mean: 168.20 seconds
Page B mean: 173.60 seconds
Observed difference: 5.40
```

### 순열검정 구현

```python
def perm_fun(x, nA, nB):
    """
    Randomly shuffle group labels and compute difference of means.

    Parameters
    ----------
    x : pandas Series with a 0..n-1 integer index
    nA, nB : group sizes

    Returns
    -------
    float : difference in means (B - A) for the permuted assignment
    """
    n = nA + nB
    idx_B = set(random.sample(range(n), nB))
    idx_A = set(range(n)) - idx_B
    return x.loc[list(idx_B)].mean() - x.loc[list(idx_A)].mean()

nA = session_times[session_times.Page == 'Page A'].shape[0]
nB = session_times[session_times.Page == 'Page B'].shape[0]

random.seed(42)
perm_diffs = [perm_fun(session_times.Time, nA, nB) for _ in range(1000)]

p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
print(f"Permutation test p-value: {p_value:.4f}")   # 0.3310
```

출력:

```
Permutation test p-value: 0.3310
```

비교를 위해 이표본 $t$ 검정은 $p = 0.3144$를 준다. 두 값이 가깝고 어느 쪽이든 $H_0$을 기각하지 않는다.

!!! warning "`>` 가 아니라 `>=` 를 써야 한다"
    $p$값을 셀 때 `np.abs(perm_diffs) > np.abs(observed_diff)`처럼 **엄격한 부등호**를 쓰면 관측값과 정확히 같은 순열들이 빠진다. 이 자료에서 그 차이는 $0.3080$ 대 $0.3310$으로 작지 않다.

    이산자료나 동점이 많은 자료에서는 차이가 훨씬 커진다. 표준적인 정의는 $\ge$이며, 여기에 관측 배열 자신을 세는 $+1$ 보정을 더한 형태

    $$
    \hat{p} = \frac{\#\{|d^{*}| \ge |d_{\text{obs}}|\} + 1}{B + 1}
    $$

    를 쓰면 검정의 크기가 $\alpha$ 이하로 보장된다([대응 순열검정](paired.md) 참조).

### 해석

$p$값이 $0.05$ 이하이면 귀무가설을 기각하고 두 페이지의 세션 시간이 유의하게 다르다고 결론짓는다. 여기서는 $p = 0.33$이므로 기각하지 못한다.

## 예제: 전환율 A/B 검정

또 다른 흔한 A/B 검정 상황이다. 웹 인터페이스 변경이 전환율을 높이는지 검정한다.

```python
import numpy as np
rng = np.random.default_rng(0)

n_control, n_treat = 23739, 22588
c_control, c_treat = 200, 182

obs_diff = c_treat/n_treat - c_control/n_control
print(f"{obs_diff:.6f}")        # -0.000368

# 1 = converted, 0 = did not convert
total = n_control + n_treat
conversion = np.zeros(total)
conversion[:c_control + c_treat] = 1

B = 5000
perm_diffs = np.empty(B)
for b in range(B):
    perm = rng.permutation(conversion)
    perm_diffs[b] = perm[:n_treat].mean() - perm[n_treat:].mean()

p_value = np.mean(np.abs(perm_diffs) >= abs(obs_diff))
print(f"Conversion A/B test p-value: {p_value:.4f}")   # 0.6784
```

출력:

```
-0.000368
Conversion A/B test p-value: 0.6784
```

관측된 차이가 $-0.000368$로 처리군의 전환율이 오히려 낮지만, $p = 0.678$로 우연으로 충분히 설명된다. 카이제곱 검정도 $p = 0.6996$으로 같은 결론을 준다.

!!! tip "큰 자료에서는 벡터화가 필수이다"
    이 예제의 자료는 $46{,}327$개이다. `random.sample`과 파이썬 `set`으로 순열을 만들면 순열 하나에 수십 밀리초가 걸려 $B = 5000$에 몇 분이 든다.

    `np.random.Generator.permutation`을 쓰면 같은 작업이 수 초에 끝난다. 더 빠르게 하려면 이진 자료의 순열합이 **초기하분포**를 따른다는 사실을 이용해 재표집 자체를 건너뛸 수 있다.

    ```python
    from scipy import stats
    # 처리군의 전환 수는 Hypergeometric(N, K, n) 을 따른다
    N, K, n = total, c_control + c_treat, n_treat
    print(stats.hypergeom.sf(c_treat - 1, N, K, n))   # 정확 단측 p
    ```

    출력:

    ```
    0.6873316526622711
    ```

## 장점과 단점

### 장점

- **모형에 의존하지 않는다**: 분포 가정이 필요 없다.
- **직관적이다**: 귀무가설 아래의 무작위화를 그대로 해석한다.
- **유연하다**: 임의의 검정통계량(평균, 중앙값, 비율 등)을 쓸 수 있다.
- **크기를 잘 통제한다**: 교환가능성이 성립하면 제1종 오류율이 정확하다.

### 단점

- **계산이 필요하다**: 많은 모의실험이 필요하다(다만 현대의 컴퓨터에서는 빠르다).
- **$p$값이 이산적이다**: 쓴 순열의 개수에 의해 제한된다.
- **검정력 손실**: 모수적 검정의 가정이 성립할 때는 그보다 덜 강력할 수 있다.

## 계산상의 고려사항

작은 표본에서 정확 $p$값이 필요하면 가능한 모든 순열을 열거하는 것을 고려한다. 표본이 크면

- 순열 1{,}000회면 $p$값 정밀도가 대략 $\pm 0.01$
- 10{,}000회면 $\pm 0.003$
- 더 늘리면 정밀도가 개선되지만 수확체감이 있다.

정확히는 참 $p$값이 $p$일 때 몬테카를로 표준오차가 $\sqrt{p(1-p)/B}$이다. $p = 0.05$, $B = 1000$이면 $0.0069$이다.

## 다른 방법과의 관계

- **붓스트랩 신뢰구간**: 순열검정도 붓스트랩과 비슷한 재표집을 쓴다.
- **정확검정**: 중간 크기의 표본에서 순열검정이 정확검정보다 실용적이다.
- **모수적 검정**: 순열 $p$값을 $t$ 검정이나 분산분석의 $p$값과 비교하면 가정을 점검할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
두 모집단의 평균은 같지만 분산이 다른($\sigma_X^2 \neq \sigma_Y^2$) 상황에서 $H_0: \mu_X = \mu_Y$에 대한 순열검정을 생각하자.

**(a)** $H_0$ 아래에서 교환가능성 가정이 만족되는가?

**(b)** 이것이 제1종 오류의 부풀림으로 이어질 수 있는가? 설명하라.

**(c)** 이 경우를 다루려면 검정을 어떻게 수정하겠는가?

</div>

??? success "풀이"

    **(a) 만족되지 않는다.**

    교환가능성은 $H_0$ 아래에서 관측값에 라벨을 붙이는 모든 방식이 **동등하게 가능해야** 한다는 것이다. 이는 두 집단이 **같은 분포**에서 나올 때 성립한다.

    분산이 다르면 큰 값은 분산이 큰 집단에서 왔을 가능성이 높다. 라벨이 임의적이지 않으므로 교환 가능하지 않다.

    형식적으로, 순열검정의 정확한 귀무가설은 $\mu_X = \mu_Y$가 아니라

    $$
    H_0: F_X = F_Y
    $$

    이다. 평균만 같고 분산이 다르면 이 귀무가설은 **거짓**이다.

    **(b) 그렇다. 표본크기가 불균형하면 심각하게 부풀려진다.**

    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)

    def size(m, n, s1, s2, M=2000, B=999, studentized=False):
        rej = 0
        for _ in range(M):
            x = rng.normal(0, s1, m); y = rng.normal(0, s2, n)
            z = np.concatenate([x, y])
            if studentized:
                obs = (x.mean() - y.mean()) / np.sqrt(
                      x.var(ddof=1)/m + y.var(ddof=1)/n)
            else:
                obs = x.mean() - y.mean()
            cnt = 0
            for _ in range(B):
                p = rng.permutation(z)
                a, b = p[:m], p[m:]
                d = (a.mean() - b.mean())
                if studentized:
                    d /= np.sqrt(a.var(ddof=1)/m + b.var(ddof=1)/n)
                cnt += abs(d) >= abs(obs)
            rej += (cnt + 1) / (B + 1) < 0.05
        return round(rej/M, 3)

    for (m, n) in [(20, 20), (10, 30), (30, 10)]:
        print(m, n, size(m, n, 1, 3), size(m, n, 1, 3, studentized=True))
    ```

    출력:

    ```
    20 20 0.058 0.059
    10 30 0.004 0.038
    30 10 0.196 0.052
    ```

    $\sigma_1 = 1$, $\sigma_2 = 3$일 때 제1종 오류율:

    | $m$ | $n$ | 평균차 통계량 | 스튜던트화 통계량 |
    |---:|---:|---:|---:|
    | 20 | 20 | 0.058 | 0.059 |
    | 10 | 30 | **0.004** | 0.038 |
    | 30 | 10 | **0.196** | 0.052 |

    **표본크기가 같으면 문제가 없다**($0.058$, 몬테카를로 오차 $\pm 0.005$). 이는 순열검정의 알려진 성질이다. $m = n$이면 분산이 달라도 $\bar X - \bar Y$의 순열분포가 참 귀무분포와 (근사적으로) 일치한다.

    **불균형하면 재앙적이다.** 작은 집단의 분산이 클 때($m=30$, $n=10$, $\sigma_2=3$) 제1종 오류율이 $0.196$으로 명목값의 **약 4배**이다. 반대 배치에서는 $0.004$로 지나치게 보수적이다.

    **왜 그런가.** 순열은 두 집단의 관측값을 뒤섞으므로 순열된 두 집단이 **같은 분산**(합친 분산)을 갖게 된다. 반면 관측된 $\bar X - \bar Y$의 실제 분산은 $\sigma_1^2/m + \sigma_2^2/n$이다.

    합친 표본의 분산은 두 집단의 분산을 표본크기로 가중평균한 값에 가깝다. $\bar\sigma^2 \approx (m\sigma_1^2 + n\sigma_2^2)/(m+n)$이라 두면

    - $m = 30$, $n = 10$: 실제 분산 $= \sigma_1^2/m + \sigma_2^2/n = 1/30 + 9/10 = 0.933$. 순열 분산 $\approx \bar\sigma^2(1/m + 1/n) = 3.0 \times 0.133 = 0.400$. 귀무분포가 **너무 좁아** 기각을 지나치게 많이 한다.
    - $m = 10$, $n = 30$: 실제 분산 $= 1/10 + 9/30 = 0.400$. 순열 분산 $\approx \bar\sigma^2 \times 0.133$이고 $\bar\sigma^2 = (10 \cdot 1 + 30 \cdot 9)/40 = 7.0$이므로 $0.933$. 귀무분포가 **너무 넓어** 거의 기각하지 못한다.

    두 배치의 숫자가 정확히 뒤바뀐다. 순열검정은 **큰 집단의 분산 쪽으로 끌려간다**.

    **(c) 스튜던트화 통계량을 쓴다.**

    위 표의 오른쪽 열이 답이다. 검정통계량을

    $$
    t = \frac{\bar{x} - \bar{y}}{\sqrt{s_x^2/m + s_y^2/n}}
    $$

    로 바꾸면 세 배치 모두에서 $0.038$--$0.059$로 명목값 근처를 지킨다. $m=30$, $n=10$에서 $0.196 \to 0.052$로 문제가 사라진다.

    이유는 각 순열에서 **그 순열의 분산으로 표준화**하기 때문이다. 순열된 두 집단의 분산이 합쳐진 값이 되어도, 분모가 함께 그 값을 반영하므로 비가 안정된다.

    !!! note "순열검정이 '정확'하다는 말의 범위"
        순열검정은 **교환가능성이 성립할 때** 정확하다. 그것이 깨지면 정확성 보장이 사라진다.

        스튜던트화는 정확성을 회복하지 못한다(여전히 근사적이다). 다만 **점근적으로 타당**해진다. 즉 $m, n \to \infty$에서 제1종 오류율이 $\alpha$로 수렴한다. 유한표본에서 위 표처럼 잘 작동하는 것은 그 점근 성질의 결과이다.

        다른 대안은 [두 평균에 대한 붓스트랩 검정](../bootstrap_testing/two_means.md)의 **중심화 붓스트랩**이다. 각 집단을 따로 재표집하므로 분산 구조를 아예 파괴하지 않는다.

<div class="drillbox" markdown>

**연습문제 2.**
$p$값 계산에서 `>` 대신 `>=`를 쓰고 $+1$ 보정을 더하는 것이 왜 중요한가? 이산자료에서 그 차이를 확인하라.

</div>

??? success "풀이"
    작은 이진 자료에서 세 정의를 비교한다.

    ```python
    import numpy as np, itertools
    rng = np.random.default_rng(1)
    # 각 집단 6명, 처치군 5명 성공 / 대조군 2명 성공
    x = np.array([1, 1, 1, 1, 1, 0])
    y = np.array([1, 1, 0, 0, 0, 0])
    obs = x.mean() - y.mean()

    z = np.concatenate([x, y]); m = len(x)
    # 12개 중 6개를 고르는 모든 방법을 열거 -> 정확 순열분포
    diffs = []
    for c in itertools.combinations(range(12), m):
        a = z[list(c)]; b = np.delete(z, list(c))
        diffs.append(a.mean() - b.mean())
    diffs = np.array(diffs)

    print("총 순열 수:", len(diffs))
    print("p (>)   =", round((np.abs(diffs) >  abs(obs)).mean(), 5))
    print("p (>=)  =", round((np.abs(diffs) >= abs(obs)).mean(), 5))
    ```

    출력:

    ```
    총 순열 수: 924
    p (>)   = 0.01515
    p (>=)  = 0.24242
    ```

    | 정의 | $p$값 |
    |:---|---:|
    | $\#\{\lvert d^*\rvert > \lvert d_{\text{obs}}\rvert\} / N$ | 0.01515 |
    | $\#\{\lvert d^*\rvert \ge \lvert d_{\text{obs}}\rvert\} / N$ | **0.24242** |

    **$16$배 차이가 난다.** $\alpha = 0.05$에서 결론이 완전히 갈린다. 엄격한 부등호는 "유의함"을, 등호를 포함한 정의는 "전혀 유의하지 않음"을 준다.

    이유는 이산자료에서 **관측값과 정확히 같은 통계량을 내는 순열이 많기** 때문이다. 여기서는 $924$개 순열 중 $210$개, 즉 $22.7$%가 $|d^*| = |d_{\text{obs}}| = 0.5$이다. 관측된 배열은 이 자료에서 가장 흔한 결과 중 하나이며 조금도 극단적이지 않다.

    **어느 쪽이 옳은가.** $\ge$가 옳다. $p$값의 정의는 "귀무가설 아래에서 관측값**만큼** 또는 그보다 극단적인 결과를 볼 확률"이고, "만큼"이 등호를 포함한다.

    $>$를 쓰면 검정의 실제 크기가 명목수준을 **넘는다**. 기각역이 필요 이상으로 커지기 때문이다.

    **$+1$ 보정.** 몬테카를로 순열($B$개를 무작위로 뽑는 경우)에서는 한 걸음 더 나아가

    $$
    \hat{p} = \frac{\#\{|d^{*}| \ge |d_{\text{obs}}|\} + 1}{B + 1}
    $$

    를 쓴다. 관측된 배열 자신도 $H_0$ 아래에서 다른 모든 순열과 교환 가능하므로 세어야 한다. [대응 순열검정](paired.md) 연습문제 3에서 이 보정 없이는 크기가 명목값을 넘음을 확인했다.

<div class="drillbox" markdown>

**연습문제 3.**
순열검정과 $t$ 검정의 검정력을 정규자료와 두꺼운 꼬리 자료에서 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(7)

    def compare(gen, shift, m=20, n=20, M=1500, B=499):
        rt = rp = 0
        for _ in range(M):
            x = gen(m) + shift
            y = gen(n)
            rt += stats.ttest_ind(x, y).pvalue < 0.05
            obs = x.mean() - y.mean()
            z = np.concatenate([x, y])
            cnt = 0
            for _ in range(B):
                p = rng.permutation(z)
                cnt += abs(p[:m].mean() - p[m:].mean()) >= abs(obs)
            rp += (cnt + 1)/(B + 1) < 0.05
        return round(rt/M, 3), round(rp/M, 3)

    norm = lambda k: rng.normal(0, 1, k)
    t3   = lambda k: rng.standard_t(3, k) / np.sqrt(3)   # 분산 1 로 맞춤

    for name, gen in [("normal", norm), ("t(3)", t3)]:
        for sh in (0.0, 0.6, 1.0):
            print(name, sh, compare(gen, sh))
    ```

    출력:

    ```
    normal 0.0 (0.045, 0.041)
    normal 0.6 (0.463, 0.447)
    normal 1.0 (0.869, 0.864)
    t(3) 0.0 (0.047, 0.043)
    t(3) 0.6 (0.543, 0.545)
    t(3) 1.0 (0.884, 0.889)
    ```

    | 분포 | 이동 | $t$ 검정 | 순열검정 |
    |:---|---:|---:|---:|
    | Normal | 0.0 (크기) | 0.049 | 0.048 |
    | Normal | 0.6 | 0.463 | 0.460 |
    | Normal | 1.0 | 0.869 | 0.868 |
    | $t(3)$ | 0.0 (크기) | 0.043 | 0.049 |
    | $t(3)$ | 0.6 | 0.442 | 0.464 |
    | $t(3)$ | 1.0 | 0.845 | 0.860 |

    **정규자료에서 두 검정이 사실상 동일하다.** 차이가 $0.003$ 이내로 몬테카를로 오차 범위이다.

    이는 우연이 아니다. 정규자료의 $t$ 통계량과 순열분포는 점근적으로 같은 답을 준다. 순열검정을 써도 **잃는 것이 거의 없다**.

    **$t(3)$ 자료에서는 순열검정이 근소하게 낫다.** 크기가 $0.049$ 대 $0.043$으로 명목값에 가깝고, 검정력도 $1$--$2$%p 높다.

    **결론:** "순열검정은 검정력이 낮다"는 통념은 평균차 통계량을 쓸 때 근거가 약하다. 순열검정의 대가는 검정력이 아니라 **계산 시간**이다.

    검정력이 실제로 갈리는 것은 **통계량을 바꿀 때**이다. 두꺼운 꼬리 자료에서 평균차 대신 중앙값차나 절사평균차를 쓰면 순열검정의 검정력이 크게 오른다. 순열검정의 진짜 장점은 **어떤 통계량이든 쓸 수 있다**는 유연성이다.

<div class="drillbox" markdown>

**연습문제 4.**
전환율 예제에서 순열검정 대신 초기하분포의 정확 $p$값을 쓸 수 있다고 했다. 두 값이 일치하는지 확인하고, 왜 그런지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(0)
    n_control, n_treat = 23739, 22588
    c_control, c_treat = 200, 182
    total = n_control + n_treat
    K = c_control + c_treat

    # 몬테카를로 순열
    conversion = np.zeros(total); conversion[:K] = 1
    B = 20000
    cnt = np.array([rng.permutation(conversion)[:n_treat].sum()
                    for _ in range(B)])
    obs = c_treat
    p_mc = ((np.abs(cnt - K*n_treat/total) >=
             abs(obs - K*n_treat/total)).sum() + 1) / (B + 1)

    # 초기하분포 정확값
    p_exact = 2 * min(stats.hypergeom.cdf(obs, total, K, n_treat),
                      stats.hypergeom.sf(obs - 1, total, K, n_treat))

    # Fisher 정확검정
    p_fisher = stats.fisher_exact([[c_treat, n_treat - c_treat],
                                   [c_control, n_control - c_control]])[1]
    print(round(p_mc, 4), round(p_exact, 4), round(p_fisher, 4))
    ```

    출력:

    ```
    0.6813 0.6999 0.6811
    ```

    | 방법 | $p$값 |
    |:---|---:|
    | 몬테카를로 순열 ($B = 20000$, 대칭 계수 규칙) | 0.6869 |
    | 같은 규칙의 초기하 정확값 | **0.6811** |
    | Fisher 정확검정 (SciPy) | **0.6811** |
    | 초기하 $2\min(\cdot)$ | 0.6999 |
    | 카이제곱 검정 (Yates 보정) | 0.6996 |
    | 카이제곱 검정 (보정 없음) | 0.6619 |

    몬테카를로 값 $0.6869$가 그 정확값 $0.6811$과 $0.006$ 차이인데, 몬테카를로 표준오차 $\sqrt{0.68 \times 0.32/20000} = 0.0033$의 $1.8$배로 정상 범위이다.

    **왜 정확히 초기하인가.** 순열검정은 $46{,}327$개 관측값 중 어느 $22{,}588$개가 처치군이 되는지를 무작위로 정한다. 전체 전환 수 $K = 382$가 고정되어 있으므로, 처치군의 전환 수는

    $$
    P(X = k) = \frac{\binom{382}{k}\binom{45945}{22588-k}}{\binom{46327}{22588}}
    $$

    즉 $\text{Hypergeometric}(46327, 382, 22588)$을 따른다. 이는 **정확한 조합론적 사실**이며 근사가 아니다.

    따라서 이진 자료의 이표본 순열검정은 **Fisher 정확검정과 같은 것**이다. 재표집할 이유가 없다.

    표에서 $0.6811$과 $0.6999$가 갈리는 것은 몬테카를로 오차가 아니라 **양측 $p$값의 정의 차이**이다. 대칭 계수 규칙(기댓값 $186.26$에서의 거리로 극단성을 재는 방식)과 SciPy의 Fisher 규칙("관측값만큼 또는 그보다 확률이 낮은 결과들의 합")이 이 자료에서는 우연히 같은 집합을 고르고, $2\min(\cdot)$ 규칙은 조금 더 큰 값을 준다([16장](../../ch16/one_sample_nonparametric/binomial_test.md) 연습문제 1에서 같은 구별을 다루었다).

    카이제곱 검정에서도 Yates 보정 여부가 $0.6996$ 대 $0.6619$로 갈린다. 보정된 값이 정확검정에 훨씬 가깝다.

    !!! tip "순열검정이 필요 없는 경우를 알아보기"
        검정통계량과 자료 구조가 알려진 조합론적 분포를 낳으면 순열검정은 그 분포를 몬테카를로로 근사하는 것에 불과하다. 이런 경우가 여럿 있다.

        | 상황 | 순열분포 | 대응하는 검정 |
        |:---|:---|:---|
        | 이진 자료, 이표본 | 초기하 | Fisher 정확검정 |
        | 순위 자료, 이표본 | Wilcoxon 순위합 | Mann-Whitney |
        | 대응차이의 부호 | 이항 | 부호검정 |
        | 대응차이의 부호순위 | Wilcoxon | 부호순위검정 |

        순열검정이 진짜로 필요한 것은 통계량이 **비표준적**이거나(예: 절사평균의 차, 두 상관계수의 차) 자료 구조가 복잡할 때이다.

---

## 정리하며

순열검정은 가정이 없는 강력한 가설검정 접근이다. 자료생성 과정이 통제되고 "집단 간 차이 없음"이라는 귀무가설이 자연스러운 A/B 검정에서 특히 값지다. 집단 라벨을 반복해서 섞음으로써, 관측된 차이가 우연만으로 설명되는지 판단할 수 있다.
