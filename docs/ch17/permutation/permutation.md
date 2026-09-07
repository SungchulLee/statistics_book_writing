# 순열검정


## 서론

**순열검정**(permutation test, 또는 무작위화검정 randomization test)은 가설검정을 위한 다재다능한 비모수 방법이다. 자료를 재배열(순열)하여 만든 분포와 관측된 검정통계량을 비교함으로써, 관측값이 귀무가설과 부합하는지 평가한다. 자료의 기저 분포에 대한 가정에 의존하지 않는다.

### 주요 특징

1. **가정이 없다**: 정규성이나 등분산성을 가정하지 않는다. 검정통계량의 분포는 전적으로 자료로부터 결정된다.
2. **유연하다**: 평균, 중앙값, 분산, 상관계수 등 어떤 통계량의 비교에도 쓸 수 있다.
3. **정확검정**: 표본이 작으면 가능한 모든 순열을 고려하므로 정확하다.
4. **근사검정**: 자료가 크면 순열의 일부만 표집하여 근사한다.

---

## 가설

- **귀무가설 ($H_0$)**: 관측된 자료가 같은 분포에서 나왔다. 또는 검정통계량이 집단 변수와 무관하다.
- **대립가설 ($H_1$)**: 관측된 자료가 다른 분포에서 나왔다. 또는 검정통계량이 집단 변수와 연관된다.

---

## 일반 절차

### 1단계: 검정통계량 선택

검정하려는 효과나 차이를 반영하는 통계량을 고른다. 평균차, 중앙값차, 상관계수 등이다.

### 2단계: 관측 검정통계량 계산

원래 자료로 검정통계량을 계산한다.

### 3단계: 순열 생성

자료 라벨이나 집단 배정을 무작위로 섞어 집단 간 연관을 끊는다.

### 4단계: 각 순열의 검정통계량 계산

각 순열에 대해 검정통계량을 계산한다.

### 5단계: 관측 통계량과 비교

관측 통계량을 순열들의 통계량 분포와 비교한다. 관측값만큼 또는 그보다 극단적인 순열의 비율이 $p$값이다.

---

## 평균차에 대한 순열검정

### 예제

**자료:**

- 집단 A: $[8, 7, 9, 10, 6]$
- 집단 B: $[5, 6, 4, 3, 7]$

**가설:**

- $H_0$: 두 집단의 평균이 같다.
- $H_1$: 두 집단의 평균이 다르다.

```python
import numpy as np
import matplotlib.pyplot as plt

def permutation_test(group_a, group_b, n_permutations=10000, seed=0):
    """
    Permutation test for a difference in means.

    Parameters
    ----------
    group_a, group_b : array-like
        Data for each group.
    n_permutations : int
        Number of random permutations.

    Returns
    -------
    p_value : float
        Two-sided p-value with the +1 correction.
    observed_diff : float
        The observed difference in means.
    perm_differences : ndarray
        The permutation distribution.
    """
    rng = np.random.default_rng(seed)
    combined = np.concatenate([group_a, group_b])
    observed_diff = np.mean(group_a) - np.mean(group_b)
    na = len(group_a)

    perm_differences = np.empty(n_permutations)
    for i in range(n_permutations):
        p = rng.permutation(combined)
        perm_differences[i] = p[:na].mean() - p[na:].mean()

    p_value = ((np.abs(perm_differences) >= abs(observed_diff)).sum() + 1) \
              / (n_permutations + 1)
    return p_value, observed_diff, perm_differences

group_a = np.array([8, 7, 9, 10, 6])
group_b = np.array([5, 6, 4, 3, 7])

p_value, observed_diff, perm_dist = permutation_test(group_a, group_b)

print(f"Observed Difference in Means: {observed_diff:.2f}")
print(f"P-value: {p_value:.4f}")

fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(perm_dist, bins=50, edgecolor='black', alpha=0.7,
        label='Permutation Distribution')
ax.axvline(observed_diff, color='red', linestyle='--', linewidth=2,
           label=f'Observed = {observed_diff:.2f}')
ax.axvline(-observed_diff, color='red', linestyle='--', linewidth=2)
ax.legend()
ax.set_xlabel('Difference in Means')
ax.set_ylabel('Frequency')
ax.set_title('Permutation Test Distribution')
plt.show()
```

**출력 예:**

```
Observed Difference in Means: 3.00
P-value: 0.0397
```

$5$% 수준에서는 유의하지 않다. 각 집단이 $5$개뿐이므로 가능한 순열이 $\binom{10}{5} = 252$가지이고, 이를 **전부 열거**하면 정확 $p$값 $10/252 = 0.0397$을 얻는다. 몬테카를로 값이 이와 일치한다.

!!! warning "$t$ 검정과의 차이를 그냥 넘기지 말 것"
    같은 자료의 이표본 $t$ 검정은 $p = 0.0171$로 **유의하다**고 답한다. 순열검정은 $p = 0.0397$이다. 두 결론이 $\alpha = 0.02$ 부근에서 갈린다.

    차이의 원인은 이 표본이 너무 작다는 것이다. 정확 순열분포에서 $|d^*| \ge 3$인 순열은 $252$개 중 $10$개뿐이고, 가능한 $p$값의 최솟값은 $2/252 = 0.0079$이다. $t$ 분포는 이 이산성을 무시하고 매끄러운 꼬리를 가정한다.

    작은 표본에서는 **정확 순열 $p$값이 더 믿을 만하다**.

---

## 상관에 대한 순열검정

두 변수 사이의 관측된 상관이 유의한지 검정한다.

```python
import numpy as np

def permutation_correlation_test(x, y, n_permutations=10000, seed=0):
    """
    Permutation test for the significance of a correlation.

    Only y is shuffled; x stays fixed. This breaks any association
    between the two variables while keeping both marginals intact.
    """
    rng = np.random.default_rng(seed)
    observed_corr = np.corrcoef(x, y)[0, 1]

    perm_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_corrs[i] = np.corrcoef(x, rng.permutation(y))[0, 1]

    p_value = ((np.abs(perm_corrs) >= abs(observed_corr)).sum() + 1) \
              / (n_permutations + 1)
    return p_value, observed_corr, perm_corrs

x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
y = np.array([2, 3, 5, 4, 6, 8, 7, 9])

p_value, observed_corr, _ = permutation_correlation_test(x, y)
print(f"Observed Correlation: {observed_corr:.4f}")   # 0.9524
print(f"Permutation P-value: {p_value:.4f}")          # 0.0011
```

$n = 8$이므로 $8! = 40{,}320$가지 순열을 모두 열거할 수 있고, 정확 $p$값은 $0.001141$이다.

---

## 대응자료에 대한 순열검정

대응자료에서는 집단 라벨을 섞는 대신 차이의 **부호**를 섞는다.

```python
import numpy as np

def paired_permutation_test(before, after, n_permutations=10000, seed=0):
    """
    Sign-flip permutation test for paired data.

    Under H0 the distribution of each difference is symmetric about 0,
    so flipping the sign of any subset of differences is equally likely.
    """
    rng = np.random.default_rng(seed)
    differences = np.asarray(after) - np.asarray(before)
    observed_mean_diff = differences.mean()

    signs = rng.choice([-1, 1], size=(n_permutations, len(differences)))
    perm_means = (signs * differences).mean(axis=1)

    p_value = ((np.abs(perm_means) >= abs(observed_mean_diff)).sum() + 1) \
              / (n_permutations + 1)
    return p_value, observed_mean_diff

before = [70, 68, 75, 80, 72, 74, 69, 77, 73, 76]
after = [72, 69, 78, 85, 75, 76, 70, 79, 74, 80]

p_value, obs_diff = paired_permutation_test(before, after)
print(f"Observed Mean Difference: {obs_diff:.2f}")   # 2.40
print(f"P-value: {p_value:.4f}")                     # 0.002
```

$2^{10} = 1{,}024$가지 부호 배정을 모두 열거하면 정확 $p$값은 $2/1024 = 0.001953$이다. 이는 이 자료에서 가능한 **최소 $p$값**이다. 열 개의 차이가 모두 양수이므로, 관측된 배정과 그 전부를 뒤집은 배정만이 $|\bar{d}^*| \ge 2.4$를 만족한다.

---

## 정확 순열검정과 근사 순열검정

| 측면 | 정확 | 근사 |
|---|---|---|
| **방법** | 가능한 모든 순열 열거 | 순열의 무작위 부분집합 |
| **가능성** | 작은 표본에서만 | 어떤 크기에서도 |
| **순열 수** | 이표본이면 $\binom{n_1+n_2}{n_1}$, 대응이면 $2^n$ | 사용자가 지정(예: 10{,}000) |
| **$p$값** | 정확 | 근사(순열을 늘리면 개선) |

크기 $5$인 두 집단이면 $\binom{10}{5} = 252$가지로 열거가 쉽다. 큰 표본에서는 무작위 순열 $10{,}000$개면 대개 충분하다.

!!! tip "최소 $p$값을 먼저 확인하라"
    정확 순열검정에서 얻을 수 있는 가장 작은 양측 $p$값은 이표본이면 $2/\binom{n_1+n_2}{n_1}$, 대응이면 $2/2^n$이다.

    | 설계 | 최소 양측 $p$값 |
    |:---|---:|
    | 이표본 $n_1 = n_2 = 4$ | $2/70 = 0.0286$ |
    | 이표본 $n_1 = n_2 = 5$ | $2/252 = 0.0079$ |
    | 대응 $n = 5$ | $2/32 = 0.0625$ |
    | 대응 $n = 6$ | $2/64 = 0.0313$ |

    **대응 $n = 5$이면 $\alpha = 0.05$에서 절대 기각할 수 없다.** 자료가 아무리 극단적이어도 그렇다. 실험을 설계할 때 이 계산을 먼저 해야 한다.

---

## 장단점

| 장점 | 단점 |
|---|---|
| 분포 가정이 없다 | 자료가 크면 계산이 무겁다 |
| 작은 자료에서 정확하다 | 정확도가 순열 수에 의존한다 |
| 어떤 검정통계량에도 유연하다 | 대응자료는 방식을 바꿔야 한다(부호 뒤집기) |
| 표본크기가 같으면 이분산에 로버스트하다 | 이분산과 불균형이 겹치면 스튜던트화가 필요하다 |

---

## 응용

1. **평균·중앙값 차 검정**: 정규성 가정 없이 집단 비교
2. **상관 검정**: 상관계수의 유의성 검정
3. **변수선택**: 기계학습·예측모형에서 중요 변수 식별
4. **유전체학과 생물학**: 유전자 발현 등 고차원 자료 분석에 널리 쓰인다
5. **시계열**: 종속자료에 대한 블록 순열검정

## 연습문제

**연습문제 1.**
이 페이지의 세 예제 — 평균차, 상관, 대응자료 — 각각에서 가능한 모든 순열을 열거하여 정확 $p$값을 구하고, 대응하는 모수적 검정과 비교하라.

??? success "풀이"
    ```python
    import numpy as np, itertools
    from scipy import stats

    # (1) 평균차: C(10,5) = 252 가지
    a = np.array([8, 7, 9, 10, 6]); b = np.array([5, 6, 4, 3, 7])
    z = np.concatenate([a, b]); obs = a.mean() - b.mean()
    d = [z[list(c)].mean() - np.delete(z, list(c)).mean()
         for c in itertools.combinations(range(10), 5)]
    d = np.array(d)
    print(len(d), (np.abs(d) >= obs - 1e-12).sum(), (np.abs(d) >= obs - 1e-12).mean())
    print(stats.ttest_ind(a, b))

    # (2) 상관: 8! = 40,320 가지
    x = np.arange(1, 9); y = np.array([2, 3, 5, 4, 6, 8, 7, 9])
    r = np.corrcoef(x, y)[0, 1]
    rs = np.array([np.corrcoef(x, y[list(p)])[0, 1]
                   for p in itertools.permutations(range(8))])
    print((np.abs(rs) >= abs(r) - 1e-12).mean(), stats.pearsonr(x, y).pvalue)

    # (3) 대응: 2^10 = 1,024 가지
    before = np.array([70, 68, 75, 80, 72, 74, 69, 77, 73, 76])
    after  = np.array([72, 69, 78, 85, 75, 76, 70, 79, 74, 80])
    dd = after - before
    S = np.array(list(itertools.product([1, -1], repeat=10)))
    m = (S * dd).mean(1)
    print((np.abs(m) >= abs(dd.mean()) - 1e-12).sum(),
          (np.abs(m) >= abs(dd.mean()) - 1e-12).mean())
    print(stats.ttest_rel(after, before))
    ```

    | 예제 | 순열 수 | 관측 통계량 | 정확 순열 $p$ | 모수적 $p$ |
    |:---|---:|---:|---:|---:|
    | 평균차 | 252 | $3.00$ | **0.0397** | 0.0171 ($t$) |
    | 상관 | 40{,}320 | $r = 0.9524$ | **0.00114** | 0.00026 ($t$) |
    | 대응 | 1{,}024 | $\bar{d} = 2.40$ | **0.00195** | 0.00032 ($t$) |

    **세 경우 모두 순열 $p$값이 $t$ 검정보다 크다.** 비율로는 $2.3$배, $4.4$배, $6.0$배이다.

    이는 우연이 아니라 구조적이다. 두 가지 이유가 겹친다.

    **첫째, 이산성.** 순열 $p$값은 유한한 격자 위의 값만 취한다. 평균차 예제에서는 $1/252 = 0.004$ 간격, 대응 예제에서는 $1/1024 = 0.00098$ 간격이다. 대응 예제의 $p = 0.00195$는 **가능한 최솟값**이다. 자료가 지금보다 열 배 극단적이어도 이보다 작아질 수 없다.

    **둘째, 순열분포의 꼬리가 짧다.** 순열분포는 관측된 값들만 재배열하므로 유계이다. 평균차의 순열분포는 $[-4.0, 4.0]$ 안에만 값을 갖는다. $t$ 분포는 무한한 꼬리를 가정하므로 극단값에 더 작은 확률을 배정한다.

    **어느 쪽을 믿을 것인가.** 순열 $p$값이다. $t$ 검정의 $p$값은 정규성이 성립할 때만 정확하며, $n = 5$에서 그 가정을 자료로 확인할 방법이 없다. 순열 $p$값은 교환가능성만으로 정확하다.

    실무적 함의는 분명하다. **아주 작은 표본에서 $t$ 검정은 유의성을 과장한다.**

---

**연습문제 2.**
대응 예제에서 정확 $p$값 $0.00195$가 "가능한 최솟값"이라고 했다. 이것이 검정력에 어떤 제약을 주는지 설명하고, $\alpha = 0.01$로 검정하려면 대응표본이 최소 몇 개 필요한지 구하라.

??? success "풀이"
    **최소 $p$값의 구조.**

    부호 뒤집기 순열검정에서 가능한 부호 배정은 $2^n$가지이다. 관측된 배정 자체는 항상 $|\bar{d}^*| = |\bar{d}_{\text{obs}}|$를 만족하고, 모든 부호를 뒤집은 배정도 그렇다($\bar{d}^* = -\bar{d}_{\text{obs}}$).

    따라서 양측 $p$값은 최소한 $2/2^n = 2^{1-n}$이다.

    | $n$ | $2^n$ | 최소 양측 $p$값 | $\alpha = 0.05$ 가능? | $\alpha = 0.01$ 가능? |
    |---:|---:|---:|:---:|:---:|
    | 4 | 16 | 0.1250 | ✗ | ✗ |
    | 5 | 32 | 0.0625 | ✗ | ✗ |
    | 6 | 64 | 0.0313 | ✓ | ✗ |
    | 7 | 128 | 0.0156 | ✓ | ✗ |
    | 8 | 256 | 0.0078 | ✓ | ✓ |
    | 10 | 1{,}024 | 0.0020 | ✓ | ✓ |

    **답: $\alpha = 0.01$에는 $n \ge 8$이 필요하다.** $2^{1-8} = 0.0078 < 0.01$이 처음 성립하는 $n$이다.

    **검정력에 대한 제약은 절대적이다.** $n = 5$이면 대립가설이 아무리 강해도, 자료가 아무리 완벽하게 한 방향을 가리켜도, $\alpha = 0.05$에서의 검정력이 **정확히 0**이다. 기각 자체가 불가능하기 때문이다.

    이는 "검정력이 낮다"와 질적으로 다르다. 표본크기 계산에서 흔히 놓치는 지점이다. 모수적 검정력 공식은 $n = 5$에서도 $0.8$ 같은 숫자를 내놓지만, 비모수 정확검정을 쓸 계획이라면 그 숫자는 무의미하다.

    **일반화.** 이표본 설계에서는 최소 $p$값이 $2/\binom{n_1+n_2}{n_1}$이다.

    | $n_1 = n_2$ | $\binom{2n}{n}$ | 최소 양측 $p$값 |
    |---:|---:|---:|
    | 3 | 20 | 0.100 |
    | 4 | 70 | 0.029 |
    | 5 | 252 | 0.0079 |
    | 6 | 924 | 0.0022 |

    이표본 설계가 대응 설계보다 순열 개수가 빨리 늘어난다. 같은 총 관측 수 $2n = 10$에 대해 이표본은 $252$가지, 대응 $n = 10$은 $1024$가지로 대응 쪽이 많지만, 대응은 관측 수가 $20$개(쌍 $10$개)이다. **관측 수를 고정하면 이표본 설계의 눈금이 더 촘촘하다.**

    다만 이는 눈금의 문제일 뿐 검정력의 전부가 아니다. 대응 설계는 개체 간 변동을 제거하므로 실제 검정력은 대개 훨씬 높다.

---

**연습문제 3.**
순열검정의 유연성을 분산 비교로 확인하라. $H_0: \sigma_X^2 = \sigma_Y^2$를 $\log(s_X^2/s_Y^2)$를 통계량으로 하는 순열검정으로 검정하고, $F$ 검정 및 Levene 검정과 제1종 오류율·검정력을 비교하라. 자료는 정규분포와 $t(3)$ 분포에서 생성한다.

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(31)

    def perm_var_p(x, y, B=499):
        obs = np.log(x.var(ddof=1) / y.var(ddof=1))
        z = np.concatenate([x, y]); m = len(x); c = 0
        for _ in range(B):
            p = rng.permutation(z)
            c += abs(np.log(p[:m].var(ddof=1) / p[m:].var(ddof=1))) >= abs(obs) - 1e-12
        return (c + 1) / (B + 1)

    def F_p(x, y):
        f = x.var(ddof=1) / y.var(ddof=1)
        d1, d2 = len(x) - 1, len(y) - 1
        return 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2))

    def run(gen, ratio, n=25, M=1500):
        r = dict(F=0, levene=0, perm=0)
        for _ in range(M):
            x = gen(n) * ratio; y = gen(n)
            r['F'] += F_p(x, y) < 0.05
            r['levene'] += stats.levene(x, y, center='median').pvalue < 0.05
            r['perm'] += perm_var_p(x, y) < 0.05
        return {k: round(v/M, 3) for k, v in r.items()}
    ```

    **제1종 오류율 ($\sigma_X/\sigma_Y = 1$, $n = 25$)**

    | 자료 | $F$ 검정 | Levene | 순열검정 |
    |:---|---:|---:|---:|
    | 정규 | 0.047 | 0.045 | 0.051 |
    | $t(3)$ | **0.318** | 0.038 | 0.044 |

    **$F$ 검정이 $t(3)$ 자료에서 붕괴한다.** 제1종 오류율이 $0.318$로 명목값의 **6배가 넘는다**. 등분산인 두 표본을 놓고 세 번에 한 번꼴로 "분산이 다르다"고 선언한다.

    이유는 $F$ 검정이 정규성에 극도로 민감하기 때문이다. $s^2$의 표집분포는 4차 적률(첨도)에 의존하는데, $F$ 분포는 정규분포의 첨도 $3$을 전제한다. $t(3)$의 첨도는 무한대이다.

    **순열검정과 Levene 검정은 크기를 지킨다**($0.044$, $0.038$). 순열검정의 정확성은 여기서도 작동한다. $\sigma_X = \sigma_Y$이고 두 표본이 같은 분포에서 오면 교환가능성이 정확히 성립한다.

    **검정력 ($\sigma_X/\sigma_Y = 2$)**

    | 자료 | $F$ 검정 | Levene | 순열검정 |
    |:---|---:|---:|---:|
    | 정규 | **0.903** | 0.807 | 0.867 |
    | $t(3)$ | ~~0.787~~ | 0.559 | **0.538** |

    **정규자료에서 $F$ 검정이 가장 강력하다**($0.903$). 이는 이론이 예측하는 바이다. 정규성이 성립하면 $F$ 검정이 최적이다. 순열검정은 $0.867$로 $0.036$을 잃을 뿐이다. Levene은 $0.807$로 더 잃는다.

    **$t(3)$ 자료에서 $F$ 검정의 $0.787$은 읽으면 안 되는 숫자이다.** 크기가 $0.318$인 검정의 검정력은 의미가 없다. 크기를 $0.05$로 보정하면 실제 검정력은 훨씬 낮다. 표에 취소선을 그은 이유이다.

    !!! note "순열검정이 여기서 특히 값진 이유"
        분산비 검정에는 이미 $F$ 검정과 Levene 검정이 있다. 그런데도 순열검정이 유용한 것은 **통계량을 자유롭게 고를 수 있기** 때문이다.

        $\log(s_X^2/s_Y^2)$ 대신 사분위범위의 비, 중앙값절대편차(MAD)의 비, 또는 문제에 특화된 임의의 산포 측도를 넣어도 절차가 그대로 작동한다. 각각에 대해 새로운 표집분포 이론을 유도할 필요가 없다.

        Levene 검정 자체가 이 아이디어의 특수한 경우이다. $|x_i - \text{med}(x)|$로 변환한 뒤 평균차를 보는 것이며, 이를 순열 틀 안에 넣으면 정규 근사 없이 정확해진다.

    !!! warning "분산 순열검정의 한계"
        $H_0: \sigma_X^2 = \sigma_Y^2$만 참이고 두 분포의 **모양이 다르면** 교환가능성이 깨진다. 위 모의실험은 두 표본을 같은 분포족에서 생성했으므로 유리한 상황이다.

        평균이 다르면서 분산만 같은 경우도 마찬가지다. 이때는 각 표본을 자기 평균으로 중심화한 뒤 순열하는 등의 보정이 필요하며, 그렇게 해도 근사적으로만 타당하다. [교환가능성에 대한 논의](foundations.md)의 연습문제 1이 같은 문제를 평균 비교에서 다룬다.
