# X-bar의 표본분포 (Uniform)

## 개요

모집단에서 크기 $n$인 확률표본을 반복해서 뽑고 매번 표본평균 $\bar{X}$를 계산하면, 그 평균들이 이루는 분포를 **$\bar{X}$의 표본분포**라 한다. 이 페이지에서는 Uniform 모집단을 사용하여 이 개념을 살펴본다. 모집단 분포가 평평한데도 $\bar{X}$의 표본분포는 모평균 주위로 모이고, 중심극한정리에 의해 $n$이 커질수록 근사적으로 정규분포가 된다.

## 모집단 모형

모집단이 구간 $[0, 1]$ 위의 연속 균등분포를 따른다고 하자:

$$
X \sim \text{Uniform}(0, 1)
$$

모평균과 모분산은:

$$
\mu = E[X] = \frac{1}{2}, \qquad \sigma^2 = \text{Var}(X) = \frac{1}{12}
$$

## 표본평균의 표본분포

이 모집단에서 독립적으로 뽑은 확률표본 $X_1, X_2, \ldots, X_n$에 대해 표본평균은:

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

독립 확률변수의 기댓값과 분산의 성질에 의해:

$$
E[\bar{X}] = \mu = \frac{1}{2}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{12n}
$$

**중심극한정리**에 의해 $n$이 크면:

$$
\bar{X} \;\dot{\sim}\; N\!\left(\frac{1}{2},\; \frac{1}{12n}\right)
$$

## 모의실험

다음 코드는 Uniform(0, 1) 모집단에서 크기 $n = 5$인 표본을 10,000개 뽑아 모집단, 하나의 표본, $\bar{X}$의 표본분포라는 세 분포를 나란히 그린다.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# Generate a large population from Uniform(0, 1)
population = np.random.uniform(size=(n_population,))

# Draw a single sample
single_sample = np.random.choice(population, size=sample_size, replace=False)

# Simulate the sampling distribution of X-bar
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# Plot
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

ax0.hist(population, bins=np.linspace(0, 1, 100))
ax0.set_title("Population Distribution")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=np.linspace(0, 1, 100))
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

## 해석

!!! note "주요 관찰"

    1. **모집단 분포**는 $[0, 1]$에서 평평하다(균등분포).
    2. 크기 5인 **하나의 표본**은 흩어진 점 몇 개일 뿐이며, 그것만으로는 모집단의 모양을 알 수 없다.
    3. $\bar{X}$의 **표본분포**는 모집단이 종 모양이 아닌데도 종 모양이고 $\mu = 0.5$를 중심으로 한다. 중심극한정리가 작동하는 모습이다.
    4. 표본분포의 퍼짐은 모집단 분포보다 $1/\sqrt{n}$배 좁다.

$n = 5$일 때 $\bar{X}$의 표준오차는:

$$
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}} = \frac{1/\sqrt{12}}{\sqrt{5}} \approx 0.129
$$

$n$이 커질수록 이 표준오차가 줄어들고 표본분포는 $\mu$ 주위로 더 촘촘하게 모인다.

## 연습문제

**연습문제 1.** $X \sim \text{Uniform}(0, 1)$에 대해 적분을 사용하여 정의로부터 $E[X]$와 $\text{Var}(X)$를 유도하라.

??? success "풀이"
    $X \sim \text{Uniform}(0, 1)$의 pdf는 $x \in [0, 1]$에서 $f(x) = 1$이다.

    $$
    E[X] = \int_0^1 x \cdot 1 \, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}
    $$

    $$
    E[X^2] = \int_0^1 x^2 \cdot 1 \, dx = \left[\frac{x^3}{3}\right]_0^1 = \frac{1}{3}
    $$

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}
    $$

    $\square$

---

**연습문제 2.** 관측값이 독립이라는 가정 아래, 평균이 $\mu$이고 분산이 $\sigma^2$인 임의의 모집단에 대해 $E[\bar{X}] = \mu$이고 $\text{Var}(\bar{X}) = \sigma^2 / n$임을 증명하라.

??? success "풀이"
    $X_1, \ldots, X_n$을 $E[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2$인 i.i.d. 확률변수라 하자.

    $$
    E[\bar{X}] = E\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n E[X_i] = \frac{1}{n} \cdot n\mu = \mu
    $$

    독립성에 의해:

    $$
    \text{Var}(\bar{X}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{1}{n^2} \cdot n\sigma^2 = \frac{\sigma^2}{n}
    $$

    $\square$

---

**연습문제 3.** 표본크기를 $n = 5$에서 $n = 20$으로 늘리면 $\bar{X}$의 표준오차는 몇 배로 줄어드는가? $n = 5$일 때에 비해 표준오차를 절반으로 줄이려면 표본크기가 얼마여야 하는가?

??? success "풀이"
    표준오차는 $\text{SE} = \sigma / \sqrt{n}$이다.

    표준오차의 비는:

    $$
    \frac{\text{SE}(n=5)}{\text{SE}(n=20)} = \frac{\sigma/\sqrt{5}}{\sigma/\sqrt{20}} = \sqrt{\frac{20}{5}} = \sqrt{4} = 2
    $$

    따라서 $n = 5$에서 $n = 20$으로 갈 때 표준오차가 2배로 줄어든다.

    $n = 5$ 대비 표준오차를 절반으로 줄이려면:

    $$
    \frac{\sigma}{\sqrt{n}} = \frac{1}{2} \cdot \frac{\sigma}{\sqrt{5}} \implies \sqrt{n} = 2\sqrt{5} \implies n = 20
    $$

    $\square$

---

**연습문제 4.** 모의실험을 $n = 5$ 대신 $n = 50$으로 수정하라. 이론적 표준오차를 계산하고 모의실험으로 얻은 10,000개 평균의 표본표준편차와 비교하라.

??? success "풀이"
    $n = 50$에서의 이론적 표준오차는:

    $$
    \text{SE} = \frac{1/\sqrt{12}}{\sqrt{50}} = \frac{1}{\sqrt{600}} \approx 0.0408
    $$

    코드로는:

    ```python
    import numpy as np
    np.random.seed(1)

    population = np.random.uniform(size=10_000)
    sample_means = [
        np.mean(np.random.choice(population, size=50, replace=False))
        for _ in range(10_000)
    ]
    empirical_se = np.std(sample_means)
    print(f"Theoretical SE: {1 / np.sqrt(600):.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")
    ```

    경험적 표준오차가 0.0408에 가깝게 나와 이론적 공식을 확인해 준다. $\square$

---

**연습문제 5.** Irwin–Hall 분포는 $X_i \sim \text{Uniform}(0,1)$이 독립일 때 합 $S_n = X_1 + X_2 + \cdots + X_n$의 분포이다. $\bar{X} = S_n / n$임을 보이고, $n = 2$에 대한 Irwin–Hall pdf를 사용하여 $n = 2$일 때 $\bar{X}$의 정확한 pdf를 구하라.

??? success "풀이"
    정의에 의해 $\bar{X} = S_n / n$이다. $n = 2$일 때 $S_2 = X_1 + X_2$의 Irwin–Hall pdf는 삼각분포이다:

    $$
    f_{S_2}(s) = \begin{cases} s & 0 \le s \le 1 \\ 2 - s & 1 < s \le 2 \\ 0 & \text{otherwise} \end{cases}
    $$

    $\bar{X} = S_2 / 2$이므로 변수변환 공식을 적용한다. $y = s/2$로 두면 $s = 2y$이고 $ds = 2\,dy$이므로:

    $$
    f_{\bar{X}}(y) = f_{S_2}(2y) \cdot 2 = \begin{cases} 4y & 0 \le y \le \tfrac{1}{2} \\ 4(1 - y) & \tfrac{1}{2} < y \le 1 \\ 0 & \text{otherwise} \end{cases}
    $$

    이는 $y = 1/2$에서 정점을 이루는 $[0, 1]$ 위의 대칭 삼각분포이다. $n = 2$에서도 이미 $\bar{X}$의 표본분포가 (아직 정규는 아니지만) 단봉이고 대칭임을 확인해 준다. $\square$
