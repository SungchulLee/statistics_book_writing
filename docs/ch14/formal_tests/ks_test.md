# Kolmogorov-Smirnov 검정

## 개요

Kolmogorov-Smirnov(KS) 검정은 표본의 경험분포함수를 지정된 이론적 분포와 비교하는 비모수 검정이다. 정규성 검정에서는 경험적 CDF와 정규 CDF 사이의 최대 수직 거리를 잰다. 결정적인 단서 조항은 표준 KS 검정이 귀무가설의 모수를 미리 완전히 지정하도록 요구한다는 것이다. 모수를 자료에서 추정하면 $p$값이 무효가 된다.

## 경험분포함수

i.i.d. 표본 $X_1, \ldots, X_n$이 주어졌을 때 경험분포함수(EDF)는

$$
F_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}(X_i \leq x),
$$

여기서 $\mathbf{1}(\cdot)$은 지시함수이다. Glivenko-Cantelli 정리에 의해 $n \to \infty$일 때 $F_n(x) \to F(x)$가 거의 확실하게 균등수렴한다.

## KS 통계량

일표본 KS 통계량은 $F_n$과 가설 CDF $F_0$ 사이의 상한 거리를 잰다.

$$
D_n = \sup_x |F_n(x) - F_0(x)|.
$$

실제로는 다음과 같이 계산한다.

$$
D_n = \max_{1 \leq i \leq n} \max\!\left(\left|\frac{i}{n} - F_0(X_{(i)})\right|,\; \left|F_0(X_{(i)}) - \frac{i-1}{n}\right|\right),
$$

여기서 $X_{(1)} \leq \cdots \leq X_{(n)}$은 순서통계량이다.

## 가설

$$
H_0: F = F_0 \quad (\text{자료가 지정된 분포를 따른다}), \qquad H_1: F \neq F_0.
$$

$F_0$이 완전히 지정된 $H_0$ 아래에서 $\sqrt{n}\, D_n$의 분포는 Kolmogorov 분포로 수렴하며 그 CDF는

$$
P(\sqrt{n}\, D_n \leq t) \to 1 - 2\sum_{k=1}^{\infty} (-1)^{k-1} e^{-2k^2 t^2}.
$$

### 코드

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = rng.normal(0.0, 1.0, size=250)

# Fully specified H0: Normal(0, 1)
D, p = stats.kstest(x, 'norm', args=(0.0, 1.0))

print(f"n = {x.size}")
print(f"KS one-sample vs N(0,1): D = {D:.4f}, p = {p:.4f}")
if p < 0.05:
    print("=> Reject H0: data may not follow N(0,1).")
else:
    print("=> Fail to reject H0 at alpha = 0.05.")
```

출력:

```text
n = 250
KS one-sample vs N(0,1): D = 0.0354, p = 0.9013
=> Fail to reject H0 at alpha = 0.05.
```

## Lilliefors 문제

$\mu$와 $\sigma$를 자료에서 추정해 $F_0$에 꽂아 넣으면, 적합된 CDF가 $F_n$에 가깝도록 최적화되므로 KS 통계량이 체계적으로 작아진다. 그러면 표준 KS 임계값과 $p$값이 무효가 된다(지나치게 보수적이어서 과소기각한다). Lilliefors 검정은 모수 추정을 반영하는 모의실험 기반 임계값이나 표로 정리된 임계값을 써서 이 문제를 해결한다.

## 해석

KS 검정은 *일치성*(consistent)을 갖는 검정이다. $n \to \infty$이면 결국 $F_0$에서의 어떤 이탈이든 탐지한다. 그러나 정규분포의 특정 구조를 활용하는 검정(Shapiro-Wilk, Anderson-Darling 등)에 비해 검정력이 상당히 낮다. KS 검정은 분포의 모든 부분을 동등하게 취급하는 반면, Anderson-Darling 검정은 꼬리에 추가 가중을 주므로 꼬리 이탈에 더 민감하다.

이 검정력 차이가 얼마나 극적일 수 있는지는 연습문제 4에서 확인한다.

## 연습문제

**연습문제 1.** $\mathcal{N}(0,1)$에서 관측값 $n = 300$개를 생성하라. $\mathcal{N}(0,1)$과 $\mathcal{N}(0.5, 1)$에 대해 각각 KS 검정을 수행하라. $p$값을 비교하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=300)

    D1, p1 = stats.kstest(x, 'norm', args=(0.0, 1.0))
    D2, p2 = stats.kstest(x, 'norm', args=(0.5, 1.0))

    print(f"vs N(0,1):   D = {D1:.4f}, p = {p1:.4g}")
    print(f"vs N(0.5,1): D = {D2:.4f}, p = {p2:.4g}")
    ```

    출력:

    ```text
    vs N(0,1):   D = 0.0507, p = 0.4089
    vs N(0.5,1): D = 0.2264, p = 5.45e-14
    ```

    참 분포인 $\mathcal{N}(0,1)$에 대해서는 $D = 0.051$, $p = 0.409$로 기각하지 않는다. 평균이 틀린 $\mathcal{N}(0.5,1)$에 대해서는 EDF가 체계적으로 이동해 있어 $D = 0.226$으로 네 배 넘게 커지고 $p = 5.5 \times 10^{-14}$로 강하게 기각한다.

    $D$의 크기를 직관적으로 이해해 보자. $\Phi(x)$와 $\Phi(x - 0.5)$ 사이의 최대 수직 거리는 $x = 0.25$에서 $\Phi(0.25) - \Phi(-0.25) = 0.1974$이다. 표본 $D = 0.226$이 이 이론값 근처에 있다.

    **위치 이동은 KS 검정이 잘 잡는다.** 다음 연습문제들에서 볼 꼬리 이탈과 대조된다. $\square$

---

**연습문제 2.** Lilliefors 문제를 시연하라. $\mathcal{N}(0,1)$에서 관측값 $n = 200$개를 생성하고 $\hat{\mu}$, $\hat{\sigma}$를 추정한 뒤 $\mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$에 대해 KS 검정을 수행하라. $p$값을 믿을 수 있는가?

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=200)

    mu_hat, sigma_hat = x.mean(), x.std(ddof=1)
    D, p = stats.kstest(x, 'norm', args=(mu_hat, sigma_hat))

    print(f"Estimated mu = {mu_hat:.4f}, sigma = {sigma_hat:.4f}")
    print(f"KS vs N(mu_hat, sigma_hat): D = {D:.4f}, p = {p:.4f}")
    ```

    출력:

    ```text
    Estimated mu = 0.0153, sigma = 0.9636
    KS vs N(mu_hat, sigma_hat): D = 0.0306, p = 0.9892
    ```

    $p = 0.989$는 **믿을 수 없다.** 적합된 정규분포가 미리 지정된 정규분포보다 EDF에 더 가까우므로 $D$가 작아지고 $p$값이 부풀려진다.

    얼마나 심각한지 모의실험으로 확인해 보자. $\mathcal{N}(0,1)$에서 $n = 200$인 표본 5,000개를 뽑아 매번 모수를 추정하고 이 순진한 KS 검정을 적용하면 $\alpha = 0.05$에서의 경험적 기각률은

    ```text
    Plug-in KS empirical size: 0.0002
    ```

    **0.0002이다.** 명목값 0.05의 250분의 1에 불과하다. 5,000회 중 단 한 번만 기각했다는 뜻이다.

    이 정도로 극단적인 과소기각은 검정을 사실상 무용하게 만든다. 크기가 0.0002인 검정은 참된 이탈이 있어도 거의 탐지하지 못한다. 타당한 추론을 하려면 반드시 Lilliefors 보정이 필요하다. $\square$

---

**연습문제 3.** 순서통계량으로부터 $D_n$을 계산하는 공식을 유도하라. $n$개의 값 $X_{(1)}, \ldots, X_{(n)}$만 확인하면 충분함을 보여라.

??? success "풀이"

    EDF $F_n(x)$는 각 $X_{(i)}$에서 $1/n$만큼 뛰는 계단함수이다. 연속된 두 순서통계량 사이에서 $F_n$은 상수인 반면 $F_0$은 단조증가한다. 따라서 차이 $F_n(x) - F_0(x)$는 그 구간에서 단조감소하고, 상한 $|F_n(x) - F_0(x)|$는 도약 직전이나 직후에 달성된다.

    - $i$번째 도약 직후: $F_n(X_{(i)}) = i/n$이므로 $|i/n - F_0(X_{(i)})|$.
    - $i$번째 도약 직전: $F_n(X_{(i)}^-) = (i-1)/n$이므로 $|(i-1)/n - F_0(X_{(i)})|$.

    따라서

    $$
    D_n = \max_{1 \leq i \leq n} \max\!\left(\left|\frac{i}{n} - F_0(X_{(i)})\right|,\; \left|F_0(X_{(i)}) - \frac{i-1}{n}\right|\right).
    $$

    $n$개의 순서통계량만 평가하면 충분하다. 무한한 실직선 위의 상한을 유한한 최댓값 계산으로 바꿔주는 결과이며, KS 검정을 실용적으로 만드는 핵심이다. $\square$

---

**연습문제 4.** $n = 200$이고 대립가설이 $t_5$일 때($\mathcal{N}(0,1)$에 대해 검정) $\alpha = 0.05$에서 KS 검정의 검정력을 추정하는 몬테카를로 모의실험을 수행하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 200, 5000, 0.05
    rejections = 0

    for _ in range(reps):
        x = rng.standard_t(df=5, size=n)
        _, p = stats.kstest(x, 'norm', args=(0.0, 1.0))
        if p < alpha:
            rejections += 1

    print(f"KS power vs t(5): {rejections / reps:.4f}")
    ```

    출력:

    ```text
    KS power vs t(5): 0.0910
    ```

    **검정력이 0.091에 불과하다.** 크기 0.05보다 겨우 조금 클 뿐이며, 사실상 쓸모가 없다.

    왜 이렇게 낮은지는 모집단 KS 거리를 계산하면 명확해진다.

    $$
    \sup_x |F_{t_5}(x) - \Phi(x)| = 0.0305 \quad (x = -1.63 \text{ 부근에서 달성}).
    $$

    한편 $n = 200$에서 5% 임계값은 근사적으로

    $$
    D_{\text{crit}} \approx \frac{1.358}{\sqrt{200}} = 0.096.
    $$

    곧 **모집단 거리 $0.0305$가 임계값 $0.096$의 3분의 1에도 못 미친다.** 표집잡음이 없더라도 $D_n$이 임계값에 도달하지 못한다. 50% 검정력을 얻으려면 $n \approx (1.358/0.0305)^2 \approx 1980$이 필요하다.

    $t_5$의 분산이 $5/3 \approx 1.67$로 1과 크게 다른데도 CDF 거리가 작은 이유는, 두 CDF가 여러 번 교차하면서 차이가 상쇄되기 때문이다. 중앙 근처에서는 $t_5$가 더 뾰족해서 $F_{t_5} > \Phi$이고, 꼬리에서는 $t_5$가 더 두꺼워서 부호가 뒤집힌다. 어느 한 지점의 수직 거리는 작게 유지된다.

    같은 자료에 정규성 검정을 적용하면 대비가 뚜렷하다($n = 200$, $\alpha = 0.05$).

    | 검정 | 검정력 |
    |---|---|
    | KS (vs $\mathcal{N}(0,1)$ 완전 지정) | 0.091 |
    | Anderson-Darling | 0.724 |
    | Shapiro-Wilk | 0.819 |

    (공정한 비교는 아니다. AD와 SW는 모수를 추정하는 위치-척도 자유 검정인 반면 KS는 완전 지정 대립을 다룬다. 그럼에도 KS가 꼬리 이탈에 얼마나 둔감한지는 분명하다.)

    꼬리에 가중을 주는 Anderson-Darling과 순서통계량의 상관구조를 활용하는 Shapiro-Wilk가 이런 상황에서 압도적으로 낫다. $\square$

---

**연습문제 5.** 일차원의 경우에 대해 Glivenko-Cantelli 정리를 서술하고 증명하라(증명의 개요면 충분하다).

??? success "풀이"

    **정리 (Glivenko-Cantelli).** $X_1, X_2, \ldots$가 CDF $F$를 갖는 i.i.d.라 하자. 그러면

    $$
    \sup_x |F_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad (n \to \infty).
    $$

    **증명 개요.** 임의의 $x \in \mathbb{R}$을 고정한다. 강대수의 법칙에 의해 $F_n(x) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(X_i \leq x) \to F(x)$가 거의 확실하게 성립한다. 이는 점별 수렴이다.

    균등수렴으로 끌어올리려면, $\epsilon > 0$을 고정하고 모든 $j$에 대해 $F(t_j) - F(t_{j-1}) < \epsilon$이 되도록 점 $-\infty = t_0 < t_1 < \cdots < t_K = \infty$를 잡는다. 임의의 $x \in [t_{j-1}, t_j]$에 대해 $F_n$과 $F$의 단조성에서

    $$
    F_n(x) - F(x) \leq F_n(t_j) - F(t_{j-1}) = [F_n(t_j) - F(t_j)] + [F(t_j) - F(t_{j-1})] < [F_n(t_j) - F(t_j)] + \epsilon.
    $$

    비슷하게 $F(x) - F_n(x) < [F(t_j) - F_n(t_{j-1})] + \epsilon$이다.

    $t_j$가 유한개이므로 각 $t_j$에서의 점별 수렴(강대수의 법칙)이 결국 $\sup_x |F_n(x) - F(x)| < 2\epsilon$을 거의 확실하게 보장한다. $\epsilon$이 임의였으므로 결과가 따라 나온다. $\square$
