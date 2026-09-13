# 코시에서 대수의법칙의 실패

## 개요

코시분포는 대수의법칙에 대한 가장 극적인 반례를 제공한다. 코시분포는 평균이 유한하지 않으므로($E[|X|] = \infty$) i.i.d. 코시 관측값의 표본평균은 수렴하지 않는다 — 표본크기와 무관하게 관측값 하나와 같은 분포를 갖는다. 이 페이지에서는 궤적 그림, 표본분포 히스토그램, Q-Q 그림으로 코시와 정규의 표본평균 거동을 대조한다.

## 코시분포

표준 코시분포의 밀도는:

$$f(x) = \frac{1}{\pi(1 + x^2)}, \quad x \in \mathbb{R}$$

핵심 성질:

- **평균이 유한하지 않다:** $E[|X|] = \int_0^\infty \frac{2x}{\pi(1+x^2)}dx = \frac{2}{\pi}\left[\frac{1}{2}\ln(1+x^2)\right]_0^\infty = \infty$
- **분산도 유한하지 않다** (평균이 존재하지 않으므로)
- **두꺼운 꼬리:** $t \to \infty$일 때 $P(|X| > t) \sim 2/(\pi t)$ (정규의 지수 감쇠와 달리 다항 감쇠)
- **특성함수:** $\varphi(t) = e^{-|t|}$

!!! danger "대수의법칙이 적용되지 않는다"
    대수의법칙은 $E[|X|] < \infty$를 요구한다. 코시에서는 이것이 깨지므로 표본평균 $\bar{X}_n$은 어떤 값으로도 수렴하지 않는다. 사실 $\bar{X}_n$은 모든 $n$에서 정확히 같은 코시분포를 갖는다.

## 표본평균의 궤적

누적평균 $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$은 정규 자료와 코시 자료에서 놀랄 만큼 다른 거동을 보인다.

<div class="codebox" markdown>

### 예제 1. 코시와 정규의 표본평균 경로 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def sample_mean_trajectories(dist, n_max=10_000, n_tries=20):
    """누적 표본평균의 경로를 n_tries개 만든다.

    코시분포는 평균이 존재하지 않으므로 큰수의 법칙이 성립하지 않는다.
    경로가 수렴하지 않고 계속 튀는 모습을 정규분포와 나란히 놓고 본다.
    """
    trajectories = []
    for _ in range(n_tries):
        if dist == "cauchy":
            data = np.random.standard_cauchy(n_max)
        else:
            data = np.random.standard_normal(n_max)
        running_mean = np.cumsum(data) / np.arange(1, n_max + 1)
        trajectories.append(running_mean)
    return trajectories

n_max = 10_000
ns = np.arange(1, n_max + 1)

cauchy_traj = sample_mean_trajectories("cauchy", n_max)
normal_traj = sample_mean_trajectories("normal", n_max)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
for traj in cauchy_traj:
    # [-50, 50]으로 잘라 낸다. 코시 경로는 수백, 수천까지 튀어 올라
    # 그대로 그리면 나머지가 한 줄로 뭉개진다.
    # **잘라 냈다는 것 자체가 코시분포의 성질을 말해 준다.**
    clipped = np.clip(traj, -50, 50)
    ax.semilogx(ns, clipped, lw=0.7, alpha=0.6)
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("n"); ax.set_ylabel("Sample mean")
ax.set_title("Cauchy: Sample Mean Trajectories")
ax.set_ylim(-50, 50)

ax = axes[1]
for traj in normal_traj:
    ax.semilogx(ns, traj, lw=0.7, alpha=0.6)
ax.axhline(0, color="red", linestyle="--", lw=2)
ax.set_xlabel("n"); ax.set_ylabel("Sample mean")
ax.set_title("Normal: Sample Mean Trajectories")
ax.set_ylim(-1, 1)

plt.tight_layout()
plt.show()
```

![Cauchy: Sample Mean Trajectories](./img/cauchy_lln_failure_27.png)

</div>

!!! note "수렴과 비수렴"
    정규분포(오른쪽 그림)에서는 $n$이 커지면 20개 궤적이 모두 눈에 띄게 0으로 수렴한다. Cauchy(왼쪽 그림)에서는 궤적이 계속 불규칙하게 떠돈다 — $n$이 커진 뒤에도 이따금 나타나는 극단 관측값이 누적평균을 "초기화"해 버린다.

## 표본평균의 분포

정규분포에서는 $n$이 커질수록 $\bar{X}_n$의 표본분포가 좁아진다(중심극한정리에 의해 표준편차가 $1/\sqrt{n}$이다). 코시에서는 $\bar{X}_n$의 분포가 전혀 좁아지지 **않는다**.

<div class="codebox" markdown>

### 예제 2. 표본크기를 키워도 좁아지지 않는 분포 { .eg }

```python
from scipy import stats

def sample_mean_distributions(dist, n_vals, n_reps=10_000):
    """표본크기별로 표본평균의 분포를 만든다.

    앞 그림이 한 경로가 시간에 따라 어떻게 움직이는지를 보였다면, 여기서는
    같은 크기의 표본을 만 번 뽑아 표본평균이 어디에 흩어지는지를 본다.
    """
    results = {}
    for n in n_vals:
        if dist == "cauchy":
            data = np.random.standard_cauchy((n_reps, n))
        else:
            data = np.random.standard_normal((n_reps, n))
        results[n] = data.mean(axis=1)
    return results

# n 을 100 배로 키워도 코시 쪽 히스토그램은 좁아지지 않는다.
# 코시 표본평균의 분포가 원래 분포와 똑같은 코시이기 때문이다.
# 표본을 늘리는 일이 아무 보탬이 되지 않는 드문 경우다.
n_vals = [100, 1000, 10_000]
cauchy_dists = sample_mean_distributions("cauchy", n_vals)
normal_dists = sample_mean_distributions("normal", n_vals)

fig, axes = plt.subplots(1, 3, figsize=(17, 4))
for col, n in enumerate(n_vals):
    ax = axes[col]
    c_means = np.clip(cauchy_dists[n], -20, 20)
    n_means = normal_dists[n]
    ax.hist(c_means, bins=80, density=True, alpha=0.6, color="coral", label="Cauchy")
    ax.hist(n_means, bins=50, density=True, alpha=0.6, color="steelblue", label="Normal")
    ax.set_title(f"Sample Mean Distribution (n = {n})")
    ax.set_xlabel("Sample mean value")
    ax.set_xlim(-5, 5)
    ax.legend()
plt.tight_layout()
plt.show()
```

![코시에서 대수의법칙의 실패](./img/cauchy_lln_failure_80.png)

</div>

!!! warning "코시분포는 집중되지 않는다"
    $n = 10{,}000$에서 정규 표본평균의 분포는 0에 뾰족하게 모여들지만(표준편차 $= 0.01$), 코시 표본평균의 분포는 $n = 100$일 때와 사실상 똑같아 보인다. 코시 자료를 더 많이 평균해도 도움이 되지 않는다.

## Q-Q 그림: 두꺼운 꼬리의 시각화

코시 분위수를 정규 분위수와 비교하는 **Q-Q 그림**은 코시 꼬리가 얼마나 극단적으로 두꺼운지 드러낸다.

<div class="codebox" markdown>

### 예제 3. Q-Q 그림으로 보는 두꺼운 꼬리 { .eg }

```python
# Q-Q 그림은 자료의 분위수를 정규분포의 분위수와 짝지어 찍는다.
# 정규자료라면 점들이 직선에 놓인다. 코시 자료는 양끝이 위아래로 크게
# 휘어 올라가는데, 그 휘어짐이 곧 두꺼운 꼬리의 눈에 보이는 모습이다.
cauchy_sample = np.random.standard_cauchy(1000)
fig, ax = plt.subplots(figsize=(6, 6))
stats.probplot(cauchy_sample, dist="norm", plot=ax)
ax.set_title("Cauchy vs Normal Q-Q Plot")
plt.tight_layout()
plt.show()
```

![Cauchy vs Normal Q-Q Plot](./img/cauchy_lln_failure_119.png)

특유의 S자(또는 하키스틱) 모양은 코시가 정규분포보다 훨씬 극단적인 값을 만들어낸다는 것을 보여준다.

</div>

## 평균이 실패하는 이유: 특성함수를 통한 증명

표준 코시의 특성함수는 $\varphi_X(t) = e^{-|t|}$이다.

i.i.d. 코시 변수 $n$개의 표본평균에 대해:

$$\varphi_{\bar{X}_n}(t) = \left[\varphi_X(t/n)\right]^n = \left[e^{-|t|/n}\right]^n = e^{-|t|}$$

이는 코시 관측값 하나의 특성함수이다. 따라서:

$$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{모든 } n \text{에 대해}$$

!!! info "안정성"
    코시분포는 지수 $\alpha = 1$인 **안정분포**이다. 안정분포에서는 i.i.d. 복사본들의 선형결합이 (축척을 조정하면) 같은 분포 형태를 갖는다. 코시는 표본평균이 관측값 하나와 같은 분포를 갖는 유일한 대칭 안정분포인데, 합 $S_n$의 척도모수가 $\sqrt{n}$이 아니라 $n$에 비례해 커져서 $1/n$로 나누는 것과 정확히 상쇄되기 때문이다.

## 코시 자료를 위한 대안 추정량

코시 자료에서 표본평균이 쓸모없다면 어떤 대안이 통할까?

| 추정량 | 수렴하는가? | 속도 |
|-----------|-----------|------|
| 표본평균 | 아니오 | 해당 없음 |
| 표본중앙값 | 예 | $O(1/\sqrt{n})$ |
| MLE (위치) | 예 | $O(1/\sqrt{n})$ |
| 절사평균 ($\alpha > 0$) | 예 | $O(1/\sqrt{n})$ |

**표본중앙값**은 코시 위치모수에 대해 일치하며 점근분산이 $\pi^2/(4n) \approx 2.47/n$이다. 위치모수의 MLE는 점근분산이 $2/n$으로 더 작아 중앙값보다 효율적이며, 중앙값 대비 MLE의 ARE는 $\pi^2/8 \approx 1.23$이다.

## 해석

- 코시분포는 **대수의법칙이 실패하는** 대표적인 예이다. 평균이 유한하다는 조건이 수학적 형식이 아니라 진짜 요구사항임을 보여준다.
- 코시 자료의 표본평균은 관측값 하나와 **같은 분포**를 가지므로 평균을 내도 전혀 나아지지 않는다.
- **두꺼운 꼬리**가 근본 원인이다: 이따금 나타나는 극단 관측값이 합을 지배하여 집중을 막는다.
- **중앙값**과 **MLE**는 통상적인 $1/\sqrt{n}$ 속도로 수렴하는 쓸 만한 대안이다.
- 실무에서 코시는 경고 역할을 한다: 추정량이 자료에 적절한지 항상 확인하라. 자료의 꼬리가 극단적으로 두꺼우면 표본평균이 오해를 부르거나 무의미할 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
적분을 직접 계산하여 표준 코시분포에서 $E[|X|] = \infty$임을 보여라.

</div>

??? success "풀이"
    $$E[|X|] = \int_{-\infty}^{\infty} \frac{|x|}{\pi(1+x^2)}\,dx = \frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx$$

    치환 $u = 1 + x^2$, $du = 2x\,dx$를 쓰면:

    $$\frac{2}{\pi}\int_0^{\infty} \frac{x}{1+x^2}\,dx = \frac{2}{\pi}\cdot\frac{1}{2}\int_1^{\infty}\frac{du}{u} = \frac{1}{\pi}\left[\ln u\right]_1^{\infty} = \frac{1}{\pi}\cdot\infty = \infty$$

    $E[|X|] = \infty$이므로 평균 $E[X]$가 존재하지 않는다. 이것이 코시에서 대수의법칙이 실패하는 근본 이유이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
특성함수를 써서, i.i.d. 표준 코시 확률변수 $n$개의 $\bar{X}_n$이 표준 코시 확률변수 하나와 같은 분포를 가짐을 증명하라.

</div>

??? success "풀이"
    표준 코시의 특성함수는 $\varphi_X(t) = e^{-|t|}$이다.

    i.i.d. $X_1, \ldots, X_n$에 대해 합 $S_n = \sum_{i=1}^n X_i$은:

    $$\varphi_{S_n}(t) = \prod_{i=1}^n \varphi_{X_i}(t) = \left(e^{-|t|}\right)^n = e^{-n|t|}$$

    이는 $\text{Cauchy}(0, n)$(척도모수가 $n$인 Cauchy)의 특성함수이다.

    표본평균 $\bar{X}_n = S_n/n$에 대해:

    $$\varphi_{\bar{X}_n}(t) = \varphi_{S_n}(t/n) = e^{-n|t/n|} = e^{-|t|}$$

    이는 정확히 표준 코시의 특성함수 $\varphi_X(t)$이다. 특성함수가 분포를 유일하게 결정하므로:

    $$\bar{X}_n \sim \text{Cauchy}(0, 1) \quad \text{모든 } n \geq 1 \text{에 대해}$$

    $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
코시와 정규분포의 꼬리를 비교하라. 각각에서 $P(|X| > 10)$은 얼마인가? 이것이 표본평균의 거동에 대해 무엇을 함의하는가?

</div>

??? success "풀이"
    **정규:** $P(|Z| > 10) = 2\mathcal{N}(-10) \approx 2 \times 7.62 \times 10^{-24} \approx 1.52 \times 10^{-23}$.

    **Cauchy:** $P(|X| > 10) = 2\left(\frac{1}{2} - \frac{1}{\pi}\arctan(10)\right) = 1 - \frac{2}{\pi}\arctan(10) \approx 1 - \frac{2}{\pi}(1.4711) \approx 1 - 0.9366 = 0.0634$.

    즉 코시에서는 $P(|X| > 10) \approx 6.3\%$로 정규보다 $10^{21}$배 넘게 크다.

    **표본평균에 대한 함의:** 코시 관측값 $n = 100$개의 표본에서는 절댓값이 10을 넘는 값이 대략 6개 나올 것으로 기대된다. 이런 극단값이 합에 불균형하게 기여하여 온건한 관측값들을 압도한다. 그런 극단값의 발생률이 $n$이 커져도 충분히 빨리 줄지 않으므로 $\bar{X}_n$의 $1/n$ 정규화가 합을 "길들이지" 못한다. 이것이 표본평균이 수렴하지 않는 직관적인 이유이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
표본중앙값은 코시 위치모수에 대해 일치한다. 그 점근분산은 얼마인가? 코시 위치모수 MLE의 점근분산과 비교하라.

</div>

??? success "풀이"
    코시 밀도 $f(x) = \frac{1}{\pi(1+x^2)}$에서 (표준 코시의 중앙값인) 0에서의 값은 $f(0) = 1/\pi$이다.

    표본중앙값의 점근분산은:

    $$\text{Var}(\text{median}) \approx \frac{1}{4nf(0)^2} = \frac{1}{4n(1/\pi)^2} = \frac{\pi^2}{4n} \approx \frac{2.467}{n}$$

    코시 위치모수 $\mu$에 대한 Fisher 정보량은:

    $$I(\mu) = \int_{-\infty}^{\infty} \frac{[f'(x-\mu)]^2}{f(x-\mu)}\,dx = \frac{1}{2}$$

    따라서 CRLB(그리고 MLE의 점근분산)는:

    $$\text{Var}(\hat{\mu}_{\text{MLE}}) \approx \frac{1}{nI(\mu)} = \frac{2}{n}$$

    MLE 대비 중앙값의 점근 상대효율은:

    $$\text{ARE} = \frac{2/n}{\pi^2/(4n)} = \frac{8}{\pi^2} \approx 0.811$$

    즉 코시 자료에서 중앙값의 효율은 MLE의 약 81%이다 — 중앙값의 단순함과 계산의 편의를 생각하면 합리적인 맞바꿈이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
"안정분포"가 무엇인지, 코시가 왜 안정분포인지 설명하라. 코시의 안정지수는 얼마이며, 그것이 꼬리 거동에 대해 무엇을 결정하는가?

</div>

??? success "풀이"
    확률변수 $X$가 지수 $\alpha \in (0, 2]$인 **안정분포**를 따른다는 것은, 임의의 $n$개 i.i.d. 복사본 $X_1, \ldots, X_n$에 대해 어떤 상수 $c_n$이 존재하여

    $$X_1 + X_2 + \cdots + X_n \overset{d}{=} n^{1/\alpha} X + c_n$$

    이 성립한다는 뜻이다. 동등하게, 이 족은 (위치와 척도를 조정하면) 덧셈에 대해 닫혀 있다. 대칭 안정분포의 특성함수는 $\varphi(t) = \exp(-c|t|^\alpha + i\delta t)$ 꼴이다.

    **코시는 $\alpha = 1$이다:**

    - 특성함수: $\varphi(t) = e^{-|t|}$, 즉 $e^{-|t|^1}$이므로 $\alpha = 1$.
    - 합의 성질: $S_n = \sum X_i$은 $\varphi_{S_n}(t) = e^{-n|t|}$이므로 $\text{Cauchy}(0, n)$에 해당한다. 이는 $n^{1/\alpha}X = n^1 X \sim \text{Cauchy}(0, n)$과 일치한다.

    **안정지수 $\alpha$가 꼬리 거동을 결정한다:**

    - ($\alpha < 2$일 때) $t \to \infty$에서 $P(|X| > t) \sim t^{-\alpha}$.
    - $\alpha = 2$: Gaussian(지수 꼬리, 모든 적률이 유한).
    - $\alpha = 1$: Cauchy($P(|X|>t) \sim 1/t$, 평균 없음).
    - $0 < \alpha < 1$: 꼬리가 더 두꺼움(평균 없음, $P(|X|>t) \sim t^{-\alpha}$).

    $\alpha$가 작을수록 꼬리가 두껍다. $\alpha < 2$이면 차수가 $\alpha$ 이상인 적률이 무한하다. $\alpha \leq 1$이면 평균이 존재하지 않고 표본평균에 대해 대수의법칙이 실패한다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
코시 위치모수의 **MLE**를 구하는 방정식을 쓰고, 그 점근분산이 중앙값보다 작음을 확인하라. 그럼에도 실무에서 중앙값을 쓰는 이유는?

</div>

??? success "풀이"
    **점수방정식.** $f(x;\theta)=1/\{\pi(1+(x-\theta)^2)\}$이므로

    $$
    \ell(\theta) = -\sum_i\ln\left\{1+(x_i-\theta)^2\right\}+\text{상수}
    $$

    $$
    \ell'(\theta) = \sum_i\frac{2(x_i-\theta)}{1+(x_i-\theta)^2} = 0
    $$

    **닫힌 해가 없고** 수치적으로 풀어야 한다.

    **피셔 정보량.**

    $$
    I_1(\theta) = \int_{-\infty}^\infty \frac{\{2u/(1+u^2)\}^2}{\pi(1+u^2)}du = \frac12
    $$

    이므로 MLE의 점근분산이 $2/n$이다.

    **중앙값.** $f(\theta)=1/\pi$이므로

    $$
    \operatorname{Var}(\tilde X) \approx \frac{1}{4nf(\theta)^2} = \frac{\pi^2}{4n} = \frac{2.467}{n}
    $$

    **비교.**

    $$
    \text{ARE}(\tilde X,\ \hat\theta_{\text{MLE}}) = \frac{2/n}{2.467/n} = \frac{8}{\pi^2} = 0.811
    $$

    **중앙값의 효율이 81%다.** MLE가 낫지만 차이가 크지 않다.

    **그럼에도 중앙값을 쓰는 이유.**

    1. **계산이 간단하다.** 정렬만 하면 되고 수치 최적화가 필요 없다.
    2. **MLE가 다봉이다.** 앞서 본 대로 코시 로그가능도는 국소 최대가 여럿일 수 있어, 잘못된 봉우리에 빠지면 결과가 나쁘다. 중앙값은 유일하게 정해진다.
    3. **중앙값이 좋은 초기값이다.** 실무에서는 중앙값에서 출발해 뉴턴 반복을 몇 번 돌리는 절충이 흔하다. 그러면 다봉 문제를 대체로 피하면서 MLE의 효율을 얻는다.
    4. **모형이 정확히 코시가 아닐 수 있다.** MLE의 우월성은 코시 가정 아래에서만 성립한다. 중앙값은 어떤 대칭 연속분포에서도 중심을 일치추정한다.

    **효율 81%가 시사하는 것.** 정규분포에서 중앙값의 효율이 64%였던 것과 견주면, **꼬리가 두꺼울수록 중앙값의 상대적 성능이 좋아진다.** 코시는 그 극단이며, 표본평균은 아예 일치하지도 않는다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
코시분포에서 표본평균이 수렴하지 않는데 **표본 절사평균**은 어떤가? 절사 비율에 따라 어떻게 달라지는지 논하라.

</div>

??? success "풀이"
    **결론부터.** 어떤 $\alpha>0$에 대해서도 $\alpha$ 절사평균은 **일치추정량**이다. 절사 비율이 아무리 작아도 그렇다.

    **왜 그런가.** 절사평균은 절사된 분포의 평균을 추정하는데, 코시분포를 양끝에서 자르면

    $$
    \int_{q_\alpha}^{q_{1-\alpha}}x\,f(x)\,dx
    $$

    가 **유한**하다. 무한을 만드는 것은 꼬리인데 그것을 잘라 냈기 때문이다. 대칭성에서 이 값이 0(위치모수)이고, 큰수의 법칙이 적용된다.

    **점근분산.** 절사 비율 $\alpha$에 대한 절사평균의 점근분산을 계산하면 $\alpha$가 작을수록 커진다.

    | $\alpha$ | 점근분산($\times n$) | 효율(MLE 대비) |
    |---|---|---|
    | 0 | $\infty$ | 0 |
    | 0.05 | 12.0 | 0.17 |
    | 0.10 | 5.8 | 0.34 |
    | 0.25 | 3.0 | 0.67 |
    | 0.38 | 2.6 | **0.77** |
    | 0.50(중앙값) | 2.47 | 0.81 |

    **$\alpha\to0$에서 분산이 발산한다.** 조금만 자르면 일치하기는 하지만 효율이 형편없다.

    **정규분포와 정반대다.** 정규에서는 $\alpha$가 커질수록 효율이 **떨어졌는데**(0에서 최대), 코시에서는 $\alpha$가 커질수록 **올라간다**(중앙값에서 최대).

    **일반 원리.** **최적 절사 비율은 꼬리의 두께가 정한다.**

    | 분포 | 최적 $\alpha$(근사) |
    |---|---|
    | 정규 | 0 |
    | $t_{10}$ | 0.05 |
    | $t_5$ | 0.10 |
    | $t_3$ | 0.20 |
    | 코시 | 0.50 |

    **실무의 함의.** 꼬리를 모르면 $\alpha=0.1\sim0.2$가 **넓은 범위에서 무난한 선택**이다. 정규에서 3~7%만 잃고 꼬리가 두꺼우면 큰 이득을 본다. 이것이 20% 절사평균이 널리 권장되는 근거다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
코시분포에서 **신뢰구간**을 만드는 방법을 두 가지 적어라. 표본평균에 기반한 방법이 왜 실패하는가?

</div>

??? success "풀이"
    **표본평균 방법이 실패하는 이유.**

    - $\bar X_n$이 $\theta$로 수렴하지 않는다. 분포가 $n$과 무관하게 $\text{Cauchy}(\theta,1)$이다.
    - $s$가 수렴하지 않고 $n$과 함께 커진다. $s/\sqrt n$이 무엇을 재는지 알 수 없다.
    - 따라서 $\bar x\pm t\,s/\sqrt n$의 포함확률이 명목값과 전혀 맞지 않으며, $n$을 키워도 나아지지 않는다.

    **방법 1 — 중앙값 기반.** $\tilde X$의 점근분포가

    $$
    \sqrt n(\tilde X-\theta)\xrightarrow{d}N\!\left(0,\ \frac{\pi^2}{4}\right)
    $$

    이므로

    $$
    \tilde x \pm 1.96\cdot\frac{\pi}{2\sqrt n}
    $$

    이 근사 95% 구간이다. $n=100$이면 반폭이 $1.96\times0.157 = 0.308$이다.

    **정확한 구간도 있다.** 중앙값의 정확한 구간은 순서통계량으로 만든다. $P(X_{(k)}<\theta<X_{(n-k+1)})$이 이항분포로 계산되므로, 분포 가정 없이 **분포무관 구간**을 얻는다.

    $$
    P\left(X_{(k)}<\theta<X_{(n-k+1)}\right) = 1-2\,P\left(\text{Binomial}(n,\tfrac12)<k\right)
    $$

    **방법 2 — 프로파일 가능도.** 코시 가능도가 정확히 알려져 있으므로

    $$
    \left\{\theta:\ 2\{\ell(\hat\theta)-\ell(\theta)\}\le3.841\right\}
    $$

    를 구간으로 쓴다. MLE 기반이라 효율이 가장 좋고, 로그가능도의 실제 모양을 반영한다. 다만 다봉이면 구간이 **연결되지 않을 수 있다.**

    **방법 3 — 부트스트랩.** 중앙값이나 절사평균에 대해서는 작동한다. **표본평균에는 쓰면 안 된다.** 극한분포가 정규가 아니므로 표준 부트스트랩이 일치하지 않는다.

    **권고.** 분포를 코시로 확신하면 프로파일 가능도, 확신할 수 없으면 **순서통계량 기반 중앙값 구간**이 가장 안전하다. 후자는 연속분포이기만 하면 정확하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
코시분포가 실제로 나타나는 상황을 세 가지 들고, 각각에서 왜 그 분포가 나오는지 설명하라.

</div>

??? success "풀이"
    **(1) 두 독립 정규의 비.** $Z_1,Z_2\sim N(0,1)$이 독립이면

    $$
    \frac{Z_1}{Z_2}\sim\text{Cauchy}(0,1)
    $$

    이다. 분모가 0 근처의 값을 가질 확률이 양수이므로 비가 폭발한다.

    **어디에 나타나는가.** 두 추정값의 비를 다룰 때다. 도구변수 추정량 $\hat\beta_{\text{IV}} = \widehat{\text{cov}}(z,y)/\widehat{\text{cov}}(z,x)$가 그 구조이며, **도구가 약하면 분모가 0 근처라 코시에 가까운 분포**가 된다. 앞서 본 약한 도구 문제의 수학적 정체다.

    피셔의 $t$ 분포도 $\nu=1$에서 코시이며, 역시 비의 구조에서 나온다.

    **(2) 회전하는 광원.** 직선에서 거리 1만큼 떨어진 점에 광원이 있고, 각도 $\Theta\sim\text{Uniform}(-\pi/2,\pi/2)$로 무작위로 빛을 쏜다. 직선에 닿는 위치가

    $$
    X = \tan\Theta \sim \text{Cauchy}(0,1)
    $$

    이다. 각도가 $\pm\pi/2$에 가까우면 아주 먼 곳에 닿으므로 꼬리가 두꺼워진다.

    **물리적 대응물.** 등대, 회전하는 입자 검출기, 방향 자료. 실제로 **공명의 선폭**이 코시분포(물리학에서는 로렌츠 분포)를 따른다.

    **(3) 안정분포의 특수한 경우.** $\alpha=1$인 대칭 안정분포가 코시다. 합에 대해 닫혀 있어

    $$
    \bar X_n \sim \text{Cauchy}(\theta,1)
    $$

    이 $n$과 무관하다. **꼬리 지수가 1인 거듭제곱 꼬리 현상**이 있으면 코시류가 나타난다. 일부 금융 수익률, 지진 규모, 네트워크 파일 크기가 그렇다(다만 대개 $\alpha$가 1보다 크다).

    **공통 구조.** 세 경우 모두 **나눗셈이나 탄젠트처럼 0 근처에서 폭발하는 변환**이 관여한다. 그것이 거듭제곱 꼬리를 만들고, 지수가 1이면 평균조차 없어진다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
자료가 코시에 가까운지 **진단**하는 방법을 적고, 코시로 판정되면 분석 전략을 어떻게 바꿔야 하는지 정리하라.

</div>

??? success "풀이"
    **진단.**

    1. **누적평균 그림.** 수렴하지 않고 계단처럼 뛰면 강한 신호다. 코시에서는 $n$이 커져도 큰 점프가 계속 일어난다.

    2. **누적 $s$ 그림.** 표본표준편차가 $n$과 함께 계속 커지면 2차 적률이 없다.

    3. **로그-로그 생존함수.** 기울기가 $-1$에 가까우면 $\alpha\approx1$로 코시류다.

    4. **Q-Q 그림.** 정규 Q-Q에서 양끝이 극도로 휘어 거의 수직이 된다. 코시 Q-Q(`stats.probplot(x, dist=stats.cauchy)`)를 그려 직선이면 확증이다.

    5. **힐 추정량.** 꼬리 지수 $\hat\alpha$를 추정한다. 1 근처면 코시류이고, 1 이하면 평균이 없다.

    **전략의 변경.**

    | 하던 것 | 바꿀 것 |
    |---|---|
    | 표본평균 | 중앙값, 절사평균(높은 $\alpha$), 또는 MLE |
    | $s/\sqrt n$ 표준오차 | 중앙값의 점근 표준오차, 순서통계량 구간 |
    | $t$ 검정 | 부호검정, 윌콕슨, 순열검정 |
    | 최소제곱 회귀 | 최소절대편차(LAD) 회귀, 분위수회귀 |
    | 상관계수 | 스피어만, 켄달 |
    | 표준 부트스트랩 | $m$-out-of-$n$ 부트스트랩, 서브샘플링 |
    | 분산·표준편차 보고 | IQR, MAD, 분위수 범위 |

    **개념적 전환.** 가장 중요한 것은 **"평균"과 "분산"이라는 언어를 버리는 것**이다. 존재하지 않는 양을 추정하려 애쓰는 대신, 언제나 존재하는 **분위수**로 문제를 다시 쓴다.

    - "평균 손실이 얼마인가" → "중앙 손실이 얼마인가", "상위 5% 손실이 얼마인가"
    - "변동성이 얼마인가" → "사분위수 범위가 얼마인가"
    - "평균이 다른가" → "분포가 다른가"(콜모고로프-스미르노프, 순열검정)

    **모형을 쓴다면.** 코시나 안정분포를 직접 적합하는 것도 방법이다. 다만 안정분포는 밀도가 닫힌 형태가 없어 특성함수 기반 추정이나 분위수 적합이 필요하다. 꼬리만 관심이면 **극단값 이론**(일반화파레토)이 더 실용적이다.

---

## 정리하며

코시분포는 큰수의 법칙에 대한 **가장 극적인 반례**다.

- **$\mathbb{E}|X|=\infty$ 이므로 수렴할 대상 자체가 없다.** 밀도가 $1/(\pi(1+x^2))$ 이라 $x f(x)\sim1/(\pi x)$ 로 양쪽 꼬리가 모두 발산한다.
- **$\bar X_n$ 이 $n$ 과 무관하게 표준 코시를 따른다.** 관측을 백만 개 모으는 것이 하나만 보는 것과 **정확히 같다.** 궤적 그림에서 누적평균이 정착하지 않고 이따금 크게 튀며, Q-Q 그림에서 두꺼운 꼬리가 드러난다.
- **정규와 대조하면 차이가 선명하다.** 같은 그림에서 정규의 누적평균은 곧게 모여들고 코시는 끝까지 헤맨다.
- **처방은 중앙값이다.** 코시분포의 중앙값은 잘 정의되고 표본중앙값이 그것으로 수렴한다. **평균이 없는 곳에서도 분위수는 언제나 존재한다.**
- **어디서 마주치는가.** 두 독립 정규의 비가 코시이므로, 회귀계수의 비·두 추정값의 비·각도 측정에서 자연스럽게 나타난다. **비를 다룰 때 조심해야 하는 이유다.**

**이것으로 7장이 끝난다.** 표본평균과 표본분산을 여러 각도에서 살피고, 불편성·효율성·로버스트성의 거래 조건을 따졌으며, 마지막으로 그 모든 것이 무너지는 경우까지 보았다.

다음 장 **구간추정**으로 넘어간다. 지금까지 점 하나로 답했다면, 이제 **불확실성의 범위를 함께 보고하는** 방법을 다룬다.
