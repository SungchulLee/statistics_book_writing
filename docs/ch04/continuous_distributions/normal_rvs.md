# 정규 난수 생성

## 개요

분포로부터 **난수(확률표본)**를 뽑는 것은 모의실험에 기반한 통계학의 기본이다. 정규분포에서는 `stats.norm.rvs()`가 $N(\mu, \sigma^2)$로부터 표본을 생성한다. 표본크기가 커질수록 표본의 히스토그램은 이론적 PDF로 수렴하며, 이는 큰수의 법칙을 직접 보여 주는 예이다.

---

## 표본 생성하기

<div class="codebox" markdown>

### 예제 1. 정규 표본을 뽑아 이론 밀도와 견주기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu = 1
sigma = 2       # scipy의 scale은 분산이 아니라 **표준편차**다

# 이론적 밀도함수. 우리가 참값을 알고 있는 곡선이다.
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 200)
y = stats.norm(loc=mu, scale=sigma).pdf(x)

# 같은 분포에서 난수 1만 개를 뽑는다.
n_samples = 10_000
samples = stats.norm(loc=mu, scale=sigma).rvs(size=n_samples)

fig, ax = plt.subplots(figsize=(12, 3))
# density=True 가 필수다. 이것이 없으면 y축이 도수(개수)가 되어
# 밀도곡선과 눈금이 달라 겹쳐 그릴 수 없다.
ax.hist(samples, bins=50, density=True, alpha=0.3, color='C0',
        label='Random Samples (rvs)')
# 히스토그램이 이 곡선을 얼마나 잘 따라가는지가 표본의 충실도다
ax.plot(x, y, 'r-', lw=2, label='Theoretical PDF')
ax.set_title(f"Normal(μ={mu}, σ={sigma}) — PDF vs. Random Samples")
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend()
ax.grid(True, linestyle=':')
plt.show()
```

![정규 난수 생성](./img/normal_rvs_11.png)

핵심 옵션은 `hist()`의 `density=True`이다. 히스토그램을 정규화하여 전체 넓이를 1로 만들어 주므로 PDF 곡선과 직접 비교할 수 있다.

</div>

---

## 히스토그램이 PDF를 근사하는 이유

$x_0$을 중심으로 폭이 $\Delta x$인 구간에 대해, 그 구간에 들어가는 표본의 기대 비율은 근사적으로 $f(x_0)\,\Delta x$이며 $f$는 PDF이다. `density=True`로 두면 $x_0$에서의 히스토그램 높이가 $f(x_0)$을 추정한다. 큰수의 법칙에 의해 이 추정값은 $n \to \infty$일 때 참 밀도로 수렴한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$N(0, 1)$에서 100개, 1000개, 10000개의 표본을 생성하고 히스토그램을 같은 그림에 겹쳐 그려라. 표본크기가 커질수록 적합이 어떻게 좋아지는지 서술하라.

</div>

??? success "풀이"
    ```python
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))
    x = np.linspace(-4, 4, 200)
    pdf = stats.norm.pdf(x)
    # 표본 크기만 10배씩 키우고 나머지는 모두 같게 둔다.
    # 구간 수(bins=30)를 고정했으므로 구간당 평균 도수가 3.3 -> 33 -> 333 으로
    # 늘어나고, 그만큼 막대 높이의 상대적 잡음이 줄어든다.
    for ax, n in zip(axes, [100, 1000, 10000]):
        samples = stats.norm.rvs(size=n)
        ax.hist(samples, bins=30, density=True, alpha=0.4)
        ax.plot(x, pdf, 'r-', lw=2)
        ax.set_title(f"n = {n}")
    ```

    ![정규 난수 생성](./img/normal_rvs_56.png)

    $n = 100$에서는 히스토그램이 들쭉날쭉하고 눈에 띄게 벗어난다. $n = 1000$에서는 종 모양을 알아볼 수 있지만 여전히 울퉁불퉁하다. $n = 10000$에서는 히스토그램이 PDF를 바짝 따라간다. 수렴 속도는 대략 $O(1/\sqrt{n})$이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$X \sim N(\mu, \sigma^2)$에서 $n$개의 표본 $X_1, \ldots, X_n$을 뽑을 때 $E[\bar{X}]$와 $\text{Var}(\bar{X})$는 무엇인가? 크기 $n = 50$인 표본평균을 10000개 생성하여 수치적으로 확인하라.

</div>

??? success "풀이"
    $E[\bar{X}] = \mu$이고 $\text{Var}(\bar{X}) = \sigma^2/n$이다.

    ```python
    np.random.seed(1)      # 시드를 고정해야 아래 출력이 재현된다

    mu, sigma, n = 5, 3, 50
    # 크기 50짜리 표본을 1만 번 뽑아 그때마다 표본평균을 기록한다
    means = [stats.norm(mu, sigma).rvs(n).mean() for _ in range(10000)]
    print(f"E[X_bar] ≈ {np.mean(means):.4f}  (theory: {mu})")
    print(f"Var(X_bar) ≈ {np.var(means):.4f}  (theory: {sigma**2/n:.4f})")
    ```

    출력:

    ```
    E[X_bar] ≈ 5.0033  (theory: 5)
    Var(X_bar) ≈ 0.1816  (theory: 0.1800)
    ```

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
`stats.norm.rvs(size=n)`과 `np.random.normal(0, 1, n)`의 차이를 설명하라. 어느 쪽을 언제 선호하겠는가?

</div>

??? success "풀이"
    둘 다 표준정규 난수를 생성한다. `np.random.normal`은 NumPy의 난수 생성기를 직접 호출하므로 단순한 경우 약간 더 빠르다. `stats.norm.rvs`는 고정된 분포 인터페이스를 사용하여 더 유연하다. 분포 객체를 한 번 만들어 두고 같은 객체에 `.rvs()`, `.pdf()`, `.cdf()` 등을 호출할 수 있다. 같은 분포에 여러 메서드가 필요하면 SciPy 인터페이스를, 촘촘한 반복문에서 속도를 최대로 내려면 NumPy 직접 호출을 사용하라.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Box-Muller 변환으로 균등확률변수로부터 표준정규 표본을 생성하라. 변환은 다음과 같다. $U_1, U_2 \sim \text{Uniform}(0,1)$이 독립일 때,

$$
Z_1 = \sqrt{-2\ln U_1}\,\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\,\sin(2\pi U_2)
$$

$Z_1$과 $Z_2$가 근사적으로 $N(0,1)$임을 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    np.random.seed(0)
    n = 10000

    # 균등난수 두 개로 독립인 표준정규난수 두 개를 만든다.
    #   U1 -> 반지름 R = sqrt(-2 ln U1)   (레일리 분포)
    #   U2 -> 각도  theta = 2*pi*U2       (0~2pi 균등)
    # 극좌표 (R, theta)를 직교좌표로 옮기면 두 성분이 각각 N(0,1)이 된다.
    U1 = np.random.uniform(0, 1, n)
    U2 = np.random.uniform(0, 1, n)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)

    print(f"Z1: 평균 {Z1.mean():+.4f}  표준편차 {Z1.std():.4f}")
    print(f"Z2: 평균 {Z2.mean():+.4f}  표준편차 {Z2.std():.4f}")
    # 같은 U1을 공유하는데도 두 출력은 독립이다. 상관계수로 확인한다.
    print(f"corr(Z1, Z2) = {np.corrcoef(Z1, Z2)[0, 1]:+.4f}")
    ```

    출력:

    ```
    Z1: 평균 +0.0126  표준편차 1.0169
    Z2: 평균 +0.0163  표준편차 1.0013
    corr(Z1, Z2) = -0.0084
    ```

    평균은 0, 표준편차는 1에 가깝고 상관계수도 0에 가깝다. **같은 `U1`을 쓰는데도 독립**이라는 점이 이 변환의 묘미다. `U1`은 반지름만 정하고 `U2`가 각도를 정하는데, 등방적인 2차원 정규분포에서는 반지름과 각도가 서로 독립이기 때문이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`np.random.seed(0)`과 `rng = np.random.default_rng(0)` 가운데 요즘 권장되는 쪽은 무엇이며 이유는 무엇인가?

</div>

??? success "풀이"
    `np.random.default_rng(0)`이 권장된다.

    `np.random.seed()`는 모듈 전체가 공유하는 **전역 상태**를 바꾼다. 내가 부른 함수 안에서 누군가 또 시드를 건드리면 내 결과가 조용히 달라지고, 병렬로 여러 작업을 돌리면 서로의 난수 흐름이 얽힌다. 어느 코드가 난수를 몇 개 썼는지에 결과가 의존하므로, 중간에 코드 한 줄을 넣기만 해도 이후 모든 값이 바뀐다.

    `default_rng`는 독립된 생성기 객체를 준다.

    ```python
    rng = np.random.default_rng(0)
    print(rng.normal(size=3))   # [ 0.12573022 -0.13210486  0.64042265]

    rng = np.random.default_rng(0)
    print(rng.normal(size=3))   # 같은 값이 다시 나온다
    ```

    생성기를 함수 인자로 넘기면 그 함수의 난수 소비가 바깥에 영향을 주지 않고, 병렬 작업마다 `rng.spawn(k)`로 서로 겹치지 않는 생성기를 나눠 줄 수도 있다. 밑바탕 알고리즘도 메르센 트위스터에서 PCG64로 바뀌어 더 빠르고 통계적 성질이 낫다.

    SciPy에서는 `stats.norm.rvs(size=n, random_state=rng)`처럼 넘긴다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
모의실험으로 확률 $p \approx 0.05$를 추정하려 한다. 95% 신뢰수준에서 오차가 $\pm 0.001$ 안에 들도록 하려면 반복 횟수 $B$가 얼마나 필요한가? 자릿수를 하나 더 얻으려면 몇 배가 필요한가?

</div>

??? success "풀이"
    $B$번의 반복에서 사건이 일어난 비율 $\hat p$는 $\text{Binomial}(B, p)/B$이므로

    $$
    \operatorname{SE}(\hat p) = \sqrt{\frac{p(1-p)}{B}}
    $$

    이다. 95% 오차한계가 $1.96\,\operatorname{SE} \le 0.001$이어야 하므로 $\operatorname{SE} \le 0.00051$이고

    $$
    B \ge \frac{p(1-p)}{\operatorname{SE}^2} = \frac{0.05 \times 0.95}{(0.00051)^2} \approx 182{,}476
    $$

    이다. 약 18만 번이 필요하다.

    오차한계가 $B^{-1/2}$에 비례하므로, 오차를 10분의 1로 줄이려면 $B$를 **100배**로 늘려야 한다. 소수점 한 자리를 더 얻는 값이 반복 100배다. 몬테카를로가 원리는 단순해도 고정밀도에는 비싼 방법인 이유이며, 대조변량·중요도추출 같은 분산감소 기법이 연구되는 까닭이기도 하다.

    거꾸로 읽는 쪽이 더 유용할 때도 많다. $B = 10{,}000$이면 $p = 0.05$ 근처에서 오차한계가 $1.96\sqrt{0.05 \times 0.95/10^4} \approx 0.0043$이다. 모의실험 결과를 "0.0500"처럼 네 자리로 적는 것은 있지도 않은 정밀도를 주장하는 셈이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표준정규 난수만 만들 수 있는 생성기로 (가) $N(\mu, \sigma^2)$ 표본과 (나) 평균 $\boldsymbol\mu$, 공분산 $\Sigma$인 다변량 정규 표본을 어떻게 만드는지 적어라.

</div>

??? success "풀이"
    **(가) 일변량.** $Z \sim N(0,1)$에 대해 $X = \mu + \sigma Z$로 두면 $E[X] = \mu$이고 $\operatorname{Var}(X) = \sigma^2\operatorname{Var}(Z) = \sigma^2$이다. 정규분포는 선형변환에 대해 닫혀 있으므로 $X \sim N(\mu, \sigma^2)$이다.

    **(나) 다변량.** $\Sigma$가 양정부호이면 촐레스키 분해로 $\Sigma = LL^\top$인 하삼각행렬 $L$을 얻는다. $\mathbf{Z}$를 성분이 독립인 표준정규 벡터라 하고

    $$
    \mathbf{X} = \boldsymbol\mu + L\mathbf{Z}
    $$

    로 두면 $E[\mathbf{X}] = \boldsymbol\mu$이고

    $$
    \operatorname{Cov}(\mathbf{X}) = L\operatorname{Cov}(\mathbf{Z})L^\top = LIL^\top = LL^\top = \Sigma
    $$

    이다. 정규벡터의 선형변환이 다시 정규벡터이므로 $\mathbf{X} \sim N(\boldsymbol\mu, \Sigma)$이다.

    ```python
    L = np.linalg.cholesky(Sigma)
    X = mu + Z @ L.T          # Z의 모양이 (n, d)일 때
    ```

    촐레스키가 실패하면($\Sigma$가 양정부호가 아니면) 고유분해를 써서 $\Sigma = Q\Lambda Q^\top$에서 $L = Q\Lambda^{1/2}$로 두고, 음수 고윳값은 0으로 자른다. `np.random.default_rng().multivariate_normal`이 내부에서 이런 처리를 한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
예제 1은 $n = 10{,}000$개 표본을 `bins=50`으로 그렸다. 스콧의 규칙 $h = 3.49\, s\, n^{-1/3}$으로 적절한 구간 폭을 구하고 구간 수를 확인하라. 구간 수를 너무 크거나 작게 잡으면 무엇이 문제인가?

</div>

??? success "풀이"
    $s \approx \sigma = 2$이고 $n = 10^4$이므로 $n^{-1/3} = 1/21.54$이고

    $$
    h = 3.49 \times 2 \times \frac{1}{21.54} \approx 0.324
    $$

    이다. 그림이 $\mu \pm 4\sigma$, 즉 폭 16 정도를 덮으므로 구간 수는 $16/0.324 \approx 49$개다. 코드의 `bins=50`이 이 규칙과 거의 일치한다.

    **구간이 너무 많으면** 구간당 도수가 적어 막대 높이가 들쭉날쭉해진다. 편향은 작지만 분산이 크다. 극단적으로 구간마다 관측값이 0개나 1개면 히스토그램은 잡음만 보여 준다.

    **구간이 너무 적으면** 매끄럽긴 하지만 분포의 모양을 뭉갠다. 봉우리가 둘인 분포가 하나로 보이거나 치우침이 사라진다. 편향은 크고 분산은 작다.

    스콧의 규칙은 적분평균제곱오차를 최소로 하도록 이 맞바꿈을 푼 결과이며, 밑에 깔린 분포가 정규분포라는 가정에서 유도되었다. 치우쳤거나 봉우리가 여럿인 자료에는 IQR을 쓰는 프리드먼-디아코니스 규칙 $h = 2\,\text{IQR}\,n^{-1/3}$이 더 강건하다. `bins='auto'`가 두 규칙 중 큰 쪽을 고른다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
박스-뮐러 변환이 정확히 독립인 표준정규 난수 둘을 만든다는 것을 야코비안으로 증명하라.

</div>

??? success "풀이"
    거꾸로 가는 편이 쉽다. $Z_1, Z_2$가 독립인 표준정규라 하면 결합밀도는

    $$
    f(z_1, z_2) = \frac{1}{2\pi}\exp\!\left(-\frac{z_1^2 + z_2^2}{2}\right)
    $$

    이다. 극좌표 $z_1 = r\cos\theta$, $z_2 = r\sin\theta$로 바꾸면 야코비안이 $r$이므로

    $$
    f(r, \theta) = \frac{r}{2\pi}e^{-r^2/2}, \qquad r > 0,\ \theta \in [0, 2\pi)
    $$

    이다. 이 밀도가 $r$만의 함수와 $\theta$만의 함수의 곱으로 쪼개지므로 $R$과 $\Theta$가 **독립**이고, $\Theta \sim \text{Uniform}(0, 2\pi)$이며 $R$의 밀도는 $re^{-r^2/2}$(레일리분포)이다.

    이제 $S = R^2$으로 두면 $ds = 2r\,dr$이므로

    $$
    f_S(s) = \frac{r e^{-r^2/2}}{2r} = \frac12 e^{-s/2}, \qquad s > 0
    $$

    로 $S \sim \text{Exp}(1/2)$이다. 지수분포의 역변환에 따라 $U_1 \sim \text{Uniform}(0,1)$일 때 $S = -2\ln U_1$이 이 분포를 따르므로 $R = \sqrt{-2\ln U_1}$이다. 또 $\Theta = 2\pi U_2$가 $[0, 2\pi)$ 위의 균등분포를 준다.

    이 둘을 직교좌표로 되돌린 것이 바로

    $$
    Z_1 = R\cos\Theta = \sqrt{-2\ln U_1}\cos(2\pi U_2), \qquad Z_2 = R\sin\Theta = \sqrt{-2\ln U_1}\sin(2\pi U_2)
    $$

    이다. 위 논증을 거꾸로 읽으면 이렇게 만든 $(Z_1, Z_2)$의 결합밀도가 정확히 독립인 표준정규 두 개의 밀도가 된다. $\square$

    **근사가 아니라 정확한 변환**이라는 점이 요점이다. 핵심은 2차원 표준정규분포가 회전에 대해 불변이라는 사실이고, 그 덕분에 반지름과 각도가 독립으로 갈라진다. 실무에서는 삼각함수가 비싸서 극좌표를 쓰지 않는 마사글리아 변형이나 지구랏을 쓴다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$N(\mu, \sigma^2)$에서 크기 $n = 20$인 표본을 $B = 10{,}000$번 뽑아 매번 $t$ 신뢰구간을 만들고, 그 구간이 참 $\mu$를 담는 비율을 세는 모의실험을 설계하라. 결과가 정확히 0.95가 아니어도 되는 이유를 말하라.

</div>

??? success "풀이"
    각 반복에서 표본을 뽑아 $\bar x \pm t_{0.975, n-1}\, s/\sqrt n$을 만들고, 참 $\mu$가 그 안에 드는지 세면 된다.

    ```python
    rng = np.random.default_rng(0)
    mu, sigma, n, B = 5, 3, 20, 10_000
    tcrit = stats.t.ppf(0.975, n - 1)

    x = rng.normal(mu, sigma, size=(B, n))          # 행마다 하나의 표본
    xbar = x.mean(axis=1)
    s = x.std(axis=1, ddof=1)                        # ddof=1 이 표본표준편차
    half = tcrit * s / np.sqrt(n)
    covered = (xbar - half <= mu) & (mu <= xbar + half)
    print(f"포함비율 = {covered.mean():.4f}")
    ```

    **정확히 0.95가 나오지 않는 이유**는 포함비율 자체가 추정값이기 때문이다. 참 포함확률이 0.95일 때 $B = 10{,}000$번 반복에서 세어 본 비율의 표준오차는

    $$
    \sqrt{\frac{0.95 \times 0.05}{10{,}000}} \approx 0.00218
    $$

    이므로, 95% 정도의 반복에서 $0.9457$과 $0.9543$ 사이의 값이 나온다. $0.948$이나 $0.953$이 나왔다고 해서 이론이 틀린 것이 아니다.

    이 모의실험이 정말 쓸모 있는 경우는 가정을 깰 때다. 자료를 지수분포나 자유도 3인 $t$ 분포에서 뽑아 같은 $t$ 구간을 만들어 보면 포함비율이 0.95에서 눈에 띄게 벗어나며, $n$을 키우면 중심극한정리 덕분에 서서히 0.95로 돌아온다. "$n$이 얼마나 커야 충분한가"라는 물음에 수치로 답하는 표준적인 방법이다.

---

## 정리하며

난수 생성은 모의실험에 기반한 통계학 전체의 출발점이다.

- **`stats.norm.rvs(loc, scale, size)`** 가 $N(\mu,\sigma^2)$ 에서 표본을 뽑는다. 표본이 커질수록 히스토그램이 이론적 밀도로 다가가며, 이것이 큰수의 법칙을 눈으로 보는 가장 간단한 방법이다.
- **재현성을 위해 씨앗을 고정한다.** 요즘 권장되는 방식은 전역 상태를 건드리는 `np.random.seed()` 가 아니라 `rng = np.random.default_rng(0)` 로 생성기를 따로 만들어 넘기는 것이다.
- **표본은 분포가 아니다.** 유한한 표본의 히스토그램은 언제나 울퉁불퉁하며, 그 울퉁불퉁함 자체가 표집변동이다. 구간 폭을 바꾸면 모양이 달라진다는 점도 함께 기억할 일이다.
- 모의실험으로 확률을 추정할 때의 정밀도는 $1/\sqrt{B}$ 로만 좋아진다. 소수점 한 자리를 더 얻으려면 반복을 **100배** 늘려야 한다.

다음 절 **$t$ 분포**로 넘어간다. 정규분포에서 $\sigma$ 를 모른 채 $s$ 로 대신할 때 나타나는 분포이며, 그 대가가 두꺼운 꼬리로 나타난다.
