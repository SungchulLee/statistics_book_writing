# 변환 시연

## 개요

자료가 정규분포를 따르지 않을 때 수학적 변환을 적용하면 정규성에 더 가까운 분포를 얻어 표준적인 모수적 절차를 쓸 수 있는 경우가 많다. 이 페이지는 흔히 쓰는 세 변환 계열(로그, 제곱근, Box-Cox)을 시연하고, 각각이 언제 적절한지 설명하며, 시각적·형식적 확인으로 개선 정도를 평가하는 방법을 보인다.

---

## 1. 왜 변환하는가

많은 통계 방법($t$ 검정, 분산분석, 선형회귀 등)이 오차의 정규성을 가정한다. 자료가 오른쪽으로 치우쳐 있거나 분산이 일정하지 않을 때 적절한 변환은

1. 분포를 대칭으로 만들고,
2. 분산을 안정시키며,
3. 정규분포로의 근사를 개선한다.

---

## 2. 로그 변환

양수이면서 오른쪽으로 치우친 자료에는 로그 변환이 가장 먼저 시도할 도구이다. $X > 0$일 때

$$
Y = \ln X.
$$

$X \sim \text{Lognormal}(\mu, \sigma^2)$이면 $Y \sim \mathcal{N}(\mu, \sigma^2)$가 정확히 성립한다. 분포가 정확히 대수정규가 아니더라도 로그 변환은 왜도를 크게 줄이는 경우가 많다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 로그 변환 전후의 왜도. $X \sim \text{Lognormal}(0,\ 0.8^2)$에서 $n = 300$개를 뽑아 $Y = \ln X$를 만든다.

**(1)** $\text{Lognormal}(\mu, \sigma^2)$의 왜도를 **닫힌 꼴로 유도**하고 $\sigma = 0.8$에서의 값을 구하시오. 답이 $\mu$에 의존하는가. 로그를 씌운 뒤의 이론 왜도는 얼마인가.

**(2)** 변환 뒤의 표본왜도가 $G_1 = 0.2933$으로 나왔다. 이론값이 **정확히 0**인데도 이만큼 남았다. 정규성 아래의 $\mathrm{SE}(G_1)$을 구해 이 잔차가 우연으로 설명되는 크기인지 판정하시오. 같은 자료에 샤피로–윌크를 걸면 결론이 같은가.

</div>

??? success "풀이"

    **(1) 해석적으로.** $X = e^{\mu + \sigma Z}$, $Z \sim \mathcal{N}(0,1)$로 쓴다. 왜도는 위치와 척도에 불변이므로 $\mu$는 $e^\mu$라는 **곱하는 상수**로만 들어가고 답에서 떨어져 나간다. 따라서 $W = e^{\sigma Z}$만 보면 된다.

    적률생성함수를 쓰면 $\mathbb{E}[W^k] = \mathbb{E}[e^{k\sigma Z}] = e^{k^2\sigma^2/2}$이다. $w = e^{\sigma^2}$으로 줄여 쓰면

    $$
    \mathbb{E}[W] = w^{1/2}, \qquad \mathbb{E}[W^2] = w^{2}, \qquad \mathbb{E}[W^3] = w^{9/2}
    $$

    이다. 분산은

    $$
    \operatorname{Var}(W) = w^2 - w = w(w-1)
    $$

    이고, 3차 중심적률은

    $$
    \mathbb{E}[(W - \mathbb{E}W)^3] = \mathbb{E}[W^3] - 3\,\mathbb{E}[W]\,\mathbb{E}[W^2] + 2(\mathbb{E}W)^3
    = w^{9/2} - 3w^{5/2} + 2w^{3/2}
    = w^{3/2}\bigl(w^3 - 3w + 2\bigr)
    $$

    이다. 괄호를 인수분해하면 $w^3 - 3w + 2 = (w-1)(w^2 + w - 2) = (w-1)^2(w+2)$이므로

    $$
    \gamma_1 = \frac{w^{3/2}(w-1)^2(w+2)}{\bigl[w(w-1)\bigr]^{3/2}}
    = \frac{w^{3/2}(w-1)^2(w+2)}{w^{3/2}(w-1)^{3/2}}
    = (w+2)\sqrt{w-1}
    $$

    을 얻는다. **$\mu$는 들어오지 않는다.** $\sigma$ 하나가 로그정규의 모양을 다 정한다.

    $\sigma = 0.8$이면 $w = e^{0.64} = 1.896481$이므로

    $$
    \gamma_1 = (1.896481 + 2)\sqrt{0.896481} = 3.896481 \times 0.946826 = 3.6893
    $$

    이다. **로그를 씌운 뒤의 이론 왜도는 정확히 0이다.** 근사가 아니다. $\ln X = \mu + \sigma Z$가 그 자체로 $\mathcal{N}(\mu, \sigma^2)$이고, 정규분포의 왜도는 0이기 때문이다. 이것이 로그정규가 변환의 교과서적 예인 까닭이다. 변환이 "왜도를 줄인다"가 아니라 **없앤다.**

    **(2) 수치적으로.** 먼저 쪽의 코드를 그대로 돌린다.

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    # 로그정규는 로그를 씌우면 정확히 정규가 되는 분포다. 변환이 왜 듣는지를
    # 보이기에 가장 깨끗한 예다.
    rng = np.random.default_rng(42)
    x = rng.lognormal(mean=0.0, sigma=0.8, size=300)
    y = np.log(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=30, density=True, alpha=0.6, edgecolor="black")
    axes[0].set_title("Original (Lognormal)")
    axes[1].hist(y, bins=30, density=True, alpha=0.6, edgecolor="black")
    axes[1].set_title("After log transform")
    plt.tight_layout()
    plt.show()

    print(f"Before: skewness = {stats.skew(x, bias=False):.4f}")
    print(f"After:  skewness = {stats.skew(y, bias=False):.4f}")
    ```

    출력:

    ```text
    Before: skewness = 3.5432
    After:  skewness = 0.2933
    ```

    ![로그 변환 전후의 히스토그램](./img/transformations_code_27.png)

    이제 남은 $0.2933$을 재어 본다.

    ```python
    import numpy as np
    from scipy import stats

    sigma = 0.8
    w = np.exp(sigma ** 2)
    print(f"w = exp(sigma^2) = {w:.6f}")
    print(f"이론 왜도 (w+2)sqrt(w-1) = {(w + 2) * np.sqrt(w - 1):.4f}")

    rng = np.random.default_rng(42)
    x = rng.lognormal(mean=0.0, sigma=sigma, size=300)
    y = np.log(x)
    n = len(x)

    # bias=False 가 G1, bias=True(기본값) 가 g1 이다. 아래 SE 는 G1 의 것이다.
    G1 = stats.skew(y, bias=False)
    g1 = stats.skew(y)
    SE = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
    print(f"변환 뒤  G1 = {G1:.4f}   g1 = {g1:.4f}")
    print(f"정규 아래 SE(G1) = {SE:.4f}   G1/SE = {G1 / SE:.4f}")
    print(f"skewtest:  z = {stats.skewtest(y)[0]:.4f}  p = {stats.skewtest(y)[1]:.4f}")
    print(f"shapiro :  W = {stats.shapiro(y)[0]:.4f}  p = {stats.shapiro(y)[1]:.4f}")
    print(f"변환 뒤 표본평균 {y.mean():.4f}, 표본표준편차 {y.std(ddof=1):.4f}  (이론 0, {sigma})")
    ```

    출력:

    ```text
    w = exp(sigma^2) = 1.896481
    이론 왜도 (w+2)sqrt(w-1) = 3.6893
    변환 뒤  G1 = 0.2933   g1 = 0.2918
    정규 아래 SE(G1) = 0.1407   G1/SE = 2.0842
    skewtest:  z = 2.0725  p = 0.0382
    shapiro :  W = 0.9927  p = 0.1515
    변환 뒤 표본평균 -0.0329, 표본표준편차 0.7442  (이론 0, 0.8)
    ```

    **(1)의 확인.** 유도한 이론 왜도 $3.6893$에 대해 표본값은 $G_1 = 3.5432$다. 차이 $0.146$은 $4\%$ 아래이고, 왜도가 큰 분포에서 $G_1$의 변동이 크다는 점을 생각하면 잘 맞는 편이다. 변환 뒤 표본평균 $-0.0329$와 표본표준편차 $0.7442$도 이론값 $0$과 $0.8$을 둘러싸고 있다.

    **(2) 우연으로 설명되는 크기이지만, 5% 수준에서는 기각된다.** $n = 300$에서

    $$
    \mathrm{SE}(G_1) = \sqrt{\frac{6n(n-1)}{(n-2)(n+1)(n+3)}}
    = \sqrt{\frac{6 \times 300 \times 299}{298 \times 301 \times 303}}
    = \sqrt{\frac{538200}{27\,178\,494}} = 0.1407
    $$

    이므로 $G_1/\mathrm{SE} = 0.2933/0.1407 = 2.084$다. `scipy.stats.skewtest`는 여기에 다고스티노 변환을 한 번 더 거쳐 $z = 2.0725$, $p = 0.0382$를 준다(둘이 조금 다른 것은 단순 $z$ 대신 변환된 통계량을 쓰기 때문이다). **$p = 0.038 < 0.05$이니 대칭성이 기각된다.**

    !!! warning "어느 판본의 왜도인가"

        위 $\mathrm{SE}$ 식은 **보정판 $G_1$의 것**이다. 쪽의 코드가 `bias=False`를 주었으므로 $G_1 = 0.2933$과 짝이 맞는다. `scipy.stats.skew`의 **기본값은 `bias=True`**여서 $g_1 = 0.2918$을 주는데, 이 값에 위 $\mathrm{SE}$를 그대로 쓰면 안 된다. 어느 판본인지 밝히지 않으면 수가 맞지 않는다.

    여기서 멈추면 이상한 결론에 이른다. **변환은 정확히 옳았다.** (1)에서 보았듯 $\ln X$는 근사적으로가 아니라 **정확히** 정규다. 그런데도 그 정규 자료의 왜도 검정이 5% 수준에서 기각한다. 이것은 변환의 결함이 아니라 **$H_0$이 참일 때 20번에 한 번 일어나는 일**이고, 씨앗 42가 그 한 번에 걸린 것이다.

    같은 자료에 샤피로–윌크를 걸면 $W = 0.9927$, $p = 0.1515$로 **기각하지 못한다.** 두 검정의 결론이 갈린다. 샤피로–윌크는 분포 전체의 모양을 보므로 왜도 하나에 몰린 신호를 희석하고, 왜도검정은 그 하나만 겨냥하므로 증폭한다. 어느 쪽이 "맞는"가를 묻는 것은 잘못된 질문이다. 참인 답을 아는 이 인공 상황에서는 **둘 다 틀렸다.** 샤피로–윌크가 우연히 옳은 결론에 닿았을 뿐이다.

    그러므로 $3.54 \to 0.29$를 "사실상 대칭이 되었다"고 읽는 것은 **크기에 대해서는 옳다.** 왜도가 12배 줄었고 남은 $0.29$는 이론값 0에서 두 표준오차 거리에 있는 표집잡음이다. 그러나 "그러므로 검정을 통과한다"로 이어서는 안 된다. **옳은 변환도 그 표본의 검정을 통과하지 못할 수 있다.**

---

## 3. 제곱근 변환

계수 자료나 아래로 0에 의해 유계인 자료에는 제곱근 변환

$$
Y = \sqrt{X}
$$

가 로그보다 온건한 교정이다. 분산이 평균과 같은 포아송 계수 자료에 자주 쓰이며, 제곱근이 분산을 근사적으로 안정시킨다.

!!! warning "제곱근 변환은 과교정할 수 있다"
    포아송 자료에서 제곱근 변환은 평균 $\lambda$가 작으면 왜도를 **음수 쪽으로 지나치게** 밀어붙인다. $\lambda = 4$일 때 원자료의 왜도는 $1/\sqrt{4} = 0.5$이지만 $\sqrt{X}$의 왜도는 $-0.636$으로 절댓값이 오히려 커진다. 연습문제 3에서 자세히 다룬다.

---

## 4. Box-Cox 변환

Box-Cox 계열은 로그 변환과 거듭제곱 변환을 모수 $\lambda$ 하나로 일반화한다.

$$
Y^{(\lambda)} =
\begin{cases}
\dfrac{X^\lambda - 1}{\lambda}, & \lambda \neq 0, \\[6pt]
\ln X, & \lambda = 0.
\end{cases}
$$

최적 $\lambda$는 최대가능도로 고른다. SciPy의 `stats.boxcox`는 변환된 자료와 적합된 $\lambda$를 반환한다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> Box-Cox가 고르는 lambda. 보기 1과 **같은 표본**($\text{Lognormal}(0,\ 0.8^2)$에서 $n = 300$, 씨앗 42)에 `stats.boxcox`를 걸면 $\hat\lambda = -0.1258$이 나온다. 참값은 $\lambda = 0$이다.

**(1)** $\hat\lambda = -0.1258$이 참값 $0$을 "잘 회복했다"고 말할 수 있는지 **가능도비 신뢰구간으로 판정**하시오. $\lambda = 0$을 가능도비검정으로 직접 검정하면 어떻게 되는가.

**(2)** 변환 뒤 표본왜도가 $-0.0018$로, 참값 $\lambda = 0$을 썼을 때의 $0.2933$(보기 1)보다 **0에 가깝다.** 참값보다 추정값이 더 좋아 보이는 까닭을 설명하시오.

</div>

??? success "풀이"

    **(1) 회복했다고 말하기 어렵다. 95% 구간이 0을 간신히 놓친다.**

    박스–콕스의 $\lambda$에는 닫힌 꼴 표준오차가 없으므로 **가능도비**로 구간을 만든다. 프로파일 로그가능도 $\ell(\lambda)$에 대해

    $$
    \bigl\{\lambda : 2\bigl[\ell(\hat\lambda) - \ell(\lambda)\bigr] \le \chi^2_{1,\,0.95}\bigr\}
    = \bigl\{\lambda : 2\bigl[\ell(\hat\lambda) - \ell(\lambda)\bigr] \le 3.8415\bigr\}
    $$

    가 95% 구간이다. `stats.boxcox`에 `alpha=0.05`를 주면 이 구간을 바로 돌려준다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    x = rng.lognormal(mean=0.0, sigma=0.8, size=300)

    # alpha 를 주면 lambda 의 가능도비 신뢰구간을 함께 돌려준다.
    y_bc, lam, ci = stats.boxcox(x, alpha=0.05)
    print(f"lambda_hat = {lam:.4f}")
    print(f"95% 가능도비 신뢰구간 = [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"0 이 구간 안에 있는가: {ci[0] < 0 < ci[1]}")

    # 가능도비 검정으로 lambda = 0 을 직접 검정한다.
    ll_hat = stats.boxcox_llf(lam, x)
    ll_0 = stats.boxcox_llf(0.0, x)
    LR = 2 * (ll_hat - ll_0)
    print(f"loglik(lambda_hat) = {ll_hat:.4f},  loglik(0) = {ll_0:.4f}")
    print(f"LR = {LR:.4f}   임계값 chi2(1, 0.95) = {stats.chi2.ppf(0.95, 1):.4f}")

    # lambda 를 바꿔 가며 왜도와 샤피로 p 를 본다.
    print("  lambda      G1     shapiro p   로그가능도")
    for L in [-0.2, -0.1258, -0.1, 0.0, 0.1]:
        t = stats.boxcox(x, lmbda=L)
        print(f"  {L:+7.4f}  {stats.skew(t, bias=False):+7.4f}   "
              f"{stats.shapiro(t)[1]:8.4f}   {stats.boxcox_llf(L, x):9.4f}")

    # 표본왜도를 정확히 0 으로 만드는 lambda 를 따로 찾는다.
    from scipy.optimize import brentq
    lam_sym = brentq(lambda L: stats.skew(stats.boxcox(x, lmbda=L), bias=False),
                     -0.5, 0.3)
    print(f"표본왜도를 0 으로 만드는 lambda = {lam_sym:.4f}")
    ```

    출력:

    ```text
    lambda_hat = -0.1258
    95% 가능도비 신뢰구간 = [-0.2496, -0.0035]
    0 이 구간 안에 있는가: False
    loglik(lambda_hat) = 101.0194,  loglik(0) = 98.9864
    LR = 4.0661   임계값 chi2(1, 0.95) = 3.8415
      lambda      G1     shapiro p   로그가능도
      -0.2000  -0.1713     0.6482    100.3258
      -0.1258  -0.0019     0.9284    101.0194
      -0.1000  +0.0577     0.8974    100.9348
      +0.0000  +0.2933     0.1515     98.9864
      +0.1000  +0.5384     0.0009     94.4061
    표본왜도를 0 으로 만드는 lambda = -0.1250
    ```

    95% 구간은 $[-0.2496,\ -0.0035]$이고 **$0$은 그 안에 없다.** 위쪽 끝이 $-0.0035$이니 $0.0035$ 차이로 벗어난 것이다. 가능도비검정도 같은 말을 한다. $2[\ell(\hat\lambda) - \ell(0)] = 2(101.0194 - 98.9864) = 4.0661$이 임계값 $3.8415$를 넘으므로 **$\lambda = 0$이 5% 수준에서 기각된다.**

    참값이 정확히 0인데 기각되었다. 보기 1에서 본 것과 **똑같은 일**이며, 우연이 아니라 같은 원인이다. 그 표본의 로그변환 자료에 오른쪽 치우침 $G_1 = 0.2933$이 표집잡음으로 남아 있었고, 약간 음수인 $\lambda$가 오른쪽 꼬리를 한 번 더 눌러 그 잡음을 지운다. 두 보기는 같은 흔들림을 서로 다른 창으로 본 것이다. $H_0$이 참일 때 20번에 한 번 일어나는 일이고 씨앗 42가 그 한 번이다.

    그래서 "$-0.126$이 0을 잘 회복했다"는 말은 **크기에 대해서만 옳다.** $0$과 $-0.126$의 차이는 실무적으로 무의미하고, 쪽 끝의 관례대로 $\hat\lambda$를 0으로 반올림해 해석 가능한 로그변환을 쓰는 것이 옳은 선택이다. 그러나 **"95% 구간이 0을 담는다"는 뜻으로 읽으면 틀린다.** 담지 않는다.

    **(2) 추정값이 참값보다 좋아 보이는 것은 과적합이다.** $\hat\lambda$는 바로 **이 표본의** 정규가능도를 최대로 만드는 값이다. 참값 $\lambda = 0$은 *모집단*에 대해 옳지만, 손에 든 표본 하나에 대해 가장 잘 맞는 값은 아니다. 표를 보면 차이가 분명하다.

    | $\lambda$ | $G_1$ | 샤피로 $p$ | 로그가능도 |
    |---|---|---|---|
    | $-0.1258 = \hat\lambda$ | $-0.0019$ | $0.9284$ | $101.0194$ |
    | $0$ (참값) | $+0.2933$ | $0.1515$ | $98.9864$ |

    $\hat\lambda$ 쪽이 왜도는 0에 $150$배 가깝고 샤피로 $p$는 6배 크다. **모든 지표에서 참값을 이긴다.** 그러나 이긴 상대는 모집단이 아니라 표본이다. 새 표본 300개를 뽑아 이 $\hat\lambda = -0.1258$을 그대로 쓰면 평균적으로 $\lambda = 0$보다 나쁠 것인데, $-0.1258$은 지난 표본의 잡음에 맞춰진 값이기 때문이다.

    표가 하나 더 보여 주는 것이 있다. 왜도를 정확히 0으로 만드는 $\lambda$는 $-0.1250$으로 $\hat\lambda = -0.1258$과 거의 같다. 이 표본에서는 **최대가능도가 고른 $\lambda$와 표본왜도를 지우는 $\lambda$가 사실상 일치한다.** 다만 이것은 이 표본에서 관측된 일치일 뿐 일반 법칙이 아니다. 박스–콕스는 왜도만 보는 것이 아니라 분포 전체의 정규가능도를 보기 때문이다.

    **교훈.** $\hat\lambda$를 고르고 나서 같은 자료에 정규성 검정을 걸면 $p$값이 낙관적으로 나온다. $\lambda$를 추정하는 데 자료를 이미 썼으므로 그 검정은 더 이상 명목 크기를 지키지 않는다. 14.4절 릴리에포르 검정이 다루는 문제와 같은 구조다. **모수를 추정하고 나면 검정의 귀무분포가 달라진다.**

자료가 대수정규이므로 참 최적값은 $\lambda = 0$(로그 변환)이고, 최대가능도 추정값 $-0.126$은 표집변동 범위 안에서 이를 잘 회복한다. $\hat{\lambda} \approx 0$이면 Box-Cox는 로그로, $\hat{\lambda} \approx 0.5$이면 제곱근으로 환원된다.

---

## 5. 해석

변환한 뒤에는 반드시 시각적 방법(히스토그램, Q-Q 그림)과 형식적 검정(Shapiro-Wilk, Anderson-Darling)을 모두 써서 정규성을 다시 확인하라. 예컨대 치우침은 없앴지만 이봉성을 만들어 낸 변환은 상황을 개선한 것이 아니다. 또한 변환된 척도에서 수행한 추론은 원래 척도로 해석하려면 역변환해야 한다는 점을 기억하라.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $\text{Lognormal}(0, 0.6)$ 분포에서 관측값 $n = 400$개를 생성하라. 로그 변환을 적용하고 원자료와 변환 자료 모두에 Shapiro-Wilk 검정을 수행하라. $p$값을 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 0.6, size=400)

    _, p_orig = stats.shapiro(x)
    _, p_log = stats.shapiro(np.log(x))

    print(f"Original:        p = {p_orig:.4g}")
    print(f"Log-transformed: p = {p_log:.4g}")
    ```

    출력:

    ```text
    Original:        p = 2.989e-20
    Log-transformed: p = 0.3173
    ```

    원자료는 $p \approx 3 \times 10^{-20}$으로 압도적으로 기각된다. 로그 변환 후에는 $p = 0.317$로 정규성에 반하는 증거가 없다. $\ln X \sim \mathcal{N}(0, 0.36)$이 정확히 성립하므로 당연한 결과이다.

    다만 $p = 0.317$은 "정규성에 반하는 증거가 없다"는 뜻이지 "1에 가까우니 완벽히 정규"라는 뜻이 아니다. $H_0$이 참이면 $p$값은 $\text{Uniform}(0,1)$을 따르므로 0.317은 전형적인 값이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> $\text{Gamma}(2, 1)$ 분포에서 뽑은 관측값 $n = 300$개에 Box-Cox 변환을 적용하라. 최적 $\hat{\lambda}$와 변환 자료의 Shapiro-Wilk $p$값을 보고하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)
    x = rng.gamma(shape=2.0, scale=1.0, size=300)

    y_bc, lam = stats.boxcox(x)
    _, p_bc = stats.shapiro(y_bc)

    print(f"Optimal lambda: {lam:.4f}")
    print(f"Shapiro-Wilk p-value after Box-Cox: {p_bc:.4f}")
    ```

    출력:

    ```text
    Optimal lambda: 0.3637
    Shapiro-Wilk p-value after Box-Cox: 0.7376
    ```

    Gamma(2,1)은 중간 정도로 오른쪽으로 치우쳐 있다(이론적 왜도 $2/\sqrt{2} = 1.414$). 최적 $\hat{\lambda} = 0.364$는 세제곱근($1/3$)과 제곱근($1/2$) 사이에 있고, 변환 후 Shapiro-Wilk $p$값은 $0.74$로 0.05를 크게 넘는다. Box-Cox가 자료를 성공적으로 정규화했음을 뜻한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> Poisson($\lambda = 4$) 관측값 $n = 500$개를 생성하라. 제곱근 변환을 적용하고 변환 전후의 표본왜도를 비교하라. 히스토그램을 나란히 그려라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(2)
    x = rng.poisson(lam=4, size=500)
    y = np.sqrt(x)

    print(f"Original skewness: {stats.skew(x, bias=False):.4f}")
    print(f"Sqrt skewness:     {stats.skew(y, bias=False):.4f}")
    print(f"Var(sqrt(X)):      {y.var(ddof=1):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=range(15), density=True, alpha=0.6, edgecolor="black")
    axes[0].set_title("Poisson(4) — original")
    axes[1].hist(y, bins=20, density=True, alpha=0.6, edgecolor="black")
    axes[1].set_title("After sqrt transform")
    plt.tight_layout()
    plt.show()
    ```

    출력:

    ```text
    Original skewness: 0.4223
    Sqrt skewness:     -0.5880
    Var(sqrt(X)):      0.2823
    ```

    ![제곱근 변환 전후의 히스토그램](./img/transformations_code_211.png)

    **여기서 제곱근 변환은 실패한다.** 원자료의 왜도 $+0.42$(이론값 $1/\sqrt{4} = 0.5$)가 변환 후 $-0.59$가 되었다. 부호가 뒤집혔을 뿐 아니라 절댓값도 커졌다. 제곱근이 왼쪽 꼬리를 지나치게 압축한 **과교정**이다.

    이유는 $\lambda = 4$가 작아서 $X = 0, 1$ 근처에 무시할 수 없는 확률질량이 있고, 그 구간에서 $\sqrt{\cdot}$의 기울기가 급격히 변하기 때문이다. $\lambda$가 커지면 문제가 사라진다.

    | $\lambda$ | 원자료 왜도 $1/\sqrt{\lambda}$ | $\sqrt{X}$의 왜도 | $\operatorname{Var}(\sqrt{X})$ |
    |---|---|---|---|
    | 4 | 0.500 | $-0.636$ | 0.306 |
    | 9 | 0.333 | $-0.218$ | 0.263 |
    | 25 | 0.200 | $-0.107$ | 0.254 |
    | 100 | 0.100 | $-0.051$ | 0.251 |

    분산 안정화 성질 $\operatorname{Var}(\sqrt{X}) \approx 1/4$는 $\lambda$와 무관하게 잘 성립한다($\lambda = 4$에서 0.306, $\lambda = 100$에서 0.251). 곧 제곱근 변환은 **분산 안정화에는 성공하지만 작은 $\lambda$에서 대칭화에는 실패한다**. 두 목적을 혼동하지 말아야 한다.

    작은 $\lambda$에서 대칭성이 필요하다면 Anscombe 변환 $2\sqrt{X + 3/8}$($\lambda = 4$에서 왜도 $-0.251$)이나 Wilson-Hilferty 계열의 $X^{2/3}$(왜도 $-0.117$)이 더 낫다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> Box-Cox 변환이 $X > 0$을 요구하는 이유를 설명하라. 자료에 0이나 음수가 포함될 때 어떤 수정을 쓸 수 있는가?

</div>

??? success "풀이"

    Box-Cox 공식 $Y^{(\lambda)} = (X^\lambda - 1)/\lambda$는 $X$를 임의의 실수 거듭제곱 $\lambda$로 올린다. $X \leq 0$이면 (정수가 아닌 $\lambda$에 대해) $X^\lambda$가 정의되지 않거나 복소수가 된다.

    자료에 0이나 음수가 있을 때 흔한 수정은 **이동된** Box-Cox 변환이다. 모든 관측값에 대해 $X + c > 0$이 되도록 상수 $c > 0$을 골라 $X + c$에 Box-Cox를 적용한다. 대안으로 **Yeo-Johnson** 변환은 $X \geq 0$과 $X < 0$에 서로 다른 공식을 써서 음수 자료를 직접 다룰 수 있도록 Box-Cox를 확장한다. SciPy에서는 `stats.yeojohnson`으로 쓸 수 있다. $\square$

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> $\lambda \to 0$인 극한을 취하여 $\lambda = 0$인 Box-Cox 변환이 $Y = \ln X$로 환원됨을 증명하라.

</div>

??? success "풀이"

    $\lambda \neq 0$에 대해

    $$
    Y^{(\lambda)} = \frac{X^\lambda - 1}{\lambda} = \frac{e^{\lambda \ln X} - 1}{\lambda}.
    $$

    L'Hôpital 규칙을 적용한다(또는 $e^{\lambda \ln X} = 1 + \lambda \ln X + O(\lambda^2)$로 전개한다).

    $$
    \lim_{\lambda \to 0} \frac{e^{\lambda \ln X} - 1}{\lambda} = \lim_{\lambda \to 0} \frac{(\ln X)\, e^{\lambda \ln X}}{1} = \ln X.
    $$

    따라서 $Y^{(0)} = \ln X$이다. 이 연속성 덕분에 Box-Cox 계열이 $\lambda = 0$에서 매끄럽게 이어지고, 최대가능도로 $\lambda$를 최적화할 때 로그 변환이 자연스러운 극한으로 포함된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span> **역변환은 평균을 중앙값으로 바꾼다.** 쪽 끝 "되돌릴 때를 대비한다"에서 역변환을 하면 평균이 중앙값이 된다고 했다. 이것을 보기 1의 표본으로 직접 재어 본다. 보기 1($\text{Lognormal}(0,\ 0.8^2)$, $n = 300$, 씨앗 42)에서 로그 변환 뒤 표본평균은 $\bar y = -0.0329$, 표본표준편차는 $s_y = 0.7442$였다. 같은 표본의 원자료 표본평균은 $\bar x = 1.3038$이다. $t_{299,\,0.975} = 1.9679$를 쓰라.

**(1)** 로그 척도에서 $\mu$의 95% $t$ 구간을 구하고 양 끝에 $\exp$를 씌워라. 그렇게 얻은 구간은 **무엇에 대한** 95% 구간인가. 그 구간이 점추정값 $e^{\bar y}$를 중심으로 대칭이 아닌 까닭도 말하라.

**(2)** $e^{\bar y}$를 $\mathbb{E}[X]$의 추정값으로 쓰면 비율로 얼마나 아래로 치우치는가. $\sigma = 0.8$에서 그 비율을 구하고, 편의를 고친 추정값 $e^{\bar y + s_y^2/2}$를 계산해 $\bar x = 1.3038$과 참값 $\mathbb{E}[X]$와 나란히 놓아라.

</div>

??? success "풀이"

    **(1) $e^\mu$, 곧 $X$의 중앙값에 대한 구간이다. 평균에 대한 구간이 아니다.**

    $\mathrm{SE}(\bar y) = s_y/\sqrt{n} = 0.7442/\sqrt{300} = 0.7442/17.3205 = 0.042966$이므로 반폭은 $1.9679 \times 0.042966 = 0.084555$이고

    $$
    \mu \in \bigl(-0.0329 - 0.0846,\ -0.0329 + 0.0846\bigr) = (-0.1174,\ 0.0517)
    $$

    이다. $\exp$는 단조증가이므로 양 끝에 씌워도 포함확률이 보존된다. 따라서

    $$
    e^\mu \in \bigl(e^{-0.1174},\ e^{0.0517}\bigr) = (0.8892,\ 1.0530)
    $$

    이 95% 구간이다. 그런데 $\text{Lognormal}(\mu, \sigma^2)$의 중앙값은 $e^\mu$이고 평균은 $e^{\mu + \sigma^2/2}$다. 로그 척도에서 재던 $\mu$는 평균의 로그가 아니라 **중앙값의 로그**였으므로, 되돌린 구간도 중앙값을 겨냥한다. 참 중앙값 $e^0 = 1$이 $(0.8892,\ 1.0530)$ 안에 들어 있어 맞게 작동한다. 참 평균 $1.3771$은 구간 **밖**이며, 이것은 실패가 아니라 겨냥한 과녁이 다르다는 뜻이다.

    점추정값은 $e^{\bar y} = e^{-0.0329} = 0.9676$이고, 이것은 원자료의 **기하평균** $\bigl(\prod_i x_i\bigr)^{1/n}$과 같은 수다. 구간의 아래쪽 간격은 $0.9676 - 0.8892 = 0.0784$, 위쪽 간격은 $1.0530 - 0.9676 = 0.0854$로 대칭이 아니다. $\exp$가 **볼록**이라 로그 척도에서 같은 길이 $0.0846$이던 두 걸음이 위쪽에서 더 크게 늘어나기 때문이다. 이 비대칭은 흠이 아니라 이득이다. 구간이 $(0, \infty)$ 안에 머무는 것이 자동으로 보장되는데, 원래 척도에서 $\pm$ 로 만든 대칭 구간은 그것을 보장하지 못한다.

    **(2) 비율로 $e^{\sigma^2/2} = 1.3771$배, 곧 $27.4\%$ 낮다.**

    $e^{\bar y}$는 $n \to \infty$에서 $e^\mu$로 간다. 참 평균과의 비는

    $$
    \frac{\mathbb{E}[X]}{e^{\mu}} = \frac{e^{\mu + \sigma^2/2}}{e^{\mu}} = e^{\sigma^2/2} = e^{0.32} = 1.3771
    $$

    이다. 곧 $e^{\bar y}$는 평균의 $1/1.3771 = 0.7262$배를 겨냥하고 있어 $27.4\%$ 낮다. **이 치우침은 표본을 키워도 줄지 않는다.** $n$을 $300$에서 $3{,}000{,}000$으로 늘려도 $e^{\bar y}$는 $1.3771$이 아니라 $1$로 간다.

    고친 추정값은 $\sigma^2$ 자리에 $s_y^2$를 꽂는다. $s_y^2/2 = 0.7442^2/2 = 0.55386/2 = 0.27693$이므로

    $$
    e^{\bar y + s_y^2/2} = e^{-0.0329 + 0.27693} = e^{0.24403} = 1.2764
    $$

    이다. 네 수를 나란히 놓는다.

    | 값 | 수치 | 겨냥하는 모수 |
    |---|---|---|
    | $e^{\bar y}$ | $0.9676$ | 중앙값 $e^\mu = 1$ |
    | $e^{\bar y + s_y^2/2}$ | $1.2764$ | 평균 $e^{\mu + \sigma^2/2} = 1.3771$ |
    | $\bar x$ | $1.3038$ | 평균 $1.3771$ |
    | 참 평균 | $1.3771$ | — |

    교정판 $1.2764$가 참값 $1.3771$보다 아직 낮은 것은 **치우침이 남아서가 아니라 이 표본의 표집잡음** 때문이다. 지수 안을 보면 분명하다. $\bar y + s_y^2/2 = 0.24403$인데 참값은 $\mu + \sigma^2/2 = 0.32$다. 이 표본은 $\bar y = -0.0329 < 0$이고 $s_y = 0.7442 < 0.8$이어서 **두 입력이 모두 아래로 흔들렸고**, 둘 다 지수 안에 들어가므로 오차가 겹친다. 차이 $-0.076$이 $e^{-0.076} = 0.9268$배, 곧 $1.3771 \times 0.9268 = 1.2764$를 만든다. $\bar x = 1.3038$도 같은 방향이다. $\operatorname{Var}(X) = e^{0.64}(e^{0.64}-1) = 1.8965 \times 0.8965 = 1.7002$이므로 $\mathrm{SE}(\bar x) = \sqrt{1.7002/300} = 0.0753$이고, $1.3038$은 참값에서 약 $1$ 표준오차 아래다.

    **두 오차의 종류가 다르다는 것이 요점이다.** $e^{\bar y}$의 $27.4\%$는 $n$과 무관한 **체계적** 어긋남이고, $1.2764$ 대 $1.3771$의 $7\%$는 $n$이 커지면 사라지는 **표집** 오차다. 변환한 척도에서 추론하고 원래 척도로 보고할 때 밝혀야 하는 것이 바로 앞쪽이다. "로그를 되돌렸으니 평균의 추정값"이라고 쓰면 중앙값을 평균이라 부르는 것이 된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span> **거듭제곱 변환은 포아송 왜도를 $(3q-2)$배 한다.** 연습문제 3은 $\text{Poisson}(\lambda)$에 $\sqrt{\cdot}$를 걸면 왜도가 $+1/\sqrt{\lambda}$에서 음수로 **지나쳐** 간다는 것을 수치로 보였고, 풀이 끝에서 $X^{2/3}$이 더 낫다고 했다. 두 사실을 하나의 유도로 묶어라.

$q > 0$에 대해 $Y = X^q$라 하고 $D = X - \lambda$로 두어 $Y$를 $\lambda$ 근방에서 **2차까지** 전개한다. 포아송의 중심적률은 $\mu_2 = \lambda$, $\mu_3 = \lambda$, $\mu_4 = \lambda + 3\lambda^2$이다.

**(1)** $\lambda \to \infty$의 최저차항에서

$$
\gamma_1(X^q) = \frac{3q - 2}{\sqrt{\lambda}} + O(\lambda^{-3/2})
$$

임을 보여라.

**(2)** $q = 1,\ \tfrac12,\ \tfrac23$을 넣어 세 결과를 읽어라. 연습문제 3의 표에 있는 정확한 값과 비교해 근사가 어디서 맞고 어디서 깨지는지 말하라. 특히 $\lambda = 4$에서 $\sqrt{X}$의 왜도가 $-0.636$으로 **절댓값이 오히려 커진** 현상이 이 근사로 설명되는가.

</div>

??? success "풀이"

    **(1)** $g(x) = x^q$라 하면 $g'(\lambda) = q\lambda^{q-1}$, $g''(\lambda) = q(q-1)\lambda^{q-2}$이다. 2차 전개는

    $$
    Y = g(\lambda + D) \approx g(\lambda) + g'(\lambda) D + \tfrac12 g''(\lambda) D^2
    $$

    이고, $\mathbb{E}[D] = 0$, $\mathbb{E}[D^2] = \lambda$이므로 $\mathbb{E}[Y] \approx g(\lambda) + \tfrac12 g'' \lambda$, 따라서

    $$
    Y - \mathbb{E}[Y] \approx g' D + \tfrac12 g''\bigl(D^2 - \lambda\bigr)
    $$

    이다. 이제 $\mu_4 - \lambda^2 = (\lambda + 3\lambda^2) - \lambda^2 = \lambda + 2\lambda^2$를 기억해 둔다.

    **분산.** 제곱해서 기댓값을 취하면

    $$
    \operatorname{Var}(Y) \approx g'^2 \lambda + g' g'' \mu_3 + \tfrac14 g''^2 \bigl(\mu_4 - \lambda^2\bigr)
    $$

    이다. 차수를 세면 첫 항은 $q^2 \lambda^{2q-1}$, 둘째 항은 $q^2(q-1)\lambda^{2q-2}$, 셋째 항은 $O(\lambda^{2q-2})$다. 곧

    $$
    \operatorname{Var}(Y) = q^2 \lambda^{2q-1}\bigl(1 + O(\lambda^{-1})\bigr)
    $$

    이고 $\bigl[\operatorname{Var}(Y)\bigr]^{3/2} = q^3 \lambda^{3q - 3/2}\bigl(1 + O(\lambda^{-1})\bigr)$이다.

    **3차 중심적률.** 세제곱해서 살아남는 최저차 항이 둘이다.

    $$
    \mathbb{E}\bigl[(Y - \mathbb{E}Y)^3\bigr] \approx g'^3 \mu_3 + 3 g'^2 \cdot \tfrac12 g'' \,\mathbb{E}\bigl[D^2(D^2 - \lambda)\bigr] = g'^3 \mu_3 + \tfrac32 g'^2 g'' \bigl(\mu_4 - \lambda^2\bigr)
    $$

    첫 항은 $q^3\lambda^{3q-3} \cdot \lambda = q^3 \lambda^{3q-2}$다. 둘째 항은

    $$
    \tfrac32 \cdot q^2\lambda^{2q-2} \cdot q(q-1)\lambda^{q-2} \cdot \bigl(\lambda + 2\lambda^2\bigr)
    = 3q^3(q-1)\lambda^{3q-2} + O(\lambda^{3q-3})
    $$

    이다. 둘을 더하면

    $$
    \mathbb{E}\bigl[(Y - \mathbb{E}Y)^3\bigr] = q^3 \lambda^{3q-2}\bigl[1 + 3(q-1)\bigr] + O(\lambda^{3q-3})
    = q^3 (3q-2)\,\lambda^{3q-2} + O(\lambda^{3q-3})
    $$

    이다. 전개에서 떨어뜨린 나머지 항들($g'g''^2$, $g''^3$, 그리고 3차 미분 $g'''$이 들어오는 항)은 모두 $O(\lambda^{3q-3})$이므로 최저차에 끼어들지 않는다. 나누면

    $$
    \gamma_1(X^q) = \frac{q^3(3q-2)\lambda^{3q-2}}{q^3 \lambda^{3q - 3/2}}\bigl(1 + O(\lambda^{-1})\bigr) = \frac{3q-2}{\sqrt{\lambda}} + O(\lambda^{-3/2})
    $$

    을 얻는다. $q$가 $q^3$으로 들어왔다가 그대로 약분되어 **$3q-2$라는 일차식 하나만 남는다.** $\square$

    **(2)** 세 값을 넣는다.

    - $q = 1$: $\gamma_1 = 1/\sqrt{\lambda}$. 포아송의 **정확한** 왜도와 같다. 전개가 올바르게 눈금을 맞추고 있다는 확인이다.
    - $q = 1/2$: $3q-2 = -1/2$이므로 $\gamma_1(\sqrt{X}) \approx -\dfrac{1}{2\sqrt{\lambda}}$. **부호가 뒤집히고 크기는 절반이 된다.** 제곱근 변환은 왜도를 0으로 데려가는 것이 아니라 0을 지나 반대쪽으로 **원래의 절반만큼** 보낸다. 연습문제 3이 본 과교정의 정체가 이것이다.
    - $q = 2/3$: $3q - 2 = 0$. 최저차항이 **사라진다.** 윌슨–힐퍼티의 $X^{2/3}$이 왜 나오는지가 여기서 설명된다. $2/3$은 세제곱근과 제곱근 사이의 임의의 값이 아니라 **$3q = 2$를 푼 유일한 지수**다.

    정확한 값과 나란히 놓는다(연습문제 3의 표를 넓힌 것이다).

    | $\lambda$ | $X$ ($q=1$) | $\sqrt{X}$ 정확 | 근사 $-1/(2\sqrt{\lambda})$ | $X^{2/3}$ 정확 |
    |---|---|---|---|---|
    | $4$ | $+0.5000$ | $-0.6357$ | $-0.2500$ | $-0.1169$ |
    | $9$ | $+0.3333$ | $-0.2179$ | $-0.1667$ | $-0.0182$ |
    | $25$ | $+0.2000$ | $-0.1070$ | $-0.1000$ | $-0.0028$ |
    | $100$ | $+0.1000$ | $-0.0508$ | $-0.0500$ | $-0.0003$ |

    $\lambda = 100$에서 $-0.0508$ 대 $-0.0500$으로 $1.6\%$ 차이, $\lambda = 25$에서 $7\%$, $\lambda = 9$에서 $31\%$다. 점근식은 $\lambda$가 커질수록 빠르게 맞아 들어간다. $X^{2/3}$ 열도 예측대로다. 최저차항이 0이고 $O(\lambda^{-3/2})$만 남으므로 $\sqrt{X}$보다 **훨씬 빨리** 0으로 간다($\lambda = 100$에서 $-0.0003$).

    **$\lambda = 4$에서는 근사가 깨지고, 그것이 중요하다.** 근사는 $-0.2500$이라고 말한다. 이 값은 원자료의 $+0.5$보다 절댓값이 **작으므로**, 점근식만 믿으면 "제곱근이 과교정은 하지만 그래도 개선은 한다"고 읽게 된다. 실제 값은 $-0.6357$로 절댓값이 **커졌다.** 곧 연습문제 3이 본 "실패"는 이 전개가 보지 못하는 현상이다.

    까닭은 전개의 전제가 무너졌기 때문이다. 전개는 $D = X - \lambda$가 $\lambda$에 비해 작아서 $g$를 2차 다항식으로 바꿔 놓아도 된다고 가정한다. $\lambda = 4$에서는 $\mathrm{SD}(X) = 2$가 평균 $4$의 절반이고, $P(X \le 1) = e^{-4}(1 + 4) = 5e^{-4} = 0.0916$만큼의 질량이 $\sqrt{\cdot}$의 기울기 $1/(2\sqrt{x})$가 치솟는 구간에 놓인다. $x = 0$ 근처에서 기울기가 발산하는 것은 어떤 유한 차수 다항식도 흉내 낼 수 없는 모양이다. **점근식은 왜도를 반대쪽으로 넘긴다는 방향을 옳게 알려 주지만, 작은 $\lambda$에서 얼마나 넘기는지는 알려 주지 못한다.** 연습문제 3이 $\lambda$별 표를 수치로 들고 있어야 했던 이유다. $\square$

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> **추정한 $\lambda$로 변환한 뒤의 섀피로–윌크는 실제로 몇 %에서 기각하는가.** 보기 2의 교훈은 "$\lambda$를 추정하는 데 자료를 이미 썼으므로 그 검정은 더 이상 명목 크기를 지키지 않는다"였다. 말로 두지 말고 **재어라.**

$\text{Lognormal}(0,\ 0.8^2)$에서 $n = 300$을 뽑는다. 이 자료는 $\lambda = 0$에서 **정확히** 정규가 되므로 귀무가설이 참이고, 기각률이 그대로 검정의 실제 크기다. 두 절차를 $\alpha = 0.01,\ 0.05,\ 0.10$에서 비교한다.

- (가) 참값 $\lambda = 0$을 꽂고 $\ln x$에 섀피로–윌크.
- (나) 같은 자료로 $\hat\lambda$를 추정하고 `stats.boxcox(x)[0]`에 섀피로–윌크.

**(1)** (나)의 기각률이 명목 수준보다 **클 것 같은가 작을 것 같은가.** 먼저 답을 적어 두고 $R = 10000$회 모의실험으로 확인하라.

**(2)** (나)를 쓰면서 크기 $0.05$를 지키려면 임계 $p$값을 얼마로 바꿔야 하는가. 그 값을 모의실험에서 읽고, 보기 2의 $p = 0.9284$를 그 기준으로 다시 판정하라.

</div>

??? success "풀이"

    **(1) 작게 나온다.** $\hat\lambda$는 바로 이 표본의 정규가능도를 최대로 만드는 값이므로, 변환된 자료는 **참값 $\lambda = 0$을 썼을 때보다 더 정규처럼 보인다.** 보기 2가 이미 그 한 표본을 보여 주었다($\lambda = 0$에서 $p = 0.1515$, $\hat\lambda$에서 $p = 0.9284$). $W$가 커지는 쪽으로 밀리면 $p$가 커지고 기각이 줄어든다. 곧 검정이 **보수적**이 된다. 14.4절 릴리에포르와 같은 구조다. 모수를 추정하면 귀무분포가 달라지는데, 추정하지 않았을 때의 임계값을 그대로 쓰면 수준이 어긋난다.

    ```python
    import numpy as np
    from scipy import stats

    # 자료가 정확히 로그정규이므로 lambda = 0 이 참값이고 ln x 는 정확히 정규다.
    # 곧 귀무가설이 참인 상황이며, 기각률은 검정의 실제 크기를 재는 것이다.
    rng = np.random.default_rng(0)
    R, n = 10000, 300
    p_true = np.empty(R)   # (가) 참값 lambda = 0 을 꽂는다
    p_hat = np.empty(R)    # (나) 같은 자료로 lambda 를 추정한다
    for r in range(R):
        x = rng.lognormal(mean=0.0, sigma=0.8, size=n)
        p_true[r] = stats.shapiro(np.log(x))[1]
        p_hat[r] = stats.shapiro(stats.boxcox(x)[0])[1]

    print(f"R = {R}, n = {n}")
    print("alpha   (가) lambda = 0   (나) lambda = hat")
    for a in [0.01, 0.05, 0.10]:
        print(f"{a:5.2f}   {(p_true < a).mean():13.4f}   {(p_hat < a).mean():15.4f}")
    print(f"p 값의 중앙값:   (가) {np.median(p_true):.4f}   (나) {np.median(p_hat):.4f}")
    q05 = np.quantile(p_hat, 0.05)
    print(f"(나) 의 p 분포 5% 분위수 = {q05:.4f}   그 임계값으로 센 기각률 = {(p_hat < q05).mean():.4f}")
    print(f"(나) 에서 p >= 0.9284 일 확률 = {(p_hat >= 0.9284).mean():.4f}")
    ```

    출력:

    ```text
    R = 10000, n = 300
    alpha   (가) lambda = 0   (나) lambda = hat
     0.01          0.0105            0.0008
     0.05          0.0515            0.0089
     0.10          0.0996            0.0254
    p 값의 중앙값:   (가) 0.5032   (나) 0.7056
    (나) 의 p 분포 5% 분위수 = 0.1602   그 임계값으로 센 기각률 = 0.0500
    (나) 에서 p >= 0.9284 일 확률 = 0.1533
    ```

    **(가)는 눈금이 맞는다.** $0.0105$, $0.0515$, $0.0996$이 각각 $0.01$, $0.05$, $0.10$과 모의오차 안에서 같다. $R = 10000$에서 기각률의 표준오차는 $\sqrt{0.05 \times 0.95/10000} = 0.0022$이므로 $0.0515$는 명목값에서 $0.7$ 표준오차 거리다. 이것은 두 가지를 동시에 확인해 준다. 섀피로–윌크의 $p$값이 정규자료에서 제대로 균등하고(중앙값 $0.5032$는 $\text{Uniform}(0,1)$의 중앙값 $0.5$), 로그정규에 로그를 씌운 자료가 정말로 정규라는 것이다.

    **(나)는 눈금이 어긋난다.** $\alpha = 0.05$에서 실제 크기가 $0.0089$다. 명목값의 **$18\%$**이고 $0.05$에서 약 $19$ 표준오차 떨어져 있으니 모의오차로 설명되지 않는다. $\alpha = 0.01$에서는 $0.0008$로 명목값의 $8\%$까지 내려간다. $p$값의 중앙값도 $0.5032$에서 $0.7056$으로 밀려 올라가 **더 이상 균등분포가 아니다.** 예측대로 보수적이다.

    여기서 "보수적이면 안전한 쪽이니 괜찮다"고 넘기면 안 된다. 크기를 $0.05$에서 $0.0089$로 잃는다는 것은 같은 만큼의 **검정력을 버린다**는 뜻이다. 자료가 어떤 $\lambda$로도 정규가 되지 않는 경우(연습문제 10처럼 이봉인 경우)를 이 절차는 둔하게 본다.

    **(2) 임계값을 $p < 0.05$에서 $p < 0.1602$로 올려야 한다.**

    $\hat\lambda$에 닫힌 꼴 귀무분포가 없으므로 모의실험이 준 분포를 직접 쓴다. (나)의 $p$값 분포의 $5\%$ 분위수가 $0.1602$이고, 그 값을 임계값으로 다시 세면 기각률이 정확히 $0.0500$이 된다(구성상 그렇게 되도록 고른 값이다). 모수적 붓스트랩으로 임계값을 되맞춘 것이며, 릴리에포르 표가 KS에 대해 해 주는 일과 같다.

    이 기준으로 보기 2를 다시 본다. $p = 0.9284$는 $0.1602$보다 훨씬 크므로 **기각하지 못한다는 결론은 그대로다.** 그러나 그 수를 읽는 느낌은 달라져야 한다. 이 절차 아래에서 $p \ge 0.9284$가 나올 확률이 $0.1533$이다. 곧 $0.9284$는 귀무가설이 참일 때 **일곱 번에 한 번쯤 나오는 평범한 값**이고, "$p$가 $0.93$이니 아주 정규답다"는 읽기는 과장이다. 보기 2의 표에서 $\hat\lambda$ 쪽 샤피로 $p$가 참값 쪽의 6배였던 것도 같은 착시다. **추정한 $\lambda$로 변환한 자료의 $p$값은 $0.05$가 아니라 $0.16$을 기준으로 읽어야 한다.** $\square$

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> **박스–콕스 가능도의 야코비안 항은 왜 있어야 하는가.** 보기 2의 코드가 쓴 `stats.boxcox_llf`는 무엇을 계산하는가. $Y^{(\lambda)}(X_i) \sim \mathcal{N}(\mu, \sigma^2)$라는 모형에서 $X$의 밀도에는 야코비안 $\lvert dy/dx \rvert = x^{\lambda-1}$이 곱해지고, $\mu$와 $\sigma^2$를 프로파일로 없애면 상수를 빼고

$$
\ell(\lambda) = -\frac{n}{2}\ln \hat\sigma^2_\lambda + (\lambda - 1)\sum_{i=1}^{n} \ln x_i, \qquad \hat\sigma^2_\lambda = \frac{1}{n}\sum_{i=1}^{n}\Bigl(y^{(\lambda)}_i - \overline{y^{(\lambda)}}\Bigr)^2
$$

가 남는다. 뒤의 항이 야코비안에서 온 것이다.

**(1)** 자료를 $x_i \mapsto c x_i$ ($c > 0$)로 바꾸면 $\ell(\lambda)$가 $\lambda$에 **의존하지 않는** 상수 $-n\ln c$만큼만 움직임을 보여라. 따라서 $\hat\lambda$는 측정 단위에 무관하다. (힌트: $Y^{(\lambda)}(cx) = c^\lambda Y^{(\lambda)}(x) + \dfrac{c^\lambda - 1}{\lambda}$.)

**(2)** 야코비안 항을 빼고 $Q(\lambda) = -\tfrac{n}{2}\ln\hat\sigma^2_\lambda$만 최대화한다고 하자. 곧 "변환 뒤 분산이 가장 작아지는 $\lambda$"를 고르는 것이다. 모든 $x_i > 1$이고 적어도 두 값이 서로 다를 때, $\lambda \to -\infty$에서 $Q(\lambda) \to +\infty$인 반면 $\ell(\lambda) \to -\infty$임을 보여라. 야코비안 항이 하는 일을 한 줄로 말하라.

</div>

??? success "풀이"

    **(1)** $\lambda \neq 0$에서 힌트의 식을 확인한다.

    $$
    Y^{(\lambda)}(cx) = \frac{(cx)^\lambda - 1}{\lambda} = \frac{c^\lambda x^\lambda - c^\lambda + c^\lambda - 1}{\lambda} = c^\lambda \cdot \frac{x^\lambda - 1}{\lambda} + \frac{c^\lambda - 1}{\lambda}
    $$

    곧 $cx$를 변환한 값은 $x$를 변환한 값의 **아핀 상**이고, 기울기가 $c^\lambda$, 절편이 $(c^\lambda-1)/\lambda$다. 표본분산은 절편에 영향받지 않고 기울기의 제곱만큼 커지므로

    $$
    \hat\sigma^2_\lambda(cx) = c^{2\lambda}\,\hat\sigma^2_\lambda(x)
    $$

    이다. 또 $\sum \ln (c x_i) = n \ln c + \sum \ln x_i$다. 둘을 넣으면

    $$
    \ell_{cx}(\lambda) = -\frac{n}{2}\Bigl[2\lambda \ln c + \ln\hat\sigma^2_\lambda(x)\Bigr] + (\lambda-1)\Bigl[n\ln c + \sum \ln x_i\Bigr]
    $$

    이고, 정리하면

    $$
    \ell_{cx}(\lambda) = \ell_x(\lambda) \underbrace{- n\lambda \ln c + n\lambda \ln c}_{=\,0} - n\ln c = \ell_x(\lambda) - n\ln c
    $$

    가 된다. **$\lambda$가 붙은 두 항이 정확히 맞비긴다.** 남은 $-n\ln c$는 $\lambda$와 무관한 상수이므로 최대점을 옮기지 못한다. 따라서 $\hat\lambda(cx) = \hat\lambda(x)$다.

    $\lambda = 0$에서도 따로 확인된다. $\ln(cx_i) = \ln c + \ln x_i$이니 분산은 그대로이고 야코비안 항은 $(0-1)\sum\ln(cx_i) = -\sum \ln x_i - n\ln c$가 되어, 역시 $-n\ln c$만 더해진다. 두 경우가 이어 붙는다.

    이 성질은 모형 선택 기준에 당연히 요구해야 하는 것이다. 보기 2의 $\hat\lambda = -0.1258$은 자료를 미터로 재든 센티미터로 재든 같은 값이어야 하고, 실제로 그렇다.

    **(2)** $t = -\lambda > 0$으로 두면

    $$
    y_i^{(-t)} = \frac{x_i^{-t} - 1}{-t} = \frac{1 - e^{-t \ln x_i}}{t}
    $$

    이고, 이것은 $x_i$에 대해 증가한다. $m = \min_i \ln x_i$, $M = \max_i \ln x_i$라 하자. 가정에서 $0 < m < M$이다. 표본의 범위는

    $$
    R_t = y^{(-t)}_{\max} - y^{(-t)}_{\min} = \frac{e^{-tm} - e^{-tM}}{t} = \frac{e^{-tm}}{t}\Bigl(1 - e^{-t(M-m)}\Bigr)
    $$

    이다. 범위가 $R$인 자료의 표본분산은 아래위로 갇힌다. 위로는 $\hat\sigma^2 \le R^2/4$이고(길이 $R$ 구간에 놓인 분포의 분산은 $R^2/4$ 이하다), 아래로는 양 끝값 가운데 적어도 하나가 평균에서 $R/2$ 이상 떨어져 있으므로 $\hat\sigma^2 \ge R^2/(4n)$이다. 로그를 씌우면

    $$
    -\frac{n}{2}\ln\hat\sigma^2_{-t} = -n \ln R_t + O(1) = n t m + n \ln t + O(1)
    $$

    을 얻는다($O(1)$은 $t$에 무관한 유계항이다. $1 - e^{-t(M-m)} \to 1$이므로 그 로그도 유계다). 그러므로

    $$
    Q(-t) = ntm + n\ln t + O(1) \longrightarrow +\infty \qquad (t \to \infty)
    $$

    이다. 야코비안 없는 기준에는 **최대점이 아예 없다.** $\lambda$를 음수로 멀리 보낼수록 좋아지기만 한다. 실제로 일어나는 일은 간단하다. $t$가 크면 모든 $x_i^{-t}$가 0으로 눌리고 $y_i \approx 1/t$가 되어 **자료 전체가 한 점으로 뭉개진다.** 분산이 0으로 가니 "변환 뒤 분산이 작다"는 기준은 만족하지만, 그런 자료가 정규라고 말하는 것은 아무 내용이 없다.

    야코비안 항을 되돌려 놓으면

    $$
    \ell(-t) = Q(-t) + (-t-1)\sum_i \ln x_i = ntm + n\ln t - (t+1)\,n\,\overline{\ln x} + O(1)
    $$

    $$
    = n t\bigl(m - \overline{\ln x}\bigr) + n\ln t + O(1)
    $$

    이다. 적어도 두 $x_i$가 다르므로 $m < \overline{\ln x}$이고, 따라서 $m - \overline{\ln x} < 0$이다. $t$에 **일차로** 음수인 항이 $n\ln t$를 압도하므로

    $$
    \ell(-t) \longrightarrow -\infty \qquad (t \to \infty)
    $$

    가 된다. 야코비안 항이 발산을 정확히 눌러서 가능도에 제대로 된 최대점을 돌려준다.

    **한 줄로.** 야코비안은 $\lambda$를 줄여 척도를 압축해 얻은 "작은 분산"에 압축한 만큼의 값을 물려서, 가능도가 변환된 척도가 아니라 **원래 척도의 밀도**를 비교하게 만든다. (1)과 (2)는 같은 사실의 두 얼굴이다. $Q$를 보면 단위를 바꿀 때 $Q_{cx}(\lambda) = Q_x(\lambda) - n\lambda \ln c$로 $\lambda$에 **기울어진** 항이 남아 최대점이 단위에 따라 움직이는데, 야코비안 항이 바로 그 기울기를 지우는 항이다. 단위에 무관해지는 것과 발산이 막히는 것이 같은 항에서 나온다. $\square$

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> **왜도와 첨도가 둘 다 0인 이봉분포 만들기.** 5절은 "치우침은 없앴지만 이봉성을 만들어 낸 변환은 상황을 개선한 것이 아니다"라고 경고한다. 어떤 변환을 걸고 나서 표본왜도와 초과첨도만 재어 둘 다 0임을 확인했다면 안심해도 되는가. **반례를 만들어라.**

세 정규밀도의 혼합

$$
f(x) = \frac{1-p}{2}\,\phi_{a,\,s}(x) + \frac{1-p}{2}\,\phi_{-a,\,s}(x) + p\,\phi_{0,\,\tau}(x)
$$

를 쓴다. 여기서 $\phi_{m,v}$는 $\mathcal{N}(m, v^2)$의 밀도다. $p = 0.05$, $a = 1$, $s = 0.4$로 둔다.

**(1)** 이 분포의 왜도가 모든 $\tau$에서 정확히 0인 까닭을 말하고, 초과첨도까지 **정확히 0**으로 만드는 $\tau$를 구하라. ($\mu_2$와 $\mu_4$를 성분별로 모아 $\mu_4 = 3\mu_2^2$을 풀면 된다.)

**(2)** 그 $\tau$로 $n = 2000$을 뽑아 왜도검정·첨도검정·자크–베라와 섀피로–윌크·앤더슨–달링을 모두 걸어라. 어느 검정이 무엇을 보고 무엇을 못 보는지 설명하라. 2장의 부등식 $\beta_2 \ge 1 + \gamma_1^2$는 이 분포에 대해 무엇을 말해 주는가.

</div>

??? success "풀이"

    **(1) $\tau = 2.1935$.**

    **왜도.** $f(-x) = f(x)$다. 세 성분 가운데 두 개가 $\pm a$에 같은 무게로 놓이고 나머지 하나가 0을 중심으로 대칭이기 때문이다. 평균은 0이고, 대칭밀도의 **모든 홀수 중심적률이 0**이므로 $\mu_3 = 0$, 따라서 $\gamma_1 = 0$이 $\tau$와 무관하게 정확히 성립한다. $\tau$를 아무렇게 골라도 왜도검정은 할 말이 없다.

    **초과첨도.** $\mathcal{N}(m, v^2)$에 대해 $\mathbb{E}[X^2] = m^2 + v^2$, $\mathbb{E}[X^4] = m^4 + 6m^2v^2 + 3v^4$다. 평균이 0이므로 중심적률이 곧 적률이고, 성분별로 모으면 $u = \tau^2$으로 두어

    $$
    \mu_2 = (1-p)\bigl(a^2 + s^2\bigr) + p\,u = A + p\,u, \qquad A = 0.95 \times 1.16 = 1.10200
    $$

    $$
    \mu_4 = (1-p)\bigl(a^4 + 6a^2s^2 + 3s^4\bigr) + 3p\,u^2 = C + 3p\,u^2, \qquad C = 0.95 \times 2.0368 = 1.93496
    $$

    이다($a^4 + 6a^2s^2 + 3s^4 = 1 + 0.96 + 0.0768 = 2.0368$). 초과첨도 0은 $\mu_4 = 3\mu_2^2$이므로

    $$
    1.93496 + 0.15\,u^2 = 3\bigl(1.10200 + 0.05\,u\bigr)^2 = 3.643212 + 0.3306\,u + 0.0075\,u^2
    $$

    $$
    0.1425\,u^2 - 0.3306\,u - 1.708252 = 0
    $$

    이다. 판별식은 $0.3306^2 + 4 \times 0.1425 \times 1.708252 = 0.109296 + 0.973704 = 1.083000$이고 $\sqrt{1.083} = 1.040673$이므로 양의 근은

    $$
    u = \frac{0.3306 + 1.040673}{2 \times 0.1425} = \frac{1.371273}{0.285} = 4.8115, \qquad \tau = \sqrt{4.8115} = 2.1935
    $$

    다. 확인하면 $\mu_2 = 1.10200 + 0.05 \times 4.8115 = 1.3426$, $\mu_4 = 1.93496 + 0.15 \times 4.8115^2 = 5.4075$이고 $3\mu_2^2 = 5.4075$로 맞는다.

    **어떻게 가능한가.** $\pm 1$에 모인 두 좁은 봉우리만 있으면 분포가 평평해져 초과첨도가 음수가 된다(등가중 두 정규 혼합은 $-2a^4/(s^2+a^2)^2$로 **항상** 음수다). 거기에 넓게 퍼진 $\mathcal{N}(0, 2.1935^2)$를 $5\%$만 섞어 꼬리를 늘이면 첨도가 끌어올려진다. 두 효과가 정확히 맞비기는 $\tau$를 고른 것이다. 섞은 무게가 $5\%$뿐이니 가운데 골은 메워지지 않는다. 밀도를 재어 보면 $f(0) = 0.0507$, $f(1) = 0.4819$로 골이 봉우리의 $10.5\%$ 높이에 지나지 않아 **봉이 분명히 둘**이다.

    **(2)**

    ```python
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    # 좌우 대칭이라 왜도는 어떤 tau 에서도 0 이다. 초과첨도까지 0 으로 맞추는 tau 를 푼다.
    # mu2 = A + p*u,  mu4 = C + 3p*u^2  (u = tau^2) 를 mu4 = 3*mu2^2 에 넣는다.
    p, a, s = 0.05, 1.0, 0.4
    A = (1 - p) * (a**2 + s**2)
    C = (1 - p) * (a**4 + 6 * a**2 * s**2 + 3 * s**4)
    c2, c1, c0 = 3 * p**2 - 3 * p, 6 * A * p, 3 * A**2 - C
    u = (-c1 - np.sqrt(c1**2 - 4 * c2 * c0)) / (2 * c2)   # 양의 근
    tau = np.sqrt(u)
    mu2, mu4 = A + p * u, C + 3 * p * u**2
    print(f"A = {A:.5f}   C = {C:.5f}   u = tau^2 = {u:.4f}   tau = {tau:.4f}")
    print(f"mu2 = {mu2:.4f}   mu4 = {mu4:.4f}   3*mu2^2 = {3 * mu2**2:.4f}   초과첨도 = {mu4 / mu2**2 - 3:.1e}")

    def f(t):
        return ((1 - p) / 2 * stats.norm.pdf(t, a, s) + (1 - p) / 2 * stats.norm.pdf(t, -a, s)
                + p * stats.norm.pdf(t, 0.0, tau))

    print(f"밀도 f(0) = {f(0.0):.4f},  f(1) = {f(1.0):.4f},  f(0)/f(1) = {f(0.0) / f(1.0):.4f}")

    rng = np.random.default_rng(0)
    n = 2000
    which, z = rng.random(n), rng.standard_normal(n)
    x = np.where(which < p, tau * z,
                 np.where(which < p + (1 - p) / 2, a + s * z, -a + s * z))

    jb = stats.jarque_bera(x)
    sw = stats.shapiro(x)
    ad = stats.anderson(x, "norm")
    print(f"G1 = {stats.skew(x, bias=False):+.4f}   G2 = {stats.kurtosis(x, bias=False):+.4f}")
    print(f"skewtest      p = {stats.skewtest(x)[1]:.4f}")
    print(f"kurtosistest  p = {stats.kurtosistest(x)[1]:.4f}")
    print(f"jarque_bera   JB = {jb[0]:.4f}   p = {jb[1]:.4f}")
    print(f"shapiro       W = {sw[0]:.4f}   p = {sw[1]:.3e}")
    print(f"anderson      A^2 = {ad.statistic:.4f}   5% 임계값 = {ad.critical_values[2]:.4f}")

    edges = np.arange(-2.0, 2.01, 0.5)
    cnt, _ = np.histogram(x, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], cnt):
        print(f"  [{lo:+4.1f}, {hi:+4.1f})  {c:4d}  {'#' * (c // 8)}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(x, bins=60, density=True, alpha=0.6, edgecolor="black")
    axes[0].set_title("왜도 0, 초과첨도 0, 그런데 봉이 둘")
    stats.probplot(x, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q 그림")
    plt.tight_layout()
    plt.show()
    ```

    출력:

    ```text
    A = 1.10200   C = 1.93496   u = tau^2 = 4.8115   tau = 2.1935
    mu2 = 1.3426   mu4 = 5.4075   3*mu2^2 = 5.4075   초과첨도 = 4.4e-16
    밀도 f(0) = 0.0507,  f(1) = 0.4819,  f(0)/f(1) = 0.1052
    G1 = +0.0077   G2 = -0.0491
    skewtest      p = 0.8886
    kurtosistest  p = 0.6829
    jarque_bera   JB = 0.2449   p = 0.8847
    shapiro       W = 0.9396   p = 8.394e-28
    anderson      A^2 = 52.4738   5% 임계값 = 0.7850
      [-2.0, -1.5)    93  ###########
      [-1.5, -1.0)   405  ##################################################
      [-1.0, -0.5)   373  ##############################################
      [-0.5, +0.0)   106  #############
      [+0.0, +0.5)   104  #############
      [+0.5, +1.0)   387  ################################################
      [+1.0, +1.5)   361  #############################################
      [+1.5, +2.0)   112  ##############
    ```

    **적률을 보는 검정은 전부 통과시킨다.** $G_1 = +0.0077$, $G_2 = -0.0491$로 둘 다 0 근처고, 왜도검정 $p = 0.8886$, 첨도검정 $p = 0.6829$, 자크–베라 $JB = 0.2449$에 $p = 0.8847$이다. $n = 2000$인데도 세 검정이 모두 "정규성에 반하는 증거가 없다"고 말한다.

    **경험분포를 보는 검정은 압도적으로 기각한다.** 섀피로–윌크는 $W = 0.9396$, $p = 8.4 \times 10^{-28}$이고, 앤더슨–달링은 $A^2 = 52.47$로 $5\%$ 임계값 $0.785$의 **67배**다. 막대그림이 왜 그런지 보여 준다. $[-1.5, -1.0)$에 $405$개, $[0.5, 1.0)$에 $387$개가 쌓여 있는데 가운데 $[-0.5, 0.5)$ 두 칸에는 $106 + 104 = 210$개뿐이다. 정규라면 가운데가 가장 두꺼워야 할 자리가 가장 얇다. Q-Q 그림에서는 이것이 중앙부의 **평평한 계단**으로 나타난다.

    **까닭.** 자크–베라와 왜도·첨도 검정이 자료에서 읽는 것은 $(G_1, G_2)$ 두 수뿐이다. 분포를 그 두 수만으로 요약하면 무한히 많은 모양이 같은 요약을 갖고, 우리는 그 가운데 하나를 **설계해서** 집었다. 반면 섀피로–윌크와 앤더슨–달링은 순서통계량 전체를 정규분위수와 맞춰 보므로, 적률이 맞아도 **순서가 맞지 않으면** 잡아낸다. 가운데가 비어 있다는 사실은 적률 두 개에 거의 흔적을 남기지 않지만 경험분포함수에는 그대로 남는다.

    이것이 꼬리가 두꺼운 경우와 **다른** 종류의 이탈이라는 점도 중요하다. 여기서 4차 적률은 정규와 정확히 같다. 꼬리의 무게가 어긋난 것이 아니라 가운데의 모양이 어긋났고, 적률 검정은 바로 그것에 눈이 없다.

    **$\beta_2 \ge 1 + \gamma_1^2$는 아무 경고도 주지 못한다.** 이 분포는 $\gamma_1 = 0$, $\beta_2 = \mu_4/\mu_2^2 = 3$이므로 부등식은 $3 \ge 1$이 되어 여유 있게 성립한다. 2장의 부등식은 $(\gamma_1, \beta_2)$ 쌍이 **불가능한** 영역에 있는지만 걸러 준다. 가능한 영역 안에 있다는 것은 정규일 수 있다는 뜻이 아니고, 여기서는 정규와 **똑같은** 점 $(0, 3)$에 앉아 있으면서 정규가 아니다.

    **결론.** 변환 뒤에 왜도와 첨도가 0이 되었다는 것은 정규성의 **필요조건**을 확인한 것일 뿐 충분조건이 아니다. 5절이 히스토그램과 Q-Q 그림을 "반드시" 함께 보라고 한 것은 수사가 아니다. 적률 두 개로 짠 그물은 구멍이 있고, 그 구멍은 작정하면 통과할 수 있을 만큼 크다. $\square$

---

## 정리하며

변환을 **적용하고 그 효과를 확인**하는 절차를 밟았다.

- **변환 전후를 반드시 비교한다.** 히스토그램·Q-Q 그림과 왜도·첨도 값, 그리고 정규성 검정을 나란히 놓아 개선 여부를 판단한다.
- **박스–콕스의 $\lambda$ 는 최대가능도로 추정된다.** `scipy.stats.boxcox` 가 최적 $\lambda$ 와 변환된 자료를 함께 돌려주며, **추정된 $\lambda$ 가 0 이나 0.5 근처면 해석하기 쉬운 로그·제곱근으로 반올림**하는 것이 관례다.
- **양수 제약을 확인한다.** 0 이나 음수가 있으면 상수를 더하거나 여–존슨으로 가야 하며, **상수를 더하는 선택이 결과를 바꾼다**는 점에 유의한다.
- **변환이 항상 통하지는 않는다.** 이봉분포나 이산성이 강한 자료는 어떤 변환으로도 정규가 되지 않으며, 그때는 부트스트랩이나 비모수로 옮긴다.
- **되돌릴 때를 대비한다.** 예측값과 구간을 원래 척도로 보고하려면 역변환이 필요하고, 그 과정에서 평균이 중앙값이 된다는 점을 밝혀야 한다.

다음 절 **응용 개관**으로 14장을 마무리한다.
