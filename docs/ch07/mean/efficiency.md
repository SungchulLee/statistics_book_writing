# 표본평균의 효율성

어떤 추정량이 불편이고 일치한다는 것을 확인하고 나면 자연스럽게 다음 질문이 따라온다: 얼마나 정밀할 수 있는가? 같은 모수의 불편추정량들 중에서도 어떤 것은 다른 것보다 분산이 작다. 가능한 가장 작은 분산 — Cramér–Rao 하한 — 에 도달하는 추정량을 **효율적**이라고 한다. 이 절에서는 표본평균이 언제, 왜 그 이름을 얻는지, 그리고 정규성이 깨지면 어떻게 되는지 살펴본다.

## 효율성의 정의

모수 $\theta$의 불편추정량 $\hat{\theta}$는 그 분산이 Cramér–Rao 하한(CRLB)과 같을 때 **효율적**이라고 한다:

$$
\operatorname{Var}(\hat{\theta}) = \frac{1}{n \, I(\theta)}
$$

여기서 $I(\theta)$는 관측값 하나에 대한 Fisher 정보량이다. 이 등식을 만족하는 불편추정량은 $\theta$의 모든 불편추정량 중에서 달성 가능한 가장 작은 분산을 갖는다.

## 정규분포 평균에 대한 CRLB

정규족 $X \sim N(\mu, \sigma^2)$은 CRLB가 성립하기 위한 정칙조건을 만족한다(받침이 $\mu$에 의존하지 않고, 로그가능도가 두 번 미분 가능하며 기댓값과 미분의 순서를 바꿀 수 있다). 관측값 하나에 대한 Fisher 정보량은 $I(\mu) = 1/\sigma^2$이므로 CRLB는 다음을 준다:

$$
\operatorname{Var}(\hat{\mu}) \geq \frac{1}{n \, I(\mu)} = \frac{\sigma^2}{n}
$$

표본평균 $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$의 분산은 $\operatorname{Var}(\bar{X}) = \sigma^2 / n$으로 이 하한과 정확히 일치한다. 정규성 아래에서 $\bar{X}$는 최대가능도추정량이자 $\mu$의 균일최소분산불편추정량(UMVUE)이므로, $\mu$의 모든 불편추정량 중에서 효율적이다.

## 점근 상대효율

표본평균은 정규성 아래에서 효율적이지만, 실제 자료는 흔히 정규 모형에서 벗어난다. 꼬리가 두껍거나 치우친 분포에서는 표본평균이 효율성의 우위를 잃는다. **점근 상대효율(ARE)**은 같은 정밀도를 얻기 위해 한 추정량이 다른 추정량에 비해 관측값이 몇 개나 필요한지를 재어 두 추정량을 비교하는 방법이다.

같은 모수의 두 추정량 $T_1$과 $T_2$에 대해 $T_2$ 대비 $T_1$의 ARE는 다음으로 정의된다:

$$
\operatorname{ARE}(T_1, T_2) = \frac{\operatorname{Var}(T_2)}{\operatorname{Var}(T_1)}
$$

$\operatorname{ARE}(T_1, T_2) > 1$이면 추정량 $T_1$이 더 효율적이다(관측값이 더 적게 필요하다). $\operatorname{ARE}(T_1, T_2) < 1$이면 $T_2$가 더 효율적이다.

### 표본평균과 중앙값의 ARE

정규분포 아래에서 점근분산은 $\operatorname{Var}(\bar{X}) = \sigma^2/n$, $\operatorname{Var}(\text{중앙값}) = \pi\sigma^2/(2n)$이므로:

$$
\operatorname{ARE}(\bar{X}, \text{중앙값}) = \frac{\operatorname{Var}(\text{중앙값})}{\operatorname{Var}(\bar{X})} = \frac{\pi}{2} \approx 1.57
$$

즉 정규성 아래에서 표본평균은 중앙값보다 약 57% 더 효율적이다. 중앙값이 $\bar{X}$의 정밀도를 따라잡으려면 관측값이 대략 1.57배 필요하다.

그러나 꼬리가 두꺼운 분포에서는 순위가 뒤집힌다. 다음 표는 몇몇 분포에서 중앙값 대비 표본평균의 ARE를 정리한 것이다:

| 분포 | $\operatorname{ARE}(\bar{X}, \text{중앙값})$ | 해석 |
|---|---|---|
| Normal | $\pi/2 \approx 1.57$ | 평균이 57% 더 효율적 |
| Double exponential (Laplace) | $1/2 = 0.50$ | 중앙값이 두 배 더 효율적 |
| Cauchy | $0$ (평균의 분산이 무한) | 중앙값이 확실히 우월 |

Laplace 분포에서는 중앙값이 표본평균의 절반에 해당하는 관측값만 있으면 된다. Cauchy 분포에서는 표본평균의 분산이 무한하여 표본크기와 무관하게 쓸모 있는 정보를 주지 못한다 — 중앙값이 분명한 선택이다.

!!! tip "실무 지침"
    바탕 분포가 근사적으로 정규이면 표본평균이 최선의 선택이다. 두꺼운 꼬리나 이상점이 있으면 절사평균이나 중앙값 같은 로버스트한 대안이 정규성 아래에서 약간의 효율을 희생하는 대신 더 나은 정밀도를 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
**중앙값과 평균의 ARE.** (a) 정규 자료에서 ARE가 $2/\pi$임을 보여라. (b) $t_3$ 자료에서는 어느 쪽이 이기는가?

</div>

??? success "풀이"
    (a) 중앙값의 점근분산은 $1/[4 f(\mu)^2 n]$이며, 여기서 $f$는 중앙값에서의 밀도이다. $N(\mu, \sigma^2)$에서 $f(\mu) = 1/(\sigma\sqrt{2\pi})$이므로 $\mathrm{AVar}(\text{중앙값}) = \pi\sigma^2/(2n)$이다. 평균은 $\sigma^2/n$. ARE = $2/\pi \approx 0.637$. 평균이 1.57배로 이긴다.

    (b) $t_3$에서 0에서의 밀도는 $\Gamma(2)/(\sqrt{3\pi}\Gamma(3/2)) \approx 0.368$이다. 평균의 분산은 유한하지만($\mathrm{Var}(X) = 3$이므로 $\mathrm{Var}(\bar X) = 3/n$) 크다. 중앙값의 점근분산은 $1/[4(0.368)^2 n] \approx 1.85/n$으로 훨씬 작다. **꼬리가 두꺼우면 중앙값이 크게 이긴다.**

    모의실험이 이를 확인해 준다: 정규에서는 평균의 분산 $\approx 1/n$ 대 중앙값 $\approx \pi/(2n)$. $t_3$에서는 평균의 분산 $3/n$이 중앙값의 $1.85/n$보다 크며, 자유도가 2에 가까워질수록 격차가 벌어진다.

<div class="drillbox" markdown>

**연습문제 2.**
**축소추정량.** $\hat\mu_\lambda = \lambda \bar X$. (a) 평균제곱오차를 유도하라. (b) 최적 $\lambda^*$. (c) $\lambda^*$를 직접 쓸 수 없는 이유는?

</div>

??? success "풀이"
    (a) 편향 = $(\lambda - 1)\mu$. 분산 = $\lambda^2 \sigma^2/n$. MSE = $(\lambda-1)^2 \mu^2 + \lambda^2 \sigma^2/n$.

    (b) $d\mathrm{MSE}/d\lambda = 2(\lambda - 1)\mu^2 + 2\lambda\sigma^2/n = 0 \Rightarrow \lambda^* = \mu^2/(\mu^2 + \sigma^2/n)$.

    항상 $\lambda^* < 1$이다. $\sigma^2/n$이 클 때(잡음이 클 때) 작아진다 — 과감하게 축소하라. 신호가 지배하면 1에 가깝게 커진다.

    (c) $\lambda^*$가 미지의 $\mu$에 의존한다. 대입한 $\hat\lambda$는 그 자체의 변동성을 갖는다. James-Stein 추정량이 이를 다룬다: 표본자료를 적응적으로 사용하는 경험적 Bayes 축소인자로, $p \ge 3$에서 MLE를 지배한다.

<div class="drillbox" markdown>

**연습문제 3.**
**James-Stein 추정량.** $p \ge 3$인 $\mathbf X \sim N(\boldsymbol\mu, I_p)$에서 $\hat{\boldsymbol\mu}_{\text{JS}} = (1 - (p-2)/\|\mathbf X\|^2)\mathbf X$와 MLE를 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    rng = np.random.default_rng(0)
    p, R = 10, 20_000
    mu = np.ones(p) * 0.5
    mse_mle = mse_js = 0.0
    for _ in range(R):
        x = mu + rng.standard_normal(p)
        js = (1 - (p - 2)/np.dot(x, x)) * x
        mse_mle += np.sum((x - mu)**2)
        mse_js += np.sum((js - mu)**2)
    print(f"MSE MLE={mse_mle/R:.3f}  JS={mse_js/R:.3f}")
    ```

    출력:

    ```
    MSE MLE=10.024  JS=3.625
    ```

    예상 결과: $p \ge 3$일 때 모든 $\boldsymbol\mu$에 대해 MSE(JS) < MSE(MLE)이다 (Stein, 1956). MLE는 차원 3 이상에서 **허용 불가능**하다 — 언제나 더 나은 추정량이 존재한다.

    실무적 영향: 현대 축소법(능형회귀, 계층적 Bayes, 라소)의 토대이다. 성분 $\mu_i$들이 서로 무관해도 함께 축소하면 전체 평균제곱오차가 개선된다.

<div class="drillbox" markdown>

**연습문제 4.**
**효율적 = CRLB 달성.** $N(\mu, \sigma^2)$에서 $\bar X$가 점근적으로만이 아니라 모든 $n$에서 효율적임(CRLB를 달성함)을 보여라.

</div>

??? success "풀이"
    정규분포 평균의 Fisher 정보량: 관측값당 $I(\mu) = 1/\sigma^2$, 전체는 $nI(\mu) = n/\sigma^2$.

    CRLB: $\mathrm{Var}(\hat\mu) \ge 1/(nI(\mu)) = \sigma^2/n$.

    $\mathrm{Var}(\bar X) = \sigma^2/n$ — **CRLB를 정확히 달성한다**.

    CRLB의 등호는 드물다 — 보통 MLE는 점근적으로만 CRLB에 도달한다. 정규분포 평균의 $\bar X$는 임의의 $n$에서 정확히 효율적인 몇 안 되는 사례이다. 점수함수 $\partial \log f/\partial\mu = (X - \mu)/\sigma^2$가 $X$에 대해 선형이어서, (CRLB 유도의 바탕이 되는) Cauchy-Schwarz 부등식이 등호로 성립하기 때문이다.

<div class="drillbox" markdown>

**연습문제 5.**
**효율성과 충분통계량.** 효율성을 충분성과 연결하라: $\bar X$는 $\mu$에 대해 충분하기 때문에 효율적이다.

</div>

??? success "풀이"
    **Rao-Blackwell 정리**에 의해, 임의의 불편추정량은 충분통계량으로 조건부기댓값을 취해 개선할 수 있다.

    ($\sigma^2$이 알려져 있을 때) $\bar X$는 $\mu$에 대해 충분하다: 가능도가 $L(\mu) = f(\bar X, \sigma^2/n) \cdot h(X_1, \ldots, X_n)$으로 인수분해되며 $h$는 $\mu$에 의존하지 않는다.

    **Lehmann-Scheffé**에 의해, 유일한 UMVUE는 완비충분통계량의 함수이다. $\bar X$는 완비 + 충분 + 불편이므로 UMVUE이다.

    **말로 하면:** 효율성은 충분함(자료를 전부 사용함) + 불편함(체계적 오차 없음)에서 따라 나온다. $\bar X$는 두 조건을 모두 만족한다.

    ($X_1$ 같은) 비효율적 추정량은 표본 전체를 쓰지 않아 정보를 버린다.

<div class="drillbox" markdown>

**연습문제 6.**
**고차원에서의 맞바꿈.** 축소추정량이 고차원에서는 MLE를 지배하지만 저차원에서는 그렇지 않은 이유는 무엇인가?

</div>

??? success "풀이"
    $p$차원에서 MLE의 위험: $\mathrm{Risk}(\hat{\boldsymbol\mu}_{\text{MLE}}) = p\sigma^2$.

    JS 축소: $\mathrm{Risk}(\hat{\boldsymbol\mu}_{\text{JS}}) = p\sigma^2 - (p-2)^2 \mathbb{E}[1/\|\mathbf X\|^2]$.

    $p \ge 3$이면 $(p-2)^2 > 0$이므로 JS가 균일하게 지배한다. $p \le 2$이면 보정항이 0이거나 음수여서 MLE가 최적으로 남는다.

    **직관적 설명:** 고차원에서 MLE는 더 많은 방향을 "탐색"하며 오차를 누적한다. 고차원 공간의 대부분이 참값에서 멀기 때문에, 0(또는 임의의 고정점) 쪽으로 아무렇게나 축소해도 이 초과 오차가 줄어든다.

    **Stein 현상:** MLE는 *오직* 차원 3 이상에서만 허용 불가능하다. $p = 3$이라는 경계는 정확하다 — $p = 2$에서는 MLE가 허용 가능하다.

    실무적 귀결: 고차원 회귀(예측변수 $p$개)에서는 축소법(능형, 라소, 엘라스틱 넷)이 비슷한 원리로 OLS를 일상적으로 능가한다.

---

## 정리하며

정규모집단에서 표본평균은 **효율적**이다. 크라메르–라오 하한을 정확히 달성한다.

$$
I(\mu)=\frac{1}{\sigma^2} \;\Longrightarrow\; \mathrm{Var}(\bar X)=\frac{\sigma^2}{n}=\frac{1}{nI(\mu)}
$$

- **모든 불편추정량 중 최선이다.** 정규성 아래에서는 더 정밀한 불편추정량이 존재하지 않는다.
- **정규성이 깨지면 최적성도 깨진다.** 중앙값 대비 점근상대효율이 분포에 따라 뒤집힌다.

| 분포 | 중앙값 점근분산 | 평균 분산 | 승자 |
|---|---|---|---|
| 정규 | $\pi/2\approx1.571$ | $1$ | 평균이 $57\%$ 우세 |
| 라플라스 | $1$ | $2$ | **중앙값이 2배 우세** |
| 코시 | 유한 | 무한 | **중앙값만 작동** |

- **밀도의 모양이 승자를 정한다.** 3장의 표본분위수 중심극한정리가 그 근거다. 중앙값의 점근분산은 $1/(4f(m)^2n)$ 이므로 **중앙에 밀도가 뾰족하게 몰린 분포에서 중앙값이 유리하다.**
- **효율성은 모형이 옳다는 전제 위의 최적성이다.** 그 전제가 흔들리면 최적이던 것이 최악이 될 수 있으며, 다음 절의 로버스트 추정량이 그 대비책이다.

다음 절 **절사평균과 윈저화 평균**으로 넘어간다.
