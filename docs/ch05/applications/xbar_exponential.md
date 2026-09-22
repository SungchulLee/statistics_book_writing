# X̄의 표본분포 (Exponential)

## 개요

앞 쪽의 균등 모집단에서는 $n = 5$만으로도 표본평균이 종 모양이 되었다. 같은 실험을 지수분포에서 하면 그렇게 되지 않는다. 이 쪽에서 볼 것은 중심극한정리가 **느리게** 듣는 모습이다.

지수분포는 오른쪽으로 심하게 치우쳐 있다. 왼쪽은 0에서 막혀 있는데 오른쪽으로는 긴 꼬리가 뻗어 있어, 작은 값이 대부분이고 아주 큰 값이 가끔 섞인다. 이런 모집단에서 다섯 개를 뽑아 평균을 내면 그 치우침이 표본평균의 분포에까지 따라 들어온다. 봉우리가 생기기는 하지만 아직 한쪽으로 기울어진 봉우리다.

물론 $n$을 키우면 결국 정규가 된다. 문제는 "결국"이 언제인가이고, 여기서는 그 답이 균등분포보다 훨씬 크다. 흔히 외우는 "$n \ge 30$이면 충분하다"는 어림이 이 모집단에서는 통하지 않는다는 것을 연습문제에서 수치로 확인하게 된다.

## 모집단 모형

각 관측값이 비율 모수 $\lambda = 1$인 지수분포에서 나온다고 하자.

$$
X \sim \text{Exp}(1), \qquad f(x) = e^{-x}, \quad x \ge 0
$$

이때 모평균과 모분산은 둘 다 1이다. 평균과 표준편차가 같다는 것 자체가 이 분포가 얼마나 넓게 퍼져 있는지를 말해 준다.

$$
\mu = E[X] = \frac{1}{\lambda} = 1, \qquad \sigma^2 = \text{Var}(X) = \frac{1}{\lambda^2} = 1
$$

## 합이 감마분포라 정확한 답을 안다

$\text{Exp}(\lambda)$에서 독립으로 뽑은 확률표본 $X_1, \ldots, X_n$에 대해서도 표본평균의 중심과 폭은 앞 쪽과 똑같은 방식으로 곧바로 나온다. 모집단이 치우쳤든 아니든 이 두 값에는 근사가 없다.

$$
E[\bar{X}] = \mu = \frac{1}{\lambda}, \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} = \frac{1}{n\lambda^2}
$$

그런데 이 모집단에는 다른 예에 없는 이점이 하나 더 있다. 모양까지 정확히 알 수 있다는 것이다.

!!! info "정확한 분포"
    합 $S_n = \sum_{i=1}^n X_i$는 감마분포를 따른다: $S_n \sim \text{Gamma}(n, \lambda)$. 따라서 $\bar{X} = S_n/n \sim \text{Gamma}(n, n\lambda)$이며 형상이 $n$, 비율이 $n\lambda$이다. $n \to \infty$일 때 중심극한정리는 다음을 보장한다:

    $$
    \bar{X} \;\dot{\sim}\; N\!\left(\frac{1}{\lambda},\; \frac{1}{n\lambda^2}\right)
    $$

덕분에 정규근사가 얼마나 빗나가는지를 어림이 아니라 정확한 값과 견주어 잴 수 있다. 연습문제 3과 9에서 그 대조가 이 쪽의 결론을 수치로 못 박는다.

## 모의실험

앞 쪽과 똑같은 그림을 이번에는 지수 모집단에서 그린다. 맨 위는 모집단, 가운데는 거기서 뽑은 **표본 하나**, 맨 아래는 그런 표본을 1만 번 뽑아 얻은 표본평균들의 분포다. 세 패널의 가로 눈금을 같게 묶어 두었으므로 퍼짐을 눈으로 직접 견줄 수 있다.

<div class="codebox" markdown>

### 예제 1. 지수 모집단에서 표본평균의 표집분포 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

sample_size = 5
n_samples = 10_000
n_population = 10_000

# Exp(1)에서 큰 모집단을 만든다. 치우친 모집단이다.
population = np.random.exponential(size=(n_population,))

# 표본을 딱 하나 뽑는다. 현실에서 우리가 실제로 갖게 되는 것이 이것뿐이다.
# 아래 가운데 패널에 점 몇 개로 그려진다.
single_sample = np.random.choice(population, size=sample_size, replace=False)

# 표본을 되풀이해 뽑으며 표본평균을 기록한다. 이 값들의 분포가 표집분포다.
sample_means = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# 모집단과 표집분포를 나란히 그린다.
# 세 패널을 sharex=True 로 묶는 것이 이 그림의 핵심 장치다.
# 가로 눈금이 같아야 세 분포의 **퍼짐**을 직접 견줄 수 있다.
#   위   모집단      : 가장 넓다
#   가운데 표본 하나  : 모집단에서 뽑은 점 몇 개
#   아래  표집분포    : 눈에 띄게 좁다. 이 좁아짐이 sigma/sqrt(n) 이다.
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

_, bins, _ = ax0.hist(population, bins=100)
ax0.set_title("Population Distribution (Exponential)")

ax1.scatter(single_sample, np.zeros_like(single_sample), s=100)
ax1.set_title(f"Sample Distribution (n = {sample_size})")

ax2.hist(sample_means, bins=bins)
ax2.set_title("Sampling Distribution of X-bar")

plt.tight_layout()
plt.show()
```

![Population Distribution (Exponential)](./img/xbar_exponential_40.png)

</div>

## 해석

맨 위의 모집단 분포는 오른쪽으로 심하게 치우쳐 있고 큰 값까지 뻗는 긴 꼬리를 갖는다. 가운데의 표본 하나는 점 다섯 개뿐인데, 운에 따라 다섯 개가 모두 작은 값일 수도 있고 하나가 유난히 클 수도 있다.

맨 아래에서는 두 가지가 동시에 보인다. 하나는 균등분포에서와 같다. 표본분포가 $n = 5$에서도 이미 모집단보다 훨씬 좁게 모여 있다. 모집단 표준편차가 $\sigma = 1$인 데 비해 표준오차는 $\text{SE}(\bar{X}) = 1/\sqrt{5} \approx 0.447$이니 폭이 절반 아래로 줄어든 셈이다.

다른 하나가 이 쪽의 차이점이다. 모양이 아직 대칭이 아니다. $n = 5$에서는 표본분포에 오른쪽 치우침이 눈에 띄게 남아 있으며, $n$을 키워야 점점 대칭이 되고 정규분포에 가까워진다. 좁아지는 일과 정규가 되는 일이 따로 논다는 점이 요점이다. 폭은 공식대로 곧바로 줄지만 모양은 천천히 고쳐진다.

!!! warning "치우친 모집단에서의 소표본"
    중심극한정리의 수렴 속도는 모집단이 얼마나 치우쳤는지에 달려 있다. 지수분포(왜도 $= 2$)에서는 정규근사가 추론에 믿을 만해지려면 $n \ge 30$ 이상이 필요할 수 있다.

### 왜 이 모집단이 현실적인가

$\text{Exp}(1)$은 인공적인 분포처럼 보이지만, 여기서 본 모양은 실제 자료에서 흔하다. **소득**이 대표적이다. 대다수가 중간값 근처에 모여 있고 소수가 아주 큰 값을 가지며, 왼쪽은 0에서 막혀 있다. 대기 시간, 보험 청구액, 웹페이지 체류 시간, 도시 인구도 같은 모양이다.

이런 자료에서는 위의 세 패널이 그대로 실무의 문제가 된다. 맨 위의 모집단을 보고 "평균 소득"을 말하면 대다수 사람의 형편을 대표하지 못한다. 평균이 중앙값보다 한참 오른쪽에 있기 때문이다. 가운데의 표본 하나는 우리가 실제로 손에 쥐는 전부인데, 다섯 명을 뽑았는데 우연히 고소득자가 한 명 끼면 표본평균이 크게 튄다. 그 튐의 크기를 재는 것이 맨 아래의 표집분포이고 그 폭이 표준오차다. 그런데 치우친 모집단에서는 $n$이 작을 때 이 분포 자체가 치우쳐 있어, 표준오차를 알아도 정규근사로 확률을 계산할 수가 없다.

그래서 소득처럼 치우친 자료에서는 **로그를 씌워 대칭으로 만든 뒤 분석하는** 방법이 흔히 쓰인다. 4장에서 본 로그정규분포가 그 배경이며, 14장에서 변환을 본격적으로 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $t < \lambda$에서 적률생성함수가 $M_X(t) = \lambda / (\lambda - t)$임을 사용하여 $X \sim \text{Exp}(\lambda)$의 평균과 분산을 유도하라.

</div>

??? success "풀이"
    적률생성함수는 $t < \lambda$에서 $M_X(t) = \frac{\lambda}{\lambda - t}$이다.

    1차 적률:

    $$
    M_X'(t) = \frac{\lambda}{(\lambda - t)^2}, \qquad E[X] = M_X'(0) = \frac{\lambda}{\lambda^2} = \frac{1}{\lambda}
    $$

    2차 적률:

    $$
    M_X''(t) = \frac{2\lambda}{(\lambda - t)^3}, \qquad E[X^2] = M_X''(0) = \frac{2\lambda}{\lambda^3} = \frac{2}{\lambda^2}
    $$

    분산:

    $$
    \text{Var}(X) = E[X^2] - (E[X])^2 = \frac{2}{\lambda^2} - \frac{1}{\lambda^2} = \frac{1}{\lambda^2}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$이면 $S_n = \sum_{i=1}^n X_i \sim \text{Gamma}(n, \lambda)$임을 보여라.

</div>

??? success "풀이"
    $X_i \sim \text{Exp}(\lambda)$의 MGF는 $M_{X_i}(t) = \frac{\lambda}{\lambda - t}$이다.

    독립성에 의해 합의 MGF는:

    $$
    M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t) = \left(\frac{\lambda}{\lambda - t}\right)^n
    $$

    이는 $\text{Gamma}(n, \lambda)$ 분포(형상 $n$, 비율 $\lambda$)의 MGF이다. MGF가 분포를 유일하게 결정하므로 $S_n \sim \text{Gamma}(n, \lambda)$이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $n = 5$, $\lambda = 1$일 때 감마분포를 사용하여 정확한 확률 $P(\bar{X} > 2)$를 계산하고 정규근사와 비교하라.

</div>

??? success "풀이"
    $S_5 \sim \text{Gamma}(5, 1)$일 때 $\bar{X} = S_5 / 5$이므로 $P(\bar{X} > 2) = P(S_5 > 10)$이다.

    Python으로:

    ```python
    from scipy import stats
    # 정확한 값: 지수표본의 합은 감마분포를 따른다.
    p_exact = 1 - stats.gamma.cdf(10, a=5, scale=1)
    # 정규근사. 평균 1, 표준오차 1/sqrt(5) 이다.
    p_normal = 1 - stats.norm.cdf(2, loc=1, scale=1/5**0.5)
    print(f"Exact (Gamma):       {p_exact:.6f}")
    print(f"Normal approximation: {p_normal:.6f}")
    ```

    출력:

    ```
    Exact (Gamma):       0.029253
    Normal approximation: 0.012674
    ```

    정확한 값은 약 0.0293이고 정규근사는 약 0.0127이다. $n = 5$는 중심극한정리가 지수분포의 치우침을 온전히 보정하기에 너무 작아, 정규근사가 오른쪽 꼬리 확률을 크게 과소추정한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 지수분포의 왜도는 $\gamma_1 = 2$이다. $\bar{X}$의 왜도가 $\gamma_1(\bar{X}) = 2/\sqrt{n}$임을 보여라. 표본크기가 얼마일 때 $\bar{X}$의 왜도가 0.5 아래로 떨어지는가?

</div>

??? success "풀이"
    왜도가 $\gamma_1$인 i.i.d. 확률변수에 대해 $\bar{X} = \frac{1}{n}\sum X_i$의 왜도는:

    $$
    \gamma_1(\bar{X}) = \frac{\gamma_1}{\sqrt{n}}
    $$

    $\bar{X}$의 3차 중심적률이 (독립인 복사본 $n$개를 더하고 $n$으로 나누므로) $\mu_3 / n^2$이고 $(\text{Var}(\bar{X}))^{3/2} = (\sigma^2/n)^{3/2}$이므로:

    $$
    \gamma_1(\bar{X}) = \frac{n \cdot \mu_3 / n^3}{(\sigma^2 / n)^{3/2}} = \frac{\mu_3}{n^2} \cdot \frac{n^{3/2}}{\sigma^3} = \frac{\mu_3}{\sigma^3 \sqrt{n}} = \frac{\gamma_1}{\sqrt{n}}
    $$

    $\text{Exp}(1)$에서 $\gamma_1 = 2$이므로 $\gamma_1(\bar{X}) = 2/\sqrt{n}$이다.

    $2/\sqrt{n} < 0.5$로 두면:

    $$
    \sqrt{n} > 4 \implies n > 16
    $$

    따라서 왜도가 0.5 아래로 떨어지려면 $n \ge 17$이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 모의실험을 $n = 5$ 대신 $n = 50$으로 반복하라. 표본평균의 히스토그램 위에 정규 밀도 $N(1, 1/50)$을 겹쳐 그리고 적합 정도를 정성적으로 서술하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats

    np.random.seed(1)
    population = np.random.exponential(size=10_000)

    sample_means = [
        np.mean(np.random.choice(population, size=50, replace=False))
        for _ in range(10_000)
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(sample_means, bins=60, density=True, alpha=0.5, label="Simulated")

    x = np.linspace(min(sample_means), max(sample_means), 200)
    ax.plot(x, stats.norm.pdf(x, loc=1, scale=1/np.sqrt(50)), "r--", lw=2,
            label="N(1, 1/50)")
    ax.legend()
    ax.set_title("Sampling Distribution of X-bar, n=50")
    plt.show()
    ```

    ![Sampling Distribution of X-bar, n=50](./img/xbar_exponential_185.png)

    $n = 50$에서는 표본평균의 히스토그램이 거의 대칭이고 $N(1, 1/50)$ 밀도를 바짝 따라간다. $\bar{X}$의 왜도가 $2/\sqrt{50} \approx 0.28$로 충분히 작아 정규근사가 아주 잘 맞는다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$\lambda$의 최대가능도추정량이 $\hat\lambda = 1/\bar X$임을 보이고, $E[\hat\lambda] = \frac{n}{n-1}\lambda$임을 유도하라. 불편으로 고치면 무엇이 되는가?

</div>

??? success "풀이"
    **최대가능도.** 로그가능도가 $\ell(\lambda) = n\ln\lambda - \lambda\sum x_i$이므로

    $$
    \ell'(\lambda) = \frac n\lambda - \sum x_i = 0 \implies \hat\lambda = \frac{n}{\sum X_i} = \frac{1}{\bar X}
    $$

    이다.

    **기대값.** 연습문제 2에서 $S_n = \sum X_i \sim \text{Gamma}(n,\lambda)$이므로

    $$
    E\!\left[\frac1{S_n}\right] = \int_0^\infty \frac1s\cdot\frac{\lambda^n s^{n-1}e^{-\lambda s}}{\Gamma(n)}ds = \frac{\lambda^n}{\Gamma(n)}\int_0^\infty s^{n-2}e^{-\lambda s}ds = \frac{\lambda^n}{\Gamma(n)}\cdot\frac{\Gamma(n-1)}{\lambda^{n-1}} = \frac{\lambda}{n-1}
    $$

    이다($n \ge 2$에서만 수렴한다). 따라서

    $$
    E[\hat\lambda] = n\,E\!\left[\frac1{S_n}\right] = \frac{n}{n-1}\lambda
    $$

    이다. $\square$

    **항상 과대추정한다.** $1/x$가 볼록함수이므로 옌센 부등식에서 예상되는 방향이다. 편향의 크기는 $\lambda/(n-1)$로, $n=5$면 25%나 된다.

    **불편 보정.**

    $$
    \tilde\lambda = \frac{n-1}{n}\hat\lambda = \frac{n-1}{\sum X_i}
    $$

    로 두면 불편이 된다. $n=1$이면 기대값이 존재하지 않으므로 보정 자체가 불가능하다.

    이 예가 보여 주는 것은 **최대가능도추정이 불변성은 가져도 불편성은 갖지 않는다**는 점이다. $\mu = 1/\lambda$의 MLE는 $\bar X$이고 이것은 불편인데, 역수를 취하는 순간 편향이 생긴다. 어느 모수화로 문제를 적느냐에 따라 불편성이 나타났다 사라진다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
지수분포에서 평균 $1/\lambda$를 추정하는 두 방법 — 표본평균 $\bar X$와 (중앙값을 $\ln 2$로 나눈) $\tilde X/\ln 2$ — 의 점근분산을 견주어라. 어느 쪽이 나은가?

</div>

??? success "풀이"
    **표본평균.** $\operatorname{Var}(X) = 1/\lambda^2$이므로

    $$
    \operatorname{Var}(\bar X) = \frac{1}{n\lambda^2}
    $$

    **중앙값 기반.** 표본중앙값의 점근분산은 일반적으로

    $$
    \operatorname{Var}(\tilde X) \approx \frac{1}{4n\{f(m)\}^2}
    $$

    이다. 지수분포에서 중앙값은 $m = \ln2/\lambda$이고 그 점의 밀도는

    $$
    f(m) = \lambda e^{-\lambda m} = \lambda e^{-\ln 2} = \frac\lambda2
    $$

    이므로 $\operatorname{Var}(\tilde X) \approx 1/(n\lambda^2)$이다. 평균의 추정량은 $\tilde X/\ln2$이므로

    $$
    \operatorname{Var}\!\left(\frac{\tilde X}{\ln2}\right) \approx \frac{1}{n\lambda^2(\ln2)^2} = \frac{1}{0.4805\,n\lambda^2}
    $$

    **비교.** 상대효율이

    $$
    \frac{\operatorname{Var}(\bar X)}{\operatorname{Var}(\tilde X/\ln2)} = (\ln2)^2 = 0.48
    $$

    로, 중앙값 기반 추정량의 분산이 **두 배 넘게** 크다. 같은 정밀도를 얻으려면 표본이 2.08배 필요하다.

    **뜻.** 모형이 정말 지수분포라면 표본평균을 써야 한다. $\bar X$가 충분통계량의 함수이자 최소분산불편추정량이기 때문이다.

    그래도 중앙값을 쓸 이유가 있다면 **모형이 틀렸을 가능성**이다. 자료에 오염이 섞이거나 꼬리가 지수보다 두꺼우면 $\bar X$는 몇 개의 큰 값에 끌려가지만 중앙값은 버틴다. 효율 52%를 보험료로 내고 강건성을 사는 거래이며, 고장시간 자료처럼 극단값의 신뢰도가 낮은 경우에는 합리적인 선택일 수 있다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
어떤 창구에 손님이 비율 $\lambda$인 포아송 과정으로 도착하고 서비스 시간이 평균 $1/\mu$인 지수분포를 따른다(M/M/1 대기행렬). 도착 간격 30개와 서비스 시간 30개를 관측해 각각 $\bar x = 2.5$분, $\bar y = 2.0$분을 얻었다. 평균 체류시간 $W = 1/(\mu-\lambda)$를 추정하고 문제점을 지적하라.

</div>

??? success "풀이"
    $\hat\lambda = 1/2.5 = 0.4$명/분, $\hat\mu = 1/2.0 = 0.5$명/분이므로

    $$
    \hat W = \frac{1}{\hat\mu-\hat\lambda} = \frac{1}{0.5-0.4} = 10\ \text{분}
    $$

    이다. 서비스 자체는 2분인데 대기까지 합치면 10분이다. 이용률 $\rho = \lambda/\mu = 0.8$의 효과다.

    **문제점 1 — 극도로 불안정하다.** $\hat W$는 두 추정값의 **차이의 역수**다. $\hat\mu - \hat\lambda = 0.1$인데 각 추정값의 표준오차가

    $$
    \operatorname{SE}(\hat\lambda) \approx \frac{\lambda}{\sqrt n} = \frac{0.4}{\sqrt{30}} = 0.073
    $$

    수준이다. 분모가 자기 표준오차보다 조금 큰 정도에 지나지 않는다. 델타 방법으로 계산하면

    $$
    \operatorname{SE}(\hat W) \approx \frac{\sqrt{\operatorname{Var}(\hat\mu)+\operatorname{Var}(\hat\lambda)}}{(\mu-\lambda)^2} \approx \frac{\sqrt{0.0083+0.0053}}{0.01} \approx 11.7\ \text{분}
    $$

    으로 추정값 자체보다 크다. 사실상 아무것도 말하지 못한다.

    **문제점 2 — 분모가 음수가 될 수 있다.** $\hat\mu < \hat\lambda$가 나오면 $\hat W$가 음수가 되고, 이는 "대기행렬이 발산한다"는 뜻이다. 표본이 작으면 실제로 $\rho<1$인데도 이런 일이 일어난다.

    **문제점 3 — 비선형 변환의 편향.** $1/(\mu-\lambda)$가 볼록이므로 $\hat W$는 $W$를 체계적으로 과대추정한다.

    **교훈.** **혼잡한 시스템($\rho \to 1$)의 성능 지표는 추정하기 대단히 어렵다.** 작은 추정오차가 결과를 폭발시킨다. 실무에서는 $\rho$를 직접 추정해 신뢰구간을 만들고, $W$는 부트스트랩이나 베이즈 방법으로 비대칭 구간을 보고하는 편이 낫다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$n=5$, $\lambda=1$일 때 $\mu = 1/\lambda$에 대한 (가) 감마분포에 기반한 정확한 95% 신뢰구간과 (나) 정규근사 구간 $\bar x \pm 1.96\,\bar x/\sqrt n$의 실제 포함확률을 비교하라.

</div>

??? success "풀이"
    **(가) 정확한 구간.** $2n\lambda\bar X = 2\lambda S_n \sim \chi^2_{2n}$이 추축량이다. $2n = 10$이므로

    $$
    P\!\left(\chi^2_{10,0.025} \le 2n\lambda\bar X \le \chi^2_{10,0.975}\right) = 0.95
    $$

    에서 $\mu = 1/\lambda$에 대해 풀면

    $$
    \left(\frac{2n\bar x}{\chi^2_{10,0.975}},\ \frac{2n\bar x}{\chi^2_{10,0.025}}\right) = \left(\frac{10\bar x}{20.483},\ \frac{10\bar x}{3.247}\right) = (0.488\bar x,\ 3.080\bar x)
    $$

    이다. **포함확률이 정확히 0.95**이며, $\bar x$를 중심으로 극도로 비대칭이다. 위로 3배까지 뻗는다.

    **(나) 정규근사 구간.** $\operatorname{SE}(\bar X) = \mu/\sqrt n$을 $\bar x/\sqrt5$로 추정하면

    $$
    \bar x \pm 1.96\frac{\bar x}{\sqrt5} = \bar x(1 \pm 0.8765) = (0.124\bar x,\ 1.877\bar x)
    $$

    이다. 실제 포함확률은 $P(0.124\bar X \le 1 \le 1.877\bar X)$, 즉 $P(0.533 \le \bar X \le 8.06)$인데 $\bar X \sim \text{Gamma}(5, \text{척도}=1/5)$이므로 계산하면 **약 0.868**이다.

    **비교.**

    | 방법 | 구간(배율) | 실제 포함확률 |
    |---|---|---|
    | 정확(카이제곱) | $(0.488,\ 3.080)\bar x$ | 0.950 |
    | 정규근사 | $(0.124,\ 1.877)\bar x$ | 0.868 |

    정규근사는 명목 95%에서 실제로는 87%밖에 담지 못한다. 게다가 **어긋남이 한쪽으로 몰려 있다.** 위쪽 한계가 너무 낮아 참값이 위로 빠져나가는 경우가 대부분이다.

    원인은 두 가지다. 첫째, $n=5$에서 $\bar X$의 왜도가 $2/\sqrt5 = 0.894$로 여전히 크다. 둘째, 표준오차를 $\mu$가 아니라 $\bar x$로 추정하면서 $\bar x$가 작게 나온 표본에서 구간이 함께 좁아지는 이중의 문제가 생긴다.

    **결론.** 정확한 추축량을 알 수 있으면 반드시 그것을 쓴다. 지수·감마·포아송처럼 카이제곱 관계가 있는 분포에서는 정확한 구간이 공짜로 얻어진다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$X_1,\dots,X_n \sim \text{Exp}(\lambda)$에서 $\min_i X_i$와 $\max_i X_i$의 분포를 구하고, $n$이 커질 때 세 통계량 $\min$, $\bar X$, $\max$가 각각 어떻게 움직이는지 견주어라.

</div>

??? success "풀이"
    **최솟값.** $P(\min > t) = \{e^{-\lambda t}\}^n = e^{-n\lambda t}$이므로

    $$
    \min_i X_i \sim \text{Exp}(n\lambda), \qquad E[\min] = \frac{1}{n\lambda}
    $$

    **최댓값.** $P(\max \le t) = (1-e^{-\lambda t})^n$이고, 순서통계량의 기대값 공식에서

    $$
    E[\max] = \frac{1}{\lambda}\sum_{k=1}^n\frac1k = \frac{H_n}{\lambda} \approx \frac{\ln n + \gamma}{\lambda}
    $$

    이다($\gamma \approx 0.5772$는 오일러 상수). 이는 최댓값을 $\min$부터 차례로 쌓아 올리는 분해에서 나오며, 무기억성 덕분에 $k$번째 구간이 $\text{Exp}((n-k+1)\lambda)$를 따른다.

    **세 통계량의 움직임.**

    | 통계량 | 중심 | 산포 | $n\to\infty$ |
    |---|---|---|---|
    | $\min$ | $\dfrac{1}{n\lambda}$ | $\dfrac{1}{n\lambda}$ | 0으로 수렴 |
    | $\bar X$ | $\dfrac1\lambda$ | $\dfrac{1}{\lambda\sqrt n}$ | $1/\lambda$로 수렴 |
    | $\max$ | $\dfrac{\ln n}{\lambda}$ | $\dfrac{\pi}{\lambda\sqrt6}$ | 발산하되 산포는 **상수** |

    셋의 성격이 완전히 다르다.

    - $\min$은 $1/n$의 속도로 0에 붙는다. 산포도 같은 속도로 줄어 상대적 불확실성은 그대로다.
    - $\bar X$는 큰수의 법칙대로 모평균에 수렴하고 산포가 $1/\sqrt n$로 준다. **유일하게 안정된 것**이다.
    - $\max$는 $\ln n$으로 천천히 발산하지만 **산포가 줄지 않는다.** 실제로 $\lambda\max - \ln n$이 굼벨분포로 수렴한다. 표본을 아무리 늘려도 최댓값의 불확실성은 그대로라는 뜻이다.

    극단값을 다룰 때 표본평균의 직관이 통하지 않는 이유가 여기 있다. **평균은 자료가 쌓이면 안정되지만 최댓값은 그렇지 않다.** 100년 홍수위나 최대 손실액 추정이 어려운 근본적인 까닭이며, 그래서 극단값 이론이라는 별도의 분야가 있다.

---

## 정리하며

지수 모집단은 중심극한정리를 시험하기 좋은 사례다. 심하게 치우쳐 있기 때문이다. $X\sim\text{Exp}(1)$이면 $\mu=1$, $\sigma^2=1$이고 왜도가 2인데, 왜도가 0이던 균등분포와 정확히 대조된다.

이 쪽에서 가져갈 것 하나는 그 치우침이 **얼마나 천천히 사라지는가**다. $\bar X$의 왜도는 $2/\sqrt n$으로 줄어든다. 제곱근으로 줄어드니 $n=30$에서도 0.37이 남아 눈에 띄고, $n=100$이 되어야 0.2다. 입문서에서 흔히 외우는 "$n\ge30$이면 충분하다"가 이 모집단에서는 성립하지 않는다.

그 어긋남을 어림이 아니라 정확한 값으로 잴 수 있다는 것이 이 예의 장점이다. 지수분포의 합은 감마분포이므로 $n\bar X\sim\text{Gamma}(n,1)$이고, 근사와 정확값을 나란히 놓고 견줄 수 있다. 그렇게 재 보면 오차가 어디에 몰려 있는지도 드러난다. **꼬리에서 가장 크다.** 3장에서 확인했듯 $n=30$의 오른쪽 꼬리에서 정규근사는 확률을 여러 배 과소평가한다. 신뢰구간의 중심부는 그런대로 맞아도 극단 분위수는 믿기 어렵다는 뜻이다.

다음 절 **정규 모집단**은 정반대의 경우다. 근사가 아예 필요 없고 모든 $n$에서 결과가 **정확**하다.
