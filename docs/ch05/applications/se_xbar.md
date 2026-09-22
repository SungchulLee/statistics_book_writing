# X̄의 표준오차

## 개요

통계량의 **표준오차**는 그 표본분포의 표준편차이다. 표본평균 $\bar{X}$에 대해 표준오차는 $\bar{X}$가 표본마다 얼마나 달라지는지를 정량화한다. 모평균 $\mu$의 추정량으로서 표본평균의 정밀도를 이해하는 데 가장 중요한 양이다. 이 페이지에서는 표준오차 공식을 유도하고, 모의실험으로 추정하며, 시각화하는 방법을 보인다.

<div class="defn" markdown>

### 정의 1. 표본평균의 표준오차 { .dfn }

평균이 $\mu$, 표준편차가 $\sigma$인 모집단에서 뽑은 확률표본 $X_1, \ldots, X_n$에 대해 $\bar{X}$의 표준오차는:

$$
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}
$$

!!! info "표준오차와 표준편차"

    - **표준편차** $\sigma$는 모집단에서 개별 관측값의 퍼짐을 잰다.
    - **표준오차** $\sigma / \sqrt{n}$는 반복추출에 걸친 표본평균 $\bar{X}$의 퍼짐을 잰다.
    - ($n > 1$이면) 표준오차는 언제나 $\sigma$보다 작고 $n$이 커질수록 줄어든다.

실무에서는 $\sigma$를 대개 모르므로 표본표준편차 $s$로 추정하여 **추정 표준오차**를 얻는다:

$$
\widehat{\text{SE}}(\bar{X}) = \frac{s}{\sqrt{n}}
$$

</div>

## 유도

i.i.d. 관측값에 대해 $\bar{X}$의 정의에서 출발하면:

$$
\text{Var}(\bar{X}) = \text{Var}\!\left(\frac{1}{n}\sum_{i=1}^n X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n}
$$

표준오차는 이 분산의 제곱근이다:

$$
\text{SE}(\bar{X}) = \sqrt{\text{Var}(\bar{X})} = \frac{\sigma}{\sqrt{n}}
$$

## 모의실험

다음 코드는 Uniform(0, 1) 모집단에서 $n = 5$로 표본평균 10,000개를 모의실험하여 경험적 표준오차를 계산하고 결과를 시각화한다.

<div class="codebox" markdown>

### 예제 1. 표본평균의 표준오차 모의실험 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# 크기 5짜리 균등표본 U(0,1)을 1만 번 뽑아 그때마다 표본평균을 기록한다.
X_bar = []
for _ in range(10_000):
    x = np.random.uniform(size=(5,))
    X_bar.append(x.mean())

# 1만 개 표본평균의 평균과 표준편차.
#   평균     -> 참 평균 0.5 에 가까워야 한다
#   표준편차 -> 이것이 표준오차다. 이론값은 sigma/sqrt(n) 이며
#              U(0,1)의 sigma = 1/sqrt(12) 이므로
#              (1/sqrt(12))/sqrt(5) = 0.1291 이 나와야 한다.
average = np.array(X_bar).mean()
standard_error = np.array(X_bar).std()

print(f"Estimated Mean of X_bar:  {average:.4f}")
print(f"Standard Error of X_bar:  {standard_error:.4f}")

# 히스토그램에 이론값을 겹쳐 그린다.
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_title("Sampling Distribution of X-bar")
ax.hist(X_bar, bins=100, density=True, alpha=0.3)
ax.vlines(average, ymin=0, ymax=5, color="k", lw=5, label="Mean")
ax.vlines(average + standard_error, ymin=0, ymax=5,
          color="k", ls="--", label="Mean +/- SE")
ax.vlines(average - standard_error, ymin=0, ymax=5,
          color="k", ls="--")
ax.legend()
plt.show()
```

출력:

```
Estimated Mean of X_bar:  0.4981
Standard Error of X_bar:  0.1287
```

![Sampling Distribution of X-bar](./img/se_xbar_45.png)

</div>

### 예상 출력

$n = 5$인 Uniform(0, 1)에 대해:

- **이론적 평균**: $\mu = 0.5$
- **이론적 표준오차**: $\sigma / \sqrt{n} = (1/\sqrt{12}) / \sqrt{5} \approx 0.1291$
- **경험적 값**은 이 이론값들에 가깝게 나와야 한다.

## 표준오차가 n에 따라 줄어드는 방식

표준오차는 $1/\sqrt{n}$로 줄어든다. 이는 다음을 뜻한다:

- $n$을 두 배로 하면 표준오차가 $\sqrt{2} \approx 1.41$배 줄어든다.
- $n$을 네 배로 하면 표준오차가 절반이 된다.
- 표준오차를 10분의 1로 줄이려면 관측값이 100배 필요하다.

| $n$ | $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$ | $n=1$ 대비 |
|---|---|---|
| 1 | $\sigma$ | 100% |
| 4 | $\sigma/2$ | 50% |
| 25 | $\sigma/5$ | 20% |
| 100 | $\sigma/10$ | 10% |
| 10,000 | $\sigma/100$ | 1% |

!!! warning "수확 체감"
    $1/\sqrt{n}$ 관계는 관측값을 하나 더할 때마다 정밀도 향상이 점점 작아짐을 뜻한다. $n = 100$에서 $n = 400$으로 (비용을 4배로) 늘려도 표준오차는 절반이 될 뿐이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 어떤 모집단의 $\sigma = 10$이다. $n = 25$, $n = 100$, $n = 400$에서 $\bar{X}$의 표준오차를 계산하고 "네 배 규칙"을 확인하라.

</div>

??? success "풀이"
    $$
    \text{SE}(n=25) = \frac{10}{\sqrt{25}} = \frac{10}{5} = 2.0
    $$

    $$
    \text{SE}(n=100) = \frac{10}{\sqrt{100}} = \frac{10}{10} = 1.0
    $$

    $$
    \text{SE}(n=400) = \frac{10}{\sqrt{400}} = \frac{10}{20} = 0.5
    $$

    $n$이 4배가 될 때마다 표준오차가 절반이 된다: $2.0 \to 1.0 \to 0.5$. 네 배 규칙이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> 어떤 연구자가 $\bar{X}$의 표준오차를 최대 0.5로 만들고자 한다. 모표준편차는 $\sigma \approx 8$로 추정된다. 최소 표본크기는 얼마인가?

</div>

??? success "풀이"
    다음이 필요하다:

    $$
    \frac{\sigma}{\sqrt{n}} \le 0.5 \implies \sqrt{n} \ge \frac{8}{0.5} = 16 \implies n \ge 256
    $$

    최소 표본크기 $n = 256$이 필요하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> $n \ge 1$에서 $\text{SE}(\bar{X})$가 $n$의 감소함수이자 볼록함수임을 증명하라. 볼록성은 표본크기를 늘릴 때의 한계 이득에 관해 무엇을 함의하는가?

</div>

??? success "풀이"
    $n > 0$에서 $f(n) = \sigma / \sqrt{n} = \sigma \cdot n^{-1/2}$이라 하자.

    1계도함수:

    $$
    f'(n) = -\frac{\sigma}{2} n^{-3/2} < 0
    $$

    따라서 $f$는 순감소한다. 표본이 클수록 언제나 표준오차가 작아진다.

    2계도함수:

    $$
    f''(n) = \frac{3\sigma}{4} n^{-5/2} > 0
    $$

    따라서 $f$는 순볼록이다. 볼록성은 $n$이 커질수록 표준오차의 감소 속도가 느려짐을 뜻한다. 실용적으로 말하면, 관측값을 하나 더 얻을 때마다 표준오차가 줄어드는 폭이 직전보다 작아진다. 표본크기를 늘리는 데에는 한계 수확 체감이 있다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 추정 표준오차 $\widehat{\text{SE}} = s / \sqrt{n}$를 사용할 때, 모집단이 정규이면 $(\bar{X} - \mu) / \widehat{\text{SE}}$가 자유도 $n - 1$인 $t$ 분포를 따름을 보여라.

</div>

??? success "풀이"
    $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$에 대해 다음을 떠올리자:

    - $Z = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0, 1)$
    - $Q = \frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$
    - $Z$와 $Q$는 독립이다.

    정의에 의해 $t$ 분포는 $Z \sim N(0,1)$, $Q \sim \chi^2(k)$, $Z \perp Q$일 때 비 $T = Z / \sqrt{Q/k}$이다.

    $$
    T = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \cdot \frac{1}{\sqrt{(n-1)S^2 / (\sigma^2(n-1))}} = \frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \cdot \frac{\sigma}{S} = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t(n-1)
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 모의실험을 Uniform(0, 1) 대신 Exponential(1) 모집단으로 바꾸어라. 이론적 표준오차 $(\sigma/\sqrt{n} = 1/\sqrt{5})$와 경험적 표준오차를 비교하라. 공식 $\text{SE} = \sigma/\sqrt{n}$은 정규가 아닌 모집단에서도 여전히 타당한가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    np.random.seed(0)

    X_bar = [np.random.exponential(size=5).mean() for _ in range(10_000)]

    empirical_se = np.std(X_bar)
    theoretical_se = 1 / np.sqrt(5)

    print(f"Theoretical SE: {theoretical_se:.4f}")
    print(f"Empirical SE:   {empirical_se:.4f}")
    ```

    출력:

    ```
    Theoretical SE: 0.4472
    Empirical SE:   0.4446
    ```

    이론적 표준오차는 $1/\sqrt{5} \approx 0.4472$이고 경험적 표준오차가 이 값에 매우 가깝게 나온다.

    그렇다. 공식 $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$은 모집단 모양과 무관하게 분산이 유한한 **임의의** 모집단에서 타당하다. $\text{Var}(\bar{X}) = \sigma^2/n$이 관측값의 독립성과 분산의 성질에서 곧바로 따라 나오기 때문이다. 이 공식은 정규성에 의존하지 않는다. 모집단 모양에 의존하는 것은 $\bar{X}$의 **분포**(정규인지 아닌지)이며, 표준오차 공식 자체는 보편적이다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$\sigma/\sqrt n$ 공식은 관측값이 **독립**일 때만 성립한다. 관측값들이 서로 상관 $\rho$를 갖는다면 $\operatorname{Var}(\bar X)$는 무엇이 되는가? $\rho > 0$이고 $n\to\infty$이면 어떻게 되는가?

</div>

??? success "풀이"
    모든 쌍의 상관이 $\rho$로 같다고 하자(등상관 구조). 그러면

    $$
    \operatorname{Var}(\bar X) = \frac{1}{n^2}\left\{\sum_i\operatorname{Var}(X_i) + \sum_{i\ne j}\operatorname{Cov}(X_i,X_j)\right\} = \frac{1}{n^2}\left\{n\sigma^2 + n(n-1)\rho\sigma^2\right\}
    $$

    이므로

    $$
    \operatorname{Var}(\bar X) = \frac{\sigma^2}{n}\left\{1+(n-1)\rho\right\}
    $$

    이다. $\rho = 0$이면 익숙한 $\sigma^2/n$으로 돌아온다.

    **$\rho > 0$이고 $n\to\infty$이면**

    $$
    \operatorname{Var}(\bar X) \to \rho\sigma^2 \ne 0
    $$

    이다. **표본을 아무리 늘려도 표준오차가 0으로 가지 않는다.** 관측값들이 같은 정보를 되풀이해 담고 있어 새 관측이 새 정보를 주지 못하기 때문이다.

    괄호 안의 $1+(n-1)\rho$를 **설계효과(DEFF)**라 하고, $n/\text{DEFF}$를 **유효 표본크기**라 한다. $\rho=0.05$이고 $n=20$인 군집이 50개면 관측이 1000개지만 유효 표본크기는

    $$
    \frac{1000}{1+19\times0.05} = \frac{1000}{1.95} = 513
    $$

    으로 절반에 지나지 않는다.

    현실에서 이런 상황이 흔하다. 같은 학교 학생들, 같은 환자에게서 반복 측정한 값, 시계열의 이웃한 관측값이 모두 그렇다. **독립을 가정하고 $\sigma/\sqrt n$을 쓰면 표준오차를 심하게 과소평가하고**, 그 결과 신뢰구간이 좁아지고 있지도 않은 유의성이 나온다. 군집 표준오차, 혼합효과 모형, 일반화추정방정식이 모두 이 문제를 다루는 도구다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
실제로는 $\sigma$를 모르므로 $\widehat{\operatorname{SE}} = s/\sqrt n$을 쓴다. 정규모집단에서 이 추정값 자체의 상대 표준오차가 근사적으로 $1/\sqrt{2(n-1)}$임을 보이고, $n=5, 30$에서 값을 구하라.

</div>

??? success "풀이"
    $\widehat{\operatorname{SE}} = S/\sqrt n$이므로 상대 변동은 $S$의 것과 같다. 정규모집단에서 $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$이므로

    $$
    \operatorname{Var}(S^2) = \frac{2\sigma^4}{n-1}
    $$

    이다. 델타 방법으로 $g(x)=\sqrt x$를 적용하면 $g'(\sigma^2) = 1/(2\sigma)$이므로

    $$
    \operatorname{Var}(S) \approx \frac{1}{4\sigma^2}\cdot\frac{2\sigma^4}{n-1} = \frac{\sigma^2}{2(n-1)}
    $$

    이고 상대 표준오차가

    $$
    \frac{\operatorname{SD}(S)}{\sigma} \approx \frac{1}{\sqrt{2(n-1)}}
    $$

    이다. $\square$

    | $n$ | 5 | 30 | 100 |
    |---|---|---|---|
    | 상대 SE | 35.4% | 13.1% | 7.1% |

    **$n=5$면 표준오차의 추정값 자체가 35%나 흔들린다.** "표준오차 = 0.13"이라고 소수점 둘째 자리까지 적는 것이 얼마나 허상인지 보여 준다.

    이 여분의 불확실성을 보정하는 것이 바로 $t$ 분포다. $z$ 대신 $t_{n-1}$을 쓰면 분모가 흔들리는 만큼 임계값을 키워 준다. **$t$ 분포의 꼬리가 두꺼운 이유가 곧 이 연습문제의 답이다.**

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
변동계수 $\hat c = S/\bar X$의 표준오차를 델타 방법으로 구하라(정규모집단 가정). 왜 이 통계량은 $\mu$가 0에 가까울 때 쓸 수 없는가?

</div>

??? success "풀이"
    이변량 델타 방법을 쓴다. $g(\mu,\sigma) = \sigma/\mu$에 대해

    $$
    \frac{\partial g}{\partial\mu} = -\frac{\sigma}{\mu^2}, \qquad \frac{\partial g}{\partial\sigma} = \frac1\mu
    $$

    이고 정규모집단에서 $\bar X$와 $S$가 **독립**이므로 공분산 항이 없다. 연습문제 7의 $\operatorname{Var}(S) \approx \sigma^2/\{2(n-1)\}$과 $\operatorname{Var}(\bar X) = \sigma^2/n$을 넣으면

    $$
    \operatorname{Var}(\hat c) \approx \frac{\sigma^2}{\mu^4}\cdot\frac{\sigma^2}{n} + \frac{1}{\mu^2}\cdot\frac{\sigma^2}{2(n-1)} = \frac{c^2}{n}\left\{c^2 + \frac{n}{2(n-1)}\right\}
    $$

    이다($c = \sigma/\mu$). $n$이 크면

    $$
    \operatorname{SE}(\hat c) \approx \frac{c}{\sqrt n}\sqrt{c^2+\frac12}
    $$

    로 간단해진다.

    **$\mu \to 0$일 때.** 위 식에서 $c = \sigma/\mu \to \infty$이므로 표준오차가 발산한다. 더 근본적으로, $\bar X$가 0을 지날 수 있으면 $\hat c$가 무한대로 튀거나 부호가 뒤집힌다. 델타 방법은 $g$를 $\mu$ 근처에서 선형근사하는 것인데, $\mu$가 0 근처면 $1/\mu$의 곡률이 폭발해 근사 자체가 무너진다.

    **실무 규칙.** 변동계수는 **비율 척도**(참 영점이 있고 값이 모두 양수인 척도)에서만 뜻이 있다. 무게, 시간, 농도, 소득에는 쓸 수 있지만 섭씨온도나 표준점수에는 쓸 수 없다. 섭씨를 화씨로 바꾸면 변동계수가 완전히 달라진다는 점을 생각해 보면 명확하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$\bar X$의 표준오차를 $\sigma$나 정규성 가정 없이 자료만으로 추정하는 방법을 적고, 예제 1의 모의실험 결과와 견주어라.

</div>

??? success "풀이"
    **부트스트랩**을 쓴다.

    ```python
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 5)              # 우리가 가진 표본은 이것뿐
    boot = [rng.choice(x, size=5, replace=True).mean() for _ in range(10_000)]
    print(f"부트스트랩 SE = {np.std(boot, ddof=1):.4f}")
    ```

    **원리.** 참 모집단에서 반복 표집하는 대신 **관측된 표본 자체를 모집단으로 삼아** 복원추출한다. 표본이 모집단을 잘 대신한다면 재표집으로 얻은 평균들의 흩어짐이 참 표집분포의 흩어짐에 가깝다.

    **예제 1과의 관계.** 예제 1의 모의실험은 참 모집단 $\text{Uniform}(0,1)$에서 1만 번 표집했다. 그것이 가능한 것은 모집단을 알기 때문이며, **현실에서는 불가능하다.** 부트스트랩은 같은 일을 표본 하나만 가지고 흉내 낸다.

    $n=5$에서는 결과가 그리 좋지 않다. 부트스트랩 SE가 이론값 0.1291보다 체계적으로 **작게** 나오는데, 복원추출이 $\sigma^2$이 아니라 $\frac{n-1}{n}\sigma^2$을 추정하기 때문이다($\sqrt{4/5} = 0.894$배). 이런 편향은 $n$이 크면 사라진다.

    **부트스트랩을 쓸 이유.** 평균이라면 $s/\sqrt n$이라는 공식이 있으니 굳이 필요 없다. 부트스트랩이 빛나는 것은 **공식이 없는 통계량**이다. 중앙값, 절사평균, 사분위수 범위, 두 추정값의 비, 상관계수, 회귀에서 파생된 복잡한 양의 표준오차를 모두 같은 절차로 얻을 수 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
표본평균이 아닌 다른 중심 추정량의 표준오차를 견주어 보자. 정규모집단에서 표본중앙값의 점근 표준오차가 $\sigma\sqrt{\pi/(2n)}$임을 보이고, 어느 쪽을 언제 써야 하는지 정리하라.

</div>

??? success "풀이"
    표본중앙값 $\tilde X$의 점근분포는

    $$
    \sqrt n(\tilde X - m) \xrightarrow{d} N\!\left(0,\ \frac{1}{4f(m)^2}\right)
    $$

    이다($m$은 모중앙값, $f$는 밀도). 정규분포에서는 $m=\mu$이고

    $$
    f(\mu) = \frac{1}{\sigma\sqrt{2\pi}}
    $$

    이므로

    $$
    \operatorname{Var}(\tilde X) \approx \frac{1}{4n}\cdot 2\pi\sigma^2 = \frac{\pi\sigma^2}{2n}, \qquad \operatorname{SE}(\tilde X) \approx \sigma\sqrt{\frac{\pi}{2n}}
    $$

    이다. $\square$

    **비교.** $\operatorname{SE}(\bar X) = \sigma/\sqrt n$이므로

    $$
    \frac{\operatorname{Var}(\bar X)}{\operatorname{Var}(\tilde X)} = \frac{2}{\pi} \approx 0.637
    $$

    로, 중앙값의 분산이 평균의 1.57배다. 정규모집단에서 중앙값을 쓰면 **표본의 36%를 버리는 셈**이다. 표본 100개로 중앙값을 내는 것이 표본 64개로 평균을 내는 것과 같다.

    **그래도 중앙값을 쓸 때.**

    | 상황 | 권장 |
    |---|---|
    | 모집단이 정규에 가깝고 이상치가 없다 | $\bar X$ |
    | 이상치나 오염이 의심된다 | $\tilde X$ 또는 절사평균 |
    | 꼬리가 두껍다($t_3$, 코시 등) | $\tilde X$가 **더 효율적**이다 |
    | 분포가 치우쳐 "전형적인 값"을 말하고 싶다 | $\tilde X$(다른 양을 추정하는 것임에 유의) |

    꼬리가 두꺼우면 순서가 뒤집힌다는 점이 중요하다. $t_3$ 모집단에서는 중앙값의 점근효율이 평균의 1.6배이고, 코시에서는 평균이 아예 수렴하지 않는 반면 중앙값은 제대로 작동한다.

    타협안으로 **절사평균**이 있다. 양끝 10~20%를 버리고 평균을 내면 정규분포에서 효율을 거의 잃지 않으면서(95% 이상) 이상치에 대한 저항성을 얻는다. 후버 M-추정량도 같은 절충이다.

---

## 정리하며

$\mathrm{SE}(\bar X)=\sigma/\sqrt n$ 은 이 책에서 가장 자주 쓰이는 공식이다.

- **유도는 두 줄이다.** $\mathrm{Var}(\bar X)=\frac1{n^2}\sum\mathrm{Var}(X_i)=\frac{\sigma^2}{n}$ 이고 제곱근을 취한다. **독립성이 필요한 곳은 가운데 등식 하나뿐이다.**
- **$\sigma$ 를 모르면 $s$ 로 바꿔 $\widehat{\mathrm{SE}}=s/\sqrt n$ 을 쓴다.** 그 대체가 $t$ 분포를 부르며, 소표본에서는 대가가 작지 않다(3장 슬러츠키 문서).
- **$\sqrt n$ 이라는 속도가 실무를 지배한다.** 오차를 절반으로 줄이려면 표본을 **네 배**, 십분의 일로 줄이려면 **백 배** 늘려야 한다. 표본크기 계산이 전부 이 관계에서 나온다.
- **줄어드는 것은 추정값의 흔들림이지 자료의 흩어짐이 아니다.** $\sigma$ 는 모집단의 성질이라 $n$ 과 무관하다.

**독립이 아니면 이 공식이 무너진다.** 3장에서 보았듯 AR(1) 자료에서는 유효표본크기가 $n(1-\phi)/(1+\phi)$ 로 줄어, $\phi=0.9$ 면 관측 만 개가 독립 오백 개 값어치밖에 안 된다.

다음 쪽부터는 네 모집단(균등·지수·정규·베르누이)에서 $\bar X$ 의 표본분포를 직접 모의실험한다. 중심극한정리가 모집단 모양을 가리지 않는다는 것과, 치우칠수록 수렴이 느리다는 것을 눈으로 확인한다.
