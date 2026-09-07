# 분산에 대한 Gaussian MLE의 편향

## 이 예가 중요한 이유

최대가능도추정은 바람직한 점근적 성질을 많이 갖는다 — 일치성, 정규성, 효율성. 그러나 유한표본에서 MLE는 편향될 수 있다. 정규분포의 분산추정량이 고전적인 예이다: $n-1$ 대신 $n$으로 나누면 참 분산을 체계적으로 과소추정한다. 이 예는 두 가지 중요한 교훈을 준다 — MLE 관점의 최적성이 불편성을 보장하지는 않는다는 것, 그리고 편향과 분산의 맞바꿈이 추정이론에서 되풀이되는 주제라는 것이다.

## 정규분포 분산의 MLE

$\mu$와 $\sigma^2$이 모두 미지인 $X_1, \ldots, X_n \overset{\text{iid}}{\sim} N(\mu, \sigma^2)$을 생각하자. 로그가능도를 $\sigma^2$에 대해 최대화하면 MLE를 얻는다:

$$
\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

표본평균으로부터의 평균 제곱편차이다. 직관적으로는 자연스러운 추정량으로 보이지만, 알고 보면 아래로 편향되어 있다.

## 편향의 유도

$E[\hat{\sigma}^2_{\text{MLE}}]$을 계산하기 위해 표준적인 대수적 분해를 쓴다. 모평균 $\mu$를 더하고 빼는 데서 출발한다:

$$
\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n \left[(X_i - \mu) - (\bar{X} - \mu)\right]^2
$$

제곱을 전개하면:

$$
= \sum_{i=1}^n (X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n (X_i - \mu) + n(\bar{X} - \mu)^2
$$

$\sum_{i=1}^n (X_i - \mu) = n(\bar{X} - \mu)$이므로 가운데 항은 $-2n(\bar{X} - \mu)^2$과 같고, 따라서

$$
\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2
$$

이제 양변에 기댓값을 취한다. 첫 항에서는 각 $(X_i - \mu)^2$의 기댓값이 $\sigma^2$이므로 합의 기댓값은 $n\sigma^2$이다. 둘째 항에서는 $\bar{X} \sim N(\mu, \sigma^2/n)$이므로 $E[(\bar{X} - \mu)^2] = \sigma^2/n$이고 $E[n(\bar{X} - \mu)^2] = \sigma^2$이다. 따라서:

$$
E\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - \sigma^2 = (n-1)\sigma^2
$$

$n$으로 나누면:

$$
E[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2
$$

MLE의 편향은

$$
\text{Bias}(\hat{\sigma}^2_{\text{MLE}}) = E[\hat{\sigma}^2_{\text{MLE}}] - \sigma^2 = -\frac{\sigma^2}{n}
$$

MLE는 참 분산을 체계적으로 과소추정한다. 편향은 음수이고 크기가 $\sigma^2/n$으로 표본크기가 커지면 줄어든다. 그러나 편향이 사라진다는 것만으로 일치성이 보장되지는 않는다 — 추정량의 분산도 사라져야 하는데, 실제로 ($1/n$의 속도로) 사라지므로 MLE는 일치한다.

## Bessel 수정

위 유도는 편향이 정확히 어디서 오는지 드러낸다: 참 평균 $\mu$ 대신 $\bar{X}$를 쓰는 데 자유도 하나가 든다. 자료로부터 $\mu$를 추정한다는 것은 편차 $X_i - \bar{X}$가 제약 $\sum_{i=1}^n (X_i - \bar{X}) = 0$을 만족한다는 뜻이므로, 자유로운 것은 $n - 1$개뿐이다.

**Bessel 수정**은 $n$ 대신 $n - 1$로 나누어 불편추정량을 만든다:

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2
$$

위 계산에 의해 $E[S^2] = \sigma^2$이 정확히 성립한다. 대부분의 통계 소프트웨어가 쓰는 표본분산이다.

## 평균제곱오차의 비교

불편성만이 좋은 추정량의 기준은 아니다. **평균제곱오차(MSE)**는 편향과 분산의 균형을 잡는다:

$$
\text{MSE}(\hat{\sigma}^2) = \text{Bias}^2(\hat{\sigma}^2) + \text{Var}(\hat{\sigma}^2)
$$

정규분포에서는 $\hat{\sigma}^2_c = \frac{1}{c}\sum_{i=1}^n (X_i - \bar{X})^2$ 형태의 추정량에 대한 평균제곱오차를 닫힌 형태로 계산할 수 있다.

**불편추정량** ($c = n - 1$):

$$
\text{MSE}(S^2) = \frac{2\sigma^4}{n-1}
$$

**MLE** ($c = n$):

$$
\text{MSE}(\hat{\sigma}^2_{\text{MLE}}) = \frac{2n - 1}{n^2}\,\sigma^4
$$

**평균제곱오차 최적 추정량** ($c = n + 1$):

$$
\text{MSE}\!\left(\frac{1}{n+1}\sum(X_i - \bar{X})^2\right) = \frac{2\sigma^4}{n+1}
$$

!!! example "n = 10에서의 수치 비교"

    $n = 10$, $\sigma^2 = 1$일 때:

    | 추정량 | 나누는 수 | 편향 | 평균제곱오차 |
    |---|---|---|---|
    | $S^2$ | $n - 1 = 9$ | 0 | $2/9 \approx 0.222$ |
    | $\hat{\sigma}^2_{\text{MLE}}$ | $n = 10$ | $-0.1$ | $19/100 = 0.190$ |
    | 평균제곱오차 최적 | $n + 1 = 11$ | $-2/11 \approx -0.182$ | $2/11 \approx 0.182$ |

    MLE는 편향되어 있음에도 $S^2$보다 평균제곱오차가 작다. 평균제곱오차 최적 추정량($n + 1$로 나누기)은 편향을 더 받아들이는 대신 분산을 크게 낮추어 더 좋은 성적을 낸다.

!!! tip "편향–분산 맞바꿈"

    이 예는 일반적인 원리를 보여준다: 분산이 충분히 줄어든다면 약간의 편향은 받아들일 만하다. 평균제곱오차 최적 추정량은 MLE도 불편추정량도 아니며, 둘 사이의 최선의 균형점이다.

## 연습문제

**연습문제 1.**
정규분포 분산의 MLE $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n(X_i - \bar{X})^2$이 편향되어 있음을 보여라. 정확한 편향을 계산하라.

??? success "풀이"
    $\sum_{i=1}^n(X_i - \bar{X})^2 / \sigma^2 \sim \chi^2_{n-1}$이고 그 평균은 $n - 1$이다. 따라서:

    $$
    E\!\left[\sum_{i=1}^n(X_i - \bar{X})^2\right] = (n-1)\sigma^2
    $$

    $$
    E[\hat{\sigma}^2_{\text{MLE}}] = \frac{1}{n}(n-1)\sigma^2 = \frac{n-1}{n}\sigma^2
    $$

    편향은:

    $$
    \text{Bias} = E[\hat{\sigma}^2_{\text{MLE}}] - \sigma^2 = -\frac{\sigma^2}{n}
    $$

    MLE는 평균적으로 $\sigma^2$을 과소추정한다. 편향은 $n \to \infty$일 때 사라지지만 작은 $n$에서는 눈에 띈다.

---

**연습문제 2.**
불편추정량 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$은 편향을 바로잡는다. $n \geq 2$에서 $\text{MSE}(\hat{\sigma}^2_{\text{MLE}}) < \text{MSE}(S^2)$임을 보여, 편향된 MLE의 평균제곱오차가 실제로 더 작음을 확인하라.

??? success "풀이"
    임의의 추정량 $\hat{\theta}$에 대해 $\text{MSE} = \text{Bias}^2 + \text{Var}$이다.

    $\sum(X_i - \bar{X})^2/\sigma^2 \sim \chi^2_{n-1}$의 분산이 $2(n-1)$이므로:

    $$
    \text{Var}(S^2) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1}
    $$

    $$
    \text{MSE}(S^2) = 0 + \frac{2\sigma^4}{n-1}
    $$

    MLE의 경우:

    $$
    \text{Var}(\hat{\sigma}^2_{\text{MLE}}) = \frac{\sigma^4}{n^2} \cdot 2(n-1) = \frac{2(n-1)\sigma^4}{n^2}
    $$

    $$
    \text{MSE}(\hat{\sigma}^2_{\text{MLE}}) = \frac{\sigma^4}{n^2} + \frac{2(n-1)\sigma^4}{n^2} = \frac{(2n-1)\sigma^4}{n^2}
    $$

    비교하면 $\frac{2n-1}{n^2} < \frac{2}{n-1}$은 $(2n-1)(n-1) < 2n^2$, 즉 $2n^2 - 3n + 1 < 2n^2$, 즉 $-3n + 1 < 0$으로 정리되며 모든 $n \geq 1$에서 성립한다. $\square$

---

**연습문제 3.**
$n = 5$, $\sigma^2 = 10$에 대해 $\hat{\sigma}^2_{\text{MLE}}$과 $S^2$의 편향, 분산, 평균제곱오차를 계산하라.

??? success "풀이"
    $n = 5$, $\sigma^2 = 10$에서 $\hat{\sigma}^2_{\text{MLE}}$:

    - 편향: $-10/5 = -2.0$
    - 분산: $2(4)(100)/25 = 32.0$
    - 평균제곱오차: $4 + 32 = 36.0$

    $S^2$의 경우:

    - 편향: $0$
    - 분산: $2(100)/4 = 50.0$
    - 평균제곱오차: $0 + 50 = 50.0$

    MLE의 평균제곱오차는 36으로, 불편추정량 $S^2$의 50보다 28% 작다. 여기서는 편향–분산 맞바꿈이 편향추정량에 유리하다: 분산 감소(50에서 32로)가 도입된 편향을 충분히 보상한다.

---

**연습문제 4.**
정규분포에서 추정량 $\hat{\sigma}^2_c = \frac{1}{n+1}\sum(X_i - \bar{X})^2$은 $\hat{\sigma}^2_{\text{MLE}}$보다 평균제곱오차가 더 작다. 그 평균제곱오차를 계산하여 MLE와 비교해 확인하라.

??? success "풀이"
    일반적인 추정량 $\hat{\sigma}^2_c = \frac{1}{c}\sum(X_i - \bar{X})^2$에서 $c = n + 1$이면:

    $$
    E[\hat{\sigma}^2_c] = \frac{n-1}{n+1}\sigma^2, \quad \text{Bias} = \frac{n-1}{n+1}\sigma^2 - \sigma^2 = -\frac{2\sigma^2}{n+1}
    $$

    $$
    \text{Var}(\hat{\sigma}^2_c) = \frac{2(n-1)\sigma^4}{(n+1)^2}
    $$

    $$
    \text{MSE}(\hat{\sigma}^2_c) = \frac{4\sigma^4}{(n+1)^2} + \frac{2(n-1)\sigma^4}{(n+1)^2} = \frac{(2n+2)\sigma^4}{(n+1)^2} = \frac{2\sigma^4}{n+1}
    $$

    MLE의 평균제곱오차 $(2n-1)\sigma^4/n^2$과 비교하면, $n = 5$일 때 MLE는 $9 \times 100/25 = 36$을 주는 반면 $c = 6$은 $200/6 \approx 33.3$을 준다. (정규 자료에서) 평균제곱오차 최적 분모는 실제로 $c = n + 1$이며, 이때 평균제곱오차가 $2\sigma^4/(n+1)$로 최소가 된다.
