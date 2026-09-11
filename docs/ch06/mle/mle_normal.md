# Normal 분포의 MLE

## 개요

$x^{(i)}$를 $N(\mu, \sigma^2)$에서 얻은 $m$개의 i.i.d. 표본이라 하자. 그러면 $\mu$와 $\sigma^2$은 다음 $\hat{\mu}$와 $\hat{\sigma}^2$으로 추정할 수 있다:

$$
\begin{array}{lll}
\hat{\mu} &=& \displaystyle\frac{\sum_{i=1}^m x^{(i)}}{m} \\[12pt]
\hat{\sigma}^2 &=& \displaystyle\frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m}
\end{array}
$$

## 유도

### 자료

$$
\{x^{(i)} : i = 1, \ldots, m\}
$$

### 모형

$$
x^{(i)} \sim N(\mu, \sigma^2)
$$

### 가능도함수

$$
L(\mu, \sigma^2) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{1}{2\sigma^2}(x^{(i)} - \mu)^2\right)
$$

### 로그가능도함수

$$
\ell(\mu, \sigma^2) = -\frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 - \frac{m}{2}\log\sigma^2 + \text{Constant}
$$

### 비용함수

$$
J(\mu, \sigma^2) = \frac{1}{2\sigma^2}\sum_{i=1}^m (x^{(i)} - \mu)^2 + \frac{m}{2}\log\sigma^2
$$

### 최대가능도 원리

$$
\text{argmax}_{\mu, \sigma^2}\; L
\quad\Leftrightarrow\quad
\text{argmax}_{\mu, \sigma^2}\; \ell
\quad\Leftrightarrow\quad
\text{argmin}_{\mu, \sigma^2}\; J
$$

### MLE 해

$$
\begin{array}{llcll}
\displaystyle\frac{\partial J}{\partial \mu} = 0
&\Rightarrow&
\displaystyle\sum_{i=1}^m (x^{(i)} - \mu) = 0
&\Rightarrow&
\displaystyle\hat{\mu} = \frac{\sum_{i=1}^m x^{(i)}}{m} \\[16pt]
\displaystyle\frac{\partial J}{\partial \sigma^2} = 0
&\Rightarrow&
\cdots
&\Rightarrow&
\displaystyle\hat{\sigma}^2 = \frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m}
\end{array}
$$

## 핵심 관찰

| 추정량 | MLE | 불편인가? |
|-----------|-----|-----------|
| $\hat{\mu}$ | $\frac{1}{m}\sum x^{(i)}$ | ✅ 그렇다 |
| $\hat{\sigma}^2$ | $\frac{1}{m}\sum (x^{(i)} - \hat{\mu})^2$ | ❌ 아니다 ($m-1$이 아니라 $m$으로 나눈다) |

!!! note "분산 MLE의 편향"
    MLE $\hat{\sigma}^2$은 $m$으로 나누므로 $\sigma^2$의 편향추정량이 된다. 불편 표본분산 $S^2$은 $m - 1$로 나눈다(Bessel 수정):

    $$
    S^2 = \frac{\sum_{i=1}^m (x^{(i)} - \hat{\mu})^2}{m - 1}
    $$

## 최소제곱과의 연결

($\sigma^2$을 고정한) $\mu$에 대한 비용함수는:

$$
J(\mu) \propto \sum_{i=1}^m (x^{(i)} - \mu)^2
$$

이는 정확히 **최소제곱** 목적함수이다. 따라서 정규분포 평균의 MLE는 최소제곱 추정값과 동등하며, 이는 MLE와 회귀분석 사이의 깊은 연결이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\sigma^2$이 알려진 표본 $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$에서 로그가능도를 $\mu$에 대해 미분하여 $\mu$의 MLE를 유도하라.

</div>

??? success "풀이"
    로그가능도는:

    $$
    \ell(\mu) = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
    $$

    $\mu$에 대해 미분하면:

    $$
    \frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = \frac{1}{\sigma^2}\left(\sum x_i - n\mu\right)
    $$

    0으로 두면:

    $$
    \sum x_i - n\mu = 0 \implies \hat{\mu} = \frac{\sum x_i}{n} = \bar{x}
    $$

    2계도함수가 $-n/\sigma^2 < 0$이므로 최댓값임이 확인된다.

<div class="drillbox" markdown>

**연습문제 2.**
MLE $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$이 편향되어 있음을 보이고 $E[\hat{\sigma}^2]$을 계산하라.

</div>

??? success "풀이"
    $S^2$을 $E[S^2] = \sigma^2$인 불편 표본분산이라 하면 $\sum(X_i - \bar{X})^2 = (n-1)S^2$이다. 따라서:

    $$
    E[\hat{\sigma}^2] = E\!\left[\frac{1}{n}\sum(X_i - \bar{X})^2\right] = \frac{1}{n}E\bigl[(n-1)S^2\bigr] = \frac{n-1}{n}\sigma^2
    $$

    편향은 $E[\hat{\sigma}^2] - \sigma^2 = -\sigma^2/n$이다. MLE는 참 분산을 과소추정한다. 이 때문에 불편추정량으로 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$을 쓰게 된다.

<div class="drillbox" markdown>

**연습문제 3.**
정규 로그가능도를 $\mu$에 대해 최대화하는 것과 잔차제곱합을 최소화하는 것 사이의 연결을 설명하라. 정규가 아닌 분포에서는 이 연결이 왜 깨지는가?

</div>

??? success "풀이"
    정규분포에서 ($\sigma^2$을 고정한) $\mu$에 대한 로그가능도는:

    $$
    \ell(\mu) = \text{const} - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
    $$

    상수와 인수 $-1/(2\sigma^2)$은 argmax를 바꾸지 않으므로 $\ell(\mu)$를 최대화하는 것은 $\sum(x_i - \mu)^2$을 최소화하는 것과 동등하다. 이것이 최소제곱 목적함수이다.

    정규가 아닌 분포에서는 로그가능도가 $(x_i - \mu)$의 다른 함수를 포함한다. 예를 들어 Laplace 분포에서는 $\ell(\mu) \propto -\sum |x_i - \mu|$이므로 MLE가 절대편차의 합을 최소화하며 평균이 아니라 중앙값을 준다. 최소제곱과의 연결은 정규분포의 이차 지수부에 특유한 것이다.

<div class="drillbox" markdown>

**연습문제 4.**
정규분포에서 얻은 관측값 5개가 3, 5, 7, 9, 11이다. $\mu$와 $\sigma^2$의 MLE를 모두 계산하라.

</div>

??? success "풀이"
    $\mu$의 MLE는 표본평균이다:

    $$
    \hat{\mu} = \frac{3+5+7+9+11}{5} = \frac{35}{5} = 7
    $$

    $\sigma^2$의 MLE는 ($n-1$이 아니라) $n$으로 나눈다:

    $$
    \hat{\sigma}^2 = \frac{1}{5}\sum(x_i - 7)^2 = \frac{(3-7)^2 + (5-7)^2 + (7-7)^2 + (9-7)^2 + (11-7)^2}{5}
    $$

    $$
    = \frac{16 + 4 + 0 + 4 + 16}{5} = \frac{40}{5} = 8
    $$

    참고: 불편추정값은 $S^2 = 40/4 = 10$이다.
