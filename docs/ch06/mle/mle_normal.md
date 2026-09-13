# 정규분포의 MLE

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

**연습문제 1.** <span class="diff med" title="중간"></span>
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

**연습문제 2.** <span class="diff med" title="중간"></span>
MLE $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i - \bar{x})^2$이 편향되어 있음을 보이고 $E[\hat{\sigma}^2]$을 계산하라.

</div>

??? success "풀이"
    $S^2$을 $E[S^2] = \sigma^2$인 불편 표본분산이라 하면 $\sum(X_i - \bar{X})^2 = (n-1)S^2$이다. 따라서:

    $$
    E[\hat{\sigma}^2] = E\!\left[\frac{1}{n}\sum(X_i - \bar{X})^2\right] = \frac{1}{n}E\bigl[(n-1)S^2\bigr] = \frac{n-1}{n}\sigma^2
    $$

    편향은 $E[\hat{\sigma}^2] - \sigma^2 = -\sigma^2/n$이다. MLE는 참 분산을 과소추정한다. 이 때문에 불편추정량으로 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$을 쓰게 된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
정규 로그가능도를 $\mu$에 대해 최대화하는 것과 잔차제곱합을 최소화하는 것 사이의 연결을 설명하라. 정규가 아닌 분포에서는 이 연결이 왜 깨지는가?

</div>

??? success "풀이"
    정규분포에서 ($\sigma^2$을 고정한) $\mu$에 대한 로그가능도는:

    $$
    \ell(\mu) = \text{const} - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
    $$

    상수와 인수 $-1/(2\sigma^2)$은 argmax를 바꾸지 않으므로 $\ell(\mu)$를 최대화하는 것은 $\sum(x_i - \mu)^2$을 최소화하는 것과 동등하다. 이것이 최소제곱 목적함수이다.

    정규가 아닌 분포에서는 로그가능도가 $(x_i - \mu)$의 다른 함수를 포함한다. 예를 들어 라플라스분포에서는 $\ell(\mu) \propto -\sum |x_i - \mu|$이므로 MLE가 절대편차의 합을 최소화하며 평균이 아니라 중앙값을 준다. 최소제곱과의 연결은 정규분포의 이차 지수부에 특유한 것이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
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

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
정규 로그가능도의 **피셔 정보행렬**을 $(\mu,\sigma^2)$에 대해 구하라. 대각행렬이 나오는데, 이것이 무엇을 뜻하는가?

</div>

??? success "풀이"
    관측값 하나의 로그밀도는

    $$
    \ln f = -\frac12\ln(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}
    $$

    이다. 2계 편도함수를 구하면

    $$
    \frac{\partial^2\ln f}{\partial\mu^2} = -\frac{1}{\sigma^2}, \qquad
    \frac{\partial^2\ln f}{\partial\mu\,\partial\sigma^2} = -\frac{x-\mu}{\sigma^4}, \qquad
    \frac{\partial^2\ln f}{\partial(\sigma^2)^2} = \frac{1}{2\sigma^4}-\frac{(x-\mu)^2}{\sigma^6}
    $$

    이고 기대값을 취하면($E[X-\mu]=0$, $E[(X-\mu)^2]=\sigma^2$)

    $$
    I_1(\mu,\sigma^2) = \begin{pmatrix} 1/\sigma^2 & 0 \\ 0 & 1/(2\sigma^4)\end{pmatrix}
    $$

    를 얻는다. 비대각 성분이 0인 것은 $E[X-\mu] = 0$ 때문이다.

    **뜻.** 정보행렬이 대각이면 두 모수가 **직교**한다고 말하며, 다음을 함의한다.

    - **MLE가 점근적으로 독립이다.** 점근공분산이 $I^{-1}/n$이고 이것도 대각이므로 $\hat\mu$와 $\hat\sigma^2$의 점근상관이 0이다(정규분포에서는 유한표본에서도 정확히 독립이다).
    - **한 모수에 대한 추론이 다른 모수의 추정오차에 영향받지 않는다.** $\sigma^2$을 모르는 것이 $\hat\mu$의 점근분산을 키우지 않는다. 실제로 $\operatorname{Var}(\hat\mu) = \sigma^2/n$으로, $\sigma^2$을 알든 모르든 같다.
    - **프로파일 가능도가 다루기 쉽다.** $\mu$를 프로파일링할 때 $\sigma^2$을 최적화해 없애도 곡률이 크게 바뀌지 않는다.

    직교하지 않는 예로 감마분포의 $(k,\theta)$가 있다. 그 경우 형상모수를 모른다는 사실이 척도모수 추정의 정밀도를 떨어뜨린다. **모수화를 직교하도록 고르면 추론이 단순해진다**는 것이 일반적인 원리이며, 감마분포에서 $(k, k\theta)$로 다시 매개하면 직교에 가까워진다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$\sigma^2$을 모를 때 $\mu$의 프로파일 로그가능도를 구하고, 그로부터 우도비 신뢰구간이 $t$ 구간과 어떻게 이어지는지 보여라.

</div>

??? success "풀이"
    **프로파일링.** 고정된 $\mu$에 대해 $\sigma^2$을 최적화한다. $\ell(\mu,\sigma^2) = -\frac n2\ln\sigma^2 - \frac{1}{2\sigma^2}\sum(x_i-\mu)^2 + \text{상수}$이므로

    $$
    \hat\sigma^2(\mu) = \frac1n\sum_i(x_i-\mu)^2
    $$

    이고, 이를 대입하면

    $$
    \ell_p(\mu) = -\frac n2\ln\left\{\frac1n\sum_i(x_i-\mu)^2\right\} - \frac n2 + \text{상수}
    $$

    이다.

    **우도비 통계량.** $\sum(x_i-\mu)^2 = \sum(x_i-\bar x)^2 + n(\bar x-\mu)^2 = (n-1)s^2 + n(\bar x-\mu)^2$이므로

    $$
    \Lambda(\mu) = 2\{\ell_p(\hat\mu)-\ell_p(\mu)\} = n\ln\left\{1+\frac{n(\bar x-\mu)^2}{(n-1)s^2}\right\} = n\ln\left(1+\frac{t^2}{n-1}\right)
    $$

    이다. 여기서 $t = \sqrt n(\bar x-\mu)/s$가 바로 $t$ 통계량이다.

    **이어짐.** $\Lambda$가 $t^2$의 **순증가함수**이므로, $\Lambda \le c$인 집합과 $|t| \le c'$인 집합이 정확히 같다. 즉

    $$
    \text{우도비 구간} = \left\{\mu:\ |t(\mu)| \le c'\right\}
    $$

    로 **$t$ 구간과 같은 모양**이다. 차이는 임계값을 어디서 가져오느냐뿐이다.

    - 우도비 구간은 $\Lambda \le \chi^2_{1,0.95} = 3.841$을 쓴다(점근 근사).
    - $t$ 구간은 $|t| \le t_{0.975,n-1}$을 쓴다(정확).

    $n$이 크면 $n\ln(1+t^2/n) \approx t^2$이고 $t_{0.975,n-1}\to1.96$이므로 두 임계값이 일치한다. $n$이 작으면 $t$ 쪽이 정확하다.

    **일반 교훈.** 정규모형에서 우도비 방법은 익숙한 $t$ 절차를 **재발견**한다. 우도비가 특별히 새로운 답을 주는 것이 아니라, 정확한 해가 없는 일반 모형으로 같은 발상을 확장하는 도구라는 점이 요점이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
선형회귀 $y_i = \mathbf{x}_i^\top\boldsymbol\beta + \varepsilon_i$, $\varepsilon_i \sim N(0,\sigma^2)$에서 $\boldsymbol\beta$와 $\sigma^2$의 MLE를 구하라. $\hat{\boldsymbol\beta}_{\text{MLE}}$가 최소제곱 추정량과 같은데 $\hat\sigma^2$은 왜 보통 쓰는 값과 다른가?

</div>

??? success "풀이"
    **로그가능도.**

    $$
    \ell(\boldsymbol\beta,\sigma^2) = -\frac n2\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i(y_i-\mathbf{x}_i^\top\boldsymbol\beta)^2
    $$

    **$\boldsymbol\beta$의 MLE.** $\sigma^2 > 0$이 곱해진 상수이므로, $\ell$을 최대화하는 것은 잔차제곱합을 **최소화**하는 것과 같다. 따라서

    $$
    \hat{\boldsymbol\beta}_{\text{MLE}} = \arg\min\sum_i(y_i-\mathbf{x}_i^\top\boldsymbol\beta)^2 = (X^\top X)^{-1}X^\top\mathbf{y} = \hat{\boldsymbol\beta}_{\text{OLS}}
    $$

    **정규 오차 가정 아래에서 최소제곱이 곧 최대가능도**다. 최소제곱이 그토록 자연스러워 보이는 이유가 여기 있다.

    **$\sigma^2$의 MLE.** $\hat{\boldsymbol\beta}$를 넣고 $\sigma^2$으로 미분하면

    $$
    \hat\sigma^2_{\text{MLE}} = \frac{\text{RSS}}{n}
    $$

    **왜 다른가.** 보통 쓰는 값은

    $$
    \hat\sigma^2_{\text{OLS}} = \frac{\text{RSS}}{n-p}
    $$

    로 $p$개의 회귀계수를 추정하느라 잃은 자유도를 반영한다. $E[\text{RSS}] = (n-p)\sigma^2$이므로 이쪽이 **불편**이고 MLE는 아래로 편향되어 있다.

    MLE가 편향되는 이유는 명확하다. 잔차 $\hat\varepsilon_i = y_i - \mathbf{x}_i^\top\hat{\boldsymbol\beta}$는 참 오차 $\varepsilon_i$보다 작다. $\hat{\boldsymbol\beta}$가 바로 그 제곱합을 최소로 하도록 골라졌기 때문이다. 그 "과적합"의 정도가 정확히 $p/n$만큼이고, $n-p$로 나누어 보정한다.

    **실무적 함의.** $p$가 $n$에 비해 작으면 차이가 무시할 만하다. 그러나 $p$가 $n$에 가까우면($p/n = 0.3$ 같은 경우) MLE가 오차분산을 30%나 과소평가한다. 고차원 회귀에서 예측오차를 심하게 낙관적으로 보게 되는 한 원인이다.

    AIC 같은 정보기준은 $\hat\sigma^2_{\text{MLE}}$를 쓰되 모수 개수에 벌점을 주어 같은 문제를 다른 방식으로 다룬다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
표본에 이상치가 하나 섞이면 $\hat\mu_{\text{MLE}} = \bar x$가 얼마나 흔들리는가? 잡음 분포를 $t_\nu$로 바꾸면 무엇이 달라지는지 점수방정식으로 설명하라.

</div>

??? success "풀이"
    **정규 잡음.** 점수방정식은

    $$
    \sum_i (x_i-\mu) = 0 \implies \hat\mu = \bar x
    $$

    이다. 잔차에 정비례하는 가중치를 주므로, 관측값 하나가 $x \to \infty$이면 $\hat\mu$도 함께 무한대로 간다. **붕괴점이 0**이고, 영향함수 $\psi(r) = r$이 유계가 아니다.

    $n=20$, 나머지가 모두 0 근처인 자료에 $x = 100$이 하나 섞이면 $\bar x$가 5만큼 움직인다.

    **$t_\nu$ 잡음.** 음의 로그밀도가 $\rho(r) = \frac{\nu+1}{2}\ln(1+r^2/\nu)$이므로 점수방정식이

    $$
    \sum_i \psi(x_i-\mu) = 0, \qquad \psi(r) = \frac{(\nu+1)r}{\nu+r^2}
    $$

    가 된다. $\psi$가 **유계**이고 $|r|\to\infty$에서 0으로 되돌아간다. 즉 잔차가 아주 큰 관측값은 추정에 거의 기여하지 않는다.

    이를 가중최소제곱 꼴로 다시 쓰면

    $$
    \hat\mu = \frac{\sum_i w_ix_i}{\sum_i w_i}, \qquad w_i = \frac{\nu+1}{\nu+(x_i-\hat\mu)^2}
    $$

    로, **가중치가 잔차의 제곱에 반비례**한다. $\hat\mu$가 양변에 나타나므로 반복해서 풀어야 하고(IRLS), 이것이 $t$ 잡음 모형을 EM으로 적합할 때 자연스럽게 나오는 알고리즘이다.

    **$\nu$의 선택.** $\nu \to \infty$이면 정규로 돌아가 강건성이 사라지고, $\nu$가 작으면 강건하지만 효율을 잃는다. 실무에서는 $\nu = 4$ 정도로 고정하거나 자료에서 함께 추정한다. $\nu$를 추정하면 자료가 "얼마나 두꺼운 꼬리를 요구하는지"를 스스로 말하게 하는 셈이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$\hat\mu$와 $\hat\sigma^2_{\text{MLE}}$의 결합 점근분포를 구하고, 델타 방법으로 $\hat\sigma = \sqrt{\hat\sigma^2}$의 점근분산을 유도하라.

</div>

??? success "풀이"
    **결합 점근분포.** MLE의 일반이론에 따라

    $$
    \sqrt n\begin{pmatrix}\hat\mu-\mu\\ \hat\sigma^2-\sigma^2\end{pmatrix} \xrightarrow{d} N\!\left(\mathbf{0},\ I_1^{-1}\right)
    $$

    이고 연습문제 5에서 $I_1 = \operatorname{diag}(1/\sigma^2,\ 1/(2\sigma^4))$이므로

    $$
    I_1^{-1} = \begin{pmatrix}\sigma^2 & 0\\ 0 & 2\sigma^4\end{pmatrix}
    $$

    이다. 즉

    $$
    \hat\mu \approx N\!\left(\mu, \frac{\sigma^2}{n}\right), \qquad \hat\sigma^2 \approx N\!\left(\sigma^2, \frac{2\sigma^4}{n}\right), \qquad \text{점근독립}
    $$

    이다. $\hat\sigma^2$의 점근분산 $2\sigma^4/n$은 앞서 본 정확한 값 $2\sigma^4/(n-1)$($S^2$의 경우)과 $O(1/n^2)$만큼 다르다.

    **델타 방법.** $g(v) = \sqrt v$이고 $g'(\sigma^2) = 1/(2\sigma)$이므로

    $$
    \operatorname{Var}(\hat\sigma) \approx \left(\frac{1}{2\sigma}\right)^2\cdot\frac{2\sigma^4}{n} = \frac{\sigma^2}{2n}
    $$

    이고

    $$
    \operatorname{SE}(\hat\sigma) \approx \frac{\sigma}{\sqrt{2n}}
    $$

    이다. $\square$

    **읽는 법.** 상대 표준오차가

    $$
    \frac{\operatorname{SE}(\hat\sigma)}{\sigma} = \frac{1}{\sqrt{2n}}
    $$

    로 $\hat\mu$의 상대 정밀도와 비교하면 $\sqrt2$배 나쁘다(변동계수가 1일 때 기준). **산포를 재는 것이 중심을 재는 것보다 어렵다**는 사실이 다시 확인된다.

    한 가지 더. $\ln\hat\sigma$의 점근분산은 델타 방법으로 $1/(2n)$이 되어 **$\sigma$에 전혀 의존하지 않는다.** 그래서 표준편차의 신뢰구간은 로그 척도에서 만들고 되돌리는 것이 자연스럽고, 그렇게 하면 자동으로 양수이면서 비대칭인 구간을 얻는다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
같은 $\mu$를 재는 두 측정기의 정밀도가 다르다. 측정기 1로 $n_1$번, 측정기 2로 $n_2$번 재었고 $\sigma_1, \sigma_2$를 안다. $\mu$의 MLE를 구하고, 그것이 단순히 전체 평균이 아닌 이유를 설명하라.

</div>

??? success "풀이"
    **로그가능도.** 두 집단의 기여를 더한다.

    $$
    \ell(\mu) = -\frac{1}{2\sigma_1^2}\sum_{i=1}^{n_1}(x_i-\mu)^2 - \frac{1}{2\sigma_2^2}\sum_{j=1}^{n_2}(y_j-\mu)^2 + \text{상수}
    $$

    미분해 0으로 두면

    $$
    \frac{n_1(\bar x-\mu)}{\sigma_1^2} + \frac{n_2(\bar y-\mu)}{\sigma_2^2} = 0
    $$

    이고

    $$
    \hat\mu = \frac{\dfrac{n_1}{\sigma_1^2}\bar x + \dfrac{n_2}{\sigma_2^2}\bar y}{\dfrac{n_1}{\sigma_1^2}+\dfrac{n_2}{\sigma_2^2}}
    $$

    를 얻는다. **각 집단 평균을 정밀도 $n_i/\sigma_i^2$으로 가중한 평균**이다.

    **왜 전체 평균이 아닌가.** 전체 평균은

    $$
    \frac{n_1\bar x + n_2\bar y}{n_1+n_2}
    $$

    로 표본크기만 반영하고 정밀도를 무시한다. $\sigma_2$가 $\sigma_1$의 10배라면 측정기 2의 관측값 하나가 측정기 1의 관측값 100분의 1만큼의 정보밖에 없는데, 전체 평균은 둘을 같게 취급한다.

    **분산.**

    $$
    \operatorname{Var}(\hat\mu) = \frac{1}{\dfrac{n_1}{\sigma_1^2}+\dfrac{n_2}{\sigma_2^2}} = \frac{1}{I_{\text{총}}}
    $$

    으로 **정보량의 역수**다. 각 관측값이 기여하는 정보 $1/\sigma_i^2$이 그대로 더해진다.

    **일반화.** 관측값마다 분산이 다르면($\sigma_i^2$) 같은 논리로

    $$
    \hat\mu = \frac{\sum_i x_i/\sigma_i^2}{\sum_i 1/\sigma_i^2}
    $$

    이고, 이것이 **가중최소제곱**의 가장 단순한 경우다. 회귀로 확장하면 $\hat{\boldsymbol\beta} = (X^\top W X)^{-1}X^\top W\mathbf{y}$($W = \operatorname{diag}(1/\sigma_i^2)$)가 되며, 이분산이 있을 때 최소제곱보다 효율적인 추정량을 준다. 메타분석의 역분산 가중도 정확히 이 공식이다.

---

## 정리하며

정규분포에서는 모수가 둘이고, 점수방정식 둘을 함께 푼다.

$$
\hat\mu = \frac{1}{m}\sum_i x^{(i)}, \qquad
\hat\sigma^2 = \frac{1}{m}\sum_i (x^{(i)}-\hat\mu)^2
$$

- **$\hat\mu$ 는 표본평균이고 불편이다.**
- **$\hat\sigma^2$ 은 $m$ 으로 나눈다. 즉 편향되어 있다.** $\mathbb{E}[\hat\sigma^2]=\frac{m-1}{m}\sigma^2$ 이므로 참값을 체계적으로 과소추정한다. 베셀 보정으로 $m-1$ 을 쓰면 불편이 되지만, **그것은 최대가능도추정량이 아니다.**
- **최대가능도는 불편성을 보장하지 않는다.** 이 예가 그 사실을 가장 선명하게 보여 준다. 불변성 때문에 그렇다 — $\sigma$ 의 최대가능도추정량이 $\hat\sigma^2$ 의 제곱근인데, 비선형 변환은 불편성을 보존하지 않는다.
- **편향이 $O(1/m)$ 이라 점근적으로는 사라진다.** 소표본에서만 문제가 되며, $m=10$ 이면 $10\%$ 과소추정이다.
- $(\sum x_i,\sum x_i^2)$ 이 충분통계량이다. 모수가 둘이니 통계량도 둘이며, 지수족의 전형적인 구조다.

다음 절 **포아송분포의 최대가능도**로 넘어간다. 계수 자료의 표준 모형이며, 여기서는 추정량이 불편이면서 동시에 효율적이다.
