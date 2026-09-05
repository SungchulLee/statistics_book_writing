# 표본분포 (단순 최소제곱)

일반적인 다중회귀 이론을 전개하기 전에 **단순선형회귀** — 예측변수가 하나인 모형 — 에서 추정량의 표본분포를 유도해 보면 도움이 된다. 단순한 경우를 직접 따라가 보면 오차의 정규성이 어떻게 추정량의 정규성으로 전파되는지, 자유도가 왜 $n - 2$인지, t-통계량이 어떻게 나오는지에 대한 직관이 생긴다. 단순한 경우의 공식은 또한 각 대수적 양(제곱합, 교차곱)의 역할을 행렬 표기가 가릴 수 있는 방식과 달리 투명하게 드러낸다.

## 단순선형회귀 모형

모형은

$$
Y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \quad i = 1, \dots, n
$$

이며, 여기서

- $x_1, \dots, x_n$은 고정된(비확률적) 예측변수 값이며 모두 같지는 않다.
- $\varepsilon_1, \dots, \varepsilon_n$은 독립인 $N(0, \sigma^2)$ 확률변수다.
- $\beta_0$(절편)과 $\beta_1$(기울기)은 미지의 모수다.
- $\sigma^2$은 미지의 오차분산이다.

$\varepsilon_i \sim N(0, \sigma^2)$이고 $x_i$가 고정되어 있으므로 각 반응은 독립적으로 $Y_i \sim N(\beta_0 + \beta_1 x_i, \sigma^2)$이다.

## 최소제곱추정량

최소제곱추정량은 $\sum_{i=1}^n (Y_i - \beta_0 - \beta_1 x_i)^2$을 최소화한다. 닫힌 형태의 해는 다음과 같다.

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}
$$

$$
\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}
$$

여기서 $\bar{x} = \frac{1}{n}\sum_i x_i$, $\bar{Y} = \frac{1}{n}\sum_i Y_i$, $S_{xx} = \sum_i(x_i - \bar{x})^2$, $S_{xy} = \sum_i(x_i - \bar{x})(Y_i - \bar{Y})$이다.

## 추정량의 선형성

핵심적인 관찰은 두 추정량이 모두 반응 $Y_1, \dots, Y_n$의 **선형함수**라는 점이다.

기울기의 경우, ($\sum_i(x_i - \bar{x})\bar{Y} = 0$이므로) $S_{xy} = \sum_i(x_i - \bar{x})Y_i$이니

$$
\hat{\beta}_1 = \sum_{i=1}^n c_i Y_i, \quad \text{where } c_i = \frac{x_i - \bar{x}}{S_{xx}}
$$

이다. 가중치 $c_i$는 유용한 두 항등식을 만족한다.

$$
\sum_{i=1}^n c_i = 0, \qquad \sum_{i=1}^n c_i^2 = \frac{1}{S_{xx}}
$$

$\hat{\beta}_1$이 독립인 정규확률변수들의 선형결합이므로 그 자체가 정규분포를 따른다.

## 기울기의 표본분포

!!! tip "정리 — 기울기 추정량의 분포"
    정규오차를 갖는 단순선형회귀 모형 아래에서

    $$
    \hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{S_{xx}}\right)
    $$

**증명.**

*평균:*

$$
E[\hat{\beta}_1] = \sum_i c_i E[Y_i] = \sum_i c_i(\beta_0 + \beta_1 x_i) = \beta_0\sum_i c_i + \beta_1\sum_i c_i x_i
$$

$\sum_i c_i = 0$이고 $\sum_i c_i x_i = \sum_i \frac{(x_i - \bar{x})x_i}{S_{xx}} = \frac{S_{xx}}{S_{xx}} = 1$이므로

$$
E[\hat{\beta}_1] = \beta_1
$$

이다. 따라서 $\hat{\beta}_1$은 **불편**이다.

*분산:*

$$
\operatorname{Var}(\hat{\beta}_1) = \sum_i c_i^2 \operatorname{Var}(Y_i) = \sigma^2 \sum_i c_i^2 = \frac{\sigma^2}{S_{xx}}
$$

*정규성:* $\hat{\beta}_1$이 독립인 정규확률변수들의 선형결합이므로 정규분포를 따른다. $\square$

## 절편의 표본분포

!!! tip "정리 — 절편 추정량의 분포"
    같은 모형 아래에서

    $$
    \hat{\beta}_0 \sim N\!\left(\beta_0,\; \sigma^2\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)\right)
    $$

**증명.**

*평균:* $E[\hat{\beta}_0] = E[\bar{Y}] - E[\hat{\beta}_1]\bar{x} = (\beta_0 + \beta_1\bar{x}) - \beta_1\bar{x} = \beta_0$. 불편이다.

*분산:* $\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1\bar{x}$이고 ($\sum_i c_i = 0$이므로) $\operatorname{Cov}(\bar{Y}, \hat{\beta}_1) = 0$이므로

$$
\operatorname{Var}(\hat{\beta}_0) = \operatorname{Var}(\bar{Y}) + \bar{x}^2\operatorname{Var}(\hat{\beta}_1) = \frac{\sigma^2}{n} + \frac{\bar{x}^2\sigma^2}{S_{xx}} = \sigma^2\!\left(\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}\right)
$$

$\square$

## 잔차제곱합과 분산의 추정

잔차제곱합은

$$
\text{SSE} = \sum_{i=1}^n (Y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2
$$

이다.

!!! tip "정리 — SSE의 분포"
    정규 단순선형회귀 모형 아래에서

    $$
    \frac{\text{SSE}}{\sigma^2} \sim \chi^2_{n-2}
    $$

    이고 SSE는 $(\hat{\beta}_0, \hat{\beta}_1)$과 독립이다.

자유도가 $n - 2$인 것은 두 개의 모수($\beta_0$과 $\beta_1$)를 추정했기 때문이다. $\sigma^2$의 불편추정량은

$$
s^2 = \frac{\text{SSE}}{n - 2}
$$

이고 $E[s^2] = \sigma^2$이다.

## t-통계량

실무에서는 $\sigma^2$을 모르므로 표준오차에서 이를 $s^2$으로 대체한다. 그러면 정규분포가 t-분포로 바뀐다.

!!! tip "정리 — 기울기의 t-분포"
    통계량

    $$
    T = \frac{\hat{\beta}_1 - \beta_1}{s / \sqrt{S_{xx}}} \sim t_{n-2}
    $$

    은 자유도 $n - 2$인 스튜던트 t-분포를 따른다.

**증명 개요.** 분자 $(\hat{\beta}_1 - \beta_1)/(\sigma/\sqrt{S_{xx}}) \sim N(0,1)$이고 분모에는 $s/\sigma = \sqrt{\text{SSE}/((n-2)\sigma^2)}$이 들어 있다. $\text{SSE}/\sigma^2 \sim \chi^2_{n-2}$이고 $\hat{\beta}_1$과 독립이므로, 이 비는 $N(0,1)/\sqrt{\chi^2_{n-2}/(n-2)}$ 형태이며 이것이 $t_{n-2}$ 분포의 정의다. $\square$

절편에 대해서도 마찬가지로

$$
\frac{\hat{\beta}_0 - \beta_0}{s\sqrt{1/n + \bar{x}^2/S_{xx}}} \sim t_{n-2}
$$

이다.

## 신뢰구간

t-분포 결과는 곧바로 신뢰구간을 준다.

$\beta_1$에 대한 $100(1 - \alpha)\%$ 신뢰구간은

$$
\hat{\beta}_1 \pm t_{\alpha/2,\,n-2} \cdot \frac{s}{\sqrt{S_{xx}}}
$$

이고, $\beta_0$에 대해서는

$$
\hat{\beta}_0 \pm t_{\alpha/2,\,n-2} \cdot s\sqrt{\frac{1}{n} + \frac{\bar{x}^2}{S_{xx}}}
$$

이다. 여기서 $t_{\alpha/2,\,n-2}$는 $t_{n-2}$ 분포의 상위 $\alpha/2$ 분위수다.

## 예

$n = 5$개의 자료점에서 $\bar{x} = 3$, $S_{xx} = 10$, $\hat{\beta}_1 = 2.5$, $\hat{\beta}_0 = 1.0$, $\text{SSE} = 6.0$이라 하자.

- **추정 분산:** $s^2 = 6.0 / 3 = 2.0$이므로 $s = \sqrt{2} \approx 1.414$.
- **기울기의 표준오차:** $\text{SE}(\hat{\beta}_1) = s/\sqrt{S_{xx}} = \sqrt{2}/\sqrt{10} = \sqrt{0.2} \approx 0.447$.
- **$H_0: \beta_1 = 0$에 대한 t-통계량:** $T = 2.5 / 0.447 \approx 5.59$이며 $t_{3}$과 비교한다.
- **기울기의 95% 신뢰구간:** $t_{0.025, 3} = 3.182$를 쓰면 $2.5 \pm 3.182 \times 0.447 \approx 2.5 \pm 1.42 = (1.08, 3.92)$.

## 요약

정규오차를 갖는 단순선형회귀에서 기울기 추정량 $\hat{\beta}_1$과 절편 추정량 $\hat{\beta}_0$은 참 모수를 평균으로 하는(즉 불편인) 정규분포를 따르며, 그 분산은 $\sigma^2$과 예측변수 값의 퍼짐 $S_{xx}$에 달려 있다. 잔차제곱합 $\text{SSE}/\sigma^2$은 $\chi^2_{n-2}$ 분포를 따르고 추정량들과 독립이다. 표준화된 추정량에서 $\sigma$를 $s = \sqrt{\text{SSE}/(n-2)}$로 바꾸면 자유도 $n - 2$인 t-통계량이 나오며, 이것이 가설검정과 신뢰구간의 토대가 된다.

## 연습문제

**연습문제 1.**
자료점이 $n = 10$개이고 $\bar{x} = 4$, $S_{xx} = 20$, $\hat{\beta}_1 = 3.0$, $\text{SSE} = 16$인 단순선형회귀에서 기울기 $\beta_1$에 대한 95% 신뢰구간을 구성하라.

??? success "연습문제 1 풀이"
    먼저 추정 분산을 계산한다.

    $$
    s^2 = \frac{\text{SSE}}{n - 2} = \frac{16}{8} = 2.0
    $$

    기울기의 표준오차는

    $$
    \text{SE}(\hat{\beta}_1) = \frac{s}{\sqrt{S_{xx}}} = \frac{\sqrt{2}}{\sqrt{20}} = \sqrt{0.1} \approx 0.3162
    $$

    이다. $t_{0.025, 8} = 2.306$을 쓰면 95% 신뢰구간은

    $$
    3.0 \pm 2.306 \times 0.3162 = 3.0 \pm 0.729 = (2.271,\; 3.729)
    $$

    이다.

---

**연습문제 2.**
$c_i = (x_i - \bar{x})/S_{xx}$에 대해 $\hat{\beta}_1 = \sum_{i=1}^n c_i Y_i$임을 증명하고, 이를 이용해 $\operatorname{Var}(\hat{\beta}_1) = \sigma^2 / S_{xx}$을 유도하라.

??? success "연습문제 2 풀이"
    최소제곱 기울기 추정량은

    $$
    \hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})(Y_i - \bar{Y})}{S_{xx}} = \frac{\sum_{i=1}^n (x_i - \bar{x})Y_i}{S_{xx}}
    $$

    이다. 마지막 등호는 $\sum(x_i - \bar{x})\bar{Y} = \bar{Y}\sum(x_i - \bar{x}) = 0$을 쓴 것이다. $c_i = (x_i - \bar{x})/S_{xx}$로 두면 $\hat{\beta}_1 = \sum c_i Y_i$이다.

    $Y_i$가 독립이고 $\operatorname{Var}(Y_i) = \sigma^2$이므로

    $$
    \operatorname{Var}(\hat{\beta}_1) = \sum_{i=1}^n c_i^2 \operatorname{Var}(Y_i) = \sigma^2 \sum_{i=1}^n \frac{(x_i - \bar{x})^2}{S_{xx}^2} = \sigma^2 \cdot \frac{S_{xx}}{S_{xx}^2} = \frac{\sigma^2}{S_{xx}}
    $$

    이다. $\square$

---

**연습문제 3.**
$\text{SSE}/\sigma^2 \sim \chi^2_{n-2}$의 자유도가 왜 $n$이 아니라 $n - 2$인지 설명하라.

??? success "연습문제 3 풀이"
    잔차제곱합은 ($\sigma^2$으로 나눈 뒤) 모자 행렬 $\mathbf{H}$에 대해 $\text{SSE} = \mathbf{Y}^T(\mathbf{I} - \mathbf{H})\mathbf{Y}/\sigma^2$로 쓸 수 있다. 행렬 $\mathbf{I} - \mathbf{H}$는 계수가 $n - 2$인 멱등행렬이다(단순회귀에서는 절편과 기울기에 대응하여 $\text{rank}(\mathbf{H}) = 2$이므로).

    멱등행렬의 계수가 결과로 나오는 카이제곱분포의 자유도와 같다. 직관적으로는, $\mathbf{Y}$에 $n$개의 독립적인 정보 조각이 있는 상태에서 출발하지만 두 모수 $\beta_0$과 $\beta_1$을 적합하는 데 자유도 2를 "써버려" $\sigma^2$을 추정할 자유도로 $n - 2$가 남는다.

---

**연습문제 4.**
$\hat{\beta}_0$과 $\hat{\beta}_1$이 상관되어 있음을 보이고 $\operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = -\bar{x}\,\sigma^2/S_{xx}$을 유도하라.

??? success "연습문제 4 풀이"
    $\hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{x}$이므로

    $$
    \operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = \operatorname{Cov}(\bar{Y} - \hat{\beta}_1 \bar{x},\; \hat{\beta}_1) = \operatorname{Cov}(\bar{Y}, \hat{\beta}_1) - \bar{x}\operatorname{Var}(\hat{\beta}_1)
    $$

    을 계산한다. 이제 $\bar{Y} = \frac{1}{n}\sum Y_i$이고 $c_i = (x_i - \bar{x})/S_{xx}$에 대해 $\hat{\beta}_1 = \sum c_i Y_i$이므로

    $$
    \operatorname{Cov}(\bar{Y}, \hat{\beta}_1) = \frac{1}{n}\sum_{i=1}^n c_i \operatorname{Var}(Y_i) = \frac{\sigma^2}{n} \sum_{i=1}^n \frac{x_i - \bar{x}}{S_{xx}} = 0
    $$

    이다($\sum(x_i - \bar{x}) = 0$이므로). 따라서

    $$
    \operatorname{Cov}(\hat{\beta}_0, \hat{\beta}_1) = 0 - \bar{x} \cdot \frac{\sigma^2}{S_{xx}} = -\frac{\bar{x}\,\sigma^2}{S_{xx}}
    $$

    이다. $\bar{x} > 0$일 때 두 추정량은 음의 상관을 갖는다. $\square$
