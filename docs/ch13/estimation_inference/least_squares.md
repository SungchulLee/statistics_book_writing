# 최소제곱 추정

이 절은 선형회귀의 최적 모수를 세 관점 — 최소제곱(OLS) 기준, 정규오차 아래의 최대가능도 원리, 그리고 벡터 미적분으로 증명하는 정규방정식 — 에서 유도한다.

---

## 1. 선형회귀의 최대가능도 추정

### 자료와 모형

선형회귀 모형을 따르는 자료 $\{(x^{(i)}, y^{(i)}): i = 1, \ldots, m\}$가 주어졌다고 하자.

$$
y^{(i)} = \alpha + \beta x^{(i)} + \varepsilon^{(i)}
$$

여기서 $\varepsilon^{(i)} \sim N(0, \sigma^2)$이고 $\sigma^2$은 고정되어 있다.

### 가능도함수

각 $y^{(i)}$가 $y^{(i)} \sim N(\alpha + \beta x^{(i)}, \sigma^2)$을 따르므로 가능도함수는 다음과 같다.

$$
L(\alpha, \beta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2\right)
$$

### 로그가능도함수

자연로그를 취하면

$$
l(\alpha, \beta) = -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2 + \text{상수}
$$

### 비용함수(제곱손실)

비용함수를 다음과 같이 정의한다.

$$
J(\alpha, \beta) = \frac{1}{m} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2
$$

### MLE와 OLS의 동등성

상수와 $\sigma^2$이 $\alpha$나 $\beta$에 의존하지 않으므로, 가능도를 최대화하는 것은 제곱손실을 최소화하는 것과 동등하다.

$$
\text{argmax}_{\alpha, \beta}\ L \quad \Leftrightarrow \quad \text{argmax}_{\alpha, \beta}\ l \quad \Leftrightarrow \quad \text{argmin}_{\alpha, \beta}\ J
$$

이는 기본이 되는 결과이다. **정규오차 가정 아래에서 OLS와 MLE는 같은 모수 추정값을 준다.**

---

## 2. 정규방정식(단순한 경우)

L2 손실의 편미분을 0으로 둔다.

$$
l = \frac{1}{n} \sum_{i=1}^n (\alpha + \beta x_i - y_i)^2
$$

$$
\begin{array}{lll}
\displaystyle \frac{\partial l}{\partial \alpha} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) = 0
& \Rightarrow &
2\alpha + 2\beta \bar{x} - 2\bar{y} = 0 \\[8pt]
\displaystyle \frac{\partial l}{\partial \beta} = \frac{2}{n} \sum_{i=1}^n \left((\alpha + \beta x_i) - y_i\right) x_i = 0
& \Rightarrow &
2\alpha \bar{x} + 2\beta \overline{x^2} - 2\overline{xy} = 0
\end{array}
$$

풀면

$$
\begin{array}{lll}
\beta &=& \displaystyle \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - (\bar{x})^2}
= \frac{s_{xy}}{s_x^2}
= \frac{\rho\, s_x\, s_y}{s_x^2}
= \rho \frac{s_y}{s_x} \\[8pt]
\alpha &=& \displaystyle -\rho \frac{s_y}{s_x} \bar{x} + \bar{y}
\end{array}
$$

---

## 3. 정규방정식(행렬 형태)

설명변수가 $d$개인 일반적인 경우:

$$
\text{argmin}_{\boldsymbol{\theta}} \; J(\boldsymbol{\theta})
\quad \Rightarrow \quad
\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}
\quad \Rightarrow \quad
\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### 벡터 미적분 항등식

유도에 다음 항등식들을 쓴다.

$$
\begin{aligned}
(1) \quad & \frac{\partial (\mathbf{a}^T \mathbf{b})}{\partial \mathbf{a}} = \mathbf{b} \\[4pt]
(2) \quad & \frac{\partial (\mathbf{a}^T \mathbf{b})}{\partial \mathbf{b}} = \mathbf{a} \\[4pt]
(3) \quad & \frac{\partial \text{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{A}} = \mathbf{B}^T \\[4pt]
(4) \quad & \frac{\partial \text{tr}(\mathbf{A}\mathbf{B})}{\partial \mathbf{B}} = \mathbf{A}^T \\[4pt]
(5) \quad & \frac{\partial |\mathbf{A}|}{\partial \mathbf{A}} = \mathbf{C} \quad \text{($\mathbf{C}$는 $\mathbf{A}$의 여인수 행렬)} \\[4pt]
(6) \quad & \frac{\partial \log|\mathbf{A}|}{\partial \mathbf{A}} = \mathbf{A}^{-T} := (\mathbf{A}^{-1})^T
\end{aligned}
$$

### 정규방정식의 증명

비용함수에서 출발한다.

$$
\begin{aligned}
J(\boldsymbol{\theta})
&= \frac{1}{2m} \| \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \|^2 \\[4pt]
&= \frac{1}{2m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y}) \\[4pt]
&= \frac{1}{2m} \left( \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \boldsymbol{\theta}^T \mathbf{X}^T \mathbf{y} - \mathbf{y}^T \mathbf{X} \boldsymbol{\theta} + \mathbf{y}^T \mathbf{y} \right)
\end{aligned}
$$

$\boldsymbol{\theta}$에 대해 미분하면

$$
\begin{aligned}
\frac{\partial J}{\partial \boldsymbol{\theta}}
&= \frac{1}{2m} \left( \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} + (\boldsymbol{\theta}^T \mathbf{X}^T \mathbf{X})^T - \mathbf{X}^T \mathbf{y} - (\mathbf{y}^T \mathbf{X})^T \right) \\[4pt]
&= \frac{1}{m} \left( \mathbf{X}^T \mathbf{X} \boldsymbol{\theta} - \mathbf{X}^T \mathbf{y} \right) \\[4pt]
&= \mathbf{0}
\end{aligned}
$$

이로부터 **정규방정식** $\mathbf{X}^T \mathbf{X} \boldsymbol{\theta} = \mathbf{X}^T \mathbf{y}$를 얻고, 닫힌 형태의 해는 다음과 같다.

$$
\hat{\boldsymbol{\theta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

!!! note "역행렬은 언제 존재하는가"
    행렬 $\mathbf{X}^T\mathbf{X}$가 가역인 것은 $\mathbf{X}$가 완전 열계수를 가질 때, 곧 어떤 설명변수도 나머지의 완전한 선형결합이 아닐 때에 한한다. 이 조건이 무너지면(다중공선성) 릿지 회귀 같은 정칙화 기법을 쓸 수 있다.

---

## 4. 예측

### 일반적인 경우

새 입력 $\mathbf{x}$가 주어지면 절편을 위해 1을 앞에 붙인 뒤 계수벡터와 곱한다.

$$
\mathbf{x}
\quad \Rightarrow \quad
\tilde{\mathbf{x}} = [1, \mathbf{x}]
\quad \Rightarrow \quad
\hat{y} = \tilde{\mathbf{x}} \hat{\boldsymbol{\theta}}
$$

### 단순선형회귀

설명변수가 하나인 특수한 경우, 예측 공식은 우아한 표준화 형태로 줄어든다.

$$
\frac{y - \bar{y}}{s_y} = \rho \frac{x - \bar{x}}{s_x}
$$

**증명**: 단순한 경우의 정규방정식에서 $\beta = \rho\, s_y / s_x$와 $\alpha = \bar{y} - \beta \bar{x}$를 얻었으므로

$$
\begin{aligned}
y &= \alpha + \beta x \\
  &= \bar{y} - \rho \frac{s_y}{s_x} \bar{x} + \rho \frac{s_y}{s_x} x \\
  &= \rho \frac{s_y}{s_x}(x - \bar{x}) + \bar{y}
\end{aligned}
$$

정리하면 $\displaystyle \frac{y - \bar{y}}{s_y} = \rho \frac{x - \bar{x}}{s_x}$이다.

---

## 5. 종합 예제: 최대가능도에서 행렬 표현까지

앞의 내용을 하나로 꿰는 완결된 유도 예제이다.

선형회귀 모형을 따르는 자료 $\{(x^{(i)}, y^{(i)}): i = 1, \ldots, m\}$가 주어졌다고 하자.

$$
y^{(i)} = \alpha + \beta x^{(i)} + \varepsilon^{(i)}
$$

여기서 $\varepsilon^{(i)} \sim N(0, \sigma^2)$이고 $\sigma^2$은 고정되어 있다.

**(a)** 이 모형의 가능도함수 $L(\alpha, \beta)$를 유도하라.

**(b)** 로그가능도함수 $l(\alpha, \beta)$를 유도하라.

**(c)** 비용함수(제곱손실)를 다음과 같이 정의할 때,

$$
J(\alpha, \beta) = \frac{1}{2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2
$$

제곱손실을 최소화하는 것과 가능도함수를 최대화하는 것의 관계를 설명하라.

**(d)** 제곱손실을 최소화하는 모수 $\hat{\alpha}$, $\hat{\beta}$가 만족해야 하는 정규방정식을 유도하라.

**(e)** 정규방정식을 풀어 $\hat{\alpha}$와 $\hat{\beta}$를 구하라.

**(f)** 설명변수가 $d$개인 다변수 경우로 모형을 확장하라.

$$
\mathbf{x}^{(i)} = (x_1^{(i)}, x_2^{(i)}, \ldots, x_d^{(i)}), \quad 1 \leq i \leq m
$$

설계행렬 $\mathbf{X}$, 모수벡터 $\boldsymbol{\theta}$, 반응벡터 $\mathbf{y}$를 써서 제곱손실함수를 행렬 연산으로 나타내라.

??? success "풀이"

    **(a) 가능도함수**

    $\varepsilon^{(i)} \sim N(0, \sigma^2)$이라 가정하면 각 $y^{(i)}$는 $y^{(i)} \sim N(\alpha + \beta x^{(i)}, \sigma^2)$을 따른다. 가능도함수는

    $$
    L(\alpha, \beta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{1}{2\sigma^2} \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2\right)
    $$

    **(b) 로그가능도함수**

    $$
    l(\alpha, \beta) = \ln L(\alpha, \beta) = -\frac{1}{2\sigma^2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2 + \text{상수}
    $$

    **(c) MLE와 OLS의 동등성**

    제곱손실함수는

    $$
    J(\alpha, \beta) = \frac{1}{2} \sum_{i=1}^m \left(y^{(i)} - \alpha - \beta x^{(i)}\right)^2
    $$

    상수와 $\sigma^2$이 $\alpha$나 $\beta$에 의존하지 않으므로 로그가능도를 최대화하는 것은 잔차제곱합을 최소화하는 것과 동등하다.

    $$
    \text{argmax}_{\alpha, \beta}\ L(\alpha, \beta) \quad \Leftrightarrow \quad \text{argmax}_{\alpha, \beta}\ l(\alpha, \beta) \quad \Leftrightarrow \quad \text{argmin}_{\alpha, \beta}\ J(\alpha, \beta)
    $$

    **(d) 정규방정식**

    편미분을 취해 0으로 두면

    1. $\displaystyle \frac{\partial J}{\partial \alpha} = \sum_{i=1}^m \left(\alpha + \beta x^{(i)} - y^{(i)}\right) = 0$

    2. $\displaystyle \frac{\partial J}{\partial \beta} = \sum_{i=1}^m \left(\alpha + \beta x^{(i)} - y^{(i)}\right) x^{(i)} = 0$

    이로부터 정규방정식을 얻는다.

    $$
    \begin{aligned}
    \alpha + \beta \bar{x} &= \bar{y} \\
    \beta \overline{x^2} + \alpha \bar{x} &= \overline{xy}
    \end{aligned}
    $$

    **(e) 정규방정식 풀기**

    $$
    \beta = \frac{\overline{xy} - \bar{x}\bar{y}}{\overline{x^2} - (\bar{x})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)} = \rho \frac{\sigma_y}{\sigma_x}
    $$

    $$
    \alpha = \bar{y} - \beta \bar{x}
    $$

    **(f) 행렬 표현**

    설계행렬 $\mathbf{X}$, 모수벡터 $\boldsymbol{\theta}$, 출력벡터 $\mathbf{y}$를 만든다.

    $$
    \mathbf{X} =
    \begin{bmatrix}
    1 & x_1^{(1)} & \cdots & x_d^{(1)} \\
    \vdots & \vdots & \ddots & \vdots \\
    1 & x_1^{(m)} & \cdots & x_d^{(m)}
    \end{bmatrix}, \quad
    \boldsymbol{\theta} =
    \begin{bmatrix}
    \beta_0 \\ \beta_1 \\ \vdots \\ \beta_d
    \end{bmatrix}, \quad
    \mathbf{y} =
    \begin{bmatrix}
    y^{(1)} \\ \vdots \\ y^{(m)}
    \end{bmatrix}
    $$

    제곱손실함수는

    $$
    J(\boldsymbol{\theta}) = \frac{1}{2m} \| \mathbf{X}\boldsymbol{\theta} - \mathbf{y} \|^2 = \frac{1}{2m} (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})^T (\mathbf{X}\boldsymbol{\theta} - \mathbf{y})
    $$

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
$\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$을 최소화하여 단순선형회귀의 OLS 추정량 $\hat{\beta}_1 = \sum(x_i - \bar{x})(y_i - \bar{y}) / \sum(x_i - \bar{x})^2$을 유도하라.

</div>

??? success "풀이"
    $S(\beta_0, \beta_1) = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2$이라 하자. 편미분을 취해 0으로 두면

    $$
    \frac{\partial S}{\partial \beta_0} = -2\sum(y_i - \beta_0 - \beta_1 x_i) = 0 \implies \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}
    $$

    $$
    \frac{\partial S}{\partial \beta_1} = -2\sum x_i(y_i - \beta_0 - \beta_1 x_i) = 0
    $$

    $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$를 대입하면

    $$
    \sum x_i(y_i - \bar{y} + \hat{\beta}_1 \bar{x} - \hat{\beta}_1 x_i) = 0
    $$

    $$
    \sum x_i(y_i - \bar{y}) = \hat{\beta}_1 \sum x_i(x_i - \bar{x})
    $$

    $\sum x_i(y_i - \bar{y}) = \sum(x_i - \bar{x})(y_i - \bar{y})$이고 $\sum x_i(x_i - \bar{x}) = \sum(x_i - \bar{x})^2$이므로

    $$
    \hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
OLS 잔차의 합이 0임을 보여라. 곧 $e_i = y_i - \hat{y}_i$일 때 $\sum_{i=1}^n e_i = 0$이다.

</div>

??? success "풀이"
    첫 번째 정규방정식($\partial S / \partial \beta_0 = 0$)에서

    $$
    \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
    $$

    $e_i = y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i$이므로 이 식이 바로 $\sum_{i=1}^n e_i = 0$이다.

    이 결과는 절편항이 있는 모든 회귀모형에서 성립한다. 반대로 절편을 강제로 0으로 둔 모형에서는 성립하지 않는다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최소제곱 해가 설계행렬 $X$의 열공간에서 무엇을 뜻하는지 기하학적으로 설명하라. 적합값 $\hat{Y}$와 잔차 $e$의 관계는 무엇인가?

</div>

??? success "풀이"
    적합값 $\hat{Y} = X\hat{\beta} = X(X^TX)^{-1}X^TY = HY$는 $Y$를 $X$의 열공간 위로 **직교사영**한 것이다. 잔차벡터 $e = Y - \hat{Y} = (I - H)Y$는 열공간과 직교한다.

    기하학적으로 OLS는 $X$의 열공간에서 유클리드 거리로 $Y$에 가장 가까운 점을 찾는다. 피타고라스 정리에서 분해 $\|Y\|^2 = \|\hat{Y}\|^2 + \|e\|^2$을 얻으며, 이는 ($Y$와 $\hat{Y}$를 중심화하면) $\text{SST} = \text{SSR} + \text{SSE}$에 해당한다.

---

## 정리하며

같은 추정량이 **세 가지 길**에서 나온다.

- **최소제곱 기준.** 잔차제곱합을 최소화한다. 분포 가정이 전혀 없다.
- **최대가능도.** 오차가 정규라고 가정하면 로그가능도의 최대화가 잔차제곱합의 최소화와 같아진다. **정규 가정 아래에서 두 원리가 만나는 것**이며, 제곱을 쓰는 관행에 근거를 준다.
- **정규방정식.** $\mathbf X^\top\mathbf X\hat{\boldsymbol\beta}=\mathbf X^\top\mathbf y$ 이며, 기하적으로는 **잔차가 열공간과 직교**한다는 조건이다.
- **세 번째 관점이 가장 일반적이다.** 정규성도 필요 없고, 0장의 사영 그림으로 곧바로 읽힌다.
- **가우스–마르코프 정리가 최소제곱을 정당화한다.** 오차가 무상관이고 등분산이면(정규성 없이도) 최소제곱이 **선형 불편추정량 중 최소분산**이다.

다음 절 **표집분포**로 넘어간다. 추정값의 불확실성을 다룬다.
