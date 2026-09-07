# 교란과 인과 시연

## 개요

이 페이지는 교란변수가 어떻게 허위상관을 만들어 내는지, 그리고 Simpson의 역설이 어떻게 처치효과의 겉보기 방향을 뒤집는지 보인다. 두 가지 상황을 모의로 만든다. 하나는 숨은 공통원인이 오도하는 연관을 만들어 내는 교란된 회귀이고, 다른 하나는 소박한 추정값의 부호가 아예 반대로 나오는 처치효과 분석이다.

---

## 1부: 교란된 회귀

### 자료생성과정

$C$가 교란변수, $T$가 처치, $Y$가 결과인 유향비순환그래프 $T \leftarrow C \rightarrow Y$를 생각하자. 처치 $T$는 $Y$에 **아무런 인과효과가 없으며**, 모든 연관은 $C$를 통해 흐른다.

다음에서 자료를 생성한다.

$$
\begin{pmatrix} T \\ C \end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix} 0 \\ 0 \end{pmatrix}, \begin{pmatrix} 1 & \rho_{TC} \\ \rho_{TC} & 1 \end{pmatrix}\right), \qquad Y = C + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 1)
$$

$Y$는 오직 $C$에만 의존하므로 $T$가 $Y$에 미치는 참 인과효과는 0이다.

```python
import numpy as np
from scipy import stats

np.random.seed(42)


def simulate_confounded_data(n=500, rho_tc=0.8):
    mean = [0, 0]
    cov = [[1, rho_tc], [rho_tc, 1]]
    tc = np.random.multivariate_normal(mean, cov, n)
    t = tc[:, 0]
    c = tc[:, 1]
    y = c + np.random.normal(0, 1, n)
    return t, c, y
```

### 짧은 회귀와 긴 회귀

**짧은 회귀**($C$를 빠뜨린 회귀)는 $Y$를 $T$에만 회귀시킨다.

$$
Y = \alpha + \beta_T^{\text{short}} T + u
$$

**긴 회귀**($C$를 통제한 회귀)는 교란변수를 포함한다.

$$
Y = \alpha + \beta_T^{\text{long}} T + \beta_C C + u
$$

```python
def compute_regressions(t, c, y):
    # Short regression
    slope_short, _, r_short, p_short, _ = stats.linregress(t, y)

    # Long regression via OLS
    X = np.column_stack([np.ones(len(t)), t, c])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    return {
        "short_slope": slope_short,
        "short_p": p_short,
        "long_beta_T": beta[1],
        "long_beta_C": beta[2],
    }
```

$\rho_{TC} = 0.8$, $n = 500$일 때의 결과:

```text
short_slope  =  0.7772   (p = 2.8e-37)
long_beta_T  = -0.1000
long_beta_C  =  1.0915
```

$T$에 아무런 인과효과가 없는데도 짧은 회귀는 $\beta_T^{\text{short}} = 0.777$이라는 압도적으로 유의한 기울기를 내놓는다. 긴 회귀는 $\beta_T^{\text{long}} \approx 0$을 올바르게 추정한다.

### 부분회귀(Frisch-Waugh-Lovell)

이와 동등한 방법으로, $T$와 $Y$를 각각 $C$에 회귀시켜 잔차를 얻은 뒤 그 잔차끼리 회귀시킬 수 있다.

$$
e_T = T - \hat{\gamma}_1 C, \qquad e_Y = Y - \hat{\gamma}_2 C
$$

$$
\beta_T^{\text{long}} = \frac{\text{Cov}(e_T, e_Y)}{\text{Var}(e_T)}
$$

```python
t_resid = t - stats.linregress(c, t).slope * c
y_resid = y - stats.linregress(c, y).slope * c
slope_partial = stats.linregress(t_resid, y_resid).slope
```

이렇게 얻은 `slope_partial`은 $-0.1000$으로 긴 회귀의 $\beta_T^{\text{long}}$과 소수점 넷째 자리까지 일치한다. **Frisch-Waugh-Lovell 정리**가 작동하는 모습이다. 긴 회귀에서 $T$의 계수는 $e_Y$를 $e_T$에 회귀시킨 기울기와 같다.

---

## 2부: Simpson의 역설과 평균처치효과

### 설정

환자에게 이진 중증도 지표가 있다(경증 = 0, 중증 = 1). 중증 환자가 처치를 받을 확률이 더 높다(적응증에 의한 교란).

$$
P(\text{처치} = 1 \mid \text{중증}) = 0.7, \qquad P(\text{처치} = 1 \mid \text{경증}) = 0.3
$$

결과는 중증도와 처치 모두에 의존한다.

$$
Y = 50 - 20 \cdot \text{중증도} + 5 \cdot \text{처치} + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, 5^2)
$$

참 처치효과는 $+5$이지만, 중증 환자는 전반적으로 결과가 나쁘다.

```python
def simpson_paradox_demo(n=1000):
    severity = np.random.binomial(1, 0.5, n)
    p_treat = np.where(severity == 1, 0.7, 0.3)
    treatment = np.random.binomial(1, p_treat)

    y = (50 - 20 * severity + 5 * treatment
         + np.random.normal(0, 5, n))

    # Naive ATE
    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    # Adjusted ATE (stratified)
    ate_mild = (y[(treatment == 1) & (severity == 0)].mean()
                - y[(treatment == 0) & (severity == 0)].mean())
    ate_severe = (y[(treatment == 1) & (severity == 1)].mean()
                  - y[(treatment == 0) & (severity == 1)].mean())
    p_severe = severity.mean()
    ate_adjusted = (1 - p_severe) * ate_mild + p_severe * ate_severe

    return ate_naive, ate_mild, ate_severe, ate_adjusted
```

### 역설

**소박한 평균처치효과(ATE)**는 중증도를 조정하지 않은 채 처치군과 비처치군을 비교한다. 결과가 나쁜 중증 환자가 처치군에 과다 대표되므로, 소박한 ATE는 **음수**가 될 수 있고, 그러면 처치가 해로운 것처럼 보인다.

**조정된 ATE**는 중증도로 층화한 뒤 가중평균을 계산한다.

$$
\text{ATE}_{\text{adj}} = P(\text{경증}) \cdot \text{ATE}_{\text{경증}} + P(\text{중증}) \cdot \text{ATE}_{\text{중증}}
$$

$n = 1000$일 때의 결과:

```text
ate_naive    = -2.62
ate_mild     =  5.22
ate_severe   =  6.07
ate_adjusted =  5.64
```

소박한 ATE는 $-2.62$로 부호가 반대이다(이론값은 $-3$이다: 처치군의 중증 비율은 $0.7$, 비처치군은 $0.3$이므로 $-20 \times 0.4 + 5 = -3$). 반면 두 하위집단의 ATE는 모두 참값 $+5$ 근처이고, 그 가중평균 $5.64$가 참값을 올바르게 복원한다.

---

## 해석

이 시연들은 인과추론의 두 가지 핵심 주제를 보여준다.

1. **누락변수 편향.** 교란변수를 통제하지 않으면 인과효과 추정값이 편향된다. 짧은 회귀는 실제로 $C$에 속하는 변동을 $T$의 탓으로 돌린다. 편향은 $\beta_T^{\text{short}} - \beta_T^{\text{long}} = \hat{\delta} \cdot \hat{\gamma}$와 같은데, 여기서 $\hat{\delta}$는 긴 회귀에서 $C$의 계수이고 $\hat{\gamma}$는 $C$를 $T$에 회귀시킨 계수이다.

2. **Simpson의 역설.** 이질적인 하위집단을 뭉뚱그리면 효과의 방향이 뒤집힐 수 있다. 소박한 ATE는 중증도에 의해 교란되어 있고, 층화한 ATE는 이 교란을 제거한다. 관찰연구에는 무작위화가 없으므로 교란변수에 대한 명시적 조정이 반드시 필요하다.

---

## 연습문제

**연습문제 1.**
교란된 회귀에서 누락변수 편향 공식을 유도하라. 다음을 보여라.

$$
\beta_T^{\text{short}} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \hat{\gamma}
$$

여기서 $\hat{\gamma}$는 $C$를 $T$에 회귀시킨 계수이다.

??? success "풀이"

    짧은 회귀가 추정하는 것은

    $$
    \beta_T^{\text{short}} = \frac{\text{Cov}(T, Y)}{\text{Var}(T)}
    $$

    $Y = \alpha + \beta_T^{\text{long}} T + \beta_C^{\text{long}} C + u$이므로

    $$
    \text{Cov}(T, Y) = \beta_T^{\text{long}} \text{Var}(T) + \beta_C^{\text{long}} \text{Cov}(T, C)
    $$

    양변을 $\text{Var}(T)$로 나누면

    $$
    \beta_T^{\text{short}} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \frac{\text{Cov}(T, C)}{\text{Var}(T)} = \beta_T^{\text{long}} + \beta_C^{\text{long}} \cdot \hat{\gamma}
    $$

    여기서 $\hat{\gamma} = \text{Cov}(T, C) / \text{Var}(T)$는 $C$를 $T$에 회귀시킨 회귀계수이다. 우리 모의실험에서는 $\beta_T^{\text{long}} \approx 0$, $\beta_C^{\text{long}} \approx 1$이고 $\text{Var}(T) = 1$이므로 $\beta_T^{\text{short}} \approx \hat{\gamma} = \rho_{TC}$이다. 실제로 표본에서 $\hat{\gamma} = 0.804$, $\beta_T^{\text{short}} = 0.777$이다. 짧은 회귀의 기울기는 **전부가 편향**이다. $\square$

---

**연습문제 2.**
$\rho_{TC} \in \{-0.9, -0.5, 0, 0.5, 0.9\}$, $n = 500$으로 교란된 회귀를 모의실험하라. 각 값에 대해 $\beta_T^{\text{short}}$와 $\beta_T^{\text{long}}$을 보고하라. $\beta_T^{\text{short}}$를 $\rho_{TC}$의 함수로 그려 대략 선형임을 확인하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rhos = [-0.9, -0.5, 0, 0.5, 0.9]
    short_slopes = []

    for rho in rhos:
        np.random.seed(42)
        n = 500
        tc = np.random.multivariate_normal([0, 0],
                                           [[1, rho], [rho, 1]], n)
        t, c = tc[:, 0], tc[:, 1]
        y = c + np.random.normal(0, 1, n)

        short_slope = stats.linregress(t, y).slope
        X = np.column_stack([np.ones(n), t, c])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        short_slopes.append(short_slope)
        print(f"rho={rho:+.1f}: short={short_slope:.3f}, "
              f"long_T={beta[1]:.3f}")

    import matplotlib.pyplot as plt
    plt.plot(rhos, short_slopes, 'o-')
    plt.xlabel('rho(T, C)')
    plt.ylabel('Short regression slope')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Omitted Variable Bias')
    plt.show()
    ```

    출력:

    ```text
    rho=-0.9: short=-0.877, long_T=0.131
    rho=-0.5: short=-0.444, long_T=0.056
    rho=+0.0: short=-0.021, long_T=0.008
    rho=+0.5: short=0.454,  long_T=-0.065
    rho=+0.9: short=0.885,  long_T=-0.140
    ```

    그림은 대략 선형인 관계 $\beta_T^{\text{short}} \approx \rho_{TC}$를 보여주며, 이는 누락변수 편향 공식을 확인해 준다. 긴 회귀의 기울기는 $\rho_{TC}$와 무관하게 0 근처에 머문다. $\rho_{TC} = 0$일 때는 짧은 회귀도 편향되지 않는데, 교란변수가 처치와 상관이 없으면 애초에 교란이 아니기 때문이다. $\square$

---

**연습문제 3.**
Simpson의 역설 시연에서 처치 배정이 중증도와 독립이라면(즉 모두에게 $P(\text{처치}) = 0.5$) 어떻게 되는가? 이를 모의실험하고 소박한 ATE가 참 효과를 올바르게 추정함을 보여라.

??? success "풀이"

    ```python
    import numpy as np

    np.random.seed(42)
    n = 1000
    severity = np.random.binomial(1, 0.5, n)
    treatment = np.random.binomial(1, 0.5, n)  # independent of severity

    y = 50 - 20 * severity + 5 * treatment + np.random.normal(0, 5, n)

    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()
    print(f"Naive ATE = {ate_naive:.2f} (true = 5)")   # 4.86
    ```

    처치가 무작위화되면(교란변수와 독립이면) 소박한 ATE는 참 ATE의 불편추정량이 된다. 이 모의실험에서 $4.86$으로 참값 $5$에 가깝게 나온다. 처치군과 대조군의 중증도 분포가 같으므로 교란이 없다. 무작위대조시험이 인과추론의 표준으로 여겨지는 이유가 바로 이것이다. $\square$

---

**연습문제 4.**
Simpson의 역설 예제를 중증도 세 수준(경증, 중등증, 중증)으로 확장하고 처치확률을 각각 0.2, 0.5, 0.8로 두어라. 역설이 여전히 일어남을 보이고 층화한 ATE를 계산하라.

??? success "풀이"

    ```python
    import numpy as np

    np.random.seed(42)
    n = 1500
    severity = np.random.choice([0, 1, 2], size=n, p=[1/3, 1/3, 1/3])
    p_treat = np.where(severity == 0, 0.2,
                       np.where(severity == 1, 0.5, 0.8))
    treatment = np.random.binomial(1, p_treat)

    y = 50 - 15 * severity + 5 * treatment + np.random.normal(0, 5, n)

    ate_naive = y[treatment == 1].mean() - y[treatment == 0].mean()

    ates = []
    for s in [0, 1, 2]:
        mask_t = (treatment == 1) & (severity == s)
        mask_c = (treatment == 0) & (severity == s)
        ate_s = y[mask_t].mean() - y[mask_c].mean()
        ates.append(ate_s)
        print(f"ATE (severity={s}): {ate_s:.2f}")

    p_sev = [np.mean(severity == s) for s in [0, 1, 2]]
    ate_adjusted = sum(p * a for p, a in zip(p_sev, ates))

    print(f"Naive ATE:    {ate_naive:.2f}")
    print(f"Adjusted ATE: {ate_adjusted:.2f}")
    print(f"True effect:  5.00")
    ```

    출력:

    ```text
    ATE (severity=0): 4.51
    ATE (severity=1): 4.17
    ATE (severity=2): 5.60
    Naive ATE:    -7.28
    Adjusted ATE:  4.76
    True effect:  5.00
    ```

    소박한 ATE는 $-7.28$로 크게 음의 방향으로 편향된다. 결과가 가장 나쁜 최중증 환자가 처치를 압도적으로 많이 받기 때문이다(이론값도 $-7$이다: $\mathbb{E}[\text{중증도} \mid T=1] = 1.4$, $\mathbb{E}[\text{중증도} \mid T=0] = 0.6$이므로 $-15 \times 0.8 + 5 = -7$). 각 하위집단의 ATE는 모두 5 근처이고 그 가중평균 $4.76$이 참 효과를 올바르게 복원한다. 역설은 교란 층의 개수와 무관하게 일반화된다. $\square$

---

**연습문제 5.**
$T$가 무작위화되어 있다면(모든 교란변수와 독립이면), 잠재결과 틀에서 평균들의 소박한 차이 $\mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0]$이 평균처치효과 $\mathbb{E}[Y(1) - Y(0)]$와 같음을 증명하라.

??? success "풀이"

    $Y(1)$과 $Y(0)$을 각각 처치와 대조에서의 잠재결과라 하자. 관측되는 결과는 $Y = T \cdot Y(1) + (1 - T) \cdot Y(0)$이다.

    ATE는 다음으로 정의된다.

    $$
    \tau = \mathbb{E}[Y(1) - Y(0)]
    $$

    $T$가 $(Y(0), Y(1))$과 독립이면(무작위화),

    $$
    \mathbb{E}[Y \mid T = 1] = \mathbb{E}[Y(1) \mid T = 1] = \mathbb{E}[Y(1)]
    $$

    이며 마지막 등식에 독립성이 쓰였다. 마찬가지로

    $$
    \mathbb{E}[Y \mid T = 0] = \mathbb{E}[Y(0) \mid T = 0] = \mathbb{E}[Y(0)]
    $$

    따라서

    $$
    \mathbb{E}[Y \mid T = 1] - \mathbb{E}[Y \mid T = 0] = \mathbb{E}[Y(1)] - \mathbb{E}[Y(0)] = \tau
    $$

    무작위화는 처치군과 대조군이 기댓값의 의미에서 비교 가능하도록 보장하여 선택편향을 없앤다. 무작위화가 없으면 일반적으로 $\mathbb{E}[Y(1) \mid T = 1] \ne \mathbb{E}[Y(1)]$이고, 평균들의 소박한 차이는 편향된다. $\square$
