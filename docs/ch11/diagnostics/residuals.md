# 잔차 분석

잔차 분석은 선형회귀 모형이 자료에 얼마나 잘 맞는지 평가하는 결정적인 단계이다. 관측값과 예측값의 차이인 잔차를 살펴봄으로써 모형의 정확도와 핵심 가정의 성립 여부를 알 수 있다.

---

## 1. 잔차의 이해

잔차는 관측값과 그에 대응하는 예측값의 차이이다.

$$
e_i = y_i - \hat{y}_i
$$

여기서 $y_i$는 관측값이고 $\hat{y}_i$는 회귀모형의 예측값이다.

$$\begin{array}{lll}
\text{예측값} && \hat{y}_i \\
\text{잔차} && y_i - \hat{y}_i \\\hline
\text{실제값} && y_i
\end{array}$$

잔차는 모형이 각 관측값에 얼마나 잘 맞는지를 드러낸다. 이상적으로는 0에 가까울수록 좋으며, 이는 예측이 관측값과 가깝게 일치함을 뜻한다.

---

## 2. 핵심 가정

잔차 분석은 신뢰할 만한 선형회귀에 필수적인 다음 가정들을 확인한다.

- **선형성**: 설명변수와 반응변수의 관계가 선형이어야 한다. 잔차를 예측값에 대해 그렸을 때 패턴이 보이지 않아야 한다.
- **독립성**: 잔차들이 서로 독립이어야 한다. 시계열 자료에서 특히 중요하다.
- **등분산성(상수분산)**: 잔차의 분산이 독립변수의 모든 수준에서 일정해야 한다.
- **정규성**: 잔차가 정규분포를 따르는 것이 이상적이다. 모형을 추론에 쓸 때 특히 그렇다.

---

## 3. 잔차그림

### 잔차–적합값 그림

이 그림은 **선형성**과 **등분산성**의 문제를 찾아내는 데 도움이 된다. 이상적으로는 잔차가 0을 지나는 수평선 주위에 패턴 없이 무작위로 흩어져 있어야 한다. 곡선 패턴은 비선형성을, 부채꼴이나 깔때기 모양은 이분산을 시사한다.

**일반 잔차그림과의 구분**: 잔차–적합값 그림은 $x$축에 적합값(예측값)을 두어 모형 전체를 점검한다. 일반적인 "잔차그림"은 $x$축에 개별 설명변수나 관측 순번을 두어 특정 설명변수와의 관계나 시간 추세를 진단하기도 한다.

| 항목 | 잔차–적합값 그림 | 일반 잔차그림 |
|---|---|---|
| **주된 목적** | 선형성과 등분산성 확인 | 개별 설명변수와의 관계 평가 |
| **X축** | 적합값(예측값) | 설명변수 또는 관측 순번 |
| **쓰는 때** | 적합 후 전반적 가정 점검 | 특정 설명변수나 시간 효과 진단 |

#### 무늬가 없다는 것을 수로 판정하기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 모형이 잘 맞을 때의 잔차. `make_regression(n_samples=100, n_features=1, noise=10)`은 $x_i \sim N(0,1)$, $y_i = \beta x_i + \varepsilon_i$, $\varepsilon_i \sim N(0, 10^2)$인 자료를 만든다(절편은 0이다).

**(1)** 그림을 보기 **전에** 잔차가 들어갈 띠의 폭을 계산하시오. $E[\sum_i e_i^2] = (n-p)\sigma^2$을 써서 잔차제곱평균의 이론값을 구하고, $\pm 2\hat\sigma$ 밖에 놓일 점이 몇 개쯤일지 말하시오. 또 모집단 $R^2$을 $\beta$와 $\sigma$로 적으시오.

**(2)** "무늬가 없다"를 눈이 아니라 **수로** 판정하는 방법 셋을 세우고 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** [11.5절 보기 1](../assumptions/checking_homoscedasticity.md)에서 $E[\mathbf e^\top \mathbf e] = \operatorname{tr}((I-H)\Sigma)$를 보였다. 등분산이면 $\Sigma = \sigma^2 I$이고 $\operatorname{tr}(I - H) = n - p$이므로

    $$
    E\!\left[\frac{1}{n}\sum_i e_i^2\right] = \frac{(n-p)\sigma^2}{n} = \frac{98 \times 100}{100} = 98
    $$

    이다. 곧 잔차의 제곱평균이 $98$ 근처, 제곱근으로 $9.90$ 근처여야 한다. **참 $\sigma = 10$보다 조금 작다.** 두 모수를 적합하느라 잔차가 그만큼 수축했기 때문이다.

    $\hat\sigma = \sqrt{\text{SSE}/(n-p)}$는 이 수축을 되돌린 값이므로 $\sigma$를 겨냥한다. 오차가 정규이면 표준화 잔차가 대략 표준정규이므로 $\pm 2\hat\sigma$ 밖에 놓일 점은 $100 \times 0.0455 \approx 5$개가 기댓값이다. 다만 잔차는 서로 독립이 아니고 분산도 $\sigma^2(1-h_{ii})$로 조금씩 다르므로 이 수는 어림이다.

    모집단 $R^2$은 신호 대 전체의 비다. $\operatorname{Var}(x) = 1$이므로

    $$
    R^2_{\text{pop}} = \frac{\beta^2 \operatorname{Var}(x)}{\beta^2\operatorname{Var}(x) + \sigma^2} = \frac{\beta^2}{\beta^2 + 100}
    $$

    이다.

    **(2) 무늬 없음을 재는 세 가지.** 잔차 그림에서 눈이 찾는 것은 셋이고, 각각에 대응하는 수가 있다.

    | 눈이 보는 것 | 수로 재면 | 기대값 |
    |---|---|---|
    | 전체 기울기 | $\operatorname{corr}(e, \hat y)$ | **항등적으로 0** |
    | 깔때기 | $\operatorname{corr}(\lvert e \rvert, \hat y)$ | 0 근처 |
    | 휘어짐 | $e$를 $\hat y, \hat y^2$에 회귀한 2차항 | 유의하지 않음 |

    첫 줄이 항등적으로 0인 것은 정규방정식 때문이며([11.5절 보기 2](../assumptions/checking_homoscedasticity.md)), **그러므로 기울기는 볼 필요가 없다.** 읽을 것은 둘째와 셋째 줄뿐이다.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.datasets import make_regression

    # 모형이 잘 맞는 경우를 먼저 본다. 잔차 그림에 아무 무늬도 없어야 한다.
    np.random.seed(0)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    data = pd.DataFrame({'X': X.flatten(), 'y': y})

    X_with_const = sm.add_constant(data['X'])
    model = sm.OLS(data['y'], X_with_const).fit()

    data['Fitted'] = model.fittedvalues
    data['Residuals'] = model.resid

    # 왼쪽은 회귀 그림, 오른쪽은 잔차 그림이다. 늘 짝으로 본다.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

    ax0.scatter(data['X'], data['y'], alpha=0.7, label='Data Points')
    ax0.plot(data['X'], data['Fitted'], color='orange', label='Regression Line')
    ax0.set_title('Regression Plot')
    ax0.set_xlabel('Predictor (X)')
    ax0.set_ylabel('Response (y)')
    ax0.legend()

    # 잔차가 0 선 둘레에 무늬 없이 흩어져 있으면 좋다. 이 자료가 그렇다.
    ax1.scatter(data['Fitted'], data['Residuals'], alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals vs. Fitted Values Plot')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')

    plt.tight_layout()
    plt.show()
    ```

    ![가정이 성립할 때의 잔차](./img/residuals_48.png)

    참값과 세 판정을 모두 수로 확인한다.

    ```python
    import numpy as np
    import statsmodels.api as sm
    from sklearn.datasets import make_regression

    # make_regression 은 참 계수를 돌려줄 수 있다. 같은 씨앗이면 같은 자료다.
    np.random.seed(0)
    _, _, true_coef = make_regression(n_samples=100, n_features=1, noise=10, coef=True)
    n, p, sigma = 100, 2, 10.0

    print(f"참 기울기 = {true_coef:.4f},   적합값 = {model.params.iloc[1]:.4f}")
    print(f"참 절편   = 0.0000,   적합값 = {model.params.iloc[0]:.4f}")
    print(f"모집단 R^2 = {true_coef ** 2 / (true_coef ** 2 + sigma ** 2):.4f},"
          f"   표본 R^2 = {model.rsquared:.4f}")

    e = model.resid.values
    print(f"\n잔차제곱평균의 이론값 sigma^2 (n-p)/n = {sigma ** 2 * (n - p) / n:.3f}")
    print(f"실측 = {(e ** 2).mean():.3f}")
    print(f"sigma_hat = sqrt(SSE/(n-p)) = {np.sqrt(model.mse_resid):.4f}")
    print(f"|e| > 2 sigma_hat 인 점의 개수 = {(np.abs(e) > 2 * np.sqrt(model.mse_resid)).sum()} / {n}")

    # "무늬 없음" 을 눈 대신 수로 판정한다
    f = model.fittedvalues.values
    print(f"\ncorr(e, fitted)  = {np.corrcoef(e, f)[0, 1]:+.2e}   (항등적으로 0)")
    print(f"corr(|e|, fitted) = {np.corrcoef(np.abs(e), f)[0, 1]:+.4f}   (등분산)")
    aux = sm.OLS(e, sm.add_constant(np.column_stack([f, f ** 2]))).fit()
    print(f"e 를 fitted, fitted^2 에 회귀한 2차항의 p = {aux.pvalues[2]:.4f}   (선형성)")
    ```

    출력:

    ```
    참 기울기 = 42.3855,   적합값 = 42.6194
    참 절편   = 0.0000,   적합값 = -0.8142
    모집단 R^2 = 0.9473,   표본 R^2 = 0.9417

    잔차제곱평균의 이론값 sigma^2 (n-p)/n = 98.000
    실측 = 114.171
    sigma_hat = sqrt(SSE/(n-p)) = 10.7936
    |e| > 2 sigma_hat 인 점의 개수 = 2 / 100

    corr(e, fitted)  = -1.94e-15   (항등적으로 0)
    corr(|e|, fitted) = +0.0223   (등분산)
    e 를 fitted, fitted^2 에 회귀한 2차항의 p = 0.4335   (선형성)
    ```

    **참값과 적합값이 잘 맞는다.** 기울기 $42.619$ 대 참값 $42.3855$, 모집단 $R^2$ $0.9473$ 대 표본 $0.9417$이다.

    **잔차제곱평균은 이론값보다 크게 나왔다.** $98$을 예상했는데 $114.17$이다. $\hat\sigma = 10.79$로 참 $\sigma = 10$보다 $8\%$ 크다는 뜻이다. $\hat\sigma^2$의 상대표준편차가 $\sqrt{2/(n-p)} = \sqrt{2/98} = 0.143$이고 실제 어긋남이 $16.5\%$이므로 $1.15$ 표준편차 거리다. $\hat\sigma$ 자체의 상대표준편차는 그 절반인 $7\%$이니, **한 표본에서 $\hat\sigma$가 $\pm 7\%$씩 흔들리는 것이 정상이며**, 잔차 띠의 폭을 보고 $\sigma$를 눈으로 읽을 때 기억해 둘 일이다.

    $\pm 2\hat\sigma$ 밖의 점이 $2$개로 기대값 $5$보다 적다. $100$개 가운데 $5$개를 기대할 때의 표준편차가 $\sqrt{100 \times 0.0455 \times 0.9545} = 2.1$이므로 역시 흔한 일이다. 띠의 폭을 $\hat\sigma$로 재면 이미 부풀려진 척도를 쓰는 셈이라 바깥 점이 더 적게 세어지는 쪽으로 기운다.

    **세 판정이 모두 깨끗하다.** $\operatorname{corr}(e, \hat y)$가 $-1.9 \times 10^{-15}$로 항등적 0이고, $\operatorname{corr}(\lvert e\rvert, \hat y) = 0.0223$으로 깔때기가 없으며, 2차항의 $p = 0.4335$로 휘어짐도 없다. **이 세 수가 이 페이지의 기준선이다.** 아래 보기들에서 같은 수를 다시 재어 무엇이 어떻게 달라지는지 본다.

#### 그림 해석

1. **무작위 흩어짐**: 잔차가 0 주위에 무작위로 흩어져 있으면 선형성 가정이 성립함을 시사한다.
2. **일정한 폭**: 적합값 전 범위에 걸쳐 폭이 대체로 일정하면 등분산성을 뒷받침한다.
3. **패턴이나 깔때기 모양**: 곡선 패턴은 비선형성을, 깔때기 모양은 이분산을 나타낸다.

### 좋은 경우: 선형 자료에 선형모형

자료가 실제로 선형이고 선형모형을 적합하면 잔차가 일정한 분산으로 무작위로 흩어진다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 회귀·잔차 그림 함수. `generate_data()`는 $x_i \sim N(0,1)$, $y_i = 1 + 2x_i + 3u_i$, $u_i \sim N(0,1)$인 자료를 $n = 50$개 만든다. 곧 참 절편 $1$, 참 기울기 $2$, 참 $\sigma = 3$이다.

**(1)** 아래 세 함수를 만들어 적합하고, 추정값이 참값에서 **몇 표준오차** 떨어져 있는지 재시오. 참 표준오차는 $\sigma/\sqrt{S_{xx}}$로 계산할 수 있다.

**(2)** 그림을 "깔때기"로 읽어도 되는가. 적합값 사분위별 잔차의 폭과 Breusch-Pagan 검정으로 판정하시오.

</div>

??? success "풀이"

    **(1) 적합과 참값.** 이 자료는 등분산이므로 참 표준오차가 닫힌 꼴로 나온다.

    $$
    \operatorname{SE}(\hat\beta_1) = \frac{\sigma}{\sqrt{S_{xx}}},
    \qquad S_{xx} = \sum_i (x_i - \bar x)^2
    $$

    $\sigma = 3$을 알고 있으므로 $S_{xx}$만 자료에서 재면 된다.

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression

    plt.rcParams["font.sans-serif"] = ["NanumGothic", "Apple SD Gothic Neo", "Malgun Gothic"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    def generate_data(n=50, noise_level=3.0, seed=0):
        """기울기 2 의 직선 자료를 만든다. noise_level 로 잡음 크기를 조절한다."""
        np.random.seed(seed)
        x = np.random.randn(n, 1)
        x.sort(axis=0)
        noise = np.random.normal(0, 1, size=x.shape)
        y = (1 + 2 * x + noise_level * noise).reshape((-1,))
        return x, y

    def perform_regression(x, y):
        """최소제곱으로 적합하고 예측값까지 돌려준다."""
        model = LinearRegression()
        model.fit(x, y)
        y_pred = model.predict(x)
        return model, y_pred

    def plot_regression_and_residuals(x, y, y_pred):
        """회귀 그림과 잔차 그림을 나란히 그린다. 아래에서 되풀이해 쓴다."""
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 3))

        ax0.plot(x, y, 'o', label="Data")
        ax0.plot(x, y_pred, '-b', label="Predicted")
        ax0.set_title('Regression Plot')
        ax0.legend()

        ax1.plot(x, y - y_pred, 'o', label="Residuals")
        ax1.set_title('Residual Plot')
        ax1.legend()

        for ax in (ax0, ax1):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_position("zero")

        plt.tight_layout()
        plt.show()

    x, y = generate_data()
    model, y_pred = perform_regression(x, y)
    plot_regression_and_residuals(x, y, y_pred)
    ```

    ![직선 자료에 직선 적합](./img/residuals_99.png)

    ```python
    import numpy as np
    import statsmodels.api as sm

    e = y - y_pred
    xf = x.ravel()
    Sxx = ((xf - xf.mean()) ** 2).sum()

    print(f"참값 (절편, 기울기) = (1, 2)")
    print(f"적합값              = ({model.intercept_:.4f}, {model.coef_[0]:.4f})")
    print(f"참 SE(기울기) = sigma/sqrt(S_xx) = {3 / np.sqrt(Sxx):.4f}   (S_xx = {Sxx:.3f})")
    print(f"(b1 - 2)/SE = {(model.coef_[0] - 2) / (3 / np.sqrt(Sxx)):+.3f}")

    print(f"\n잔차제곱평균 이론 sigma^2 (n-p)/n = {9 * 48 / 50:.3f},  실측 {(e ** 2).mean():.3f}")

    # 정말 깔때기인가. 적합값 사분위로 잘라 잔차의 폭을 잰다.
    q = np.quantile(y_pred, [0, 0.25, 0.5, 0.75, 1.0])
    print()
    for k in range(4):
        sel = (y_pred >= q[k]) & (y_pred <= q[k + 1])
        print(f"  Q{k + 1}: n = {sel.sum()}, sd(e) = {e[sel].std(ddof=1):.4f}")
    print(f"\ncorr(|e|, fitted) = {np.corrcoef(np.abs(e), y_pred)[0, 1]:+.4f}")
    from statsmodels.stats.diagnostic import het_breuschpagan
    lm, p = het_breuschpagan(e, sm.add_constant(xf))[:2]
    print(f"Breusch-Pagan: LM = {lm:.4f}, p = {p:.4f}")

    # 씨앗을 바꾸어 400번 되풀이하면 기울기는 2 둘레에 모인다.
    slopes = []
    for s in range(400):
        xs, ys = generate_data(seed=s)
        slopes.append(perform_regression(xs, ys)[0].coef_[0])
    print(f"\n씨앗 400개의 기울기: 평균 {np.mean(slopes):.4f}, 표준편차 {np.std(slopes):.4f}")
    ```

    출력:

    ```
    참값 (절편, 기울기) = (1, 2)
    적합값              = (0.8125, 2.8868)
    참 SE(기울기) = sigma/sqrt(S_xx) = 0.3769   (S_xx = 63.340)
    (b1 - 2)/SE = +2.352

    잔차제곱평균 이론 sigma^2 (n-p)/n = 8.640,  실측 5.770

      Q1: n = 13, sd(e) = 1.6495
      Q2: n = 12, sd(e) = 2.7632
      Q3: n = 12, sd(e) = 3.2007
      Q4: n = 13, sd(e) = 2.1579

    corr(|e|, fitted) = +0.1255
    Breusch-Pagan: LM = 0.6560, p = 0.4180

    씨앗 400개의 기울기: 평균 1.9988, 표준편차 0.4458
    ```

    **(1)의 답: 이 표본은 꽤 빗나갔다.** 기울기 추정값이 $2.887$로 참값 $2$에서 참 표준오차의 $2.35$배 떨어져 있다. $\hat\sigma$도 작게 나왔다. 잔차제곱평균이 $5.770$으로 이론값 $8.640$의 $2/3$다. 까닭은 둘이다. 이 씨앗이 뽑은 잡음 $u_i$의 표본표준편차가 $0.876$으로 참값 $1$보다 작았고, 게다가 $\operatorname{corr}(u, x) = 0.384$로 $x$와 같은 방향으로 기울어 **그 몫이 기울기에 흡수되었다.** 잡음의 제곱합이 $338.3$(기대 $441$)인데 적합 뒤에는 $288.5$(기대 $432$)만 남은 것이 흡수된 양이다.

    씨앗을 $400$개 바꾸어 보면 기울기의 평균이 $1.9988$로 참값 $2$에 맞는다. **추정은 불편이고, 다만 이 한 표본이 $2.35$ 표준오차 쪽에 떨어졌을 뿐**이다(양측으로 $P = 0.019$, 쉰 번에 한 번쯤 일어나는 일이다). 씨앗 간 표준편차 $0.4458$은 이 표본의 참 SE $0.3769$보다 조금 큰데, $S_{xx}$ 자체가 표본마다 달라지기 때문이다.

    **(2)의 답: 깔때기가 아니다.** 적합값 사분위별 잔차의 표준편차가 $1.65, 2.76, 3.20, 2.16$으로 **단조가 아니다.** 가운데가 가장 넓고 양끝이 좁은 모양인데, 이것은 이분산이 아니라 $x$가 정규분포라 가운데에 점이 몰려 있는 데서 오는 모습이다. 구간당 $12$–$13$개로 센 표준편차의 상대오차가 $1/\sqrt{2 \times 12} = 20\%$나 되므로 이 정도 들쭉날쭉은 잡음이다.

    형식적 검정도 같은 말을 한다. $\operatorname{corr}(\lvert e\rvert, \hat y) = 0.126$이고 Breusch-Pagan이 $p = 0.418$이다. **자료를 등분산으로 만들었으니 옳은 판정이며, "오른쪽으로 갈수록 퍼진다"고 읽는다면 그림에서 없는 것을 본 것이다.** 점 $50$개의 산점도에서 폭의 변화를 눈으로 판정하는 일이 얼마나 미덥지 못한지 보여 주는 보기이기도 하다.

### 나쁜 경우: 다항 자료에 선형모형

자료가 다항 관계를 갖는데 선형모형만 적합하면 잔차에 뚜렷한 곡선 패턴이 나타난다. 선형성이 위배되었다는 신호이다.

<div class="storybox" markdown>

**그림으로 보는 통계학 1.** $8$ 분을 넘기지 않은 사람

<figure markdown>
  ![케플러의 얼굴과 별이 뜬 하늘 아래 망원경이 늘어선 천문대를 그린 1971년 루마니아 우표](./img/cn__stamp_romania1971.png){ width=270 align=right }
  <figcaption markdown>루마니아 $1971$ 년, 액면 $55$ 바니. 「$400$ ANI DE LA NAŞTERE」— 탄생 $400$ 주년 — 이라 적혀 있다. *(Wikimedia Commons, PD)*</figcaption>
</figure>

잔차에 남은 패턴을 그냥 넘기지 않아 모형을 통째로 바꾼 가장 유명한 사람이 **요하네스 케플러**다. 그때까지 천 년이 넘도록 행성은 **완전한 원**을 그린다고 여겨졌다. 원이 아니면 하늘이 아름답지 않다고 생각했기 때문이다.

케플러에게는 스승 튀코 브라헤가 평생 모은 관측 자료가 있었다. 망원경이 나오기 전이라 맨눈으로 잰 것인데도 그때까지 누구의 것보다 정확했다. 케플러는 화성의 자리를 원 궤도로 맞춰 보았고, 아무리 애써도 **각도로 $8$ 분**이 남았다.

$8$ 분은 보름달 지름의 $4$ 분의 $1$ 도 안 되는 각이다. 누구라도 관측 오차로 넘길 만했다. **그는 넘기지 않았다.** 자기 계산이 아니라 자료를 믿기로 하고 원을 버렸고, 궤도가 **타원**이며 태양이 그 **한 초점**에 있다는 것을 찾아냈다. 뒷날 그는 이렇게 적었다 — 이 $8$ 분이 천문학 전체를 다시 세우는 길을 열어 주었다고.

아래 보기 3 이 하는 일이 꼭 그것이다. 직선을 맞추었는데 잔차가 뚜렷한 곡선을 그리면, 그 곡선은 **잡음이 아니라 아직 적지 않은 항**이다. 케플러의 $8$ 분이 잔차그림의 패턴이고, 원에서 타원으로 옮긴 것이 모형을 바꾼 일이다. **타원은 아름다워서 고른 것이 아니라, 자료가 그렇게 말해서 받아들인 것이다.**

케플러는 궤도의 어디에 있는가를 묻는 문제에도, 하늘이 아니라 술통의 부피를 따지는 문제에도 같은 손을 댔다. 하던 일은 늘 같다 — 남이 눈대중으로 넘긴 자리를 끝까지 재어 본 것이다.

*확인한 곳 — 우표의 「POSTA ROMANA」·액면 「$55$ B」·「$400$ ANI DE LA NASTERE」·「$1971$」·「KEPLER」·도안가 이름 「VLASTO」는 위 사진에서 직접 읽었다 · 화성 궤도에 남은 $8$ 분의 오차와 튀코의 관측 자료는 [MacTutor 「Johannes Kepler」](https://mathshistory.st-andrews.ac.uk/Biographies/Kepler/) · [Wikipedia 「Kepler's laws of planetary motion — History」](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#History)*

</div>

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 이차 자료를 직선으로 맞추면. `generate_data(d=2)`는 $y_i = 1 + 2x_i + 3x_i^2 + 3u_i$를 만든다. $x_i \sim N(0,1)$이다.

**(1)** 이 자료에 직선을 맞추면 $n \to \infty$에서 기울기가 $2$, 절편이 $4$로 수렴함을 보이시오($E[x] = E[x^3] = 0$, $E[x^2] = 1$을 쓴다). 또 잔차가

$$
e_i = \underbrace{\big[3x_i^2 \text{ 를 } (1, x) \text{ 에 사영하고 남은 것}\big]}_{\text{곡률}} + \underbrace{\big[3u_i \text{ 를 } (1, x) \text{ 에 사영하고 남은 것}\big]}_{\text{잡음}}
$$

로 **정확히** 갈린다는 것을 보이시오.

**(2)** 두 조각을 실제로 계산해 잔차의 흩어짐 가운데 곡률이 차지하는 몫을 재시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 직선 적합의 극한 기울기는 모집단 공분산의 비다.

    $$
    \beta_1^* = \frac{\operatorname{Cov}(x, y)}{\operatorname{Var}(x)}
    = \frac{\operatorname{Cov}(x,\, 2x + 3x^2)}{1}
    = 2\operatorname{Var}(x) + 3\operatorname{Cov}(x, x^2)
    $$

    인데 $\operatorname{Cov}(x, x^2) = E[x^3] - E[x]E[x^2] = 0 - 0 = 0$이므로 $\beta_1^* = 2$다. **표준정규는 대칭이라 $x$와 $x^2$이 무상관이고, 그래서 이차항이 기울기를 전혀 건드리지 않는다.** 절편은

    $$
    \beta_0^* = E[y] - \beta_1^* E[x] = (1 + 0 + 3 \cdot 1) - 0 = 4
    $$

    이다. **기울기는 멀쩡하고 절편만 $1$에서 $4$로 밀린다.** 이차항의 평균 $3E[x^2] = 3$이 통째로 절편에 얹힌 것이다.

    잔차의 분해는 최소제곱이 선형사영이라는 데서 바로 나온다. $M$을 $(1, x)$ 위로의 사영을 빼는 행렬이라 하면 $\mathbf e = M\mathbf y$이고

    $$
    \mathbf y = \mathbf 1 + 2\mathbf x + 3\mathbf x^2 + 3\mathbf u
    \quad\Longrightarrow\quad
    \mathbf e = M\mathbf 1 + 2M\mathbf x + 3M\mathbf x^2 + 3M\mathbf u
    = 3M\mathbf x^2 + 3M\mathbf u
    $$

    이다. $M\mathbf 1 = M\mathbf x = \mathbf 0$이기 때문이다. **앞의 항이 곡률의 몫이고 뒤의 항이 잡음의 몫이며, 둘의 합이 잔차와 정확히 같다.** 앞의 항은 $x$의 결정함수이므로 잔차 그림에서 **매끄러운 포물선 모양**으로 나타난다. 이것이 "U자"의 정체다.

    **(2) 수치적으로.**

    ```python
    def generate_data(n=50, noise_level=3.0, d=1, seed=0):
        """차수 d 인 다항 자료를 만든다. d=1 이면 앞과 같은 직선이다."""
        np.random.seed(seed)
        x = np.random.randn(n, 1)
        x.sort(axis=0)
        noise = np.random.normal(0, 1, size=x.shape)
        y = (1 + np.sum([(k+1) * x**k for k in range(1, d+1)], axis=0) + noise_level * noise).reshape((-1,))
        return x, y

    # 이차 자료를 직선으로 맞춘다. 회귀 그림만 보면 그럴듯해 보이지만
    # 잔차 그림에는 굽은 무늬가 또렷하게 남는다. 잔차 그림을 보는 까닭이다.
    x, y = generate_data(d=2)
    model, y_pred = perform_regression(x, y)
    plot_regression_and_residuals(x, y, y_pred)
    ```

    ![이차 관계에서의 잔차](./img/residuals_147.png)

    잔차가 U자를 그린다. 이제 (1)의 분해로 그 U자의 크기를 잰다.

    ```python
    import numpy as np
    import statsmodels.api as sm

    xf = x.ravel()
    e = y - y_pred

    print(f"직선 적합 (절편, 기울기) = ({model.intercept_:.4f}, {model.coef_[0]:.4f})")
    print(f"모집단 극한값            = (4, 2)")
    print(f"표본 적률: E[x] = {xf.mean():+.4f}, E[x^2] = {(xf ** 2).mean():.4f}, "
          f"E[x^3] = {(xf ** 3).mean():+.4f}")

    # 잔차를 두 조각으로 정확히 가른다: 곡률의 몫과 잡음의 몫
    curv = 3 * xf ** 2
    curv_part = curv - sm.OLS(curv, sm.add_constant(xf)).fit().fittedvalues
    noise_part = e - curv_part
    print(f"\n두 조각의 합이 잔차와 같은가: 최대 오차 {np.abs(e - (curv_part + noise_part)).max():.1e}")
    print(f"곡률 조각의 표준편차 = {curv_part.std():.4f}")
    print(f"잡음 조각의 표준편차 = {noise_part.std():.4f}")
    print(f"잔차의 분산 가운데 곡률이 차지하는 몫 = "
          f"{np.corrcoef(e, curv_part)[0, 1] ** 2:.4f}")
    print(f"곡률 조각의 범위 = [{curv_part.min():.3f}, {curv_part.max():.3f}]")
    ```

    출력:

    ```
    직선 적합 (절편, 기울기) = (4.6161, 3.2857)
    모집단 극한값            = (4, 2)
    표본 적률: E[x] = +0.1406, E[x^2] = 1.2866, E[x^3] = +0.3493

    두 조각의 합이 잔차와 같은가: 최대 오차 0.0e+00
    곡률 조각의 표준편차 = 4.6361
    잡음 조각의 표준편차 = 2.4021
    잔차의 분산 가운데 곡률이 차지하는 몫 = 0.8136
    곡률 조각의 범위 = [-3.816, 16.768]
    ```

    **분해가 정확하다.** 두 조각의 합과 잔차의 차이가 **정확히 0**이다. 근사가 아니라 항등식이다.

    **잔차의 $81.4\%$가 곡률이다.** 곡률 조각의 표준편차가 $4.636$으로 잡음 조각의 $2.402$보다 두 배 가까이 크다. 잔차 그림에서 보이는 U자는 "약간의 경향"이 아니라 **잔차를 지배하는 성분**이며, 그래서 $n = 50$뿐인데도 눈에 또렷하다. 곡률 조각이 $-3.82$에서 $+16.77$까지 가는데, 음의 범위가 짧고 양의 범위가 긴 **비대칭 U자**인 것도 그림과 맞는다. $x$가 0 근처일 때 $3x^2$이 바닥에 깔리고 $\lvert x\rvert$가 커지면 급히 치솟기 때문이다.

    **극한값과는 어느 정도만 맞는다.** 적합값이 $(4.616,\ 3.286)$으로 극한값 $(4,\ 2)$에서 꽤 떨어져 있다. 표본 적률을 보면 이유가 보인다. $n = 50$에서 $E[x] = 0.141$, $E[x^3] = 0.349$로 둘 다 0이 아니다. 극한값 유도가 쓴 대칭성이 이 표본에서는 깨져 있고, 그 비대칭이 $\operatorname{Cov}(x, x^2) \ne 0$을 만들어 기울기를 끌어올렸다. **"$x^2$은 기울기를 건드리지 않는다"는 모집단에서만 성립하며, 유한한 표본에서는 그만큼 샌다.**

#### 선형 대 이차 잔차 비교

모형 오설정을 더 잘 진단하려면 경쟁 모형들의 잔차를 직접 비교하는 것이 유용하다. 이차 관계를 따르는 자료를 생각하자.

<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 평활선으로 본 굽은 잔차. 자료는 $x \sim U(-3,3)$, $y = 2 + 0.5x - 1.5x^2 + \varepsilon$, $\varepsilon \sim N(0,1)$이다.

**(1)** 직선 적합의 극한 계수와 **모집단 $R^2$**을 두 모형 각각에 대해 구하시오. $x \sim U(-3,3)$이면 $\operatorname{Var}(x) = 3$, $E[x^4] = 81/5$이다.

**(2)** 두 모형을 적합해 (1)과 맞추시오. 표본 $R^2 = 0.083$이 모집단 값 $0.042$의 두 배인데, 이것을 이상하게 보아야 하는가.

</div>

??? success "풀이"

    **(1) 해석적으로.** $x \sim U(-3,3)$이므로 $E[x] = 0$, $\operatorname{Var}(x) = 6^2/12 = 3$, $E[x^2] = 3$, $E[x^3] = 0$(대칭), $E[x^4] = 3^4/5 = 16.2$다. 따라서

    $$
    \operatorname{Var}(x^2) = E[x^4] - (E[x^2])^2 = 16.2 - 9 = 7.2
    $$

    이다. 직선 적합의 극한 기울기는 $\operatorname{Cov}(x, x^2) = E[x^3] = 0$이므로

    $$
    \beta_1^* = \frac{\operatorname{Cov}(x,\, 0.5x - 1.5x^2)}{\operatorname{Var}(x)} = 0.5
    $$

    이고 절편은 $\beta_0^* = E[y] = 2 - 1.5 E[x^2] = 2 - 4.5 = -2.5$다.

    모집단 $R^2$에는 $\operatorname{Var}(y)$가 필요하다. $x$와 $x^2$이 무상관이므로 교차항이 없고

    $$
    \operatorname{Var}(y) = 0.5^2 \operatorname{Var}(x) + 1.5^2 \operatorname{Var}(x^2) + 1
    = 0.75 + 16.2 + 1 = 17.95
    $$

    이다. 직선 모형이 설명하는 것은 $0.5x$ 부분뿐이므로

    $$
    R^2_{\text{직선}} = \frac{0.75}{17.95} = 0.0418,
    \qquad
    R^2_{\text{이차}} = \frac{0.75 + 16.2}{17.95} = 0.9443
    $$

    이다. **직선 모형이 못 잡는 $16.2$가 전부 $x^2$ 항이고, 그것이 $\operatorname{Var}(y)$의 $90\%$다.**

    **(2) 수치적으로.**

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.nonparametric.smoothers_lowess import lowess

    # 같은 이야기를 평활선까지 얹어 더 또렷하게 본다.
    np.random.seed(42)
    x = np.random.uniform(-3, 3, 100)
    y_true = 2 + 0.5 * x - 1.5 * x**2
    y = y_true + np.random.normal(0, 1, len(x))

    # 왼쪽에 쓸 선형 모형
    X_linear = sm.add_constant(x)
    model_linear = sm.OLS(y, X_linear).fit()
    residuals_linear = model_linear.resid
    y_pred_linear = model_linear.fittedvalues

    # 오른쪽에 쓸 이차 모형
    X_quad = sm.add_constant(np.column_stack([x, x**2]))
    model_quad = sm.OLS(y, X_quad).fit()
    residuals_quad = model_quad.resid
    y_pred_quad = model_quad.fittedvalues

    # 잔차에 평활선을 얹어 그린다
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 선형 모형의 잔차
    ax1.scatter(y_pred_linear, residuals_linear, alpha=0.6)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)

    # 평활선이 굽어 있으면 남은 구조가 있다는 신호다.
    lowess_result = lowess(residuals_linear, y_pred_linear, frac=0.3)
    ax1.plot(lowess_result[:, 0], lowess_result[:, 1], 'b-', linewidth=2.5,
             label='LOWESS Trend')

    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Linear Model: Clear Non-linearity Pattern')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 이차항을 넣고 나면 평활선이 평평해진다. 무늬가 사라진 것이다.
    ax2.scatter(y_pred_quad, residuals_quad, alpha=0.6)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)

    # 평활선을 얹는다
    lowess_result_quad = lowess(residuals_quad, y_pred_quad, frac=0.3)
    ax2.plot(lowess_result_quad[:, 0], lowess_result_quad[:, 1], 'b-', linewidth=2.5,
             label='LOWESS Trend')

    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Quadratic Model: Non-linearity Removed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 요약 비교
    print("Model Comparison:")
    print(f"Linear Model R²:    {model_linear.rsquared:.4f}")
    print(f"Quadratic Model R²: {model_quad.rsquared:.4f}")
    print(f"Linear Model RSS:    {np.sum(residuals_linear**2):.2f}")
    print(f"Quadratic Model RSS: {np.sum(residuals_quad**2):.2f}")
    ```

    출력:

    ```
    Model Comparison:
    Linear Model R²:    0.0830
    Quadratic Model R²: 0.9534
    Linear Model RSS:    1530.56
    Quadratic Model RSS: 77.72
    ```

    ![선형 모형과 이차 모형의 잔차와 평활선](./img/residuals_166.png)

    (1)의 극한값과 맞춰 본다.

    ```python
    import numpy as np
    import statsmodels.api as sm

    # 모집단 적률 (x ~ U(-3,3))
    VarX = 36 / 12
    EX4 = 3 ** 4 / 5
    VarX2 = EX4 - VarX ** 2
    VarY = 0.5 ** 2 * VarX + 1.5 ** 2 * VarX2 + 1.0
    print(f"Var(x) = {VarX:.4f},  Var(x^2) = {VarX2:.4f},  Var(y) = {VarY:.4f}")
    print(f"직선 적합의 극한 (절편, 기울기) = ({2 - 1.5 * VarX:.4f}, {0.5:.4f})")
    print(f"실제 적합                      = ({model_linear.params[0]:.4f}, "
          f"{model_linear.params[1]:.4f})")
    print(f"\n모집단 R^2:  직선 {0.5 ** 2 * VarX / VarY:.4f},"
          f"  이차 {(0.5 ** 2 * VarX + 1.5 ** 2 * VarX2) / VarY:.4f}")
    print(f"표본   R^2:  직선 {model_linear.rsquared:.4f},  이차 {model_quad.rsquared:.4f}")

    # 표본 R^2 이 0.083 인 것이 이상한가. 같은 모형에서 2000번 뽑아 본다.
    r2s = []
    for s in range(2000):
        rng = np.random.default_rng(s)
        xx = rng.uniform(-3, 3, 100)
        yy = 2 + 0.5 * xx - 1.5 * xx ** 2 + rng.normal(0, 1, 100)
        r2s.append(sm.OLS(yy, sm.add_constant(xx)).fit().rsquared)
    r2s = np.array(r2s)
    print(f"\n직선 R^2 의 모의분포: 평균 {r2s.mean():.4f}, 표준편차 {r2s.std():.4f}")
    print(f"0.0830 은 그 분포의 {(r2s < model_linear.rsquared).mean():.1%} 분위")
    ```

    출력:

    ```
    Var(x) = 3.0000,  Var(x^2) = 7.2000,  Var(y) = 17.9500
    직선 적합의 극한 (절편, 기울기) = (-2.5000, 0.5000)
    실제 적합                      = (-2.7512, 0.6626)

    모집단 R^2:  직선 0.0418,  이차 0.9443
    표본   R^2:  직선 0.0830,  이차 0.9534

    직선 R^2 의 모의분포: 평균 0.0553, 표준편차 0.0493
    0.0830 은 그 분포의 75.9% 분위
    ```

    **유도한 극한값이 맞는다.** 직선 적합이 $(-2.751,\ 0.663)$으로 극한값 $(-2.5,\ 0.5)$ 근처이고, 모집단 $R^2$ $0.0418$과 $0.9443$이 표본값 $0.0830$과 $0.9534$와 같은 자리에 있다.

    **표본 $R^2 = 0.083$이 모집단 값의 두 배인 것은 이상하지 않다.** 같은 모형에서 $2000$번 뽑아 보면 직선 $R^2$의 평균이 $0.0553$, 표준편차가 $0.0493$이고 $0.0830$은 그 분포의 $76$번째 백분위수다. **$R^2$이 0 근처일 때는 표집분포가 오른쪽으로 길게 늘어져 평균이 모집단 값보다 위에 놓인다.** 모형에 아무 설명력이 없어도 $R^2$의 기댓값이 $1/(n-1)$만큼 양수인 것과 같은 현상이다. 작은 $R^2$을 보고 "그래도 $4\%$는 설명한다"고 말하기 전에 이 분포를 떠올려야 한다.

    **핵심은 $R^2$이 아니라 잔차의 무늬다.** $R^2 = 0.083$이라는 수는 "얼마나 못 맞히는가"만 말하고 "왜 못 맞히는가"는 말하지 않는다. 그 까닭을 알려 주는 것이 잔차 그림이고, 거기에 얹은 평활선이다.

**핵심 통찰**: 잔차를 지나는 LOWESS(국소가중 산점도 평활) 평활곡선이 위배 패턴을 뚜렷이 드러낸다. 선형모형에서는 이 곡선이 0 아래로 내려갔다가 위로 올라가며, 체계적인 과소예측과 과대예측이 일어나고 있음을 나타낸다. 이차 모형의 잔차는 무작위로 흩어져 비선형성의 형태가 제대로 포착되었음을 보여준다. $R^2$가 0.083에서 0.953으로 뛰고 잔차제곱합이 1530.56에서 77.72로 20분의 1 수준이 되는 것이 그 차이를 수치로 보여준다.

### 해결: 다항회귀

참 자료생성과정에 맞추어 다항 특성을 추가하면 잔차의 패턴이 해소된다.

<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> 다항회귀로 고치기. 보기 3의 자료에 이번에는 $x$와 $x^2$을 모두 넣는다.

**(1)** 보기 3에서 잔차를 $3M\mathbf x^2 + 3M\mathbf u$로 갈랐다. 이제 $\mathbf x^2$까지 설계행렬에 넣으면 **곡률 조각이 정확히 0이 된다**는 것을 보이시오.

**(2)** 두 적합의 잔차를 견주어 (1)을 확인하고, $\hat\sigma$가 참값 $3$에 가까워지는지 보시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 이제 사영을 빼는 행렬 $M_2$가 $(\mathbf 1, \mathbf x, \mathbf x^2)$ **셋 모두**를 지운다. 그러므로

    $$
    \mathbf e = M_2 \mathbf y = M_2\big(\mathbf 1 + 2\mathbf x + 3\mathbf x^2 + 3\mathbf u\big) = 3 M_2 \mathbf u
    $$

    로 **곡률 조각이 통째로 사라지고 잡음의 몫만 남는다.** 보기 3에서 잔차의 $81\%$를 차지하던 성분이 설계행렬 안으로 들어갔기 때문이다. 이것이 "다항회귀도 선형모형"이라는 말의 쓸모다. 설계행렬에 열을 하나 더 쌓는 것만으로 곡률을 완전히 흡수한다.

    따라서 $\hat\sigma^2 = \mathbf e^\top\mathbf e/(n-3)$이 이제 $\sigma^2 = 9$를 겨냥한다. 보기 3의 직선 적합에서는 곡률이 잔차에 남아 있었으므로 $\hat\sigma$가 참값보다 크게 나왔을 것이다. **모형을 잘못 세우면 $\hat\sigma$가 부풀고, 그 부푼 $\hat\sigma$로 계산한 모든 표준오차와 구간이 함께 부푼다.**

    **(2) 수치적으로.**

    ```python
    def perform_regression(x, y, d=1):
        """차수 d 의 다항회귀. x, x^2, ... 를 열로 쌓아 넣기만 하면 된다.

        항이 x 의 거듭제곱일 뿐 계수에 대해서는 여전히 선형이므로,
        최소제곱을 그대로 쓸 수 있다. 다항회귀도 선형모형인 까닭이다.
        """
        x_poly = np.concatenate([x**k for k in range(1, d+1)], axis=1)
        model = LinearRegression()
        model.fit(x_poly, y)
        y_pred = model.predict(x_poly)
        return model, y_pred

    x, y = generate_data(d=2)
    model, y_pred = perform_regression(x, y, d=2)
    plot_regression_and_residuals(x, y, y_pred)
    ```

    ![이차 자료에 이차 적합](./img/residuals_250.png)

    왼쪽 회귀 그림의 곡선이 점구름을 따라 휘고, 오른쪽 잔차 그림에는 보기 3의 U자가 없다. 수로 확인한다.

    ```python
    import numpy as np
    import statsmodels.api as sm

    xf = x.ravel()
    e2 = y - y_pred                                        # 이차 적합의 잔차
    _, y_lin = perform_regression(x, y, d=1)
    e1 = y - y_lin                                         # 보기 3 의 직선 적합 잔차

    print(f"이차 적합 계수 = (절편 {model.intercept_:.4f}, "
          f"x {model.coef_[0]:.4f}, x^2 {model.coef_[1]:.4f})   참값 (1, 2, 3)")
    print(f"\n{'':>10}{'직선':>12}{'이차':>12}")
    print(f"{'RSS':>10}{(e1 ** 2).sum():>12.2f}{(e2 ** 2).sum():>12.2f}")
    print(f"{'sigma_hat':>10}{np.sqrt((e1 ** 2).sum() / 48):>12.4f}"
          f"{np.sqrt((e2 ** 2).sum() / 47):>12.4f}   (참값 3)")

    # 보기 3 의 곡률 조각이 정말 사라졌는가
    curv = 3 * xf ** 2
    print(f"\ncorr(e, x^2):  직선 {np.corrcoef(e1, curv)[0, 1]:+.4f},"
          f"  이차 {np.corrcoef(e2, curv)[0, 1]:+.2e}")
    aux1 = sm.OLS(e1, sm.add_constant(np.column_stack([xf, xf ** 2]))).fit()
    print(f"e 를 (1, x, x^2) 에 회귀한 R^2:  직선 {aux1.rsquared:.4f}")
    ```

    출력:

    ```
    이차 적합 계수 = (절편 0.5384, x 2.8580, x^2 3.2162)   참값 (1, 2, 3)

                        직선          이차
           RSS     1518.09      282.93
     sigma_hat      5.6238      2.4535   (참값 3)

    corr(e, x^2):  직선 +0.8978,  이차 +4.02e-16
    e 를 (1, x, x^2) 에 회귀한 R^2:  직선 0.8136
    ```

    **곡률이 정확히 사라졌다.** $\operatorname{corr}(e, x^2)$이 직선 적합에서 $+0.898$이었던 것이 이차 적합에서 $4 \times 10^{-16}$이 된다. 근사가 아니라 정규방정식이 강제하는 0이다. $\mathbf x^2$이 설계행렬의 열이 된 순간 잔차는 그와 직교할 수밖에 없다.

    **$\hat\sigma$가 제자리로 돌아온다.** 직선 적합의 $\hat\sigma = 5.624$는 참값 $3$의 거의 두 배였는데, 이차 적합에서 $2.454$가 된다. 보기 2에서 본 대로 이 씨앗의 잡음이 작게 뽑혀 참값보다 조금 아래이지만, 직선 적합의 $5.62$와는 비교가 되지 않는다. **RSS도 $1518$에서 $283$으로 다섯 분의 일이 되었고**, 줄어든 $1235$가 바로 곡률이 차지하고 있던 몫이다. 보기 3에서 잰 "잔차의 $81.4\%$가 곡률"과 맞춰 보면 $1518 \times 0.814 = 1236$으로 들어맞는다.

    계수도 참값에 가깝다. $(0.538,\ 2.858,\ 3.216)$ 대 $(1,\ 2,\ 3)$인데, $n = 50$에 $\sigma = 3$이라 이 정도 흔들림은 자연스럽다.

!!! tip "참고"
    [Transforming nonlinear data (Khan Academy)](https://www.khanacademy.org/math/ap-statistics/bivariate-data-ap/assessing-fit-least-squares-regression/v/transforming-nonlinear-data)

<div class="storybox" markdown>

**그림으로 보는 통계학 2.** 가로축을 하나 잘못 고르면

이 절의 그림은 모두 가로축이 **적합값** $\hat y$ 다. 「잔차–적합값 그림」이라는 이름부터 그렇다. 그런데 왜 하필 적합값인가. 손에 있는 것 가운데 가장 자연스러워 보이는 것은 오히려 **관측값** $y$ 인데 말이다.

까닭이 식 하나에 있다. 절편이 있는 최소제곱에서 정규방정식은 $\mathbf X^\top \mathbf e = \mathbf 0$ 이고, 그래서

$$
\operatorname{Cov}(e,\ \hat y) = 0
$$

이 **언제나 정확히** 성립한다. 잔차 대 적합값 그림에 직선을 맞추면 기울기가 $0$ 이라는 뜻이다. 그러므로 그 그림에 보이는 굽음이나 나팔 모양은 **남김없이 자료가 말하는 것**이다. 가로축이 공짜로 만들어 낸 무늬가 섞여 있지 않다.

관측값을 놓으면 사정이 다르다. $y = \hat y + e$ 이므로

$$
\operatorname{Cov}(e,\ y) = \operatorname{Cov}(e,\ \hat y) + \operatorname{Var}(e) = \operatorname{Var}(e)
$$

이고, 따라서 잔차를 관측값에 회귀한 기울기는

$$
\frac{\operatorname{Var}(e)}{\operatorname{Var}(y)} = 1 - R^2,
\qquad
\operatorname{corr}(e,\ y) = \sqrt{1 - R^2}
$$

다. **$0$ 이 아니다. 그리고 그 값은 자료가 아니라 $R^2$ 이 정한다.**

![같은 적합을 적합값과 관측값에 대해 각각 그린 모습과, 관측값 쪽 기울기가 1-R² 위에 놓이는 모습](./img/residual_vs_observed.png)

왼쪽은 **같은 적합**을 두 가로축으로 그린 것이다. $R^2 = 0.743$ 인 이 적합에서 적합값 쪽 기울기는 $-3.4 \times 10^{-16}$ — 반올림 오차다. 관측값 쪽은 $0.2568$ 이고, $1 - R^2 = 0.2568$ 이다. 소수 넷째 자리까지 같다.

오른쪽이 그것이 우연이 아님을 보인다. 목표 $R^2$ 를 바꿔 가며 자료를 $400$ 번씩 새로 만들어 기울기를 재면

| 잰 $R^2$ | 잔차 대 관측값의 기울기 | $1 - R^2$ |
|---|---|---|
| $0.1031$ | $0.8969$ | $0.8969$ |
| $0.2999$ | $0.7001$ | $0.7001$ |
| $0.5040$ | $0.4960$ | $0.4960$ |
| $0.6999$ | $0.3001$ | $0.3001$ |
| $0.9003$ | $0.0997$ | $0.0997$ |

로 네 자리까지 포개진다.

그래서 이 그림을 잘못 그렸을 때 벌어지는 일이 고약하다. 오른쪽으로 올라가는 띠가 또렷하게 보이고, **「모형이 큰 값을 과소예측한다」**는 읽기가 자연스럽게 따라온다. 그런데 그 띠는 모형의 결함이 아니라 $R^2$ 을 **다시 적은 것**이다. 모형이 완벽히 옳아도 나오고, 나쁜 모형일수록 **더 또렷해지고**($R^2$ 이 작으니까), 모형을 고쳐 $R^2$ 을 올리면 저절로 옅어진다. 곧 그 그림은 **모형에 대해 아무것도 말해 주지 않으면서 말하는 것처럼 보인다.**

덧붙여 둘 것이 있다. 잘못된 가로축은 **관측값 하나뿐**이다. 설명변수에 대해 그리는 것은 괜찮다 — $\mathbf X^\top\mathbf e = \mathbf 0$ 이 $\operatorname{Cov}(e,\ x_j) = 0$ 도 함께 주기 때문이다. 잔차를 설명변수에 대해 그리는 진단(성분+잔차 그림 따위)이 성립하는 까닭이 그것이다. 가로축에 놓아도 되는 것과 안 되는 것을 가르는 것은 취향이 아니라 **정규방정식**이다.

</div>

---

## 4. 척도–위치 그림

척도–위치 그림은 표준화 잔차 절댓값의 제곱근을 적합값에 대해 그려 **등분산성**을 확인한다. 그림 전체에 걸쳐 폭이 일정하면 상수분산을 뒷받침한다.

<div class="exbox" markdown>

**보기 6.** <span class="diff easy" title="쉬움"></span> 척도–위치 그림까지. 보기 1과 같은 자료에 그림 셋을 나란히 그린다.

**(1)** 등분산이 성립하면 척도–위치 그림의 점들이 **어느 높이**에 모여야 하는가. $Z \sim N(0,1)$에 대해 $E[\lvert Z\rvert^s] = 2^{s/2}\Gamma\!\big(\tfrac{s+1}{2}\big)/\sqrt\pi$를 써서 $E[\sqrt{\lvert Z\rvert}\,]$를 닫힌 꼴로 구하시오.

**(2)** 그림을 그리고 (1)의 높이를 확인하시오. 제곱근이 치우침을 없앴는지도 왜도로 재시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 주어진 적률 공식에 $s = 1/2$를 넣는다.

    $$
    E\big[\sqrt{\lvert Z \rvert}\,\big] = \frac{2^{1/4}\,\Gamma(3/4)}{\sqrt\pi}
    $$

    $\Gamma(3/4) = 1.225417$, $2^{1/4} = 1.189207$, $\sqrt\pi = 1.772454$이므로

    $$
    E\big[\sqrt{\lvert Z \rvert}\,\big] = \frac{1.189207 \times 1.225417}{1.772454} = 0.822179
    $$

    이다. **등분산이면 척도–위치 그림의 점들이 높이 $0.822$ 둘레에 평평한 띠를 이룬다.** 이 수를 알아 두면 그림의 세로축을 읽을 때 기준이 생긴다. 띠가 그보다 높은 쪽으로 기울면 그 자리의 $\sigma$가 크다는 뜻이고, 세로축이 $\sqrt\sigma$ 척도이므로 **높이의 비를 제곱하면 $\sigma$의 비**가 된다([11.5절 보기 5](../assumptions/checking_homoscedasticity.md)).

    같은 공식으로 $s = 1$을 넣으면 $E[\lvert Z\rvert] = \sqrt{2/\pi} = 0.7979$이고 $s = 2$면 $E[Z^2] = 1$이다. 세 변환 가운데 제곱근만이 분포의 치우침을 거의 없앤다는 것은 (2)에서 왜도로 확인한다.

    **(2) 수치적으로.**

    ```python
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.datasets import make_regression

    np.random.seed(0)
    X, y = make_regression(n_samples=100, n_features=1, noise=10)
    data = pd.DataFrame({'X': X.flatten(), 'y': y})

    X_with_const = sm.add_constant(data['X'])
    model = sm.OLS(data['y'], X_with_const).fit()

    data['Fitted'] = model.fittedvalues
    data['Residuals'] = model.resid
    # 척도-위치 그림을 위해 잔차를 표준화하고 절댓값의 제곱근을 취한다.
    data['Standardized Residuals'] = data['Residuals'] / np.std(data['Residuals'])
    data['Sqrt Abs Standardized Residuals'] = np.sqrt(np.abs(data['Standardized Residuals']))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 5))

    # 회귀 그림
    ax0.scatter(data['X'], data['y'], alpha=0.7, label='Data Points')
    ax0.plot(data['X'], data['Fitted'], color='orange', label='Regression Line')
    ax0.set_title('Regression Plot')
    ax0.set_xlabel('Predictor (X)')
    ax0.set_ylabel('Response (y)')
    ax0.legend()

    # 잔차 대 적합값
    ax1.scatter(data['Fitted'], data['Residuals'], alpha=0.7)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_title('Residuals vs. Fitted Values Plot')
    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')

    # 척도-위치 그림은 등분산성만 본다. 부호를 없앴으므로 점들이 이루는 띠의
    # 높이가 일정한지만 보면 된다.
    ax2.scatter(data['Fitted'], data['Sqrt Abs Standardized Residuals'], alpha=0.7)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Scale-Location Plot')
    ax2.set_xlabel('Fitted Values')
    ax2.set_ylabel(r'$\sqrt{|\text{Standardized Residuals}|}$')

    plt.tight_layout()
    plt.show()
    ```

    ![잔차 진단 종합](./img/residuals_270.png)

    세 그림을 함께 보면 어느 가정이 어디서 깨지는지 한눈에 들어온다. 오른쪽 그림의 높이를 (1)의 값과 맞춘다.

    ```python
    import math
    import numpy as np
    from scipy.stats import skew
    from statsmodels.nonparametric.smoothers_lowess import lowess

    theory = 2 ** 0.25 * math.gamma(0.75) / math.sqrt(math.pi)
    v = data['Sqrt Abs Standardized Residuals'].values
    print(f"E[sqrt|Z|] = 2^(1/4) Gamma(3/4) / sqrt(pi) = {theory:.6f}")
    print(f"그림의 세로축 평균                        = {v.mean():.6f}")
    print(f"(n = {len(v)}, 표준오차 = {v.std(ddof=1) / math.sqrt(len(v)):.6f})")

    # 띠가 평평한가: 평활선의 양 끝
    f = data['Fitted'].values
    sm_line = lowess(v, f, frac=0.6)
    print(f"\n평활선 왼쪽 끝 {sm_line[0, 1]:.4f}  →  오른쪽 끝 {sm_line[-1, 1]:.4f}")
    print(f"제곱한 비 (sd 의 비로 읽는다) = {(sm_line[-1, 1] / sm_line[0, 1]) ** 2:.4f}")

    # 제곱근이 치우침을 없앴는가
    print(f"\n왜도:  |r| = {skew(np.abs(data['Standardized Residuals'].values)):+.4f},"
          f"  sqrt|r| = {skew(v):+.4f}")
    ```

    출력:

    ```
    E[sqrt|Z|] = 2^(1/4) Gamma(3/4) / sqrt(pi) = 0.822179
    그림의 세로축 평균                        = 0.843058
    (n = 100, 표준오차 = 0.033396)

    평활선 왼쪽 끝 0.7258  →  오른쪽 끝 0.6442
    제곱한 비 (sd 의 비로 읽는다) = 0.7876

    왜도:  |r| = +0.5204,  sqrt|r| = -0.0001
    ```

    **유도한 높이 $0.822179$가 맞는다.** 실측 평균이 $0.843058$로 표준오차 $0.0334$의 $0.63$배 안에 들어온다. 그림의 점구름이 어느 높이에 모여야 하는지를 **자료를 보기 전에** 알고 있었던 셈이다.

    **띠는 평평하다.** 평활선이 $0.726$에서 $0.644$로 조금 내려가는데, 제곱한 $0.788$이 $\sigma$의 비다. 곧 오른쪽 끝의 $\sigma$가 왼쪽의 $79\%$라는 말인데, 보기 1에서 Breusch-Pagan 대신 쓴 $\operatorname{corr}(\lvert e\rvert, \hat y) = 0.022$가 이미 "이 기울기는 잡음"이라고 말해 주었다. **등분산 자료에서도 평활선은 이 정도 기운다**는 것을 기억해 두어야 진짜 이분산과 구별할 수 있다. [11.5절 보기 5](../assumptions/checking_homoscedasticity.md)의 이분산 자료에서는 같은 비가 $7.90$이었다.

    **제곱근이 치우침을 없앴다.** $\lvert r\rvert$의 왜도가 $+0.520$인데 $\sqrt{\lvert r\rvert}$의 왜도는 $-0.0001$이다. 표본 하나에서 이렇게까지 0에 붙은 것은 우연이지만, 방향은 분명하다. 반정규는 오른쪽으로 치우쳐 있고 제곱근이 그것을 거의 대칭으로 펴 준다. **치우친 세로축에 평활선을 얹으면 큰 값 몇 개가 선을 끌어올려 없는 추세를 만들어 낸다.** 제곱근을 쓰는 이유가 그것이다.

### 왜 제곱근을 쓰는가

등분산 아래에서 $r_i$ 는 대략 표준정규이므로 $\lvert r_i \rvert$ 는 **반정규**가 되어 0 에서
잘린 채 오른쪽으로 치우친다. 치우친 세로축에 평활선을 얹으면 추세가 아닌 치우침을 읽게 된다.
제곱근은 그 치우침을 거의 없애는 변환이고, 그래서 세로축은 $\sqrt{\sigma_i}$ 에 비례한다 —
**평활선 두 값의 비를 제곱하면 그 자리의 $\sigma$ 비**가 된다.

세 변환($\lvert r_i \rvert$ · $\sqrt{\lvert r_i \rvert}$ · $r_i^2$)의 왜도를 재어 이것을 수로 확인하고,
평활선이 올라간 크기를 $\sigma$ 의 비로 번역해 참값과 맞추는 것은 11.5절
[선형회귀의 등분산성 확인](../assumptions/checking_homoscedasticity.md) 보기 5 다.

---

## 5. 가정 위배에 대한 대처

잔차 분석에서 위배가 드러나면

- **변수변환**: 종속변수나 독립변수에 로그나 제곱근 변환을 적용해 비선형성과 이분산에 대처한다.
- **가중최소제곱(WLS)**: 분산에 따라 관측값마다 다른 가중치를 주어 비상수 분산을 직접 다룬다.
- **로버스트 회귀**: 이상점의 영향을 최소화해 가정 이탈에 더 견고하게 만든다.
- **다항 특성**: 잔차그림이 비선형성을 시사하면 다항 항을 추가한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 잔차그림에서 잔차가 뚜렷한 패턴 없이 0 주위에 무작위로 흩어져 있다. 모형 가정에 대해 무엇을 결론지을 수 있는가? 이 그림이 다루지 **못하는** 가정은 무엇인가?

</div>

??? success "풀이"
    잔차–적합값 그림이 0 주위의 무작위 흩어짐을 보이면 **선형성**(체계적 곡률 없음)과 **등분산성**(일정한 폭) 가정을 뒷받침한다. 오차의 평균이 대략 0임도 확인해 준다.

    그러나 이 그림은 다음을 다루지 **못한다**. (1) 잔차의 **정규성**(Q-Q 그림이나 히스토그램이 필요하다), (2) **독립성**(잔차–순서 그림이나 Durbin-Watson 검정이 필요하다), (3) **이상점과 영향점**(Cook 거리나 지렛대 진단이 필요하다. 영향점 하나가 직선을 자기 쪽으로 끌어당겨 "보기 좋은" 잔차그림을 만들 수 있기 때문이다).

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
원잔차, 표준화 잔차, 스튜던트화(외부 스튜던트화) 잔차의 차이를 설명하라. 이상점 탐지에는 어느 것이 가장 적절하며 왜 그런가?

</div>

??? success "풀이"

    - **원잔차:** $e_i = Y_i - \hat{Y}_i$. 분산이 서로 다르므로($\text{Var}(e_i) = \sigma^2(1 - h_{ii})$) 관측값끼리 직접 비교하면 오도할 수 있다.

    - **표준화(내부 스튜던트화) 잔차:** $r_i = e_i / (\hat{\sigma}\sqrt{1 - h_{ii}})$. 각 잔차를 그 추정 표준편차로 나눈다. 모형 가정 아래에서 근사적으로 $N(0,1)$을 따른다.

    - **외부 스튜던트화 잔차:** $t_i = e_i / (\hat{\sigma}_{(i)}\sqrt{1 - h_{ii}})$. 여기서 $\hat{\sigma}_{(i)}$는 관측값 $i$를 뺀 뒤 추정한 값이다. 자유도 $n - p - 2$의 정확한 $t$ 분포를 따른다.

    이상점 탐지에는 **외부 스튜던트화 잔차**가 가장 적절하다. 문제의 이상점 자체 때문에 부풀려지지 않은 분산 추정을 쓰므로 그 관측값이 정말 극단적인지를 더 정직하게 평가한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
척도–위치 그림에 뚜렷한 상승 추세가 나타난다. 이것이 무엇을 나타내며 표준적인 잔차–적합값 그림과 어떻게 다른지 기술하라.

</div>

??? success "풀이"
    **척도–위치 그림**은 $\sqrt{|\text{표준화 잔차}|}$를 적합값에 대해 그린다. 상승 추세는 잔차의 **흩어짐**(절대 크기로 잰)이 적합값과 함께 커진다는 뜻이며 **이분산**을 나타낸다.

    표준적인 잔차–적합값 그림과 다른 점은 잔차의 부호가 아니라 **크기**에만 집중한다는 것이다. 표준 그림에서도 이분산이 깔때기 모양으로 나타날 수 있지만, 척도–위치 그림은 부호를 없애고 제곱근 척도로 그려지는 양의 분산을 안정화하므로 분산의 증가를 훨씬 쉽게 탐지할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
보기 1은 $n = 100$, $p = 2$, 참 $\sigma = 10$인 자료에서 잔차제곱평균 $\frac1n\sum_i e_i^2 = 114.171$과 $\hat\sigma = \sqrt{\text{SSE}/(n-p)} = 10.7936$을 함께 출력했다.

**(1)** 두 수가 서로 어떤 관계인지 식으로 적고, $114.171$만 가지고 $\hat\sigma = 10.7936$을 되살려 보시오.

**(2)** 어떤 독자가 $114.171$을 참값 $\sigma^2 = 100$과 견주어 "$14.2\%$ 높다"고 읽었다. 그런데 풀이는 어긋남을 $16.5\%$라고 적었다. 두 수가 왜 다른지 설명하고, 그 차이를 식으로 맞추시오.

**(3)** 이 쪽의 자료에서는 둘의 차이가 작다. $n$으로 나눈 값이 $\sigma^2$을 $20\%$나 낮게 잡으려면 $n$과 $p$가 어떤 관계여야 하는가. 그런 $(n, p)$를 하나 들고, 그런 상황에서 잔차 띠의 폭으로 $\sigma$를 눈대중하는 일이 왜 위험한지 한 줄로 쓰시오.

</div>

??? success "풀이"

    **(1) 두 수의 관계.** 분모만 다르다. $\text{SSE} = \sum_i e_i^2$이므로

    $$
    \frac{1}{n}\sum_i e_i^2 = \frac{\text{SSE}}{n}
    = \frac{n-p}{n}\cdot\frac{\text{SSE}}{n-p}
    = \hat\sigma^2\,\frac{n-p}{n}
    $$

    이다. 곧 잔차제곱평균은 $\hat\sigma^2$을 $(n-p)/n = 98/100 = 0.98$배 한 값이다. 거꾸로 풀면

    $$
    \hat\sigma = \sqrt{\frac{n}{n-p}\cdot\frac1n\sum_i e_i^2}
    = \sqrt{\frac{100}{98}\times 114.171}
    = \sqrt{116.5010} = 10.7936
    $$

    으로 출력된 값이 정확히 되살아난다.

    **(2) 분모를 섞어 읽은 탓이다.** $\frac1n\sum e_i^2$이 겨냥하는 것은 $\sigma^2$이 아니라 $(n-p)\sigma^2/n = 98$이다. 그러므로 어긋남은

    $$
    \frac{114.171}{98} = 1.16501
    \qquad\text{즉 } 16.5\%
    $$

    이고, 이것은 $\hat\sigma^2/\sigma^2 = 116.5010/100 = 1.16501$과 **같은 수**다. 반면 독자가 계산한 것은

    $$
    \frac{114.171}{100} = 1.14171
    \qquad\text{즉 } 14.2\%
    $$

    이다. 두 비의 관계는 바로 (1)의 인수다.

    $$
    1.16501 \times \frac{n-p}{n} = 1.16501 \times 0.98 = 1.14171
    $$

    곧 독자는 **$2\%$의 구조적 수축을 표집오차로 착각**했다. 부호가 반대 방향이어서 실제 어긋남 $16.5\%$를 $14.2\%$로 깎아 보고한 셈이다.

    **(3) $p/n$이 커질 때.** $(n-p)/n = 0.8$, 곧 $p = n/5$여야 한다. 예컨대 $n = 10$, $p = 2$가 그렇다. $n = 20$, $p = 4$도 같다.

    이때 잔차 띠의 폭은 $\sigma$가 아니라 $\sigma\sqrt{0.8} = 0.894\sigma$ 쪽에 맞춰 좁아져 보인다. **적합하는 모수가 많아지면 잔차는 그만큼 수축하므로, 띠의 폭을 그대로 $\sigma$로 읽으면 $\sigma$를 체계적으로 낮게 잡는다.** 보기 1에서 $\pm 2\hat\sigma$ 밖의 점이 기대값 $5$개보다 적게 세어진 것과 같은 방향의 편향이며, $p/n$이 작을 때만 무시할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
보기 6은 척도–위치 그림의 평활선이 $0.7258$에서 $0.6442$로 내려가는 것을 보고 **제곱한 비** $0.7876$을 "$\sigma$의 비"로 읽었다. 11.2절의 이분산 자료에서는 같은 비가 $7.90$이었다.

**(1)** 왜 높이의 비를 **제곱**해야 $\sigma$의 비가 되는지, 세로축이 무엇인지에서 출발해 보이시오. 국소적으로 오차의 표준편차가 $\sigma(x)$라고 두면 된다.

**(2)** 11.2절의 $\sigma$ 비 $7.90$은 척도–위치 그림에서 높이 몇 배로 보이는가. 또 참 $\sigma$가 그림의 왼쪽에서 오른쪽으로 꼭 **두 배**가 되는 자료라면 평활선의 높이는 몇 배가 되는가.

**(3)** (2)의 답을 보고, 척도–위치 그림이 이분산을 **찾기 쉽게** 만든다는 설명과 어떻게 양립하는지 쓰시오.

</div>

??? success "풀이"

    **(1) 세로축이 $\sqrt{\lvert\cdot\rvert}$이기 때문이다.** 세로축은 $v_i = \sqrt{\lvert r_i\rvert}$이고 $r_i$는 표준화 잔차다. 국소 표준편차가 $\sigma(x_i)$이면 $e_i \approx \sigma(x_i) Z_i$이고, 표준화는 자료 전체의 한 수 $\hat\sigma$로 나누므로

    $$
    r_i \approx \frac{\sigma(x_i)}{\hat\sigma} Z_i,
    \qquad
    v_i = \sqrt{\lvert r_i \rvert} \approx \sqrt{\frac{\sigma(x_i)}{\hat\sigma}}\,\sqrt{\lvert Z_i\rvert}
    $$

    이다. 양변의 기댓값을 보기 6 (1)의 상수와 함께 쓰면

    $$
    E[v_i] \approx 0.822179 \sqrt{\frac{\sigma(x_i)}{\hat\sigma}}
    $$

    이므로 두 자리 $x_a, x_b$에서의 높이의 비는 $\sqrt{\sigma(x_b)/\sigma(x_a)}$다. **제곱해야 $\sigma$의 비가 된다.** 전체를 나눈 $\hat\sigma$는 비를 취할 때 약분되어 사라지므로, 표준화를 어떻게 했든 비는 영향을 받지 않는다는 점도 함께 따라온다.

    **(2) 제곱근만큼 눌린다.** $\sigma$ 비 $7.90$은 높이의 비

    $$
    \sqrt{7.90} = 2.811
    $$

    로 보인다. 평평한 자료의 높이가 $0.822$이므로 넓은 쪽 끝이 $0.822 \times 2.811 = 2.31$ 근처까지 올라간다는 뜻이다. $\sigma$가 두 배가 되는 자료라면

    $$
    \sqrt{2} = 1.414
    \qquad\text{곧 } 41.4\%
    $$

    만 올라간다. 거꾸로 평활선이 $10\%$ 올라간 것을 보았다면 $\sigma$는 $1.1^2 = 1.21$배, 곧 $21\%$ 늘어난 것이다.

    **(3) 눌리는 대신 깨끗해진다.** 제곱근은 신호를 **줄인다.** 보통의 잔차그림에서는 $\sigma$가 두 배일 때 띠의 폭도 두 배($100\%$ 증가)로 벌어지는데, 척도–위치 그림에서는 같은 일이 $41\%$로 줄어든다. 그러니 "더 잘 보인다"는 말은 신호를 키운다는 뜻이 아니다.

    얻는 것은 **잡음 쪽**이다. 보기 6에서 $\lvert r\rvert$의 왜도 $+0.520$이 $\sqrt{\lvert r\rvert}$에서 $-0.0001$로 떨어졌다. 치우친 세로축에서는 큰 값 몇 개가 평활선을 끌어당겨 **없는 추세를 만들어 내는데**, 제곱근이 그 꼬리를 눌러 평활선을 믿을 수 있게 한다. 곧 분자(신호)는 제곱근만큼 줄지만 분모(평활선의 흔들림)가 더 많이 줄어 **신호 대 잡음비가 올라간다.** 부호를 없애 폭 하나만 보게 만든 것도 같은 방향의 이득이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
보기 6의 코드는 표준화를 `data['Residuals'] / np.std(data['Residuals'])`로, 곧 $n$으로 나눈 표준편차로 했다. 연습문제 2의 정의 $r_i = e_i/(\hat\sigma\sqrt{1-h_{ii}})$와는 다른 식이다.

**(1)** 절편이 있는 적합에서 $\sum_i e_i = 0$임을 쓰면 `np.std`의 값이 $\sqrt{\text{SSE}/n}$이다. 이것이 $\hat\sigma\sqrt{1 - p/n}$과 **정확히** 같음을 보이고, $\operatorname{tr}(H) = p$를 써서 $p/n$이 모자값의 **평균**임을 보이시오. 보기 1의 수 $\hat\sigma = 10.7936$과 $\frac1n\sum e_i^2 = 114.171$로 확인하시오.

**(2)** 그러므로 코드의 지름길은 어떤 관측값에서 정확하고 어떤 관측값에서 틀리는가. 틀리는 크기를 $h_{ii}$의 함수로 적고, $h_{ii} = 0.10$인 점과 $h_{ii} = 0.5$인 점에서 값을 넣어 보시오. 척도–위치 그림의 **높이**로는 그 어긋남이 얼마가 되는가.

**(3)** 보기 6의 그림에서 이 지름길이 문제를 일으켰다고 볼 수 있는가. 판단의 근거를 쓰시오.

</div>

??? success "풀이"

    **(1) 두 분모는 평균 모자값에서 정확히 일치한다.** 절편이 있으면 $\bar e = 0$이므로 `np.std`(기본 `ddof=0`)는

    $$
    \sqrt{\frac1n\sum_i (e_i - \bar e)^2} = \sqrt{\frac{\text{SSE}}{n}}
    $$

    이다. 한편 $\hat\sigma^2 = \text{SSE}/(n-p)$이므로

    $$
    \hat\sigma\sqrt{1 - \frac{p}{n}}
    = \sqrt{\frac{\text{SSE}}{n-p}\cdot\frac{n-p}{n}}
    = \sqrt{\frac{\text{SSE}}{n}}
    $$

    로 **항등적으로 같다.** 근사가 아니다.

    모자값의 평균은 $p/n$이다. $H = X(X^\top X)^{-1}X^\top$에서

    $$
    \operatorname{tr}(H) = \operatorname{tr}\big((X^\top X)^{-1} X^\top X\big) = \operatorname{tr}(I_p) = p
    \quad\Longrightarrow\quad
    \frac1n\sum_i h_{ii} = \frac{p}{n}
    $$

    이고, 이것이 보기 1에서 쓴 $\operatorname{tr}(I - H) = n - p$의 다른 얼굴이다. 곧 **코드의 분모는 "모든 점의 $h_{ii}$를 평균값으로 바꿔 치운" 올바른 분모**다.

    수로 확인하면 $\sqrt{114.171} = 10.6851$이고

    $$
    \hat\sigma\sqrt{1 - p/n} = 10.7936 \times \sqrt{0.98} = 10.7936 \times 0.98995 = 10.6851
    $$

    로 일치한다.

    **(2) 지렛대가 평균을 벗어나는 점에서 틀린다.** 두 표준화의 비는

    $$
    \frac{r_i^{\text{코드}}}{r_i^{\text{정의}}}
    = \frac{\hat\sigma\sqrt{1-h_{ii}}}{\hat\sigma\sqrt{1-p/n}}
    = \sqrt{\frac{1-h_{ii}}{1-p/n}}
    $$

    이다. $h_{ii} = p/n$인 점에서는 정확히 $1$이고, $h_{ii} > p/n$이면 $1$보다 작다. 곧 **지렛대가 큰 점은 코드가 실제보다 작게 그린다.** 보기 6은 $p/n = 0.02$이므로

    - $h_{ii} = 0.10$: $\sqrt{0.90/0.98} = \sqrt{0.91837} = 0.9583$, 곧 $4.2\%$ 작게.
    - $h_{ii} = 0.5$: $\sqrt{0.50/0.98} = \sqrt{0.51020} = 0.7143$, 곧 $28.6\%$ 작게.

    세로축은 다시 제곱근을 취하므로 어긋남이 또 절반(로그 눈금에서)으로 줄어든다. 높이의 비는 $\sqrt{0.9583} = 0.9789$와 $\sqrt{0.7143} = 0.8452$, 곧 $2.1\%$와 $15.5\%$다.

    **(3) 보기 6에서는 문제가 없다.** $n = 100$, $p = 2$인 단순회귀이고 $x_i \sim N(0,1)$이라 $h_{ii} = \frac1n + (x_i-\bar x)^2/S_{xx}$가 모두 작다. $\lvert x_i - \bar x\rvert$가 표본표준편차의 $3$배인 극단점이라도 $S_{xx} \approx n$이므로 $h_{ii} \approx 0.01 + 0.09 = 0.10$이고, (2)에서 보았듯 높이로는 $2\%$ 어긋난다. 보기 6의 평활선이 양 끝에서 $0.726$과 $0.644$로 $11\%$ 차이가 난 것에 비하면 **읽기에 영향을 줄 크기가 아니다.**

    위험한 쪽은 반대 상황이다. $n$이 작거나 설명변수가 많아 $p/n$이 크거나, 한두 점이 멀리 떨어져 $h_{ii}$가 $0.5$에 이르는 자료라면 지름길이 **바로 그 점을 작게 그려** 가장 보고 싶은 이상점을 숨긴다. 연습문제 2가 이상점 탐지에 외부 스튜던트화 잔차를 쓰라고 한 것과 같은 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
보기 2는 적합값 사분위별 잔차의 표준편차를 $1.6495,\ 2.7632,\ 3.2007,\ 2.1579$(각 구간 $n = 13, 12, 12, 13$)로 재고, 단조가 아니므로 이분산이 아니라고 읽었다.

**(1)** 정리하며 절의 $\operatorname{Var}(e_i) = \sigma^2(1 - h_{ii})$를 쓰면, 이 자료에서 사분위별 잔차의 폭이 **어느 방향으로** 기울어야 하는지 예측할 수 있다. 방향을 말하고, 그 효과의 크기를 $h_{ii}$의 상한으로 어림하시오.

**(2)** 정규 표본에서 $m$개로 센 표준편차는 $\operatorname{sd}(\log s) \approx 1/\sqrt{2(m-1)}$를 따른다. 가장 넓은 구간과 가장 좁은 구간의 비 $3.2007/1.6495$가 몇 표준오차인지 재고, 네 구간 가운데서 양끝을 **골라낸** 것까지 셈에 넣어 판정하시오.

**(3)** Breusch-Pagan의 $p = 0.418$은 (2)의 물음에 답해 주는가. 이 검정이 무엇에 대해 검정력을 갖는지 생각해 답하시오.

</div>

??? success "풀이"

    **(1) 가운데가 넓어야 한다, 아주 조금.** $\hat y$가 $x$의 증가함수이므로 적합값의 사분위는 곧 $x$의 사분위다. 단순회귀에서

    $$
    h_{ii} = \frac1n + \frac{(x_i - \bar x)^2}{S_{xx}}
    $$

    이므로 $x$가 극단일수록 $h_{ii}$가 크고, $\operatorname{Var}(e_i) = \sigma^2(1-h_{ii})$가 작다. 곧 **Q1과 Q4의 잔차가 Q2, Q3보다 좁은 것이 구조적으로 옳은 방향**이며, 관측된 $1.65,\ 2.76,\ 3.20,\ 2.16$의 모양과 방향이 맞는다.

    크기는 그러나 미미하다. $n = 50$, $S_{xx} = 63.34$이고 가장 먼 점이 $\lvert x - \bar x\rvert \approx 2.5$라 해도

    $$
    h_{ii} \le \frac{1}{50} + \frac{2.5^2}{63.34} = 0.02 + 0.0987 = 0.119
    $$

    이고, 표준편차의 인수는 $\sqrt{1 - 0.119} = 0.939$다. 곧 **가장 극단적인 한 점조차 $\sigma$보다 $6\%$ 좁을 뿐**이고, 구간 평균으로는 그보다 훨씬 작다. $1.65$ 대 $3.20$의 $94\%$ 차이를 이것으로 설명할 수는 없다.

    **(2) 고른 것까지 넣으면 잡음이다.** 비의 로그는

    $$
    \log\frac{3.2007}{1.6495} = \log 1.9404 = 0.6629
    $$

    이고, 두 구간이 독립이라 보면 표준오차가

    $$
    \sqrt{\frac{1}{2(13-1)} + \frac{1}{2(12-1)}}
    = \sqrt{0.04167 + 0.04545} = \sqrt{0.08712} = 0.2952
    $$

    이다. 따라서 $z = 0.6629/0.2952 = 2.25$이고, **미리 정해 둔 한 쌍**이었다면 양측 $p = 0.025$로 유의하다.

    그러나 이 쌍은 네 구간을 다 본 뒤 가장 넓은 것과 가장 좁은 것을 **골라낸** 것이다. 비교 가능한 쌍이 $\binom{4}{2} = 6$개이므로 Bonferroni로 고치면

    $$
    p_{\text{조정}} \approx 6 \times 0.0247 = 0.148
    $$

    로 유의하지 않다. 표준정규 $4$개의 범위가 평균 $2.06$ 표준편차라는 사실과도 맞는다. 관측된 $2.25$는 **네 개를 뽑았을 때 으레 나오는 벌어짐**이다. 따라서 보기 2의 판정이 옳고, 다만 그 근거는 "단조가 아니다"보다 이 다중성 계산이 정확하다.

    **(3) 답해 주지 않는다. 다른 것을 검정한 것이다.** 보기 2가 쓴 것은 설계행렬 $(\mathbf 1, \mathbf x)$에 대한 Breusch-Pagan, 곧 $e_i^2$을 $1$과 $x_i$에 회귀한 보조검정이다. 이것은 분산이 $x$에 대해 **단조(선형)로** 변하는 대안에만 검정력을 갖는다.

    그런데 (1)과 (2)가 다룬 모양은 **가운데가 부푼 비단조** 형태다. 그런 모양은 $e_i^2$과 $x_i$의 상관을 거의 0으로 남기므로, 분산이 정말 그렇게 변해도 Breusch-Pagan은 통과한다. 이 대안을 잡으려면 보조회귀에 $x^2$을 넣어야 한다.

    요컨대 $p = 0.418$은 **"분산에 $x$ 방향의 기울기가 없다"**는 말이고, "분산이 일정하다"는 말이 아니다. $\operatorname{corr}(\lvert e\rvert, \hat y) = 0.126$도 똑같이 기울기만 본다. 비단조 모양에 대한 판정은 (2)의 구간별 셈과 다중성 보정에서 나왔다. **검정의 $p$를 읽을 때는 그 검정이 어느 대안을 겨누는지 함께 읽어야 한다.**

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
보기 5는 같은 자료에 대한 직선 적합과 이차 적합에서 $\text{RSS}_1 = 1518.09$, $\text{RSS}_2 = 282.93$을 얻고, 직선 적합의 잔차 $\mathbf e_1$을 $(\mathbf 1, \mathbf x, \mathbf x^2)$에 회귀한 보조회귀의 $R^2$이 $0.8136$이라고 출력했다. 이 $0.8136$은 보기 3이 "잔차의 분산 가운데 곡률의 몫"으로 잰 수와 **같은 수**다.

**(1)** $X_1 = (\mathbf 1, \mathbf x)$가 $X_2 = (\mathbf 1, \mathbf x, \mathbf x^2)$에 포개져 있다는 사실만 써서

$$
R^2_{\text{보조}} = 1 - \frac{\text{RSS}_2}{\text{RSS}_1}
$$

임을 증명하고, 세 수가 서로 맞는지 확인하시오. 보기 5가 "줄어든 $1235$가 곡률의 몫"이라 한 것이 왜 근사가 아니라 항등식인지도 말하시오.

**(2)** 보기 5는 $\operatorname{corr}(\mathbf e_1, \mathbf x^2) = 0.8978$도 출력했다. 이 값의 제곱은 $0.8061$로 $0.8136$보다 작다. 둘 중 어느 쪽이 항상 크고, 왜 그런가. 이 자료에서 그 차이가 생긴 구체적인 원인은 무엇인가.

</div>

??? success "풀이"

    **(1) 보조회귀의 잔차가 이차 적합의 잔차와 같다.** $P_1, P_2$를 각각 $X_1, X_2$의 열공간 위로의 사영이라 하고 $M_k = I - P_k$라 하자. $\operatorname{col}(X_1) \subset \operatorname{col}(X_2)$이므로

    $$
    P_2 P_1 = P_1
    $$

    이다. 이제 $\mathbf e_1 = M_1\mathbf y$를 $X_2$에 회귀하면 적합값이

    $$
    P_2 \mathbf e_1 = P_2 M_1 \mathbf y = (P_2 - P_2P_1)\mathbf y = (P_2 - P_1)\mathbf y
    $$

    이고 잔차가

    $$
    \mathbf e_1 - P_2\mathbf e_1 = M_1\mathbf y - (P_2 - P_1)\mathbf y = (I - P_2)\mathbf y = M_2\mathbf y = \mathbf e_2
    $$

    다. 한편 $\mathbf 1 \in \operatorname{col}(X_1)$이라 $\bar e_1 = 0$이므로 보조회귀의 총제곱합은 $\lVert \mathbf e_1\rVert^2 = \text{RSS}_1$이고 잔차제곱합은 $\lVert \mathbf e_2\rVert^2 = \text{RSS}_2$다. 따라서

    $$
    R^2_{\text{보조}} = 1 - \frac{\text{RSS}_2}{\text{RSS}_1}
    $$

    이다. $\square$

    수로 맞춰 보면

    $$
    1 - \frac{282.93}{1518.09} = 1 - 0.18638 = 0.81362
    $$

    로 출력된 $0.8136$과 같다. 또 식을 옮기면 $\text{RSS}_1 - \text{RSS}_2 = \text{RSS}_1 R^2_{\text{보조}}$이므로

    $$
    1518.09 \times 0.81362 = 1235.16 = 1518.09 - 282.93
    $$

    이고, **좌변과 우변이 자리까지 같다.** 근사가 아닌 까닭은 둘이 같은 사영의 두 가지 이름이기 때문이다. $\mathbf x^2$을 설계행렬에 넣어 줄어든 RSS와, 직선 잔차 안에 들어 있던 $(\mathbf 1,\mathbf x,\mathbf x^2)$ 설명 가능한 몫은 **같은 벡터 $(P_2-P_1)\mathbf y$의 길이의 제곱**이다.

    **(2) 보조회귀의 $R^2$이 항상 크거나 같다.** 중심화한 $\mathbf x^2$을 $\tilde{\mathbf v}$라 하면 $\tilde{\mathbf v} \in \operatorname{col}(X_2)$이므로 $\tilde{\mathbf v} = P_2\tilde{\mathbf v}$이고, Cauchy-Schwarz로

    $$
    \lvert \mathbf e_1^\top \tilde{\mathbf v}\rvert
    = \lvert (P_2\mathbf e_1)^\top \tilde{\mathbf v}\rvert
    \le \lVert P_2 \mathbf e_1\rVert \, \lVert \tilde{\mathbf v}\rVert
    $$

    이다. 양변을 $\lVert \mathbf e_1\rVert \lVert\tilde{\mathbf v}\rVert$로 나누면

    $$
    \operatorname{corr}(\mathbf e_1, \mathbf x^2)^2
    \le \frac{\lVert P_2\mathbf e_1\rVert^2}{\lVert \mathbf e_1\rVert^2}
    = R^2_{\text{보조}}
    $$

    이고, 등호는 $\tilde{\mathbf v}$가 $P_2\mathbf e_1$과 평행할 때만 성립한다. 실제로 $0.8978^2 = 0.8061 \le 0.8136$이다.

    **차이의 원인은 $\mathbf x^2$ 안에 남아 있는 $\mathbf x$ 성분이다.** $\mathbf e_1$은 $\mathbf 1$과 $\mathbf x$에 직교하므로 설명에 쓰이는 것은 $\mathbf x^2$에서 $(\mathbf 1, \mathbf x)$를 걷어낸 $M_1\mathbf x^2$뿐이다. 보기 3이 `curv_part`로 계산한 바로 그 벡터이고, 그래서 보기 3의 $0.8136$과 보조회귀의 $0.8136$이 같은 수였다. 반면 $\operatorname{corr}(\mathbf e_1, \mathbf x^2)$은 평균만 뺀 $\mathbf x^2$을 쓰므로 $\mathbf x$ 방향 성분이 분모에 그대로 남아 상관을 희석한다. 이 표본에서 그 성분이 0이 아닌 까닭은 보기 3이 지적한 대로 $E[x^3]$의 표본값이 $+0.3493$이어서 $\operatorname{Cov}(x, x^2) \ne 0$이기 때문이다. **대칭이 완벽한 모집단에서라면 두 수가 일치한다.**

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
보기 4의 자료는 $x \sim U(-3,3)$, $y = 2 + 0.5x - 1.5x^2 + \varepsilon$, $\varepsilon \sim N(0,1)$이고, 직선 적합의 극한이 $(\beta_0^*, \beta_1^*) = (-2.5,\ 0.5)$, 모집단 $R^2$이 $0.0418$이었다. 기울기가 참 일차항 계수 $0.5$와 똑같이 나온 것은 $x$의 분포가 대칭이어서 $\operatorname{Cov}(x, x^2) = 0$이었기 때문이다.

관측 구간만 $x \sim U(0,3)$으로 옮기고 나머지는 그대로 둔다. $E[x^k] = 3^k/(k+1)$을 쓴다.

**(1)** $\operatorname{Var}(x)$, $\operatorname{Cov}(x, x^2)$, $\operatorname{Var}(x^2)$를 구하시오.

**(2)** 직선 적합의 극한 계수 $(\beta_0^*, \beta_1^*)$와 $\operatorname{Var}(y)$, 그리고 직선·이차 두 모형의 모집단 $R^2$을 구하시오.

**(3)** 두 구간의 결과를 견주어라. 어느 구간에서 직선 모형의 $R^2$이 더 높은가. 그 구간에서 직선 모형이 더 나은 모형인가. 잔차 가운데 곡률이 차지하는 몫까지 계산해 답하시오.

</div>

??? success "풀이"

    **(1) 적률.** $E[x] = 3/2 = 1.5$, $E[x^2] = 9/3 = 3$, $E[x^3] = 27/4 = 6.75$, $E[x^4] = 81/5 = 16.2$이므로

    $$
    \operatorname{Var}(x) = 3 - 1.5^2 = 0.75
    $$

    $$
    \operatorname{Cov}(x, x^2) = E[x^3] - E[x]E[x^2] = 6.75 - 1.5 \times 3 = 2.25
    $$

    $$
    \operatorname{Var}(x^2) = E[x^4] - (E[x^2])^2 = 16.2 - 9 = 7.2
    $$

    이다. $\operatorname{Var}(x^2)$는 두 구간에서 같은데($7.2$), **$\operatorname{Cov}(x,x^2)$가 $0$에서 $2.25$로 바뀐 것이 모든 차이의 뿌리다.**

    **(2) 극한 계수와 $R^2$.**

    $$
    \beta_1^* = \frac{\operatorname{Cov}(x,\ 0.5x - 1.5x^2)}{\operatorname{Var}(x)}
    = \frac{0.5 \times 0.75 - 1.5 \times 2.25}{0.75}
    = \frac{0.375 - 3.375}{0.75} = -4.0
    $$

    $$
    \beta_0^* = E[y] - \beta_1^* E[x] = (2 + 0.75 - 4.5) - (-4.0)(1.5) = -1.75 + 6 = 4.25
    $$

    이다. 반응변수의 분산은 교차항이 살아 있으므로

    $$
    \operatorname{Var}(y) = 0.5^2(0.75) + 1.5^2(7.2) - 2(0.5)(1.5)(2.25) + 1
    = 0.1875 + 16.2 - 3.375 + 1 = 14.0125
    $$

    이고, 직선 모형이 설명하는 몫은 $(\beta_1^*)^2\operatorname{Var}(x) = 16 \times 0.75 = 12$다. 따라서

    $$
    R^2_{\text{직선}} = \frac{12}{14.0125} = 0.8564,
    \qquad
    R^2_{\text{이차}} = \frac{14.0125 - 1}{14.0125} = \frac{13.0125}{14.0125} = 0.9286
    $$

    이다.

    **(3) $R^2$은 올라가고 모형은 더 나빠진다.**

    | 항목 | $x \sim U(-3,3)$ | $x \sim U(0,3)$ |
    |---|---|---|
    | $\operatorname{Cov}(x, x^2)$ | $0$ | $2.25$ |
    | 극한 기울기 $\beta_1^*$ | $0.5$ | $-4.0$ |
    | $R^2_{\text{직선}}$ | $0.0418$ | $0.8564$ |
    | $R^2_{\text{이차}}$ | $0.9443$ | $0.9286$ |
    | 잔차분산 중 곡률의 몫 | $16.2/17.2 = 0.942$ | $1.0125/2.0125 = 0.503$ |

    오른쪽 구간에서 직선 $R^2$이 $0.042$에서 $0.856$으로 **스무 배**가 된다. 그런데도 직선 모형은 조금도 나아지지 않았다. 자료는 여전히 완전한 이차식이고, 직선은 여전히 틀렸다.

    더 나쁜 것은 **기울기가 쓸모를 잃었다**는 점이다. $U(-3,3)$에서는 $\beta_1^* = 0.5$로 참 일차항 계수와 같았는데, $U(0,3)$에서는 $-4.0$으로 부호까지 뒤집힌다. 이차항이 기울기에 보태는 양이 정확히

    $$
    -1.5 \times \frac{\operatorname{Cov}(x,x^2)}{\operatorname{Var}(x)} = -1.5 \times \frac{2.25}{0.75} = -4.5
    $$

    이고, $0.5 + (-4.5) = -4.0$이다. **$R^2$이 높아진 까닭이 바로 기울기가 망가진 까닭이다.** $x^2$의 변동이 $x$와 상관되면서 그 큰 분산($\operatorname{Var}(x^2) = 7.2$)의 대부분이 직선 항으로 빨려 들어갔고, 설명된 분산은 그만큼 늘었다.

    잔차에 남은 곡률의 몫도 $0.942$에서 $0.503$으로 반 토막이 난다. 곧 **$U(0,3)$에서는 잔차그림의 U자가 상대적으로 얕아 눈에 덜 띈다.** 높은 $R^2$과 흐려진 무늬가 겹쳐 "직선으로 충분하다"는 결론을 부르기 쉬운 가장 나쁜 조합이다. 보기 4의 결론 "핵심은 $R^2$이 아니라 잔차의 무늬다"는 이 구간에서 한층 더 들어맞는다. $R^2$은 적합의 좋음이 아니라 **설명변수가 어느 구간에서 관측되었는지**에 먼저 반응한다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
보기 1은 $\operatorname{corr}(e, \hat y)$가 **항등적으로 0**이므로 "기울기는 볼 필요가 없다"고 했고, 정리하며 절은 잔차의 합이 언제나 0이라고 했다. 두 주장은 모두 설계행렬에 **절편 열이 있다**는 조건에 기댄다.

**(1)** 정규방정식에서 나오는 것은 $\sum_i e_i \hat y_i = 0$이다. 이것만으로는 Pearson 상관 $\operatorname{corr}(e, \hat y)$가 0이 되지 않는다. 상관의 분자를 전개해 어느 조건이 더 필요한지 밝히시오.

**(2)** 절편 없이 원점을 지나는 적합 $\hat y_i = b x_i$를 $x = (1, 2, 3)$, $y = (3, 3, 3)$에 대해 하시오. $b$, 잔차, $\sum_i e_i$, $\sum_i e_i \hat y_i$, 그리고 $\operatorname{corr}(e, \hat y)$를 모두 구하시오.

**(3)** (2)의 잔차그림에 나타난 기울기는 **무엇을 알려 주는가.** 절편이 있는 적합에서는 같은 기울기가 아무것도 알려 주지 못하는 것과 견주어 쓰시오.

</div>

??? success "풀이"

    **(1) 잔차의 평균이 0이어야 한다.** Pearson 상관의 분자는 중심화한 내적이다.

    $$
    \sum_i (e_i - \bar e)(\hat y_i - \bar{\hat y})
    = \sum_i e_i \hat y_i - n\,\bar e\,\bar{\hat y}
    $$

    정규방정식 $X^\top \mathbf e = \mathbf 0$은 $\mathbf e$가 $\operatorname{col}(X)$에 직교한다는 말이고, $\hat{\mathbf y} \in \operatorname{col}(X)$이므로 첫 항 $\sum_i e_i\hat y_i = 0$은 언제나 성립한다. 그러나 둘째 항이 남는다. 분자는

    $$
    -\,n\,\bar e\,\bar{\hat y}
    $$

    이므로, 상관이 0이 되려면 $\bar e = 0$(또는 $\bar{\hat y} = 0$)이 필요하다. 그리고 $\bar e = 0$은 $\mathbf 1 \in \operatorname{col}(X)$, 곧 **절편 열이 있을 때** 직교성 $\mathbf 1^\top \mathbf e = 0$으로부터 나온다. **"항등적으로 0"의 진짜 근거는 직교성 하나가 아니라 직교성에 절편이 더해진 것이다.**

    **(2) 반례.** 최소제곱으로 $b$를 구하면

    $$
    b = \frac{\sum_i x_i y_i}{\sum_i x_i^2} = \frac{3 + 6 + 9}{1 + 4 + 9} = \frac{18}{14} = \frac97 = 1.285714
    $$

    이고 적합값과 잔차가

    $$
    \hat{\mathbf y} = \Big(\tfrac97,\ \tfrac{18}{7},\ \tfrac{27}{7}\Big)
    = (1.2857,\ 2.5714,\ 3.8571),
    \qquad
    \mathbf e = \Big(\tfrac{12}{7},\ \tfrac37,\ -\tfrac67\Big)
    = (1.7143,\ 0.4286,\ -0.8571)
    $$

    이다. 두 합을 재면

    $$
    \sum_i e_i = \frac{12 + 3 - 6}{7} = \frac97 = 1.2857 \ne 0
    $$

    $$
    \sum_i e_i \hat y_i = \frac{12 \cdot 9 + 3 \cdot 18 + (-6)\cdot 27}{49} = \frac{108 + 54 - 162}{49} = 0
    $$

    로, **직교성은 멀쩡한데 잔차의 합이 0이 아니다.** 상관은 $e_i = 3 - bx_i$가 $x_i$의 일차함수이고 $\hat y_i = bx_i$이므로 두 벡터가 완전한 음의 일차관계다. 중심화해 보면 $\hat y_i - \bar{\hat y} = b(x_i - \bar x) = \frac97(-1, 0, 1)$이고 $e_i - \bar e = -b(x_i - \bar x) = \frac97(1, 0, -1)$이므로

    $$
    \operatorname{corr}(e, \hat y) = -1
    $$

    이다. 항등적으로 0이라던 양이 **가능한 가장 먼 값**을 가진다.

    **(3) 절편이 빠졌다는 신호다.** 이 자료의 참 평균은 $y = 3$으로 일정하다. 곧 옳은 모형은 절편만 있는 모형인데 그 절편을 강제로 0으로 묶었으므로, 적합선이 점구름을 가로질러 왼쪽을 낮게 오른쪽을 높게 지나간다. 그 체계적 어긋남이 잔차그림에 **기울기 $-1$의 완전한 직선**으로 찍힌 것이다. 원점 통과 적합에서는 잔차–적합값 그림의 기울기가 **실제 진단 정보**이며, 빠진 절편을 잡아내는 데 쓸 수 있다.

    절편이 있는 적합에서는 사정이 정반대다. 기울기가 모형이 맞든 틀리든 항등적으로 0이므로, 그 기울기를 보고는 아무것도 알 수 없다. 보기 1이 $\operatorname{corr}(e,\hat y) = -1.94\times10^{-15}$를 적고 **"그러므로 기울기는 볼 필요가 없다"**고 한 것이 그 뜻이고, 읽을 것은 깔때기($\operatorname{corr}(\lvert e\rvert, \hat y)$)와 휘어짐(2차항) 뿐이다. **같은 그림의 같은 특징이 설계행렬에 따라 정보일 수도, 아무것도 아닐 수도 있다.**

---

## 정리하며

잔차 $e_i=y_i-\hat y_i$ 가 **모형 진단의 주재료**다.

- **잔차가 오차의 추정값이다.** 참 오차 $\varepsilon_i$ 는 관측할 수 없고 잔차로 대신 본다. 다만 **잔차는 오차와 달리 서로 상관되어 있고 분산도 일정하지 않다**($\mathrm{Var}(e_i)=\sigma^2(1-h_{ii})$).
- **그래서 표준화·스튜던트화 잔차를 쓴다.** 모자값 $h_{ii}$ 로 보정해 비교 가능하게 만든 것이며, 대략 $|r|>2$ 를 눈여겨본다.
- **네 가지 그림이 기본이다.** 적합값 대 잔차(선형성·등분산성), Q-Q(정규성), 순서 대 잔차(독립성), 척도–위치 그림.
- **패턴이 없어야 정상이다.** 무작위한 구름 모양이면 좋고, 곡선·부채꼴·주기가 보이면 무언가 놓친 것이다.
- **잔차의 합은 언제나 $0$ 이다.** 절편이 있으면 자동으로 그렇게 되므로, 그 사실 자체는 아무 정보도 아니다.

다음 절 **[다중공선성과 분산팽창인자](multicollinearity.md)**로 넘어간다. 설명변수끼리 강하게 상관되면 추정과 추론이 함께 흔들린다.
