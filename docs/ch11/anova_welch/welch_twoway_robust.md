# 이원배치 Welch 분산분석 (로버스트 HC3)

## 개요

이원배치 요인 설계의 오차가 이분산이면, 공통 분산 가정에 기반한 표준 분산분석 $F$-검정을 믿을 수 없다. 실용적인 대안은 완전 요인 설정으로 OLS 모형을 적합한 뒤 HC3 이분산 일치 공분산 추정량과 Wald $F$-검정을 결합하여 주효과와 교호작용을 검정하는 것이다. 이 접근은 전용 Welch-James 구현 없이도 이원배치 분산분석의 로버스트한 대응물을 제공한다.

## 로버스트 OLS 접근

모형은 표준적인 이원배치 요인 모형이다:

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

여기서 $\text{Var}(\varepsilon_{ijk})$는 더 이상 일정하다고 가정하지 않는다. OLS 계수 추정값 $\hat{\boldsymbol{\beta}}$는 여전히 불편이며 일치성을 갖지만, 고전적인 공분산행렬 $\hat{\sigma}^2 (X^\top X)^{-1}$은 타당하지 않다. HC3 추정량은 이를 다음으로 대체한다:

$$
\widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} \left(\sum_{i=1}^{n} \frac{\hat{e}_i^2}{(1 - h_{ii})^2} \mathbf{x}_i \mathbf{x}_i^\top \right) (X^\top X)^{-1}
$$

여기서 $h_{ii}$는 햇 행렬 $H = X(X^\top X)^{-1}X^\top$의 $i$번째 대각 성분이고 $\hat{e}_i$는 $i$번째 OLS 잔차이다. 이 추정량은 이분산 아래에서 일치성을 가지며 HC0나 HC1보다 소표본 성능이 좋다.

## 각 항에 대한 Wald F-검정

주효과나 교호작용을 검정하려면 그 항의 계수에 해당하는 행을 $R$이 고르는 결합 선형 가설 $R\boldsymbol{\beta} = \mathbf{0}$을 세운다. Wald $F$-통계량은

$$
F_W = \frac{1}{q} (R\hat{\boldsymbol{\beta}})^\top \bigl(R\, \widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}})\, R^\top\bigr)^{-1} (R\hat{\boldsymbol{\beta}})
$$

이며 $q$는 제약의 수($R$의 행 수)이다. $H_0$ 아래에서 $F_W$는 근사적으로 $F_{q, \nu}$를 따르고 $\nu$는 조정된 분모 자유도이다.

## 코드 예제

```python
import pandas as pd
from statsmodels.formula.api import ols

# 칸마다 반복이 **둘 이상** 있어야 한다. 칸당 하나뿐이면 3x3 설계에서
# 모수 9개로 관측값 9개를 완전히 맞혀 버려 잔차 자유도가 0이 되고,
# 지렛값 h_ii가 1이 되어 HC3의 1/(1-h_ii)^2 이 발산한다.
data = {
    "Temperature": ["High"]*6 + ["Low"]*6 + ["Medium"]*6,
    "Fertilizer":  ["A", "A", "B", "B", "C", "C"] * 3,
    "Growth":      [12, 13, 15, 18, 14, 15,
                    10,  9, 13, 12, 11, 13,
                    14, 16, 16, 21, 15, 17],
}
df = pd.DataFrame(data)

model = ols("Growth ~ C(Temperature) * C(Fertilizer)", data=df).fit()
rob = model.get_robustcov_results(cov_type="HC3")

# Temperature의 주효과 검정.
# ":"가 든 이름을 빼야 한다. 교호작용 항의 이름도 "C(Temperature)["로 시작하므로
# 그냥 startswith만 쓰면 교호작용까지 함께 검정해 자유도가 2가 아니라 6이 된다.
pnames = model.params.index.tolist()
temp_params = [p for p in pnames
               if p.startswith("C(Temperature)[") and ":" not in p]
constraint = ", ".join([f"{t} = 0" for t in temp_params])
print("Main effect: Temperature")
print(rob.f_test(constraint))

# 교호작용 검정
inter_params = [p for p in pnames if ":" in p]
constraint_inter = ", ".join([f"{t} = 0" for t in inter_params])
print("Interaction: Temperature x Fertilizer")
print(rob.f_test(constraint_inter))
```

출력:

```
Main effect: Temperature
<F test: F=8.055555555555552, p=0.009878581991016048, df_denom=9, df_num=2>
Interaction: Temperature x Fertilizer
<F test: F=0.15370680044593157, p=0.956506962060376, df_denom=9, df_num=4>
```

주효과는 유의하고($p = 0.0099$) 교호작용은 아니다($p = 0.957$). 분자 자유도가 각각 2와 4로, 수준 수에서 계산한 $a - 1 = 2$와 $(a-1)(b-1) = 4$에 맞는다. 이 자유도를 확인하는 것이 제약을 제대로 걸었는지 점검하는 가장 쉬운 방법이다.

## 표준 분산분석과의 비교

참고를 위해 (로버스트하지 않은) 표준 분산분석표를 얻을 수 있다:

```python
import statsmodels.api as sm

print(sm.stats.anova_lm(model, typ=2))
```

출력:

```
                                 sum_sq   df      F    PR(>F)
C(Temperature)                81.444444  2.0  14.66  0.001475
C(Fertilizer)                 36.777778  2.0   6.62  0.017060
C(Temperature):C(Fertilizer)   2.555556  4.0   0.23  0.914666
Residual                      25.000000  9.0    NaN       NaN
```

표준 분산분석은 Temperature의 $F$를 14.66으로, HC3 Wald 검정은 8.06으로 준다. 두 값이 이만큼 다른 것은 분산이 칸마다 다르다는 신호다. 실제로 이 자료에서 B 비료의 칸들이 다른 칸보다 흩어져 있다.

방향도 눈여겨보라. 로버스트 검정이 더 **작은** $F$를 준다. 표준 검정이 표준오차를 과소평가해 효과를 부풀리고 있었다는 뜻이다.

분산이 같으면 HC3 Wald 검정과 표준 분산분석이 비슷한 결과를 준다. 두 결과가 어긋난다면 이분산이 표준 검정에 영향을 주고 있다는 뜻이다.

## 해석

- **HC3 대 HC0:** HC3는 각 제곱 잔차를 (HC0처럼 그대로 두지 않고) $(1 - h_{ii})^2$으로 나눈다. 지렛값이 큰 점의 잔차가 작아지는 경향을 이 상향 조정이 보정하여 소표본에서 포함확률을 개선한다.
- **언제 이 접근을 쓰는가:** 형식적 검정(Levene, Bartlett)이나 시각적 검토(잔차 그림)가 분산이 다름을 시사할 때마다 표준 $F$-검정보다 HC3 기반 Wald 검정이 낫다.
- **한계:** 칸 크기가 아주 작으면 개별 지렛값 $h_{ii}$가 1에 가까워져 HC3 추정량이 불안정해진다. 극단적으로 칸당 $n = 1$이면 포화모형이 되어 $h_{ii} = 1$, 잔차 0이 되고 HC3의 $1/(1-h_{ii})^2$이 발산해 계산 자체가 불가능하다. 위 예제에서 칸마다 반복을 둘씩 둔 이유가 이것이다. 칸 크기가 클수록 로버스트 추정량의 신뢰성이 높아진다.

## 연습문제

**연습문제 1.**
표준오차가 틀리게 되는데도 이분산 아래에서 OLS 계수 추정값이 여전히 불편인 이유를 설명하라. OLS의 어떤 성질이 쓰이는가?

??? success "풀이"
    OLS 추정량은 $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1} X^\top \mathbf{y}$이다. 기댓값을 취하면

    $$
    E[\hat{\boldsymbol{\beta}}] = (X^\top X)^{-1} X^\top E[\mathbf{y}] = (X^\top X)^{-1} X^\top X \boldsymbol{\beta} = \boldsymbol{\beta}
    $$

    이다. 이 유도는 추정량의 선형성과 $E[\mathbf{y}] = X\boldsymbol{\beta}$(조건부 평균의 올바른 설정)만 쓸 뿐 등분산성 가정을 쓰지 않는다. 따라서 관측값마다 $\text{Var}(\varepsilon_i) = \sigma_i^2$이 달라도 OLS 추정값은 불편이다. Gauss-Markov 정리는 등분산성 아래에서만 OLS가 BLUE(최우수 선형 불편 추정량)임을 보장하므로, 등분산성이 없으면 OLS는 여전히 불편이지만 더 이상 효율적이지는 않다.

---

**연습문제 2.**
수준이 High, Low, Medium(Low가 기준)인 모형에서 Temperature의 주효과를 검정하기 위한 제약행렬 $R$을 써라. $R$의 행은 몇 개인가?

??? success "풀이"
    Low가 기준 수준이면 모형에는 Temperature[T.High]와 Temperature[T.Medium]의 지시 계수가 들어간다. Temperature의 주효과를 검정한다는 것은

    $$
    H_0: \beta_{\text{T.High}} = 0 \text{ and } \beta_{\text{T.Medium}} = 0
    $$

    을 검정한다는 뜻이다. 제약행렬은 전체 모수 벡터 $\boldsymbol{\beta} = (\beta_0, \beta_{\text{T.High}}, \beta_{\text{T.Medium}}, \beta_{\text{F.B}}, \beta_{\text{F.C}}, \ldots)^\top$에서 이 두 계수를 고른다. Temperature[T.High]가 두 번째 모수이고 Temperature[T.Medium]이 세 번째라면

    $$
    R = \begin{pmatrix} 0 & 1 & 0 & 0 & \cdots & 0 \\ 0 & 0 & 1 & 0 & \cdots & 0 \end{pmatrix}
    $$

    이다. 행렬 $R$은 (기준이 아닌 수준마다 하나씩) $q = a - 1 = 2$개의 행을 가지며, 주효과의 자유도 2에 대응한다.

---

**연습문제 3.**
HC3는 $(1 - h_{ii})^2$으로 나누고 HC2는 $(1 - h_{ii})$로 나눈다. $(1 - h_{ii})^2$ 보정의 직관과 그것이 소표본 성능을 개선하는 이유를 설명하라.

??? success "풀이"
    OLS 잔차는 $\hat{e}_i = y_i - \hat{y}_i = (1 - h_{ii})\varepsilon_i + (\text{다른 } \varepsilon_j \text{에 관한 항})$이다. 따라서 (교차항을 무시하면) $E[\hat{e}_i^2] \approx (1 - h_{ii})^2 \sigma_i^2$이고, 이는 $\hat{e}_i^2$이 $\sigma_i^2$을 $(1 - h_{ii})^2$배만큼 체계적으로 과소추정한다는 뜻이다.

    - **HC0**은 $\hat{e}_i^2$을 그대로 써서 아래로 편향된다.
    - **HC2**는 $(1 - h_{ii})$로 나누어 $E[\hat{e}_i^2 / (1 - h_{ii})] \approx (1 - h_{ii})\sigma_i^2$이 되므로 여전히 편향된다.
    - **HC3**는 $(1 - h_{ii})^2$으로 나누어 $\hat{e}_i^2 / (1 - h_{ii})^2 \approx \sigma_i^2$이 되므로 $\sigma_i^2$의 거의 불편한 추정값을 준다.

    지렛값이 큰 점($h_{ii}$가 큰 점)일수록 잔차가 가장 심하게 축소된다. HC3의 더 강한 보정은 이런 영향력 있는 관측값이 분산 추정에 적절히 기여하도록 하며, 몇몇 점이 큰 지렛값을 가질 수 있는 소표본에서 특히 중요하다.

---

**연습문제 4.**
예제 코드의 설계는 칸당 관측값이 $n = 1$뿐이다($3 \times 3$ 설계에 관측값 9개, 모수 9개). 이 경우 HC3 추정량이 왜 문제가 되는지 설명하고 믿을 만한 추론을 위한 최소 칸 크기를 제안하라.

??? success "풀이"
    칸당 $n = 1$이고 관측값 9개에 모수 9개를 적합하면 햇 행렬이 $H = I$(항등행렬)가 되어 모든 관측값에서 $h_{ii} = 1$이다. HC3의 분모 $(1 - h_{ii})^2 = 0$이 되어 추정량이 정의되지 않는다(0으로 나눔).

    $h_{ii}$가 정확히 1은 아니더라도 1에 가까우면 HC3 추정값이 극도로 커지고 불안정해진다. 일반적으로 HC3는 제곱 잔차가 의미 있는 분산 추정값을 주려면 잔차 자유도가 충분해야 한다.

    흔한 권고는 HC3가 안정적으로 작동하려면 칸당 적어도 $n = 3$에서 $5$개의 관측값이 필요하다는 것이다. $3 \times 3$ 설계에서 칸당 $n \ge 5$이면($N = 45$, $p = 9$) 최대 지렛값이 1보다 충분히 낮게 억제되어 HC3 추정량이 잘 작동한다.

---

**연습문제 5.**
이분산 아래에서 HC3 샌드위치 추정량이 $\text{Var}(\hat{\boldsymbol{\beta}})$에 대해 일치성을 가짐을, 즉 $n \to \infty$일 때 OLS 추정량의 참 분산으로 수렴함을 증명하라.

??? success "풀이"
    이분산 아래에서 OLS 추정량의 참 분산은

    $$
    \text{Var}(\hat{\boldsymbol{\beta}}) = (X^\top X)^{-1} X^\top \Omega\, X\, (X^\top X)^{-1}
    $$

    이며 $\Omega = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$이다. HC3 추정량은 $\Omega$를 $\hat{\Omega}_{\text{HC3}} = \text{diag}(\hat{e}_i^2 / (1 - h_{ii})^2)$로 대체한다.

    정칙 조건 아래에서 $n \to \infty$일 때: (1) $h_{ii} \le p/n \to 0$이므로 각 지렛값이 0으로 가고 $(1 - h_{ii})^2 \to 1$이다. (2) OLS의 일치성에 의해 각 $i$에서 $\hat{e}_i^2 \to \varepsilon_i^2$이다. (3) 큰 수의 법칙에 의해 표본평균 $(1/n) X^\top \hat{\Omega}_{\text{HC3}} X \to (1/n) X^\top \Omega\, X$이다.

    따라서 $\widehat{\text{Cov}}_{\text{HC3}}(\hat{\boldsymbol{\beta}})$는 확률적으로 $\text{Var}(\hat{\boldsymbol{\beta}})$로 수렴한다. HC3 보정은 (같은 점근 성질을 갖는) HC0에 대한 유한표본 개선이며 $n \to \infty$에서는 같은 극한으로 수렴한다. $\square$
