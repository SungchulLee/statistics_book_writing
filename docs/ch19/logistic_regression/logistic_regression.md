# 로지스틱 회귀 실습


## 개요

이 절에서는 scikit-learn과 statsmodels 두 가지로 파이썬에서 로지스틱 회귀를 실습한다. 공부
시간과 시험 결과(합격/불합격)를 연결하는 인공자료를 생성해 모형을 적합하고, 오즈비와 신뢰구간을
계산하며, 가능도비 검정을 수행하고, 혼동행렬·ROC 곡선·정밀도-재현율 곡선으로 예측을 평가한 뒤
유든의 J 통계량으로 문턱을 고른다.

## 자료 생성

이항 결과 $y_i \in \{0,1\}$을 잠재 선형모형을 시그모이드에 통과시켜 생성한다.

$$
z_i = -3 + 0.7\,x_i + 0.3\,\varepsilon_i, \qquad
p_i = \frac{1}{1+e^{-z_i}}, \qquad
y_i \sim \operatorname{Bernoulli}(p_i)
$$

여기서 $x_i$는 $[1,10]$에서 균등하게 뽑은 공부 시간이고 $\varepsilon_i \sim N(0,1)$이다.

```python
import numpy as np

np.random.seed(42)
n = 300
hours_studied = np.random.uniform(1, 10, n)
noise = np.random.normal(0, 1, n)
logit = -3 + 0.7 * hours_studied + 0.3 * noise
prob = 1 / (1 + np.exp(-logit))
passed = np.random.binomial(1, prob)

X = hours_studied.reshape(-1, 1)
y = passed
```

!!! note "이 자료는 로지스틱 모형을 정확히 따르지 않는다"
    선형예측자에 $0.3\varepsilon_i$가 더해져 있으므로, $x$만 관측하는 분석자의 관점에서
    $P(Y=1\mid x)$는 로지스틱 함수들의 혼합이지 로지스틱 함수 자체가 아니다. 이런 관측되지
    않은 이질성은 계수를 0 쪽으로 **감쇠**시킨다. $\sigma = 0.3$은 작아서 효과가 미미하지만,
    적합된 기울기가 참값 $0.7$과 정확히 일치하지 않는 데에는 표집오차뿐 아니라 이 요인도 있다.

## scikit-learn으로 적합하기

scikit-learn의 `LogisticRegression`은 기본적으로 벌점 로그가능도를 최대화한다(C = 1.0, L2
벌점). 추정된 절편과 기울기는 로지스틱 모형에 그대로 대응한다.

$$
\log\frac{P(Y=1\mid x)}{1-P(Y=1\mid x)} = \hat\beta_0 + \hat\beta_1\,x
$$

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

print(f"Intercept: {model.intercept_[0]:.4f}")
print(f"Coefficient (hours): {model.coef_[0][0]:.4f}")
print(f"Train accuracy: {model.score(X_train, y_train):.3f}")
print(f"Test accuracy:  {model.score(X_test, y_test):.3f}")

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)
```

출력:

```
Intercept: -2.9986
Coefficient (hours): 0.7099
Train accuracy: 0.805
Test accuracy:  0.833
```

출력은 절편 $-2.9986$, 기울기 $0.7099$, 훈련 정확도 $0.805$, 검정 정확도 $0.833$이다.

## statsmodels로 추론하기

statsmodels는 최대가능도 추정을 통해 표준오차, 왈드 검정, 신뢰구간을 제공한다(기본적으로 벌점을
주지 않는다).

```python
import statsmodels.api as sm
from scipy import stats

X_sm = sm.add_constant(hours_studied)
logit_model = sm.Logit(y, X_sm)
result = logit_model.fit(disp=0)
def print_summary(res):
    """summary()의 Date/Time 칸은 실행할 때마다 달라지므로 비우고 출력한다."""
    lines = []
    for line in str(res.summary()).split("\n"):
        if line.startswith(("Date:", "Time:")):
            lines.append(line[:19].ljust(38) + line[38:])
        else:
            lines.append(line)
    print("\n".join(lines))


print_summary(result)
```

출력:

```
Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                  300
Model:                          Logit   Df Residuals:                      298
Method:                           MLE   Df Model:                            1
Date:                                   Pseudo R-squ.:                  0.3820
Time:                                   Log-Likelihood:                -123.42
converged:                       True   LL-Null:                       -199.70
Covariance Type:            nonrobust   LLR p-value:                 4.760e-35
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const         -3.2519      0.408     -7.979      0.000      -4.051      -2.453
x1             0.7572      0.083      9.130      0.000       0.595       0.920
==============================================================================
```

!!! warning "두 결과를 나란히 비교하기 전에"
    위 statsmodels 코드는 **전체 자료 300건**에 적합하지만 scikit-learn 코드는 **훈련자료
    210건**에만 적합했다. 계수가 다르게 나오는 것($-3.2519$ 대 $-2.9986$)은 알고리즘 차이가
    아니라 대부분 이 때문이다. 조건을 맞춰 훈련자료에만 적합하면,

    | 적합 | $\hat\beta_0$ | $\hat\beta_1$ |
    |---|---|---|
    | sklearn, `C=1.0`(L2 벌점) | $-2.9986$ | $0.7099$ |
    | sklearn, `penalty=None` | $-3.0275$ | $0.7162$ |
    | statsmodels(벌점 없음) | $-3.0238$ | $0.7155$ |

    이 되어 statsmodels와 벌점 없는 sklearn이 사실상 일치한다. 남는 차이 $0.7162$ 대
    $0.7099$가 기본 L2 벌점이 만든 축소다.

### 오즈비

로짓 연결은 오즈의 로그이므로, 계수를 지수화하면 오즈비가 된다.

$$
\text{OR}_j = e^{\hat\beta_j}
$$

공부 시간이 한 단위 늘면 합격 오즈에 $e^{\hat\beta_1}$이 곱해진다.

```python
import numpy as np

print("Odds Ratios:")
print(np.exp(result.params))

print("95% CI for Odds Ratios:")
print(np.exp(result.conf_int()))
```

출력:

```
Odds Ratios:
[0.0387011  2.13232758]
95% CI for Odds Ratios:
[[0.01740966 0.08603126]
 [1.81241582 2.50870736]]
```

전체 자료 적합에서 $\hat\beta_1 = 0.7572$이므로 오즈비는 $e^{0.7572} = 2.1323$이고 95%
신뢰구간은 $(1.8124,\ 2.5087)$이다. 즉 공부 시간 한 시간마다 합격 오즈가 약 두 배가 된다.
구간이 1을 포함하지 않으므로 효과는 유의하다.

### 가능도비 검정

가능도비 검정은 적합모형을 영모형(절편만)과 비교한다.

$$
\Lambda = -2\bigl[\ell(\hat{\boldsymbol\beta}_0) - \ell(\hat{\boldsymbol\beta})\bigr]
\;\sim\; \chi^2_1
$$

```python
null_model = sm.Logit(y, sm.add_constant(np.ones(n))).fit(disp=0)
lr_stat = -2 * (null_model.llf - result.llf)
lr_pvalue = stats.chi2.sf(lr_stat, df=1)
print(f"Likelihood Ratio Test: chi2 = {lr_stat:.4f}, p = {lr_pvalue:.6f}")
```

출력:

```
Likelihood Ratio Test: chi2 = 152.5683, p = 0.000000
```

결과는 $\Lambda = 152.57$, $p = 4.8 \times 10^{-35}$로 영가설을 압도적으로 기각한다.

## 혼동행렬과 분류 보고서

기본 문턱 $\tau = 0.5$에서 혼동행렬은

$$
\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}
$$

이고 표준 지표들은 다음과 같다.

$$
\text{Accuracy} = \frac{\text{TP}+\text{TN}}{n}, \qquad
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \qquad
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}
$$

$$
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}
$$

```python
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, precision_score,
                              recall_score, f1_score)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print(f"TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))
```

출력:

```
Confusion Matrix:
[[30  7]
 [ 8 45]]
TN=30, FP=7, FN=8, TP=45
Accuracy:  0.833
Precision: 0.865
Recall:    0.849
F1 Score:  0.857
              precision    recall  f1-score   support

           0       0.79      0.81      0.80        37
           1       0.87      0.85      0.86        53

    accuracy                           0.83        90
   macro avg       0.83      0.83      0.83        90
weighted avg       0.83      0.83      0.83        90
```

검정자료 90건에서 TN $= 30$, FP $= 7$, FN $= 8$, TP $= 45$이고, 정확도 $0.833$,
정밀도 $0.865$, 재현율 $0.849$, $F_1 = 0.857$이다.

## ROC 곡선과 AUC

ROC 곡선은 문턱을 변화시키며 FPR에 대한 TPR을 그린다. 곡선 아래 면적(AUC)이 판별력을 요약한다.

$$
\text{AUC} = \int_0^1 \text{TPR}\bigl(\text{FPR}\bigr)\,d(\text{FPR})
$$

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"AUC = {auc:.4f}")
```

출력:

```
AUC = 0.9001
```

AUC $= 0.9001$로, 설명변수가 단 하나인 모형치고는 매우 좋은 판별력이다.

## 정밀도-재현율 곡선

양성 범주가 드물 때는 정밀도-재현율 곡선이 ROC 곡선보다 유용한 정보를 주는 경우가 많다. 평균
정밀도(AP)가 이 곡선을 요약한다.

$$
\text{AP} = \sum_{k} (R_k - R_{k-1})\,P_k
$$

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)
print(f"Average Precision = {ap:.4f}")
```

출력:

```
Average Precision = 0.9272
```

AP $= 0.9272$이다. 이 자료는 검정자료의 양성 비율이 $53/90 = 0.589$로 오히려 양성이 다수이므로,
AP의 무작위 기준선도 $0.589$로 높다는 점을 함께 보아야 한다.

## 문턱 선택

기본 문턱 $\tau = 0.5$가 항상 최적인 것은 아니다. **유든의 J 통계량**은
$J = \text{TPR} - \text{FPR}$를 최대화하는 문턱을 고른다.

```python
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]
print(f"Optimal threshold (Youden's J): {optimal_threshold:.3f}")
print(f"  TPR = {tpr[optimal_idx]:.3f}, FPR = {fpr[optimal_idx]:.3f}")
```

출력:

```
Optimal threshold (Youden's J): 0.716
  TPR = 0.811, FPR = 0.054
```

유든의 J가 고른 문턱은 $0.716$이고 그때 TPR $= 0.811$, FPR $= 0.054$다.

아래 코드는 문턱에 따라 정확도, 정밀도, 재현율, $F_1$이 어떻게 변하는지 보여준다.

```python
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_t = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_t)
    prec = precision_score(y_test, y_pred_t, zero_division=0)
    rec = recall_score(y_test, y_pred_t, zero_division=0)
    f1 = f1_score(y_test, y_pred_t, zero_division=0)
    print(f"tau={threshold:.1f}  Acc={acc:.3f}  Prec={prec:.3f}  "
          f"Rec={rec:.3f}  F1={f1:.3f}")
```

출력:

```
tau=0.3  Acc=0.744  Prec=0.721  Rec=0.925  F1=0.810
tau=0.4  Acc=0.756  Prec=0.763  Rec=0.849  F1=0.804
tau=0.5  Acc=0.833  Prec=0.865  Rec=0.849  F1=0.857
tau=0.6  Acc=0.844  Prec=0.898  Rec=0.830  F1=0.863
tau=0.7  Acc=0.856  Prec=0.935  Rec=0.811  F1=0.869
```

| $\tau$ | 정확도 | 정밀도 | 재현율 | $F_1$ |
|---|---|---|---|---|
| 0.3 | $0.744$ | $0.721$ | $0.925$ | $0.810$ |
| 0.4 | $0.756$ | $0.763$ | $0.849$ | $0.804$ |
| 0.5 | $0.833$ | $0.865$ | $0.849$ | $0.857$ |
| 0.6 | $0.844$ | $0.898$ | $0.830$ | $0.863$ |
| 0.7 | $0.856$ | $0.935$ | $0.811$ | $0.869$ |

## 해석

- 공부 시간의 계수가 양수이므로, 공부를 더 하면 합격의 로그오즈(따라서 확률)가 올라간다.
- 계수를 지수화하면 오즈비가 된다. 한 시간이 늘 때마다 합격 오즈에 $e^{\hat\beta_1} = 2.13$이
  곱해진다.
- 가능도비 검정은 공부 시간이 효과가 없다는 영가설을 기각한다.
- AUC는 모형이 두 범주를 모든 문턱에 걸쳐 얼마나 잘 분리하는지를 하나의 수로 요약한다.
- 유든의 J는 위양성과 위음성의 비용이 같을 때 문턱을 고르는 원리적인 방법을 제공한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$n = 500$이고 특성이 두 개인 인공자료를 참 모형
$\log\frac{p}{1-p} = -1 + 0.5\,x_1 - 0.3\,x_2$로부터 생성하라. scikit-learn으로 로지스틱
회귀를 적합해 추정된 계수를 참값과 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    np.random.seed(0)
    n = 500
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    logit = -1 + 0.5 * x1 - 0.3 * x2
    p = 1 / (1 + np.exp(-logit))
    y = np.random.binomial(1, p)

    X = np.column_stack([x1, x2])
    model = LogisticRegression(penalty=None, solver='lbfgs')
    model.fit(X, y)
    print(f"Intercept: {model.intercept_[0]:.4f} (true: -1)")
    print(f"Coef x1:   {model.coef_[0][0]:.4f} (true: 0.5)")
    print(f"Coef x2:   {model.coef_[0][1]:.4f} (true: -0.3)")
    ```

    출력:

    ```
    Intercept: -0.7256 (true: -1)
    Coef x1:   0.4547 (true: 0.5)
    Coef x2:   -0.0607 (true: -0.3)
    ```

    (표준오차와 신뢰구간은 같은 자료에 `sm.Logit`을 적합해 얻은 값이다. sklearn은 이를
    제공하지 않는다.)

    | 모수 | 참값 | 추정치 | 표준오차 | 95% 신뢰구간 |
    |---|---|---|---|---|
    | $\beta_0$ | $-1.0$ | $-0.7256$ | $0.0982$ | $(-0.918,\ -0.533)$ |
    | $\beta_1$ | $0.5$ | $0.4546$ | $0.1011$ | $(0.257,\ 0.653)$ |
    | $\beta_2$ | $-0.3$ | $-0.0607$ | $0.0998$ | $(-0.256,\ 0.135)$ |

    $\beta_1$은 참값에 가깝지만 **$\beta_2$는 심하게 빗나갔다.** 추정치 $-0.0607$은 참값
    $-0.3$에서 표준오차의 $2.4$배만큼 떨어져 있고, 95% 신뢰구간이 참값을 **포함하지 못한다.**
    이는 오류가 아니라 20번에 한 번쯤 일어나는 일이며, 하필 이 난수 씨앗에서 일어난 것이다.

    표본을 키우면 사라진다. 같은 코드를 $n = 5000$으로 돌리면
    $\hat\beta = (-0.996,\ 0.524,\ -0.305)$, 표준오차 $0.033$ 수준으로 셋 다 참값에 잘
    맞는다.

    **교훈:** $n = 500$에 이항 결과이면 실효 정보량은 생각보다 훨씬 적다. 여기서 사건 수는
    약 130건이므로 계수당 표준오차가 $0.1$ 수준이고, 크기 $0.3$인 효과는 신호 대 잡음비가
    3에 불과하다. 단일 표본의 점추정치를 참값처럼 읽지 말고 반드시 신뢰구간과 함께 보아야
    한다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
유든의 $J = \text{TPR} - \text{FPR}$를 최대화하는 것이 ROC 곡선에서 대각선까지의 수직거리가
가장 큰 지점을 찾는 것과 같음을 대수적으로 보여라.

</div>

??? success "풀이"

    ROC 그림의 대각선은 직선 $\text{TPR} = \text{FPR}$이다. 곡선 위의 점
    $(\text{FPR}, \text{TPR})$에서 대각선까지의 수직거리는, $x = \text{FPR}$에서 대각선의 값이
    $\text{FPR}$ 자신이므로

    $$
    d = \text{TPR} - \text{FPR}
    $$

    이다. 따라서 $J = \text{TPR} - \text{FPR}$를 최대화하는 것은 곡선에서 대각선까지의
    수직거리를 최대화하는 것과 정확히 같다.

    한 가지 덧붙이면, 유든의 J를 최대화하는 문턱은 **기울기가 1인 접선이 ROC 곡선에 닿는
    지점**이기도 하다. ROC 곡선의 기울기는 그 문턱에서의 가능도비
    $f_1(\tau)/f_0(\tau)$와 같으므로, $J$ 최적점은 가능도비가 1이 되는 곳이다. 이는
    $C_{FP} = C_{FN}$이고 유병률이 $0.5$일 때의 베이즈 규칙과 일치한다. 즉 유든의 J는
    비용과 유병률을 모두 대칭으로 가정한 특수한 선택이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
어떤 로지스틱 회귀가 한 환자에게 $\hat{p} = 0.72$를 출력했다. 문턱 0.5에서의 혼동행렬은
$\text{TP}=80$, $\text{FP}=15$, $\text{FN}=20$, $\text{TN}=85$였다. 정확도, 정밀도, 재현율,
$F_1$을 계산하라.

</div>

??? success "풀이"

    $$
    \text{Accuracy} = \frac{80+85}{200} = 0.825
    $$

    $$
    \text{Precision} = \frac{80}{80+15} = \frac{80}{95} \approx 0.842
    $$

    $$
    \text{Recall} = \frac{80}{80+20} = \frac{80}{100} = 0.800
    $$

    $$
    F_1 = \frac{2 \times 0.842 \times 0.800}{0.842 + 0.800} \approx 0.821
    $$

    동등하게 $F_1 = \dfrac{2 \times 80}{2 \times 80 + 15 + 20} = \dfrac{160}{195} = 0.8205$
    로 계산해도 된다. 이 형태가 반올림 오차를 피할 수 있어 더 안전하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
$F_1$ 점수가 정밀도와 재현율의 조화평균임을 증명하라.

</div>

??? success "풀이"

    양수 $a$와 $b$의 조화평균은

    $$
    H = \frac{2}{\frac{1}{a} + \frac{1}{b}} = \frac{2ab}{a+b}
    $$

    이다. $a = \text{Precision}$, $b = \text{Recall}$로 두면

    $$
    H = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}} = F_1
    $$

    이다. 이는 $F_1$이 정밀도와 재현율의 극단적인 불균형에 산술평균보다 큰 벌칙을 준다는 것을
    보여준다. 둘 중 하나가 0에 가까우면 조화평균은 급격히 아래로 끌려간다. $\square$

<div class="drillbox" markdown>

**연습문제 5.**
가능도비 통계량 $\Lambda = -2[\ell_0 - \ell_1]$은 $H_0$ 아래에서 $\chi^2_p$를 따르며, $p$는
전체 모형이 추가로 갖는 모수의 개수다. 영모형의 로그가능도가 $-180$, 설명변수 3개를 가진
전체 모형의 로그가능도가 $-160$이라 하자. $\Lambda$를 계산하고 $\alpha = 0.01$에서 $H_0$을
기각하는지 답하라.

</div>

??? success "풀이"

    $$
    \Lambda = -2\bigl[(-180) - (-160)\bigr] = -2(-20) = 40
    $$

    $H_0$ 아래에서 $\Lambda \sim \chi^2_3$이고 $\alpha = 0.01$의 임계값은
    $\chi^2_{3,0.99} = 11.34$다. $40 \gg 11.34$이므로 $H_0$을 기각하고, 세 설명변수 중 적어도
    하나가 결과에 통계적으로 유의한 효과를 갖는다고 결론짓는다.

    동등하게 $p\text{-값} = P(\chi^2_3 \geq 40) = 1.07 \times 10^{-8}$로 0.01보다 훨씬 작다.

    !!! note "유의성과 크기는 다르다"
        $\Lambda = 40$은 세 변수가 **통계적으로** 유의하다고 말할 뿐 그 효과가 **실질적으로**
        크다는 뜻은 아니다. 이탈도가 $360$에서 $320$으로 줄었다면 맥패든 유사 $R^2$는
        $1 - 320/360 = 0.11$에 불과하다. 표본이 크면 아주 작은 개선도 유의해진다. 언제나
        p-값과 효과크기를 함께 보고하라. $\square$
