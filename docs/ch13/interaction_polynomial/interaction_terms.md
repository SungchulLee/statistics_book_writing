# 교호작용 항

## 왜 교호작용 항을 쓰는가

현실의 많은 상황에서 한 변수가 결과에 미치는 효과는 일정하지 않고 다른 변수에 따라 달라진다. **교호작용 항**은 둘 이상의 독립변수가 종속변수에 미치는 이런 결합 효과를 모형화한다.

교호작용이 자연스럽게 나타나는 예:

- 운동과 식단이 체중 감량에 미치는 효과를 연구할 때, 운동의 영향은 특정 식단을 하는 사람에게서 더 클 수 있다.
- 매출 예측에서 마케팅 캠페인의 효과는 경기 상황이나 계절에 따라 달라질 수 있다.
- 금융에서 금리 변화가 자산 가격에 미치는 효과는 현재의 변동성 국면에 의존할 수 있다.

교호작용이 존재하는데도 모형화하지 않으면, 각 설명변수의 효과가 다른 설명변수의 값과 무관하게 동일하다고 가정하는 셈이므로 회귀 결과가 오도할 수 있다.

---

## 수학적 정식화

두 변수 $X_1$과 $X_2$의 교호작용은 곱항으로 회귀모형에 추가된다.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 (X_1 \times X_2) + \epsilon
$$

여기서

- $Y$는 종속변수,
- $\beta_1$과 $\beta_2$는 $X_1$과 $X_2$의 **주효과**,
- $\beta_3$은 $X_1$과 $X_2$ 사이의 **교호작용 효과**,
- $\epsilon$은 오차항이다.

항 $X_1 \times X_2$가 교호작용을 포착한다. $X_1$이 $Y$에 미치는 부분효과는 더 이상 상수 $\beta_1$이 아니라 $\beta_1 + \beta_3 X_2$가 되어 $X_2$의 수준에 의존한다.

!!! note "주효과를 함께 넣기"
    교호작용 항을 포함할 때는 대응하는 두 주효과($X_1$과 $X_2$)도 일반적으로 모형에 남겨야 한다. 교호작용을 넣으면서 주효과를 빼면 계수가 편향되고 해석할 수 없게 된다.

---

## 교호작용 항의 해석

$\beta_3$의 부호와 크기가 교호작용의 성격을 정한다.

- **양의 교호작용 계수 ($\beta_3 > 0$)**: 두 설명변수가 함께 증가할 때 $Y$에 미치는 결합 효과가 각각의 효과를 더한 것보다 **크다**. 두 설명변수가 서로를 강화한다.

- **음의 교호작용 계수 ($\beta_3 < 0$)**: 두 설명변수의 결합 효과가 각각의 효과를 더한 것보다 **작다**. 한 설명변수가 다른 쪽의 효과를 억누른다.

- **교호작용이 0 ($\beta_3 = 0$)**: $X_1$이 $Y$에 미치는 효과가 $X_2$에 의존하지 않는다. 모형은 교호작용이 없는 가법모형으로 줄어든다.

교호작용 항이 통계적으로 유의하다는 것은 한 변수와 결과의 관계가 다른 변수에 따라 달라진다는 뜻이다.

---

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 공부 시간, 수면, 성적. 공부 시간과 수면이 학생의 시험 성적에 미치는 효과를 연구한다고 하자. 교호작용 항이 없으면 모형은 학생이 얼마나 자든 공부 시간 추가의 이득이 동일하다고 가정한다.

공부 시간과 수면의 교호작용 항은 수면이 부족할 때 공부 시간 추가의 이득이 **줄어든다**는 사실을 드러낼 수 있다. 형식적으로 쓰면

$$
\text{Score} = \beta_0 + \beta_1 \cdot \text{StudyHours} + \beta_2 \cdot \text{Sleep} + \beta_3 \cdot (\text{StudyHours} \times \text{Sleep}) + \epsilon
$$

$\beta_3 > 0$이면 잠을 더 잘수록 공부의 이득이 커진다. $\beta_3 < 0$이면 적게 자는 학생에게는 공부를 더 해도 수익체감이 나타난다.

</div>

<div class="codebox" markdown>

### 예제 1. 교호작용 항 { .eg }

두 예제에서 쓸 자료를 먼저 읽는다. ISLR 교재의 Advertising과 Credit이다.

```python
import pandas as pd

# Advertising: 광고비(TV, Radio, Newspaper)와 매출(Sales), 200개 시장
advertising = pd.read_csv("https://www.statlearning.com/s/Advertising.csv",
                          index_col=0)
advertising = advertising.rename(columns={"radio": "Radio",
                                          "newspaper": "Newspaper",
                                          "sales": "Sales"})

# Credit: 신용카드 잔액(Balance)과 소득(Income), 학생 여부(Student), 400명
credit = pd.read_csv("https://raw.githubusercontent.com/vincentarelbundock/"
                     "Rdatasets/master/csv/ISLR/Credit.csv", index_col=0)

print(advertising[["TV", "Radio", "Sales"]].describe().round(2).to_string())
print()
print(credit.groupby("Student")[["Income", "Balance"]].mean().round(2).to_string())
```

출력:

```
           TV   Radio   Sales
count  200.00  200.00  200.00
mean   147.04   23.26   14.02
std     85.85   14.85    5.22
min      0.70    0.00    1.60
25%     74.38    9.98   10.38
50%    149.75   22.90   12.90
75%    218.82   36.52   17.40
max    296.40   49.60   27.00

         Income  Balance
Student                 
No        44.99   480.37
Yes       47.29   876.82
```

Advertising은 시장 200곳의 광고비와 매출, Credit은 400명의 소득과 잔액이다.

Credit 자료에서 학생의 평균 잔액이 877달러로 비학생의 480달러보다 훨씬 높은데 소득은 비슷하다는 점을 눈여겨보라. 아래 교호작용 모형이 이 차이를 어떻게 나누는지 볼 것이다.

</div>

<div class="codebox" markdown>

### 예제 2. 마케팅 효과 (TV와 Radio) { .eg }

광고 분석의 고전적 예는 TV와 Radio 광고 지출이 매출에 미치는 영향이다. 주효과만 있는 모형은 각 매체가 독립적인 효과를 갖는다고 가정한다.

$$
\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \epsilon
$$

그러나 **상승효과**가 있을 수 있다. TV와 Radio에 함께 광고하면 각각의 효과를 더한 것보다 더 효과적일 수 있다. 교호작용 항이 이를 포착한다.

$$
\text{Sales} = \beta_0 + \beta_1 \cdot \text{TV} + \beta_2 \cdot \text{Radio} + \beta_3 \cdot (\text{TV} \times \text{Radio}) + \epsilon
$$

**해석**:

- $\beta_3 > 0$이면 TV와 Radio 광고를 결합할 때 각 매체가 따로 주는 것을 넘어서는 상승효과가 매출에 생긴다.
- $\beta_3 < 0$이면 효과가 체감한다. 두 매체에 동시에 많이 쓰는 것은 덜 효율적일 수 있다.

이 모형은 statsmodels의 식(formula) 문법으로 편리하게 적합할 수 있다.

```python
import statsmodels.formula.api as smf

# Using formula syntax (R-like)
# The * operator includes main effects and the interaction
model = smf.ols('Sales ~ TV * Radio', data=advertising).fit()
# summary()는 실행 날짜와 시각을 함께 찍으므로 계수 표만 인쇄한다.
print(model.summary().tables[1])
print(f"R^2 = {model.rsquared:.4f}")
```

출력:

```
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept      6.7502      0.248     27.233      0.000       6.261       7.239
TV             0.0191      0.002     12.699      0.000       0.016       0.022
Radio          0.0289      0.009      3.241      0.001       0.011       0.046
TV:Radio       0.0011   5.24e-05     20.727      0.000       0.001       0.001
==============================================================================
R^2 = 0.9678
```

교호작용 항 `TV:Radio`의 계수가 0.0011이고 $t = 20.7$로 압도적으로 유의하다. TV와 라디오 광고 사이에 상승효과가 있다는 뜻이다.

크기를 가늠해 보자. 라디오에 0을 쓸 때 TV 1단위의 효과는 0.0191이지만, 라디오에 30을 쓰면 $0.0191 + 0.0011 \times 30 = 0.052$로 2.7배가 된다. **교호작용이 있으면 주효과를 단독으로 해석할 수 없다**는 말의 뜻이 이것이다.

$R^2$도 0.968로, 교호작용 없는 모형(0.897)보다 크게 높다.

!!! warning "`statsmodels.api`에는 소문자 `ols`가 없다"
    식 인터페이스는 `statsmodels.formula.api`(관례적으로 `smf`)에 있다. `statsmodels.api`(관례적으로 `sm`)에는 배열을 받는 대문자 `sm.OLS`만 있으므로 `sm.ols(...)`를 호출하면 `AttributeError`가 난다.

</div>

<div class="codebox" markdown>

### 예제 3. 소득과 학생 여부의 교호작용 { .eg }

신용카드 잔액이 소득과 학생 여부에 어떻게 의존하는지 살피는 모형을 생각하자. **질적 변수**(학생: 예/아니오)가 연속변수(소득)와 교호작용할 수 있다.

$$
\text{Balance} = \beta_0 + \beta_1 \cdot \text{Income} + \beta_2 \cdot \text{Student} + \beta_3 \cdot (\text{Income} \times \text{Student}) + \epsilon
$$

여기서 Student는 1(예) 또는 0(아니오)으로 부호화한다.

**해석**:

- $\beta_1$: **학생이 아닌 사람**에게 Income이 Balance에 미치는 효과는 $\beta_1$이다.
- $\beta_1 + \beta_3$: **학생**에게 Income이 Balance에 미치는 효과는 $\beta_1 + \beta_3$이다.
- $\beta_3 \neq 0$이면 Income과 Balance의 관계가 **학생 여부에 따라 다르다**.

이런 교호작용은 서로 다른 집단이 같은 설명변수에 다르게 반응하는지를 드러내며, 세분화와 표적 분석에 결정적인 통찰을 준다.

```python
# Example with categorical variable
# statsmodels automatically encodes categorical variables
model = smf.ols('Balance ~ Income + C(Student) + Income:C(Student)',
                data=credit).fit()
print(model.summary().tables[1])
print(f"R^2 = {model.rsquared:.4f}")
```

출력:

```
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
Intercept                  200.6232     33.698      5.953      0.000     134.373     266.873
C(Student)[T.Yes]          476.6758    104.351      4.568      0.000     271.524     681.827
Income                       6.2182      0.592     10.502      0.000       5.054       7.382
Income:C(Student)[T.Yes]    -1.9992      1.731     -1.155      0.249      -5.403       1.404
============================================================================================
R^2 = 0.2799
```

교호작용 항의 계수가 $-2.00$이지만 $p = 0.249$로 유의하지 않다. 소득이 잔액에 미치는 효과가 학생과 비학생에서 다르다는 증거가 약하다는 뜻이다.

주효과는 강하다. 학생이라는 것만으로 잔액이 평균 477달러 높고(`C(Student)[T.Yes]`), 소득이 1(천 달러) 늘 때마다 6.22달러 는다.

`C(Student)[T.Yes]`라는 이름은 patsy가 No를 기준(reference)으로 삼았다는 뜻이다. 기준 수준이 무엇인지 확인하지 않으면 계수의 부호를 거꾸로 읽게 된다.

이런 교호작용을 시각화하면 흔히 기울기가 다른 두 회귀직선(집단마다 하나씩)이 나타나며, Income이 Balance에 미치는 차별적 효과를 보여준다.

</div>

## 고차 및 다원 교호작용

교호작용은 변수 쌍에만 국한되지 않는다. $X_1$, $X_2$, $X_3$ 사이의 **3원 교호작용**은 다음 형태이다.

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_1 X_2 + \beta_5 X_1 X_3 + \beta_6 X_2 X_3 + \beta_7 X_1 X_2 X_3 + \epsilon
$$

실무에서 3원 이상의 고차 교호작용은 해석하기 어려워 드물게만 쓴다. 대부분의 응용 연구는 2원 교호작용에 집중한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
모형 $Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$에서 $X_1$이 $Y$에 미치는 효과가 $X_2$에 따라 어떻게 변하는지의 관점에서 $\beta_3$을 해석하라.

</div>

??? success "풀이"
    $X_1$이 $Y$에 미치는 부분효과는

    $$
    \frac{\partial E[Y]}{\partial X_1} = \beta_1 + \beta_3 X_2
    $$

    따라서 $\beta_3$은 $X_2$가 한 단위 늘어날 때 **$X_1$의 기울기가 변하는 양**을 나타낸다. $\beta_3 > 0$이면 $X_2$가 커질수록 $X_1$이 $Y$에 미치는 효과가 강해진다. $\beta_3 < 0$이면 약해진다. $\beta_3 = 0$이면 $X_2$와 무관하게 $X_1$의 효과가 동일하다(교호작용 없음).

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
연봉을 예측하는 모형에 경력, 학위(이진: 0 = 학위 없음, 1 = 학위 있음), 그리고 그 교호작용이 들어 있다. 추정된 식은 $\hat{Y} = 30000 + 2000 X_1 + 10000 X_2 + 1500 X_1 X_2$이다. 학위가 있는 사람과 없는 사람에 대해 회귀식을 따로 쓰고 그 차이를 해석하라.

</div>

??? success "풀이"
    **학위 없음** ($X_2 = 0$): $\hat{Y} = 30000 + 2000 X_1$. 경력 1년 추가마다 연봉이 \$2,000 오른다.

    **학위 있음** ($X_2 = 1$): $\hat{Y} = (30000 + 10000) + (2000 + 1500) X_1 = 40000 + 3500 X_1$. 경력 1년 추가마다 연봉이 \$3,500 오른다.

    교호작용 항(\$1,500)은 학위가 경력의 수익률을 연간 \$1,500만큼 키운다는 뜻이다. 학위 소지자는 출발점도 높고(\$40,000 대 \$30,000) 경력 1년당 얻는 것도 많다(\$3,500 대 \$2,000).

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
주효과($X_1$이나 $X_2$)를 빼면서 교호작용 항($X_1 X_2$)은 남겨 두는 것이 일반적으로 부적절한 이유를 설명하라. 이는 어떤 통계적 원칙을 어기는가?

</div>

??? success "풀이"
    이는 **위계 원칙**(주변성 원칙)을 어긴다. 교호작용 항을 포함한다면 그것을 이루는 모든 저차 항도 함께 있어야 한다는 원칙이다.

    교호작용을 남긴 채 주효과를 빼면 남은 항들의 해석이 달라진다. 예를 들어 $Y = \beta_0 + \beta_2 X_2 + \beta_3 X_1 X_2 + \varepsilon$에서 $X_1$을 빼면, 모형은 $X_2 = 0$일 때 $X_1$의 효과가 0이라고 강제하게 되는데 이는 강하고 대개 정당화되지 않는 제약이다. 또한 교호작용 계수가 $X_2$의 부호화 방식에 의존하게 되어 해석 가능성이 무너진다.

---

## 정리하며

교호작용 항은 한 설명변수의 효과가 다른 설명변수의 수준에 의존하도록 허용함으로써 다중회귀의 틀을 확장한다. 현실의 여러 관계를 정확히 모형화하는 데 필수적이며, 이론이나 분야 지식이 설명변수의 효과가 순수하게 가법적이지 않다고 시사할 때는 반드시 검정해 보아야 한다.
