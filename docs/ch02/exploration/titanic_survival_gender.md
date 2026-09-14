# 사례 연구: 타이타닉 생존과 성별

앞의 두 절은 **한 변수**를 들여다보는 도구였다. 이 절은 하나의 실제 자료에 하나의 질문을 놓고, **두 변수의 관계를 탐색하는 과정 전체**를 처음부터 끝까지 따라간다.

자료는 1912년 타이타닉호 승객 891명의 기록이고, 질문은 하나다.

> **성별과 생존은 관계가 있는가? 있다면 얼마나 강한가?**

같은 질문에 답하는 방법이 여섯 가지다. 상관계수, 분할표, 생존율, 오즈비, 검정, 그림. **여섯 가지가 모두 같은 $2\times2$ 표 하나에서 나오는데, 전달하는 정보의 양은 크게 다르다.** 이 절의 목적은 그 차이를 보이는 것이다.

!!! note "이 절에서 쓰는 도구는 다른 절에 각각 설명되어 있다"
    분할표와 독립성은 [모자이크 그림과 도수분포표](../visualization/mosaic.md), 막대그림은 [막대그림](../visualization/bar_charts.md), 쌍그림은 [쌍그림](../visualization/pair_plots.md) 절에서 다룬다. 여기서는 그 도구들을 **한 자료에 차례로 적용해 보는 것**이 목적이므로, 각 도구의 원리는 해당 절에 미룬다.

    자료는 인터넷에서 내려받는다. 네트워크가 없으면 실행되지 않지만, **출력과 그림을 모두 실어 두었으므로 읽는 데는 지장이 없다.**

## 1. 자료와 첫 점검

<div class="codebox" markdown>

### 예제 1. 자료를 불러오고 결측부터 확인하기 { .eg }

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# 타이타닉 승객 명부. index_col 로 승객번호를 색인으로 삼는다.
URL = ("https://raw.githubusercontent.com/datasciencedojo/"
       "datasets/master/titanic.csv")
df = pd.read_csv(URL, index_col="PassengerId")

print(f"승객 {df.shape[0]}명, 변수 {df.shape[1]}개\n")

# 무엇이든 하기 전에 결측부터 센다.
# 결측이 있는 변수를 모르고 분석하면 n 이 조용히 줄어든다.
na = df.isna().sum()
print("결측이 있는 변수")
for name, cnt in na[na > 0].items():
    print(f"  {name:9s} {cnt:4d}개  ({cnt / len(df):.1%})")

# 성별을 0/1 로 부호화한다. 1=남성, 0=여성 으로 정했다.
# 이 선택이 뒤에 나올 상관계수의 '부호'를 정한다는 점을 기억해 두자.
df["Sex_int"] = (df["Sex"] == "male").astype(int)

print(f"\n성별 분포")
print(df["Sex"].value_counts().to_string())
print(f"\n생존 분포 (0=사망, 1=생존)")
print(df["Survived"].value_counts().sort_index().to_string())
print(f"\n전체 생존율 {df['Survived'].mean():.4f}")
```

```text
승객 891명, 변수 11개

결측이 있는 변수
  Age        177개  (19.9%)
  Cabin      687개  (77.1%)
  Embarked     2개  (0.2%)

성별 분포
Sex
male      577
female    314

생존 분포 (0=사망, 1=생존)
Survived
0    549
1    342

전체 생존율 0.3838
```

**결측을 먼저 세는 것이 습관이 되어야 한다.** `Age`가 177개(19.9%) 비어 있는데, 이 사실을 모르고 나이를 쓰는 분석을 하면 **표본이 891명에서 714명으로 조용히 줄어든다.** 6절에서 실제로 그런 일이 일어난다.

**이 절의 주된 질문에는 결측이 없다.** `Survived`와 `Sex`는 891명 모두 기록되어 있으므로, 5절까지는 전수를 쓴다.

**성별 부호화는 자의적이다.** 남성을 1로 둘 수도 여성을 1로 둘 수도 있고, 어느 쪽을 골라도 분석의 내용은 같다. 다만 **상관계수의 부호가 뒤집힌다.** 바로 다음에서 이 점이 문제가 된다.

</div>

## 2. 상관계수 하나로 요약하면

두 변수가 모두 0과 1뿐이지만, 피어슨 상관계수를 계산하는 데는 아무 문제가 없다. 공식이 요구하는 것은 두 수치 변수일 뿐이다.

<div class="codebox" markdown>

### 예제 2. 두 이진 변수의 상관은 파이 계수다 { .eg }

```python
# Survived 와 Sex_int 는 둘 다 0/1 이지만 피어슨 공식은 그대로 적용된다.
corr = df[["Survived", "Sex_int"]].corr()
print(corr.round(4).to_string())

r = corr.loc["Survived", "Sex_int"]
print(f"\n상관계수 {r:.4f}")

# 두 이진 변수의 피어슨 상관에는 '파이 계수'라는 이름이 따로 붙어 있다.
# 파이 계수는 카이제곱 통계량과 다음 관계로 이어져 있다.
#       phi = sqrt(chi2 / n)
# 이것을 직접 확인해 본다. (연속성 보정을 끈 카이제곱을 써야 맞는다)
from scipy.stats import chi2_contingency

tab = pd.crosstab(df["Sex"], df["Survived"])
res = chi2_contingency(tab, correction=False)
n = tab.values.sum()
print(f"카이제곱(보정 없음) {res.statistic:.4f}")
print(f"sqrt(chi2/n) = {np.sqrt(res.statistic / n):.4f}   |r| = {abs(r):.4f}")

# 부호를 뒤집어 보면 -- 부호화만 바꾸었을 뿐인데 부호가 바뀐다.
df["Sex_female"] = (df["Sex"] == "female").astype(int)
print(f"\n1=남성 으로 부호화: r = {np.corrcoef(df['Survived'], df['Sex_int'])[0, 1]:+.4f}")
print(f"1=여성 으로 부호화: r = {np.corrcoef(df['Survived'], df['Sex_female'])[0, 1]:+.4f}")
```

```text
          Survived  Sex_int
Survived    1.0000  -0.5434
Sex_int    -0.5434   1.0000

상관계수 -0.5434
카이제곱(보정 없음) 263.0506
sqrt(chi2/n) = 0.5434   |r| = 0.5434

1=남성 으로 부호화: r = -0.5434
1=여성 으로 부호화: r = +0.5434
```

**$\varphi=\sqrt{\chi^2/n}$ 가 소수점 넷째 자리까지 맞는다.** 두 이진 변수의 피어슨 상관은 **파이 계수**이며, 카이제곱 통계량을 $n$으로 나눈 것의 제곱근과 정확히 같다. 5절의 검정과 여기의 상관계수가 사실은 **같은 하나의 수**인 셈이다.

**그런데 $-0.5434$라는 숫자가 무엇을 말하는가.**

| 묻는 것 | $-0.5434$가 답하는가 |
|---|---|
| 관계가 있는가 | 그렇다 |
| 방향은 | **부호화가 정한다** |
| 여성의 생존율은 | **모른다** |
| 남성의 생존율은 | **모른다** |
| 몇 배 차이인가 | **모른다** |

**부호가 자의적이라는 것이 특히 곤란하다.** 남성을 1로 두면 $-0.54$, 여성을 1로 두면 $+0.54$다. **자료가 아니라 코드 한 줄이 부호를 정한다.** "음의 상관"이라는 서술은 여기서 아무 의미가 없다.

!!! warning "파이 계수는 $\pm1$에 도달하지 못할 수 있다"
    두 변수의 주변 비율이 다르면 $\varphi$의 최댓값이 1보다 작아진다. 여기서는 생존 비율이 0.3838, 여성 비율이 0.3524이므로

    $$
    \varphi_{\max}=\frac{\min(p,q)-pq}{\sqrt{p(1-p)q(1-q)}}=0.9347
    $$

    이다. 관측된 $|\varphi|=0.5434$는 **도달 가능한 최댓값의 58%**다. 이것을 모르고 "0.54면 중간 정도"라고 읽으면 강도를 과소평가한다. 자세한 내용은 [점이연 상관과 파이 계수](../../ch12/correlation/special_cases.md) 절에 있다.

</div>

## 3. 분할표가 훨씬 많이 말한다

<div class="codebox" markdown>

### 예제 3. 분할표, 생존율, 오즈비 { .eg }

```python
# margins=True 를 주면 행·열의 합계가 함께 나온다.
print("도수")
print(pd.crosstab(df["Sex"], df["Survived"], margins=True).to_string())

# normalize="index" 는 '각 행의 합이 1이 되도록' 나눈다.
# 즉 성별 안에서의 생존율이다. 무엇을 무엇으로 나누는지가 핵심이다.
pct = pd.crosstab(df["Sex"], df["Survived"], normalize="index") * 100
print("\n성별 안에서의 생존율 (%)")
print(pct.round(2).to_string())

f_rate = pct.loc["female", 1]
m_rate = pct.loc["male", 1]
print(f"\n여성 생존율 {f_rate:.2f}%")
print(f"남성 생존율 {m_rate:.2f}%")
print(f"차이       {f_rate - m_rate:.2f}%포인트")
print(f"비(위험비)  {f_rate / m_rate:.4f}배")

# 오즈비: (여성의 생존 오즈) / (남성의 생존 오즈)
a = tab.loc["female", 1]; b = tab.loc["female", 0]   # 여성 생존 / 사망
c = tab.loc["male", 1];   d = tab.loc["male", 0]     # 남성 생존 / 사망
print(f"\n여성 오즈 {a}/{b} = {a / b:.4f}")
print(f"남성 오즈 {c}/{d} = {c / d:.4f}")
print(f"오즈비    {(a * d) / (b * c):.4f}")
```

```text
도수
Survived    0    1  All
Sex                    
female     81  233  314
male      468  109  577
All       549  342  891

성별 안에서의 생존율 (%)
Survived      0      1
Sex                   
female    25.80  74.20
male      81.11  18.89

여성 생존율 74.20%
남성 생존율 18.89%
차이       55.31%포인트
비(위험비)  3.9280배

여성 오즈 233/81 = 2.8765
남성 오즈 109/468 = 0.2329
오즈비    12.3507
```

**이제 비로소 말할 수 있는 것이 생겼다.**

| 요약 | 값 | 읽는 법 |
|---|---|---|
| 상관계수 | $-0.5434$ | 관계가 있다 |
| **생존율** | **74.20% 대 18.89%** | **여성 넷 중 셋, 남성 다섯 중 하나** |
| 차이 | 55.31%포인트 | 절대적 격차 |
| 위험비 | 3.93배 | 여성이 약 4배 |
| **오즈비** | **12.35** | 오즈로는 12배 |

**같은 표에서 나온 세 가지 "배율"이 3.93과 12.35로 크게 다르다.** 틀린 것이 아니라 **다른 것을 재고 있다.**

$$
\text{위험비}=\frac{0.7420}{0.1889}=3.93,
\qquad
\text{오즈비}=\frac{0.7420/0.2580}{0.1889/0.8111}=12.35
$$

**위험비는 확률의 비, 오즈비는 오즈의 비**다. 생존율이 0에 가까울 때는 둘이 비슷해지지만, 여기처럼 74%나 되면 크게 벌어진다. **어느 쪽을 보고하든 이름을 정확히 붙여야 한다.**

**전체 생존율 38.38%는 여기서 아무 쓸모가 없다.** 두 집단의 값이 74%와 19%인데 그 평균을 말하는 것은 **양쪽 어느 쪽도 설명하지 않는 수**다.

</div>

## 4. 독립성 검정

<div class="codebox" markdown>

### 예제 4. 카이제곱 독립성 검정 { .eg }

```python
from scipy.stats import chi2_contingency

# scipy 의 기본값은 correction=True 이며, 2x2 표에 예이츠 연속성 보정을 건다.
# 두 경우를 모두 계산해 비교해 본다.
for corr_flag in [True, False]:
    res = chi2_contingency(tab, correction=corr_flag)
    label = "보정 있음(기본값)" if corr_flag else "보정 없음"
    print(f"{label:16s} chi2 = {res.statistic:8.4f}   p = {res.pvalue:.4e}")

res = chi2_contingency(tab, correction=False)
print(f"\n자유도 {res.dof}")
print("\n귀무가설(독립)이 맞다면 기대되는 도수")
exp = pd.DataFrame(res.expected_freq, index=tab.index, columns=tab.columns)
print(exp.round(2).to_string())

print("\n관측 - 기대")
print((tab - exp).round(2).to_string())

# 검정통계량이 어디서 나오는지 한 칸씩 뜯어본다.
print("\n칸별 기여도 (관측-기대)^2 / 기대")
cells = ((tab - exp) ** 2 / exp)
print(cells.round(2).to_string())
print(f"합계 {cells.values.sum():.4f}")
```

```text
보정 있음(기본값)       chi2 = 260.7170   p = 1.1974e-58
보정 없음            chi2 = 263.0506   p = 3.7117e-59

자유도 1

귀무가설(독립)이 맞다면 기대되는 도수
Survived       0       1
Sex                     
female    193.47  120.53
male      355.53  221.47

관측 - 기대
Survived       0       1
Sex                     
female   -112.47  112.47
male      112.47 -112.47

칸별 기여도 (관측-기대)^2 / 기대
Survived      0       1
Sex                    
female    65.39  104.96
male      35.58   57.12
합계 263.0506
```

**$p=3.7\times10^{-59}$이다.** 독립이라는 가정 아래 이만큼 치우친 표가 나올 확률이 사실상 0이라는 뜻이다.

**네 칸의 "관측 $-$ 기대"가 모두 절댓값 112.47로 같다.** 우연이 아니다. $2\times2$ 표에서 행·열 합이 고정되면 **자유롭게 움직일 수 있는 칸이 하나뿐**이고, 한 칸이 정해지면 나머지 셋이 따라 정해진다. 자유도가 1인 이유가 이것이다.

**연속성 보정은 여기서 거의 영향이 없다.** 263.05가 260.72로 바뀔 뿐이다. 기대도수가 모두 120을 넘을 만큼 표본이 크기 때문이다. 기대도수가 5 미만인 칸이 있을 때라야 보정이 의미를 갖는다.

!!! danger "$p$값은 효과의 크기를 말하지 않는다"
    $p=3.7\times10^{-59}$은 **"관계가 있다"**는 것만 말한다. **"얼마나 강한가"는 말하지 않는다.**

    같은 생존율 차이(74.20% 대 18.89%)를 유지한 채 표본을 10분의 1로 줄이면 $p$는 $10^{-6}$ 수준으로 커지지만, **효과의 크기는 조금도 변하지 않는다.** $p$값은 효과와 표본 크기를 뒤섞은 수다.

    그러므로 검정 결과는 **언제나 효과크기와 함께** 보고한다. 여기서는 생존율 차이 55.31%포인트, 오즈비 12.35, $\varphi=0.54$가 그것이다.

</div>

## 5. 같은 표를 네 가지로 그리기

<div class="codebox" markdown>

### 예제 5. 네 가지 그림이 각각 다른 것을 강조한다 { .eg }

```python
import matplotlib.pyplot as plt
import seaborn as sns

# 네 칸에서 성별 순서를 하나로 고정한다.
# 칸마다 순서가 다르면 독자가 같은 그림 안에서 두 번 방향을 바꿔 읽어야 한다.
ORDER = ["female", "male"]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (1) 도수 -- 집단 크기가 다르다는 사실이 보인다
sns.countplot(data=df, x="Sex", hue="Survived", order=ORDER,
              ax=axes[0, 0], palette="Set2")
axes[0, 0].set_title("(1) count by gender")
axes[0, 0].set_xlabel("gender"); axes[0, 0].set_ylabel("count")

# (2) 생존율 -- 집단 크기를 지우고 비율만 남긴다
rate = df.groupby("Sex")["Survived"].mean().reindex(ORDER) * 100
rate.plot(kind="bar", ax=axes[0, 1], color=["coral", "skyblue"],
          edgecolor="black")
axes[0, 1].set_title("(2) survival rate (%)")
axes[0, 1].set_xlabel("gender"); axes[0, 1].set_ylabel("survival rate (%)")
axes[0, 1].tick_params(axis="x", rotation=0)
axes[0, 1].grid(True, alpha=0.3, axis="y"); axes[0, 1].set_ylim(0, 85)
for i, v in enumerate(rate):                     # 막대 위에 값을 적어 준다
    axes[0, 1].text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")

# (3) 누적 막대 -- 집단 크기와 구성비를 한 막대에 함께 담는다
tab_o = pd.crosstab(df["Sex"], df["Survived"]).reindex(ORDER)
tab_o.plot(kind="bar", stacked=True, ax=axes[1, 0],
           color=["indianred", "seagreen"], edgecolor="black")
axes[1, 0].set_title("(3) stacked count")
axes[1, 0].set_xlabel("gender"); axes[1, 0].set_ylabel("count")
axes[1, 0].tick_params(axis="x", rotation=0)
axes[1, 0].legend(["did not survive", "survived"], title="Survived")

# (4) 비율 열지도 -- 네 칸의 수를 그대로 읽게 한다
# vmin/vmax 를 0과 1로 고정해야 색의 진하기가 비율과 맞는다.
prop = pd.crosstab(df["Sex"], df["Survived"],
                   normalize="index").reindex(ORDER)
sns.heatmap(prop, annot=True, fmt=".2%", cmap="Blues", ax=axes[1, 1],
            vmin=0, vmax=1, cbar_kws={"label": "proportion"},
            linewidths=1, linecolor="black")
axes[1, 1].set_title("(4) proportion heatmap")
axes[1, 1].set_xlabel("Survived (0=no, 1=yes)")
axes[1, 1].set_ylabel("gender")

fig.suptitle("Titanic: one 2x2 table, four views", y=1.00)
fig.tight_layout()
plt.show()
```

![같은 분할표를 네 가지로 그린 그림](./img/titanic_gender_four.png)

**네 그림이 같은 표에서 나왔는데 강조점이 다르다.**

| 그림 | 잘 보이는 것 | 가려지는 것 |
|---|---|---|
| (1) 도수 | **집단 크기가 다르다**(남 577, 여 314) | 비율 비교가 어렵다 |
| (2) 생존율 | **74.2% 대 18.9%** | 표본 크기가 사라진다 |
| (3) 누적 | 크기와 구성을 함께 | 위쪽 조각의 길이 비교가 어렵다 |
| (4) 열지도 | 네 수를 **정확히** | 크기 감각이 없다 |

**질문이 "얼마나 차이 나는가"이면 (2)가 답한다.** 이 절의 질문에는 (2)가 가장 곧바른 그림이다.

**그러나 (2)만 보면 위험하다.** 비율만 남기고 $n$을 지우므로, **여성이 3명뿐이어도 똑같은 그림**이 나온다. (1)이나 (3)을 함께 놓아 표본 크기를 함께 보여야 한다.

**누적 막대 (3)의 약점은 위쪽 조각이다.** 아래 조각(사망)은 모두 바닥에서 시작하므로 길이를 비교하기 쉽지만, 위 조각(생존)은 시작점이 제각각이라 눈으로 견주기 어렵다. 이것이 누적 막대의 일반적인 한계다.

**색 선택에 관하여.** 열지도에 `RdYlGn`(빨강-노랑-초록)을 쓰는 관행이 있는데, **적록색각 이상이 있는 독자에게는 읽히지 않는다.** 남성의 8% 정도가 여기에 해당하므로, 파랑 계열이나 명도 차가 뚜렷한 색표를 쓰는 편이 안전하다.

</div>

## 6. 쌍그림은 왜 잘 안 되는가

여러 변수를 한꺼번에 훑을 때는 쌍그림이 표준 도구다. 여기에도 써 보자. 나이까지 넣어 세 변수를 본다.

<div class="codebox" markdown>

### 예제 6. 이진 변수에 쌍그림을 쓰면 { .eg }

```python
sub = df[["Survived", "Age", "Sex_int"]]

# seaborn 은 결측이 있는 행을 말없이 버린다. 몇 명이 남는지 직접 센다.
print(f"원래 {len(sub)}명")
print(f"Age 결측 {sub['Age'].isna().sum()}명")
print(f"쌍그림에 실제로 쓰이는 관측 {sub.dropna().shape[0]}명")
print(f"  -> 전체의 {sub.dropna().shape[0] / len(sub):.1%}")

# 버려진 177명이 남은 714명과 다른 사람들인지 확인한다.
# 무작위로 빠진 것이 아니라면 결과가 편향된다.
miss = sub["Age"].isna()
print(f"\n나이가 기록된 승객의 생존율   {df.loc[~miss, 'Survived'].mean():.4f}")
print(f"나이가 빠진 승객의 생존율     {df.loc[miss, 'Survived'].mean():.4f}")
print(f"나이가 기록된 승객의 남성 비율 {df.loc[~miss, 'Sex_int'].mean():.4f}")
print(f"나이가 빠진 승객의 남성 비율   {df.loc[miss, 'Sex_int'].mean():.4f}")

g = sns.pairplot(sub, diag_kind="hist",
                 plot_kws={"alpha": 0.6, "s": 18},
                 diag_kws={"bins": 30})
g.figure.suptitle("pairplot: Survived, Age, Sex_int", y=1.01)
plt.show()
```

```text
원래 891명
Age 결측 177명
쌍그림에 실제로 쓰이는 관측 714명
  -> 전체의 80.1%

나이가 기록된 승객의 생존율   0.4062
나이가 빠진 승객의 생존율     0.2938
나이가 기록된 승객의 남성 비율 0.6345
나이가 빠진 승객의 남성 비율   0.7006
```

![쌍그림: 이진 변수에서는 칸이 점 몇 개로 무너진다](./img/titanic_pairplot.png)

**쌍그림이 여기서 두 번 실패한다.**

**첫째, 이진 변수의 산점도는 그림이 아니다.** `Survived`와 `Sex_int`의 칸을 보면 **점이 네 개**다. 891명의 자료가 네 점으로 겹쳐 버렸다. 나머지 칸들도 두 줄로 늘어선 띠일 뿐이어서, 띠의 길이는 나이의 범위를 말하지만 **어느 쪽에 사람이 많은지는 전혀 보이지 않는다.**

**둘째, 177명이 조용히 사라졌다.** 그리고 사라진 사람들이 무작위가 아니다.

| | 생존율 | 남성 비율 |
|---|---|---|
| 나이가 **기록된** 714명 | 0.4062 | 0.6345 |
| 나이가 **빠진** 177명 | **0.2938** | **0.7006** |

**나이가 기록되지 않은 승객은 남성 비율이 높고 생존율이 낮다.** 무작위 결측이 아니므로, 이 714명으로 얻은 결론을 891명 전체에 그대로 옮길 수 없다.

**쌍그림은 연속형 변수를 위한 도구다.** 이진·범주형 변수에는 이 절의 5절에서 쓴 도구들 — 분할표, 막대그림, 모자이크 그림 — 이 맞는다. 자세한 논의는 [쌍그림](../visualization/pair_plots.md) 절에 있다.

**그래도 `Age` 칸 하나는 쓸모가 있다.** 대각선의 나이 히스토그램은 20대에 봉우리가 있고 오른쪽으로 긴 꼬리를 가진 모양을 보여 준다. **세 변수 중 연속형인 하나에 대해서만 쌍그림이 제 일을 한 것이다.**

</div>

## 7. 여섯 가지 요약을 나란히

같은 $2\times2$ 표에서 나온 여섯 가지를 한자리에 모으면 이렇다.

| 요약 | 값 | 강점 | 약점 |
|---|---|---|---|
| 상관계수 $\varphi$ | $-0.5434$ | 한 수, 비교 가능 | **부호가 자의적**, 해석 불가 |
| $\chi^2$ 검정 | $p=3.7\times10^{-59}$ | 우연 여부 판정 | **크기를 말하지 않음** |
| 분할표 | 81 / 233 / 468 / 109 | **원자료 전부** | 읽는 데 시간이 든다 |
| **생존율** | **74.20% 대 18.89%** | **바로 이해된다** | 표본 크기가 안 보임 |
| 위험비 | 3.93배 | 직관적 | 기저 비율에 의존 |
| 오즈비 | 12.35 | 회귀와 연결 | **일상어가 아니다** |

**보고해야 할 것을 하나만 고른다면 생존율이다.** "여성 74%, 남성 19%"는 통계를 모르는 사람도 곧바로 이해하고, 다른 다섯 가지를 모두 되살릴 수 있는 정보를 담고 있다.

**상관계수는 여섯 중 가장 정보가 적다.** 그런데도 "관계의 강도"를 묻는 질문에 가장 먼저 계산되는 일이 잦다. **한 수로 줄이는 대가가 무엇인지 알고 줄여야 한다.**

!!! tip "이 사례에서 멈추지 말 것"
    이 절은 **성별과 생존의 연관**을 보였을 뿐, **성별 때문에 살았다**는 것을 보이지 않았다.

    타이타닉에서 여성의 생존율이 높았던 데에는 "여성과 어린이 먼저"라는 대피 관행이 작용했겠지만, **객실 등급**도 함께 움직인다. 1등실 승객은 구명정에 가깝고 여성 비율도 달랐다. 등급을 나누어 보면 이야기가 달라질 수 있고, 그런 층별 분석에서 전체 경향이 뒤집히는 현상이 **심슨의 역설**이다.

    연관에서 인과로 넘어가는 문제는 [12장](../../ch12/causation/causation.md)에서 본격적으로 다룬다.

## 정리하며

하나의 $2\times2$ 표를 여섯 가지 방법으로 요약하고 네 가지로 그려 보았다.

- **결측부터 센다.** `Age`의 19.9% 결측이 6절에서 표본을 714명으로 줄였고, 그 177명은 무작위가 아니었다.
- **두 이진 변수의 피어슨 상관은 파이 계수**이며 $\varphi=\sqrt{\chi^2/n}$로 검정통계량과 이어져 있다. 수치로 확인했다($0.5434$).
- **상관계수의 부호는 부호화가 정한다.** 자료가 정하는 것이 아니다.
- **생존율(74.20% 대 18.89%)이 가장 잘 전달되는 요약**이고, 상관계수가 가장 정보가 적다.
- **$p$값과 효과크기는 다른 것을 잰다.** $p=3.7\times10^{-59}$은 크기에 대해 아무 말도 하지 않는다.
- **같은 표를 그리는 방법마다 강조점이 다르다.** 비율 그림은 표본 크기를 지우므로 도수 그림과 함께 놓는다.
- **쌍그림은 연속형 변수를 위한 도구다.** 이진 변수에서는 칸이 네 점으로 무너진다.

다음 장에서는 이런 탐색에서 얻은 인상을 **확률의 언어**로 다루기 시작한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
예제 3의 분할표에서 **행 기준 백분율**과 **열 기준 백분율**, **전체 기준 백분율**을 각각 계산하고, 세 가지가 답하는 질문이 어떻게 다른지 말하라.

</div>

??? success "풀이"
    **세 가지 정규화가 모두 다른 조건부확률**이다.

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import pandas as pd

    URL = ("https://raw.githubusercontent.com/datasciencedojo/"
           "datasets/master/titanic.csv")
    df = pd.read_csv(URL, index_col="PassengerId")

    for how, label in [("index", "행 기준 (성별 안에서)"),
                       ("columns", "열 기준 (생존 여부 안에서)"),
                       ("all", "전체 기준")]:
        t = pd.crosstab(df["Sex"], df["Survived"], normalize=how) * 100
        print(f"\n{label}")
        print(t.round(2).to_string())
    ```

    ```text

    행 기준 (성별 안에서)
    Survived      0      1
    Sex                   
    female    25.80  74.20
    male      81.11  18.89

    열 기준 (생존 여부 안에서)
    Survived      0      1
    Sex                   
    female    14.75  68.13
    male      85.25  31.87

    전체 기준
    Survived      0      1
    Sex                   
    female     9.09  26.15
    male      52.53  12.23
    ```

    **세 표가 서로 다른 질문에 답한다.**

    | 정규화 | 읽는 법 | 예 |
    |---|---|---|
    | **행**($P(\text{생존}\mid\text{성별})$) | "여성 중 몇 %가 살았나" | **74.20%** |
    | **열**($P(\text{성별}\mid\text{생존})$) | "생존자 중 몇 %가 여성인가" | **68.13%** |
    | 전체($P(\text{성별},\text{생존})$) | "전체 승객 중 몇 %가 여성이고 생존했나" | 26.15% |

    **74.20%와 68.13%를 혼동하는 것이 가장 흔한 실수**다. 두 수는 조건과 결과가 뒤바뀐 별개의 확률이다.

    $$
    P(\text{생존}\mid\text{여성})=0.7420,
    \qquad
    P(\text{여성}\mid\text{생존})=0.6813
    $$

    **베이즈 정리가 둘을 잇는다.**

    $$
    P(\text{여성}\mid\text{생존})
    =\frac{P(\text{생존}\mid\text{여성})\,P(\text{여성})}{P(\text{생존})}
    =\frac{0.7420\times0.3524}{0.3838}=0.6813
    $$

    **검산이 맞는다.**

    **이 절의 질문에는 행 기준이 맞다.** "성별이 생존에 영향을 주었는가"를 묻고 있으므로 **성별을 조건으로** 놓아야 한다.

    **열 기준이 맞는 질문도 있다.** "구조된 사람들의 구성은 어땠는가"를 묻는다면 열 기준이다.

    **전체 기준은 주변확률을 되살릴 때 쓴다.** 행을 더하면 성별 비율(35.24%, 64.76%), 열을 더하면 생존 비율(61.62%, 38.38%)이 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
예제 2에서 $\varphi=\sqrt{\chi^2/n}$임을 수치로 확인했다. 이 등식을 $2\times2$ 표에서 **대수적으로 증명**하라.

</div>

??? success "풀이"
    **표기.** $2\times2$ 표를

    | | $Y=0$ | $Y=1$ | 합 |
    |---|---|---|---|
    | $X=0$ | $a$ | $b$ | $a+b$ |
    | $X=1$ | $c$ | $d$ | $c+d$ |
    | 합 | $a+c$ | $b+d$ | $n$ |

    라 두자.

    **1단계 — 파이 계수의 닫힌 꼴.** $X$와 $Y$가 0/1이므로

    $$
    \overline{XY}=\frac{d}{n},\quad
    \bar X=\frac{c+d}{n},\quad
    \bar Y=\frac{b+d}{n}
    $$

    이고, 0/1 변수는 $X^2=X$이므로 $\operatorname{Var}(X)=\bar X(1-\bar X)$다. 따라서

    $$
    \varphi=\frac{\overline{XY}-\bar X\bar Y}
    {\sqrt{\bar X(1-\bar X)\,\bar Y(1-\bar Y)}}
    $$

    분자를 정리하면

    $$
    \frac{d}{n}-\frac{(c+d)(b+d)}{n^2}
    =\frac{nd-(c+d)(b+d)}{n^2}
    =\frac{ad-bc}{n^2}
    $$

    ($n=a+b+c+d$를 대입해 전개하면 교차항이 상쇄된다.) 분모는

    $$
    \sqrt{\frac{(c+d)(a+b)}{n^2}\cdot\frac{(b+d)(a+c)}{n^2}}
    =\frac{\sqrt{(a+b)(c+d)(a+c)(b+d)}}{n^2}
    $$

    이므로

    $$
    \boxed{\;\varphi=\frac{ad-bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}\;}
    $$

    **2단계 — 카이제곱의 닫힌 꼴.** 기대도수는 $E_{11}=(a+b)(a+c)/n$ 등이고, 네 칸의 "관측 $-$ 기대"가 모두 $\pm(ad-bc)/n$으로 같다. 이를 대입하면 잘 알려진 결과

    $$
    \chi^2=\frac{n\,(ad-bc)^2}{(a+b)(c+d)(a+c)(b+d)}
    $$

    를 얻는다.

    **3단계 — 결합.** 두 식을 견주면

    $$
    \chi^2=n\varphi^2
    \quad\Longleftrightarrow\quad
    \varphi=\sqrt{\frac{\chi^2}{n}} \qquad\square
    $$

    ```python
    import numpy as np
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(df["Sex"], df["Survived"])
    a, b = tab.loc["female", 0], tab.loc["female", 1]
    c, d = tab.loc["male", 0],   tab.loc["male", 1]
    n = a + b + c + d

    phi_formula = (a * d - b * c) / np.sqrt(
        (a + b) * (c + d) * (a + c) * (b + d))
    chi2_formula = n * (a * d - b * c) ** 2 / (
        (a + b) * (c + d) * (a + c) * (b + d))

    df["Sex_int"] = (df["Sex"] == "male").astype(int)
    print(f"공식으로 구한 phi   {phi_formula:+.6f}")
    print(f"numpy 상관계수      {np.corrcoef(df['Survived'], df['Sex_int'])[0, 1]:+.6f}")
    print(f"공식으로 구한 chi2  {chi2_formula:.6f}")
    print(f"scipy chi2          "
          f"{chi2_contingency(tab, correction=False).statistic:.6f}")
    print(f"n * phi^2           {n * phi_formula**2:.6f}")
    ```

    ```text
    공식으로 구한 phi   -0.543351
    numpy 상관계수      -0.543351
    공식으로 구한 chi2  263.050574
    scipy chi2          263.050574
    n * phi^2           263.050574
    ```

    **세 값이 소수점 여섯 자리까지 맞는다.**

    **부호만 반대인 이유**는 표의 행 순서(female이 먼저)와 `Sex_int`의 부호화(male이 1)가 서로 반대이기 때문이다. **$\chi^2$는 제곱이라 부호를 잃으므로**, $\sqrt{\chi^2/n}$로는 크기만 얻고 방향은 따로 정해야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
연습문제 2의 공식 $\varphi=(ad-bc)/\sqrt{\cdots}$를 보면 **$\varphi=0$인 것과 독립인 것이 동치**임을 알 수 있다. 이를 설명하고, $ad=bc$가 무엇을 뜻하는지 밝혀라.

</div>

??? success "풀이"
    **분자만 보면 된다.** 분모는 주변도수의 곱이므로 어느 칸도 비어 있지 않은 한 양수다. 따라서

    $$
    \varphi=0
    \quad\Longleftrightarrow\quad
    ad-bc=0
    \quad\Longleftrightarrow\quad
    ad=bc
    $$

    **$ad=bc$는 오즈비가 1이라는 뜻**이다.

    $$
    \text{OR}=\frac{ad}{bc}=1
    $$

    **이것이 곧 독립이다.** 표기를 확률로 바꾸면

    $$
    \frac{a}{a+b}=\frac{c}{c+d}
    \quad\Longleftrightarrow\quad
    P(Y=0\mid X=0)=P(Y=0\mid X=1)
    $$

    **즉 $X$를 알아도 $Y$의 조건부확률이 바뀌지 않는다.**

    ```python
    import numpy as np

    # ad = bc 인 표를 일부러 만들어 본다.
    # 행 비율을 똑같이 유지하면 자동으로 그렇게 된다.
    tabs = {
        "독립 (비율 동일)":      np.array([[30, 70], [60, 140]]),
        "약한 연관":            np.array([[30, 70], [55, 145]]),
        "이 절의 타이타닉":      np.array([[81, 233], [468, 109]]),
    }
    for lab, T in tabs.items():
        a, b, c, d = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
        n = T.sum()
        phi = (a * d - b * c) / np.sqrt(
            (a + b) * (c + d) * (a + c) * (b + d))
        print(f"{lab:16s} ad={a * d:6d}  bc={b * c:6d}  "
              f"OR={a * d / (b * c):7.4f}  phi={phi:+.4f}")
        print(f"{'':16s} 1행 비율 {a / (a + b):.4f}, "
              f"2행 비율 {c / (c + d):.4f}")
    ```

    ```text
    독립 (비율 동일)       ad=  4200  bc=  4200  OR= 1.0000  phi=+0.0000
                     1행 비율 0.3000, 2행 비율 0.3000
    약한 연관            ad=  4350  bc=  3850  OR= 1.1299  phi=+0.0262
                     1행 비율 0.3000, 2행 비율 0.2750
    이 절의 타이타닉        ad=  8829  bc=109044  OR= 0.0810  phi=-0.5434
                     1행 비율 0.2580, 2행 비율 0.8111
    ```

    **첫 행에서 두 비율이 0.3000으로 정확히 같고 $\varphi$가 정확히 0이다.**

    **세 지표가 같은 것을 다르게 말한다.**

    | 지표 | 독립일 때 | 타이타닉 |
    |---|---|---|
    | $ad-bc$ | $0$ | $8829-109044<0$ |
    | 오즈비 | $1$ | 0.0810 |
    | $\varphi$ | $0$ | $-0.5434$ |

    **오즈비 0.0810의 역수가 12.35**로, 예제 3에서 구한 값이다(표의 행 순서가 반대라 역수로 나왔다). 행과 열의 순서를 어떻게 잡느냐에 따라 $\text{OR}$이 $12.35$로도 $1/12.35$로도 나오므로, **어느 쪽을 분자로 두었는지 반드시 밝혀야 한다.**

    **모자이크 그림이 이 사실을 눈으로 보여 준다.** $ad=bc$이면 두 열의 분할선 높이가 같아져 **선이 일직선으로 이어진다.** [모자이크 그림](../visualization/mosaic.md) 절의 3번에서 다룬 내용이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
$p$값이 효과크기를 말하지 않는다는 4절의 주장을 **수치로 확인**하라. 생존율 차이를 그대로 유지한 채 표본 크기만 바꾸어 보라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy.stats import chi2_contingency

    # 타이타닉의 비율(여성 74.20%, 남성 18.89%, 여성 35.24%)을 그대로 유지하고
    # 전체 인원만 바꾼다. 효과의 크기는 조금도 변하지 않는다.
    f_rate, m_rate, f_share = 0.7420, 0.1889, 0.3524

    print(f"{'n':>7s}{'chi2':>10s}{'p':>12s}{'phi':>9s}"
          f"{'생존율 차':>10s}{'유의':>6s}")
    for n in [20, 50, 100, 300, 891, 5000]:
        nf = round(n * f_share); nm = n - nf
        fs = round(nf * f_rate); ms = round(nm * m_rate)
        T = np.array([[nf - fs, fs], [nm - ms, ms]])
        res = chi2_contingency(T, correction=False)
        phi = np.sqrt(res.statistic / n)
        diff = fs / nf - ms / nm
        print(f"{n:>7d}{res.statistic:>10.3f}{res.pvalue:>12.2e}"
              f"{phi:>9.4f}{diff:>10.4f}"
              f"{'예' if res.pvalue < 0.05 else '아니오':>6s}")
    ```

    ```text
          n      chi2           p      phi     생존율 차    유의
         20     6.282    1.22e-02   0.5604    0.5604     예
         50    13.981    1.85e-04   0.5288    0.5347     예
        100    30.092    4.12e-08   0.5486    0.5582     예
        300    88.890    4.17e-21   0.5443    0.5546     예
        891   263.051    3.71e-59   0.5434    0.5531     예
       5000  1474.237    0.00e+00   0.5430    0.5528     예
    ```

    **$\varphi$와 생존율 차이는 $n$과 무관하게 거의 일정하다.**

    | $n$ | $\varphi$ | 생존율 차 | $p$ |
    |---|---|---|---|
    | 20 | 0.5604 | 0.560 | $1.2\times10^{-2}$ |
    | 100 | 0.5486 | 0.558 | $4.1\times10^{-8}$ |
    | 891 | 0.5434 | 0.553 | $3.7\times10^{-59}$ |
    | 5000 | 0.5430 | 0.553 | $\approx0$ |

    **효과크기는 0.55 근처에 붙박이인데 $p$값은 $10^{-2}$에서 $10^{-59}$까지 57자릿수를 움직인다.**

    **$\chi^2$이 $n$에 정비례한다**는 것이 그 이유다. 연습문제 2에서

    $$
    \chi^2=n\varphi^2
    $$

    이었으므로, **$\varphi$를 고정하면 $\chi^2$은 $n$에 비례해 커진다.** $n$을 5000으로 키우면 $\chi^2=5000\times0.5430^2\approx1474$이다.

    **역방향의 함정이 더 위험하다.** 아주 작은 효과도 $n$만 크면 유의해진다.

    ```python
    print("\n효과가 아주 작을 때 (생존율 40% 대 38%)")
    print(f"{'n':>8s}{'phi':>9s}{'p':>12s}{'유의':>6s}")
    for n in [100, 1_000, 10_000, 100_000]:
        nf = n // 2; nm = n - nf
        fs = round(nf * 0.40); ms = round(nm * 0.38)
        T = np.array([[nf - fs, fs], [nm - ms, ms]])
        res = chi2_contingency(T, correction=False)
        print(f"{n:>8d}{np.sqrt(res.statistic / n):>9.4f}"
              f"{res.pvalue:>12.2e}{'예' if res.pvalue < 0.05 else '아니오':>6s}")
    ```

    ```text

    효과가 아주 작을 때 (생존율 40% 대 38%)
           n      phi           p    유의
         100   0.0205    8.38e-01   아니오
        1000   0.0205    5.17e-01   아니오
       10000   0.0205    4.03e-02     예
      100000   0.0205    8.97e-11     예
    ```

    **$\varphi=0.02$라는 무시해도 좋을 효과가 $n=10{,}000$에서 유의해진다.** $n=100{,}000$이면 $p=9.0\times10^{-11}$이다.

    **결론 셋.**

    1. **$p$값은 효과와 표본 크기를 뒤섞은 수**다. 하나만으로는 어느 쪽이 큰지 알 수 없다.
    2. **유의성은 크기의 증거가 아니다.** 큰 표본에서는 하찮은 차이도 유의하다.
    3. **검정 결과는 언제나 효과크기와 함께** 보고한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
2절에서 $\varphi_{\max}=0.9347$이라 했다. 이 값을 직접 계산하고, **주변 비율을 바꾸면 $\varphi_{\max}$가 어떻게 변하는지** 보여라.

</div>

??? success "풀이"
    **$\varphi$의 최댓값.** 행 비율 $p$, 열 비율 $q$가 고정되어 있을 때, 겹침을 최대로 하면

    $$
    \varphi_{\max}=\frac{\min(p,q)-pq}{\sqrt{p(1-p)q(1-q)}}
    $$

    ```python
    import numpy as np

    def phi_max(p, q):
        return (min(p, q) - p * q) / np.sqrt(p * (1 - p) * q * (1 - q))

    # 타이타닉의 주변 비율
    p = (df["Survived"] == 1).mean()          # 생존 비율
    q = (df["Sex"] == "female").mean()        # 여성 비율
    print(f"생존 비율 p = {p:.4f},  여성 비율 q = {q:.4f}")
    print(f"phi_max    = {phi_max(p, q):.4f}")
    print(f"관측 |phi| = 0.5434")
    print(f"도달률     = {0.5434 / phi_max(p, q):.4f}\n")

    print("주변 비율이 다르면 phi_max 가 달라진다")
    print(f"{'p':>6s}{'q':>6s}{'phi_max':>10s}")
    for pp, qq in [(0.5, 0.5), (0.5, 0.35), (0.38, 0.35),
                   (0.5, 0.1), (0.1, 0.9), (0.05, 0.5)]:
        print(f"{pp:>6.2f}{qq:>6.2f}{phi_max(pp, qq):>10.4f}")
    ```

    ```text
    생존 비율 p = 0.3838,  여성 비율 q = 0.3524
    phi_max    = 0.9347
    관측 |phi| = 0.5434
    도달률     = 0.5814

    주변 비율이 다르면 phi_max 가 달라진다
         p     q   phi_max
      0.50  0.50    1.0000
      0.50  0.35    0.7338
      0.38  0.35    0.9373
      0.50  0.10    0.3333
      0.10  0.90    0.1111
      0.05  0.50    0.2294
    ```

    **$p=q$일 때만 $\varphi_{\max}=1$이다.**

    | $p$ | $q$ | $\varphi_{\max}$ |
    |---|---|---|
    | 0.50 | 0.50 | **1.0000** |
    | 0.38 | 0.35 | **0.9373** |
    | 0.50 | 0.35 | 0.7338 |
    | **0.10** | **0.90** | **0.1111** |

    **타이타닉은 운이 좋은 경우다.** 생존 비율 0.384와 여성 비율 0.352가 가까워 $\varphi_{\max}=0.935$로 1에 가깝다(표의 $0.38/0.35$ 행이 0.9373으로 이를 확인해 준다). 그래서 $\varphi=0.543$을 **거의 액면 그대로** 읽어도 큰 무리가 없다.

    **$p=0.1$, $q=0.9$라면 사정이 전혀 다르다.** 두 변수가 **완벽하게 연관되어도** $\varphi$가 0.111을 넘을 수 없다. 이때 $\varphi=0.09$를 "약한 연관"이라 부르면 **완전히 틀린 해석**이다. 도달률로는 81%다.

    **그래서 $\varphi$ 대신 오즈비를 쓰는 관행이 생겼다.** 오즈비는 주변 비율에 영향받지 않는다.

    ```python
    print("\n주변 비율만 바꾸고 연관의 '세기'는 유지했을 때")
    print(f"{'표':>28s}{'phi':>9s}{'OR':>9s}")
    for lab, T in [("균형 (50/50)", np.array([[40, 10], [10, 40]])),
                   ("한쪽이 드묾",   np.array([[80, 20], [ 2,  8]]))]:
        a, b, c, d = T[0, 0], T[0, 1], T[1, 0], T[1, 1]
        phi = (a * d - b * c) / np.sqrt(
            (a + b) * (c + d) * (a + c) * (b + d))
        print(f"{lab:>28s}{phi:>9.4f}{a * d / (b * c):>9.4f}")
    ```

    ```text

    주변 비율만 바꾸고 연관의 '세기'는 유지했을 때
                               표      phi       OR
                      균형 (50/50)   0.6000  16.0000
                          한쪽이 드묾   0.3960  16.0000
    ```

    **오즈비는 16으로 같은데 $\varphi$는 0.600과 0.396으로 다르다.** 주변 비율이 치우친 쪽에서 $\varphi$가 눌린 것이다.

    **실무 권고.** 서로 다른 표의 연관 강도를 **비교**할 때는 오즈비를, 하나의 표를 **묘사**할 때는 생존율 차이를 쓴다. $\varphi$는 주변 비율을 함께 밝히지 않으면 오해를 부른다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
6절에서 나이 결측이 무작위가 아님을 보았다. 결측을 **버리는 것**과 **평균으로 채우는 것**이 각각 어떤 왜곡을 낳는지 수치로 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd

    miss = df["Age"].isna()
    print(f"나이 결측 {miss.sum()}명 / 전체 {len(df)}명\n")

    print("결측 여부에 따른 차이")
    print(f"{'':14s}{'n':>6s}{'생존율':>9s}{'남성 비율':>10s}{'1등실 비율':>11s}")
    for lab, m in [("나이 있음", ~miss), ("나이 없음", miss)]:
        sub = df[m]
        print(f"{lab:14s}{len(sub):>6d}{sub['Survived'].mean():>9.4f}"
              f"{(sub['Sex'] == 'male').mean():>10.4f}"
              f"{(sub['Pclass'] == 1).mean():>11.4f}")
    ```

    ```text
    나이 결측 177명 / 전체 891명

    결측 여부에 따른 차이
                       n      생존율     남성 비율     1등실 비율
    나이 있음            714   0.4062    0.6345     0.2605
    나이 없음            177   0.2938    0.7006     0.1695
    ```

    **나이가 빠진 승객은 남성이 많고(0.7006 대 0.6345), 1등실이 적고(0.1695 대 0.2605), 덜 살아남았다(0.2938 대 0.4062).** 완전 무작위 결측(MCAR)이 아니다.

    **이제 두 처리 방식을 비교한다.**

    ```python
    # (가) 버리기 (listwise deletion)
    drop = df.dropna(subset=["Age"])

    # (나) 평균으로 채우기 (mean imputation)
    fill = df.copy()
    fill["Age"] = fill["Age"].fillna(df["Age"].mean())

    print(f"{'':16s}{'n':>6s}{'평균 나이':>10s}{'나이 SD':>9s}"
          f"{'나이-생존 상관':>13s}")
    for lab, d in [("원자료(관측만)", df.dropna(subset=["Age"])),
                   ("(가) 버리기", drop),
                   ("(나) 평균 대치", fill)]:
        r = np.corrcoef(d["Age"], d["Survived"])[0, 1]
        print(f"{lab:16s}{len(d):>6d}{d['Age'].mean():>10.4f}"
              f"{d['Age'].std(ddof=1):>9.4f}{r:>13.4f}")

    print(f"\n전체 생존율")
    print(f"  참값(891명)        {df['Survived'].mean():.4f}")
    print(f"  (가) 버린 뒤        {drop['Survived'].mean():.4f}")
    print(f"  (나) 대치 후        {fill['Survived'].mean():.4f}")
    ```

    ```text
                         n     평균 나이    나이 SD     나이-생존 상관
    원자료(관측만)           714   29.6991  14.5265      -0.0772
    (가) 버리기            714   29.6991  14.5265      -0.0772
    (나) 평균 대치          891   29.6991  13.0020      -0.0698

    전체 생존율
      참값(891명)        0.3838
      (가) 버린 뒤        0.4062
      (나) 대치 후        0.3838
    ```

    **두 방법이 서로 다른 것을 망친다.**

    | | 표본 크기 | 생존율 | 나이 SD | 나이-생존 상관 |
    |---|---|---|---|---|
    | 참값 | 891 | **0.3838** | — | — |
    | **(가) 버리기** | **714** | **0.4062** | 14.53 | $-0.0772$ |
    | **(나) 평균 대치** | 891 | 0.3838 | **13.00** | $-0.0698$ |

    **(가) 버리기는 생존율을 0.3838에서 0.4062로 부풀린다.** 덜 살아남은 사람들이 통째로 빠졌기 때문이다. 5.8%의 상대적 과대평가다.

    **(나) 평균 대치는 표본 크기와 평균은 지키지만 산포를 줄인다.** 나이의 표준편차가 14.53에서 13.00으로 **10.5% 축소**되었다. 177명 전원에게 똑같은 값(29.70)을 주었으니 당연하다.

    **분산이 줄면 상관도 희석된다.** 나이-생존 상관이 $-0.0772$에서 $-0.0698$로 약해졌다.

    ```python
    # 대치된 177명이 한 점에 쌓였는지 확인한다.
    v = fill.loc[miss, "Age"]
    print(f"\n대치된 177명의 나이: 최소 {v.min():.4f}, 최대 {v.max():.4f}, "
          f"표준편차 {v.std(ddof=1):.4f}")
    print(f"29.70세 근처(±0.5세) 인원: 원자료 "
          f"{((df['Age'] - 29.6991).abs() < 0.5).sum()}명 -> 대치 후 "
          f"{((fill['Age'] - 29.6991).abs() < 0.5).sum()}명")
    ```

    ```text

    대치된 177명의 나이: 최소 29.6991, 최대 29.6991, 표준편차 0.0000
    29.70세 근처(±0.5세) 인원: 원자료 25명 -> 대치 후 202명
    ```

    **29.7세 근처($\pm0.5$세)에 25명이던 것이 202명이 되었다.** 없던 봉우리가 한 칸에 솟는다. **자료에 없던 구조를 만들어 낸 것**이다.

    **권고 넷.**

    1. **결측률과 결측 패턴을 항상 보고**한다.
    2. **결측 여부로 나누어 다른 변수를 비교**한다. 여기서 한 것이 그 점검이다.
    3. **평균 대치는 쓰지 않는다.** 산포를 줄이고 가짜 봉우리를 만든다.
    4. **다중대치(MI)**가 표준 해법이다. 불확실성까지 반영한다.

    **이 절의 주된 분석에는 영향이 없다.** `Survived`와 `Sex`에는 결측이 없으므로 1~5절은 891명 전수를 썼다. 문제는 나이를 끌어들인 6절부터 생긴다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
객실 등급(`Pclass`)을 함께 고려하면 성별의 효과가 어떻게 보이는가? **등급별로 나누어** 분석하고, 전체 분석과 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    print("전체 (등급 무시)")
    all_tab = pd.crosstab(df["Sex"], df["Survived"])
    f = all_tab.loc["female", 1] / all_tab.loc["female"].sum()
    m = all_tab.loc["male", 1] / all_tab.loc["male"].sum()
    print(f"  여성 {f:.4f}  남성 {m:.4f}  차이 {f - m:+.4f}  "
          f"OR {(all_tab.loc['female', 1] * all_tab.loc['male', 0]) / (all_tab.loc['female', 0] * all_tab.loc['male', 1]):.4f}")

    print("\n등급별")
    print(f"{'등급':>5s}{'n':>6s}{'여성 생존율':>12s}{'남성 생존율':>12s}"
          f"{'차이':>9s}{'오즈비':>10s}")
    for pc in [1, 2, 3]:
        sub = df[df["Pclass"] == pc]
        t = pd.crosstab(sub["Sex"], sub["Survived"])
        fr = t.loc["female", 1] / t.loc["female"].sum()
        mr = t.loc["male", 1] / t.loc["male"].sum()
        orr = (t.loc["female", 1] * t.loc["male", 0]) / \
              (t.loc["female", 0] * t.loc["male", 1])
        print(f"{pc:>5d}{len(sub):>6d}{fr:>12.4f}{mr:>12.4f}"
              f"{fr - mr:>+9.4f}{orr:>10.4f}")

    print("\n등급 자체의 효과 (성별 무시)")
    print(f"{'등급':>5s}{'생존율':>10s}{'여성 비율':>11s}")
    for pc in [1, 2, 3]:
        sub = df[df["Pclass"] == pc]
        print(f"{pc:>5d}{sub['Survived'].mean():>10.4f}"
              f"{(sub['Sex'] == 'female').mean():>11.4f}")
    ```

    ```text
    전체 (등급 무시)
      여성 0.7420  남성 0.1889  차이 +0.5531  OR 12.3507

    등급별
       등급     n      여성 생존율      남성 생존율       차이       오즈비
        1   216      0.9681      0.3689  +0.5992   51.9037
        2   184      0.9211      0.1574  +0.7636   62.4510
        3   491      0.5000      0.1354  +0.3646    6.3830

    등급 자체의 효과 (성별 무시)
       등급       생존율      여성 비율
        1    0.6296     0.4352
        2    0.4728     0.4130
        3    0.2424     0.2933
    ```

    **성별의 효과가 세 등급 모두에서 살아남는다.** 방향이 뒤집히지 않으므로 **심슨의 역설은 일어나지 않았다.**

    | 등급 | 여성 | 남성 | 차이 | 오즈비 |
    |---|---|---|---|---|
    | 1 | 0.9681 | 0.3689 | $+0.599$ | **51.90** |
    | 2 | 0.9211 | 0.1574 | $+0.764$ | **62.45** |
    | 3 | 0.5000 | 0.1354 | $+0.365$ | 6.38 |
    | **전체** | 0.7420 | 0.1889 | $+0.553$ | **12.35** |

    **1·2등실의 층별 오즈비가 전체 오즈비보다 훨씬 크다.** 51.90, 62.45인데 전체는 12.35다. 이것이 **비붕괴성(non-collapsibility)**이다.

    **왜 그런가.** 등급이 **성별과 생존 양쪽에 연결**되어 있다.

    ```text
    3등실:  생존율 0.2424 (가장 낮음),  여성 비율 0.2933 (가장 낮음)
    1등실:  생존율 0.6296 (가장 높음),  여성 비율 0.4352 (가장 높음)
    ```

    **등급이 교란변수 노릇을 한다.** 전체 오즈비 12.35는 성별의 효과와 등급의 효과가 섞인 값이다.

    **차이 척도로 보면 이야기가 조금 다르다.** 1·2등실에서는 차이가 0.60, 0.76으로 큰데 3등실은 0.36으로 작다. **효과가 등급에 따라 달라지는 것**이며, 이것은 교란이 아니라 **상호작용**이다.

    ```python
    # 3등실에서 차이가 작은 이유: 여성 생존율 자체가 크게 낮다
    print(f"\n등급별 여성 생존율: "
          f"1등 {df[(df.Pclass == 1) & (df.Sex == 'female')]['Survived'].mean():.4f}, "
          f"2등 {df[(df.Pclass == 2) & (df.Sex == 'female')]['Survived'].mean():.4f}, "
          f"3등 {df[(df.Pclass == 3) & (df.Sex == 'female')]['Survived'].mean():.4f}")
    print(f"등급별 남성 생존율: "
          f"1등 {df[(df.Pclass == 1) & (df.Sex == 'male')]['Survived'].mean():.4f}, "
          f"2등 {df[(df.Pclass == 2) & (df.Sex == 'male')]['Survived'].mean():.4f}, "
          f"3등 {df[(df.Pclass == 3) & (df.Sex == 'male')]['Survived'].mean():.4f}")
    ```

    ```text

    등급별 여성 생존율: 1등 0.9681, 2등 0.9211, 3등 0.5000
    등급별 남성 생존율: 1등 0.3689, 2등 0.1574, 3등 0.1354
    ```

    **1·2등실 여성은 거의 전원(97%, 92%) 살아남았다.** 3등실 여성은 절반이다. **등급이 여성에게 훨씬 크게 작용했다** — 남성은 세 등급에서 0.37, 0.16, 0.14로 덜 벌어진다.

    **결론 셋.**

    1. **성별의 효과는 등급을 통제해도 남는다.** 층별로 모두 같은 방향이다.
    2. **전체 오즈비는 층별 오즈비의 "평균"이 아니다.** 12.35가 6.38~62.50 범위 밖은 아니지만 단순 평균과도 다르다.
    3. **효과의 크기가 층마다 다르다.** 하나의 수로 요약하면 이 정보를 잃는다.

    **이런 층별 분석에서 방향까지 뒤집히는 경우**가 심슨의 역설이며, [생태학적 상관](../../ch12/ecological_correlation/simpsons_paradox.md) 절에서 다룬다. $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
생존율 차이 55.31%포인트에 **신뢰구간**을 붙여라. 오즈비의 구간도 구하라.

</div>

??? success "풀이"
    **두 비율의 차이.** 표준오차는

    $$
    \operatorname{SE}=\sqrt{\frac{\hat p_1(1-\hat p_1)}{n_1}+\frac{\hat p_2(1-\hat p_2)}{n_2}}
    $$

    **오즈비는 로그 척도에서** 구간을 만든 뒤 되돌린다. 오즈비의 표본분포가 오른쪽으로 크게 치우쳐 있기 때문이다.

    $$
    \operatorname{SE}(\log\text{OR})=\sqrt{\frac1a+\frac1b+\frac1c+\frac1d}
    $$

    ```python
    import numpy as np
    from scipy.stats import norm

    tab = pd.crosstab(df["Sex"], df["Survived"])
    a, b = tab.loc["female", 1], tab.loc["female", 0]   # 여성 생존/사망
    c, d = tab.loc["male", 1],   tab.loc["male", 0]     # 남성 생존/사망
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    z = norm.ppf(0.975)

    # (1) 비율 차이
    se_d = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    print(f"여성 생존율 {p1:.4f} (n={n1}),  남성 생존율 {p2:.4f} (n={n2})")
    print(f"차이 {diff:.4f},  SE {se_d:.4f}")
    print(f"95% CI [{diff - z * se_d:.4f}, {diff + z * se_d:.4f}]")

    # (2) 위험비 -- 로그 척도에서
    se_lr = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    lr = np.log(p1 / p2)
    print(f"\n위험비 {p1 / p2:.4f}")
    print(f"95% CI [{np.exp(lr - z * se_lr):.4f}, {np.exp(lr + z * se_lr):.4f}]")

    # (3) 오즈비 -- 로그 척도에서
    se_lo = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    lo = np.log((a * d) / (b * c))
    print(f"\n오즈비 {np.exp(lo):.4f},  log(OR) {lo:.4f},  SE {se_lo:.4f}")
    print(f"95% CI [{np.exp(lo - z * se_lo):.4f}, {np.exp(lo + z * se_lo):.4f}]")

    # 로그를 거치지 않으면 어떻게 되는지 비교해 본다.
    se_naive = np.exp(lo) * se_lo      # 델타법 근사
    print(f"\n(비교) 원 척도에서 대칭 구간을 만들면"
          f" [{np.exp(lo) - z * se_naive:.4f}, {np.exp(lo) + z * se_naive:.4f}]")
    print("   -> 폭이 로그 구간과 다르고, 하한이 음수가 될 수도 있다")
    ```

    ```text
    여성 생존율 0.7420 (n=314),  남성 생존율 0.1889 (n=577)
    차이 0.5531,  SE 0.0296
    95% CI [0.4951, 0.6111]

    위험비 3.9280
    95% CI [3.2770, 4.7084]

    오즈비 12.3507,  log(OR) 2.5137,  SE 0.1672
    95% CI [8.9000, 17.1393]

    (비교) 원 척도에서 대칭 구간을 만들면 [8.3038, 16.3975]
       -> 폭이 로그 구간과 다르고, 하한이 음수가 될 수도 있다
    ```

    **세 구간 모두 "차이 없음"을 한참 벗어난다.**

    | 지표 | 추정값 | 95% 신뢰구간 | 귀무값 |
    |---|---|---|---|
    | 비율 차이 | 0.5531 | $[0.4951,\ 0.6111]$ | 0 |
    | 위험비 | 3.9280 | $[3.2770,\ 4.7084]$ | 1 |
    | 오즈비 | 12.3507 | $[8.9000,\ 17.1393]$ | 1 |

    **오즈비의 구간이 12.35를 가운데 두고 대칭이 아니다.** 아래로 3.45, 위로 4.79다. **로그 척도에서 대칭인 구간을 지수로 되돌렸기 때문**이며, 이것이 올바른 방식이다.

    $$
    \log\text{OR}=2.5137\pm1.96\times0.1672=[2.1860,\ 2.8414]
    $$

    $$
    \exp[2.1860,\ 2.8414]=[8.9000,\ 17.1393]
    $$

    **원 척도에서 대칭 구간을 만들면 $[8.30,\ 16.40]$으로 달라진다.** 오즈비가 더 작고 표본이 더 작을 때는 **하한이 음수**가 되어 아예 말이 안 되는 구간이 나온다.

    **구간의 폭이 정보량을 말해 준다.** 비율 차이의 구간 폭이 0.116(11.6%포인트)이다. $n=891$로도 이 정도이므로, **차이가 55%포인트나 되지 않았다면** 구간이 0을 포함했을 수도 있다.

    **보고 형식.**

    ```text
    타이타닉 승객 891명의 성별과 생존

      여성 74.20% (233/314),  남성 18.89% (109/577)
      차이   55.31%포인트,  95% CI [49.51, 61.11]
      오즈비 12.35,          95% CI [8.90, 17.14]
      chi2(1) = 263.05, p < 1e-58

    이는 관찰자료이며, 객실 등급이 성별·생존 양쪽과
    연관되어 있다(연습문제 7). 인과적 해석에는 주의가 필요하다.
    ```

    **마지막 문단이 빠지면 좋은 보고가 아니다.** $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
5절의 네 그림 중 **(2) 생존율 막대그림만 보면 위험하다**고 했다. 그 위험을 구체적인 예로 보여라.

</div>

??? success "풀이"
    **비율만 그리면 $n$이 사라진다.** 표본이 3명이든 300명이든 똑같은 막대가 나온다.

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # 같은 생존율 74% 대 19% 를 주지만 표본 크기만 다른 세 가지 자료
    # (이름, 여성 생존, 여성 전체, 남성 생존, 남성 전체)
    cases = [("n=8",     3,   4,   1,   4),
             ("n=88",   30,  40,   9,  48),
             ("n=891", 233, 314, 109, 577)]

    z = norm.ppf(0.975)
    print(f"{'자료':>8s}{'여성 생존율':>12s}{'남성 생존율':>12s}"
          f"{'차이':>9s}{'차이의 95% CI':>24s}")
    for lab, a, n1, c, n2 in cases:
        p1, p2 = a / n1, c / n2
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        d = p1 - p2
        print(f"{lab:>8s}{p1:>12.4f}{p2:>12.4f}{d:>+9.4f}"
              f"{f'[{d - z * se:+.4f}, {d + z * se:+.4f}]':>24s}")
    ```

    ```text
          자료      여성 생존율      남성 생존율       차이              차이의 95% CI
         n=8      0.7500      0.2500  +0.5000      [-0.1001, +1.1001]
        n=88      0.7500      0.1875  +0.5625      [+0.3887, +0.7363]
       n=891      0.7420      0.1889  +0.5531      [+0.4951, +0.6111]
    ```

    **세 자료의 막대 높이가 사실상 같다.** 0.75/0.25, 0.75/0.19, 0.74/0.19다. **막대그림만 보면 구별할 수 없다.**

    **그러나 구간의 폭이 완전히 다르다.**

    | 자료 | 차이 | 95% CI | 폭 | 0을 포함? |
    |---|---|---|---|---|
    | $n=8$ | $+0.500$ | $[-0.100,\ +1.100]$ | **2.200** | **그렇다** |
    | $n=88$ | $+0.563$ | $[+0.389,\ +0.736]$ | 0.348 | 아니다 |
    | $n=891$ | $+0.553$ | $[+0.495,\ +0.611]$ | **0.116** | 아니다 |

    **$n=8$에서는 구간이 0을 포함한다.** 같은 높이의 막대인데 **"차이가 없을 수도 있다"**는 결론이 나온다. 구간이 $[-0.10,\ +1.10]$으로, 상한이 1을 넘어 **비율 차이로는 불가능한 값까지 뻗는다.** 정규근사가 이렇게 작은 표본에서는 통하지 않는다는 신호이기도 하다.

    **$n=891$의 구간이 $n=8$보다 19배 좁다.** 표준오차가 $\sqrt{n}$에 반비례하므로 $\sqrt{891/8}\approx10.6$배가 기본이고, 여기에 두 자료의 비율 자체가 달라 차이가 더 벌어졌다.

    **처방 넷.**

    1. **막대 위에 $n$을 함께 적는다.** `"74.2% (233/314)"` 형식이 가장 좋다.
    2. **오차막대를 붙인다.** 신뢰구간을 시각적으로 전달한다.
    3. **도수 그림을 나란히 놓는다.** 5절의 (1)이나 (3)이 그 역할을 한다.
    4. **표본이 아주 작으면 막대그림을 쓰지 않는다.** 점 몇 개를 그대로 보이는 편이 정직하다.

    **가장 나쁜 사례가 실무에서 흔하다.**

    ```text
    "A안 전환율 12%, B안 전환율 8% -- A안 채택"

    → A안 3/25, B안 2/25 였다면?
      차이 0.04, 95% CI [-0.14, +0.22]
      아무것도 말할 수 없는 자료다
    ```

    **비율만 보이고 분모를 숨기는 그림은 의심해야 한다.** $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
두 범주형 변수의 관계를 탐색하는 **절차와 보고 지침**을 정리하라.

</div>

??? success "풀이"
    **절차 일곱 단계.**

    ```text
    1. 결측을 센다        ─ 변수별 결측률, 결측 여부에 따른 차이
    2. 분할표를 만든다     ─ 도수 + 행 기준 비율
    3. 효과크기를 구한다   ─ 비율 차이, 위험비, 오즈비
    4. 신뢰구간을 붙인다   ─ 오즈비는 반드시 로그 척도에서
    5. 검정한다           ─ 카이제곱 (기대도수 5 미만이면 피셔)
    6. 그린다             ─ 비율 그림 + 도수 그림을 함께
    7. 층화해 본다        ─ 교란이 의심되는 변수로 나누어
    ```

    **이 사례의 핵심 수치 여섯.**

    | 항목 | 값 |
    |---|---|
    | 생존율 | **74.20% 대 18.89%** |
    | 차이 | 55.31%포인트, CI $[49.51,\ 61.11]$ |
    | 오즈비 | 12.35, CI $[8.90,\ 17.14]$ |
    | $\varphi$ | $-0.5434$($\varphi_{\max}=0.9347$) |
    | $\chi^2(1)$ | 263.05, $p=3.7\times10^{-59}$ |
    | 층화 후 | 세 등급 모두 같은 방향(오즈비 6.4~62.4) |

    **요약 지표의 선택.**

    | 목적 | 지표 |
    |---|---|
    | **일반 독자에게 전달** | **두 비율을 그대로**(74% 대 19%) |
    | 절대적 영향의 크기 | 비율 차이 + CI |
    | **연구 간 비교·회귀** | **오즈비** + CI |
    | 상관행렬에 넣기 | $\varphi$(주변 비율 병기) |
    | 우연 여부 판정 | $\chi^2$ 검정 |

    **흔한 실수 여섯.**

    | 실수 | 사실 |
    |---|---|
    | 상관계수만 보고 | 부호가 **부호화에 의존**, 해석 불가 |
    | $p$값을 효과크기로 | 표본 크기에 좌우된다 |
    | $P(A\mid B)$와 $P(B\mid A)$ 혼동 | 74.20% vs 68.13% |
    | 오즈비를 "몇 배 더 산다"로 | 그것은 **위험비**(3.93) |
    | 비율 막대만 보이기 | $n$이 사라진다 |
    | 층화 없이 인과 주장 | 등급이 교란한다 |

    **네 번째가 언론 보도에서 가장 흔하다.** 오즈비 12.35를 "12배 더 살아남았다"고 쓰면 틀렸다. **생존율의 비는 3.93배**다.

    **보고 형식.**

    ```text
    타이타닉 승객 891명 (결측 없음: Survived, Sex)

      여성 74.20% (233/314) 생존
      남성 18.89% (109/577) 생존

      차이   55.31%포인트  [49.51, 61.11]
      위험비  3.93         [3.28,  4.71]
      오즈비 12.35         [8.90, 17.14]
      chi2(1) = 263.05, p < 1e-58

      객실 등급으로 층화해도 방향은 세 등급 모두 같았으나
      효과 크기는 달랐다(오즈비 51.9 / 62.4 / 6.4).

      관찰자료이므로 인과적 해석은 하지 않는다.
    ```

    **표본 크기, 효과크기, 구간, 층화, 인과에 대한 유보** — 다섯 가지가 모두 들어간 것이 좋은 보고다.

    **한 문장.** 두 범주형 변수의 관계는 **분할표 하나에 전부 들어 있고**, 상관계수·검정·그림은 그 표를 각각 다른 방식으로 줄인 것이므로, **무엇을 줄였는지 알고 골라야 한다.** $\square$
