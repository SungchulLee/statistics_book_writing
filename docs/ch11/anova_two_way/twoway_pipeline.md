# 이원배치 분산분석 파이프라인

## 개요

이원배치 분산분석은 일원배치 설계를 확장하여 두 요인의 효과와 그 교호작용이 연속형 반응에 미치는 영향을 동시에 살핀다. 이 페이지는 ToothGrowth 자료로 완전한 파이프라인을 따라간다. 이원배치 분산분석(제II형) 적합, 각 주효과와 교호작용에 대한 Tukey HSD 사후검정, 교호작용 그림 작성. 두 요인은 보충제 종류(OJ 대 VC)와 용량 수준(0.5, 1.0, 2.0)이다.

## 이원배치 분산분석 모형

수준이 $a$개인 요인 $A$와 수준이 $b$개인 요인 $B$에 대한 칸 평균 모형은

$$
y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \varepsilon_{ijk}
$$

이며 $\alpha_i$는 요인 $A$의 주효과, $\beta_j$는 요인 $B$의 주효과, $(\alpha\beta)_{ij}$는 교호작용 효과, $\varepsilon_{ijk} \sim N(0, \sigma^2)$이다.

제II형 분산분석표는 세 가지 귀무가설을 검정한다:

| 원천 | $H_0$ | $df$ |
|---|---|---|
| 요인 $A$ | 모든 $\alpha_i = 0$ | $a - 1$ |
| 요인 $B$ | 모든 $\beta_j = 0$ | $b - 1$ |
| $A \times B$ | 모든 $(\alpha\beta)_{ij} = 0$ | $(a-1)(b-1)$ |
| 잔차 | | $N - ab$ |

## 1단계: 모형 적합

```python
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

url = ('https://raw.githubusercontent.com/vincentarelbundock/'
       'Rdatasets/master/csv/datasets/ToothGrowth.csv')
df = pd.read_csv(url, usecols=[1, 2, 3])

model = ols('len ~ C(supp) + C(dose) + C(supp):C(dose)', data=df).fit()
aov2 = anova_lm(model, typ=2)
print(aov2)
```

제II형 제곱합은 각 주효과를 다른 주효과로 조정하되 교호작용은 무시하고 검정한다. 설계가 균형이거나 거의 균형일 때 권장된다.

## 2단계: 주효과에 대한 Tukey HSD

사후검정은 한 요인의 어느 수준이 다른지 찾아낸다. 각 주효과에 대해 Tukey HSD를 따로 수행한다.

```python
from statsmodels.stats.multicomp import pairwise_tukeyhsd

print(pairwise_tukeyhsd(endog=df['len'], groups=df['dose'], alpha=0.05))
print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp'], alpha=0.05))
```

## 3단계: 교호작용에 대한 Tukey HSD

$a \times b$개의 칸 평균을 모두 비교하려면 결합 집단 변수를 만들어 교호작용 칸에 Tukey HSD를 수행한다.

```python
df['supp_dose'] = df['supp'].astype(str) + "_" + df['dose'].astype(str)
print(pairwise_tukeyhsd(endog=df['len'], groups=df['supp_dose'], alpha=0.05))
```

칸이 $a \times b = 2 \times 3 = 6$개이므로 쌍별 비교는 $\binom{6}{2} = 15$개이다. Tukey 절차는 이 15개 전체에 걸쳐 가족단위 오류율을 동시에 통제한다.

## 4단계: 교호작용 그림

교호작용 그림은 한 요인을 가로축에 두고 다른 요인의 각 수준을 별도의 선으로 그려 칸 평균을 보여준다. 선이 평행하지 않으면 교호작용을 시사한다.

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot

fig, ax = plt.subplots(figsize=(8, 4))
interaction_plot(df['dose'], df['supp'], df['len'], ax=ax,
                 markers=['o', 's'], linestyles=['--', '-.'])
ax.set_title("Interaction: dose x supp")
ax.set_xlabel("dose")
ax.set_ylabel("len")
plt.tight_layout()
plt.show()
```

## 해석

- **용량의 주효과:** 용량에 대한 분산분석 $p$-값이 작고 Tukey HSD가 유의한 쌍별 차이를 보이면 용량이 클수록 치아 성장이 크다는 뜻이다.
- **보충제의 주효과:** supp의 $p$-값이 유의하면 두 보충제 종류(OJ 대 VC)가 서로 다른 평균 치아 길이를 낳음을 나타낸다.
- **교호작용:** 교호작용이 유의하면 용량의 효과가 보충제 종류에 (또는 그 반대로) 의존한다는 뜻이다. ToothGrowth 자료에서는 용량 2.0에서 OJ와 VC의 결과가 비슷하지만 낮은 용량에서는 다르며, 교호작용 그림에서 선이 수렴하는 모습으로 나타난다.
- **제II형 대 제III형:** 교호작용이 있는 상태에서 주효과를 검정할 사전 이유가 없다면 제II형이 적절하다. 교호작용이 유의하고 설계가 불균형이면 (교호작용을 포함한 다른 모든 효과를 통제하고 각 효과를 검정하는) 제III형이 나을 수 있다.

## 연습문제

**연습문제 1.**
칸당 관측값이 $n = 10$개인 $2 \times 3$ 요인 설계에서 분산분석표의 각 원천별 자유도와 전체 자유도를 진술하라.

??? success "풀이"
    요인 $A$의 수준이 $a = 2$개, 요인 $B$의 수준이 $b = 3$개, 칸당 $n = 10$이므로 $N = 2 \times 3 \times 10 = 60$이다.

    | 원천 | $df$ |
    |---|---|
    | 요인 $A$ | $a - 1 = 1$ |
    | 요인 $B$ | $b - 1 = 2$ |
    | $A \times B$ | $(a-1)(b-1) = 2$ |
    | 잔차 | $N - ab = 60 - 6 = 54$ |
    | 전체 | $N - 1 = 59$ |

---

**연습문제 2.**
제I형, 제II형, 제III형 제곱합의 차이를 설명하라. 어떤 조건에서 세 유형이 동일한 결과를 주는가?

??? success "풀이"

    - **제I형(순차):** 각 효과를 그보다 앞서 들어간 효과들로만 조정하여 검정한다. 결과가 모형의 항 순서에 의존한다.
    - **제II형:** 각 주효과를 다른 주효과로 조정하되 교호작용으로는 조정하지 않고 검정한다. 교호작용은 두 주효과로 조정한 뒤 검정한다.
    - **제III형:** 각 효과를 교호작용을 포함한 다른 모든 효과로 조정하여 검정한다.

    설계가 **균형**(칸 크기가 같음)이고 모형이 완전히 지정되면 세 유형이 동일한 결과를 준다. 균형 설계에서는 제곱합이 직교하므로 항의 입력 순서가 문제되지 않고 다른 항으로 조정해도 달라지지 않는다. 불균형 설계에서는 세 유형이 상당히 다를 수 있다.

---

**연습문제 3.**
교호작용 그림에서 OJ와 VC의 선이 용량 2.0에서 수렴한다. 이는 교호작용 항에 대해 무엇을 함의하는가? OJ와 VC의 차이가 용량 0.5와 2.0에서 같은지 검정하는 대비를 써라.

??? success "풀이"
    수렴한다는 것은 용량이 커질수록 보충제의 효과가 줄어든다는 뜻이며, 이는 교호작용의 한 형태이다. OJ–VC 차이가 용량 0.5와 2.0에서 같은지 검정하는 대비는

    $$
    \psi = (\mu_{\text{OJ},0.5} - \mu_{\text{VC},0.5}) - (\mu_{\text{OJ},2.0} - \mu_{\text{VC},2.0})
    $$

    이다. $H_0: \psi = 0$ 아래에서 보충제의 효과가 두 용량에서 같다. 결과가 유의하면 OJ–VC 차이의 크기가 용량 수준에 따라 달라진다는 뜻이며, 이것이 바로 분산분석 모형의 교호작용 항이 담아내는 것이다.

---

**연습문제 4.**
균형 설계의 이원배치 분산분석에서 총제곱합이

$$
SST = SS_A + SS_B + SS_{AB} + SSE
$$

로 분해됨을 보여라. 이 분해에 필요한 독립성 가정을 진술하라.

??? success "풀이"
    항등식

    $$
    y_{ijk} - \bar{y}_{\cdot\cdot\cdot} = (\bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot\cdot\cdot}) + (\bar{y}_{\cdot j\cdot} - \bar{y}_{\cdot\cdot\cdot}) + (\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot}) + (y_{ijk} - \bar{y}_{ij\cdot})
    $$

    에서 시작한다. 제곱하여 모든 $i, j, k$에 대해 합하면 (균형 설계에서 성립하는) 직교성 덕분에 모든 교차항이 사라져

    $$
    \sum_{i,j,k} (y_{ijk} - \bar{y}_{\cdot\cdot\cdot})^2 = bn\sum_i (\bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot\cdot\cdot})^2 + an\sum_j (\bar{y}_{\cdot j\cdot} - \bar{y}_{\cdot\cdot\cdot})^2 + n\sum_{i,j}(\bar{y}_{ij\cdot} - \bar{y}_{i\cdot\cdot} - \bar{y}_{\cdot j\cdot} + \bar{y}_{\cdot\cdot\cdot})^2 + \sum_{i,j,k}(y_{ijk} - \bar{y}_{ij\cdot})^2
    $$

    가 된다. 즉 $SST = SS_A + SS_B + SS_{AB} + SSE$이다.

    이 분해에는 (1) 균형 설계(칸당 $n$이 같음)와 (2) 오차 $\varepsilon_{ijk}$가 독립이고 공통 분산 $\sigma^2$을 갖는다는 가정이 필요하다. 독립성은 $SSE / \sigma^2 \sim \chi^2_{N-ab}$이고 $SSE$가 $SS_A$, $SS_B$, $SS_{AB}$와 독립임을 보장하며, 이는 $F$-검정이 정확한 $F$-분포를 갖는 데 필요하다. $\square$

---

**연습문제 5.**
교호작용은 유의한데 한 주효과가 유의하지 않을 때, 여러 교과서가 그 주효과를 해석하지 말라고 경고한다. 구체적인 수치 예로 이유를 설명하라.

??? success "풀이"
    교호작용이 유의하다는 것은 한 요인의 효과가 다른 요인의 수준에 의존한다는 뜻이다. 이 상황에서 다른 요인의 수준에 걸쳐 평균을 낸 주효과는 어느 집단의 실제 경험도 대표하지 못할 수 있다.

    **예:** 칸 평균이 다음과 같은 $2 \times 2$ 설계를 생각하자:

    | | $B_1$ | $B_2$ |
    |---|---|---|
    | $A_1$ | 10 | 20 |
    | $A_2$ | 20 | 10 |

    $A$의 주변평균은 $\bar{y}_{1\cdot} = 15$, $\bar{y}_{2\cdot} = 15$이므로 $A$의 주효과는 0이다. 그러나 $A$에는 분명 큰 효과가 있다. $A_1$에서 $A_2$로 가면 $B_1$ 조건에서는 반응이 10만큼 커지고 $B_2$ 조건에서는 10만큼 작아진다. 이 반대 방향의 효과가 주변평균에서 상쇄되어 주효과 검정이 무의미해진다. 여기서 정보를 담고 있는 양은 교호작용이며, 올바른 해석은 $A$의 효과 방향이 $B$의 수준에 따라 뒤집힌다는 것이다.
