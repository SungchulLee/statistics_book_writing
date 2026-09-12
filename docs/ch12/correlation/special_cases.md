# 점이연 상관과 파이 계수

Pearson 상관계수는 두 연속형 변수에 대해 정의되지만, 실무의 많은 상황에는 이진(이분) 변수가 등장한다. 한쪽 또는 양쪽이 이진일 때에도 Pearson 공식이 그대로 적용되며, 고유한 이름과 해석을 갖는 특수한 계수가 나온다. **점이연 계수**는 이진 변수 하나와 연속형 변수 하나의 경우를, **파이 계수**는 두 이진 변수의 경우를 다룬다.

---

## 점이연 상관

### 동기

이진 집단 변수(예: 처치 대 대조)와 연속형 결과(예: 시험 점수) 사이의 연관을 재고 싶다고 하자. 점이연 상관은 두 집단 사이에서 연속형 변수가 얼마나 다른지를 익숙한 $[-1, 1]$ 척도로 수치화한다.

<div class="defn" markdown>

**정의 1.** [점이연 상관계수]

이진 변수를 $X \in \{0, 1\}$로 부호화하고 집단 0에 $n_0$개, 집단 1에 $n_1$개의 관측값이 있으며 $n = n_0 + n_1$이라 하자. $Y$를 연속형 변수라 하면 **점이연 상관**은

$$
r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{S_Y} \sqrt{\frac{n_0 \, n_1}{n^2}}
$$

이다. 여기서 $\bar{Y}_0$과 $\bar{Y}_1$은 $Y$의 집단 평균이고, $S_Y$는 $Y$ 전체의 표준편차이다. Pearson 상관과 정확히 일치하려면 $S_Y$를 $n$으로 나눈 형태 $S_Y = \sqrt{\frac{1}{n}\sum (Y_i - \bar{Y})^2}$로 잡아야 한다.

$X$가 0과 1만 취할 때 이는 $X$와 $Y$ 사이의 Pearson 상관과 대수적으로 동일하다.

</div>

### 이표본 t-검정과의 연결

점이연 상관은 이표본 t-검정통계량과 직접 관련된다. $t$가 자유도 $n - 2$의 등분산 이표본 t-통계량이면

$$
r_{pb} = \frac{t}{\sqrt{t^2 + (n - 2)}}
$$

이다. 따라서 $H_0\!: r_{pb} = 0$을 검정하는 것은 이표본 t-검정으로 $H_0\!: \mu_0 = \mu_1$을 검정하는 것과 동치이다.

### 해석

- $r_{pb} > 0$: 집단 1의 $Y$ 값이 더 큰 경향이 있다.
- $r_{pb} < 0$: 집단 0의 $Y$ 값이 더 큰 경향이 있다.
- $r_{pb} = 0$: 두 집단의 평균에 차이가 없다.
- $r_{pb}^2$: $Y$의 분산 중 집단 소속으로 설명되는 비율.

---

## 파이 계수

### 동기

두 변수가 모두 이진일 때(예: 성별과 합격/불합격) $2 \times 2$ 분할표에 대한 연관 측도가 필요하다. **파이 계수**는 두 이진 변수 사이의 Pearson 상관이며 연관의 강도를 재는 자연스러운 측도이다.

<div class="defn" markdown>

**정의 2.** [파이 계수]

$2 \times 2$ 표를 생각하자:

|  | $Y = 1$ | $Y = 0$ | 합계 |
|:---:|:---:|:---:|:---:|
| $X = 1$ | $a$ | $b$ | $a + b$ |
| $X = 0$ | $c$ | $d$ | $c + d$ |
| 합계 | $a + c$ | $b + d$ | $n$ |

**파이 계수**는

$$
\phi = \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}
$$

이며, 이는 $X$와 $Y$를 0/1로 부호화한 자료에 대해 계산한 Pearson 상관과 대수적으로 동일하다.

</div>

### 카이제곱과의 연결

파이 계수는 $2 \times 2$ 표의 Pearson 카이제곱 통계량과 관련된다:

$$
\chi^2 = n \, \phi^2
$$

따라서

$$
\phi = \sqrt{\frac{\chi^2}{n}}
$$

이며 부호는 연관의 방향으로 정한다($ad > bc$이면 $\phi > 0$).

$H_0\!: \phi = 0$을 검정하는 것은 $2 \times 2$ 표의 카이제곱 독립성 검정과 동치이다.

### 성질

1. **범위.** $-1 \le \phi \le 1$이지만, $\phi$가 $\pm 1$에 도달할 수 있는 것은 $X$와 $Y$의 주변분포가 같을 때뿐이다(즉 $a + b = a + c$이고 $c + d = b + d$일 때).

2. **도달 가능한 최댓값.** 주변분포가 다르면 $|\phi|$가 $1$보다 작은 값으로 제한된다. 최댓값은 주변 비율에 달려 있다.

3. **대칭성.** $\phi_{XY} = \phi_{YX}$.

---

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 점이연 상관. 어떤 연구자가 학생 10명을 스터디 그룹($X = 1$) 또는 무개입($X = 0$)에 배정하고 시험 점수를 기록했다:

| 집단 ($X$) | 점수 ($Y$) |
|:---:|:---|
| 0 | 65, 70, 68, 72, 66 |
| 1 | 78, 82, 85, 80, 76 |

계산하면 $\bar{Y}_0 = 68.2$, $\bar{Y}_1 = 80.2$, $S_Y = 6.65$($n$으로 나눈 표준편차), $n_0 = n_1 = 5$, $n = 10$이다.

$$
r_{pb} = \frac{80.2 - 68.2}{6.65} \sqrt{\frac{5 \times 5}{100}} = \frac{12.0}{6.65} \times 0.5 \approx 0.903
$$

강한 양의 $r_{pb}$는 스터디 그룹의 점수가 상당히 높았음을 나타낸다.

</div>

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 파이 계수. 어떤 조사가 200명에게 운동 습관($X$: 규칙적으로 운동함)과 수면의 질($Y$: 잘 잔다고 보고함)을 물었다:

|  | 잘 잠 ($Y=1$) | 못 잠 ($Y=0$) | 합계 |
|:---:|:---:|:---:|:---:|
| 운동함 ($X=1$) | 60 | 30 | 90 |
| 운동 안 함 ($X=0$) | 40 | 70 | 110 |
| 합계 | 100 | 100 | 200 |

$$
\phi = \frac{(60)(70) - (30)(40)}{\sqrt{(90)(110)(100)(100)}} = \frac{4200 - 1200}{\sqrt{99{,}000{,}000}} = \frac{3000}{9949.87} \approx 0.301
$$

규칙적인 운동과 좋은 수면의 질 사이에 중간 정도의 양의 연관이 있다.

</div>

## 더 큰 표를 위한 관련 측도

파이 계수는 $2 \times 2$ 표에 특화되어 있다. 더 큰 분할표에는 다음과 같은 관련 측도가 있다:

- **Cramér의 V**: $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))}$으로 파이를 $r \times c$ 표로 일반화한다.
- **분할계수**: $C = \sqrt{\chi^2 / (\chi^2 + n)}$이며 위로 1보다 작은 값에서 제한된다.

이들은 [카이제곱 검정](../../ch10/test/independence.md) 장에서 다룬다.

---

## Python으로 계산하기

```python
import numpy as np
from scipy import stats

# Point-biserial example
group = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
scores = np.array([65, 70, 68, 72, 66, 78, 82, 85, 80, 76])

r_pb, p_val = stats.pointbiserialr(group, scores)
print(f"Point-biserial r = {r_pb:.4f}, p-value = {p_val:.4f}")

# This equals Pearson r on the same data
r_pearson, _ = stats.pearsonr(group, scores)
print(f"Pearson r        = {r_pearson:.4f}")

# Phi coefficient via chi-square
table = np.array([[60, 30], [40, 70]])
chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
phi = np.sqrt(chi2 / table.sum())
print(f"Phi coefficient  = {phi:.4f}")
```

출력:

```
Point-biserial r = 0.9029, p-value = 0.0003
Pearson r        = 0.9029
Phi coefficient  = 0.3015
```

점이연 상관과 Pearson 상관이 **정확히 같다**. 점이연 상관은 별개의 공식이 아니라, 한 변수가 0/1일 때의 Pearson 상관에 붙인 이름일 뿐이다.

파이 계수도 마찬가지로 두 이진 변수에 대한 Pearson 상관과 같다. 값이 0.30으로 작은 것은 다른 자료($2 \times 2$ 표)를 쓰기 때문이지 계수의 성질 때문이 아니다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
성별(0 = 여성, 1 = 남성)과 시험 점수 사이의 점이연 상관을 계산하라. 자료는 여성 $\{78, 82, 85, 88\}$, 남성 $\{90, 92, 88, 95\}$이다.

</div>

??? success "풀이"
    $X$ = 성별(0/1), $Y$ = 점수라 하자. 여성 평균 $\bar{Y}_0 = 83.25$, 남성 평균 $\bar{Y}_1 = 91.25$, 전체 평균 $\bar{Y} = 87.25$, $n_0 = n_1 = 4$, $n = 8$이다.

    점이연 상관은

    $$
    r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{S_Y}\sqrt{\frac{n_0 n_1}{n^2}}
    $$

    이며 $S_Y$는 $n$으로 나눈 표준편차이다.

    $\sum(Y_i - 87.25)^2 = 85.5625 + 27.5625 + 5.0625 + 0.5625 + 7.5625 + 22.5625 + 0.5625 + 60.0625 = 209.5$

    $S_Y = \sqrt{209.5/8} = \sqrt{26.1875} = 5.117$

    $$
    r_{pb} = \frac{91.25 - 83.25}{5.117}\sqrt{\frac{16}{64}} = \frac{8}{5.117} \times 0.5 \approx 0.782
    $$

    `scipy.stats.pointbiserialr`가 주는 값과 일치한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
점이연 상관이 이진 변수와 연속형 변수 사이의 Pearson 상관과 동등함을 보여라.

</div>

??? success "풀이"
    $X \in \{0, 1\}$이고 $P(X = 1) = p = n_1/n$이라 하자. Pearson 상관은

    $$
    r = \frac{\sum(X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum(X_i - \bar{X})^2 \sum(Y_i - \bar{Y})^2}}
    $$

    이다. $\bar{X} = p$이므로 집단 1에서는 $(X_i - p) = 1 - p$, 집단 0에서는 $-p$이다. 분자는

    $$
    \sum(X_i - p)(Y_i - \bar{Y}) = (1-p)\sum_{i \in \text{group 1}}(Y_i - \bar{Y}) - p\sum_{i \in \text{group 0}}(Y_i - \bar{Y})
    $$

    가 된다. $\sum_{i \in \text{group 1}}(Y_i - \bar{Y}) = n_1(\bar{Y}_1 - \bar{Y})$이고 집단 0도 마찬가지이므로 이는 $n \cdot p(1-p)(\bar{Y}_1 - \bar{Y}_0)$으로 단순해진다. 분모에는 $\sqrt{n \cdot p(1-p) \cdot \sum(Y_i - \bar{Y})^2}$이 들어가고, 그 비가 점이연 공식을 준다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
다음 $2 \times 2$ 표에 대해 파이 계수를 계산하라:

|  | 합격 | 불합격 |
|---|---|---|
| 공부함 | 40 | 10 |
| 공부 안 함 | 20 | 30 |

</div>

??? success "풀이"
    칸이 $a, b, c, d$인 $2 \times 2$ 표에서 $a = 40, b = 10, c = 20, d = 30$이다.

    $$
    \phi = \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}
    $$

    $$
    = \frac{40 \times 30 - 10 \times 20}{\sqrt{50 \times 50 \times 60 \times 40}} = \frac{1200 - 200}{\sqrt{6{,}000{,}000}} = \frac{1000}{2449.5} \approx 0.408
    $$

    파이 계수 0.41은 공부와 합격 사이에 중간 정도의 양의 연관이 있음을 나타낸다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
$2 \times 2$ 표에서 파이 계수와 카이제곱 검정통계량의 관계를 설명하라. 하나로부터 다른 하나를 어떻게 얻는가?

</div>

??? success "풀이"
    파이 계수와 카이제곱 통계량은 직접 관련된다:

    $$
    \chi^2 = n\phi^2 \quad \Leftrightarrow \quad \phi = \sqrt{\chi^2/n}
    $$

    여기서 $n$은 전체 표본크기이다. 연습문제 3에서 $\phi = 0.408$, $n = 100$이므로

    $$
    \chi^2 = 100 \times 0.408^2 = 100 \times 0.1665 = 16.65
    $$

    이다. 즉 파이 계수는 카이제곱 통계량을 정규화한 값이다. $\chi^2$은 표본크기에 의존하지만(같은 연관이라도 $n$이 크면 $\chi^2$이 커진다) $\phi$는 척도에 자유롭고 $-1$과 $1$ 사이에 있다. 자유도 1의 카이제곱 검정과 $\phi = 0$의 검정은 동치이다.

---

## 정리하며

점이연 계수와 파이 계수는 이진 자료에 대한 Pearson 상관의 특수한 경우이다. 점이연 계수는 이진 집단 변수와 연속형 결과 사이의 연관을 재며 그 검정은 이표본 t-검정과 동치이다. 파이 계수는 $2 \times 2$ 표에서 두 이진 변수 사이의 연관을 재며 그 검정은 카이제곱 독립성 검정과 동치이다. 이런 연결을 알면 서로 달라 보이던 여러 통계 절차가 상관이라는 하나의 틀 아래 통합된다.
