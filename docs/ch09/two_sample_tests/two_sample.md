# 이표본 검정

## 1. 이표본 z 검정

이표본 z-검정은 모분산을 알고 있을 때 독립인 두 표본의 평균이 유의하게 다른지 판단한다. 서로 다른 조건의 두 집단을 비교할 때 쓰며, 표본이 정규분포를 따르고 독립이라고 가정한다.

### A. 가설

- **귀무가설**: $H_0: \mu_1 = \mu_2$
- **대립가설**:
    - 양측: $H_a: \mu_1 \neq \mu_2$
    - 단측(큼): $H_a: \mu_1 > \mu_2$
    - 단측(작음): $H_a: \mu_1 < \mu_2$

### B. 검정통계량

$$ z = \frac{(\bar{x}_1 - \bar{x}_2) - (\mu_1 - \mu_2)}{\sqrt{\frac{\sigma_1^2}{n_1} + \frac{\sigma_2^2}{n_2}}} $$

$H_0$ 아래에서($\mu_1 - \mu_2 = 0$) 표본이 크면 표본표준편차를 써서:

$$ z = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$

### C. 판정 규칙

- **양측**: $|z| > z_{\alpha/2}$이면 $H_0$을 기각한다.
- **단측(큼)**: $z > z_{\alpha}$이면 $H_0$을 기각한다.
- **단측(작음)**: $z < -z_{\alpha}$이면 $H_0$을 기각한다.

### D. p-값

- 양측: $p\text{-값} = 2P(Z \geq |z|)$
- 단측(큼): $p\text{-값} = P(Z \geq z)$
- 단측(작음): $p\text{-값} = P(Z \leq z)$

### E. 예제

<div class="codebox" markdown>

#### 예제 1. 두 비율의 차이 — 간단한 예 { .eg }

```python
import numpy as np
from scipy import stats

n_f, n_s = 100, 100
x_bar_f, x_bar_s = 1.85, 1.65
s_f, s_s = 1.3, 1.2

# n이 각각 100이라 크므로 sigma 자리에 표본표준편차를 넣고 z를 쓴다.
statistic = (x_bar_f - x_bar_s) / np.sqrt(s_f**2/n_f + s_s**2/n_s)
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"statistic : {statistic:.4f}")
print(f"p value   : {p_value:.4f}")
```

출력:

```
statistic : 1.1305
p value   : 0.2583
```

출생아 수 평균이 1.85와 1.65로 0.2 차이지만 개인차(표준편차 1.3, 1.2)가 커서 기각하지 못한다.

</div>

---

## 2. 이표본 t 검정

이표본 t-검정(독립표본 t-검정)은 독립인 두 집단의 평균이 유의하게 다른지 판단한다. 모분산을 모르고 서로 같다고 가정할 때 쓴다.

### A. 가설

- **귀무가설**: $H_0: \mu_1 = \mu_2$
- **대립가설**:
    - 양측: $H_a: \mu_1 \neq \mu_2$
    - 단측(큼): $H_a: \mu_1 > \mu_2$
    - 단측(작음): $H_a: \mu_1 < \mu_2$

### B. 검정통계량 (합동분산)

$$ t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \cdot \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} $$

여기서 합동 표준편차는:

$$ s_p = \sqrt{\frac{(n_1 - 1) s_1^2 + (n_2 - 1) s_2^2}{n_1 + n_2 - 2}} $$

이 통계량은 자유도 $n_1 + n_2 - 2$인 t-분포를 따른다.

### C. 판정 규칙

- **양측**: $|t| > t_{\alpha/2, n_1+n_2-2}$이면 $H_0$을 기각한다.
- **단측(큼)**: $t > t_{\alpha, n_1+n_2-2}$이면 $H_0$을 기각한다.
- **단측(작음)**: $t < -t_{\alpha, n_1+n_2-2}$이면 $H_0$을 기각한다.

### D. 예제

<div class="codebox" markdown>

#### 예제 2. 급여의 성별 격차 { .eg }

시장조사자들이 남성 관리자와 여성 관리자의 평균 급여를 비교한다.

$$H_0 : \mu_{\text{men}} = \mu_{\text{women}} \quad\text{vs}\quad H_1: \mu_{\text{men}} > \mu_{\text{women}}$$

</div>

| | 밭 A | 밭 B |
|:---:|:---:|:---:|
| 평균 | 1.3 m | 1.6 m |
| 표준편차 | 0.5 m | 0.3 m |
| n | 22 | 24 |

$$H_0 : \mu_A = \mu_B \quad\text{vs}\quad H_1: \mu_A \neq \mu_B$$

<div class="codebox" markdown>

#### 예제 3. 서로 다른 두 밭의 토마토 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

X_1_bar, X_2_bar = 1.3, 1.6
s_1, s_2 = 0.5, 0.3
n_1, n_2 = 22, 24

# 표준편차가 0.5와 0.3으로 다르므로 합동하지 않고 Welch를 쓴다.
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)

# Welch-Satterthwaite 자유도.
# 분모에 n_i가 아니라 **n_i - 1**이 들어간다는 점에 주의하라.
# n으로 잘못 쓰면 자유도가 부풀어 기각하기 쉬워진다.
top = (s_1**2 / n_1 + s_2**2 / n_2)**2
bottom = (s_1**2 / n_1)**2 / (n_1 - 1) + (s_2**2 / n_2)**2 / (n_2 - 1)
df = top / bottom

p_value = 2 * stats.t(df).cdf(-abs(statistic))
print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0")
else:
    print("Fail to reject H_0")
```

출력:

```
df = 33.7874
statistic = -2.4403
p_value   = 0.0201
Reject H_0
```

자유도가 33.79로 정수가 아니다. Welch 자유도는 근사값이라 정수일 이유가 없다. 합동 검정이었다면 $n_1 + n_2 - 2 = 44$였을 것이고, 분산이 달라 정보량을 보수적으로 잡은 결과가 이 차이다.

</div>

| | France | Switzerland |
|:---:|:---:|:---:|
| 평균 | 1.85 | 1.65 |
| 표준편차 | 1.3 | 1.2 |
| n | 100 | 100 |

합동분산을 쓰면:

<div class="codebox" markdown>

#### 예제 4. 출생아 수 — 프랑스와 스위스 { .eg }

```python
X_1_bar, X_2_bar = 1.85, 1.65
s_1, s_2 = 1.3, 1.2
n_1, n_2 = 100, 100

# 합동분산은 두 표본분산을 자유도로 가중평균한 것이다.
# 여기서는 n이 같아 단순 평균과 같아진다.
s_p_square = ((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 + n_2 - 2)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
df = n_1 + n_2 - 2
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")
```

출력:

```
df = 198.0000
statistic = 1.1305
p_value   = 0.2596
```

같은 자료의 앞선 $z$-검정과 통계량이 1.1305로 정확히 같고 p-값만 0.2583에서 0.2596으로 바뀌었다. 자유도 198이면 $t$가 정규분포와 거의 구별되지 않기 때문이다.

</div>

<div class="codebox" markdown>

#### 예제 5. 두 품종의 배 (Bosc와 Anjou) { .eg }

| | Bosc | Anjou |
|:---:|:---:|:---:|
| 평균 | 120 | 116 |
| 표준편차 | 15 | 13 |
| n | 65 | 65 |

$\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$의 99% 신뢰구간은 $4 \pm 6.44$, 즉 $(-2.44, 10.44)$이다. 신뢰구간이 0을 포함하므로 $\alpha = 0.01$에서 $H_0$을 기각하지 못한다.

</div>

---

## 3. Welch의 t 검정

**Welch의 t-검정**은 분산이 다르고 표본크기도 다를 수 있는 상황을 감안한, 표준 이표본 t-검정의 로버스트한 변형이다.

### 공식

$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$$

자유도는 **Welch-Satterthwaite 식**으로 근사한다:

$$df = \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{\frac{\left( \frac{s_1^2}{n_1} \right)^2}{n_1 - 1} + \frac{\left( \frac{s_2^2}{n_2} \right)^2}{n_2 - 1}}$$

### 언제 쓰는가

- 두 집단의 분산이 눈에 띄게 다를 때.
- 두 집단의 표본크기가 크게 다를 때.
- 모분산을 모를 때.

<div class="codebox" markdown>

#### 예제 6. 합동 t-검정 구현 { .eg }

```python
import numpy as np
from scipy.stats import ttest_ind

team_a = [120, 118, 125, 130, 115, 122, 121, 119, 117, 123, 124, 126, 127, 118, 116]
team_b = [135, 132, 137, 140, 136, 130, 134, 138, 139, 133, 131, 142, 141,
           129, 128, 135, 137, 136, 134, 132]

# 표본크기가 15와 20으로 다르다. 이런 상황이 Welch를 쓸 이유다.
stat, p_value = ttest_ind(team_a, team_b, equal_var=False)
print(f"Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: The means are significantly different.")
else:
    print("Fail to reject H0.")
```

출력:

```
Test Statistic: -9.4407
P-value: 0.0000
Reject H0: The means are significantly different.
```

두 팀의 평균이 121.4와 135.0으로 13.6 차이인데 팀 안의 산포는 표준편차 4 남짓이라 $t$가 $-9.44$까지 간다. 집단 간 차이가 집단 안 산포보다 훨씬 크면 표본이 작아도 분명하게 갈린다.

</div>

### 표준 이표본 t-검정과의 비교

| 항목 | 표준 t-검정 | Welch t-검정 |
|---|---|---|
| 분산 가정 | 등분산 | 등분산 가정 없음 |
| 표본크기 | 비슷한 크기를 전제 | 크기가 달라도 된다 |
| 자유도 | 고정: $n_1 + n_2 - 2$ | Welch-Satterthwaite로 근사 |

---

## 4. 이표본 비율 검정

이표본 비율 검정은 이진 결과에 대해 독립인 두 집단의 비율에 유의한 차이가 있는지 판단한다.

### A. 가설

- **귀무가설**: $H_0: p_1 = p_2$
- **대립가설**:
    - 양측: $H_a: p_1 \neq p_2$
    - 단측(큼): $H_a: p_1 > p_2$
    - 단측(작음): $H_a: p_1 < p_2$

### B. 검정통계량

합동 비율:

$$\hat{p}_{\text{pool}} = \frac{x_1 + x_2}{n_1 + n_2}$$

검정통계량:

$$z = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}_{\text{pool}} (1 - \hat{p}_{\text{pool}}) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}$$

### C. 예제

| | A 지구 | B 지구 | 합계 |
|:---:|:---:|:---:|:---:|
| 예 | 58 | 52 | 110 |
| 아니오 | 42 | 48 | 90 |

$$H_0 : p_A = p_B \quad\text{vs}\quad H_1: p_A \neq p_B$$

<div class="codebox" markdown>

#### 예제 7. 새 법률에 대한 지지 { .eg }

```python
import numpy as np
from scipy import stats

positive_A, positive_B = 58, 52
n_A, n_B = 100, 100
p_hat_A, p_hat_B = positive_A / n_A, positive_B / n_B

# H0가 "두 비율이 같다"이므로 그 공통값을 전체를 합쳐 추정한다.
# 신뢰구간을 만들 때는 이렇게 합동하지 않는다. 목적이 다르기 때문이다.
p_pooled = (positive_A + positive_B) / (n_A + n_B)
statistic = (p_hat_A - p_hat_B) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_A + 1/n_B))
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0")
else:
    print("Fail to reject H_0")
```

출력:

```
statistic = 0.8528
p_value = 0.3938
Fail to reject H_0
```

지지율이 58%와 52%로 6%p 차이인데도 기각하지 못한다. 지구당 100명으로는 이 정도 차이를 가려낼 수 없다. 비율의 차이를 검정하려면 평균의 차이보다 훨씬 큰 표본이 필요하다.

</div>

<div class="codebox" markdown>

#### 예제 8. Derrick의 지지율 { .eg }

Derrick은 총리 지지율이 11월보다 12월에 낮은지 검정한다.

$$H_0 : p_{\text{Nov}} = p_{\text{Dec}} \quad\text{vs}\quad H_1: p_{\text{Nov}} > p_{\text{Dec}}$$

</div>

<div class="codebox" markdown>

#### 예제 9. 10센트와 5센트 동전 { .eg }

Kiley는 10센트 동전과 5센트 동전이 앞면을 보일 가능성이 같은지 검정한다.

$$H_0 : p_{\text{Dime}} = p_{\text{Nickel}} \quad\text{vs}\quad H_1: p_{\text{Dime}} \neq p_{\text{Nickel}}$$

연구자들이 2000년에서 2015년 사이에 근시 유병률이 높아졌는지 검정한다. 2000년: 400명 중 132명. 2015년: 600명 중 228명.

$$H_0 : p_{2000} = p_{2015} \quad\text{vs}\quad H_1: p_{2000} < p_{2015}$$

</div>

<div class="codebox" markdown>

#### 예제 10. 근시 비율의 변화 { .eg }

```python
n_2000, n_2015 = 400, 600
positive_2000, positive_2015 = 132, 228
p_hat_2000, p_hat_2015 = positive_2000 / n_2000, positive_2015 / n_2015

p_pooled = (positive_2000 + positive_2015) / (n_2000 + n_2015)
statistic = (p_hat_2000 - p_hat_2015) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_2000 + 1/n_2015))
# H1이 p_2000 < p_2015 이므로 왼쪽 꼬리를 센다.
p_value = stats.norm().cdf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: significant increase in myopia")
else:
    print("Fail to reject H_0")
```

출력:

```
statistic = -1.6137
p_value = 0.0533
Fail to reject H_0
```

$p = 0.0533$으로 0.05를 아슬아슬하게 넘겨 기각하지 못한다. 유병률이 33%에서 38%로 5%p 늘었지만 표본 1,000명으로는 부족하다.

이런 경계 사례를 "효과가 없다"로 읽으면 안 된다. 0.0533과 0.0467 사이에 실질적인 차이는 없다. 기각 여부라는 이분법 대신 신뢰구간과 효과크기를 함께 보고하는 편이 낫다.

</div>

수의사들이 수컷 고양이 259마리 중 24마리, 암컷 241마리 중 14마리가 이환된 자료로 $H_0: p_{\text{male}} = p_{\text{female}}$ 대 $H_1: p_{\text{male}} > p_{\text{female}}$을 검정한다.

<div class="codebox" markdown>

#### 예제 11. 고양이 질병 — 암수 비교 { .eg }

```python
# 두 비율이 같다는 귀무가설 아래에서는 둘을 합쳐 하나의 비율로 보는 것이
# 맞다. 아래 p_pooled 가 그것이며, 표준오차를 이 값으로 만든다.
positive_male, positive_female = 24, 14
n_male, n_female = 259, 241
p_hat_male, p_hat_female = positive_male / n_male, positive_female / n_female

p_pooled = (positive_male + positive_female) / (n_male + n_female)
statistic = (p_hat_male - p_hat_female) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_male + 1/n_female))
p_value = stats.norm().sf(statistic)      # H1: p_male > p_female 이므로 오른쪽 꼬리

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")
```

출력:

```
statistic = 1.4577
p_value = 0.0725
```

이환율이 9.3%와 5.8%로 수컷 쪽이 1.6배 높지만 $p = 0.0725$로 기각하지 못한다. 이환된 개체가 24마리와 14마리뿐이라, 500마리를 조사했어도 비교의 정밀도를 좌우하는 것은 전체 개체 수가 아니라 이 사건 수다.

</div>

<div class="codebox" markdown>

#### 예제 12. 대면 수업과 온라인 수업 { .eg }

$p_{\text{in\_person}} - p_{\text{online}}$의 95% 신뢰구간이 $(-0.04, 0.14)$이다. 구간이 0을 포함하므로 $H_0: p_{\text{in\_person}} = p_{\text{online}}$을 기각하지 못한다.

</div>

---

## 5. Mann-Whitney U 검정 (Wilcoxon 순위합 검정)

Mann-Whitney U 검정은 독립인 두 집단의 분포를 비교하는 비모수 검정이다. 모수적 검정의 가정이 충족되지 않을 때(예: 비정규성이나 순서형 자료) 유용하다.

### 핵심 특징

- 독립인 두 집단의 분포가 같은지 검정한다.
- 가정: 독립인 집단, 순서형/구간/비율 척도 자료, 확률표본.
- **귀무가설**: 두 집단의 분포가 같다.
- **대립가설**: 분포가 다르거나, 한 집단이 더 큰 값을 갖는 경향이 있다.

### 검정 방법

1. 모든 자료를 합쳐 **순위**를 매긴다(동점은 평균 순위를 준다).
2. 순위합 $R_1$과 $R_2$를 계산한다.
3. U-통계량을 계산한다:
    - $U_1 = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1$
    - $U_2 = n_1 n_2 + \frac{n_2(n_2+1)}{2} - R_2$
4. 검정통계량: $U = \min(U_1, U_2)$.
5. 표본이 크면($n_1, n_2 > 20$) Z-점수를 쓰는 정규근사를 적용한다.

### 해석

- $p < 0.05$이면 $H_0$을 기각한다. 두 집단의 분포가 유의하게 다르다.
- $p$가 크면 $H_0$을 기각하지 못한다.

### 어느 집단이 더 큰가?

$H_0$을 기각했다면 두 집단의 **평균 순위**를 비교한다. 평균 순위가 높은 집단이 더 큰 값을 갖는 경향이 있다.

### 참고

Mann-Whitney U 검정과 Wilcoxon 순위합 검정은 통계적으로 동등하다. 용어는 소프트웨어마다 다르다(예: SPSS에서는 "Mann-Whitney U", R에서는 "Wilcoxon rank-sum").

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
독립인 두 표본: 집단 1 ($n_1 = 25$, $\bar{x}_1 = 78$, $s_1 = 10$), 집단 2 ($n_2 = 30$, $\bar{x}_2 = 72$, $s_2 = 12$). 합동 표준오차(등분산 가정)를 써서 $\alpha = 0.05$에서 $H_0: \mu_1 = \mu_2$의 이표본 $t$-검정을 하라.

</div>

??? success "풀이"
    합동분산은:

    $$
    s_p^2 = \frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2} = \frac{24 \times 100 + 29 \times 144}{53} = \frac{2400 + 4176}{53} = \frac{6576}{53} \approx 124.08
    $$

    합동 표준오차는:

    $$
    \text{SE} = \sqrt{s_p^2\left(\frac{1}{n_1} + \frac{1}{n_2}\right)} = \sqrt{124.08 \times (0.04 + 0.0333)} = \sqrt{124.08 \times 0.0733} = \sqrt{9.095} \approx 3.016
    $$

    검정통계량은:

    $$
    t = \frac{78 - 72}{3.016} = \frac{6}{3.016} \approx 1.989
    $$

    $df = 53$에서 $t_{53, 0.025} \approx 2.006$이다. $|t| = 1.989 < 2.006$이므로 $\alpha = 0.05$에서 아슬아슬하게 $H_0$을 **기각하지 못한다**.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
합동 $t$-검정보다 Welch의 $t$-검정을 언제 선호해야 하는지, 그리고 분산이 다른데 합동 검정을 쓰면 무슨 일이 생기는지 설명하라.

</div>

??? success "풀이"
    두 집단의 **분산이 다를 때**($\sigma_1^2 \neq \sigma_2^2$) Welch의 $t$-검정을 선호해야 한다. 등분산을 가정하지 않고 자유도에 Welch-Satterthwaite 근사를 쓴다.

    분산이 다른데 합동 $t$-검정을 쓰면 합동 분산추정값이 부정확해진다: 서로 다른 두 분산을 평균하므로, 특히 표본크기까지 다르면 오해를 부를 수 있다. 분산이 큰 집단의 표본이 더 작으면 제1종 오류율이 $\alpha$ 위로 부푼다. 분산이 큰 집단의 표본이 더 크면 검정이 보수적이 된다(제1종 오류가 $\alpha$ 아래). Welch 검정은 두 문제를 모두 피하므로 기본으로 널리 권장된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
어떤 연구에서 두 집단 사이에 $p = 0.001$로 통계적으로 유의한 차이를 찾았고 평균 차이는 0.5 단위였다. 두 집단의 표준편차는 모두 50이다. 이 결과의 실질적 유의성을 논하라.

</div>

??? success "풀이"
    표준화 효과크기는 $d = 0.5/50 = 0.01$로 극도로 작다. $p$-값이 아주 작지만(통계적 유의성이 높지만) 표준편차 50에 비해 0.5 단위의 차이는 실질적으로 무시할 만하다.

    이는 **통계적 유의성**과 **실질적 유의성**의 구분을 보여준다. 표본이 충분히 크면 사소한 차이도 통계적 유의성을 얻을 수 있다. $p$-값은 차이가 정확히 0일 가능성이 낮다고 말할 뿐, 그 차이가 의미 있다고 말하지 않는다. 연구자는 항상 효과크기를 보고하고 그 크기가 응용 맥락에서 중요할 만한지 따져야 한다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이표본 $t$-검정보다 Mann-Whitney U 검정을 선호해야 하는 때는 언제인가?

</div>

??? success "풀이"
    다음의 경우 Mann-Whitney U 검정(Wilcoxon 순위합 검정)을 선호해야 한다:

    1. **자료가 정규분포를 따르지 않을 때.** 특히 중심극한정리가 $t$-검정을 충분히 보호하지 못하는 작은 표본에서 그렇다.
    2. **자료가 구간/비율 척도가 아니라 순서형일 때**(예: Likert 척도 평가). 평균은 의미가 없지만 순위는 의미가 있다.
    3. **이상점이 있어** $t$-검정에 과도한 영향을 줄 때. Mann-Whitney 검정은 순위에 기반하므로 이상점에 로버스트하다.
    4. **분포가 치우쳐 있고** 관심이 평균 자체보다 중심경향이나 확률적 순서의 비교에 있을 때.

    정규성 가정이 성립할 때는 Mann-Whitney 검정이 $t$-검정보다 검정력이 낮으므로(점근 상대효율이 $3/\pi \approx 0.955$), 자료가 분명히 정규이면 $t$-검정을 택한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 페이지가 소개한 다섯 검정($z$, 합동 $t$, Welch $t$, 만·휘트니, 그리고 순열검정)을 **하나의 자료에 모두** 적용하고, 왜 결론이 갈리는지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    A = np.array([4.2, 5.1, 3.8, 6.0, 4.7, 5.5, 3.9, 4.4, 5.8,
                  4.1, 6.3, 4.9, 5.2, 3.6, 4.6, 12.4, 5.0, 4.3])
    B = np.array([5.6, 6.2, 5.9, 7.1, 6.5, 5.4, 6.8, 5.1, 6.0,
                  7.4, 5.7, 6.1, 6.6, 5.3, 6.9, 5.8, 6.4, 7.0])
    n1, n2 = len(A), len(B)

    print(f"A: n={n1}  평균 {A.mean():.4f}  SD {A.std(ddof=1):.4f}  "
          f"중앙값 {np.median(A):.4f}")
    print(f"B: n={n2}  평균 {B.mean():.4f}  SD {B.std(ddof=1):.4f}  "
          f"중앙값 {np.median(B):.4f}\n")

    z = (A.mean() - B.mean()) / np.sqrt(A.var(ddof=1) / n1 + B.var(ddof=1) / n2)
    print(f"z 검정       z={z:8.4f}                 p={2 * stats.norm.sf(abs(z)):.4f}")

    r = stats.ttest_ind(A, B, equal_var=True)
    print(f"합동 t       t={r.statistic:8.4f}  df={n1 + n2 - 2:6d}   p={r.pvalue:.4f}")

    r = stats.ttest_ind(A, B, equal_var=False)
    print(f"Welch t      t={r.statistic:8.4f}  df={r.df:6.2f}   p={r.pvalue:.4f}")

    u = stats.mannwhitneyu(A, B)
    print(f"만·휘트니    U={u.statistic:8.1f}                 p={u.pvalue:.4f}")

    res = stats.permutation_test(
        (A, B), lambda x, y, axis=0: x.mean(axis) - y.mean(axis),
        permutation_type='independent', n_resamples=99_999,
        random_state=1, alternative='two-sided')
    print(f"순열검정                                p={res.pvalue:.4f}")
    ```

    ```text
    A: n=18  평균 5.2111  SD 1.9517  중앙값 4.8000
    B: n=18  평균 6.2111  SD 0.6685  중앙값 6.1500

    z 검정       z= -2.0565                 p=0.0397
    합동 t       t= -2.0565  df=    34   p=0.0475
    Welch t      t= -2.0565  df= 20.93   p=0.0524
    만·휘트니    U=    45.5                 p=0.0002
    순열검정                                p=0.0337
    ```

    **$p$가 0.0002에서 0.0524까지 흩어진다.** 0.05를 기준으로 결론이 갈린다.

    **관찰 1 — 세 검정통계량이 완전히 같다**($-2.0565$). $n_1=n_2$이면 합동 표준오차와 비합동 표준오차가 **정확히 같기** 때문이다.

    $$
    s_p^2\Bigl(\frac1n+\frac1n\Bigr)=\frac{s_1^2+s_2^2}{2}\cdot\frac2n=\frac{s_1^2}{n}+\frac{s_2^2}{n}
    $$

    **달라지는 것은 참조분포뿐**이다. $z$(무한 자유도) → $t_{34}$ → $t_{20.93}$ 순으로 꼬리가 두꺼워져 $p$가 커진다.

    **관찰 2 — 만·휘트니만 자릿수가 다르다**($p=0.0002$). 자료를 보면 이유가 보인다. **A에 12.4라는 이상점이 하나** 있다. 이 값이

    - **평균을 끌어올려** 두 집단의 평균 차이를 줄이고,
    - **표준편차를 1.95로 부풀려** 표준오차를 크게 만든다.

    순위로 바꾸면 12.4는 그저 "가장 큰 값"일 뿐이라 영향이 사라진다.

    ```python
    A2 = A[A < 10]                                   # 12.4 를 빼면
    print(f"A(이상점 제외): n={len(A2)}  평균 {A2.mean():.4f}  "
          f"SD {A2.std(ddof=1):.4f}")
    print(f"  Welch p = {stats.ttest_ind(A2, B, equal_var=False).pvalue:.6f}")
    ```

    ```text
    A(이상점 제외): n=17  평균 4.7882  SD 0.7921
      Welch p = 0.000003
    ```

    **관측값 하나가 $p$를 0.052에서 0.000003로 바꾼다.**

    **관찰 3 — 순열검정도 이상점에 취약하다**($p=0.0337$). 평균 차이를 통계량으로 쓰기 때문이다. **순열은 분포 가정을 없앨 뿐 통계량의 성질을 바꾸지 않는다.**

    **무엇이 옳은가.** 답은 "12.4가 무엇인가"에 달려 있다.

    | 12.4의 정체 | 올바른 처리 |
    |---|---|
    | 기록 오류 | 원자료를 확인해 고치거나 제외 |
    | 실제로 일어난 극단값 | **남겨 두고** 로버스트 검정 |
    | 다른 모집단에서 온 값 | 모집단 정의를 다시 검토 |

    **보고 원칙.** 이상점이 결론을 뒤집을 때는 **양쪽 결과를 모두 보고**한다. "이상점을 포함하면 $p=0.052$, 제외하면 $p<0.001$"이라고 쓰는 것이 정직하다. 유리한 쪽만 고르는 것이 $p$-해킹이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
만·휘트니 검정의 **정확분포와 정규근사**가 언제 갈리는지, 그리고 **동점**이 있으면 어떻게 되는지 확인하라. 아주 작은 표본에서 이 검정의 한계는 무엇인가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    x = np.array([3.1, 4.5, 2.8, 5.0, 3.9, 4.2])
    y = np.array([5.5, 6.1, 4.8, 7.0, 5.9, 6.3])
    for method in ["exact", "asymptotic"]:
        r = stats.mannwhitneyu(x, y, method=method)
        print(f"{method:12s} U={r.statistic:5.1f}  p={r.pvalue:.6f}")

    print("\n동점이 있는 자료 (정수로 반올림)")
    xt = np.array([3, 4, 3, 5, 4, 4])
    yt = np.array([5, 6, 5, 7, 6, 6])
    r = stats.mannwhitneyu(xt, yt, method="asymptotic")
    print(f"  정규근사(동점 보정)  U={r.statistic:5.1f}  p={r.pvalue:.6f}")
    r = stats.mannwhitneyu(xt, yt, method="exact")
    print(f"  정확법(동점 무시)    U={r.statistic:5.1f}  p={r.pvalue:.6f}")

    print("\n두 집단이 완전히 분리됐을 때 도달 가능한 최소 양측 p")
    for k in [3, 4, 5, 6, 8]:
        a, b = np.arange(k), np.arange(k) + 100
        print(f"  n1=n2={k}: p = {stats.mannwhitneyu(a, b, method='exact').pvalue:.6f}")
    ```

    ```text
    exact        U=  1.0  p=0.004329
    asymptotic   U=  1.0  p=0.008239

    동점이 있는 자료 (정수로 반올림)
      정규근사(동점 보정)  U=  1.0  p=0.006845
      정확법(동점 무시)    U=  1.0  p=0.004329

    두 집단이 완전히 분리됐을 때 도달 가능한 최소 양측 p
      n1=n2=3: p = 0.100000
      n1=n2=4: p = 0.028571
      n1=n2=5: p = 0.007937
      n1=n2=6: p = 0.002165
      n1=n2=8: p = 0.000155
    ```

    **1 — 소표본에서 정규근사가 두 배 보수적이다.** $n_1=n_2=6$에서 정확법 0.0043, 정규근사 0.0082다. **작은 표본에서는 반드시 `method="exact"`를 쓴다.**

    **2 — 동점이 있으면 정확법을 쓸 수 없다.** `scipy`는 오류를 내지 않고 **동점을 무시한 값**(0.004329)을 돌려준다. 동점 보정을 한 정규근사(0.006845)와 다르다. **조용히 틀린 값을 주므로 주의해야 한다.**

    동점이 있으면

    - **정규근사 + 동점 보정**을 쓰거나,
    - **순열검정**(동점을 그대로 다룸)을 쓴다.

    **3 — 아주 작은 표본에서는 기각 자체가 불가능하다.** $n_1=n_2=3$이면 두 집단이 **완전히 분리돼도** 양측 $p$의 최솟값이 0.10이다. 순위의 배열이 $\binom{6}{3}=20$가지뿐이라 가장 극단적인 경우의 확률이 $2/20=0.1$이다.

    | $n_1=n_2$ | 도달 가능한 최소 $p$ | $\alpha=0.05$에서 |
    |---|---|---|
    | 3 | 0.100 | **기각 불가능** |
    | 4 | 0.029 | 완전 분리에서만 가능 |
    | 5 | 0.008 | 가능 |
    | 8 | 0.00016 | 여유 있음 |

    **이것이 비모수 검정의 근본적 한계**다. 순위만 쓰므로 가능한 결과의 가짓수가 유한하고, 표본이 작으면 그 가짓수가 너무 적다.

    **실무 지침.**

    | 조건 | 권장 |
    |---|---|
    | $n_1,n_2\ge4$이고 동점 없음 | `method="exact"` |
    | 동점 있음 | `method="asymptotic"`(동점 보정 포함) |
    | $n_1,n_2<4$ | **비모수로는 불가능** — 다른 설계를 찾는다 |
    | $n$이 크고 동점 많음 | 순열검정 또는 정규근사 |

    **연속형 자료를 반올림해 기록하면 인위적 동점이 생긴다.** 가능하면 원래 정밀도로 기록하는 것이 낫다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
**순열검정이 이 페이지의 여러 검정을 하나의 틀로 묶는다**는 것을 보여라. 같은 순열 절차에 통계량만 바꿔 넣어 $t$ 검정과 만·휘트니를 재현하라.

</div>

??? success "풀이"
    **순열검정의 논리.** $H_0$가 "두 집단의 분포가 같다"이면, **집단 이름표는 자료와 무관**하다. 따라서 이름표를 임의로 섞어 만든 통계량들의 분포가 곧 귀무분포다.

    $$
    p=\frac{1+\#\{\text{섞은 자료의 통계량}\ \ge\ \text{관측 통계량}\}}{1+B}
    $$

    **어떤 통계량이든 쓸 수 있다**는 것이 핵심이다.

    ```python
    import numpy as np
    from scipy import stats

    A = np.array([4.2, 5.1, 3.8, 6.0, 4.7, 5.5, 3.9, 4.4, 5.8,
                  4.1, 6.3, 4.9, 5.2, 3.6, 4.6, 12.4, 5.0, 4.3])
    B = np.array([5.6, 6.2, 5.9, 7.1, 6.5, 5.4, 6.8, 5.1, 6.0,
                  7.4, 5.7, 6.1, 6.6, 5.3, 6.9, 5.8, 6.4, 7.0])

    def perm_p(x, y, stat, B_=19_999, seed=0):
        """같은 순열 틀에 통계량만 갈아 끼운다."""
        rng = np.random.default_rng(seed)
        pooled = np.r_[x, y]
        n1 = len(x)
        obs = abs(stat(pooled[:n1], pooled[n1:]))
        cnt = 1
        for _ in range(B_):
            z = rng.permutation(pooled)
            cnt += abs(stat(z[:n1], z[n1:])) >= obs
        return cnt / (B_ + 1)

    stats_to_try = {
        "평균 차이":      lambda a, b: a.mean() - b.mean(),
        "합동 t 통계량":  lambda a, b: stats.ttest_ind(a, b, equal_var=True).statistic,
        "Welch t 통계량": lambda a, b: stats.ttest_ind(a, b, equal_var=False).statistic,
        # 순위합은 양수이므로 abs() 가 통하도록 기댓값을 빼서 중심화한다
        "순위합(만·휘트니)": lambda a, b: (stats.rankdata(np.r_[a, b])[:len(a)].sum()
                                        - len(a) * (len(a) + len(b) + 1) / 2),
        "중앙값 차이":    lambda a, b: np.median(a) - np.median(b),
        "20% 절사평균 차이": lambda a, b: (stats.trim_mean(a, 0.2)
                                          - stats.trim_mean(b, 0.2)),
    }
    for name, f in stats_to_try.items():
        print(f"{name:20s} 순열 p = {perm_p(A, B, f):.4f}")

    print(f"\n참고: 합동 t 검정      p = "
          f"{stats.ttest_ind(A, B, equal_var=True).pvalue:.4f}")
    print(f"      만·휘트니 검정   p = {stats.mannwhitneyu(A, B).pvalue:.4f}")
    ```

    ```text
    평균 차이                순열 p = 0.0318
    합동 t 통계량             순열 p = 0.0318
    Welch t 통계량          순열 p = 0.0317
    순위합(만·휘트니)           순열 p = 0.0002
    중앙값 차이               순열 p = 0.0001
    20% 절사평균 차이          순열 p = 0.0001

    참고: 합동 t 검정      p = 0.0475
          만·휘트니 검정   p = 0.0002
    ```

    **세 가지가 드러난다.**

    **1 — 평균 차이·합동 $t$·Welch $t$가 거의 같은 $p$를 준다**(0.0318, 0.0318, 0.0317). $n_1=n_2$이면 세 통계량이 순열 안에서 **같은 순서**를 매기기 때문이다. 남는 차이는 몬테카를로 오차뿐이다.

    **그리고 이 값(0.032)이 합동 $t$ 검정의 $p$(0.0475)와 다르다.** 순열은 $t$ 분포를 쓰지 않고 **이 자료에서 직접 만든 귀무분포**를 쓰기 때문이다. 이상점 때문에 $t$ 분포 근사가 잘 맞지 않는 상황이라 차이가 제법 크다.

    **2 — 순위합 통계량을 쓰면 만·휘트니가 그대로 재현된다**(0.0002 대 0.0002). **만·휘트니 검정은 "순위합을 통계량으로 쓰는 순열검정"이다.**

    다만 순위합은 항상 양수이므로 `abs()`로 양측을 만들려면 **기댓값 $n_1(n_1+n_2+1)/2$를 먼저 빼야** 한다. 이 중심화를 빠뜨리면 $p$가 1에 가깝게 나온다.

    **3 — 로버스트 통계량들이 훨씬 작은 $p$를 준다**(중앙값 0.0001, 절사평균 0.0001). 이상점의 영향을 받지 않기 때문이다. **같은 순열 틀에서 통계량만 바꿨는데 $p$가 300배 차이난다** — 검정의 힘이 어디서 오는지 보여 준다.

    **순열검정이 주는 자유.**

    | 얻는 것 | 설명 |
    |---|---|
    | **분포 가정 불필요** | 정규성을 요구하지 않는다 |
    | **통계량 자유** | 중앙값, 절사평균, 지니계수, 무엇이든 |
    | **정확한 수준** | 유한표본에서 정확하다(모든 순열을 쓰면) |
    | **해석 가능** | "우연히 이런 차이가 날 확률" 그대로 |

    **치르는 대가.**

    1. **교환가능성이 필요하다.** $H_0$가 "분포가 완전히 같다"여야 한다. 분산이 다르면 엄밀한 수준이 보장되지 않는다(그래서 Welch 통계량을 넣어도 완전하지는 않다).
    2. **신뢰구간을 직접 주지 않는다.** 구간이 필요하면 붓스트랩이나 검정을 반전시키는 방법을 쓴다.
    3. **계산량이 있다.** 다만 요즘 하드웨어에서는 거의 문제가 되지 않는다.

    **가장 중요한 교훈.** 검정을 고르는 일은 **통계량을 고르는 일**이다. 분포 가정은 그 통계량의 귀무분포를 얻기 위한 수단일 뿐이고, 순열은 그 수단을 자료 자체에서 만들어 낸다. **"어떤 요약값으로 두 집단을 비교할 것인가"가 본질적 질문**이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 5의 자료에 쓸 수 있는 **효과크기**를 모두 계산하고, 서로 왜 다른지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    A = np.array([4.2, 5.1, 3.8, 6.0, 4.7, 5.5, 3.9, 4.4, 5.8,
                  4.1, 6.3, 4.9, 5.2, 3.6, 4.6, 12.4, 5.0, 4.3])
    B = np.array([5.6, 6.2, 5.9, 7.1, 6.5, 5.4, 6.8, 5.1, 6.0,
                  7.4, 5.7, 6.1, 6.6, 5.3, 6.9, 5.8, 6.4, 7.0])
    n1, n2 = len(A), len(B)

    sp = np.sqrt(((n1 - 1) * A.var(ddof=1) + (n2 - 1) * B.var(ddof=1))
                 / (n1 + n2 - 2))
    d = (B.mean() - A.mean()) / sp
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))
    glass = (B.mean() - A.mean()) / A.std(ddof=1)      # 대조군 SD 기준
    t = stats.ttest_ind(A, B, equal_var=True).statistic
    r_pb = abs(t) / np.sqrt(t**2 + n1 + n2 - 2)
    u = stats.mannwhitneyu(A, B).statistic
    r_rb = 1 - 2 * u / (n1 * n2)
    cles = 1 - u / (n1 * n2)

    print(f"{'원 척도 평균 차이':<22s} {B.mean() - A.mean():8.4f}")
    print(f"{'원 척도 중앙값 차이':<22s} {np.median(B) - np.median(A):8.4f}")
    print(f"{'20% 절사평균 차이':<22s} "
          f"{stats.trim_mean(B, 0.2) - stats.trim_mean(A, 0.2):8.4f}")
    print(f"{'Cohen d':<22s} {d:8.4f}")
    print(f"{'Hedges g':<22s} {g:8.4f}")
    print(f"{'Glass Δ':<22s} {glass:8.4f}")
    print(f"{'점이연 상관 r':<22s} {r_pb:8.4f}")
    print(f"{'순위이연 상관':<22s} {r_rb:8.4f}")
    print(f"{'CLES  P(B > A)':<22s} {cles:8.4f}")
    ```

    ```text
    원 척도 평균 차이               1.0000
    원 척도 중앙값 차이              1.3500
    20% 절사평균 차이              1.3917
    Cohen d                  0.6855
    Hedges g                 0.6703
    Glass Δ                  0.5124
    점이연 상관 r                 0.3326
    순위이연 상관                  0.7191
    CLES  P(B > A)           0.8596
    ```

    **"효과의 크기"가 0.33부터 0.86까지 나온다.** 모두 같은 자료다.

    **왜 이렇게 다른가.**

    | 지표 | 무엇으로 나누는가 | 이상점의 영향 |
    |---|---|---|
    | Cohen $d$ | 합동 SD(1.46) | **큼** — 12.4가 SD를 부풀림 |
    | Glass $\Delta$ | A의 SD(1.95) | **더 큼** |
    | 점이연 $r$ | $t$의 단조변환 | 큼 |
    | 순위이연 $r$ | 순위만 사용 | **없음** |
    | CLES | 순위만 사용 | **없음** |

    **순위 기반 지표(0.72, 0.86)가 훨씬 크다.** 순위로 보면 두 집단이 거의 겹치지 않기 때문이다. **CLES 0.86**은 "임의로 고른 B의 값이 임의의 A보다 클 확률이 86%"라는 뜻으로, 가장 직관적인 표현이다.

    **원 척도가 여전히 가장 중요하다.** 평균 차이 1.0과 중앙값 차이 1.35 중 어느 쪽이 의미 있는지는 **측정 단위가 무엇인가**에 달려 있다. 표준화 지표는 원 척도를 대체하는 것이 아니라 보완한다.

    **평균 차이(1.00)와 중앙값 차이(1.35)가 35% 다르다.** 이 자체가 **분포가 대칭이 아님을 알리는 신호**다.

    **선택 기준.**

    | 목적 | 지표 |
    |---|---|
    | **해석과 의사결정** | 원 척도 차이 + 신뢰구간 |
    | 다른 연구와 비교 | Hedges $g$ |
    | 메타분석 입력 | Hedges $g$ + 그 분산 |
    | 비전문가에게 설명 | **CLES** |
    | 이상점·치우침이 있음 | 순위이연, 절사평균 차이 |
    | 소표본($n<20$) | Hedges $g$(Cohen $d$는 위로 편향) |

    **주의 둘.**

    1. **관례적 구간(0.2/0.5/0.8)을 기계적으로 적용하지 않는다.** 분야에 따라 $d=0.2$가 대단히 큰 효과일 수 있다.
    2. **효과크기에도 신뢰구간을 붙인다.** $n=18$씩이면 $d$의 구간이 대략 $\pm0.67$로 매우 넓다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
이표본 $t$ 검정이 **더미변수를 쓴 단순회귀와 같다**는 것을 보여라. 이 관점에서 Welch 검정은 무엇에 대응하는가?

</div>

??? success "풀이"
    **모형.** 집단 표시 $G_i\in\{0,1\}$에 대해

    $$
    Y_i=\beta_0+\beta_1G_i+\varepsilon_i,\qquad \varepsilon_i\sim N(0,\sigma^2)
    $$

    를 적합하면

    - $\hat\beta_0=\bar Y_{G=0}$ (0집단의 평균),
    - $\hat\beta_1=\bar Y_{G=1}-\bar Y_{G=0}$ (두 평균의 차이),
    - $\hat\beta_1$의 $t$ 통계량 $=$ **합동 이표본 $t$ 통계량**.

    ```python
    import numpy as np
    from scipy import stats

    A = np.array([4.2, 5.1, 3.8, 6.0, 4.7, 5.5, 3.9, 4.4, 5.8,
                  4.1, 6.3, 4.9, 5.2, 3.6, 4.6, 12.4, 5.0, 4.3])
    B = np.array([5.6, 6.2, 5.9, 7.1, 6.5, 5.4, 6.8, 5.1, 6.0,
                  7.4, 5.7, 6.1, 6.6, 5.3, 6.9, 5.8, 6.4, 7.0])

    y = np.r_[A, B]
    g = np.r_[np.zeros(len(A)), np.ones(len(B))]
    X = np.column_stack([np.ones(len(y)), g])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    XtXinv = np.linalg.inv(X.T @ X)
    s2 = resid @ resid / (n - p)
    se = np.sqrt(np.diag(s2 * XtXinv))
    t = beta[1] / se[1]

    print(f"절편 β0  = {beta[0]:.4f}   (A의 평균 {A.mean():.4f})")
    print(f"기울기 β1 = {beta[1]:.4f}   (B - A = {B.mean() - A.mean():.4f})")
    print(f"β1 의 SE  = {se[1]:.4f},  t = {t:.4f},  df = {n - p}, "
          f"p = {2 * stats.t.sf(abs(t), n - p):.4f}")
    r = stats.ttest_ind(B, A, equal_var=True)
    print(f"합동 t 검정:        t = {r.statistic:.4f},  p = {r.pvalue:.4f}")

    sst, sse = ((y - y.mean())**2).sum(), resid @ resid
    F = ((sst - sse) / 1) / (sse / (n - p))
    print(f"\nANOVA  F = {F:.4f} = t² = {t**2:.4f},  R² = {1 - sse / sst:.4f}")
    print(f"점이연 상관 r = {np.corrcoef(g, y)[0, 1]:.4f}, "
          f"r² = {np.corrcoef(g, y)[0, 1]**2:.4f}")

    # HC3 이분산 로버스트 표준오차
    h = np.einsum('ij,jk,ik->i', X, XtXinv, X)
    u = resid / (1 - h)
    V = XtXinv @ (X.T @ np.diag(u**2) @ X) @ XtXinv
    se_r = np.sqrt(np.diag(V))
    print(f"\nHC3 로버스트 SE = {se_r[1]:.4f},  t = {beta[1] / se_r[1]:.4f}")
    rw = stats.ttest_ind(B, A, equal_var=False)
    print(f"Welch: t = {rw.statistic:.4f}, df = {rw.df:.2f}, "
          f"SE = {(B.mean() - A.mean()) / rw.statistic:.4f}")
    ```

    ```text
    절편 β0  = 5.2111   (A의 평균 5.2111)
    기울기 β1 = 1.0000   (B - A = 1.0000)
    β1 의 SE  = 0.4863,  t = 2.0565,  df = 34, p = 0.0475
    합동 t 검정:        t = 2.0565,  p = 0.0475

    ANOVA  F = 4.2291 = t² = 4.2291,  R² = 0.1106
    점이연 상관 r = 0.3326, r² = 0.1106

    HC3 로버스트 SE = 0.5004,  t = 1.9985
    Welch: t = 2.0565, df = 20.93, SE = 0.4863
    ```

    **완전히 일치한다.** 그리고 덤으로

    - $F=t^2$ — **일원배치 ANOVA도 같은 것**이다(집단이 2개일 때).
    - $R^2=r_{pb}^2=0.1106$ — **점이연 상관의 제곱**이 결정계수다.

    **Welch는 무엇에 대응하는가.** 회귀에서 등분산 가정을 버리면 **이분산 로버스트 표준오차**(화이트/HC 계열)를 쓴다. 위에서 HC3 SE가 0.5004로, 합동 SE 0.4863보다 크다.

    **다만 완전히 같지는 않다.**

    | | Welch | HC3 회귀 |
    |---|---|---|
    | 표준오차 | 0.4863 | 0.5004 |
    | 자유도 | 20.93(새터스웨이트) | 34 또는 정규근사 |

    $n_1=n_2$이면 Welch의 SE가 합동 SE와 **정확히 같아지고**, 차이는 자유도뿐이다. HC3는 반대로 SE를 조정하고 자유도는 그대로 둔다. **두 방법은 이분산을 다른 방식으로 다룬다** — 소표본에서는 Welch가, 대표본에서는 둘이 같아진다.

    **이 관점이 주는 이득.**

    1. **공변량을 넣을 수 있다.** $Y=\beta_0+\beta_1G+\beta_2X+\varepsilon$은 공분산분석(ANCOVA)이고, 교란을 보정하면서 집단 차이를 본다.
    2. **집단이 셋 이상이어도 된다.** 더미를 늘리면 ANOVA다.
    3. **군집·시계열 구조를 다룰 수 있다.** 군집 로버스트 표준오차, 혼합모형으로 확장된다.
    4. **결과가 이진·계수여도 된다.** 로지스틱·포아송 회귀로 자연스럽게 넘어간다.

    **"$t$ 검정, ANOVA, 상관, 회귀는 모두 같은 선형모형"**이라는 관점은 통계학을 배우는 데 가장 유용한 통합 시각 중 하나다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 페이지의 다섯 검정을 **언제 쓸지** 한 장으로 정리하라.

</div>

??? success "풀이"

    **결정 흐름.**

    ```text
    독립인 두 집단을 비교한다
        │
        ├─ 결과가 이진(성공/실패)인가
        │      └─ 예 ──→ 두 비율 z 검정
        │                 (기대도수가 작으면 피셔 정확검정)
        │
        └─ 연속형·순서형
             │
             ├─ σ 를 정말 아는가 (오랜 이력·기기 명세)
             │      └─ 예 ──→ 이표본 z 검정   ※ 드물다
             │
             ├─ 그림을 그린다 (상자그림·정규분위수그림)
             │
             ├─ 이상점이나 두꺼운 꼬리가 있는가
             │      ├─ 예 ──→ 만·휘트니, 절사평균 검정, 순열검정
             │      └─ 아니오
             │             └─→ Welch t 검정  ← 기본 선택
             │
             └─ 표본이 아주 작은가(<10)
                    └─→ 순열검정(정확한 수준)
    ```

    **다섯 검정 비교표.**

    | 검정 | 가정 | 검정 대상 | 언제 |
    |---|---|---|---|
    | **$z$ 검정** | $\sigma$ 기지, 정규 또는 대표본 | $\mu_1-\mu_2$ | $\sigma$를 실제로 알 때 |
    | **합동 $t$** | 정규, **등분산** | $\mu_1-\mu_2$ | 등분산이 설계로 보장될 때 |
    | **Welch $t$** | 정규(로버스트) | $\mu_1-\mu_2$ | **기본** |
    | **만·휘트니** | 독립, 연속(동점 적음) | $P(Y>X)$ | 순위가 관심, 이상점 |
    | **순열검정** | **교환가능성** | 통계량에 따라 | 소표본, 특이 통계량 |

    **세 가지 핵심 권고.**

    **1 — 기본값은 Welch $t$다.** 등분산일 때 잃는 것이 거의 없고, 아닐 때 얻는 것이 크다. `scipy`의 `equal_var=False`가 기본값인 이유다.

    **2 — 사전검정으로 검정을 고르지 않는다.** 등분산 검정도, 정규성 검정도 문지기로 쓰지 않는다. **자료의 성격과 연구 질문**으로 미리 정한다.

    **3 — 검정 대상이 다르다는 점을 잊지 않는다.** 만·휘트니로 바꾸는 것은 "더 안전한 검정"으로 바꾸는 것이 아니라 **다른 질문으로 바꾸는 것**이다.

    **어떤 검정을 쓰든 함께 보고할 것.**

    - [ ] 각 집단의 $n$, 중심, 산포
    - [ ] **차이의 추정값과 신뢰구간**
    - [ ] 효과크기(원 척도와 표준화 척도 모두)
    - [ ] 검정통계량, 자유도, $p$-값
    - [ ] 검정을 고른 **이유**
    - [ ] 이상점·결측의 처리 방식
    - [ ] 표본크기를 어떻게 정했는지

    **마지막으로 — 가장 중요한 가정은 독립성이다.** 정규성과 등분산은 방법으로 피해 갈 수 있지만, 관측이 독립이 아니면 **어떤 이표본 검정도 유효하지 않다.** 자료가 어떻게 수집됐는지를 먼저 보라.

---

## 정리하며

이표본 검정 네 가지를 **한 표로** 모은다.

| 비교 대상 | 조건 | 참조 분포 |
|---|---|---|
| 평균 차 | $\sigma$ 기지 | $N(0,1)$ |
| 평균 차 | $\sigma$ 미지 | $t$ (웰치 권장) |
| 비율 차 | 대표본 | $N(0,1)$, 합동비율 |
| 분산 비 | 정규모집단 | $F_{n_1-1,n_2-1}$ |

- **모두 독립을 전제한다.** 같은 대상에서 두 번 측정했다면 대응 검정으로 가야 하며, 다음 절의 주제다.
- **강건성의 순서는 평균 > 비율 > 분산**이다. 앞의 둘은 중심극한정리의 보호를 받지만 분산 검정은 그렇지 않다.
- **검정과 신뢰구간을 함께 보고한다.** $p$ 값은 "차이가 있는가"에만 답하고 "얼마나"에는 답하지 않는다.
- **표본크기가 다르면 조심한다.** 불균형 설계에서 합동 방법과 웰치의 차이가 가장 크게 벌어진다.

다음 절부터 이 검정들의 **구현과 실제 자료 적용**을 다룬다.
