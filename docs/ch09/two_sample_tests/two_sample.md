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

```python
import numpy as np
from scipy import stats

n_f, n_s = 100, 100
x_bar_f, x_bar_s = 1.85, 1.65
s_f, s_s = 1.3, 1.2

statistic = (x_bar_f - x_bar_s) / np.sqrt(s_f**2/n_f + s_s**2/n_s)
p_value = stats.norm().sf(abs(statistic)) * 2

print(f"statistic : {statistic:.4f}")
print(f"p value   : {p_value:.4f}")
```

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

#### 예제: 급여의 성별 격차

시장조사자들이 남성 관리자와 여성 관리자의 평균 급여를 비교한다.

$$H_0 : \mu_{\text{men}} = \mu_{\text{women}} \quad\text{vs}\quad H_1: \mu_{\text{men}} > \mu_{\text{women}}$$

#### 예제: 서로 다른 두 밭의 토마토

| | 밭 A | 밭 B |
|:---:|:---:|:---:|
| 평균 | 1.3 m | 1.6 m |
| 표준편차 | 0.5 m | 0.3 m |
| n | 22 | 24 |

$$H_0 : \mu_A = \mu_B \quad\text{vs}\quad H_1: \mu_A \neq \mu_B$$

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

X_1_bar, X_2_bar = 1.3, 1.6
s_1, s_2 = 0.5, 0.3
n_1, n_2 = 22, 24

# Welch's approach (unequal variances)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)

# Welch-Satterthwaite degrees of freedom
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

#### 예제: 출생아 수 (France 대 Switzerland)

| | France | Switzerland |
|:---:|:---:|:---:|
| 평균 | 1.85 | 1.65 |
| 표준편차 | 1.3 | 1.2 |
| n | 100 | 100 |

합동분산을 쓰면:

```python
X_1_bar, X_2_bar = 1.85, 1.65
s_1, s_2 = 1.3, 1.2
n_1, n_2 = 100, 100

s_p_square = ((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 + n_2 - 2)
statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
df = n_1 + n_2 - 2
p_value = 2 * stats.t(df).cdf(-abs(statistic))

print(f"{df = :.4f}")
print(f"{statistic = :.4f}")
print(f"{p_value   = :.4f}")
```

#### 예제: 두 품종의 배 (Bosc와 Anjou)

| | Bosc | Anjou |
|:---:|:---:|:---:|
| 평균 | 120 | 116 |
| 표준편차 | 15 | 13 |
| n | 65 | 65 |

$\mu_{\text{Bosc}} - \mu_{\text{Anjou}}$의 99% 신뢰구간은 $4 \pm 6.44$, 즉 $(-2.44, 10.44)$이다. 신뢰구간이 0을 포함하므로 $\alpha = 0.01$에서 $H_0$을 기각하지 못한다.

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

### Python 구현

```python
import numpy as np
from scipy.stats import ttest_ind

team_a = [120, 118, 125, 130, 115, 122, 121, 119, 117, 123, 124, 126, 127, 118, 116]
team_b = [135, 132, 137, 140, 136, 130, 134, 138, 139, 133, 131, 142, 141,
           129, 128, 135, 137, 136, 134, 132]

stat, p_value = ttest_ind(team_a, team_b, equal_var=False)
print(f"Test Statistic: {stat:.4f}")
print(f"P-value: {p_value:.4f}")

alpha = 0.05
if p_value < alpha:
    print("Reject H0: The means are significantly different.")
else:
    print("Fail to reject H0.")
```

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

#### 예제: 새 법률에 대한 지지

| | A 지구 | B 지구 | 합계 |
|:---:|:---:|:---:|:---:|
| 예 | 58 | 52 | 110 |
| 아니오 | 42 | 48 | 90 |

$$H_0 : p_A = p_B \quad\text{vs}\quad H_1: p_A \neq p_B$$

```python
import numpy as np
from scipy import stats

positive_A, positive_B = 58, 52
n_A, n_B = 100, 100
p_hat_A, p_hat_B = positive_A / n_A, positive_B / n_B

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

#### 예제: Derrick의 지지율

Derrick은 총리 지지율이 11월보다 12월에 낮은지 검정한다.

$$H_0 : p_{\text{Nov}} = p_{\text{Dec}} \quad\text{vs}\quad H_1: p_{\text{Nov}} > p_{\text{Dec}}$$

#### 예제: 10센트와 5센트 동전

Kiley는 10센트 동전과 5센트 동전이 앞면을 보일 가능성이 같은지 검정한다.

$$H_0 : p_{\text{Dime}} = p_{\text{Nickel}} \quad\text{vs}\quad H_1: p_{\text{Dime}} \neq p_{\text{Nickel}}$$

#### 예제: 근시

연구자들이 2000년에서 2015년 사이에 근시 유병률이 높아졌는지 검정한다. 2000년: 400명 중 132명. 2015년: 600명 중 228명.

$$H_0 : p_{2000} = p_{2015} \quad\text{vs}\quad H_1: p_{2000} < p_{2015}$$

```python
n_2000, n_2015 = 400, 600
positive_2000, positive_2015 = 132, 228
p_hat_2000, p_hat_2015 = positive_2000 / n_2000, positive_2015 / n_2015

p_pooled = (positive_2000 + positive_2015) / (n_2000 + n_2015)
statistic = (p_hat_2000 - p_hat_2015) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_2000 + 1/n_2015))
p_value = stats.norm().cdf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")

alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: significant increase in myopia")
else:
    print("Fail to reject H_0")
```

#### 예제: 고양이 질병

수의사들이 수컷 고양이 259마리 중 24마리, 암컷 241마리 중 14마리가 이환된 자료로 $H_0: p_{\text{male}} = p_{\text{female}}$ 대 $H_1: p_{\text{male}} > p_{\text{female}}$을 검정한다.

```python
positive_male, positive_female = 24, 14
n_male, n_female = 259, 241
p_hat_male, p_hat_female = positive_male / n_male, positive_female / n_female

p_pooled = (positive_male + positive_female) / (n_male + n_female)
statistic = (p_hat_male - p_hat_female) / (np.sqrt(p_pooled * (1 - p_pooled)) * np.sqrt(1/n_male + 1/n_female))
p_value = stats.norm().sf(statistic)

print(f"{statistic = :.4f}")
print(f"{p_value = :.4f}")
```

#### 예제: 대면 수업과 온라인 수업

$p_{\text{in\_person}} - p_{\text{online}}$의 95% 신뢰구간이 $(-0.04, 0.14)$이다. 구간이 0을 포함하므로 $H_0: p_{\text{in\_person}} = p_{\text{online}}$을 기각하지 못한다.

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

**연습문제 1.**
독립인 두 표본: 집단 1 ($n_1 = 25$, $\bar{x}_1 = 78$, $s_1 = 10$), 집단 2 ($n_2 = 30$, $\bar{x}_2 = 72$, $s_2 = 12$). 합동 표준오차(등분산 가정)를 써서 $\alpha = 0.05$에서 $H_0: \mu_1 = \mu_2$의 이표본 $t$-검정을 하라.

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

---

**연습문제 2.**
합동 $t$-검정보다 Welch의 $t$-검정을 언제 선호해야 하는지, 그리고 분산이 다른데 합동 검정을 쓰면 무슨 일이 생기는지 설명하라.

??? success "풀이"
    두 집단의 **분산이 다를 때**($\sigma_1^2 \neq \sigma_2^2$) Welch의 $t$-검정을 선호해야 한다. 등분산을 가정하지 않고 자유도에 Welch-Satterthwaite 근사를 쓴다.

    분산이 다른데 합동 $t$-검정을 쓰면 합동 분산추정값이 부정확해진다: 서로 다른 두 분산을 평균하므로, 특히 표본크기까지 다르면 오해를 부를 수 있다. 분산이 큰 집단의 표본이 더 작으면 제1종 오류율이 $\alpha$ 위로 부푼다. 분산이 큰 집단의 표본이 더 크면 검정이 보수적이 된다(제1종 오류가 $\alpha$ 아래). Welch 검정은 두 문제를 모두 피하므로 기본으로 널리 권장된다.

---

**연습문제 3.**
어떤 연구에서 두 집단 사이에 $p = 0.001$로 통계적으로 유의한 차이를 찾았고 평균 차이는 0.5 단위였다. 두 집단의 표준편차는 모두 50이다. 이 결과의 실질적 유의성을 논하라.

??? success "풀이"
    표준화 효과크기는 $d = 0.5/50 = 0.01$로 극도로 작다. $p$-값이 아주 작지만(통계적 유의성이 높지만) 표준편차 50에 비해 0.5 단위의 차이는 실질적으로 무시할 만하다.

    이는 **통계적 유의성**과 **실질적 유의성**의 구분을 보여준다. 표본이 충분히 크면 사소한 차이도 통계적 유의성을 얻을 수 있다. $p$-값은 차이가 정확히 0일 가능성이 낮다고 말할 뿐, 그 차이가 의미 있다고 말하지 않는다. 연구자는 항상 효과크기를 보고하고 그 크기가 응용 맥락에서 중요할 만한지 따져야 한다.

---

**연습문제 4.**
이표본 $t$-검정보다 Mann-Whitney U 검정을 선호해야 하는 때는 언제인가?

??? success "풀이"
    다음의 경우 Mann-Whitney U 검정(Wilcoxon 순위합 검정)을 선호해야 한다:

    1. **자료가 정규분포를 따르지 않을 때.** 특히 중심극한정리가 $t$-검정을 충분히 보호하지 못하는 작은 표본에서 그렇다.
    2. **자료가 구간/비율 척도가 아니라 순서형일 때**(예: Likert 척도 평가). 평균은 의미가 없지만 순위는 의미가 있다.
    3. **이상점이 있어** $t$-검정에 과도한 영향을 줄 때. Mann-Whitney 검정은 순위에 기반하므로 이상점에 로버스트하다.
    4. **분포가 치우쳐 있고** 관심이 평균 자체보다 중심경향이나 확률적 순서의 비교에 있을 때.

    정규성 가정이 성립할 때는 Mann-Whitney 검정이 $t$-검정보다 검정력이 낮으므로(점근 상대효율이 $3/\pi \approx 0.955$), 자료가 분명히 정규이면 $t$-검정을 택한다.
