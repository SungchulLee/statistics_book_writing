# 비율의 표본분포

## 개요

**표본비율 $\hat{p}$의 표본분포**는 이항 모집단에서 확률표본을 반복해서 뽑을 때 성공 비율이 어떻게 달라지는지를 기술한다. 모비율에 관한 추론의 토대이며, 여론조사, 품질관리, 임상시험, A/B 테스트가 모두 이에 의존한다.

## 수학적 정의

$X_1, \dots, X_n$을 i.i.d. $\text{Bernoulli}(p)$라 하자. 여기서 $X_i = 1$(성공) 또는 $X_i = 0$(실패)이다. 표본비율은:

$$
\hat{p} = \frac{1}{n}\sum_{i=1}^n X_i = \frac{\text{number of successes}}{n}
$$

## 성질

### 기댓값 (불편성)

$$
E[\hat{p}] = p
$$

표본비율은 모비율의 **불편추정량**이다.

### 분산과 표준오차

$\text{Var}(X_i) = p(1-p)$이므로:

$$
\text{Var}(\hat{p}) = \frac{p(1-p)}{n}, \qquad
\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}
$$

!!! note
    $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$과 달리 $\hat{p}$의 표준오차는 모수 $p$ 자체에 의존한다. 실무에서는 $p$를 모르므로 $\hat{p}$로 대체한다:

    $$
    \widehat{\text{SE}}(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
    $$

### 모양 (정규근사)

중심극한정리에 의해 $n$이 충분히 크면:

$$
\frac{\hat{p} - p}{\sqrt{p(1-p)/n}} \xrightarrow{d} N(0, 1)
$$

정규근사가 타당하기 위한 **경험 법칙**:

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

성공 횟수와 실패 횟수가 모두 종 모양 근사를 쓰기에 충분히 크도록 보장한다.

## 표준오차 계산

<div class="probox" markdown>

**문제.** <span class="diff easy" title="쉬움"></span> 참 비율이 $p = 0.4$이고 표본크기가 $n = 100$이다.

</div>

??? success "풀이"
    $$
    \text{SE}(\hat{p}) = \sqrt{\frac{0.4 \times 0.6}{100}} = \sqrt{0.0024} \approx 0.049
    $$

    크기 100인 표본을 반복해서 뽑으면 $\hat{p}$는 참값 $p = 0.4$ 주위로 대체로 0.049 정도 달라진다.

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 브랜드 선호. 어떤 모집단에서 60%가 브랜드 A를 선호한다. $n = 100$일 때 $P(\hat{p} > 0.65)$를 구하라.

</div>

??? success "풀이"

    $$
    \text{SE} = \sqrt{\frac{0.60 \times 0.40}{100}} \approx 0.049
    $$

    $$
    Z = \frac{0.65 - 0.60}{0.049} \approx 1.02
    $$

    $$
    P(\hat{p} > 0.65) = P(Z > 1.02) \approx 0.154
    $$

    ```python
    from scipy import stats
    print(f"P(p_hat > 0.65) = {stats.norm.sf(1.02):.4f}")
    ```

    출력:

    ```
    P(p_hat > 0.65) = 0.1539
    ```
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 소표본 — 정확값과 근사값. 어떤 도시에서 30%가 대중교통을 선호한다. $n = 10$일 때 $P(\hat{p} > 0.35)$를 구하라.

</div>

??? success "풀이"
    **정확한 이항 계산.** $\hat{p} > 0.35$는 $X \geq 4$를 뜻하며 여기서 $X \sim \text{Binomial}(10, 0.3)$이다:

    $$
    P(X \geq 4) = 1 - P(X \leq 3)
    $$

    $$
    P(X = 0) = 0.0282, \quad P(X = 1) = 0.1211, \quad P(X = 2) = 0.2335, \quad P(X = 3) = 0.2668
    $$

    $$
    P(X \geq 4) = 1 - 0.6496 = 0.3504
    $$

    **정규근사.** 조건을 확인하면 $np = 3 < 5$이므로 정규근사가 미덥지 않다.

    $$
    \text{SE} = \sqrt{\frac{0.3 \times 0.7}{10}} \approx 0.1449, \qquad
    Z = \frac{0.35 - 0.30}{0.1449} \approx 0.345
    $$

    $$
    P(\hat{p} > 0.35) \approx P(Z > 0.345) \approx 0.365
    $$

    **비교:**

    | 방법 | 결과 |
    |--------|--------|
    | 정확한 binomial | 0.3504 |
    | 정규근사 | 0.3650 |

    표본이 작은데도 근사가 꽤 가깝지만, $np < 5$일 때는 정확한 binomial 계산이 낫다.

    ```python
    from scipy import stats

    # 정확한 값: 이항분포에서 바로 구한다.
    exact = 1 - stats.binom(n=10, p=0.3).cdf(3)
    print(f"Exact: {exact:.4f}")

    # 정규근사로 구한 값. 둘을 견준다.
    approx = stats.norm.sf(0.345)
    print(f"Normal approx: {approx:.4f}")
    ```

    출력:

    ```
    Exact: 0.3504
    Normal approx: 0.3650
    ```
## 두 비율의 차

비율이 $p_1$과 $p_2$인 두 모집단에서 독립인 표본을 뽑으면:

$$
Z = \frac{(\hat{p}_1 - \hat{p}_2) - (p_1 - p_2)}{\sqrt{\frac{p_1(1-p_1)}{n_1} + \frac{p_2(1-p_2)}{n_2}}} \approx N(0, 1)
$$

**신뢰구간:**

$$
(\hat{p}_1 - \hat{p}_2) \pm z_{\alpha/2} \sqrt{\frac{\hat{p}_1(1-\hat{p}_1)}{n_1} + \frac{\hat{p}_2(1-\hat{p}_2)}{n_2}}
$$

## 모의실험: p-hat의 표본분포

<div class="codebox" markdown>

### 예제 1. 표본비율의 표집분포 모의실험 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

np.random.seed(1)

# 모집단은 0과 1 두 값뿐인 베르누이다. binom(n=1)이 곧 베르누이다.
population = stats.binom(n=1, p=0.4).rvs(100_000)
sample_size = 1_000
n_samples = 10_000

# 0/1 자료의 평균이 곧 비율이다. 그래서 p-hat 은 특별한 통계량이 아니라
# **표본평균의 한 경우**이며, 중심극한정리가 그대로 적용된다.
sample_proportions = [
    np.mean(np.random.choice(population, size=sample_size, replace=False))
    for _ in range(n_samples)
]

# 이 그림의 요점은 위아래의 **모양 차이**다.
# 모집단은 막대 두 개뿐인 가장 극단적인 비정규 분포인데,
# 표본비율의 표집분포는 매끄러운 종 모양이 된다.
fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 6))

# bins=3 인 이유: 값이 0과 1뿐이라 구간을 잘게 나눌 필요가 없다.
ax0.hist(population, bins=3, density=True, alpha=0.5)
ax0.set_title('Population Distribution (Bernoulli, p = 0.4)', fontsize=16)

ax1.hist(sample_proportions, bins=50, density=True, alpha=0.5)
ax1.set_title(rf'Sampling Distribution of $\hat{{p}}$ (n = {sample_size})', fontsize=16)

for ax in (ax0, ax1):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
```

</div>

## 대학원 수준의 보충

- 표본이 작거나 비율이 극단적일 때($p$가 0이나 1에 가까울 때)는 **이항분포**를 직접 사용해야 한다.
- **Wilson 구간**이 Wald 구간($\hat{p} \pm z^* \cdot \widehat{\text{SE}}$)보다 대체로 선호된다. 특히 $n$이 작거나 $p$가 극단적일 때 포함확률 성질이 더 좋다.
- **Agresti–Coull 구간**은 Wald 구간을 계산하기 전에 가상의 성공 2회와 실패 2회를 더하는 방식으로, 간단하면서도 포함확률을 개선한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
모집단의 $p = 0.60$이고 표본 $n = 100$이다. $P(\hat p > 0.65)$를 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n} = \sqrt{0.24/100} \approx 0.049$. $Z = (0.65 - 0.60)/0.049 \approx 1.02$.

    $P(\hat p > 0.65) = 1 - \Phi(1.02) \approx 0.154$로 약 15.4%이다.

    조건: $np = 60 \ge 10$이고 $n(1-p) = 40 \ge 10$이므로 정규근사가 타당하다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**소표본의 문제.** 모집단의 $p = 0.30$이고 표본 $n = 10$이다. $P(\hat p > 0.35)$를 정확한 방법과 정규근사로 각각 계산하라. 여기서 정규근사가 통하는지 아니면 실패하는지 이유를 설명하라.

</div>

??? success "풀이"
    정확한 계산: $\hat p > 0.35$ ⟺ $X \ge 4$이며 $X \sim \mathrm{Binomial}(10, 0.3)$이다.

    $P(X < 4) = P(X = 0,1,2,3) = 0.028 + 0.121 + 0.233 + 0.267 = 0.650$이므로 $P(X \ge 4) = 0.350$이다.

    정규근사: $\mathrm{SE} = \sqrt{0.21/10} \approx 0.145$. $Z = 0.05/0.145 \approx 0.345$. $P(Z > 0.345) \approx 0.365$.

    차이: 정확값 0.350 대 정규근사 0.365로 약 1.5퍼센트포인트 차이가 난다. ($p = 0.3$에서 binomial이 그리 심하게 치우쳐 있지 않아) 정규근사가 그런대로 통하지만, $np = 3$이 작아 통상적인 경험 법칙($np \ge 10$)을 위반한다. 정확도를 높이려면 연속성 수정을 적용하거나 정확한 binomial을 사용하라.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**$\hat p$의 불편성을 증명하고 표준오차를 구하라.** $\mathbb{E}[\hat p] = p$이고 $\mathrm{Var}(\hat p) = p(1-p)/n$임을 보여라.

</div>

??? success "풀이"
    $X_i \sim \mathrm{Bernoulli}(p)$가 i.i.d.일 때 $X = \sum X_i$에 대해 $\hat p = X/n$이다.

    $\mathbb{E}[\hat p] = \mathbb{E}[X]/n = np/n = p$로 불편이다.

    $\mathrm{Var}(\hat p) = \mathrm{Var}(X)/n^2 = np(1-p)/n^2 = p(1-p)/n$.

    $\mathrm{SE}(\hat p) = \sqrt{p(1-p)/n}$.

    $\square$

    표준오차는 $p = 1/2$에서 최대가 된다(최악의 경우). $p = 1/2$일 때 $\mathrm{SE} = 1/(2\sqrt n)$이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**표본크기 설계.** $p$를 모를 때 95% 신뢰수준에서 오차한계 $\pm 3$퍼센트포인트로 $p$를 추정하려면 표본크기가 얼마여야 하는가?

</div>

??? success "풀이"
    오차한계: $\mathrm{ME} = z_{0.975} \cdot \mathrm{SE} = 1.96 \sqrt{p(1-p)/n} \le 0.03$.

    최악의 경우인 $p = 1/2$에서 $\mathrm{SE} = 1/(2\sqrt n)$이므로 $1.96/(2\sqrt n) \le 0.03 \Rightarrow \sqrt n \ge 1.96/0.06 \approx 32.67 \Rightarrow n \ge 1068$이다.

    이것이 여론조사에서 "$n \approx 1000$" 규칙이 나온 배경이다. $n = 1000$이면 $p$가 무엇이든 95% 신뢰수준의 오차한계가 최대 $\pm 3.1$퍼센트포인트이다.

    $p$가 0.5에서 멀다고 짐작되면(가령 $p \approx 0.1$) $p(1-p)$가 0.25 대신 0.09가 되어 $n \approx 0.09 \cdot 1068/0.25 \approx 385$면 충분하다. $p$의 대략적인 값을 알면 필요한 $n$이 줄어든다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
**Wilson 점수 구간.** 이항 비율에서 표준적인 Wald 신뢰구간 $\hat p \pm z \sqrt{\hat p(1-\hat p)/n}$보다 **Wilson 점수** 신뢰구간이 선호되는 이유는 무엇인가?

</div>

??? success "풀이"
    **Wald 신뢰구간의 문제:**

    - 특히 $p = 0$이나 $p = 1$ 근처에서 포함확률이 비대칭이다.
    - 0 아래나 1 위로 뻗는 구간이 나올 수 있다(예: $\hat p = 0.05, n = 50$이면 CI = $0.05 \pm 0.06 = (-0.01, 0.11)$).
    - 포함확률이 $n$에 따라 크게 요동쳐 명목 수준 $1 - \alpha$에서 멀어진다.

    **Wilson 점수 구간:**

    $$
    p_{\mathrm{Wilson}} = \frac{\hat p + z^2/(2n) \pm z\sqrt{\hat p(1-\hat p)/n + z^2/(4n^2)}}{1 + z^2/n}
    $$

    표준오차에서 $p$를 $\hat p$로 대체하는 대신 검정을 역으로 풀어 $|p - \hat p| \le z\sqrt{p(1-p)/n}$을 $p$에 대해 해결한 것이다.

    **장점:** 항상 $[0, 1]$ 안에 머물고, 경계 근처에서 포함확률이 훨씬 좋으며, 현대적 관행에서 권장된다(R의 `prop.test`, Python의 `statsmodels.stats.proportion.proportion_confint(method="wilson")`).

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**비율의 차.** 독립인 두 표본에서 $n_1$로부터 $\hat p_1$을, $n_2$로부터 $\hat p_2$를 얻었다. $\hat p_1 - \hat p_2$의 표준오차를 유도하라.

</div>

??? success "풀이"
    독립성에 의해 $\mathrm{Var}(\hat p_1 - \hat p_2) = \mathrm{Var}(\hat p_1) + \mathrm{Var}(\hat p_2) = p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2$이다.

    $\mathrm{SE}(\hat p_1 - \hat p_2) = \sqrt{p_1(1-p_1)/n_1 + p_2(1-p_2)/n_2}$.

    추론에서는($H_0: p_1 = p_2 = p$를 검정할 때) 표준오차에 합동추정량 $\hat p_{\text{pool}} = (X_1 + X_2)/(n_1 + n_2)$를 대입한다.

    신뢰구간에서는(동일성을 가정하지 않고 $p_1 - p_2$를 추정할 때) $\hat p_1$과 $\hat p_2$를 각각 대입한다:

    $$
    \mathrm{CI}: (\hat p_1 - \hat p_2) \pm z\sqrt{\hat p_1(1-\hat p_1)/n_1 + \hat p_2(1-\hat p_2)/n_2}
    $$

    이것이 A/B 테스트와 임상시험에서 쓰는 두 비율 $z$ 검정과 신뢰구간의 바탕이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
A/B 테스트에서 A안은 250명 중 90명(36%), B안은 300명 중 84명(28%)이 전환했다. 차이의 95% 신뢰구간과 $z$ 검정을 수행하라. 검정에서 쓰는 표준오차와 신뢰구간에서 쓰는 표준오차가 왜 다른가?

</div>

??? success "풀이"
    $\hat p_1 = 0.36$, $\hat p_2 = 0.28$, 차이는 $0.08$이다.

    **신뢰구간.** 각 비율을 따로 추정한 표준오차를 쓴다.

    $$
    \operatorname{SE}_{\text{CI}} = \sqrt{\frac{0.36(0.64)}{250}+\frac{0.28(0.72)}{300}} = \sqrt{0.000922+0.000672} = 0.03992
    $$

    $$
    0.08 \pm 1.96\times0.03992 = 0.08\pm0.0782 = (0.0018,\ 0.1582)
    $$

    **검정.** $H_0: p_1=p_2$ 아래에서는 공통 비율이 있으므로 두 표본을 합쳐 추정한다.

    $$
    \hat p = \frac{90+84}{550} = 0.3164, \qquad \operatorname{SE}_0 = \sqrt{\hat p(1-\hat p)\left(\frac{1}{250}+\frac{1}{300}\right)} = 0.03983
    $$

    $$
    z = \frac{0.08}{0.03983} = 2.009, \qquad p\text{-값} = 0.0446
    $$

    5%에서 기각한다.

    **두 표준오차가 다른 이유.** 검정은 **귀무가설이 참이라는 전제 아래** 통계량의 분포를 구한다. $p_1=p_2=p$이므로 두 표본을 합친 $\hat p$가 그 공통값의 더 나은 추정값이고, 이를 쓰면 검정력이 높아진다.

    신뢰구간은 어떤 가설도 전제하지 않는다. $p_1$과 $p_2$가 다를 수 있다고 보고 각각 따로 추정해야 한다.

    두 표준오차가 비슷해서 여기서는 결론이 같지만, **언제나 그런 것은 아니다.** 특히 두 비율이 크게 다르면 신뢰구간이 0을 담지 않는데 검정은 기각하지 못하거나 그 반대인 경우가 생긴다. 보고할 때는 **신뢰구간을 우선하는 편이 낫다.** 효과의 크기와 불확실성을 함께 보여 주기 때문이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 7의 자료로 **위험비**(상대위험도)와 **오즈비**를 계산하고, 절대 차이를 포함한 세 척도가 각각 무엇을 말해 주는지 비교하라.

</div>

??? success "풀이"

    | 척도 | 정의 | 값 |
    |---|---|---|
    | 절대 차이 | $p_1-p_2$ | $0.36-0.28 = 0.08$ |
    | 위험비 | $p_1/p_2$ | $0.36/0.28 = 1.286$ |
    | 오즈비 | $\dfrac{p_1/(1-p_1)}{p_2/(1-p_2)}$ | $\dfrac{0.5625}{0.3889} = 1.446$ |

    **절대 차이(8%포인트)** 는 실무적 의사결정에 가장 직접적이다. 방문자 10,000명이면 전환이 800건 더 생긴다. 다만 기준선이 다른 상황끼리 옮겨 쓸 수 없다.

    **위험비(1.29배)** 는 "28%가 36%로 올랐다"는 상대적 개선을 말한다. 기준선이 달라도 대체로 안정적이라 여러 연구를 합칠 때 유용하다. 그러나 **상대적 크기만으로는 실질적 중요성을 판단할 수 없다.** 0.001%가 0.00129%가 되어도 위험비는 똑같이 1.29다.

    **오즈비(1.45배)** 는 절대 차이나 위험비보다 **언제나 더 극적으로 보인다**($p$가 0.5에서 멀지 않을 때). 오즈비를 위험비처럼 읽으면 효과를 과장하게 된다. 그럼에도 널리 쓰이는 이유는 두 가지다. 첫째, 로지스틱 회귀의 계수가 곧 로그 오즈비다. 둘째, 환자-대조군 연구에서는 위험을 추정할 수 없지만 오즈비는 추정할 수 있다.

    **보고 원칙.** 상대 척도 하나만 적는 것은 좋지 않은 관행이다. **기준선 위험과 절대 차이를 반드시 함께 적어야** 독자가 실질적 크기를 판단할 수 있다. "위험이 두 배로 늘었다"는 말이 0.001%에서 0.002%인지 20%에서 40%인지에 따라 뜻이 전혀 다르다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
다섯 개 지역의 지지율을 각각 95% 신뢰구간으로 보고하려 한다. 다섯 구간이 **동시에** 참값을 담을 확률은 얼마인가? 본페로니 보정을 적용하면 각 구간은 어떻게 달라지는가?

</div>

??? success "풀이"
    각 구간이 참값을 담을 확률이 0.95이고 다섯 지역이 독립이라면

    $$
    P(\text{모두 담음}) = 0.95^5 = 0.774
    $$

    로 77%에 지나지 않는다. 다섯 구간 중 적어도 하나가 빗나갈 확률이 23%다. 지역이 20곳이면 $0.95^{20} = 0.36$으로 떨어진다.

    **본페로니 보정.** 동시 포함확률을 0.95 이상으로 만들려면 각 구간의 유의수준을 $\alpha/k = 0.05/5 = 0.01$로 낮춘다. 임계값이

    $$
    z_{1-0.01/2} = z_{0.995} = 2.576
    $$

    으로 1.96에서 2.576으로 커지고, **각 구간의 폭이 31% 넓어진다.**

    근거는 본페로니 부등식이다. 각 구간이 빗나갈 확률이 $\alpha/k$ 이하면

    $$
    P(\text{적어도 하나 빗나감}) \le \sum_{i=1}^k \frac{\alpha}{k} = \alpha
    $$

    이다. 독립을 요구하지 않는다는 점이 이 방법의 큰 장점이다.

    **언제 보정해야 하는가.** 판단의 기준은 **결론을 어떻게 쓰느냐**다.

    - "다섯 지역 중 어디가 가장 높은가", "어느 한 곳이라도 50%를 넘는가"처럼 **전체를 아우르는 주장**을 하려면 보정해야 한다.
    - 각 지역을 독립적인 별개 연구로 보고하고 독자가 개별적으로 읽는다면 보정하지 않아도 된다.

    본페로니는 $k$가 크면 지나치게 보수적이다. 그때는 홀름 절차(단계적으로 완화), 튜키 HSD(모든 쌍 비교), 벤저미니-호크버그(거짓발견율 통제) 같은 대안을 쓴다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
전화 조사에서 1000명에게 연락해 200명이 응답했고 그중 110명(55%)이 찬성했다. 표준오차와 신뢰구간을 계산하되, 이 구간이 담지 못하는 오차가 무엇인지 논하라.

</div>

??? success "풀이"
    응답자 200명만 보면

    $$
    \hat p = 0.55, \quad \operatorname{SE} = \sqrt{\frac{0.55\times0.45}{200}} = 0.0352, \quad \text{95\% CI} = (0.481,\ 0.619)
    $$

    이다.

    **이 구간이 담는 것은 표집오차뿐이다.** 응답률이 20%이므로 800명의 의견을 전혀 모른다. 그 800명이 응답자와 다르다면 구간이 아무리 좁아도 참값을 담지 못한다.

    **무응답 편향의 크기.** 응답자 비율을 $p_R$, 무응답자 비율을 $p_M$, 응답률을 $r$이라 하면 참 비율은

    $$
    p = r\,p_R + (1-r)p_M
    $$

    이다. 편향은

    $$
    p_R - p = (1-r)(p_R-p_M)
    $$

    으로, **응답률이 낮을수록, 두 집단이 다를수록 커진다.** 여기서는 $1-r = 0.8$이므로 두 집단의 차이가 5%포인트만 되어도 편향이 4%포인트다. 이는 표집오차 한계 7%포인트와 맞먹는다.

    **결정적인 차이는 이것이다.** 표집오차는 표본을 늘리면 줄어들지만 **무응답 편향은 전혀 줄지 않는다.** 10,000명에게 연락해 2,000명이 응답하면 구간은 $\pm2.2$%포인트로 좁아지지만 편향 4%포인트는 그대로다. 표본을 늘릴수록 **틀린 답에 더 정밀하게 수렴**한다.

    **대처.** 응답률 자체를 높이는 것이 최선이다(재접촉, 유인 제공). 그다음은 인구통계 정보로 가중치를 조정하는 사후층화인데, 이는 "관측된 특성이 같으면 응답 성향도 같다"는 검증 불가능한 가정에 기댄다. 무응답자 일부를 집중적으로 추적 조사해 두 집단의 차이를 직접 추정하는 방법도 있다.

    **보고할 때는 반드시 응답률을 함께 적어야 한다.** 응답률 없는 조사 결과는 오차한계만 그럴듯한 숫자일 뿐이다.

---

## 정리하며

| 성질 | 결과 |
|----------|--------|
| $E[\hat{p}]$ | $p$ (불편) |
| $\text{Var}(\hat{p})$ | $p(1-p)/n$ |
| $\text{SE}(\hat{p})$ | $\sqrt{p(1-p)/n}$ |
| 정규근사가 타당한 조건 | $np \geq 5$이고 $n(1-p) \geq 5$ |
| $\bar{X}$와의 핵심 차이 | 표준오차가 모수 자체에 의존한다 |
| $n$이 작을 때 | 정규근사 대신 정확한 binomial을 사용 |

이어지는 세 쪽에서 이 결과를 모의실험으로 확인한다. $\hat p$는 베르누이 모집단의 $\bar X$이므로 중심극한정리가 그대로 적용되지만, 값이 $0, 1/n, 2/n, \ldots$ 로만 놓인다는 **이산성**이 남는다. 표본크기의 효과, $p$가 극단일 때의 붕괴, 그리고 신뢰구간의 실제 포함률을 차례로 본다.
