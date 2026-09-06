# 다중검정 보정

## 개요

여러 가설을 동시에 검정하면 거짓 양성이 적어도 하나 나올 확률이 빠르게 커진다. 다중검정 보정은 p-값이나 유의수준 문턱을 조정하여 **가족단위 오류율**(FWER)이나 **거짓발견율**(FDR) 같은 오류율을 통제한다. 유전체학, 뇌영상을 비롯해 수천 개의 검정을 한꺼번에 수행하는 어떤 상황에서든 이 보정을 이해하는 것이 필수적이다.

## 다중검정 문제

독립인 검정 $m$개를 각각 수준 $\alpha$에서 수행하면 제1종 오류가 적어도 한 번 생길 확률은

$$
\text{FWER} = 1 - (1 - \alpha)^m.
$$

$\alpha = 0.05$에서 $m = 20$개 검정이면:

$$
\text{FWER} = 1 - 0.95^{20} \approx 0.642.
$$

모든 귀무가설이 참이어도 거짓 양성이 적어도 하나 나올 확률이 약 64%라는 뜻이다.

## 흔한 보정

### Bonferroni 보정

$p_i < \alpha/m$이면 $H_{0,i}$를 기각한다. FWER을 수준 $\alpha$로 통제하지만 보수적이다.

### Holm–Bonferroni 방법

p-값을 $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$으로 정렬한다. 조건을 만족하는 가장 큰 지표를 $k$라 할 때, 모든 $i = 1, \dots, k$에 대해

$$
p_{(i)} < \frac{\alpha}{m - i + 1}
$$

이면 $H_{0,(i)}$를 기각한다. FWER을 여전히 통제하면서 Bonferroni보다 균일하게 더 강력하다.

### Benjamini–Hochberg (BH) 절차

FDR을 수준 $q$로 통제한다. p-값을 정렬하고 다음을 만족하는 가장 큰 $k$를 찾는다.

$$
p_{(k)} \leq \frac{k}{m}\,q.
$$

가설 $H_{0,(1)}, \dots, H_{0,(k)}$를 모두 기각한다.

## 코드

```python
import numpy as np
from scipy import stats

np.random.seed(42)

n = 100
data = np.random.normal(loc=0, scale=1, size=n)

print(f"Sample size: {n}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std:  {data.std(ddof=1):.4f}")
```

### `statsmodels`로 보정 적용하기

```python
from statsmodels.stats.multitest import multipletests

# Suppose we have m p-values from m independent tests
m = 50
p_values = np.random.uniform(0, 1, m)
# Inject some true signals
p_values[:5] = np.random.uniform(0, 0.005, 5)

# Bonferroni
_, p_bonf, _, _ = multipletests(p_values, method="bonferroni")

# Holm
_, p_holm, _, _ = multipletests(p_values, method="holm")

# Benjamini-Hochberg
_, p_bh, _, _ = multipletests(p_values, method="fdr_bh")

alpha = 0.05
print(f"Rejections (uncorrected): {np.sum(p_values < alpha)}")
print(f"Rejections (Bonferroni):  {np.sum(p_bonf < alpha)}")
print(f"Rejections (Holm):        {np.sum(p_holm < alpha)}")
print(f"Rejections (BH):          {np.sum(p_bh < alpha)}")
```

### 해석

- **Bonferroni**가 가장 보수적이다: FWER 통제를 보장하려고 적은 수의 가설만 기각한다.
- **Holm**은 단계적 하강 절차로 Bonferroni보다 균일하게 더 강력하다.
- **BH**는 기각 중 거짓 발견의 기대 비율(FDR)을 통제하며 셋 중 가장 관대한 경향이 있다. 거짓 발견을 어느 정도 허용하는 대가로 더 큰 검정력을 준다.

## 연습문제

**연습문제 1.** 어떤 연구자가 $\alpha = 0.05$에서 독립인 검정 $m = 100$개를 수행한다. 모든 귀무가설이 참이라면 거짓 양성이 적어도 하나 나올 확률은? Bonferroni 보정 문턱은 얼마여야 하는가?

??? success "연습문제 1 풀이"

    FWER은

    $$
    1 - (1-0.05)^{100} = 1 - 0.95^{100} \approx 1 - 0.00592 = 0.994.
    $$

    거짓 양성이 적어도 하나 나올 확률이 약 99.4%이다. Bonferroni 문턱은 $\alpha/m = 0.05/100 = 0.0005$이다. $\square$

---

**연습문제 2.** $m=5$개 검정에서 정렬된 p-값 $p_{(1)} = 0.001$, $p_{(2)} = 0.008$, $p_{(3)} = 0.039$, $p_{(4)} = 0.041$, $p_{(5)} = 0.23$이 주어졌다. FDR 수준 $q = 0.05$에서 BH 절차를 적용하라. 어느 가설이 기각되는가?

??? success "연습문제 2 풀이"

    BH 문턱 $k\,q/m$을 계산한다:

    | $k$ | $p_{(k)}$ | $k \cdot 0.05 / 5$ | $p_{(k)} \leq$ 문턱? |
    |-----|-----------|---------------------|---------------------------|
    | 1   | 0.001     | 0.01                | 예                       |
    | 2   | 0.008     | 0.02                | 예                       |
    | 3   | 0.039     | 0.03                | 아니오                        |
    | 4   | 0.041     | 0.04                | 아니오                        |
    | 5   | 0.23      | 0.05                | 아니오                        |

    $p_{(k)} \leq k\,q/m$을 만족하는 가장 큰 $k$는 $k=2$이다. $H_{0,(1)}$과 $H_{0,(2)}$를 기각한다. $\square$

---

**연습문제 3.** Bonferroni 보정이 FWER을 수준 $\alpha$로 통제함을 증명하라. 즉 각 검정을 수준 $\alpha/m$에서 수행하면 $P(\text{잘못된 기각이 적어도 하나}) \leq \alpha$임을 보여라.

??? success "연습문제 3 풀이"

    $V_i$를 $i$번째 참인 귀무가설이 잘못 기각되는 사건이라 하자. 합집합 한계(Boole 부등식)에 의해

    $$
    P\!\left(\bigcup_{i=1}^{m_0} V_i\right) \leq \sum_{i=1}^{m_0} P(V_i) \leq \sum_{i=1}^{m_0} \frac{\alpha}{m} = \frac{m_0\,\alpha}{m} \leq \alpha,
    $$

    여기서 $m_0 \leq m$은 참인 귀무가설의 개수이다. 이 부등식은 검정 사이의 종속성과 무관하게 성립한다. $\square$

---

**연습문제 4.** $\alpha=0.05$에서 p-값 $p_1=0.01$, $p_2=0.04$, $p_3=0.03$, $p_4=0.005$에 Holm 절차를 적용하라. 어느 가설이 기각되는가?

??? success "연습문제 4 풀이"

    p-값을 정렬한다: $p_{(1)}=0.005$, $p_{(2)}=0.01$, $p_{(3)}=0.03$, $p_{(4)}=0.04$.

    1단계: $p_{(1)} = 0.005$를 $\alpha/(m-1+1) = 0.05/4 = 0.0125$와 비교한다. $0.005 < 0.0125$이므로 $H_{0,(1)}$을 기각한다.

    2단계: $p_{(2)} = 0.01$을 $0.05/3 \approx 0.0167$과 비교한다. $0.01 < 0.0167$이므로 $H_{0,(2)}$를 기각한다.

    3단계: $p_{(3)} = 0.03$을 $0.05/2 = 0.025$와 비교한다. $0.03 > 0.025$이므로 멈춘다.

    $H_{0,(1)}$과 $H_{0,(2)}$($p_4=0.005$와 $p_1=0.01$에 해당)를 기각한다. $H_{0,(3)}$과 $H_{0,(4)}$는 기각하지 못한다. $\square$

---

**연습문제 5.** BH 절차가 Bonferroni보다 강력한 이유를 직관적으로 설명하라. BH 절차가 명목 수준에서 FDR을 통제하지 못하는 조건은 무엇인가?

??? success "연습문제 5 풀이"

    **BH가 더 강력한 이유:** Bonferroni는 모든 검정에 고정된 문턱 $\alpha/m$을 쓰는데 $m$이 커지면 이 값이 극도로 작아진다. BH는 순위에 따라 커지는 적응적 문턱 $k\alpha/m$을 쓰므로 정렬된 목록의 중간에 있는 p-값이 덜 엄격한 장벽을 마주한다. 참 신호가 많으면(그 p-값들이 0 근처에 몰려 있으면) BH의 단계적 상승 방식이 그것들을 포착하지만 Bonferroni의 균일한 문턱은 놓친다.

    **BH가 실패할 수 있는 경우:** BH 절차는 독립성 또는 부분집합 각각에 대한 양의 회귀 종속성(PRDS) 아래에서 FDR을 수준 $q$로 통제함이 증명되어 있다. 검정통계량 사이에 강한 음의 종속성이 있으면 실제 FDR이 명목 수준 $q$를 넘을 수 있다. 이런 경우에는 $q$를 $q / \sum_{i=1}^m 1/i$로 바꾸는 Benjamini–Yekutieli(BY) 보정을 대신 써야 한다. $\square$
