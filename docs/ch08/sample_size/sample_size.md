# 표본크기 계산

## 개요

자료를 모으기 전에 적절한 표본크기를 고르는 일은 연구 설계에서 가장 중요한 단계에 속한다. 표본이 너무 작으면 신뢰구간이 넓고 검정력이 낮으며, 불필요하게 크면 자원을 낭비한다. 이 페이지에서는 평균이나 비율을 추정할 때 $n$을 정하는 핵심 공식을 제시하고 Python 코드로 예시한다.

## 평균 추정을 위한 표본크기

신뢰수준 $1 - \alpha$에서 오차한계 $E$ 이내로 모평균 $\mu$를 추정하려면 최소 표본크기는

$$
n = \left\lceil \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2 \right\rceil
$$

여기서 $\sigma$는 모표준편차(또는 계획용 추정값)이고 $\lceil \cdot \rceil$은 천장함수이다.

### 유도

$z$-구간의 오차한계 식

$$
E = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

에서 출발해 $n$에 대해 풀면

$$
\sqrt{n} = \frac{z_{\alpha/2} \cdot \sigma}{E}
\quad\Longrightarrow\quad
n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2
$$

$n$은 정수여야 하므로 올림한다.

## 비율 추정을 위한 표본크기

오차한계 $E$ 이내로 모비율 $p$를 추정할 때 보수적인 공식은($p(1-p)$를 최대화하는 $p = 0.5$를 쓴다)

$$
n = \left\lceil \frac{z_{\alpha/2}^2}{4E^2} \right\rceil
$$

계획용 추정값 $p_0$이 있다면 더 조인 공식은

$$
n = \left\lceil \frac{z_{\alpha/2}^2 \, p_0(1 - p_0)}{E^2} \right\rceil
$$

## Python 코드

```python
import numpy as np
from scipy import stats

np.random.seed(42)

# --- 평균 추정을 위한 표본크기 ---
# 자료를 모으기 전이라 s가 없으므로 t가 아니라 z를 쓴다.
# sigma도 모르니 계획용 추정값을 넣는다. 이 값이 틀리면 계산 전체가 틀어지므로
# 표본크기 계산에서 가장 약한 고리는 언제나 sigma의 사전 추정이다.
sigma_est = 15
for E in [1, 2, 3, 5]:
    for conf in [0.90, 0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z * sigma_est / E) ** 2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")

# --- 비율 추정을 위한 표본크기 (보수적으로 p=0.5) ---
# p(1-p)가 p=0.5에서 최대 0.25이므로 sqrt(0.25) = 1/2이 되어
# 공식에 2E가 나타난다. 참 p가 무엇이든 안전한 크기다.
print("\nSample size for proportion (conservative p=0.5):")
for E in [0.01, 0.03, 0.05]:
    for conf in [0.95, 0.99]:
        z = stats.norm.ppf(1 - (1 - conf) / 2)
        n_needed = int(np.ceil((z / (2 * E)) ** 2))
        print(f"  E=+/-{E}, {conf*100:.0f}% conf -> n = {n_needed}")
```

출력:

```
  E=+/-1, 90% conf -> n = 609
  E=+/-1, 95% conf -> n = 865
  E=+/-1, 99% conf -> n = 1493
  E=+/-2, 90% conf -> n = 153
  E=+/-2, 95% conf -> n = 217
  E=+/-2, 99% conf -> n = 374
  E=+/-3, 90% conf -> n = 68
  E=+/-3, 95% conf -> n = 97
  E=+/-3, 99% conf -> n = 166
  E=+/-5, 90% conf -> n = 25
  E=+/-5, 95% conf -> n = 35
  E=+/-5, 99% conf -> n = 60

Sample size for proportion (conservative p=0.5):
  E=+/-0.01, 95% conf -> n = 9604
  E=+/-0.01, 99% conf -> n = 16588
  E=+/-0.03, 95% conf -> n = 1068
  E=+/-0.03, 99% conf -> n = 1844
  E=+/-0.05, 95% conf -> n = 385
  E=+/-0.05, 99% conf -> n = 664
```

비율 쪽 표를 보면 여론조사가 왜 대개 표본 1,000명 남짓인지 알 수 있다. 95% 신뢰수준에서 오차한계 $\pm 3$%p를 맞추는 데 1,068명이 필요하고, 이를 $\pm 1$%p로 조이려면 9,604명이 필요하다. 아홉 배의 비용으로 정밀도를 세 배 얻는 셈이다.

## 해석

- 필요한 표본크기는 임계값 $z_{\alpha/2}$와 모집단 산포 $\sigma$의 **제곱**에 비례해 커지고, 원하는 오차한계 $E$의 제곱에 반비례해 작아진다. 오차한계를 절반으로 줄이면 필요한 $n$이 네 배가 된다.
- 비율에서 보수적인 선택 $p = 0.5$는 참 $p$가 무엇이든 충분한 정밀도를 보장하지만 표본이 커질 수 있다는 대가를 치른다.
- 실무에서 $\sigma$를 정확히 아는 경우는 드물다. 흔한 전략은 예비조사, 선행 연구, 또는 범위 규칙 $\sigma \approx \text{범위}/4$를 쓰는 것이다.

## 연습문제

**연습문제 1.** 어떤 엔지니어가 케이블의 평균 인장강도를 95% 신뢰수준에서 $\pm 5$ MPa 이내로 추정해야 한다. 예비조사에서 $\sigma \approx 20$ MPa로 나타났다. 필요한 표본크기는?

??? success "풀이"

    임계값은 $z_{0.025} = 1.96$이다. 공식을 적용하면

    $$
    n = \left\lceil \left(\frac{1.96 \times 20}{5}\right)^2 \right\rceil
      = \left\lceil (7.84)^2 \right\rceil
      = \left\lceil 61.47 \right\rceil
      = 62
    $$

    적어도 관측값 62개가 필요하다. $\square$

---

**연습문제 2.** 어떤 조사자가 99% 신뢰수준에서 후보 지지율을 $\pm 3$ 퍼센트포인트 이내로 추정하려 한다. 보수적인 접근으로 필요한 표본크기를 구하라.

??? success "풀이"

    $E = 0.03$이고 $z_{0.005} = 2.576$이면,

    $$
    n = \left\lceil \frac{(2.576)^2}{4 \times (0.03)^2} \right\rceil
      = \left\lceil \frac{6.6358}{0.0036} \right\rceil
      = \left\lceil 1843.3 \right\rceil
      = 1844
    $$

    조사자에게 적어도 응답자 1844명이 필요하다. $\square$

---

**연습문제 3.** 모든 $p \in [0,1]$에 대해 $p(1-p) \le 1/4$임을 보이고, 비율에서 $p = 0.5$가 가장 보수적인(가장 큰) 표본크기를 주는 이유를 설명하라.

??? success "풀이"

    $f(p) = p(1-p) = p - p^2$이라 하자. 미분하면 $f'(p) = 1 - 2p$이고 $p = 1/2$에서 0이 된다. $f''(p) = -2 < 0$이므로 $p = 1/2$은 $[0,1]$에서의 전역 최댓값이며

    $$
    f(1/2) = \frac{1}{2}\left(1 - \frac{1}{2}\right) = \frac{1}{4}
    $$

    표본크기 공식은 $n = z_{\alpha/2}^2 \, p(1-p) / E^2$이다. $p(1-p)$가 분자에 있으므로 $p = 1/2$에서의 최댓값이 가장 큰 $n$을 주며, 참 $p$가 무엇이든 충분한 정밀도를 보장한다. $\square$

---

**연습문제 4.** 처음에는 오차한계 $\pm 4$로 계획했다가 나중에 $\pm 2$가 필요하다고 결정했다면 필요한 표본크기는 몇 배가 되는가? 이 관계를 일반적으로 증명하라.

??? success "풀이"

    표본크기 공식은 $n = (z_{\alpha/2} \sigma / E)^2$이다. 오차한계 $E_1$에 대응하는 것을 $n_1$, $E_2$에 대응하는 것을 $n_2$라 하자. 그러면

    $$
    \frac{n_2}{n_1} = \frac{(z_{\alpha/2} \sigma / E_2)^2}{(z_{\alpha/2} \sigma / E_1)^2} = \left(\frac{E_1}{E_2}\right)^2
    $$

    오차한계를 절반으로 하면($E_2 = E_1/2$) $n_2 / n_1 = 4$이다. 필요한 표본크기가 **네 배**가 된다. $E_1 = 4, E_2 = 2$인 구체적인 경우:

    $$
    \frac{n_2}{n_1} = \left(\frac{4}{2}\right)^2 = 4
    $$

    $\square$

---

**연습문제 5.** 어떤 연구자가 계획용 추정값 $p_0 = 0.30$을 갖고 있으며 오차한계 $\pm 0.04$인 $p$의 95% 신뢰구간을 원한다. 계획용 추정값 공식과 보수적인 공식의 표본크기를 비교하라. 관측값을 몇 개나 아끼는가?

??? success "풀이"

    $z_{0.025} = 1.96$, $E = 0.04$일 때:

    **보수적인 경우** ($p = 0.5$):

    $$
    n_{\text{cons}} = \left\lceil \frac{(1.96)^2}{4(0.04)^2} \right\rceil = \left\lceil \frac{3.8416}{0.0064} \right\rceil = \left\lceil 600.25 \right\rceil = 601
    $$

    **계획용 추정값** ($p_0 = 0.30$):

    $$
    n_{\text{plan}} = \left\lceil \frac{(1.96)^2 \times 0.30 \times 0.70}{(0.04)^2} \right\rceil = \left\lceil \frac{3.8416 \times 0.21}{0.0016} \right\rceil = \left\lceil \frac{0.8067}{0.0016} \right\rceil = \left\lceil 504.2 \right\rceil = 505
    $$

    계획용 추정값을 쓰면 관측값 $601 - 505 = 96$개를 아낀다. $\square$
