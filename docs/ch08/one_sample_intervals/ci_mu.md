# μ의 신뢰구간

## 일표본 z 신뢰구간

경영, 의료, 교육을 비롯한 현실의 여러 응용에서 확률표본으로 모평균 $\mu$를 추정하는 일은 흔히 필수적이다. 모분산을 알고 있으면 표준정규분포를 이용해 신뢰구간을 구성할 수 있다.

### 공식

모분산 $\sigma^2$을 알고 있다면, 크기 $n$인 표본에 기반한 모평균 $\mu$의 신뢰구간은

$$
\bar{X} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}
$$

여기서

- $\bar{X}$는 표본평균,
- $\alpha$는 유의수준($\text{유의수준} = 1 - \text{신뢰수준}$),
- $z_{\alpha/2}$는 $P(Z > z_{\alpha/2}) = \alpha/2$를 만족하는 표준정규분포의 임계값,
- $\sigma$는 알려진 모표준편차,
- $n$은 표본크기이다.

양 $\sigma / \sqrt{n}$은 표본평균의 표준오차이다.

### 타당성 조건

$$
\bar{x}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \text{이 크다, 예: } n \ge 30, \text{ 그래야 중심극한정리 근사가 통한다} \\
n \text{이 } N \text{에 비해 작다, 예: } n \le 0.1N, \text{ 그래야 i.i.d. 근사가 통한다}
\end{cases}
$$

$\sigma$를 모르고 $n$이 클 때 $\sigma$ 자리에 표본표준편차 $s$를 쓰는 것은 대수의법칙으로 정당화된다:

$$
\bar{x}\pm z_{\alpha/2}\frac{s}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \ge 30 \text{ (중심극한정리)} \\
n \ge 30 \text{ (} s \approx \sigma \text{를 위한 대수의법칙)} \\
n \le 0.1N \text{ (i.i.d.)}
\end{cases}
$$

<div class="codebox" markdown>

**예제 1.** 일표본 z 신뢰구간 계산

```python
import scipy.stats as stats
import numpy as np

# 주어진 자료
n = 40
sample_mean = 85
sigma = 12          # 알고 있는 모표준편차
confidence_level = 0.95

# 임계값. 0.95가 아니라 1 - 0.05/2 = 0.975를 넣는다.
# 양쪽 꼬리에 alpha/2씩 나눠 주기 때문이다.
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# 표준오차는 자료의 산포가 아니라 **표본평균**의 산포다. sqrt(n)으로 나눈다.
standard_error = sigma / np.sqrt(n)
margin_of_error = z_critical * standard_error

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence_interval = }")
```

출력:

```
confidence_interval = (81.28122980617263, 88.71877019382737)
```

</div>

---

## 일표본 t 신뢰구간

실무에서는 모분산 $\sigma^2$을 모르는 경우가 많다. 이때는 표본분산 $s^2$으로 분산을 추정하는데, 이 과정에서 추가적인 불확실성이 생긴다. 이를 반영하기 위해 정규분포 대신 $t$-분포를 쓴다.

### 공식

모분산을 모를 때 모평균 $\mu$의 신뢰구간은

$$
\bar{X} \pm t_{\alpha/2, \, n-1} \times \frac{s}{\sqrt{n}}
$$

여기서

- $\bar{X}$는 표본평균,
- $\alpha$는 유의수준($\text{유의수준} = 1 - \text{신뢰수준}$),
- $n - 1$은 $t$-분포의 자유도,
- $t_{\alpha/2, \, n-1}$은 $P(T > t_{\alpha/2, \, n-1}) = \alpha/2$를 만족하는 $t$-분포의 임계값,
- $s$는 표본표준편차,
- $n$은 표본크기이다.

### 타당성 조건

$$
\bar{x}\pm t_{\alpha/2,n-1}\frac{s}{\sqrt{n}}
\quad\text{if}\quad
\begin{cases}
n \text{이 작다, 예: } n < 30, \text{ 그래서 중심극한정리 근사가 통하지 않는다} \\
\text{모집단 분포가 정규이다, 그래서 표본분포를 정확히 안다} \\
n \le 0.1N \text{ (i.i.d.)}
\end{cases}
$$

<div class="codebox" markdown>

**예제 2.** 일표본 t 신뢰구간 계산

```python
import scipy.stats as stats
import numpy as np

# 주어진 자료
n = 25
sample_mean = 50
sample_std = 8      # 모표준편차가 아니라 자료에서 얻은 표본표준편차
confidence_level = 0.95

# 앞의 z-구간과 달라지는 곳은 여기 한 줄뿐이다.
# 자유도가 n-1인 것은 편차를 참 평균이 아니라 x_bar에서 쟀기 때문이다.
degrees_of_freedom = n - 1
t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)

standard_error = sample_std / np.sqrt(n)
margin_of_error = t_critical * standard_error

confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print(f"{confidence_interval = }")
```

출력:

```
confidence_interval = (46.697762301395166, 53.302237698604834)
```

</div>

$t_{0.025,\,24} = 2.0639$로 $z_{0.025} = 1.9600$보다 5.3% 크다. 같은 $\bar x$와 같은 산포에서 구간이 그만큼 넓어지며, 이것이 $\sigma$를 모른다는 사실의 값이다.

---

## 보기

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 모평균의 95% 신뢰구간 (분산을 아는 경우). 어떤 모집단에서 크기 $n = 40$인 확률표본을 모았다고 하자. 표본평균은 $\bar{X} = 85$이고 알려진 모표준편차는 $\sigma = 12$이다. 모평균 $\mu$의 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    95% 신뢰수준에서 임계값 $z_{\alpha/2}$는 약 1.96이다. 대입하면:

    $$
    85 \pm 1.96 \times \frac{12}{\sqrt{40}}
    $$

    표준오차: $\text{SE} = 12 / \sqrt{40} \approx 1.8974$. 오차한계: $1.96 \times 1.8974 \approx 3.717$.

    $$
    \boxed{(81.283,\ 88.717)}
    $$

    참 모평균 $\mu$가 $(81.283, 88.717)$ 안에 있다고 95% 신뢰한다.
<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 모평균의 95% 신뢰구간 (큰 표본). 성인 남성 100명의 확률표본에서 평균 키 175 cm, 표준편차 6 cm를 얻었다. 모평균 키의 95% 신뢰구간을 계산하라.

</div>

??? success "풀이"
    $n = 100 \ge 30$이므로 표본표준편차와 함께 표준정규분포를 쓴다.

    $$
    \text{SE} = \frac{6}{\sqrt{100}} = 0.6, \qquad \text{ME} = 1.96 \times 0.6 = 1.176
    $$

    $$
    \boxed{(173.82,\ 176.18) \text{ cm}}
    $$

    ```python
    import numpy as np
    import scipy.stats as stats

    n = 100
    x_bar = 175
    s = 6
    confidence_level = 0.95
    alpha = 1 - confidence_level

    # n = 100 >= 30 이므로 s를 sigma처럼 쓰고 z를 쓴다.
    # 이 표본크기에서 t를 써도 임계값이 1.984로 1.960과 1% 남짓 차이다.
    z_star = stats.norm().ppf(1 - alpha / 2)
    standard_error = s / np.sqrt(n)
    margin_of_error = z_star * standard_error

    print(f"{confidence_level:.0%} confidence interval: {x_bar} ± {margin_of_error:.2f}")
    ```

    출력:

    ```
    95% confidence interval: 175 ± 1.18
    ```
<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 표본크기의 결정 (천문학자). 한 천문학자가 멀리 있는 별까지의 거리를 측정한다. 측정값은 i.i.d.이고 평균이 $d$(실제 거리), 분산이 4 광년이다. 추정값이 95% 신뢰수준에서 $\pm 0.5$ 광년 이내로 정확하려면 측정을 몇 번 해야 하는가?

</div>

??? success "풀이"
    다음이 필요하다.

    $$
    1.96 \sqrt{\frac{4}{n}} \leq 0.5
    $$

    $n$에 대해 풀면:

    $$
    n \geq \frac{4 \times 1.96^2}{0.5^2} = 61.4656
    $$

    $n$은 정수여야 하므로 적어도 **62번의 측정**이 필요하다.
<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 모평균의 95% 신뢰구간 (분산을 모르는 작은 표본). 크기 $n = 25$인 확률표본에서 $\bar{X} = 50$, $s = 8$을 얻었다. $\mu$의 95% 신뢰구간을 구성하라.

</div>

??? success "풀이"
    자유도 $df = 24$, 95% 신뢰수준에서 $t_{\alpha/2, 24} \approx 2.064$이다.

    $$
    \text{SE} = \frac{8}{\sqrt{25}} = 1.6, \qquad \text{ME} = 2.064 \times 1.6 \approx 3.302
    $$

    $$
    \boxed{(46.698,\ 53.302)}
    $$
<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> t*의 계산. 관측값이 $n = 15$개일 때 98% 신뢰구간의 임계값 $t_*$는 얼마인가?

</div>

??? success "풀이"

    ```python
    import scipy.stats as stats

    confidence_level = 0.98
    alpha = 1 - confidence_level
    n = 15
    df = n - 1

    # 98% 구간이므로 한쪽 꼬리에 1%씩 남긴다. 즉 왼쪽 누적확률 0.99 지점.
    t_star = stats.t(df=df).ppf(1 - alpha / 2)
    print(f"{t_star = :.4f}")
    ```

    출력:

    ```
    t_star = 2.6245
    ```

    같은 98%라도 정규분포라면 $z = 2.3263$이다. 자유도가 14밖에 안 되어 꼬리가 두껍기 때문에 임계값이 13% 커졌다.
<div class="exbox" markdown>

**보기 6.** <span class="diff easy" title="쉬움"></span> 도장 두께. Felix는 자동차 부품에서 무작위로 50개 지점을 골라 도막 두께를 측정했다. 표본에서 $\bar{x} = 148$ 마이크론, $s = 3.3$ 마이크론을 얻었고 95% 신뢰구간 $(147.1, 148.9)$ 마이크론을 구성했다. 평균 두께가 목표값 150 마이크론과 일치한다고 보는 것이 그럴듯한가?

</div>

??? success "풀이"
    아니다. 신뢰구간 $(147.1, 148.9)$가 목표 두께 150 마이크론을 포함하지 않기 때문이다. 자료는 평균 두께가 목표에서 유의하게 벗어난다는 증거를 준다.
---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
병 50개에서 $\bar X = 503$ mL, $s = 5$ mL이다. $\mu$의 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $n$이 크므로 $z$를 쓴다: $\mathrm{SE} = 5/\sqrt{50} \approx 0.707$. 오차한계 = $1.96 \cdot 0.707 \approx 1.39$.

    신뢰구간: $(501.6, 504.4)$ mL.

    목표값 500 mL가 신뢰구간 밖이다 — 이 배치의 평균이 목표를 넘는다는 증거이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
**표본크기 계획.** 목표 오차한계 10 km, $\sigma = 15$ km, 신뢰수준 90%. 필요한 $n$은?

</div>

??? success "풀이"
    $z_{0.05} = 1.645$. $n = (1.645 \cdot 15/10)^2 = (2.47)^2 \approx 6.09$. 올림하면 $n = 7$.

    아주 작아서 실행 가능하다. 신뢰수준을 높이면(95%) $z = 1.96$이고 $n \approx 9$이다. $\sigma$가 더 크면(예: 30) $n \approx 25$이다.

    표본크기는 $\sigma^2$과 $z^2$에 비례한다 — 둘 다에 대해 이차이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
**작은 표본에서의 $t$-구간.** 병 $n = 25$개에서 $\bar X = 500$ mL, $s = 10$ mL이다. 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\sigma$를 모르므로 $t$를 쓴다: $t_{0.025, 24} = 2.064$. 오차한계 = $2.064 \cdot 10/\sqrt{25} = 4.13$.

    신뢰구간: $(495.87, 504.13)$.

    $t_{0.025, 24} > z_{0.025}$임에 유의하라 — 같은 신뢰수준에서 $t$ 구간이 $z$ 구간보다 넓은 것은 $s$의 불확실성을 반영하기 때문이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
**$\sigma$를 아는 시험점수.** $n = 36$, $\bar X = 78$, $\sigma = 12$. 95% 신뢰구간을 구하라.

</div>

??? success "풀이"
    $\sigma$를 알므로 $z$를 쓴다: 오차한계 = $1.96 \cdot 12/\sqrt{36} = 1.96 \cdot 2 = 3.92$.

    신뢰구간: $(74.08, 81.92)$.

    $\sigma$를 알면 표준오차 공식 외에는 표본크기가 $z$ 구간의 형태를 바꾸지 않는다. 가장 깔끔한 상황이지만 현실적인 경우는 드물다(보통은 $\sigma$를 모른다).

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**신뢰수준의 맞바꿈.** 90% 신뢰구간이 $(72, 78)$이다. (a) 해석. (b) 99% 신뢰구간이 더 넓은 이유.

</div>

??? success "풀이"
    (a) 올바른 해석: "이 표본추출과 신뢰구간 구성을 여러 번 반복하면 그 결과 구간들 중 약 90%가 참 평균을 담는다." "$\mu \in (72, 78)$일 확률이 90%이다"가 아니다 — 모수는 고정되어 있다.

    (b) 99% 신뢰구간은 $z_{0.005} = 2.576$을 쓰고 90% 신뢰구간은 $z_{0.05} = 1.645$를 쓴다. 너비의 비: $2.576/1.645 \approx 1.57$. 99% 신뢰구간이 57% 더 넓다.

    더 높은 확신을 위해서는 더 큰 변동을 감당해야 하므로 구간이 넓어진다. 신뢰수준이 낮으면 구간이 조이지만 덜 믿을 만하다. 정밀도와 확신 사이의 맞바꿈이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff easy" title="쉬움"></span>
**전구.** $n = 25$, $\bar X = 1200$시간, $s = 150$시간. (a) 95% 신뢰구간. (b) 99% 신뢰구간. (c) 오차한계 20시간을 위한 $n$.

</div>

??? success "풀이"
    (a) $t_{0.025, 24} = 2.064$. 오차한계 = $2.064 \cdot 150/5 = 61.9$. 신뢰구간: $(1138.1, 1261.9)$.

    (b) $t_{0.005, 24} = 2.797$. 오차한계 = $2.797 \cdot 30 = 83.9$. 신뢰구간: $(1116.1, 1283.9)$. 예상대로 더 넓다.

    (c) ($n$이 커질 것이므로) $z$ 근사를 쓰면 $n = (1.96 \cdot 150/20)^2 = (14.7)^2 \approx 216.1$. 올림하면 $n = 217$.

    오차한계 20을 달성하려면 원래의 $n = 25$보다 거의 9배 많은 표본이 필요하다. 오차한계를 약 62에서 20으로 줄이려면 자료가 $217/25 \approx 8.7$배 필요하다 — 오차한계 감소에 대해 이차이다.

---

## 정리하며

$\sigma$ 를 알 때 $\mu$ 의 신뢰구간은 $z$ 임계값을 쓴다.

$$
\bar X \pm z_{\alpha/2}\cdot\frac{\sigma}{\sqrt n}
$$

- **$z_{0.025}=1.96$** 이 가장 자주 쓰이는 값이다. $0.95$ 가 아니라 $0.975$ 의 분위수라는 점이 요령이며, 양쪽 꼬리에 $2.5\%$ 씩 남기기 때문이다.
- **폭이 $2z_{\alpha/2}\sigma/\sqrt n$ 이고 자료에 의존하지 않는다.** $\sigma$ 를 알고 있으므로 표본을 보기 전에 구간의 폭이 정해진다. 표본크기 설계가 가능한 이유다.
- **$\sigma$ 를 아는 경우는 실무에서 드물다.** 이 구간의 가치는 주로 개념적이며, 다음 단계인 $t$ 구간의 출발점이 된다.
- **정규성이 필요한가.** 모집단이 정규면 모든 $n$ 에서 정확하고, 아니면 중심극한정리에 기대는 근사다. 3장에서 보았듯 치우친 모집단에서는 $n\ge30$ 으로도 부족할 수 있다.

다음 절 **$p$ 의 신뢰구간**으로 넘어간다. 비율에서는 표준오차가 추정하려는 모수 자체에 의존해 새로운 문제가 생긴다.
