# 정규분포의 백분위점 함수 (분위수 함수)

## 개요

**백분위점 함수**(PPF)는 **분위수 함수** 또는 **역 CDF**라고도 하며, 다음 물음에 답한다. 누적확률 $q$가 주어졌을 때 $P(X \le x) = q$를 만족하는 값 $x$는 무엇인가?

$$
\text{ppf}(q) = F^{-1}(q) = \inf\{x : F(x) \ge q\}
$$

표준정규분포에서 가장 중요한 분위수는 $\mathcal{N}^{-1}(0.975) \approx 1.96$이며, 양측 95% 신뢰구간의 임계값이다.

---

## 코드

<div class="codebox" markdown>

### 예제 1. 백분위점 함수로 분위수 구하기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

mu, sigma = 0, 1
prob = 0.975      # 95% 신뢰구간의 한쪽 끝. 양쪽 꼬리에 2.5%씩 남긴다.

dist = stats.norm(loc=mu, scale=sigma)
x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 1000)
pdf = dist.pdf(x)

# ppf는 CDF의 역함수다. "누적확률이 이만큼 되는 지점은 어디인가"에 답한다.
#   cdf: 값 -> 확률
#   ppf: 확률 -> 값
# ppf(0.975)가 그 유명한 1.96 이며, 신뢰구간 공식의 z값이 여기서 나온다.
z = dist.ppf(prob)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(x, pdf, color='b', lw=2, label='PDF')
ax.plot([z, z], [0, dist.pdf(z)], color='k', lw=3)   # 경계선
# 왼쪽 97.5%를 칠한다. 칠해진 넓이가 곧 확률이라는 점이 요점이다.
ax.fill_between(x[x <= z], pdf[x <= z], 0,
                interpolate=True, color='r', alpha=0.25,
                label=f"P(X ≤ {z:.2f}) = {prob}")
ax.text(z + 0.05, dist.pdf(z) / 2,
        f"ppf({prob}) = {z:.4f}", fontsize=11, va='center')
ax.set_title(f"Normal({mu}, {sigma}) — PPF (Quantile Function)")
ax.legend(loc='upper left', frameon=False)
plt.tight_layout()
plt.show()
```

![정규분포의 백분위점 함수 (분위수 함수)](./img/normal_ppf_17.png)

</div>

---

## 표준정규분포의 흔한 분위수

| $q$ | $\mathcal{N}^{-1}(q)$ | 용도 |
|---|---|---|
| 0.500 | 0 | 중앙값 |
| 0.900 | 1.282 | 단측 90% 신뢰구간 |
| 0.950 | 1.645 | 단측 95% 신뢰구간 |
| 0.975 | 1.960 | 양측 95% 신뢰구간 |
| 0.995 | 2.576 | 양측 99% 신뢰구간 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
PPF를 사용하여 $Z \sim N(0,1)$에 대해 $P(-z \le Z \le z) = 0.99$를 만족하는 $z$를 구하라.

</div>

??? success "풀이"
    양쪽 꼬리에 각각 0.5%를 남기므로 $P(Z \le z) = 0.995$가 필요하다:

    $$
    z = \mathcal{N}^{-1}(0.995) \approx 2.576
    $$

    따라서 표준정규분포의 99%가 $\pm 2.576$ 사이에 놓인다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$X \sim N(100, 225)$일 때 $X$의 90 백분위수를 구하라.

</div>

??? success "풀이"
    여기서 $\mu = 100$, $\sigma = 15$이다. 90 백분위수는:

    $$
    x_{0.90} = \mu + \sigma \cdot \mathcal{N}^{-1}(0.90) = 100 + 15 \times 1.282 \approx 119.2
    $$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
연속분포에 대해 모든 $q \in (0, 1)$에서 $F(F^{-1}(q)) = q$임을 증명하라.

</div>

??? success "풀이"
    $x_q = F^{-1}(q) = \inf\{x : F(x) \ge q\}$라 하자. $F$가 연속이고 비감소이며 치역이 $(0,1)$이므로 집합 $\{x : F(x) \ge q\}$는 닫힌 반직선 $[x_q, \infty)$이다. $F$의 연속성에 의해 $F(x_q) = q$이다(하한이 달성된다). 따라서 $F(F^{-1}(q)) = F(x_q) = q$이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
PPF와 생존함수의 관계를 설명하라. $P(X > z) = 0.05$를 만족하는 $z$는 어떻게 계산하겠는가?

</div>

??? success "풀이"
    생존함수는 $S(x) = 1 - F(x)$이다. $P(X > z) = 0.05$이면 $P(X \le z) = 0.95$이므로 $z = F^{-1}(0.95)$이다.

    SciPy에서는 `z = stats.norm.ppf(0.95)`이며, 동등하게 `z = stats.norm.isf(0.05)`로도 구할 수 있다. 여기서 `isf`는 역생존함수이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
$N(\mu, \sigma^2)$의 사분위수 범위를 $\sigma$로 나타내라. $X \sim N(100, 225)$의 $Q_1$과 $Q_3$을 구하라.

</div>

??? success "풀이"
    $\mathcal{N}^{-1}(0.75) \approx 0.6745$이고 대칭성에서 $\mathcal{N}^{-1}(0.25) = -0.6745$이므로

    $$
    \text{IQR} = \sigma\left\{\mathcal{N}^{-1}(0.75) - \mathcal{N}^{-1}(0.25)\right\} = 2 \times 0.6745\,\sigma \approx 1.349\,\sigma
    $$

    이다. $\mu = 100$, $\sigma = 15$이면

    $$
    Q_1 = 100 - 15(0.6745) \approx 89.88, \qquad Q_3 = 100 + 15(0.6745) \approx 110.12
    $$

    이다.

    거꾸로 읽으면 $\hat\sigma = \text{IQR}/1.349$가 된다. 표본표준편차와 달리 극단값에 끌려가지 않는 강건한 척도 추정량이고, 상자그림의 수염 길이 $1.5 \times \text{IQR}$이 정규분포에서 약 $2.7\sigma$에 해당해 바깥값이 0.7%쯤 나오도록 맞춰져 있는 것도 같은 계산에서 나온다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
$n = 5$인 자료의 정규 Q-Q 그림을 그리려 한다. 플로팅 위치 $(i - 0.5)/n$을 쓸 때 가로축에 놓일 이론 분위수 다섯 개를 구하라. 왜 $i/n$을 그냥 쓰지 않는가?

</div>

??? success "풀이"
    $p_i = (i-0.5)/5 = 0.1, 0.3, 0.5, 0.7, 0.9$이므로 이론 분위수는

    $$
    -1.282,\quad -0.524,\quad 0,\quad 0.524,\quad 1.282
    $$

    이다.

    $p_i = i/n$을 쓰면 마지막 값이 $p_5 = 1$이 되고 $\mathcal{N}^{-1}(1) = \infty$이라 가장 큰 관측값을 그릴 수 없다. 경험적 누적분포함수의 계단 한가운데를 대표값으로 잡는 것이 $(i-0.5)/n$이며, 이렇게 하면 양끝이 $(0,1)$ 안에 머문다.

    실제로는 이보다 정교한 블롬(Blom)의 위치 $(i - 3/8)/(n + 1/4)$가 더 널리 쓰인다. 이 경우 분위수가 $-1.180, -0.497, 0, 0.497, 1.180$으로 조금 안쪽으로 당겨진다. 정규분포의 순서통계량 기대값에 더 가깝게 맞춘 것이고, `scipy.stats.probplot`의 기본값이다. 표본이 커지면 어느 쪽을 쓰든 차이가 사라진다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
새 제품의 수명 표준편차가 $\sigma = 15$시간으로 알려져 있다. 평균 수명을 95% 신뢰수준에서 오차한계 2시간 안으로 추정하려면 표본이 몇 개 필요한가?

</div>

??? success "풀이"
    평균이 알려진 분산에서의 신뢰구간은 $\bar X \pm z_{0.975}\,\sigma/\sqrt n$이므로 오차한계가

    $$
    E = z_{0.975}\frac{\sigma}{\sqrt n} \le 2
    $$

    이면 된다. $n$에 대해 풀면

    $$
    n \ge \left(\frac{z_{0.975}\,\sigma}{E}\right)^2 = \left(\frac{1.96 \times 15}{2}\right)^2 = 216.09
    $$

    이다. 표본크기는 정수이고 부등식을 만족해야 하므로 **올림**해서 $n = 217$이다.

    두 가지를 눈여겨본다. 첫째, $n$이 오차한계의 **제곱에 반비례**한다. 정밀도를 두 배로 높이려면 표본을 네 배로 늘려야 한다. 둘째, 반올림이 아니라 올림이다. 216으로 하면 오차한계가 2를 아주 조금 넘는다. 실무에서는 $\sigma$가 추정값일 때 $t$ 분위수를 쓰거나 여유를 더 두기도 한다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
정규분포의 분위수함수가 $F^{-1}_{\mu,\sigma}(q) = \mu + \sigma\,\mathcal{N}^{-1}(q)$임을 보여라. 같은 방법으로 로그정규분포의 분위수함수를 구하라.

</div>

??? success "풀이"
    $X \sim N(\mu, \sigma^2)$이면 $Z = (X-\mu)/\sigma \sim N(0,1)$이므로

    $$
    q = P(X \le x) = P\!\left(Z \le \frac{x-\mu}{\sigma}\right) = \mathcal{N}\!\left(\frac{x-\mu}{\sigma}\right)
    $$

    이다. 양변에 $\mathcal{N}^{-1}$을 적용하면 $(x-\mu)/\sigma = \mathcal{N}^{-1}(q)$, 즉 $x = \mu + \sigma\,\mathcal{N}^{-1}(q)$이다. $\square$

    일반적으로 **분위수함수는 증가하는 변환과 맞바꿀 수 있다.** $g$가 증가함수이고 $Y = g(X)$이면 $F_Y^{-1}(q) = g(F_X^{-1}(q))$이다. $\{Y \le g(x)\}$와 $\{X \le x\}$가 같은 사건이기 때문이다.

    로그정규분포는 $Y = e^X$이고 지수함수가 증가함수이므로

    $$
    F_Y^{-1}(q) = \exp\!\left(\mu + \sigma\,\mathcal{N}^{-1}(q)\right)
    $$

    이다. $q = 0.5$를 넣으면 중앙값 $e^\mu$가 나온다. 평균에는 이런 성질이 없다는 점이 중요하다. $E[g(X)] \ne g(E[X])$이지만 분위수는 그대로 옮겨 간다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$X \sim \text{Binomial}(10, 0.5)$에서 $F^{-1}(0.3)$을 구하고, $F(F^{-1}(0.3)) = 0.3$이 성립하지 않음을 확인하라. 연습문제 3의 결론이 왜 깨지는가? 그럼에도 $F^{-1}(U) \sim F$가 성립함을 보여라.

</div>

??? success "풀이"
    $F(3) = 0.1719$, $F(4) = 0.3770$이므로 $F(x) \ge 0.3$이 되는 가장 작은 정수는 4이다. 따라서

    $$
    F^{-1}(0.3) = 4, \qquad F(F^{-1}(0.3)) = 0.3770 \ne 0.3
    $$

    이다.

    연습문제 3의 증명은 "$F$가 연속이므로 하한에서 $F(x_q) = q$가 달성된다"는 데 기댔다. 이산분포의 $F$는 계단함수라 $0.3$이라는 값을 아예 지나가지 않고 $0.1719$에서 $0.3770$으로 뛴다. 도달할 수 없는 값의 역상을 물었으니 등호가 성립할 수 없다. 일반적으로 성립하는 것은 $F(F^{-1}(q)) \ge q$라는 부등식뿐이다.

    **그래도 역변환은 작동한다.** 하한 정의에서 다음 동치가 성립한다.

    $$
    F^{-1}(u) \le x \iff u \le F(x)
    $$

    ($\Leftarrow$는 $x$가 집합 $\{t : F(t) \ge u\}$에 속하므로 하한이 $x$ 이하. $\Rightarrow$는 $F$가 우연속이라 하한이 집합에 속하고, $F$가 비감소이므로 $u \le F(F^{-1}(u)) \le F(x)$.) 따라서 $U \sim \text{Uniform}(0,1)$에 대해

    $$
    P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)
    $$

    이다. $F^{-1}(U)$가 정확히 $F$를 따른다. $\square$

    하한으로 정의하는 까닭이 여기에 있다. 등호 $F(F^{-1}(q))=q$를 포기하는 대신 위 동치를 얻고, 그 덕분에 역변환 추출법이 연속·이산·혼합 분포 모두에서 통한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
정규난수를 만들 때 역변환 추출법 $X = \mathcal{N}^{-1}(U)$를 쓰지 않고 박스-뮐러 변환을 쓰는 경우가 많다. 이유를 설명하고, 그럼에도 역변환법을 써야 하는 상황을 하나 들어라.

</div>

??? success "풀이"
    정규분포의 $\mathcal{N}^{-1}$에는 닫힌 형태가 없다. 유리함수 근사나 뉴턴 반복으로 계산해야 하므로 한 번 호출이 초월함수 몇 개를 쓰는 것보다 비싸고, 극단 꼬리에서는 정밀도도 떨어진다.

    박스-뮐러 변환은 $U_1, U_2 \sim \text{Uniform}(0,1)$에서

    $$
    Z_1 = \sqrt{-2\ln U_1}\cos(2\pi U_2), \qquad Z_2 = \sqrt{-2\ln U_1}\sin(2\pi U_2)
    $$

    로 독립인 표준정규 난수를 **두 개씩** 만든다. 극좌표로 옮기면 반지름의 제곱이 지수분포를 따르고 각도가 균등분포를 따른다는 사실을 쓴 것이며, 모두 초등함수라 훨씬 빠르다. 실제 난수 라이브러리는 이보다도 빠른 지구랏(ziggurat) 알고리즘을 쓴다.

    **역변환법이 필요한 경우**는 난수 하나와 그 분위수의 대응을 유지해야 할 때다. 대표적으로 준난수를 쓰는 준몬테카를로나, 서로 다른 시나리오에서 같은 $U$를 재사용해 분산을 줄이는 공통난수 기법, 대조변량 $1-U$를 쓰는 기법이 그렇다. 코퓰라로 상관구조를 만들 때도 균등난수와 분위수의 일대일 대응이 방법의 핵심이므로 역변환을 써야 한다.

---

## 정리하며

분위수함수 $\text{ppf}(q)=F^{-1}(q)$ 는 누적분포함수를 거꾸로 읽는다. **확률을 주면 값을 돌려준다.**

- **$\mathcal{N}^{-1}(0.975)\approx1.96$** 이 이 책에서 가장 자주 쓰이는 수다. 양측 $95\%$ 신뢰구간의 임계값이며, $0.95$ 가 아니라 $0.975$ 를 넣는다는 점이 요령이다. 양쪽 꼬리에 $2.5\%$ 씩 남기기 때문이다.
- **하한(inf)으로 정의하는 이유**는 $F$ 가 계단이거나 평평한 구간을 가질 수 있기 때문이다. 정규분포처럼 연속이고 순증가하면 보통의 역함수와 같다.
- 신뢰구간의 임계값, 가설검정의 기각역 경계, Q-Q 그림의 이론 분위수가 모두 이 함수를 호출한 결과다.
- **역변환 표본추출**의 핵심 부품이기도 하다. $U\sim\text{Uniform}(0,1)$ 일 때 $F^{-1}(U)$ 가 $F$ 를 따른다.

다음 절 **생존함수**는 반대쪽 꼬리를 본다. $1-F(x)$ 를 직접 계산하는 것이 왜 수치적으로 더 나은지가 요점이다.
