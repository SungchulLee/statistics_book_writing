# 역변환 표본추출

## 개요

**역변환 표본추출**은 역 CDF(분위수 함수)를 구할 수 있는 임의의 분포에서 확률표본을 생성하는 방법이다. 핵심 결과는 다음과 같다.

$U \sim \text{Uniform}(0, 1)$이고 $F$가 역함수 $F^{-1}$을 갖는 CDF이면:

$$
X = F^{-1}(U) \sim F
$$

이 하나의 착상이 계산통계학과 Monte Carlo 모의실험의 상당 부분을 떠받친다.

---

## 증명

임의의 $x$에 대해:

$$
P(X \le x) = P(F^{-1}(U) \le x) = P(U \le F(x)) = F(x)
$$

마지막 등식은 $U \sim \text{Uniform}(0,1)$이므로 $P(U \le p) = p$라는 사실을 사용한다. $\square$

---

## 지수분포

지수분포의 CDF는 $F(x) = 1 - e^{-\lambda x}$이다. 역함수를 구하면:

$$
F^{-1}(u) = -\frac{\ln(1 - u)}{\lambda}
$$

<div class="codebox" markdown>

### 예제 1. 역변환 표집으로 지수분포 만들기 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)
n = 10_000
lam = 1.0

# 역변환 표집의 전부가 이 두 줄이다.
#   1단계: 균등난수 U를 뽑는다.
#   2단계: 목표 분포의 CDF 역함수 F^{-1} 을 U에 적용한다.
# 지수분포는 F(x) = 1 - e^{-lam x} 이므로 F^{-1}(u) = -ln(1-u)/lam 이다.
u = np.random.uniform(0, 1, n)
x_exp = -np.log(1 - u) / lam

# 세 패널로 과정을 분해해 보여 준다.
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 왼쪽: 출발점. 완전히 평평한 균등분포다.
axes[0].hist(u, bins=50, density=True, color="lightgray", edgecolor="black")
axes[0].set_title("Step 1: U ~ Uniform(0,1)")

t = np.linspace(0, np.max(x_exp), 300)
axes[1].hist(x_exp, bins=50, density=True, color="steelblue",
             edgecolor="white", alpha=0.7, label="Transformed")
axes[1].plot(t, stats.expon.pdf(t, scale=1/lam), "r-", lw=2.5, label="Exp PDF")
axes[1].set_title("Step 2: X = -ln(1-U)/λ")
axes[1].legend()

# 오른쪽: 변환 함수 자체를 그린다.
# 이 곡선의 **기울기**가 왜 평평한 입력이 치우친 출력이 되는지 설명한다.
# u가 1에 가까워질수록 곡선이 가팔라져, 좁은 u 구간이 넓은 x 구간으로 늘어난다.
# 늘어난 구간에서는 밀도가 낮아지므로 오른쪽 꼬리가 얇아지는 것이다.
# 0과 1은 각각 -inf, +inf 로 발산하므로 격자에서 살짝 안쪽으로 잡는다.
u_grid = np.linspace(0.001, 0.999, 300)
axes[2].plot(u_grid, -np.log(1 - u_grid) / lam, "b-", lw=2)
axes[2].set_title("Inverse CDF: F⁻¹(u)")
axes[2].set_xlabel("u")
axes[2].set_ylabel("x")

plt.tight_layout()
plt.show()
```

![Step 1: U ~ Uniform(0,1)](./img/inverse_transform_sampling_37.png)

</div>

---

## 코시분포

표준 코시분포의 CDF는 $F(x) = \frac{1}{2} + \frac{1}{\pi}\arctan(x)$이다. 역함수를 구하면:

$$
F^{-1}(u) = \tan\!\left(\pi\!\left(u - \frac{1}{2}\right)\right)
$$

<div class="codebox" markdown>

### 예제 2. 역변환 표집으로 코시분포 만들기 { .eg }

```python
# 코시분포의 CDF는 F(x) = 1/2 + arctan(x)/pi 이므로
# 그 역함수는 F^{-1}(u) = tan(pi*(u - 1/2)) 이다.
u = np.random.uniform(0, 1, n)
x_cauchy = np.tan(np.pi * (u - 0.5))

fig, ax = plt.subplots(figsize=(10, 4))
# 코시분포는 꼬리가 너무 무거워 평균조차 존재하지 않는다.
# 1만 개를 뽑으면 수백, 수천 단위의 값이 나오므로 그대로 그리면
# 히스토그램이 한 칸에 뭉쳐 아무것도 안 보인다.
# 그래서 [-20, 20]으로 잘라 **가운데 부분만** 그린다.
# 자른 값들은 양 끝 막대에 쌓이므로 그 두 막대는 해석하지 말아야 한다.
x_clipped = np.clip(x_cauchy, -20, 20)
ax.hist(x_clipped, bins=80, density=True, color="coral",
        edgecolor="white", alpha=0.7, label="Transformed")
t2 = np.linspace(-20, 20, 500)
ax.plot(t2, stats.cauchy.pdf(t2), "k-", lw=2.5, label="Cauchy PDF")
ax.set_title("Cauchy via Inverse Transform")
ax.set_xlim(-20, 20)
ax.legend()
plt.tight_layout()
plt.show()
```

![Cauchy via Inverse Transform](./img/inverse_transform_sampling_94.png)

코시 예는 역변환 표본추출이 유한한 평균조차 없는 두꺼운 꼬리 분포에서도 작동함을 보여 준다.

</div>

---

## 언제 사용하는가

| 상황 | 권장 방법 |
|---|---|
| $F^{-1}$이 닫힌 형태로 주어짐 | 역변환 (빠르고 정확) |
| $F^{-1}$ 계산 비용이 큼 | 기각표본추출이나 MCMC 고려 |
| 이산분포 | 누적 PMF 문턱값 사용 |
| 다변량 분포 | 조건부 분해나 전용 알고리즘 사용 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$\text{Uniform}(a, b)$ 분포의 역 CDF를 유도하고 역변환 공식을 쓰라.

</div>

??? success "풀이"
    CDF는 $F(x) = (x - a)/(b - a)$이다. $u = F(x)$로 두고 풀면:

    $$
    x = a + (b - a)u = F^{-1}(u)
    $$

    따라서 $U \sim \text{Uniform}(0,1)$이면 $X = a + (b-a)U \sim \text{Uniform}(a, b)$이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
역변환 표본추출로 $\text{Bernoulli}(p)$ 분포에서 표본을 생성하라. 같은 착상이 임의의 이산분포로 어떻게 확장되는지 설명하라.

</div>

??? success "풀이"
    $U \sim \text{Uniform}(0,1)$을 생성한다. $U \le p$이면 $X = 1$로, 그렇지 않으면 $X = 0$으로 둔다.

    값이 $x_1, x_2, \ldots$이고 확률이 $p_1, p_2, \ldots$인 일반적인 이산분포에서는 누적확률 $c_k = \sum_{i=1}^k p_i$를 계산한다. $U \le c_k$를 만족하는 가장 작은 첨자 $k$에 대해 $X = x_k$로 둔다. 이는 $[0,1]$을 길이 $p_k$인 구간들로 나누고 각 구간을 $x_k$에 대응시키는 것이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
역변환 결과를 증명하라. $U \sim \text{Uniform}(0,1)$이고 $F$가 연속이고 순증가하는 CDF이면 $F^{-1}(U) \sim F$이다.

</div>

??? success "풀이"
    임의의 $x \in \mathbb{R}$에 대해:

    $$
    P(F^{-1}(U) \le x) = P(U \le F(x))
    $$

    이 등식은 $F$가 순증가하므로 $F^{-1}(u) \le x \iff u \le F(x)$라는 사실을 사용한다. $U \sim \text{Uniform}(0,1)$이므로:

    $$
    P(U \le F(x)) = F(x)
    $$

    따라서 $P(F^{-1}(U) \le x) = F(x)$이고, 이는 $F^{-1}(U)$의 CDF가 $F$임을 뜻한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
레일리분포는 $x \ge 0$에 대해 CDF가 $F(x) = 1 - e^{-x^2/(2\sigma^2)}$이다. 역 CDF를 유도하고 역변환 표본추출로 레일리 표본을 생성하는 코드를 작성하라.

</div>

??? success "풀이"
    $u = 1 - e^{-x^2/(2\sigma^2)}$의 역함수를 구하면:

    $$
    e^{-x^2/(2\sigma^2)} = 1 - u \implies x = \sigma\sqrt{-2\ln(1-u)}
    $$

    코드:

    ```python
    import numpy as np

    np.random.seed(0)
    sigma = 1.0
    u = np.random.uniform(0, 1, 10000)
    # 레일리 분포의 CDF는 F(x) = 1 - exp(-x^2 / (2 sigma^2)) 이므로
    # 이를 x에 대해 풀면 F^{-1}(u) = sigma * sqrt(-2 ln(1-u)) 다.
    x = sigma * np.sqrt(-2 * np.log(1 - u))

    # 레일리 분포의 평균은 sigma*sqrt(pi/2) ≈ 1.2533,
    # 분산은 (4-pi)/2 * sigma^2 ≈ 0.4292 다.
    print(f"표본평균 {x.mean():.4f}  (이론 {sigma*np.sqrt(np.pi/2):.4f})")
    print(f"표본분산 {x.var():.4f}  (이론 {(4-np.pi)/2*sigma**2:.4f})")
    ```

    출력:

    ```
    표본평균 1.2453  (이론 1.2533)
    표본분산 0.4306  (이론 0.4292)
    ```

    이는 Box-Muller 변환의 한 성분과 밀접하게 관련된다. $R = \sqrt{-2\ln U}$가 레일리분포를 따른다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
모든 역변환 공식에서 $U$를 $1 - U$로 바꾸어도 출력의 분포가 달라지지 않는 이유를 설명하라. 실무에서 이것이 왜 유용한가?

</div>

??? success "풀이"
    $U \sim \text{Uniform}(0,1)$이면 $1 - U$도 $\text{Uniform}(0,1)$이다(균등분포는 0.5를 중심으로 대칭이다). 따라서 $F^{-1}(U)$에서 $U$를 $1 - U$로 바꾸어도 같은 분포가 나온다.

    이는 공식을 간단하게 만들어 주므로 유용하다. 지수분포에서는 $-\ln(1-U)/\lambda$를 $-\ln(U)/\lambda$로 바꿀 수 있어 뺄셈 한 번을 아낀다. 실무에서는 $\ln(0) = -\infty$를 주는 $U = 0$이라는 경계 사례도 피하게 되는데, $1 - U = 0$이 될 확률은 0이기 때문이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
파레토분포의 CDF는 $x \ge x_m$에 대해 $F(x) = 1 - (x_m/x)^\alpha$이다. 역변환 공식을 유도하고, $\alpha$가 작을수록 극단값이 자주 나오는 이유를 그 공식으로 설명하라.

</div>

??? success "풀이"
    $u = 1 - (x_m/x)^\alpha$를 풀면 $(x_m/x)^\alpha = 1-u$이므로

    $$
    x = x_m (1-u)^{-1/\alpha}, \qquad \text{즉}\quad X = x_m\,U^{-1/\alpha}
    $$

    이다(마지막은 $1-U$를 $U$로 바꾼 것이다).

    지수 $-1/\alpha$가 전부를 말해 준다. $U$가 0에 가까운 작은 값이 나오면 $U^{-1/\alpha}$가 폭발하는데, 그 폭발의 세기를 $1/\alpha$가 정한다. $\alpha = 2$이면 $U = 0.01$일 때 배율이 $10$이지만, $\alpha = 0.5$이면 같은 $U$에서 배율이 $10^4$이다.

    지수분포의 $-\ln U$와 견주면 차이가 분명하다. 로그는 아무리 작은 $U$에도 아주 천천히 자라지만 거듭제곱은 빠르게 자란다. 그래서 지수분포는 꼬리가 지수적으로 얇고 파레토분포는 거듭제곱 꼬리를 갖는다. 소득·도시 인구·인터넷 트래픽처럼 "몇몇이 전체를 지배하는" 현상에 파레토분포가 쓰이는 까닭이다. $\alpha \le 2$이면 분산이, $\alpha \le 1$이면 평균조차 존재하지 않는다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$F^{-1}$을 구할 수 없을 때 쓰는 **기각표집**의 절차를 적고 그 타당성을 보여라. 표준정규분포를 표준 코시분포에서 기각표집으로 뽑을 때의 수용률을 구하라.

</div>

??? success "풀이"
    **절차.** 목표밀도 $f$, 뽑기 쉬운 제안밀도 $g$, 그리고 모든 $x$에서 $f(x) \le M g(x)$인 상수 $M$을 준비한다.

    1. $Y \sim g$를 뽑는다.
    2. $U \sim \text{Uniform}(0,1)$을 뽑는다.
    3. $U \le f(Y)/\{M g(Y)\}$이면 $Y$를 받아들이고, 아니면 1로 돌아간다.

    **타당성.** 한 번의 시도에서 받아들여지면서 $Y \le x$일 확률은

    $$
    \int_{-\infty}^x g(y)\,\frac{f(y)}{Mg(y)}\,dy = \frac1M\int_{-\infty}^x f(y)\,dy = \frac{F(x)}{M}
    $$

    이다. $x \to \infty$로 두면 한 번에 받아들여질 확률이 $1/M$임을 알 수 있고, 따라서 받아들여졌다는 조건 아래의 분포는

    $$
    P(Y \le x \mid \text{수용}) = \frac{F(x)/M}{1/M} = F(x)
    $$

    이다. 정확히 목표분포를 준다. $\square$

    **수용률.** 표준정규 $\varphi$와 표준 코시 $g(x) = 1/\{\pi(1+x^2)\}$에 대해

    $$
    \frac{\varphi(x)}{g(x)} = \frac{\pi(1+x^2)}{\sqrt{2\pi}}e^{-x^2/2}
    $$

    이다. 로그를 미분해 최대를 찾으면 $x = \pm1$에서 최대이고

    $$
    M = \sqrt{\frac{2\pi}{e}} \approx 1.5203
    $$

    이다. 수용률은 $1/M \approx 0.658$로, 평균 1.52번 시도해야 하나를 얻는다.

    제안분포를 정규분포로 하고 목표를 코시로 잡는 반대 방향은 **불가능하다.** 코시의 꼬리가 훨씬 두꺼워 $f/g$가 유계가 아니므로 그런 $M$이 존재하지 않는다. **제안분포는 목표분포보다 꼬리가 두꺼워야 한다**는 것이 기각표집의 철칙이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
$X \sim F$를 구간 $[a, b]$로 절단한 분포에서 표본을 뽑는 방법을 역변환으로 설계하라. 단순히 "범위 밖이면 버리고 다시 뽑기"와 견주면 어떤 점이 나은가?

</div>

??? success "풀이"
    절단분포의 CDF는 $a \le x \le b$에서

    $$
    F_{[a,b]}(x) = \frac{F(x) - F(a)}{F(b) - F(a)}
    $$

    이다. $u = F_{[a,b]}(x)$를 풀면

    $$
    x = F^{-1}\Big(F(a) + u\,\{F(b) - F(a)\}\Big)
    $$

    을 얻는다. 즉 **$[0,1]$의 균등난수를 $[F(a), F(b)]$의 균등난수로 늘린 뒤 $F^{-1}$에 넣으면** 된다.

    ```python
    lo, hi = dist.cdf(a), dist.cdf(b)
    x = dist.ppf(lo + np.random.uniform(0, 1, n) * (hi - lo))
    ```

    **버리고 다시 뽑기와의 비교.** 기각 방식은 한 번에 성공할 확률이 $F(b)-F(a)$이므로 절단 구간이 좁으면 치명적으로 느려진다. $P(a \le X \le b) = 10^{-6}$이라면 평균 백만 번을 뽑아야 하나를 얻는다. 역변환 방식은 구간이 아무리 좁아도 **언제나 한 번에 끝난다.** 또 뽑는 횟수가 고정되어 있어 실행 시간이 예측 가능하고, 난수 하나와 결과 하나가 일대일로 대응하므로 공통난수 같은 분산감소 기법과도 맞는다.

    한 가지 주의할 점은 수치 정밀도다. 꼬리 깊숙한 곳을 절단하면 $F(a)$와 $F(b)$가 둘 다 1에 붙어 그 차이에서 유효숫자가 날아간다. 그런 경우에는 생존함수 쪽에서 같은 계산을 하거나 로그 척도로 다루어야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
두 설계 $A$와 $B$의 성능 차이 $E[h_A(X)] - E[h_B(X)]$를 모의실험으로 추정한다. 두 시나리오에 **같은** 균등난수를 쓰면(공통난수) 왜 추정의 분산이 줄어드는지 보이고, 어떤 조건에서 오히려 해가 되는지 말하라.

</div>

??? success "풀이"
    각 시나리오에서 얻은 추정량을 $\hat\theta_A$, $\hat\theta_B$라 하면 차이의 분산은

    $$
    \operatorname{Var}(\hat\theta_A - \hat\theta_B) = \operatorname{Var}(\hat\theta_A) + \operatorname{Var}(\hat\theta_B) - 2\operatorname{Cov}(\hat\theta_A, \hat\theta_B)
    $$

    이다. 서로 다른 난수를 쓰면 공분산이 0이라 두 분산이 그대로 더해진다. 같은 난수를 쓰면 공분산이 양수가 되어 그만큼 줄어든다.

    **왜 양수가 되는가.** 역변환을 쓰면 $X_A = F_A^{-1}(U)$, $X_B = F_B^{-1}(U)$로 같은 $U$에서 두 입력이 나온다. $F_A^{-1}$과 $F_B^{-1}$이 모두 증가함수이므로 $U$가 크면 두 입력이 함께 크고, 작으면 함께 작다. 즉 $(X_A, X_B)$가 양으로 결합되어 있다. $h_A$와 $h_B$가 둘 다 같은 방향으로 단조라면 출력도 양으로 결합되고 공분산이 양수가 된다. **엄밀히는** 증가함수의 쌍에 대해 $\operatorname{Cov}\{h_A(F_A^{-1}(U)),\, h_B(F_B^{-1}(U))\} \ge 0$임이 FKG 부등식(또는 체비쇼프의 합 부등식)으로 보장된다.

    "같은 난수"란 같은 씨앗을 뜻하는 것이 아니라 **같은 무작위성이 같은 역할에 배정된다**는 뜻이다. 대기행렬 모의실험이라면 $k$번째 손님의 도착 간격과 $k$번째 서비스 시간이 두 시나리오에서 같은 균등난수에서 나와야 한다. 여기서 역변환이 필수가 된다. 기각표집이나 박스-뮐러는 소비하는 난수의 개수가 실행마다 달라져 대응이 어긋난다.

    **해가 되는 경우.** 두 시나리오의 반응이 서로 반대 방향이면 공분산이 음수가 되어 분산이 오히려 커진다. 또 난수 배정이 어긋나 한쪽에서만 난수를 더 쓰면 그 뒤로 모든 대응이 밀려 상관이 사라진다. 이럴 때는 난수 흐름을 역할별로 분리해 각각 별도의 생성기(`rng.spawn`)에서 뽑는 것이 안전하다.

    같은 원리의 사촌이 **대조변량**이다. 한 번은 $U$로, 한 번은 $1-U$로 돌려 평균 내면 음의 상관 덕분에 분산이 줄어든다. 연습문제 5에서 본 $U$와 $1-U$의 교체 가능성이 여기서 쓸모를 얻는다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$K$개의 값을 갖는 이산분포에서 난수 하나를 뽑는 비용을 연습문제 2의 방법으로 계산하면 얼마인가? 더 빠른 방법을 두 가지 들어라.

</div>

??? success "풀이"
    연습문제 2의 방법은 누적확률 $c_1, c_2, \dots$를 앞에서부터 훑으며 $U \le c_k$인 첫 $k$를 찾는다. 평균 비교 횟수가 $\sum_k k\,p_k$이므로 최악의 경우 $O(K)$이다. $K$가 크면(예를 들어 어휘 크기가 5만인 언어모형에서 다음 낱말을 뽑는 경우) 난수 하나마다 수만 번의 비교가 필요해 감당하기 어렵다.

    **방법 1 — 이분탐색.** 누적확률 배열 $c$는 정렬되어 있으므로 이분탐색으로 $O(\log K)$에 찾을 수 있다. `np.searchsorted(c, u)` 한 줄이면 되고, 벡터화되어 여러 난수를 한 번에 처리한다. 전처리로 누적합을 만드는 데 $O(K)$가 들지만 한 번만 하면 된다. 가장 손쉬운 개선이다.

    **방법 2 — 앨리어스 방법.** $O(K)$ 전처리로 표를 만들어 두면 그 뒤로는 **난수 하나당 $O(1)$**에 뽑는다. 착상은 이렇다. 확률 $K$개를 높이 $1/K$인 상자 $K$개에 나누어 담되, 각 상자에 많아야 두 종류만 들어가게 배치한다(평균보다 큰 것을 쪼개 작은 것의 빈자리를 채우는 식이다). 뽑을 때는 상자 번호를 균등하게 하나 고르고 그 안에서 동전을 한 번 던지면 끝이다.

    선택 기준은 분포가 얼마나 자주 바뀌느냐다. 같은 분포에서 아주 많이 뽑는다면 앨리어스 방법이 압도적이고, 확률이 매번 달라진다면(예를 들어 매 단계 확률을 새로 계산하는 깁스 표집) 전처리 비용이 아까우므로 이분탐색이나 선형 탐색이 낫다. NumPy의 `rng.choice`는 기본적으로 누적합과 이분탐색을 쓴다.

---

## 정리하며

역변환 표본추출은 한 줄로 요약된다. **$U\sim\text{Uniform}(0,1)$ 이면 $F^{-1}(U)\sim F$ 다.**

- **증명도 한 줄이다.** $P(F^{-1}(U)\le x)=P(U\le F(x))=F(x)$ 이며, 마지막 등식이 균등분포의 정의다.
- **분위수함수를 구할 수 있으면 어떤 분포든 뽑을 수 있다.** 지수분포는 $F^{-1}(u)=-\ln(1-u)/\lambda$ 처럼 손으로 풀리고, 그렇지 않은 경우에는 수치적 역함수를 쓴다.
- **정규분포에는 잘 쓰지 않는다.** $\Phi^{-1}$ 에 닫힌 형태가 없어 비싸기 때문이며, 실제로는 박스–뮐러 변환이나 지구라트 알고리즘을 쓴다.
- **이산분포에도 그대로 통한다.** $F$ 가 계단이면 $F^{-1}(u)=\inf\{x:F(x)\ge u\}$ 가 계단의 위치를 골라 준다.
- **역함수를 구할 수 없을 때**는 기각 표집이나 MCMC 로 넘어간다.

**이 하나의 착상이 계산통계학의 상당 부분을 떠받친다.** 뒤에 나올 부트스트랩·순열검정·몬테카를로 모의실험이 모두 "균등난수에서 원하는 분포를 만든다"는 이 단계 위에 서 있다.

다음 절 **결합분포**로 넘어간다. 지금까지 변수 하나를 다뤘다면, 이제 둘 이상이 함께 움직이는 경우를 본다.
