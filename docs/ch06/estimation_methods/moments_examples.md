# 흔한 분포에 대한 적률법

적률법은 적률을 모수로 나타낼 수 있는 임의의 분포에서 모수를 추정하는 체계적인 방법을 제공한다. 처음 $p$개의 모집단 적률을 대응하는 표본 적률과 같다고 두고 그 연립방정식을 풀면 직관적이고 계산하기 쉬운 닫힌 형태의 추정량을 얻는다. 이 페이지에서는 정규분포, 감마분포, 베타분포라는 세 가지 중요한 분포족에 대해 적률법 추정량을 유도하며, 대수적 복잡도가 커지는 순서로 일반적인 절차를 보인다.

이 페이지에서 표본 적률을 다음과 같이 정의한다:

$$
M_1 = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i, \quad M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

$M_2$가 ($1/(n-1)$이 아니라) $1/n$을 분모로 쓴다는 점에 유의하라. 적률법은 **모집단** 적률을 그 **표본 대응물**과 맞추는데, 모집단의 2차 중심적률 $E[(X - \mu)^2] = \sigma^2$에 대응하는 것은 Bessel 수정된 표본분산 $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$이 아니라 $M_2$이기 때문이다.

---

## 정규분포

정규분포는 모수가 곧 처음 두 적률이기 때문에 적률법에서 가장 단순한 경우이다. $X \sim N(\mu, \sigma^2)$에 대해 모집단 적률은:

$$
E[X] = \mu, \quad E[(X - \mu)^2] = \sigma^2
$$

**적률방정식 세우기.** 모집단 적률을 대응하는 표본 적률과 같다고 둔다:

$$
\mu = M_1 = \bar{X}
$$

$$
\sigma^2 = M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**풀기.** 이 연립방정식은 이미 풀려 있다. 각 방정식이 곧바로 추정량을 준다:

$$
\hat{\mu}_{\text{MoM}} = \bar{X}, \quad \hat{\sigma}^2_{\text{MoM}} = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

!!! note "정규분포에서 적률법과 MLE"
    정규분포의 적률법 추정량은 MLE와 동일하다. 둘 다 $\hat{\mu} = \bar{X}$와 $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$을 준다. 적률법의 분산추정량은 편향되어 있지만(기댓값이 $\frac{n-1}{n}\sigma^2$이다) $n \to \infty$일 때 일치한다.

??? example "수치 예제"
    $n = 5$개의 값 $2, 4, 6, 8, 10$을 관측했다고 하자.

    $$
    \hat{\mu}_{\text{MoM}} = \bar{X} = \frac{2 + 4 + 6 + 8 + 10}{5} = 6
    $$

    $$
    \hat{\sigma}^2_{\text{MoM}} = \frac{(2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2}{5} = \frac{16 + 4 + 0 + 4 + 16}{5} = 8
    $$

    적률법 추정값은 $\hat{\mu} = 6$, $\hat{\sigma}^2 = 8$이다(Bessel 수정된 $S^2 = 10$과 비교된다).

---

## 감마분포

감마분포는 대기 시간, 강우량, 보험 청구액처럼 양수이고 오른쪽으로 치우친 자료를 모형화하는 데 널리 쓰인다. 적률법 추정에 두 방정식의 연립을 풀어야 하므로 정규분포보다 배울 것이 많은 예이다.

**형상–척도 모수화**를 사용한다. $X \sim \text{Gamma}(\alpha, \beta)$에서 $\alpha > 0$은 형상모수이고 $\beta > 0$은 척도모수이다. 모집단 적률은:

$$
E[X] = \alpha\beta, \quad \text{Var}(X) = \alpha\beta^2
$$

**적률방정식 세우기.** 모집단 적률을 표본 적률과 같다고 둔다:

$$
\alpha\beta = M_1 = \bar{X}
$$

$$
\alpha\beta^2 = M_2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**풀기.** 두 번째 식을 첫 번째 식으로 나누어 $\beta$를 분리한다:

$$
\frac{\alpha\beta^2}{\alpha\beta} = \frac{M_2}{M_1} \implies \beta = \frac{M_2}{M_1} = \frac{M_2}{\bar{X}}
$$

이를 첫 번째 식에 대입하여 $\alpha$를 구한다:

$$
\alpha = \frac{M_1}{\beta} = \frac{\bar{X}}{M_2/\bar{X}} = \frac{\bar{X}^2}{M_2}
$$

적률법 추정량은:

$$
\hat{\beta}_{\text{MoM}} = \frac{M_2}{\bar{X}}, \quad \hat{\alpha}_{\text{MoM}} = \frac{\bar{X}^2}{M_2}
$$

!!! tip "추정량의 해석"
    형상 추정값 $\hat{\alpha}$는 변동계수의 제곱을 뒤집은 것이다: $\hat{\alpha} = (\bar{X}/\sqrt{M_2})^2 = 1/\hat{CV}^2$. 형상모수가 클수록 평균에 비해 변동이 작은 분포이다.

??? example "수치 예제"
    대기 시간(분)을 관측하여 $3.2, 5.1, 4.7, 6.3, 2.8, 4.9$를 얻었다고 하자.

    $$
    \bar{X} = \frac{3.2 + 5.1 + 4.7 + 6.3 + 2.8 + 4.9}{6} = \frac{27.0}{6} = 4.5
    $$

    $$
    M_2 = \frac{(3.2-4.5)^2 + (5.1-4.5)^2 + (4.7-4.5)^2 + (6.3-4.5)^2 + (2.8-4.5)^2 + (4.9-4.5)^2}{6}
    $$

    $$
    = \frac{1.69 + 0.36 + 0.04 + 3.24 + 2.89 + 0.16}{6} = \frac{8.38}{6} \approx 1.397
    $$

    $$
    \hat{\beta}_{\text{MoM}} = \frac{1.397}{4.5} \approx 0.310, \quad \hat{\alpha}_{\text{MoM}} = \frac{4.5^2}{1.397} = \frac{20.25}{1.397} \approx 14.50
    $$

    추정값은 형상 $\hat{\alpha} \approx 14.5$, 척도 $\hat{\beta} \approx 0.31$인 감마분포를 시사한다.

---

## 베타분포

베타분포는 구간 $(0, 1)$ 위의 자료를 모형화하므로 비율, 비율값, 베이즈 사전분포 설정에 유용하다. 적률법 추정에는 조금 더 복잡한 대수 조작이 필요하다.

$a, b > 0$인 $X \sim \text{Beta}(a, b)$에 대해 모집단 적률은:

$$
E[X] = \frac{a}{a + b}, \quad \text{Var}(X) = \frac{ab}{(a+b)^2(a+b+1)}
$$

**적률방정식 세우기.** $\mu = E[X]$, $\sigma^2 = \text{Var}(X)$라 하고 표본 적률과 같다고 둔다:

$$
\frac{a}{a+b} = M_1 = \bar{X}
$$

$$
\frac{ab}{(a+b)^2(a+b+1)} = M_2
$$

**풀기.** 첫 번째 식에서 $a = \bar{X}(a + b)$이므로 $b = a(1 - \bar{X})/\bar{X}$이고 $a + b = a/\bar{X}$이다. 편의상 $s = a + b$라 두면 $a = s\bar{X}$, $b = s(1 - \bar{X})$이다.

분산 방정식에 대입하면:

$$
\frac{s\bar{X} \cdot s(1 - \bar{X})}{s^2(s + 1)} = M_2 \implies \frac{\bar{X}(1 - \bar{X})}{s + 1} = M_2
$$

$s$에 대해 풀면:

$$
s + 1 = \frac{\bar{X}(1 - \bar{X})}{M_2} \implies s = \frac{\bar{X}(1 - \bar{X})}{M_2} - 1
$$

$a = s\bar{X}$이고 $b = s(1 - \bar{X})$이므로:

$$
\hat{a}_{\text{MoM}} = \bar{X}\left(\frac{\bar{X}(1 - \bar{X})}{M_2} - 1\right)
$$

$$
\hat{b}_{\text{MoM}} = (1 - \bar{X})\left(\frac{\bar{X}(1 - \bar{X})}{M_2} - 1\right)
$$

!!! warning "타당성 조건"
    베타분포의 적률법 추정량은 $M_2 < \bar{X}(1 - \bar{X})$를 요구하며, 이 조건이 $s > 0$을, 따라서 $\hat{a}, \hat{b} > 0$을 보장한다. 표본분산이 $\bar{X}(1 - \bar{X})$를 넘으면 적률법이 실패한다. 그 평균을 갖는 어떤 베타분포로도 담아낼 수 없을 만큼 자료가 퍼져 있기 때문이다.

??? example "수치 예제"
    비율값 $0.3, 0.5, 0.4, 0.6, 0.35, 0.55$를 관측했다고 하자.

    $$
    \bar{X} = \frac{0.3 + 0.5 + 0.4 + 0.6 + 0.35 + 0.55}{6} = \frac{2.70}{6} = 0.45
    $$

    $$
    M_2 = \frac{(0.3-0.45)^2 + \cdots + (0.55-0.45)^2}{6} = \frac{0.0225 + 0.0025 + 0.0025 + 0.0225 + 0.01 + 0.01}{6} = \frac{0.07}{6} \approx 0.01167
    $$

    타당성 확인: $\bar{X}(1-\bar{X}) = 0.45 \times 0.55 = 0.2475 > 0.01167$ (충족).

    $$
    s = \frac{0.2475}{0.01167} - 1 = 21.21 - 1 = 20.21
    $$

    $$
    \hat{a}_{\text{MoM}} = 0.45 \times 20.21 \approx 9.09, \quad \hat{b}_{\text{MoM}} = 0.55 \times 20.21 \approx 11.12
    $$

    적률법 추정값은 $\text{Beta}(9.1, 11.1)$ 분포를 시사하며, 이는 단봉이고 (평균이 0.45로) 오른쪽으로 약간 치우쳐 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
감마분포를 따르는 청구액 자료: 2.1, 0.8, 3.5, 1.2, 5.7, 0.4, 2.8, 1.9, 4.3, 0.6. (a) $\alpha, \beta$의 적률법 추정값. (b) scipy를 이용한 MLE. (c) 시각적 비교.

</div>

??? success "풀이"
    (a) 표본 적률: $\bar x = 2.33$, $m_2 \approx 2.77$.

    $\mathbb{E}[X] = \alpha\beta$, $\mathrm{Var}(X) = \alpha\beta^2$인 Gamma$(\alpha, \beta)$에 대해:

    $\hat\beta = m_2/\bar x \approx 1.19$, $\hat\alpha = \bar x^2/m_2 \approx 1.96$.

    (b) `scipy.stats.gamma.fit(x, floc=0)`으로 구한 MLE: 전형적인 출력은 $\hat\alpha \approx 1.69$, $\hat\beta \approx 1.38$.

    (c) 히스토그램에 두 밀도를 겹쳐 그린다. $n = 10$에서는 두 추정량 모두 분산이 크고 곡선이 비슷하지만 동일하지는 않다. 구별하려면 자료가 더 필요하다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**Uniform$(a, b)$의 적률법.** $\mathbb{E}[X], \mathbb{E}[X^2]$을 사용하여 적률법 추정량을 유도하라.

</div>

??? success "풀이"
    모집단 적률: $\mathbb{E}[X] = (a+b)/2$, $\mathrm{Var}(X) = (b-a)^2/12$. 따라서 $\mathbb{E}[X^2] = (a+b)^2/4 + (b-a)^2/12$이다.

    표본을 모집단과 같다고 두면 $\bar X = (a+b)/2$이고 $m_2 = (b-a)^2/12$이며, $m_2$는 ($n$을 쓴) 표본분산이다.

    두 번째 식에서 $b - a = \sqrt{12 m_2}$이다. $a + b = 2\bar X$와 결합하면:

    $\hat a_{\text{MoM}} = \bar X - \sqrt{3 m_2}$, $\hat b_{\text{MoM}} = \bar X + \sqrt{3 m_2}$.

    **문제점:** 어떤 관측값이 $[\hat a, \hat b]$ 밖에 있으면 이 추정값은 불가능하다. MLE는 순서통계량으로 이를 처리한다: $\hat a_{\text{MLE}} = \min X_i$, $\hat b_{\text{MLE}} = \max X_i$로 언제나 실현 가능하다.

    MLE가 지키는 경계 제약을 적률법이 어길 수 있음을 보여 준다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**Beta$(\alpha, \beta)$의 적률법.** $\mathbb{E}[X], \mathrm{Var}(X)$로부터 유도하라.

</div>

??? success "풀이"
    Beta$(\alpha, \beta)$: $\mathbb{E}[X] = \alpha/(\alpha+\beta)$, $\mathrm{Var}(X) = \alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$.

    $m = \bar X$, $v = m_2$(표본분산)라 하고 풀면:

    $\alpha + \beta = m(1 - m)/v - 1$. 따라서 $\hat\alpha = m \cdot (m(1-m)/v - 1)$, $\hat\beta = (1-m)(m(1-m)/v - 1)$.

    **타당성:** $v < m(1-m)$이 필요하다(베타분포의 분산은 같은 평균을 갖는 베르누이분포의 분산으로 위에서 유계이다). 표본에서 $v > m(1-m)$이면 적률법이 실패한다(맞는 베타분포가 없다). 과대산포가 심한 자료에는 적률법을 쓸 수 없다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**로그정규분포의 적률법.** $Y \sim N(\mu, \sigma^2)$일 때 $X = e^Y$가 주어졌을 때 $\mu, \sigma$의 적률법 추정량을 유도하라.

</div>

??? success "풀이"
    $\mathbb{E}[X] = e^{\mu + \sigma^2/2}$, $\mathrm{Var}(X) = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2}$.

    변동계수의 제곱을 계산하면 $\mathrm{CV}^2 = \mathrm{Var}(X)/(\mathbb{E}[X])^2 = e^{\sigma^2} - 1$이다.

    따라서 $\hat\sigma^2_{\text{MoM}} = \ln(1 + \mathrm{CV}^2_{\text{sample}}) = \ln(1 + s^2/\bar X^2)$이다.

    $\hat\mu_{\text{MoM}} = \ln \bar X - \hat\sigma^2_{\text{MoM}}/2$.

    **더 쉬운 대안 (MLE 방식):** 먼저 변환한다. $Y_i = \ln X_i$가 i.i.d. $N(\mu, \sigma^2)$이므로 표준적으로 $\hat\mu = \bar Y$, $\hat\sigma^2 = s_Y^2$이다. 적률법보다 깔끔하고 효율적이다. 원래 척도에서 적률법을 쓰면 편리한 로그변환을 포기하는 셈이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**적률법이 비효율적일 수 있는 이유.** Uniform$(0, \theta)$에서 적률법과 MLE를 비교하라.

</div>

??? success "풀이"
    적률법: $\bar X = \theta/2 \Rightarrow \hat\theta_{\text{MoM}} = 2\bar X$. $\mathrm{Var}(\hat\theta_{\text{MoM}}) = 4 \mathrm{Var}(\bar X) = 4 \theta^2/(12n) = \theta^2/(3n)$.

    MLE: $\hat\theta_{\text{MLE}} = X_{(n)}$. 편향 보정한 $((n+1)/n) X_{(n)}$은 $\mathrm{Var} = \theta^2/[n(n+2)]$이다.

    비: $\mathrm{Var}(\hat\theta_{\text{MoM}})/\mathrm{Var}(\hat\theta_{\text{MLE}}) = (n+2)/3 \to \infty$.

    이 문제에서 $n \to \infty$일 때 적률법은 MLE보다 *무한히* 나쁘다. 표본평균은 최댓값에 관한 정보를 버리지만 MLE는 그것을 활용한다.

    **일반적인 교훈:** 적률법은 소박하며 저차 적률만 사용한다. MLE는 완전한 가능도를 사용하므로, 분포가 적률 너머의 "구조"를 가질 때, 특히 지지집합이 유계이거나(균등분포의 끝점) 꼬리가 두꺼울 때 훨씬 효율적일 수 있다.

    적률법은 MLE를 다루기 어렵거나 반복적 MLE 최적화의 출발점이 필요할 때 여전히 유용하다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**일반화 적률법 (GMM).** 모수보다 적률 조건이 많을 때 이들을 결합하는 가중 방식을 제안하라.

</div>

??? success "풀이"
    $\theta \in \mathbb{R}^k$이고 $j = 1, \ldots, m$에 대해 $m \ge k$개의 적률 조건 $\mathbb{E}[g_j(X; \theta)] = 0$이 있다고 하자.

    **GMM 추정량:**

    $$
    \hat\theta = \arg\min_\theta \left[\sum_i \mathbf g(X_i; \theta)\right]^T W \left[\sum_i \mathbf g(X_i; \theta)\right]
    $$

    여기서 $W$는 양의 정부호 $m \times m$ 가중행렬이다.

    **최적 $W$:** 적률 조건의 공분산행렬의 역행렬 $W^* = \Omega^{-1}$이다. 이것이 점근분산을 최소화한다.

    **2단계 GMM:**

    1. $W = I$(단위행렬)로 초기 $\hat\theta^{(1)}$을 구한다.
    2. $\hat\theta^{(1)}$에서 $\Omega$를 추정하고 $W = \hat\Omega^{-1}$로 둔다.
    3. 다시 최적화한다.

    GMM은 실증경제학의 토대이며(Hansen의 1982년 논문은 노벨상을 받았다) 내생변수보다 도구변수가 많을 때의 도구변수 추정을 떠받친다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
$\text{Weibull}(k,\lambda)$의 적률법 추정량을 유도하라. 왜 닫힌 해가 없으며 어떻게 푸는가?

</div>

??? success "풀이"
    **적률.** 앞서 본 대로

    $$
    E[X] = \lambda\Gamma\!\left(1+\frac1k\right), \qquad E[X^2] = \lambda^2\Gamma\!\left(1+\frac2k\right)
    $$

    이다.

    **$\lambda$를 소거한다.** 변동계수를 만들면 $\lambda$가 약분된다.

    $$
    \frac{s^2}{\bar x^2} = \frac{\Gamma(1+2/k)}{\Gamma(1+1/k)^2}-1 =: h(k)
    $$

    **닫힌 해가 없는 이유.** 우변이 **감마함수의 비**라 $k$에 대해 대수적으로 풀리지 않는다. $h$는 $k$의 순감소함수($k$가 크면 분포가 덜 퍼진다)이므로 해는 유일하다.

    **푸는 방법.**

    1. 표본에서 $\hat c = s/\bar x$(변동계수)를 계산한다.
    2. $h(k) = \hat c^2$을 수치적으로 푼다. 이분법이나 뉴턴법이면 충분하며, $h$가 단조라 수렴이 안정적이다.
    3. 그다음 $\hat\lambda = \bar x/\Gamma(1+1/\hat k)$로 얻는다.

    ```python
    from scipy.optimize import brentq
    from scipy.special import gamma
    h = lambda k: gamma(1 + 2/k) / gamma(1 + 1/k)**2 - 1
    k_hat = brentq(lambda k: h(k) - (s/xbar)**2, 0.1, 50)
    lam_hat = xbar / gamma(1 + 1/k_hat)
    ```

    **어림 공식.** $\hat k \approx \hat c^{-1.086}$이라는 근사가 널리 쓰인다($0.1 < \hat c < 1$ 범위에서 꽤 정확하다). 수치해의 좋은 초기값이 된다.

    **덧붙임.** 와이불의 MLE도 닫힌 해가 없어 수치적으로 풀어야 하므로, 적률법의 "계산이 쉽다"는 장점이 여기서는 크지 않다. 그래도 **MLE 반복의 초기값**으로 쓰기에 좋고, 중도절단이 없는 완전 자료에서는 결과가 크게 다르지 않다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
관측값 10개짜리 감마 자료에서 적률법과 MLE가 꽤 다른 값을 주었다. 어느 쪽을 믿을지 판단하는 절차를 적어라.

</div>

??? success "풀이"
    **1단계 — 차이가 표집변동 수준인가.** 부트스트랩으로 두 추정값을 함께 재표집해 차이 $\hat\alpha_{\text{MoM}}-\hat\alpha_{\text{MLE}}$의 분포를 본다. 그 분포가 0을 넉넉히 담으면 우연으로 설명된다. $n=10$이면 두 추정값의 표준오차가 모두 커서 대개 이 경우에 해당한다.

    **2단계 — 모형 적합도를 본다.** 두 추정값으로 각각 감마 밀도를 그려 히스토그램이나 경험분포와 겹쳐 본다. Q-Q 그림이 더 예민하다. 한쪽이 뚜렷이 나쁘면 그쪽을 버린다.

    **3단계 — 이상치를 확인한다.** 적률법은 $s^2$을 쓰므로 큰 값 하나에 크게 흔들린다. 최댓값을 빼고 다시 계산해 보아 적률법 추정값만 크게 움직이면 그것이 원인이다.

    **4단계 — 원칙적으로는 MLE.** 감마 모형이 맞다면 MLE가 점근효율적이고, $n=10$처럼 작은 표본에서도 대개 적률법보다 MSE가 작다. 다만 MLE도 소표본에서 $\alpha$를 과대추정하는 편향이 알려져 있어, 편향 보정($\hat\alpha_{\text{corr}} = \hat\alpha(n-3)/n$ 꼴)을 고려할 만하다.

    **5단계 — 그러나 모형을 의심한다.** 두 방법의 차이가 크고 적합도도 나쁘면 **감마분포 자체가 틀렸을** 가능성이 높다. 로그정규, 와이불, 역가우스를 함께 적합해 AIC로 비교한다.

    **보고.** $n=10$에서는 어느 방법을 쓰든 추정값의 신뢰구간이 매우 넓다. **점추정값의 차이를 논하기 전에 구간을 먼저 보여 주는 것**이 정직하다. 두 구간이 크게 겹친다면 "방법에 따라 다르다"는 논의 자체가 과잉 해석이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**혼합분포**의 모수를 적률법으로 추정하는 방법을 두 성분 정규혼합에서 설명하라. 피어슨이 이 문제를 어떻게 풀었는지와 현대적 방법을 비교하라.

</div>

??? success "풀이"
    **모형.** $f(x) = \pi\phi(x;\mu_1,\sigma_1^2)+(1-\pi)\phi(x;\mu_2,\sigma_2^2)$로 모수가 **5개**($\pi,\mu_1,\mu_2,\sigma_1^2,\sigma_2^2$)다.

    **적률법.** 5개 모수이므로 1차부터 5차까지의 적률을 맞춰야 한다.

    $$
    m_k = \pi\,E[X_1^k]+(1-\pi)E[X_2^k], \qquad k=1,\dots,5
    $$

    **피어슨의 방법(1894).** 그는 이 연립방정식을 대수적으로 정리해 **9차 다항방정식** 하나로 줄였다. 게의 이마 너비 자료가 두 종의 혼합인지를 판정하려는 것이 동기였고, 손으로 그 방정식을 풀었다. 통계사에서 혼합모형의 출발점으로 꼽히는 작업이다.

    **문제점.**

    - **5차 적률을 써야 한다.** 표집변동이 엄청나서 추정값이 극도로 불안정하다. $n$이 수백이어도 믿기 어렵다.
    - **9차 방정식의 실근이 여럿**일 수 있고, 그중 유효한 것을 고르는 기준이 명확하지 않다.
    - **모수공간을 벗어나기 쉽다.** 음의 분산이나 $\pi \notin [0,1]$이 흔히 나온다.

    **현대적 방법 — EM 알고리즘.**

    - **E 단계**: 각 관측값이 성분 1에 속할 사후확률 $\gamma_i = \dfrac{\pi\phi(x_i;\mu_1,\sigma_1^2)}{f(x_i)}$를 계산한다.
    - **M 단계**: $\gamma_i$를 가중치로 삼아 가중평균·가중분산으로 모수를 갱신하고 $\hat\pi = \bar\gamma$로 둔다.

    **비교.**

    | | 피어슨 적률법 | EM |
    |---|---|---|
    | 안정성 | 매우 나쁨 | 좋음 |
    | 고차 적률 | 5차까지 필요 | 불필요 |
    | 다봉성 | 근 선택 문제 | 초기값 문제(비슷한 성격) |
    | 성분 수 확장 | 사실상 불가능 | 용이 |

    **그래도 적률법의 쓸모.** EM의 **초기값**으로 쓸 수 있고, 성분이 잘 분리된 경우에는 빠르게 합리적인 답을 준다. 또 고차 적률을 쓰지 않는 변형(분위수 기반, 특성함수 기반)이 계산 통계에서 다시 연구되고 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
적률법을 **모의실험으로 검증**하는 절차를 적어라. 특히 편향과 분산이 표본크기와 함께 어떻게 변하는지 보려면 무엇을 그려야 하는가?

</div>

??? success "풀이"
    **절차.**

    ```python
    rng = np.random.default_rng(0)
    alpha_true, beta_true = 2.0, 3.0
    ns = [10, 20, 50, 100, 200, 500, 1000]
    B = 5000
    for n in ns:
        est = np.empty((B, 2))
        for b in range(B):
            x = rng.gamma(alpha_true, beta_true, n)
            m, v = x.mean(), x.var(ddof=1)
            est[b] = [m**2 / v, v / m]          # (alpha_hat, beta_hat)
        bias = est.mean(axis=0) - [alpha_true, beta_true]
        sd = est.std(axis=0, ddof=1)
        print(n, bias, sd)
    ```

    **그려야 할 것.**

    1. **$\log n$ 대 $\log|\text{편향}|$.** 기울기가 편향의 수렴 속도를 준다. 적률법의 편향은 보통 $O(1/n)$이므로 **기울기 $-1$인 직선**이 나와야 한다. 그렇지 않으면 일치성이 의심스럽다.
    2. **$\log n$ 대 $\log(\text{표준편차})$.** 기울기가 $-1/2$이어야 한다. 정칙 문제의 표준 속도다.
    3. **$n\cdot\operatorname{Var}$ 대 $n$.** 이 값이 상수로 안정되면 그 상수가 점근분산이고, 이론값과 맞는지 확인할 수 있다. 가장 예민한 진단이다.
    4. **각 $n$에서의 추정값 분포.** 상자그림이나 히스토그램. 치우침이 언제 사라지는지, 정규근사가 언제부터 통하는지를 눈으로 본다.
    5. **MLE와의 MSE 비.** $n$에 대해 그리면 점근상대효율로 수렴하는지 볼 수 있다.

    **주의할 점.**

    - **모수공간을 벗어난 추정값을 어떻게 처리했는지 반드시 기록한다.** 버리면 편향이 왜곡되고, 자르면 그것도 하나의 추정량이다.
    - **$B$를 충분히 크게.** 편향은 분산보다 작은 양이므로 정밀하게 재려면 $B$가 커야 한다. 편향의 표준오차가 $\operatorname{SD}/\sqrt B$임을 기억한다.
    - **여러 참값에서 반복한다.** $\alpha=2$에서의 결론이 $\alpha=0.5$나 $\alpha=20$에서도 성립하는지 확인한다. 감마분포는 $\alpha$에 따라 모양이 크게 달라진다.

---

## 정리하며

세 분포에 적률법을 적용하며 **대수적 난이도가 올라가는 순서**를 보았다.

- **정규분포**가 가장 쉽다. 모수가 곧 처음 두 적률이라 $\hat\mu=\bar X$, $\hat\sigma^2=M_2$ 가 바로 나온다.
- **감마분포**는 평균 $\alpha\theta$ 와 분산 $\alpha\theta^2$ 을 연립해 풀면 $\hat\theta=M_2/\bar X$, $\hat\alpha=\bar X^2/M_2$ 를 얻는다. 최대가능도는 디감마함수가 들어가 수치해가 필요하므로, **여기서 적률법의 계산 이점이 뚜렷하다.**
- **베타분포**는 대수가 가장 번거롭지만 여전히 닫힌 형태가 나온다.
- **$M_2$ 가 $1/n$ 로 나눈다는 점이 중요하다.** 적률법은 모집단 적률 $\mathbb{E}[(X-\mu)^2]$ 에 대응하는 표본량을 맞추는 것이므로, 베셀 보정한 $S^2$ 이 아니라 $M_2$ 를 쓴다. **따라서 적률법 분산추정량은 편향되어 있다.**
- **추정값이 모수공간을 벗어날 수 있다.** 감마·베타에서 표본 적률 조합에 따라 음수 형상모수가 나올 수 있으며, 최대가능도에서는 생기지 않는 문제다.

다음 절 **일반화 적률법**으로 넘어간다. 적률 조건이 모수보다 많을 때 그 정보를 버리지 않고 모두 쓰는 방법이다.
