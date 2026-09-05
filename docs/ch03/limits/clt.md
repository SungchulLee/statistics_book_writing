# 중심극한정리

## 개요

**중심극한정리(CLT)** 는 확률과 통계 전체에서 가장 중요한 결과 중 하나다. 모집단의 평균과 분산이 유한하기만 하면, 충분히 많은 i.i.d. 확률변수의 표본평균이 갖는 표본분포는 원래 분포가 무엇이든 **상관없이** 근사적으로 정규분포를 따른다는 것이다.

---

## 중심극한정리의 진술

$X_1, X_2, \ldots, X_n$이 평균 $\mu$, 분산 $\sigma^2$인 i.i.d. 확률변수이면 표준화된 표본평균이 표준정규분포로 분포수렴한다.

$$
\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty
$$

동등하게, $n$이 크면 표본평균이 근사적으로 정규분포를 따른다.

$$
\bar{X} \sim N\left(\mu, \frac{\sigma^2}{n}\right)
$$

합 $S_n = \sum_{i=1}^n X_i$의 관점에서는

$$
S_n \approx N(n\mu, \, n\sigma^2)
$$

이다.

---

## 큰수의 법칙에서 중심극한정리로

큰수의 법칙은 표본평균이 **어디로** 수렴하는지를 알려준다: $\bar{X} \to \mu$. 중심극한정리는 $\mu$ 주위의 변동이 **얼마나 빨리**, **어떤 모양으로** 움직이는지를 알려준다.

$$
\frac{\sqrt{n}}{\sigma}(\bar{X} - \mu) = \frac{S_n - n\mu}{\sqrt{n\sigma^2}} \xrightarrow{d} N(0, 1)
$$

큰수의 법칙은 편차 $\bar{X} - \mu \to 0$이라고 말한다. $\sqrt{n}$으로 척도를 다시 맞추면 중심극한정리가 이 편차들이 자명하지 않은 정규 구조를 가짐을 드러낸다.

---

## 실무 지침

### 최소 표본 크기 (n >= 30)

흔히 인용되는 어림법칙은 $n \geq 30$이면 중심극한정리 근사가 성립할 만큼 "충분히 크다"는 것이다.

- 모집단이 대략 대칭이면 $n \approx 15$–20으로도 충분할 수 있다.
- 모집단이 치우쳤거나 꼬리가 두꺼우면 $n \geq 40$–50이 필요할 수 있다.
- 30이라는 수는 정리가 아니라 **관례**다.

### 비율에 대한 정규근사

표본비율(이항 자료)을 다룰 때 중심극한정리는 다음을 요구한다.

$$
np \geq 5 \quad \text{and} \quad n(1-p) \geq 5
$$

일부 교과서는 더 엄격한 $np \geq 10$, $n(1-p) \geq 10$을 쓴다.

### 독립성을 위한 10% 조건

크기 $N$인 유한 모집단에서 비복원으로 표집하면 뽑기가 독립이 아니다. 유한모집단 수정은 다음과 같다.

$$
\text{Var}(\bar{X}_n) = \frac{\sigma^2}{n} \cdot \frac{N - n}{N - 1}
$$

표집비율이 작으면 이 수정은 무시할 만하다.

$$
\frac{n}{N} \leq 10\%
$$

이것이 성립하면 표본을 i.i.d.로 안전하게 다룰 수 있다.

---

## 정규근사

### 이항분포에 대한 근사

$n$이 큰 $X \sim \text{Binomial}(n, p)$에 대해

$$
X \approx N(np, \, np(1-p))
$$

이다. **연속성 수정**을 적용하면

$$
P(X \leq k) \approx P\left(Z \leq \frac{k + 0.5 - np}{\sqrt{np(1-p)}}\right)
$$

이다.

### 포아송분포에 대한 근사

$\lambda$가 큰 $X \sim \text{Poisson}(\lambda)$에 대해

$$
X \approx N(\lambda, \lambda)
$$

이다.

---

## 작동하는 중심극한정리: 균등분포와 지수분포

중심극한정리는 원래 분포와 무관하게 작동한다. 그러나 **수렴 속도**는 분포의 모양에 달려 있다.

- **대칭 분포**(예: 균등분포)는 정규성으로 아주 빠르게 수렴한다.
- **치우친 분포**(예: 지수분포)는 더 느리게 수렴한다. 극단값의 영향이 더 두드러져 더 큰 표본이 필요하다.

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def demonstrate_clt(distribution_type, sample_size, n_simulations=10_000):
    """Demonstrate CLT convergence for a given distribution."""
    np.random.seed(0)

    if distribution_type == 'uniform':
        data = np.mean(stats.uniform().rvs((sample_size, n_simulations)), axis=0)
        label = 'Uniform(0,1)'
    elif distribution_type == 'exponential':
        data = np.mean(stats.expon().rvs((sample_size, n_simulations)), axis=0)
        label = 'Exponential(1)'

    mu, sigma = data.mean(), data.std()

    fig, ax = plt.subplots(figsize=(12, 3))
    _, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.3, color='blue',
                         label=f'Sample Means (n={sample_size})')
    ax.plot(bins, stats.norm(mu, sigma).pdf(bins), '--r', lw=2, label='Normal PDF')
    ax.set_title(f'CLT: Sample Means from {label}')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()

# Demonstrate with both distributions
demonstrate_clt('uniform', sample_size=5)
demonstrate_clt('exponential', sample_size=5)
```

---

## 응용

중심극한정리는 여러 핵심 통계 절차를 떠받친다.

- **가설검정:** $z$-검정과 $t$-검정은 검정통계량의 표본분포가 근사적으로 정규라고 가정한다.
- **신뢰구간:** 정규분포 분위수로 구성하며, 중심극한정리가 이를 정당화한다.
- **품질관리:** 제품 측정값의 표본평균이 기준을 충족하는지 평가한다.

---

## 종합

정규근사를 적용하기 전에 다음 조건들을 확인하라.

| 조건 | 어림법칙 |
|:---|:---|
| 표본 크기 | $n \geq 30$ (모집단이 거의 정규가 아닌 한) |
| 비율 | $np \geq 5$ 그리고 $n(1-p) \geq 5$ |
| 유한모집단 표집 | $n/N \leq 10\%$ |

이들은 엄밀한 정리가 아니라 이상적인 수학의 세계와 실제 자료 분석을 잇는, 널리 채택된 **실무 지침**이다.

---

## 핵심 요약

- 중심극한정리는 모집단 분포와 무관하게 $n$이 크면 표본평균이 근사적으로 정규임을 보장한다.
- 수렴 속도는 원래 분포의 왜도에 달려 있다.
- 실무 조건($n \geq 30$, 성공/실패 도수, 10% 규칙)이 근사의 신뢰성을 확보해 준다.
- 중심극한정리는 신뢰구간, 가설검정, 그리고 응용통계 대부분의 이론적 척추다.

## 연습문제

**연습문제 1.**
어떤 기계가 $\mu = 500$ ml, $\sigma = 10$ ml로 병을 채운다(정규분포가 아니다). (a) 중심극한정리에 따른 $\bar X_{36}$의 분포는? (b) $P(\bar X_{36} > 503)$은? (c) $P(|\bar X_n - 500| < 2) \ge 0.95$가 되려면 $n$은?

??? success "연습문제 1 풀이"
    (a) $\bar X_{36} \approx N(500, 100/36) = N(500, 2.778)$. 표준오차 $= 10/6 \approx 1.67$.

    (b) $Z = (503 - 500)/(10/6) = 1.8$이므로 $P(Z > 1.8) \approx 1 - 0.9641 = 0.0359$. 약 3.6%다.

    (c) $2/(\sigma/\sqrt n) \ge z_{0.025} = 1.96 \Rightarrow 2\sqrt n / 10 \ge 1.96 \Rightarrow n \ge 96.04$이어야 하므로 $n \ge 97$이다.

---

**연습문제 2.**
평균 0, 분산 1이고 0의 근방에서 적률생성함수 $M(t)$가 유한한 i.i.d. $X_i$에 대해 **적률생성함수 방법으로 중심극한정리를 증명하라**. $\sqrt n \bar X_n$의 적률생성함수가 $e^{t^2/2}$로 수렴함을 보여라.

??? success "연습문제 2 풀이"
    표준화된 합: $Z_n = \sqrt n \bar X_n = (X_1 + \cdots + X_n)/\sqrt n$.

    그 적률생성함수: $M_{Z_n}(t) = \mathbb{E}[e^{t Z_n}] = \prod_i \mathbb{E}[e^{t X_i / \sqrt n}] = [M(t/\sqrt n)]^n$.

    $M(t/\sqrt n)$을 0 주위로 전개한다. $\mu = 0$, $\sigma^2 = 1$을 쓰면 $M(u) = 1 + \mu u + (\sigma^2 + \mu^2) u^2/2 + O(u^3) = 1 + u^2/2 + O(u^3)$이다.

    따라서 $M(t/\sqrt n) = 1 + t^2/(2n) + O(n^{-3/2})$이다.

    $n$제곱을 취하면 ($(1 + a/n + o(1/n))^n \to e^a$를 이용해) $n \to \infty$일 때 $M_{Z_n}(t) = (1 + t^2/(2n) + O(n^{-3/2}))^n \to e^{t^2/2}$이다.

    극한 적률생성함수 $e^{t^2/2}$이 $N(0, 1)$의 것이다. 적률생성함수 수렴 정리에 의해 $Z_n \xrightarrow{d} N(0, 1)$이다. $\square$

    참고: 이 증명은 근방에서 적률생성함수가 유한할 것을 요구하므로 꼬리가 두꺼운 일부 분포를 배제한다. ($\phi(t) = \mathbb{E}[e^{itX}]$를 쓰는) 특성함수 증명은 분산이 유한한 모든 분포로 일반화된다.

---

**연습문제 3.**
심하게 치우친 분포에서 **중심극한정리의 느린 수렴을 보여라.** $X_i$가 평균 1인 지수분포를 따른다고 하자. $n = 30$일 때 $\bar X_n$의 왜도는 얼마인가? 정규 극한의 대칭성과 비교하라.

??? success "연습문제 3 풀이"
    Exponential(1)의 왜도는 2다(오른쪽으로 치우침). i.i.d. 합에서 *표준화된 합*의 왜도는 $\gamma_n = \gamma_1 / \sqrt n$으로 줄어든다.

    $$
    \mathrm{Skew}(\bar X_n) = \frac{\mathrm{Skew}(X)}{\sqrt n}
    $$

    $n = 30$이면 $\mathrm{Skew}(\bar X_{30}) = 2/\sqrt{30} \approx 0.365$이다.

    이는 여전히 0에서 상당히 떨어진 값이다. 정규 극한의 왜도는 0이다. $n = 30$에서 지수분포의 표본평균은 눈에 띄게 오른쪽으로 치우쳐 있다. 실무적 함의: 관례적인 $n \ge 30$ 어림법칙은 심하게 치우친 분포에는 *충분하지 않다*. 실제로는 $n \ge 100$ 이상이거나 대안 기법(부트스트랩, 정확법)이 필요하다.

    수렴 속도는 **베리–에센 정리**가 지배한다: $\sup_x |F_{\bar X_n}(x) - \Phi(x)| \le C \cdot \mathbb{E}|X|^3 / (\sigma^3 \sqrt n)$. 분자의 3차 적률이 왜도/비대칭성을 포착한다. 심하게 치우친 분포는 3차 절대적률이 커서 중심극한정리의 수렴이 느리다.

---

**연습문제 4.**
**다변량 중심극한정리.** $\mathbf X_i \in \mathbb{R}^d$가 평균 $\boldsymbol\mu$, 공분산 $\boldsymbol\Sigma$인 i.i.d.라 하자. 다변량 중심극한정리를 진술하고, 그것이 다변량 정규분포에 근거한 신뢰타원체를 왜 정당화하는지 설명하라.

??? success "연습문제 4 풀이"
    **다변량 중심극한정리:**

    $$
    \sqrt n (\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} N_d(\mathbf 0, \boldsymbol\Sigma)
    $$

    이며 수렴은 $\mathbb{R}^d$에서의 분포수렴(모든 성분의 결합분포)이다.

    증명 개요: **크라메르–월드 장치**를 쓴다. 다변량 수렴은 모든 선형 사영이 (일변량으로) 수렴할 때에 한해 성립한다. 임의의 $\mathbf a \in \mathbb{R}^d$에 대해 스칼라 사영에 일변량 중심극한정리를 적용하면 $\sqrt n \, \mathbf a^T(\bar{\mathbf X}_n - \boldsymbol\mu) \xrightarrow{d} N(0, \mathbf a^T \boldsymbol\Sigma \mathbf a)$이고, 이로부터 $N_d(\mathbf 0, \boldsymbol\Sigma)$로의 다변량 수렴이 따라온다.

    **신뢰타원체의 정당화:** $\bar{\mathbf X}_n \approx N_d(\boldsymbol\mu, \boldsymbol\Sigma/n)$이면 $n(\bar{\mathbf X}_n - \boldsymbol\mu)^T \boldsymbol\Sigma^{-1}(\bar{\mathbf X}_n - \boldsymbol\mu) \approx \chi^2_d$이다. 집합 $\{\boldsymbol\mu : n(\bar{\mathbf X}_n - \boldsymbol\mu)^T \boldsymbol\Sigma^{-1}(\bar{\mathbf X}_n - \boldsymbol\mu) \le \chi^2_{d, 0.95}\}$은 참 평균을 95%의 점근 확률로 덮는 타원체다. 호텔링의 $T^2$ 검정과 다변량 신뢰영역이 모두 이 위에 서 있다.

---

**연습문제 5.**
**린데베르그 중심극한정리**는 동일분포가 아니어도 독립이기만 하면 되는 확률변수를 허용한다. **린데베르그 조건**을 진술하고 언제 성립하는지 설명하라.

??? success "연습문제 5 풀이"
    $X_1, X_2, \ldots$가 독립이고(동일분포일 필요는 없다) $\mathbb{E}[X_i] = 0$, $\mathrm{Var}(X_i) = \sigma_i^2$, $s_n^2 = \sum_{i=1}^n \sigma_i^2$이라 하자. **린데베르그 조건**은 다음과 같다.

    $$
    \forall\, \varepsilon > 0: \quad \frac{1}{s_n^2}\sum_{i=1}^n \mathbb{E}\!\left[X_i^2 \mathbf 1(|X_i| > \varepsilon s_n)\right] \to 0
    $$

    직관: 개별 $X_i$의 "꼬리"(즉 $|X_i|$가 $\varepsilon s_n$을 넘는 부분)가 기여하는 총 분산이 무시할 만해진다는 것이다. 어느 한 $X_i$도 분산을 지배하지 않는다.

    **성립하는 경우:** (1) 분산이 유한한 i.i.d.인 경우 자명하게 성립한다. (2) $X_i$가 균등하게 유계이고 $s_n \to \infty$인 경우. (3) 최대 분산이 $\max_i \sigma_i^2 / s_n^2 \to 0$을 만족하는 임의의 열.

    **실패하는 경우:** 한 항이 총 분산에서 사라지지 않는 몫을 차지하면(예: 가우시안 $Y$에 대해 $X_n = n \cdot Y$여서 $X_n$이 지배하는 경우) 극한이 가우시안이 아니라 안정분포가 된다.

    린데베르그 중심극한정리는 i.i.d. 중심극한정리를 포괄하며, 독립이지만 이질적인 기여들의 합에 적용된다. 동일분포가 아닌 오차를 갖는 회귀에 핵심적이다.

---

**연습문제 6.**
중심극한정리는 **유한한 분산**을 요구한다. 그 이유와, 분산이 무한할 때(꼬리가 두꺼운 분포) 어떻게 되는지 논하라. **안정분포**와 **알파-안정 중심극한정리**를 언급하라.

??? success "연습문제 6 풀이"
    **유한한 분산이 필수인 이유:** 표준화 $(S_n - n\mu)/\sqrt{n\sigma^2}$이 암묵적으로 $\sigma^2 < \infty$를 가정한다. 분산이 무한하면 이 표준화가 정의되지 않는다. 개별 기여의 "무시가능성"을 확립할 수 없으므로 린데베르그–펠러 틀이 무너진다.

    **꼬리가 두꺼운 경우의 행동:** $\alpha < 2$인 **안정분포** $S_\alpha(\sigma, \beta)$의 흡인 영역에 있는 분포는 거듭제곱 법칙 꼬리 $P(|X| > x) \sim x^{-\alpha}$를 갖는다. $\alpha \le 2$이면 분산이 무한하고 $\alpha \le 1$이면 평균이 무한하다.

    **알파-안정 중심극한정리:** 꼬리 지수가 $0 < \alpha \le 2$인 i.i.d. $X_i$에 대해 정규화된 합

    $$
    \frac{S_n - n a_n}{n^{1/\alpha}} \xrightarrow{d} S_\alpha(\sigma, \beta)
    $$

    이 $\alpha$-안정분포로 수렴한다. 정규화가 $\sqrt n$이 아니라 $n^{1/\alpha}$임에 주목하라. $\alpha = 2$이면 가우시안 중심극한정리가 복원되고, $\alpha = 1$이면 코시 극한을 얻으며, $\alpha < 1$이면 평균조차 수렴하지 않는다.

    **실무적 함의:** 금융 수익률은 흔히 $\alpha \approx 1.5$–$1.8$이다(꼬리는 두껍지만 분산이 유한하므로 가우시안 중심극한정리가 느리게나마 적용된다). 네트워크 패킷 크기, 파일 크기, 대기 지연은 흔히 $\alpha < 2$여서(진짜로 꼬리가 두꺼워서) 가우시안 기반 신뢰구간이 타당하지 않다. 알맞은 도구는 두꺼운 꼬리를 고려하는 것들이다. 조심스러운 부트스트랩, 분위수 추정량, 알파-안정 모형 등이다.
