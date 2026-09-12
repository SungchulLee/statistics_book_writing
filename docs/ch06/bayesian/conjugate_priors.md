# 켤레 사전분포

베이즈 추론에서 사후분포를 계산하려면 적분 $\int f(x \mid \theta) \pi(\theta) \, d\theta$를 평가해야 하는데, 임의의 사전분포–가능도 조합에서는 이것이 어려운 경우가 많다. 켤레 사전분포는 우아한 지름길을 제공한다. 사전분포가 가능도에 맞춰진 특정 분포족에 속하면 사후분포도 반드시 같은 족에 속하며 모수만 갱신된다. 이 닫힌 형태의 결과 덕분에 단순한 모형에서는 수치적분이나 MCMC가 필요 없어지며, 켤레 사전분포는 다루기 쉬운 베이즈 분석의 토대가 된다.

<div class="defn" markdown>

**정의 1.** [켤레 사전분포]

$x = (x_1, \ldots, x_n)$을 가능도가 $f(x \mid \theta)$인 분포에서 얻은 i.i.d. 표본이라 하자. 사전분포족 $\mathcal{F}$가 가능도 $f(x \mid \theta)$에 대해 **켤레**라는 것은, 모든 사전분포 $\pi(\theta) \in \mathcal{F}$에 대해 사후분포 $\pi(\theta \mid x) \in \mathcal{F}$임을 뜻한다.

다시 말해 자료를 관측하면 사전분포의 모수만 바뀔 뿐 함수 형태는 바뀌지 않는다. "갱신에 대해 닫혀 있다"는 이 성질이 켤레족을 실무에서 유용하게 만든다.

</div>

## 흔한 켤레 쌍

다음 표는 가장 자주 마주치는 켤레 쌍을 정리한 것이다. 각 경우에서 Gamma 분포는 비율 모수화를 사용하며, $\text{Gamma}(\alpha, \beta)$의 밀도는 $\theta^{\alpha - 1} e^{-\beta \theta}$에 비례한다.

| 가능도 | 켤레 사전분포 | 사후분포 |
|---|---|---|
| Bernoulli/Binomial | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) |
| Normal ($\sigma^2$ 알려짐) | Normal($\mu_0, \sigma_0^2$) | Normal$\!\left(\frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2},\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}\right)$ |
| Exponential | Gamma($\alpha, \beta$) | Gamma($\alpha + n, \beta + \sum x_i$) |

모든 행에서 사후분포가 사전분포와 같은 분포 형태를 가지며, 모수만 자료의 요약통계량(표본합, 표본크기, 표본평균)으로 갱신됨에 주목하라.

## Beta-Binomial 켤레 쌍의 유도

켤레성이 왜 작동하는지 보기 위해 Beta-Binomial 경우를 살펴보자. $X \mid p \sim \text{Binomial}(n, p)$이고 $p \sim \text{Beta}(\alpha, \beta)$라 하자. 사전밀도는

$$
\pi(p) \propto p^{\alpha - 1}(1 - p)^{\beta - 1}
$$

이고 $n$번의 시행에서 $k$번 성공했을 때의 가능도는

$$
f(k \mid p) \propto p^{k}(1 - p)^{n - k}
$$

베이즈 정리에 의해 사후분포는 이 둘의 곱에 비례한다:

$$
\pi(p \mid k) \propto p^{\alpha - 1}(1 - p)^{\beta - 1} \cdot p^{k}(1 - p)^{n - k} = p^{(\alpha + k) - 1}(1 - p)^{(\beta + n - k) - 1}
$$

이는 $\text{Beta}(\alpha + k, \beta + n - k)$ 분포의 핵이며 켤레성이 확인된다.

!!! example "실제로 쓰는 Beta-Binomial"
    어떤 동전의 앞면 확률 $p$가 미지라고 하자. $[0, 1]$의 모든 값에 같은 가중을 주는 균등 사전분포 $p \sim \text{Beta}(1, 1)$에서 시작한다. $n = 10$번 던져 $k = 7$번 앞면을 관측한 뒤 사후분포는

    $$
    p \mid k = 7 \sim \text{Beta}(1 + 7, 1 + 3) = \text{Beta}(8, 4)
    $$

    사후평균은 $8 / 12 \approx 0.667$로, 사전평균 $0.5$와 표본비율 $0.7$ 사이에 놓인다. 자료가 쌓일수록 사후분포는 $p$의 참값 주위로 모인다.

## 켤레 사전분포를 언제 쓰는가

켤레 사전분포는 다음 경우에 가장 유용하다:

- **해석적 다루기 쉬움**이 필요할 때. 예를 들어 새 자료가 시간에 따라 들어오고 사후분포를 빠르게 다시 계산해야 하는 순차적 갱신 상황이 그렇다.
- **해석 가능성**이 중요할 때. 사전분포의 모수가 흔히 "가상 관측값"이라는 자연스러운 해석을 갖는다(예: Beta 사전분포의 $\alpha$와 $\beta$는 사전의 성공 횟수와 실패 횟수처럼 작동한다).

다만 켤레 사전분포는 사전분포족의 선택을 가능도에 맞추도록 제한하므로, 실제 사전 믿음을 언제나 표현하지는 못한다. 복잡한 모형이거나 사전분포의 유연성이 중요할 때는 켤레가 아닌 사전분포와 MCMC나 변분추론 같은 계산 방법을 함께 쓰는 편이 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
Beta 분포는 Binomial 가능도의 켤레 사전분포이다. 사전분포가 $\text{Beta}(2, 5)$이고 10번의 시행에서 3번 성공을 관측했을 때 사후분포와 사후평균을 구하라.

</div>

??? success "풀이"
    $\text{Beta}(\alpha, \beta)$ 사전분포와 $n$번의 시행에서 $k$번 성공했을 때 사후분포는 $\text{Beta}(\alpha + k, \beta + n - k)$이다.

    여기서 $\alpha = 2$, $\beta = 5$, $k = 3$, $n = 10$이므로:

    $$
    \text{Posterior} = \text{Beta}(2 + 3, 5 + 7) = \text{Beta}(5, 12)
    $$

    사후평균은:

    $$
    E[p \mid \text{data}] = \frac{\alpha + k}{\alpha + k + \beta + n - k} = \frac{5}{5 + 12} = \frac{5}{17} \approx 0.294
    $$

    이는 사전평균 $2/7 \approx 0.286$과 표본비율 $3/10 = 0.3$ 사이의 절충이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Gamma 분포가 Poisson 가능도의 켤레 사전분포임을 보여라. $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$이고 $\lambda \sim \text{Gamma}(\alpha, \beta)$일 때 $\lambda$의 사후분포를 유도하라.

</div>

??? success "풀이"
    $n$개 관측값에 대한 Poisson 가능도는:

    $$
    L(\lambda) = \prod_{i=1}^n \frac{\lambda^{x_i} e^{-\lambda}}{x_i!} \propto \lambda^{\sum x_i} e^{-n\lambda}
    $$

    (비율 모수화를 쓴) Gamma$(\alpha, \beta)$ 사전분포는:

    $$
    \pi(\lambda) \propto \lambda^{\alpha - 1} e^{-\beta\lambda}
    $$

    사후분포는:

    $$
    \pi(\lambda \mid \mathbf{x}) \propto \lambda^{\sum x_i} e^{-n\lambda} \cdot \lambda^{\alpha - 1} e^{-\beta\lambda} = \lambda^{\alpha + \sum x_i - 1} e^{-(\beta + n)\lambda}
    $$

    이는 $\text{Gamma}(\alpha + \sum x_i, \beta + n)$ 분포의 핵이며 켤레성이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
사전분포가 "무정보"이거나 "약한 정보"라는 것이 무슨 뜻인지 설명하라. $\sigma^2$이 알려진 정규분포의 평균 $\mu$에 대해 약한 정보의 켤레 사전분포 예를 들라.

</div>

??? success "풀이"
    **무정보**(또는 "막연한") 사전분포는 사전 지식을 최소로 담아 자료가 사후분포를 지배하도록 하려는 것이다. **약한 정보**의 사전분포는 특정 값을 강하게 선호하지 않으면서 모수를 합리적인 범위로 제약한다.

    $\sigma^2$이 알려진 정규분포의 평균 $\mu$에 대해 켤레 사전분포는 $\mu \sim N(\mu_0, \tau^2)$이다. 약한 정보를 주려면 $\tau$를 자료의 척도에 비해 매우 크게 잡는다. 예를 들어 $\mu$가 $-100$에서 $100$ 사이일 것으로 예상하면 $\mu_0 = 0$, $\tau = 100$을 쓸 수 있다. 이 사전분포는 $n$이 어느 정도만 되어도 사후분포에 거의 영향을 주지 않지만 극단적인 추정값은 막아 준다.

    $\tau \to \infty$인 극한에서는 비정상(improper) 평평한 사전분포 $\pi(\mu) \propto 1$을 얻으며 이는 무정보이다. 그러면 사후평균이 MLE $\bar{x}$와 같아진다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
켤레 Normal-Gamma 사전분포를 쓰면 정규분포의 평균과 정밀도 $(\mu, \tau)$에 대한 사후분포도 Normal-Gamma이다. 켤레 사전분포가 계산상 편리한 이유를 직관적으로 서술하고 한계를 하나 들라.

</div>

??? success "풀이"
    **편리함:** 켤레 사전분포가 계산상 편리한 이유는 사후분포가 사전분포와 같은 분포족에 속하고 모수만 바뀌기 때문이다. 이는 다음을 뜻한다:

    - 사후분포를 닫힌 형태로 쓸 수 있다(수치적분이 필요 없다).
    - 새 자료로 갱신하는 것이 초모수를 갱신하는 것으로 끝난다: $(\alpha, \beta) \to (\alpha', \beta')$.
    - 순차적 갱신이 아주 쉽다. 관측값이 하나 들어올 때마다 초모수를 조금씩 갱신하면 된다.

    **한계:** 켤레 사전분포가 분석자의 실제 사전 믿음을 정확히 표현하지 못할 수 있다. 예를 들어 이항 비율에 대한 Beta 사전분포는 단봉(또는 U자형)인데, 분석자는 참 확률이 (0.2 근처 아니면 0.8 근처처럼) 두 봉우리를 갖는다고 믿을 수도 있다. 믿음을 켤레족에 억지로 맞추면 계산의 편의를 위해 표현력을 희생하는 셈이다. 현대의 MCMC 방법은 임의의 사전분포를 허용하므로 켤레성의 필요를 줄여 준다.

---

## 정리하며

켤레 사전분포는 **갱신에 대해 닫혀 있는** 분포족이다. 사전분포와 사후분포가 같은 족에 속하고 모수만 바뀐다.

| 가능도 | 켤레 사전분포 | 갱신 |
|---|---|---|
| 베르누이 · 이항 | 베타 | $a+\sum x_i$, $b+n-\sum x_i$ |
| 포아송 | 감마 | $\alpha+\sum x_i$, $\beta+n$ |
| 정규(분산 기지) | 정규 | 정밀도의 가중평균 |
| 정규(평균 기지) | 역감마 | — |

- **덕분에 적분이 필요 없다.** 수치적분이나 MCMC 없이 사후분포를 손으로 적을 수 있으며, 그것이 단순한 모형에서 켤레족을 쓰는 이유다.
- **갱신이 요약통계량만으로 이루어진다.** 표본합·표본크기가 사전 모수에 더해지는 꼴이며, **사전분포를 "가상의 관측"으로 읽을 수 있다.** $\text{Beta}(a,b)$ 는 성공 $a$ 번, 실패 $b$ 번을 미리 본 것과 같다.
- **충분통계량과 지수족이 그 배경이다.** 켤레 사전분포가 존재하는 것은 가능도가 지수족일 때이며, 우연이 아니다.
- **편의를 위한 선택이라는 점을 잊지 말 것.** 켤레족이 실제 믿음을 잘 표현하지 못하면 계산 편의를 위해 잘못된 사전분포를 쓰는 셈이 된다. 오늘날은 MCMC 로 임의의 사전분포를 다룰 수 있다.

다음 절 **최대사후확률 추정**으로 넘어간다. 사후분포를 하나의 수로 요약하는 방법이며, 정칙화와의 연결이 드러난다.
