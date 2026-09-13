# 켤레 사전분포

베이즈 추론에서 사후분포를 계산하려면 적분 $\int f(x \mid \theta) \pi(\theta) \, d\theta$를 평가해야 하는데, 임의의 사전분포–가능도 조합에서는 이것이 어려운 경우가 많다. 켤레 사전분포는 우아한 지름길을 제공한다. 사전분포가 가능도에 맞춰진 특정 분포족에 속하면 사후분포도 반드시 같은 족에 속하며 모수만 갱신된다. 이 닫힌 형태의 결과 덕분에 단순한 모형에서는 수치적분이나 MCMC가 필요 없어지며, 켤레 사전분포는 다루기 쉬운 베이즈 분석의 토대가 된다.

<div class="defn" markdown>

### 정의 1. 켤레 사전분포 { .dfn }

$x = (x_1, \ldots, x_n)$을 가능도가 $f(x \mid \theta)$인 분포에서 얻은 i.i.d. 표본이라 하자. 사전분포족 $\mathcal{F}$가 가능도 $f(x \mid \theta)$에 대해 **켤레**라는 것은, 모든 사전분포 $\pi(\theta) \in \mathcal{F}$에 대해 사후분포 $\pi(\theta \mid x) \in \mathcal{F}$임을 뜻한다.

다시 말해 자료를 관측하면 사전분포의 모수만 바뀔 뿐 함수 형태는 바뀌지 않는다. "갱신에 대해 닫혀 있다"는 이 성질이 켤레족을 실무에서 유용하게 만든다.

</div>

## 흔한 켤레 쌍

다음 표는 가장 자주 마주치는 켤레 쌍을 정리한 것이다. 각 경우에서 감마분포는 비율 모수화를 사용하며, $\text{Gamma}(\alpha, \beta)$의 밀도는 $\theta^{\alpha - 1} e^{-\beta \theta}$에 비례한다.

| 가능도 | 켤레 사전분포 | 사후분포 |
|---|---|---|
| Bernoulli/Binomial | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson | Gamma($\alpha, \beta$) | Gamma($\alpha + \sum x_i, \beta + n$) |
| Normal ($\sigma^2$ 알려짐) | Normal($\mu_0, \sigma_0^2$) | Normal$\!\left(\frac{\sigma^2 \mu_0 + n\sigma_0^2 \bar{x}}{\sigma^2 + n\sigma_0^2},\; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + n\sigma_0^2}\right)$ |
| Exponential | Gamma($\alpha, \beta$) | Gamma($\alpha + n, \beta + \sum x_i$) |

모든 행에서 사후분포가 사전분포와 같은 분포 형태를 가지며, 모수만 자료의 요약통계량(표본합, 표본크기, 표본평균)으로 갱신됨에 주목하라.

## 베타-이항 켤레 쌍의 유도

켤레성이 왜 작동하는지 보기 위해 베타-이항 경우를 살펴보자. $X \mid p \sim \text{Binomial}(n, p)$이고 $p \sim \text{Beta}(\alpha, \beta)$라 하자. 사전밀도는

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

!!! example "실제로 쓰는 베타-이항"
    어떤 동전의 앞면 확률 $p$가 미지라고 하자. $[0, 1]$의 모든 값에 같은 가중을 주는 균등 사전분포 $p \sim \text{Beta}(1, 1)$에서 시작한다. $n = 10$번 던져 $k = 7$번 앞면을 관측한 뒤 사후분포는

    $$
    p \mid k = 7 \sim \text{Beta}(1 + 7, 1 + 3) = \text{Beta}(8, 4)
    $$

    사후평균은 $8 / 12 \approx 0.667$로, 사전평균 $0.5$와 표본비율 $0.7$ 사이에 놓인다. 자료가 쌓일수록 사후분포는 $p$의 참값 주위로 모인다.

## 켤레 사전분포를 언제 쓰는가

켤레 사전분포는 다음 경우에 가장 유용하다:

- **해석적 다루기 쉬움**이 필요할 때. 예를 들어 새 자료가 시간에 따라 들어오고 사후분포를 빠르게 다시 계산해야 하는 순차적 갱신 상황이 그렇다.
- **해석 가능성**이 중요할 때. 사전분포의 모수가 흔히 "가상 관측값"이라는 자연스러운 해석을 갖는다(예: 베타 사전분포의 $\alpha$와 $\beta$는 사전의 성공 횟수와 실패 횟수처럼 작동한다).

다만 켤레 사전분포는 사전분포족의 선택을 가능도에 맞추도록 제한하므로, 실제 사전 믿음을 언제나 표현하지는 못한다. 복잡한 모형이거나 사전분포의 유연성이 중요할 때는 켤레가 아닌 사전분포와 MCMC나 변분추론 같은 계산 방법을 함께 쓰는 편이 낫다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
베타분포는 이항 가능도의 켤레 사전분포이다. 사전분포가 $\text{Beta}(2, 5)$이고 10번의 시행에서 3번 성공을 관측했을 때 사후분포와 사후평균을 구하라.

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
감마분포가 포아송 가능도의 켤레 사전분포임을 보여라. $X_1, \dots, X_n \overset{\text{iid}}{\sim} \text{Poisson}(\lambda)$이고 $\lambda \sim \text{Gamma}(\alpha, \beta)$일 때 $\lambda$의 사후분포를 유도하라.

</div>

??? success "풀이"
    $n$개 관측값에 대한 포아송 가능도는:

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
켤레 정규-감마 사전분포를 쓰면 정규분포의 평균과 정밀도 $(\mu, \tau)$에 대한 사후분포도 정규-감마이다. 켤레 사전분포가 계산상 편리한 이유를 직관적으로 서술하고 한계를 하나 들라.

</div>

??? success "풀이"
    **편리함:** 켤레 사전분포가 계산상 편리한 이유는 사후분포가 사전분포와 같은 분포족에 속하고 모수만 바뀌기 때문이다. 이는 다음을 뜻한다:

    - 사후분포를 닫힌 형태로 쓸 수 있다(수치적분이 필요 없다).
    - 새 자료로 갱신하는 것이 초모수를 갱신하는 것으로 끝난다: $(\alpha, \beta) \to (\alpha', \beta')$.
    - 순차적 갱신이 아주 쉽다. 관측값이 하나 들어올 때마다 초모수를 조금씩 갱신하면 된다.

    **한계:** 켤레 사전분포가 분석자의 실제 사전 믿음을 정확히 표현하지 못할 수 있다. 예를 들어 이항 비율에 대한 베타 사전분포는 단봉(또는 U자형)인데, 분석자는 참 확률이 (0.2 근처 아니면 0.8 근처처럼) 두 봉우리를 갖는다고 믿을 수도 있다. 믿음을 켤레족에 억지로 맞추면 계산의 편의를 위해 표현력을 희생하는 셈이다. 현대의 MCMC 방법은 임의의 사전분포를 허용하므로 켤레성의 필요를 줄여 준다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**지수족과 켤레 사전분포**의 일반 관계를 서술하라. 왜 지수족에는 언제나 켤레 사전분포가 존재하는가?

</div>

??? success "풀이"
    **지수족.** 밀도를

    $$
    f(x;\boldsymbol\eta) = h(x)\exp\left\{\boldsymbol\eta^\top\mathbf{T}(x)-A(\boldsymbol\eta)\right\}
    $$

    로 쓰자. $n$개 관측의 가능도는

    $$
    L(\boldsymbol\eta) \propto \exp\left\{\boldsymbol\eta^\top\sum_i\mathbf{T}(x_i)-nA(\boldsymbol\eta)\right\}
    $$

    이다. **자연통계량의 합과 표본크기만이 들어온다.**

    **켤레 사전분포.** 가능도와 같은 모양으로 두면 된다.

    $$
    \pi(\boldsymbol\eta;\boldsymbol\nu,\kappa) \propto \exp\left\{\boldsymbol\eta^\top\boldsymbol\nu-\kappa A(\boldsymbol\eta)\right\}
    $$

    그러면 사후분포가

    $$
    \pi(\boldsymbol\eta\mid x) \propto \exp\left\{\boldsymbol\eta^\top\left(\boldsymbol\nu+\sum_i\mathbf{T}(x_i)\right)-(\kappa+n)A(\boldsymbol\eta)\right\}
    $$

    로 **같은 족에 머물고**, 갱신이 단순한 덧셈이다.

    $$
    \boldsymbol\nu \to \boldsymbol\nu+\sum_i\mathbf{T}(x_i), \qquad \kappa\to\kappa+n
    $$

    **왜 언제나 존재하는가.** 위 구성이 기계적이기 때문이다. 가능도가 $\boldsymbol\eta$에 대해 지수-선형 꼴이므로, 같은 꼴의 사전분포를 쓰면 지수의 인수가 더해질 뿐 모양이 바뀌지 않는다. 적분이 유한하도록 $(\boldsymbol\nu,\kappa)$를 제한하면 정상 분포가 된다.

    **해석.** $\kappa$가 **사전 표본크기**이고 $\boldsymbol\nu$가 **사전 자료의 충분통계량 합**이다. 베타-이항의 $(a+b, a)$, 감마-포아송의 $(\beta, \alpha)$가 모두 이 구조다.

    | 가능도 | 자연통계량 | 켤레 사전분포 |
    |---|---|---|
    | 베르누이/이항 | $x$ | 베타 |
    | 포아송 | $x$ | 감마 |
    | 지수/감마(비율) | $x$ | 감마 |
    | 정규($\sigma$ 기지) | $x$ | 정규 |
    | 정규($\mu$ 기지) | $x^2$ | 역감마 |
    | 정규(둘 다 미지) | $(x,x^2)$ | 정규-역감마 |
    | 다항 | 도수 벡터 | 디리클레 |

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**디리클레-다항** 켤레를 유도하고, 그것이 베타-이항의 자연스러운 확장임을 보여라. 텍스트 분석에서 어떻게 쓰이는가?

</div>

??? success "풀이"
    **모형.** $K$개 범주의 도수 $\mathbf{n}=(n_1,\dots,n_K)$가 다항분포를 따르고

    $$
    \boldsymbol\pi \sim \text{Dirichlet}(\alpha_1,\dots,\alpha_K), \qquad \pi(\boldsymbol\pi)\propto\prod_k \pi_k^{\alpha_k-1}
    $$

    이라 하자.

    **사후분포.** 가능도 $\propto\prod_k\pi_k^{n_k}$를 곱하면

    $$
    \pi(\boldsymbol\pi\mid\mathbf{n}) \propto \prod_k\pi_k^{\alpha_k+n_k-1}
    $$

    으로 **$\text{Dirichlet}(\alpha_1+n_1,\dots,\alpha_K+n_K)$** 다. 도수가 그냥 더해진다.

    **사후평균.**

    $$
    E[\pi_k\mid\mathbf{n}] = \frac{\alpha_k+n_k}{\alpha_0+n}, \qquad \alpha_0=\sum_k\alpha_k,\ n=\sum_k n_k
    $$

    **베타-이항의 확장임.** $K=2$로 두면 디리클레가 베타가 되고 다항이 이항이 된다. 위 식이 정확히 $(a+k)/(a+b+n)$이다.

    **텍스트 분석에서의 쓰임.**

    - **언어모형의 평활.** 낱말 $k$의 확률을 $n_k/n$으로 추정하면 훈련자료에 없던 낱말의 확률이 0이 되어 전체 문장의 확률이 0이 된다. 디리클레 사전분포를 쓰면

      $$
      \hat\pi_k = \frac{n_k+\alpha}{n+K\alpha}
      $$

      로 **아무도 0이 되지 않는다.** $\alpha=1$이면 라플라스 평활, $\alpha<1$이면 리드스톤 평활이다.

    - **나이브 베이즈 분류.** 각 분류마다 낱말 분포를 디리클레-다항으로 추정한다. 평활이 없으면 새 낱말 하나에 분류가 무너진다.

    - **잠재 디리클레 배분(LDA).** 문서마다 주제 분포가 디리클레, 주제마다 낱말 분포가 디리클레다. 켤레 구조 덕분에 **붕괴 깁스 표집**이 가능해진다. $\boldsymbol\pi$를 해석적으로 적분해 없애고 주제 배정만 표집하면 되므로 훨씬 효율적이다.

    **$\alpha$의 역할.** $\alpha$가 작으면(1 미만) 사후분포가 **희소**해진다. 몇몇 범주에 질량이 몰리고 나머지는 0에 가까워진다. LDA에서 $\alpha$를 작게 두는 것이 "문서 하나가 소수의 주제만 다룬다"는 믿음을 반영한 것이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
켤레 사전분포의 **한계**를 세 가지 들고, 그럼에도 여전히 유용한 이유를 적어라.

</div>

??? success "풀이"
    **한계 1 — 사전 믿음을 표현하지 못할 수 있다.** 켤레족은 모양이 정해져 있다. 베타분포는 봉우리가 하나뿐이므로 "$p$가 0.2 근처거나 0.8 근처일 것"이라는 이봉 믿음을 담을 수 없다. 그런 믿음이 실제로 합리적인 경우가 있다(두 부류의 개체가 섞여 있을 때).

    **한계 2 — 꼬리가 얇다.** 정규-정규 켤레에서 사전분포의 꼬리가 가능도와 같은 속도로 줄어들므로, **자료가 사전분포와 크게 어긋나면 사후분포가 둘 사이 어중간한 곳에 놓인다.** 꼬리가 두꺼운 사전분포($t$ 분포)를 쓰면 자료가 강하게 말할 때 사전분포가 물러나는 바람직한 거동을 보이는데, 켤레성을 잃는다. 이를 **사전분포의 강건성** 문제라 한다.

    **한계 3 — 모형이 조금만 복잡해져도 깨진다.** 계층모형, 회귀계수, 결측자료, 비선형 모형에서는 완전 켤레가 성립하지 않는 경우가 대부분이다. 현실의 모형은 대개 켤레가 아니다.

    **그럼에도 유용한 이유.**

    - **해석의 틀을 준다.** 사전 표본크기, 축소 가중치, 정밀도 덧셈 같은 개념이 켤레 구조에서 명확히 드러난다. 복잡한 모형에서도 이 직관이 통한다.
    - **깁스 표집의 부품이 된다.** 전체 모형이 켤레가 아니어도 **조건부분포가 켤레**인 경우가 많다. 그러면 각 단계에서 해석적으로 표집할 수 있어 MCMC가 훨씬 효율적이다. 계층모형의 표준 구현이 이 방식이다.
    - **빠른 근사와 검산.** 복잡한 모형을 돌리기 전에 켤레 근사로 대략의 답을 얻어 두면, 최종 결과가 터무니없는지 바로 알아챈다.
    - **닫힌 형태의 주변가능도.** 베이즈 인수나 모형 비교를 계산할 수 있다.

    **실무 권고.** 켤레를 **기본값이 아니라 출발점**으로 삼는다. 사전 믿음이 켤레족으로 표현되면 쓰고, 아니면 계산을 조금 더 들여 적절한 사전분포를 쓴다. 요즘은 HMC 같은 도구가 있어 켤레성에 얽매일 이유가 예전보다 훨씬 적다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
**제프리스 사전분포**를 정의하고, 베르누이와 정규 평균에서 각각 구하라. "무정보"라는 표현이 왜 조심스러운가?

</div>

??? success "풀이"
    **정의.**

    $$
    \pi_J(\theta) \propto \sqrt{I(\theta)}
    $$

    (다모수면 $\sqrt{|I(\boldsymbol\theta)|}$.)

    **핵심 성질 — 모수화 불변.** $\phi = g(\theta)$로 바꾸면 $I(\phi) = I(\theta)(d\theta/d\phi)^2$이므로

    $$
    \pi_J(\phi) = \pi_J(\theta)\left|\frac{d\theta}{d\phi}\right|
    $$

    로 **정확히 변수변환 공식을 만족한다.** 즉 $\theta$에서 제프리스 사전분포를 정한 뒤 $\phi$로 바꾸든, $\phi$에서 직접 정하든 같은 결과가 나온다. 균등 사전분포에는 이 성질이 없다.

    **베르누이.** $I(p) = 1/\{p(1-p)\}$이므로

    $$
    \pi_J(p) \propto p^{-1/2}(1-p)^{-1/2} = \text{Beta}(1/2,\ 1/2)
    $$

    **정규 평균($\sigma$ 기지).** $I(\mu)=1/\sigma^2$으로 $\mu$에 무관하므로

    $$
    \pi_J(\mu) \propto 1
    $$

    로 $\mathbb{R}$ 위의 평평한(비정상) 사전분포다.

    **정규 척도.** $I(\sigma) = 2/\sigma^2$이므로 $\pi_J(\sigma)\propto1/\sigma$, 즉 $\ln\sigma$에 평평한 사전분포다.

    **"무정보"가 조심스러운 이유.**

    1. **완전히 정보가 없는 사전분포는 존재하지 않는다.** 제프리스 사전분포도 정보를 담는다. 베르누이의 $\text{Beta}(0.5,0.5)$는 **경계를 선호**한다(밀도가 0과 1에서 발산). 균등 사전분포와 다른 결론을 준다.
    2. **비정상일 수 있다.** 사후분포가 정상인지 매번 확인해야 하고, 베이즈 인수는 아예 계산할 수 없다.
    3. **다모수에서 문제가 생긴다.** 정규분포에 그대로 적용하면 $\pi_J(\mu,\sigma)\propto1/\sigma^2$이 나오는데, 표준적으로 쓰는 $1/\sigma$와 다르고 성질도 나쁘다. 그래서 실무에서는 모수를 나눠 다루는 **참조 사전분포**를 쓴다.
    4. **"객관적"이라는 주장은 과하다.** 어떤 불변성을 원하느냐에 따라 답이 달라진다.

    **권고.** "무정보"보다 **"약정보"**나 **"참조"**라는 표현이 정확하다. 그리고 어떤 사전분포든 그것이 함의하는 바를 **사전 예측분포로 확인**하는 것이 좋은 관행이다. 사전분포에서 자료를 생성해 보아 터무니없는 자료가 나오면 그 사전분포는 무정보가 아니라 잘못된 정보를 담고 있는 것이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
**감마-포아송 켤레**를 유도하고, 그 사후 예측분포가 음이항임을 보여라. 이것이 과대산포와 어떻게 이어지는가?

</div>

??? success "풀이"
    **사후분포.** $X_i \sim \text{Poisson}(\lambda)$, $\lambda\sim\text{Gamma}(\alpha,\beta)$(비율 모수화)라 하면

    $$
    \pi(\lambda\mid\mathbf{x}) \propto \lambda^{\sum x_i}e^{-n\lambda}\cdot\lambda^{\alpha-1}e^{-\beta\lambda} = \lambda^{\alpha+\sum x_i-1}e^{-(\beta+n)\lambda}
    $$

    이므로

    $$
    \lambda\mid\mathbf{x} \sim \text{Gamma}\!\left(\alpha+\textstyle\sum x_i,\ \beta+n\right)
    $$

    이다. 사후평균은

    $$
    \frac{\alpha+\sum x_i}{\beta+n} = \frac{\beta}{\beta+n}\cdot\frac\alpha\beta+\frac{n}{\beta+n}\cdot\bar x
    $$

    로 역시 가중평균이며, $\beta$가 **사전 관측 기간**의 역할을 한다.

    **사후 예측분포.** 다음 관측 $\tilde X$에 대해

    $$
    P(\tilde X=k\mid\mathbf{x}) = \int_0^\infty \frac{e^{-\lambda}\lambda^k}{k!}\,\pi(\lambda\mid\mathbf{x})\,d\lambda
    $$

    인데, 이는 앞서 계산한 감마-포아송 혼합이므로

    $$
    \tilde X \mid\mathbf{x} \sim \text{NegBinom}\!\left(r=\alpha+\textstyle\sum x_i,\ p=\frac{\beta+n}{\beta+n+1}\right)
    $$

    이다.

    **과대산포와의 연결.** 음이항의 분산이 평균보다 크다.

    $$
    \operatorname{Var}(\tilde X\mid\mathbf{x}) = E[\tilde X\mid\mathbf{x}]\left(1+\frac{1}{\beta+n}\right) > E[\tilde X\mid\mathbf{x}]
    $$

    **$\lambda$를 모른다는 불확실성이 예측분산을 부풀린다.** $n\to\infty$이면 여분의 항이 0으로 가서 포아송으로 돌아온다.

    **두 가지 과대산포를 구분해야 한다.**

    - **여기의 것**은 **인식론적** 불확실성이다. $\lambda$는 하나인데 우리가 모를 뿐이며, 자료가 쌓이면 사라진다.
    - **앞서 본 음이항 회귀의 과대산포**는 **실재하는** 이질성이다. 개체마다 $\lambda$가 정말로 다르며, 자료를 아무리 모아도 사라지지 않는다.

    두 경우 모두 음이항이 나오지만 뜻이 다르다. 예측을 할 때는 둘 다 반영해야 하고, 모수를 추정할 때는 구분해야 한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
켤레 사전분포를 **혼합**하면(예: 베타분포 둘의 혼합) 어떤 성질이 유지되고 어떤 유연성을 얻는가?

</div>

??? success "풀이"
    **혼합 사전분포.**

    $$
    \pi(\theta) = \sum_{j=1}^J w_j\,\pi_j(\theta), \qquad \sum_j w_j=1
    $$

    각 $\pi_j$가 켤레라 하자.

    **유지되는 성질 — 사후분포도 같은 족의 혼합이다.**

    $$
    \pi(\theta\mid x) = \sum_j \tilde w_j\,\pi_j(\theta\mid x), \qquad \tilde w_j = \frac{w_j\,m_j(x)}{\sum_l w_l\,m_l(x)}
    $$

    여기서 $m_j(x)=\int f(x\mid\theta)\pi_j(\theta)d\theta$는 성분 $j$의 주변가능도다.

    **두 가지가 갱신된다.** 각 성분이 켤레 규칙대로 갱신되고, **가중치도 갱신된다.** 자료를 잘 설명하는 성분의 가중치가 올라간다.

    **얻는 유연성.**

    1. **다봉 사전 믿음.** "효과가 없거나(0 근처) 상당히 크거나" 같은 믿음을 표현할 수 있다. 스파이크-슬랩 사전분포가 이 구조이며, 변수 선택의 베이즈 판이다.
    2. **꼬리 두껍게 만들기.** 척도가 다른 정규분포들을 섞으면 꼬리가 두꺼워진다. 실제로 $t$ 분포가 정규의 척도 혼합이다. **자료가 사전분포와 어긋날 때 사전분포가 물러나는** 강건한 거동을 얻는다.
    3. **임의의 사전분포 근사.** 성분을 충분히 많이 쓰면 어떤 사전분포든 원하는 정밀도로 근사할 수 있다. 켤레 계산의 편리함을 유지하면서 유연성을 얻는 셈이다.

    **대가.**

    - 성분 수 $J$와 각 성분의 모수를 정해야 한다.
    - 사후 요약값이 닫힌 형태이긴 하나 식이 길어진다.
    - 성분이 많으면 계산 이득이 줄어 차라리 MCMC를 쓰는 편이 나을 수 있다.

    **실무의 대표적 쓰임.** 임상시험의 **강건 사전분포**다. 과거 자료에 기반한 정보적 성분과 평평한 성분을 $0.8:0.2$로 섞어 두면, 새 자료가 과거와 비슷하면 정보를 빌려 오고 크게 다르면 자동으로 평평한 성분이 지배한다. 규제기관이 요구하는 "사전분포에 대한 보험"을 제공하는 장치다.

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
