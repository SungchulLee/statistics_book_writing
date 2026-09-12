# MAP 추정

사후분포 $\pi(\theta \mid \mathbf{x})$를 계산한 뒤에는 갱신된 믿음을 요약하는 하나의 점추정값이 필요한 경우가 많다. 사후평균과 사후중앙값도 흔히 쓰이지만, 사후최빈값인 **최대사후확률(MAP)** 추정값에는 특별한 매력이 있다. 베이즈 추론을 벌점 최적화와 직접 연결하여 베이즈적 사고와 빈도주의적 사고 사이의 다리를 놓기 때문이다.

<div class="defn" markdown>

**정의 1.** [MAP 추정량]

MAP 추정량은 사후밀도를 최대화하는 모수값을 고른다:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; \pi(\theta \mid \mathbf{x})
$$

베이즈 정리가 $\pi(\theta \mid \mathbf{x}) = f(\mathbf{x} \mid \theta)\,\pi(\theta) / f(\mathbf{x})$를 주고 주변가능도 $f(\mathbf{x})$가 $\theta$에 의존하지 않으므로, 사후분포를 최대화하는 것은 분자를 최대화하는 것과 동등하다:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; f(\mathbf{x} \mid \theta)\,\pi(\theta)
$$

(최대점을 보존하는 단조변환인) 로그를 취하면 다음이 된다:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta}\; \bigl[\log f(\mathbf{x} \mid \theta) + \log \pi(\theta)\bigr]
$$

따라서 MAP 추정값은 로그가능도에 로그 사전분포 항을 더한 것을 최대화한다. 이 덧셈 구조가 MAP와 MLE, 그리고 정칙화의 연결을 이해하는 열쇠이다.

</div>

## MLE와의 관계

MAP 목적함수는 MLE와 $\log \pi(\theta)$가 더해진 것만 다르다. 표본크기 $n$이 크면 로그가능도 $\log f(\mathbf{x} \mid \theta) = \sum_{i=1}^n \log f(x_i \mid \theta)$는 $n$에 비례해 커지는 반면 로그 사전분포는 $\theta$의 고정된 함수로 남는다. 그 결과, 표준적인 정칙 조건 아래에서 사전분포가 참 모수값 근방에서 양수이기만 하면 사전분포의 영향이 사라지고 MAP 추정값이 MLE로 수렴한다:

$$
\hat{\theta}_{\text{MAP}} \to \hat{\theta}_{\text{MLE}} \quad \text{as } n \to \infty
$$

!!! example "정규분포 평균에서의 MAP와 MLE"
    사전분포가 $\mu \sim N(0, \sigma_0^2)$인 $X_1, \ldots, X_n \overset{iid}{\sim} N(\mu, 1)$이라 하자. MAP 추정값은

    $$
    \hat{\mu}_{\text{MAP}} = \frac{n\sigma_0^2}{n\sigma_0^2 + 1}\,\bar{X}
    $$

    $n = 1$이고 $\sigma_0^2 = 1$이면 MAP 추정값은 $\bar{X}/2$로 사전평균 $0$과 자료 사이의 절충이다. $n = 100$이면 MAP 추정값이 약 $0.99\,\bar{X}$로 MLE $\hat{\mu}_{\text{MLE}} = \bar{X}$와 거의 같다.

## 정칙화와의 관계

로그 사전분포 항 $\log \pi(\theta)$는 특정 모수값을 억제하는 벌점 역할을 한다. 사전분포족이 다르면 벌점의 구조도 달라진다.

**Gaussian 사전분포와 L2 정칙화.** $\theta_j \overset{iid}{\sim} N(0, \sigma_0^2)$이면 로그 사전분포는

$$
\log \pi(\theta) = \text{const} - \frac{1}{2\sigma_0^2}\sum_j \theta_j^2
$$

따라서 $\log f(\mathbf{x} \mid \theta) + \log \pi(\theta)$를 최대화하는 것은 음의 로그가능도에 $\lambda = 1/(2\sigma_0^2)$인 L2 벌점 $\lambda \|\theta\|_2^2$을 더해 최소화하는 것과 동등하다. 이것이 정확히 능형회귀이다.

**Laplace 사전분포와 L1 정칙화.** $\theta_j \overset{iid}{\sim} \text{Laplace}(0, b)$이면 로그 사전분포는

$$
\log \pi(\theta) = \text{const} - \frac{1}{b}\sum_j |\theta_j|
$$

이를 최대화하는 것은 음의 로그가능도에 $\lambda = 1/b$인 L1 벌점 $\lambda \|\theta\|_1$을 더해 최소화하는 것과 동등하다. 이것이 정확히 Lasso 회귀이며, L1 벌점이 일부 계수를 정확히 0으로 만들기 때문에 희소성을 촉진한다.

| 사전분포 | 로그 사전분포 벌점 | 정칙화 |
|---|---|---|
| $N(0, \sigma_0^2)$ | $-\frac{1}{2\sigma_0^2}\|\theta\|_2^2$ | Ridge (L2) |
| Laplace$(0, b)$ | $-\frac{1}{b}\|\theta\|_1$ | Lasso (L1) |

이 대응은 흔히 과대적합에 관한 순수 빈도주의 논증으로 동기가 부여되는 정칙화 추정에 자연스러운 베이즈적 해석이 있음을 드러낸다. 벌점은 모수의 크기에 대한 사전 믿음을 담고 있는 것이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Bernoulli에 대한 베이즈적 추정량 $\hat\theta_B = (\sum X_i + a)/(n + a + b)$를 생각하자. (a) 편향과 분산. (b) $a = b = \sqrt n/2$일 때 편향되어 있지만 일치함을 보여라. (c) $\theta = 0.5, n = 10$에서 MLE와 평균제곱오차를 비교하라.

</div>

??? success "풀이"
    (a) $S = \sum X_i$라 하면 $\mathbb{E}[S] = n\theta$, $\mathrm{Var}(S) = n\theta(1-\theta)$이다.

    $\mathbb{E}[\hat\theta_B] = (n\theta + a)/(n + a + b)$. $\mathrm{Bias}(\hat\theta_B) = (a - (a+b)\theta)/(n + a + b)$.

    $\mathrm{Var}(\hat\theta_B) = n\theta(1-\theta)/(n + a + b)^2$.

    (b) $a = b = \sqrt n/2$이면 $\mathrm{Bias} = \sqrt n(0.5 - \theta)/(n + \sqrt n) \to 0$이고 분산도 $\to 0$이므로 일치한다.

    (c) $\theta = 0.5, n = 10$에서 편향은 0이다(사전분포의 중심이 $0.5$이고 실제 $\theta$도 그 값이다). $\mathrm{Var}(\hat\theta_B) \approx 0.0144$이고 MLE는 $\mathrm{Var}(\hat p) = 0.025$이다. $\theta = 0.5$에서 베이즈 추정량의 평균제곱오차가 더 작다. 사전분포 쪽으로의 축소가 다른 곳에서의 편향을 대가로 분산을 줄인다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$\mathrm{Beta}(\alpha, \beta)$ 사전분포를 쓴 $X \sim \mathrm{Binomial}(n, p)$에 대해 **MAP 추정량을 유도하라.**

</div>

??? success "풀이"
    사후분포: $\pi(p \mid x) \propto p^x(1-p)^{n-x} \cdot p^{\alpha-1}(1-p)^{\beta-1} = p^{x+\alpha-1}(1-p)^{n-x+\beta-1}$.

    이는 (켤레성에 의해) $\mathrm{Beta}(x + \alpha, n - x + \beta)$이다.

    Beta$(\alpha', \beta')$의 최빈값은 둘 다 $> 1$일 때 $(\alpha' - 1)/(\alpha' + \beta' - 2)$이다.

    $\hat p_{\mathrm{MAP}} = (x + \alpha - 1)/(n + \alpha + \beta - 2)$.

    **특수한 경우:**

    - $\alpha = \beta = 1$ (균등 사전분포): $\hat p_{\mathrm{MAP}} = x/n = \hat p_{\mathrm{MLE}}$. 평평한 사전분포는 MLE를 복원한다.
    - $\alpha = \beta = 0.5$ (Jeffreys 사전분포): 0.5 쪽으로 약한 축소.
    - $\alpha = \beta$가 클 때: $\alpha/(\alpha + \beta) = 0.5$ 쪽으로 강한 축소.

    MAP 점추정량은 사후평균과 다르다: $\hat p_{\mathrm{Bayes,mean}} = (x + \alpha)/(n + \alpha + \beta)$. 대칭 손실(제곱오차)에는 평균이, 0-1 손실에는 최빈값이 선호된다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**MAP와 MLE.** MAP가 MLE와 같아지는 때는 언제인가? 같지 않은 때는?

</div>

??? success "풀이"
    MAP는 $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$를 최대화하고 MLE는 $L(\theta)$만 최대화한다.

    **MAP = MLE**일 필요충분조건은 $\pi(\theta)$가 지지집합에서 상수인 것, 즉 평평한(비정상) 사전분포인 것이다. 달리 말하면 사전분포가 무정보인 경우이다.

    **MAP $\ne$ MLE**인 것은 사전분포에 모양이 있을 때이다. 예를 들어 Beta(2, 2)는 경계보다 0.5 근처에 더 많은 질량을 두므로 MLE에 비해 MAP를 0.5 쪽으로 끌어당긴다.

    정보가 있는 사전분포에서 MAP는 **사전평균/최빈값 쪽으로의 축소**를 들여온다. 축소의 강도는 $1/n$에 비례한다. 소표본에서는 정보가 있는 사전분포가 지배하고 대표본에서는 자료가 지배한다.

    **실용적 활용:** $n$이 매우 작거나 관측된 계수가 극단적일 때 약한 정보의 사전분포를 쓴 MAP가 MLE보다 안정적이다. 예를 들어 $n$번 던져 앞면이 $X = 0$번 나왔을 때 $p$의 MLE는 $\hat p = 0$인데(정확히 0일 수는 없다), Beta(1.5, 1.5)를 쓴 MAP는 $\hat p = 0.5/(n + 1)$로 작지만 0이 아니다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**켤레 사전분포.** 켤레 사전분포가 계산상 편리한 이유는 무엇인가? Beta-Bernoulli 외의 예를 하나 들라.

</div>

??? success "풀이"
    가능도족 $L(\theta; X)$에 대한 **켤레 사전분포**란 사후분포 $\pi(\theta \mid X)$가 같은 족에 속하게 하는 사전분포 $\pi(\theta)$이다.

    **편리함:**

    - 사후분포를 해석적으로 계산할 수 있어 수치적분이 필요 없다.
    - 새 자료로 갱신하는 것이 초모수를 갱신하는 것으로 끝난다(예: Beta $\to$ 모수가 이동한 Beta).
    - 순차적/온라인 갱신이 아주 쉽다.

    **예:**

    | 가능도 | 켤레 사전분포 | 사후분포 |
    |---|---|---|
    | Bernoulli$(p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + x, \beta + n - x)$ |
    | Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum X_i, \beta + n)$ |
    | Normal$(\mu, \sigma^2)$, $\sigma$ 알려짐 | Normal$(\mu_0, \tau_0^2)$ | Normal (갱신됨) |
    | Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + n, \beta + \sum X_i)$ |

    **현대적 단서:** MCMC와 HMC를 쓸 수 있게 되면서 켤레성은 더 이상 필수가 아니다. 임의의 사전분포와 가능도를 쓰고 사후분포를 수치적으로 표본추출할 수 있다. 켤레성은 교육적으로 다루기 쉽다는 점과 기준선 역할로서 여전히 가치가 있다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
**사후평균과 MAP.** $\pi(\theta \mid x) = \mathrm{Beta}(20, 5)$에 대해 둘을 모두 계산하라. 왜 서로 다를 수 있는가?

</div>

??? success "풀이"
    $\mathrm{Beta}(\alpha, \beta)$에서:

    - 평균: $\alpha/(\alpha + \beta) = 20/25 = 0.80$.
    - 최빈값: $(\alpha - 1)/(\alpha + \beta - 2) = 19/23 \approx 0.826$.

    $\mathrm{Beta}(20, 5)$가 비대칭이기 때문에 둘이 다르다. 질량이 1 쪽에 몰려 있고 왜도는 음수이다(왼쪽으로 치우쳐 있다). 대칭인 사후분포(예: 정규분포)에서는 평균 = 최빈값 = 중앙값이다. 치우친 사후분포에서는 세 값이 갈라지며, 이 경우처럼 왜도가 음수이면 평균 < 중앙값 < 최빈값의 순서가 된다.

    **어느 것을 쓸 것인가?**

    - **제곱오차 손실:** 사후평균이 최적이다(기대 제곱오차를 최소화한다).
    - **0-1 손실:** 사후최빈값(MAP)이 최적이다.
    - **절대오차 손실:** 사후중앙값이 최적이다.

    손실함수에 맞추어 점추정량을 고르면 된다. 실무에서는 어떤 점 요약보다 사후 **분포** 자체가 더 많은 정보를 담으므로, 가능하면 사후분포 전체를 보고해야 한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**비정상 사전분포.** $\int \pi(\theta) d\theta = \infty$인 사전분포(예: $\mathbb{R}$ 위의 $\pi(\mu) = 1$)를 **비정상(improper)**이라 한다. 그래도 타당한 사후분포를 계산할 수 있는가? 언제 그러한가?

</div>

??? success "풀이"
    그렇다. *사후분포*가 정상(proper)이기만 하면 된다. 조건은 분자 $L(\theta) \pi(\theta)$가 $\theta$에 대해 유한한 적분을 갖는 것이다:

    $$
    \int L(\theta) \pi(\theta) d\theta < \infty
    $$

    그러면 $\pi(\theta \mid x) \propto L(\theta) \pi(\theta)$를 정규화할 수 있다.

    **흔한 비정상 사전분포:**

    - 위치모수에 대한 $\pi(\mu) = 1$ ($\mathbb{R}$ 위의 Lebesgue 측도).
    - 분산/척도 모수에 대한 $\pi(\sigma) = 1/\sigma$ (척도에 대한 Jeffreys).
    - Bernoulli에 대한 $\pi(p) \propto 1/\sqrt{p(1-p)}$ (Bernoulli에 대한 Jeffreys).

    **비정상 사전분포의 문제:**

    - 자료가 정보를 주지 못하면 사후분포도 비정상이 될 수 있다.
    - 베이즈 인자와 모형 비교가 깨질 수 있다(주변가능도가 정의되지 않는다).
    - 직관에 반하는 역설이 생길 수 있다(Lindley 역설).

    **현대적 관행:** 비정상 사전분포를 근사하면서도 정상성을 유지하는 약한 정보의 *정상* 사전분포(예: 위치모수에 대한 $N(0, 10^4)$)를 쓴다. "사전 정보를 최소화한다"는 의도를 유지하면서 함정을 피한다.

---

## 정리하며

최대사후확률 추정은 사후분포의 **최빈값**을 고른다.

$$
\hat\theta_{\text{MAP}} = \arg\max_\theta\; f(\mathbf x\mid\theta)\,\pi(\theta)
$$

- **로그를 취하면 정체가 드러난다.** $\log f(\mathbf x\mid\theta)+\log\pi(\theta)$ 를 최대화하는 것이며, 첫 항이 로그가능도이고 **둘째 항이 벌점**이다. 즉 MAP 는 **벌점 최대가능도**다.
- **정칙화가 사전분포의 다른 이름이다.** 정규 사전분포를 두면 $L_2$ 벌점(능형회귀)이 되고, 라플라스 사전분포를 두면 $L_1$ 벌점(라쏘)이 된다. 18장의 정칙화 회귀가 이 사실 위에 서 있다.
- **평평한 사전분포면 최대가능도와 같아진다.** 베이즈와 빈도주의가 만나는 지점이다.
- **사후평균·사후중앙값과 다르다.** 사후분포가 치우쳐 있으면 셋이 갈리며, 최빈값은 **손실함수의 관점에서 0–1 손실에 대응**한다. 제곱손실이면 사후평균, 절대손실이면 사후중앙값이 최적이다.
- **점추정은 사후분포를 버리는 일이다.** MAP 값 하나만 보고하면 불확실성 정보가 사라진다. 베이즈의 장점은 분포 전체에 있다.

다음 절 **베이즈 추정 시연**에서 두 켤레 모형을 실제로 돌려 보며, 자료가 쌓일수록 사후분포가 어떻게 좁아지는지 본다.
