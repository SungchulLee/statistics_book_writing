# 카이제곱분포와 이차형식

고전 통계의 여러 검정통계량 — 적합도 통계량, 분산비, 잔차제곱합 — 은 정규확률벡터의 이차형식이다. 그런 이차형식이 언제 카이제곱분포를 따르는지 이해하는 것이 이 검정통계량들의 정확한 분포를 유도하는 데 필수적이다. 핵심 결과는 (앞 절들에서 다룬) **멱등행렬** 구조를 카이제곱분포와 연결한다. $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬이고 $\mathbf{z}$가 표준정규벡터이면 $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$이다.

## 카이제곱분포 복습

<div class="defn" markdown>

**정의 1.** [카이제곱분포]

$Z_1, Z_2, \dots, Z_k$가 독립인 표준정규 확률변수($Z_i \sim N(0,1)$)이면

$$
Q = \sum_{i=1}^k Z_i^2 \sim \chi^2_k
$$

이다. 이 분포는 **자유도**가 $k$다. 벡터 표기로는 $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_k)$이면 $\mathbf{z}^T\mathbf{z} \sim \chi^2_k$이다.

</div>

핵심 성질:

- $E[Q] = k$이고 $\operatorname{Var}(Q) = 2k$
- $Q_1 \sim \chi^2_{k_1}$과 $Q_2 \sim \chi^2_{k_2}$가 독립이면 $Q_1 + Q_2 \sim \chi^2_{k_1 + k_2}$
- 카이제곱분포는 감마분포의 특수한 경우다: $\chi^2_k = \text{Gamma}(k/2, 2)$

## 정규벡터의 이차형식

확률벡터 $\mathbf{z} \in \mathbb{R}^n$의 **이차형식**은 대칭행렬 $\mathbf{A} \in \mathbb{R}^{n \times n}$에 대한 $\mathbf{z}^T\mathbf{A}\mathbf{z}$ 꼴의 식이다. $\mathbf{z}^T\mathbf{A}\mathbf{z} = \mathbf{z}^T\bigl(\frac{\mathbf{A} + \mathbf{A}^T}{2}\bigr)\mathbf{z}$이므로 일반성을 잃지 않고 $\mathbf{A}$를 대칭으로 가정할 수 있다.

$\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$일 때 $\mathbf{z}^T\mathbf{A}\mathbf{z}$의 분포는 $\mathbf{A}$의 고윳값에 달려 있다.

### 대각화 접근

스펙트럼 정리에 의해 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$이며 $\mathbf{Q}$는 직교행렬, $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$이다. $\mathbf{w} = \mathbf{Q}^T\mathbf{z}$로 두면

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} = \mathbf{w}^T\boldsymbol{\Lambda}\mathbf{w} = \sum_{i=1}^n \lambda_i W_i^2
$$

이다. $\mathbf{Q}$가 직교행렬이고 $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이므로, 회전된 벡터 $\mathbf{w} = \mathbf{Q}^T\mathbf{z}$도 $N(\mathbf{0}, \mathbf{I}_n)$을 따른다(표준정규분포는 직교변환에 불변이다). 따라서 $W_1, \dots, W_n$은 독립인 $N(0,1)$ 확률변수이고, 이 이차형식은 독립인 $\chi^2_1$ 변수들의 가중합이다.

## 기본 카이제곱 정리

<div class="thmbox" markdown>

### 정리 1. 멱등행렬을 갖는 이차형식 { .thm }

$\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{A} \in \mathbb{R}^{n \times n}$이 계수 $r$인 대칭 멱등행렬이라 하자. 그러면

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r
$$

</div>

??? proof "증명"

    $\mathbf{A}$가 대칭이고 멱등이므로 (멱등행렬 이론에 따라) 고윳값이 모두 0 또는 1이다. $\operatorname{rank}(\mathbf{A}) = r$이므로 정확히 $r$개의 고윳값이 1이고 $n - r$개가 0이다.

    대각화 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$와 $\mathbf{w} = \mathbf{Q}^T\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$을 쓰면

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^n \lambda_i W_i^2 = \sum_{i:\,\lambda_i = 1} W_i^2
    $$

    이다. 이는 독립인 $\chi^2_1$ 변수 $r$개의 합이므로 $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$이다. $\square$

## 일반적인 필요충분조건

멱등 조건은 충분할 뿐 아니라, (척도를 제외하면) 카이제곱분포를 얻기 위해 필요하기도 하다.

<div class="thmbox" markdown>

### 정리 2. 코크런 조건 { .thm }

$\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고 $\mathbf{A}$가 대칭인 $n \times n$ 행렬이라 하자. $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$일 필요충분조건은 $\mathbf{A}$가 $\operatorname{rank}(\mathbf{A}) = r$인 멱등행렬인 것이다.

</div>

**증명 개요(필요성).** $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$이면 적률생성함수가 $\chi^2_r$의 것과 일치해야 한다. $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2$의 적률생성함수는 $\prod_{i=1}^n(1 - 2\lambda_i t)^{-1/2}$이고 $\chi^2_r$의 것은 $(1 - 2t)^{-r/2}$이다. 이 둘이 같으려면 정확히 $r$개의 고윳값이 1이고 나머지가 0이어야 하며, 이는 $\mathbf{A}$가 계수 $r$인 멱등행렬이라는 뜻이다. $\square$

## 이차형식의 독립성

<div class="thmbox" markdown>

### 정리 3. 크레이그 정리 { .thm }

$\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고 $\mathbf{A}$와 $\mathbf{B}$가 대칭인 $n \times n$ 행렬이라 하자. 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$와 $\mathbf{z}^T\mathbf{B}\mathbf{z}$가 **독립**일 필요충분조건은

$$
\mathbf{A}\mathbf{B} = \mathbf{0}
$$

인 것이다.

</div>

??? proof "증명 개요"

    $(\mathbf{z}^T\mathbf{A}\mathbf{z}, \mathbf{z}^T\mathbf{B}\mathbf{z})$의 결합 적률생성함수가 주변 적률생성함수의 곱으로 인수분해될 필요충분조건이 $\mathbf{A}\mathbf{B} = \mathbf{0}$이다. $\mathbf{A}$와 $\mathbf{B}$를 동시에 대각화해 보면 알 수 있다. 조건 $\mathbf{A}\mathbf{B} = \mathbf{0}$은 두 이차형식이 독립인 $W_i^2$ 항들의 서로 겹치지 않는 부분집합만을 포함하도록 보장한다. $\square$

## 코크런 정리

코크런 정리는 카이제곱 결과와 독립성 결과를 제곱합 분해에 관한 하나의 강력한 진술로 결합한다.

<div class="thmbox" markdown>

### 정리 4. 코크런 정리 { .thm }

$\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고

$$
\mathbf{z}^T\mathbf{z} = \mathbf{z}^T\mathbf{A}_1\mathbf{z} + \mathbf{z}^T\mathbf{A}_2\mathbf{z} + \cdots + \mathbf{z}^T\mathbf{A}_k\mathbf{z}
$$

이라 하자. 여기서 $\mathbf{A}_1, \dots, \mathbf{A}_k$는 $\operatorname{rank}(\mathbf{A}_i) = r_i$이고 $r_1 + r_2 + \cdots + r_k = n$인 대칭 양반정치행렬이다. 그러면 이차형식 $\mathbf{z}^T\mathbf{A}_1\mathbf{z}/\sigma^2, \dots, \mathbf{z}^T\mathbf{A}_k\mathbf{z}/\sigma^2$은 서로 독립이고 $\mathbf{z}^T\mathbf{A}_i\mathbf{z}/\sigma^2 \sim \chi^2_{r_i}$이다.

</div>

코크런 정리는 분산분석 F-검정을 떠받치는 이론적 원동력이다. 회귀제곱합과 잔차제곱합이 ($\sigma^2$으로 나눈 뒤) 독립인 카이제곱 확률변수임을 보장하며, 이것이 F-통계량을 구성하는 데 필요하다.

## 예 — 잔차제곱합

$\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$인 선형모형 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$에서,

잔차벡터는 $\mathbf{H} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T$에 대해 $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$이다. 잔차제곱합은

$$
\text{SSE} = \mathbf{e}^T\mathbf{e} = \mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}
$$

이다. $(\mathbf{I} - \mathbf{H})\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$이므로 $\text{SSE} = \boldsymbol{\varepsilon}^T(\mathbf{I} - \mathbf{H})\boldsymbol{\varepsilon}$으로 쓸 수 있다. 행렬 $\mathbf{M} = \mathbf{I} - \mathbf{H}$는 대칭 멱등이고 $\operatorname{rank}(\mathbf{M}) = n - p$이다. $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma$로 두면

$$
\frac{\text{SSE}}{\sigma^2} = \mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{n-p}
$$

이다. 이것이 $\sigma^2$의 불편추정값을 얻기 위해 SSE를 $n - p$로 나누는 이유를 설명한다.

$$
s^2 = \frac{\text{SSE}}{n - p}, \qquad E[s^2] = \sigma^2
$$

## 비중심 카이제곱분포

$\boldsymbol{\mu} \neq \mathbf{0}$인 $\mathbf{z} \sim N(\boldsymbol{\mu}, \mathbf{I}_n)$이고 $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬이면, 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$는 **비중심 카이제곱분포**를 따른다.

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r(\delta)
$$

여기서 비중심 모수는 $\delta = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$이다. 비중심 카이제곱분포는 F-검정의 검정력 계산과 대립가설 아래에서 회귀제곱합의 분포에 등장한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
$\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{P}$가 $\operatorname{rank}(\mathbf{P}) = r$인 대칭 멱등행렬이라 하자. $\mathbf{Z}^T \mathbf{P} \mathbf{Z}$의 분포를 진술하고 이를 적용하라. 관측값이 $n = 20$인 단순선형회귀에서 $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$일 때 $\mathbf{e}^T \mathbf{e}/\sigma^2$의 분포를 구하라.

</div>

??? success "풀이"
    기본 카이제곱 정리에 의해 $\mathbf{Z}^T \mathbf{P} \mathbf{Z} \sim \chi^2_r$이다.

    단순선형회귀에서 $\mathbf{M} = \mathbf{I} - \mathbf{H}$는 계수가 $n - 2 = 18$인 대칭 멱등행렬이다. $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 아래에서 ($\mathbf{M}\mathbf{X} = \mathbf{0}$을 쓰면) 잔차는 $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\boldsymbol{\varepsilon}$이다. 따라서

    $$
    \frac{\mathbf{e}^T\mathbf{e}}{\sigma^2} = \frac{\boldsymbol{\varepsilon}^T\mathbf{M}\boldsymbol{\varepsilon}}{\sigma^2} = \mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{18}
    $$

    이며, 여기서 $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma \sim N(\mathbf{0}, \mathbf{I}_n)$이다.

<div class="drillbox" markdown>

**연습문제 2.**
$\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{A}$가 고윳값 $\lambda_1, \ldots, \lambda_n$을 갖는 대칭행렬일 때, $\mathbf{z}^T\mathbf{A}\mathbf{z}$의 적률생성함수가 0의 어떤 근방에서

$$
M(t) = \prod_{i=1}^n (1 - 2\lambda_i t)^{-1/2}
$$

임을 보여라. 이를 이용해 $\mathbf{A}$가 계수 $r$인 멱등행렬일 때 적률생성함수가 $(1 - 2t)^{-r/2}$으로 환원됨을 확인하라.

</div>

??? success "풀이"
    $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$로 대각화한다. $\mathbf{w} = \mathbf{Q}^T\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$으로 두면

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^n \lambda_i W_i^2
    $$

    이다. $W_i$들이 독립이므로 적률생성함수가 인수분해된다.

    $$
    M(t) = \prod_{i=1}^n \mathbb{E}[e^{t \lambda_i W_i^2}] = \prod_{i=1}^n (1 - 2\lambda_i t)^{-1/2}
    $$

    여기서 $\chi^2_1$의 적률생성함수 $\mathbb{E}[e^{tW^2}] = (1 - 2t)^{-1/2}$을 썼다.

    $\mathbf{A}$가 멱등이고 고윳값 중 $r$개가 1, 나머지가 0이면 1인 고윳값만 기여하므로 $M(t) = (1 - 2t)^{-r/2}$, 즉 $\chi^2_r$의 적률생성함수가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
$\mathbf{y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I}_n)$이라 하자. $H_0: \boldsymbol{\beta} = \mathbf{0}$ 아래에서 회귀제곱합이 $\mathrm{SSR}/\sigma^2 = \mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$이고 $\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$과 독립임을 보여라.

</div>

??? success "풀이"
    $H_0$ 아래에서 $\mathbf{y} = \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이다. $\mathbf{z} = \mathbf{y}/\sigma$로 두면 $\mathrm{SSR}/\sigma^2 = \mathbf{z}^T\mathbf{H}\mathbf{z}$이고 $\mathbf{H}$는 계수 $p$인 대칭 멱등행렬이므로 $\chi^2_p$를 따른다.

    마찬가지로 $\mathrm{SSE}/\sigma^2 = \mathbf{z}^T\mathbf{M}\mathbf{z}$이고 $\mathbf{M}$은 계수 $n - p$인 대칭 멱등행렬이므로 $\chi^2_{n-p}$를 따른다.

    독립성은 크레이그 정리에서 따라온다: $\mathbf{H}\mathbf{M} = \mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{0}$. $\square$

    이것이 바로 전체 F-검정의 설정이다. $H_0$ 아래에서 $F = (\mathrm{SSR}/p)/(\mathrm{SSE}/(n-p)) \sim F_{p, n-p}$이다.

<div class="drillbox" markdown>

**연습문제 4.**
흔한 분산분석 분해는 $\mathbf{y}^T\mathbf{y}$를 두 개의 이차형식으로 쪼갠다. $\mathbf{A}_1 = \mathbf{H}$, $\mathbf{A}_2 = \mathbf{I} - \mathbf{H}$로 두자. 코크런 정리가 적용됨을 확인하고, 각 조각의 카이제곱분포와 독립성을 결론지어라.

</div>

??? success "풀이"
    분해는 $\mathbf{y}^T\mathbf{y} = \mathbf{y}^T\mathbf{H}\mathbf{y} + \mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}$이다.

    코크런 정리의 가정 확인:

    - $\mathbf{A}_1 + \mathbf{A}_2 = \mathbf{H} + (\mathbf{I} - \mathbf{H}) = \mathbf{I}$. ✓
    - 각 $\mathbf{A}_i$가 대칭 양반정치다. ✓
    - $\operatorname{rank}(\mathbf{A}_1) + \operatorname{rank}(\mathbf{A}_2) = p + (n - p) = n$. ✓

    결론($\mathbf{y} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 아래에서): $\mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$이고 $\mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}/\sigma^2 \sim \chi^2_{n-p}$이며 둘은 독립이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.**
$\mathbf{z} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$이고 $\mathbf{A}$가 대칭일 때 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$의 **기댓값**을 계산하라. 그런 다음 $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$인 경우로 특수화하여 $\mathbb{E}[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \sigma^2 \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$임을 보여라.

</div>

??? success "풀이"
    $\mathbf{w} \sim N(\mathbf{0}, \boldsymbol{\Sigma})$에 대해 $\mathbf{z} = \boldsymbol{\mu} + \mathbf{w}$로 쓴다. 전개하면($\mathbf{A}$의 대칭성을 쓴다)

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu} + 2\boldsymbol{\mu}^T\mathbf{A}\mathbf{w} + \mathbf{w}^T\mathbf{A}\mathbf{w}
    $$

    이다. 가운데 항은 평균이 0이다. 마지막 항에 대해서는

    $$
    \mathbb{E}[\mathbf{w}^T\mathbf{A}\mathbf{w}] = \mathbb{E}[\operatorname{tr}(\mathbf{A}\mathbf{w}\mathbf{w}^T)] = \operatorname{tr}(\mathbf{A}\,\mathbb{E}[\mathbf{w}\mathbf{w}^T]) = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma})
    $$

    이다. 따라서 $\mathbb{E}[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \operatorname{tr}(\mathbf{A}\boldsymbol{\Sigma}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$이고, $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$이면 이것이 $\sigma^2 \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$가 된다. $\square$

    통계적 쓰임: $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$인 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ 아래에서 $\mathbf{M}\mathbf{X}\boldsymbol{\beta} = \mathbf{0}$이므로 $\mathbb{E}[\mathrm{SSE}] = \sigma^2 \operatorname{tr}(\mathbf{M}) = \sigma^2(n - p)$이다. $n - p$로 나누면 불편인 $\hat{\sigma}^2$을 얻는다.

<div class="drillbox" markdown>

**연습문제 6.**
대립가설 아래에서 $\boldsymbol{\mu} \ne \mathbf{0}$인 $\mathbf{z} \sim N(\boldsymbol{\mu}, \mathbf{I}_n)$이고 $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬이면, 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$는 **비중심** 카이제곱분포 $\chi^2_r(\delta)$를 따른다. 비중심 모수 $\delta$를 찾고 F-검정의 검정력 계산에서 그 역할을 설명하라.

</div>

??? success "풀이"
    $\mathbf{w} \sim N(\mathbf{0}, \mathbf{I}_n)$에 대해 $\mathbf{z} = \boldsymbol{\mu} + \mathbf{w}$로 분해한다. $\mathbf{A} = \mathbf{Q}\operatorname{diag}(\mathbf{1}_r, \mathbf{0}_{n-r})\mathbf{Q}^T$로 대각화하고 $\tilde{\boldsymbol{\mu}} = \mathbf{Q}^T \boldsymbol{\mu}$, $\tilde{\mathbf{w}} = \mathbf{Q}^T \mathbf{w}$로 두면

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^r (\tilde{\mu}_i + \tilde{W}_i)^2
    $$

    이다. 이것이 바로 비중심 $\chi^2_r$의 정의이며 비중심 모수는

    $$
    \delta = \sum_{i=1}^r \tilde{\mu}_i^2 = \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}
    $$

    이다.

    **F-검정 검정력에서의 역할:** $H_1: \boldsymbol{\beta} \ne \mathbf{0}$ 아래에서 F 분자의 $\chi^2$이 $\delta = \boldsymbol{\beta}^T\mathbf{X}^T\mathbf{X}\boldsymbol{\beta}/\sigma^2$인 비중심 분포가 된다. $\delta$가 클수록(귀무가설에서 멀수록) F-통계량의 분포가 큰 값 쪽으로 이동하여 기각 확률이 높아진다. 즉 검정력이 커진다. 이것이 표본 크기를 계획할 때 검정력 계산기에 넣는 공식이다.

<div class="drillbox" markdown>

**연습문제 7.**
크레이그 정리($\mathbf{A}\mathbf{B} = \mathbf{O}$이면 두 이차형식이 독립)를 모의실험으로 확인하라. $\mathbf{A}\mathbf{B} \neq \mathbf{O}$인 경우와 대비하라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n, B = 6, 300_000
    Z = rng.normal(size=(B, n))                 # z ~ N(0, I)

    A = np.diag([1., 1., 0., 0., 0., 0.])       # 좌표 1,2 를 본다
    Bm = np.diag([0., 0., 1., 1., 1., 0.])      # 좌표 3,4,5 를 본다 (겹치지 않음)
    C = np.diag([1., 1., 1., 0., 0., 0.])       # 좌표 1,2,3 (A 와 겹친다)

    q = lambda M: np.einsum('bi,ij,bj->b', Z, M, Z)
    qa, qb, qc = q(A), q(Bm), q(C)

    print("A B = O 인가:", np.allclose(A @ Bm, 0),
          "  corr(q_A, q_B) =", round(np.corrcoef(qa, qb)[0, 1], 5))
    print("A C = O 인가:", np.allclose(A @ C, 0),
          "  corr(q_A, q_C) =", round(np.corrcoef(qa, qc)[0, 1], 5))
    ```

    출력:

    ```
    A B = O 인가: True   corr(q_A, q_B) = 0.00164
    A C = O 인가: False   corr(q_A, q_C) = 0.81566
    ```

    $\mathbf{A}\mathbf{B} = \mathbf{O}$일 때 상관이 $0.002$로 사실상 0이고, 겹치는 $\mathbf{C}$에서는 $0.816$으로 강하게 상관된다.

    **왜 그런가.** 대각행렬 예에서는 직관이 명확하다. $\mathbf{A}\mathbf{B} = \mathbf{O}$은 두 이차형식이 **서로 다른 좌표만** 쓴다는 뜻이고, 정규분포에서 서로 다른 좌표는 독립이다. 일반적인 경우에도 두 행렬을 동시에 대각화하면 같은 구조가 드러난다.

    이것이 분산분석에서 제곱합들이 독립인 근거다. 서로 직교하는 부분공간으로의 사영은 곱이 $\mathbf{O}$이므로 독립이고, 그래서 카이제곱의 비가 $F$ 분포가 된다. $\square$

<div class="drillbox" markdown>

**연습문제 8.**
$\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{A}$가 대칭이면 $\operatorname{Var}(\mathbf{z}^T\mathbf{A}\mathbf{z}) = 2\operatorname{tr}(\mathbf{A}^2)$임을 확인하라. $\mathbf{A}$가 멱등일 때 이것이 카이제곱의 분산과 맞음을 보여라.

</div>

??? success "풀이"
    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n, B = 6, 300_000
    Z = rng.normal(size=(B, n))

    M = np.diag([1., 1., 1., 1., 1., 1.]).copy()
    M[0, 0], M[1, 1] = 2., 3.
    M[0, 1] = M[1, 0] = 1.                      # 대칭이지만 멱등은 아님

    q = np.einsum('bi,ij,bj->b', Z, M, Z)
    print(f"E  모의 {q.mean():.4f}   tr(M)      {np.trace(M):.4f}")
    print(f"Var 모의 {q.var():.4f}   2 tr(M^2)  {2*np.trace(M @ M):.4f}")
    ```

    출력:

    ```
    E  모의 8.9960   tr(M)      9.0000
    Var 모의 38.0284   2 tr(M^2)  38.0000
    ```

    평균이 $\operatorname{tr}(\mathbf{M})$, 분산이 $2\operatorname{tr}(\mathbf{M}^2)$과 맞는다.

    **멱등인 경우.** $\mathbf{A}^2 = \mathbf{A}$이므로

    $$
    \operatorname{Var}(\mathbf{z}^T\mathbf{A}\mathbf{z}) = 2\operatorname{tr}(\mathbf{A}^2) = 2\operatorname{tr}(\mathbf{A}) = 2r
    $$

    로 $\chi^2_r$의 분산과 정확히 일치한다. 평균도 $\operatorname{tr}(\mathbf{A}) = r$이다.

    **거꾸로 읽으면 유용하다.** 이차형식이 카이제곱이 **아닌** 경우에도 평균과 분산은 이 공식으로 계산된다. 그래서 근사적으로 $\chi^2$에 맞추는 새터스웨이트 근사가 가능하다. 자유도를 $\nu = 2(\operatorname{tr}\mathbf{A})^2/\operatorname{tr}(\mathbf{A}^2)$로 잡으면 평균과 분산이 맞아떨어진다. 웰치 $t$ 검정의 자유도가 정수가 아닌 이유가 여기에 있다. $\square$

<div class="drillbox" markdown>

**연습문제 9.**
공분산이 $\boldsymbol{\Sigma} \neq \sigma^2\mathbf{I}$인 일반적인 경우에는 $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$일 필요충분조건이 $\mathbf{A}\boldsymbol{\Sigma}$가 멱등인 것이다. 마할라노비스 이차형식으로 확인하라.

</div>

??? success "풀이"
    $\mathbf{A} = \boldsymbol{\Sigma}^{-1}$로 두면 $\mathbf{A}\boldsymbol{\Sigma} = \mathbf{I}$로 멱등이고 계수가 $p$이므로 $\mathbf{z}^T\boldsymbol{\Sigma}^{-1}\mathbf{z} \sim \chi^2_p$여야 한다.

    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    Sigma = np.array([[2., 0.5], [0.5, 1.]])
    Sinv = np.linalg.inv(Sigma)

    Z = rng.normal(size=(300_000, 2)) @ np.linalg.cholesky(Sigma).T   # z ~ N(0, Sigma)
    q = np.einsum('bi,ij,bj->b', Z, Sinv, Z)

    print("A Sigma 가 멱등인가:", np.allclose((Sinv @ Sigma) @ (Sinv @ Sigma), Sinv @ Sigma))
    print(f"평균 모의 {q.mean():.4f}   chi2_2 이론 2")
    print(f"분산 모의 {q.var():.4f}   chi2_2 이론 4")
    ```

    출력:

    ```
    A Sigma 가 멱등인가: True
    평균 모의 2.0044   chi2_2 이론 2
    분산 모의 4.0198   chi2_2 이론 4
    ```

    평균 $2$, 분산 $4$로 $\chi^2_2$와 맞는다.

    **이것이 마할라노비스 거리의 근거다.** $(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu}) \sim \chi^2_p$이므로, 이 값이 $\chi^2_p$의 상위 백분위수를 넘는 점을 다변량 이상치로 판정할 수 있다. $\boldsymbol{\Sigma}^{-1}$로 가중하는 것은 백색화($\boldsymbol{\Sigma}^{-1/2}$를 곱하는 것)와 같고, 백색화 뒤에는 표준정규가 되어 제곱합이 카이제곱이 된다. $\square$

<div class="drillbox" markdown>

**연습문제 10.**
회귀에서 $F = \dfrac{\text{SSR}/p'}{\text{SSE}/(n-p)}$가 두 이차형식의 비임을 이용해, $H_0$ 아래에서 실제로 $F_{p', n-p}$ 분포를 따름을 모의실험으로 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)
    n, p = 20, 3                       # 절편 + 설명변수 2개
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    P1 = np.ones((n, n)) / n           # 절편만 있는 모형의 사영
    A = H - P1                         # 회귀 부분, 계수 p-1 = 2
    M = np.eye(n) - H                  # 잔차 부분, 계수 n-p = 17

    print("A M = O 인가:", np.allclose(A @ M, 0), "  (독립성 조건)")
    print("계수:", np.linalg.matrix_rank(A), np.linalg.matrix_rank(M))

    B = 200_000
    Y = rng.normal(size=(B, n))        # H0: beta = 0 (절편만), sigma = 1
    ssr = np.einsum('bi,ij,bj->b', Y, A, Y)
    sse = np.einsum('bi,ij,bj->b', Y, M, Y)
    F = (ssr / 2) / (sse / 17)

    print(f"\n평균 모의 {F.mean():.4f}   이론 {17/(17-2):.4f}")
    for q in (0.5, 0.9, 0.95, 0.99):
        print(f"  q={q:<5} 모의 {np.quantile(F, q):7.4f}   F(2,17) {stats.f.ppf(q, 2, 17):7.4f}")
    ```

    출력:

    ```
    A M = O 인가: True   (독립성 조건)
    계수: 2 17

    평균 모의 1.1353   이론 1.1333
      q=0.5   모의  0.7235   F(2,17)  0.7222
      q=0.9   모의  2.6458   F(2,17)  2.6446
      q=0.95  모의  3.6142   F(2,17)  3.5915
      q=0.99  모의  6.1267   F(2,17)  6.1121
    ```

    분위수가 $F_{2,17}$과 잘 맞는다.

    **이 한 문제에 이 절의 내용이 모두 들어 있다.** $\mathbf{A}$와 $\mathbf{M}$이 대칭 멱등이라 각 이차형식이 카이제곱이 되고(계수 $2$와 $17$), $\mathbf{A}\mathbf{M} = \mathbf{O}$이라 크레이그 정리에 의해 둘이 독립이며, 독립인 두 카이제곱을 자유도로 나눈 비가 $F$ 분포의 정의다.

    $F$ 검정이 성립하려면 세 가지가 모두 필요하다는 점에 유의하라. 정규성(카이제곱이 되려면), 멱등성(자유도가 정수가 되려면), 직교성(독립이 되려면). 하나라도 깨지면 $F$ 분포는 근사에 지나지 않는다. $\square$

---

## 정리하며

이차형식의 카이제곱분포는 사영행렬의 멱등 구조 위에 서 있다. $\mathbf{z}$가 표준정규이고 $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬일 때 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$는 자유도 $r$인 카이제곱을 따른다. 크레이그 정리가 독립성 조건($\mathbf{A}\mathbf{B} = \mathbf{0}$)을 주고, 코크런 정리가 총제곱합의 분해에 대해 이 결과들을 통합한다. 이 결과들이 선형회귀 틀에서 F-검정, t-검정, 분산분석의 이론적 토대를 제공한다.
