# 카이제곱분포와 이차형식

고전 통계의 여러 검정통계량 — 적합도 통계량, 분산비, 잔차제곱합 — 은 정규확률벡터의 이차형식이다. 그런 이차형식이 언제 카이제곱분포를 따르는지 이해하는 것이 이 검정통계량들의 정확한 분포를 유도하는 데 필수적이다. 핵심 결과는 (앞 절들에서 다룬) **멱등행렬** 구조를 카이제곱분포와 연결한다. $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬이고 $\mathbf{z}$가 표준정규벡터이면 $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$이다.

## 카이제곱분포 복습

!!! info "정의 — 카이제곱분포"
    $Z_1, Z_2, \dots, Z_k$가 독립인 표준정규 확률변수($Z_i \sim N(0,1)$)이면

    $$
    Q = \sum_{i=1}^k Z_i^2 \sim \chi^2_k
    $$

    이다. 이 분포는 **자유도**가 $k$다. 벡터 표기로는 $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_k)$이면 $\mathbf{z}^T\mathbf{z} \sim \chi^2_k$이다.

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

!!! info "정리 — 멱등행렬을 갖는 이차형식"
    $\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{A} \in \mathbb{R}^{n \times n}$이 계수 $r$인 대칭 멱등행렬이라 하자. 그러면

    $$
    \mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r
    $$

**증명.** $\mathbf{A}$가 대칭이고 멱등이므로 (멱등행렬 이론에 따라) 고윳값이 모두 0 또는 1이다. $\operatorname{rank}(\mathbf{A}) = r$이므로 정확히 $r$개의 고윳값이 1이고 $n - r$개가 0이다.

대각화 $\mathbf{A} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$와 $\mathbf{w} = \mathbf{Q}^T\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$을 쓰면

$$
\mathbf{z}^T\mathbf{A}\mathbf{z} = \sum_{i=1}^n \lambda_i W_i^2 = \sum_{i:\,\lambda_i = 1} W_i^2
$$

이다. 이는 독립인 $\chi^2_1$ 변수 $r$개의 합이므로 $\mathbf{z}^T\mathbf{A}\mathbf{z} \sim \chi^2_r$이다. $\square$

## 일반적인 필요충분조건

멱등 조건은 충분할 뿐 아니라, (척도를 제외하면) 카이제곱분포를 얻기 위해 필요하기도 하다.

!!! info "정리 — 코크런 조건"
    $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고 $\mathbf{A}$가 대칭인 $n \times n$ 행렬이라 하자. $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$일 필요충분조건은 $\mathbf{A}$가 $\operatorname{rank}(\mathbf{A}) = r$인 멱등행렬인 것이다.

**증명 개요(필요성).** $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2 \sim \chi^2_r$이면 적률생성함수가 $\chi^2_r$의 것과 일치해야 한다. $\mathbf{z}^T\mathbf{A}\mathbf{z}/\sigma^2$의 적률생성함수는 $\prod_{i=1}^n(1 - 2\lambda_i t)^{-1/2}$이고 $\chi^2_r$의 것은 $(1 - 2t)^{-r/2}$이다. 이 둘이 같으려면 정확히 $r$개의 고윳값이 1이고 나머지가 0이어야 하며, 이는 $\mathbf{A}$가 계수 $r$인 멱등행렬이라는 뜻이다. $\square$

## 이차형식의 독립성

!!! info "정리 — 크레이그 정리"
    $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고 $\mathbf{A}$와 $\mathbf{B}$가 대칭인 $n \times n$ 행렬이라 하자. 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$와 $\mathbf{z}^T\mathbf{B}\mathbf{z}$가 **독립**일 필요충분조건은

    $$
    \mathbf{A}\mathbf{B} = \mathbf{0}
    $$

    인 것이다.

**증명 개요.** $(\mathbf{z}^T\mathbf{A}\mathbf{z}, \mathbf{z}^T\mathbf{B}\mathbf{z})$의 결합 적률생성함수가 주변 적률생성함수의 곱으로 인수분해될 필요충분조건이 $\mathbf{A}\mathbf{B} = \mathbf{0}$이다. $\mathbf{A}$와 $\mathbf{B}$를 동시에 대각화해 보면 알 수 있다. 조건 $\mathbf{A}\mathbf{B} = \mathbf{0}$은 두 이차형식이 독립인 $W_i^2$ 항들의 서로 겹치지 않는 부분집합만을 포함하도록 보장한다. $\square$

## 코크런 정리

코크런 정리는 카이제곱 결과와 독립성 결과를 제곱합 분해에 관한 하나의 강력한 진술로 결합한다.

!!! info "정리 — 코크런 정리"
    $\mathbf{z} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이고

    $$
    \mathbf{z}^T\mathbf{z} = \mathbf{z}^T\mathbf{A}_1\mathbf{z} + \mathbf{z}^T\mathbf{A}_2\mathbf{z} + \cdots + \mathbf{z}^T\mathbf{A}_k\mathbf{z}
    $$

    이라 하자. 여기서 $\mathbf{A}_1, \dots, \mathbf{A}_k$는 $\operatorname{rank}(\mathbf{A}_i) = r_i$이고 $r_1 + r_2 + \cdots + r_k = n$인 대칭 양반정치행렬이다. 그러면 이차형식 $\mathbf{z}^T\mathbf{A}_1\mathbf{z}/\sigma^2, \dots, \mathbf{z}^T\mathbf{A}_k\mathbf{z}/\sigma^2$은 서로 독립이고 $\mathbf{z}^T\mathbf{A}_i\mathbf{z}/\sigma^2 \sim \chi^2_{r_i}$이다.

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

## 요약

이차형식의 카이제곱분포는 사영행렬의 멱등 구조 위에 서 있다. $\mathbf{z}$가 표준정규이고 $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬일 때 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$는 자유도 $r$인 카이제곱을 따른다. 크레이그 정리가 독립성 조건($\mathbf{A}\mathbf{B} = \mathbf{0}$)을 주고, 코크런 정리가 총제곱합의 분해에 대해 이 결과들을 통합한다. 이 결과들이 선형회귀 틀에서 F-검정, t-검정, 분산분석의 이론적 토대를 제공한다.

## 연습문제

**연습문제 1.**
$\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{P}$가 $\operatorname{rank}(\mathbf{P}) = r$인 대칭 멱등행렬이라 하자. $\mathbf{Z}^T \mathbf{P} \mathbf{Z}$의 분포를 진술하고 이를 적용하라. 관측값이 $n = 20$인 단순선형회귀에서 $\mathbf{e} = (\mathbf{I} - \mathbf{H})\mathbf{y}$일 때 $\mathbf{e}^T \mathbf{e}/\sigma^2$의 분포를 구하라.

??? success "풀이"
    기본 카이제곱 정리에 의해 $\mathbf{Z}^T \mathbf{P} \mathbf{Z} \sim \chi^2_r$이다.

    단순선형회귀에서 $\mathbf{M} = \mathbf{I} - \mathbf{H}$는 계수가 $n - 2 = 18$인 대칭 멱등행렬이다. $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 아래에서 ($\mathbf{M}\mathbf{X} = \mathbf{0}$을 쓰면) 잔차는 $\mathbf{e} = \mathbf{M}\mathbf{y} = \mathbf{M}\boldsymbol{\varepsilon}$이다. 따라서

    $$
    \frac{\mathbf{e}^T\mathbf{e}}{\sigma^2} = \frac{\boldsymbol{\varepsilon}^T\mathbf{M}\boldsymbol{\varepsilon}}{\sigma^2} = \mathbf{z}^T\mathbf{M}\mathbf{z} \sim \chi^2_{18}
    $$

    이며, 여기서 $\mathbf{z} = \boldsymbol{\varepsilon}/\sigma \sim N(\mathbf{0}, \mathbf{I}_n)$이다.

---

**연습문제 2.**
$\mathbf{z} \sim N(\mathbf{0}, \mathbf{I}_n)$이고 $\mathbf{A}$가 고윳값 $\lambda_1, \ldots, \lambda_n$을 갖는 대칭행렬일 때, $\mathbf{z}^T\mathbf{A}\mathbf{z}$의 적률생성함수가 0의 어떤 근방에서

$$
M(t) = \prod_{i=1}^n (1 - 2\lambda_i t)^{-1/2}
$$

임을 보여라. 이를 이용해 $\mathbf{A}$가 계수 $r$인 멱등행렬일 때 적률생성함수가 $(1 - 2t)^{-r/2}$으로 환원됨을 확인하라.

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

---

**연습문제 3.**
$\mathbf{y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I}_n)$이라 하자. $H_0: \boldsymbol{\beta} = \mathbf{0}$ 아래에서 회귀제곱합이 $\mathrm{SSR}/\sigma^2 = \mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$이고 $\mathrm{SSE}/\sigma^2 \sim \chi^2_{n-p}$과 독립임을 보여라.

??? success "풀이"
    $H_0$ 아래에서 $\mathbf{y} = \boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I}_n)$이다. $\mathbf{z} = \mathbf{y}/\sigma$로 두면 $\mathrm{SSR}/\sigma^2 = \mathbf{z}^T\mathbf{H}\mathbf{z}$이고 $\mathbf{H}$는 계수 $p$인 대칭 멱등행렬이므로 $\chi^2_p$를 따른다.

    마찬가지로 $\mathrm{SSE}/\sigma^2 = \mathbf{z}^T\mathbf{M}\mathbf{z}$이고 $\mathbf{M}$은 계수 $n - p$인 대칭 멱등행렬이므로 $\chi^2_{n-p}$를 따른다.

    독립성은 크레이그 정리에서 따라온다: $\mathbf{H}\mathbf{M} = \mathbf{H}(\mathbf{I} - \mathbf{H}) = \mathbf{H} - \mathbf{H}^2 = \mathbf{0}$. $\square$

    이것이 바로 전체 F-검정의 설정이다. $H_0$ 아래에서 $F = (\mathrm{SSR}/p)/(\mathrm{SSE}/(n-p)) \sim F_{p, n-p}$이다.

---

**연습문제 4.**
흔한 분산분석 분해는 $\mathbf{y}^T\mathbf{y}$를 두 개의 이차형식으로 쪼갠다. $\mathbf{A}_1 = \mathbf{H}$, $\mathbf{A}_2 = \mathbf{I} - \mathbf{H}$로 두자. 코크런 정리가 적용됨을 확인하고, 각 조각의 카이제곱분포와 독립성을 결론지어라.

??? success "풀이"
    분해는 $\mathbf{y}^T\mathbf{y} = \mathbf{y}^T\mathbf{H}\mathbf{y} + \mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}$이다.

    코크런 정리의 가정 확인:

    - $\mathbf{A}_1 + \mathbf{A}_2 = \mathbf{H} + (\mathbf{I} - \mathbf{H}) = \mathbf{I}$. ✓
    - 각 $\mathbf{A}_i$가 대칭 양반정치다. ✓
    - $\operatorname{rank}(\mathbf{A}_1) + \operatorname{rank}(\mathbf{A}_2) = p + (n - p) = n$. ✓

    결론($\mathbf{y} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 아래에서): $\mathbf{y}^T\mathbf{H}\mathbf{y}/\sigma^2 \sim \chi^2_p$이고 $\mathbf{y}^T(\mathbf{I} - \mathbf{H})\mathbf{y}/\sigma^2 \sim \chi^2_{n-p}$이며 둘은 독립이다. $\square$

---

**연습문제 5.**
$\mathbf{z} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$이고 $\mathbf{A}$가 대칭일 때 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$의 **기댓값**을 계산하라. 그런 다음 $\boldsymbol{\Sigma} = \sigma^2\mathbf{I}$인 경우로 특수화하여 $\mathbb{E}[\mathbf{z}^T\mathbf{A}\mathbf{z}] = \sigma^2 \operatorname{tr}(\mathbf{A}) + \boldsymbol{\mu}^T\mathbf{A}\boldsymbol{\mu}$임을 보여라.

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

---

**연습문제 6.**
대립가설 아래에서 $\boldsymbol{\mu} \ne \mathbf{0}$인 $\mathbf{z} \sim N(\boldsymbol{\mu}, \mathbf{I}_n)$이고 $\mathbf{A}$가 계수 $r$인 대칭 멱등행렬이면, 이차형식 $\mathbf{z}^T\mathbf{A}\mathbf{z}$는 **비중심** 카이제곱분포 $\chi^2_r(\delta)$를 따른다. 비중심 모수 $\delta$를 찾고 F-검정의 검정력 계산에서 그 역할을 설명하라.

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
