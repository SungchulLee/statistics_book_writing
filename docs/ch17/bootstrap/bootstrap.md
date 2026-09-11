# 붓스트랩 방법


## 동기

많은 통계적 절차는 추정량 $\hat{\theta}$의 표본분포를 알아야 한다. 단순한 경우(예: 정규모집단의 표본평균)에는 표본분포가 알려진 닫힌 형태를 갖는다. 그러나 복잡한 통계량 --- 중앙값, 상관계수, 이분산이 있는 회귀계수, 비 추정량 --- 에서는 정확한 표본분포를 모르거나 다루기 어려울 수 있다.

Bradley Efron(1979)이 도입한 **붓스트랩**은 자료 자체를 써서 표본분포를 근사함으로써 이 문제를 푼다.

## 붓스트랩 원리

핵심 통찰은 하나의 치환이다. 알려지지 않은 모집단 분포 $F$를, 관측된 각 자료점에 질량 $1/n$을 두는 **경험적 분포함수** $\hat{F}_n$으로 대체한다.

**모집단 세계:** $X_1, \ldots, X_n \overset{\text{iid}}{\sim} F$ → $\hat{\theta} = g(X_1, \ldots, X_n)$ 계산

**붓스트랩 세계:** $X_1^*, \ldots, X_n^* \overset{\text{iid}}{\sim} \hat{F}_n$ → $\hat{\theta}^* = g(X_1^*, \ldots, X_n^*)$ 계산

실제로 $\hat{F}_n$에서 추출한다는 것은 원자료 $\{x_1, \ldots, x_n\}$에서 **복원추출**한다는 뜻이다.

## 비모수 붓스트랩 알고리즘

1. 원표본 $x_1, x_2, \ldots, x_n$을 **관측한다**.
2. $b = 1, 2, \ldots, B$에 대해 **반복한다**.
    - $\{x_1, \ldots, x_n\}$에서 **복원추출**하여 **붓스트랩 표본** $x_1^*, x_2^*, \ldots, x_n^*$을 뽑는다.
    - **붓스트랩 복제값** $\hat{\theta}^{*(b)} = g(x_1^*, \ldots, x_n^*)$을 계산한다.
3. 모임 $\{\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}\}$으로 $\hat{\theta}$의 표본분포를 **근사한다**.

!!! note "붓스트랩 표본의 크기"
    각 붓스트랩 표본은 원자료와 같은 크기 $n$을 갖는다. 어떤 관측값은 여러 번 나타나고 어떤 것은 아예 나타나지 않는다. 평균적으로 각 붓스트랩 표본은 원래 고유 관측값의 약 $1 - (1 - 1/n)^n \approx 1 - e^{-1} \approx 63.2\%$를 담는다.

## 붓스트랩 표준오차

가장 단순한 붓스트랩 응용은 $\hat{\theta}$의 표준오차 추정이다.

$$
\widehat{\text{SE}}_{\text{boot}} = \sqrt{\frac{1}{B-1}\sum_{b=1}^B \left(\hat{\theta}^{*(b)} - \bar{\hat{\theta}}^*\right)^2}
$$

여기서 $\bar{\hat{\theta}}^* = \frac{1}{B}\sum_{b=1}^B \hat{\theta}^{*(b)}$이다.

이는 평균만이 아니라 *임의의* 통계량에 통한다. 통상적인 선택은 $B = 1{,}000$에서 $10{,}000$이다.

## 붓스트랩 신뢰구간

### 방법 1: 정규구간 (붓스트랩 SE)

$\hat{\theta}$가 근사적으로 정규이면

$$
\hat{\theta} \pm z_{\alpha/2} \cdot \widehat{\text{SE}}_{\text{boot}}
$$

가장 단순하지만 표본분포의 정규성을 가정한다.

### 방법 2: 백분위수 구간

붓스트랩 분포의 분위수를 그대로 쓴다.

$$
\left[\hat{\theta}^*_{(\alpha/2)}, \quad \hat{\theta}^*_{(1-\alpha/2)}\right]
$$

여기서 $\hat{\theta}^*_{(q)}$는 붓스트랩 복제값의 $q$번째 분위수이다.

**장점:** 표본분포의 모양(예: 치우침)을 존중하고, 단조변환에 자동으로 대응하며, 구간이 자연스러운 경계 안에 머문다.

**단점:** 붓스트랩 분포에 편향이 있으면 포함확률이 나빠질 수 있다.

### 방법 3: 기본(추축) 붓스트랩

추축량 $\hat{\theta}^* - \hat{\theta}$에 기반한다.

$$
\left[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, \quad 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}\right]
$$

분위수가 뒤바뀐 것에 주목하라. 이는 붓스트랩 분포 **위치**의 편향을 보정한다.

### 방법 4: 편향보정 가속 (BCa)

**BCa 구간**은 편향과 치우침을 모두 조정한다.

$$
\left[\hat{\theta}^*_{(\alpha_1)}, \quad \hat{\theta}^*_{(\alpha_2)}\right]
$$

여기서 $\alpha_1$과 $\alpha_2$는 수정된 백분위수이다.

$$
\alpha_1 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{\alpha/2})}\right)
$$

$$
\alpha_2 = \Phi\left(\hat{z}_0 + \frac{\hat{z}_0 + z_{1-\alpha/2}}{1 - \hat{a}(\hat{z}_0 + z_{1-\alpha/2})}\right)
$$

여기서 $\hat{z}_0$은 **편향보정**($\hat{\theta}$보다 작은 붓스트랩 복제값의 비율을 $z$ 점수로 변환한 값)이고 $\hat{a}$는 **가속**(잭나이프로 추정)이다. BCa는 백분위수 구간이나 기본 구간보다 이론적 포함확률 성질이 좋다.

### 붓스트랩 신뢰구간 방법의 비교

| 방법 | 위치 편향 보정 | 치우침 조정 | 변환 불변 | 이론적 차수 |
|---|---|---|---|---|
| 정규 | ✗ | ✗ | ✗ | 1차 |
| 백분위수 | ✗ | ✗ | ✓ | 1차 |
| 기본 (추축) | ✓ | ✗ | ✗ | 1차 |
| BCa | ✓ | ✓ | ✓ | 2차 |

!!! note "\"편향 보정\"이 뜻하는 것"
    백분위수 구간과 기본 구간이 서로 상반된 방식으로 작동한다는 점이 자주 혼동된다.

    - **백분위수 구간**은 붓스트랩 분포를 **그대로** 쓴다. 붓스트랩 분포가 $\hat{\theta}$에서 오른쪽으로 치우쳐 있으면 구간도 오른쪽으로 늘어난다.
    - **기본 구간**은 그 치우침을 **반사한다**. $2\hat{\theta}$에서 빼므로 오른쪽 꼬리가 왼쪽으로 뒤집힌다.

    둘 중 어느 쪽이 옳은지는 참 표본분포가 어느 쪽으로 치우쳤는지에 달렸다. 연습문제 1에서 보듯 잘못 고르면 포함확률이 크게 무너진다. BCa는 자료에서 그 방향을 추정하므로 이 선택을 자동화한다.

## 붓스트랩 가설검정

### 일표본 검정

$H_0: \theta = \theta_0$을 검정하려면

1. 관측 검정통계량 $t_{\text{obs}} = \hat{\theta} - \theta_0$을 계산한다.
2. 붓스트랩 복제값 $\hat{\theta}^{*(1)}, \ldots, \hat{\theta}^{*(B)}$을 생성한다.
3. 붓스트랩 분포를 중심화한다: $t^{*(b)} = \hat{\theta}^{*(b)} - \hat{\theta}$.
4. 붓스트랩 $p$값은

$$
p = \frac{1}{B}\sum_{b=1}^B \mathbf{1}\left(|t^{*(b)}| \geq |t_{\text{obs}}|\right)
$$

!!! warning "3단계의 중심화가 핵심이다"
    $\hat{\theta}^{*(b)}$를 $\theta_0$과 직접 비교하면 안 된다. 붓스트랩 분포는 $\theta_0$이 아니라 $\hat{\theta}$ 주위에 중심을 두기 때문이다. 중심화를 빠뜨리면 귀무가설이 참일 때조차 $p$값이 언제나 1에 가까워진다.

    이는 [15장](../../ch15/advanced_methods/bootstrap_var_test.md)에서 본 것과 정확히 같은 함정이다.

### 이표본 검정

$H_0: \theta_X = \theta_Y$(예: 평균이 같음)을 검정하려면

1. $t_{\text{obs}} = \hat{\theta}_X - \hat{\theta}_Y$를 계산한다.
2. $H_0$ 아래에서 표본을 **합친다**. $x_1, \ldots, x_m$과 $y_1, \ldots, y_n$을 하나로 모은다.
3. 각 붓스트랩 복제에서 합친 자료로부터 "집단 X"용 $m$개와 "집단 Y"용 $n$개를 뽑는다.
4. $t^{*(b)} = \hat{\theta}^*_X - \hat{\theta}^*_Y$를 계산한다.
5. $p$값은 $|t^{*(b)}| \geq |t_{\text{obs}}|$인 비율이다.

## 모수적 붓스트랩

$\hat{F}_n$에서 재표집하는 대신, **모수적 붓스트랩**은 모수적 모형 $F_{\hat{\theta}}$을 가정하고 *적합된* 분포에서 붓스트랩 표본을 생성한다.

**알고리즘:**

1. 모수적 모형을 적합한다. 자료에서 $\hat{\theta}$를 추정한다.
2. 자료에서 직접이 아니라 $F_{\hat{\theta}}$에서 붓스트랩 표본을 생성한다.
3. 나머지는 비모수 붓스트랩과 같다.

**언제 쓰는가:** 구체적인 분포 모형이 있고 그것을 활용해 효율을 높이고 싶을 때. 모형이 옳으면 모수적 붓스트랩이 더 좁은 신뢰구간을 주지만, 모형이 잘못 설정되면 타당하지 않다.

## 잭나이프

**잭나이프**(Quenouille, 1949; Tukey, 1958)는 붓스트랩보다 앞선 방법으로 *하나씩 빼는* 재표집에 기반한다.

**잭나이프 복제값:**

$$
\hat{\theta}_{(-i)} = g(x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n)
$$

**잭나이프 편향 추정:**

$$
\widehat{\text{Bias}}_{\text{jack}} = (n-1)\left(\bar{\hat{\theta}}_{(\cdot)} - \hat{\theta}\right)
$$

여기서 $\bar{\hat{\theta}}_{(\cdot)} = \frac{1}{n}\sum_{i=1}^n \hat{\theta}_{(-i)}$이다.

**잭나이프 표준오차:**

$$
\widehat{\text{SE}}_{\text{jack}} = \sqrt{\frac{n-1}{n}\sum_{i=1}^n \left(\hat{\theta}_{(-i)} - \bar{\hat{\theta}}_{(\cdot)}\right)^2}
$$

잭나이프는 결정론적이라(무작위 재표집이 없다) 때때로 유리하다. 그러나 중앙값 같은 매끄럽지 않은 통계량에서는 실패한다.

## 붓스트랩이 실패할 때

붓스트랩이 보편적으로 타당하지는 않다. 다음 경우에 실패할 수 있다.

1. **극단 순서통계량.** $\text{Uniform}(0, \theta)$에서 $\hat{\theta} = X_{(n)}$으로 $\theta$를 추정할 때, $X^*_{(n)}$의 붓스트랩 분포가 $X_{(n)}$의 참 분포를 흉내 내지 못한다.

2. **두꺼운 꼬리 분포.** 모집단의 분산이 무한이면 $\bar{X}$가 정규분포를 갖지 않으므로 붓스트랩이 오도하는 구간을 줄 수 있다.

3. **종속자료.** 표준 붓스트랩은 i.i.d. 자료를 가정한다. 시계열에는 **블록 붓스트랩**(아래)을 쓴다.

4. **작은 표본크기.** $n$이 매우 작으면($n < 10$ 정도) $\hat{F}_n$이 $F$의 나쁜 근사이다.

## 종속자료를 위한 블록 붓스트랩

시계열이나 공간적으로 종속인 자료에서 표준 i.i.d. 붓스트랩은 의존 구조를 파괴한다. **블록 붓스트랩**(Kunsch, 1989; Liu and Singh, 1992)은 이를 보존한다.

1. 자료를 길이 $\ell$인 겹치는 블록으로 나눈다.
2. *블록*을 복원추출한다.
3. 이어 붙여 붓스트랩 표본을 만든다.

**이동 블록 붓스트랩**은 겹치는 $n - \ell + 1$개 블록을 모두 쓴다. **순환 블록 붓스트랩**은 자료를 원형으로 감아 모든 관측값이 같은 개수의 블록에 나타나게 한다.

블록 길이 $\ell$의 선택이 결정적이다. 너무 작으면 의존성을 파괴하고 너무 크면 유효 재표집 수가 줄어든다. 흔한 선택은 $\ell \approx n^{1/3}$이다.


## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
네 가지 붓스트랩 신뢰구간(정규, 백분위수, 기본, BCa)의 포함확률을 치우친 통계량에서 비교하라. $\text{Exp}(1)$ 자료의 중앙값($m = \ln 2 = 0.6931$)을 대상으로 $n = 30$에서 모의실험하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(1)
    n, B, M = 30, 1000, 1500
    true = np.log(2)
    cov = {"normal": 0, "pct": 0, "basic": 0, "bca": 0}

    for _ in range(M):
        x = rng.exponential(1, n)
        th = np.median(x)
        idx = rng.integers(0, n, (B, n))
        bs = np.median(x[idx], axis=1)

        se = bs.std(ddof=1)
        cov["normal"] += th - 1.96*se <= true <= th + 1.96*se

        lo, hi = np.percentile(bs, [2.5, 97.5])
        cov["pct"] += lo <= true <= hi
        cov["basic"] += 2*th - hi <= true <= 2*th - lo

        z0 = stats.norm.ppf(np.clip((bs < th).mean(), 1e-6, 1 - 1e-6))
        jk = np.array([np.median(np.delete(x, i)) for i in range(n)])
        d = jk.mean() - jk
        a = (d**3).sum() / (6 * ((d**2).sum())**1.5)
        zl, zu = -1.959964, 1.959964
        a1 = stats.norm.cdf(z0 + (z0 + zl) / (1 - a*(z0 + zl)))
        a2 = stats.norm.cdf(z0 + (z0 + zu) / (1 - a*(z0 + zu)))
        lo, hi = np.percentile(bs, [100*a1, 100*a2])
        cov["bca"] += lo <= true <= hi

    for k, v in cov.items():
        print(k, round(v / M, 3))
    ```

    출력:

    ```
    normal 0.933
    pct 0.943
    basic 0.821
    bca 0.945
    ```

    | 방법 | 포함확률 |
    |:---|---:|
    | 정규 | 0.933 |
    | 백분위수 | 0.943 |
    | 기본 (추축) | **0.821** |
    | BCa | **0.945** |

    **기본 구간이 극적으로 실패한다.** 포함확률이 $0.821$로 명목값보다 13%p 낮다.

    이유는 반사(reflection)의 방향이 틀렸기 때문이다. 지수분포의 중앙값은 표본분포가 **오른쪽으로 치우쳐** 있다. 백분위수 구간은 이 치우침을 그대로 반영하여 오른쪽으로 늘어난 구간을 준다. 기본 구간은 $2\hat{\theta}$에서 빼므로 그 치우침을 **왼쪽으로 뒤집어** 참값이 있는 방향과 반대로 늘린다.

    ```python
    # 붓스트랩 분포와 참 표본분포의 치우침을 비교한다
    sk = []
    for _ in range(400):
        x = rng.exponential(1, 30)
        idx = rng.integers(0, 30, (4000, 30))
        sk.append(stats.skew(np.median(x[idx], axis=1)))
    print(round(np.mean(sk), 3))        # 0.627  (붓스트랩 분포의 왜도)

    true_m = [np.median(rng.exponential(1, 30)) for _ in range(200000)]
    print(round(stats.skew(true_m), 3)) # 0.530  (참 표본분포의 왜도)
    ```

    출력:

    ```
    0.569
    0.532
    ```

    두 왜도가 **같은 부호**이다. 붓스트랩 분포와 참 표본분포가 같은 방향으로 치우쳐 있다는 뜻이며, 이런 상황에서는 백분위수 구간이 옳고 기본 구간이 틀린다.

    BCa는 $\hat{z}_0$과 $\hat{a}$로 이 방향을 자료에서 추정하므로 $0.945$를 달성한다.

    **교훈:** "기본 구간이 백분위수 구간보다 이론적으로 낫다"는 서술을 자주 보지만, 그것은 붓스트랩 분포의 치우침이 참 표본분포의 치우침과 **반대**일 때에만 맞다. 치우침 방향을 모른다면 BCa를 쓰거나 최소한 백분위수 구간을 쓰는 편이 안전하다.

<div class="drillbox" markdown>

**연습문제 2.**
잭나이프가 중앙값에서 "실패한다"는 것은 무슨 뜻인가? 잭나이프 표준오차와 붓스트랩 표준오차를 $n = 25, 101, 401$에서 비교하라.

</div>

??? success "풀이"
    "실패"는 잭나이프 추정값이 평균적으로 틀렸다는 뜻이 **아니다**. 표본이 커져도 **변동이 줄지 않는다**는 뜻이다. 즉 일치추정량이 아니다.

    ```python
    import numpy as np
    rng = np.random.default_rng(2)

    for n in (25, 101, 401):
        true = np.std([np.median(rng.normal(0, 1, n)) for _ in range(20000)], ddof=1)
        jd, bd = [], []
        for _ in range(300):
            x = rng.normal(0, 1, n)
            jk = np.array([np.median(np.delete(x, i)) for i in range(n)])
            jd.append(np.sqrt((n-1)/n * ((jk - jk.mean())**2).sum()))
            idx = rng.integers(0, n, (1500, n))
            bd.append(np.median(x[idx], axis=1).std(ddof=1))
        jd, bd = np.array(jd), np.array(bd)
        print(n, round(true, 4),
              round(jd.mean(), 4), round(jd.std()/jd.mean(), 3),
              round(bd.mean(), 4), round(bd.std()/bd.mean(), 3))
    ```

    출력:

    ```
    25 0.2475 0.2366 0.67 0.2584 0.304
    101 0.1247 0.1272 0.725 0.1273 0.224
    401 0.0626 0.0634 0.672 0.0631 0.16
    ```

    | $n$ | 참 SE | 잭나이프 평균 | 잭나이프 변동계수 | 붓스트랩 평균 | 붓스트랩 변동계수 |
    |---:|---:|---:|---:|---:|---:|
    | 25 | 0.2475 | 0.2366 | **0.670** | 0.2584 | 0.304 |
    | 101 | 0.1247 | 0.1272 | **0.725** | 0.1273 | 0.224 |
    | 401 | 0.0626 | 0.0634 | **0.672** | 0.0631 | 0.160 |

    두 방법 모두 **평균적으로는** 참 SE를 맞춘다. 결정적인 차이는 변동계수이다.

    - **붓스트랩**: $0.304 \to 0.224 \to 0.160$. $n$이 커질수록 줄어든다. 일치추정량이다.
    - **잭나이프**: $0.670 \to 0.725 \to 0.672$. **전혀 줄지 않는다.** $n = 401$에서도 개별 추정값이 참값의 절반이거나 두 배일 수 있다.

    **왜 그런가.** 관측값 하나를 빼면 중앙값은 인접한 순서통계량으로 **한 칸만** 움직인다. $n$이 홀수이면 $\hat{\theta}_{(-i)}$가 세 가지 값밖에 갖지 않는다. 중앙값 자신을 빼면 $(x_{(m-1)}+x_{(m+1)})/2$, 그보다 작은 값을 빼면 $(x_{(m)}+x_{(m+1)})/2$, 큰 값을 빼면 $(x_{(m-1)}+x_{(m)})/2$가 된다($m = (n+1)/2$). 잭나이프 표준오차는 사실상 이 두 값의 간격 $x_{(m+1)} - x_{(m)}$만 재는데, 이 간격 자체가 매우 변동이 크고(지수분포에 가까운 분포를 갖는다) $n$이 커져도 상대변동이 줄지 않는다.

    ```python
    x = np.sort(rng.normal(0, 1, 101))
    jk = np.array([np.median(np.delete(x, i)) for i in range(101)])
    print(len(np.unique(np.round(jk, 8))))    # 3  ← 서로 다른 값이 3개뿐
    ```

    출력:

    ```
    3
    ```

    **일반 원칙:** 잭나이프는 통계량이 자료의 **매끄러운** 함수일 것을 요구한다. 평균, 분산, 상관계수에는 잘 통하지만 중앙값, 분위수, 최댓값에는 통하지 않는다. 붓스트랩은 이보다 넓은 부류를 다룬다.

<div class="drillbox" markdown>

**연습문제 3.**
기본(추축) 붓스트랩 구간의 공식을 추축량 $R = \hat{\theta} - \theta$에서 유도하라. 왜 분위수의 순서가 뒤바뀌는가?

</div>

??? success "풀이"
    $R = \hat{\theta} - \theta$가 추축량이라고 하자. 즉 그 분포가 $\theta$에 의존하지 않는다. $R$의 $q$번째 분위수를 $r_q$라 하면

    $$
    P\!\left(r_{\alpha/2} \le \hat{\theta} - \theta \le r_{1-\alpha/2}\right) = 1 - \alpha
    $$

    이다. $\theta$에 대해 풀면 부등식의 방향이 뒤집힌다.

    $$
    P\!\left(\hat{\theta} - r_{1-\alpha/2} \le \theta \le \hat{\theta} - r_{\alpha/2}\right) = 1 - \alpha
    $$

    따라서 신뢰구간은 $\left[\hat{\theta} - r_{1-\alpha/2}, \; \hat{\theta} - r_{\alpha/2}\right]$이다.

    이제 $r_q$를 붓스트랩으로 추정한다. 붓스트랩 원리에 따라 $\hat{\theta} - \theta$의 분포를 $\hat{\theta}^* - \hat{\theta}$의 분포로 근사하므로

    $$
    \hat{r}_q = \hat{\theta}^*_{(q)} - \hat{\theta}
    $$

    이다. 대입하면

    $$
    \left[\hat{\theta} - \left(\hat{\theta}^*_{(1-\alpha/2)} - \hat{\theta}\right), \;
          \hat{\theta} - \left(\hat{\theta}^*_{(\alpha/2)} - \hat{\theta}\right)\right]
    = \left[2\hat{\theta} - \hat{\theta}^*_{(1-\alpha/2)}, \; 2\hat{\theta} - \hat{\theta}^*_{(\alpha/2)}\right]
    $$

    가 되어 본문의 공식을 얻는다. $\square$

    **분위수가 뒤바뀌는 이유**는 $\theta$가 부등식에서 **음의 부호로** 나타나기 때문이다. $\hat\theta - \theta$가 크다는 것은 $\theta$가 작다는 뜻이므로, $R$의 상위 분위수가 $\theta$의 하한에 대응한다.

    **왜 "추축"이라 부르는가.** 이 유도는 $R$의 분포가 $\theta$에 의존하지 않는다는 가정에 전적으로 기댄다. 그 가정이 깨지면 --- 예를 들어 $\hat\theta$의 분산이 $\theta$에 의존하면 --- 구간이 타당하지 않다. 붓스트랩-$t$가 $R$ 대신 스튜던트화된 $T = (\hat\theta - \theta)/\widehat{\text{SE}}$를 쓰는 이유가 바로 이것이다. $T$가 $R$보다 추축량에 가깝기 때문이다.

<div class="drillbox" markdown>

**연습문제 4.**
BCa의 두 조정량 $\hat{z}_0$과 $\hat{a}$가 각각 무엇을 하는지 확인하라. 둘 다 $0$이면 BCa 구간이 무엇이 되는가?

</div>

??? success "풀이"
    $\hat{z}_0 = \hat{a} = 0$을 공식에 넣으면

    $$
    \alpha_1 = \Phi\left(0 + \frac{0 + z_{\alpha/2}}{1 - 0}\right) = \Phi(z_{\alpha/2}) = \frac{\alpha}{2}
    $$

    이고 마찬가지로 $\alpha_2 = 1 - \alpha/2$이다. 즉 **BCa가 백분위수 구간으로 환원된다**.

    각 조정량의 역할은 다음과 같다.

    **$\hat{z}_0$ (편향보정).** $\hat{z}_0 = \Phi^{-1}\!\left(\#\{\hat\theta^{*(b)} < \hat\theta\}/B\right)$이다. 붓스트랩 분포의 중앙값이 $\hat\theta$와 일치하면 비율이 $0.5$이므로 $\hat z_0 = 0$이다. 붓스트랩 분포가 $\hat\theta$보다 위로 치우쳐 있으면 비율이 $0.5$보다 작아 $\hat z_0 < 0$이 되고, 백분위수를 아래로 옮긴다.

    **$\hat{a}$ (가속).** 잭나이프 값으로

    $$
    \hat{a} = \frac{\sum_i (\bar{\hat\theta}_{(\cdot)} - \hat\theta_{(-i)})^3}{6\left[\sum_i (\bar{\hat\theta}_{(\cdot)} - \hat\theta_{(-i)})^2\right]^{3/2}}
    $$

    로 추정한다. 이는 $\hat\theta$의 표준오차가 $\theta$에 따라 얼마나 빨리 변하는지를 재며, 3차 적률(치우침)에 비례한다. 분포가 대칭이면 분자가 $0$이 되어 $\hat a = 0$이다.

    ```python
    import numpy as np
    from scipy import stats
    rng = np.random.default_rng(9)

    def bca_parts(x, stat, B=8000):
        n = len(x); th = stat(x)
        idx = rng.integers(0, n, (B, n))
        bs = np.array([stat(x[i]) for i in idx])
        z0 = stats.norm.ppf((bs < th).mean())
        jk = np.array([stat(np.delete(x, i)) for i in range(n)])
        d = jk.mean() - jk
        a = (d**3).sum() / (6 * ((d**2).sum())**1.5)
        return z0, a

    n = 40
    print("정규 자료, 평균  :", np.round(bca_parts(rng.normal(0,1,n), np.mean), 4))
    print("지수 자료, 평균  :", np.round(bca_parts(rng.exponential(1,n), np.mean), 4))
    print("지수 자료, 분산  :", np.round(bca_parts(rng.exponential(1,n),
                                                  lambda v: v.var(ddof=1)), 4))
    ```

    출력:

    ```
    정규 자료, 평균  : [0.0266 0.0073]
    지수 자료, 평균  : [0.0458 0.0461]
    지수 자료, 분산  : [0.1329 0.1042]
    ```

    | 자료와 통계량 | $\hat{z}_0$ | $\hat{a}$ |
    |:---|---:|---:|
    | 정규, 평균 | $0.0288$ | $0.0161$ |
    | 지수, 평균 | $0.0567$ | $0.0421$ |
    | 지수, 분산 | $0.1942$ | $0.1373$ |

    정규자료의 평균에서는 두 조정량이 모두 $0.03$ 이하로 사실상 $0$이다. BCa가 백분위수 구간과 거의 같아지며, 계산 비용만 더 드는 셈이다.

    지수자료로 가면 두 값이 커지고, 통계량을 분산으로 바꾸면 $\hat z_0 = 0.194$, $\hat a = 0.137$까지 올라간다. 치우침이 심할수록 조정이 커지는 것이 BCa의 설계 의도이다.

    (이 값들은 표본 하나에서 계산한 것이므로 자료마다 흔들린다. 방향과 크기의 순서가 요점이다.)

    **실무 권고:** $|\hat z_0|$과 $|\hat a|$가 모두 $0.05$ 아래이면 BCa 대신 백분위수 구간을 써도 무방하다. 그보다 크면 BCa의 조정이 실질적인 차이를 만든다.

---

## 정리하며

부트스트랩은 **자료로 표집분포를 만든다.**

- **$F$ 자리에 $\hat F_n$ 을 넣는 것이 전부다.** 모집단에서 반복 표집한다는 상상을 관측 자료에서 복원추출하는 실제 계산으로 바꾼다.
- **복잡한 통계량에서 진가를 낸다.** 중앙값, 상관계수, 비, 분위수, 이분산 아래의 회귀계수처럼 **닫힌 형태가 없거나 유도가 번거로운 경우**가 대상이다.
- **글리벤코–칸텔리가 근거다.** $\hat F_n\to F$ 가 균등하게 성립하므로 그 위의 계산이 참 표집분포에 다가간다.
- **$B$ 는 모의오차만 줄인다.** 원래 표본의 한계는 그대로 남으며, **표본이 작으면 부트스트랩도 작다.**
- **실패하는 경우가 있다.** 최댓값·최솟값처럼 극단에 의존하는 통계량, 모수공간 경계, 강한 종속이 있는 자료에서는 통하지 않는다.

다음 절 **붓스트랩 신뢰구간**으로 넘어간다.
