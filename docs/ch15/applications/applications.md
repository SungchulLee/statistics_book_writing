# 회귀와 분산분석에서의 응용


분산의 동일성 검정은 여러 통계 방법, 특히 회귀분석과 분산분석에서 결정적인 단계이다. 두 기법 모두 분산의 동질성(등분산성)에 대한 가정이 타당하고 신뢰할 만한 추론을 보장하는 근본적인 역할을 한다.

---

## 1. 회귀에서의 분산 검정

회귀분석의 핵심 가정 가운데 하나가 **등분산성**이다. 잔차의 분산이 설명변수의 모든 수준에서 일정해야 한다. 이 가정이 위배되면 추론이 무효가 되어 잘못된 결론으로 이어질 수 있다.

!!! warning "이분산은 계수를 편향시키지 않는다"
    "이분산이 있으면 회귀계수 추정값이 편향된다"는 서술을 흔히 보는데 **틀렸다**. Gauss-Markov 정리의 불편성 부분은 등분산을 요구하지 않는다. $E[\varepsilon \mid X] = 0$이면 이분산이 있어도 OLS는 불편이고 일치성을 갖는다.

    이분산이 망가뜨리는 것은 **표준오차**이다. OLS 표준오차 공식 $s^2(\mathbf{X}'\mathbf{X})^{-1}$이 등분산을 가정하므로, 이분산 아래에서 이 값이 편향된다. 그 결과 $t$ 통계량, $p$값, 신뢰구간이 모두 틀린다.

    실무적 함의가 다르다. 계수가 편향된다면 결과를 폐기해야 하지만, 표준오차만 문제라면 **로버스트 표준오차로 고칠 수 있다.** 계수 추정값은 그대로 쓴다.

모의실험으로 이 주장을 확인해 보자. $y = 2 + x + \varepsilon$에서 오차의 표준편차를 $0.2 + 0.3x^2$으로 두어 $x$가 커질수록 산포가 급격히 커지게 했다. $n = 100$짜리 자료를 4000번 생성해 매번 OLS로 적합한다.

![이분산 아래에서 기울기 추정값의 분포와 구간의 실제 포함률](./img/hetero_unbiased_se.png)

왼쪽이 자료 한 벌이다. 오른쪽으로 갈수록 점들이 나팔처럼 벌어지는 전형적인 이분산이다. 그런데 OLS 적합선(빨강 점선)은 참 회귀직선(검정)과 거의 겹친다. 분산이 들쭉날쭉해도 **최소제곱은 여전히 올바른 직선을 찾아낸다.**

오른쪽이 4000번의 기울기 추정값이다. 분포의 중심이 $0.9981$로 참값 1과 사실상 같다. $E[\varepsilon \mid X] = 0$만 성립하면 불편성은 이분산과 무관하게 유지되며, 이것이 가우스–마르코프 정리에서 등분산이 **효율성** 부분에만 쓰인다는 사실의 구체적 의미다.

문제는 그 아래 숫자다. **명목 95% 신뢰구간이 참값을 덮은 비율이 OLS 표준오차로는 $0.889$에 그친다.** 100번 중 11번은 참값이 구간 밖에 있다는 뜻이고, 그만큼 $p$값도 작게 나온다. 이분산 자료에서 "$p < 0.05$이므로 유의하다"는 결론이 실제보다 자주 나오는 이유가 이것이다.

HC3 로버스트 표준오차로 바꾸면 포함률이 $0.952$로 회복된다. **계수는 그대로 두고 표준오차 계산만 바꾸었을 뿐인데 추론이 정상으로 돌아온다.** 이분산을 발견했을 때 모형을 버리거나 변수를 변환할 필요가 없는 경우가 많다는 실무적 결론이 여기서 나온다. 다만 효율성은 여전히 손해이므로, 오차 구조를 안다면 가중최소제곱이 더 정밀한 추정값을 준다.

### 등분산성과 이분산

**등분산성:** 잔차의 분산이 독립변수의 모든 수준에서 일정하다.

$$
\text{Var}(\epsilon_i) = \sigma^2 \quad \text{(모든 } i \text{에 대해)}
$$

**이분산:** 잔차의 분산이 독립변수에 따라 달라진다.

$$
\text{Var}(\epsilon_i) = f(x_i)
$$

이분산은 회귀계수 추정의 비효율성과 잘못된 $p$값으로 이어져 가설검정에 영향을 준다. 따라서 회귀분석을 진행하기 전에 이분산을 검정하는 것이 중요하다.

### 등분산성 검정

등분산성을 확인하는 검정으로 **Breusch-Pagan 검정**과 **White 검정**이 있다. 이 검정들은 잔차와 설명변수의 관계를 조사한다.

### Breusch-Pagan 검정

Breusch-Pagan 검정은 잔차의 분산이 모형의 독립변수와 관련되어 있는지 검정하여 이분산을 탐지한다.

**가설:**

- $H_0$: 잔차가 등분산이다. 곧 분산이 일정하다: $\text{Var}(\epsilon_i) = \sigma^2$
- $H_1$: 잔차분산이 독립변수의 함수이다: $\text{Var}(\epsilon_i) = f(x_i)$

**Python 구현:**

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 회귀에서의 등분산 검정. $n = 10$짜리 자료에 `Y ~ X`를 적합하고 `het_breuschpagan` 으로 Breusch-Pagan 검정을 돌린다.

**(1)** 이 함수가 돌려주는 LM 통계량과 F 통계량이 **제곱잔차를 설명변수에 회귀시킨 보조회귀**의 $nR^2$과 $F$임을 밝히고, 두 수를 $R^2$ 하나로부터 손으로 조립해 함수의 값과 맞추시오. `robust=False`로 바뀌는 것은 무엇인가.

**(2)** 두 판본의 **실제 크기**를 등분산 자료에서 $n = 10, 50, 200$으로 재고, 어느 쪽이 표본을 키워 고쳐지고 어느 쪽이 고쳐지지 않는지 밝히시오. 또 $n = 10$에서 검정력이 얼마인지 재어 본문의 $p = 0.777$을 어떻게 읽어야 하는지 결론하시오.

</div>

??? success "풀이"

    **(1) 보조회귀 하나가 전부다.** 원래 회귀 $y_i = \beta_0 + \beta_1 x_i + \varepsilon_i$의 OLS 잔차를 $e_i$라 하자. 등분산이면 $E[\varepsilon_i^2]$이 $x_i$와 무관해야 하므로, 제곱잔차를 설명변수에 회귀시켜

    $$
    e_i^2 = \gamma_0 + \gamma_1 x_i + u_i
    $$

    의 기울기가 $0$인지 보면 된다. 이 **보조회귀**의 결정계수를 $R_{\text{aux}}^2$, 설명변수 개수를 $k$라 하면 코엔커(1981)의 **스튜던트화 BP 통계량**은

    $$
    \text{LM} = n R_{\text{aux}}^2 \;\xrightarrow{\;d\;}\; \chi^2_k \quad (H_0 \text{ 아래})
    $$

    이고, F 판본은 그 보조회귀 자체의 F 통계량

    $$
    F = \frac{R_{\text{aux}}^2/k}{(1-R_{\text{aux}}^2)/(n-k-1)} \;\sim\; F(k,\; n-k-1)
    $$

    이다. **둘 다 $R_{\text{aux}}^2$ 하나의 함수**이므로, $R_{\text{aux}}^2$만 알면 두 수를 손으로 적을 수 있다. 여기서는 $k = 1$, $n = 10$이다.

    `robust=False`로 바꾸면 **원래 Breusch-Pagan(1979)** 통계량이 나온다. 그쪽은 제곱잔차를 $\hat\sigma^2 = \frac1n\sum e_i^2$으로 나눈 $g_i = e_i^2/\hat\sigma^2$을 설명변수에 회귀시킨 뒤 설명제곱합의 절반

    $$
    \text{BP} = \tfrac12 \,\text{ESS}_g \;\xrightarrow{\;d\;}\; \chi^2_k
    $$

    을 쓴다. **$\chi^2_k$ 극한이 오차의 정규성에 기대고 있다는 것이 이 판본의 약점**이고, 코엔커 판본은 $\hat\sigma^2$ 대신 $g$의 실제 표본분산으로 눈금을 맞춰 그 의존을 없앤 것이다. 두 판본이 같은 자료에서 다른 수를 준다.

    **(2) 먼저 함수를 그대로 돌린다.**

    ```python
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
    import pandas as pd

    data = pd.DataFrame({
        'X': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Y': [2, 4, 6, 8, 10, 9, 15, 16, 18, 20]
    })

    model = smf.ols('Y ~ X', data=data).fit()

    # 회귀에서 등분산성은 잔차의 퍼짐이 적합값에 따라 달라지지 않는다는 가정이다.
    # Breusch-Pagan 은 잔차의 제곱을 설명변수에 회귀해 그 관계를 찾는다.
    lm_stat, lm_p, f_stat, f_p = het_breuschpagan(model.resid, model.model.exog)

    print(f"Breusch-Pagan LM = {lm_stat:.4f}, p = {lm_p:.4f}")
    print(f"Breusch-Pagan F  = {f_stat:.4f}, p = {f_p:.4f}")
    ```

    출력:

    ```text
    Breusch-Pagan LM = 0.0803, p = 0.7769
    Breusch-Pagan F  = 0.0648, p = 0.8055
    ```

    **이제 (1)을 손으로 조립해 맞추고, 두 판본의 크기와 검정력을 잰다.**

    ```python
    import numpy as np
    import statsmodels.api as sm
    from scipy import stats

    # --- (1) 손으로 조립해 het_breuschpagan 과 맞춘다. 위 model 을 이어받는다. ---
    e = model.resid.values
    X = model.model.exog                       # [1, x]
    n, k = len(e), X.shape[1] - 1

    aux = sm.OLS(e ** 2, X).fit()              # 보조회귀: e^2 을 x 에 회귀
    R2 = aux.rsquared
    lm_sm, p_lm, f_sm, p_f = het_breuschpagan(model.resid, X)

    print(f"보조회귀 R^2  = {R2:.8f}")
    print(f"n R^2 (손)    = {n * R2:.8f}      LM  (statsmodels) = {lm_sm:.8f}")
    F_hand = (R2 / k) / ((1 - R2) / (n - k - 1))
    print(f"F     (손)    = {F_hand:.8f}      F   (statsmodels) = {f_sm:.8f}")
    print(f"p(chi2_{k})     = {stats.chi2.sf(n * R2, k):.6f}      p   (statsmodels) = {p_lm:.6f}")
    print(f"p(F({k},{n - k - 1}))     = {stats.f.sf(F_hand, k, n - k - 1):.6f}      p   (statsmodels) = {p_f:.6f}")

    # 정규성을 쓰는 원래 BP(1979) 는 robust=False 다. (1/2)*ESS of e^2/sigmahat^2
    sig2 = (e ** 2).mean()
    g = e ** 2 / sig2
    ess0 = ((sm.OLS(g, X).fit().fittedvalues - g.mean()) ** 2).sum()
    print(f"\n원래 BP = ESS/2 = {ess0 / 2:.8f}      robust=False  = "
          f"{het_breuschpagan(model.resid, X, robust=False)[0]:.8f}")


    # --- (2) 두 판본의 크기와 검정력을 모의실험으로 잰다 ---
    def bp_both(E, X):
        """E: (R,n) 잔차.  (코엔커 nR^2, 원래 BP) 를 한꺼번에."""
        R, n = E.shape
        P = X @ np.linalg.pinv(X)
        G = E ** 2
        Gc = G - G.mean(axis=1, keepdims=True)
        ess = ((Gc @ P.T) ** 2).sum(axis=1)     # 중심화한 g 의 사영 제곱합 = ESS
        tss = (Gc ** 2).sum(axis=1)
        return n * ess / tss, 0.5 * ess / G.mean(axis=1) ** 2


    def size_power(n, R, dist, slope, seed):
        rng = np.random.default_rng(seed)
        x = np.linspace(1, 10, n)
        X = np.column_stack([np.ones(n), x])
        if dist == "Normal":
            u = rng.normal(size=(R, n))
        elif dist == "t(3)":
            u = rng.standard_t(3, size=(R, n))
        else:                                    # 로그정규를 평균 0, 분산 1 로 표준화
            z = rng.lognormal(0, 1, size=(R, n))
            u = (z - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)
        y = 2 + x + u * (1 + slope * x)          # slope = 0 이면 등분산
        P = X @ np.linalg.pinv(X)
        lm, bp = bp_both(y - y @ P.T, X)
        return (stats.chi2.sf(lm, 1) < 0.05).mean(), (stats.chi2.sf(bp, 1) < 0.05).mean()


    print("\n등분산 자료에서의 실제 크기 (명목 0.05, 4만 회)")
    print(f"{'n':>5}  {'errors':<8}{'Koenker':>10}{'orig BP':>10}")
    for nn in (10, 50, 200):
        for d in ("Normal", "t(3)", "LogN"):
            a, b = size_power(nn, 40_000, d, 0.0, seed=1)
            print(f"{nn:>5}  {d:<8}{a:>10.4f}{b:>10.4f}")

    print("\n이분산 자료에서의 검정력 (정규오차, sd = 1 + 0.5x)")
    for nn in (10, 30, 100, 300):
        a, b = size_power(nn, 40_000, "Normal", 0.5, seed=2)
        print(f"  n = {nn:>3}:  코엔커 {a:.4f}   원래 BP {b:.4f}")
    ```

    출력:

    ```text
    보조회귀 R^2  = 0.00803084
    n R^2 (손)    = 0.08030838      LM  (statsmodels) = 0.08030838
    F     (손)    = 0.06476684      F   (statsmodels) = 0.06476684
    p(chi2_1)     = 0.776880      p   (statsmodels) = 0.776880
    p(F(1,8))     = 0.805534      p   (statsmodels) = 0.805534

    원래 BP = ESS/2 = 0.23674242      robust=False  = 0.23674242

    등분산 자료에서의 실제 크기 (명목 0.05, 4만 회)
        n  errors     Koenker   orig BP
       10  Normal      0.0580    0.0237
       10  t(3)        0.0663    0.0637
       10  LogN        0.1103    0.1262
       50  Normal      0.0507    0.0463
       50  t(3)        0.0428    0.2449
       50  LogN        0.0523    0.4062
      200  Normal      0.0502    0.0489
      200  t(3)        0.0409    0.3776
      200  LogN        0.0435    0.5475

    이분산 자료에서의 검정력 (정규오차, sd = 1 + 0.5x)
      n =  10:  코엔커 0.1656   원래 BP 0.1189
      n =  30:  코엔커 0.5497   원래 BP 0.6179
      n = 100:  코엔커 0.9913   원래 BP 0.9958
      n = 300:  코엔커 1.0000   원래 BP 1.0000
    ```

    **(1)의 조립이 여덟 자리까지 맞는다.** 보조회귀의 $R_{\text{aux}}^2 = 0.00803084$ 하나에서 $n R^2 = 0.08030838$과 $F = 0.06476684$가 나오고, 둘 다 `het_breuschpagan` 의 값과 소수 여덟째 자리까지 같다. $p$값도 $\chi^2_1$과 $F(1,8)$에서 각각 $0.776880$과 $0.805534$로 맞는다. **같은 $R^2$을 두 참조분포에 들이댄 것이 LM 과 F 판본의 차이 전부**이고, 그래서 $n = 10$처럼 작은 표본에서 두 $p$값이 $0.78$과 $0.81$로 갈린다. $\chi^2$ 쪽은 $n \to \infty$의 극한이므로 작은 표본에서는 F 쪽이 낫다.

    원래 BP 도 맞는다. 손으로 센 $\tfrac12 \text{ESS}_g = 0.23674242$가 `robust=False` 의 값과 같다. **같은 자료에서 $0.0803$과 $0.2367$, 곧 세 배 차이**다. 어느 판본을 돌렸는지 모르면 통계량을 보고할 수도 없다.

    **(2) 표가 이 장의 주제를 그대로 되풀이한다.** 등분산 자료의 실제 크기를 보라.

    - **코엔커 판본은 $n$을 키우면 고쳐진다.** $n = 10$에서 $0.058$–$0.110$으로 흐트러져 있다가 $n = 50$에서 $0.043$–$0.052$, $n = 200$에서 $0.041$–$0.050$으로 세 오차분포 모두 명목값에 모인다.
    - **원래 BP 는 $n$을 키우면 더 나빠진다.** $t(3)$에서 $0.064 \to 0.245 \to 0.378$, 로그정규에서 $0.126 \to 0.406 \to 0.548$이다. 표본을 스무 배 늘리는 동안 거짓 양성률이 다섯 배, 네 배로 올라갔다.

    **이것이 15장이 되풀이해 온 바로 그 실패다.** $F$ 검정과 Bartlett 검정도 같은 일을 하며, 5.3절이 그 극한 오류율이 모집단 첨도만의 함수이고 표본을 키워도 낫지 않음을 유도해 두었다. 분산에 관한 통계량은 네 번째 적률에 기대므로 중심극한정리의 보호를 받지 못한다. 원래 BP 가 정규성 아래에서만 $\chi^2$인 것도 같은 이유이고, 코엔커의 눈금 바꾸기가 그 의존을 끊어 준 것이다. 그러므로 **`het_breuschpagan` 의 기본값 `robust=True` 를 건드리지 말라.**

    **$n = 10$에서는 어느 판본도 쓸 수 없다.** 마지막 표가 그 까닭이다. 오차 표준편차가 $x = 1$에서 $1.5$, $x = 10$에서 $6$으로 **네 배** 벌어지는 뚜렷한 이분산인데도 $n = 10$에서 코엔커의 검정력이 $0.1656$이다. 여섯 번에 한 번만 잡아낸다. 같은 이분산을 $n = 100$에서는 $0.9913$으로 거의 늘 잡는다.

    그러므로 본문 자료의 $p = 0.777$은 **"등분산이다"는 증거가 아니다.** 등분산이어도 $p$가 크고, 네 배짜리 이분산이 있어도 다섯 번 중 네 번은 $p$가 크다. 두 경우를 $n = 10$으로는 구별할 수 없다. 게다가 $n = 10$에서 코엔커의 실제 크기가 정규오차에서조차 $0.058$, 로그정규에서 $0.110$이라 **검정 자체도 아직 눈금이 맞지 않는다.** 이 자료에 대해 할 수 있는 정직한 말은 "판정할 수 없다" 하나뿐이다.

**해석:**

- $p$값이 0.05보다 작으면 귀무가설을 기각하고 이분산이 존재한다고 결론짓는다.
- $p$값이 0.05보다 크면 귀무가설을 기각하지 못하고 잔차가 등분산과 일관된다고 본다.

여기서는 $p = 0.777$로 기각하지 못한다. 다만 $n = 10$으로 매우 작아 검정력이 사실상 없으므로, **"등분산성이 확인되었다"가 아니라 "이 자료로는 판정할 수 없다"**가 옳은 결론이다.

### 이분산의 해결책

이분산이 탐지되면 다음 해결책이 가능하다.

**1. 종속변수의 변환:** 로그나 제곱근 변환이 분산을 안정시킬 수 있다.

$$
Y' = \log(Y) \quad \text{또는} \quad Y' = \sqrt{Y}
$$

**2. 가중최소제곱(WLS):** 추정된 분산에 기반해 관측값에 가중치를 부여하여, 분산이 작은 관측값에 더 큰 비중을 준다.

$$
\hat{\beta} = (X^\top W X)^{-1} X^\top W Y
$$

**3. 로버스트 표준오차:** 회귀계수를 바꾸지 않으면서 이분산을 반영하는 로버스트 표준오차를 쓴다. 셋 중 가장 간단하고 가장 널리 쓰인다.

---

## 2. 분산분석에서의 분산 검정

분산분석은 셋 이상 집단의 평균을 비교하여 유의한 차이가 있는지 판정한다. 핵심 가정은 집단들의 분산이 같다는 것(**분산의 동질성**)이다. 이 가정이 위배되면 분산분석 결과가 오도할 수 있다.

### 분산분석에서의 분산 동질성

**가설:**

- $H_0$: 집단들의 분산이 같다: $\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$
- $H_1$: 적어도 한 집단의 분산이 다르다: 적어도 한 쌍의 $i \neq j$에 대해 $\sigma_i^2 \neq \sigma_j^2$

### 분산분석에서의 Levene 검정

Levene 검정은 분산분석의 등분산 가정을 확인하는 데 자주 쓰인다. 분산이 같지 않으면(분산의 이질성) Welch 분산분석 같은 대안을 써야 한다.

**Python 구현:**

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 평균이 달라도 분산이 같으면. 세 집단의 평균은 $14, 26, 36$으로 크게 다르지만 표본분산은 똑같다.

**(1)** Levene 통계량 $W$가 **집단별 위치이동에 불변**임을 보이시오. 곧 각 집단에 제멋대로의 상수 $a_i$를 더해도 $W$가 변하지 않는다. 어느 중심을 쓰든 성립하는가. 또 `scipy.stats.levene`을 `center` 없이 부르면 실제로 돌아가는 검정은 무엇인가.

**(2)** 불변성을 수치로 확인하고, 이 보기 자료가 불변성보다 **더 강한** 뜻에서 퇴화되어 있음을 밝히시오. 그리고 집단별로 중심을 빼는 대신 **전체 중앙값 하나로** 빼면 어떤 일이 벌어지는지 보이시오.

</div>

??? success "풀이"

    **(1) 중심 함수가 위치이동에 공변하면 그만이다.** 집단 $i$의 자료를 $y_{i1}, \ldots, y_{in_i}$, 중심 함수를 $c(\cdot)$이라 하면 Levene 검정은 변환값

    $$
    z_{ij} = \lvert y_{ij} - c(y_i) \rvert
    $$

    에 보통의 일원분산분석을 돌려 $W$를 얻는다. **$W$는 $z$만의 함수다.** 그러므로 $z$가 변하지 않으면 $W$도 변하지 않는다.

    이제 집단 $i$에 상수 $a_i$를 더한다. 중심 함수가 **위치이동에 공변**, 곧

    $$
    c(y_i + a_i \mathbf{1}) = c(y_i) + a_i
    $$

    를 만족하면

    $$
    \lvert (y_{ij} + a_i) - c(y_i + a_i\mathbf{1}) \rvert
    = \lvert y_{ij} + a_i - c(y_i) - a_i \rvert
    = \lvert y_{ij} - c(y_i) \rvert
    = z_{ij}
    $$

    로 $z$가 **그대로**다. 따라서 $W$도 그대로다. $\square$

    세 중심 모두 공변한다.

    - **평균**은 선형이므로 $\overline{y_i + a_i} = \bar y_i + a_i$.
    - **중앙값**은 순서통계량이 통째로 $a_i$만큼 밀리므로 중앙값도 $a_i$만큼 밀린다.
    - **절사평균**도 순서가 보존되어 잘라낼 꼬리가 바뀌지 않으므로 $a_i$만큼 밀린다.

    **그러므로 Levene 검정은 집단평균의 차이를 아예 보지 않는다.** 중심을 빼는 단계가 위치 정보를 전부 걷어 내기 때문이고, 이것이 등분산 검정이 평균 검정과 섞이지 않는 이유다.

    **`center` 를 주지 않으면 Brown-Forsythe 가 돌아간다.** `scipy.stats.levene` 의 `center` 기본값은 `'median'` 이다. 그러므로 위 코드의 `levene(group1, group2, group3)` 은 Levene(1960)의 원래 검정이 아니라 **중앙값 중심의 Brown-Forsythe 변형**이다. 두 선택은 치우친 자료에서 꽤 다른 답을 주므로(15.5절), 어느 것을 돌렸는지 늘 밝혀 두어야 한다. 이 보기 자료는 대칭이라 둘이 일치한다.

    **(2) 확인한다.**

    ```python
    import numpy as np
    from scipy.stats import levene

    # 세 집단의 평균은 크게 다르지만 분산은 똑같다. 등분산 검정이 평균 차이에
    # 휘둘리지 않는다는 것을 보이는 자료다.
    group1 = [10, 12, 14, 16, 18]
    group2 = [22, 24, 26, 28, 30]
    group3 = [32, 34, 36, 38, 40]

    print("variances:", [np.var(g, ddof=1) for g in (group1, group2, group3)])

    # Levene 은 각 값에서 제 집단의 중심을 뺀 절대편차에 분산분석을 돌린다.
    # 중심을 빼는 단계가 평균 차이를 걷어 내므로, 평균이 아무리 달라도
    # 분산만 같으면 기각되지 않는다.
    test_stat, p_value = levene(group1, group2, group3)
    print(f"Levene's test statistic: {test_stat}")
    print(f"P-value: {p_value}")
    ```

    출력:

    ```text
    variances: [10.0, 10.0, 10.0]
    Levene's test statistic: 0.0
    P-value: 1.0
    ```

    이제 (1)의 세 주장을 차례로 확인한다.

    ```python
    import inspect
    import numpy as np
    from scipy.stats import levene, f_oneway

    # (0) 기본값 확인: 아무 인자도 주지 않은 위의 levene(...) 은 무엇이었나.
    print("levene 의 center 기본값 =", inspect.signature(levene).parameters["center"].default)

    groups = [np.array([10, 12, 14, 16, 18], float),
              np.array([22, 24, 26, 28, 30], float),
              np.array([32, 34, 36, 38, 40], float)]

    # (1) 집단별 중심을 뺀 절대편차 Z. 세 집단이 완전히 같은 다중집합이다.
    for g in groups:
        print(f"  {g.astype(int)}  ->  Z = {np.abs(g - np.median(g)).astype(int)}")

    for c in ("mean", "median", "trimmed"):
        kw = {"proportiontocut": 0.1} if c == "trimmed" else {}
        r = levene(*groups, center=c, **kw)
        print(f"  center={c:>8}:  W = {r.statistic:.4f},  p = {r.pvalue:.4f}")

    # 평균은 얼마나 다른가. 원자료에 분산분석을 돌려 본다.
    r0 = f_oneway(*groups)
    print(f"\n집단평균 = {[g.mean() for g in groups]}")
    print(f"원자료에 분산분석(평균 비교):  F = {r0.statistic:.4f},  p = {r0.pvalue:.3e}")

    # (2) 위치이동 불변성. 퇴화되지 않은 자료로 보여야 "0 = 0" 이 아님이 분명해진다.
    nd = [np.array([10, 12, 14, 16, 18], float),    # 공차 2, 표본분산 10.0
          np.array([22, 25, 28, 31, 34], float),    # 공차 3, 표본분산 22.5
          np.array([32, 37, 42, 47, 52], float)]    # 공차 5, 표본분산 62.5
    base = levene(*nd, center="median")
    print(f"\n이동 전:  W = {base.statistic:.10f},  p = {base.pvalue:.10f}")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(1000):
        shifted = [g + s for g, s in zip(nd, rng.normal(0, 1e4, 3))]
        worst = max(worst, abs(levene(*shifted, center="median").statistic - base.statistic))
    print(f"임의 이동 1000 번 뒤 통계량의 최대 변화 = {worst:.3e}")

    # (3) 집단별로 중심을 빼지 않으면(전체 중앙값 하나로 빼면) 어떻게 되는가.
    allv = np.concatenate(groups)
    Zg = [np.abs(g - np.median(allv)) for g in groups]
    print(f"\n전체 중앙값 = {np.median(allv):.1f}")
    for z in Zg:
        print(f"  Z = {z.astype(int)}")
    r = f_oneway(*Zg)
    print(f"그 Z 에 분산분석:  F = {r.statistic:.4f},  p = {r.pvalue:.6f}")
    ```

    출력:

    ```text
    levene 의 center 기본값 = median
      [10 12 14 16 18]  ->  Z = [4 2 0 2 4]
      [22 24 26 28 30]  ->  Z = [4 2 0 2 4]
      [32 34 36 38 40]  ->  Z = [4 2 0 2 4]
      center=    mean:  W = 0.0000,  p = 1.0000
      center=  median:  W = 0.0000,  p = 1.0000
      center= trimmed:  W = 0.0000,  p = 1.0000

    집단평균 = [14.0, 26.0, 36.0]
    원자료에 분산분석(평균 비교):  F = 60.6667,  p = 5.314e-07

    이동 전:  W = 1.8947368421,  p = 0.1926999286
    임의 이동 1000 번 뒤 통계량의 최대 변화 = 1.517e-13

    전체 중앙값 = 26.0
      Z = [16 14 12 10  8]
      Z = [4 2 0 2 4]
      Z = [ 6  8 10 12 14]
    그 Z 에 분산분석:  F = 16.8772,  p = 0.000325
    ```

    **기본값이 `median` 이다.** 그러므로 위에서 돌아간 것은 Brown-Forsythe 검정이다. 다만 이 자료는 세 중심이 모두 같은 답($W = 0$)을 주므로 구별되지 않는다.

    **불변성이 수치로 확인된다.** 퇴화되지 않은 자료(공차 $2, 3, 5$, 표본분산 $10.0,\ 22.5,\ 62.5$)에서 $W = 1.8947368421$이고, 집단마다 표준편차 $10^4$의 정규난수를 더해 **1000번** 흔들어도 통계량의 최대 변화가 $1.5\times10^{-13}$이다. 이것은 $10^4$을 더했다 뺀 데서 생긴 부동소수점 반올림이며, 수학적으로는 정확히 $0$이어야 하는 양이다. **집단평균을 제멋대로 옮겨도 $W$가 꿈쩍하지 않는다.**

    그 대비가 선명하다. 같은 자료의 집단평균이 $14,\ 26,\ 36$으로 크게 다르고, 원자료에 분산분석을 돌리면 $F = 60.67$, $p = 5.3\times10^{-7}$으로 평균 차이를 압도적으로 기각한다. **평균에 대해서는 이렇게 강한 증거가 있는데 분산에 대해서는 $W = 0$, $p = 1$이다.** 두 질문이 완전히 분리되어 있다.

    **이 자료는 불변성보다 강한 뜻에서 퇴화되어 있다.** 불변성은 "이동해도 $W$가 변하지 않는다"는 말일 뿐이고, 이동하지 않은 자료에서 $W$가 $0$이어야 할 이유는 없다. 위의 퇴화되지 않은 자료가 $W = 1.89$를 주는 것이 그 증거다. 여기서 $W = 0$이 나오는 까닭은 세 집단의 $Z$가 모두 $\{4,2,0,2,4\}$로 **같은 다중집합**이라 집단간 제곱합이 정확히 $0$이기 때문이다. 아래 상자와 연습문제 2 가 그 까닭을 등차수열의 분산 공식으로 적어 둔다.

    **집단별로 빼는 것이 핵심이다.** 마지막 묶음이 그것을 보인다. 집단 중심 대신 **전체 중앙값 $26$ 하나**로 빼면 $Z$가 $\{16,14,12,10,8\}$, $\{4,2,0,2,4\}$, $\{6,8,10,12,14\}$로 집단마다 전혀 달라지고, 그 $Z$에 분산분석을 돌리면 $F = 16.88$, $p = 0.000325$로 **"분산이 다르다"고 기각한다.** 분산은 세 집단이 정확히 같은데도 그렇다. 새어 든 것은 분산 차이가 아니라 **평균 차이**다. (1)의 증명에서 $c(y_i)$가 집단마다 따로 계산되어야 $a_i$가 약분된 것이고, 상수 하나를 쓰면 $\lvert y_{ij} + a_i - c_0 \rvert$에서 $a_i$가 남는다. **Levene 검정에서 집단별 중심화는 선택이 아니라 검정의 정의 그 자체다.**

!!! note "이 보기 자료는 퇴화되어 있다"
    세 집단이 모두 등차수열 $\{a, a+2, a+4, a+6, a+8\}$의 형태이므로 **표본분산이 정확히 10으로 동일**하다. 중앙값으로부터의 절대편차도 세 집단 모두 $\{4, 2, 0, 2, 4\}$로 같다.

    그래서 Levene 통계량이 **정확히 0**, $p$값이 **정확히 1**이 된다. 검정을 시연하는 자료로는 적절하지 않다. 분산 차이가 있는 자료를 쓰려면 예컨대 `group3 = [26, 31, 36, 41, 46]`처럼 간격을 바꾸면 된다.

**해석:**

- $p$값이 0.05보다 작으면 귀무가설을 기각하고 분산이 같지 않다고 결론짓는다.
- $p$값이 0.05보다 크면 귀무가설을 기각하지 못한다. 이는 "분산이 같다"의 증명이 아니라 "다르다는 증거가 없다"는 뜻이다.

### 분산 이질성의 해결책

Levene 검정이 이분산을 시사하면 집단 간 등분산을 가정하지 않는 **Welch 분산분석**을 적용할 수 있다.

!!! danger "`anova_lm(..., robust='hc3')`은 Welch 분산분석이 아니다"
    다음 코드가 Welch 분산분석으로 소개되는 경우가 있으나 **틀렸다**.

    ```python
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # 집단마다 분산이 다른 자료 (표준편차 1, 2, 4)
    rng = np.random.default_rng(0)
    data = pd.DataFrame({
        "Score": np.concatenate([rng.normal(10, 1, 20),
                                 rng.normal(12, 2, 20),
                                 rng.normal(14, 4, 20)]),
        "Group": np.repeat(["A", "B", "C"], 20),
    })

    # Welch 분산분석이 아니다 — 등분산을 가정한 보통의 분산분석이다
    model = sm.formula.ols('Score ~ Group', data=data).fit()
    print(sm.stats.anova_lm(model, typ=2, robust='hc3'))
    ```

    출력:

    ```
                  sum_sq    df          F        PR(>F)
    Group     424.427668   2.0  30.736759  8.794817e-10
    Residual  393.541447  57.0        NaN           NaN
    ```

    표가 나오기는 하지만 이것은 Welch 분산분석이 아니다. 분모 자유도가 $57 = N - k$ 그대로이고, Welch라면 등분산이 깨진 만큼 자유도가 줄어들어야 한다.

    이것은 OLS 적합에 이분산 일치 공분산 행렬(HC3)을 적용한 **Wald 형태의 분산분석표**이다. Welch 분산분석과 다음 점에서 다르다.

    | | Welch 분산분석 | `anova_lm(robust='hc3')` |
    |---|---|---|
    | 집단평균의 가중 | 정밀도 $n_i/s_i^2$로 가중 | 가중하지 않음(OLS) |
    | 분모 자유도 | Welch-Satterthwaite 근사 | $N - k$ 그대로 |
    | 소표본 성질 | 잘 연구되어 있음 | 근사가 거칠 수 있음 |

    Welch 분산분석은 SciPy에 없으므로 직접 구현하거나(15.7절 [분산분석 사전검정](anova_pretest.md) 페이지 참조) `pingouin.welch_anova`를 쓴다.

**Welch 분산분석의 Python 구현:**

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> Welch 분산분석. 집단 셋, 집단당 $n_i = 3$인 자료 $\{10,12,14\}$, $\{22,24,26\}$, $\{32,34,36\}$에 Welch 와 고전 분산분석을 모두 돌린다.

**(1)** 세 표본분산을 구하고, Welch 의 $A$, $\Lambda$, $F_W$, 분모 자유도 $\nu_2$를 **분수로 정확히** 계산하시오. 표본분산이 모두 같을 때 $A$가 고전적 $F$와 일치하는 까닭도 밝히시오.

**(2)** $d_1 = 2$인 $F$ 분포의 꼬리확률이 닫힌 꼴임을 보이고, 그것으로 두 $p$값을 **분수로** 적으시오. 두 $p$값이 갈리는 몫을 **통계량이 작아진 탓**과 **자유도가 줄어든 탓**으로 나누면 어느 쪽이 주범인가.

</div>

??? success "풀이"

    **(1) 손으로 끝까지 간다.** 세 집단이 모두 $\{a, a+2, a+4\}$ 꼴이므로 평균이 $a+2$, 편차가 $-2, 0, 2$, 제곱합이 $8$이고

    $$
    s_i^2 = \frac{8}{3-1} = 4 \qquad (i = 1,2,3)
    $$

    으로 **세 표본분산이 정확히 같다.** 집단평균은 $12, 24, 34$다.

    가중치는 $w_i = n_i/s_i^2 = 3/4$로 모두 같고 $\sum_j w_j = 9/4$이므로, 가중평균이 단순평균으로 떨어진다.

    $$
    \bar X_w = \frac{12 + 24 + 34}{3} = \frac{70}{3}
    $$

    이제 $A$를 계산한다. $12 - \frac{70}{3} = -\frac{34}{3}$, $24 - \frac{70}{3} = \frac{2}{3}$, $34 - \frac{70}{3} = \frac{32}{3}$이므로

    $$
    A = \frac{1}{k-1}\sum_i w_i(\bar X_i - \bar X_w)^2
    = \frac{1}{2}\cdot\frac{3}{4}\cdot\frac{34^2 + 2^2 + 32^2}{9}
    = \frac{3}{8}\cdot\frac{2184}{9}
    = 91
    $$

    이다. $\Lambda$는 $w_i/\sum_j w_j = 1/3$이 세 번 같으므로

    $$
    \Lambda = \frac{1}{k^2-1}\sum_i \frac{1}{n_i-1}\left(1 - \frac{w_i}{\sum_j w_j}\right)^2
    = \frac{1}{8}\cdot 3 \cdot \frac{1}{2}\left(\frac23\right)^2
    = \frac{1}{8}\cdot\frac{2}{3} = \frac{1}{12}
    $$

    이고, 따라서

    $$
    F_W = \frac{A}{1 + 2(k-2)\Lambda} = \frac{91}{1 + \frac16} = \frac{91 \cdot 6}{7} = 78,
    \qquad
    \nu_2 = \frac{1}{3\Lambda} = 4
    $$

    이다. **네 수가 모두 정수 또는 깔끔한 분수로 떨어진다.** 균형자료에서 표본분산이 모두 같을 때의 일반형

    $$
    \Lambda = \frac{k-1}{k(k+1)(n-1)}
    $$

    에 $k = 3$, $n = 3$을 넣어도 $\frac{2}{3 \cdot 4 \cdot 2} = \frac{1}{12}$로 같다(15.7절 [분산분석 사전검정](anova_pretest.md) 연습문제 3).

    **$A$가 고전적 $F$와 같아지는 까닭.** 모든 $s_i^2 = s^2$, 모든 $n_i = n$이면 $w_i = n/s^2$으로 같아져 $\bar X_w$가 전체평균 $\bar X$가 되고

    $$
    A = \frac{1}{k-1}\cdot\frac{n}{s^2}\sum_i (\bar X_i - \bar X)^2 = \frac{\text{MSB}}{s^2}
    $$

    이다. 한편 균형자료의 합동분산은 집단분산의 단순평균이므로 $\text{MSE} = s^2$이다. 그러므로

    $$
    A = \frac{\text{MSB}}{\text{MSE}} = F_{\text{고전}}
    $$

    이고, 실제로 $A = 91$이 고전적 $F = 91$과 같다. **그러면 두 통계량의 차이는 보정항 하나로 줄어든다.**

    $$
    \frac{F_W}{F_{\text{고전}}} = \frac{1}{1 + 2(k-2)\Lambda} = \frac{6}{7}
    $$

    $91 \times \frac67 = 78$이다.

    **(2) $d_1 = 2$이면 꼬리확률이 닫힌 꼴이다.** $F = \dfrac{U/2}{V/\nu}$에서 $U \sim \chi^2_2$, $V \sim \chi^2_\nu$가 독립이라 하자. $\chi^2_2$는 평균 $2$인 지수분포이므로 $U/2 \sim \text{Exp}(1)$이고, $E \sim \text{Exp}(1)$에 대해 $P(E > t) = e^{-t}$다. $V$로 조건을 걸면

    $$
    P(F > x) = P\!\left(\frac U2 > \frac{xV}{\nu}\right) = E\!\left[e^{-xV/\nu}\right] = M_V\!\left(-\frac x\nu\right)
    $$

    이고, $\chi^2_\nu$의 적률생성함수가 $M_V(t) = (1-2t)^{-\nu/2}$이므로

    $$
    P(F > x) = \left(1 + \frac{2x}{\nu}\right)^{-\nu/2}
    $$

    이다. 이제 두 $p$값을 분수로 적을 수 있다.

    $$
    p_W = \left(1 + \frac{2 \cdot 78}{4}\right)^{-2} = 40^{-2} = \frac{1}{1600} = 0.000625
    $$

    $$
    p_{\text{고전}} = \left(1 + \frac{2 \cdot 91}{6}\right)^{-3} = \left(\frac{94}{3}\right)^{-3} = \frac{27}{830584} = 3.250725\times10^{-5}
    $$

    **두 $p$값의 비가 $19.2$배다.** 그 몫을 쪼개려면 중간 두 경우를 더 재면 된다. 같은 닫힌 꼴로

    $$
    p(F=78,\ \nu=6) = 27^{-3} = \frac{1}{19683}, \qquad
    p(F=91,\ \nu=4) = \left(\frac{93}{2}\right)^{-2} = \frac{4}{8649}
    $$

    이다. 고전값을 $1$로 보면 통계량만 $91 \to 78$로 바꾼 쪽이 $1.563$배, 자유도만 $6 \to 4$로 줄인 쪽이 $14.227$배다. **주범은 자유도다.** 보정항 $6/7$이 통계량을 깎는 몫은 작고, $\nu_2$가 $6$에서 $4$로 내려앉는 몫이 압도적이다.

    **확인한다.**

    ```python
    import numpy as np
    from scipy import stats

    def welch_anova(groups):
        """Welch 의 일원배치 분산분석. 등분산을 가정하지 않는다."""
        k = len(groups)
        n = np.array([len(g) for g in groups])
        m = np.array([np.mean(g) for g in groups])
        v = np.array([np.var(g, ddof=1) for g in groups])
        w = n / v
        m_w = np.sum(w * m) / np.sum(w)
        A = np.sum(w * (m - m_w) ** 2) / (k - 1)
        lam = np.sum((1 - w / np.sum(w)) ** 2 / (n - 1)) / (k ** 2 - 1)
        F = A / (1 + 2 * (k - 2) * lam)
        df2 = 1 / (3 * lam)
        return F, k - 1, df2, stats.f.sf(F, k - 1, df2)

    groups = [[10, 12, 14], [22, 24, 26], [32, 34, 36]]
    F, df1, df2, p = welch_anova(groups)
    print(f"Welch ANOVA: F = {F:.4f}, df = ({df1}, {df2:.2f}), p = {p:.6f}")

    # 고전적 분산분석과 견준다. 분산이 크게 다르면 두 p-값이 갈린다.
    print(f"Classical ANOVA: {stats.f_oneway(*groups)}")
    ```

    출력:

    ```
    Welch ANOVA: F = 78.0000, df = (2, 4.00), p = 0.000625
    Classical ANOVA: F_onewayResult(statistic=91.0, pvalue=3.2507247912312294e-05)
    ```

    함수가 $F = 78.0000$, $\nu_2 = 4.00$, $p = 0.000625$를 주고 (1)·(2)의 손계산과 그대로 맞는다. 분수까지 맞춰 보려면 `fractions` 로 같은 길을 따라가면 된다.

    ```python
    from fractions import Fraction as Fr

    import numpy as np
    from scipy import stats

    # --- (1) 분수로 정확히 따라가 본다. 위의 welch_anova 를 그대로 쓴다. ---
    g = [[10, 12, 14], [22, 24, 26], [32, 34, 36]]
    k, n = len(g), len(g[0])
    m = [Fr(sum(x), len(x)) for x in g]
    v = [Fr(sum((Fr(t) - mi) ** 2 for t in x), len(x) - 1) for x, mi in zip(g, m)]
    print(f"집단평균   = {[str(x) for x in m]}")
    print(f"표본분산   = {[str(x) for x in v]}")

    w = [Fr(n, 1) / vi for vi in v]
    sw = sum(w)
    mw = sum(wi * mi for wi, mi in zip(w, m)) / sw
    A = sum(wi * (mi - mw) ** 2 for wi, mi in zip(w, m)) / (k - 1)
    lam = sum((1 - wi / sw) ** 2 / Fr(n - 1) for wi in w) / (k ** 2 - 1)
    F_W = A / (1 + 2 * (k - 2) * lam)
    print(f"w_i = {[str(x) for x in w]},  sum w = {sw},  가중평균 = {mw}")
    print(f"A = {A},   Lambda = {lam},   F_W = {F_W},   nu2 = {1 / (3 * lam)}")
    print(f"균형·등분산일 때의 Lambda 공식 (k-1)/(k(k+1)(n-1)) = {Fr(k - 1, k * (k + 1) * (n - 1))}")
    print(f"F_W / F_classical = 1/(1+2(k-2)Lambda) = {1 / (1 + 2 * (k - 2) * lam)}")

    # --- (2) d1 = 2 이면 꼬리확률이 닫힌 꼴이다:  sf(x) = (1 + 2x/nu)^(-nu/2) ---
    def f2_sf(x, nu):
        return Fr(1) / (1 + Fr(2) * Fr(x) / Fr(nu)) ** Fr(nu, 2)

    print()
    for FF, d2, name in [(91, 6, "고전     "), (78, 6, "통계량만 "),
                         (91, 4, "자유도만 "), (78, 4, "Welch    ")]:
        exact = f2_sf(FF, d2)
        print(f"{name} F={FF}, df=(2,{d2}):  p = {str(exact):>10} = {float(exact):.6e}"
              f"   scipy {stats.f.sf(FF, 2, d2):.6e}")

    base = float(f2_sf(91, 6))
    print(f"\n고전 대비 배율:  통계량만 {float(f2_sf(78, 6)) / base:.3f} 배,"
          f"  자유도만 {float(f2_sf(91, 4)) / base:.3f} 배,"
          f"  둘 다 {float(f2_sf(78, 4)) / base:.3f} 배")
    ```

    출력:

    ```text
    집단평균   = ['12', '24', '34']
    표본분산   = ['4', '4', '4']
    w_i = ['3/4', '3/4', '3/4'],  sum w = 9/4,  가중평균 = 70/3
    A = 91,   Lambda = 1/12,   F_W = 78,   nu2 = 4
    균형·등분산일 때의 Lambda 공식 (k-1)/(k(k+1)(n-1)) = 1/12
    F_W / F_classical = 1/(1+2(k-2)Lambda) = 6/7

    고전      F=91, df=(2,6):  p =  27/830584 = 3.250725e-05   scipy 3.250725e-05
    통계량만  F=78, df=(2,6):  p =    1/19683 = 5.080526e-05   scipy 5.080526e-05
    자유도만  F=91, df=(2,4):  p =     4/8649 = 4.624812e-04   scipy 4.624812e-04
    Welch     F=78, df=(2,4):  p =     1/1600 = 6.250000e-04   scipy 6.250000e-04

    고전 대비 배율:  통계량만 1.563 배,  자유도만 14.227 배,  둘 다 19.226 배
    ```

    **손계산이 한 자리도 어긋나지 않는다.** $A = 91$, $\Lambda = 1/12$, $F_W = 78$, $\nu_2 = 4$가 모두 분수로 정확히 떨어지고, 균형·등분산일 때의 일반 공식 $(k-1)/(k(k+1)(n-1))$도 $1/12$을 준다. 보정비는 정확히 $6/7$이다.

    **네 $p$값 모두 닫힌 꼴이다.** $d_1 = 2$ 덕분에 $27/830584$, $1/19683$, $4/8649$, $1/1600$으로 적히고 네 값이 `scipy` 와 여섯 자리까지 같다. $\nu/2$가 정수가 되는 $\nu = 4, 6$이어서 지수가 정수 거듭제곱으로 떨어진 것이다.

    **배율의 분해가 (2)의 답을 준다.** 고전 $p$를 $1$로 보면 통계량만 $91 \to 78$로 바꾼 쪽이 $1.563$배, 자유도만 $6 \to 4$로 줄인 쪽이 $14.227$배, 둘 다 하면 $19.226$배다. **자유도가 주범이고 통계량의 몫은 작다.** 두 효과가 곱으로 분해되지는 않지만($1.563 \times 14.227 = 22.2 \neq 19.2$) 어느 쪽이 크냐는 분명하다.

    이것이 "Welch 가 등분산 아래에서 잃는 것"의 정체다. **통계량을 $6/7$로 깎는 것은 사소하고, 분모 자유도를 $N - k = 6$에서 $\nu_2 = 4$로 내려놓는 것이 값을 치른다.** $p$값이 $3.3\times10^{-5}$에서 $6.3\times10^{-4}$로 올라가 **19 배**가 되었다. 다만 두 값 모두 어떤 관례적 수준에서도 기각하므로 **이 자료의 결론은 바뀌지 않는다.**

    **$n_i = 3$이기 때문에 대가가 이렇게 크다.** $\nu_2 = 1/(3\Lambda)$에 균형·등분산 공식을 넣으면

    $$
    \nu_2 = \frac{k(k+1)(n-1)}{3(k-1)}
    $$

    이고 $n$에 비례하므로, 집단당 관측값을 늘리면 $\nu_2$가 $N - k = k(n-1)$에 비해 덜 뒤처진다. $k = 3$이면 $\nu_2/(N-k) = (k+1)/(3(k-1)) = 2/3$으로 비율 자체는 $n$과 무관하지만, $F$ 분포는 분모 자유도가 $30$을 넘으면 거의 변하지 않으므로 **절대적인 대가가 사라진다.** 15.7절 [분산분석 사전검정](anova_pretest.md) 연습문제 2 가 재어 둔 검정력 손실 $2$–$3$퍼센트포인트가 그 크기다.

    거꾸로 말하면 **$n_i = 3$에서는 $\nu_2 = 4$라 Welch-Satterthwaite 근사 자체를 믿기 어렵다.** 이 보기는 공식을 손으로 따라가 보기에 좋은 자료일 뿐, 실제 추론에 쓸 자료가 아니다.

**해석:**

Welch 분산분석은 집단 간 등분산을 가정하지 않고 집단평균을 비교하는 F 통계량과 $p$값을 제공한다. $p$값이 0.05보다 작으면 집단평균에 유의한 차이가 있다고 결론짓는다.

---

## 3. 흐름: 회귀와 분산분석에서의 분산 검정

세 처치집단의 평균을 분산분석으로 비교하고 자료에 회귀분석을 수행하는 상황을 생각하자.

1. **분산분석:** 처치집단에 Levene 검정(또는 Brown-Forsythe 검정)을 수행한다. 분산이 같으면 표준 분산분석으로, 아니면 Welch 분산분석으로 진행한다.
2. **회귀:** 회귀모형을 적합한 뒤 Breusch-Pagan 검정으로 이분산을 확인한다. 이분산이 있으면 로버스트 표준오차를 적용하거나 종속변수를 변환한다.

이 단계들이 모형의 가정을 충족시켜 더 정확하고 신뢰할 만한 통계적 추론으로 이어진다.

!!! tip "더 단순한 대안"
    위 흐름은 두 단계 절차의 문제(15.7절 [분산분석 사전검정](anova_pretest.md) 참조)를 안고 있다. 실무에서는 다음이 더 간단하고 안전하다.

    1. **분산분석:** 사전검정 없이 처음부터 Welch 분산분석을 쓴다.
    2. **회귀:** 사전검정 없이 처음부터 로버스트 표준오차(HC3)를 쓴다.

    두 경우 모두 가정이 성립할 때 잃는 것이 적고(검정력 2~3%p), 가정이 깨졌을 때 얻는 것이 크다. 검정 결과에 따라 방법을 바꾸는 자료 의존적 절차 자체를 없애는 것이 핵심이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
"이분산이 있으면 회귀계수가 편향된다"는 서술이 왜 틀렸는지 설명하고, 실제로 무엇이 편향되는지 밝혀라.

</div>

??? success "풀이"
    **OLS 추정량의 불편성.** $\hat{\boldsymbol{\beta}} = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\mathbf{Y}$이고 $\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$이므로

    $$
    \hat{\boldsymbol{\beta}} = \boldsymbol{\beta} + (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\varepsilon}.
    $$

    $E[\boldsymbol{\varepsilon} \mid \mathbf{X}] = \mathbf{0}$이면

    $$
    E[\hat{\boldsymbol{\beta}} \mid \mathbf{X}] = \boldsymbol{\beta} + (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'E[\boldsymbol{\varepsilon}\mid\mathbf{X}] = \boldsymbol{\beta}.
    $$

    이 유도 어디에도 $\operatorname{Var}(\varepsilon_i)$이 등장하지 않는다. **불편성은 오차의 분산 구조와 무관하다.**

    **편향되는 것은 분산 추정량이다.** 참 공분산행렬은

    $$
    \operatorname{Var}(\hat{\boldsymbol{\beta}}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\boldsymbol{\Omega}\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}, \quad \boldsymbol{\Omega} = \operatorname{diag}(\sigma_1^2, \ldots, \sigma_n^2)
    $$

    인데 OLS는 $\boldsymbol{\Omega} = \sigma^2 \mathbf{I}$를 가정하여

    $$
    \widehat{\operatorname{Var}}_{\text{OLS}}(\hat{\boldsymbol{\beta}}) = s^2(\mathbf{X}'\mathbf{X})^{-1}
    $$

    을 쓴다. $\boldsymbol{\Omega} \neq \sigma^2\mathbf{I}$이면 이 추정값이 참값과 다르며, 15.7절 [회귀에서의 분산 검정](regression_variance.md) 연습문제 2에서 보았듯 **어느 방향으로든** 틀릴 수 있다.

    **추가로 잃는 것: 효율성.** OLS는 여전히 불편이지만 최소분산은 아니다. 참 $\boldsymbol{\Omega}$를 알면 GLS가 더 작은 분산을 갖는다. 다만 실무에서 $\boldsymbol{\Omega}$를 모르는 경우가 많으므로, 효율성 손실을 감수하고 OLS + 로버스트 표준오차를 쓰는 것이 표준 관행이다.

    **왜 이 구분이 중요한가.** 계수가 편향된다고 믿으면 결과를 폐기하거나 모형을 다시 설정해야 한다고 생각하게 된다. 실제로는 **표준오차 계산 방식만 바꾸면 된다.** `cov_type='HC3'` 한 줄이면 해결된다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
본문의 Levene 보기 자료가 왜 퇴화되어 있는지 보이고, 분산 차이가 있는 자료로 바꾸어 검정을 다시 수행하라.

</div>

??? success "풀이"
    **퇴화의 이유.** 세 집단이 모두 공차 2인 등차수열이다.

    $$
    g_1 = \{10,12,14,16,18\}, \quad g_2 = \{22,24,26,28,30\}, \quad g_3 = \{32,34,36,38,40\}.
    $$

    등차수열의 분산은 공차와 항의 개수에만 의존하고 시작값과 무관하다. 공차 $d$, 항 $n = 5$이면

    $$
    s^2 = \frac{d^2 n(n+1)}{12} = \frac{4 \times 5 \times 6}{12} = 10.
    $$

    세 집단 모두 $s^2 = 10$이다.

    더 나아가 중앙값으로부터의 절대편차도 세 집단 모두 $\{4, 2, 0, 2, 4\}$로 **완전히 동일**하다. Levene 검정은 이 편차들의 집단평균을 비교하므로, 집단간 제곱합이 정확히 0이 되어 $W = 0$, $p = 1$이 나온다.

    **분산 차이를 만든 자료.**

    ```python
    import numpy as np
    from scipy.stats import levene, bartlett, fligner

    g1 = [10, 12, 14, 16, 18]      # d = 2, var = 10
    g2 = [22, 25, 28, 31, 34]      # d = 3, var = 22.5
    g3 = [32, 37, 42, 47, 52]      # d = 5, var = 62.5

    print("variances:", [round(np.var(g, ddof=1), 2) for g in (g1, g2, g3)])
    # 이름있는 튜플을 그대로 찍으면 유효숫자가 너무 많다. 자리수를 맞춰 출력한다.
    for name, res in [("Levene (median)", levene(g1, g2, g3)),
                      ("Levene (mean)  ", levene(g1, g2, g3, center='mean')),
                      ("Bartlett       ", bartlett(g1, g2, g3)),
                      ("Fligner-Killeen", fligner(g1, g2, g3))]:
        print(f"{name}: stat = {res.statistic:.4f}, p = {res.pvalue:.4f}")
    ```

    출력:

    ```text
    variances: [10.0, 22.5, 62.5]
    Levene (median): stat = 1.8947, p = 0.1927
    Levene (mean)  : stat = 1.8947, p = 0.1927
    Bartlett       : stat = 2.9323, p = 0.2308
    Fligner-Killeen: stat = 3.5932, p = 0.1659
    ```

    이제 검정통계량이 0이 아니고 $p$값도 1이 아니다. 다만 **어느 검정도 기각하지 못한다**($p$값이 0.17~0.23).

    분산비가 $62.5/10 = 6.25$배로 상당한데도 그렇다. 각 집단 $n = 5$, 총 15개 관측값으로는 검정력이 매우 낮기 때문이다. 15.4절 [유도와 카이제곱 근사](../bartlett_test/derivation.md) 연습문제 2에서 총 30개로도 4배 차이를 겨우 탐지했음을 떠올리면 당연한 결과이다.

    (평균 중심과 중앙값 중심 Levene의 결과가 완전히 같다. 다섯 개 등차수열에서는 평균과 중앙값이 일치하기 때문이다.)

    **교육적 함의.** 보기 자료를 만들 때는 (1) 보이려는 현상이 실제로 존재하는지, (2) 그것을 탐지할 만한 표본크기인지 확인해야 한다. 등차수열처럼 규칙적인 자료는 의도치 않은 퇴화를 낳기 쉽다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`sm.stats.anova_lm(model, typ=2, robust='hc3')`이 Welch 분산분석과 어떻게 다른지 구체적으로 설명하고, 두 결과를 비교하라.

</div>

??? success "풀이"
    **본질적 차이.** 두 절차 모두 이분산에 대응하지만 방식이 다르다.

    - **`anova_lm(robust='hc3')`:** OLS로 적합한 뒤, 계수의 공분산행렬만 HC3 샌드위치 추정량으로 바꾸어 Wald 검정을 수행한다. 집단평균의 **추정 방식은 그대로**이고, 분모 자유도도 $N - k$를 그대로 쓴다.
    - **Welch 분산분석:** 집단평균을 정밀도 $w_i = n_i/s_i^2$로 **가중**하여 결합하고, Welch-Satterthwaite 근사로 분모 자유도를 줄인다.

    ```python
    import numpy as np, pandas as pd
    import statsmodels.api as sm
    from scipy import stats

    data = pd.DataFrame({'Group': list('AAABBBCCC'),
                         'Score': [10, 12, 14, 22, 24, 26, 32, 34, 36]})
    m = sm.formula.ols('Score ~ Group', data=data).fit()
    print(sm.stats.anova_lm(m, typ=2, robust='hc3'))

    groups = [[10, 12, 14], [22, 24, 26], [32, 34, 36]]
    print(stats.f_oneway(*groups))
    ```

    출력:

    ```text
                  sum_sq   df          F    PR(>F)
    Group     485.333333  2.0  60.666667  0.000105
    Residual   24.000000  6.0        NaN       NaN
    F_onewayResult(statistic=91.0, pvalue=3.2507247912312294e-05)
    ```

    고전 분산분석의 $F = 91.0$이고 HC3 판은 $F = 60.67$이다. HC3 쪽이 더 보수적이다.

    **이 자료에서는 세 집단의 분산이 정확히 같다**($\{10,12,14\}$의 분산 = $\{22,24,26\}$의 분산 = $\{32,34,36\}$의 분산 = 4). 그러므로 Welch 분산분석은 고전 분산분석과 사실상 같은 결과를 낼 것이고, HC3 판만 다르게 나온다.

    **왜 HC3가 더 보수적인가.** HC3는 잔차를 $(1-h_{ii})$로 나누어 지렛대 보정을 한다. $n = 9$, 모수 3개로 자유도가 매우 적으므로 $h_{ii}$가 커서($1/3$) 보정이 크게 작용한다. 작은 표본에서 HC3는 알려진 대로 보수적이다.

    **결론.** 세 절차가 모두 다른 답을 준다. 이름이 "robust"라고 해서 Welch와 같은 것이 아니며, 무엇을 쓰는지 명확히 하고 그 소표본 성질을 알아야 한다. $n = 9$처럼 극단적으로 작은 자료에서는 어느 것도 신뢰하기 어렵고, 순열검정이 더 나은 선택이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
본문의 두 단계 흐름 대신 "처음부터 Welch + HC3"를 쓰는 대안이 제시되었다. 각 방식의 장단점을 정리하고 언제 어느 쪽을 택할지 논하라.

</div>

??? success "풀이"

    | | 두 단계 (사전검정 후 선택) | 처음부터 로버스트 |
    |---|---|---|
    | 제1종 오류 조절 | 왜곡 가능(15.7절 표: 0.057) | 안정적(0.044) |
    | 검정력 (가정 성립 시) | 최대 | 2~3%p 손실 |
    | 자료 의존적 선택 | 있음 | 없음 |
    | 사전등록 가능성 | 어려움 | 쉬움 |
    | 보고의 투명성 | 낮음(선택 과정 서술 필요) | 높음 |
    | 계산 복잡도 | 검정 두 번 | 한 번 |

    **처음부터 로버스트를 택할 상황(대부분).**

    - 탐색적 분석이 아닌 확증적 분석
    - 사전등록된 연구
    - 가정의 성립 여부가 불확실한 관찰자료
    - 표본이 작아 사전검정의 검정력이 낮은 경우

    **두 단계를 택할 만한 상황(드묾).**

    - 표본이 매우 커서 사전검정의 검정력이 충분한 경우
    - 등분산성 자체가 과학적 관심사인 경우(이때는 사전검정이 아니라 **보고할 결과**이다)
    - 물리적·이론적 근거로 등분산을 강하게 기대할 수 있고 최대 검정력이 필요한 경우

    **핵심 원칙.** 통계적 절차의 선택은 **자료를 보기 전에** 정해야 한다. 자료를 보고 방법을 고르면, 아무리 각 방법이 개별적으로 타당해도 결합된 절차의 오류율이 통제되지 않는다.

    "처음부터 로버스트"의 진짜 장점은 검정력이나 크기 수치가 아니라 **이 선택 문제를 아예 없앤다는 것**이다. 자료를 보고 결정할 일이 없으면 자료 의존적 편향도 없다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
보기 2 의 마지막 묶음에서 집단별 중앙값 대신 **전체 중앙값 하나**로 빼자 $F = 16.8772$, $p = 0.000325$ 로 "분산이 다르다"고 기각했다. 세 집단의 표본분산은 정확히 같았다. 그 누설이 얼마나 큰지 재어 보자.

$g_1 = \{10,12,14,16,18\}$, $g_2 = \{22,24,26,28,30\}$ 은 그대로 두고 셋째 집단만 $g_3 = \{32+t,\,34+t,\,36+t,\,38+t,\,40+t\}$ 로 $t \ge -6$ 만큼 옮긴다. 세 표본분산은 $t$ 와 무관하게 모두 $10$ 이다.

**(1)** $t \ge -6$ 이면 15 개 값의 전체 중앙값이 $t$ 와 무관하게 $26$ 임을 보이고, $Z_{ij} = \lvert y_{ij} - 26 \rvert$ 에 일원분산분석을 돌린 통계량이

$$
F(t) = \frac{25t^2 + 140t + 1924}{114}
$$

임을 유도하시오. $t = 0$ 에서 본문의 $16.8772$ 가 나오는지 확인하시오.

**(2)** 보기 3 (2)의 닫힌 꼴로 $p$값을 적고, 기각을 면하게 해 주는 $t$ 가 있는지 판정하시오. $t \to \infty$ 에서 $F$ 가 어떤 속도로 커지는지도 밝히고, 집단별 중심화를 뺀 "레빈 검정"이 실제로 무엇을 검정하고 있는지 한 줄로 쓰시오.

</div>

??? success "풀이"

    **(1) 중앙값이 $t$ 에 걸리지 않는다.** 값이 15 개이므로 전체 중앙값은 크기순 8 번째다. $t \ge -6$ 이면 셋째 집단의 최솟값이 $32 + t \ge 26$ 이므로 $26$ 보다 작은 값은 $g_1$ 의 다섯 개와 $g_2$ 의 $22, 24$ 뿐이다. 곧 7 개가 $26$ 아래에 있고 8 번째가 $g_2$ 의 $26$ 이다. 그러므로 전체 중앙값은 $t$ 와 무관하게 $c_0 = 26$ 이다.

    이제 $Z_{ij} = \lvert y_{ij} - 26 \rvert$ 를 적는다.

    $$
    Z_1 = \{16,14,12,10,8\}, \qquad Z_2 = \{4,2,0,2,4\}, \qquad Z_3 = \{6+t,\,8+t,\,10+t,\,12+t,\,14+t\}
    $$

    집단평균은 $\bar Z_1 = 12$, $\bar Z_2 = \frac{12}{5}$, $\bar Z_3 = 10 + t$ 이고, 집단내 제곱합은 $t$ 에 걸리지 않는다. $Z_1$ 과 $Z_3$ 은 공차 $2$ 의 등차수열이라 각각 $4 \times 10 = 40$, $Z_2$ 는

    $$
    (4-2.4)^2 + (2-2.4)^2 + (0-2.4)^2 + (2-2.4)^2 + (4-2.4)^2 = 2.56+0.16+5.76+0.16+2.56 = 11.2
    $$

    이다. 그러므로 $\text{SSW} = 40 + 11.2 + 40 = 91.2$, 자유도 $15-3 = 12$, $\text{MSW} = 7.6$ 으로 **$t$ 와 무관한 상수**다.

    집단간 제곱합만 $t$ 에 걸린다. $\bar Z_{\cdot} = \frac{1}{3}\left(12 + \frac{12}{5} + 10 + t\right) = \frac{122+5t}{15}$ 이므로

    $$
    \sum_i (\bar Z_i - \bar Z_\cdot)^2 = \sum_i \bar Z_i^2 - 3\bar Z_\cdot^2 = \frac{6244}{25} + 20t + t^2 - \frac{(122+5t)^2}{75} = \frac{3848}{75} + \frac{56}{15}t + \frac{2}{3}t^2
    $$

    이고 $\text{MSB} = \frac{5}{2}\sum_i(\bar Z_i - \bar Z_\cdot)^2$ 이다. 따라서

    $$
    F(t) = \frac{\text{MSB}}{7.6} = \frac{25}{76}\left(\frac{3848}{75} + \frac{56}{15}t + \frac{2}{3}t^2\right) = \frac{962}{57} + \frac{70}{57}t + \frac{25}{114}t^2 = \frac{25t^2+140t+1924}{114}
    $$

    이다. $t = 0$ 에서 $\frac{1924}{114} = 16.877193$ 으로 본문의 $F = 16.8772$ 와 맞는다. $\square$

    **(2) 닫힌 꼴로 $p$값을 적는다.** 분자 자유도가 $d_1 = 2$ 이고 분모 자유도가 $\nu = 12$ 이므로 보기 3 (2)의 공식 $P(F > x) = (1+2x/\nu)^{-\nu/2}$ 에서

    $$
    p(t) = \left(1 + \frac{F(t)}{6}\right)^{-6}
    $$

    이다. $t = 0$ 이면 $F = \frac{962}{57}$, $1 + \frac F6 = \frac{652}{171}$ 이라

    $$
    p(0) = \left(\frac{171}{652}\right)^{6} = 3.2546\times10^{-4}
    $$

    으로 본문의 $0.000325$ 와 맞는다.

    **기각을 면하게 해 주는 $t$ 는 없다.** $p = 0.05$ 가 되는 임계값은 $\left(1+\frac F6\right)^{-6} = 0.05$, 곧

    $$
    F_{0.05} = 6\left(20^{1/6} - 1\right) = 6(1.64752 - 1) = 3.8851
    $$

    이다. 한편 $F(t)$ 는 위로 열린 포물선이고 꼭짓점이 $t = -\frac{140}{50} = -2.8$ 에 있으므로 최솟값이

    $$
    F(-2.8) = \frac{25(7.84) - 392 + 1924}{114} = \frac{1728}{114} = 15.1579
    $$

    이다. $15.16 > 3.89$ 이므로 허용되는 모든 $t \ge -6$ 에서 기각한다. 실제로 $t = -6$ 이면 $F = \frac{1984}{114} = 17.4035$, $t = 0$ 이면 $16.8772$, $t = 5$ 이면 $28.5$, $t = 20$ 이면 $129.1579$ 다.

    **$t\to\infty$ 에서 $F$ 는 이차로 커진다.** $F(t) \sim \frac{25}{114}t^2 = 0.2193\,t^2$ 이므로 $p \to 0$ 이고, 집단평균을 멀리 떼어 놓을수록 "분산이 다르다"는 증거가 끝없이 세진다. 세 표본분산은 그동안 내내 $10$ 으로 똑같다.

    **무엇을 검정하고 있었는가.** 전체 중심 $c_0$ 하나로 빼면 $\bar Z_i \approx \lvert \mu_i - c_0\rvert$ 이므로, 그 분산분석은 분산이 아니라 **집단평균이 전체 중심에서 얼마나 떨어져 있는지**를 비교한다 — 평균 검정의 변종이다. 보기 2 (1)의 증명에서 $a_i$ 가 약분된 것은 $c(y_i)$ 가 집단마다 따로 계산되었기 때문이고, 상수 하나를 쓰면 $\lvert y_{ij} + a_i - c_0\rvert$ 에 $a_i$ 가 그대로 남는다. 그 남은 $a_i$ 가 $t^2$ 으로 자라는 것이 위 식이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
1 절에서 $\sigma(x) = 0.2 + 0.3x^2$ 일 때 명목 95% 구간의 포함률이 $0.889$ 로 **모자랐다**. 그런데 연습문제 1 은 OLS 표준오차가 "어느 방향으로든" 틀릴 수 있다고 적었다. 그 방향을 정하는 것이 무엇인지 정확히 재어 보자.

설계점을 $x = (-2,-1,0,1,2)$ 로 고정하고 $\operatorname{Var}(\varepsilon_i) = \sigma_i^2$, $E[\varepsilon_i] = 0$, 서로 독립이라 하자. 이때 $\bar x = 0$, $S_{xx} = 10$, $n = 5$ 다.

**(1)** 기울기의 참 분산과 OLS 가 쓰는 추정량의 기댓값이

$$
\operatorname{Var}(\hat\beta_1) = \frac{\sum_i x_i^2\sigma_i^2}{S_{xx}^2}, \qquad E\!\left[\frac{s^2}{S_{xx}}\right] = \frac{\sum_i (1-h_{ii})\sigma_i^2}{(n-2)S_{xx}}
$$

임을 보이고 $h_{ii}$ 다섯 개를 구하시오.

**(2)** $\sigma^2 = (3,1,1,1,3)$ 과 $\sigma^2 = (1,1,7,1,1)$ 에 대해 두 수를 계산하고 표준오차가 몇 배 틀리는지 적으시오. 두 경우의 방향이 반대인 까닭을 두 가중벡터로 설명하고, 이 설계에서 OLS 표준오차가 **정확히 맞는** 조건을 $\sigma_i^2$ 의 부등식 하나로 적으시오.

</div>

??? success "풀이"

    **(1) 두 식을 유도한다.** $\bar x = 0$ 이므로 $\hat\beta_1 = \sum_i x_i y_i / S_{xx}$ 이고

    $$
    \hat\beta_1 - \beta_1 = \sum_i \frac{x_i}{S_{xx}}\varepsilon_i \;\Longrightarrow\; \operatorname{Var}(\hat\beta_1) = \frac{\sum_i x_i^2\sigma_i^2}{S_{xx}^2}
    $$

    이다. 참 분산은 $\sigma_i^2$ 을 **$x_i^2$ 으로 가중**한다.

    한편 잔차는 $\mathbf{e} = (\mathbf{I}-\mathbf{H})\boldsymbol\varepsilon$ 이고 $\mathbf{I}-\mathbf{H}$ 가 멱등이므로

    $$
    E\!\left[\sum_i e_i^2\right] = E[\boldsymbol\varepsilon'(\mathbf{I}-\mathbf{H})\boldsymbol\varepsilon] = \operatorname{tr}\!\left((\mathbf{I}-\mathbf{H})\boldsymbol\Omega\right) = \sum_i (1-h_{ii})\sigma_i^2
    $$

    이고, $s^2 = \sum e_i^2/(n-2)$ 를 $S_{xx}$ 로 나눈 것이 OLS 의 분산추정량이다. 추정량은 $\sigma_i^2$ 을 **$1-h_{ii}$ 로 가중**한다. $\square$

    단순회귀의 지렛대는 $h_{ii} = \frac1n + \frac{x_i^2}{S_{xx}} = 0.2 + \frac{x_i^2}{10}$ 이므로

    $$
    (h_{11},\ldots,h_{55}) = (0.6,\; 0.3,\; 0.2,\; 0.3,\; 0.6), \qquad \sum_i h_{ii} = 2 = k
    $$

    이다.

    **(2) 두 경우를 계산한다.**

    $\sigma^2 = (3,1,1,1,3)$ — 분산이 **양 끝에** 몰린 경우.

    $$
    \operatorname{Var}(\hat\beta_1) = \frac{4(3)+1+0+1+4(3)}{100} = \frac{26}{100} = 0.26
    $$

    $$
    E[s^2] = \frac{0.4(3)+0.7+0.8+0.7+0.4(3)}{3} = \frac{4.6}{3} = 1.5333, \qquad E\!\left[\frac{s^2}{S_{xx}}\right] = 0.15333
    $$

    비는 $\frac{0.26}{0.15333} = \frac{39}{23} = 1.6957$ 이고 표준오차로는 $\sqrt{1.6957} = 1.3022$ 배다. **OLS 표준오차가 참값의 $1/1.3022 = 0.768$ 배, 곧 23% 작다.** 구간이 너무 좁아 포함률이 명목값에 모자란다.

    $\sigma^2 = (1,1,7,1,1)$ — 분산이 **가운데에** 몰린 경우.

    $$
    \operatorname{Var}(\hat\beta_1) = \frac{4+1+0+1+4}{100} = 0.10
    $$

    $$
    E[s^2] = \frac{0.4+0.7+7(0.8)+0.7+0.4}{3} = \frac{7.8}{3} = 2.6, \qquad E\!\left[\frac{s^2}{S_{xx}}\right] = 0.26
    $$

    비는 $\frac{0.10}{0.26} = \frac{5}{13} = 0.3846$ 이고 표준오차로는 $\sqrt{0.3846} = 0.6202$ 배다. **이번에는 OLS 표준오차가 참값의 $1/0.6202 = 1.61$ 배, 곧 61% 크다.** 구간이 너무 넓어 포함률이 명목값을 넘는다. 분산의 들쭉날쭉함은 두 번째가 더 심한데(분산비가 $7$ 대 $1$ 로 첫 경우의 $3$ 대 $1$ 보다 크다) 틀리는 방향은 반대다.

    **까닭은 두 가중벡터가 다르다는 것뿐이다.** 참 분산의 가중치는 $x_i^2/S_{xx} = (0.4,\,0.1,\,0,\,0.1,\,0.4)$ 이고 추정량의 가중치는 $(1-h_{ii})/(n-2) = (0.4,\,0.7,\,0.8,\,0.7,\,0.4)/3$ 이다. 가운데 점을 보라. **$x = 0$ 의 분산은 기울기의 산포에 아무 기여도 하지 않는데($0$), $s^2$ 에서는 가장 큰 몫($0.8/3 = 0.267$)을 차지한다.** 그러므로 분산이 양 끝에 몰리면 $s^2$ 이 참 산포를 과소평가하고, 가운데에 몰리면 과대평가한다.

    **정확히 맞는 조건.** 두 가중벡터의 차는

    $$
    \frac{x_i^2}{S_{xx}} - \frac{1-h_{ii}}{n-2} = \frac{1}{3}(0.8,\,-0.4,\,-0.8,\,-0.4,\,0.8) \;\propto\; (2,\,-1,\,-2,\,-1,\,2)
    $$

    이므로, 이 설계에서 OLS 표준오차가 편향되지 않을 필요충분조건은

    $$
    2\sigma_1^2 - \sigma_2^2 - 2\sigma_3^2 - \sigma_4^2 + 2\sigma_5^2 = 0
    $$

    이고, 좌변이 양수면 과소추정, 음수면 과대추정이다. 등분산 $\sigma_i^2 \equiv \sigma^2$ 이면 $2-1-2-1+2 = 0$ 으로 성립한다(이분산이어도 이 식만 맞으면 OLS 표준오차는 불편이다). 첫 경우는 $6-1-2-1+6 = 8 > 0$ 이라 과소, 둘째는 $2-1-14-1+2 = -12 < 0$ 이라 과대다.

    **1 절의 자료는 첫 경우다.** $\sigma(x) = 0.2+0.3x^2$ 은 $\lvert x\rvert$ 가 커질수록 커지므로 분산이 양 끝에 몰리고, 좌변이 양수가 되어 OLS 표준오차가 작게 나온다. 그래서 포함률이 $0.889$ 로 모자랐다. 실제 자료의 이분산은 대개 이 모양($\lvert x\rvert$ 와 함께 커지는 쪽)이라 "OLS 는 $p$값을 작게 낸다"는 경험칙이 생긴 것인데, **그것은 정리가 아니라 흔한 경우일 뿐**이다. 가운데가 더 퍼진 자료에서는 OLS 가 오히려 보수적이다. 어느 쪽이든 HC3 는 $\sigma_i^2$ 을 $x_i^2$ 가중으로 그대로 집어넣으므로 두 경우 모두 고쳐 준다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
보기 1 (2)의 표에서 `robust=False` 판본(원래 BP)은 $t(3)$ 과 로그정규에서 표본을 키울수록 **더** 나빠졌다. 왜 그런지, 그리고 극한값이 얼마인지 적률로 계산하자. 설명변수가 하나인 단순회귀, $H_0$ 아래 $\operatorname{Var}(\varepsilon_i) = \sigma^2$, 오차의 표준화 4 차 적률을 $\beta_2 = E[\varepsilon^4]/\sigma^4$ 라 한다.

**(1)** $g_i = e_i^2/\hat\sigma^2$ 의 보조회귀에서 $\text{ESS}_g = \left(\sum_i (x_i-\bar x)g_i\right)^2/S_{xx}$ 임을 보이고, $\operatorname{Var}(\varepsilon_i^2/\sigma^2) = \beta_2 - 1$ 을 써서

$$
\text{BP} = \tfrac12\text{ESS}_g \;\xrightarrow{\;d\;}\; \frac{\beta_2-1}{2}\,\chi^2_1
$$

임을 보이시오. 코엔커 판본 $nR^2$ 에서는 이 상수가 왜 사라지는가.

**(2)** 명목 $\alpha = 0.05$ 에서 원래 BP 의 극한 크기가 $P\!\left(\chi^2_1 > 2\chi^2_{1,0.95}/(\beta_2-1)\right)$ 임을 쓰고, 정규($\beta_2 = 3$)·균일($\beta_2 = 1.8$)·지수($\beta_2 = 9$)·로그정규($\beta_2 = 113.94$)에서 값을 구하시오. 또 극한 크기가 명목값의 두 배가 되는 $\beta_2$ 를 구하고, 2 장의 하한 $\beta_2 \ge 1 + \gamma_1^2$ 과 묶어 "비대칭도가 얼마를 넘으면 반드시 그만큼 나빠진다"는 꼴의 결론을 적으시오. 끝으로 보기 1 표의 로그정규 칸이 $n = 200$ 에서 $0.5475$ 에 그친 것을 (2)의 예측과 견주어 설명하시오.

</div>

??? success "풀이"

    **(1) 보조회귀의 설명제곱합을 적는다.** 보조회귀 $g_i = \gamma_0 + \gamma_1 x_i + u_i$ 의 OLS 기울기는 $\hat\gamma_1 = \sum_i (x_i-\bar x)g_i/S_{xx}$ 이고, 설명제곱합은 $\text{ESS}_g = \hat\gamma_1^2 S_{xx}$ 이므로

    $$
    \text{ESS}_g = \frac{\left(\sum_i (x_i-\bar x)g_i\right)^2}{S_{xx}}
    $$

    이다.

    이제 극한을 본다. $\hat\sigma^2 = \frac1n\sum_i e_i^2 \to \sigma^2$ 이고 $e_i = \varepsilon_i + O_p(n^{-1/2})$ 이므로 $g_i = \varepsilon_i^2/\sigma^2 + o_p(1)$ 이다. $u_i = \varepsilon_i^2/\sigma^2$ 이라 두면 $E[u_i] = 1$ 이고

    $$
    \operatorname{Var}(u_i) = \frac{E[\varepsilon^4]}{\sigma^4} - 1 = \beta_2 - 1
    $$

    이다. $\sum_i (x_i - \bar x) = 0$ 이라 평균 $1$ 은 떨어지므로, $\max_i (x_i-\bar x)^2/S_{xx} \to 0$ 인 설계에서 린데베르그 중심극한정리가

    $$
    \frac{\sum_i (x_i-\bar x)u_i}{\sqrt{S_{xx}}} \;\xrightarrow{\;d\;}\; N(0,\;\beta_2-1)
    $$

    을 준다. 제곱하면 $\text{ESS}_g \to (\beta_2-1)\chi^2_1$ 이고 절반이 BP 다. $\square$

    정규오차라면 $\beta_2 = 3$ 이라 상수가 $\frac{3-1}{2} = 1$ 이고 $\text{BP}\to\chi^2_1$ 이다. **교과서의 참조분포는 정규성에서만 맞는다.** 보기 1 (1)이 "이 판본의 약점은 $\chi^2_k$ 극한이 정규성에 기댄다는 것"이라 한 말의 정확한 형태가 이 상수다.

    **코엔커는 그 상수를 추정해 약분한다.** $nR^2 = n\,\text{ESS}_g/\text{TSS}_g$ 이고 $\text{TSS}_g = \sum_i (g_i-\bar g)^2$ 은 큰 수의 법칙으로 $n\operatorname{Var}(u) = n(\beta_2-1)$ 에 가까워지므로

    $$
    nR^2 = \frac{\text{ESS}_g}{\text{TSS}_g/n} \;\xrightarrow{\;d\;}\; \frac{(\beta_2-1)\chi^2_1}{\beta_2-1} = \chi^2_1
    $$

    이다. **4 차 적률을 가정하지 않고 표본에서 재어 나눈 것이 차이의 전부**이고, 그래서 보기 1 의 표에서 코엔커 열만 세 오차분포 모두 $0.05$ 로 모인다.

    **(2) 극한 크기를 수로 적는다.** $\text{BP} \to \frac{\beta_2-1}{2}\chi^2_1$ 이고 임계값은 $c = \chi^2_{1,0.95} = 3.8415$ 이므로

    $$
    P(\text{BP} > c) \to P\!\left(\chi^2_1 > \frac{2c}{\beta_2-1}\right)
    $$

    이다. 네 경우를 넣는다.

    | 오차분포 | $\beta_2$ | 문턱 $2c/(\beta_2-1)$ | 극한 크기 |
    |---|---|---|---|
    | 균일 | $1.8$ | $9.6037$ | $0.0019$ |
    | 정규 | $3$ | $3.8415$ | $0.0500$ |
    | 지수 | $9$ | $0.9604$ | $0.3271$ |
    | 로그정규 | $113.94$ | $0.0680$ | $0.7942$ |

    **첨도가 $3$ 보다 작으면 지나치게 보수적이고, 크면 거짓 양성이 폭발한다.** 균일오차에서는 $0.05$ 가 $0.0019$ 로 내려앉아(26 분의 1) 이분산을 찾아내지 못하고, 지수오차에서는 세 번에 한 번 거짓 경보가 난다.

    **두 배가 되는 첨도.** $P(\chi^2_1 > q) = 0.10$ 인 $q$ 는 $\chi^2_{1,0.90} = 2.7055$ 이므로

    $$
    \frac{2c}{\beta_2-1} = 2.7055 \;\Longrightarrow\; \beta_2 = 1 + \frac{2(3.8415)}{2.7055} = 3.8397
    $$

    이다. **첨도가 $3.84$ 만 되어도 1 종 오류율이 두 배다.** 정규분포에서 $0.84$ 만큼 벗어난 자리이니 실무에서 흔한 수준이다.

    **2 장의 하한을 끼운다.** $\beta_2 \ge 1 + \gamma_1^2$ 이므로 $\gamma_1^2 \ge 2.8397$, 곧

    $$
    \lvert \gamma_1 \rvert \ge 1.6851 \;\Longrightarrow\; \beta_2 \ge 3.8397 \;\Longrightarrow\; \text{극한 크기} \ge 0.10
    $$

    이다. **비대칭도의 절댓값이 $1.6851$ 을 넘는 오차분포라면 첨도를 따져 볼 필요도 없이 원래 BP 의 1 종 오류율은 적어도 두 배다.** 지수분포는 $\gamma_1 = 2$ 로 여기에 걸리고 실제로 $0.327$ 이다. 거꾸로 보수적($\beta_2 < 3$)이 되려면 $\lvert\gamma_1\rvert < \sqrt2 = 1.4142$ 가 필요조건이다.

    **표의 로그정규 칸은 아직 극한에 닿지 않았다.** 예측은 $0.7942$ 인데 $n = 200$ 에서 $0.5475$ 다. 틀린 것이 아니라 **느린 것**이다. (1)의 중심극한정리는 $\varepsilon$ 이 아니라 $u = \varepsilon^2/\sigma^2$ 에 적용되므로 수렴 속도는 $u$ 의 비대칭도, 곧 $\varepsilon$ 의 6 차·8 차 적률에 달려 있다. 로그정규 $e^{Z}$ 는 $E[e^{8Z}] = e^{32} \approx 8\times10^{13}$ 으로 터무니없이 무거워 $n = 200$ 은 극한 근처가 아니다. 표의 $0.126 \to 0.406 \to 0.548$ 이 **느리게 오르는 중**인 것이 그 증거다. $t(3)$ 은 $\beta_2 = \infty$ 이므로 극한 크기가 $1$ 이고, 역시 $0.064 \to 0.245 \to 0.378$ 로 오르는 중이다.

    **이것이 "표본을 키워도 낫지 않는다"의 정확한 뜻이다.** $n$ 이 커지면서 통계량이 $\chi^2_1$ 로 가는 것이 아니라 $\frac{\beta_2-1}{2}$ 배 부풀려진 분포로 간다. 분모를 표본에서 재는 코엔커 판본만이 그 배율을 지운다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
보기 2 (1)이 보인 대로 Levene 검정은 $z_{ij} = \lvert y_{ij} - c(y_i)\rvert$ 에 일원분산분석을 돌린 것이다. 집단 크기를 극단으로 보내 보자. 모든 집단이 $n_i = 2$ 라 하자.

**(1)** 중심이 평균이든 중앙값이든 절사평균이든 $z_{i1} = z_{i2} = \lvert y_{i1}-y_{i2}\rvert/2$ 임을 보이고, 그 결과 $W$ 의 분모가 **항등적으로 $0$** 임을 보이시오. $\{10,12\}$, $\{20,30\}$, $\{5,25\}$ 에 `scipy.stats.levene` 을 돌리면 어떤 값이 나오는가.

**(2)** 그런데 Bartlett 검정과 $F$ 검정은 $n_i = 2$ 에서도 돌아간다. $(y_{i1}-y_{i2})^2/2$ 이 $\sigma_i^2$ 의 불편추정량이고 자유도가 $1$ 임을 보인 뒤 같은 세 집단에서 Bartlett 통계량과 $p$값을 손으로 구하시오. 두 사실을 묶어 로버스트 검정이 **무엇을 대가로 가정을 덜 쓰는지** 한 줄로 쓰시오.

</div>

??? success "풀이"

    **(1) 두 점이면 모든 중심이 중점이다.** $n_i = 2$ 이면 평균도 중앙값도 $(y_{i1}+y_{i2})/2$ 이고, 양쪽에서 같은 비율을 잘라내는 절사평균도 (잘라낼 점이 없거나 둘 다 잘리므로) 같은 중점이다. 따라서 $d_i = \lvert y_{i1}-y_{i2}\rvert$ 라 쓰면

    $$
    z_{i1} = \left\lvert y_{i1} - \frac{y_{i1}+y_{i2}}{2}\right\rvert = \frac{d_i}{2} = z_{i2}
    $$

    이다. **집단 안에서 $z$ 가 상수**이므로 집단내 제곱합이 집단마다 $0$ 이고, 합쳐도 $0$ 이다. 곧

    $$
    W = \frac{\sum_i n_i(\bar z_i - \bar z_\cdot)^2/(k-1)}{\sum_i\sum_j (z_{ij}-\bar z_i)^2/(N-k)} = \frac{\text{분자}}{0}
    $$

    에서 분모가 자료와 무관하게 $0$ 이다. 분자는 $\bar z_i = d_i/2$ 가 집단마다 다르기만 하면 양수다.

    주어진 자료에서 $d = (2,\,10,\,20)$ 이므로 $\bar z = (1,\,5,\,10)$, $\bar z_\cdot = \frac{32}{6} = \frac{16}{3}$ 이고

    $$
    \text{SSB} = 2\left[\left(1-\tfrac{16}{3}\right)^2 + \left(5-\tfrac{16}{3}\right)^2 + \left(10-\tfrac{16}{3}\right)^2\right] = 2\cdot\frac{169+1+196}{9} = \frac{732}{9}
    $$

    로 $\text{MSB} = \frac{366}{9} = 40.667 > 0$ 이다. 그러므로 `scipy.stats.levene` 은 `RuntimeWarning: divide by zero` 를 내면서 `statistic=inf, pvalue=0.0` 을 돌려준다. **분산이 조금만 달라 보이면 $p = 0$ 으로 무조건 기각한다.** ($d_i$ 가 모두 같으면 분자까지 $0$ 이 되어 `nan` 이다.) 어느 쪽도 검정이 아니다. Levene 계열이 쓸 수 있는 최소 집단 크기는 $n_i = 3$ 이고, 그때도 집단마다 $z$ 가 세 개뿐이라 분모가 거의 비어 있다.

    **(2) 고전적 검정은 두 점으로도 산다.** $y_{i1}, y_{i2} \sim N(\mu_i, \sigma_i^2)$ 이 독립이면 $y_{i1}-y_{i2}\sim N(0, 2\sigma_i^2)$ 이므로

    $$
    E\!\left[\frac{(y_{i1}-y_{i2})^2}{2}\right] = \frac{2\sigma_i^2}{2} = \sigma_i^2, \qquad \frac{(y_{i1}-y_{i2})^2}{2\sigma_i^2}\sim\chi^2_1
    $$

    이다. 자유도는 $n_i - 1 = 1$ 이므로 제곱합이 곧 표본분산이고, 그 값이 위의 양과 같다.

    $$
    s_i^2 = \frac{1}{n_i-1}\sum_j (y_{ij}-\bar y_i)^2 = 2\left(\frac{d_i}{2}\right)^2 = \frac{d_i^2}{2}
    $$

    자료에서 $s^2 = (2,\,50,\,200)$ 이다. Bartlett 은 $\nu_i = 1$, $\nu = 3$, 합동분산 $s_p^2 = \frac{2+50+200}{3} = 84$ 로

    $$
    B = \frac{\nu\ln s_p^2 - \sum_i \nu_i \ln s_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_i \frac{1}{\nu_i} - \frac1\nu\right)} = \frac{3\ln 84 - (\ln 2 + \ln 50 + \ln 200)}{1 + \frac16\left(3 - \frac13\right)}
    $$

    이다. 분자는 $13.29245 - 9.90349 = 3.38896$, 분모는 $1 + \frac16\cdot\frac83 = \frac{13}{9}$ 이므로

    $$
    B = \frac{9}{13}(3.38896) = 2.34621
    $$

    이고, $\chi^2_2$ 의 꼬리는 $P(\chi^2_2 > b) = e^{-b/2}$ 이므로

    $$
    p = e^{-1.17310} = 0.30941
    $$

    이다. 기각하지 못한다 — 자유도가 집단마다 $1$ 뿐이니 분산비가 $100$ 배인데도 증거가 모자란다. 그래도 **수가 나오고 그 수가 뜻을 가진다**는 점이 Levene 과 다르다.

    **대가는 자유도다.** Levene 계열은 모집단 적률을 가정하는 대신 **집단 안의 산포를 한 번 더 표본으로 쓴다.** $z_{ij}$ 의 집단내 변동이 분모가 되므로, 그 변동을 잴 자유도가 집단마다 따로 필요하다. $n_i = 2$ 에서는 그 자유도가 $0$ 이라 검정이 아예 정의되지 않고, Bartlett 은 $\chi^2_1$ 이라는 **가정된 분포**로 그 자리를 메워 살아남는다. 15 장이 되풀이한 맞바꿈이 집단 크기의 극단에서는 이렇게 극단적으로 드러난다. **분산을 재는 일에서 "가정을 덜 쓰기"는 곧 "자료를 더 쓰기"다.**

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
3 절의 흐름은 "BP 로 이분산을 확인하고, 있으면 로버스트 표준오차, 없으면 OLS" 다. 15.7 절이 분산분석에서 이 두 단계 절차를 비판했듯 회귀에서도 같은 일이 벌어지는지 재어 보자.

$x$ 를 $[0,2]$ 의 $n = 50$ 등간격 점, $y = 2 + x + \sigma(x)Z$, $Z\sim N(0,1)$ 로 두고 기울기의 명목 95% 구간이 참값 $1$ 을 덮는 비율을 세 규칙 — (가) 늘 OLS, (나) 늘 HC3, (다) BP($\alpha=0.05$)를 통과하면 OLS·기각하면 HC3 — 로 잰다. $\sigma(x) = 0.8$, $0.3+0.5x$, $0.2+0.3x^2$ 세 자료에서 10 만 번씩 돌린다.

**(1)** 세 규칙의 포함률을 표로 내고, 무조건 포함률만 보면 **사전검정 규칙이 왜 무해해 보이는지** 밝히시오. 어느 자료에서 사전검정 규칙이 명목값에 가장 가까운가.

**(2)** 이번에는 사전검정의 판정으로 표본을 나누어 포함률을 다시 재시오. 사전검정이 "등분산이다" 하고 통과시킨 표본에서 OLS 구간의 포함률은 얼마인가. 같은 표본에서 HC3 로 바꾸면 고쳐지는가. 고쳐지지 않는다면 기울기의 참 산포와 두 표준오차의 평균을 재어 까닭을 밝히고, 세 규칙 가운데 무엇을 쓸지 결론하시오.

</div>

??? success "풀이"

    **재어 본다.** 단순회귀이므로 $\hat\beta_1 = \mathbf{w}'\mathbf{y}$, $w_i = (x_i-\bar x)/S_{xx}$ 로 모든 것을 행렬 한 번에 벡터화할 수 있다.

    ```python
    import numpy as np
    from scipy import stats


    def simulate(n, sdfun, R=100_000, seed=7):
        """x in [0,2] 에서 y = 2 + x + sd(x)Z 를 R 벌 만들고
        기울기의 95% 구간과 BP 사전검정을 한꺼번에 계산한다."""
        x = np.linspace(0, 2, n)
        xc = x - x.mean()
        Sxx = (xc ** 2).sum()
        h = 1 / n + xc ** 2 / Sxx                  # 단순회귀의 지렛대
        w = xc / Sxx                               # b1 = w'y
        rng = np.random.default_rng(seed)
        Y = 2 + x + rng.normal(size=(R, n)) * sdfun(x)
        b1 = Y @ w
        b0 = Y.mean(axis=1) - b1 * x.mean()
        E = Y - b0[:, None] - b1[:, None] * x
        se_ols = np.sqrt((E ** 2).sum(axis=1) / (n - 2) / Sxx)
        se_hc3 = np.sqrt((w ** 2 * E ** 2 / (1 - h) ** 2).sum(axis=1))

        # 코엔커 판 Breusch-Pagan: e^2 을 x 에 회귀시킨 nR^2
        G = E ** 2
        Gc = G - G.mean(axis=1, keepdims=True)
        lm = n * (Gc @ xc) ** 2 / (Sxx * (Gc ** 2).sum(axis=1))
        keep = stats.chi2.sf(lm, 1) >= 0.05        # 통과 = "등분산이다" 하고 OLS 를 쓴다

        t = stats.t.ppf(0.975, n - 2)
        return (keep, np.abs(b1 - 1) <= t * se_ols, np.abs(b1 - 1) <= t * se_hc3,
                b1, se_ols, se_hc3)


    n = 50
    settings = [("sd=0.8 (homo)", lambda x: 0.8 + 0 * x),
                ("sd=0.3+0.5x", lambda x: 0.3 + 0.5 * x),
                ("sd=0.2+0.3x^2", lambda x: 0.2 + 0.3 * x ** 2)]

    print(f"n = {n}, 10 만 회.  nominal 95% 구간의 실제 포함률")
    print(f"{'data':<16}{'BP pass':>9}{'OLS':>8}{'HC3':>8}{'pretest':>9}")
    rows = []
    for name, f in settings:
        keep, ok_o, ok_h, b1, so, sh = simulate(n, f)
        pre = np.where(keep, ok_o, ok_h)
        print(f"{name:<16}{keep.mean():>9.4f}{ok_o.mean():>8.4f}"
              f"{ok_h.mean():>8.4f}{pre.mean():>9.4f}")
        rows.append((name, keep, ok_o, ok_h, b1, so, sh))

    print("\n사전검정 결과로 표본을 나누어 본 포함률")
    print(f"{'data':<16}{'#pass':>8}{'OLS|pass':>10}{'HC3|pass':>10}{'HC3|rej':>9}")
    for name, keep, ok_o, ok_h, b1, so, sh in rows:
        print(f"{name:<16}{keep.sum():>8d}{ok_o[keep].mean():>10.4f}"
              f"{ok_h[keep].mean():>10.4f}{ok_h[~keep].mean():>9.4f}")

    print("\nsd=0.2+0.3x^2 자료: 기울기의 참 산포와 표준오차의 평균")
    name, keep, ok_o, ok_h, b1, so, sh = rows[2]
    print(f"  all  : SD(b1) = {b1.std():.4f}, "
          f"E[se_ols] = {so.mean():.4f}, E[se_hc3] = {sh.mean():.4f}")
    print(f"  pass : SD(b1) = {b1[keep].std():.4f}, "
          f"E[se_ols] = {so[keep].mean():.4f}, E[se_hc3] = {sh[keep].mean():.4f}")
    ```

    출력:

    ```text
    n = 50, 10 만 회.  nominal 95% 구간의 실제 포함률
    data              BP pass     OLS     HC3  pretest
    sd=0.8 (homo)      0.9497  0.9499  0.9517   0.9506
    sd=0.3+0.5x        0.1528  0.9391  0.9475   0.9510
    sd=0.2+0.3x^2      0.0084  0.8988  0.9432   0.9434

    사전검정 결과로 표본을 나누어 본 포함률
    data               #pass  OLS|pass  HC3|pass  HC3|rej
    sd=0.8 (homo)      94970    0.9497    0.9508   0.9676
    sd=0.3+0.5x        15283    0.9121    0.8894   0.9580
    sd=0.2+0.3x^2        841    0.8169    0.7943   0.9445

    sd=0.2+0.3x^2 자료: 기울기의 참 산포와 표준오차의 평균
      all  : SD(b1) = 0.2013, E[se_ols] = 0.1670, E[se_hc3] = 0.2020
      pass : SD(b1) = 0.2115, E[se_ols] = 0.1414, E[se_hc3] = 0.1383
    ```

    **(1) 무조건 포함률만 보면 사전검정이 흠잡을 데가 없다.** 세 자료에서 $0.9506$, $0.9510$, $0.9434$ 로 모두 명목값 근처이고, **약한 이분산 자료에서는 세 규칙 가운데 가장 명목값에 가깝기까지 하다**($0.9510$ 대 HC3 의 $0.9475$, OLS 의 $0.9391$). 이것이 두 단계 절차가 끈질기게 살아남는 까닭이다.

    까닭은 두 가지다. 첫째, **강한 이분산 자료에서는 사전검정이 거의 늘 기각한다.** BP 통과율이 $0.0084$ 이므로 (다)는 $99.2\%$ 의 경우 (나)와 같은 규칙이고, 그래서 $0.9434$ 와 $0.9432$ 로 사실상 같은 수가 나온다. 사전검정이 잘하는 것이 아니라 **일을 거의 하지 않는** 것이다. 둘째, 약한 이분산에서는 두 치우침이 서로 상쇄된다. 기각한 표본에서 HC3 가 $0.9580$ 으로 과대포함하고 통과한 표본에서 OLS 가 $0.9121$ 로 과소포함하는데, $15\%$ 와 $85\%$ 로 섞으니 $0.9510$ 이 된다. **우연한 상쇄를 정당성으로 읽어서는 안 된다.**

    등분산 자료의 열이 또 하나를 말해 준다. BP 통과율 $0.9497$ 은 사전검정의 실제 크기가 $0.0503$ 으로 제대로 맞는다는 뜻이고, 세 규칙의 포함률도 $0.9499$, $0.9517$, $0.9506$ 으로 모두 같다. **가정이 성립할 때 늘 HC3 가 잃는 것이 사실상 없다.**

    **(2) 사전검정의 판정으로 나누어 보면 이야기가 뒤집힌다.** 사전검정의 값어치는 "통과했으니 이 표본에서는 OLS 를 써도 좋다"는 **조건부 보증**이다. 그 보증을 그대로 재면

    - 약한 이분산: 통과한 표본에서 OLS 포함률 $0.9121$
    - 강한 이분산: 통과한 표본에서 OLS 포함률 $0.8169$

    로, 무조건 포함률($0.9391$, $0.8988$)보다 **더 나쁘다.** 통과시킨 표본만 모으면 $95\%$ 구간이 $82\%$ 밖에 덮지 않는다.

    **같은 표본에서 HC3 로 바꾸어도 고쳐지지 않는다.** $0.8894$, $0.7943$ 으로 오히려 조금 더 낮다. 그러므로 범인은 표준오차 공식이 아니다. 마지막 묶음이 그것을 보인다. 강한 이분산 자료에서 기울기의 참 산포는 전체 $0.2013$, 통과한 표본에서 $0.2115$ 로 **오히려 커졌는데**, 평균 표준오차는 OLS 가 $0.1670\to0.1414$, HC3 가 $0.2020\to0.1383$ 으로 **둘 다 크게 작아졌다.** (전체에서 HC3 의 $0.2020$ 이 참 산포 $0.2013$ 과 거의 같다는 점도 확인해 두자 — HC3 는 무조건적으로는 정확하다.)

    까닭은 BP 가 통과시키는 표본의 정체다. **통과한 표본은 이분산이 없는 표본이 아니라 이분산이 우연히 눈에 띄지 않은 표본**이고, 그것은 큰 $x$ 쪽 잔차가 작게 나온 표본이다. 잔차가 작으니 어떤 공식으로 재도 표준오차가 작게 나오는데, 기울기의 오차는 그만큼 줄지 않는다. **조건부로 보는 순간 어떤 표준오차도 눈금이 맞지 않는다.**

    **결론: (나) 늘 HC3 다.** (나)는 애초에 조건부 주장을 하지 않는다. "이 규칙을 반복하면 $95\%$ 를 덮는다"만 주장하고 그 주장이 세 자료에서 $0.9517$, $0.9475$, $0.9432$ 로 맞는다. (다)는 "통과했으니 OLS 로 충분하다"는 조건부 주장을 덧붙이는데 그 주장이 $0.82$ 로 틀린다. 틀린 주장을 하나 더 하면서 얻는 것은 없다 — 등분산 자료에서도 (나)가 잃는 것이 없으니까. 본문 마지막 상자의 "사전검정 없이 처음부터 로버스트 표준오차" 권고가 회귀에서도 수로 확인된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
본문 표는 `anova_lm(..., robust='hc3')` 이 집단평균을 "가중하지 않음(OLS)" 이라 적었다. 그 칸을 정확히 따져 보자. 일원배치를 셀평균 모형 $y_{ij} = \mu_i + \varepsilon_{ij}$ ($i = 1,\ldots,k$, $j = 1,\ldots,n_i$)로 적합한다.

**(1)** OLS 적합값이 집단평균이므로 지렛대가 $h_{ij} = 1/n_i$ 임을 보이고, HC3 샌드위치가 주는 공분산행렬이 대각이며

$$
\widehat{\operatorname{Var}}_{\text{HC3}}(\bar y_i) = \frac{s_i^2}{n_i-1}
$$

임을 보이시오($s_i^2$ 은 집단 $i$ 의 표본분산). 보통의 $s_i^2/n_i$ 와 몇 배 다른가.

**(2)** $H_0: \mu_1 = \cdots = \mu_k$ 의 Wald 통계량이 $Q = \min_c \sum_i u_i(\bar y_i - c)^2$, $u_i = 1/\widehat{\operatorname{Var}}(\bar y_i)$ 임을 쓰고, 그러므로 HC3 분산분석의 $F$ 가 **Welch 의 $A$ 에서 $n_i$ 를 $n_i-1$ 로 바꾼 것**임을 밝히시오. 본문 표의 "가중하지 않음" 은 어떤 뜻에서 옳고 어떤 뜻에서 오해를 부르는가. 균형·등분산일 때 $F_{\text{HC3}} = \frac{n-1}{n}F_{\text{고전}}$ 임을 보여 보기 3 자료의 $F$ 와 $p$ 를 분수로 적고, 불균형·이분산 자료에서 손계산과 `statsmodels` 를 맞추시오.

</div>

??? success "풀이"

    **(1) 셀평균 모형의 모자행렬은 블록대각이다.** 설계행렬 $\mathbf{X}$ 의 열이 집단 지시자이므로 $\mathbf{X}'\mathbf{X} = \operatorname{diag}(n_1,\ldots,n_k)$, $\hat\mu_i = \bar y_i$ 이고, 모자행렬의 $i$ 번째 블록은 $\frac{1}{n_i}\mathbf{J}_{n_i}$ 다. 따라서

    $$
    h_{ij} = \frac{1}{n_i}, \qquad e_{ij} = y_{ij} - \bar y_i
    $$

    이다. (처치 코딩으로 적합해도 열공간이 같으므로 모자행렬은 그대로다.)

    HC3 샌드위치는 $e_{ij}^2$ 을 $(1-h_{ij})^2$ 으로 나누어 끼운다.

    $$
    \widehat{\operatorname{Var}}_{\text{HC3}}(\hat{\boldsymbol\mu}) = (\mathbf{X}'\mathbf{X})^{-1}\mathbf{X}'\operatorname{diag}\!\left(\frac{e_{ij}^2}{(1-h_{ij})^2}\right)\mathbf{X}(\mathbf{X}'\mathbf{X})^{-1}
    $$

    가운데 행렬의 $(i,i')$ 성분은 서로 다른 집단의 지시자를 곱하므로 $i \ne i'$ 에서 $0$ 이고, 대각성분은

    $$
    \sum_j \frac{e_{ij}^2}{(1-1/n_i)^2} = \left(\frac{n_i}{n_i-1}\right)^2 (n_i-1)s_i^2
    $$

    이다. 양쪽을 $1/n_i$ 씩 곱하면

    $$
    \widehat{\operatorname{Var}}_{\text{HC3}}(\bar y_i) = \frac{1}{n_i^2}\cdot\frac{n_i^2}{(n_i-1)^2}(n_i-1)s_i^2 = \frac{s_i^2}{n_i-1}
    $$

    로 **대각이고 집단마다 따로**다. $\square$

    보통의 $s_i^2/n_i$ 보다 정확히 $\frac{n_i}{n_i-1}$ 배 크다. $n_i = 3$ 이면 $1.5$ 배, $n_i = 10$ 이면 $1.11$ 배, $n_i\to\infty$ 면 $1$ 배다. 연습문제 3 이 "$h_{ii}$ 가 $1/3$ 이라 보정이 크게 작용한다"고 한 말의 정확한 형태가 이 한 줄이고, **HC3 의 보수성이 소표본에서만 문제가 되는 까닭**도 여기 있다. 자유도 하나를 잃는 셈이니, HC3 는 $\bar y_i$ 의 분산을 "$n_i$ 개가 아니라 $n_i - 1$ 개로 평균한 것처럼" 센다.

    **(2) Wald 통계량은 가중제곱합이다.** $\hat{\boldsymbol\mu} = (\bar y_1,\ldots,\bar y_k)'$, $\hat{\mathbf{V}} = \operatorname{diag}(1/u_1,\ldots,1/u_k)$ 라 하자. 귀무가설 $\mathbf{C}\boldsymbol\mu = \mathbf{0}$($\mathbf{C}$ 는 $(k-1)\times k$ 꼴의 대비행렬)의 Wald 통계량은 제약집합까지의 $\hat{\mathbf{V}}^{-1}$ 거리

    $$
    Q = (\mathbf{C}\hat{\boldsymbol\mu})'(\mathbf{C}\hat{\mathbf{V}}\mathbf{C}')^{-1}(\mathbf{C}\hat{\boldsymbol\mu}) = \min_{\mu_1=\cdots=\mu_k}(\hat{\boldsymbol\mu}-\boldsymbol\mu)'\hat{\mathbf{V}}^{-1}(\hat{\boldsymbol\mu}-\boldsymbol\mu) = \min_c \sum_i u_i(\bar y_i - c)^2
    $$

    로 쓰인다(어느 대비행렬을 쓰든 같은 값이다). 최소화하는 $c$ 는 가중평균 $\bar y_u = \sum_i u_i\bar y_i/\sum_i u_i$ 이고 최솟값은 $\sum_i u_i(\bar y_i-\bar y_u)^2$ 이다. 보고되는 $F$ 는 $Q/(k-1)$ 이고 참조분포는 $F(k-1,\,N-k)$ 다.

    이것을 Welch 의 $A = \frac{1}{k-1}\sum_i w_i(\bar X_i - \bar X_w)^2$, $w_i = n_i/s_i^2$ 과 나란히 놓으면 **모양이 똑같고 가중치만 다르다.**

    | | Welch | HC3 분산분석 |
    |---|---|---|
    | 가중치 | $w_i = n_i/s_i^2$ | $u_i = (n_i-1)/s_i^2$ |
    | 보정항 | $1+2(k-2)\Lambda$ 로 나눔 | 없음 |
    | 분모 자유도 | $\nu_2 = 1/(3\Lambda)$ | $N-k$ |

    **그러므로 본문 표의 "가중하지 않음" 은 점추정에 대해서만 옳다.** 집단평균의 추정값은 $\bar y_i$ 이고 Welch 도 같은 값을 쓴다 — 이 점에서 두 방법은 애초에 다르지 않다. 그러나 **검정통계량은 $1/\widehat{\operatorname{Var}}$ 로 가중한다.** `anova_lm(robust='hc3')` 도 가중하고 있으므로, 그 칸은 "Welch 는 가중하는데 HC3 는 안 한다"는 오해를 부른다. 실제 차이는 가중치에 $n_i$ 를 쓰는지 $n_i-1$ 을 쓰는지, 그리고 **자유도를 깎는지 안 깎는지**다.

    **균형·등분산일 때의 정확한 배율.** $n_i = n$, $s_i^2 = s^2$ 이면 $u_i = (n-1)/s^2$ 이 모두 같아 $\bar y_u = \bar y$ 가 되고

    $$
    Q = \frac{n-1}{s^2}\sum_i (\bar y_i - \bar y)^2 = \frac{n-1}{n}\cdot\frac{n\sum_i(\bar y_i-\bar y)^2}{s^2} = \frac{n-1}{n}(k-1)\frac{\text{MSB}}{\text{MSE}}
    $$

    이다(균형자료에서 $\text{MSE} = s^2$ 임은 보기 3 (1)에서 보았다). 그러므로

    $$
    F_{\text{HC3}} = \frac{n-1}{n}F_{\text{고전}}
    $$

    이다. 보기 3 자료는 $n = 3$, $F_{\text{고전}} = 91$ 이므로

    $$
    F_{\text{HC3}} = \frac23\cdot 91 = \frac{182}{3} = 60.6667
    $$

    이고, $d_1 = 2$ 와 $\nu = N-k = 6$ 에 보기 3 (2)의 닫힌 꼴을 쓰면

    $$
    p = \left(1 + \frac{2}{6}\cdot\frac{182}{3}\right)^{-3} = \left(\frac{191}{9}\right)^{-3} = \frac{729}{6967871} = 1.0462\times10^{-4}
    $$

    이다. 연습문제 3 의 출력에 적힌 $F = 60.666667$, $p = 0.000105$ 와 맞는다. 보기 2 자료는 $n = 5$ 이므로 $F_{\text{HC3}} = \frac45\cdot\frac{182}{3} = 48.5333$ 이어야 한다.

    **확인한다.**

    ```python
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy import stats


    def hc3_wald_F(groups):
        """일원배치에서 anova_lm(robust='hc3') 의 F 를 손으로 조립한다.
        HC3 는 Var(ybar_i) 를 s_i^2/n_i 가 아니라 s_i^2/(n_i - 1) 로 추정한다."""
        n = np.array([len(g) for g in groups], float)
        m = np.array([np.mean(g) for g in groups])
        s2 = np.array([np.var(g, ddof=1) for g in groups])
        u = (n - 1) / s2                      # 1 / Var_HC3(ybar_i)
        mu = (u * m).sum() / u.sum()          # u 로 가중한 전체평균
        return (u * (m - mu) ** 2).sum() / (len(groups) - 1)


    def welch_F(groups):
        """비교용 Welch. 가중치가 w_i = n_i/s_i^2 이고 보정항과 nu2 가 붙는다."""
        k = len(groups)
        n = np.array([len(g) for g in groups], float)
        m = np.array([np.mean(g) for g in groups])
        s2 = np.array([np.var(g, ddof=1) for g in groups])
        w = n / s2
        mw = (w * m).sum() / w.sum()
        A = (w * (m - mw) ** 2).sum() / (k - 1)
        lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum() / (k ** 2 - 1)
        return A / (1 + 2 * (k - 2) * lam), 1 / (3 * lam)


    def sm_hc3_F(groups):
        """statsmodels 가 실제로 내놓는 값."""
        lab = np.concatenate([[chr(65 + i)] * len(g) for i, g in enumerate(groups)])
        d = pd.DataFrame({"Score": np.concatenate(groups), "Group": lab})
        fit = sm.formula.ols("Score ~ Group", data=d).fit()
        return sm.stats.anova_lm(fit, typ=2, robust="hc3")["F"].iloc[0]


    cases = [
        ("보기 3 자료", [[10, 12, 14], [22, 24, 26], [32, 34, 36]]),
        ("보기 2 자료", [[10, 12, 14, 16, 18], [22, 24, 26, 28, 30],
                         [32, 34, 36, 38, 40]]),
        ("불균형·이분산 자료", [[10, 12, 14, 20], [22, 24, 30],
                                [32, 34, 36, 38, 50, 41]]),
    ]

    for name, g in cases:
        g = [np.asarray(x, float) for x in g]
        k, N = len(g), sum(len(x) for x in g)
        nn = [len(x) for x in g]
        F_sm, F_hand = sm_hc3_F(g), hc3_wald_F(g)
        F_cls = stats.f_oneway(*g).statistic
        F_w, nu2 = welch_F(g)
        bal = ("   (n-1)/n = %.6f" % ((nn[0] - 1) / nn[0])) if len(set(nn)) == 1 else ""
        print(f"{name}   n_i = {nn}")
        print(f"  statsmodels HC3 F = {F_sm:.8f}     손으로 조립한 F = {F_hand:.8f}")
        print(f"  고전 F = {F_cls:.6f}   F_HC3/F_고전 = {F_hand / F_cls:.6f}{bal}")
        print(f"  p(HC3)   = {stats.f.sf(F_hand, k - 1, N - k):.8f}"
              f"   (df = {k - 1}, {N - k})")
        print(f"  p(Welch) = {stats.f.sf(F_w, k - 1, nu2):.8f}"
              f"   (F_W = {F_w:.4f}, nu2 = {nu2:.4f})")

    g = [np.asarray(x, float) for x in cases[2][1]]
    n = np.array([len(x) for x in g], float)
    s2 = np.array([np.var(x, ddof=1) for x in g])
    print("\n불균형·이분산 자료의 두 가중치")
    print(f"  n_i       = {n}")
    print(f"  s_i^2     = {np.round(s2, 4)}")
    print(f"  HC3   u_i = (n_i-1)/s_i^2 = {np.round((n - 1) / s2, 6)}")
    print(f"  Welch w_i =     n_i/s_i^2 = {np.round(n / s2, 6)}")
    ```

    출력:

    ```text
    보기 3 자료   n_i = [3, 3, 3]
      statsmodels HC3 F = 60.66666667     손으로 조립한 F = 60.66666667
      고전 F = 91.000000   F_HC3/F_고전 = 0.666667   (n-1)/n = 0.666667
      p(HC3)   = 0.00010462   (df = 2, 6)
      p(Welch) = 0.00062500   (F_W = 78.0000, nu2 = 4.0000)
    보기 2 자료   n_i = [5, 5, 5]
      statsmodels HC3 F = 48.53333333     손으로 조립한 F = 48.53333333
      고전 F = 60.666667   F_HC3/F_고전 = 0.800000   (n-1)/n = 0.800000
      p(HC3)   = 0.00000177   (df = 2, 12)
      p(Welch) = 0.00001975   (F_W = 56.0000, nu2 = 8.0000)
    불균형·이분산 자료   n_i = [4, 3, 6]
      statsmodels HC3 F = 20.69519328     손으로 조립한 F = 20.69519328
      고전 F = 24.596251   F_HC3/F_고전 = 0.841396
      p(HC3)   = 0.00027899   (df = 2, 10)
      p(Welch) = 0.00157428   (F_W = 23.3892, nu2 = 5.8933)

    불균형·이분산 자료의 두 가중치
      n_i       = [4. 3. 6.]
      s_i^2     = [18.6667 17.3333 41.5   ]
      HC3   u_i = (n_i-1)/s_i^2 = [0.160714 0.115385 0.120482]
      Welch w_i =     n_i/s_i^2 = [0.214286 0.173077 0.144578]
    ```

    **손계산이 여덟 자리까지 맞는다.** 세 자료 모두 `statsmodels` 의 $F$ 와 $\sum_i u_i(\bar y_i-\bar y_u)^2/(k-1)$ 이 소수 여덟째 자리까지 같다. **불균형·이분산 자료에서도 맞는다**는 것이 중요하다 — 그 자료에서는 $u_i$ 가 $(0.1607,\,0.1154,\,0.1205)$ 로 집단마다 다르므로, 맞는다는 사실 자체가 HC3 분산분석이 **가중을 하고 있다**는 증거다.

    균형자료의 배율도 정확하다. 보기 3 자료는 $F_{\text{HC3}}/F_{\text{고전}} = 0.666667 = \frac23 = \frac{n-1}{n}$, 보기 2 자료는 $0.8 = \frac45$ 다. $p$값은 $0.00010462$ 로 손으로 적은 $\frac{729}{6967871}$ 과 같고 본문 표의 $0.000105$ 와 맞는다. 불균형 자료에서는 배율이 $0.8414$ 로 어느 $\frac{n_i-1}{n_i}$ 와도 같지 않다 — 가중치가 집단마다 달라지면 단순한 배율로 떨어지지 않는다.

    **그래서 두 방법의 차이는 자유도다.** 세 자료 모두에서 HC3 의 $p$값이 Welch 보다 **작다**($1.05\times10^{-4}$ 대 $6.25\times10^{-4}$, $1.8\times10^{-6}$ 대 $2.0\times10^{-5}$, $2.8\times10^{-4}$ 대 $1.6\times10^{-3}$). 가중치의 $n_i \to n_i-1$ 은 통계량을 깎아 HC3 를 **보수적으로** 만들지만, 분모 자유도를 $N-k$ 로 그대로 두는 것이 반대 방향으로 더 크게 작용한다. 보기 3 (2)가 이미 재어 둔 대로 **값을 치르는 쪽은 자유도**이고($14.2$ 배 대 $1.6$ 배), HC3 는 그 값을 치르지 않는다.

    **결론.** 이름에 robust 가 붙었다고 Welch 가 되는 것이 아니다. HC3 분산분석은 가중도 하고 샌드위치도 쓰지만 **Welch–Satterthwaite 자유도 보정을 빼먹은** 통계량이므로, 집단이 작고 분산이 많이 다를 때 $p$값이 Welch 보다 작게 나온다. 등분산이 깨진 소표본에서 1 종 오류율을 지키려면 자유도를 깎는 쪽, 곧 Welch 를 써야 한다.

---

## 정리하며

분산 검정이 **어디에 쓰이는지** 정리한다.

- **회귀의 등분산성 확인**이 가장 흔한 용도다. 위반하면 계수는 불편이지만 표준오차가 틀리며, 13장에서 본 대로 로버스트 표준오차나 가중최소제곱으로 대처한다.
- **분산분석의 가정 확인**이 두 번째다. 다만 11장에서 논했듯 **사전 검정으로 방법을 고르는 2단계 절차는 권하지 않으며**, 웰치를 기본으로 쓰는 편이 낫다.
- **분산 자체가 관심사인 경우**가 세 번째다. 품질관리의 공정 변동, 금융의 변동성 비교가 그렇다. 이때는 검정이 보조가 아니라 목적이다.
- **어느 용도든 정규성 확인이 먼저다.** 이 장 전체가 그 점을 반복했으며, 고전적 검정과 로버스트 검정의 선택이 거기서 갈린다.
- **검정보다 그림이 먼저다.** 집단별 상자그림과 적합값 대 잔차 그림이 대부분의 판단을 해 준다.

**이것으로 15장이 끝난다.** 일표본 카이제곱 검정에서 시작해 $F$·바틀렛의 고전적 검정과 그 취약성, 레빈 계열의 로버스트 대안, 그리고 부트스트랩·베이즈·가능도비의 현대적 접근까지 보았다.

다음 장 **비모수 검정**으로 넘어간다. 분포 가정을 아예 최소화하는 방법들이며, 이 장의 로버스트 검정이 그 예고편이었다.
