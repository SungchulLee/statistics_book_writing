# Bartlett 검정

!!! note "이 주제를 다루는 다른 곳"
    분산분석의 등분산성 사전확인이라는 좁은 맥락에서 같은 검정을 짧게 쓰는 예가
    **11.5 가정**에 있다.

## 개요

Bartlett 검정은 여러 집단이 같은 분산을 갖는다는 귀무가설(등분산성)을 확인한다. 등분산에 대한 고전적 검정 가운데 자료가 정말로 정규분포를 따를 때 가장 강력하다. 그러나 정규성 이탈에 매우 민감하여, 치우쳤거나 꼬리가 두꺼운 자료에 적용하면 거짓 양성률이 부풀려진다.

## 검정 설정

정규 모집단에서 뽑은 크기 $n_1, \ldots, n_k$인 독립 집단 $k$개가 주어졌을 때 가설은

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{대} \quad H_1 : \sigma_i^2 \text{이 모두 같지는 않다}.
$$

## 검정통계량

$S_i^2$을 집단 $i$의 표본분산, $N = \sum_{i=1}^k n_i$라 하고 합동분산을

$$
S_p^2 = \frac{1}{N - k} \sum_{i=1}^{k} (n_i - 1) S_i^2
$$

로 정의한다. Bartlett 검정통계량은

$$
T = \frac{(N - k)\ln S_p^2 - \sum_{i=1}^{k}(n_i - 1)\ln S_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k}\frac{1}{n_i - 1} - \frac{1}{N - k}\right)}.
$$

$H_0$과 정규성 아래에서 근사적으로 $T \sim \chi^2(k-1)$이다.

SciPy는 `scipy.stats.bartlett`을 직접 제공한다.

<div class="codebox" markdown>

**예제 1.** 분산 차이를 키워 가며

```python
import numpy as np
import scipy.stats as stats

# 한쪽의 표준편차를 1 로 두고 다른 쪽을 조금씩 키워 가며 검정한다.
# 5% 차이는 잡아내지 못하고 20% 쯤 되어야 걸린다. 등분산 검정의 검정력이
# 생각보다 낮다는 것을 보여 주는 대목이다.
size, seed = 100, 1
x = stats.norm(loc=0, scale=1).rvs(size, random_state=seed)

for scale in [1.00, 1.05, 1.10, 1.15, 1.20]:
    y = stats.norm(loc=1, scale=scale).rvs(size, random_state=seed)
    stat, pval = stats.bartlett(x, y)
    print(f"sigma_y={scale:.2f}  chi2={stat:.4f}  p={pval:.3f}")
```

출력:

```text
sigma_y=1.00  chi2=-0.0000  p=1.000
sigma_y=1.05  chi2=0.2344  p=0.628
sigma_y=1.10  chi2=0.8934  p=0.345
sigma_y=1.15  chi2=1.9179  p=0.166
sigma_y=1.20  chi2=3.2564  p=0.071
```

</div>

!!! note "$p$값이 F 검정과 정확히 같다"
    이 표의 $p$값 $(1.000, 0.628, 0.345, 0.166, 0.071)$은 [분산 동일성에 대한 F 검정](../f_test/f_test_variances.md) 페이지의 같은 자료에 대한 $p$값과 **소수 셋째 자리까지 동일**하다.

    우연이 아니다. $k = 2$이고 $n_1 = n_2$일 때 Bartlett 통계량은 F 통계량의 단조함수이므로 두 검정이 항상 같은 결론을 낸다. 연습문제 5에서 증명한다.

    첫 행의 `chi2=-0.0000`은 부동소수점 오차이다. 두 표본이 같은 seed를 쓰므로 $S_1^2 = S_2^2$이 되어 이론적으로 정확히 0이어야 한다(15.4절 연습문제 1의 등호 조건).

    이 예제도 F 검정 페이지와 마찬가지로 `x`와 `y`가 같은 seed를 공유하여 표집변동이 제거되어 있다.

## 해석

- 두 집단의 분산이 같으면 검정통계량이 0에 가깝고 $p$값이 크다.
- 분산비가 1에서 멀어질수록 $T$가 커지고 $p$값이 작아진다.
- Bartlett 검정은 정규성 가정이 충분히 정당화될 때만 적용해야 한다. 비정규성 아래에서는 Levene이나 Brown-Forsythe 검정이 더 안전하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 크기 $n_1 = 10$, $n_2 = 12$, $n_3 = 15$인 세 집단의 표본분산이 $S_1^2 = 4.1$, $S_2^2 = 3.8$, $S_3^2 = 5.6$이다. 합동분산 $S_p^2$과 Bartlett 검정통계량 $T$를 손으로 계산하라(로그는 계산기를 써도 좋다).

</div>

??? success "풀이"

    전체 표본크기는 $N = 10 + 12 + 15 = 37$이고 $k = 3$이다. 합동분산은

    $$
    S_p^2 = \frac{9(4.1) + 11(3.8) + 14(5.6)}{37 - 3} = \frac{36.9 + 41.8 + 78.4}{34} = \frac{157.1}{34} = 4.6206.
    $$

    $T$의 분자는

    $$
    34 \ln(4.6206) - [9\ln(4.1) + 11\ln(3.8) + 14\ln(5.6)]
    $$

    $$
    = 34(1.5305) - [9(1.4110) + 11(1.3350) + 14(1.7228)]
    $$

    $$
    = 52.037 - [12.699 + 14.685 + 24.119] = 52.037 - 51.503 = 0.535.
    $$

    보정인자는

    $$
    C = 1 + \frac{1}{6}\left(\frac{1}{9} + \frac{1}{11} + \frac{1}{14} - \frac{1}{34}\right) = 1 + \frac{1}{6}(0.1111 + 0.0909 + 0.0714 - 0.0294) = 1.0407.
    $$

    따라서 $T = 0.535 / 1.0407 = 0.514$이다.

    ```python
    import numpy as np
    import scipy.stats as stats

    n = np.array([10, 12, 15])
    s2 = np.array([4.1, 3.8, 5.6])
    k, nu = len(n), n - 1

    sp2 = np.sum(nu * s2) / np.sum(nu)
    num = np.sum(nu) * np.log(sp2) - np.sum(nu * np.log(s2))
    C = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / nu) - 1 / np.sum(nu))
    T = num / C
    print(f"Sp2 = {sp2:.4f}, numerator = {num:.4f}, C = {C:.4f}")
    print(f"T = {T:.4f}, p = {stats.chi2.sf(T, k - 1):.4f}")
    ```

    출력:

    ```text
    Sp2 = 4.6206, numerator = 0.5351, C = 1.0407
    T = 0.5142, p = 0.7733
    ```

    $\chi^2(2)$와 비교하면 $p = 0.773$으로 큰 값이므로 등분산을 기각하지 못한다. 표본분산의 최대·최소 비가 $5.6/3.8 = 1.47$로 작으므로 예상되는 결과이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> Python으로 $N(0,1)$, $N(0,1.5)$, $N(0,2)$에서 각각 크기 50인 세 집단을 생성하라. Bartlett 검정을 적용하고 검정통계량과 $p$값을 보고하라. 기대한 결과와 일치하는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(42)
    g1 = rng.normal(0, 1.0, 50)
    g2 = rng.normal(0, 1.5, 50)
    g3 = rng.normal(0, 2.0, 50)

    print("sample variances:",
          [round(np.var(g, ddof=1), 3) for g in (g1, g2, g3)])
    stat, pval = stats.bartlett(g1, g2, g3)
    print(f"Bartlett: chi2={stat:.3f}, p={pval:.4g}")
    ```

    출력:

    ```text
    sample variances: [0.59, 1.322, 4.032]
Bartlett: chi2=43.960, p=2.846e-10
    ```

    참 분산이 1, 2.25, 4로 크게 다르므로 $p$값이 $2.8 \times 10^{-10}$으로 매우 작고 $H_0$을 올바르게 기각한다.

    **주의: `normal(0, s)`의 두 번째 인자는 표준편차이다.** 문제 서술의 "$N(0,1.5)$"는 표기 관례에 따라 분산 1.5를 뜻할 수도 있으나, 여기서는 코드대로 **표준편차 1.5**로 읽어 참 분산이 $2.25$이다. 정규분포를 쓸 때 두 번째 인자가 분산인지 표준편차인지는 늘 확인해야 한다. NumPy와 SciPy는 표준편차를, 많은 교과서 표기 $N(\mu, \sigma^2)$은 분산을 쓴다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> Bartlett 검정이 비정규성에 민감한 이유를 설명하라. 특히 첨도가 $S^2$의 분포에, 나아가 검정통계량에 어떤 영향을 주는지 논하라.

</div>

??? success "풀이"

    Bartlett 통계량의 유도는 $(n_i - 1)S_i^2/\sigma_i^2 \sim \chi^2(n_i - 1)$을 가정하는데, 이는 자료가 정규일 때만 정확히 성립한다. 초과첨도 $\kappa > 0$(두꺼운 꼬리)인 분포에서는 $S^2$의 분산이 부풀려진다.

    $$
    \operatorname{Var}(S^2) \approx \frac{2\sigma^4}{n-1} + \frac{\kappa \sigma^4}{n}.
    $$

    추가항 $\kappa\sigma^4/n$ 때문에 $S_i^2$이 $\chi^2$ 기준분포가 예측하는 것보다 더 변동한다. 결과적으로 $\ln S_i^2$의 집단간 변동이 부풀려지고, $H_0$ 아래에서 $T$가 $\chi^2(k-1)$보다 확률적으로 커진다. 그래서 기각률이 명목 $\alpha$를 훨씬 넘는다.

    **로그가 문제를 순수하게 만든다.** 15.4절 연습문제 2에서 보았듯 델타 방법으로

    $$
    \operatorname{Var}(\ln S^2) \approx \frac{\gamma_2 + 2}{n}
    $$

    이고, 이 값은 $\sigma^2$에 **전혀 의존하지 않고 오직 $\gamma_2$에만 의존한다.** 로그 변환이 척도 정보를 지우고 모양 정보만 남기므로, 첨도의 영향이 감쇄 없이 그대로 통계량에 실린다.

    **정량적으로.** 정규성 아래에서 $\operatorname{Var}(\ln S^2) = 2/n$인데 $\gamma_2 = 6$이면 $8/n$으로 네 배가 된다. $T$의 각 항이 네 배로 부풀려지므로 $T$ 자체가 대략 네 배가 되고, $\chi^2$ 임계값을 훨씬 자주 넘게 된다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 5,000회 반복 모의실험을 수행하라. 표준 대수정규분포에서 크기 20인 세 집단을 생성한다(귀무가설 아래에서 등분산). $\alpha = 0.05$로 Bartlett 검정을 적용하고 거짓 양성률을 추정하라. Levene 검정과 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    import scipy.stats as stats

    rng = np.random.default_rng(0)
    n_sims, n, alpha = 5000, 20, 0.05
    rej_bart, rej_lev = 0, 0

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, n)
        g2 = rng.lognormal(0, 1, n)
        g3 = rng.lognormal(0, 1, n)
        _, p_b = stats.bartlett(g1, g2, g3)
        _, p_l = stats.levene(g1, g2, g3)      # median-centred (Brown-Forsythe)
        if p_b < alpha:
            rej_bart += 1
        if p_l < alpha:
            rej_lev += 1

    print(f"Bartlett false-positive rate: {rej_bart/n_sims:.4f}")
    print(f"Levene   false-positive rate: {rej_lev/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett false-positive rate: 0.6748
    Levene   false-positive rate: 0.0392
    ```

    **Bartlett의 거짓 양성률이 $0.675$이다.** 명목값의 **13배**이며, 등분산인 자료의 3분의 2에서 "분산이 다르다"고 잘못 판정한다. 이는 이 장에서 관찰한 가장 극단적인 크기 왜곡이다.

    이유는 $\text{Lognormal}(0,1)$의 초과첨도가

    $$
    \gamma_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 6 = e^4 + 2e^3 + 3e^2 - 6 = 110.9
    $$

    로 극단적이기 때문이다. $t_5$의 6이나 지수분포의 6과 비교하면 열여덟 배가 넘는다.

    Levene(SciPy 기본값인 중앙값 중심, 곧 Brown-Forsythe)은 $0.039$로 명목값 근처를 유지한다. 15.5절에서 반복해 본 결론이 여기서도 확인된다.

    **실무적 결론.** 소득, 자산가격, 대기시간처럼 대수정규에 가까운 자료에 Bartlett 검정을 쓰는 것은 사실상 난수 생성기를 돌리는 것과 다름없다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $k = 2$이고 $n_1 = n_2 = n$일 때 Bartlett 검정통계량이 F 통계량 $F = S_1^2/S_2^2$의 단조함수임을 증명하라. 곧 두 검정이 $H_0$ 기각 여부에서 항상 일치함을 보여라.

</div>

??? success "풀이"

    $k = 2$이고 표본크기가 같으면 $S_p^2 = (S_1^2 + S_2^2)/2$이다. $r = S_1^2/S_2^2$이라 쓰면 $S_p^2 = S_2^2(1 + r)/2$이다.

    $T$의 분자는

    $$
    2(n-1)\ln S_p^2 - (n-1)\ln S_1^2 - (n-1)\ln S_2^2 = (n-1)\bigl[2\ln S_p^2 - \ln S_1^2 - \ln S_2^2\bigr].
    $$

    $S_p^2 = S_2^2(1+r)/2$와 $S_1^2 = rS_2^2$을 대입하면

    $$
    2\ln\frac{S_2^2(1+r)}{2} - \ln(rS_2^2) - \ln S_2^2 = 2\ln\frac{1+r}{2} + 2\ln S_2^2 - \ln r - 2\ln S_2^2 = 2\ln\frac{1+r}{2} - \ln r.
    $$

    $S_2^2$이 완전히 소거되어 분자가 $r$만의 함수가 된다.

    $$
    g(r) = (n-1)\left[2\ln\frac{1+r}{2} - \ln r\right].
    $$

    보정인자 $C$도 $n$에만 의존하는 상수이므로 $T = g(r)/C$ 역시 $r$만의 함수이다.

    **단조성.** $g$를 미분하면

    $$
    g'(r) = (n-1)\left[\frac{2}{1+r} - \frac{1}{r}\right] = (n-1)\cdot\frac{2r - (1+r)}{r(1+r)} = (n-1)\cdot\frac{r-1}{r(1+r)}.
    $$

    $r < 1$이면 $g' < 0$이고 $r > 1$이면 $g' > 0$이다. 곧 $g$는 $r = 1$에서 최솟값 $g(1) = (n-1)[2\ln 1 - \ln 1] = 0$을 갖고 양쪽으로 **엄격히 증가**한다.

    따라서 기각역 $\{T > \chi^2_{1-\alpha}(1)\}$은 어떤 상수 $c_L < 1 < c_U$에 대해 $\{r < c_L\} \cup \{r > c_U\}$로 대응된다. 이것이 정확히 양측 F 검정의 기각역 형태이므로 두 검정은 항상 일치한다.

    **수치 확인.** 본문 코드의 출력에서 Bartlett의 $p$값 $(1.000, 0.628, 0.345, 0.166, 0.071)$이 F 검정 페이지의 $p$값과 완전히 같다. 위 증명의 직접적 확인이다.

    **$n_1 \neq n_2$이면 어떻게 되는가.** 그때는 $S_p^2$이 가중평균이 되어 $S_2^2$이 소거되지 않으므로 $T$가 $r$만의 함수가 아니다. 두 검정이 조금씩 다른 결론을 낼 수 있다. 다만 $H_0$ 아래에서 두 통계량의 상관이 매우 높으므로 실무에서 차이가 드러나는 경우는 드물다. $\square$

---

## 정리하며

`scipy.stats.bartlett` 로 **구현과 확인**을 했다.

- **집단별 배열을 넘기면 통계량과 $p$ 값이 나온다.** `bartlett(g1, g2, g3)` 형태다.
- **표본크기가 달라도 된다.** 자유도로 가중하므로 불균형 설계에서도 쓸 수 있다.
- **각 집단에 관측이 최소 2 개는 있어야 한다.** 분산을 계산할 수 없으면 오류가 난다.
- **정규 자료로 먼저 검증한다.** 등분산일 때 기각률이 명목 $\alpha$ 에 맞는지 확인하면 코드가 옳게 작동함을 알 수 있다.
- **분산비를 바꿔 가며 검정력을 본다.** 비가 커질수록 기각률이 오르는 것이 정상이며, 그 속도가 검정력 곡선이다.

다음 절 **바틀렛 민감도**에서 비정규 자료로 돌려 본다.
