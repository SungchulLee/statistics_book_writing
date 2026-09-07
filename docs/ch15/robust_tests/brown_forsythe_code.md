# Brown-Forsythe 검정 (scipy)

## 개요

Brown-Forsythe 검정은 여러 집단의 분산 동일성을 평가하는, Bartlett 검정의 로버스트 대안이다. Levene 검정의 변형으로 중심화에 집단평균 대신 집단중앙값을 써서 정규성 이탈에 저항한다. SciPy에서는 `scipy.stats.levene`에 `center='median'` 인자를 주어 쓴다.

---

## 검정 설정

크기 $n_1, \ldots, n_k$인 독립 집단 $k$개가 주어졌을 때 가설은

$$
H_0 : \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2 \quad \text{대} \quad H_1 : \sigma_i^2 \text{이 모두 같지는 않다}
$$

## 작동 방식

Brown-Forsythe 검정은 원래의 관측값을 집단중앙값으로부터의 절대편차로 변환한다.

$$
z_{ij} = |x_{ij} - \tilde{x}_i|
$$

여기서 $\tilde{x}_i$는 집단 $i$의 중앙값이다. 그런 다음 변환값 $z_{ij}$에 일원분산분석 $F$ 검정을 수행한다.

$$
W = \frac{\sum_{i=1}^{k} n_i (\bar{z}_{i\cdot} - \bar{z}_{\cdot\cdot})^2 \,/\, (k - 1)}{\sum_{i=1}^{k}\sum_{j=1}^{n_i} (z_{ij} - \bar{z}_{i\cdot})^2 \,/\, (N - k)}
$$

여기서 $\bar{z}_{i\cdot}$은 집단 $i$의 변환값 평균, $\bar{z}_{\cdot\cdot}$은 전체평균, $N = \sum n_i$이다. $H_0$ 아래에서 $W$는 근사적으로 $F(k-1, N-k)$이다.

---

## 왜 중앙값인가

원래의 Levene 검정은 중심화에 집단 **평균**을 쓴다($z_{ij} = |x_{ij} - \bar{x}_i|$). Brown과 Forsythe(1974)는 평균을 중앙값으로 바꾸면 다음과 같은 검정이 됨을 보였다.

- 치우쳤거나 꼬리가 두꺼운 분포에서 올바른 제1종 오류율을 유지한다.
- 정규성 아래에서 Levene 검정에 견줄 만한 검정력을 갖는다.
- 중앙값이 위치의 로버스트 추정량이므로 이상점의 영향을 덜 받는다.

---

## 코드

SciPy에서는 함수 호출 한 번으로 끝난다.

```python
import numpy as np
from scipy.stats import levene

g1 = np.array([12, 15, 14, 10, 13, 14, 12, 11], dtype=float)
g2 = np.array([22, 25, 20, 18, 24, 23, 19, 21], dtype=float)
g3 = np.array([32, 35, 34, 30, 33, 34, 32, 31], dtype=float)

print("variances:", [round(np.var(g, ddof=1), 4) for g in (g1, g2, g3)])

W, p = levene(g1, g2, g3, center='median')
print(f"Brown-Forsythe W = {W:.6f}, p-value = {p:.6f}")

W_mean, p_mean = levene(g1, g2, g3, center='mean')
print(f"Levene (mean)  W = {W_mean:.6f}, p-value = {p_mean:.6f}")
```

출력:

```text
variances: [2.8393, 6.0, 2.8393]
Brown-Forsythe W = 1.107595, p-value = 0.348894
Levene (mean)  W = 1.121795, p-value = 0.344444
```

---

## 해석

- $W$ 통계량이 크면($p$값이 작으면) $H_0$을 기각하고 집단분산이 모두 같지는 않다고 결론짓는다.
- 이 세 집단은 주로 위치가 다르고 산포는 비슷하므로 $W$가 작고 $p$가 크게 나온다. 등분산을 기각하지 못한다.
- 자료의 정규성이 불확실할 때 등분산성 검정의 권장 기본값이 Brown-Forsythe 검정이다.

표본분산이 $2.84$, $6.00$, $2.84$로 최대·최소 비가 $2.1$인데도 유의하지 않다. 각 집단 $n = 8$로는 검정력이 매우 낮기 때문이다.

---

## 연습문제

**연습문제 1.** 두 집단 $\mathbf{x}_1 = (2, 4, 6)$과 $\mathbf{x}_2 = (1, 5, 9, 13)$에 대해 Brown-Forsythe 검정통계량 $W$를 손으로 계산하라. 집단중앙값, 절대편차, 편차의 집단평균, $F$ 비의 각 단계를 보여라.

??? success "풀이"

    집단중앙값: $\tilde{x}_1 = 4$, $\tilde{x}_2 = 7$.

    중앙값으로부터의 절대편차:

    - 집단 1: $z_{11} = |2-4| = 2$, $z_{12} = |4-4| = 0$, $z_{13} = |6-4| = 2$
    - 집단 2: $z_{21} = |1-7| = 6$, $z_{22} = |5-7| = 2$, $z_{23} = |9-7| = 2$, $z_{24} = |13-7| = 6$

    집단평균: $\bar{z}_1 = (2+0+2)/3 = 4/3$, $\bar{z}_2 = (6+2+2+6)/4 = 4$.

    전체평균: $\bar{z} = (2+0+2+6+2+2+6)/7 = 20/7 = 2.8571$.

    집단간 제곱합:

    $$
    \text{SS}_B = 3\left(\tfrac{4}{3} - \tfrac{20}{7}\right)^2 + 4\left(4 - \tfrac{20}{7}\right)^2 = 3(1.5238)^2 + 4(1.1429)^2
    $$

    $$
    = 3(2.3220) + 4(1.3061) = 6.9660 + 5.2245 = 12.1905
    $$

    집단내 제곱합:

    $$
    \text{SS}_W = (2-\tfrac{4}{3})^2 + (0-\tfrac{4}{3})^2 + (2-\tfrac{4}{3})^2 + (6-4)^2 + (2-4)^2 + (2-4)^2 + (6-4)^2
    $$

    $$
    = 0.4444 + 1.7778 + 0.4444 + 4 + 4 + 4 + 4 = 18.6667
    $$

    $$
    W = \frac{12.1905 / 1}{18.6667 / 5} = \frac{12.1905}{3.7333} = 3.2653
    $$

    ```python
    import numpy as np
    from scipy.stats import levene, f

    x1 = np.array([2, 4, 6.])
    x2 = np.array([1, 5, 9, 13.])
    print(levene(x1, x2, center='median'))
    print(f"F critical (1, 5) at alpha=0.05: {f.ppf(0.95, 1, 5):.4f}")
    ```

    출력:

    ```text
    LeveneResult(statistic=3.2653061224489797, pvalue=0.13057297707695592)
    F critical (1, 5) at alpha=0.05: 6.6079
    ```

    $F(1, 5)$의 $\alpha = 0.05$ 임계값이 $6.61$이므로 $3.27 < 6.61$이 되어 $H_0$을 기각하지 못한다($p = 0.131$).

    표본분산이 $4$ 대 $26.67$로 6.7배나 차이 나는데도 그렇다. $n_1 = 3$, $n_2 = 4$라는 극단적으로 작은 표본에서는 어떤 검정도 무력하다. $\square$

---

**연습문제 2.** 모의실험 연구를 수행하라. 표준 대수정규분포에서 크기 30인 세 집단을 생성한다(모두 등분산). Bartlett 검정과 Brown-Forsythe 검정을 $\alpha = 0.05$에서 3,000회 반복 적용하여 거짓 양성률을 비교하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import bartlett, levene

    rng = np.random.default_rng(42)
    rej_bart, rej_bf = 0, 0
    n_sims = 3000

    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, 30)
        g2 = rng.lognormal(0, 1, 30)
        g3 = rng.lognormal(0, 1, 30)
        _, p_b = bartlett(g1, g2, g3)
        _, p_bf = levene(g1, g2, g3, center='median')
        if p_b < 0.05:
            rej_bart += 1
        if p_bf < 0.05:
            rej_bf += 1

    print(f"Bartlett false-positive rate:       {rej_bart/n_sims:.4f}")
    print(f"Brown-Forsythe false-positive rate: {rej_bf/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett false-positive rate:       0.7303
    Brown-Forsythe false-positive rate: 0.0307
    ```

    **Bartlett의 거짓 양성률이 $0.730$이다.** 명목값의 **15배**로, 등분산인 자료의 4분의 3에서 잘못 기각한다.

    $\text{Lognormal}(0,1)$의 초과첨도가 $110.9$로 극단적이기 때문이다. 15.8절 [Bartlett 검정](../bartlett_test/bartlett_test_code.md) 연습문제 4에서 $n = 20$일 때 $0.675$였는데, $n = 30$으로 키우니 오히려 $0.730$으로 **악화**되었다. 15.3절에서 확인한 "표본을 키우면 나빠진다"는 성질이 여기서도 나타난다.

    Brown-Forsythe는 $0.031$로 명목값보다 약간 보수적이지만 완전히 통제된다. **로버스트 중심화의 효과가 24배 차이로 나타난다.** $\square$

---

**연습문제 3.** Brown-Forsythe 검정이 위치 이동에 불변임을 증명하라. 곧 집단 $i$의 모든 관측값에 상수 $c_i$를 더해도 검정통계량 $W$가 변하지 않음을 보여라.

??? success "풀이"

    집단 $i$의 모든 $j$에 대해 $x_{ij}' = x_{ij} + c_i$라 하자. 집단중앙값도 같은 양만큼 이동한다. $\tilde{x}_i' = \tilde{x}_i + c_i$. 변환값은

    $$
    z_{ij}' = |x_{ij}' - \tilde{x}_i'| = |(x_{ij} + c_i) - (\tilde{x}_i + c_i)| = |x_{ij} - \tilde{x}_i| = z_{ij}
    $$

    모든 $z_{ij}$ 값이 변하지 않으므로 검정통계량 $W$도 동일하다. 이는 Brown-Forsythe 검정이 집단평균이 아니라 각 집단 내부의 산포에만 의존함을 확인해 준다.

    **왜 이것이 중요한 성질인가.** 분산 검정이 위치에 의존한다면, 집단평균이 크게 다른 자료에서 분산 차이와 평균 차이를 구별할 수 없게 된다. 본문 예제에서 세 집단의 평균이 $12.6$, $21.5$, $32.6$으로 크게 다른데도 $W$가 오직 산포만 반영한다.

    **척도에는 불변이 아니다.** 모든 집단에 같은 상수 $c$를 **곱하면** 모든 $z_{ij}$가 $c$배가 되지만, $W$는 분자와 분모 모두 $c^2$배가 되므로 여전히 불변이다. 그러나 집단마다 **다른** 상수를 곱하면 분산비가 바뀌므로 $W$가 달라진다. 이는 당연하며, 그것이 바로 이 검정이 탐지하려는 것이다. $\square$

---

**연습문제 4.** $N(0, 1)$, $N(0, 1.5)$, $N(0, 2)$, $N(0, 3)$(표준편차 기준)에서 각각 $n = 25$인 네 집단을 생성하여 Brown-Forsythe 검정을 적용하라. $W$와 $p$값을 보고하라. 그런 다음 각 집단을 $n = 100$으로 늘려 반복하라. 표본크기가 검정력에 어떤 영향을 주는가?

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import levene

    rng = np.random.default_rng(42)

    for n in [25, 100]:
        g1 = rng.normal(0, 1.0, n)
        g2 = rng.normal(0, 1.5, n)
        g3 = rng.normal(0, 2.0, n)
        g4 = rng.normal(0, 3.0, n)
        W, p = levene(g1, g2, g3, g4, center='median')
        print(f"n={n:3d}: W={W:.4f}, p={p:.6g}")
    ```

    출력:

    ```text
    n= 25: W=8.3766, p=5.29567e-05
    n=100: W=29.2350, p=4.28901e-17
    ```

    $n = 25$에서 이미 $p = 5.3 \times 10^{-5}$로 강하게 기각한다. $n = 100$에서는 $p = 4.3 \times 10^{-17}$이다.

    표준편차 비가 $3:1$, 곧 분산비가 $9:1$로 매우 크기 때문이다. 15.4절 연습문제 4에서 세 집단 총 30개로 4배 차이를 겨우 탐지했음을 떠올리면, 여기 네 집단 총 100개로 9배 차이를 탐지하는 것은 쉬운 편이다.

    **표본크기의 효과.** $W$가 $8.38$에서 $29.24$로 3.5배 커졌다. 표본크기가 네 배가 되면 검정통계량이 대략 네 배가 되어야 하는데(집단간 제곱합이 $n$에 비례), 3.5배는 그에 부합한다. 각 $S_i^2$의 표집변동이 줄어 참 분산 차이가 더 뚜렷해진다.

    **주의.** 이 예제는 단일 실행이므로 검정력이 아니라 하나의 표본에서의 결과이다. 검정력을 논하려면 여러 번 반복하여 기각 비율을 세어야 한다. $\square$

---

**연습문제 5.** Brown-Forsythe 검정은 중심화에 중앙값을 쓴다. 대안으로 **절사평균**(예: 10% 절사)을 쓸 수 있다. $z_{ij} = |x_{ij} - \bar{x}_{i,\text{trim}}|$을 계산하고 $z$ 값에 일원분산분석 $F$ 검정을 수행하는 판을 구현하라. $t(3)$ 분포 자료에서 중앙값 기반 판과 거짓 양성률을 비교하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy.stats import levene, trim_mean, f_oneway

    rng = np.random.default_rng(0)
    rej_med, rej_trim = 0, 0
    n_sims = 3000

    for _ in range(n_sims):
        g1 = rng.standard_t(3, 30)
        g2 = rng.standard_t(3, 30)
        g3 = rng.standard_t(3, 30)

        # Median centering (Brown-Forsythe)
        _, p_med = levene(g1, g2, g3, center='median')
        if p_med < 0.05:
            rej_med += 1

        # Trimmed-mean centering: deviations of the FULL sample
        # from the trimmed mean (data are NOT discarded)
        z_groups = [np.abs(g - trim_mean(g, 0.1)) for g in (g1, g2, g3)]
        _, p_trim = f_oneway(*z_groups)
        if p_trim < 0.05:
            rej_trim += 1

    print(f"Median centering FPR:       {rej_med/n_sims:.4f}")
    print(f"Trimmed mean centering FPR: {rej_trim/n_sims:.4f}")
    ```

    출력:

    ```text
    Median centering FPR:       0.0383
    Trimmed mean centering FPR: 0.0447
    ```

    $t(3)$ 분포에서 두 방법 모두 거짓 양성률을 0.05 근처로 조절한다. 절사평균 판이 $0.045$로 명목값에 조금 더 가깝다.

    대칭이면서 꼬리가 중간 정도인 자료에서는 절사평균이 중앙값보다 효율적인 위치 추정량이므로 대립가설 아래 검정력이 조금 높을 수 있다. 다만 극단적인 치우침에는 중앙값이 더 로버스트하다.

    !!! danger "이 구현과 SciPy의 `center='trimmed'`는 다르다"
        위 코드는 절사평균을 **중심으로만** 쓰고 편차는 **모든 관측값**에 대해 계산한다. 자료를 버리지 않는다.

        반면 **SciPy의 `stats.levene(..., center='trimmed')`는 자료 자체를 잘라낸다.** 각 집단에서 양 끝 10%를 제거한 뒤 남은 관측값으로만 검정을 수행한다. 15.5절 [Brown-Forsythe 검정](./brown_forsythe.md)에서 측정했듯 그 결과 **정규 자료에서도 크기가 0.12~0.19까지 부풀려진다**.

        | 방법 | 자료 처리 | 정규 자료 크기 |
        |---|---|---|
        | `center='median'` | 전부 사용 | 0.04 |
        | 위 코드(절사평균 중심) | 전부 사용 | 0.05 |
        | `center='trimmed'` | 양 끝 10% 제거 | **0.15** |

        가장 큰 편차들을 제거하면 F 통계량의 분모(집단내 제곱합)가 과도하게 줄어드는데, $F$ 기준분포가 이 선택 효과를 보정하지 않기 때문이다.

        절사평균 중심화를 쓰고 싶다면 위 코드처럼 **직접 구현**해야 한다. $\square$
