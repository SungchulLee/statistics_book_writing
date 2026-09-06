# Bartlett 검정 비정규성 민감도

## 개요

분산 동일성에 대한 Bartlett 검정은 정규성 아래에서 일양최강력 검정이지만, 정규성 이탈에 민감하기로 악명 높다. 자료가 치우쳤거나 꼬리가 두꺼우면 Bartlett 검정의 거짓 양성률이 부풀려져, 명목 유의수준 $\alpha$가 시사하는 것보다 훨씬 자주 등분산 귀무가설을 기각한다. 이 페이지는 모의실험으로 이 민감도를 시연하고 대수정규 자료에서 Bartlett 검정을 Levene 검정 및 Fligner-Killeen 검정과 비교한다.

## 문제

$H_0: \sigma_1^2 = \sigma_2^2 = \cdots = \sigma_k^2$과 정규성 가정 아래에서 Bartlett 통계량 $T$는 근사적으로 $\chi^2(k-1)$ 분포를 갖는다. 자료가 비정규이면 초과첨도가 $S_i^2$의 변동을 부풀리므로 $T$의 실제 분포가 $\chi^2(k-1)$보다 확률적으로 커진다. 그 결과

$$
P(T > \chi^2_{1-\alpha}(k-1) \mid H_0, \text{비정규 자료}) \gg \alpha.
$$

## 모의실험 설계

이 효과를 보이려면

1. 모수가 같은 **대수정규** 분포에서 크기 $n$인 집단 $k = 3$개를 생성한다(등분산 귀무가설이 참으로 성립한다).
2. 수준 $\alpha = 0.05$에서 Bartlett, Levene, Fligner-Killeen 검정을 적용한다.
3. 여러 번 반복하여 기각 비율(경험적 거짓 양성률)을 기록한다.

올바르게 보정되어 있다면 모든 검정이 약 5%로 기각해야 한다.

## 코드

```python
import numpy as np
from scipy.stats import bartlett, levene, fligner

rng = np.random.default_rng(0)


def simulate_once(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    """Generate k groups from lognormal (skewed) or normal."""
    if skew:
        groups = [rng.lognormal(mean=0.0, sigma=s, size=n) for s in sigmas]
    else:
        groups = [rng.normal(loc=0.0, scale=s, size=n) for s in sigmas]
    return groups


def trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True):
    """Run all three tests on one simulated dataset."""
    g1, g2, g3 = simulate_once(n=n, sigmas=sigmas, skew=skew)
    _, p_bartlett = bartlett(g1, g2, g3)
    _, p_levene = levene(g1, g2, g3, center='mean')
    _, p_fligner = fligner(g1, g2, g3)
    return p_bartlett, p_levene, p_fligner


# Run simulation
n_sims, alpha = 5000, 0.05
ps_b, ps_lv, ps_fl = [], [], []

for _ in range(n_sims):
    p_b, p_l, p_f = trial(n=20, sigmas=(1.0, 1.0, 1.0), skew=True)
    ps_b.append(p_b)
    ps_lv.append(p_l)
    ps_fl.append(p_f)

fp_b = np.mean(np.array(ps_b) < alpha)
fp_lv = np.mean(np.array(ps_lv) < alpha)
fp_fl = np.mean(np.array(ps_fl) < alpha)

print("False-positive rates under skewed (lognormal) data:")
print(f"  Bartlett       : {fp_b:.4f}")
print(f"  Levene (mean)  : {fp_lv:.4f}")
print(f"  Fligner-Killeen: {fp_fl:.4f}")
```

출력:

```text
False-positive rates under skewed (lognormal) data:
  Bartlett       : 0.6748
  Levene (mean)  : 0.2470
  Fligner-Killeen: 0.1028
```

## 해석

| 검정 | 거짓 양성률 | 명목값의 배수 |
|---|---|---|
| Bartlett | **0.675** | 13.5 |
| Levene (평균) | **0.247** | 4.9 |
| Fligner-Killeen | **0.103** | 2.1 |

- **Bartlett 검정**의 거짓 양성률이 $0.675$이다. 명목 $\alpha = 0.05$의 **13.5배**이며, 등분산인 자료의 3분의 2에서 기각한다. 정규성이 의심스러울 때 등분산 진단 도구로 쓸 수 없다.
- **Levene 검정**(평균 중심)도 $0.247$로 크게 부풀려진다. 대수정규의 강한 치우침 때문에 평균 중심화가 무너진다.
- **Fligner-Killeen**도 $0.103$으로 명목값의 두 배이다. 순위 기반이라 다른 두 검정보다는 훨씬 낫지만 **완전히 통제되지는 않는다**.

!!! warning "여기서 통제되는 검정은 Brown-Forsythe뿐이다"
    위 코드는 세 검정만 비교하지만, 같은 조건에서 **Brown-Forsythe**(`center='median'`)를 추가하면 거짓 양성률이 $0.030$으로 유일하게 통제된다.

    $\text{Lognormal}(0,1)$은 왜도 $6.185$, 초과첨도 $110.9$로 극단적으로 치우쳐 있다. 이 정도 치우침에서는 순위 변환(Fligner-Killeen)조차 부족하고 **중앙값 중심화**가 결정적이다. 자세한 비교는 15.8절 [로버스트 분산 검정 비교](robust_tests_comparison.md)를 보라.

핵심은 Bartlett의 비정규성 민감도가 사소한 불편이 아니라 **근본적으로 잘못된 결론**으로 이어질 수 있다는 점이다.

## 연습문제

**연습문제 1.** `skew=False`(정규 자료)로 모의실험을 실행하여 세 검정의 거짓 양성률이 모두 0.05에 가까움을 확인하라.

??? success "연습문제 1 풀이"

    ```python
    import numpy as np
    from scipy.stats import bartlett, levene, fligner

    rng = np.random.default_rng(0)
    n_sims, alpha = 5000, 0.05
    rej = {"Bartlett": 0, "Levene": 0, "Fligner": 0}

    for _ in range(n_sims):
        g1 = rng.normal(0, 1, 20)
        g2 = rng.normal(0, 1, 20)
        g3 = rng.normal(0, 1, 20)
        _, p = bartlett(g1, g2, g3)
        if p < alpha: rej["Bartlett"] += 1
        _, p = levene(g1, g2, g3, center='mean')
        if p < alpha: rej["Levene"] += 1
        _, p = fligner(g1, g2, g3)
        if p < alpha: rej["Fligner"] += 1

    for name, count in rej.items():
        print(f"{name}: {count/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett: 0.0508
    Levene: 0.0556
    Fligner: 0.0364
    ```

    세 비율 모두 0.05 근처로, 정규성 아래에서 모든 검정이 올바르게 보정되어 있음을 확인해 준다.

    Bartlett($0.0508$)이 가장 정확하다. 15.4절 유도 페이지에서 본 Bartlett 보정인자 $C$가 유한표본 오차를 제거하기 때문이다. Fligner-Killeen($0.0364$)은 다소 보수적인데, 순위 변환이 정보를 버린 대가이다.

    **대조가 뚜렷하다.** 정규 자료에서 Bartlett의 크기는 $0.051$로 완벽하지만 대수정규 자료에서는 $0.675$이다. **가정이 성립할 때 가장 좋고 깨질 때 가장 나쁜 것**이 이 검정의 특징이다. $\square$

---

**연습문제 2.** 대수정규 분포를 유지한 채 표본크기를 $n = 100$으로 늘려라. Bartlett 검정의 부풀려진 거짓 양성률이 개선되는가? 이유를 설명하라.

??? success "연습문제 2 풀이"

    ```python
    import numpy as np
    from scipy.stats import bartlett

    rng = np.random.default_rng(0)
    n_sims = 5000
    rej = 0
    for _ in range(n_sims):
        g1 = rng.lognormal(0, 1, 100)
        g2 = rng.lognormal(0, 1, 100)
        g3 = rng.lognormal(0, 1, 100)
        _, p = bartlett(g1, g2, g3)
        if p < 0.05:
            rej += 1
    print(f"Bartlett FP rate (n=100): {rej/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett FP rate (n=100): 0.8134
    ```

    거짓 양성률이 개선되지 않을 뿐 아니라 **악화된다**. $n = 20$의 $0.675$에서 $n = 100$의 $0.813$으로 올라간다.

    | $n$ | 20 | 30 | 100 |
    |---|---|---|---|
    | Bartlett 거짓 양성률 | 0.675 | 0.732 | 0.813 |

    ($n = 30$ 값은 15.8절 [로버스트 분산 검정 비교](robust_tests_comparison.md)에서 얻었다.)

    표본이 커지면 분산 추정이 정밀해지지만 검정의 통계적 검정력도 커진다. 근본 문제는 Bartlett의 $\chi^2$ 기준분포가 비정규 자료에 대해 **틀렸다**는 것이며, 이는 표본크기로 개선되지 않는다. 편향이 유한표본 문제가 아니라 점근적이다.

    15.3절 연습문제 4의 유도로 설명하면, 실제 $\operatorname{Var}(\ln S^2)$와 명목값의 비율 $(\gamma_2+2)/2$가 $n$에 의존하지 않는다. $n$이 커지면 명목 분포는 좁아지는데 실제 분포는 같은 비율로 넓은 상태를 유지하므로 불일치의 상대적 크기가 커진다. $\square$

---

**연습문제 3.** 대수정규 대신 $t(3)$ 자료를 쓰도록 모의실험을 수정하라. Bartlett 검정이 여전히 부풀려지는가? 심각도가 대수정규의 경우와 어떻게 비교되는가?

??? success "연습문제 3 풀이"

    ```python
    import numpy as np
    from scipy.stats import bartlett, t as tdist

    rng = np.random.default_rng(0)
    n_sims = 5000
    rej = 0
    for _ in range(n_sims):
        g1 = tdist(df=3).rvs(20, random_state=rng)
        g2 = tdist(df=3).rvs(20, random_state=rng)
        g3 = tdist(df=3).rvs(20, random_state=rng)
        _, p = bartlett(g1, g2, g3)
        if p < 0.05:
            rej += 1
    print(f"Bartlett FP rate (t(3)): {rej/n_sims:.4f}")
    ```

    출력:

    ```text
    Bartlett FP rate (t(3)): 0.4076
    ```

    $t(3)$에서 $0.408$로 심하게 부풀려지지만, 대수정규의 $0.675$보다는 **덜하다**.

    | 분포 | 왜도 | 초과첨도 | Bartlett 크기 |
    |---|---|---|---|
    | $\mathcal{N}(0,1)$ | 0 | 0 | 0.051 |
    | $t(3)$ | 0 | $\infty$ | 0.408 |
    | $\text{Lognormal}(0,1)$ | 6.19 | 110.9 | 0.675 |

    **흥미로운 대비.** $t(3)$은 네 번째 적률이 아예 존재하지 않으므로 초과첨도가 $\infty$인데도, 유한한 $\gamma_2 = 110.9$인 대수정규보다 덜 나쁘다.

    이유는 **유한표본 거동**의 차이이다. $t(3)$은 대칭이고 극단값이 드물게 나타나므로, 대부분의 $n = 20$ 표본에서 표본첨도가 이론값보다 훨씬 작게 나온다. 반면 대수정규는 모든 적률이 존재하고 치우침이 일관되게 나타나므로 왜곡이 매번 발생한다.

    대수정규는 **치우침** 문제가 있고 $t(3)$은 **첨도** 문제가 있다. 둘 다 Bartlett의 정규성 가정을 위반하지만 방식이 다르며, 실무에서는 치우침이 더 큰 위협임을 보여준다. $\square$

---

**연습문제 4.** 모수 $\sigma$인 대수정규분포의 분산은 $(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}$이다. 모든 집단이 같은 $\sigma$와 $\mu$를 쓰면 모분산이 같다. $\mu = 0, \sigma = 1$인 대수정규의 모분산, 왜도, 첨도를 계산하라.

??? success "연습문제 4 풀이"

    $X \sim \text{Lognormal}(\mu, \sigma^2)$이고 $\mu = 0, \sigma = 1$일 때

    $$
    E[X] = e^{\mu + \sigma^2/2} = e^{1/2} = 1.6487.
    $$

    $$
    \operatorname{Var}(X) = (e^{\sigma^2} - 1)e^{2\mu + \sigma^2} = (e - 1)e^1 = 1.7183 \times 2.7183 = 4.6708.
    $$

    왜도는

    $$
    \gamma_1 = (e^{\sigma^2} + 2)\sqrt{e^{\sigma^2} - 1} = (e + 2)\sqrt{e - 1} = 4.7183 \times 1.3108 = 6.1849.
    $$

    초과첨도는

    $$
    \gamma_2 = e^{4\sigma^2} + 2e^{3\sigma^2} + 3e^{2\sigma^2} - 6 = e^4 + 2e^3 + 3e^2 - 6
    $$

    $$
    = 54.598 + 40.171 + 22.167 - 6 = 110.936.
    $$

    엄청난 왜도와 첨도가 Bartlett 검정이 그토록 극적으로 실패하는 이유를 설명한다. 자료가 이 정도로 비정규이면 $T$에 대한 $\chi^2$ 근사가 터무니없이 부정확하다.

    **정량화.** 15.4절 연습문제 2에서 $\operatorname{Var}(\ln S^2) \approx (\gamma_2+2)/n$임을 보았다. 여기서 $\gamma_2 = 110.9$이므로 팽창 인자가

    $$
    \frac{\gamma_2 + 2}{2} = \frac{112.9}{2} = 56.5
    $$

    이다. $\ln S_i^2$의 실제 분산이 정규 이론값의 **56배**이며, 표준편차로는 7.5배이다. Bartlett 통계량이 명목 $\chi^2_2$보다 그만큼 큰 값을 내므로 거의 항상 기각한다.

    $\sigma$에 따른 민감도도 극적이다.

    | $\sigma$ | 0.25 | 0.5 | 1.0 | 1.5 |
    |---|---|---|---|---|
    | $\gamma_2$ | 1.10 | 5.90 | 110.9 | 10,075 |

    $\sigma$가 0.25에서 1.5로 여섯 배가 될 때 $\gamma_2$가 만 배 가까이 커진다. $\gamma_2$의 지배항이 $e^{4\sigma^2}$이라 $\sigma^2$에 지수적으로 반응하기 때문이다. $\square$

---

**연습문제 5.** 관측된 첨도에 기반하여 임계값을 조정하는 Bartlett 검정의 보정판을 제안하라. 이 접근의 실현 가능성과 한계를 논하라.

??? success "연습문제 5 풀이"

    한 가지 접근은 Box(1953)의 보정으로, $\chi^2(k-1)$ 기준을 $\chi^2(k-1)/C$로 바꾼다.

    $$
    C = 1 + \frac{\hat{\kappa}}{2(k+1)}\left(\sum_{i=1}^k \frac{1}{n_i - 1} - \frac{1}{N-k}\right) + \text{고차항},
    $$

    여기서 $\hat{\kappa}$는 집단들에 걸쳐 합동한 공통 초과첨도의 추정값이다.

    ```python
    import numpy as np
    from scipy.stats import bartlett, kurtosis, chi2

    def bartlett_corrected(*groups):
        stat, _ = bartlett(*groups)
        k = len(groups)
        all_data = np.concatenate(groups)
        kappa_hat = kurtosis(all_data, fisher=True)
        ns = [len(g) for g in groups]
        N = sum(ns)
        correction = max(1 + kappa_hat / (2 * (k + 1)) *
                         (sum(1 / (n - 1) for n in ns) - 1 / (N - k)), 0.5)
        adjusted_stat = stat / correction
        return adjusted_stat, chi2(df=k - 1).sf(adjusted_stat)
    ```

    **실제로 작동하는가.** 위 구현으로 크기를 측정하면($k=3$, $n=20$, $R = 5000$)

    ```text
    Corrected Bartlett size, Normal data:      0.0510
    Corrected Bartlett size, Lognormal data:   0.6488
    ```

    !!! danger "이 보정은 문제를 거의 해결하지 못한다"
        대수정규 자료에서 보정 전 $0.675$가 보정 후 $0.649$로 **4% 개선에 그친다.** 여전히 명목값의 13배이다.

        정규 자료에서는 $0.051$로 크기를 망가뜨리지 않으므로 무해하기는 하다. 그러나 필요한 상황에서 무력하다.

        **왜 실패하는가.** 세 가지 이유가 겹친다.

        1. **보정 항이 너무 작다.** 괄호 안의 $\sum 1/(n_i-1) - 1/(N-k)$가 $n = 20$에서 $3/19 - 1/57 = 0.1404$에 불과하다. $\hat\kappa = 110$이어도 $C = 1 + \frac{110}{8}(0.1404) = 2.93$이다. 그런데 필요한 보정 배수는 $(\gamma_2+2)/2 = 56.5$이다. **20분의 1 수준이다.**

        2. **첨도 추정 자체가 불가능하다.** $\text{Lognormal}(0,1)$의 참 $\gamma_2 = 110.9$인데, $n = 60$짜리 합동표본의 표본첨도는 중앙값이 $7.1$이고 10~90백분위수가 $2.2$~$24.7$이다. 표본첨도가 참값을 심하게 과소추정한다(치우친 분포에서 표본 적률이 아래로 편향된다). 보정할 양을 알 수 없으므로 보정이 성립하지 않는다.

        3. **Box의 보정은 중간 정도의 비정규성을 위한 것이다.** 1차 근사이므로 $\gamma_2$가 작을 때만 유효하다. $\gamma_2 = 110$에서는 고차항이 지배한다.

    **한계 정리.**

    1. 첨도 추정 자체가 신뢰할 만하려면 큰 표본이 필요하고, 두꺼운 꼬리에서는 큰 표본으로도 불안정하다.
    2. 보정은 집단들이 하나의 공통 첨도를 갖는다고 가정한다. 집단마다 분포가 다르면 근사적일 뿐이다.
    3. 극단적으로 비정규인 자료에서는 어떤 단순 보정도 문제를 완전히 고치지 못한다.

    **결론: 보정하지 말고 검정을 바꾸라.** Brown-Forsythe나 Fligner-Killeen 같은 비모수 검정이 훨씬 신뢰할 만하다. 위 실험에서 Brown-Forsythe는 같은 대수정규 자료에서 $0.030$이었다. 보정된 Bartlett의 $0.649$와 비교할 수 없는 수준이다.

    이는 통계학의 일반적 교훈이기도 하다. **잘못된 모형을 보정으로 구제하려 하기보다 처음부터 가정이 약한 방법을 쓰는 편이 낫다.** $\square$
