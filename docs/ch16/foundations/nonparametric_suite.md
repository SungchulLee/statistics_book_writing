# 비모수 검정 모음

## 개요

이 페이지는 핵심 비모수 검정들을 하나의 참고 자료로 모아, 각 검정을 언제 쓰는지와
서로 어떻게 연결되는지를 정리한다. 비모수 방법은 분포 가정을 최소한으로만 한다 ---
대개 관측값이 독립이고 연속분포에서 나왔다는 것뿐이다. 정규성을 정당화할 수 없거나,
자료가 순서형이거나, 이상치가 우려될 때 없어서는 안 될 도구이다.

## 비모수 검정의 분류

아래 표는 주요 검정을 실험설계에 따라 정리한 것이다.

| 설계 | 검정 | 사용하는 정보 |
|---|---|---|
| 일표본 / 대응 | 부호검정 | 차이의 부호 |
| 일표본 / 대응 | Wilcoxon 부호순위 | 차이의 부호순위 |
| 독립 이표본 | Wilcoxon 순위합 | 합친 표본의 순위 |
| 독립 이표본 | Mann--Whitney $U$ | 쌍별 비교 (순위합과 동치) |
| $k$개 독립표본 | Kruskal--Wallis $H$ | 합친 표본의 순위 |
| $k$개 독립표본 | Mood 중앙값검정 | 전체 중앙값 위/아래 도수 |
| 무작위성 | Wald--Wolfowitz 런 검정 | 이진 수열의 런 |

## 검정력과 가정

각 검정은 일반성과 검정력 사이의 스펙트럼 위에 놓인다.

**부호검정.** 가정이 가장 약하다. 차이가 독립이고 연속이기만 하면 된다.
차이의 부호만 쓴다.

$$
Z_{\text{sign}} = \frac{2\,n_+ - n}{\sqrt{n}},
$$

여기서 $n_+$는 양의 차이 개수, $n$은 0이 아닌 차이의 개수이다. 사실상 이항검정이다.

**Wilcoxon 부호순위검정.** 차이의 분포가 중앙값을 중심으로 **대칭**이라는 가정을
추가한다. 부호에 더해 순위 크기까지 쓰므로 더 많은 정보를 회수한다.

$$
W^{+} = \sum_{i:\,D_i > 0} \operatorname{rank}(|D_i|),
$$

귀무 평균과 분산은

$$
\operatorname{E}[W^+] = \frac{n'(n'+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n'(n'+1)(2n'+1)}{24}.
$$

**순위합 / Mann--Whitney.** 독립인 두 표본에서 $N$개 관측값 전체에 순위를 매긴다.
통계량 $U$는 승수를 센다.

$$
U = \sum_{i=1}^{m}\sum_{j=1}^{n} \mathbf{1}[X_i > Y_j].
$$

귀무가설 아래에서 $\operatorname{E}[U] = mn/2$이다.

**Kruskal--Wallis $H$.** 순위합을 $k$개 집단으로 일반화한다.

$$
H = \frac{12}{N(N+1)}\sum_{j=1}^{k}\frac{R_j^2}{n_j} - 3(N+1) \;\sim\; \chi^2_{k-1}.
$$

**Mood 중앙값검정.** 각 관측값을 "전체 중앙값 위/아래"라는 이진 지시값으로만 쓰는,
다집단 검정 중 가장 로버스트한 방법이다.

## 선택 지침

다음 흐름이 선택 과정을 요약한다.

1. **표본이 대응인가 독립인가?**
      - 대응 $\to$ 2단계로.
      - 독립 $\to$ 3단계로.
2. **대응: 차이의 분포가 대칭인가?**
      - 예 $\to$ Wilcoxon 부호순위 (더 강력).
      - 아니오 또는 모름 $\to$ 부호검정 (더 안전).
3. **독립: 집단이 몇 개인가?**
      - 두 집단 $\to$ Mann--Whitney $U$ 또는 Wilcoxon 순위합.
      - 셋 이상 $\to$ Kruskal--Wallis $H$ (또는 추가 로버스트성이 필요하면 Mood 중앙값검정).

## 예제: 여러 검정을 함께 적용하기

아래 코드는 같은 대응자료에 여러 검정을 적용하여 $p$값을 직접 비교한다.

```python
import numpy as np
from scipy import stats

paired_data = np.array([
    [93, 76], [70, 72], [81, 75], [65, 68], [79, 65],
    [54, 54], [94, 88], [91, 81], [77, 65], [65, 57],
    [95, 86], [89, 87], [78, 78], [80, 77], [76, 76]
])

post, pre = paired_data[:, 0], paired_data[:, 1]

# Sign test (normal approximation)
diffs = post - pre
nonzero = diffs[diffs != 0]
n_plus = (nonzero > 0).sum()
n = len(nonzero)
z_sign = (2 * n_plus - n) / np.sqrt(n)
p_sign = 2 * stats.norm.cdf(-abs(z_sign))
print(f"Sign test:          Z = {z_sign:.4f}, p = {p_sign:.4f}")
# Sign test:          Z = 2.3094, p = 0.0209

# Wilcoxon signed-rank test
stat_sr, p_sr = stats.wilcoxon(post, pre, alternative="two-sided",
                                method="approx", zero_method="pratt")
print(f"Signed-rank test:   W = {stat_sr}, p = {p_sr:.4f}")
# Signed-rank test:   W = 11.0, p = 0.0086

# Wilcoxon rank-sum test -- WRONG for this data, shown for contrast only
stat_rs, p_rs = stats.ranksums(post, pre)
print(f"Rank-sum test:      Z = {stat_rs:.4f}, p = {p_rs:.4f}")
# Rank-sum test:      Z = 1.4725, p = 0.1409
```

출력:

```
Sign test:          Z = 2.3094, p = 0.0209
Signed-rank test:   W = 11.0, p = 0.0086
Rank-sum test:      Z = 1.4725, p = 0.1409
```

| 검정 | $p$값 | 적절한가 |
|:---|---:|:---|
| Wilcoxon 부호순위 | $0.0086$ | 예 |
| 부호검정 | $0.0209$ | 예 (더 보수적) |
| Wilcoxon 순위합 | $0.1409$ | **아니오** |

대응자료에 적절한 두 검정 중에서는 부호순위검정이 정보를 더 많이 쓰므로 $p$값이 작다.

!!! danger "순위합검정의 $p$값은 비교 대상이 아니다"
    순위합검정의 $0.1409$가 가장 크다고 해서 "가장 보수적인 검정"이라고 읽으면
    안 된다. 이 검정은 **대응 구조를 버렸으므로 애초에 잘못 적용된 것**이다.
    학생 15명의 처치 전 점수와 후 점수는 독립이 아니다.

    대응자료를 독립표본으로 분석하면 개인차가 모두 잡음으로 들어가 검정력을 크게
    잃는다. 여기서는 그 손실이 $p$값 16배로 나타났다.

## 해석

- **비모수 $\neq$ 무가정.** 모든 검정이 여전히 독립성을 요구한다. 부호순위검정은
  대칭성을 추가로 요구한다. Kruskal--Wallis는 집단 분포의 모양이 같지 않으면
  위치 이동만이 아니라 임의의 분포적 차이를 검정한다.
- **점근상대효율(ARE).** 정규성 아래에서 Wilcoxon 부호순위검정의 대응 $t$ 검정 대비
  ARE는 $3/\pi \approx 0.955$로 검정력 손실이 매우 작다. 두꺼운 꼬리 분포에서는
  ARE가 $1$을 넘는다.
- **다중비교.** Kruskal--Wallis가 $H_0$을 기각하면 사후 쌍별 검정(예: Dunn 검정)에
  Bonferroni나 Holm 보정을 적용하여 어느 집단이 다른지 밝힌다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 한 연구자가 대응 관측값 12개를 갖고 있는데 차이가 대칭인지 확신할 수
없다. 부호검정과 Wilcoxon 부호순위검정 중 무엇을 써야 하는가? 근거와 맞교환을 설명하라.

</div>

??? success "풀이"

    **부호검정**을 써야 한다. Wilcoxon 부호순위검정은 $H_0$ 아래에서 차이의 분포가
    중앙값을 중심으로 대칭임을 요구한다. 이 가정이 깨지면 $W^+$의 귀무분포가 틀리고
    제1종 오류가 부풀려질 수 있다.

    맞교환은 **검정력**이다. 부호검정은 차이의 부호만 쓰고 크기 정보를 버리므로,
    대칭성이 실제로 성립한다면 검정력이 낮다. 그러나 이 경우에는 연구자가 대칭성을
    확인할 수 없으므로 제1종 오류를 통제하는 안전성이 검정력보다 중요하다.

    표본이 더 크다면 차이의 히스토그램으로 대칭성을 평가한 뒤 부호순위검정으로
    바꿀 수 있다. 다만 $n = 12$로는 히스토그램으로 대칭성을 판정하기 어렵다.
    맥락 지식(예: "이 측정값은 바닥효과가 있어 차이가 오른쪽으로 치우친다")이
    있다면 그것이 더 믿을 만한 근거이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** 정규성 아래에서 부호검정의 일표본 $t$ 검정 대비 점근상대효율이
$2/\pi \approx 0.637$임을 보여라.

</div>

??? success "풀이"

    ARE는 두 검정의 **효능**(efficacy) 제곱의 비이다. 검정통계량 $T_n$의 효능은

    $$
    c = \lim_{n \to \infty} \frac{\frac{d}{d\delta}\operatorname{E}_\delta[T_n]\big|_{\delta=0}}{\sqrt{n \operatorname{Var}_0(T_n)}}
    $$

    으로 정의된다.

    **$t$ 검정.** $T_n = \bar{D}$이고 $\operatorname{E}_\delta[\bar D] = \delta$이므로
    미분값이 $1$이다. $\operatorname{Var}_0(\bar D) = \sigma^2/n$이므로

    $$
    c_t = \frac{1}{\sqrt{n \cdot \sigma^2/n}} = \frac{1}{\sigma}.
    $$

    **부호검정.** $T_n = \hat{p} = n_+/n$이라 하자. $D_i \sim \mathcal{N}(\delta, \sigma^2)$이면

    $$
    \operatorname{E}_\delta[\hat p] = P(D_i > 0) = \Phi\!\left(\frac{\delta}{\sigma}\right)
    $$

    이고, $\delta = 0$에서 미분하면

    $$
    \frac{d}{d\delta}\Phi\!\left(\frac{\delta}{\sigma}\right)\bigg|_{\delta=0}
    = \frac{1}{\sigma}\varphi(0) = \frac{1}{\sigma\sqrt{2\pi}} = f(0)
    $$

    이다($f$는 $D_i$의 밀도함수). $\operatorname{Var}_0(\hat p) = 1/(4n)$이므로

    $$
    c_{\text{sign}} = \frac{f(0)}{\sqrt{n \cdot 1/(4n)}} = 2 f(0) = \frac{2}{\sigma\sqrt{2\pi}}.
    $$

    **ARE.**

    $$
    \text{ARE}(\text{부호}, t) = \left(\frac{c_{\text{sign}}}{c_t}\right)^2
    = \left(\frac{2/(\sigma\sqrt{2\pi})}{1/\sigma}\right)^2
    = \frac{4}{2\pi} = \frac{2}{\pi} \approx 0.637. \quad \square
    $$

    일반적으로 밀도 $f$와 표준편차 $\sigma$를 갖는 대칭분포에서
    $\text{ARE}(\text{부호}, t) = 4\sigma^2 f(0)^2$이다. 이 공식이 유용한 것은
    다른 분포에서도 바로 계산할 수 있기 때문이다.

    | 분포 | $\sigma^2$ | $f(0)$ | ARE |
    |:---|:---:|:---:|---:|
    | Normal | $1$ | $1/\sqrt{2\pi} = 0.3989$ | $0.637$ |
    | Laplace ($\text{Var} = 1$) | $1$ | $1/\sqrt{2} = 0.7071$ | $2.000$ |
    | Uniform$(-\sqrt3, \sqrt3)$ | $1$ | $1/(2\sqrt3) = 0.2887$ | $0.333$ |

    Laplace에서 부호검정이 $t$ 검정보다 **두 배 효율적**임에 주목하라. 꼬리가
    두꺼울수록 $f(0)$이 커지고 부호검정이 유리해진다.

<div class="drillbox" markdown>

**연습문제 3.** 독립인 세 집단이 다음 자료를 냈다.

- 집단 A: $5, 8, 12, 15$
- 집단 B: $7, 11, 14, 18, 20$
- 집단 C: $3, 6, 9$

Kruskal--Wallis 검정과 Mood 중앙값검정을 파이썬으로 수행하고 $p$값을 비교하라.

</div>

??? success "풀이"

    ```python
    from scipy import stats

    a = [5, 8, 12, 15]
    b = [7, 11, 14, 18, 20]
    c = [3, 6, 9]

    print(stats.kruskal(a, b, c))
    # KruskalResult(statistic=4.0295, pvalue=0.13335)

    r = stats.median_test(a, b, c)
    print(r.statistic, r.pvalue, r.median)
    print(r.table)
    # 4.8  0.09072  10.0
    # [[2 4 0]
    #  [2 1 3]]
    ```

    출력:

    ```
    KruskalResult(statistic=4.029487179487184, pvalue=0.13335459246779172)
    4.8 0.0907179532894125 10.0
    [[2 4 0]
     [2 1 3]]
    ```

    | 검정 | 통계량 | $p$값 |
    |:---|---:|---:|
    | Kruskal--Wallis | $H = 4.029$ | $0.1334$ |
    | Mood 중앙값 | $\chi^2 = 4.800$ | $0.0907$ |

    **놀랍게도 Mood 중앙값검정의 $p$값이 더 작다.** "Kruskal--Wallis가 언제나 더
    강력하다"는 통념이 이 자료에서는 성립하지 않는다.

    이유는 이 자료의 구조에 있다. 전체 중앙값은 $10$이고, 집단 C는 값이 $3, 6, 9$로
    **전부 중앙값 아래**이다. Mood 검정에게 이는 매우 강한 신호이다($0$ 대 $3$).
    반면 Kruskal--Wallis는 집단 C의 순위가 $1, 2, 4$로 낮긴 하지만 집단 A의
    $3, 5, 8, 10$과 상당히 겹친다는 것도 함께 본다.

    더 근본적으로, 표본이 $4, 5, 3$으로 매우 작아 두 검정 모두 $\chi^2$ 근사가
    믿을 만하지 않다. 이 크기에서 $p$값의 순서를 두고 검정력을 논하는 것은
    무리이다.

    **교훈:** "검정 A가 검정 B보다 강력하다"는 진술은 언제나 **분포와 대립가설의
    종류에 대한 평균**을 두고 하는 말이다. 개별 자료에서는 어느 쪽이든 이길 수 있다.
    자료를 본 뒤 $p$값이 작은 검정을 고르면 실제 유의수준이 명목값을 크게 넘는다.
    $\square$

<div class="drillbox" markdown>

**연습문제 4.** Kruskal--Wallis가 단순한 중앙값 검정이 아닌 이유를 설명하라.
실제로 무엇을 검정하며, 어떤 추가 가정 아래에서 위치이동 검정이 되는가?

</div>

??? success "풀이"

    Kruskal--Wallis 검정의 귀무가설은 $H_0{:}\; F_1 = F_2 = \cdots = F_k$,
    즉 모든 집단이 같은 분포에서 왔다는 것이다. 이는 **분포의 동일성**에 대한
    검정이지 중앙값이 같은지에 대한 검정이 아니다. $H_0$을 기각한다는 것은
    집단들이 위치, 척도, 모양 또는 그 조합에서 다르다는 뜻일 수 있다.

    분포들이 위치만 다르다는 **위치이동 가정**($F_j(x) = F(x - \mu_j)$, $F$는 공통 모양)을
    추가하면 분포가 다를 수 있는 유일한 통로가 평균/중앙값이 되므로, 이 경우
    Kruskal--Wallis가 중앙값(또는 평균) 동일성 검정이 된다.

    위치이동 가정이 없으면 유의한 Kruskal--Wallis 결과가 중심이 아니라 산포의
    차이를 반영할 수 있고, 이를 중앙값 검정으로 해석하는 연구자를 오도한다.

    [Kruskal-Wallis](../multi_group_nonparametric/kruskal_wallis.md) 연습문제 3에서
    이를 정량적으로 확인했다. 중앙값이 모두 정확히 0이지만 왜도가 다른 세 집단에서
    $n = 100$일 때 기각률이 $0.614$까지 올라간다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** Kruskal--Wallis 결과가 유의한 뒤 어느 쌍이 다른지 알고 싶다.
Dunn 검정을 기술하고 다중비교 보정이 왜 필요한지 설명하라.

</div>

??? success "풀이"

    **Dunn 검정**은 원래의 Kruskal--Wallis 합친 순위에서 얻은 평균순위를 이용해
    $\binom{k}{2}$개의 모든 집단 쌍을 비교한다. 평균순위가 $\bar{R}_i$, $\bar{R}_j$인
    집단 $i$와 $j$에 대해

    $$
    Z_{ij} = \frac{\bar{R}_i - \bar{R}_j}{\sqrt{\frac{N(N+1)}{12}\left(\frac{1}{n_i} + \frac{1}{n_j}\right)}}
    $$

    이며 $N$은 전체 표본크기이다. 각 $Z_{ij}$를 표준정규분포와 비교한다.

    **다중비교 보정**(Bonferroni, Holm, Benjamini--Hochberg 등)이 필요한 이유는
    $\binom{k}{2}$개의 검정을 수행하면 집단별 오류율이 부풀려지기 때문이다.
    $m$개의 쌍을 각각 유의수준 $\alpha$로 검정하면 검정들이 독립일 때 적어도 하나가
    거짓 기각될 확률이 $1 - (1 - \alpha)^m$까지 올라간다.

    | $k$ | $m = \binom{k}{2}$ | 보정 없는 FWER |
    |---:|---:|---:|
    | 3 | 3 | $0.143$ |
    | 5 | 10 | $0.401$ |
    | 8 | 28 | $0.762$ |

    $k = 8$이면 모든 집단이 동일해도 76%의 확률로 "유의한" 쌍을 하나 이상 찾게 된다.

    Bonferroni 보정은 각 쌍별 검정에 $\alpha/m$을 쓰므로 전체 집단별 오류율을
    $\alpha$ 이하로 보장한다. Holm 절차는 FWER을 똑같이 통제하면서 균일하게 더
    강력하므로 대체로 더 낫다.

    !!! note "Dunn 검정의 쌍은 독립이 아니다"
        위 표의 $1 - (1-\alpha)^m$은 검정들이 **독립일 때**의 값이다. Dunn 검정의
        쌍별 비교는 같은 순위를 공유하므로 서로 상관되어 있고, 실제 FWER은 이보다
        낮다. 그럼에도 보정 없이는 $\alpha$를 넘으므로 보정이 필요하다.
        Bonferroni는 독립 여부와 무관하게 유효하다는 것이 그 장점이다. $\square$
