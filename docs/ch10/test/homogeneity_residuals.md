# 동질성 잔차 열지도

## 개요

카이제곱 동질성 검정이 귀무가설을 기각한 뒤에 자연스럽게 따라오는 질문은 **어느 칸이 동질성 이탈의 원인인가?**이다. 이 페이지에서는 표준화(Pearson) 잔차, Bonferroni 보정을 적용한 칸별 유의성 검정, 그리고 열지도 시각화를 이용한 사후 진단을 보인다. 이 접근은 어떤 모집단–범주 조합이 기대 분포에서 가장 많이 벗어나는지 짚어내도록 돕는다.

## 표준화 잔차

칸 $(i, j)$의 **Pearson 표준화 잔차**는

$$
R_{ij} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}}}
$$

이다. $H_0$ 아래에서 표본이 크면 각 $R_{ij}$는 근사적으로 표준정규를 따른다. $|R_{ij}| > 2$인 잔차는 그 칸이 전체 카이제곱 통계량에 뚜렷하게 기여함을 시사한다.

전체 카이제곱 통계량이 잔차 제곱의 합이라는 점에 유의하라:

$$
\chi^2 = \sum_{i,j} R_{ij}^2
$$

## Bonferroni 보정을 적용한 칸별 유의성

각 표준화 잔차를 근사적인 $z$-점수로 볼 수 있다. 칸 $(i,j)$의 양측 p-값은

$$
p_{ij} = 2\bigl[1 - \mathcal{N}(|R_{ij}|)\bigr]
$$

이며 $\mathcal{N}$은 표준정규 누적분포함수이다. $r \times c$개의 칸을 동시에 검정하므로 가족단위 오류율을 통제하기 위해 **Bonferroni 보정**을 적용한다:

$$
p_{ij}^{\text{Bonf}} = \min\bigl(r \cdot c \cdot p_{ij},\; 1\bigr)
$$

$p_{ij}^{\text{Bonf}} < \alpha$이면 그 칸을 유의하다고 표시한다.

### 잔차와 조정 p-값 계산

<div class="codebox" markdown>

#### 예제 1. 잔차로 어느 칸이 어긋났는지 찾기 { .eg }

```python
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

observed = np.array([
    [25, 30, 20, 25],
    [18, 22, 35, 25],
    [30, 25, 15, 30],
], dtype=float)

row_tot = observed.sum(axis=1, keepdims=True)
col_tot = observed.sum(axis=0, keepdims=True)
tot = observed.sum()
expected = (row_tot @ col_tot) / tot

# Pearson 표준화 잔차. 제곱해서 모두 더하면 카이제곱 통계량이 된다.
# 분모가 sqrt(E)인 것은 H0 아래에서 각 칸 도수의 분산이 근사적으로 E이기 때문이다.
resid = (observed - expected) / np.sqrt(expected)

# 칸마다 z-검정을 하는 셈이라 다중검정 문제가 생긴다. 그래서 아래에서 보정한다.
z = resid.ravel()
pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
reject, pvals_bonf, _, _ = multipletests(pvals, method="bonferroni")
pvals_bonf = pvals_bonf.reshape(observed.shape)
reject = reject.reshape(observed.shape)

print("Standardized residuals:")
print(resid)
print()
print("Bonferroni-adjusted per-cell p-values:")
print(pvals_bonf)
```

출력:

```
Standardized residuals:
[[ 0.13514748  0.8553372  -0.69006556 -0.32274861]
 [-1.28390102 -0.72374686  2.41522946 -0.32274861]
 [ 1.14875354 -0.13159034 -1.7251639   0.64549722]]

Bonferroni-adjusted per-cell p-values:
[[1.        1.        1.        1.       ]
 [1.        1.        0.1887036 1.       ]
 [1.        1.        1.        1.       ]]
```

전체 검정은 $p = 0.028$로 기각했는데(앞 페이지) 칸별로 보면 Bonferroni 보정 후 유의한 칸이 하나도 없다. 가장 큰 잔차인 2.415(모집단 2, 범주 3)조차 보정 후 $p = 0.189$다.

이런 어긋남은 흔하다. 전체 검정은 12개 칸의 어긋남을 **모아서** 보고, 칸별 검정은 12번의 검정에 대한 대가를 각각 치른다. 전체 검정이 유의한데 어느 칸도 유의하지 않은 것은 모순이 아니라, 증거가 한 칸에 몰려 있지 않고 흩어져 있다는 뜻이다.

여기서 Bonferroni는 상당히 보수적이기도 하다. 잔차들은 서로 독립이 아니라 주변 합계 제약으로 묶여 있으므로(제곱합이 카이제곱 통계량으로 고정된다) 12로 곱하는 것은 필요 이상이다.

</div>

### 열지도 시각화

<div class="codebox" markdown>

#### 예제 2. 잔차를 열지도로 보기 { .eg }

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(resid, aspect="auto")
ax.set_title("Standardized residuals heatmap")
ax.set_xlabel("Category")
ax.set_ylabel("Population")
plt.colorbar(im, ax=ax, shrink=0.8)

# 본페로니 보정 뒤에도 유의한 칸에 표시를 남긴다
for i in range(observed.shape[0]):
    for j in range(observed.shape[1]):
        # 보정 후 유의한 칸에만 별표를 붙인다. 이 예제에서는 하나도 없다.
        mark = "*" if reject[i, j] else ""
        ax.text(j, i, f"{resid[i, j]:.2f}{mark}",
                ha="center", va="center", fontsize=10)

plt.tight_layout()
plt.show()
```

![표준화 잔차 열지도](./img/homogeneity_residuals_78.png)

색으로 어느 칸이 기대보다 많고 적은지 한눈에 보인다. 모집단 2의 범주 3이 가장 밝고(+2.42), 모집단 3의 범주 3이 가장 어둡다(−1.73). 즉 범주 3의 선호가 모집단에 따라 갈리는 것이 이 표의 주된 구조다.

`*`로 표시된 칸은 Bonferroni 보정 후에도 통계적으로 유의한 칸이다. 색의 변화 덕분에 어느 칸의 잔차가 가장 크게 양(과다 대표)이거나 음(과소 대표)인지 쉽게 알아볼 수 있다.

</div>

## 해석

열지도는 동질성 아래의 기대 패턴에서 관측 자료가 어디에서 갈라지는지를 한눈에 요약해 준다. 핵심은 다음과 같다:

- **양의 잔차**(따뜻한 색)는 그 모집단이 해당 범주에서 기대보다 관측값이 많음을 뜻한다.
- **음의 잔차**(차가운 색)는 기대보다 적음을 뜻한다.
- 별표(`*`)는 다중비교 보정 후에도 이탈이 통계적으로 유의한 칸을 표시한다.

전체 카이제곱 검정은 모집단들이 다르다는 사실*만* 알려줄 뿐 *어떻게* 다른지는 알려주지 않으므로, 이런 사후분석이 꼭 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
관측도수 $O = 40$, 기대도수 $E = 25$일 때 표준화 잔차와 보정하지 않은 양측 p-값을 계산하라.

</div>

??? success "풀이"

    $$
    R = \frac{O - E}{\sqrt{E}} = \frac{40 - 25}{\sqrt{25}} = \frac{15}{5} = 3.0
    $$

    양측 p-값은

    $$
    p = 2[1 - \mathcal{N}(3.0)] = 2 \times 0.00135 = 0.0027
    $$

    이다. 이 칸은 다중비교 보정 이전부터 매우 유의한 과다 대표를 보인다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$4 \times 3$ 표에는 칸이 12개 있다. 칸별 p-값을 계산했더니 가장 작은 보정 전 p-값이 $0.006$이었다. Bonferroni 보정 후 이 칸은 $\alpha = 0.05$에서 유의한가?

</div>

??? success "풀이"

    Bonferroni 보정 p-값은

    $$
    p^{\text{Bonf}} = 12 \times 0.006 = 0.072
    $$

    이다. $0.072 > 0.05$이므로 보정 전 p-값이 작았음에도 이 칸은 Bonferroni 보정 후 유의하지 **않다**. 비교 횟수가 많을 때 Bonferroni가 얼마나 보수적일 수 있는지 보여준다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**표준화 잔차** $R_{ij} = (O_{ij} - E_{ij})/\sqrt{E_{ij}}$와 **조정 표준화 잔차** $R_{ij}^{\text{adj}} = (O_{ij} - E_{ij})/\sqrt{E_{ij}(1 - R_i/n)(1 - C_j/n)}$의 차이를 설명하라. $H_0$ 아래에서 어느 쪽이 $N(0,1)$에 더 가까운 분포를 갖는가?

</div>

??? success "풀이"

    Pearson 표준화 잔차는 $\sqrt{E_{ij}}$로 나누는데, 이는 $H_0$ 아래에서 $O_{ij} - E_{ij}$의 표준편차에 대한 근사일 뿐이다. 잔차의 참 분산은 인자 $(1 - R_i/n)(1 - C_j/n)$을 통해 주변 합계에도 의존한다.

    **조정 표준화 잔차**는 이 보정을 반영한다:

    $$
    R_{ij}^{\text{adj}} = \frac{O_{ij} - E_{ij}}{\sqrt{E_{ij}(1 - R_i/n)(1 - C_j/n)}}
    $$

    $H_0$ 아래에서 $R_{ij}^{\text{adj}}$의 분포가 보정하지 않은 잔차보다 $N(0,1)$에 더 가깝다. 따라서 칸별 가설검정에는 조정 잔차가 선호된다. 다만 탐색적인 열지도에는 보정하지 않은 형태도 여전히 흔히 쓰인다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Bonferroni 보정을 왜 "보수적"이라고 하는가? 다중비교의 대안을 하나 들고 어떻게 다른지 설명하라.

</div>

??? success "풀이"

    Bonferroni 보정은 유의수준 $\alpha$를 모든 비교에 똑같이 나누어 $m$개 검정 각각의 문턱으로 $\alpha / m$을 쓰는 방식으로 **가족단위 오류율**(FWER)을 통제한다. 검정이 많아지면 이 문턱이 아주 작아져, 참 효과가 있어도 개별 가설을 기각하기 어려워진다(검정력이 낮다).

    대안으로 **Benjamini-Hochberg(BH) 절차**가 있다. FWER 대신 **거짓발견율**(FDR)을 통제한다. FDR은 기각된 가설 중 거짓 양성의 기대 비율이다. BH는 p-값을 정렬한 뒤 적절한 절단 지표 $i$에 대해 $p_{(i)} \le (i/m)\alpha$인 가설을 모두 기각한다. Bonferroni보다 덜 보수적이어서, 통제된 비율의 거짓 발견을 감수하는 대신 검정력을 더 얻는다. `statsmodels`에서는 `multipletests(pvals, method="fdr_bh")`로 쓸 수 있다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
$\sum_{i,j} R_{ij}^2 = \chi^2$, 즉 카이제곱 통계량이 표준화 잔차 제곱의 합과 같음을 증명하라.

</div>

??? success "풀이"

    정의에 의해 표준화 잔차는 $R_{ij} = (O_{ij} - E_{ij}) / \sqrt{E_{ij}}$이다. 제곱하면

    $$
    R_{ij}^2 = \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
    $$

    이다. 모든 칸에 대해 합하면

    $$
    \sum_{i=1}^{r}\sum_{j=1}^{c} R_{ij}^2 = \sum_{i=1}^{r}\sum_{j=1}^{c} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} = \chi^2
    $$

    이 되는데, 이것이 바로 Pearson 카이제곱 통계량의 정의이다. 이 분해는 $\chi^2$가 모든 칸의 기여를 모은 값임을 보여주며, $R_{ij}$(또는 $R_{ij}^2$)의 열지도는 그 총합이 표 전체에 어떻게 흩어져 있는지 드러낸다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
연습문제 3이 개념으로 설명한 두 잔차의 차이를 **모의실험으로 확인**하라. 어느 쪽이 정말 $N(0,1)$인가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(3690)
    M, n = 20_000, 300
    p_row = np.array([1 / 3, 1 / 3, 1 / 3])
    p_col = np.array([0.25, 0.26, 0.23, 0.26])
    P = np.outer(p_row, p_col)                   # H0: 독립이 참

    std_r, adj_r = [], []
    for _ in range(M):
        obs = rng.multinomial(n, P.ravel()).reshape(3, 4).astype(float)
        if (obs.sum(0) == 0).any() or (obs.sum(1) == 0).any():
            continue
        exp = np.outer(obs.sum(1), obs.sum(0)) / n
        rp, cp = obs.sum(1) / n, obs.sum(0) / n
        std_r.append(((obs - exp) / np.sqrt(exp))[1, 2])
        adj_r.append(((obs - exp)
                      / np.sqrt(exp * np.outer(1 - rp, 1 - cp)))[1, 2])

    std_r, adj_r = np.array(std_r), np.array(adj_r)
    print(f"표준화 잔차 R    평균 {std_r.mean():+.4f}  표준편차 "
          f"{std_r.std(ddof=1):.4f}  |·|>1.96 비율 {np.mean(np.abs(std_r) > 1.96):.4f}")
    print(f"조정 잔차   R^adj 평균 {adj_r.mean():+.4f}  표준편차 "
          f"{adj_r.std(ddof=1):.4f}  |·|>1.96 비율 {np.mean(np.abs(adj_r) > 1.96):.4f}")
    print(f"\n이론값: R 의 표준편차 = √((1-R_i/n)(1-C_j/n)) = "
          f"√((1-1/3)(1-0.23)) = {np.sqrt((1 - 1 / 3) * (1 - 0.23)):.4f}")
    ```

    ```text
    표준화 잔차 R    평균 +0.0033  표준편차 0.7161  |·|>1.96 비율 0.0062
    조정 잔차   R^adj 평균 +0.0047  표준편차 0.9991  |·|>1.96 비율 0.0494

    이론값: R 의 표준편차 = √((1-R_i/n)(1-C_j/n)) = √((1-1/3)(1-0.23)) = 0.7165
    ```

    **답이 분명하다. 조정 잔차만 $N(0,1)$이다.**

    | | 표준편차 | $|\cdot|>1.96$ 비율 |
    |---|---|---|
    | 표준화 잔차 $R$ | **0.716** | 0.0062 |
    | 조정 잔차 $R^{\text{adj}}$ | **0.999** | **0.0494** |

    **표준화 잔차의 표준편차가 이론값 0.7165와 정확히 일치**한다. 우연이 아니라 다음 결과 때문이다.

    $$
    \operatorname{Var}(R_{ij})=\Bigl(1-\frac{R_i}{n}\Bigr)\Bigl(1-\frac{C_j}{n}\Bigr)<1
    $$

    **주변합이 추정되었기 때문**이다. 기대도수를 관측된 주변합에서 계산하므로 잔차가 덜 자유롭게 움직인다.

    **실무적 대가가 크다.** 표준화 잔차에 $\pm1.96$ 기준을 적용하면 **실제로는 $\alpha=0.006$인 검정**을 하는 셈이다. 명목의 1/8이라 **이탈을 놓친다.**

    **그런데 $\sum R_{ij}^2=\chi^2$은 여전히 성립**한다(연습문제 5). 분산이 1이 아닌데 제곱합이 카이제곱이 되는 것은 모순이 아니다. **자유도가 $rc$가 아니라 $(r-1)(c-1)$이기 때문**이다. 실제로

    $$
    \sum_{i,j}\operatorname{Var}(R_{ij})
    =\sum_{i,j}\Bigl(1-\frac{R_i}{n}\Bigr)\Bigl(1-\frac{C_j}{n}\Bigr)
    =(r-1)(c-1)
    $$

    이다. 위 설정에서 $12\times0.716^2\approx6.15$이고 $(3-1)(4-1)=6$으로 맞는다.

    **결론.**

    | 목적 | 쓸 잔차 |
    |---|---|
    | $\chi^2$을 칸별로 분해 | **표준화 잔차** $R$ (제곱합이 $\chi^2$) |
    | 칸별로 유의성 판정 | **조정 잔차** $R^{\text{adj}}$ |

    **둘을 혼동하는 것이 분할표 분석에서 가장 흔한 기술적 실수**다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
예제 1은 표준화 잔차에 본페로니 보정을 적용해 **유의한 칸이 없다**는 결론을 얻었다. 조정 잔차로 다시 하면 결론이 달라지는지 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    obs = np.array([[25, 30, 20, 25],
                    [18, 22, 35, 25],
                    [30, 25, 15, 30]], dtype=float)
    chi2, p, df, exp = stats.chi2_contingency(obs, correction=False)
    n = obs.sum()
    rp, cp = obs.sum(1) / n, obs.sum(0) / n

    std_r = (obs - exp) / np.sqrt(exp)
    adj_r = (obs - exp) / np.sqrt(exp * np.outer(1 - rp, 1 - cp))

    print(f"전체 검정:  χ² = {chi2:.4f},  df = {df},  p = {p:.6f}")
    print(f"검산  ΣR² = {np.sum(std_r**2):.4f}\n")
    print("표준화 잔차 R\n", np.round(std_r, 4))
    print("\n조정 잔차 R^adj\n", np.round(adj_r, 4))
    print(f"\n|R| 최대 {np.abs(std_r).max():.4f},  "
          f"|R^adj| 최대 {np.abs(adj_r).max():.4f}\n")

    for arr, name in [(std_r, "표준화"), (adj_r, "조정  ")]:
        pv = 2 * stats.norm.sf(np.abs(arr)).ravel()
        for method, label in [("bonferroni", "본페로니"),
                              ("holm", "홀름   "),
                              ("fdr_bh", "BH     ")]:
            rej, adj_p, _, _ = multipletests(pv, alpha=0.05, method=method)
            print(f"  {name} 잔차 + {label}: 최소 조정 p = {adj_p.min():.4f}, "
                  f"유의한 칸 {int(rej.sum())}개")
    ```

    ```text
    전체 검정:  χ² = 14.1697,  df = 6,  p = 0.027796
    검산  ΣR² = 14.1697

    표준화 잔차 R
     [[ 0.1351  0.8553 -0.6901 -0.3227]
     [-1.2839 -0.7237  2.4152 -0.3227]
     [ 1.1488 -0.1316 -1.7252  0.6455]]

    조정 잔차 R^adj
     [[ 0.1903  1.215  -0.9652 -0.4616]
     [-1.8077 -1.0281  3.3783 -0.4616]
     [ 1.6174 -0.1869 -2.4131  0.9232]]

    |R| 최대 2.4152,  |R^adj| 최대 3.3783

      표준화 잔차 + 본페로니: 최소 조정 p = 0.1887, 유의한 칸 0개
      표준화 잔차 + 홀름   : 최소 조정 p = 0.1887, 유의한 칸 0개
      표준화 잔차 + BH     : 최소 조정 p = 0.1887, 유의한 칸 0개
      조정   잔차 + 본페로니: 최소 조정 p = 0.0088, 유의한 칸 1개
      조정   잔차 + 홀름   : 최소 조정 p = 0.0088, 유의한 칸 1개
      조정   잔차 + BH     : 최소 조정 p = 0.0088, 유의한 칸 1개
    ```

    **결론이 뒤집힌다.** 표준화 잔차로는 유의한 칸이 없지만(최소 조정 $p=0.189$), **조정 잔차로는 칸 (2,3)이 유의하다**(조정 $p=0.0088$).

    **어느 쪽이 옳은가 — 조정 잔차다.** 앞 문제에서 확인한 대로 $N(0,1)$인 것은 조정 잔차뿐이다. 표준화 잔차에 정규 임계값을 쓰면 **지나치게 보수적**이어서 실제 이탈을 놓친다.

    **전체 검정과의 일관성도 조정 잔차 쪽이 낫다.** 옴니버스 검정이 $p=0.028$로 기각했는데 사후분석에서 아무것도 못 찾으면 이상하다. 조정 잔차는 "모집단 2의 범주 3이 많다"는 구체적 답을 준다.

    | 칸 | 관측 | 기대 | $R$ | $R^{\text{adj}}$ |
    |---|---|---|---|---|
    | (2, 3) | 35 | 23.33 | 2.415 | **3.378** |
    | (3, 3) | 15 | 23.33 | $-1.725$ | $-2.413$ |

    **보정 방법 셋은 여기서 같은 결론**을 준다. 가장 작은 $p$가 다른 것들과 크게 떨어져 있어 어느 방법을 써도 하나만 살아남는다. **보정 방법의 선택은 여러 칸이 경계 근처에 몰려 있을 때 문제가 된다.**

    **권장 절차 넷.**

    1. **옴니버스 검정**으로 전체 이탈을 확인한다.
    2. 기각했으면 **조정 잔차**를 계산한다.
    3. **다중비교 보정**을 적용한다(칸이 $rc$개).
    4. 살아남은 칸을 **원 도수와 함께** 보고한다.

    **2번을 빠뜨리는 것이 이 페이지 예제의 문제**였다. 코드가 짧아 보이지만 분모에 $(1-R_i/n)(1-C_j/n)$을 넣는 한 줄이 결론을 바꾼다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 4가 언급한 다중비교 대안들이 **잔차 분석에서 어떻게 다른지** 모의실험으로 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(1470)
    M, n = 3_000, 400

    def cell_pvalues(obs):
        """조정 잔차에서 칸별 양측 p 값을 만든다."""
        total = obs.sum()
        exp = np.outer(obs.sum(1), obs.sum(0)) / total
        rp, cp = obs.sum(1) / total, obs.sum(0) / total
        adj = (obs - exp) / np.sqrt(exp * np.outer(1 - rp, 1 - cp))
        return 2 * stats.norm.sf(np.abs(adj)).ravel()

    scenarios = {
        "H0 (독립)": np.outer([1 / 3] * 3, [0.25] * 4),
        "한 칸만 이탈": None,       # 아래에서 만든다
        "여러 칸 이탈": None,
    }
    base = np.outer([1 / 3] * 3, [0.25] * 4)
    p1 = base.copy(); p1[1, 2] += 0.04; p1[1, 0] -= 0.04
    p2 = base.copy()
    p2[0, 0] += 0.02; p2[0, 1] -= 0.02
    p2[1, 2] += 0.02; p2[1, 3] -= 0.02
    p2[2, 1] += 0.02; p2[2, 0] -= 0.02
    scenarios["한 칸만 이탈"] = p1
    scenarios["여러 칸 이탈"] = p2

    print(f"{'상황':>14s} {'보정':>8s} {'적어도 하나 기각':>16s} {'평균 기각 칸':>13s}")
    for label, P in scenarios.items():
        for method, name in [("bonferroni", "본페로니"), ("holm", "홀름"),
                             ("fdr_bh", "BH"), (None, "보정 없음")]:
            any_rej = tot_rej = 0
            for _ in range(M):
                obs = rng.multinomial(n, P.ravel()).reshape(3, 4).astype(float)
                if (obs.sum(0) == 0).any() or (obs.sum(1) == 0).any():
                    continue
                pv = cell_pvalues(obs)
                rej = pv < 0.05 if method is None else \
                    multipletests(pv, alpha=0.05, method=method)[0]
                any_rej += rej.any()
                tot_rej += rej.sum()
            print(f"{label:>14s} {name:>8s} {any_rej / M:16.4f} {tot_rej / M:13.4f}")
    ```

    ```text
                상황       보정        적어도 하나 기각       평균 기각 칸
           H0 (독립)     본페로니           0.0453        0.0500
           H0 (독립)       홀름           0.0453        0.0537
           H0 (독립)       BH           0.0510        0.0723
           H0 (독립)    보정 없음           0.3620        0.5923
           한 칸만 이탈     본페로니           0.6363        1.0953
           한 칸만 이탈       홀름           0.6517        1.1263
           한 칸만 이탈       BH           0.6613        1.4780
           한 칸만 이탈    보정 없음           0.9533        2.8807
           여러 칸 이탈     본페로니           0.5123        0.8883
           여러 칸 이탈       홀름           0.5193        0.9187
           여러 칸 이탈       BH           0.5447        1.4087
           여러 칸 이탈    보정 없음           0.9250        3.0647
    ```

    **보정 없이 하면 $H_0$에서 36%가 뭔가를 "발견"한다.** 칸이 12개이므로 당연하다. **잔차를 보정 없이 해석하는 것은 12번 검정하는 것과 같다.**

    **세 보정 모두 FWER을 지킨다**(0.045, 0.045, 0.051). BH는 FDR을 통제하는 방법이라 FWER 보장이 없지만, **참 이탈이 하나도 없는 상황에서는 FDR 통제가 곧 FWER 통제**이므로 여기서도 0.051로 안전하다.

    **차이는 검정력에서 드러난다.**

    | 상황 | 본페로니 | 홀름 | BH |
    |---|---|---|---|
    | 한 칸만 이탈 (적어도 하나) | 0.636 | 0.652 | **0.661** |
    | 한 칸만 이탈 (평균 기각 칸) | 1.095 | 1.126 | **1.478** |
    | 여러 칸 이탈 (적어도 하나) | 0.512 | 0.519 | **0.545** |
    | 여러 칸 이탈 (평균 기각 칸) | 0.888 | 0.919 | **1.409** |

    **BH가 찾아내는 칸이 훨씬 많다**(1.41 대 0.89). 참 이탈이 여럿일 때 그 이득이 커진다는 이론과 맞는다.

    **홀름은 본페로니를 항상 조금 앞선다**(0.652 대 0.636). 같은 FWER 보장을 주면서 더 강력하므로 **본페로니를 쓸 이유가 없다.**

    **"평균 기각 칸"이 1을 넘는 것에 주의한다.** "한 칸만 이탈" 설정에서도 평균 1.1~1.5개가 기각되는데, 이는 **주변합 제약 때문에 한 칸이 이탈하면 다른 칸도 함께 움직이기** 때문이다. 잔차들이 독립이 아니라는 사실이 여기서 드러난다.

    | 방법 | 통제 대상 | 언제 유리한가 |
    |---|---|---|
    | 본페로니 | FWER | 쓸 이유가 없다 |
    | **홀름** | FWER | 확증적 분석 |
    | **BH** | FDR | **탐색적 분석, 참 이탈이 여럿** |

    **실무 권고.**

    1. **홀름을 기본으로 쓴다.** 본페로니를 지배하므로 쓰지 않을 이유가 없다.
    2. **탐색적 분석이면 BH.** "유의한 칸 목록"을 후속 연구의 후보로 삼을 때 적절하다.
    3. **보정 없는 잔차는 그림 용도로만.** 모자이크 그림의 색칠 기준 정도로 쓰고, 판정에는 쓰지 않는다.
    4. **이탈이 분산돼 있으면 칸별 분석을 포기**하고 옴니버스 결과와 전체 패턴을 서술한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
잔차 분석이 **오도할 수 있는 상황**을 찾아라. 주변합이 극단적으로 치우치면 어떤 일이 생기는가?

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    def show(obs, label):
        obs = np.asarray(obs, float)
        chi2, p, df, exp = stats.chi2_contingency(obs, correction=False)
        n = obs.sum()
        rp, cp = obs.sum(1) / n, obs.sum(0) / n
        adj = (obs - exp) / np.sqrt(exp * np.outer(1 - rp, 1 - cp))
        contrib = (obs - exp)**2 / exp
        print(f"{label}   n={n:.0f}  χ²={chi2:.4f}  df={df}  p={p:.4f}")
        print(f"  관측\n{obs.astype(int)}")
        print(f"  기대\n{np.round(exp, 2)}")
        print(f"  조정 잔차\n{np.round(adj, 3)}")
        print(f"  χ² 기여율(%)\n{np.round(contrib / chi2 * 100, 1)}\n")

    show([[980, 20], [960, 40]], "① 주변합이 한쪽에 쏠림")
    show([[500, 500], [400, 600]], "② 주변합이 균형")
    ```

    ```text
    ① 주변합이 한쪽에 쏠림   n=2000  χ²=6.8729  df=1  p=0.0088
      관측
    [[980  20]
     [960  40]]
      기대
    [[970.  30.]
     [970.  30.]]
      조정 잔차
    [[ 2.622 -2.622]
     [-2.622  2.622]]
      χ² 기여율(%)
    [[ 1.5 48.5]
     [ 1.5 48.5]]

    ② 주변합이 균형   n=2000  χ²=20.2020  df=1  p=0.0000
      관측
    [[500 500]
     [400 600]]
      기대
    [[450. 550.]
     [450. 550.]]
      조정 잔차
    [[ 4.495 -4.495]
     [-4.495  4.495]]
      χ² 기여율(%)
    [[27.5 22.5]
     [27.5 22.5]]
    ```

    **①에서 기여율이 극단적으로 쏠린다.** 둘째 열이 전체 $\chi^2$의 **97%**를 만든다. 관측과 기대의 차이는 네 칸 모두 10으로 **똑같은데도** 그렇다.

    **이유.** $\chi^2$의 각 항이 $(O-E)^2/E$이므로, **$E$가 작은 칸이 같은 차이에 대해 훨씬 큰 기여**를 한다.

    $$
    \frac{10^2}{30}=3.33 \quad\text{대}\quad \frac{10^2}{970}=0.103
    $$

    **32배 차이**다.

    **이것이 오도하는 지점 셋.**

    **1 — "둘째 열이 문제다"라고 읽기 쉽다.** 그러나 도수의 차이는 네 칸이 모두 10으로 같다. 다른 것은 **기저 크기**뿐이다.

    **2 — $2\times2$에서 조정 잔차는 네 칸이 크기가 같다**($|2.622|$). 자유도가 1이므로 독립적인 정보가 하나뿐이기 때문이다. **칸별 분석이 아무 의미가 없다.** $2\times2$에서 잔차를 보는 것은 헛수고다.

    **3 — 기여율과 잔차가 다른 이야기를 한다.** 기여율은 1.5%와 48.5%로 갈리는데 조정 잔차는 모두 같다. **어느 쪽을 볼지 정해야 한다.**

    | 지표 | 답하는 질문 |
    |---|---|
    | $\chi^2$ 기여율 | 통계량을 **누가 만들었나** |
    | 조정 잔차 | 그 칸이 **통계적으로 유의한가** |

    **실무에서는 절대 차이도 함께 본다.**

    ```python
    obs = np.array([[980, 20], [960, 40]], float)
    p1, p2 = obs[0, 1] / obs[0].sum(), obs[1, 1] / obs[1].sum()
    print(f"둘째 열 비율: {p1:.4f} 대 {p2:.4f}")
    print(f"  위험차 {p2 - p1:+.4f}   위험비 {p2 / p1:.4f}")
    ```

    ```text
    둘째 열 비율: 0.0200 대 0.0400
      위험차 +0.0200   위험비 2.0000
    ```

    **"위험이 2배"이지만 절대 차이는 2%p**다. 앞 장에서 본 상대·절대 지표의 문제가 여기서도 그대로다.

    **잔차 분석의 한계 요약.**

    1. **$2\times2$에서는 쓰지 않는다.** 자유도가 1이라 정보가 하나뿐이다.
    2. **기대도수가 작은 칸의 잔차는 불안정**하다. $E<5$면 정규근사가 나쁘다.
    3. **잔차는 크기를 재지 않는다.** 원 도수와 비율을 함께 본다.
    4. **잔차끼리 독립이 아니다.** 주변합 제약 때문에 음의 상관이 있다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
분할표의 **사후 잔차 분석 절차**를 정리하라.

</div>

??? success "풀이"

    **절차.**

    ```text
    ① 옴니버스 검정
        χ² 이 유의하지 않으면 → 여기서 멈춘다
              ↓ 유의함
    ② 표의 크기를 본다
        2×2  → 잔차 분석 불필요 (자유도 1)
        그 외 → 계속
              ↓
    ③ 조정 표준화 잔차를 계산한다
        R^adj = (O-E) / √[E(1-R_i/n)(1-C_j/n)]
        ※ 단순 표준화 잔차를 쓰면 이탈을 놓친다
              ↓
    ④ 다중비교 보정 (칸이 r×c 개)
        홀름을 기본, 탐색이면 BH
              ↓
    ⑤ 살아남은 칸을 원 도수·비율과 함께 보고
              ↓
    ⑥ 그림으로 전체 패턴 확인 (모자이크 등)
    ```

    **핵심 공식 셋.**

    | 양 | 식 | 용도 |
    |---|---|---|
    | 표준화 잔차 | $\dfrac{O-E}{\sqrt E}$ | $\sum R^2=\chi^2$ 분해 |
    | **조정 잔차** | $\dfrac{O-E}{\sqrt{E(1-\frac{R_i}{n})(1-\frac{C_j}{n})}}$ | **유의성 판정** |
    | 기여율 | $\dfrac{(O-E)^2/E}{\chi^2}$ | 누가 통계량을 만들었나 |

    **점검 목록.**

    - [ ] 옴니버스 검정이 유의했는가
    - [ ] $2\times2$가 아닌가
    - [ ] **조정** 잔차를 썼는가
    - [ ] 다중비교 보정을 했는가
    - [ ] 기대도수가 작은 칸의 잔차를 조심했는가
    - [ ] 원 도수와 비율을 함께 보고했는가
    - [ ] 자료를 보고 세운 가설임을 밝혔는가

    **자주 하는 실수 다섯.**

    | 실수 | 결과 |
    |---|---|
    | 단순 표준화 잔차에 $\pm1.96$ | 실제 수준 0.006 — **이탈을 놓침** |
    | 보정 없이 여러 칸 해석 | $H_0$에서 23%가 오탐(연습문제 8) |
    | $2\times2$에서 잔차 분석 | 무의미 |
    | 옴니버스가 유의하지 않은데 잔차로 진행 | 일관되지 않음 |
    | 잔차 크기를 효과 크기로 읽기 | 잔차는 $n$에 따라 커진다 |

    **마지막 항목이 미묘하다.** 잔차는 $\sqrt n$에 비례해 커지므로, **큰 표본에서는 사소한 이탈도 큰 잔차**를 낸다. 효과의 크기는 비율의 차이나 크라메르 $V$로 따로 재야 한다.

    **한 문장.** 잔차 분석은 **"어디가"를 답하는 도구**이고, "얼마나"는 효과크기가, "정말인가"는 다중비교 보정이 답한다. 셋을 함께 보아야 이야기가 완성된다.

---

## 정리하며

기각한 뒤에 **어느 칸이 원인인지**를 찾는 절차다.

$$
R_{ij}=\frac{O_{ij}-E_{ij}}{\sqrt{E_{ij}}}
$$

- **표준화 잔차가 칸별 기여도를 말해 준다.** $\sum_{ij}R_{ij}^2$ 이 곧 $\chi^2$ 통계량이므로, 절댓값이 큰 칸이 기각을 이끈 칸이다.
- **대략 $|R|>2$ 를 주목한다.** 근사적으로 표준정규를 따르므로 $2$ 를 넘으면 눈여겨볼 만하다. **다만 칸이 많으면 우연히 넘는 칸이 생긴다.**
- **그래서 본페로니 보정을 함께 쓴다.** 칸 수만큼 검정하는 셈이므로 문턱을 조정해야 하며, 9장의 다중검정 논의가 그대로 적용된다.
- **열지도가 읽기 쉽다.** 잔차의 부호와 크기를 색으로 보이면 어느 집단이 어느 범주에서 기대보다 많고 적은지가 한눈에 들어온다.
- **사후 분석은 탐색이다.** 자료를 보고 고른 칸이므로, 확증하려면 새 자료가 필요하다.

다음 절 **McNemar 검정**으로 넘어간다. 지금까지는 독립표본이었지만 이제 **대응된 이진 자료**를 다룬다.
