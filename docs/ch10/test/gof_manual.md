# 카이제곱 적합도 검정 (수동)

## 개요

이 페이지에서는 편의 함수 `scipy.stats.chisquare`에 의존하지 않고 NumPy와 SciPy로 카이제곱 적합도 검정통계량과 p-값을 **직접** 계산하는 방법을 보인다. 예제는 세 결과가 똑같이 그럴듯한 가위바위보 자료를 쓴다. p-값에 해당하는 꼬리 확률을 강조한 카이제곱 분포 시각화도 포함한다.

## 가설

- **귀무가설** ($H_0$): 결과가 모든 범주에 걸쳐 균등(같은) 분포를 따른다.
- **대립가설** ($H_A$): 결과가 균등분포를 따르지 않는다.

## 검정통계량

범주가 $k$개이고 관측도수가 $O_i$, 기대도수가 $E_i$일 때 카이제곱 통계량은

$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}
$$

이다. 모든 기대도수가 적어도 5 이상이면 $H_0$ 아래에서 이 통계량은 근사적으로 자유도

$$
\text{df} = k - 1
$$

인 $\chi^2$ 분포를 따른다.

아래 스크립트는 통계량을 처음부터 계산하고 기각역을 색칠한 $\chi^2$ 밀도를 그린다.

<div class="codebox" markdown>

### 예제 1. 적합도 검정 손계산 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

observed_counts = np.array([4, 13, 7])
# H0가 균등분포이므로 기대도수는 모두 n/k, 즉 관측도수의 평균과 같다.
expected_counts = np.ones(3) * observed_counts.mean()
degrees_of_freedom = observed_counts.shape[0] - 1   # 총합이 고정이라 하나를 잃는다

# 분모가 관측도수가 아니라 **기대도수**인 것에 주의하라.
# H0 아래에서 각 칸 도수의 분산이 근사적으로 E_i이기 때문이다(연습문제 4).
chi_square_statistic = np.sum(
    (observed_counts - expected_counts) ** 2 / expected_counts
)
# 카이제곱 적합도 검정은 언제나 우측검정이므로 sf(위쪽 꼬리)를 쓴다.
p_value = stats.chi2(degrees_of_freedom).sf(chi_square_statistic)

print(f"Chi-square Statistic = {chi_square_statistic:.4f}")
print(f"p-value = {p_value:.4f}")
```

출력:

```
Chi-square Statistic = 5.2500
p-value = 0.0724
```

기대도수가 모두 8로 5를 넘으므로 카이제곱 근사를 써도 되는 상황이다.

**단계별 설명:**

1. **관측도수**를 NumPy 배열 $[4,\;13,\;7]$로 저장한다.
2. $H_0$ 아래의 **기대도수**는 각각 $24 / 3 = 8$이다.
3. **검정통계량**을 성분별로 계산한다:

$$
\chi^2 = \frac{(4-8)^2}{8} + \frac{(13-8)^2}{8} + \frac{(7-8)^2}{8} = 2 + 3.125 + 0.125 = 5.25
$$

4. **p-값**은 $\chi^2(2)$ 분포의 생존함수(위쪽 꼬리 확률)를 $5.25$에서 평가한 값이다.

</div>

### 시각화

<div class="codebox" markdown>

#### 예제 2. 검정 결과를 그림으로 { .eg }

```python
fig, ax = plt.subplots(figsize=(12, 4))

# 왼쪽 구간 — 기각하지 않는 쪽
x_left = np.linspace(0, chi_square_statistic, 100)
y_left = stats.chi2(degrees_of_freedom).pdf(x_left)
ax.plot(x_left, y_left, linewidth=3)
x_fill_left = np.concatenate([[0], x_left, [chi_square_statistic], [0]])
y_fill_left = np.concatenate([[0], y_left, [0], [0]])
ax.fill(x_fill_left, y_fill_left, alpha=0.1)

# 오른쪽 꼬리 — 기각역
x_right = np.linspace(chi_square_statistic, 20, 100)
y_right = stats.chi2(degrees_of_freedom).pdf(x_right)
ax.plot(x_right, y_right, linewidth=3)
x_fill_right = np.concatenate(
    [[chi_square_statistic], x_right, [20], [chi_square_statistic]]
)
y_fill_right = np.concatenate([[0], y_right, [0], [0]])
ax.fill(x_fill_right, y_fill_right, alpha=0.1)

# p-값을 표시한다
ax.annotate(
    f"p-value = {p_value:.02%}",
    xy=((12.5 + 15.0) / 2, 0.01),
    xytext=(16.5, 0.10),
    fontsize=15,
    arrowprops=dict(width=0.2, headwidth=8),
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["bottom"].set_position("zero")
ax.spines["left"].set_position("zero")
plt.tight_layout()
plt.show()
```

![카이제곱 분포와 p-값](./img/gof_manual_66.png)

색칠된 오른쪽 꼬리는 $H_0$ 아래에서 $5.25$만큼 또는 그보다 극단적인 $\chi^2$ 값을 관측할 확률을 나타낸다.

</div>

## 해석

$\chi^2 = 5.25$, $\text{df} = 2$이므로 p-값은 약 $0.0724$이다. 통상적인 유의수준 $\alpha = 0.05$에서 $H_0$을 **기각하지 못한다**. 결과가 균등분포에서 벗어난다고 결론지을 만한 증거가 충분하지 않다.

기억할 점:

- 수동 계산은 `scipy.stats.chisquare`가 내부에서 무엇을 하는지 드러낸다.
- 시각화는 검정통계량, $\chi^2$ 분포, p-값 사이의 연결을 분명히 해 준다.
- 카이제곱 근사를 믿기 전에 기대도수 조건($E_i \ge 5$)을 반드시 확인하라.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
육면체 주사위를 120번 굴렸다. 눈 1부터 6까지의 관측도수가 $[15, 22, 18, 25, 20, 20]$이다. 카이제곱 통계량을 손으로 계산하고 자유도를 구하라.

</div>

??? success "풀이"

    $H_0$ 아래에서 각 눈의 기대도수는 $E_i = 120/6 = 20$이다. 통계량은

    $$
    \chi^2 = \frac{(15-20)^2}{20} + \frac{(22-20)^2}{20} + \frac{(18-20)^2}{20} + \frac{(25-20)^2}{20} + \frac{(20-20)^2}{20} + \frac{(20-20)^2}{20}
    $$

    $$
    = \frac{25}{20} + \frac{4}{20} + \frac{4}{20} + \frac{25}{20} + 0 + 0 = 1.25 + 0.20 + 0.20 + 1.25 + 0 + 0 = 2.90
    $$

    자유도: $\text{df} = 6 - 1 = 5$. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
p-값을 계산할 때 `stats.chi2(df).cdf(statistic)`이 아니라 `stats.chi2(df).sf(statistic)`을 쓰는 이유를 설명하라.

</div>

??? success "풀이"

    카이제곱 적합도 검정은 언제나 **우측**검정이다. 통계량이 크면 $H_0$으로부터의 이탈을 뜻한다. 따라서 p-값은 위쪽 꼬리 확률인 $P(\chi^2 \ge \text{관측 통계량})$이다. 생존함수 `sf`는 $F$가 누적분포함수일 때 $1 - F(x)$를 계산하므로 정확히 이 위쪽 꼬리 넓이를 준다. `cdf`를 그대로 쓰면 우리가 필요한 것의 여집합인 왼쪽 꼬리 확률을 얻게 된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
기대 비율이 균등하지 않고 $(p_1, p_2, p_3) = (0.5, 0.3, 0.2)$이며 관측값이 총 24개라고 하자. 같은 관측도수 $[4, 13, 7]$에 대해 기대도수와 카이제곱 통계량을 다시 계산하라.

</div>

??? success "풀이"

    기대도수: $E_1 = 0.5 \times 24 = 12$, $E_2 = 0.3 \times 24 = 7.2$, $E_3 = 0.2 \times 24 = 4.8$.

    $$
    \chi^2 = \frac{(4-12)^2}{12} + \frac{(13-7.2)^2}{7.2} + \frac{(7-4.8)^2}{4.8}
    $$

    $$
    = \frac{64}{12} + \frac{33.64}{7.2} + \frac{4.84}{4.8} = 5.333 + 4.672 + 1.008 = 11.014
    $$

    $\text{df} = 2$에서 p-값은 약 $0.0041$이다. $\alpha = 0.05$에서 $H_0$을 기각한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
카이제곱 통계량의 각 항 분모에 (관측도수가 아니라) 기대도수가 와야 하는 이유는 무엇인가? 관측도수를 쓰면 무엇이 잘못되는가?

</div>

??? success "풀이"

    카이제곱 통계량은 각 제곱편차 $(O_i - E_i)^2$을 **기대**도수 $E_i$로 표준화한다. $H_0$ 아래에서 범주 $i$의 도수의 분산이 근사적으로 $E_i$이기 때문이다(다항분포에서 각 칸의 도수는 평균 $E_i$인 포아송으로 근사되고, 포아송 확률변수의 분산은 평균과 같다). 따라서 $E_i$로 나누면 표본분포가 근사적으로 $\chi^2$인 양이 만들어진다.

    관측도수를 쓰면, 우연히 관측도수가 아주 작은 범주에서 (0에 가까운 수로 나누기 때문에) 기여가 인위적으로 부풀려질 수 있고, 결과 통계량은 더 이상 $\chi^2$ 분포를 따르지 않는다. 이론적 정당화는 특정하게 $H_0$ 아래의 기대도수에 의존한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
범주가 $k = 2$개일 때 카이제곱 적합도 통계량이 일표본 비율 z-검정통계량의 제곱으로 환원됨을 증명하라.

</div>

??? success "풀이"

    두 범주의 관측도수를 $O_1$과 $O_2 = n - O_1$, 기대도수를 $E_1 = np_0$과 $E_2 = n(1 - p_0)$이라 하자. 그러면

    $$
    \chi^2 = \frac{(O_1 - np_0)^2}{np_0} + \frac{(O_2 - n(1-p_0))^2}{n(1-p_0)}
    $$

    이다. $O_2 = n - O_1$이므로 두 번째 분자는 $(n - O_1 - n + np_0)^2 = (np_0 - O_1)^2 = (O_1 - np_0)^2$이다. $(O_1 - np_0)^2$을 묶어내면

    $$
    \chi^2 = (O_1 - np_0)^2 \left[\frac{1}{np_0} + \frac{1}{n(1-p_0)}\right] = (O_1 - np_0)^2 \cdot \frac{1}{np_0(1-p_0)}
    $$

    이다. $\hat{p} = O_1/n$이라 쓰면 $O_1 - np_0 = n(\hat{p} - p_0)$이므로

    $$
    \chi^2 = \frac{n^2(\hat{p} - p_0)^2}{np_0(1-p_0)} = \left(\frac{\hat{p} - p_0}{\sqrt{p_0(1-p_0)/n}}\right)^2 = z^2
    $$

    이 되어 표준적인 일표본 비율 z-검정통계량의 제곱과 정확히 같다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
예제 1의 $\chi^2=5.25$가 **어느 범주에서 왔는지** 분해하고, 표준화 잔차로 해석하라.

</div>

??? success "풀이"
    **카이제곱은 항의 합이므로 기여도를 나눌 수 있다.**

    ```python
    import numpy as np
    from scipy import stats

    obs = np.array([4, 13, 7])
    n = obs.sum()
    exp = np.full(3, n / 3)
    term = (obs - exp)**2 / exp

    print("범주별 기여")
    for i, (o, e, t) in enumerate(zip(obs, exp, term), 1):
        print(f"  범주{i}: O={o:3d}  E={e:6.3f}  (O-E)={o - e:+7.3f}  "
              f"기여 {t:7.4f}  ({t / term.sum() * 100:5.1f}%)")
    print(f"  합계 χ² = {term.sum():.4f}\n")

    resid = (obs - exp) / np.sqrt(exp)                     # 표준화 잔차
    adj = (obs - exp) / np.sqrt(exp * (1 - 1 / 3))         # 조정 표준화 잔차
    print(f"표준화 잔차      {np.round(resid, 4).tolist()}")
    print(f"조정 표준화 잔차  {np.round(adj, 4).tolist()}")
    print(f"본페로니 임계값(범주 3개): {stats.norm.ppf(1 - 0.025 / 3):.4f}")
    ```

    ```text
    범주별 기여
      범주1: O=  4  E= 8.000  (O-E)= -4.000  기여  2.0000  ( 38.1%)
      범주2: O= 13  E= 8.000  (O-E)= +5.000  기여  3.1250  ( 59.5%)
      범주3: O=  7  E= 8.000  (O-E)= -1.000  기여  0.1250  (  2.4%)
      합계 χ² = 5.2500

    표준화 잔차      [-1.4142, 1.7678, -0.3536]
    조정 표준화 잔차  [-1.7321, 2.1651, -0.433]
    본페로니 임계값(범주 3개): 2.3940
    ```

    **범주 2가 통계량의 60%를 만든다.** 13번 나와야 할 자리에 8번이 기대되었으니 5만큼 많다.

    **왜 표준화 잔차와 조정 잔차가 다른가.** 단순 표준화 잔차 $(O-E)/\sqrt E$의 분산은 1이 아니라

    $$
    \operatorname{Var}\Bigl(\frac{O_i-E_i}{\sqrt{E_i}}\Bigr)
    =\frac{np_i(1-p_i)}{np_i}=1-p_i
    $$

    이다. 여기서는 $1-1/3=2/3$이므로 잔차가 실제보다 작게 보인다. $\sqrt{1-p_i}$로 한 번 더 나눈 것이 **조정 표준화 잔차**이고, 이것이 $H_0$ 아래에서 근사적으로 $N(0,1)$이다.

    **판정.** 조정 잔차의 최댓값이 2.165다.

    - **보정 없이 보면** $2.165>1.96$이라 "범주 2가 유의하게 많다"고 말하고 싶어진다.
    - **본페로니로 보정하면** 임계값이 2.394라 유의하지 않다.

    **전체 검정이 $p=0.072$로 유의하지 않았으므로, 개별 칸을 유의하다고 말하는 것은 일관되지 않는다.** 옴니버스 검정을 통과하지 못하면 사후 비교로 넘어가지 않는 것이 원칙이다.

    **기여도 분해의 쓸모.**

    1. **어디를 봐야 할지** 알려준다. 범주 3은 사실상 무관하다.
    2. **후속 연구의 방향**을 준다. "범주 2가 많은 것 같다"는 가설을 세워 새 자료로 확인한다.
    3. **다만 자료를 보고 만든 가설**이므로, 같은 자료로 검정하면 안 된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
$n=24$, $k=3$에서 카이제곱 통계량의 **정확한 분포**를 열거해 구하고, 근사가 얼마나 잘 맞는지 확인하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from math import lgamma
    from scipy import stats

    n, k = 24, 3
    p = np.ones(k) / k
    exp = n * p

    # 가능한 모든 (a, b, c) 를 열거해 χ² 값별 확률을 모은다
    table = {}
    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - a - b
            t = round((((np.array([a, b, c]) - exp)**2) / exp).sum(), 9)
            log_p = (lgamma(n + 1) - lgamma(a + 1) - lgamma(b + 1)
                     - lgamma(c + 1) + n * np.log(1 / 3))
            table[t] = table.get(t, 0.0) + np.exp(log_p)

    ts = np.array(sorted(table))
    pr = np.array([table[t] for t in ts])
    sf = np.cumsum(pr[::-1])[::-1]              # 오른쪽 꼬리 확률

    print(f"가능한 도수 벡터 {(n + 1) * (n + 2) // 2}개, "
          f"서로 다른 χ² 값 {len(ts)}개")
    print(f"통계량의 평균 {(ts * pr).sum():.4f}   (df = {k - 1})\n")

    obs_stat = (((np.array([4, 13, 7]) - exp)**2) / exp).sum()
    i = int(np.argmin(np.abs(ts - obs_stat)))
    print(f"예제 1 [4, 13, 7]:  χ² = {obs_stat:.4f}")
    print(f"   정확 p = {sf[i]:.4f}   근사 p = {stats.chi2.sf(obs_stat, k - 1):.4f}\n")

    print("0.05 근처에서 도달 가능한 p 값")
    mask = (sf > 0.01) & (sf < 0.15)
    for t, s in zip(ts[mask], sf[mask]):
        print(f"   χ² = {t:7.4f}   정확 p = {s:.4f}   "
              f"근사 p = {stats.chi2.sf(t, k - 1):.4f}")
    ```

    ```text
    가능한 도수 벡터 325개, 서로 다른 χ² 값 44개
    통계량의 평균 2.0000   (df = 2)

    예제 1 [4, 13, 7]:  χ² = 5.2500
       정확 p = 0.0800   근사 p = 0.0724

    0.05 근처에서 도달 가능한 p 값
       χ² =  4.0000   정확 p = 0.1481   근사 p = 0.1353
       χ² =  4.7500   정확 p = 0.1197   근사 p = 0.0930
       χ² =  5.2500   정확 p = 0.0800   근사 p = 0.0724
       χ² =  6.2500   정확 p = 0.0499   근사 p = 0.0439
       χ² =  6.7500   정확 p = 0.0412   근사 p = 0.0342
       χ² =  7.0000   정확 p = 0.0338   근사 p = 0.0302
       χ² =  7.7500   정확 p = 0.0213   근사 p = 0.0208
       χ² =  9.0000   정확 p = 0.0134   근사 p = 0.0111
       χ² =  9.2500   정확 p = 0.0115   근사 p = 0.0098
    ```

    **통계량의 평균이 정확히 2.0000이다.** $n=24$라는 작은 표본에서도 **1차 적률은 정확히 맞는다.** 자유도가 곧 평균이라는 성질은 근사가 아니라 여기서 정확히 성립한다.

    **꼬리는 다르다.** $\chi^2=4.75$에서 정확 0.1197, 근사 0.0930으로 **29% 차이**난다. **근사가 일관되게 작은 쪽**이라 약간 과대기각한다.

    **도달 가능한 $p$ 값이 44개뿐이다.** 325개의 표가 있지만 대칭 때문에 $\chi^2$ 값이 겹친다. $p=0.05$를 정확히 달성할 수 없고, 가장 가까운 것이 $\chi^2=6.25$에서의 **0.0499**다.

    **우연히도 이 값이 0.05에 극히 가깝다.** 그래서 $n=24$·$k=3$에서 "$\chi^2>6.25$이면 기각"이라는 규칙은 실제 수준이 0.0499로 거의 정확하다. 다만 이는 **우연**이고, 다른 $n$에서는 그렇지 않다.

    **실무적 함의 셋.**

    1. **$p$가 0.03~0.08이면 근사를 의심**한다. 그 구간에서 근사 오차가 결론을 바꿀 수 있다.
    2. **소표본에서는 정확 $p$를 계산**한다. $k=3$이면 위 코드가 즉시 끝난다.
    3. **"기대도수 5 이상"을 만족해도**($E_i=8$) 근사가 완벽하지는 않다. 조건은 최소한이지 충분조건이 아니다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 5가 증명한 $k=2$의 항등식을 **수치로 확인**하고, 이 사실이 왜 유용한지 설명하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    print(f"{'k/n':>10s} {'χ²':>12s} {'z²':>12s} {'차이':>10s} "
          f"{'χ² 의 p':>9s} {'z 의 양측 p':>12s}")
    for k, n in [(13, 24), (60, 100), (7, 20), (505, 1000)]:
        p0 = 0.5
        obs = np.array([k, n - k])
        exp = np.array([n * p0, n * (1 - p0)])
        chi2 = ((obs - exp)**2 / exp).sum()
        z = (k / n - p0) / np.sqrt(p0 * (1 - p0) / n)
        print(f"{f'{k}/{n}':>10s} {chi2:12.6f} {z**2:12.6f} "
              f"{abs(chi2 - z**2):10.2e} {stats.chi2.sf(chi2, 1):9.4f} "
              f"{2 * stats.norm.sf(abs(z)):12.4f}")
    ```

    ```text
           k/n           χ²           z²         차이    χ² 의 p     z 의 양측 p
         13/24     0.166667     0.166667   3.05e-16    0.6831       0.6831
        60/100     4.000000     4.000000   1.78e-15    0.0455       0.0455
          7/20     1.800000     1.800000   6.66e-16    0.1797       0.1797
      505/1000     0.100000     0.100000   1.80e-16    0.7518       0.7518
    ```

    **차이가 부동소수점 오차 수준**($10^{-16}$)이다. 항등식이 정확히 성립한다.

    $p$ 값도 완전히 같다. $Z\sim N(0,1)$이면 $Z^2\sim\chi^2_1$이고, **$|Z|>c$와 $Z^2>c^2$가 같은 사건**이기 때문이다.

    **왜 유용한가 — 네 가지.**

    **1 — 양측성의 이해.** 카이제곱 검정은 오른쪽 꼬리만 보지만, $k=2$에서 그것은 $z$ 검정의 **양측**에 해당한다. "카이제곱은 언제나 단측"이라는 말이 오해를 부르는 지점이다. 제곱하면서 방향 정보가 사라졌을 뿐이다.

    **2 — 단측 검정으로 바꿀 수 있다.** $k=2$일 때는 $z$ 형태로 돌아가 방향을 지정할 수 있다. $k\ge3$에서는 "방향"이라는 개념이 없어 불가능하다.

    **3 — 신뢰구간과 연결된다.** 비율의 윌슨 구간은 $z$ 검정을 뒤집어 얻는데, 이는 곧 카이제곱 검정을 뒤집는 것과 같다.

    **4 — 검정력 계산의 다리.** 비율 검정의 익숙한 표본크기 공식을 카이제곱 쪽으로 옮겨 쓸 수 있다.

    **일반화.** 이 관계는 $2\times2$ 분할표에서도 성립한다. 합동 $z$ 검정통계량의 제곱이 정확히 (보정 없는) 카이제곱 통계량이다.

    $$
    \chi^2=z_{\text{pooled}}^2
    $$

    **"같은 검정의 다른 표현"**이 통계학에는 아주 많다. 이런 항등식을 발견할 때마다 개념의 그물이 촘촘해진다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
손계산 결과를 `scipy` 함수와 **대조하는 습관**을 코드로 만들어라. 기대비율이 균등하지 않거나 모수를 추정한 경우까지 포함하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    def gof_by_hand(obs, p_exp=None, n_estimated=0, alpha=0.05):
        """카이제곱 적합도 검정을 직접 계산한다.

        p_exp        : 기대비율. None 이면 균등분포.
        n_estimated  : 자료에서 추정한 모수의 개수 (자유도에서 뺀다).
        """
        obs = np.asarray(obs, float)
        n, k = obs.sum(), len(obs)
        p_exp = np.full(k, 1 / k) if p_exp is None else np.asarray(p_exp, float)
        exp = n * p_exp
        chi2 = ((obs - exp)**2 / exp).sum()
        df = k - 1 - n_estimated
        return {"chi2": chi2, "df": df, "p": stats.chi2.sf(chi2, df),
                "exp": exp, "E_min": exp.min(),
                "reject": stats.chi2.sf(chi2, df) < alpha}

    cases = [
        ("예제 1 (균등)", [4, 13, 7], None, 0),
        ("불균등 기대비율", [4, 13, 7], [0.5, 0.3, 0.2], 0),
        ("주사위 120회", [15, 22, 18, 25, 20, 20], None, 0),
        ("포아송 적합(λ 추정)", [109, 65, 22, 4],
         [0.5434, 0.3314, 0.1011, 0.0241], 1),
    ]
    for label, obs, p_exp, r in cases:
        out = gof_by_hand(obs, p_exp, r)
        chk = stats.chisquare(f_obs=obs,
                              f_exp=None if p_exp is None
                              else np.array(p_exp) * sum(obs),
                              ddof=r)
        print(f"{label}")
        print(f"  손계산   χ²={out['chi2']:8.4f}  df={out['df']}  "
              f"p={out['p']:.4f}  E_min={out['E_min']:.3f}")
        print(f"  scipy    χ²={chk.statistic:8.4f}  p={chk.pvalue:.4f}")
        print(f"  일치? {np.isclose(out['chi2'], chk.statistic)} / "
              f"{np.isclose(out['p'], chk.pvalue)}")
    ```

    ```text
    예제 1 (균등)
      손계산   χ²=  5.2500  df=2  p=0.0724  E_min=8.000
      scipy    χ²=  5.2500  p=0.0724
      일치? True / True
    불균등 기대비율
      손계산   χ²= 11.0139  df=2  p=0.0041  E_min=4.800
      scipy    χ²= 11.0139  p=0.0041
      일치? True / True
    주사위 120회
      손계산   χ²=  2.9000  df=5  p=0.7154  E_min=20.000
      scipy    χ²=  2.9000  p=0.7154
      일치? True / True
    포아송 적합(λ 추정)
      손계산   χ²=  0.3219  df=2  p=0.8514  E_min=4.820
      scipy    χ²=  0.3219  p=0.8514
      일치? True / True
    ```

    **네 경우 모두 일치한다.** 통계량은 언제나 같고, **$p$ 값은 자유도를 맞춰 주어야 같아진다.**

    **`ddof` 인자가 핵심이다.** `scipy.stats.chisquare`의 자유도는

    $$
    \text{df}=k-1-\texttt{ddof}
    $$

    이다. 기본값이 0이므로, **모수를 추정했다면 반드시 `ddof`를 넘겨야** 한다. 넷째 경우에서 `ddof=1`을 빠뜨리면 $p$가 0.851이 아니라 0.956으로 나온다.

    **대조 습관이 잡아 주는 실수들.**

    | 실수 | 증상 |
    |---|---|
    | 기대도수의 합이 $n$과 다름 | `scipy`가 오류를 낸다 |
    | 자유도를 잘못 셈 | 통계량은 같은데 $p$가 다르다 |
    | 비율과 도수를 혼동 | 통계량이 터무니없이 다르다 |
    | 배열 순서가 어긋남 | 통계량이 조용히 달라진다 |

    **마지막이 가장 위험하다.** 관측과 기대의 **범주 순서가 어긋나도** 오류가 나지 않고 조용히 틀린 값이 나온다. 그래서 두 배열을 만들 때 **같은 순서 목록에서 생성**하는 것이 안전하다.

    **손계산을 배우는 이유.** 실무에서는 `scipy`를 쓰지만, 무엇이 계산되는지 알아야 **$p$ 값이 이상할 때 원인을 찾을 수 있다.** 위 함수처럼 중간값($E_{\min}$, df)을 함께 돌려주도록 만들어 두면 진단이 쉬워진다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
손으로 카이제곱 적합도 검정을 할 때의 **계산 순서와 점검 목록**을 정리하라.

</div>

??? success "풀이"

    **계산 순서.**

    ```text
    ① n 과 k 를 센다
         n = 관측도수의 합,  k = 범주의 수
              ↓
    ② 기대비율 p_i 를 정한다
         ├─ 균등 → p_i = 1/k
         ├─ 이론값 → 문제에서 주어짐 (예: 3:1 분리비)
         └─ 모형 적합 → 추정한 모수로 계산 (추정 개수 r 을 기록)
              ↓
    ③ E_i = n p_i 를 계산한다
         └─ 검산: Σ E_i = n 이어야 한다
              ↓
    ④ E_min 을 확인한다
         └─ 5 미만이면 병합 규칙을 적용
              ↓
    ⑤ 항별로 (O_i - E_i)² / E_i 를 계산해 더한다
         └─ 검산: Σ(O_i - E_i) = 0 이어야 한다
              ↓
    ⑥ df = k - 1 - r
              ↓
    ⑦ p = P(χ²_df > 통계량)   ← 오른쪽 꼬리
              ↓
    ⑧ 잔차를 보고 어디가 어긋났는지 확인한다
    ```

    **두 개의 검산이 중요하다.**

    | 검산 | 틀리면 |
    |---|---|
    | $\sum E_i=n$ | 기대비율의 합이 1이 아니다 |
    | $\sum(O_i-E_i)=0$ | 산술 오류 |

    **셋째 검산.** 통계량이 자유도와 비슷하면 "예상대로", 자유도의 서너 배면 "이탈이 크다"는 감각을 갖는다. 통계량이 자유도보다 **훨씬 작아도** 이상하다(적합이 너무 좋다).

    **점검 목록.**

    - [ ] 관측이 **도수**인가(비율·평균이 아니라)
    - [ ] 각 관측이 정확히 한 범주에 들어가는가
    - [ ] $\sum E_i = n$ 인가
    - [ ] $E_{\min}$ 은 얼마인가
    - [ ] 모수를 추정했다면 **자유도를 줄였는가**
    - [ ] **오른쪽 꼬리**를 썼는가(`sf`이지 `cdf`가 아님)
    - [ ] 잔차를 확인했는가

    **자주 하는 산술 실수 넷.**

    1. **제곱을 빠뜨림.** $(O-E)$를 그대로 더하면 항상 0이다.
    2. **$E$가 아니라 $O$로 나눔.** 전혀 다른 통계량이 된다.
    3. **`cdf`를 씀.** $p$ 값이 $1-p$로 나온다. 값이 0.9 이상이면 이 실수를 의심한다.
    4. **자유도를 $k$로 씀.** 항상 $k-1$에서 시작한다.

    **셋째 실수를 알아채는 법.** $p$ 값이 0.95처럼 크게 나왔는데 관측과 기대가 눈에 띄게 다르다면 꼬리를 뒤집어 쓴 것이다. **통계량과 자유도를 함께 보면** 금방 드러난다.

    **손계산의 가치.** 실무에서는 한 줄이면 끝나지만, 손으로 한 번 해 보면 **분모의 $E_i$가 하는 일**, **자유도가 줄어드는 이유**, **어느 범주가 기여하는지**가 몸에 익는다. 그 감각이 있어야 결과가 이상할 때 원인을 짚을 수 있다.

---

## 정리하며

적합도 검정을 **손으로 계산**해 보며 내부를 확인했다.

- **세 단계뿐이다.** 기대도수 $E_i=np_i$ 를 만들고, $\sum(O_i-E_i)^2/E_i$ 를 더하고, $\chi^2_{k-1}$ 의 오른쪽 꼬리 확률을 구한다.
- **$p$ 값은 생존함수로 계산한다.** `chi2.sf(stat, df)` 이며, `1 - cdf` 는 꼬리에서 정밀도를 잃는다(4장).
- **그림이 이해를 돕는다.** 밀도곡선에 통계량 위치를 표시하고 오른쪽 꼬리를 칠하면 $p$ 값이 무엇인지 한눈에 보인다.
- **칸별 기여도를 함께 보는 습관.** $(O_i-E_i)^2/E_i$ 를 범주별로 나열하면 어느 범주가 기각을 이끌었는지 드러난다. **통계량 하나만으로는 알 수 없다.**
- **직접 구현의 가치는 검산이다.** 다음 절의 라이브러리 결과와 맞춰 보면 자유도나 기대도수 설정의 실수를 잡아낼 수 있다.

다음 절 **카이제곱 적합도 검정 (scipy)** 로 넘어간다.
