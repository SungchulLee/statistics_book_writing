# S²의 표준오차

## 개요

표본평균 $\bar{X}$가 표본 간 변동성을 재는 표준오차를 갖듯이 표본분산 $S^2$도 표준오차를 갖는다. 모분산 $\sigma^2$을 추정하거나 그에 관해 추론할 때 $S^2$의 정밀도를 이해하는 것이 중요하다. 이 페이지에서는 $S^2$의 표준오차를 유도하고, 균등 모집단에서 모의실험으로 추정하며, 그림으로 나타낸다.

<div class="defn" markdown>

### 정의 1. 표본분산의 표준오차 { .dfn }

**$S^2$의 표준오차**는 그 표본분포의 표준편차이다:

$$
\text{SE}(S^2) = \sqrt{\text{Var}(S^2)}
$$

정규모집단에서는 닫힌 형태로 주어진다:

$$
\text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}}
$$

더 일반적으로, 4차 적률이 유한한 임의의 모집단에 대해:

$$
\text{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right)
$$

여기서 $\mu_4 = E[(X - \mu)^4]$는 4차 중심적률이다.

</div>

## Uniform(0, 1) 모집단

$X \sim \text{Uniform}(0, 1)$에 대해:

$$
\sigma^2 = \frac{1}{12}, \qquad \mu_4 = \frac{1}{80}
$$

$n = 5$이면:

$$
\text{Var}(S^2) = \frac{1}{5}\left(\frac{1}{80} - \frac{2}{4} \cdot \frac{1}{144}\right) = \frac{1}{5}\left(\frac{1}{80} - \frac{1}{288}\right)
$$

$$
= \frac{1}{5} \cdot \frac{288 - 80}{80 \times 288} = \frac{1}{5} \cdot \frac{208}{23040} = \frac{208}{115200} \approx 0.001806
$$

$$
\text{SE}(S^2) \approx \sqrt{0.001806} \approx 0.0425
$$

## 모의실험

다음 코드는 Uniform(0, 1) 모집단에서 $n = 5$로 $S^2$ 값을 10,000개 모의실험하고 추정된 평균과 표준오차를 함께 시각화한다.

<div class="codebox" markdown>

### 예제 1. 표본분산의 표준오차 모의실험 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# 크기 5짜리 균등표본을 1만 번 뽑아 그때마다 S^2 을 기록한다.
S_square = []
for _ in range(10_000):
    x = np.random.uniform(size=(5,))
    sigma = x.std(ddof=1)     # ddof=1 이라야 표본표준편차다
    S_square.append(sigma ** 2)

# 1만 개 S^2 값의 **평균**과 **표준편차**를 낸다.
#   평균  -> S^2 의 중심. 참 분산 1/12 ≈ 0.0833 에 가까워야 한다(불편성).
#   표준편차 -> 그것이 곧 S^2 의 **표준오차**다.
# 표준오차란 "통계량의 표집분포의 표준편차"이므로,
# 통계량 값을 잔뜩 모아 그 표준편차를 재면 그것이 표준오차다.
average = np.array(S_square).mean()
standard_error = np.array(S_square).std()

print(f"Estimated Mean of S^2:   {average:.4f}")
print(f"Standard Error of S^2:   {standard_error:.4f}")

# 히스토그램에 이론값을 겹쳐 그린다.
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_title("Sampling Distribution of S^2")
ax.hist(S_square, bins=100, density=True, alpha=0.3)
# 평균과 평균 ± 1 표준오차를 세로선으로 표시한다.
# S^2 의 분포는 오른쪽으로 치우쳐 있어 이 구간이 대칭이 아님에 주의하라.
ax.vlines(average, ymin=0, ymax=12, color="k", lw=5, label="Mean")
ax.vlines(average + standard_error, ymin=0, ymax=12,
          color="k", ls="--", label="Mean +/- SE")
ax.vlines(average - standard_error, ymin=0, ymax=12,
          color="k", ls="--")
ax.legend()
plt.show()
```

출력:

```
Estimated Mean of S^2:   0.0838
Standard Error of S^2:   0.0425
```

![Sampling Distribution of S^2](./img/se_s2_55.png)

</div>

### 예상 출력

$n = 5$인 Uniform(0, 1)에 대해:

- **이론적 평균**: $E[S^2] = \sigma^2 = 1/12 \approx 0.0833$
- **이론적 표준오차**: 약 $0.0425$
- 히스토그램은 오른쪽으로 치우쳐 있다($S^2 \ge 0$이므로). 분산의 표본분포에서 전형적인 모습이다.

## 해석

!!! note "주요 관찰"

    1. $S^2$의 표본분포는 근사적으로 대칭인 $\bar{X}$의 분포와 달리 **오른쪽으로 치우쳐** 있다.
    2. 모의실험한 $S^2$ 값들의 평균이 $\sigma^2 = 1/12$에 가까워 $S^2$이 불편임을 확인해 준다.
    3. $S^2$의 표준오차가 평균보다 훨씬 작아 $n = 5$에서도 그럭저럭 정밀함을 보여 준다.
    4. 평균 $\pm$ 표준오차의 점선은 $S^2$ 값의 전형적인 범위를 나타낸다. 분포가 치우쳐 있으므로 평균 $+$ 표준오차보다 큰 값이 평균 $-$ 표준오차보다 작은 값보다 더 흔하다.

!!! warning "S-squared의 표준오차는 모집단 모양에 의존한다"
    $\sigma$와 $n$에만 의존하는 $\text{SE}(\bar{X}) = \sigma/\sqrt{n}$과 달리, $S^2$의 표준오차는 모집단의 4차 적률(첨도)에 의존한다. 꼬리가 두꺼운 모집단일수록 $S^2$ 값의 변동이 커진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> $n = 10$인 $N(0, 1)$ 모집단에 대해 이론적 $E[S^2]$과 $\text{SE}(S^2)$을 계산하라.

</div>

??? success "풀이"
    $N(0, 1)$에서 $\sigma^2 = 1$이다.

    $$
    E[S^2] = \sigma^2 = 1
    $$

    $$
    \text{SE}(S^2) = \sigma^2 \sqrt{\frac{2}{n-1}} = 1 \cdot \sqrt{\frac{2}{9}} = \sqrt{\frac{2}{9}} = \frac{\sqrt{2}}{3} \approx 0.4714
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 카이제곱 결과 $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$을 사용하여 정규모집단에서 $\text{Var}(S^2) = 2\sigma^4 / (n-1)$을 유도하라.

</div>

??? success "풀이"
    $Q = (n-1)S^2/\sigma^2$이라 하자. 그러면 $Q \sim \chi^2(n-1)$이고 $\text{Var}(Q) = 2(n-1)$이다.

    $S^2 = \sigma^2 Q / (n-1)$이므로:

    $$
    \text{Var}(S^2) = \left(\frac{\sigma^2}{n-1}\right)^2 \text{Var}(Q) = \frac{\sigma^4}{(n-1)^2} \cdot 2(n-1) = \frac{2\sigma^4}{n-1}
    $$

    따라서:

    $$
    \text{SE}(S^2) = \sqrt{\frac{2\sigma^4}{n-1}} = \sigma^2 \sqrt{\frac{2}{n-1}}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> Uniform(0, 1)의 4차 중심적률이 $\mu_4 = 1/80$임을 보여라.

</div>

??? success "풀이"
    $\mu = 1/2$인 $X \sim \text{Uniform}(0, 1)$에 대해:

    $$
    \mu_4 = E[(X - \mu)^4] = \int_0^1 \left(x - \frac{1}{2}\right)^4 dx
    $$

    $u = x - 1/2$로 치환하면 $du = dx$이고 적분 범위는 $-1/2$에서 $1/2$이 된다:

    $$
    \mu_4 = \int_{-1/2}^{1/2} u^4 \, du = \left[\frac{u^5}{5}\right]_{-1/2}^{1/2} = \frac{(1/2)^5}{5} - \frac{(-1/2)^5}{5} = \frac{2 \cdot (1/32)}{5} = \frac{1}{80}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $n$이 작을 때 $S^2$의 표본분포가 오른쪽으로 치우치는 이유와 $n$이 커질수록 더 대칭이 되는 이유를 설명하라.

</div>

??? success "풀이"
    표본분산 $S^2$은 아래로 0에서 유계이지만 (원리상) 위로는 유한한 한계가 없다. $n$이 작으면 제약 $S^2 \ge 0$이 왼쪽 꼬리를 잘라 내는 "바닥"을 만드는 반면, 이따금 나타나는 극단적인 관측값이 $S^2$을 큰 값으로 밀어 올려 긴 오른쪽 꼬리를 만든다.

    $n$이 커지면 두 가지 효과가 분포를 대칭화한다:

    1. **$S^2$에 대한 중심극한정리**: $n$이 크면 $S^2$은 약하게 의존하는 많은 항(제곱편차)의 합에 가까우므로, 중심극한정리 유형의 논증에 의해 그 분포가 정규분포로 수렴한다.
    2. **카이제곱의 수렴**: 정규모집단에서 $(n-1)S^2/\sigma^2 \sim \chi^2(n-1)$이다. $\chi^2(k)$의 왜도는 $2\sqrt{2/k}$이며 $k = n - 1$이 커지면 0으로 간다.

    두 효과 모두 $n$이 클 때 $S^2$의 분포가 근사적으로 대칭(정규)이 되고 표준오차가 퍼짐을 잘 요약해 줌을 뜻한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> Exponential(1) 모집단으로 모의실험을 반복하라. $\text{Exp}(1)$에서 $\sigma^2 = 1$, $\mu_4 = 9$임에 유의하여 $S^2$의 경험적 표준오차를 이론값과 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    np.random.seed(0)

    # 크기 5짜리 지수분포 표본에서 S^2 을 1만 번 계산한다.
    # ddof=1 이 표본분산(n-1로 나눔)을 뜻한다.
    S_square = [np.random.exponential(size=5).var(ddof=1) for _ in range(10_000)]
    empirical_se = np.std(S_square)

    # 이론값 Var(S^2) = (1/n)(mu4 - (n-3)/(n-1) * sigma^4)
    # Exp(1)에서 sigma^2 = 1, mu4 = 9 이므로 (1/5)(9 - 2/4) = 1.7
    theoretical_se = np.sqrt(1.7)
    print(f"경험적 SE = {empirical_se:.4f}")
    print(f"이론적 SE = {theoretical_se:.4f}")
    ```

    출력:

    ```
    경험적 SE = 1.3237
    이론적 SE = 1.3038
    ```

    n = 5로 아주 작은데도 두 값이 잘 맞는다. 다만 4차 적률이 들어가는 공식이라
    수렴이 느리므로, 표본이 작으면 이 정도(1.5%) 차이는 정상이다.

    $\sigma^2 = 1$, $\mu_4 = 9$, $n = 5$로 이론값을 계산하면:

    $$
    \text{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right) = \frac{1}{5}\left(9 - \frac{2}{4}\right) = \frac{1}{5} \cdot 8.5 = 1.7
    $$

    $$
    \text{SE}(S^2) = \sqrt{1.7} \approx 1.304
    $$

    비교하자면 정규이론 표준오차는 $\sigma^2 \sqrt{2/(n-1)} = \sqrt{2/4} = \sqrt{0.5} \approx 0.707$이다.

    지수 모집단의 표준오차가 정규이론 값보다 약 1.84배 큰데, 이는 지수분포의 두꺼운 꼬리(초과첨도 $= 6$)를 반영한다. 모의실험의 경험적 표준오차는 1.304에 가깝게 나온다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
일반 모집단에서

$$
\operatorname{Var}(S^2) = \frac{1}{n}\left(\mu_4 - \frac{n-3}{n-1}\sigma^4\right)
$$

임이 알려져 있다. 정규모집단($\mu_4 = 3\sigma^4$)에 넣어 $2\sigma^4/(n-1)$이 나옴을 확인하고, 이 식이 초과첨도로 어떻게 다시 쓰이는지 보여라.

</div>

??? success "풀이"
    **정규모집단.** $\mu_4 = 3\sigma^4$을 넣으면

    $$
    \operatorname{Var}(S^2) = \frac{\sigma^4}{n}\left(3 - \frac{n-3}{n-1}\right) = \frac{\sigma^4}{n}\cdot\frac{3(n-1)-(n-3)}{n-1} = \frac{\sigma^4}{n}\cdot\frac{2n}{n-1} = \frac{2\sigma^4}{n-1}
    $$

    으로 카이제곱 결과와 정확히 일치한다. ✓

    **초과첨도 형태.** $\gamma_2 = \mu_4/\sigma^4 - 3$으로 두면 $\mu_4 = (\gamma_2+3)\sigma^4$이므로

    $$
    \operatorname{Var}(S^2) = \frac{\sigma^4}{n}\left(\gamma_2 + 3 - \frac{n-3}{n-1}\right) = \frac{\sigma^4}{n}\left(\gamma_2 + \frac{2n}{n-1}\right) \approx \frac{\sigma^4}{n}(\gamma_2+2)
    $$

    이다(마지막은 $n$이 클 때).

    **읽는 법.** $S^2$의 표준오차를 정하는 것은 **분산이 아니라 4차 적률, 곧 첨도**다.

    | 모집단 | $\gamma_2$ | $\operatorname{Var}(S^2)$의 배율 |
    |---|---|---|
    | 균등 | $-1.2$ | $0.8\sigma^4/n$ |
    | 정규 | 0 | $2\sigma^4/n$ |
    | 지수 | 6 | $8\sigma^4/n$ |
    | $t_5$ | 6 | $8\sigma^4/n$ |

    지수 모집단에서 $S^2$의 분산이 정규 이론값의 **네 배**다. 표준오차로는 두 배다. 이것이 앞 연습문제에서 본 1.84배($n=5$라 $2n/(n-1)$ 보정이 들어간 값)의 정체다.

    실무적 함의가 크다. **평균에 대한 추론은 중심극한정리가 구해 주지만 분산에 대한 추론은 그렇지 않다.** 첨도는 자체로 추정하기 어려운 양이라(6차 적률이 필요하다), 분산의 신뢰구간을 자료에서 안전하게 만드는 일은 생각보다 까다롭다. 부트스트랩이 유용한 대목이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
표본표준편차 $S$는 $\sigma$의 **편향된** 추정량이다. 정규모집단에서 $E[S] = c_4\sigma$이고

$$
c_4 = \sqrt{\frac{2}{n-1}}\cdot\frac{\Gamma(n/2)}{\Gamma\{(n-1)/2\}}
$$

임이 알려져 있다. $n=2, 5, 10, 25$에서 편향의 크기를 구하고, $S^2$은 불편인데 $S$는 왜 아닌지 설명하라.

</div>

??? success "풀이"

    | $n$ | 2 | 5 | 10 | 25 | 50 |
    |---|---|---|---|---|---|
    | $c_4$ | 0.7979 | 0.9400 | 0.9727 | 0.9896 | 0.9949 |
    | 편향 | $-20.2\%$ | $-6.0\%$ | $-2.7\%$ | $-1.0\%$ | $-0.5\%$ |

    언제나 **과소추정**하며, $n$이 커지면 사라진다.

    **$S^2$은 불편인데 $S$는 아닌 이유.** 제곱근이 **오목함수**이기 때문이다. 옌센 부등식에 따라

    $$
    E[S] = E\!\left[\sqrt{S^2}\right] < \sqrt{E[S^2]} = \sqrt{\sigma^2} = \sigma
    $$

    이다. 등호는 $S^2$이 상수일 때만 성립하는데, 그것은 $n=\infty$인 경우다.

    **일반 원리.** **불편성은 비선형 변환에서 보존되지 않는다.** $E[\hat\theta]=\theta$라 해도 $E[g(\hat\theta)] \ne g(\theta)$이며, $g$가 오목이면 아래로, 볼록이면 위로 치우친다. 앞서 본 $\hat\lambda = 1/\bar X$가 볼록 변환이라 과대추정하는 것과 정확히 반대 방향이다.

    **보정.** $S/c_4$를 쓰면 불편이 되고, 공정능력지수나 관리도에서 실제로 이 보정을 쓴다. 다만 보정한 추정량이 평균제곱오차 기준으로는 오히려 나쁠 수 있다는 점을 기억할 일이다. 편향을 없애면서 분산이 늘기 때문이다.

    실무에서 $n \ge 25$면 편향이 1% 미만이라 무시해도 좋다. 문제가 되는 것은 $n$이 아주 작을 때, 특히 여러 개의 작은 표본에서 얻은 $s$를 평균 내는 경우다. 편향이 상쇄되지 않고 그대로 쌓인다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연습문제 2의 결과에서 델타 방법으로 $\operatorname{SE}(S)$를 구하고, $S$의 **상대** 표준오차가 $1/\sqrt{2(n-1)}$임을 보여라. $\sigma$를 10% 이내로 추정하려면 표본이 몇 개 필요한가?

</div>

??? success "풀이"
    $g(x) = \sqrt x$에 대해 $g'(\sigma^2) = 1/(2\sigma)$이므로

    $$
    \operatorname{Var}(S) \approx \left(\frac{1}{2\sigma}\right)^2\operatorname{Var}(S^2) = \frac{1}{4\sigma^2}\cdot\frac{2\sigma^4}{n-1} = \frac{\sigma^2}{2(n-1)}
    $$

    이고

    $$
    \operatorname{SE}(S) \approx \frac{\sigma}{\sqrt{2(n-1)}}, \qquad \frac{\operatorname{SE}(S)}{\sigma} \approx \frac{1}{\sqrt{2(n-1)}}
    $$

    이다. $\square$

    **필요 표본크기.** 상대 표준오차가 0.10 이하이려면

    $$
    \frac{1}{\sqrt{2(n-1)}} \le 0.10 \implies 2(n-1) \ge 100 \implies n \ge 51
    $$

    이다.

    **평균과의 대비가 인상적이다.** 평균을 상대오차 10%로 추정하려면 $\sigma/(\sqrt n|\mu|) \le 0.1$, 즉 변동계수가 0.5라면 $n \ge 25$면 된다. 그런데 표준편차 자체를 같은 정밀도로 알려면 51개가 필요하고, **분산**을 10% 이내로 알려면 $\sqrt{2/(n-1)} \le 0.1$에서 $n \ge 201$이 필요하다.

    | 추정 대상 | 상대오차 10%에 필요한 $n$ |
    |---|---|
    | 평균(CV=0.5) | 25 |
    | 표준편차 | 51 |
    | 분산 | 201 |

    **산포를 정밀하게 아는 것은 중심을 아는 것보다 훨씬 비싸다.** 표본크기 계산이 보통 평균을 기준으로 이루어지므로, 같은 연구에서 분산에 관한 결론은 훨씬 불확실하다는 점을 잊지 말아야 한다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$S^2$의 분포가 $n$이 커질 때 정규분포로 가는 속도를 왜도로 따져 보자. 정규모집단에서 $(n-1)S^2/\sigma^2 \sim \chi^2_{n-1}$이므로 $S^2$의 왜도는 얼마인가? $n=10, 30, 100$에서 값을 구하고 $\bar X$의 경우와 견주어라.

</div>

??? success "풀이"
    카이제곱분포의 왜도가 $\sqrt{8/k}$이고 선형변환은 왜도를 바꾸지 않으므로, $k=n-1$에서

    $$
    \text{왜도}(S^2) = \sqrt{\frac{8}{n-1}}
    $$

    이다.

    | $n$ | 10 | 30 | 100 | 200 |
    |---|---|---|---|---|
    | 왜도$(S^2)$ | 0.943 | 0.525 | 0.284 | 0.200 |
    | 왜도$(\bar X)$, 정규모집단 | 0 | 0 | 0 | 0 |

    **정규모집단에서 $\bar X$는 왜도가 정확히 0인데 $S^2$은 $n=100$에서도 0.284나 남아 있다.**

    이 차이가 실무에서 갖는 뜻은 분명하다.

    - $\bar X$의 신뢰구간은 정규모집단에서 **정확**하고, 비정규모집단에서도 중심극한정리가 빠르게 구해 준다.
    - $S^2$의 신뢰구간은 대칭으로 만들면 안 된다. $\bar s^2 \pm z\cdot\operatorname{SE}$ 꼴은 $n$이 꽤 커도 포함확률이 어긋나며, 특히 아래쪽 한계가 음수가 되기도 한다. 앞서 본 카이제곱 기반 구간이 비대칭인 이유가 이것이다.
    - 분산에 대한 검정에서 단측·양측의 임계값을 대칭으로 잡으면 틀린다.

    한마디로 **$S^2$은 $\bar X$보다 정규성에 훨씬 늦게 다가간다.** 분산을 다룰 때는 $n$이 크다는 이유만으로 정규근사를 믿으면 안 된다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
$\operatorname{SE}(S^2)$를 자료만으로 추정하는 두 가지 방법을 적고, 각각이 어떤 가정에 기대는지 밝혀라.

</div>

??? success "풀이"
    **방법 1 — 정규이론 대입.**

    $$
    \widehat{\operatorname{SE}}(S^2) = s^2\sqrt{\frac{2}{n-1}}
    $$

    $(n-1)S^2/\sigma^2\sim\chi^2_{n-1}$을 쓰고 미지의 $\sigma^2$을 $s^2$으로 대신한 것이다.

    - **가정**: 모집단이 **정규**여야 한다. 이것이 강한 가정이다. 연습문제 6에서 보았듯 초과첨도가 6인 모집단에서는 참 표준오차가 이 값의 두 배다.
    - **장점**: 계산이 한 줄이고 $n$이 작아도 쓸 수 있다.

    **방법 2 — 적률 대입.**

    $$
    \widehat{\operatorname{SE}}(S^2) = \sqrt{\frac{1}{n}\left(\hat\mu_4 - \frac{n-3}{n-1}s^4\right)}, \qquad \hat\mu_4 = \frac1n\sum_i(x_i-\bar x)^4
    $$

    연습문제 6의 일반 공식에 표본 4차 적률을 넣은 것이다.

    - **가정**: 정규성을 요구하지 않는다. 다만 모집단의 **8차 적률**이 유한해야 $\hat\mu_4$ 자체가 안정적이다.
    - **단점**: $\hat\mu_4$의 추정오차가 매우 크다. 4제곱이 들어가므로 관측값 하나가 결과를 좌우할 수 있고, $n$이 작으면 쓸 수 없다.

    **방법 3 — 부트스트랩.** $S^2$을 재표본마다 계산해 그 표준편차를 쓴다.

    - **가정**: 분포 가정이 거의 없다. 관측이 독립이고 $n$이 너무 작지 않으면 된다.
    - **장점**: 자동으로 첨도의 영향을 반영한다. $n \ge 30$쯤이면 대체로 가장 믿을 만하다.

    **권고.** 정규성을 확신할 수 있으면 방법 1이 가장 효율적이다. 그렇지 않으면 부트스트랩이 안전하다. 방법 2는 이론적으로 옳지만 실무에서 $\hat\mu_4$의 불안정성 때문에 잘 쓰이지 않는다.

---

## 정리하며

$S^2$ 에도 표준오차가 있고, 정규모집단에서는 닫힌 형태로 주어진다.

$$
\mathrm{SE}(S^2) = \sigma^2\sqrt{\frac{2}{n-1}}
$$

- **$\bar X$ 와 달리 4차적률이 필요하다.** 일반적인 모집단에서는 $\mathrm{Var}(S^2)\approx(\mu_4-\sigma^4)/n$ 이며, 정규분포의 $\mu_4=3\sigma^4$ 을 넣으면 위 공식이 나온다.
- **초과첨도가 곧 벌점이다.** $\gamma_2$ 가 양수면 실제 표준오차가 정규 공식보다 크고, 그 차이가 $\sqrt{(\gamma_2+2)/2}$ 배다. $\gamma_2=6$ 이면 **두 배**다.
- **상대 정밀도가 $\sqrt{2/(n-1)}$** 이므로 분산을 $10\%$ 정밀도로 추정하려면 $n\approx200$ 이 필요하다. 같은 정밀도로 평균을 추정하는 것보다 훨씬 많은 자료가 든다.
- **분산·상관·첨도로 갈수록 필요한 적률의 차수가 올라가고 추정이 어려워진다.** 그 순서가 곧 신뢰도의 순서다.

다음 쪽부터는 네 모집단에서 $S^2$ 의 표본분포를 직접 모의실험한다. $\bar X$ 쪽과 달리 **카이제곱 결과가 정규모집단에서만 성립**하므로, 모집단을 바꾸면 결론도 바뀐다.
