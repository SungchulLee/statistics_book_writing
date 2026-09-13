# 스튜던트 t 분포

## 개요

스튜던트 $t$ 분포는 알려진 모표준편차 $\sigma$ 대신 **표본표준편차** $S$를 사용하여 정규모집단의 평균을 추정할 때 나타난다. $\sigma$를 추정하면서 생기는 추가적인 불확실성을 반영한다.

---

<div class="defn" markdown>

### 정의 1. 스튜던트 t 분포 { .dfn }

$Z \sim N(0,1)$과 $V \sim \chi^2_d$가 독립이라 하자. 그러면 다음 비:

$$
T = \frac{Z}{\sqrt{V/d}} \sim t_d
$$

는 자유도 $d$인 스튜던트 $t$ 분포를 따른다.

---

</div>

## 자유도

자유도 $d = n - 1$은 표본분산을 추정하는 데 사용된 독립적인 정보의 개수를 반영한다.

- **$d$가 작을 때**: 정규분포보다 꼬리가 두꺼워 더 큰 불확실성을 나타낸다.
- **$d$가 클 때 ($> 30$)**: $N(0, 1)$과 사실상 구별되지 않는다.

---

## 성질

$$
\begin{aligned}
\text{Mean} &= 0 \quad \text{for } d > 1 \\
\text{Variance} &= \frac{d}{d - 2} \quad \text{for } d > 2
\end{aligned}
$$

$d \to \infty$일 때 분산은 1에 가까워지고 $t_d \to N(0, 1)$이다.

---

## PDF

$$
f_T(x) = \frac{1}{\sqrt{d}\,B\!\left(\tfrac{1}{2}, \tfrac{d}{2}\right)} \left(1 + \frac{x^2}{d}\right)^{-\frac{d+1}{2}}
$$

여기서 $B(\cdot, \cdot)$는 베타함수이다.

??? proof "증명 개요"


    $T = Z / \sqrt{V/d}$에 대해 $(Z, V)$의 결합밀도에 변수변환을 적용한다. Jacobian 인수는 $\sqrt{v/d}$이고, $\chi^2$ 변수를 적분해 없애면 $T$의 주변밀도가 위 형태가 된다. 조건부분포 $V | T = t$는 감마분포로 나타난다.

    ---

## 두꺼운 꼬리

$t$ 분포는 정규분포보다 **꼬리가 두꺼워** 극단값이 나타날 가능성이 더 크다:

<div class="codebox" markdown>

### 예제 1. t 분포의 두꺼운 꼬리 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, (ax_full, ax_tail) = plt.subplots(1, 2, figsize=(12, 3))
x = np.linspace(-4, 4, 200)

# 왼쪽: 전체 모습. 두 곡선이 거의 겹쳐 보인다.
ax_full.plot(x, stats.norm().pdf(x), label='Normal')
ax_full.plot(x, stats.t(df=10).pdf(x), label='t(10)')
ax_full.set_title('Full PDF')
ax_full.legend()

# 오른쪽: 꼬리만 확대. x[-50:] 은 x 배열의 마지막 50개, 즉 오른쪽 끝이다.
# 전체 그림에서는 밀도가 너무 작아 안 보이던 차이가 여기서 드러난다.
# **꼬리는 언제나 확대해서 봐야 한다.** 검정에서 문제가 되는 곳이 바로 꼬리다.
ax_tail.plot(x[-50:], stats.norm().pdf(x[-50:]), label='Normal')
ax_tail.plot(x[-50:], stats.t(df=10).pdf(x[-50:]), label='t(10)')
ax_tail.set_title('Right Tail (zoomed)')
ax_tail.legend()

plt.tight_layout()
plt.show()
```

![Full PDF](./img/student_t_61.png)

</div>

### 정규분포로의 수렴

<div class="codebox" markdown>

#### 예제 2. 자유도가 커지면 t가 정규로 간다 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

fig, ax = plt.subplots(figsize=(12, 3))
x = np.linspace(-3, 3, 200)

# 자유도를 키우면 t가 정규분포로 수렴한다.
# df=1 은 코시분포로 평균조차 없고, df=2 부터 평균이 생기며,
# df>2 라야 분산 df/(df-2) 가 존재한다. df=20 이면 이미 정규와 거의 같다.
for df in [1, 2, 5, 10, 20]:
    ax.plot(x, stats.t(df).pdf(x), label=f'df={df}')
ax.plot(x, stats.norm().pdf(x), 'r--', lw=2, label='Normal')     # 극한
ax.legend()
ax.set_title('t-Distribution Converges to Normal as df Increases')
plt.show()
```

![t-Distribution Converges to Normal as df Increases](./img/student_t_85.png)

</div>

---

## 왜 t인가?

모집단이 정규이고 $\sigma$를 모를 때 $\sigma$를 $S$로 대체하면:

$$
\frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n-1}
$$

이렇게 되는 이유는 다음과 같다:

1. $\bar{X} \sim N(\mu, \sigma^2/n)$이므로 $\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \sim N(0,1)$이다.
2. $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2_{n-1}$이다.
3. $\bar{X}$와 $S^2$이 **독립**이다(정규분포만의 특별한 성질).
4. 비 $\frac{N(0,1)}{\sqrt{\chi^2_{n-1}/(n-1)}}$은 정의에 의해 $t_{n-1}$이다.

<div class="codebox" markdown>

### 예제 3. 왜 t 분포가 필요한가 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
n, mu, sigma = 10, 0, 10
n_sim = 10_000

# (n, n_sim) 배열이므로 **열 하나가 표본 하나**다. axis=0 으로 집계한다.
samples = np.random.normal(mu, sigma, (n, n_sim))
x_bar = samples.mean(axis=0)
s = samples.std(axis=0, ddof=1)

# t 통계량. 분모에 참 sigma(=10)가 아니라 **표본표준편차 s** 를 넣는 것이 요점이다.
# sigma를 썼다면 이 값은 정확히 N(0,1)을 따랐을 것이다.
# s를 쓰면 분모 자체가 흔들리므로 분포가 넓어지고 꼬리가 두꺼워진다.
# 그 넓어진 정도를 정확히 기술하는 것이 t(n-1) 이다.
t_stats = (x_bar - mu) / (s / np.sqrt(n))

fig, ax = plt.subplots(figsize=(12, 3))
bins = np.arange(-6, 6, 0.1)
ax.hist(t_stats, bins=bins, density=True, alpha=0.7, label=f'Simulated $t_{{{n-1}}}$')
ax.plot(bins, stats.t(n-1).pdf(bins), '--r', lw=2, label=f'$t_{{{n-1}}}$ PDF')
ax.legend()
ax.spines[['top', 'right']].set_visible(False)
plt.show()
```

![스튜던트 t 분포](./img/student_t_118.png)

</div>

---

## t의 역할 이해하기

### n이 클 때: 중심극한정리가 z를 정당화한다

$n$이 크면 $S \approx \sigma$가 되어 $t_{n-1}$과 $N(0,1)$의 차이가 무시할 만하다. 실무에서는 $z$를 써도 똑같이 좋다.

### n이 작을 때: t가 빛나지만 정규성 아래에서만

$t$ 분포는 $n$이 작을 때 가장 중요하다. 두꺼운 꼬리가 $\sigma$ 대신 $S$를 쓰면서 생긴 추가 변동성을 제대로 반영한다. 다만 이 결과는 **모집단이 정규일 때만 정확하다**.

### 정규가 아닌 모집단

모집단이 치우쳐 있거나 꼬리가 두꺼우면 $n$이 작을 때 $t$ 근사는 **나쁘다**. $t$도 $z$도 믿을 수 없으며, 로버스트 방법이나 비모수 방법이 낫다.

### 요약

| 상황 | 권장 방법 |
|:---|:---|
| $n$이 큼 | $z$를 사용한다. $t$ 보정은 무시할 만하다 |
| $n$이 작고 모집단이 정규 | $t$가 정확하며 적절하다 |
| $n$이 작고 모집단이 정규가 아님 | 로버스트/비모수 방법을 사용한다 |

---

## 확률표본

<div class="codebox" markdown>

### 예제 4. t 분포에서 표본추출 { .eg }

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(0)
df = 5
data = stats.t(df).rvs(10_000)

fig, ax = plt.subplots(figsize=(12, 3))
# 구간을 [-5, 5]로 고정한다. t(5)는 꼬리가 두꺼워 표본에 |x| > 20 인 값도 나오는데,
# 자동 구간에 맡기면 그 이상치 때문에 가운데가 한두 칸으로 뭉개진다.
bins = np.linspace(-5, 5, 101)
ax.hist(data, bins=bins, density=True, histtype='step', label='t Samples')
ax.plot(bins, stats.t(df).pdf(bins), '--b', lw=2, label='t PDF')
# 같은 표본의 평균·표준편차로 맞춘 정규분포를 함께 그린다.
# 가운데는 잘 맞지만 꼬리에서 벌어지는 것이 요점이다.
ax.plot(bins, stats.norm(data.mean(), data.std()).pdf(bins),
        '--r', lw=2, label='Normal Approx')
ax.legend()
plt.show()
```

![스튜던트 t 분포](./img/student_t_169.png)

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
급여가 $\mu = \$40{,}000$인 정규분포를 따른다. 표본 $n = 9$, $s = \$8{,}000$일 때 $P(\bar X \ge \$45{,}000)$을 계산하라.

</div>

??? success "풀이"
    $\mathrm{SE} = s/\sqrt n = 8000/3 \approx 2667$. 검정통계량: $t = (45000 - 40000)/2667 \approx 1.875$.

    $t_8$ 아래에서 $P(T \ge 1.875) \approx 0.048$로 약 4.8%이다.

    참고: $\sigma$를 모르고 표본에서 $s$를 추정하여 추가적인 불확실성이 들어오므로 $z$ 대신 $t$를 사용한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
**$t$ 분포 유도.** $Z \sim N(0, 1)$과 $V \sim \chi^2_\nu$가 독립이면 $T = Z/\sqrt{V/\nu} \sim t_\nu$임을 보여라.

</div>

??? success "풀이"
    정의에 의해 $t_\nu$는 독립인 $Z \sim N(0, 1)$과 $V \sim \chi^2_\nu$에 대한 $Z/\sqrt{V/\nu}$의 분포이다.

    PDF의 유도: $V = v$로 조건화한다. $V = v$가 주어지면 $T = Z/\sqrt{v/\nu}$이므로 $T \mid V \sim N(0, \nu/v)$이다. 밀도는:

    $$
    f_{T \mid V}(t \mid v) = \frac{1}{\sqrt{2\pi \nu/v}} e^{-vt^2/(2\nu)}
    $$

    $T$의 주변분포: $V \sim \chi^2_\nu$의 밀도를 사용하여 $v$에 대해 적분한다. 결과는:

    $$
    f_T(t) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)} \left(1 + \frac{t^2}{\nu}\right)^{-(\nu+1)/2}
    $$

    $t$ 밀도는 $\sim t^{-(\nu+1)}$의 다항식 꼬리를 가지며, 정규분포의 $e^{-t^2/2}$보다 두껍다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**$t$는 정규분포에 가까워진다.** $\nu \to \infty$일 때 $t_\nu \to N(0, 1)$임을 보여라.

</div>

??? success "풀이"
    $V \sim \chi^2_\nu$인 $t$의 정의 $T = Z/\sqrt{V/\nu}$에서 출발한다. 큰수의 법칙에 의해 $V/\nu = (1/\nu)\sum_{i=1}^\nu Z_i^2 \to 1$이 확률수렴한다. 따라서 $\sqrt{V/\nu} \to 1$이고 $T \to Z \sim N(0, 1)$이다.

    더 정확히는 Slutsky 정리에 의해 $T = Z/\sqrt{V/\nu} \xrightarrow{d} Z/1 = Z$이다.

    **실무:** $\nu \ge 30$이면 $t$ 분포는 정규분포와 거의 구별되지 않는다. $t_{30}$의 임계값은 $z$의 임계값과 2% 이내로 일치한다. 이것이 $t$ 대신 $z$를 쓰는 $n \ge 30$ 경험 법칙의 근거이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**$z$ 대신 $t$를 쓰는 이유.** 어떤 통계학자가 $z = (\bar X - \mu_0)/(\sigma/\sqrt n)$을 계산하려다 $\sigma$를 모른다는 것을 깨닫고 $s$로 대체했다. 그 결과 $t = (\bar X - \mu_0)/(s/\sqrt n) \sim t_{n-1}$임을 보여라.

</div>

??? success "풀이"
    귀무가설 $\mu = \mu_0$ 아래에서 $\bar X \sim N(\mu_0, \sigma^2/n)$이므로 $Z = (\bar X - \mu_0)/(\sigma/\sqrt n) \sim N(0, 1)$이다.

    표본분산 $s^2$을 $\sigma^2$으로 척도조정하면 카이제곱 분포를 따른다: (정규 자료에 대해) $(n-1)s^2/\sigma^2 \sim \chi^2_{n-1}$.

    정규 자료에서 $\bar X$와 $s^2$은 독립이다(정규성에만 특유한 자명하지 않은 사실이다).

    따라서:

    $$
    t = \frac{\bar X - \mu_0}{s/\sqrt n} = \frac{(\bar X - \mu_0)/(\sigma/\sqrt n)}{\sqrt{((n-1) s^2/\sigma^2)/(n-1)}} = \frac{Z}{\sqrt{V/(n-1)}}
    $$

    이며 $V \sim \chi^2_{n-1}$은 $Z$와 독립이다. $t$의 정의에 의해 이는 $t_{n-1}$이다.

    $\sigma$ 대신 $s$를 쓰면 분모에 카이제곱이 들어온다. $t$ 분포는 정규분포보다 두꺼운 꼬리로 이 추가 불확실성을 반영한다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
**$t$의 두꺼운 꼬리.** $t_3$에 대해 $P(|T| > 2)$와 $P(|T| > 4)$를 계산하고 정규분포와 비교하라.

</div>

??? success "풀이"
    $t_3$: $P(|T| > 2) = 2 \cdot P(T > 2)$. $t_3$ 분포표에서 $P(T > 2) \approx 0.07$이므로 $P(|T| > 2) \approx 0.14$이다.

    $P(|T| > 4) \approx 2 \cdot 0.014 = 0.028$.

    정규분포: $P(|Z| > 2) \approx 0.046$, $P(|Z| > 4) \approx 6 \times 10^{-5}$.

    $|x| = 4$에서 비교하면 정규분포의 확률은 $6 \times 10^{-5}$로 사실상 0인데 $t_3$의 확률은 $0.028$로 400배가 넘는다. $t_3$은 정규분포보다 극단적인 결과에 훨씬 많은 확률을 부여한다.

    **실무적 귀결:** 같은 유의수준에서 소표본 $t$ 검정은 $z$ 검정보다 검정력이 낮다. 두꺼운 꼬리를 상쇄하기 위해 임계값이 더 크기 때문이다. 이는 가정($\sigma$를 안다는 것)과 꼬리에 대한 보수성 사이의 맞바꿈이다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**Welch의 $t$ 검정.** 분산이 다른 독립인 두 표본에 대해 Welch 검정은 $t = (\bar X_1 - \bar X_2)/\sqrt{s_1^2/n_1 + s_2^2/n_2}$를 사용하고 자유도는 Welch–Satterthwaite 공식으로 근사한다. 이 공식을 쓰고 왜 정수가 아닌지 설명하라.

</div>

??? success "풀이"
    **Welch–Satterthwaite 자유도:**

    $$
    \nu_{WS} = \frac{(s_1^2/n_1 + s_2^2/n_2)^2}{(s_1^2/n_1)^2/(n_1 - 1) + (s_2^2/n_2)^2/(n_2 - 1)}
    $$

    이 공식은 선형결합 $s_1^2/n_1 + s_2^2/n_2$의 분포를 자유도 $\nu_{WS}$인 척도조정된 카이제곱으로 근사한다. 근사는 적률맞춤 방식이다. $\chi^2$ 근사의 처음 두 적률을 정확한 분포의 것과 일치시킨다.

    **왜 정수가 아닌가:** $\nu_{WS}$는 *표본* 분산 $s_1^2, s_2^2$에 의존하는데, 이들은 임의의 양의 실수를 취할 수 있다. 우연이 아니고서는 정수가 나오지 않는다.

    구현: $\nu_{WS}$를 내림한(보수적인) 값으로 $t_{\nu_{WS}}$ 임계값을 쓰거나, 정수가 아닌 자유도를 받아들이는 소프트웨어에서는 그대로 사용한다.

    Welch 검정은 R(`t.test`)과 SciPy(`scipy.stats.ttest_ind(equal_var=False)`)에서 **기본 두 표본 $t$ 검정**이다. 원래의 스튜던트 $t$ 검정이 요구하는 등분산 가정을 필요로 하지 않기 때문이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
같은 사람 10명에게 훈련 전후 점수를 재어 차이(후 − 전)를 얻었다.

$$
2.1,\ -0.5,\ 3.2,\ 1.8,\ 0.4,\ 2.6,\ -1.1,\ 1.5,\ 2.9,\ 0.7
$$

훈련 효과가 있는지 유의수준 5%로 검정하고 95% 신뢰구간을 구하라. 이 자료를 두 표본 $t$ 검정으로 다루면 무엇이 문제인가?

</div>

??? success "풀이"
    **대응표본 $t$ 검정**은 차이를 한 표본으로 보고 $\mu_d = 0$을 검정한다.

    $$
    \bar d = 1.360, \qquad s_d = 1.450, \qquad n = 10
    $$

    $$
    t = \frac{1.360}{1.450/\sqrt{10}} = \frac{1.360}{0.4586} = 2.966, \qquad \text{자유도 }9
    $$

    임계값이 $t_{0.975,9} = 2.262$이고 $2.966 > 2.262$이므로 기각한다. p-값은 0.0158이다.

    **신뢰구간.**

    $$
    1.360 \pm 2.262 \times 0.4586 = 1.360 \pm 1.037 = (0.323,\ 2.397)
    $$

    0을 담지 않아 검정과 일치한다.

    **두 표본 검정으로 다루면.** 전과 후를 독립인 두 집단으로 보게 되는데, 같은 사람에게서 나온 값이므로 **독립이 아니다.** 개인차(어떤 사람은 원래 점수가 높고 어떤 사람은 낮다)가 두 집단의 분산에 그대로 들어가 분모를 키우고, 그 결과 검정력이 크게 떨어진다.

    대응 설계의 요점이 바로 이것이다. **차이를 취하면 개인차가 상쇄된다.** 위 자료에서 전후 점수의 산포가 각각 크더라도 차이의 산포는 1.45로 작을 수 있고, 그만큼 작은 효과도 잡아낼 수 있다. 대가는 자유도가 $2n-2 = 18$에서 $n-1 = 9$로 줄어드는 것인데, 상관이 웬만큼 있으면 이 손해보다 이득이 훨씬 크다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
효과크기 $d = (\mu-\mu_0)/\sigma = 0.5$를 유의수준 5%(양측)에서 검정력 80%로 검출하려면 한 표본 $t$ 검정에 표본이 몇 개 필요한가? 근사 공식을 유도하고 이유를 설명하라.

</div>

??? success "풀이"
    $\sigma$를 안다고 보고 $z$ 근사로 시작한다. $H_1$ 아래에서 검정통계량의 평균이 $d\sqrt n$만큼 이동하므로, 검정력 $1-\beta$를 얻으려면

    $$
    d\sqrt n \ge z_{1-\alpha/2} + z_{1-\beta}
    $$

    여야 한다. 따라서

    $$
    n \ge \left(\frac{z_{1-\alpha/2}+z_{1-\beta}}{d}\right)^2 = \left(\frac{1.96+0.84}{0.5}\right)^2 = \left(\frac{2.80}{0.5}\right)^2 = 31.4
    $$

    로 $n = 32$다. $t$ 분포를 쓰면 임계값이 조금 커서 실제로는 $n = 34$가 필요하다(`statsmodels`의 `TTestPower`로 확인할 수 있다).

    **왜 두 분위수를 더하는가.** 그림으로 보면 분명하다. 귀무분포와 대립분포가 $d\sqrt n$만큼 떨어져 있는데, 임계값이 귀무분포 중심에서 $z_{1-\alpha/2}$만큼 오른쪽에 있고, 대립분포의 중심에서 그 임계값까지가 $z_{1-\beta}$만큼 왼쪽이어야 검정력이 $1-\beta$가 된다. 두 거리를 더한 것이 분포 간 거리다.

    **읽어 둘 점.**

    - $n$이 **효과크기의 제곱에 반비례**한다. 효과가 절반이면 표본이 4배 필요하다. $d=0.2$(작은 효과)면 $n \approx 197$, $d=0.8$(큰 효과)면 $n\approx 13$이다(모두 $z$ 근사값).
    - 검정력 90%를 원하면 $z_{0.90}=1.282$를 써서 $n\approx43$으로 는다.
    - 효과크기를 **표준화된 단위**로 지정한다는 점이 중요하다. $\sigma$를 몰라도 "표준편차의 몇 배를 검출하고 싶은가"는 정할 수 있다.

    사후검정력(관측된 효과크기로 계산한 검정력)은 계산하지 말아야 한다. p-값의 단조함수라 새로운 정보가 없고, "유의하지 않았으니 검정력이 낮았다"는 순환논법이 된다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$t$ 검정의 정규성 가정이 의심스러울 때 쓸 수 있는 비모수 대안을 두 가지 들고, 각각이 무엇을 가정하고 무엇을 검정하는지 밝혀라.

</div>

??? success "풀이"
    **(1) 부호검정.** 차이의 **부호**만 본다. $H_0$: 중앙값이 0, 즉 $P(D>0)=1/2$. 양의 차이 개수가 $\text{Binomial}(n, 1/2)$를 따르는지를 검정한다.

    - **가정**: 관측이 독립이고 연속분포(동점이 없음)이면 된다. 분포의 모양에 대한 가정이 없다.
    - **검정 대상**: 중앙값.
    - **연습문제 7 자료**: 양의 차이가 8개, 음의 차이가 2개다. $P(X\ge8) \times 2 = 0.109$로 5%에서 유의하지 않다.
    - **약점**: 차이의 **크기**를 완전히 버리므로 검정력이 낮다. 위에서 $t$ 검정은 유의했는데 부호검정은 그렇지 않은 것이 그 대가다.

    **(2) 윌콕슨 부호순위 검정.** 차이의 절댓값에 순위를 매기고 부호를 붙여 더한다.

    - **가정**: 차이의 분포가 어떤 점을 중심으로 **대칭**이어야 한다. 부호검정보다 강한 가정이지만 정규성보다는 훨씬 약하다.
    - **검정 대상**: 대칭 중심(= 중앙값 = 평균).
    - **장점**: 크기 정보를 순위 형태로 살려 쓰므로 부호검정보다 검정력이 높다. 정규분포에서도 $t$ 검정 대비 점근상대효율이 $3/\pi \approx 0.955$로 손해가 거의 없고, 꼬리가 두꺼운 분포에서는 오히려 $t$ 검정보다 낫다.

    **고르는 기준.** 대칭성을 믿을 수 있으면 윌콕슨이 대체로 최선이다. 심하게 치우친 자료라면 부호검정이나 **부트스트랩**을 쓴다. 부트스트랩은 평균을 그대로 다룰 수 있고 분포 가정이 거의 없다는 장점이 있다.

    한 가지 흔한 오해를 짚어 둔다. **비모수 검정은 "가정이 없는" 검정이 아니다.** 독립성은 여전히 필요하고, 윌콕슨은 대칭성을, 두 표본 맨-휘트니는 두 분포의 모양이 같다는 조건을 (위치이동 해석을 원한다면) 요구한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$t$ 검정은 모집단의 **왜도**와 **첨도** 가운데 어느 쪽에 더 민감한가? 이유를 $t$ 통계량의 구조에서 설명하라.

</div>

??? success "풀이"
    **왜도에 훨씬 민감하다.**

    **왜 그런가.** $t = \sqrt n(\bar X-\mu)/S$에서 분자와 분모가 **독립이 아니라는 것**이 열쇠다. 정규모집단에서는 $\bar X$와 $S^2$이 독립이지만(가이어리 정리), 모집단이 치우쳐 있으면 그렇지 않다.

    구체적으로 $\operatorname{Cov}(\bar X, S^2) = \mu_3/n$이다. 모집단의 3차 중심적률이 곧 공분산을 만든다. 오른쪽으로 치우친 모집단($\mu_3>0$)이면 $\bar X$가 우연히 클 때 $S$도 함께 크게 나온다. 그러면 분자가 커져도 분모가 함께 커져 $t$가 눌리고, 반대로 $\bar X$가 작을 때는 $S$도 작아 $t$가 과장된다.

    그 결과 $t$ 통계량의 분포 자체가 **왼쪽으로 치우친다**(모집단이 오른쪽으로 치우쳤을 때). 에지워스 전개로 보면 오차의 주항이

    $$
    P(T \le t) = \Phi(t) + \frac{\gamma_1}{6\sqrt n}(2t^2+1)\varphi(t) + O(n^{-1})
    $$

    로 **왜도 $\gamma_1$에 비례하고 $n^{-1/2}$ 차수**다.

    **첨도의 영향.** 첨도는 $n^{-1}$ 차수의 항에만 들어간다. 수렴이 한 차수 빠르므로 같은 $n$에서 영향이 훨씬 작다.

    **실무적 귀결.**

    - **단측 검정이 양측보다 위험하다.** 왜도가 만드는 오차는 한쪽 꼬리를 부풀리고 다른 쪽을 줄이므로, 양측에서는 두 오차가 상당 부분 상쇄되지만 단측에서는 그대로 남는다. 치우친 자료에서 명목 5% 단측 검정의 실제 오류율이 8~10%까지 갈 수 있다.
    - **비대칭 신뢰구간이 필요할 수 있다.** 존슨의 왜도 보정이나 부트스트랩-$t$, BCa 부트스트랩이 이 치우침을 고친다.
    - **꼬리가 두꺼운 자료**(왜도는 0이지만 첨도가 큰 경우, 예를 들어 $t_3$)에서는 $t$ 검정의 제1종 오류율이 비교적 잘 유지된다. 문제는 오류율이 아니라 **검정력**이다. 이상치가 $S$를 부풀려 검정력을 떨어뜨리므로, 이때는 절사평균 $t$ 검정이나 윌콕슨 검정이 낫다.

    한 문장으로 줄이면 **치우침은 유의수준을 망가뜨리고, 두꺼운 꼬리는 검정력을 망가뜨린다.**

---

## 정리하며

- $t$ 분포는 $\sigma$를 $S$로 추정하면서 생기는 불확실성을 반영한다.
- 정규분포보다 꼬리가 두꺼우며, 특히 자유도가 작을 때 그렇다.
- $d \to \infty$일 때 $t$ 분포는 $N(0,1)$로 수렴한다.
- $t$ 결과의 정확성은 모집단의 정규성에 결정적으로 의존한다.
