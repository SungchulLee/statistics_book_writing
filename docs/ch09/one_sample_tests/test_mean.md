# 일표본 평균 검정

## 개요

일표본 평균 검정은 모평균 $\mu$가 가설의 값 $\mu_0$과 같은지 판단한다. 모분산을 알 때는 **z-검정**을, 모르고 표본에서 추정할 때는 **t-검정**을 쓴다. 두 검정 모두 자료가 정규분포를 따르거나 표본이 중심극한정리를 적용할 만큼 크다는 가정에 기댄다.

## 검정의 구성

**가설:**

- 양측: $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu \neq \mu_0$
- 단측: $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu > \mu_0$ (또는 $H_1\colon \mu < \mu_0$)

**z-검정** ($\sigma$를 아는 경우): 검정통계량은

$$
Z = \frac{\bar{X} - \mu_0}{\sigma / \sqrt{n}} \sim N(0,1).
$$

**t-검정** ($\sigma$를 모르고 $S$로 추정하는 경우): 검정통계량은

$$
T = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t_{n-1}.
$$

## 코드

```python
import math
from scipy.stats import t as tdist, norm

def test_mean_one_sample(xbar, n, mu0=0.0, sd=None, known_sigma=None,
                         alt="two-sided", alpha=0.05):
    """known_sigma를 주면 z-검정, 아니면 표본 sd로 t-검정.

    원자료가 아니라 요약통계량(xbar, n, sd)만 받는다.
    검정에 필요한 것이 그것뿐이기 때문이다.
    돌려주는 값은 (통계량, p-값, 기각 여부, 이름)이다.
    """
    if known_sigma is not None:
        se = known_sigma / math.sqrt(n)
        z = (xbar - mu0) / se
        if alt == "two-sided":
            # 작은 쪽 꼬리를 골라 두 배 한다. z의 부호를 따지지 않아도 되고
            # 어느 쪽으로 치우쳐도 같은 식이 쓰인다.
            p = 2 * min(norm.cdf(z), 1 - norm.cdf(z))
        elif alt == "less":
            p = norm.cdf(z)
        else:
            p = 1 - norm.cdf(z)
        return z, p, (p < alpha), "z-test"

    if sd is None:
        raise ValueError("Provide sd for t-test or known_sigma for z-test.")
    se = sd / math.sqrt(n)
    df = n - 1               # sd를 자료에서 추정했으므로 자유도 하나를 잃는다
    t = (xbar - mu0) / se
    if alt == "two-sided":
        p = 2 * min(tdist.cdf(t, df), 1 - tdist.cdf(t, df))
    elif alt == "less":
        p = tdist.cdf(t, df)
    else:
        p = 1 - tdist.cdf(t, df)
    return t, p, (p < alpha), f"t-test (df={df})"
```

**보기 1.**

```python
stat, p, reject, label = test_mean_one_sample(
    xbar=3.2, n=25, mu0=3.0, sd=1.1, alt="greater"
)
print(label, "stat:", stat, "p:", p, "reject:", reject)

# 같은 자료를 sigma=1.1을 안다고 가정하고 z-검정으로도 해 본다.
stat_z, p_z, reject_z, label_z = test_mean_one_sample(
    xbar=3.2, n=25, mu0=3.0, known_sigma=1.1, alt="greater"
)
print(label_z, "stat:", stat_z, "p:", p_z, "reject:", reject_z)
```

출력:

```
t-test (df=24) stat: 0.9090909090909097 p: 0.18617076763866547 reject: False
z-test stat: 0.9090909090909097 p: 0.18165107044344886 reject: False
```

통계량은 같고 p-값만 다르다. 산포로 넣은 숫자가 1.1로 같으니 분자와 분모가 같을 수밖에 없고, 달라지는 것은 그 통계량을 어느 분포에 견주느냐뿐이다. $t_{24}$가 정규분포보다 꼬리가 두꺼워 같은 통계량에 더 큰 p-값을 준다. $\sigma$를 모른다는 사실의 값이 여기서는 0.0045만큼이다.

### 해석

이 예제에서는 $\bar{x} = 3.2$, $s = 1.1$, $n = 25$로 $H_0\colon \mu = 3.0$을 $H_1\colon \mu > 3.0$에 대해 검정한다. 검정통계량은

$$
T = \frac{3.2 - 3.0}{1.1/\sqrt{25}} = \frac{0.2}{0.22} \approx 0.909.
$$

자유도 24에서 단측 p-값은 약 0.186이다. 통상적인 $\alpha = 0.05$를 넘으므로 $H_0$을 기각하지 못한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** 관측값 $n = 36$개의 표본에서 $\bar{x} = 52$이고 모표준편차가 $\sigma = 6$으로 알려져 있다. $\alpha = 0.05$에서 $H_0\colon \mu = 50$ 대 $H_1\colon \mu \neq 50$을 검정하라.

</div>

??? success "풀이"

    z-검정통계량은

    $$
    Z = \frac{52 - 50}{6/\sqrt{36}} = \frac{2}{1} = 2.0.
    $$

    양측 p-값은 $2\,P(Z \geq 2.0) = 2(0.0228) = 0.0456$이다. $0.0456 < 0.05$이므로 $H_0$을 기각한다. $\mu \neq 50$이라는 유의한 증거가 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** $n = 10$, $\bar{x} = 15.3$, $s = 2.5$일 때 $\alpha = 0.01$에서 $H_0\colon \mu = 14$ 대 $H_1\colon \mu > 14$를 검정하라.

</div>

??? success "풀이"

    t-검정통계량은

    $$
    T = \frac{15.3 - 14}{2.5/\sqrt{10}} = \frac{1.3}{0.7906} \approx 1.644.
    $$

    $\text{df} = 9$에서 단측 p-값은 $P(T_9 \geq 1.644) \approx 0.068$이다. $0.068 > 0.01$이므로 1% 수준에서 $H_0$을 기각하지 못한다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** $\sigma$를 모를 때 z-검정 대신 t-검정을 쓰는 이유를 설명하라. $n \to \infty$이면 t-분포는 어떻게 되는가?

</div>

??? success "풀이"

    $\sigma$를 모르면 $S$로 추정한다. $S$ 자체가 확률변수이므로 비 $(\bar{X}-\mu_0)/(S/\sqrt{n})$은 표준정규보다 꼬리가 두껍다. $t_{n-1}$ 분포가 이 추가 불확실성을 반영한다. $n \to \infty$이면 대수의법칙에 의해 $S \to \sigma$가 거의 확실하게 성립하므로 $S/\sqrt{n}$이 $\sigma/\sqrt{n}$처럼 행동하고 $t_{n-1} \to N(0,1)$이 된다. 형식적으로 $\nu \to \infty$일 때 $t_\nu \xrightarrow{d} N(0,1)$이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** 유의수준 $\alpha$에서 단측 t-검정 $H_0\colon \mu = \mu_0$ 대 $H_1\colon \mu > \mu_0$의 기각역을 유도하라.

</div>

??? success "풀이"

    $H_0$ 아래에서 $T = (\bar{X} - \mu_0)/(S/\sqrt{n}) \sim t_{n-1}$이다. $T$가 클 때 $H_0$을 기각하고 $H_1\colon \mu > \mu_0$을 택한다. 기각역은

    $$
    T > t_{\alpha,\,n-1},
    $$

    여기서 $t_{\alpha,\,n-1}$은 $t_{n-1}$ 분포의 $(1-\alpha)$ 분위수, 즉 $P(T_{n-1} > t_{\alpha,\,n-1}) = \alpha$인 값이다. 동등하게 p-값 $P(T_{n-1} \geq t_{\text{obs}}) < \alpha$일 때 기각한다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** 어떤 제조사가 강봉의 평균 인장강도가 적어도 5000 psi라고 주장한다. 강봉 $n = 20$개의 표본에서 $\bar{x} = 4917$, $s = 200$을 얻었다. $\alpha = 0.05$에서 이 주장이 뒷받침되는지 검정하라.

</div>

??? success "풀이"

    $H_0\colon \mu \geq 5000$ 대 $H_1\colon \mu < 5000$을 검정한다. 검정통계량은

    $$
    T = \frac{4917 - 5000}{200/\sqrt{20}} = \frac{-83}{44.72} \approx -1.856.
    $$

    $\text{df} = 19$에서 단측 p-값은 $P(T_{19} \leq -1.856) \approx 0.039$이다. $0.039 < 0.05$이므로 $H_0$을 기각한다. 평균 인장강도가 5000 psi보다 작다는 유의한 증거가 있어 제조사의 주장과 어긋난다. $\square$
