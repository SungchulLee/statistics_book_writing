# 도박사의 역설: 큰수의 법칙이 실패할 때

## 개요

큰수의 법칙(LLN)은 표본평균이 모평균으로 수렴함을 보장하지만, 모평균이 **유한할** 때에만 그렇다. **상트페테르부르크 역설**이 고전적인 반례를 제공한다. 기댓값이 무한한 게임에서는 표본평균이 안정되는 대신 발산한다. 이 절에서는 이 역설을 모의실험하고, 큰수의 법칙이 성립하는 유계 변형과 대비한다.

---

## 1. 상트페테르부르크 게임

### 규칙

공정한 동전을 첫 앞면이 나올 때까지 반복해서 던진다. $k$번째 던지기에서 처음 앞면이 나오면 $2^k$달러를 받는다.

기대 상금은 다음과 같다.

$$
E[X] = \sum_{k=1}^{\infty} 2^k \cdot \left(\frac{1}{2}\right)^k = \sum_{k=1}^{\infty} 1 = \infty
$$

$E[X] = \infty$이므로 큰수의 법칙이 적용되지 않는다. 표본평균이 수렴할 유한한 값이 없다.

### 역설

기댓값이 무한한데도 개별 라운드는 대부분 아주 적게 지급한다(50%는 \$2, 75%는 \$4 이하). 그러나 뒷면이 길게 이어지는 드문 사건이 어마어마한 상금을 낳아 평균을 지배한다. 아무리 여러 라운드를 해도 극단적인 결과 하나가 표본평균을 극적으로 바꿀 수 있다.

---

## 2. 모의실험: 무한한 평균

독립적인 수열 100개를 각각 최대 10,000라운드까지 진행하며 진행 중인 표본평균을 추적한다.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def st_petersburg_sample_means(n_max=10_000, tries=100, n_grid=200):
    """Each round: first heads on flip k → win 2^k. E[X] = infinity."""
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = 2.0 ** flips
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results

infinite_results = st_petersburg_sample_means()
```

---

## 3. 유계 변형: 유한한 평균

이제 상금을 $2^{10} = 1024$달러로 제한하자. 이렇게 절단하면 $E[X]$가 유한해져 큰수의 법칙이 적용된다.

$$
E[X_{\text{bounded}}] = \sum_{k=1}^{10} 2^k \cdot \left(\frac{1}{2}\right)^k + 1024 \cdot \sum_{k=11}^{\infty} \left(\frac{1}{2}\right)^k = 10 + 1024 \cdot \frac{1}{1024} = 11
$$

```python
def bounded_game_sample_means(n_max=10_000, tries=100, n_grid=200):
    """Same game but capped at 2^10 = 1024. E[X] is now finite."""
    n_vals = np.unique(np.logspace(1, np.log10(n_max), n_grid).astype(int))
    results = []
    for n in n_vals:
        flips = np.random.geometric(0.5, size=(n, tries))
        winnings = np.minimum(2.0 ** flips, 1024.0)
        means = winnings.mean(axis=0)
        results.append((n, means))
    return results

bounded_results = bounded_game_sample_means()
```

---

## 4. 시각화

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: infinite mean — divergence
ax = axes[0]
for n, means in infinite_results:
    ax.loglog(n * np.ones(len(means)), means, ".", color="black", ms=2, alpha=0.5)
ax.set_xlabel("n (number of rounds)")
ax.set_ylabel("Sample mean of winnings")
ax.set_title("St. Petersburg Game (E[X] = ∞)\nSample mean does NOT converge")

# Panel 2: finite mean — convergence
ax = axes[1]
for n, means in bounded_results:
    ax.semilogx(n * np.ones(len(means)), means, ".", color="steelblue", ms=2, alpha=0.5)
true_mean = 11.0
ax.axhline(true_mean, color="red", linestyle="--", lw=2,
           label=f"E[X] = {true_mean:.1f}")
ax.set_xlabel("n (number of rounds)")
ax.set_ylabel("Sample mean of winnings")
ax.set_title("Bounded Game (E[X] < ∞)\nSample mean converges (LLN)")
ax.legend()

plt.tight_layout()
plt.show()
```

---

## 5. 해석

### 왼쪽 패널 (무한한 평균)

표본평균의 구름이 $n$이 커져도 좁아지지 **않고** 로그–로그 척도에서 계속 퍼진다. $n = 10{,}000$에서도 모의실험 실행마다 평균이 크게 다르다. 이것이 큰수의 법칙 실패의 특징이다. 모평균이 무한하면 표본평균은 일치성 있는 추정량이 아니다.

### 오른쪽 패널 (유한한 평균)

표본평균의 구름이 $n$이 커질수록 빨간 선($E[X] = 11$) 주위로 **수축한다**. $n = 10{,}000$쯤이면 100번의 모의실험이 사실상 모두 11 근처의 값에 일치한다. 큰수의 법칙이 기대대로 작동하는 모습이다.

### 결정적인 조건

두 패널의 차이는 단 하나의 수학적 조건, 즉 **유한한 기댓값**이다. 상트페테르부르크 게임과 그 유계 변형은 꼬리의 행동만 다른데도 질적으로 정반대인 통계적 행동을 낳는다.

!!! warning "유한한 평균은 선택 사항이 아니다"
    큰수의 법칙은 흔히 "평균은 수렴한다"고 느슨하게 진술된다. 이는 오도한다. 평균은 모평균이 존재하고 유한할 때에만 수렴한다. 금융, 보험, 네트워크 트래픽의 꼬리가 두꺼운 분포는 이 조건을 위반할 수 있다.

---

## 6. 이론과의 연결

큰수의 법칙에는 두 형태가 있다.

- **약한 큰수의 법칙(WLLN):** 유한한 분산을 요구한다(또는 더 약하게, 절단 논증을 통해 유한한 평균). 확률수렴을 준다.
- **강한 큰수의 법칙(SLLN):** 유한한 평균($E[|X|] < \infty$)만 요구한다. 거의 확실한 수렴을 준다.

상트페테르부르크 게임은 $E[X] = \infty$이므로 어느 형태도 적용되지 않는다. 유계 변형은 평균과 분산이 모두 유한하므로 두 형태가 다 성립한다.

!!! note "두꺼운 꼬리와 무한한 평균"
    어떤 분포는 꼬리가 두꺼워도(예: $\alpha > 1$인 파레토) 평균이 유한할 수 있으며, 그런 경우 큰수의 법칙이 적용된다. 구분해야 할 것은 두꺼운 꼬리(느린 감쇠)와 적분 불가능한 꼬리(무한한 평균)다. 큰수의 법칙을 깨뜨리는 것은 후자뿐이다.

---

## 연습문제

**연습문제 1.**
정의로부터 상트페테르부르크 게임 상금의 기댓값을 계산하라. 급수의 표준 수렴 판정법은 어느 단계에서 실패하는가?

??? success "연습문제 1 풀이"
    상금은 $K \sim \text{Geometric}(1/2)$일 때 $X = 2^K$이다. 기댓값은

    $$
    E[X] = \sum_{k=1}^{\infty} 2^k \cdot P(K = k) = \sum_{k=1}^{\infty} 2^k \cdot \frac{1}{2^k} = \sum_{k=1}^{\infty} 1
    $$

    이다. 이것은 $1 + 1 + 1 + \cdots$ 형태의 급수로 발산한다. 항이 0으로 가지 않으므로 기본적인 발산 판정법($a_k \not\to 0$이면 $\sum a_k$가 발산)만으로도 기댓값이 무한함이 확인된다.

---

**연습문제 2.**
어떤 정수 $M \ge 1$에 대해 상금을 $2^M$으로 제한한다고 하자. $E[X_{\text{bounded}}]$를 $M$의 함수로 나타내는 공식을 유도하라.

??? success "연습문제 2 풀이"
    $k \le M$이면 상금이 $2^k$이고 확률이 $(1/2)^k$다. $k > M$이면 상금이 $2^M$이고 확률이 $(1/2)^k$다. 따라서

    $$
    E[X_{\text{bounded}}] = \sum_{k=1}^{M} 2^k \cdot \frac{1}{2^k} + 2^M \sum_{k=M+1}^{\infty} \frac{1}{2^k}
    $$

    이다. 첫 합은 $M$이다. 둘째 합은 기하급수로

    $$
    2^M \cdot \frac{1/2^{M+1}}{1 - 1/2} = 2^M \cdot \frac{1}{2^M} = 1
    $$

    이다. 따라서 $E[X_{\text{bounded}}] = M + 1$이다.

    $M = 10$이면 $E[X] = 11$로 모의실험과 일치한다.

---

**연습문제 3.**
모수가 $\alpha$인 파레토분포는 $x \ge 1$에서 밀도가 $f(x) = \alpha / x^{\alpha+1}$이다. $\alpha$가 어떤 값일 때 $E[X]$가 존재하는가? $\text{Var}(X)$는 어떤 값일 때 존재하는가? 각 경우에 큰수의 법칙의 어떤 형태가 적용되는가?

??? success "연습문제 3 풀이"
    $r$차 적률은

    $$
    E[X^r] = \int_1^{\infty} \frac{\alpha \, x^r}{x^{\alpha+1}} \, dx = \alpha \int_1^{\infty} x^{r - \alpha - 1} \, dx
    $$

    이며, $r - \alpha - 1 < -1$, 즉 $r < \alpha$일 때에 한해 수렴한다.

    - $E[X]$가 존재할 필요충분조건은 $\alpha > 1$이다. 강한 큰수의 법칙이 적용된다.
    - $\text{Var}(X)$가 존재할 필요충분조건은 $E[X^2] < \infty$, 즉 $\alpha > 2$이다. (유한 분산의) 약한 큰수의 법칙이 적용되고 중심극한정리도 적용된다.
    - $1 < \alpha \le 2$이면 평균은 유한하지만 분산이 무한하다. 강한 큰수의 법칙은 여전히 성립하지만 중심극한정리는 표준 형태로는 적용되지 않는다(안정분포를 쓰는 일반화된 중심극한정리가 필요하다).
    - $\alpha \le 1$이면 평균이 무한하고 어느 큰수의 법칙도 적용되지 않는다.

---

**연습문제 4.**
분산 $\sigma^2$이 유한하다고 가정하고 체비쇼프 부등식을 이용해 약한 큰수의 법칙을 증명하라.

??? success "연습문제 4 풀이"
    $X_1, \ldots, X_n$이 평균 $\mu$, 분산 $\sigma^2$인 i.i.d.라 하자. 그러면 $E[\bar{X}] = \mu$이고 $\text{Var}(\bar{X}) = \sigma^2 / n$이다.

    체비쇼프 부등식에 의해

    $$
    P(|\bar{X} - \mu| \ge \varepsilon) \le \frac{\text{Var}(\bar{X})}{\varepsilon^2} = \frac{\sigma^2}{n\varepsilon^2}
    $$

    이다. $n \to \infty$이면 고정된 임의의 $\varepsilon > 0$에 대해 우변이 0으로 가므로

    $$
    P(|\bar{X} - \mu| \ge \varepsilon) \to 0
    $$

    이다. 이것이 바로 확률수렴 $\bar{X} \xrightarrow{P} \mu$이다. $\square$

---

**연습문제 5.**
모의실험에서 유계 게임은 복원 표집(`np.random.geometric`)을 쓴다. 대신 $n$라운드의 고정된 수열을 진행하며 진행 중인 평균을 계산한다면 그림이 달라지겠는가? 모의실험 설계와 한 도박사의 경험 사이의 차이를 설명하라.

??? success "연습문제 5 풀이"
    모의실험은 각 격자점마다 길이 $n$인 **독립적인** 수열 100개를 뽑아 각각의 표본평균을 그린다. 이는 $\bar{X}_n$의 **표본분포**, 즉 가상의 여러 도박사에 걸친 변동성을 보여준다.

    한 도박사가 $n$라운드를 진행하면 진행 중인 평균의 경로 하나가 나온다. 이 경로는 강한 큰수의 법칙이 보장하는 **거의 확실한** 수렴을 보여줄 것이다. 하나의 궤적이 결국 $E[X]$ 근처에서 안정된다.

    그림은 달라진다. 각 $n$에서의 점 구름 대신 100개의 개별 궤적(선)이 각각 참 평균으로 수렴하는 모습이 보일 것이다. 구름 표현은 추정량의 **분포**를 강조하고, 궤적 표현은 **경로별** 행동을 강조한다. 둘 다 큰수의 법칙을 보여주지만 서로 보완적인 관점에서 그렇게 한다.

---

**연습문제 6.**
**도박사의 오류**는 나쁜 결과가 이어진 뒤에는 "좋은 결과가 나올 차례"라고 믿는 잘못된 생각이다. 큰수의 법칙이 이를 정당화하지 *않음*을 형식적으로 보여라. "평균은 기댓값으로 수렴한다"의 올바른 해석을 진술하라.

??? success "연습문제 6 풀이"
    $X_1, X_2, \ldots$를 공정한 동전 던지기 i.i.d.라 하자. 각 $X_i$가 독립이므로 이력과 무관하게 $P(X_{n+1} = \text{H} \mid X_1, \ldots, X_n) = P(X_{n+1} = \text{H}) = 1/2$이다. 뒷면이 열 번 연속 나왔어도 열한 번째는 여전히 50 대 50이다. 동전에는 기억이 없다.

    **큰수의 법칙이 실제로 말하는 것:** $\bar X_n \to \mu$가 거의 확실하게 성립한다. *평균*이 $\mu$에 가까워진다. 그러나 *합* $\sum X_i - n\mu$는 0으로 돌아오지 않는다. **반복로그의 법칙**에 의해 거의 확실하게 $\limsup |\sum X_i - n\mu|/\sqrt{2n\log\log n} = \sigma$이다. 합의 변동은 $\sqrt n$처럼 커지며 유계가 아니다.

    따라서 뒷면 열 번 뒤에 평균 $\bar X_{10} = -1$은 더 많이 던지면서 $\mu$ 쪽으로 이동하지만, 그것은 *미래의 던지기가 보상해서가 아니다*. 앞선 뒷면 열 번이 교정되는 것이 아니라 희석될 뿐이다. 모든 던지기는 홀로 선다.

    **도박사의 오류**는 이를 "이제 뒷면이 나에게 앞면을 빚졌다"로 오해한다. 카지노는 룰렛, 슬롯머신, 복권 전략에서 이를 이용한다. 올바른 진술은 이렇다. *장기 빈도*는 *확률*과 같지만, 특정한 짧은 구간이 무언가를 "빚지고" 있는 것은 아니다.

---

**연습문제 7.**
**유계** 확률변수($|X_i| \le M$)에 대해 $\{|\bar X_n - \mu| > \varepsilon\}$에 **보렐–칸텔리 보조정리**를 적용하여 강한 큰수의 법칙이 따라옴을 보여라.

??? success "연습문제 7 풀이"
    (범위가 $\le 2M$인 유계 $X_i$에 대한) 회프딩 부등식에 의해

    $$
    P(|\bar X_n - \mu| > \varepsilon) \le 2 e^{-n\varepsilon^2/(2M^2)}
    $$

    이다. $n$에 대해 합하면 임의의 $\varepsilon > 0$에 대해 $\sum_n P(|\bar X_n - \mu| > \varepsilon) < \infty$이다(기하 꼬리라 합할 수 있다).

    **보렐–칸텔리 보조정리 I**에 의해 $P(|\bar X_n - \mu| > \varepsilon \text{ 가 무한히 자주}) = 0$이다. 따라서 확률 1로 이런 사건이 유한 번만 일어나며, 이는 $\bar X_n \to \mu$가 거의 확실하게 성립함과 동등하다. $\square$

    콜모고로프의 일반적인 강한 큰수의 법칙은 1차 적률의 유한성만 요구하지만 증명이 더 섬세하다(절단 논증). 유계성 가정 아래에서는 회프딩–보렐–칸텔리 증명이 가장 깔끔한 길이다.
