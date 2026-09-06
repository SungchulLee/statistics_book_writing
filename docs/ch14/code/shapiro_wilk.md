# Shapiro-Wilk 검정

## 개요

Shapiro-Wilk 검정은 작은 표본과 중간 표본에서 가장 강력한 정규성 검정으로 널리 인정받는다. 정렬된 표본값이 기대되는 정규 순서통계량과 얼마나 잘 맞는지를 재어 0과 1 사이의 통계량 $W$를 만든다. $W$가 1에 가까우면 정규성과 일관되고, 유의하게 작으면 기각으로 이어진다. SciPy는 $n \leq 5000$으로 검정을 제한한다.

## 검정통계량

순서통계량 $X_{(1)} \leq X_{(2)} \leq \cdots \leq X_{(n)}$이 주어졌을 때 Shapiro-Wilk 통계량은

$$
W = \frac{\left(\sum_{i=1}^{n} a_i\, X_{(i)}\right)^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2},
$$

여기서 계수 $a_1, \ldots, a_n$은 정규 순서통계량의 기댓값과 공분산행렬에서 유도된다. 구체적으로 $m = (m_1, \ldots, m_n)^T$를 표준정규 순서통계량의 기댓값 벡터, $V$를 그 공분산행렬이라 하면

$$
a = \frac{V^{-1} m}{(m^T V^{-1} V^{-1} m)^{1/2}}.
$$

이 정규화에 의해 $\sum_i a_i^2 = 1$이다. 또한 $m$이 반대칭이므로($m_i = -m_{n+1-i}$) $a$도 반대칭이고 따라서 $\sum_i a_i = 0$이다. 두 성질 모두 뒤에서 쓰인다.

## 직관

분자 $(\sum a_i X_{(i)})^2$은 순서통계량의 가중 선형결합으로, 정규성 아래에서 분모(총제곱합)를 가깝게 따라가야 한다. 자료가 정규이면 정렬된 값들이 기대 정규 순서통계량과 잘 정렬되어 $W \approx 1$이 된다. 치우침, 두꺼운 꼬리, 다봉성 같은 정규성 이탈은 이 정렬을 깨뜨려 $W$를 줄인다.

## 가설

$$
H_0: X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2), \qquad H_1: \text{자료가 정규분포를 따르지 않는다}.
$$

$W$가 작으면(동등하게 $p$값이 작으면) $H_0$을 기각한다.

### 코드

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=240),
                    rng.lognormal(0, 0.6, size=60)])

W, p = stats.shapiro(x)
g1 = stats.skew(x, bias=False)
g2 = stats.kurtosis(x, fisher=True, bias=False)

print(f"Sample size n = {x.size}")
print(f"Shapiro-Wilk: W = {W:.4f}, p-value = {p:.4g}")
print(f"Skewness g1 = {g1:.4f}, Excess kurtosis g2 = {g2:.4f}")
if p < 0.05:
    print("=> Reject normality at alpha = 0.05.")
else:
    print("=> Fail to reject normality at alpha = 0.05.")
```

출력:

```text
Sample size n = 300
Shapiro-Wilk: W = 0.9750, p-value = 4.27e-05
Skewness g1 = 0.3707, Excess kurtosis g2 = 1.8565
=> Reject normality at alpha = 0.05.
```

## 강점과 한계

**강점:**

- $n \leq 5000$에서 흔히 쓰는 정규성 검정 중 검정력이 가장 높다.
- 폭넓은 대립가설(치우침, 첨도, 다봉성)에 민감하다.
- 임계값과 $p$값 근사가 잘 확립되어 있다.

**한계:**

- SciPy는 $n > 5000$에서 $p$값이 부정확할 수 있다고 경고한다.
- $n$이 아주 크면 사소한 이탈에도 기각한다. 이때는 시각적 방법과 효과 크기가 더 유익하다.
- 이탈의 *유형*을 알려주지 않는다(진단에는 Q-Q 그림과 짝지어 쓰라).

## 해석

혼합 예제에서 대수정규 성분이 치우침과 두꺼운 꼬리를 들여온다. $W = 0.975$로 1보다 뚜렷하게 작고 $p = 4.3 \times 10^{-5}$이다.

$W = 0.975$가 "1에 가깝다"고 느껴질 수 있으나, $W$의 척도는 그렇게 읽으면 안 된다. 정규자료 $n = 300$에서 $W$의 귀무분포는 0.99 근처에 매우 좁게 몰려 있으므로 $0.975$는 상당히 작은 값이다. $W$와 함께 보조 요약값($g_1 = 0.37$, $g_2 = 1.86$)을 보고하면 왜 정규성이 기각되었는지(주로 두꺼운 꼬리) 독자가 이해할 수 있다.

## 연습문제

**연습문제 1.** 표준정규 관측값 $n = 100$개를 생성하라. $W$와 $p$값을 계산하라. 20회 반복하여 $W$ 값들의 범위를 보고하라.

??? success "연습문제 1 풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    W_values, p_values = [], []
    for trial in range(20):
        x = rng.normal(0, 1, size=100)
        W, p = stats.shapiro(x)
        W_values.append(W)
        p_values.append(p)

    print(f"W range: [{min(W_values):.4f}, {max(W_values):.4f}]")
    print(f"Mean W: {np.mean(W_values):.4f}")
    print(f"Rejections at 0.05: {sum(p < 0.05 for p in p_values)} / 20")
    ```

    출력:

    ```text
    W range: [0.9714, 0.9950]
    Mean W: 0.9847
    Rejections at 0.05: 3 / 20
    ```

    $n = 100$인 정규자료에서 $W$는 $0.971$~$0.995$ 범위에 있고 평균은 $0.985$이다. $W$가 1에 매우 가까운 좁은 구간에 몰린다는 점이 중요하다. $W = 0.97$은 "거의 1"이 아니라 이미 분포의 아래쪽 꼬리에 있다.

    !!! warning "20회 중 3회가 기각되었다"
        자료가 모두 참으로 정규인데도 세 번(시행 2, 3, 4번)이 $\alpha = 0.05$에서 기각했다. 기댓값은 $20 \times 0.05 = 1$회이므로 3회는 많은 편이다. 이항분포로 계산하면 $P(X \geq 3) = 0.075$이니 놀랄 일까지는 아니고 우연의 범위 안이다.

        더 중요한 교훈은 **다중검정**이다. 20번의 독립 검정에서 적어도 한 번 잘못 기각할 확률은

        $$
        1 - 0.95^{20} = 0.64
        $$

        로 64%에 이른다. 여러 변수나 여러 집단에 정규성 검정을 반복 적용할 때는 이 점을 반드시 고려해야 한다. 다중비교 보정 없이 "어느 집단 하나가 정규성 검정에 실패했다"고 결론짓는 것은 위험하다. $\square$

---

**연습문제 2.** (a) $\mathcal{N}(0,1)$, (b) $t_5$, (c) $\text{Lognormal}(0, 0.5)$, (d) $\text{Uniform}(0,1)$에서 뽑은 관측값 $n = 50$개에 대한 Shapiro-Wilk $p$값을 비교하라.

??? success "연습문제 2 풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(42)
    distributions = {
        "N(0,1)": rng.normal(0, 1, 50),
        "t(5)": rng.standard_t(5, 50),
        "Lognormal(0,0.5)": rng.lognormal(0, 0.5, 50),
        "Uniform(0,1)": rng.uniform(0, 1, 50),
    }

    for name, x in distributions.items():
        W, p = stats.shapiro(x)
        print(f"{name:>20}: W = {W:.4f}, p = {p:.4g}")
    ```

    출력:

    ```text
                  N(0,1): W = 0.9841, p = 0.7301
                    t(5): W = 0.9769, p = 0.4281
        Lognormal(0,0.5): W = 0.7623, p = 1.316e-07
            Uniform(0,1): W = 0.9180, p = 0.001999
    ```

    | 분포 | $W$ | $p$ | 판정 |
    |---|---|---|---|
    | $\mathcal{N}(0,1)$ | 0.9841 | 0.730 | 기각 못 함 (정답) |
    | $t_5$ | 0.9769 | 0.428 | **기각 못 함 (놓침)** |
    | $\text{Lognormal}(0,0.5)$ | 0.7623 | $1.3 \times 10^{-7}$ | 기각 (정답) |
    | $\text{Uniform}(0,1)$ | 0.9180 | 0.0020 | 기각 (정답) |

    치우친 대수정규는 압도적으로 기각되고, 저첨인 균등분포도 기각된다. 서로 다른 유형의 이탈에 모두 민감함을 보여준다.

    그러나 **$t_5$는 놓친다**($p = 0.428$). $t_5$의 초과첨도는 $6$으로 결코 작지 않은데도 $n = 50$에서는 탐지되지 않았다. 이는 우연이 아니라 전형적인 결과이다. 연습문제 4에서 보듯 이 조건의 검정력은 0.36에 불과하다. 곧 $t_5$ 표본의 64%가 놓친다.

    두꺼운 꼬리는 치우침보다 탐지하기 어렵다. 정보가 소수의 극단 관측값에만 담겨 있기 때문이다. $\square$

---

**연습문제 3.** 항상 $W \leq 1$이 성립하는 이유를 설명하라. 어떤 조건에서 $W = 1$이 정확히 성립하는가?

??? success "연습문제 3 풀이"

    핵심은 계수 $a$의 두 성질이다. $\sum_i a_i = 0$이고 $\sum_i a_i^2 = 1$이다.

    첫 번째 성질에서 $\sum_i a_i \bar{X} = 0$이므로 분자를 중심화된 형태로 다시 쓸 수 있다.

    $$
    \sum_{i=1}^n a_i X_{(i)} = \sum_{i=1}^n a_i \bigl(X_{(i)} - \bar{X}\bigr).
    $$

    이제 Cauchy-Schwarz 부등식을 적용한다.

    $$
    \left(\sum_i a_i (X_{(i)} - \bar{X})\right)^2 \leq \left(\sum_i a_i^2\right)\left(\sum_i (X_{(i)} - \bar{X})^2\right) = \sum_i (X_i - \bar{X})^2.
    $$

    마지막 등호는 $\sum a_i^2 = 1$이고 정렬은 제곱합을 바꾸지 않기 때문이다. 양변을 $\sum (X_i - \bar{X})^2$으로 나누면 $W \leq 1$을 얻는다.

    **등호 조건.** Cauchy-Schwarz의 등호는 두 벡터가 평행할 때, 곧 어떤 상수 $c$에 대해

    $$
    X_{(i)} - \bar{X} = c\, a_i \quad (\forall i)
    $$

    일 때 성립한다. 곧 중심화된 정렬 자료가 계수 벡터 $a$의 정확한 배수여야 한다.

    연속형 자료에서 이 사건의 확률은 0이므로 $W = 1$은 실제로 관측되지 않는다. $W$는 1보다 엄격히 작되 정규자료에서는 1에 매우 가깝다. $\square$

---

**연습문제 4.** $t_5$ 분포에서 뽑은 관측값 $n = 50$개에 대해 Shapiro-Wilk 검정과 ($\mathcal{N}(0,1)$에 대한) KS 검정의 검정력을 비교하는 몬테카를로 연구를 수행하라.

??? success "연습문제 4 풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n, reps, alpha = 50, 5000, 0.05
    rej_sw, rej_ks = 0, 0

    for _ in range(reps):
        x = rng.standard_t(df=5, size=n)
        _, p_sw = stats.shapiro(x)
        _, p_ks = stats.kstest(x, 'norm', args=(0, 1))
        if p_sw < alpha:
            rej_sw += 1
        if p_ks < alpha:
            rej_ks += 1

    print(f"Shapiro-Wilk power: {rej_sw / reps:.4f}")
    print(f"KS power:           {rej_ks / reps:.4f}")
    ```

    출력:

    ```text
    Shapiro-Wilk power: 0.3616
    KS power:           0.0622
    ```

    Shapiro-Wilk의 검정력 $0.362$가 KS의 $0.062$보다 **약 5.8배** 높다. 두꺼운 꼬리 대립가설에 대해 가장 강력한 범용 정규성 검정이라는 평판을 확인해 준다.

    두 숫자를 해석할 때 주의할 점이 있다.

    - KS의 $0.062$는 명목 크기 $0.05$보다 겨우 조금 클 뿐이다. 사실상 **탐지 능력이 없다**. 완전히 지정된 $\mathcal{N}(0,1)$에 대한 검정임에도 그렇다.
    - Shapiro-Wilk의 $0.362$도 절대 기준으로는 높지 않다. $t_5$ 표본의 **64%를 놓친다**. 초과첨도가 6인 분포인데도 $n = 50$으로는 부족하다는 뜻이다.

    실무적 교훈: 작은 표본에서 정규성 검정이 기각하지 못했다고 해서 안심해서는 안 된다. 검정력이 낮아서일 가능성이 크다. $\square$

---

**연습문제 5.** 큰 표본에서는 자료가 정규성에서 조금만 벗어나도 Shapiro-Wilk 검정이 유의하지 않게 나오기가 거의 불가능하다. 이런 상황에서 $p$값보다 $W$ 통계량 자체를 (효과 크기로) 보고하는 편이 더 유익한 이유를 논하라.

??? success "연습문제 5 풀이"

    $n$이 커지면 Shapiro-Wilk 검정의 검정력이 점점 커진다. $H_0$ 아래에서 $W$의 분산이 줄어들고, 아주 작은 이탈(예: $\gamma_2 = 0.1$)만으로도 $W$가 임계값 아래로 안정적으로 떨어진다. 정규가 아닌 어떤 분포든, 정규에 아무리 가깝든 $p$값이 0으로 간다. 그러면 $p$값이 실용적으로 무의미해진다. "표본이 이탈을 탐지할 만큼 크다"는 말밖에 해주지 않기 때문이다.

    반면 $W$ 통계량 자체는 이탈의 *크기*를 전달한다. $W = 0.998$이면서 $p < 0.001$이라면, 통계적으로 유의하더라도 자료의 모양이 정규에 극도로 가깝다는 뜻이다. 반대로 $W = 0.92$는 훨씬 실질적인 이탈을 뜻한다.

    $W$를 $g_1$, $g_2$와 함께 보고하면 분석자가 그 편차가 자신의 응용에서 실질적으로 의미 있는지 판단할 수 있다(예: $t$ 검정이 여전히 근사적으로 타당할지).

    척도 감각을 위한 참고: 연습문제 1에서 보았듯 정규자료 $n = 100$에서 $W$는 $0.97$~$0.995$ 범위에 있다. 곧 $W$의 "정상 범위"는 매우 좁으므로 $W = 0.95$처럼 얼핏 1에 가까워 보이는 값도 실은 큰 이탈이다. $W$를 효과 크기로 읽을 때는 이 척도를 염두에 두어야 한다. $\square$
