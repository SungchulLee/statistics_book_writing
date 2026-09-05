# 베이즈 정리

## 개요

**베이즈 정리**는 새로운 증거를 관측했을 때 확률을 갱신하는 체계적인 방법을 제공한다. 조건의 방향을 뒤집어, $P(B \mid A)$가 주어졌을 때 $P(A \mid B)$를 계산한다. 이 정리는 베이즈 통계의 토대이며 의학 진단, 스팸 필터링, 기계학습, 금융에 널리 응용된다.

---

## 진술

$P(B) > 0$인 사건 $A$와 $B$에 대해

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

이다. 각 항에는 고유한 이름이 있다.

$$
\underbrace{P(A \mid B)}_{\text{Posterior}} = \frac{\overbrace{P(B \mid A)}^{\text{Likelihood}} \cdot \overbrace{P(A)}^{\text{Prior}}}{\underbrace{P(B)}_{\text{Evidence}}}
$$

---

## 유도

조건부확률의 정의에서 출발하면

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

이다. 분모는 흔히 전확률의 법칙으로 전개한다.

$$
P(B) = P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)
$$

그러면 전개된 형태를 얻는다.

$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid A^c) \cdot P(A^c)}
$$

---

## 일반형 (여러 가설)

$A_1, A_2, \ldots, A_n$이 표본공간 $\Omega$를 분할하면

$$
P(A_i \mid B) = \frac{P(B \mid A_i) \cdot P(A_i)}{\sum_{j=1}^{n} P(B \mid A_j) \cdot P(A_j)}
$$

이다.

---

## 예제

### 예: 의학 진단

어떤 질병이 인구의 1%에 발생한다. 검사의 민감도는 95%, 특이도는 90%다. 어떤 사람이 양성 판정을 받았다면 실제로 그 질병이 있을 확률은 얼마인가?

$$
\begin{aligned}
P(\text{disease} \mid \text{positive}) &= \frac{P(\text{positive} \mid \text{disease}) \cdot P(\text{disease})}{P(\text{positive})} \\[6pt]
&= \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.10 \times 0.99} \\[6pt]
&= \frac{0.0095}{0.1085} \approx 0.0876
\end{aligned}
$$

꽤 정확해 보이는 검사인데도 양성 결과가 실제로 질병이 있을 확률은 8.76%에 불과하다. 이 반직관적인 결과는 질병이 드물어서 양성 대부분이 거짓양성이기 때문에 생긴다.

### 예: 항아리에서 공 뽑기

항아리 두 개가 있다. A 항아리에는 빨간 공 3개와 파란 공 7개가, B 항아리에는 빨간 공 8개와 파란 공 2개가 있다. 항아리를 무작위로(50/50) 골라 빨간 공을 뽑았다. 그것이 B 항아리에서 나왔을 확률은 얼마인가?

$$
\begin{aligned}
P(B \mid \text{red}) &= \frac{P(\text{red} \mid B) \cdot P(B)}{P(\text{red} \mid A) \cdot P(A) + P(\text{red} \mid B) \cdot P(B)} \\[6pt]
&= \frac{0.8 \times 0.5}{0.3 \times 0.5 + 0.8 \times 0.5} = \frac{0.40}{0.55} \approx 0.727
\end{aligned}
$$

### 예: 스팸 필터링

이메일의 40%가 스팸이라고 하자. "free"라는 단어는 스팸 이메일의 80%, 비스팸 이메일의 10%에 나타난다. 어떤 이메일에 "free"가 들어 있다면 그것이 스팸일 확률은 얼마인가?

$$
P(\text{spam} \mid \text{"free"}) = \frac{0.80 \times 0.40}{0.80 \times 0.40 + 0.10 \times 0.60} = \frac{0.32}{0.38} \approx 0.842
$$

---

## 파이썬으로 살펴보기

```python
import numpy as np

def bayes_theorem(prior, likelihood, evidence):
    """Apply Bayes' theorem."""
    posterior = (likelihood * prior) / evidence
    return posterior

# Medical diagnosis example
prior_disease = 0.01
sensitivity = 0.95
specificity = 0.90
false_positive_rate = 1 - specificity

p_positive = sensitivity * prior_disease + false_positive_rate * (1 - prior_disease)
posterior = bayes_theorem(prior_disease, sensitivity, p_positive)

print(f"P(disease | positive) = {posterior:.4f}")
print(f"Despite a 95% sensitive test, only {posterior*100:.1f}% of positives truly have the disease.")
```

```python
import numpy as np
import matplotlib.pyplot as plt

def bayes_update_visualization():
    """Visualize how the posterior changes with prevalence."""
    prevalences = np.linspace(0.001, 0.5, 200)
    sensitivity = 0.95
    specificity = 0.90

    posteriors = []
    for prev in prevalences:
        p_pos = sensitivity * prev + (1 - specificity) * (1 - prev)
        post = (sensitivity * prev) / p_pos
        posteriors.append(post)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(prevalences * 100, np.array(posteriors) * 100, lw=2)
    ax.set_xlabel('Prevalence (%)')
    ax.set_ylabel('P(Disease | Positive) (%)')
    ax.set_title("Bayes' Theorem: Posterior vs. Prevalence")
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()

bayes_update_visualization()
```

---

## 핵심 요약

- 베이즈 정리는 **조건의 방향을 뒤집는다**. $P(B \mid A)$로부터 $P(A \mid B)$를 계산한다.
- **사전확률**은 초기 믿음을, **사후확률**은 증거를 관측한 뒤 갱신된 믿음을 반영한다.
- 낮은 기저율(드문 사건)이 지배할 수 있다. 좋은 검사를 쓰더라도 양성 대부분이 거짓양성일 수 있다.
- 베이즈 정리는 모수를 사전분포를 갖는 확률변수로 다루고 자료로 갱신하는 베이즈 추론의 토대다.

## 연습문제

**연습문제 1.**
어떤 희귀질환 검사의 유병률이 $P(D) = 0.001$, 민감도가 $P(+ \mid D) = 0.99$, 특이도가 $P(- \mid D^c) = 0.95$다. (a) $P(D \mid +)$를 계산하라. (b) 왜 그렇게 낮은지 설명하라. (c) 유병률이 0.05일 때 다시 계산하라.

??? success "연습문제 1 풀이"
    (a) $P(+) = 0.99 \cdot 0.001 + 0.05 \cdot 0.999 = 0.00099 + 0.04995 = 0.05094$이므로 $P(D \mid +) = 0.00099/0.05094 \approx 0.019$(약 1.9%)이다.

    (b) 질병이 드물다. 1000명 중 건강한 999명에게 5%의 거짓양성 *비율*이 적용되면 거짓양성이 $\approx 50$명 생기는데, 이는 참양성 $\approx 1$명보다 훨씬 많다. 양성 집단이 거짓양성으로 뒤덮인다.

    (c) 유병률이 0.05이면 $P(+) = 0.99 \cdot 0.05 + 0.05 \cdot 0.95 = 0.097$이고 $P(D \mid +) = 0.0495/0.097 \approx 0.510$이다. 사후확률이 2%에서 51%로 뛴다. 가능도비가 같아도 사전확률이 사후확률을 얼마나 강하게 좌우하는지 극명하게 보여준다.

---

**연습문제 2.**
어떤 공장에 기계 세 대가 있어 생산량의 50%, 30%, 20%를 만들고 불량률은 각각 2%, 3%, 5%다. (a) 전체 불량률을 계산하라. (b) 불량품 하나가 주어졌을 때 그것이 3번 기계에서 나왔을 확률은 얼마인가?

??? success "연습문제 2 풀이"
    (a) $P(D) = 0.02 \cdot 0.50 + 0.03 \cdot 0.30 + 0.05 \cdot 0.20 = 0.010 + 0.009 + 0.010 = 0.029$이다. 전체 불량률은 2.9%다.

    (b) $P(M_3 \mid D) = (0.05 \cdot 0.20)/0.029 = 0.010/0.029 \approx 0.345$이다. 3번 기계는 생산량의 20%만 만들지만 불량의 34.5%를 낸다. 불량률이 평균의 2.5배이기 때문이다.

---

**연습문제 3.**
조건부확률의 정의에서 **베이즈 정리를 증명하라**. 그런 다음 **승산 형태**를 진술하라: $P(H \mid E)/P(H^c \mid E) = [P(E \mid H)/P(E \mid H^c)] \cdot [P(H)/P(H^c)]$.

??? success "연습문제 3 풀이"
    조건부확률로부터 $P(A \mid B) = P(A \cap B)/P(B)$이고 $P(B \mid A) = P(A \cap B)/P(A)$이다. 따라서 $P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$이고

    $$
    P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
    $$

    를 얻는다.

    **승산 형태:** 같은 $E$에 대해 $H$의 베이즈 식을 $H^c$의 베이즈 식으로 나누면

    $$
    \frac{P(H \mid E)}{P(H^c \mid E)} = \frac{P(E \mid H) P(H) / P(E)}{P(E \mid H^c) P(H^c) / P(E)} = \underbrace{\frac{P(E \mid H)}{P(E \mid H^c)}}_{\text{likelihood ratio}} \cdot \underbrace{\frac{P(H)}{P(H^c)}}_{\text{prior odds}}
    $$

    이다.

    **사후승산 = 가능도비 × 사전승산.** 이 형태는 $P(E)$를 계산할 필요가 없어 베이즈 추론, 법정에서의 증거 평가, 의학적 의사결정의 일꾼이 된다.

---

**연습문제 4.**
어떤 동전은 공정하거나($P = 0.5$) **아니면** 앞면 확률이 $P = 0.7$로 치우쳐 있다. 사전확률은 같다. 10번 던져 앞면이 8번 나오는 것을 관측했다. 이 동전이 치우쳐 있을 사후확률을 계산하라.

??? success "연습문제 4 풀이"
    $B$ = 치우침, $F$ = 공정, $E$ = 10번 중 앞면 8번 관측이라 하자.

    가능도: $P(E \mid F) = \binom{10}{8} 0.5^8 \cdot 0.5^2 = 45 \cdot (0.5)^{10} \approx 0.0439$.

    $P(E \mid B) = \binom{10}{8} 0.7^8 \cdot 0.3^2 = 45 \cdot 0.0576 \cdot 0.09 \approx 0.2335$.

    베이즈(또는 승산 형태)에 의해

    $$
    P(B \mid E) = \frac{P(E \mid B) P(B)}{P(E \mid B) P(B) + P(E \mid F) P(F)} = \frac{0.2335 \cdot 0.5}{0.2335 \cdot 0.5 + 0.0439 \cdot 0.5} = \frac{0.2335}{0.2774} \approx 0.842
    $$

    이다. 동전이 치우쳐 있을 사후확률이 약 84%다.

    승산 형태로도 같은 결과를 바로 얻는다. 사전승산 1:1, 가능도비 $0.2335/0.0439 \approx 5.32$, 사후승산 5.32:1, 사후확률 $5.32/(5.32 + 1) \approx 0.842$.

---

**연습문제 5.**
**순차적 갱신.** $E_1$ 다음에 새로운 증거 $E_2$가 도착한다. $E_1$을 관측한 뒤 $E_2$를 관측하는 베이즈 갱신이 (가설이 주어졌을 때 조건부 독립으로 다룰 때) 결합 가능도로 한 번에 갱신하는 것과 수학적으로 동등함을 보여라.

??? success "연습문제 5 풀이"
    $E_1$ 이후의 사후확률: $P(H \mid E_1) \propto P(E_1 \mid H) P(H)$.

    $E_2$를 관측한 뒤에는 $E_1$ 이후의 사후확률을 새로운 사전확률로 삼는다.

    $$
    P(H \mid E_1, E_2) \propto P(E_2 \mid H, E_1) P(H \mid E_1)
    $$

    조건부 독립 $P(E_2 \mid H, E_1) = P(E_2 \mid H)$를 가정하면

    $$
    P(H \mid E_1, E_2) \propto P(E_2 \mid H) P(E_1 \mid H) P(H) = P(E_1, E_2 \mid H) P(H)
    $$

    인데, 이것이 바로 결합 가능도에 베이즈를 적용한 것이다. 따라서 순차적 갱신과 일괄 갱신이 같은 사후확률을 준다. 조건부 독립 아래에서 베이즈 틀은 내적으로 일관적이다.

    이것이 온라인 갱신을 정당화한다. 관측값을 하나씩 처리하면 되고 처음부터 다시 계산할 필요가 없다. 갱신 후의 사후확률이 충분한 정보다.

---

**연습문제 6.**
**기저율 무시.** **피고인의 오류**는 이렇게 주장한다. "검찰의 DNA가 피고인의 것과 일치했다. 무작위로 일치할 확률은 100만분의 1이다. 따라서 피고인은 합리적 의심을 넘어 유죄다." 이 논증이 왜 잘못되었으며 베이즈 정리는 그 오류를 어떻게 드러내는가?

??? success "연습문제 6 풀이"
    이 논증은 $P(\text{일치} \mid \text{무죄})$(100만분의 1)를 정작 중요한 역방향 조건인 $P(\text{무죄} \mid \text{일치})$와 혼동한다.

    **베이즈 정리가 그 간극을 드러낸다.** $G$ = 유죄, $M$ = DNA 일치라 하자. 용의자가 $10^6$명 규모의 데이터베이스에서 특정되었다고 하자(동등하게, 피고인이 $10^6$명의 후보 중 하나라는 사전확률 $P(G) = 10^{-6}$). 그러면 $P(M \mid G^c) = 10^{-6}$, $P(M \mid G) \approx 1$이다. 베이즈에 의해

    $$
    P(G \mid M) = \frac{P(M \mid G) P(G)}{P(M \mid G) P(G) + P(M \mid G^c) P(G^c)} = \frac{1 \cdot 10^{-6}}{1 \cdot 10^{-6} + 10^{-6} \cdot (1 - 10^{-6})} \approx 0.5
    $$

    이다. 유죄일 확률이 약 50%에 불과하여 "합리적 의심을 넘어"와는 거리가 멀다. 100만분의 1이라는 수치는 사전승산을 무시한다. 사전승산을 반영하고 나면 사후확률은 훨씬 약해진다. 확신에 찬 평결에 이르려면 DNA 일치 외의 추가 증거가 필요하다.

    이는 희귀질환 예제와 같은 오류 유형이다. 기저율을 잊는 것이다. 현실의 배심원과 정책결정자가 이 오류를 일상적으로 범한다. 베이즈 틀이 그 교정을 제공한다.
