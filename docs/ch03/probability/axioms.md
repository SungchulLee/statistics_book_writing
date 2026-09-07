# 확률의 공리

## 개요

확률의 공리는 결과와 사건에 "무게"(확률)를 부여한다는 직관적 발상을 형식화한다. 여기서는 가장 직관적인 것에서 가장 엄밀한 것까지 서로 동등한 세 가지 정식화를 제시한다.

---

## 소박한 확률 공리

이 공리들은 핵심 규칙을 이해하기 쉬운 형태로 담아낸다.

1. **비음성:** 임의의 사건 $A$에 대해

$$
P(A) \geq 0
$$

2. **정규화:** 표본공간 전체의 확률은 1이다.

$$
P(\Omega) = 1
$$

3. **가법성:** 서로 배반인 두 사건 $A$와 $B$(즉 $A \cap B = \emptyset$)에 대해

$$
P(A \cup B) = P(A) + P(B)
$$

---

## 콜모고로프의 확률 공리

**확률측도** $P$는 사건 위에 정의된 실숫값 함수로 다음을 만족한다.

$$
\begin{aligned}
(1) &\quad P(\Omega) = 1, \quad P(\emptyset) = 0 \\[6pt]
(2) &\quad 0 \leq P(A) \leq 1 \quad \text{for any event } A \\[6pt]
(3) &\quad P\left(\bigcup_{i=1}^{\infty} A_i\right) = \sum_{i=1}^{\infty} P(A_i) \quad \text{for any sequence of disjoint events } A_i
\end{aligned}
$$

소박한 공리와의 핵심 차이는 공리 (3)이다. **가산가법성**은 유한 가법 규칙을 서로소인 사건들의 무한(가산) 모임으로 확장한다.

---

## 예제

### 예: 짝수 또는 홀수가 나오기

공정한 육면체 주사위를 굴릴 때 $A = \{2, 4, 6\}$(짝수), $B = \{1, 3, 5\}$(홀수)라 하자. $A \cap B = \emptyset$이므로

$$
P(A \cup B) = P(A) + P(B) = \frac{3}{6} + \frac{3}{6} = 1
$$

이다. $A \cup B = \Omega$이므로 정규화 공리를 만족한다.

---

## 확률의 해석

### 확률 0.7

내일 비가 올 확률이 0.7이라는 것은 비가 올 가능성이 70%라는 뜻이다. 같은 기상 조건의 비슷한 날 10일 중 약 7일에 비가 올 것으로 기대한다.

### 확률 0.05

잘 섞은 카드 한 벌에서 (비복원으로) 에이스를 연달아 두 장 뽑을 확률이 0.05라는 것은 가능성이 5%라는 뜻이다. 100번 반복하면 약 5번 성공할 것으로 기대한다.

### 확률 0

확률이 0이라는 것은 그 사건이 불가능하다는 뜻이다. 예를 들어 보통의 육면체 주사위에서 7이 나올 확률은 0인데, 그 결과가 표본공간에 없기 때문이다.

---

## 파이썬으로 살펴보기

```python
import numpy as np

def verify_axioms(probabilities):
    """Verify Kolmogorov's axioms for a discrete probability distribution."""
    # Axiom 1: Non-negativity
    assert all(p >= 0 for p in probabilities), "Non-negativity violated"

    # Axiom 2: Normalization
    total = sum(probabilities)
    assert np.isclose(total, 1.0), f"Normalization violated: total = {total}"

    # Axiom 3: Additivity (verified by construction for disjoint events)
    print("All axioms satisfied!")
    print(f"  Total probability: {total:.4f}")
    print(f"  Min probability:   {min(probabilities):.4f}")
    print(f"  Max probability:   {max(probabilities):.4f}")

# Fair die
fair_die = [1/6] * 6
verify_axioms(fair_die)

# Loaded die
loaded_die = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
verify_axioms(loaded_die)
```

---

## 핵심 요약

- 콜모고로프의 공리는 확률론 전체에 엄밀한 수학적 토대를 제공한다.
- 세 공리(정규화, 비음성, 가산가법성)만으로 모든 확률 규칙을 유도할 수 있다.
- 확률은 장기적 빈도(빈도주의)로 해석할 수도 있고 믿음의 정도(베이즈)로 해석할 수도 있다.

## 연습문제

**연습문제 1.**
확률의 세 공리만을 사용해 임의의 사건 $A$에 대해 $P(A^c) = 1 - P(A)$임을 증명하라.

??? success "풀이"
    $A$와 $A^c$는 서로 배반이고 $A \cup A^c = \Omega$이므로 가법성 공리에 의해

    $$
    P(A \cup A^c) = P(A) + P(A^c)
    $$

    이다. 정규화 공리에 의해 $P(\Omega) = 1$이므로

    $$
    1 = P(A) + P(A^c)
    $$

    이고, 정리하면

    $$
    P(A^c) = 1 - P(A)
    $$

    이다. $\square$

---

**연습문제 2.**
공리로부터 임의의 두 사건 $A$와 $B$에 대해 다음이 성립함을 증명하라.

$$
P(A \cup B) = P(A) + P(B) - P(A \cap B)
$$

??? success "풀이"
    $A \cup B$를 서로소인 합집합으로 쓴다. $A \cup B = A \cup (B \cap A^c)$이고 $A$와 $B \cap A^c$는 서로소임에 유의하라. 가법성 공리에 의해

    $$
    P(A \cup B) = P(A) + P(B \cap A^c)
    $$

    이다. 마찬가지로 $B = (B \cap A) \cup (B \cap A^c)$도 서로소인 합집합이므로

    $$
    P(B) = P(B \cap A) + P(B \cap A^c)
    $$

    이고, $P(B \cap A^c)$에 대해 풀면

    $$
    P(B \cap A^c) = P(B) - P(A \cap B)
    $$

    이다. 이를 대입하면

    $$
    P(A \cup B) = P(A) + P(B) - P(A \cap B)
    $$

    를 얻는다. $\square$

---

**연습문제 3.**
어떤 학생이 $P(A) = 0.4$, $P(B) = 0.5$, $P(A \cup B) = 0.8$이라고 주장한다. 다른 학생은 $P(A) = 0.7$, $P(B) = 0.6$, $P(A \cap B) = 0.1$이라고 주장한다. 각 배정이 공리와 모순되지 않는지 판정하라.

??? success "풀이"
    **첫 번째 학생:** 포함배제를 쓰면 $P(A \cap B) = P(A) + P(B) - P(A \cup B) = 0.4 + 0.5 - 0.8 = 0.1$이다. $0 \leq 0.1 \leq \min(0.4, 0.5)$이고 모든 확률이 $[0,1]$에 있으므로 이 배정은 공리와 **모순되지 않는다**.

    **두 번째 학생:** $P(A \cup B) = P(A) + P(B) - P(A \cap B) = 0.7 + 0.6 - 0.1 = 1.2$여야 한다. 그러나 정규화 공리는 $P(A \cup B) \leq P(\Omega) = 1$을 요구한다. $1.2 > 1$이므로 이 배정은 공리를 **위반하며** 따라서 불가능하다.

---

**연습문제 4.**
공리를 사용해 $A \subseteq B$이면 $P(A) \leq P(B)$임을(확률의 단조성) 증명하라.

??? success "풀이"
    $A \subseteq B$이므로 $B = A \cup (B \cap A^c)$로 쓸 수 있고, $A$와 $B \cap A^c$는 서로소다. 가법성 공리에 의해

    $$
    P(B) = P(A) + P(B \cap A^c)
    $$

    이다. 비음성 공리에 의해 $P(B \cap A^c) \geq 0$이므로

    $$
    P(B) = P(A) + P(B \cap A^c) \geq P(A)
    $$

    이다. $\square$

---

**연습문제 5.**
**본페로니 부등식.** 임의의 사건 $A_1, \ldots, A_n$에 대해 $P(\bigcup_i A_i) \le \sum_i P(A_i)$임을 증명하라. 이것은 언제 유용한가?

??? success "풀이"
    $n$에 대한 귀납법으로 증명한다. 기저 단계 $n = 1$: 자명하게 $P(A_1) \le P(A_1)$이다.

    귀납 단계: $P(\bigcup_{i=1}^k A_i) \le \sum_{i=1}^k P(A_i)$라고 가정하자. 포함배제에 의해

    $$
    P(A_1 \cup \cdots \cup A_{k+1}) = P(\bigcup_{i=1}^k A_i) + P(A_{k+1}) - P((\bigcup_{i=1}^k A_i) \cap A_{k+1})
    $$

    이고 마지막 항이 음이 아니므로

    $$
    P(\bigcup_{i=1}^{k+1} A_i) \le P(\bigcup_{i=1}^k A_i) + P(A_{k+1}) \le \sum_{i=1}^k P(A_i) + P(A_{k+1}) = \sum_{i=1}^{k+1} P(A_i)
    $$

    이다. $\square$

    **용도:** **다중비교**에서 각각 수준 $\alpha$로 $n$번의 가설검정을 수행한다고 하자. 적어도 한 번 잘못 기각할 확률은 본페로니에 의해 $P(\bigcup_i \text{Reject}_i \mid H_0) \le n\alpha$다. 따라서 수준 $\alpha/n$으로 검정하면 가족단위 오류율이 $\le \alpha$가 된다. 본페로니 보정은 보수적이지만 의존 구조와 무관하게 언제나 타당하다.

---

**연습문제 6.**
**$\sigma$-가법성 대 유한 가법성.** 차이를 진술하고 측도론적 확률이 왜 더 강한 성질을 요구하는지 설명하라.

??? success "풀이"
    **유한 가법성:** 서로소인 (유한 모임) $A_1, \ldots, A_n$에 대해 $P(\bigcup_{i=1}^n A_i) = \sum_{i=1}^n P(A_i)$.

    **$\sigma$-가법성**(가산가법성, 콜모고로프의 공리 3): 서로소인 사건의 *가산무한* 열 $A_1, A_2, \ldots$에 대해 $P(\bigcup_{i=1}^\infty A_i) = \sum_{i=1}^\infty P(A_i)$.

    **왜 $\sigma$-가법성인가?** 유용한 여러 결과가 이를 필요로 한다.

    - **측도의 연속성**: $A_1 \subseteq A_2 \subseteq \ldots$이고 $A = \bigcup A_n$이면 $P(A) = \lim P(A_n)$이다. 극한정리에 결정적이다.
    - **확률밀도의 존재**: 연속분포를 정의하려면 임의로 잘게 나눈 부분들에 확률을 부여해야 하는데, 이는 가산 연산이다.
    - **강한 큰수의 법칙**: 가산 합집합으로 정의되는 거의 확실한 수렴을 필요로 한다.

    유한 가법성만으로는 역설적인 배정이 허용된다(예: 직관적이지만 $\sigma$-가법적이지 않은 정수 위의 "균등" 분포). 콜모고로프는 이런 것들을 배제하고 확률을 측도론과 연결하기 위해 $\sigma$-가법성을 택했다.
