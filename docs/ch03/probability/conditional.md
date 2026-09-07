# 조건부확률

## 개요

**조건부확률**은 다른 사건이 일어났음을 알게 되었을 때 어떤 사건의 확률이 어떻게 달라지는지를 정량화한다. 확률론에서 가장 중요한 개념 중 하나로, 베이즈적 추론, 통계적 추론, 불확실성 아래의 의사결정의 토대를 이룬다.

---

## 정의

사건 $B$가 주어졌을 때 사건 $A$의 **조건부확률**은($P(B) > 0$일 때) 다음과 같다.

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

**해석:** $B$에 속한 결과들에 배정된 전체 무게(확률) 가운데 $A$에도 속하는 무게의 비율이 $P(A \mid B)$다. $B$로 조건을 거는 것은 사실상 **표본공간을 $\Omega$에서 $B$로 제한**하고 확률을 다시 정규화하는 일이다.

---

## 직관: 표본공간 갱신하기

$B$로 조건을 걸면 $B$ 바깥의 모든 결과를 버리고 남은 확률의 합이 1이 되도록 다시 척도를 맞춘다.

$$
P(A \mid B) = \frac{\text{Weight of bricks in } A \cap B}{\text{Weight of bricks in } B}
$$

이는 "$B$가 일어났음을 안다면 $B$의 확률 중 얼마가 $A$에 속하는가?"라고 묻는 것과 같다.

---

## 곱셈 규칙

정의를 정리하면 **곱셈 규칙**을 얻는다.

$$
P(A \cap B) = P(A \mid B) \cdot P(B) = P(B \mid A) \cdot P(A)
$$

이는 사건들의 사슬로 확장된다.

$$
P(A \cap B \cap C) = P(A) \cdot P(B \mid A) \cdot P(C \mid A \cap B)
$$

---

## 전확률의 법칙

$B_1, B_2, \ldots, B_n$이 $\Omega$의 **분할**을 이루면(즉 서로 배반이고 그 합집합이 $\Omega$이면) 임의의 사건 $A$에 대해

$$
P(A) = \sum_{i=1}^{n} P(A \mid B_i) \cdot P(B_i)
$$

이다. 이는 각 시나리오 $B_i$를 따로 고려하여 $A$의 확률을 분해한다.

---

## 예제

### 예: 카드 뽑기

표준 52장 카드 한 벌에서 카드 한 장을 뽑는다. $A$ = "카드가 킹", $B$ = "카드가 그림 카드(J, Q, K)"라 하자.

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{4/52}{12/52} = \frac{4}{12} = \frac{1}{3}
$$

모든 킹이 그림 카드이므로 $A \cap B = A$이고, 카드가 그림 카드임을 알면 가능성이 12장으로 좁혀지는데 그중 4장이 킹이다.

### 예: 주사위 두 개

공정한 주사위 두 개를 굴린다. $A$ = "합이 8", $B$ = "첫 주사위가 3"이라 하자.

- $P(B) = 1/6$
- $A \cap B = \{(3, 5)\}$이므로 $P(A \cap B) = 1/36$

$$
P(A \mid B) = \frac{1/36}{1/6} = \frac{1}{6}
$$

### 예: 의학 검사 (전확률)

어떤 질병이 인구의 1%에 발생한다. 검사의 민감도는 95%($P(\text{양성} \mid \text{질병}) = 0.95$)이고 특이도는 90%($P(\text{음성} \mid \text{질병 없음}) = 0.90$)다.

검사가 양성일 확률은 다음과 같다.

$$
\begin{aligned}
P(\text{positive}) &= P(\text{positive} \mid \text{disease}) \cdot P(\text{disease}) + P(\text{positive} \mid \text{no disease}) \cdot P(\text{no disease}) \\
&= 0.95 \times 0.01 + 0.10 \times 0.99 \\
&= 0.0095 + 0.099 = 0.1085
\end{aligned}
$$

실제로 질병이 있는 사람은 1%뿐인데도 인구의 약 10.85%가 양성 판정을 받게 된다.

---

## 파이썬으로 살펴보기

```python
import numpy as np

def conditional_probability_simulation(n_simulations=100_000):
    """Simulate conditional probability with two dice."""
    np.random.seed(42)

    die1 = np.random.randint(1, 7, size=n_simulations)
    die2 = np.random.randint(1, 7, size=n_simulations)
    total = die1 + die2

    # P(sum=8 | die1=3)
    mask_B = die1 == 3
    mask_A_and_B = (die1 == 3) & (total == 8)

    p_conditional = mask_A_and_B.sum() / mask_B.sum()
    print(f"Simulated P(sum=8 | die1=3) = {p_conditional:.4f}")
    print(f"Theoretical P(sum=8 | die1=3) = {1/6:.4f}")

conditional_probability_simulation()
```

```python
import numpy as np

def medical_test_simulation(n_people=1_000_000):
    """Simulate the medical testing example using total probability."""
    np.random.seed(42)

    prevalence = 0.01
    sensitivity = 0.95
    false_positive_rate = 0.10

    has_disease = np.random.rand(n_people) < prevalence
    test_positive = np.where(
        has_disease,
        np.random.rand(n_people) < sensitivity,
        np.random.rand(n_people) < false_positive_rate
    )

    p_positive = test_positive.mean()
    print(f"Simulated P(positive) = {p_positive:.4f}")
    print(f"Theoretical P(positive) = {0.1085:.4f}")

medical_test_simulation()
```

---

## 핵심 요약

- 조건부확률 $P(A \mid B)$는 $B$를 관측한 뒤 $A$에 대한 우리의 믿음을 갱신한다.
- 조건을 거는 것은 표본공간을 $B$로 제한하고 확률을 다시 정규화하는 일이다.
- 곱셈 규칙은 결합확률과 조건부확률을 잇는다.
- 전확률의 법칙은 표본공간의 분할에 걸쳐 $P(A)$를 분해한다.

## 연습문제

**연습문제 1.**
항아리에 빨간 구슬 4개와 파란 구슬 6개가 있다. 구슬 두 개를 비복원으로 뽑는다. 첫 구슬이 파란색이라는 조건에서 두 번째 구슬이 빨간색일 확률은 얼마인가?

??? success "풀이"
    $B_1$ = "첫 구슬이 파란색", $R_2$ = "두 번째 구슬이 빨간색"이라 하자.

    파란 구슬 하나를 뽑고 나면 항아리에는 빨간 구슬 4개와 파란 구슬 5개(총 9개)가 남는다. 따라서

    $$
    P(R_2 \mid B_1) = \frac{4}{9}
    $$

    이다.

---

**연습문제 2.**
어떤 공장에서 기계 A가 제품의 60%를, 기계 B가 40%를 생산한다. 기계 A의 불량률은 2%이고 기계 B의 불량률은 5%다. 제품 하나를 무작위로 골랐다. 전확률의 법칙을 사용해 그 제품이 불량일 확률을 계산하라.

??? success "풀이"
    $A$ = "기계 A가 생산", $B$ = "기계 B가 생산", $D$ = "불량"이라 하자. 다음이 주어져 있다.

    $$
    P(A) = 0.60, \quad P(B) = 0.40
    $$

    $$
    P(D \mid A) = 0.02, \quad P(D \mid B) = 0.05
    $$

    전확률의 법칙에 의해

    $$
    P(D) = P(D \mid A) P(A) + P(D \mid B) P(B) = 0.02 \times 0.60 + 0.05 \times 0.40 = 0.012 + 0.020 = 0.032
    $$

    이다. 전체 불량률은 3.2%다.

---

**연습문제 3.**
$P(B) > 0$이면 $P(\cdot \mid B)$가 확률의 세 공리를 만족함을 증명하라. 즉 조건부확률 자체가 제한된 표본공간 위의 타당한 확률측도임을 보여라.

??? success "풀이"
    $P(\cdot \mid B)$에 대해 세 공리를 확인한다.

    **비음성:** 임의의 사건 $A$에 대해 $P(A \cap B) \geq 0$이고 $P(B) > 0$이므로

    $$
    P(A \mid B) = \frac{P(A \cap B)}{P(B)} \geq 0
    $$

    **정규화:**

    $$
    P(\Omega \mid B) = \frac{P(\Omega \cap B)}{P(B)} = \frac{P(B)}{P(B)} = 1
    $$

    **가산가법성:** $A_1, A_2, \ldots$가 서로 배반이면 $A_1 \cap B, A_2 \cap B, \ldots$도 서로 배반이므로

    $$
    P\!\left(\bigcup_i A_i \mid B\right) = \frac{P\!\left(\bigcup_i (A_i \cap B)\right)}{P(B)} = \frac{\sum_i P(A_i \cap B)}{P(B)} = \sum_i P(A_i \mid B)
    $$

    세 공리가 모두 성립하므로 $P(\cdot \mid B)$는 타당한 확률측도다. $\square$

---

**연습문제 4.**
공정한 주사위 두 개를 굴린다. $A$ = "합이 10 이상", $B$ = "두 주사위 모두 5 이상"이라 하자. $P(A \mid B)$를 계산하라.

??? success "풀이"
    먼저 사건 $B$ = "두 주사위 모두 5 이상"을 파악한다. 각 주사위가 5 또는 6일 수 있으므로 $B = \{(5,5),(5,6),(6,5),(6,6)\}$이고 $|B| = 4$, $P(B) = 4/36$이다.

    다음으로 $A \cap B$는 $B$의 결과 중 합이 10 이상인 것이다.

    - $(5,5)$: 합 $= 10$ (해당)
    - $(5,6)$: 합 $= 11$ (해당)
    - $(6,5)$: 합 $= 11$ (해당)
    - $(6,6)$: 합 $= 12$ (해당)

    $B$의 네 결과가 모두 합이 $\geq 10$이므로 $A \cap B = B$이고 $P(A \cap B) = 4/36$이다.

    $$
    P(A \mid B) = \frac{P(A \cap B)}{P(B)} = \frac{4/36}{4/36} = 1
    $$

    두 주사위가 모두 5 이상이면 합은 반드시 10 이상이다.

---

**연습문제 5.**
결합확률에 대한 **연쇄 법칙**은 $P(A_1, A_2, A_3) = P(A_1) P(A_2 \mid A_1) P(A_3 \mid A_1, A_2)$이다. 이를 사용해 표준 52장 카드 한 벌에서 비복원으로 카드 3장을 뽑을 때 $P(\text{하트 3연속})$을 계산하라.

??? success "풀이"
    $H_i$를 "$i$번째 카드가 하트"라 하자. 52장 한 벌에 하트는 13장이다.

    $$
    P(H_1, H_2, H_3) = P(H_1) P(H_2 \mid H_1) P(H_3 \mid H_1, H_2)
    $$

    $P(H_1) = 13/52 = 1/4$이다.

    하트 한 장을 뽑고 나면 51장 중 하트가 12장이므로 $P(H_2 \mid H_1) = 12/51$이다.

    하트 두 장을 뽑고 나면 50장 중 하트가 11장이므로 $P(H_3 \mid H_1, H_2) = 11/50$이다.

    결합확률: $P(H_1, H_2, H_3) = (1/4)(12/51)(11/50) = 132/10200 = 11/850 \approx 0.0129$.

    하트를 연속 3장 뽑을 확률은 약 1.3%다.

    **일반적인 연쇄 법칙:** $P(A_1, \ldots, A_n) = \prod_{i=1}^n P(A_i \mid A_1, \ldots, A_{i-1})$. 이것이 순차적 확률 모형 — 마르코프 연쇄, 은닉 마르코프 모형, 순차적 베이즈 갱신 — 의 토대다.

---

**연습문제 6.**
**몬티 홀 문제.** 문이 셋인데 하나 뒤에는 자동차가, 둘 뒤에는 염소가 있다. 당신은 1번 문을 골랐다. (각 문 뒤에 무엇이 있는지 아는) 진행자가 3번 문을 열어 염소를 보여주고 바꿀 기회를 준다. 바꿀 때와 그대로 둘 때 이길 확률은 각각 얼마인가?

??? success "풀이"
    $C_i$를 "자동차가 $i$번 문 뒤에 있다"라 하자($i = 1, 2, 3$에 대해 균등한 사전확률 $P(C_i) = 1/3$). $H_3$을 "진행자가 3번 문을 연다"라 하자.

    진행자의 행동: 당신이 자동차를 골랐다면(1번 문, $C_1$) 진행자는 2번과 3번 중 무작위로 고르므로 $P(H_3 \mid C_1) = 1/2$이다. 자동차가 2번 문 뒤에 있으면($C_2$) 진행자는 3번을 열 수밖에 없으므로 $P(H_3 \mid C_2) = 1$이다. 자동차가 3번 문 뒤에 있으면 진행자가 그 문을 열 수 없으므로 $P(H_3 \mid C_3) = 0$이다.

    베이즈에 의해

    $P(C_1 \mid H_3) = (1/2)(1/3) / P(H_3) = (1/6)/P(H_3)$.

    $P(C_2 \mid H_3) = (1)(1/3) / P(H_3) = (1/3)/P(H_3)$.

    $P(C_3 \mid H_3) = (0)(1/3) / P(H_3) = 0$.

    정규화하면 $P(H_3) = 1/6 + 1/3 + 0 = 1/2$이므로 $P(C_1 \mid H_3) = 1/3$, $P(C_2 \mid H_3) = 2/3$, $P(C_3 \mid H_3) = 0$이다.

    **그대로 두기**: 확률 $P(C_1 \mid H_3) = 1/3$로 이긴다.
    **바꾸기**: 확률 $P(C_2 \mid H_3) = 2/3$로 이긴다.

    바꾸면 이길 확률이 두 배가 된다. 직관은 이렇다. 처음 고른 문이 맞을 확률은 1/3이었고, 진행자의 선택이 정보를 주기 때문에 진행자가 열지 않은 문이 나머지 2/3의 확률을 떠안는다. 이 문제는 널리 알려졌을 때 수학자들조차 헷갈리게 한 것으로 유명하다. 베이즈 정리로 형식화하기 전까지는 답이 틀린 것처럼 느껴진다.
