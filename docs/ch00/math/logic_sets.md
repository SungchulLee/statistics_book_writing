# 집합, 함수, 논리

이 절은 책 전체에서 쓰이는 기초적인 수학 언어를 세운다. 논리, 집합, 함수를 정확히 정의해 두면 나중에 확률공간(집합, $\sigma$-대수, 측도의 세 쌍), 확률변수(가측함수), 통계적 추론(논리 구조를 이용해 모집단에 대한 주장을 다루는 일)을 도입할 때 모호함이 생기지 않는다.

## 정의

### 명제와 연결사

**명제(proposition)** 는 참이거나 거짓인 서술문이다. 기본 연결사는 다음과 같다.

| 기호 | 이름 | 읽는 법 | 참이 되는 조건 |
|---|---|---|---|
| $\neg P$ | 부정 | "$P$가 아니다" | $P$가 거짓일 때에 한해 참 |
| $P \land Q$ | 논리곱 | "$P$ 그리고 $Q$" | 둘 다 참일 때에 한해 참 |
| $P \lor Q$ | 논리합 | "$P$ 또는 $Q$"(포함적) | 적어도 하나가 참일 때에 한해 참 |
| $P \Rightarrow Q$ | 함의 | "$P$이면 $Q$이다" | $P$가 참이고 $Q$가 거짓일 때만 거짓 |
| $P \Leftrightarrow Q$ | 쌍조건 | "$P$일 필요충분조건은 $Q$" | $P$와 $Q$의 진리값이 같을 때에 한해 참 |

대우 $\neg Q \Rightarrow \neg P$는 $P \Rightarrow Q$와 논리적으로 동치다. 역 $Q \Rightarrow P$는 동치가 **아니며** 따로 증명해야 한다.

### 집합과 집합 연산

**집합(set)** 은 서로 구별되는 대상들의 순서 없는 모임이다. "$x$가 $A$의 원소이다"를 $x \in A$로 쓴다. 공집합은 $\emptyset$이다. 부분집합은 $A \subseteq B$로 쓴다. 표준적인 연산은 다음과 같다.

$$
A \cup B = \{x : x \in A \text{ or } x \in B\}, \qquad A \cap B = \{x : x \in A \text{ and } x \in B\}
$$

$$
A \setminus B = \{x : x \in A \text{ and } x \notin B\}, \qquad A^c = \Omega \setminus A
$$

**멱집합(power set)** $2^A$는 $A$의 모든 부분집합으로 이루어진 집합이다. **곱집합(데카르트 곱)** 은 $A \times B = \{(a, b) : a \in A, b \in B\}$이다.

### 함수

**함수(function)** $f: A \to B$는 각 $x \in A$에 정확히 하나의 $f(x) \in B$를 대응시킨다. $A$를 정의역, $B$를 공역이라 한다. **상(image)** 은 $f(A) = \{f(x) : x \in A\} \subseteq B$이다. 함수가

- $f(x_1) = f(x_2) \Rightarrow x_1 = x_2$이면 **단사(injective, 일대일)** 이다.
- $f(A) = B$이면 **전사(surjective, 위로의)** 이다.
- 둘 다이면 **전단사(bijective)** 이다.

전단사는 "원소의 개수가 같다"를 엄밀하게 표현한 것이며, 농도(가산성)를 정의하는 근거가 된다.

## 설명

### 한정기호와 그 부정

**한정기호(quantifier)** 는 "모든"($\forall$)과 "존재한다"($\exists$)를 형식화한다. 부정을 취하면 둘이 뒤바뀌고 안쪽 술어가 부정된다.

$$
\neg(\forall\, x \in A,\; P(x)) \;\Leftrightarrow\; \exists\, x \in A \text{ s.t. } \neg P(x)
$$

$$
\neg(\exists\, x \in A,\; P(x)) \;\Leftrightarrow\; \forall\, x \in A,\; \neg P(x)
$$

한정기호의 순서가 중요하다. $\forall x \exists y\, P(x, y)$(각 $x$마다 어떤 $y$가 통한다)는 $\exists y \forall x\, P(x, y)$(하나의 $y$가 모든 $x$에 통한다)보다 논리적으로 약하다. 수렴의 정의 $\forall \varepsilon \, \exists N \, \forall n > N$이 이런 중첩 구조를 가지며, 앞의 두 한정기호를 뒤바꾸면 균등수렴이 되어 엄격히 더 강한 조건이 된다.

### 드모르간 법칙

집합에 대해,

$$
(A \cup B)^c = A^c \cap B^c, \qquad (A \cap B)^c = A^c \cup B^c
$$

이는 임의의(심지어 비가산인) 모임으로 일반화된다.

$$
\left(\bigcup_{\alpha} A_\alpha\right)^{\!c} = \bigcap_{\alpha} A_\alpha^c, \qquad \left(\bigcap_{\alpha} A_\alpha\right)^{\!c} = \bigcup_{\alpha} A_\alpha^c
$$

이것은 "적어도 하나"와 "모두" 사이를 오가는 핵심 도구이며, 확률에서 사건들의 합집합의 여집합을 계산할 때 늘 쓰이는 수법이다.

### 가산성과 그 귀결

어떤 집합이 $\mathbb{N}$과 전단사 대응되면 **가산무한(countably infinite)** 이라 한다(예: $\mathbb{Z}$, $\mathbb{Q}$, 임의 구간 안의 유리수 전체). 그렇지 않으면 **비가산(uncountable)** 이다. 칸토어의 대각선 논법은 $\mathbb{R}$이 비가산임을 보인다.

이 구분은 확률의 토대가 된다.

- 가산 표본공간에서는 확률측도가 확률질량함수로 결정되며, 어떤 사건의 확률이든 합으로 주어진다.
- 비가산 표본공간(예: 실숫값 측정)에서는 개별 결과의 확률이 0이고, 확률은 밀도의 적분으로 정의된다.

### 책 전체에서 쓰이는 함수

- **지시함수** $\mathbf{1}_A(x)$는 $x \in A$이면 $1$, 아니면 $0$이다. 집합(사건)과 수(확률변수)를 잇는 다리다: $\mathbb{E}[\mathbf{1}_A] = P(A)$.
- **지수함수** $e^x$와 **로그** $\ln x$ — 적률생성함수, 로그가능도, 엔트로피에 등장한다.
- **로지스틱 / 시그모이드** $\sigma(x) = 1/(1 + e^{-x})$ — 실직선을 $(0, 1)$로 보낸다. 로짓의 역함수이며 로지스틱 회귀의 토대다.
- **감마함수** $\Gamma(\alpha) = \int_0^\infty t^{\alpha - 1} e^{-t}\, dt$ — 계승을 일반화하며 감마, 베타, $t$, $\chi^2$ 밀도를 정규화한다.

!!! note "공허한 참"
    $P$가 거짓이면 $Q$가 무엇이든 $P \Rightarrow Q$는 참이다. 확률이 0인 사건에 조건을 걸 때 이 점이 중요하다. 영집합 위에서는 어떤 진술이든 "참"이므로, 확률에 관한 진술은 거의 확실하게 성립하는 것으로 해석해야 한다.

## 예제

```python
import numpy as np

# === De Morgan's law with finite sets ===
omega = set(range(1, 11))
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7}

A_c, B_c = omega - A, omega - B
lhs = omega - (A | B)   # (A ∪ B)^c
rhs = A_c & B_c          # A^c ∩ B^c
print(f"(A ∪ B)^c = {lhs}")
print(f"A^c ∩ B^c = {rhs}")
print(f"Equal: {lhs == rhs}")

# === Indicator function and its connection to probability ===
rng = np.random.default_rng(42)
samples = rng.integers(low=1, high=11, size=10_000)
prob_A = np.mean([s in A for s in samples])
print(f"P(A) estimate: {prob_A:.3f} (true 1/2 since |A|=5 of 10)")

# === Bijection demonstration: N <-> Z ===
def bijection_N_to_Z(n):
    # n=1 -> 0, n=2 -> 1, n=3 -> -1, n=4 -> 2, n=5 -> -2, ...
    return n // 2 if n % 2 == 0 else -(n // 2)

print([bijection_N_to_Z(n) for n in range(1, 11)])
```

## 연습문제

**연습문제 1.**
다음 각 진술의 형식적 부정을 써라.

**(a)** $\forall\, x \in \mathbb{R},\; x^2 \geq 0$
**(b)** $\exists\, \varepsilon > 0 \text{ such that } \forall\, n \in \mathbb{N},\; a_n > \varepsilon$
**(c)** $\forall\, \varepsilon > 0,\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon$

??? success "풀이"
    (a) $\exists\, x \in \mathbb{R} \text{ such that } x^2 < 0$.
    (b) $\forall\, \varepsilon > 0,\; \exists\, n \in \mathbb{N} \text{ such that } a_n \leq \varepsilon$.
    (c) 바깥쪽부터 안쪽으로 차례로 부정한다. 부정은

    $$
    \exists\, \varepsilon > 0 \text{ such that } \forall\, N \in \mathbb{N},\; \exists\, n > N \text{ with } |a_n - L| \geq \varepsilon
    $$

    이며, 이것이 바로 "$a_n$이 $L$로 수렴하지 **않는다**"는 진술이다.

---

**연습문제 2.**
$A = \{1, 2, 3, 4, 5\}$, $B = \{3, 4, 5, 6, 7\}$, $\Omega = \{1, 2, 3, 4, 5, 6, 7, 8\}$이라 하자.

**(a)** $A \cap B$, $A \cup B$, $A \setminus B$, $A^c$, 그리고 $A \triangle B := (A \setminus B) \cup (B \setminus A)$를 계산하라.
**(b)** 드모르간 법칙 $(A \cup B)^c = A^c \cap B^c$를 확인하라.

??? success "풀이"
    (a) $A \cap B = \{3, 4, 5\}$, $A \cup B = \{1, 2, 3, 4, 5, 6, 7\}$, $A \setminus B = \{1, 2\}$, $A^c = \{6, 7, 8\}$, $A \triangle B = \{1, 2, 6, 7\}$.

    (b) $(A \cup B)^c = \{8\}$이고 $A^c \cap B^c = \{6, 7, 8\} \cap \{1, 2, 8\} = \{8\}$이다. 둘 다 $\{8\}$과 같다. $\square$

---

**연습문제 3.**
$\cup$, $\cap$, 여집합의 정의를 이용해 드모르간 법칙 $(A \cup B)^c = A^c \cap B^c$를 기본 원리로부터 증명하라.

??? success "풀이"
    양쪽 포함관계를 보여 집합이 같음을 증명한다.

    ($\subseteq$): $x \in (A \cup B)^c$라 하자. 그러면 $x \notin A \cup B$이므로 $x \notin A$이고 **또한** $x \notin B$이다. 따라서 $x \in A^c$이고 $x \in B^c$, 즉 $x \in A^c \cap B^c$이다.

    ($\supseteq$): $x \in A^c \cap B^c$라 하자. 그러면 $x \notin A$이고 $x \notin B$이므로 $x$는 어느 집합에도 속하지 않고 따라서 $x \notin A \cup B$, 즉 $x \in (A \cup B)^c$이다.

    양쪽 포함관계가 모두 성립하므로 두 집합은 같다. $\square$

---

**연습문제 4.**
$f: A \to B$라 하자. $f$가 단사일 필요충분조건이 모든 부분집합 쌍 $S_1, S_2 \subseteq A$에 대해

$$
f(S_1 \cap S_2) = f(S_1) \cap f(S_2)
$$

가 성립하는 것임을 증명하라.

??? success "풀이"
    ($\Rightarrow$) $f$가 단사라고 하자. 포함관계 $f(S_1 \cap S_2) \subseteq f(S_1) \cap f(S_2)$는 임의의 함수에 대해 성립한다($x \in S_1 \cap S_2$에 대해 $y = f(x)$이면 $y$는 $f(S_1)$과 $f(S_2)$ 모두에 속한다). 반대 방향으로, $y \in f(S_1) \cap f(S_2)$라 하자. 그러면 어떤 $x_1 \in S_1$, $x_2 \in S_2$에 대해 $y = f(x_1) = f(x_2)$이다. 단사성에 의해 $x_1 = x_2$이므로 $x_1 \in S_1 \cap S_2$이고 $y \in f(S_1 \cap S_2)$이다.

    ($\Leftarrow$) 모든 부분집합에 대해 이 항등식이 성립한다고 하자. $f(x_1) = f(x_2) = y$인 임의의 $x_1, x_2 \in A$를 잡고 $S_1 = \{x_1\}$, $S_2 = \{x_2\}$라 두자. 그러면 $f(S_1) \cap f(S_2) = \{y\}$이므로 가정에 의해 $f(S_1 \cap S_2) = \{y\}$이다. 이것이 공집합이 아니려면 $S_1 \cap S_2 \ne \emptyset$이어야 하므로 $x_1 = x_2$가 강제된다. $\square$

---

**연습문제 5.**
유리수 전체의 집합 $\mathbb{Q}$가 가산임을 보여라.

??? success "풀이"
    $\mathbb{Q}$에서 $\mathbb{N}$으로 가는 단사를 제시하면 충분하다(그러면 $\mathbb{Q}$는 많아야 가산이고, 무한집합임은 분명하다). 모든 양의 유리수는 $p, q \in \mathbb{N}$인 기약분수 $p/q$로 유일하게 쓸 수 있다. 다음과 같이 정의하자.

    $$
    \phi\!\left(\tfrac{p}{q}\right) = 2^p\, 3^q
    $$

    소인수분해의 유일성에 의해 $\phi$는 $\mathbb{Q}_{>0}$ 위에서 단사다. 여기에 $\mathbb{Q}$에서 $\{0\} \cup \mathbb{Q}_{>0} \cup \mathbb{Q}_{<0}$으로 가는 전단사(예: 양의 유리수와 음의 유리수를 $0, q_1, -q_1, q_2, -q_2, \ldots$처럼 번갈아 배열)를 합성하면 단사 $\mathbb{Q} \hookrightarrow \mathbb{N}$을 얻는다. $\square$

---

**연습문제 6.**
확률에서 흔한 오류는 "$P(A \mid B) > P(A)$이면 $A$가 $B$를 유발했다"고 주장하는 것이다. $P(B \mid A)$를 $P(A \mid B)$, $P(A)$, $P(B)$로 계산하여 **추론의 방향** 문제를 형식화하고, 원래 진술이 왜 근거가 없는지 평이한 말로 설명하라.

??? success "풀이"
    베이즈 규칙에 의해

    $$
    P(B \mid A) = \frac{P(A \mid B)\, P(B)}{P(A)}
    $$

    이다. 가정 $P(A \mid B) > P(A)$는 $A$와 $B$에 대해 대칭이다. 양변에 $P(B)/P(A)$를 곱하면 $P(B \mid A) > P(B)$도 성립함을 알 수 있다. 따라서 "$A$와 $B$가 양의 연관을 갖는다"는 사실은 둘 중 어느 쪽이 (또는 어느 쪽이든) 다른 쪽을 유발하는지에 대해 아무것도 말해주지 않는다. 인과성은 반사실이나 개입에 관한 진술이며 조건부확률 진술만으로는 추론할 수 없다(이것이 "상관관계는 인과관계를 뜻하지 않는다"의 형식적 대응물이다). $\square$
