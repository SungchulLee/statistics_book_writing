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

출력:

```
(A ∪ B)^c = {8, 9, 10}
A^c ∩ B^c = {8, 9, 10}
Equal: True
P(A) estimate: 0.506 (true 1/2 since |A|=5 of 10)
[0, 1, -1, 2, -2, 3, -3, 4, -4, 5]
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
다음 각 진술의 형식적 부정을 써라.

**(a)** $\forall\, x \in \mathbb{R},\; x^2 \geq 0$
**(b)** $\exists\, \varepsilon > 0 \text{ such that } \forall\, n \in \mathbb{N},\; a_n > \varepsilon$
**(c)** $\forall\, \varepsilon > 0,\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon$

</div>

??? success "풀이"
    (a) $\exists\, x \in \mathbb{R} \text{ such that } x^2 < 0$.
    (b) $\forall\, \varepsilon > 0,\; \exists\, n \in \mathbb{N} \text{ such that } a_n \leq \varepsilon$.
    (c) 바깥쪽부터 안쪽으로 차례로 부정한다. 부정은

    $$
    \exists\, \varepsilon > 0 \text{ such that } \forall\, N \in \mathbb{N},\; \exists\, n > N \text{ with } |a_n - L| \geq \varepsilon
    $$

    이며, 이것이 바로 "$a_n$이 $L$로 수렴하지 **않는다**"는 진술이다.

<div class="drillbox" markdown>

**연습문제 2.**
$A = \{1, 2, 3, 4, 5\}$, $B = \{3, 4, 5, 6, 7\}$, $\Omega = \{1, 2, 3, 4, 5, 6, 7, 8\}$이라 하자.

**(a)** $A \cap B$, $A \cup B$, $A \setminus B$, $A^c$, 그리고 $A \triangle B := (A \setminus B) \cup (B \setminus A)$를 계산하라.
**(b)** 드모르간 법칙 $(A \cup B)^c = A^c \cap B^c$를 확인하라.

</div>

??? success "풀이"
    (a) $A \cap B = \{3, 4, 5\}$, $A \cup B = \{1, 2, 3, 4, 5, 6, 7\}$, $A \setminus B = \{1, 2\}$, $A^c = \{6, 7, 8\}$, $A \triangle B = \{1, 2, 6, 7\}$.

    (b) $(A \cup B)^c = \{8\}$이고 $A^c \cap B^c = \{6, 7, 8\} \cap \{1, 2, 8\} = \{8\}$이다. 둘 다 $\{8\}$과 같다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
$\cup$, $\cap$, 여집합의 정의를 이용해 드모르간 법칙 $(A \cup B)^c = A^c \cap B^c$를 기본 원리로부터 증명하라.

</div>

??? success "풀이"
    양쪽 포함관계를 보여 집합이 같음을 증명한다.

    ($\subseteq$): $x \in (A \cup B)^c$라 하자. 그러면 $x \notin A \cup B$이므로 $x \notin A$이고 **또한** $x \notin B$이다. 따라서 $x \in A^c$이고 $x \in B^c$, 즉 $x \in A^c \cap B^c$이다.

    ($\supseteq$): $x \in A^c \cap B^c$라 하자. 그러면 $x \notin A$이고 $x \notin B$이므로 $x$는 어느 집합에도 속하지 않고 따라서 $x \notin A \cup B$, 즉 $x \in (A \cup B)^c$이다.

    양쪽 포함관계가 모두 성립하므로 두 집합은 같다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
$f: A \to B$라 하자. $f$가 단사일 필요충분조건이 모든 부분집합 쌍 $S_1, S_2 \subseteq A$에 대해

$$
f(S_1 \cap S_2) = f(S_1) \cap f(S_2)
$$

가 성립하는 것임을 증명하라.

</div>

??? success "풀이"
    ($\Rightarrow$) $f$가 단사라고 하자. 포함관계 $f(S_1 \cap S_2) \subseteq f(S_1) \cap f(S_2)$는 임의의 함수에 대해 성립한다($x \in S_1 \cap S_2$에 대해 $y = f(x)$이면 $y$는 $f(S_1)$과 $f(S_2)$ 모두에 속한다). 반대 방향으로, $y \in f(S_1) \cap f(S_2)$라 하자. 그러면 어떤 $x_1 \in S_1$, $x_2 \in S_2$에 대해 $y = f(x_1) = f(x_2)$이다. 단사성에 의해 $x_1 = x_2$이므로 $x_1 \in S_1 \cap S_2$이고 $y \in f(S_1 \cap S_2)$이다.

    ($\Leftarrow$) 모든 부분집합에 대해 이 항등식이 성립한다고 하자. $f(x_1) = f(x_2) = y$인 임의의 $x_1, x_2 \in A$를 잡고 $S_1 = \{x_1\}$, $S_2 = \{x_2\}$라 두자. 그러면 $f(S_1) \cap f(S_2) = \{y\}$이므로 가정에 의해 $f(S_1 \cap S_2) = \{y\}$이다. 이것이 공집합이 아니려면 $S_1 \cap S_2 \ne \emptyset$이어야 하므로 $x_1 = x_2$가 강제된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.**
유리수 전체의 집합 $\mathbb{Q}$가 가산임을 보여라.

</div>

??? success "풀이"
    $\mathbb{Q}$에서 $\mathbb{N}$으로 가는 단사를 제시하면 충분하다(그러면 $\mathbb{Q}$는 많아야 가산이고, 무한집합임은 분명하다). 모든 양의 유리수는 $p, q \in \mathbb{N}$인 기약분수 $p/q$로 유일하게 쓸 수 있다. 다음과 같이 정의하자.

    $$
    \phi\!\left(\tfrac{p}{q}\right) = 2^p\, 3^q
    $$

    소인수분해의 유일성에 의해 $\phi$는 $\mathbb{Q}_{>0}$ 위에서 단사다. 여기에 $\mathbb{Q}$에서 $\{0\} \cup \mathbb{Q}_{>0} \cup \mathbb{Q}_{<0}$으로 가는 전단사(예: 양의 유리수와 음의 유리수를 $0, q_1, -q_1, q_2, -q_2, \ldots$처럼 번갈아 배열)를 합성하면 단사 $\mathbb{Q} \hookrightarrow \mathbb{N}$을 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 6.**
확률에서 흔한 오류는 "$P(A \mid B) > P(A)$이면 $A$가 $B$를 유발했다"고 주장하는 것이다. $P(B \mid A)$를 $P(A \mid B)$, $P(A)$, $P(B)$로 계산하여 **추론의 방향** 문제를 형식화하고, 원래 진술이 왜 근거가 없는지 평이한 말로 설명하라.

</div>

??? success "풀이"
    베이즈 규칙에 의해

    $$
    P(B \mid A) = \frac{P(A \mid B)\, P(B)}{P(A)}
    $$

    이다. 가정 $P(A \mid B) > P(A)$는 $A$와 $B$에 대해 대칭이다. 양변에 $P(B)/P(A)$를 곱하면 $P(B \mid A) > P(B)$도 성립함을 알 수 있다. 따라서 "$A$와 $B$가 양의 연관을 갖는다"는 사실은 둘 중 어느 쪽이 (또는 어느 쪽이든) 다른 쪽을 유발하는지에 대해 아무것도 말해주지 않는다. 인과성은 반사실이나 개입에 관한 진술이며 조건부확률 진술만으로는 추론할 수 없다(이것이 "상관관계는 인과관계를 뜻하지 않는다"의 형식적 대응물이다). $\square$

<div class="drillbox" markdown>

**연습문제 7.**
칸토어의 대각선 논법으로 $(0,1)$이 비가산임을 증명하라. 이 사실이 확률론에서 갖는 귀결은 무엇인가?

</div>

??? success "풀이"
    귀류법을 쓴다. $(0,1)$이 가산이라 가정하면 그 원소를 모두 나열할 수 있다.

    $$
    x_1 = 0.d_{11}d_{12}d_{13}\cdots,\quad
    x_2 = 0.d_{21}d_{22}d_{23}\cdots,\quad
    x_3 = 0.d_{31}d_{32}d_{33}\cdots,\;\ldots
    $$

    이제 새로운 수 $y = 0.e_1e_2e_3\cdots$를 **대각선을 피해서** 만든다.

    $$
    e_k = \begin{cases} 5 & d_{kk} \ne 5 \\ 6 & d_{kk} = 5 \end{cases}
    $$

    그러면 $y \in (0,1)$이지만, 모든 $k$에 대해 $y$는 $k$번째 자리에서 $x_k$와 다르므로 $y \ne x_k$이다. 즉 $y$는 목록에 없다. 이는 목록이 $(0,1)$ 전체를 담는다는 가정에 모순이다. $\square$

    !!! note "자릿수 $5$와 $6$만 쓴 이유"
        $0$과 $9$를 피한 것은 $0.4999\cdots = 0.5000\cdots$처럼 십진 표현이 두 가지인 수 때문이다. $5$와 $6$만 쓰면 $y$의 표현이 유일해져 "자리가 다르면 수도 다르다"가 확실해진다. 증명에서 가장 자주 빠뜨리는 곳이다.

    **확률론에서의 귀결.** $(0,1)$ 위의 균등분포를 생각하자. 각 점의 확률이 어떤 상수 $c$로 같아야 할 것 같지만,

    - $c > 0$이면 가산개의 점만 모아도 확률의 합이 $\infty$가 되어 $1$을 넘는다.
    - $c = 0$이면, 만약 $(0,1)$이 가산이었을 경우 가산가법성에 의해 전체 확률이 $0$이 되어 모순이다.

    비가산이기 때문에 두 번째 모순이 발생하지 않는다. **가산가법성은 비가산 합집합에는 적용되지 않으므로**, 모든 점의 확률이 $0$이면서도 전체의 확률이 $1$일 수 있다. 연속형 확률변수에서 $P(X = a) = 0$인데도 $P(a < X < b) > 0$인 것이 이상하지 않은 이유가 바로 이것이다.

<div class="drillbox" markdown>

**연습문제 8.**
지시함수의 대수를 이용해 포함–배제 원리를 유도하라. 먼저 $\mathbf{1}_{A \cap B} = \mathbf{1}_A \mathbf{1}_B$와 $\mathbf{1}_{A^c} = 1 - \mathbf{1}_A$를 확인한 뒤,

$$
\mathbf{1}_{A_1 \cup \cdots \cup A_n} = 1 - \prod_{i=1}^n (1 - \mathbf{1}_{A_i})
$$

을 전개하고 기댓값을 취하라.

</div>

??? success "풀이"
    **기본 항등식.** $\mathbf{1}_{A\cap B}(x) = 1$일 필요충분조건은 $x \in A$이고 $x \in B$인 것이며, 이는 $\mathbf{1}_A(x)\mathbf{1}_B(x) = 1$과 같다. 두 값이 모두 $\{0,1\}$에 있으므로 곱이 곧 논리곱이다. 여집합은 정의에서 바로 나온다.

    **합집합.** $x$가 어느 $A_i$에도 속하지 않을 필요충분조건은 모든 $i$에 대해 $1 - \mathbf{1}_{A_i}(x) = 1$인 것이므로

    $$
    \mathbf{1}_{(\bigcup A_i)^c} = \prod_{i=1}^n(1 - \mathbf{1}_{A_i})
    $$

    이고, 여집합을 취하면 주어진 식이 된다. 드모르간 법칙을 곱으로 쓴 것이다.

    **전개.** $n = 3$일 때 곱을 펼치면

    $$
    1 - (1-\mathbf{1}_{A})(1-\mathbf{1}_{B})(1-\mathbf{1}_{C})
    = \mathbf{1}_A + \mathbf{1}_B + \mathbf{1}_C - \mathbf{1}_{A}\mathbf{1}_{B} - \mathbf{1}_{A}\mathbf{1}_{C} - \mathbf{1}_{B}\mathbf{1}_{C} + \mathbf{1}_{A}\mathbf{1}_{B}\mathbf{1}_{C}
    $$

    이다. 여기서 $\mathbb{E}[\mathbf{1}_S] = P(S)$를 쓰고 곱을 교집합으로 되돌리면

    $$
    P(A\cup B\cup C) = P(A)+P(B)+P(C)-P(A\cap B)-P(A\cap C)-P(B\cap C)+P(A\cap B\cap C)
    $$

    를 얻는다. 일반형의 부호 $(-1)^{k+1}$은 곱을 전개할 때 나오는 $(-1)^k$에서 그대로 온다. $\square$

    ```python
    import numpy as np

    rng = np.random.default_rng(3)
    N = 200_000
    A = rng.random(N) < 0.5
    B = rng.random(N) < 0.3
    C = rng.random(N) < 0.2

    union = (A | B | C).mean()
    incl_excl = (A.mean() + B.mean() + C.mean()
                 - (A & B).mean() - (A & C).mean() - (B & C).mean()
                 + (A & B & C).mean())
    product = (1 - (1 - A) * (1 - B) * (1 - C)).mean()   # 지시함수 곱 형태

    print(f"P(A∪B∪C) 직접   = {union:.6f}")
    print(f"포함–배제       = {incl_excl:.6f}")
    print(f"지시함수 곱     = {product:.6f}")
    ```

    출력:

    ```
    P(A∪B∪C) 직접   = 0.718550
    포함–배제       = 0.718550
    지시함수 곱     = 0.718550
    ```

    세 값이 부동소수점 오차 안에서 일치한다. 세 번째 계산이 특히 눈여겨볼 만한데, **집합 연산을 산술로 바꾸어** 벡터화된 코드 한 줄로 끝냈다.

    이 기법은 유도 도구 이상이다. 조합론에서 완전순열의 개수를 세거나 확률에서 본페로니 부등식(전개를 도중에 자르면 상계와 하계가 번갈아 나온다)을 얻을 때도 같은 전개를 쓴다.

<div class="drillbox" markdown>

**연습문제 9.**
한정기호의 순서가 왜 중요한지 구체적으로 보여라. $f_n(x) = x^n$을 $[0,1)$ 위에서 생각하자.

**(a)** $f_n \to 0$이 점별로 성립함을 보여라.
**(b)** 이 수렴이 균등하지 **않음**을 보여라.
**(c)** 두 진술의 한정기호 구조가 어떻게 다른지 설명하라.

</div>

??? success "풀이"
    **(a) 점별 수렴.** $x \in [0,1)$을 고정하면 $|x| < 1$이므로 $x^n \to 0$이다. 주어진 $\varepsilon$에 대해 $N = \lceil \ln\varepsilon / \ln x\rceil$로 두면 된다.

    **(b) 균등수렴의 실패.** 각 $n$에 대해

    $$
    \sup_{x \in [0,1)} |f_n(x) - 0| = \sup_{x\in[0,1)} x^n = 1
    $$

    이다($x \to 1^-$일 때 상한에 접근한다. 최댓값은 아니다). 이 상한이 $n$과 무관하게 $1$이므로 $0$으로 가지 않는다.

    **(c) 한정기호 구조.**

    $$
    \text{점별: } \forall \varepsilon\; \forall x\; \exists N\; \forall n>N \;\; |f_n(x)| < \varepsilon
    $$

    $$
    \text{균등: } \forall \varepsilon\; \exists N\; \forall x\; \forall n>N \;\; |f_n(x)| < \varepsilon
    $$

    차이는 오직 $\exists N$과 $\forall x$의 **순서**뿐이다. 점별에서는 $N$이 $x$에 의존해도 되지만, 균등에서는 하나의 $N$이 모든 $x$에 동시에 통해야 한다. 여기서는 $x$가 $1$에 가까워질수록 필요한 $N$이 한없이 커지므로 그런 $N$이 없다. $\square$

    ```python
    import numpy as np

    print("고정된 x 에서는 0 으로 간다 (점별 수렴)")
    print(f"{'n':>6}{'0.5^n':>12}{'0.9^n':>12}{'0.99^n':>12}{'sup':>8}")
    for n in (10, 100, 1000, 10_000):
        print(f"{n:>6}{0.5 ** n:>12.3e}{0.9 ** n:>12.3e}{0.99 ** n:>12.3e}{1.0:>8.1f}")

    print("\n|x^N| < 0.01 을 만들려면 N 이 얼마나 커야 하는가")
    for x in (0.5, 0.9, 0.99, 0.999):
        N = int(np.ceil(np.log(0.01) / np.log(x)))
        print(f"  x = {x:<6}: N = {N:>5}")
    ```

    출력:

    ```
    고정된 x 에서는 0 으로 간다 (점별 수렴)
         n       0.5^n       0.9^n      0.99^n     sup
        10   9.766e-04   3.487e-01   9.044e-01     1.0
       100   7.889e-31   2.656e-05   3.660e-01     1.0
      1000  9.333e-302   1.748e-46   4.317e-05     1.0
     10000   0.000e+00   0.000e+00   2.249e-44     1.0

    |x^N| < 0.01 을 만들려면 N 이 얼마나 커야 하는가
      x = 0.5   : N =     7
      x = 0.9   : N =    44
      x = 0.99  : N =   459
      x = 0.999 : N =  4603
    ```

    표의 마지막 열이 핵심이다. 각 열은 $0$으로 내려가지만 상한은 $n$이 아무리 커도 $1$에 붙박여 있다. 그리고 필요한 $N$은 $x \to 1^-$에서 발산한다.

    **통계에서 왜 중요한가.** 글리벤코–칸텔리 정리는 경험분포함수가 참 분포함수로 **균등하게** 수렴한다고 말한다($\sup_x |F_n(x) - F(x)| \to 0$, 거의 확실하게). 점별 수렴만으로는 부족한데, 콜모고로프–스미르노프 검정 같은 방법이 바로 그 상한을 검정통계량으로 쓰기 때문이다.

<div class="drillbox" markdown>

**연습문제 10.**
로지스틱 함수 $\sigma(x) = 1/(1+e^{-x})$가 $\mathbb{R}$에서 $(0,1)$로 가는 전단사임을 보이고 역함수를 구하라. 항등식 $\sigma'(x) = \sigma(x)(1-\sigma(x))$와 $\sigma(-x) = 1-\sigma(x)$도 증명하라.

</div>

??? success "풀이"
    **전단사.** $y = 1/(1+e^{-x})$를 $x$에 대해 푼다.

    $$
    1 + e^{-x} = \frac{1}{y}
    \;\Longrightarrow\;
    e^{-x} = \frac{1-y}{y}
    \;\Longrightarrow\;
    x = \ln\frac{y}{1-y}
    $$

    각 $y \in (0,1)$에 대해 이 $x$가 유일하게 존재하므로 $\sigma$는 전단사이고, 역함수는 **로짓** $\operatorname{logit}(y) = \ln\frac{y}{1-y}$이다. 이것이 로지스틱 회귀에서 확률을 실직선으로 옮겨 선형모형을 세울 수 있게 해 주는 다리다.

    **도함수.** 몫의 미분 또는 연쇄법칙으로

    $$
    \sigma'(x) = \frac{e^{-x}}{(1+e^{-x})^2}
    = \frac{1}{1+e^{-x}}\cdot\frac{e^{-x}}{1+e^{-x}}
    = \sigma(x)\left(1-\sigma(x)\right)
    $$

    이다. 마지막 등호는 $\dfrac{e^{-x}}{1+e^{-x}} = 1 - \dfrac{1}{1+e^{-x}}$에서 나온다.

    **대칭성.** $\sigma(-x) = \dfrac{1}{1+e^{x}} = \dfrac{e^{-x}}{e^{-x}+1} = 1-\sigma(x)$이다. $\square$

    ```python
    import numpy as np
    from scipy.special import expit

    x = np.array([-3., -1., 0., 1., 3.])
    p = expit(x)

    print("sigma(x)        :", p.round(6))
    print("sigma(x)+sigma(-x):", (p + expit(-x)).round(12))
    print("logit(sigma(x)) :", np.log(p / (1 - p)).round(10))

    h = 1e-6
    print("\n수치 도함수 :", ((expit(x + h) - expit(x - h)) / (2 * h)).round(8))
    print("sigma(1-sigma):", (p * (1 - p)).round(8))
    ```

    출력:

    ```
    sigma(x)        : [0.047426 0.268941 0.5      0.731059 0.952574]
    sigma(x)+sigma(-x): [1. 1. 1. 1. 1.]
    logit(sigma(x)) : [-3. -1.  0.  1.  3.]

    수치 도함수 : [0.04517666 0.19661193 0.25       0.19661193 0.04517666]
    sigma(1-sigma): [0.04517666 0.19661193 0.25       0.19661193 0.04517666]
    ```

    **두 가지 실무적 함의.**

    도함수가 $\sigma(1-\sigma)$라는 것은 최댓값이 $x = 0$에서 $0.25$이고 양쪽 꼬리에서 $0$으로 죽는다는 뜻이다. $|x|$가 크면 기울기가 사실상 사라지므로, 신경망에서 시그모이드를 은닉층 활성함수로 쓸 때 기울기 소실 문제가 생긴다. 로지스틱 회귀의 피셔 정보에도 같은 인자가 나타나며, 이는 $\hat{p}$이 $0$이나 $1$에 가까울 때 계수 추정이 불안정해지는 이유다.

    $x = -800$처럼 큰 음수에서 $1/(1+e^{-x})$를 그대로 계산하면 `exp` 가 넘쳐 경고가 난다. `scipy.special.expit` 은 부호에 따라 $e^{x}/(1+e^{x})$ 형태로 갈아타 이를 피한다. 로그가능도를 직접 구현할 때는 이런 수치적으로 안정한 형태를 쓰라. $\square$

