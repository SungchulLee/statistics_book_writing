# 조건부 독립

## 개요

**조건부 독립**은 조건이 되는 사건을 도입하여 독립 개념을 확장한다. 두 사건이 전체적으로는 종속이면서 추가 정보로 조건을 걸면 독립이 될 수도 있고, 그 반대일 수도 있다. 이 개념은 그래프 모형, 베이즈 망, 인과적 추론의 핵심이다.

---

## 정의

사건 $A$와 $B$가 사건 $C$($P(C) > 0$)가 주어졌을 때 **조건부 독립**이라는 것은

$$
P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)
$$

를 뜻한다. 동등하게, $P(B \cap C) > 0$이면

$$
P(A \mid B \cap C) = P(A \mid C)
$$

이다.

**해석:** $C$가 일어났음을 알고 나면, $B$가 일어났다는 사실을 추가로 알아도 $A$에 대해 더 알게 되는 것이 없다.

조건부 독립을 $A \perp\!\!\!\perp B \mid C$로 표기한다.

---

## 독립은 조건부 독립을 함의하지 않는다

두 사건이 (조건 없이) 독립이면서 조건을 건 뒤에는 종속이 될 수 있다. 이를 **버크슨의 역설** 또는 **해명 효과(explaining away)** 라 한다.

### 예: 하나의 결과를 낳는 두 원인

화재경보기($C$)가 화재($A$)나 탄 토스트($B$) 중 하나로 울릴 수 있다고 하자. 두 원인은 독립이다.

$$
P(A \cap B) = P(A) \cdot P(B)
$$

그러나 경보가 울렸다는 조건($C$)에서 화재가 없음을 알게 되면 탄 토스트일 가능성이 높아진다.

$$
P(B \mid A^c \cap C) > P(B \mid C)
$$

따라서 $A \perp\!\!\!\perp B$이지만 $A \not\perp\!\!\!\perp B \mid C$이다.

---

## 조건부 독립은 독립을 함의하지 않는다

역으로, $C$가 주어졌을 때 조건부 독립이면서 조건 없이는 독립이 아닐 수 있다.

### 예: 혼합에서 뽑기

동전을 무작위로 고른다. 1번 동전은 $P(\text{H}) = 0.3$, 2번 동전은 $P(\text{H}) = 0.7$이다. $C$가 어느 동전을 골랐는지를 나타내고 $A$와 $B$가 두 번 던진 결과라 하자.

동전($C$)이 주어지면 두 던지기는 독립이다.

$$
P(A \cap B \mid C) = P(A \mid C) \cdot P(B \mid C)
$$

그러나 동전을 모르면 두 던지기는 종속이다. 첫 던지기가 앞면이면 앞면 쪽으로 치우친 동전이 선택되었을 가능성이 높아지고, 그러면 두 번째 던지기가 앞면일 확률도 높아진다.

---

## 예제

### 예: 같은 시험을 치르는 학생들

두 학생 $A$와 $B$가 같은 시험을 친다. $A_{\text{pass}}$와 $B_{\text{pass}}$를 각자 합격하는 사건이라 하고, $C$ = "시험이 쉬웠다"라 하자.

$C$가 주어지면 학생 $A$의 합격 여부는 학생 $B$에 대해 정보를 거의 주지 않는다(둘의 능력은 별개다). 그러나 조건 없이는, $A$가 합격했음을 알면 시험이 쉬웠을 가능성이 높아지고, 그러면 $B$의 합격 가능성도 높아진다.

$$
A_{\text{pass}} \perp\!\!\!\perp B_{\text{pass}} \mid C \quad \text{but} \quad A_{\text{pass}} \not\perp\!\!\!\perp B_{\text{pass}}
$$

### 예: 합이 알려진 주사위

공정한 주사위 두 개를 굴린다. $A$ = "1번 주사위가 4", $B$ = "2번 주사위가 3"이라 하자. 이들은 독립이다. 그러나 $C$ = "합이 7"로 조건을 걸면

$$
P(A \mid C) = \frac{1}{6}, \quad P(B \mid C) = \frac{1}{6}, \quad P(A \cap B \mid C) = \frac{1}{6}
$$

이다. 여기서 $P(A \cap B \mid C) = 1/6 \neq (1/6)(1/6)$이므로 $A$와 $B$는 $C$가 주어졌을 때 조건부 독립이 **아니다**. 사실 1번 주사위가 4이고 합이 7이면 2번 주사위가 3임이 확실히 결정된다.

---

## 관계 요약

| 상황 | $A \perp\!\!\!\perp B$ | $A \perp\!\!\!\perp B \mid C$ |
|:---|:---:|:---:|
| 독립이고 조건을 걸어도 독립 | ✓ | ✓ |
| 독립이지만 조건을 걸면 종속 (버크슨) | ✓ | ✗ |
| 종속이지만 조건을 걸면 독립 | ✗ | ✓ |
| 종속이고 조건을 걸어도 종속 | ✗ | ✗ |

네 가지 상황이 모두 가능하다. 독립과 조건부 독립 사이에는 어느 방향으로도 **논리적 함의가 없다**.

---

## 파이썬으로 살펴보기

```python
import numpy as np

def mixture_coin_simulation(n_simulations=200_000):
    """Demonstrate conditional independence in a mixture model."""
    np.random.seed(42)

    # Choose coin: coin 0 has P(H)=0.3, coin 1 has P(H)=0.7
    coin = np.random.randint(0, 2, size=n_simulations)
    p_heads = np.where(coin == 0, 0.3, 0.7)

    flip1 = np.random.rand(n_simulations) < p_heads
    flip2 = np.random.rand(n_simulations) < p_heads

    # Unconditional: P(flip2=H | flip1=H) vs P(flip2=H)
    p_f2 = flip2.mean()
    p_f2_given_f1 = flip2[flip1].mean()
    print("=== Unconditional (marginal) ===")
    print(f"P(flip2=H) = {p_f2:.4f}")
    print(f"P(flip2=H | flip1=H) = {p_f2_given_f1:.4f}")
    print(f"Not independent: {abs(p_f2 - p_f2_given_f1) > 0.01}\n")

    # Conditional on coin 0
    mask_c0 = coin == 0
    p_f2_c0 = flip2[mask_c0].mean()
    p_f2_given_f1_c0 = flip2[mask_c0 & flip1].mean()
    print("=== Conditional on coin 0 (P(H)=0.3) ===")
    print(f"P(flip2=H | coin=0) = {p_f2_c0:.4f}")
    print(f"P(flip2=H | flip1=H, coin=0) = {p_f2_given_f1_c0:.4f}")
    print(f"Conditionally independent: {abs(p_f2_c0 - p_f2_given_f1_c0) < 0.02}")

mixture_coin_simulation()
```

```python
import numpy as np

def berkson_paradox_simulation(n_simulations=200_000):
    """Demonstrate Berkson's paradox: independent events become
    dependent after conditioning on a shared effect."""
    np.random.seed(42)

    # A = fire (rare), B = burnt toast (common), C = alarm
    p_fire = 0.01
    p_toast = 0.10

    fire = np.random.rand(n_simulations) < p_fire
    toast = np.random.rand(n_simulations) < p_toast
    alarm = fire | toast  # alarm if either occurs

    # Unconditional independence
    p_fire_given_toast = fire[toast].mean()
    print(f"P(fire) = {fire.mean():.4f}")
    print(f"P(fire | toast) = {p_fire_given_toast:.4f}")
    print(f"Unconditionally independent: {abs(fire.mean() - p_fire_given_toast) < 0.005}\n")

    # Conditional on alarm: explaining away
    p_fire_given_alarm = fire[alarm].mean()
    p_fire_given_alarm_no_toast = fire[alarm & ~toast].mean()
    print(f"P(fire | alarm) = {p_fire_given_alarm:.4f}")
    print(f"P(fire | alarm, no toast) = {p_fire_given_alarm_no_toast:.4f}")
    print(f"Conditionally dependent (explaining away): "
          f"{abs(p_fire_given_alarm - p_fire_given_alarm_no_toast) > 0.01}")

berkson_paradox_simulation()
```

---

## 핵심 요약

- 조건부 독립은 $C$를 알고 나면 $A$와 $B$가 서로에 대해 아무 정보도 담지 않는다는 뜻이다.
- 독립은 조건부 독립을 함의하지 **않으며** 그 역도 마찬가지다.
- 버크슨의 역설은 공통 결과로 조건을 걸면 그 독립적인 원인들 사이에 종속성이 생김을 보여준다.
- 조건부 독립은 베이즈 망과 마르코프 모형 뒤에 놓인 구조적 가정이다.

## 연습문제

**연습문제 1.**
구름 $C$, 비 $R$, 우산 $U$에 대해 $P(C) = 0.4$, $P(R \mid C) = 0.6$, $P(R \mid C^c) = 0.1$, $P(U \mid R) = 0.9$, $P(U \mid R^c) = 0.2$이고 $U \perp\!\!\!\perp C \mid R$을 가정한다. (a) $C$와 $U$는 독립인가? (b) $C \perp\!\!\!\perp U \mid R$을 확인하라.

??? success "풀이"
    (a) $P(R) = 0.6 \cdot 0.4 + 0.1 \cdot 0.6 = 0.30$이고 $P(U) = 0.9 \cdot 0.3 + 0.2 \cdot 0.7 = 0.41$이다.

    $P(U \mid C) = P(U \mid R) P(R \mid C) + P(U \mid R^c) P(R^c \mid C) = 0.9 \cdot 0.6 + 0.2 \cdot 0.4 = 0.62$이다.

    $0.62 \ne 0.41$이므로 $C$와 $U$는 조건 없이는 독립이 **아니다**.

    (b) 가정에 의해 $P(U \mid R, C) = P(U \mid R)$이다. 따라서

    $$
    P(C \cap U \mid R) = P(U \mid R, C) P(C \mid R) = P(U \mid R) P(C \mid R)
    $$

    이므로 $C \perp\!\!\!\perp U \mid R$이다. 비가 구름과 우산 사이의 연관을 "가려낸다". 비가 오는지 알고 나면 구름은 우산에 대해 더 이상의 정보를 주지 않는다.

---

**연습문제 2.**
**버크슨의 역설.** $A, B$가 입원 $C$에 대한 서로 독립인 두 위험요인이라 하자. 전체적으로는 독립인데도 $C$를 조건으로 하면 $A$와 $B$가 음의 상관을 갖게 됨을 보여라.

??? success "풀이"
    $A, B \in \{0, 1\}$이 독립인 Bernoulli($p$)이고 $A = 1$ 또는 $B = 1$이면 입원한다고 하자(즉 $C = A \cup B$).

    $P(A = 1 \mid C) = P(A = 1, C)/P(C) = P(A = 1)/P(C) = p/(2p - p^2)$이며, $p = 0.5$이면 $0.5/0.75 = 2/3$이다.

    $P(A = 1 \mid B = 1, C) = P(A = 1 \mid B = 1) = p = 0.5$이다($B = 1$이면 $C$가 따라오므로).

    $P(A = 1 \mid B = 0, C) = 1$이다($B = 0$인 입원자 중에서는 $A$가 반드시 1이어야 하므로).

    따라서 $C$가 주어졌을 때 $B = 1$임을 알면 $P(A = 1)$이 2/3에서 1/2로 *줄어들고*, $B = 0$임을 알면 1로 올라간다. 조건 없이는 독립인데도 $C$를 조건으로 하면 둘이 음으로 연관된다. **공통 결과(충돌변수)로 조건을 걸면 허위 연관이 생긴다.**

    현실적 영향: 입원 환자로 국한된 의학 연구는 독립적인 위험요인들 사이에서 음의 연관을 흔히 발견한다. 버크슨의 병원 편향이다.

---

**연습문제 3.**
**혼합분포.** 동전을 무작위로 고르는데 1번 동전은 $P(H) = 0.3$, 2번 동전은 $P(H) = 0.7$이다. 고른 동전을 두 번 던져 $X_1, X_2$를 얻는다. $X_1 \perp\!\!\!\perp X_2$는 성립하지 않지만 $X_1 \perp\!\!\!\perp X_2 \mid \text{동전}$은 성립함을 보여라.

??? success "풀이"
    동전 선택이 주어지면 두 던지기는 조건부 독립이다(같은 동전이고, 어느 동전인지 주어지면 던지기끼리 독립이다).

    조건 없이는 다음과 같다.

    $P(X_1 = H) = 0.5 \cdot 0.3 + 0.5 \cdot 0.7 = 0.5$이고 $X_2$도 같다.

    $P(X_1 = H, X_2 = H) = 0.5 \cdot 0.3^2 + 0.5 \cdot 0.7^2 = 0.045 + 0.245 = 0.29 \ne 0.25 = 0.5 \cdot 0.5$.

    따라서 조건 없이는 $X_1$과 $X_2$가 *양의 상관*을 갖는다. 첫 던지기가 앞면이면 앞면 쪽으로 치우친 동전이 선택되었을 가능성이 높아 두 번째도 앞면일 가능성이 커진다.

    양의 상관 $\rho(X_1, X_2) > 0$은 잠재적인 동전 선택에 관한 정보를 반영한다. 동전 선택을 알고 나면(즉 그것으로 조건을 걸면) 상관이 사라진다. 이것이 통계학에서 **잠재변수 모형**의 근거다. 관측된 상관이 관측되지 않은 공통 원인으로 설명될 수 있다.

---

**연습문제 4.**
**마르코프 연쇄.** 마르코프 연쇄 $X_0, X_1, X_2, \ldots$는 $X_n \perp\!\!\!\perp \{X_0, \ldots, X_{n-2}\} \mid X_{n-1}$을 만족한다. 이 성질을 말로 표현하고 그것이 어떻게 다루기 쉬운 추론을 가능하게 하는지 설명하라.

??? success "풀이"
    **마르코프 성질**(말로): 현재($X_{n-1}$)가 주어지면 미래($X_n$)는 과거($X_0, \ldots, X_{n-2}$)와 조건부 독립이다. 현재가 과거로부터 미래를 "가려낸다".

    **다루기 쉬운 추론:** $(X_0, \ldots, X_T)$의 결합분포가 다음과 같이 분해된다.

    $$
    P(X_0, X_1, \ldots, X_T) = P(X_0) \prod_{t=1}^T P(X_t \mid X_{t-1})
    $$

    각 조건부확률이 전체 이력이 아니라 바로 앞 상태에만 의존한다. 상태공간의 크기가 $k$로 유한하면 초기 상태 확률 $k$개와 전이 확률 $k^2$개만 필요하고, $T + 1$개 변수에 대한 완전히 일반적인 결합분포가 요구하는 지수적으로 많은 모수가 필요 없다.

    마르코프 연쇄는 언어 모형, 은닉 마르코프 모형, MCMC 표집, 대기행렬 이론, 페이지랭크의 바탕이 된다. 이들을 다루기 쉽게 만드는 것이 바로 이 조건부 독립 구조다.

---

**연습문제 5.**
**공통 원인 구조.** 세 변수 $X, Y, Z$가 "사슬" $X \to Z \to Y$을 이룬다. 분해 $P(X, Y, Z) = P(X) P(Z \mid X) P(Y \mid Z)$로부터 $X \perp\!\!\!\perp Y \mid Z$를 확인하라.

??? success "풀이"
    $P(Y \mid X, Z) = P(X, Y, Z)/P(X, Z) = P(X) P(Z \mid X) P(Y \mid Z)/(P(X) P(Z \mid X)) = P(Y \mid Z)$이다.

    $P(Y \mid X, Z) = P(Y \mid Z)$가 $X$에 의존하지 않으므로 $Y$는 $Z$가 주어졌을 때 $X$와 조건부 독립이다.

    동등한 방식: $P(X \cap Y \mid Z) = P(X \mid Z) P(Y \mid Z, X) = P(X \mid Z) P(Y \mid Z)$.

    **인과적 해석:** $X$가 오직 $Z$를 *통해서만* $Y$에 영향을 준다면($X \to Y$의 직접 화살표가 없다면) $Z$를 알고 난 뒤에는 $X$가 $Y$에 대해 더 이상의 정보를 주지 않는다. 이것이 인과 그래프의 d-분리 규칙에서 말하는 "사슬" 패턴이다.

    이와 대조적으로 **충돌변수** $X \to Z \leftarrow Y$에서는 $Z$로 조건을 걸면 $X$와 $Y$ 사이에 허위 종속성이 생긴다. 사슬 패턴과 정반대다. 사슬, 갈래, 충돌변수를 구별하는 것이 그래프 인과모형의 핵심이다.

---

**연습문제 6.**
**"해명" 효과.** 결과 $E$에 대해 두 원인 $A, B$가 있는 베이즈 망에서 $A$와 $B$의 사전확률이 독립이라 하자. $E$를 관측한 뒤 $A$가 일어났음을 알게 되면 $B$도 일어났을 사후확률이 *줄어든다*. $P(A) = P(B) = 0.1$, $P(E \mid A, B) = 1$, $P(E \mid A, B^c) = 0.8$, $P(E \mid A^c, B) = 0.8$, $P(E \mid A^c, B^c) = 0$으로 보여라.

??? success "풀이"
    결합확률을 계산한다.

    $P(A, B, E) = 0.1 \cdot 0.1 \cdot 1 = 0.01$.
    $P(A, B^c, E) = 0.1 \cdot 0.9 \cdot 0.8 = 0.072$.
    $P(A^c, B, E) = 0.9 \cdot 0.1 \cdot 0.8 = 0.072$.
    $P(A^c, B^c, E) = 0.9 \cdot 0.9 \cdot 0 = 0$.

    $P(E) = 0.01 + 0.072 + 0.072 + 0 = 0.154$.

    사후확률:

    $P(B \mid E) = (0.01 + 0.072)/0.154 = 0.082/0.154 \approx 0.532$.

    $P(B \mid A, E) = P(A, B, E)/P(A, E) = 0.01/(0.01 + 0.072) = 0.01/0.082 \approx 0.122$.

    따라서 결과를 관측한 뒤 $A$도 일어났음을 알게 되면 $B$의 확률이 53%에서 12%로 *떨어진다*. 관측된 결과가 $A$에 의해 "해명된" 것이다. $A$가 $E$를 일으켰음을 알고 나면 대안 원인 $B$의 가능성이 낮아진다.

    **현실 사례:** 경보가 울렸을 때 처음에는 도둑이나 지진을 의심한다. 라디오에서 방금 지진이 났다는 소식을 들으면 경보가 설명되고 도둑 가설의 가능성이 낮아진다. 이것이 베이즈적 추론에서 "경쟁하는 설명"의 형식적 기제다.
