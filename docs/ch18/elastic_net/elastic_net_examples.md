# 엘라스틱넷 보기

## 개요

엘라스틱넷은 $L_1$(라쏘) 벌점과 $L_2$(능형) 벌점을 하나의 정칙화 틀로 결합한다. 라쏘의
희소성 유도 성질을 물려받으면서, 상관된 설명변수 집단을 다루는 능형회귀의 능력도 함께 갖는다.
이 절에서는 엘라스틱넷 목적함수를 세우고 정규직교 경우의 해를 유도하며, 두 조율모수를 고르는
실무 지침을 다룬다.

---

## 1. 엘라스틱넷 목적함수

$X \in \mathbb{R}^{n \times p}$와 $y \in \mathbb{R}^n$이 주어졌을 때 엘라스틱넷은

$$
\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \frac{1}{2n}\| y - X\beta \|_2^2 + \lambda \left[ \alpha \|\beta\|_1 + \frac{1 - \alpha}{2} \|\beta\|_2^2 \right] \right\}
$$

를 푼다. 여기서 $\lambda \ge 0$은 전체 정칙화 강도를, $\alpha \in [0, 1]$은 두 벌점의 배합을
조절한다.

- $\alpha = 1$: 순수 라쏘.
- $\alpha = 0$: 순수 능형회귀.
- $0 < \alpha < 1$: 엘라스틱넷.

---

## 2. 정규직교 계획에서의 해

$X^\top X = n I_p$일 때 $j$번째 계수의 엘라스틱넷 추정치는

$$
\hat{\beta}_j^{\text{EN}} = \frac{1}{1 + \lambda(1 - \alpha)}\, S\!\left(\hat{\beta}_j^{\text{OLS}},\; \lambda \alpha\right)
$$

로 단순해진다. 여기서 $S(\cdot, \cdot)$는 연성 문턱 연산자다. 즉 엘라스틱넷은 먼저 연성
문턱을 적용하고(라쏘 단계) 그다음 축소 배율을 곱한다(능형 단계).

---

## 3. 집단 선택

라쏘에 대한 엘라스틱넷의 큰 장점은 상관된 설명변수에서의 행동이다. 여러 설명변수가 강하게
상관되어 있을 때,

- **라쏘**는 하나만 고르고 나머지를 0으로 만드는 경향이 있다(불안정한 선택).
- **엘라스틱넷**은 집단 전체를 함께 선택하거나 함께 배제하는 경향이 있다.

이 **그룹 효과**는 벌점의 $L_2$ 성분이 강볼록하기 때문에 생기며, Zou와 Hastie(2005)가
증명하였다.

---

## 4. 코드: 기본 시연

서로 거의 같은 설명변수 셋을 만들어 놓고 라쏘와 엘라스틱넷을 나란히 적합한다. 두 방법이
갈리는 자리가 어디인지 계수를 직접 보는 것이 목적이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 라쏘와 엘라스틱넷의 집단 선택. 공통 신호 $z$에 작은 잡음을 더해 서로 거의 같은 설명변수 $x_0, x_1, x_2$를 만들고 잡음변수 셋을 덧붙인다. 참 모형은 $y = 3z + \varepsilon$이다.

**(1)** 세 계수의 합이 $s > 0$으로 고정되고 셋이 모두 음이 아닐 때, $L_1$ 벌점과 $L_2$ 벌점이 **배분**에 따라 각각 어떻게 달라지는지 구하시오. 이로부터 라쏘의 배분이 불안정한 까닭을 설명하시오.

**(2)** 라쏘와 엘라스틱넷을 적합해 (1)의 예측이 맞는지 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** $\beta_0 + \beta_1 + \beta_2 = s$이고 $\beta_j \ge 0$이라 하자.

    $L_1$ 벌점은

    $$
    \lvert\beta_0\rvert + \lvert\beta_1\rvert + \lvert\beta_2\rvert = \beta_0+\beta_1+\beta_2 = s
    $$

    로 **배분과 무관하게 언제나 $s$다.** $(s, 0, 0)$이든 $(s/3, s/3, s/3)$이든 벌점이 똑같다. 즉 $L_1$은 이 평면 위에서 완전히 평평하고 어떤 배분도 선호하지 않는다. 잔차제곱합 쪽도 세 변수가 사실상 같으므로 거의 평평하다. **목적함수가 평평하면 배분을 결정하는 것은 자료의 잡음뿐이고**, 자료를 다시 뽑을 때마다 답이 달라진다.

    $L_2$ 벌점은 다르다. $\beta_j = s/3 + e_j$($\sum e_j = 0$)로 쓰면

    $$
    \sum_j \beta_j^2 = \sum_j\left(\frac{s}{3}+e_j\right)^2
    = \frac{s^2}{3} + \frac{2s}{3}\sum_j e_j + \sum_j e_j^2
    = \frac{s^2}{3} + \sum_j e_j^2
    $$

    이고 $\sum_j e_j^2 \ge 0$이므로 **균등배분 $\beta_j = s/3$에서 유일한 최소 $s^2/3$을 갖는다.** 등호는 $e_j$가 모두 0일 때만이다. 따라서 $\alpha < 1$이어서 $L_2$ 성분이 조금이라도 섞이면 평평하던 바닥이 기울어지고 배분이 곧바로 하나로 정해진다. 이것이 **그룹 효과**다.

    **(2) 수치적으로.**

    ```python
    import numpy as np
    from sklearn.linear_model import ElasticNet, Lasso

    rng = np.random.default_rng(42)

    # 서로 거의 같은 설명변수 셋(x0, x1, x2)을 일부러 만든다. 라쏘는 이런 집단에서
    # 하나만 남기고 나머지를 0으로 보내지만, 엘라스틱넷은 셋을 함께 살린다.
    n = 100
    z = rng.normal(size=n)
    X = np.column_stack([
        z + rng.normal(0, 0.05, n),      # x0
        z + rng.normal(0, 0.05, n),      # x1
        z + rng.normal(0, 0.05, n),      # x2
        rng.normal(size=(n, 3)),         # x3, x4, x5 — 잡음 변수
    ])
    y = 3 * z + rng.normal(0, 1, n)

    # l1_ratio 는 두 벌점의 배합비다. 1 이면 순수 라쏘, 0 이면 순수 능형이다.
    lasso = Lasso(alpha=0.5).fit(X, y)
    enet = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(X, y)

    print("계수 (x0~x2 가 서로 거의 같은 변수):")
    print(f"  라쏘      : {np.round(lasso.coef_, 3)}")
    print(f"  엘라스틱넷: {np.round(enet.coef_, 3)}")
    # 두 방법 모두 잡음 변수는 0 으로 보낸다. 갈리는 곳은 x0~x2 안에서다.
    # 라쏘는 셋에 제멋대로 나누어 주고, 엘라스틱넷은 거의 똑같이 나누어 준다.
    # 이 고르기가 곧 집단 선택이며, 능형 벌점이 하는 일이다.
    print(f"x0~x2 계수의 표준편차 — 라쏘 {lasso.coef_[:3].std():.3f}, "
          f"엘라스틱넷 {enet.coef_[:3].std():.3f}")

    # --- (1) 의 유도를 수로 확인한다 ---
    print(f"x0~x2 사이의 상관 = {np.corrcoef(X[:, :3], rowvar=False)[0, 1]:.4f}")
    for name, c in [("라쏘", lasso.coef_[:3]), ("엘라스틱넷", enet.coef_[:3])]:
        s = c.sum()
        print(f"{name:6s} 합 s = {s:.4f}   L1 = {np.abs(c).sum():.4f}   "
              f"L2^2 = {np.sum(c**2):.4f}   균등배분의 L2^2 = s^2/3 = {s**2/3:.4f}")
    ```

    출력:

    ```
    계수 (x0~x2 가 서로 거의 같은 변수):
      라쏘      : [ 0.505  0.555  0.941  0.    -0.    -0.   ]
      엘라스틱넷: [ 0.705  0.707  0.71  -0.    -0.    -0.   ]
    x0~x2 계수의 표준편차 — 라쏘 0.195, 엘라스틱넷 0.002
    x0~x2 사이의 상관 = 0.9959
    라쏘     합 s = 2.0008   L1 = 2.0008   L2^2 = 1.4480   균등배분의 L2^2 = s^2/3 = 1.3344
    엘라스틱넷  합 s = 2.1227   L1 = 2.1227   L2^2 = 1.5019   균등배분의 L2^2 = s^2/3 = 1.5019
    ```

    ![한 자료에서의 계수 비교와 200번 반복했을 때 집단 내 표준편차의 분포](./img/group_split.png)

    **유도가 수에서 그대로 확인된다.** 두 방법 모두 $L_1$ 값이 세 계수의 합과 소수 넷째 자리까지 **똑같다**($2.0008$과 $2.1227$). (1)에서 말한 "$L_1$은 배분과 무관하다"가 이 등식이다. 반면 $L_2$ 제곱합은 엘라스틱넷에서 $1.5019$로 균등배분의 하한 $s^2/3 = 1.5019$와 **일치**하고, 라쏘에서는 $1.4480$으로 그 하한 $1.3344$보다 $8.5\%$ 크다. 라쏘는 균등배분에서 벗어나 있고 엘라스틱넷은 바닥에 정확히 앉아 있다.

    왼쪽 그림은 위 출력을 막대로 옮긴 것이다. 초록 띠 안의 세 변수는 쌍별 상관이 모두 $0.995$를 넘는 사실상 같은 변수이고, 참값은 셋을 합쳐 $3$이다. 두 방법 모두 **잡음변수 세 개를 정확히 0으로 만들었고 세 계수의 합도 $2.00$ 대 $2.12$로 비슷하다.** 갈리는 곳은 그 합을 어떻게 쪼개느냐뿐이다. 라쏘는 $0.505,\ 0.555,\ 0.941$로 마지막 변수에 두 배 가까이 몰아주고, 엘라스틱넷은 $0.705,\ 0.707,\ 0.710$으로 소수 셋째 자리까지 고르게 나눈다.

    합이 $3$에 못 미치는 것도 읽어 둘 일이다. 벌점이 $\lambda = 0.5$로 세기 때문에 **살아남은 계수도 참값보다 작게 나온다.** 변수를 고르는 일과 계수를 바르게 재는 일은 서로 다른 문제다.

    오른쪽은 그 결과를 200번의 반복으로 확인한 것이다. 세 계수의 표준편차를 재면 엘라스틱넷은 거의 전부 $0.05$ 미만에 쌓이고(중앙값 $0.015$) 라쏘는 $0.2$에서 $1.3$까지 넓게 퍼진다(중앙값 $0.881$). **위 보기에서 본 $0.195$는 라쏘 치고 얌전한 축에 속한 자료였다는 뜻이다.** 자료를 다시 뽑았을 뿐인데 라쏘의 배분은 매번 달라지고, 엘라스틱넷의 배분은 매번 같다. 예측만이 목적이라면 둘 다 쓸 만하지만, 계수를 읽고 보고해야 한다면 이 차이가 결론의 재현성을 가른다.

실무에서는 `sklearn.linear_model.ElasticNetCV`로 $\lambda$와 $\alpha$를 교차검증으로 함께
고른다.

---

## 5. 엘라스틱넷의 좌표하강

좌표하강에서 $j$번째 계수의 갱신식은

$$
\beta_j \leftarrow \frac{S\!\left(X_j^\top r_j / n,\; \lambda \alpha\right)}{1 + \lambda(1 - \alpha)}
$$

이며, 여기서 $r_j = y - X_{-j}\beta_{-j}$는 부분잔차다. 라쏘 갱신식과 비교하면 능형 성분에서
나온 분모 $1 + \lambda(1 - \alpha)$만이 유일한 차이다.

---

## 6. 조율모수의 선택

엘라스틱넷에는 초모수가 $\lambda$와 $\alpha$ 두 개 있다. 흔히 쓰는 전략은 다음과 같다.

1. $\alpha$ 값의 격자를 고정한다(예: $\{0.1, 0.5, 0.7, 0.9, 0.95\}$).
2. 각 $\alpha$에 대해 $\lambda$ 격자 위에서 교차검증한다.
3. CV 오차가 가장 작은 $(\alpha, \lambda)$ 쌍을 고른다.

---

## 7. 해석

- **희소성 + 안정성.** 엘라스틱넷은 변수선택(일부 계수가 정확히 0)을 하면서도 설명변수가
  상관되어 있을 때 라쏘보다 안정적이다.
- **유일한 해.** 라쏘와 달리 엘라스틱넷 목적함수는 $\alpha < 1$일 때 강볼록이므로 해가 항상
  유일하다.
- **계산 비용.** 엘라스틱넷의 좌표하강은 분모만 살짝 바뀔 뿐 본질적으로 라쏘와 같다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> scikit-learn의 `ElasticNetCV`를 써서 $n = 200$, $p = 50$이고 참 계수 중 5개만
0이 아니며 앞의 10개 설명변수끼리 쌍별 상관이 $\rho = 0.95$인 인공자료에 엘라스틱넷을
적합하라. 선택된 $\alpha$, $\lambda$, 그리고 0이 아닌 계수의 개수를 보고하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    n, p = 200, 50

    # 앞의 열 변수는 서로 상관된 덩어리를 이룬다
    Sigma = np.eye(p)
    for i in range(10):
        for j in range(10):
            if i != j:
                Sigma[i, j] = 0.95
    L = np.linalg.cholesky(Sigma)
    X = np.random.randn(n, p) @ L.T

    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n)

    X_s = StandardScaler().fit_transform(X)

    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
        n_alphas=100, cv=5, max_iter=10000
    )
    enet_cv.fit(X_s, y)

    n_nonzero = np.sum(np.abs(enet_cv.coef_) > 1e-6)
    print(f"Selected alpha (l1_ratio): {enet_cv.l1_ratio_}")
    print(f"Selected lambda:           {enet_cv.alpha_:.6f}")
    print(f"Non-zero coefficients:     {n_nonzero}")
    ```

    출력:

    ```
    Selected alpha (l1_ratio): 0.95
    Selected lambda:           0.014625
    Non-zero coefficients:     41
    ```

    실행 결과는 $\alpha = 0.95$, $\lambda = 0.014625$이고 0이 아닌 계수는 **41개**다.
    참 신호가 5개뿐인데 41개가 살아남은 것은 실수가 아니라 교차검증의 성질이다. CV는 예측오차를
    최소화하므로 계수가 아주 작은 잡음변수를 남겨 두는 데 대한 벌칙이 거의 없고, 그 결과
    $\lambda$가 선택 관점에서는 지나치게 작게 잡힌다. 다만 상관 블록 안(앞의 10개 변수)에서는
    6개가 함께 선택되어 그룹 효과가 실제로 나타난다. 순수 라쏘였다면 이 블록에서 보통 한두 개만
    남는다.

    **희소성이 목표라면** CV 최소점을 그대로 쓰지 말고 1-표준오차 규칙(`cv_tuning.md` 참조),
    사후 라쏘, 또는 안정성 선택을 함께 써야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 상관계수 $\rho$가 1에 가까운 두 설명변수 $x_1$, $x_2$를 생각하자. 라쏘는 왜
둘 중 하나만 고르는 반면 엘라스틱넷은 둘 다 고르는 경향이 있는지 정성적으로 설명하고, 제약
영역의 기하와 연결지어라.

</div>

??? success "풀이"

    라쏘의 제약영역 $\|\beta\|_1 \le t$는 좌표축 위에 뾰족한 꼭짓점을 갖는다.
    $x_1 \approx x_2$이면 손실함수의 등고선은 직선 $\beta_1 = \beta_2$와 거의 평행한, 길게
    늘어난 타원이 된다. 이 타원과 마름모꼴 $L_1$ 공이 처음 닿는 곳은 대개 꼭짓점이고, 그곳에서는
    한 계수가 0이다. 그래서 라쏘는 둘 중 하나만 고른다.

    엘라스틱넷의 제약영역은 $L_1$ 마름모와 $L_2$ 공을 섞은 모양이다. $L_2$ 성분이 꼭짓점을
    둥글게 만들므로 $(\beta_1, \beta_2) = (c, c)$ 부근의 경계가 매끄럽다. 길게 늘어난 타원은
    두 계수가 모두 0이 아닌 이 매끄러운 경계에서 처음 닿을 가능성이 더 크고, 그 결과 집단
    선택이 일어난다.

    형식적으로 Zou와 Hastie(2005)는 $x_i^\top x_j / n = \rho$이고
    $\hat{\beta}_i, \hat{\beta}_j \ne 0$이면
    $|\hat{\beta}_i - \hat{\beta}_j| \le \frac{\|y\|_1}{\lambda(1-\alpha)n}\sqrt{2(1 - \rho)}$
    임을 증명하였다. 상관이 높을수록 두 계수가 가까워진다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 엘라스틱넷 목적함수에서 출발하여 좌표하강 갱신식
$\beta_j \leftarrow S(X_j^\top r_j / n,\, \lambda\alpha) / (1 + \lambda(1 - \alpha))$를
유도하라.

</div>

??? success "풀이"

    $\beta_j$를 제외한 모든 계수를 고정하자. $\beta_j$만의 함수로 본 목적함수는

    $$
    g(\beta_j) = \frac{1}{2n}\|r_j - X_j \beta_j\|_2^2 + \lambda\alpha |\beta_j| + \frac{\lambda(1 - \alpha)}{2}\beta_j^2
    $$

    이며, 여기서 $r_j = y - X_{-j}\beta_{-j}$이다. 이차항을 전개하고 $\beta_j$가 없는 항을
    무시하면

    $$
    g(\beta_j) = \frac{1}{2}\!\left(\frac{\|X_j\|^2}{n} + \lambda(1-\alpha)\right)\beta_j^2 - \frac{X_j^\top r_j}{n}\,\beta_j + \lambda\alpha|\beta_j| + C
    $$

    를 얻는다. $\|X_j\|^2/n = 1$이 되도록 표준화되어 있다고 하면,
    $\frac{1}{2}a\, z^2 - b\, z + \lambda\alpha|z|$ 꼴의 함수는 $a = 1 + \lambda(1-\alpha)$일 때

    $$
    \beta_j^* = \frac{S(b,\, \lambda\alpha)}{a} = \frac{S(X_j^\top r_j/n,\, \lambda\alpha)}{1 + \lambda(1-\alpha)}
    $$

    에서 최소가 된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> $\alpha < 1$이고 $\lambda > 0$이면 엘라스틱넷 목적함수가 강볼록임을 증명하고,
해가 유일함을 결론지어라.

</div>

??? success "풀이"

    엘라스틱넷 목적함수는 $f(\beta) = h(\beta) + \lambda\alpha\|\beta\|_1$이고, 여기서

    $$
    h(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \frac{\lambda(1-\alpha)}{2}\|\beta\|_2^2
    $$

    이다. $h$의 헤세행렬은 $\nabla^2 h = \frac{1}{n}X^\top X + \lambda(1-\alpha)I_p$이다.
    $\alpha < 1$이고 $\lambda > 0$이면 $\lambda(1-\alpha)I_p$가 양정치이므로 $\nabla^2 h$도
    양정치이고, 따라서 $h$는 강볼록이다.

    $f = h + \lambda\alpha\|\cdot\|_1$는 강볼록함수와 볼록함수의 합이므로 강볼록이다. 강볼록
    함수는 최소점을 많아야 하나 가지며, $f$의 강제성(coercivity)이 존재성을 보장한다. 따라서
    엘라스틱넷 해는 유일하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 정규직교 계획($X^\top X = nI_p$)에서 엘라스틱넷 추정량이
$\hat{\beta}_j^{\text{EN}} = \frac{1}{1+\lambda(1-\alpha)}\,S(\hat{\beta}_j^{\text{OLS}},\, \lambda\alpha)$
로 쓰임을 보이고, 두 연산(연성 문턱 뒤 재척도)을 기하학적으로 해석하라.

</div>

??? success "풀이"

    $X^\top X = nI_p$이면 엘라스틱넷 목적함수는 $p$개의 독립된 일변량 문제로 분리된다.

    $$
    \min_{\beta_j} \left\{ \frac{1}{2}(\hat{\beta}_j^{\text{OLS}} - \beta_j)^2 + \lambda\alpha|\beta_j| + \frac{\lambda(1-\alpha)}{2}\beta_j^2 \right\}.
    $$

    이차항을 합치면

    $$
    \min_{\beta_j} \left\{ \frac{1 + \lambda(1-\alpha)}{2}\beta_j^2 - \hat{\beta}_j^{\text{OLS}}\beta_j + \lambda\alpha|\beta_j| \right\}
    $$

    이고, 연습문제 3의 근접 연산자 결과에 의해 해는

    $$
    \hat{\beta}_j^{\text{EN}} = \frac{S(\hat{\beta}_j^{\text{OLS}},\, \lambda\alpha)}{1 + \lambda(1-\alpha)}
    $$

    이다.

    **기하학적 해석:** 연성 문턱은 OLS 추정치를 0 쪽으로 평행이동하고 작은 값은 정확히 0으로
    잘라 낸다(희소성을 만드는 라쏘 단계). 이어서 $1 + \lambda(1-\alpha)$로 나누는 것은 살아남은
    계수를 0 쪽으로 균일하게 축소한다(추가 축소를 주는 능형 단계). 두 연산이 합쳐져 희소성과
    연속적 축소를 동시에 얻는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span> 정규직교 계획($X^\top X = nI_p$)에서 $p = 4$이고
$\hat{\beta}^{\text{OLS}} = (3.0,\; 0.8,\; -0.3,\; 0.1)$이며 $\lambda = 0.4$로 고정되어 있다고 하자.

**(1)** 2 절의 공식으로 $\alpha = 1,\ 0.5,\ 0$ 세 경우의 $\hat{\beta}^{\text{EN}}$을 손으로 구하고, 각 경우 0이 아닌 계수의 개수를 적으시오.

**(2)** $\lambda = 0.4$를 고정한 채 $\alpha$만 움직일 때, 0이 아닌 계수가 **정확히 두 개**가 되는 $\alpha$의 범위를 구하시오.

**(3)** (1)의 수를 보면 $\alpha$가 작아질수록 **살아남는 변수는 늘어나는데 가장 큰 계수는 더 많이 줄어든다.** 이 두 가지가 왜 함께 일어나는지 공식의 어느 부분이 무엇을 하는지로 설명하시오.

</div>

??? success "풀이"

    2 절의 공식은
    $\hat{\beta}_j^{\text{EN}} = S(\hat{\beta}_j^{\text{OLS}},\, \lambda\alpha) / (1 + \lambda(1-\alpha))$
    이다. $\lambda = 0.4$이므로 문턱은 $\lambda\alpha = 0.4\alpha$이고 배율의 분모는
    $1 + 0.4(1-\alpha)$이다.

    **(1) $\alpha = 1$** (순수 라쏘). 문턱 $0.4$, 분모 $1$.
    $S$를 각 성분에 적용하면 $(3.0-0.4,\ 0.8-0.4,\ 0,\ 0) = (2.6,\ 0.4,\ 0,\ 0)$이다
    ($\lvert -0.3\rvert = 0.3 < 0.4$, $\lvert 0.1\rvert = 0.1 < 0.4$이므로 둘은 0으로 잘린다).

    $$
    \hat{\beta}^{\text{EN}} = (2.6,\; 0.4,\; 0,\; 0), \qquad \text{0이 아닌 계수 2개}
    $$

    **$\alpha = 0.5$.** 문턱 $0.2$, 분모 $1 + 0.4 \times 0.5 = 1.2$.
    $S$를 적용하면 $(2.8,\ 0.6,\ -0.1,\ 0)$이고 이를 $1.2$로 나누면

    $$
    \hat{\beta}^{\text{EN}} = \left(\tfrac{2.8}{1.2},\; \tfrac{0.6}{1.2},\; -\tfrac{0.1}{1.2},\; 0\right) = (2.3333,\; 0.5000,\; -0.0833,\; 0), \qquad \text{3개}
    $$

    **$\alpha = 0$** (순수 능형). 문턱 $0$이므로 아무것도 잘리지 않고, 분모는 $1 + 0.4 = 1.4$다.

    $$
    \hat{\beta}^{\text{EN}} = \frac{1}{1.4}(3.0,\; 0.8,\; -0.3,\; 0.1) = (2.1429,\; 0.5714,\; -0.2143,\; 0.0714), \qquad \text{4개}
    $$

    **(2)** 변수 $j$가 살아남는 조건은 $\lvert \hat{\beta}_j^{\text{OLS}}\rvert > \lambda\alpha = 0.4\alpha$다. 배율의 분모는 양수이므로 **살아남는지 아닌지는 문턱만이 결정한다.** $\lvert \hat{\beta}_j^{\text{OLS}}\rvert$를 크기순으로 적으면 $3.0 > 0.8 > 0.3 > 0.1$이다. 정확히 두 개가 살아남으려면 $3.0$과 $0.8$은 문턱을 넘고 $0.3$과 $0.1$은 넘지 못해야 하므로

    $$
    0.3 \le 0.4\alpha < 0.8 \quad\Longleftrightarrow\quad 0.75 \le \alpha < 2
    $$

    이고, $\alpha \in [0, 1]$과 교차하면

    $$
    \alpha \in [0.75,\; 1]
    $$

    이다. 즉 $\alpha = 0.75$ 아래로 내려가는 순간 세 번째 변수가 되살아난다.

    **(3)** 공식은 **서로 다른 일을 하는 두 조각의 곱**이다.

    - 문턱 $\lambda\alpha$는 **몇 개를 죽일지**를 정한다. $\alpha$가 작아지면 $\lambda\alpha$가 작아져서 죽는 변수가 줄어든다(2개 → 3개 → 4개).
    - 배율 $1/(1+\lambda(1-\alpha))$는 **살아남은 것을 얼마나 줄일지**를 정한다. $\alpha$가 작아지면 $\lambda(1-\alpha)$가 커져서 분모가 커지고 축소가 세진다($1 \to 1.2 \to 1.4$).

    $\lambda$를 고정한 채 $\alpha$를 내리면 $\lambda\alpha$는 줄고 $\lambda(1-\alpha)$는 늘기 때문에 **두 효과가 반대 방향으로 간다.** 가장 큰 계수에서 이것이 눈에 보인다.

    $$
    2.6 \;>\; 2.3333 \;>\; 2.1429
    $$

    $\alpha = 1$일 때 $3.0$은 $0.4$만 깎여 $2.6$이 되지만, $\alpha = 0$일 때는 $1.4$로 나뉘어 $2.1429$가 된다. **연성 문턱은 큰 계수에 상수만큼의 상처를 주고, 비례 축소는 큰 계수에 비례해서 상처를 준다.** 그래서 큰 계수에는 능형 쪽이 더 아프고 작은 계수에는 라쏘 쪽이 더 아프다. $\alpha$는 "희소성을 얼마나 원하는가"의 손잡이지만, $\lambda$를 함께 조정하지 않으면 희소성을 포기하는 대가로 **살아남은 계수의 편향까지 키운다.** $\square$

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span> 7 절은 엘라스틱넷 해가 $\alpha < 1$일 때 유일하지만 라쏘 해는 그렇지 않다고 말한다. 이것을 가장 단순한 경우에서 끝까지 계산해 보자. $p = 2$이고 두 열이 **완전히 똑같다**고 하자. 즉 $x_1 = x_2 = x$이고 $\lVert x \rVert_2^2 = n$이며 $c := x^\top y / n > 0$이다.

**(1)** $\alpha = 1$(순수 라쏘)일 때 목적함수가 $\beta_1 + \beta_2$를 통해서만 자료에 의존함을 보이고, $c > \lambda$이면 **해집합이 선분 전체**임을 구하시오. 유일하게 정해지는 양은 무엇인가?

**(2)** $0 < \alpha < 1$일 때 해를 닫힌 꼴로 구하고, 두 계수가 **정확히 같음**을 보이시오.

**(3)** 두 열을 하나로 합쳐(즉 $x$ 하나만 쓰는 $p = 1$ 문제로) 2 절의 공식을 쓴 결과와 (2)의 $\beta_1 + \beta_2$를 비교하시오. $c = 1.2$, $\lambda = 0.4$, $\alpha = 0.5$로 수를 넣어 확인하고, 변수를 복제하는 것이 두 벌점 가운데 **어느 쪽을 약화시키는지** 말하시오.

</div>

??? success "풀이"

    $s := \beta_1 + \beta_2$라 쓰자. $x_1 = x_2 = x$이므로 $X\beta = (\beta_1+\beta_2)x = sx$이고, $\lVert x\rVert_2^2 = n$을 쓰면 적합 항은

    $$
    \frac{1}{2n}\lVert y - sx\rVert_2^2 = \frac{1}{2n}\lVert y\rVert_2^2 - \frac{x^\top y}{n}s + \frac{\lVert x\rVert_2^2}{2n}s^2 = \text{const} - cs + \frac{s^2}{2}
    $$

    이다. **적합 항은 $s$만의 함수다.**

    **(1)** $\alpha = 1$이면 목적함수는

    $$
    F(\beta_1,\beta_2) = \text{const} - cs + \frac{s^2}{2} + \lambda\left(\lvert\beta_1\rvert + \lvert\beta_2\rvert\right)
    $$

    이다. $s$를 고정하면 삼각부등식에서 $\lvert\beta_1\rvert + \lvert\beta_2\rvert \ge \lvert s \rvert$이고, **등호는 두 계수의 부호가 같을 때(또는 하나가 0일 때)만** 성립한다. 따라서 $s > 0$인 직선 위에서 최솟값은 $\beta_1, \beta_2 \ge 0$인 선분에서 달성되고 그 값은

    $$
    \varphi(s) = \text{const} - cs + \frac{s^2}{2} + \lambda s = \text{const} + \frac{s^2}{2} - (c - \lambda)s
    $$

    이다. $\varphi'(s) = s - (c-\lambda) = 0$에서 $s^\ast = c - \lambda$이고, $c > \lambda$이므로 이것은 양수다($s \le 0$ 쪽은 $\varphi$가 증가하므로 후보가 아니다). 그러므로 해집합은

    $$
    \left\{(\beta_1, \beta_2) = (t,\; s^\ast - t) \;:\; 0 \le t \le s^\ast \right\}
    $$

    이라는 **선분 전체**이고 원소가 무한히 많다. 유일하게 정해지는 양은 $\beta_1 + \beta_2 = c - \lambda$, 따라서 적합값 $\hat{y} = s^\ast x$와 예측도 유일하다. **개별 계수는 정해지지 않고 합과 예측만 정해진다** — 라쏘의 계수를 그대로 읽고 "이 변수가 중요하다"고 말할 수 없는 까닭이 여기에 있다.

    **(2)** $0 < \alpha < 1$이면 벌점에 $\frac{\lambda(1-\alpha)}{2}(\beta_1^2+\beta_2^2)$이 더해진다. $s$를 고정하고 이 항만 최소화하면, $\beta_1 = s/2 + e$, $\beta_2 = s/2 - e$로 쓸 때

    $$
    \beta_1^2 + \beta_2^2 = \frac{s^2}{2} + 2e^2
    $$

    이므로 $e = 0$, 즉 $\beta_1 = \beta_2 = s/2$에서 **유일한** 최소 $s^2/2$를 갖는다(보기 1 (1)에서 세 변수에 대해 한 계산과 같다). 따라서 $s > 0$인 직선 위의 최솟값은

    $$
    \psi(s) = \text{const} + \frac{s^2}{2} - (c - \lambda\alpha)s + \frac{\lambda(1-\alpha)}{2}\cdot\frac{s^2}{2} = \text{const} + \frac{1}{2}\left(1 + \frac{\lambda(1-\alpha)}{2}\right)s^2 - (c-\lambda\alpha)s
    $$

    이고, $\psi'(s) = 0$에서

    $$
    s^\ast = \frac{(c - \lambda\alpha)_+}{1 + \frac{\lambda(1-\alpha)}{2}} = \frac{2\,(c-\lambda\alpha)_+}{2 + \lambda(1-\alpha)}, \qquad \hat{\beta}_1 = \hat{\beta}_2 = \frac{(c-\lambda\alpha)_+}{2+\lambda(1-\alpha)}
    $$

    이다. $L_2$ 항이 $s$를 쪼개는 방식까지 혼자 결정해 버리므로 해가 하나로 못 박힌다 — 이것이 연습문제 4의 강볼록성이 이 예에서 드러나는 모습이고, 상관이 $\rho = 1$인 극단에서의 **그룹 효과**다.

    **(3)** 열을 하나로 합치면 $p = 1$인 정규직교 문제이므로 2 절의 공식이 그대로 쓰여

    $$
    \hat{\beta}^{\text{merge}} = \frac{(c - \lambda\alpha)_+}{1 + \lambda(1-\alpha)}
    $$

    이다. (2)의 합과 비교하면

    $$
    \frac{s^\ast}{\hat{\beta}^{\text{merge}}} = \frac{2\,(1+\lambda(1-\alpha))}{2+\lambda(1-\alpha)} = \frac{2 + 2\lambda(1-\alpha)}{2+\lambda(1-\alpha)} > 1 \quad (\lambda(1-\alpha) > 0)
    $$

    이므로 **복제된 쪽의 합이 더 크다.** 수를 넣자. $c = 1.2$, $\lambda = 0.4$, $\alpha = 0.5$이면 $\lambda\alpha = 0.2$, $\lambda(1-\alpha) = 0.2$이므로

    $$
    \hat{\beta}_1 = \hat{\beta}_2 = \frac{1.2 - 0.2}{2 + 0.2} = \frac{1.0}{2.2} = 0.4545, \qquad s^\ast = 0.9091
    $$

    이고 합친 쪽은 $\hat{\beta}^{\text{merge}} = 1.0/1.2 = 0.8333$이다.

    까닭은 두 벌점이 복제에 다르게 반응하기 때문이다. 합이 $s$로 고정될 때 복제된 두 계수의

    - $L_1$ 값은 $\lvert\beta_1\rvert+\lvert\beta_2\rvert = s$로 **합치기 전과 같다**(변하지 않는다).
    - $L_2$ 값은 $\beta_1^2+\beta_2^2 = s^2/2$로 **합치기 전 $s^2$의 절반**이다.

    즉 **변수를 둘로 복제하면 $L_1$ 벌점은 그대로이고 $L_2$ 벌점만 반으로 깎인다.** 그래서 라쏘의 합 $c - \lambda$는 복제와 무관하지만, 능형 성분이 섞인 엘라스틱넷은 복제된 쪽에서 축소를 덜 받는다. 상관된 변수 묶음이 클수록 엘라스틱넷이 그 묶음에 주는 총계수가 커지는 경향도 같은 이유다. $\square$

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 연습문제 1의 코드는 `ElasticNetCV`를 부르기 전에 `StandardScaler`로 설명변수를 표준화한다. 왜 그 한 줄이 없으면 안 되는지를 끝까지 계산해 보자. 설명변수가 서로 직교하지만 **척도가 다른** 계획

$$
X^\top X = n\,\operatorname{diag}(d_1, \dots, d_p), \qquad d_j = \frac{\lVert x_j \rVert_2^2}{n}
$$

을 생각하고 $c_j := x_j^\top y / n$이라 쓰자.

**(1)** 엘라스틱넷 해가 좌표별로 분리되어 $\hat{\beta}_j^{\text{EN}} = S(c_j,\, \lambda\alpha) / \left(d_j + \lambda(1-\alpha)\right)$임을 보이고, 변수 $j$가 **탈락하는 조건**을 $c_j$로 쓰시오. $d_j = 1$이면 2 절의 공식으로 돌아감을 확인하시오.

**(2)** $x_j$를 $\gamma x_j$로 바꾸는 것은 **단위만 바꾸는 일**이어서 OLS의 적합값을 전혀 바꾸지 않는다. 그런데 (1)의 탈락 조건은 바뀜을 보이시오. $\lambda = 0.5$, $\alpha = 1$이고 $d_1 = 1$, $c_1 = 0.3$(길이를 미터로 재었다)인 변수를 센티미터로 바꾸어($\gamma = 100$) 라쏘가 이 변수를 **버리는 것에서 거의 OLS 값으로 살리는 것으로** 뒤집히는 것을 수로 확인하시오.

**(3)** 표준화하면 (1)의 탈락 조건이 어떤 꼴이 되는가? `sklearn`의 `Lasso`와 `ElasticNet`은 표준화를 해 주지 않는다는 사실과 묶어, 표준화를 왜 **반드시** 해야 하는지 한 문장으로 적으시오.

</div>

??? success "풀이"

    **(1)** $X^\top X$가 대각이므로 적합 항이 좌표별로 쪼개진다.

    $$
    \frac{1}{2n}\lVert y - X\beta\rVert_2^2 = \frac{1}{2n}\lVert y \rVert_2^2 - \sum_j \frac{x_j^\top y}{n}\beta_j + \sum_j \frac{\lVert x_j\rVert_2^2}{2n}\beta_j^2 = \text{const} + \sum_j\left(\frac{d_j}{2}\beta_j^2 - c_j\beta_j\right)
    $$

    벌점도 좌표별로 쪼개지므로 $j$번째 문제는

    $$
    \min_{\beta_j}\left\{\frac{d_j + \lambda(1-\alpha)}{2}\beta_j^2 - c_j\beta_j + \lambda\alpha\lvert\beta_j\rvert\right\}
    $$

    이다. 연습문제 3에서 쓴 $\frac{a}{2}z^2 - bz + \lambda\alpha\lvert z\rvert$의 최소점 $S(b,\lambda\alpha)/a$를 $a = d_j + \lambda(1-\alpha)$, $b = c_j$에 적용하면

    $$
    \hat{\beta}_j^{\text{EN}} = \frac{S(c_j,\, \lambda\alpha)}{d_j + \lambda(1-\alpha)}
    $$

    이다. $d_j = 1$이면 $c_j = \hat{\beta}_j^{\text{OLS}}$이고 분모가 $1+\lambda(1-\alpha)$가 되어 2 절의 공식과 일치한다. 분모는 언제나 양수이므로 **탈락 조건은 분자만의 일**이다.

    $$
    \hat{\beta}_j^{\text{EN}} = 0 \quad\Longleftrightarrow\quad \lvert c_j \rvert = \frac{\lvert x_j^\top y\rvert}{n} \le \lambda\alpha
    $$

    즉 변수의 운명을 정하는 것은 OLS 계수 $\hat{\beta}_j^{\text{OLS}} = c_j/d_j$가 아니라 $y$와의 **내적** $c_j$다.

    **(2)** $x_j \mapsto \gamma x_j$($\gamma > 0$)이면

    $$
    d_j \mapsto \gamma^2 d_j, \qquad c_j \mapsto \gamma c_j, \qquad \hat{\beta}_j^{\text{OLS}} = \frac{c_j}{d_j} \mapsto \frac{\gamma c_j}{\gamma^2 d_j} = \frac{\hat{\beta}_j^{\text{OLS}}}{\gamma}
    $$

    이다. OLS에서는 계수가 $1/\gamma$배로 바뀌고 열은 $\gamma$배가 되었으므로 기여 $\gamma x_j \cdot \hat{\beta}_j^{\text{OLS}}/\gamma = x_j\hat{\beta}_j^{\text{OLS}}$가 **그대로**다. 단위를 바꾼 것일 뿐이니 당연하다. 그러나 엘라스틱넷의 탈락 조건은 $\lvert\gamma c_j\rvert \le \lambda\alpha$로 바뀌어 **$\gamma$에 의존한다.** $\lambda\alpha$는 그대로인데 비교 대상이 $\gamma$배가 되기 때문이다.

    수로 보자. $\lambda = 0.5$, $\alpha = 1$이므로 문턱은 $\lambda\alpha = 0.5$이고 분모는 $d_1 + 0 = d_1$이다.

    - **미터.** $d_1 = 1$, $c_1 = 0.3$. $\lvert c_1\rvert = 0.3 \le 0.5$이므로 $\hat{\beta}_1^{\text{EN}} = 0$. 라쏘는 **"효과 없음"**이라고 보고한다(OLS는 $\hat{\beta}_1^{\text{OLS}} = 0.3$).
    - **센티미터.** $\gamma = 100$이므로 $d_1' = 10^4$, $c_1' = 30$. 이제 $30 > 0.5$이므로

    $$
    \hat{\beta}_1^{\text{EN}} = \frac{30 - 0.5}{10^4} = 0.00295 \ (\text{cm당})
    $$

    이고, 미터 단위로 환산하면 $100 \times 0.00295 = 0.295$다. **OLS 값 $0.3$과 사실상 같다** — 라쏘가 이 변수를 거의 손대지 않고 살렸다는 뜻이다.

    경계는 $\lvert\gamma c_1\rvert = \lambda\alpha$, 즉 $\gamma = 0.5/0.3 = 1.667$이다. 길이의 단위를 $1.667$배보다 잘게 쪼개는 순간 결론이 "효과 없음"에서 "효과 있음"으로 넘어간다. **자료는 한 글자도 바뀌지 않았고 바뀐 것은 자(尺)뿐이다.**

    덧붙여, $p = 2$에서 $c = (0.3,\ 0.6)$, $d = (1,\ 1)$이면 문턱 $0.5$는 두 번째 변수만 남긴다. 첫 변수를 센티미터로 바꾸면 $c' = (30,\ 0.6)$이 되어 둘 다 남고 $\lvert c_j\rvert$의 **순위까지 뒤집힌다.** 벌점회귀에서 "큰 계수가 중요한 변수"라는 읽기는 척도를 맞춘 뒤에만 뜻을 가진다.

    **(3)** 모든 열을 평균 0, $\lVert x_j\rVert_2^2 = n$(표본표준편차 1)로 표준화하면 $d_j = 1$이고 $c_j = \hat{\beta}_j^{\text{OLS}}$이므로 탈락 조건은

    $$
    \lvert \hat{\beta}_j^{\text{OLS}} \rvert \le \lambda\alpha
    $$

    가 된다. 이제 모든 변수가 **같은 자로 재어진 같은 문턱**을 놓고 경쟁하고, 어느 변수를 어떤 단위로 기록했는지는 결과에 영향을 주지 않는다.

    `sklearn`의 `Lasso`·`ElasticNet`·`ElasticNetCV`는 `fit` 안에서 표준화를 해 주지 않으므로(`normalize` 인수는 없어졌다) 표준화는 **사용자의 책임**이다. 한 문장으로: **$L_1$·$L_2$ 벌점은 척도에 불변이 아니어서 표준화를 빼먹으면 단위 선택이 변수선택을 대신하게 되고, 라이브러리는 그 잘못을 경고해 주지 않는다.**

    (교차검증과 함께 쓸 때는 한 걸음 더 조심해야 한다. 표준화에 쓰는 평균과 표준편차는 **훈련 폴드만으로** 계산해야 하고, 전체 자료로 먼저 표준화하면 검증 폴드의 정보가 새 나가 CV 오차를 과소평가한다. `Pipeline(StandardScaler(), ElasticNetCV())` 꼴이 안전한 이유다.) $\square$

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 엘라스틱넷의 $\alpha = 0$ 끝점, 곧 능형회귀가 **언제나** OLS보다 나은 $\lambda$를 가짐을 보이자. 정규직교 계획 $X^\top X = nI_p$에서 $y = X\beta + \varepsilon$, $\varepsilon \sim N(0, \sigma^2 I_n)$이라 하면 $\hat{\beta}^{\text{OLS}} = X^\top y/n \sim N(\beta,\, \tau^2 I_p)$이고 $\tau^2 = \sigma^2/n$이다. $\alpha = 0$에서 2 절의 공식은 $\hat{\beta}^{\text{EN}} = \hat{\beta}^{\text{OLS}} / (1+\lambda)$다.

**(1)** 총 평균제곱오차 $M(\lambda) = \sum_{j=1}^p \mathbb{E}\big[(\hat{\beta}_j^{\text{EN}} - \beta_j)^2\big]$를 편향–분산 분해로 $\lambda$의 함수로 적으시오.

**(2)** $M'(\lambda)$를 구해 $M'(0) < 0$임을 보이고($\sigma > 0$인 한 언제나), 최소점 $\lambda^\ast$와 그때의 값을 구하시오. $M(\lambda^\ast) = M(0)/(1+\lambda^\ast)$라는 깔끔한 꼴로 정리될 것이다.

**(3)** 연습문제 1의 참 계수 $\beta = (3, -2, 4, -1, 2)$, $p = 5$, $n = 200$, $\sigma = 1$에 수를 넣어 $\lambda^\ast$와 이득의 크기를 구하시오. 그리고 이득이 **절반 이상**(즉 $M(\lambda^\ast) \le M(0)/2$)이 되는 조건을 $p$, $n$, $\sigma$, $\lVert\beta\rVert_2$로 쓰고 해석하시오.

</div>

??? success "풀이"

    **(1)** $\hat{\beta}_j^{\text{EN}} = \hat{\beta}_j^{\text{OLS}}/(1+\lambda)$이고 $\hat{\beta}_j^{\text{OLS}} \sim N(\beta_j, \tau^2)$이므로

    $$
    \mathbb{E}\hat{\beta}_j^{\text{EN}} = \frac{\beta_j}{1+\lambda}, \qquad \text{편향} = \frac{\beta_j}{1+\lambda} - \beta_j = -\frac{\lambda\beta_j}{1+\lambda}, \qquad \operatorname{Var} = \frac{\tau^2}{(1+\lambda)^2}
    $$

    이다. 편향–분산 분해(5 장 정리 1)로 좌표별 MSE는 $\dfrac{\lambda^2\beta_j^2 + \tau^2}{(1+\lambda)^2}$이고, $B := \lVert\beta\rVert_2^2 = \sum_j \beta_j^2$라 쓰면

    $$
    M(\lambda) = \frac{\lambda^2 B + p\tau^2}{(1+\lambda)^2}, \qquad \tau^2 = \frac{\sigma^2}{n}
    $$

    이다. $\lambda = 0$에서 $M(0) = p\tau^2 = p\sigma^2/n$으로 OLS의 MSE이고, $\lambda \to \infty$에서 $M \to B$로 $\hat{\beta} = 0$의 MSE다.

    **(2)** 미분한다.

    $$
    M'(\lambda) = \frac{2\lambda B(1+\lambda)^2 - (\lambda^2 B + p\tau^2)\cdot 2(1+\lambda)}{(1+\lambda)^4} = \frac{2\left[\lambda B(1+\lambda) - \lambda^2 B - p\tau^2\right]}{(1+\lambda)^3} = \frac{2\left(\lambda B - p\tau^2\right)}{(1+\lambda)^3}
    $$

    분자가 $\lambda$의 **일차식**으로 깔끔하게 줄어든다. 따라서

    $$
    M'(0) = -2p\tau^2 = -\frac{2p\sigma^2}{n} < 0
    $$

    이다. $\sigma > 0$이면 $\lambda = 0$에서 $M$은 **반드시 감소**하므로, 충분히 작은 $\lambda > 0$에 대해 $M(\lambda) < M(0)$이다. 즉 **참 계수가 무엇이든 OLS를 이기는 능형 모수가 존재한다.** 까닭은 분해를 보면 분명하다. $\lambda = 0$ 근방에서 편향제곱은 $\lambda^2$의 차수로 ($O(\lambda^2)$) 늘지만 분산은 $1/(1+\lambda)^2 \approx 1-2\lambda$로 **일차**로 줄기 때문이다. **처음 한 걸음은 거래가 아니라 공짜다.**

    $M'$의 부호는 $\lambda B - p\tau^2$의 부호와 같으므로 $M$은 $\lambda < \lambda^\ast$에서 감소, $\lambda > \lambda^\ast$에서 증가한다.

    $$
    \lambda^\ast = \frac{p\tau^2}{B} = \frac{p\sigma^2}{n\lVert\beta\rVert_2^2}
    $$

    이 값을 넣으면 분자가 $\dfrac{p^2\tau^4}{B^2}B + p\tau^2 = p\tau^2\left(\dfrac{p\tau^2}{B}+1\right) = p\tau^2(1+\lambda^\ast)$이므로

    $$
    M(\lambda^\ast) = \frac{p\tau^2(1+\lambda^\ast)}{(1+\lambda^\ast)^2} = \frac{p\tau^2}{1+\lambda^\ast} = \frac{M(0)}{1+\lambda^\ast}
    $$

    이다. **이득의 비율이 $\lambda^\ast$ 하나로 요약된다.**

    **(3)** $B = 9+4+16+1+4 = 34$, $p = 5$, $\tau^2 = \sigma^2/n = 1/200 = 0.005$이므로

    $$
    \lambda^\ast = \frac{5 \times 0.005}{34} = \frac{0.025}{34} = 7.3529 \times 10^{-4}
    $$

    $$
    M(0) = p\tau^2 = 0.025, \qquad M(\lambda^\ast) = \frac{0.025}{1.00073529} = 0.0249816
    $$

    이다. 줄어든 양은 $1 - 1/(1+\lambda^\ast) = 0.000735$, 즉 **고작 $0.07\%$다.** 신호 $B = 34$가 잡음 $p\tau^2 = 0.025$보다 1000배 이상 크니 축소할 거리가 거의 없다. ($p = 50$으로 늘리면 $\lambda^\ast = 0.00735$, 이득 $0.73\%$로 열 배가 된다. 추정할 모수가 많아질수록 능형이 벌어 가는 몫이 커진다.)

    이득이 절반 이상이려면 $M(0)/(1+\lambda^\ast) \le M(0)/2$, 즉 $\lambda^\ast \ge 1$이어야 하므로

    $$
    \frac{p\sigma^2}{n} \ge \lVert\beta\rVert_2^2
    $$

    이다. 좌변은 OLS의 총분산이고 우변은 신호의 총세기다. **능형이 OLS를 크게 이기는 것은 신호가 잡음에 묻히는 곳**, 곧 $p$가 $n$에 비해 크거나 $\sigma$가 크거나 계수가 작은 곳이다. 반대로 $n$이 넉넉하고 신호가 센 (3)의 설정에서는 OLS도 이미 충분히 좋아 정칙화가 벌 것이 없다. **"편향을 사서 분산을 판다"는 거래는 분산이 비쌀 때만 남는 장사다.** $\square$

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 앞의 세 문제에서 손으로 얻은 닫힌 꼴을 `sklearn`으로 확인한다. 이 쪽의 목적함수와 `sklearn`의 `ElasticNet`은 `alpha` $= \lambda$, `l1_ratio` $= \alpha$로 **정확히 같은 것**을 최소화한다는 점을 먼저 확인하라.

**(1)** QR 분해로 $X^\top X = nI_p$인 계획을 만들고 잡음 없이 $y = X\beta$로 두어 $\hat{\beta}^{\text{OLS}} = (2.5,\ 0.9,\ -0.45,\ 0.15)$가 정확히 나오게 하라. $\lambda = 0.5$를 고정하고 $\alpha \in \{1,\ 0.8,\ 0.5,\ 0.2\}$에서 2 절의 닫힌 꼴과 `ElasticNet`의 계수를 나란히 찍어 일치를 보이고, 문턱·배율·0이 아닌 계수의 개수가 연습문제 6에서 설명한 대로 움직이는지 확인하라.

**(2)** 똑같은 열 두 개로 이루어진 계획에서 연습문제 7의 결과를 확인하라. 라쏘는 **개별 계수가 유일하지 않으므로 합만** 찍고, 엘라스틱넷은 개별 계수를 찍어 둘이 같은지 보라. 엘라스틱넷의 합을 열을 하나로 합친 경우와도 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import ElasticNet, Lasso

    rng = np.random.default_rng(0)


    def soft(b, t):
        """연성 문턱 S(b, t) = sign(b) (|b| - t)_+."""
        return np.sign(b) * np.maximum(np.abs(b) - t, 0.0)


    def clean(v):
        """출력 직전에 -0.0 과 수치 먼지를 0 으로 눌러 둔다."""
        return np.where(np.abs(v) < 1e-10, 0.0, v)


    # === (1) 정규직교 계획에서 2 절의 닫힌 꼴을 확인한다 ===
    # QR 분해로 X^T X = n I 를 정확히 만든다.
    n, p = 200, 4
    Q, _ = np.linalg.qr(rng.normal(size=(n, p)))
    X = np.sqrt(n) * Q
    beta_ols = np.array([2.5, 0.9, -0.45, 0.15])
    y = X @ beta_ols  # 잡음이 없으면 OLS 추정치가 beta_ols 와 정확히 같다

    print(f"X^T X / n 이 I 에서 벗어난 양 < 1e-10: "
          f"{np.abs(X.T @ X / n - np.eye(p)).max() < 1e-10}")
    print("OLS = X^T y / n = ["
          + " ".join(f"{v:7.4f}" for v in clean(X.T @ y / n)) + "]")
    print()

    lam = 0.5
    print("alpha  문턱  배율   닫힌 꼴                            sklearn                            일치  0 아닌 개수")
    for a in [1.0, 0.8, 0.5, 0.2]:
        closed = soft(beta_ols, lam * a) / (1.0 + lam * (1.0 - a))
        en = ElasticNet(alpha=lam, l1_ratio=a, fit_intercept=False,
                        max_iter=100000, tol=1e-12).fit(X, y)
        same = np.abs(closed - en.coef_).max() < 1e-10
        nz = int(np.sum(np.abs(en.coef_) > 1e-8))
        cs = " ".join(f"{v:7.4f}" for v in clean(closed))
        ss = " ".join(f"{v:7.4f}" for v in clean(en.coef_))
        print(f"{a:4.2f}  {lam*a:5.2f} {1+lam*(1-a):5.2f}  [{cs}]  [{ss}]  {str(same):5s}     {nz}")

    # === (2) 완전히 겹친 두 열: 라쏘의 합과 엘라스틱넷의 개별 계수 ===
    x = X[:, 0]                      # ||x||^2 = n
    X2 = np.column_stack([x, x])     # 똑같은 열 두 개
    y2 = 1.2 * x                     # c = x^T y2 / n = 1.2
    c = x @ y2 / n
    print()
    print(f"c = x^T y / n = {c:.4f}")

    lam, a = 0.4, 1.0                # 순수 라쏘
    las = Lasso(alpha=lam * a, fit_intercept=False, max_iter=100000, tol=1e-12).fit(X2, y2)
    print(f"라쏘     계수의 합 = {las.coef_.sum():.4f}   닫힌 꼴 (c - la)_+ = {max(c - lam*a, 0.0):.4f}")

    lam, a = 0.4, 0.5                # 엘라스틱넷
    en2 = ElasticNet(alpha=lam, l1_ratio=a, fit_intercept=False,
                     max_iter=100000, tol=1e-12).fit(X2, y2)
    each = (c - lam * a) / (2.0 + lam * (1.0 - a))
    print(f"엘라스틱넷 계수     = [{en2.coef_[0]:.4f} {en2.coef_[1]:.4f}]   "
          f"닫힌 꼴 (c - la)_+/(2 + l(1-a)) = {each:.4f}")
    print(f"엘라스틱넷 계수의 합 = {en2.coef_.sum():.4f}   "
          f"열을 하나로 합쳤을 때 = {soft(np.array([c]), lam*a)[0]/(1+lam*(1-a)):.4f}")
    ```

    출력:

    ```text
    X^T X / n 이 I 에서 벗어난 양 < 1e-10: True
    OLS = X^T y / n = [ 2.5000  0.9000 -0.4500  0.1500]

    alpha  문턱  배율   닫힌 꼴                            sklearn                            일치  0 아닌 개수
    1.00   0.50  1.00  [ 2.0000  0.4000  0.0000  0.0000]  [ 2.0000  0.4000  0.0000  0.0000]  True      2
    0.80   0.40  1.10  [ 1.9091  0.4545 -0.0455  0.0000]  [ 1.9091  0.4545 -0.0455  0.0000]  True      3
    0.50   0.25  1.25  [ 1.8000  0.5200 -0.1600  0.0000]  [ 1.8000  0.5200 -0.1600  0.0000]  True      3
    0.20   0.10  1.40  [ 1.7143  0.5714 -0.2500  0.0357]  [ 1.7143  0.5714 -0.2500  0.0357]  True      4

    c = x^T y / n = 1.2000
    라쏘     계수의 합 = 0.8000   닫힌 꼴 (c - la)_+ = 0.8000
    엘라스틱넷 계수     = [0.4545 0.4545]   닫힌 꼴 (c - la)_+/(2 + l(1-a)) = 0.4545
    엘라스틱넷 계수의 합 = 0.9091   열을 하나로 합쳤을 때 = 0.8333
    ```

    **모수 대응.** `sklearn`이 최소화하는 것은
    $\frac{1}{2n}\lVert y - Xw\rVert_2^2 + \texttt{alpha}\cdot\texttt{l1\_ratio}\,\lVert w\rVert_1 + \frac{\texttt{alpha}(1-\texttt{l1\_ratio})}{2}\lVert w\rVert_2^2$
    이고 이 쪽의 1 절 목적함수는
    $\frac{1}{2n}\lVert y - X\beta\rVert_2^2 + \lambda\alpha\lVert\beta\rVert_1 + \frac{\lambda(1-\alpha)}{2}\lVert\beta\rVert_2^2$
    이므로 `alpha` $= \lambda$, `l1_ratio` $= \alpha$로 **항마다 똑같다.** 그래서 변환 없이 바로 비교할 수 있다. (교과서에서 흔히 쓰는 $\lVert y-X\beta\rVert_2^2 + \tilde{\lambda}\lVert\beta\rVert_1$ 꼴과 맞추려면 $\tilde{\lambda} = n\lambda\alpha$로 고쳐야 한다. 두 규약을 섞으면 수가 전부 틀어진다.)

    **(1)** 네 줄 모두 `일치 True`다 — 닫힌 꼴과 좌표하강이 배정밀도 안에서 같은 답을 준다. 문턱 $\lambda\alpha$는 $0.5 \to 0.4 \to 0.25 \to 0.10$으로 줄고 배율의 분모 $1+\lambda(1-\alpha)$는 $1.00 \to 1.10 \to 1.25 \to 1.40$으로 늘며, 0이 아닌 계수는 $2 \to 3 \to 3 \to 4$로 **늘어난다.** 연습문제 6 (3)에서 말한 두 효과가 그대로다. 가장 큰 계수를 따라가면

    $$
    2.0000 \;>\; 1.9091 \;>\; 1.8000 \;>\; 1.7143
    $$

    으로 **변수를 더 많이 살리는 쪽이 살아남은 계수를 더 세게 줄인다.** $\alpha$를 내릴 때 $\lambda$를 함께 내려 주지 않으면 희소성과 맞바꾸는 것이 편향이라는 뜻이다. $\lvert \hat{\beta}_3^{\text{OLS}}\rvert = 0.45$가 $\alpha = 1$($0.5$)에서는 죽고 $\alpha = 0.8$($0.4$)에서는 살아나는 것도 눈여겨볼 만하다 — $\lambda$를 그대로 둔 채 $\alpha$만 건드려도 선택 집합이 바뀐다.

    **(2)** 라쏘의 계수 합은 $0.8000$으로 닫힌 꼴 $(c - \lambda\alpha)_+ = 1.2 - 0.4 = 0.8$과 일치한다. **개별 계수는 찍지 않았다.** 연습문제 7 (1)에서 보았듯 $(t,\ 0.8-t)$가 모두 최적해이므로 좌표하강이 어느 $t$에서 멈출지는 열의 순서와 구현에 달린 일이고, 그 수를 책에 싣는 것은 재현되지 않는 값을 싣는 것이다. 유일하게 정해진 양만 보고해야 한다.

    엘라스틱넷은 두 계수가 $0.4545$로 **소수 넷째 자리까지 같고** 닫힌 꼴 $(1.2-0.2)/(2+0.2) = 0.4545$와 일치한다. $\alpha < 1$이 목적함수를 강볼록으로 만들어(연습문제 4) 해가 하나로 못 박힌 결과다. 합은 $0.9091$로, 두 열을 하나로 합쳐 2 절의 공식을 쓴 $0.8333$보다 크다. 연습문제 7 (3)에서 유도한 대로 **복제가 $L_1$ 벌점은 그대로 두고 $L_2$ 벌점만 반으로 깎기** 때문이다.

    두 결과를 한 줄로 묶으면 이렇다. **라쏘는 예측은 주지만 계수를 주지 못하고, 엘라스틱넷은 둘 다 준다.** 상관된 변수 묶음에서 계수를 읽어 보고해야 한다면 $\alpha$를 1보다 작게 두어야 하는 실질적인 이유다. $\square$

---

## 정리하며

엘라스틱넷은 **두 벌점을 섞는다.**

$$
\lambda\left(\alpha\|\boldsymbol\beta\|_1+\frac{1-\alpha}{2}\|\boldsymbol\beta\|_2^2\right)
$$

- **라쏘의 희소성과 능형의 안정성을 함께 얻는다.** $\alpha=1$ 이면 라쏘, $\alpha=0$ 이면 능형이며 그 사이를 연속적으로 잇는다.
- **그룹 효과가 핵심 장점이다.** 상관된 변수들을 **함께 선택하거나 함께 버린다.** 라쏘가 하나만 고르고 나머지를 버리는 불안정성을 $L_2$ 항이 완화한다.
- **$p\gg n$ 에서 유용하다.** 라쏘의 "최대 $n$ 개" 제약을 넘어설 수 있다.
- **조율 모수가 둘이다.** $\lambda$ 와 $\alpha$ 를 2차원 격자로 교차검증해야 하므로 계산이 는다.
- **유전체학처럼 상관된 변수 집단이 있는 자료**에서 특히 자주 쓰인다.

다음 절 **교차검증과 λ 조율**로 넘어간다.
