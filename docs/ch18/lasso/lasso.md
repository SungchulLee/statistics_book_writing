# 라쏘 회귀 (L1 정칙화)


## 1. 동기: 희소성과 변수선택

능형회귀는 계수를 0 쪽으로 축소하지만 결코 정확히 0으로 만들지 않는다. 많은 응용에서, 특히 $p$가 클 때 우리는 무관한 계수를 정확히 0으로 만들어 유의미한 변수를 자동으로 **선택**하는 추정량을 원한다.

Tibshirani(1996)가 도입한 **라쏘**(Least Absolute Shrinkage and Selection Operator)는 L1 벌점을 써서 정칙화와 변수선택을 동시에 달성한다.

---

## 2. 라쏘의 정식화

$$
\hat{\boldsymbol{\beta}}_{\text{lasso}} = \arg\min_{\boldsymbol{\beta}} \left\{ \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 + \lambda \|\boldsymbol{\beta}\|_1 \right\}
$$

동등한 제약 형태는

$$
\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2 \quad \text{제약} \quad \sum_{j=1}^p |\beta_j| \leq t
$$

이다.

<div class="storybox" markdown>

**그림으로 보는 통계학 1.** 한 단계를 덜어 낸 것이 전부였다

티브시라니는 $1996$ 년 논문이 나온 지 $15$ 년 뒤에 그 논문을 돌아보는 글을 썼다. 「생각의 내력」이라는 절은 이렇게 시작한다. **"라쏘는 그저 $\ell_1$ 노름 벌점을 붙인 회귀이고, $\ell_1$ 노름은 오래전부터 있었다! 가장 직접적으로 영향을 준 것은 레오 브라이먼의 비음 개럿이었다."**

브라이먼이 $1995$ 년에 내놓은 **비음 개럿**은 이렇게 생겼다. 먼저 보통의 최소제곱 $\hat\beta_j$ 를 구해 놓고, 거기에 곱할 비음 계수 $c_j$ 를 고른다.

$$
\min_{\mathbf c}\ \sum_{i=1}^{n}\Bigl(y_i - \sum_j c_j x_{ij}\hat\beta_j\Bigr)^{2}
\quad \text{제약} \quad c_j \ge 0,\ \ \sum_{j=1}^{p} c_j \le t
$$

위 $\S 2$ 의 라쏘와 나란히 놓으면 **거의 같은 물건**이다. 둘 다 $\ell_1$ 꼴 제약으로 계수를 깎고 일부를 정확히 $0$ 으로 만든다. 다른 것은 개럿이 $\hat\beta_j$ 라는 **디딤돌을 하나 먼저 놓는다**는 점뿐이다.

티브시라니가 그다음에 적은 문장이 전부다. **"이것은 $p > N$ 일 때 정의되지 않는데($1995$ 년에는 뜨거운 주제가 아니었다!), 그래서 나는 두 단계를 하나로 합쳤다(캐나다 사람으로서 좀 더 순한 이름도 쓰고 싶었다)."** 개럿은 목을 조르는 형구이고 올가미는 던지는 밧줄이다.

![p/n 을 바꿔 가며 잰 비음 개럿과 라쏘의 계수, 시험오차, 변수선택 F₁](./img/garrote_lasso.png)

그 한 걸음이 무엇을 바꾸었는지 재 보았다. $n = 100$, 참 계수 다섯 개, 잡음 표준편차 $1$ 로 두고 $p$ 를 $10$ 에서 $160$ 까지 늘린다. 둘 다 $5$ 겹 교차검증으로 조율했고 접마다 최소제곱을 다시 구했다. $10$ 번 되풀이한 평균이다.

| $p/n$ | 개럿 RMSE | 라쏘 RMSE | 개럿 $F_1$ | 라쏘 $F_1$ |
|---|---|---|---|---|
| $0.10$ | $1.034$ | $1.046$ | $0.930$ | $0.783$ |
| $0.40$ | $1.034$ | $1.104$ | $\mathbf{0.982}$ | $\mathbf{0.595}$ |
| $0.85$ | $1.029$ | $1.106$ | $0.879$ | $0.450$ |
| $0.99$ | $1.204$ | $1.124$ | $0.425$ | $0.483$ |
| $1.60$ | **정의되지 않음** | $1.166$ | — | $0.429$ |

예상과 다른 결과가 나왔다. **$p$ 가 $n$ 보다 넉넉히 작은 구간에서는 브라이먼의 것이 더 낫다.** 예측오차는 $1.03$ 대 $1.10$ 으로 조금 앞서는 정도지만, **고른 변수집합의 $F_1$ 은 $0.98$ 대 $0.60$ 으로 크게 앞선다**(그림 ③). 까닭은 어렵지 않다. 교차검증은 예측오차를 최소로 하는 $\lambda$ 를 고르고, 그 $\lambda$ 에서 라쏘의 지지집합은 **너무 넓은 것으로 잘 알려져 있다.** 개럿은 $\hat\beta_j$ 가 이미 잡음을 한 번 걸러 준 뒤에 깎기 때문에 가짜를 덜 집는다.

그러니 $1995$ 년의 기준으로 보면 개럿이 밀릴 이유가 없었다. 밀린 것은 $p/n$ 이 $1$ 에 닿을 때다. $p/n = 0.99$ 에서 개럿의 $F_1$ 이 $0.425$ 로 주저앉고, $p \ge n$ 에서는 $\hat\beta$ 자체가 없으므로 **방법이 성립하지 않는다**(그림 ②의 붉은 구역). 같은 자리에서 라쏘는 $p/n$ 이 $0.1$ 에서 $1.6$ 으로 $16$ 배 커지는 동안 RMSE $1.05$–$1.17$ 에 그대로 머문다.

$1995$ 년에 뜨거운 주제가 아니었던 그 조건이 그 뒤 이십 년 동안 통계학에서 가장 뜨거운 주제가 되었다. 유전자 수만 개에 표본 수백 개, 파장 수천 곳에 시료 수십 개, 특징 수백만 개에 문서 수만 개. 라쏘가 개럿을 이긴 것은 더 좋은 추정량이어서가 아니라 **디딤돌을 딛지 않아서**다. 좋은 방법이 사라지는 방식은 대개 이렇다 — 더 나은 것이 나와서가 아니라, 그 방법이 서 있던 전제가 사라져서다.

_확인한 곳 — 티브시라니의 $2011$ 년 『왕립통계학회지 B』 논문 「라쏘를 통한 회귀 축소와 변수선택: 회고」($73$권 $3$호 $273$–$282$쪽) 원문. 인용한 두 문장과 개럿의 정식화, 브라이먼 $1995$ 년 논문이 『테크노메트릭스』 $37$권 $738$–$754$쪽이라는 서지를 모두 그 글에서 확인했다. 그림과 표의 수는 모두 직접 모의실험해 계산한 값이다_

</div>

---

## 3. L1이 희소성을 낳는 이유: 기하적 논증

L1 제약집합은 2차원에서 **마름모**, 고차원에서 교차다면체다. 그 꼭짓점이 좌표축 위에 놓인다. OLS 등고선이 타원이므로 타원과 마름모의 첫 접촉이 대개 꼭짓점에서 일어나고, 그곳에서 하나 이상의 좌표가 정확히 0이다.

반면 L2 제약(구)에는 꼭짓점이 없어 접촉점이 좌표축 위에 놓이는 일이 거의 없다.

| 벌점 | 제약 모양 | 축 위의 꼭짓점 | 정확한 0 |
|---|---|---|---|
| L1 (라쏘) | 마름모 / 교차다면체 | 있음 | 있음 |
| L2 (능형) | 구 / 초구 | 없음 | 없음 |

자세한 논증은 [기하적 해석 (L1 벌점)](geometry.md)을 보라.

---

## 4. 부분미분과 연성 문턱 연산자

L1 노름 $|\beta_j|$는 $\beta_j = 0$에서 **미분 불가능**하므로 **부분미분**을 쓴다.

$$
\partial |\beta_j| = \begin{cases} \{+1\} & \beta_j > 0 \\ [-1, +1] & \beta_j = 0 \\ \{-1\} & \beta_j < 0 \end{cases}
$$

**직교 설계**($\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$)라는 특수한 경우에 라쏘 해는 닫힌 형태를 갖는다.

$$
\hat{\beta}_j^{\text{lasso}} = S_\lambda(\hat{\beta}_j^{\text{OLS}}) = \text{sign}(\hat{\beta}_j^{\text{OLS}})\max(|\hat{\beta}_j^{\text{OLS}}| - \lambda, 0)
$$

이것이 **연성 문턱** 연산자다. $[-\lambda, \lambda]$ 안의 계수는 정확히 0이 되고 그보다 큰 계수는 $\lambda$만큼 축소된다.

능형(직교인 경우)과 비교하면

$$
\hat{\beta}_j^{\text{ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \lambda}
$$

이다. 능형은 비례 축소를, 라쏘는 잘라내기를 동반한 평행이동 축소를 적용한다.

![세 축소 연산자의 모양과 계수 12개에 적용했을 때의 이동](./img/shrinkage_operators.png)

왼쪽은 세 방법이 OLS 계수를 어떤 값으로 바꾸는지를 $\lambda = 1$에서 그린 것이다. 회색 점선이 아무것도 하지 않는 경우($y = x$)이고, 세 곡선이 그 아래로 얼마나 내려오는지가 축소의 양이다. 능형(파랑)은 직선 $z/2$로, 어디서나 절반으로 줄인다. 라쏘(주황)는 주황색 띠 안에서는 완전히 납작하고 밖에서는 $y = x$와 나란한 직선이다. 최량 부분집합(초록)은 띠 밖에서 점선과 겹친다. **문턱을 넘은 계수를 건드리지 않는 유일한 방법**이다.

세 곡선의 차이가 실전에서 어떻게 나타나는지는 오른쪽이 보여준다. OLS 계수 12개를 크기순으로 늘어놓고 라쏘가 어디로 옮기는지 화살표로 표시했다. $2.90 \to 1.90$, $-3.57 \to -2.57$처럼 **모든 화살표의 길이가 정확히 $\lambda = 1$로 같다.** 띠 안에 있던 다섯 개($0.92,\ 0.77,\ 0.41,\ 0.25,\ -0.19$)는 화살표조차 보이지 않는데, 옮겨갈 곳이 0이기 때문이다.

같은 자리의 파란 사각형(능형)을 보면 대조가 분명해진다. $0.25$였던 계수는 능형에서 $0.13$이 되어 여전히 모형 안에 남는다. 큰 계수 $2.90$은 능형에서 $1.45$로 라쏘의 $1.90$보다 **더 많이** 깎였다. 비례 축소는 큰 계수를 더 많이 깎고 작은 계수는 살려 두며, 평행이동 축소는 모두에게 같은 양을 요구해 작은 것부터 쓰러뜨린다.

여기서 라쏘의 두 얼굴이 한꺼번에 드러난다. 0이 된 다섯 개는 **변수선택**이고, 살아남은 일곱 개가 각각 $\lambda$만큼 작아진 것은 **축소 편향**이다. 라쏘라는 이름(Least Absolute Shrinkage and Selection Operator)이 두 단어를 모두 담고 있는 이유이며, 선택된 계수를 효과크기로 읽으면 안 되는 이유이기도 하다.

---

## 5. 좌표하강 알고리즘

일반적인(직교가 아닌) 계획행렬에서는 닫힌 형태 해가 없다. 표준 알고리즘은 **좌표하강**(Friedman et al., 2010)이다.

1. $\hat{\boldsymbol{\beta}} = \mathbf{0}$으로 초기화한다.
2. $j = 1, \ldots, p$를 순회하며 부분잔차를 계산하고 $\hat{\beta}_j \leftarrow S_\lambda\left(\frac{1}{n}\sum_i x_{ij} r_i^{(j)}\right)$로 갱신한다.
3. 수렴할 때까지 반복한다.

각 좌표 갱신이 연성 문턱 형태이므로 알고리즘이 효율적이다. R의 `glmnet`과 파이썬의 `sklearn.linear_model.Lasso`가 이 알고리즘을 쓴다. 자세한 내용은 [좌표하강 알고리즘](coordinate_descent.md)을 보라.

### 좌표하강으로 풀어 보기

다음은 순환 좌표하강으로 라쏘를 푸는 순수 NumPy 구현이다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 좌표하강으로 라쏘 풀기. 나머지 계수를 고정한 채 $\beta_j$ 하나만 최적화하는 일을 돌아가며 되풀이한다.

**(1)** $\beta_j$에 대한 일변량 문제를 풀어 갱신식을 유도하시오. 아래 코드는 `beta[j] = soft_threshold(rho_j, lam)` 로 끝나는데, 이 한 줄이 옳으려면 $X$에 어떤 조건이 필요한가.

**(2)** 구현을 scikit-learn의 `Lasso` 와 맞추어 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 부분잔차를 $r_j = y - X\beta + x_j\beta_j$라 두면 $\beta_j$만 남긴 목적함수는

    $$
    f(\beta_j) = \frac{1}{2n}\lVert r_j - x_j\beta_j\rVert_2^2 + \lambda\lvert\beta_j\rvert + \text{상수}
    $$

    이다. 제곱항을 펼치고 $\beta_j$와 무관한 부분을 상수로 묶으면

    $$
    f(\beta_j) = \frac{a_j}{2}\beta_j^2 - \rho_j\beta_j + \lambda\lvert\beta_j\rvert + \text{상수},
    \qquad
    a_j = \frac{\lVert x_j\rVert_2^2}{n},
    \quad
    \rho_j = \frac{x_j^\top r_j}{n}
    $$

    이다. $a_j > 0$이므로 $f$는 강볼록이고 최솟값이 하나뿐이다. $\beta_j \ne 0$인 곳에서 미분하면 $a_j\beta_j - \rho_j + \lambda\operatorname{sign}(\beta_j) = 0$이고, $\beta_j = 0$이 최적일 조건은 하위기울기가 0을 품는 것, 곧 $\lvert\rho_j\rvert \le \lambda$다. 셋을 합치면

    $$
    \hat\beta_j = \frac{1}{a_j}\,S(\rho_j, \lambda),
    \qquad
    S(\rho,\lambda) = \operatorname{sign}(\rho)\,(\lvert\rho\rvert - \lambda)_+
    $$

    이다. **분모 $a_j$가 코드에는 없다.** 그러므로 이 구현이 옳으려면

    $$
    a_j = \frac{\lVert x_j\rVert_2^2}{n} = 1 \quad \text{(모든 } j\text{)}
    $$

    이어야 하고, 이는 각 열을 평균 0·표준편차 1로 **표준화했을 때** 정확히 성립한다(모분산 꼴로 $n$으로 나누는 표준화여야 한다). 코드가 `X` 를 미리 표준화하라고 적어 둔 것은 벌점의 공평성 때문만이 아니라 **갱신식 자체가 그것을 가정하기 때문**이다.

    **(2) 수치적으로.**

    ```python
    import numpy as np

    def soft_threshold(rho, lam):
        """연성 문턱 연산자. 좌표하강의 갱신식에 쓰인다.

        |rho| 가 lam 보다 작으면 0 으로 보내고, 크면 그만큼 깎아 0 쪽으로 당긴다.
        라쏘가 계수를 정확히 0 으로 만들 수 있는 까닭이 이 평평한 구간에 있다.
        능형의 갱신식에는 이런 구간이 없어 0 이 될 수 없다.
        """
        if rho > lam:
            return rho - lam
        elif rho < -lam:
            return rho + lam
        return 0.0

    def lasso_cd(X, y, lam, max_iter=1000, tol=1e-6):
        """좌표하강으로 라쏘를 푼다.

        L1 벌점은 0 에서 미분이 되지 않아 정규방정식 같은 닫힌 해가 없다.
        대신 계수를 하나씩 돌아가며 나머지를 고정한 채 최적화하면, 각 단계가
        연성 문턱 한 줄로 끝난다. 라쏘의 표준적인 푸는 법이다.

        매개변수
        --------
        X   : (n, p) 설계행렬. 미리 표준화해야 한다.
        y   : (n,)   반응벡터
        lam : L1 벌점 모수

        돌려주는 값
        ----------
        beta : (p,) 계수벡터
        """
        n, p = X.shape
        beta = np.zeros(p)
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                # j 번째 변수의 몫만 되살린 부분잔차. 나머지 변수의 설명은
                # 이미 빼 놓은 상태이므로, 여기에 j 만 단순회귀하는 셈이 된다.
                r_j = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j] @ r_j / n
                beta[j] = soft_threshold(rho_j, lam)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta

    # --- (1) 의 조건과 결과를 확인한다 ---
    from sklearn.linear_model import Lasso

    np.random.seed(42)
    n, p = 150, 10
    X_raw = np.random.randn(n, p)
    beta_true = np.array([4.0, -3.0, 2.0] + [0.0] * 7)
    y = X_raw @ beta_true + np.random.randn(n) * 2
    X = (X_raw - X_raw.mean(axis=0)) / X_raw.std(axis=0)

    print("||x_j||^2 / n =", np.round((X ** 2).sum(axis=0) / n, 6))
    for lam in (1.0, 0.2212, 0.05):
        mine = lasso_cd(X, y, lam)
        sk = Lasso(alpha=lam, max_iter=100000, tol=1e-12).fit(X, y).coef_
        print(f"lam = {lam:<7} 최대 차이 = {np.abs(mine - sk).max():.3e}")
    ```

    출력:

    ```
    ||x_j||^2 / n = [1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    lam = 1.0     최대 차이 = 2.753e-08
    lam = 0.2212  최대 차이 = 4.225e-08
    lam = 0.05    최대 차이 = 6.010e-08
    ```

    **$a_j = \lVert x_j\rVert^2/n$이 열 개 모두 정확히 1이다.** `np.std` 가 기본적으로 $n$으로 나누므로 표준화의 부산물로 이 조건이 공짜로 따라온다. 그래서 분모를 생략한 갱신식이 옳고, 세 $\lambda$에서 scikit-learn과 $10^{-8}$ 수준까지 같은 답이 나온다. 남은 차이는 수렴 판정 `tol=1e-6` 때문이지 식이 달라서가 아니다.

    만약 `X_raw` 를 그대로 넣었다면 $a_j \ne 1$이므로 같은 코드가 **틀린 답**을 준다. 표준화는 벌점을 공평하게 걸기 위한 권고이기 전에 이 구현의 **전제조건**이다.

    각 단계에서 알고리즘은 부분잔차 $r_j = y - X\beta + X_j \beta_j$를 계산하고, 일변량 최소제곱 기울기 $\rho_j = X_j^\top r_j / n$을 구한 뒤 연성 문턱을 적용한다.

---

## 6. 라쏘 해 경로

$\lambda$가 $\lambda_{\max}$(모든 계수가 0)에서 0(OLS)까지 변하면 계수가 하나씩 모형에 들어온다. 경로 $\hat{\beta}_j(\lambda)$는 $\lambda$에 대해 **조각별 선형**이다(에프론 외, 2004의 LARS 결과).

$$
\lambda_{\max} = \frac{1}{n}\|\mathbf{X}^\top\mathbf{y}\|_\infty
$$

이며 $\lambda \geq \lambda_{\max}$에서 라쏘 해는 $\hat{\boldsymbol{\beta}} = \mathbf{0}$이다.

### 경로를 격자 위에서 그려 보기

$\lambda$ 격자를 큰 값에서 작은 값으로 훑으면 계수들이 어떤 순서로 모형에 들어오는지 볼 수 있다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 계수 경로 구하기. 보기 1의 해법기를 $\lambda$ 격자 위에서 되풀이해 경로를 얻는다.

**(1)** 경로가 모두 0에서 출발하는 지점 $\lambda_{\max}$를 구하고, 격자로 읽은 "변수가 들어온 $\lambda$"가 **참 꺾임점과 다를 수밖에 없는** 까닭을 말하시오.

**(2)** 격자가 준 진입 $\lambda$와 LARS가 준 정확한 꺾임점을 나란히 놓고 비교하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** 보기 1의 갱신식에서 모든 $\beta_j$가 0일 때 $r_j = y$이므로 $\rho_j = x_j^\top y / n$이다. 연성 문턱이 모든 좌표를 0으로 두려면 $\lvert\rho_j\rvert \le \lambda$가 모든 $j$에서 성립해야 하므로

    $$
    \lambda_{\max} = \max_j \frac{\lvert x_j^\top y\rvert}{n}
    $$

    이고, 영벡터는 이 $\lambda$ 이상에서 고정점이 된다. 이 값보다 조금 낮추면 최댓값을 달성하던 좌표가 처음으로 0을 벗어난다.

    경로의 참 꺾임점은 이렇게 **연속적인 $\lambda$**에 대해 정의된다. 그러나 `lasso_path` 는 미리 정한 유한한 격자점에서만 해를 구하므로, 진입을 감지하는 자리는 언제나 **꺾임점 아래 첫 격자점**이다. 따라서 격자로 읽은 진입 $\lambda$는 참값보다 **반드시 작거나 같다.** 격자를 촘촘히 할수록 가까워지지만 같아지지는 않는다.

    **(2) 수치적으로.**

    ```python
    def lasso_path(X, y, lambdas):
        """lambda 격자 위에서 계수 경로를 구한다.

        lambda 를 크게 잡으면 모든 계수가 0 이고, 줄여 갈수록 하나씩 살아난다.
        먼저 살아나는 변수가 그만큼 중요하다는 뜻이라, 경로 그림 자체가
        변수 선택의 이야기를 담는다.
        """
        coefs = []
        for lam in lambdas:
            beta = lasso_cd(X, y, lam)
            coefs.append(beta.copy())
        return np.array(coefs)

    # --- (1) 의 lambda_max 와 꺾임점을 확인한다 ---
    lambdas = np.logspace(1, -2, 60)
    path = lasso_path(X, y, lambdas)
    n_nz = (np.abs(path) > 1e-8).sum(axis=1)

    lam_max = np.abs(X.T @ (y - y.mean())).max() / n
    print(f"이론 lambda_max = {lam_max:.4f}")
    print(f"격자에서 계수가 모두 0 인 가장 작은 lambda = "
          f"{lambdas[n_nz == 0][-1]:.4f},  바로 다음 격자점 = {lambdas[(n_nz > 0).argmax()]:.4f}")

    from sklearn.linear_model import lars_path
    knots, active, _ = lars_path(X, y - y.mean(), method='lasso')
    print("정확한 꺾임점 앞 넷 =", np.round(knots[:4], 4))
    print("그 자리에서 들어온 변수 =", [f"x{j+1}" for j in active[:4]])
    print("격자가 준 진입 lambda  =",
          [f"{lambdas[(np.abs(path[:, j]) > 1e-8).argmax()]:.4f}" for j in range(4)])
    ```

    출력:

    ```
    이론 lambda_max = 4.2017
    격자에서 계수가 모두 0 인 가장 작은 lambda = 4.4062,  바로 다음 격자점 = 3.9194
    정확한 꺾임점 앞 넷 = [4.2017 3.3635 2.1672 0.1892]
    그 자리에서 들어온 변수 = ['x1', 'x2', 'x3', 'x4']
    격자가 준 진입 lambda  = ['3.9194', '3.1012', '1.9415', '0.1867']
    ```

    **유도한 $\lambda_{\max} = 4.2017$이 LARS의 첫 꺾임점과 소수 넷째 자리까지 같다.** 격자는 이 값을 $4.4062$와 $3.9194$ 사이에 가두기만 할 뿐 집어내지 못한다. 두 격자점의 비가 $10^{3/59} = 1.124$이기 때문이다.

    네 변수의 진입을 나란히 놓으면 **격자값이 언제나 참 꺾임점보다 작다.** $4.2017 > 3.9194$, $3.3635 > 3.1012$, $2.1672 > 1.9415$, $0.1892 > 0.1867$로 네 번 모두 그렇고, 이는 (1)에서 말한 그대로다. 우연이 아니라 격자 탐색의 구조가 보장하는 방향이다.

    들어오는 순서가 $x_1, x_2, x_3$으로 참 계수의 크기 $(4, -3, 2)$ 순서와 같다는 점, 그리고 네 번째 진입이 $2.1672$에서 $0.1892$로 **열 배 넘게 떨어진 뒤에야** 일어난다는 점이 이 자료의 신호가 깨끗하다는 증거다.

---

## 7. λ의 선택

능형에서처럼 **교차검증**으로 고른다. 표준 관행은 $\lambda_{\max}$에서 $0.001 \cdot \lambda_{\max}$까지 로그 척도로 100개 정도의 격자를 잡아 전체 해 경로를 계산하고, 각 점에서 CV 오차를 구해 최적 $\lambda$를 고르는 것이다.

### 교차검증으로 람다를 고르기

5-겹 교차검증으로 각 후보 $\lambda$의 예측오차를 추정한다.

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 교차검증 MSE. 함수의 주석은 "벌점이 강할수록 훈련오차는 반드시 커진다"고 적어 두었다.

**(1)** 그 "반드시"를 증명하시오. 곧 $\lambda_1 < \lambda_2$이면 두 해의 잔차제곱합이 $\mathrm{RSS}(\hat\beta_{\lambda_1}) \le \mathrm{RSS}(\hat\beta_{\lambda_2})$임을 보이시오. 이로부터 $\lambda$를 훈련오차로 고를 수 없는 까닭이 나오는가.

**(2)** 훈련오차와 교차검증 오차를 나란히 계산해 확인하고, 이 구현에서 눈여겨볼 점을 지적하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** $L(\beta,\lambda) = \frac{1}{2n}\mathrm{RSS}(\beta) + \lambda\lVert\beta\rVert_1$라 두고 $\hat\beta_1 = \hat\beta_{\lambda_1}$, $\hat\beta_2 = \hat\beta_{\lambda_2}$라 하자. 각자가 자기 $\lambda$에서 최소이므로

    $$
    L(\hat\beta_1,\lambda_1) \le L(\hat\beta_2,\lambda_1),
    \qquad
    L(\hat\beta_2,\lambda_2) \le L(\hat\beta_1,\lambda_2)
    $$

    이다. 두 부등식을 더하면 $\frac{1}{2n}\mathrm{RSS}$ 항이 양변에서 같아져 모두 지워지고

    $$
    \lambda_1\lVert\hat\beta_1\rVert_1 + \lambda_2\lVert\hat\beta_2\rVert_1
    \;\le\;
    \lambda_1\lVert\hat\beta_2\rVert_1 + \lambda_2\lVert\hat\beta_1\rVert_1
    $$

    곧 $(\lambda_2-\lambda_1)\big(\lVert\hat\beta_2\rVert_1 - \lVert\hat\beta_1\rVert_1\big) \le 0$만 남는다. $\lambda_2 > \lambda_1$이므로

    $$
    \lVert\hat\beta_2\rVert_1 \le \lVert\hat\beta_1\rVert_1
    $$

    이다. 이것을 첫 부등식에 넣으면

    $$
    \frac{\mathrm{RSS}(\hat\beta_1)}{2n}
    \le \frac{\mathrm{RSS}(\hat\beta_2)}{2n} + \lambda_1\big(\lVert\hat\beta_2\rVert_1 - \lVert\hat\beta_1\rVert_1\big)
    \le \frac{\mathrm{RSS}(\hat\beta_2)}{2n}
    $$

    이 되어 $\mathrm{RSS}(\hat\beta_1) \le \mathrm{RSS}(\hat\beta_2)$다. **훈련오차는 $\lambda$에 대해 비감소함수다.**

    그러므로 훈련오차를 최소화하는 $\lambda$는 언제나 격자의 **가장 작은 값**이고, 선택 기준으로 쓸 수 없다. 자료를 떼어 두고 재는 수밖에 없으며 그것이 교차검증이다.

    **(2) 수치적으로.**

    ```python
    def cv_lasso(X, y, lambdas, folds=5):
        """lambda 마다 k겹 교차검증 MSE 를 구한다.

        벌점이 강할수록 훈련오차는 반드시 커진다. 그러니 lambda 는 훈련자료가
        아니라 떼어 놓은 자료에서 재야 고를 수 있다.
        """
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        fold_size = n // folds
        cv_mse = np.zeros(len(lambdas))

        for k in range(folds):
            val_idx = indices[k * fold_size:(k + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]
            for i, lam in enumerate(lambdas):
                beta = lasso_cd(X_tr, y_tr, lam)
                pred = X_va @ beta
                cv_mse[i] += np.mean((y_va - pred) ** 2)
        return cv_mse / folds

    # --- (1) 의 단조성을 확인한다 ---
    np.random.seed(42)
    grid = np.logspace(1, -2, 30)
    train_mse = np.array([np.mean((y - X @ lasso_cd(X, y, l)) ** 2) for l in grid])
    print(f"훈련 MSE 가 lambda 에 대해 단조증가인가? "
          f"{bool(np.all(np.diff(train_mse[::-1]) >= -1e-12))}")
    print("훈련 MSE (lambda = 10, 1, 0.2212, 0.01):",
          [f"{np.mean((y - X @ lasso_cd(X, y, l)) ** 2):.3f}" for l in (10, 1, 0.2212, 0.01)])

    cv = cv_lasso(X, y, grid)
    print(f"CV MSE  최솟값 {cv.min():.3f} at lambda = {grid[cv.argmin()]:.4f}")
    print(f"CV MSE  양 끝: lambda = {grid[0]:.2f} -> {cv[0]:.3f},  "
          f"lambda = {grid[-1]:.2f} -> {cv[-1]:.3f}")
    ```

    출력:

    ```
    훈련 MSE 가 lambda 에 대해 단조증가인가? True
    훈련 MSE (lambda = 10, 1, 0.2212, 0.01): ['35.154', '6.848', '4.398', '4.157']
    CV MSE  최솟값 4.622 at lambda = 0.1743
    CV MSE  양 끝: lambda = 10.00 -> 35.154,  lambda = 0.01 -> 4.964
    ```

    **단조성이 격자 30점 전체에서 성립한다.** 훈련 MSE는 $\lambda$가 줄수록 $35.154 \to 6.848 \to 4.398 \to 4.157$로 끝까지 내려가고, 훈련오차만 보면 가장 작은 $\lambda$가 언제나 이긴다.

    교차검증 곡선은 다르다. $\lambda = 10$에서 $35.154$(계수가 전부 0이라 훈련오차와 같다)로 시작해 $\lambda = 0.1743$에서 $4.622$로 바닥을 치고, $\lambda = 0.01$에서 $4.964$로 **다시 올라간다.** 훈련오차가 $4.398$에서 $4.157$로 내려가는 바로 그 구간에서 검증오차는 거꾸로 간다. 이 어긋남이 과적합이고, 교차검증이 하는 일은 그것을 드러내는 것이다.

    같은 자료인데 보기 4가 보고하는 $\hat\lambda = 0.2212$, 최소 CV MSE $4.774$와 여기의 $0.1743$, $4.622$가 다르다. **겹을 나누는 난수가 다르기 때문이다.** 보기 4는 자료를 새로 뽑느라 난수를 소비한 뒤 겹을 나누고, 여기서는 씨앗을 고정한 직후에 나눈다. 아래 셋째 항목이 지적하는 바로 그 취약함이며, 한 번의 교차검증이 고른 $\lambda$를 소수점까지 믿어서는 안 된다는 뜻이기도 하다.

    **구현에서 눈여겨볼 점 셋.**

    - `np.random.shuffle` 이 전역 난수 상태를 쓴다. 함수 안에 씨앗이 없으므로 **호출 전에 씨앗을 고정하지 않으면 결과가 매번 달라진다.** 위 코드가 재현되는 것은 바로 앞에서 `np.random.seed(42)` 를 불렀기 때문이다.
    - `fold_size = n // folds` 라서 $n$이 `folds` 로 나누어떨어지지 않으면 **나머지 관측이 어느 검증 겹에도 들어가지 않는다.** 여기서는 $150 = 5 \times 30$이라 버려지는 것이 없지만 일반적으로는 손실이다.
    - 표준화를 **함수 바깥에서 전체 자료로** 해 두었다. 엄밀하게는 훈련 겹의 평균과 표준편차만 써야 하며, 지금 꼴은 검증 겹의 정보가 조금 새어 든다. $n$이 크면 영향이 작지만 원리상 결함이다.

---

## 8. 계수를 0 으로 보내므로 변수선택이 된다

**1. 희소성.** 일부 $\hat{\beta}_j = 0$인 희소해를 만들어 자동 변수선택이 가능하다.

**2. 편향.** 라쏘 계수는 0 쪽으로 편향된다. 선택된 변수에 대해서도 점근적으로 편향이 사라지지 않는다.

**3. 일치성.** **비대표 조건**(Zhao and Yu, 2006) 아래에서 라쏘는 확률 1로 올바른 변수 집합을 고른다. 이 조건이 없으면 모형선택 일치성이 깨질 수 있다.

**4. 예측.** 희소성 가정 아래에서 거의 최적인 예측오차율을 달성한다.

### 경로와 교차검증을 한 번에

설명변수 10개 중 3개만 실제로 관련 있는 인공자료를 만들어 전체 절차를 돌려 본다.

<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> 경로와 교차검증 실행. $n = 150$, $p = 10$이고 참 계수는 $(4, -3, 2, 0, \dots, 0)$, 잡음은 $N(0, 2^2)$이다. 설계는 독립 표준정규라 거의 직교다.

**(1)** 설계가 **정확히** 직교라면($X^\top X/n = I$) 라쏘 해가 최소제곱 해의 무엇이 되는가. 이로부터 교차검증이 고른 $\hat\lambda$에서 세 계수가 얼마가 될지 예측하시오. 또 교차검증 MSE가 내려갈 수 없는 바닥은 얼마인가.

**(2)** 전체 절차를 돌려 두 예측을 확인하시오.

</div>

??? success "풀이"

    **(1) 해석적으로.** $X^\top X/n = I$이면 보기 1의 갱신식에서 $\rho_j = x_j^\top(y - X\beta)/n + \beta_j$이고 비대각 성분이 없으므로 한 바퀴만 돌면 수렴한다. 그 값은

    $$
    \hat\beta_j^{\text{lasso}} = S\!\left(\hat\beta_j^{\text{OLS}},\, \lambda\right)
    = \operatorname{sign}(\hat\beta_j^{\text{OLS}})\left(\lvert\hat\beta_j^{\text{OLS}}\rvert - \lambda\right)_+
    $$

    이다. **라쏘는 최소제곱 해를 연성 문턱에 통과시킨 것**이고, 살아남은 계수는 절댓값이 정확히 $\lambda$만큼 깎인다. 이 자료는 완전히 직교하지는 않으므로 등식이 아니라 좋은 근사다.

    바닥은 잡음에서 온다. 잡음이 $N(0, 2^2)$이므로 새 관측에 대한 예측오차는 $\sigma^2 = 4$ 아래로 내려갈 수 없다(cv_tuning 절의 보기와 같은 논증이다). 교차검증 MSE가 $4$에 얼마나 가까운지가 곧 추정이 얼마나 잘되었는지의 척도다.

    **(2) 수치적으로.**

    ```python
    import matplotlib.pyplot as plt

    np.random.seed(42)

    # 열 변수 중 참으로 쓰이는 것은 앞의 셋뿐이다. 교차검증으로 고른 라쏘가
    # 그 셋을 찾아내는지 보는 것이 목적이다.
    n, p = 150, 10
    X_raw = np.random.randn(n, p)
    beta_true = np.array([4.0, -3.0, 2.0, 0, 0, 0, 0, 0, 0, 0])
    y = X_raw @ beta_true + np.random.randn(n) * 2

    # L1 벌점은 계수의 크기를 그대로 재므로, 변수의 단위가 다르면 벌점이
    # 불공평하게 걸린다. 표준화는 선택이 아니라 필수다.
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0)
    X = (X_raw - X_mean) / X_std

    # lambda 격자는 로그 눈금으로 잡는다. 벌점이 곱셈으로 작동하기 때문이다.
    lambdas = np.logspace(1, -2, 60)
    path = lasso_path(X, y, lambdas)

    # 교차검증으로 lambda 고르기
    lambdas_cv = np.logspace(1, -2, 30)
    mse_cv = cv_lasso(X, y, lambdas_cv)
    best_idx = int(np.argmin(mse_cv))
    best_lam = lambdas_cv[best_idx]

    beta_best = lasso_cd(X, y, best_lam)
    n_nonzero = np.sum(np.abs(beta_best) > 1e-8)

    print(f"Best lambda (5-fold CV):  {best_lam:.4f}")
    print(f"Non-zero coefficients:    {n_nonzero}  (true: 3)")
    print(f"Min CV MSE:               {mse_cv[best_idx]:.3f}")

    # --- 직교 근사가 예측하는 축소량과 견준다 ---
    from sklearn.linear_model import LinearRegression
    ols = LinearRegression().fit(X, y).coef_
    pred = np.sign(ols) * np.maximum(np.abs(ols) - best_lam, 0)
    print(f"X'X/n 이 단위행렬에서 벗어난 최대 크기 = {np.abs(X.T @ X / n - np.eye(p)).max():.4f}")
    print(f"  최소제곱          {np.round(ols[:3], 4)}")
    print(f"  연성문턱 예측      {np.round(pred[:3], 4)}")
    print(f"  라쏘 실제          {np.round(beta_best[:3], 4)}")
    print(f"잡음 바닥 sigma^2 = 4,  최소 CV MSE = {mse_cv[best_idx]:.3f}  "
          f"(차이 {mse_cv[best_idx] - 4:.3f})")
    ```

    출력:

    ```
    Best lambda (5-fold CV):  0.2212
    Non-zero coefficients:    3  (true: 3)
    Min CV MSE:               4.774
    X'X/n 이 단위행렬에서 벗어난 최대 크기 = 0.1857
      최소제곱          [ 3.7252 -3.129   1.8683]
      연성문턱 예측      [ 3.504  -2.9078  1.6471]
      라쏘 실제          [ 3.5463 -2.9119  1.6434]
    잡음 바닥 sigma^2 = 4,  최소 CV MSE = 4.774  (차이 0.774)
    ```

    **직교 근사가 잘 맞는다.** 연성 문턱이 예측한 $(3.504,\ -2.908,\ 1.647)$과 실제 라쏘 해 $(3.546,\ -2.912,\ 1.643)$의 차이가 셋 다 $0.05$ 아래다. $X^\top X/n$의 비대각 성분이 최대 $0.186$으로 0이 아니기 때문에 정확히 같지는 않으며, 가장 크게 어긋난 첫 계수의 차이 $0.042$가 그 비직교성의 값이다.

    추정된 계수 $(3.546,\, -2.912,\, 1.643,\, 0, \dots, 0)$은 참값 $(4, -3, 2, 0, \dots, 0)$을 모두 0 쪽으로 축소한 값이다. **그런데 축소의 출발점은 참값이 아니라 최소제곱 추정값이다.** 최소제곱이 이미 $(3.725, -3.129, 1.868)$로 참값에서 흔들려 있었고, 라쏘는 거기서 다시 $\hat\lambda = 0.2212$씩 깎았다. 두 몫을 섞어 "라쏘의 편향"이라 부르면 안 된다.

    최소 CV MSE $4.774$는 잡음 바닥 $4$보다 $0.774$ 높다. **어떤 방법도 $4$ 아래로는 내려갈 수 없으므로** 이 자료에서 남은 개선의 여지는 그 $0.774$뿐이다. 모형을 더 다듬기 전에 잡음 바닥부터 가늠해 보아야 하는 이유다.

    ![보기 4 자료의 라쏘 정칙화 경로와 5-겹 교차검증 곡선](./img/path_cv.png)

    같은 자료를 그림으로 옮긴 것이다. 왼쪽 경로에서 굵은 세 선이 참 변수, 가는 회색 일곱 선이 잡음변수다. 오른쪽에서 고른 $\hat\lambda = 0.2212$를 두 그림에 보라색 세로선으로 표시했다.

    **경로를 오른쪽에서 왼쪽으로 읽어야 한다.** $\lambda_{\max} = 4.20$보다 큰 곳에서는 모든 계수가 0이다. $\lambda$를 낮추면 $x_1$이 가장 먼저 들어오고 $x_2$, $x_3$이 뒤따르며(보기 2에서 꺾임점을 $4.2017$, $3.3635$, $2.1672$로 정확히 구했다) 들어오는 순서가 참 계수의 크기 순서 $(4.0,\ 3.0,\ 2.0)$와 같다. 반면 잡음변수 일곱 개는 $\lambda = 0.19$ 아래로 내려가야 겨우 움직이기 시작한다. **신호와 잡음 사이에 $2.17$부터 $0.19$까지 열 배 넘는 빈 구간이 있고, 교차검증이 고른 $0.2212$가 그 구간 안에 들어온다.** 변수선택이 성공하는 자료란 이런 간격이 있는 자료다.

    오른쪽 CV 곡선은 왼쪽이 평평하고 오른쪽이 가파른 L자다. 과소정칙화는 싸고 과대정칙화는 비싸다. $\lambda$를 키우면 계수가 축소될 뿐 아니라 변수가 아예 사라지기 때문이다.

---

## 9. 상관된 집단에서 하나만 임의로 고른다

1. **집단 선택.** 강하게 상관된 설명변수에서 라쏘는 하나만 고르고 나머지를 버리는 경향이 있으며, 그 선택이 임의적이다. 능형은 모두 남기고 계수 질량을 분배한다.
2. **$p > n$ 한계.** $p > n$일 때 라쏘는 최대 $n$개의 변수만 고를 수 있다.
3. **선택된 계수의 편향.** 0이 아닌 계수가 체계적으로 0 쪽으로 편향된다. 사후 라쏘 OLS 로 그 눌림을 되돌릴 수 있지만 **그 자리에 선택 편향이 들어온다** — 18.7절 [실무 지침](../overview/guidelines.md) $\S 6$ 의 상자가 약한 신호에서 $+71.7\%$ 를 잰다.

---

## 10. 베이즈 해석

라쏘는 **라플라스(이중지수) 사전분포** 아래의 MAP 추정량이다.

$$
\beta_j \overset{\text{iid}}{\sim} \text{Laplace}(0, b)
$$

라플라스 분포는 0에서 뾰족하고 가우스보다 꼬리가 두꺼워 MAP 추정에서 정확한 희소성을 촉진한다.

| 정칙화 | 사전분포 | 0에서의 밀도 |
|---|---|---|
| 능형 (L2) | $N(0, \tau^2)$ | 매끄럽고 유한 |
| 라쏘 (L1) | Laplace$(0, b)$ | 뾰족하고 유한 |
| 최량 부분집합 | 스파이크–앤–슬랩 | 0에서 점질량 |

!!! warning "척도모수를 $1/\lambda$로 쓰지 말 것"
    간이 서술에서 흔히 $\beta_j \sim \text{Laplace}(0, 1/\lambda)$라고 쓰지만 이는 정확하지 않다. [베이즈 해석](bayesian.md)에서 유도했듯 올바른 관계는

    $$
    \lambda = \frac{\sigma^2}{nb} \quad\Longleftrightarrow\quad b = \frac{\sigma^2}{n\lambda}
    $$

    이다. 잡음분산 $\sigma^2$과 표본크기 $n$이 모두 들어간다. $\sigma^2 = n = 1$인 특수한 규격화에서만 $b = 1/\lambda$가 된다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
직교 설계에서 라쏘와 능형의 축소 규칙이 각각 "평행이동"과 "비례"임을 수치로 확인하고, 작은 계수에 미치는 영향이 왜 크게 다른지 설명하라.

</div>

??? success "풀이"
    자세한 검증은 [라쏘의 정식화와 희소성](formulation.md) 연습문제 1에 있다. 요지만 옮기면, $\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$인 자료에서

    | | $\hat\beta = 2.889$ | $\hat\beta = 0.086$ |
    |:---|---:|---:|
    | 능형 ($\times 0.980$) | 2.833 ($-2$%) | 0.085 ($-2$%) |
    | 라쏘 ($-0.05$) | 2.839 ($-2$%) | 0.036 ($-58$%) |

    **큰 계수에는 두 방법의 효과가 같지만 작은 계수에는 전혀 다르다.**

    능형은 모든 계수를 같은 **비율**로 깎으므로 작은 계수의 상대적 크기가 보존된다. 라쏘는 모든 계수에서 같은 **양**을 빼므로, 그 양보다 작은 계수는 0이 되어 사라진다.

    이것이 "능형은 축소만 하고 라쏘는 선택도 한다"는 말의 산술적 내용이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
라쏘의 세 가지 한계(집단 선택, $p > n$, 편향) 중 엘라스틱넷이 해결하는 것과 해결하지 못하는 것을 구분하라.

</div>

??? success "풀이"
    | 한계 | 엘라스틱넷이 해결하는가 |
    |:---|:---|
    | 집단 선택 (상관된 변수에서 임의로 하나 선택) | **해결한다.** L2 항이 그룹 효과를 만들어 상관된 변수를 함께 유지한다. |
    | $p > n$일 때 최대 $n$개 제한 | **해결한다.** L2 항이 목적함수를 강볼록으로 만들어 해가 유일해지고, $n$개보다 많은 변수를 선택할 수 있다. |
    | 선택된 계수의 편향 | **해결하지 못한다.** L1 항이 여전히 연성 문턱을 적용하므로 편향이 남는다. 오히려 L2 항이 축소를 더한다. |

    셋째 항목이 중요하다. 엘라스틱넷의 갱신식

    $$
    \hat{\beta}_j \leftarrow \frac{S_{\alpha\lambda}(z_j)}{1 + (1-\alpha)\lambda}
    $$

    을 보면 분자에서 $\alpha\lambda$만큼 깎고 분모에서 또 나눈다. **두 번 축소된다.**

    Zou와 Hastie(2005)는 이 "이중 축소"가 예측을 해칠 수 있다고 보고, 계수에 $(1 + \lambda_2)$를 곱해 되돌리는 **보정 엘라스틱넷**(corrected elastic net)을 제안했다. sklearn의 `ElasticNet`은 이 보정을 하지 **않는다**.

    편향이 문제라면 엘라스틱넷으로 변수를 고른 뒤 그 변수들로 OLS를 다시 적합하는 것이 여전히 최선이다([변수선택 도구로서의 라쏘](feature_selection.md) 연습문제 1 참조).

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
직교 설계 $\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$ 에서 OLS 추정값이 $\hat{\boldsymbol{\beta}}^{\text{OLS}} = (0.9,\ 0.3)$ 이고 $\lambda = 0.5$ 다.
(가) 4 절의 연성 문턱 공식으로 라쏘 해를 구하라.
(나) 이제 둘째 설명변수의 **단위만** 바꾸어 $\tilde{\mathbf{x}}_2 = c\,\mathbf{x}_2$ 로 기록한다(센티미터를 밀리미터로 적으면 $c = 10$). 벌점은 그대로 $\lambda = 0.5$ 를 쓴다. 새 해를 원래 단위로 환산한 값이 $S_{\lambda/c}(\hat\beta_2^{\text{OLS}})$ 임을 보이라.
(다) $c = 10$ 과 $c = 0.1$ 에서 그 값을 구하고, 어느 쪽에서 둘째 변수가 모형에 들어오는지 말하라.
(라) 표준화가 이 문제를 없애는 까닭을 한 줄로 적고, sklearn 의 `Lasso` 를 쓸 때 반드시 해야 할 일을 말하라.

</div>

??? success "풀이"
    **(가)** $S_{0.5}(0.9) = 0.9 - 0.5 = 0.4$ 이고 $\lvert 0.3\rvert < 0.5$ 이므로 둘째는 0 이다.

    $$
    \hat{\boldsymbol{\beta}}^{\text{lasso}} = (0.4,\ 0)
    $$

    **(나)** 열을 $c$ 배 해도 직교성은 깨지지 않으므로 목적함수는 여전히 좌표별로 쪼개진다. $z_2 = \mathbf{x}_2^\top\mathbf{y}/n = \hat\beta_2^{\text{OLS}}$ 라 두고 $\mathbf{x}_2^\top\mathbf{x}_2 = n$ 을 쓰면, 새 계수 $\gamma_2$ 에 걸리는 항은

    $$
    \frac{1}{2n}\left(-2c\gamma_2\,\mathbf{x}_2^\top\mathbf{y} + c^2\gamma_2^2\,\mathbf{x}_2^\top\mathbf{x}_2\right) + \lambda\lvert\gamma_2\rvert = -c z_2\gamma_2 + \frac{c^2}{2}\gamma_2^2 + \lambda\lvert\gamma_2\rvert
    $$

    이다. 양변을 $c^2 > 0$ 으로 나누어도 최소점은 같으므로

    $$
    \frac{1}{2}\gamma_2^2 - \frac{z_2}{c}\gamma_2 + \frac{\lambda}{c^2}\lvert\gamma_2\rvert \quad\Longrightarrow\quad \hat\gamma_2 = S_{\lambda/c^2}\!\left(\frac{z_2}{c}\right)
    $$

    이다. 원래 단위로 환산한 계수는 적합값 $\gamma_2\tilde{\mathbf{x}}_2 = (c\gamma_2)\mathbf{x}_2$ 에서 읽듯 $c\hat\gamma_2$ 이고, 연성 문턱의 양차수 성질 $c\,S_t(u) = S_{ct}(cu)$ ($c>0$) 에서

    $$
    c\hat\gamma_2 = c\,S_{\lambda/c^2}(z_2/c) = S_{\lambda/c}(z_2) \qquad \square
    $$

    **단위를 $c$ 배 크게 적으면 그 변수의 유효 문턱이 $\lambda/c$ 로 줄어든다.**

    **(다)**

    | $c$ | 새 단위의 OLS | 유효 문턱 $\lambda/c$ | 원래 단위로 환산한 라쏘 |
    |:---|---:|---:|---:|
    | $1$ | $0.3$ | $0.5$ | $0$ |
    | $10$ | $0.03$ | $0.05$ | $0.25$ |
    | $0.1$ | $3.0$ | $5.0$ | $0$ |

    $c = 10$ 에서 둘째 변수가 **모형에 들어온다.** 자료도 적합도도 바뀐 것이 없고 그 변수를 밀리미터로 적었을 뿐인데 변수선택의 결론이 뒤집힌다. $c = 0.1$ 에서는 여전히 0 이다(문턱이 $5$ 로 커졌다). 첫째 변수는 단위를 건드리지 않았으므로 두 경우 모두 $0.4$ 다.

    **(라)** 표준화는 모든 열의 $\mathbf{x}_j^\top\mathbf{x}_j$ 를 같게 만들어 **모든 변수가 같은 유효 문턱 $\lambda$ 를 받게** 한다. 그래야 $\lambda$ 가 "계수의 크기" 라는 하나의 자로 읽힌다. sklearn 의 `Ridge`·`Lasso` 는 표준화를 해 주지 않으므로 `StandardScaler` 를 **파이프라인에 묶어** 직접 넣어야 한다(그리고 그 표준화는 교차검증 폴드 안에서 일어나야 한다 — 연습문제 9).

    큰 수로 적힌 변수가 벌점을 덜 받는다는 것이 요점이다. 단위가 뒤섞인 자료에 표준화 없이 라쏘를 돌리면 "소득(원)" 이 "나이(년)" 를 밀어내는데, 그것은 통계가 아니라 자가 한 일이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
4 절 그림의 계수 $\hat\beta^{\text{OLS}} = 0.25$ 와 $2.90$ 을 쓴다(직교 설계).
(가) 능형으로 $0.25$ 를 $0.001$ 이하로 줄이려면 $\lambda$ 가 얼마 이상이어야 하는가. 그 $\lambda$ 에서 $2.90$ 은 어떤 값이 되는가.
(나) 라쏘로 $0.25$ 를 정확히 0 으로 만드는 가장 작은 $\lambda$ 를 구하고, 그때 $2.90$ 이 되는 값을 구하라.
(다) 능형이 두 계수의 비를 $\lambda$ 와 무관하게 보존함을 보이고, 이것이 능형으로 변수선택을 할 수 없는 까닭임을 설명하라.

</div>

??? success "풀이"
    **(가)** $0.25/(1+\lambda) \le 0.001 \iff 1+\lambda \ge 250 \iff \lambda \ge 249$. 그 $\lambda = 249$ 에서

    $$
    \frac{2.90}{250} = 0.0116
    $$

    이다. 원래 값의 $0.4\%$ 만 남는다. 작은 계수 하나를 사실상 지우는 값을 치르고 **큰 계수의 $99.6\%$ 를 버렸다.**

    **(나)** 연성 문턱은 $\lvert\hat\beta\rvert \le \lambda$ 에서 0 이므로 가장 작은 값은 $\lambda = 0.25$ 다. 그때

    $$
    S_{0.25}(2.90) = 2.90 - 0.25 = 2.65
    $$

    로 큰 계수의 $91.4\%$ 가 남는다.

    | | $0.25$ 를 지우는 데 든 $\lambda$ | 그때 남은 $2.90$ | 남은 비율 |
    |:---|---:|---:|---:|
    | 라쏘 | $0.25$ | $2.65$ | $91.4\%$ |
    | 능형 | $249$ (근사적으로만) | $0.0116$ | $0.4\%$ |

    **(다)** 능형은 모든 좌표에 같은 인자 $1/(1+\lambda)$ 를 곱하므로

    $$
    \frac{\hat\beta_1^{\text{ridge}}}{\hat\beta_2^{\text{ridge}}} = \frac{\hat\beta_1^{\text{OLS}}/(1+\lambda)}{\hat\beta_2^{\text{OLS}}/(1+\lambda)} = \frac{\hat\beta_1^{\text{OLS}}}{\hat\beta_2^{\text{OLS}}} = \frac{2.90}{0.25} = 11.6
    $$

    로 $\lambda$ 가 사라진다. $\lambda \to \infty$ 에서도 비는 $11.6$ 이고, 두 계수는 **함께** 0 으로 간다. 하나만 0 이 되는 $\lambda$ 는 존재하지 않는다.

    라쏘는 같은 양을 빼므로 비가 $\lambda$ 에 따라 바뀐다. $\lambda = 0.25$ 에서 비는 $2.65/0 = \infty$ 로 **발산하고**, 바로 이 발산이 "변수 하나가 모형에서 빠졌다" 는 말의 산술적 내용이다.

    **능형은 선별적으로 지울 수 없다 — 하나를 지우려면 전부를 지워야 한다.** 비례 축소와 평행이동 축소의 차이가 변수선택 능력의 유무로 곧바로 번역되는 자리다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
직교 설계 $\mathbf{X}^\top\mathbf{X} = n\mathbf{I}$ 와 $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon} \sim N(\mathbf{0}, \sigma^2\mathbf{I})$ 를 둔다. 능형 추정량 $\hat{\boldsymbol{\beta}}^{\text{ridge}} = \hat{\boldsymbol{\beta}}^{\text{OLS}}/(1+\lambda)$ 의 총 평균제곱오차를 $\text{MSE}(\lambda) = \sum_{j=1}^p E(\hat\beta_j^{\text{ridge}} - \beta_j)^2$ 로 쓴다.
(가) $v = \sigma^2/n$, $S = \lVert\boldsymbol{\beta}\rVert^2$ 라 두고 $\text{MSE}(\lambda)$ 를 닫힌 꼴로 구하라.
(나) $\text{MSE}'(0) < 0$ 임을 보이고, 최적 $\lambda^*$ 와 그때의 MSE 를 구하라.
(다) $p = 10$, $n = 100$, $\sigma^2 = 1$ 에서 $S = 0.5$ 와 $S = 10$ 의 두 경우에 $\lambda^*$ 와 MSE 감소율을 수로 구하고, 능형이 언제 크게 이기는지 말하라.

</div>

??? success "풀이"
    **(가)** 직교 설계에서 $\hat{\boldsymbol{\beta}}^{\text{OLS}} = \mathbf{X}^\top\mathbf{y}/n$ 이고

    $$
    \text{Var}(\hat{\boldsymbol{\beta}}^{\text{OLS}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1} = \frac{\sigma^2}{n}\mathbf{I} = v\mathbf{I}
    $$

    이다. 5 장 정리 1 의 편향–분산 분해를 좌표마다 적용하면 편향이 $\beta_j/(1+\lambda) - \beta_j = -\lambda\beta_j/(1+\lambda)$, 분산이 $v/(1+\lambda)^2$ 이므로

    $$
    \text{MSE}(\lambda) = \sum_{j=1}^p\left[\frac{\lambda^2\beta_j^2}{(1+\lambda)^2} + \frac{v}{(1+\lambda)^2}\right] = \frac{\lambda^2 S + pv}{(1+\lambda)^2}
    $$

    이다. $\text{MSE}(0) = pv$ 가 OLS 의 값이고, $\lambda\to\infty$ 에서 $S$ (모두 0 으로 보낸 추정량의 오차) 로 간다.

    **(나)** 미분하면 분자가 $2\lambda S(1+\lambda)^2 - (\lambda^2 S + pv)\cdot 2(1+\lambda)$ 이므로 $(1+\lambda)$ 를 약분해

    $$
    \text{MSE}'(\lambda) = \frac{2\left[\lambda S(1+\lambda) - \lambda^2 S - pv\right]}{(1+\lambda)^3} = \frac{2(\lambda S - pv)}{(1+\lambda)^3}
    $$

    를 얻는다. 따라서 $\text{MSE}'(0) = -2pv < 0$ 이다. **$\lambda = 0$ 에서 기울기가 음수이므로 OLS 를 이기는 $\lambda > 0$ 이 반드시 존재한다.** 또 도함수의 부호가 $\lambda S - pv$ 의 부호와 같으므로 유일한 최소점은

    $$
    \lambda^* = \frac{pv}{S} = \frac{p\sigma^2}{n\lVert\boldsymbol{\beta}\rVert^2}
    $$

    이고, 대입하면

    $$
    \text{MSE}(\lambda^*) = \frac{(pv)^2/S + pv}{\left((S+pv)/S\right)^2} = \frac{pv\,S}{S + pv} = pv\cdot\frac{S}{S+pv} < pv
    $$

    이다. $\square$

    **(다)** $v = 1/100 = 0.01$, $pv = 0.1$ 이 OLS 의 MSE 다.

    | $S = \lVert\boldsymbol{\beta}\rVert^2$ | $\lambda^* = pv/S$ | $\text{MSE}(\lambda^*)$ | OLS 대비 |
    |---:|---:|---:|---:|
    | $0.5$ | $0.2$ | $0.1 \times \frac{0.5}{0.6} = 0.08333$ | $-16.7\%$ |
    | $10$ | $0.01$ | $0.1 \times \frac{10}{10.1} = 0.09901$ | $-1.0\%$ |

    **참 계수가 작을 때 이득이 크다.** 이득의 크기가 $S/(S+pv)$ 라는 한 덩어리에 들어 있고, 이것은 "신호 $S$ 대 잡음 $pv$" 의 비로 읽힌다. 신호가 잡음에 비해 크면 축소할 것이 없고($\lambda^* \to 0$), 작으면 0 쪽으로 끌어당기는 것이 압도적으로 유리하다.

    이기는 $\lambda$ 가 **반드시 있다**는 것과 그 $\lambda$ 를 **알 수 있다**는 것은 다른 이야기임도 눈여겨볼 만하다. $\lambda^*$ 가 미지의 $S$ 와 $\sigma^2$ 에 의존하므로 실제로는 교차검증으로 더듬어 찾아야 한다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
직교 설계, $\sigma = 1$, $n = 100$ 이면 $\hat\beta_j^{\text{OLS}} \sim N(\beta_j,\ 0.01)$ 로 표준편차가 $0.1$ 이다. 연성 문턱 $S_\lambda$ 를 적용한다.
(가) $\lambda = 0.2$ 에서 참값이 $\beta_j = 0$ 인 계수가 정확히 0 으로 추정될 확률을 구하라.
(나) 같은 $\lambda$ 에서 참값이 $\beta_j = 0.5$ 인 계수가 0 으로 추정될 확률을 구하라.
(다) 참값이 0 인 계수가 $100$ 개일 때 그중 **적어도 하나**가 살아남을 확률을 $\lambda = 0.2$ 와 $\lambda = 0.3$ 에서 구하라.
(라) (다)의 확률을 $0.05$ 이하로 만드는 가장 작은 $\lambda$ 를 구하고, 그 $\lambda$ 에서 $\beta_j = 0.5$ 를 잃을 확률을 구하라. 결과를 $\sigma\sqrt{2\log p\,/\,n}$ 과 비교하라.

</div>

??? success "풀이"
    $\hat\beta_j$ 의 표준편차를 $s = 0.1$ 로 쓴다. 연성 문턱이 0 을 주는 사건은 $\lvert\hat\beta_j\rvert \le \lambda$ 다.

    **(가)** $\beta_j = 0$ 이면

    $$
    P(\lvert\hat\beta_j\rvert \le 0.2) = 2\Phi\!\left(\frac{0.2}{0.1}\right) - 1 = 2\Phi(2) - 1 = 2(0.97725) - 1 = 0.9545
    $$

    이다.

    **(나)** $\beta_j = 0.5$ 이면

    $$
    P(-0.2 \le \hat\beta_j \le 0.2) = \Phi\!\left(\frac{0.2-0.5}{0.1}\right) - \Phi\!\left(\frac{-0.2-0.5}{0.1}\right) = \Phi(-3) - \Phi(-7) = 0.001350
    $$

    이다($\Phi(-7) \approx 1.3\times 10^{-12}$ 로 무시할 수 있다). **영계수의 $95.4\%$ 를 지우면서 진짜 효과는 $0.14\%$ 만 잃는다** — 이것이 라쏘가 쓸모 있는 이유의 전부다.

    **(다)** 독립이므로 전부 0 이 될 확률은 $(0.9545)^{100}$ 이고

    $$
    P(\text{적어도 하나 살아남음}) = 1 - (0.9545)^{100} = 1 - 0.00951 = 0.9905
    $$

    이다. $\lambda = 0.3$ 이면 한 개가 0 이 될 확률이 $2\Phi(3)-1 = 0.99730$ 이므로

    $$
    1 - (0.99730)^{100} = 1 - 0.7631 = 0.2369
    $$

    다. 변수를 $100$ 개 보면 $3$ 표준편차 문턱으로도 거짓 양성이 네 번에 한 번은 나온다.

    **(라)** $\left(2\Phi(\lambda/s) - 1\right)^{100} \ge 0.95$ 를 풀면

    $$
    2\Phi(\lambda/s) - 1 \ge 0.95^{1/100} = 0.9994872 \quad\Longrightarrow\quad \Phi(\lambda/s) \ge 0.9997436
    $$

    이고 $\lambda/s \ge 3.4740$ 이므로 $\lambda \ge 0.3474$ 다. 그 $\lambda$ 에서

    $$
    P(\beta_j = 0.5 \text{ 를 잃음}) = \Phi\!\left(\frac{0.347-0.5}{0.1}\right) - \Phi\!\left(\frac{-0.347-0.5}{0.1}\right) = \Phi(-1.526) = 0.0635
    $$

    이다. 거짓 양성을 $5\%$ 로 묶는 대가로 **진짜 효과를 놓칠 확률이 $0.14\%$ 에서 $6.4\%$ 로 $47$ 배 커졌다.**

    | $\lambda$ | 영계수 하나를 지울 확률 | $100$ 개 중 거짓 양성 | $\beta = 0.5$ 를 잃을 확률 |
    |---:|---:|---:|---:|
    | $0.2$ | $0.9545$ | $0.9905$ | $0.00135$ |
    | $0.3$ | $0.99730$ | $0.2369$ | $0.02275$ |
    | $0.3474$ | $0.9994872$ | $0.0500$ | $0.0635$ |

    **비교.** $\sigma\sqrt{2\log p / n} = \sqrt{2\log 100}/10 = 3.0349/10 = 0.3035$ 로 우리가 구한 $0.3474$ 와 같은 크기다. 이론이 권하는 $\lambda \asymp \sigma\sqrt{2\log p/n}$ 가 바로 이 다중성 계산의 결과이고, $\log p$ 가 들어가는 까닭도 여기 있다 — 변수 수가 늘면 문턱을 $\sqrt{\log p}$ 로 천천히 올려야 거짓 양성이 묶인다.

    8 절의 "편향" 과 "일치성" 이 한 화면에 같이 있는 셈이다. 희소성은 공짜가 아니라 **중간 크기의 진짜 효과를 버리는 값**으로 산다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
능형의 적합값은 $\hat{\mathbf{y}} = \mathbf{H}_\lambda\mathbf{y}$ 로 선형이고, 유효 자유도를 $\operatorname{tr}(\mathbf{H}_\lambda)$ 로 정의한다. 라쏘의 유효 자유도는 0 이 아닌 계수의 개수가 불편추정이 된다(Zou·Hastie·Tibshirani, 2007).
(가) 직교 설계에서 $\operatorname{tr}(\mathbf{H}_\lambda) = p/(1+\lambda)$ 임을 보이고 $\lambda \to 0$, $\lambda \to \infty$ 의 극한을 확인하라.
(나) 4 절 그림의 $p = 12$, $\lambda = 1$ 에서 능형과 라쏘의 유효 자유도를 각각 구하라.
(다) 라쏘는 계수 다섯 개를 0 으로 만들었는데 자유도가 능형보다 **크다.** 이 역설을 풀이하라.
(라) 능형의 자유도를 라쏘와 같게 맞추는 $\lambda$ 를 구하고, 그 $\lambda$ 에서 계수 $0.25$ 와 $2.90$ 이 어떻게 되는지 적으라.

</div>

??? success "풀이"
    **(가)** 직교 설계에서 능형 계수가 $\hat{\boldsymbol{\beta}}^{\text{ridge}} = \hat{\boldsymbol{\beta}}^{\text{OLS}}/(1+\lambda)$ 이므로

    $$
    \hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}}^{\text{ridge}} = \frac{1}{1+\lambda}\mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y} = \frac{1}{1+\lambda}\mathbf{P}\mathbf{y}
    $$

    이고 $\mathbf{P}$ 는 $\mathbf{X}$ 의 열공간으로의 사영행렬이다(부록 A). 사영행렬의 대각합은 계수(rank)와 같아 $\operatorname{tr}(\mathbf{P}) = p$ 이므로

    $$
    \operatorname{tr}(\mathbf{H}_\lambda) = \frac{p}{1+\lambda} \qquad \square
    $$

    $\lambda \to 0$ 에서 $p$ (OLS 가 모수 $p$ 개를 다 쓴다), $\lambda\to\infty$ 에서 $0$ (상수모형) 이다. 일반 설계의 $\sum_j d_j^2/(d_j^2 + n\lambda)$ 에 $d_j^2 = n$ 을 넣은 것과 같다.

    **(나)** $\lambda = 1$ 에서

    | | 유효 자유도 |
    |:---|---:|
    | 능형 | $12/(1+1) = 6$ |
    | 라쏘 | 0 이 아닌 계수의 개수 $= 7$ |

    그림에서 띠 안의 다섯 개가 0 이 되었으므로 라쏘는 $12 - 5 = 7$ 이다.

    **(다)** 자유도는 "쓰는 변수의 개수" 가 아니라 **적합값이 자료에 반응하는 총량** $\sum_i \partial\hat y_i/\partial y_i$ 다. 두 방법이 그 총량을 전혀 다르게 배분한다.

    - 라쏘는 문턱을 넘은 7 개에 대해서는 $\hat\beta_j = \hat\beta_j^{\text{OLS}} \mp \lambda$ 이므로 $\partial\hat\beta_j/\partial\hat\beta_j^{\text{OLS}} = 1$ 이다. **살아남은 변수는 OLS 와 똑같은 민감도로 자료를 따라간다.** 7 개가 온전히 1 씩 기여하고 나머지 5 개가 0 을 기여해 합이 7 이다.
    - 능형은 12 개 모두를 쓰지만 각각의 민감도가 $1/(1+\lambda) = 0.5$ 로 **반씩만** 자료를 따라간다. $12 \times 0.5 = 6$ 이다.

    변수 다섯 개를 완전히 버린 쪽이 더 복잡한 것은, 남긴 변수를 **아무 제약 없이** 쓰기 때문이다. 자유도를 쓸 때 "변수 개수" 로 바꿔 읽으면 안 되는 이유가 이 한 줄에 있다. 라쏘에서는 그 둘이 우연히 일치하는데, 그것도 연성 문턱의 기울기가 정확히 1 이라는 특수한 사정 때문이다(그림 왼쪽에서 라쏘 곡선이 띠 밖에서 점선과 나란한 것이 바로 그 기울기 1 이다).

    **(라)** $12/(1+\lambda) = 7$ 에서

    $$
    \lambda = \frac{12}{7} - 1 = \frac{5}{7} = 0.7143
    $$

    이고 축소인자는 $1/(1+5/7) = 7/12 = 0.5833$ 이다.

    | 계수 | 라쏘 ($\lambda = 1$) | 능형 ($\lambda = 5/7$) |
    |---:|---:|---:|
    | $2.90$ | $1.90$ | $1.6917$ |
    | $0.25$ | $0$ | $0.1458$ |

    **자유도가 똑같이 7 인데 모형에 든 변수의 개수는 7 대 12 다.** 같은 복잡도 예산을 라쏘는 몇 개에 몰아주고 능형은 전부에 얇게 펴 바른다. 정보기준($\text{AIC} = n\log\hat\sigma^2 + 2\,\mathrm{df}$ 꼴)으로 두 방법을 비교할 때 이 값을 넣어야 하며, 라쏘 쪽에 0 이 아닌 개수를 그대로 넣어도 되는 것은 (다)에서 본 기울기 1 덕분이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
9 절의 "집단 선택" 한계를 $p = 2$ 에서 수로 본다. $\frac{1}{n}\mathbf{X}^\top\mathbf{X} = \begin{pmatrix}1 & \rho\\ \rho & 1\end{pmatrix}$, $\mathbf{z} = \frac{1}{n}\mathbf{X}^\top\mathbf{y} = (z+\delta,\ z-\delta)$ 로 두 설명변수가 거의 대등하다.
(가) 두 계수가 모두 양수인 라쏘 해를 닫힌 꼴로 구하고, 두 계수의 차가 $2\delta/(1-\rho)$ 임을 보이라.
(나) $\rho = 0.99$, $z = 0.5$, $\lambda = 0.1$ 에서 $\delta = 0$ 과 $\delta = 0.003$ 의 해를 각각 구하라($\delta = 0.003$ 에서는 KKT 조건으로 활성집합을 확인하라). 한쪽이 0 이 되기 시작하는 $\delta^*$ 도 구하라.
(다) 엘라스틱넷(L2 벌점 $\frac{\lambda_2}{2}\lVert\boldsymbol{\beta}\rVert^2$ 추가, $\lambda_2 = 0.1$)에서 같은 $\delta = 0.003$ 의 해와 $\delta^*$ 를 구하고, 증폭계수가 어떻게 바뀌는지 적으라.

</div>

??? success "풀이"
    $\mathbf{G} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ 라 두면 목적함수가

    $$
    \frac{1}{2n}\lVert\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\rVert^2 + \lambda\lVert\boldsymbol{\beta}\rVert_1 = \text{상수} - \boldsymbol{\beta}^\top\mathbf{z} + \frac{1}{2}\boldsymbol{\beta}^\top\mathbf{G}\boldsymbol{\beta} + \lambda\lVert\boldsymbol{\beta}\rVert_1
    $$

    이다.

    **(가)** 두 계수가 모두 양수이면 부분미분이 $+1$ 로 확정되므로 정상조건은 $\mathbf{G}\boldsymbol{\beta} = \mathbf{z} - \lambda\mathbf{1}$ 이다. $\mathbf{G}^{-1} = \frac{1}{1-\rho^2}\begin{pmatrix}1 & -\rho\\ -\rho & 1\end{pmatrix}$ 을 곱하면

    $$
    \hat\beta_1 = \frac{(z+\delta-\lambda) - \rho(z-\delta-\lambda)}{1-\rho^2} = \frac{(1-\rho)(z-\lambda) + (1+\rho)\delta}{(1-\rho)(1+\rho)} = \frac{z-\lambda}{1+\rho} + \frac{\delta}{1-\rho}
    $$

    이고 같은 계산으로 $\hat\beta_2 = \frac{z-\lambda}{1+\rho} - \frac{\delta}{1-\rho}$ 다. 따라서

    $$
    \hat\beta_1 - \hat\beta_2 = \frac{2\delta}{1-\rho} \qquad \square
    $$

    **자료의 비대칭 $\delta$ 가 $1/(1-\rho)$ 배로 증폭된다.** $\rho \to 1$ 에서 증폭계수가 발산한다.

    **(나)** $\rho = 0.99$ 이면 $\frac{z-\lambda}{1+\rho} = \frac{0.4}{1.99} = 0.20101$ 이고 증폭계수는 $1/(1-0.99) = 100$ 이다.

    - $\delta = 0$: $\hat{\boldsymbol{\beta}} = (0.20101,\ 0.20101)$. 완전히 대등하게 나눈다.
    - $\delta = 0.003$: 공식은 $0.20101 \pm 100(0.003) = 0.20101 \pm 0.3$ 이므로 $(0.50101,\ -0.09899)$ 인데 **둘째가 음수**라 "둘 다 양수" 라는 가정이 깨졌다. 그러므로 $\hat\beta_2 = 0$ 인 해를 시험한다. 그때 첫 좌표는 $\hat\beta_1 = z_1 - \lambda = 0.503 - 0.1 = 0.403$ 이고, 비활성 좌표의 KKT 조건은

    $$
    \left\lvert z_2 - (\mathbf{G}\hat{\boldsymbol{\beta}})_2\right\rvert = \lvert 0.497 - 0.99(0.403)\rvert = \lvert -0.09803\rvert = 0.09803 \le 0.1 = \lambda
    $$

    로 만족된다(활성 좌표도 $z_1 - (\mathbf{G}\hat{\boldsymbol{\beta}})_1 = 0.503 - 0.403 = 0.1 = \lambda$ 로 정확히 맞는다). 목적함수가 볼록이고 $\rho<1$ 이라 강볼록이므로 이 해가 유일하다.

    $$
    \hat{\boldsymbol{\beta}} = (0.403,\ 0)
    $$

    - 한쪽이 0 이 되는 경계는 $\frac{z-\lambda}{1+\rho} = \frac{\delta}{1-\rho}$ 에서

    $$
    \delta^* = \frac{(1-\rho)(z-\lambda)}{1+\rho} = \frac{0.01 \times 0.4}{1.99} = 0.002010
    $$

    이다.

    | $\delta$ | $\hat\beta_1$ | $\hat\beta_2$ |
    |---:|---:|---:|
    | $0$ | $0.20101$ | $0.20101$ |
    | $0.002$ | $0.40101$ | $0.00101$ |
    | $0.003$ | $0.403$ | $0$ |

    **자료가 $0.6\%$ 만큼 기울었는데 결론이 "둘 다 $0.20$" 에서 "하나만 $0.40$" 으로 뒤집힌다.** 표본을 다시 뽑으면 $\delta$ 의 부호가 바뀔 수 있고, 그러면 선택되는 변수도 바뀐다. 9 절이 말한 "그 선택이 임의적이다" 가 이 $\delta^* = 0.002$ 라는 작은 수다.

    **(다)** L2 항을 더하면 정상조건의 행렬이 $\mathbf{G} + \lambda_2\mathbf{I}$ 로 바뀌므로 (가)의 계산에서 대각원소 $1$ 이 $1+\lambda_2 = 1.1$ 로 바뀐다.

    $$
    \hat\beta_{1,2} = \frac{z-\lambda}{1+\lambda_2+\rho} \pm \frac{\delta}{1+\lambda_2-\rho} = \frac{0.4}{2.09} \pm \frac{0.003}{0.11} = 0.19139 \pm 0.02727
    $$

    이므로 $\hat{\boldsymbol{\beta}} = (0.21866,\ 0.16411)$ 로 **둘 다 살아남고 거의 같다.** 경계도

    $$
    \delta^* = \frac{(1+\lambda_2-\rho)(z-\lambda)}{1+\lambda_2+\rho} = \frac{0.11\times 0.4}{2.09} = 0.02105
    $$

    로 $10$ 배 커졌다.

    | | 증폭계수 | $\delta^*$ | $\delta = 0.003$ 의 해 |
    |:---|---:|---:|:---|
    | 라쏘 | $1/(1-\rho) = 100$ | $0.00201$ | $(0.403,\ 0)$ |
    | 엘라스틱넷 | $1/(1+\lambda_2-\rho) = 9.09$ | $0.02105$ | $(0.21866,\ 0.16411)$ |

    엘라스틱넷의 묶음 효과가 하는 일이 정확히 **$1-\rho$ 를 $1+\lambda_2-\rho$ 로 바꾸는 것**이다. 상관이 1 에 다가가도 $\lambda_2$ 가 분모의 바닥을 받쳐 주므로 증폭이 발산하지 않는다. 11 장의 다중공선성에서 $\mathbf{X}^\top\mathbf{X}$ 의 역행렬이 폭발하던 것을 능형이 $\lambda_2\mathbf{I}$ 로 막던 것과 같은 산술이며, 여기서는 그 폭발이 **추정값의 분산**이 아니라 **선택의 불안정**으로 나타났을 뿐이다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
7 절처럼 교차검증으로 $\lambda$ 를 고르는데, 편의상 격자의 끝 $\lambda \ge \lambda_{\max}$ 를 본다. 거기서 모든 계수가 0 이므로 예측값은 절편뿐이다(절편은 벌점에서 뺀다). 표본 $\mathbf{y}$ 에 하나 빼기 교차검증(LOOCV)을 적용한다.
(가) 정직한 LOOCV 에서 $i$ 번째 잔차가 $\frac{n}{n-1}(y_i - \bar y)$ 임을 보이라.
(나) 흔한 실수대로 **전체 자료의 평균으로 먼저 중심화**한 뒤 LOOCV 를 하면 각 폴드의 예측값이 0 이 된다. 이때의 CV 오차와 (가)의 정직한 CV 오차의 비를 $n$ 의 식으로 구하라.
(다) $n = 5$, $\mathbf{y} = (1, 2, 3, 4, 10)$ 에서 두 값을 모두 구하고 과소평가율을 백분율로 적으라. $n = 10$ 이면 어떻게 되는가.
(라) 이 결과가 "표준화와 변수선택을 폴드 안에서 하라" 는 규칙에 대해 말해 주는 것을 적으라.

</div>

??? success "풀이"
    **(가)** $\bar y_{(-i)} = \frac{1}{n-1}\sum_{k\ne i} y_k = \frac{n\bar y - y_i}{n-1}$ 이므로

    $$
    y_i - \bar y_{(-i)} = y_i - \frac{n\bar y - y_i}{n-1} = \frac{(n-1)y_i - n\bar y + y_i}{n-1} = \frac{n(y_i - \bar y)}{n-1} \qquad \square
    $$

    빼낸 관측값이 평균을 자기 쪽으로 끌어당기지 못하므로 잔차가 $\frac{n}{n-1}$ 배 커진다.

    **(나)** 정직한 쪽은

    $$
    \text{CV}_{\text{정직}} = \frac{1}{n}\sum_{i=1}^n\left(\frac{n}{n-1}\right)^2 (y_i-\bar y)^2 = \left(\frac{n}{n-1}\right)^2 \cdot \frac{1}{n}\sum_i (y_i-\bar y)^2
    $$

    이고, 미리 중심화한 쪽은 $\tilde y_i = y_i - \bar y$ 를 0 으로 예측하므로

    $$
    \text{CV}_{\text{누수}} = \frac{1}{n}\sum_{i=1}^n (y_i - \bar y)^2
    $$

    이다. 따라서

    $$
    \frac{\text{CV}_{\text{누수}}}{\text{CV}_{\text{정직}}} = \left(\frac{n-1}{n}\right)^2
    $$

    로 **$n$ 에만 의존하고 자료에는 전혀 의존하지 않는다.**

    **(다)** $\bar y = 20/5 = 4$ 이고 $\sum(y_i-\bar y)^2 = 9+4+1+0+36 = 50$ 이므로 $\text{CV}_{\text{누수}} = 50/5 = 10$ 이다. 정직한 쪽은 잔차를 하나씩 적으면

    | $i$ | $y_i$ | $\bar y_{(-i)}$ | 잔차 | 제곱 |
    |---:|---:|---:|---:|---:|
    | 1 | $1$ | $4.75$ | $-3.75$ | $14.0625$ |
    | 2 | $2$ | $4.50$ | $-2.50$ | $6.2500$ |
    | 3 | $3$ | $4.25$ | $-1.25$ | $1.5625$ |
    | 4 | $4$ | $4.00$ | $0$ | $0$ |
    | 5 | $10$ | $2.50$ | $7.50$ | $56.2500$ |

    합이 $78.125$ 이므로 $\text{CV}_{\text{정직}} = 78.125/5 = 15.625$ 다. 비는 $10/15.625 = 0.64 = (4/5)^2$ 로 공식과 맞고, **오차를 $36\%$ 과소평가**했다. $n = 10$ 이면 $(9/10)^2 = 0.81$ 로 $19\%$ 다.

    **(라)** 여기서 새 나간 정보는 **평균 하나**뿐이고 적합한 모형은 계수가 전부 0 인 **가장 단순한 모형**이다. 그런데도 $n = 5$ 에서 오차가 $36\%$ 깎였다. 표준화(각 열의 평균과 표준편차 $2p$ 개), 변수선택(상관이 큰 변수 고르기), 결측값 대체를 전체 자료로 먼저 해 버리면 새 나가는 양이 이보다 훨씬 크다.

    실용적 결론은 둘이다.

    1. 중심화·표준화·선별은 **훈련 폴드만으로** 추정하고 검증 폴드에는 그 값을 **적용만** 한다. sklearn 이라면 `Pipeline(StandardScaler(), Lasso())` 을 `cross_val_score` 에 넣어야 하고, 스케일러를 먼저 `fit_transform` 해 두면 안 된다.
    2. 누수는 모든 $\lambda$ 에서 오차를 깎지만 그 정도가 같지 않으므로, 피해가 "CV 곡선이 통째로 내려간다" 로 끝나지 않고 **고르는 $\lambda$ 자체가 옮겨간다.** 보고하는 성능이 낙관적인 것보다 이쪽이 더 나쁘다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
9 절의 "$p>n$ 일 때 라쏘는 최대 $n$ 개의 변수만 고를 수 있다" 를 따져 본다.
(가) $n = 1$, $p = 2$, $\mathbf{X} = (1\ \ 1)$, $y = 2$, $\lambda = 1$ 인 라쏘의 **해집합 전체**를 구하라.
(나) 일반적으로 어떤 라쏘 해의 활성집합 $A$ 가 $\lvert A\rvert > n$ 이면 0 이 아닌 계수가 더 적은 해가 존재함을 보이라.
(다) 엘라스틱넷($\lambda_1 = \lambda_2 = 1$)에서 (가)의 해를 구하고, 왜 유일해지는지 설명하라.

</div>

??? success "풀이"
    **(가)** $n = 1$ 이므로 목적함수는

    $$
    f(\boldsymbol{\beta}) = \frac{1}{2}(2 - \beta_1 - \beta_2)^2 + \lambda\left(\lvert\beta_1\rvert + \lvert\beta_2\rvert\right)
    $$

    이다. 적합값이 $s = \beta_1 + \beta_2$ 에만 의존함을 눈여겨본다. $s$ 를 고정했을 때 $\lVert\boldsymbol{\beta}\rVert_1$ 의 최솟값은 삼각부등식 $\lvert\beta_1\rvert+\lvert\beta_2\rvert \ge \lvert s\rvert$ 에서 $\lvert s\rvert$ 이고, 두 좌표의 부호가 같고(또는 하나가 0 이고) 합이 $s$ 이면 등호가 성립한다. 그러므로

    $$
    \min_{\boldsymbol{\beta}} f = \min_s \left\{\frac{1}{2}(2-s)^2 + \lambda\lvert s\rvert\right\}
    $$

    이고 이 1 차원 문제의 해는 연성 문턱 $s = S_\lambda(2) = 2 - \lambda = 1$ 이다(값은 $\frac12 + 1 = 1.5$). 등호가 성립하는 $\boldsymbol{\beta}$ 를 모두 모으면

    $$
    \text{해집합} = \{(t,\ 1-t) : 0 \le t \le 1\}
    $$

    로 **선분 전체**다. 해가 유일하지 않고, 양 끝점 $(1, 0)$ 과 $(0, 1)$ 은 0 이 아닌 계수가 $1 = n$ 개뿐인 **희소해**다. 두 열이 완전히 같으니 라쏘는 어느 것을 고를 근거가 없고, 실제로 좌표하강은 순회 순서에 따라, LARS 는 수치오차에 따라 다른 끝점을 내놓는다. 연습문제 8 에서 증폭계수 $1/(1-\rho)$ 가 $\rho = 1$ 에서 발산한 것의 극한이 이 선분이다.

    **(나)** $A = \{j : \hat\beta_j \ne 0\}$ 이고 $\lvert A\rvert > n$ 이라 하자. $\mathbf{X}_A$ 는 $n \times \lvert A\rvert$ 행렬이고 열의 개수가 행의 개수보다 많으므로 열이 선형종속이다. 즉 $\mathbf{v} \ne \mathbf{0}$ 이 있어 $\mathbf{X}_A\mathbf{v} = \mathbf{0}$ 이다. $\mathbf{v}$ 를 $A$ 밖에서 0 으로 확장해 $\boldsymbol{\beta}(t) = \hat{\boldsymbol{\beta}} + t\mathbf{v}$ 를 본다.

    적합값은 $\mathbf{X}\boldsymbol{\beta}(t) = \mathbf{X}\hat{\boldsymbol{\beta}}$ 로 $t$ 와 무관하다. 한편 $\lvert t\rvert$ 가 작으면 $A$ 의 좌표들이 부호를 바꾸지 않으므로

    $$
    \lVert\boldsymbol{\beta}(t)\rVert_1 = \lVert\hat{\boldsymbol{\beta}}\rVert_1 + t\sum_{j\in A} v_j\operatorname{sign}(\hat\beta_j) = \lVert\hat{\boldsymbol{\beta}}\rVert_1 + t\,c, \qquad c := \mathbf{v}_A^\top\operatorname{sign}(\hat{\boldsymbol{\beta}}_A)
    $$

    로 $t$ 의 **선형함수**다. $\hat{\boldsymbol{\beta}}$ 가 최적이므로 $t>0$ 과 $t<0$ 양쪽에서 $f$ 가 줄지 않아야 하는데 $f(\boldsymbol{\beta}(t)) = \text{상수} + \lambda t c$ 이니 $c \ge 0$ 과 $-c \ge 0$ 이 동시에 필요하다. 즉 $c = 0$ 이고 **$f$ 는 이 직선 위에서 상수**다.

    그러므로 $t$ 를 (부호가 처음 바뀌는 곳까지) 어느 방향으로든 키워도 모두 최적해다. $t$ 를 $t_0 = \min\{\lvert\hat\beta_j/v_j\rvert : j \in A,\ v_j \ne 0\}$ 까지 (그 최소를 주는 좌표가 0 이 되는 부호로) 움직이면 적어도 한 좌표가 정확히 0 이 되고 다른 좌표는 부호를 유지하므로, 활성집합이 적어도 하나 줄어든 해를 얻는다. $\lvert A\rvert > n$ 인 동안 이 과정을 반복할 수 있으므로 **0 이 아닌 계수가 $n$ 개 이하인 해가 반드시 존재한다.** $\square$

    덧붙여, 위 논증은 $\lvert A\rvert > n$ 인 해가 있으면 해가 **유일하지 않다**는 것도 보여 준다. $p>n$ 에서 라쏘가 $n$ 개를 넘겨 고르는 일이 없는 것은 알고리즘의 사정이 아니라 목적함수의 성질이다.

    **(다)** 엘라스틱넷의 목적함수는

    $$
    g(\boldsymbol{\beta}) = \frac{1}{2}(2-\beta_1-\beta_2)^2 + \lambda_1\left(\lvert\beta_1\rvert+\lvert\beta_2\rvert\right) + \frac{\lambda_2}{2}\left(\beta_1^2+\beta_2^2\right)
    $$

    다. $\frac{\lambda_2}{2}\lVert\boldsymbol{\beta}\rVert^2$ 항 때문에 $g$ 가 **강볼록**이고(헤세행렬이 $\begin{pmatrix}1&1\\1&1\end{pmatrix} + \lambda_2\mathbf{I} \succeq \lambda_2\mathbf{I} \succ 0$) 따라서 최소점이 유일하다. $g$ 가 두 좌표의 교환에 대해 대칭이므로 그 유일한 최소점도 대칭이어야 한다. $\beta_1 = \beta_2 = t \ge 0$ 을 넣으면

    $$
    g = \frac{1}{2}(2-2t)^2 + 2\lambda_1 t + \lambda_2 t^2
    $$

    이고 미분하면 $-4(1-t) + 2\lambda_1 + 2\lambda_2 t = 0$ 에서

    $$
    t = \frac{2-\lambda_1}{2+\lambda_2} = \frac{1}{3}
    $$

    이다. 즉 $\hat{\boldsymbol{\beta}}^{\text{EN}} = (1/3,\ 1/3)$ 로 **유일하고 대칭**이다. 라쏘의 선분 $\{(t, 1-t)\}$ 가 한 점으로 모인 셈이고, 그 점이 선분의 중점 $(0.5, 0.5)$ 가 아니라 안쪽인 것은 연습문제 2 에서 본 **이중 축소** 때문이다($\lVert\hat{\boldsymbol{\beta}}^{\text{EN}}\rVert_1 = 2/3 < 1$).

    세 사실이 한 예에 다 들어 있다. 라쏘는 완전히 상관된 변수에서 **유일하지 않고**, 희소한 해를 **고를 수는 있지만 근거 없이** 고르며, 엘라스틱넷은 유일성과 묶음 효과를 사 오는 대가로 **축소를 더 한다.**

---

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff med" title="중간"></span> $S(\rho, \lambda)$가 $\lambda |\cdot|$의 근접 연산자임을 손으로 확인하라. 즉
$S(\rho, \lambda) = \arg\min_{z} \left\{ \frac{1}{2}(z - \rho)^2 + \lambda |z| \right\}$
임을 보여라.

</div>

??? success "풀이"

    $g(z) = \frac{1}{2}(z - \rho)^2 + \lambda |z|$라 하고 세 경우로 나눈다.

    **경우 1: $z > 0$.** $g(z) = \frac{1}{2}(z-\rho)^2 + \lambda z$이고
    $g'(z) = z - \rho + \lambda = 0$에서 $z^* = \rho - \lambda$를 얻는다. 이 값이 양수인 것은
    $\rho > \lambda$일 때뿐이다.

    **경우 2: $z < 0$.** $g(z) = \frac{1}{2}(z-\rho)^2 - \lambda z$이고
    $g'(z) = z - \rho - \lambda = 0$에서 $z^* = \rho + \lambda$를 얻는다. 이 값이 음수인 것은
    $\rho < -\lambda$일 때뿐이다.

    **경우 3: $z = 0$.** 부분미분 조건 $0 \in \{-\rho\} + \lambda[-1, 1]$은
    $|\rho| \le \lambda$를 요구한다.

    셋을 합치면 $z^* = \text{sign}(\rho)\max(|\rho| - \lambda, 0) = S(\rho, \lambda)$이다.
    $\square$

---

<div class="drillbox" markdown>

**연습문제 12.** <span class="diff med" title="중간"></span> 좌표하강 알고리즘에서 $\beta_j$를 갱신하기 전에 부분잔차
$r_j = y - X\beta + X_j \beta_j$를 계산해야 하는 이유를 설명하라. 대신 전체 잔차
$r = y - X\beta$를 쓰면 무엇이 잘못되는가?

</div>

??? success "풀이"

    $\beta_j$에 대한 좌표하강 갱신은 다른 계수를 모두 고정한 채 라쏘 목적함수를 $\beta_j$에
    대해 최소화하는 것이다. 부분잔차 $r_j$는 현재 적합에서 $j$번째 변수의 기여를 제거하므로,
    갱신은 $X_j$와 **$X_j$ 자신의 기여를 뺀** 잔차 사이의 상관에만 의존하게 된다.

    전체 잔차 $r = y - X\beta$를 쓰면 $X_j \beta_j$가 이미 빠져 있는 상태이므로 $j$번째 변수의
    기여를 이중으로 차감하는 셈이 되어 갱신이 편향된다. 구체적으로 $\rho_j$가
    $X_j^\top(y - X_{-j}\beta_{-j})/n$이 아니라 $X_j^\top(y - X\beta)/n$이 되고, 반복열은 라쏘
    해로 수렴하지 않는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 13.** <span class="diff med" title="중간"></span> `lasso_cd` 함수가 온기 시작(즉 항상 0에서 출발하는 대신 초기값 $\beta^{(0)}$을
받도록)을 지원하도록 수정하라. 온기 시작이 정칙화 경로 계산을 왜 빠르게 하는지 설명하라.

</div>

??? success "풀이"

    ```python
    def lasso_cd_warm(X, y, lam, beta_init=None, max_iter=1000, tol=1e-6):
        n, p = X.shape
        beta = beta_init.copy() if beta_init is not None else np.zeros(p)
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                r_j = y - X @ beta + X[:, j] * beta[j]
                rho_j = X[:, j] @ r_j / n
                beta[j] = soft_threshold(rho_j, lam)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta
    ```

    $\lambda$를 큰 값에서 작은 값으로 훑으며 정칙화 경로를 계산할 때, 해 사상의 연속성에 의해
    $\lambda_k$에서의 해는 $\lambda_{k+1}$에서의 해와 가깝다. 직전 해를 초기값으로 쓰면 좌표하강
    반복 횟수가 수백에서 몇 회로 줄어드는 것이 보통이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 14.** <span class="diff med" title="중간"></span> 관측치보다 설명변수가 많은 자료($n = 100$, $p = 200$)에서 계수 5개만 0이 아니게
생성하라. 교차검증으로 고른 $\lambda$에서 라쏘를 적합하고, 선택된 변수 중 참양성과 위양성의
개수를 보고하라.

</div>

??? success "풀이"

    ```python
    import numpy as np

    np.random.seed(0)
    n, p = 100, 200
    X = np.random.randn(n, p)
    X = (X - X.mean(0)) / X.std(0)
    beta_true = np.zeros(p)
    beta_true[:5] = [3, -2, 4, -1, 2]
    y = X @ beta_true + np.random.randn(n) * 1.5

    lambdas_cv = np.logspace(1, -2, 30)
    mse_cv = cv_lasso(X, y, lambdas_cv)
    best_lam = lambdas_cv[np.argmin(mse_cv)]
    beta_hat = lasso_cd(X, y, best_lam)

    selected = np.abs(beta_hat) > 1e-8
    true_support = np.abs(beta_true) > 0

    tp = np.sum(selected & true_support)
    fp = np.sum(selected & ~true_support)
    print(f"True positives:  {tp}/5")
    print(f"False positives: {fp}")
    ```

    출력:

    ```
    True positives:  5/5
    False positives: 11
    ```

    실행 결과는 $\hat{\lambda} = 0.2807$에서 참양성 5/5, 위양성 11개(선택된 변수 총 16개)다.
    즉 라쏘는 $p > n$인 고차원 희소 상황에서도 참
    신호를 **모두** 되찾지만, 예측오차를 최소화하는 $\lambda$는 잡음변수도 상당수 함께
    들여보낸다. 이는 라쏘 자체의 결함이라기보다 교차검증이 **예측**을 최적화하기 때문이다.
    지지집합을 정확히 복원하려면 더 큰 $\lambda$(1-표준오차 규칙)나 사후 라쏘, 안정성 선택
    같은 추가 절차가 필요하다.

    한 가지 덧붙일 것이 있다. 위 코드는 표준화를 **폴드 밖에서** 한 번에 해 놓고 `cv_lasso` 에
    넘긴다. 검증 폴드의 평균·표준편차가 훈련에 새어 들어간 것이므로 엄밀하게는 누수다. 여기서는
    열마다 평균·표준편차 두 수만 새고 설명변수가 반응과 무관하게 생성되었으므로 $\hat\lambda$가
    눈에 띄게 달라지지 않지만, 실제 자료에서는 표준화를 폴드 **안**에서 훈련 부분만으로 추정해
    검증 부분에 적용해야 한다. `sklearn` 의 `Pipeline` 에 `StandardScaler` 를 넣는 까닭이
    바로 이것이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 15.** <span class="diff med" title="중간"></span> $X$의 열이 완전계수(full column rank)이면 라쏘 해가 유일하지만, $X$의 열이
일차종속이면 유일하지 않을 수 있음을 증명하라.

</div>

??? success "풀이"

    라쏘 목적함수는

    $$
    f(\beta) = \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1
    $$

    이다. 첫 항은 헤세행렬이 $\frac{1}{n}X^\top X$인 볼록 이차식이다. $X$가 완전계수이면
    $X^\top X$는 양정치이므로 $f$는 **강볼록**이다. 강볼록 함수는 최소점을 많아야 하나 가지므로
    해가 유일하다.

    $X$의 열이 일차종속이면 $X^\top X$는 양반정치일 뿐이다. 이차항은 볼록이지만 강볼록이 아니고
    $L_1$ 항 역시 볼록이지만 강볼록이 아니다. 합은 볼록이므로 최소점의 집합은 볼록집합이지만,
    한 점보다 많을 수 있다.

    **반례:** $X = [x \mid x]$(동일한 두 열)라 하자. 이때 $X\beta = x(\beta_1 + \beta_2)$이므로
    목적함수는 $s = \beta_1 + \beta_2$에만 의존하고, 주어진 $s$에 대해 $\|\beta\|_1$은
    $\beta_1, \beta_2$의 부호가 같을 때 최솟값 $|s|$를 갖는다. 따라서 문제는

    $$
    \min_{s}\ \frac{1}{2n}\|y - sx\|_2^2 + \lambda |s|
    $$

    로 환원되고, 그 최적해 $s^*$는 유일하다. 그러나 원래 문제의 해집합은
    $\{(\beta_1, \beta_2) : \beta_1 + \beta_2 = s^*,\ \beta_1\beta_2 \ge 0\}$이라는 선분 전체다.
    예컨대 $s^* = 1$이면 $(\alpha, 1-\alpha)$가 모든 $\alpha \in [0,1]$에 대해 최적이다.
    즉 해가 유일하지 않다. $\square$

---

<div class="drillbox" markdown>

**연습문제 16.** <span class="diff med" title="중간"></span> 보기 1(1)은 `lasso_cd` 의 갱신식이 옳으려면
$a_j = \lVert x_j \rVert_2^2 / n = 1$이어야 함을 보였다. 그 조건이 깨지면 **정확히 얼마나** 틀리는지를
수로 짚어라. $p = 1$로 두고 $\lVert x \rVert_2^2/n = 1$인 열 $x$와 상수 $c > 0$에 대해 설계를
$\tilde{x} = cx$라 하자(변수의 단위를 바꾼 것이다). $c_0 = x^\top y / n$이라 쓴다.

**(1)** 같은 $\lambda$에서 $\tilde{x}$에 대한 **참** 라쏘 해 $\tilde\beta$를 $c_0, \lambda, c$로 적어라.
단위를 바꾸기 전의 해 $\hat\beta = S(c_0, \lambda)$에 대해 $\tilde\beta = \hat\beta / c$가 성립하는가.

**(2)** `lasso_cd` 에 $\tilde{x}$를 넣으면 무엇이 돌아오는가. 그것이 참값의 몇 배인지 $c$로 적어라.

**(3)** $c_0 = 0.8$, $\lambda = 0.3$, $c = 2$에서 (1)(2)에 나온 세 수를 적어라. 끝으로, $\lambda$를
다시 고르는 것만으로 `lasso_cd` 의 오류를 $p \ge 2$에서 고칠 수 있는가.

</div>

??? success "풀이"

    **(1)** 목적함수는 $\frac{1}{2n}\lVert y - cx\beta \rVert_2^2 + \lambda\lvert\beta\rvert$이다.
    $\gamma = c\beta$로 바꾸어 쓰면

    $$
    \frac{1}{2n}\lVert y - x\gamma \rVert_2^2 + \frac{\lambda}{c}\lvert\gamma\rvert
    $$

    이고, $\lVert x\rVert_2^2/n = 1$이므로 이것은 보기 4(1)의 직교 꼴이다. 따라서
    $\gamma^* = S(c_0,\ \lambda/c)$이고

    $$
    \tilde\beta = \frac{1}{c}\,S\!\left(c_0,\ \frac{\lambda}{c}\right)
    = \frac{\operatorname{sign}(c_0)}{c}\left(\lvert c_0\rvert - \frac{\lambda}{c}\right)_+
    $$

    이다. **$\tilde\beta = \hat\beta/c$는 성립하지 않는다.** 단위를 $c$배 하면 계수는 $1/c$배가
    되어야 하는데, 벌점 $\lambda\lvert\beta\rvert$는 그 작아진 계수에 걸리므로 **유효 벌점이
    $\lambda/c$로 함께 묽어진다.** 적합을 그대로 보존하려면 $\lambda \mapsto \lambda/c$로 벌점까지
    같이 바꾸어야 하고, 그때에만 $\tilde\beta = \hat\beta/c$다. 이것이 "라쏘는 척도동등이 아니다"의
    가장 작은 보기다.

    **(2)** $p = 1$이면 부분잔차가 $r_1 = y - \tilde{x}\beta + \tilde{x}\beta = y$이므로
    $\rho_1 = \tilde{x}^\top y / n = c\,c_0$이고, 코드는 분모 $a_1 = c^2$ 없이

    $$
    \beta^{\text{code}} = S(c\,c_0,\ \lambda)
    $$

    를 돌려준다(다음 바퀴에도 같은 값이 나오므로 고정점이고 반복이 끝난다). 한편

    $$
    c^2\,\tilde\beta = c\,S\!\left(c_0, \frac{\lambda}{c}\right)
    = \operatorname{sign}(c_0)\left(c\lvert c_0\rvert - \lambda\right)_+
    = S(c\,c_0,\ \lambda)
    $$

    이므로

    $$
    \beta^{\text{code}} = c^2\,\tilde\beta
    $$

    이다. **오류는 정확히 $c^2 = a_1$배다.** 길이를 센티미터에서 미터로 바꾸면($c = 1/100$)
    코드의 답이 참값의 $10^{-4}$배가 된다.

    **(3)** $c_0 = 0.8$, $\lambda = 0.3$, $c = 2$에서

    - 참 라쏘 해: $\tilde\beta = S(0.8,\ 0.15)/2 = 0.65/2 = 0.325$
    - 척도동등이 성립했다면: $\hat\beta/c = S(0.8,\ 0.3)/2 = 0.5/2 = 0.25$
    - 코드가 돌려주는 값: $S(1.6,\ 0.3) = 1.3 = 4 \times 0.325$

    세 수가 모두 다르다. 가운데 수와 첫 수의 차이는 **벌점이 척도에 불변이 아니라서** 생기고,
    셋째 수와 첫 수의 차이는 **코드가 $a_j$를 빠뜨려서** 생긴다. 두 결함은 서로 다른 것이다.

    $\lambda$를 다시 고르는 것으로는 고칠 수 없다. $p = 1$이면 $S(1.6, \lambda') = 0.325$가 되도록
    $\lambda' = 1.275$를 쓰면 되지만, $p \ge 2$에서 열마다 $c_j$가 다르면 좌표별로
    $\lambda/c_j^2$ 꼴의 **서로 다른** 벌점이 필요하다. 하나의 $\lambda$로는 불가능하다. 고치는 길은
    갱신식에 $1/a_j$를 넣거나, 보기 1처럼 **먼저 표준화해서 $a_j = 1$로 만드는 것**뿐이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 17.** <span class="diff hard" title="어려움"></span> 보기 2의 코드는 $\lambda_{\max}$를
`np.abs(X.T @ (y - y.mean())).max() / n` 로 재면서, 해는 `lasso_cd(X, y, lam)` 로 **중심화하지 않은**
`y` 에 대해 구하고 절편도 두지 않는다. 보기 4의 자료에서 $\bar y = 0.2049 \ne 0$이므로 이것은
어긋나 보인다.

**(1)** 두 식이 같은 수를 주는 까닭을 보이고, 더 나아가 `lasso_cd(X, y, lam)` 와
`lasso_cd(X, y - y.mean(), lam)` 이 **모든 $\lambda$에서 같은 해**를 준다는 것을 증명하라.

**(2)** 그렇다면 절편은 어디로 갔는가. 절편을 벌점에서 제외한 모형
$\min_{\beta_0, \beta} \frac{1}{2n}\lVert y - \beta_0\mathbf{1} - X\beta \rVert_2^2 + \lambda\lVert\beta\rVert_1$
의 해에서 $\hat\beta_0$을 구하고, $\hat\beta$가 (1)의 해와 같음을 보여라.

**(3)** `X` 자리에 중심화하지 않은 `X_raw` 를 넣었다면 (1)(2) 중 무엇이 깨지는가. 보기 4의
자료에서 그 어긋남을 수로 보여라($\lambda_{\max}$를 두 식으로 재면 각각 $3.7689$와 $3.7585$다).

</div>

??? success "풀이"

    **(1)** 보기 4가 `X = (X_raw - X_mean) / X_std` 로 열마다 평균을 뺐으므로 $X$의 모든 열이
    중심화되어 있고, 따라서

    $$
    X^\top \mathbf{1} = 0
    $$

    이다. 이 한 줄이 전부다. 먼저 $\lambda_{\max}$는

    $$
    X^\top(y - \bar y\mathbf{1}) = X^\top y - \bar y\,X^\top\mathbf{1} = X^\top y
    $$

    이므로 중심화하든 않든 **똑같은 벡터**이고 두 식의 값이 같다.

    해도 같다. $\mathbf{1}^\top(y - \bar y\mathbf{1}) = 0$과 $\mathbf{1}^\top X\beta = 0$을 쓰면

    $$
    \lVert y - X\beta \rVert_2^2
    = \lVert (y - \bar y\mathbf{1}) - X\beta \rVert_2^2
      + 2\bar y\,\mathbf{1}^\top\!\big[(y - \bar y\mathbf{1}) - X\beta\big] + n\bar y^2
    = \lVert (y - \bar y\mathbf{1}) - X\beta \rVert_2^2 + n\bar y^2
    $$

    이 되어 교차항이 사라진다. 두 목적함수가 $\beta$와 무관한 상수 $\bar y^2/2$만큼 차이 나므로
    최소점의 집합이 **완전히 같다.** 실제로 보기 4의 자료에 $\lambda = 0.2212$를 넣고 두 가지로
    풀면 계수 차이가 $10^{-15}$ 수준으로 부동소수점 오차뿐이다.

    **(2)** $\beta$를 고정하고 $\beta_0$에 대해 최소화하면 $\hat\beta_0 = \frac{1}{n}\mathbf{1}^\top(y - X\beta)$
    이고, $\mathbf{1}^\top X = 0$이므로

    $$
    \hat\beta_0 = \bar y = 0.2049
    $$

    다. **$\beta$가 무엇이든 절편은 $\bar y$다.** 이것을 다시 넣으면 남는 문제가

    $$
    \min_{\beta}\ \frac{1}{2n}\lVert (y - \bar y\mathbf{1}) - X\beta \rVert_2^2 + \lambda\lVert\beta\rVert_1
    $$

    이고 (1)에 의해 그 해가 `lasso_cd(X, y, lam)` 의 해와 같다. 곧 **열을 중심화하면 절편이
    기울기 문제에서 완전히 분리된다.** 보기 4가 절편을 보고하지 않은 것은 빠뜨린 것이 아니라
    $\bar y$로 이미 알고 있기 때문이고, 절편에 벌점을 걸지 않는 관례가 이 분리를 깨지 않는
    까닭도 같다.

    **(3)** `X_raw` 를 쓰면 $X_{\text{raw}}^\top\mathbf{1} \ne 0$이므로 (1)과 (2)가 **동시에** 깨진다.
    보기 4의 자료에서 $\max_j \lvert \mathbf{1}^\top x_j \rvert / n = 0.1340$으로 0이 아니고,
    그 결과

    $$
    \frac{\lVert X_{\text{raw}}^\top(y - \bar y\mathbf{1})\rVert_\infty}{n} = 3.7689
    \ne
    \frac{\lVert X_{\text{raw}}^\top y\rVert_\infty}{n} = 3.7585
    $$

    로 $\lambda_{\max}$가 어느 식을 쓰는지에 따라 달라진다. 더 나쁜 것은 (2)인데,
    $\hat\beta_0 = \bar y - \frac{1}{n}\mathbf{1}^\top X_{\text{raw}}\hat\beta$가 $\hat\beta$에 의존하므로
    절편과 기울기가 뒤엉킨다. 절편 없이 중심화하지 않은 $y$로 푼 해는 절편 있는 모형의 해와
    다르고, 그 차이는 $\lambda$가 커질수록 벌어진다. 거기에 연습문제 16의 $a_j \ne 1$ 오류까지
    겹친다.

    **표준화는 벌점을 공평하게 걸기 위한 권고에 그치지 않는다.** 중심화는 절편을 떼어 내고,
    척도 맞추기는 갱신식을 옳게 만든다. 둘 다 이 구현의 전제조건이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 18.** <span class="diff hard" title="어려움"></span> 상관된 두 변수에서 라쏘가 하나를
버리는 자리를 종이에서 끝까지 구한다. 표준화된 설계 $X \in \mathbb{R}^{n \times 2}$가

$$
\frac{X^\top X}{n} = \begin{pmatrix} 1 & 0.8 \\ 0.8 & 1 \end{pmatrix},
\qquad
\frac{X^\top y}{n} = \begin{pmatrix} 1.0 \\ 0.9 \end{pmatrix}
$$

를 만족한다고 하자. 이 쪽의 목적함수
$\frac{1}{2n}\lVert y - X\beta\rVert_2^2 + \lambda\lVert\beta\rVert_1$에 대해 답하라.

**(1)** $\lambda_{\max}$를 구하라.

**(2)** 모든 $\lambda > 0$에 대한 라쏘 해 $\hat\beta(\lambda)$를 구간으로 나누어 닫힌 꼴로 적어라.
특히 $\hat\beta_2$가 0이 되는 $\lambda$를 구하고, 상관계수 $\rho$와 $c = X^\top y/n$에 대한
일반식 $\lambda_2^\ast = (c_2 - \rho c_1)/(1 - \rho)$를 유도하라.

**(3)** $\hat\beta_2$가 떨어진 뒤 $\hat\beta_1$이 $x_2$를 아예 본 적 없는 단순회귀의 연성 문턱과
같아짐을 확인하라. 같은 설계에서 능형 해 $\hat\beta^{\text{ridge}}(\lambda) = (X^\top X/n + \lambda I)^{-1}X^\top y/n$을
구해 $\lambda = 0,\ 0.5,\ 10$에서 두 방법을 견주고, 무엇이 다른지 한 줄로 적어라.

</div>

??? success "풀이"

    $G = X^\top X/n$, $c = X^\top y/n$으로 쓰면 목적함수는 상수를 뺀 꼴로

    $$
    \frac{1}{2}\beta^\top G\beta - c^\top\beta + \lambda\lVert\beta\rVert_1
    $$

    이고, 최적조건(하위기울기)은 $j$마다

    $$
    (G\beta)_j - c_j + \lambda\, \partial\lvert\beta_j\rvert \ni 0,
    \qquad
    \partial\lvert\beta_j\rvert =
    \begin{cases}
    \{\operatorname{sign}(\beta_j)\} & \beta_j \ne 0,\\
    [-1, 1] & \beta_j = 0
    \end{cases}
    $$

    이다.

    **(1)** $\beta = 0$이 해일 조건은 $\lvert c_j \rvert \le \lambda$가 모두에서 성립하는 것이므로

    $$
    \lambda_{\max} = \max_j \lvert c_j \rvert = \max(1.0,\ 0.9) = 1.0
    $$

    이다.

    **(2)** 먼저 두 좌표가 모두 양수인 구간을 찾는다. 그때 조건은 $G\beta = c - \lambda\mathbf{1}$,
    곧

    $$
    \beta_1 + 0.8\beta_2 = 1.0 - \lambda,
    \qquad
    0.8\beta_1 + \beta_2 = 0.9 - \lambda
    $$

    이다. 두 식을 빼면 $0.2(\beta_1 - \beta_2) = 0.1$에서

    $$
    \beta_1 - \beta_2 = 0.5
    $$

    로 **$\lambda$와 무관한** 차이가 나오고, 더하면 $1.8(\beta_1 + \beta_2) = 1.9 - 2\lambda$에서

    $$
    \beta_1 + \beta_2 = \frac{1.9 - 2\lambda}{1.8}
    $$

    이다. 따라서

    $$
    \hat\beta_1 = 0.25 + \frac{1.9 - 2\lambda}{3.6},
    \qquad
    \hat\beta_2 = -0.25 + \frac{1.9 - 2\lambda}{3.6}.
    $$

    $\lambda = 0$을 넣으면 최소제곱 해 $\hat\beta^{\text{OLS}} = (7/9,\ 5/18) = (0.7778,\ 0.2778)$이다.
    이 꼴이 유효한 것은 $\hat\beta_2 \ge 0$인 동안, 곧

    $$
    \frac{1.9 - 2\lambda}{3.6} \ge 0.25
    \iff 1.9 - 2\lambda \ge 0.9
    \iff \lambda \le 0.5
    $$

    일 때다. $\lambda > 0.5$에서는 $\hat\beta_2 = 0$이고 첫 조건만 남아
    $\hat\beta_1 = S(c_1, \lambda) = 1 - \lambda$다. 이것이 정말 해인지는 둘째 좌표의 하위기울기
    조건으로 확인한다.

    $$
    \lvert (G\hat\beta)_2 - c_2 \rvert = \lvert 0.8(1-\lambda) - 0.9 \rvert = 0.1 + 0.8\lambda \le \lambda
    \iff \lambda \ge 0.5
    $$

    로 정확히 맞아떨어진다. $\lambda \ge 1$에서는 (1)에 의해 둘 다 0이다. 모아 적으면

    $$
    \hat\beta(\lambda) =
    \begin{cases}
    \left(0.25 + \dfrac{1.9 - 2\lambda}{3.6},\ \ -0.25 + \dfrac{1.9 - 2\lambda}{3.6}\right), & 0 \le \lambda \le 0.5,\\[3mm]
    (1 - \lambda,\ 0), & 0.5 \le \lambda \le 1,\\[2mm]
    (0,\ 0), & \lambda \ge 1
    \end{cases}
    $$

    이고 $\lambda = 0.5$에서 $(0.5,\ 0)$으로, $\lambda = 1$에서 $(0,\ 0)$으로 두 꺾임점이 모두
    연속이다.

    일반식은 같은 계산을 문자로 하면 나온다. $G = \begin{pmatrix} 1 & \rho \\ \rho & 1\end{pmatrix}$에서
    $\hat\beta = G^{-1}(c - \lambda\mathbf{1})$이므로

    $$
    \hat\beta_2 = \frac{(c_2 - \lambda) - \rho(c_1 - \lambda)}{1 - \rho^2}
    = \frac{(c_2 - \rho c_1) - \lambda(1 - \rho)}{1 - \rho^2}
    $$

    이고 $\hat\beta_2 \ge 0 \iff \lambda \le \lambda_2^\ast = \dfrac{c_2 - \rho c_1}{1 - \rho}$다.
    여기서는 $(0.9 - 0.8)/(1 - 0.8) = 0.5$다. **분모 $1 - \rho$가 열쇠다.** $\rho \to 1$이면
    $\lambda_2^\ast \to \pm\infty$ 쪽으로 터지면서, 아주 조금만 약한 변수도 거의 모든 $\lambda$에서
    즉시 떨어진다. 상관이 높을수록 라쏘가 둘 중 하나를 더 매몰차게 버린다는 뜻이다.

    **(3)** $\lambda > 0.5$에서 $\hat\beta_1 = 1 - \lambda = S(c_1, \lambda)$인데, 이것은 $x_1$만 넣은
    단순회귀($\lVert x_1\rVert_2^2/n = 1$이므로 직교설계)의 라쏘 해와 **글자까지 같다.** 한 변수가
    떨어지면 남은 변수는 상관 $0.8$을 처음부터 없었던 것처럼 푼다.

    능형은 $\big((1+\lambda)^2 - 0.64\big)$를 분모로 하여

    $$
    \hat\beta^{\text{ridge}}(\lambda)
    = \frac{1}{(1+\lambda)^2 - 0.64}
      \begin{pmatrix} 0.28 + \lambda \\ 0.1 + 0.9\lambda \end{pmatrix}
    $$

    이다. 값을 넣으면

    | $\lambda$ | 라쏘 | 능형 | 능형의 $\hat\beta_2/\hat\beta_1$ |
    |---|---|---|---|
    | $0$ | $(0.7778,\ 0.2778)$ | $(0.7778,\ 0.2778)$ | $0.357$ |
    | $0.5$ | $(0.5000,\ 0)$ | $(0.4845,\ 0.3416)$ | $0.705$ |
    | $10$ | $(0,\ 0)$ | $(0.0854,\ 0.0756)$ | $0.885$ |

    **능형의 $\hat\beta_2$는 어떤 $\lambda$에서도 0이 아니다.** 분자 $0.1 + 0.9\lambda$가
    $\lambda \ge 0$에서 양수이기 때문이다. 게다가 $\lambda = 0$에서 $0.2778$이던 것이 $\lambda = 0.5$에서
    $0.3416$으로 **올라간다** — 상관된 짝의 계수를 서로 끌어당겨 비율을 $0.357 \to 0.705 \to 0.885$로
    1에 가깝게 만드는 것이 능형이 하는 일이고, 그래서 능형은 변수선택을 하지 못한다. 라쏘는
    거꾸로 $\lambda = 0.5$에서 짝을 끊어 하나만 남긴다. 같은 "축소"라는 말로 덮을 수 없는 두 가지
    행동이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 19.** <span class="diff hard" title="어려움"></span> 연습문제 5가 증명한 비유일성을 수로
보여라. 연습문제 13의 `lasso_cd_warm` 에 $X = [x \mid x]$(완전히 같은 두 열)와 여러 초기값을 주어,

**(1)** 초기값에 따라 **개별 계수가 달라지고**,

**(2)** 그런데도 $\hat\beta_1 + \hat\beta_2$, 적합값 $X\hat\beta$, 목적함수값은 **모두 같고**,

**(3)** 그 합이 연습문제 15에서 환원한 일변량 문제의 닫힌 꼴 $s^\ast = S(x^\top y/n,\ \lambda)$와 **같음**

을 확인하라. $m = 200$, $\lambda = 0.3$, $y = 2x + \varepsilon$, $\varepsilon \sim N(0, 0.5^2)$으로 하라.
끝으로, 왜 개별 계수는 싣지 말고 합만 실어야 하는지 한 줄로 적어라.

</div>

??? success "풀이"

    ```python
    rng = np.random.default_rng(0)
    m = 200
    x_d = rng.standard_normal(m)
    x_d = (x_d - x_d.mean()) / x_d.std()          # ||x||^2 / m = 1
    y_d = 2.0 * x_d + rng.standard_normal(m) * 0.5
    X_d = np.column_stack([x_d, x_d])             # 두 열이 완전히 같다
    lam_d = 0.3

    # 환원된 일변량 문제의 닫힌 꼴: s* = S(x'y/m, lam)
    c_d = x_d @ y_d / m
    s_star = soft_threshold(c_d, lam_d)
    print(f"x'y/m = {c_d:.4f},   닫힌 꼴 s* = S(x'y/m, lam) = {s_star:.4f}")

    def obj_d(b):
        """이 쪽의 라쏘 목적함수."""
        return np.sum((y_d - X_d @ b) ** 2) / (2 * m) + lam_d * np.abs(b).sum()

    ref = None
    for b0 in (np.zeros(2), np.array([1.0, 1.0]),
               np.array([-1.0, 3.0]), np.array([0.6, 0.6])):
        b = lasso_cd_warm(X_d, y_d, lam_d, beta_init=b0, max_iter=100000, tol=1e-12)
        if ref is None:
            ref = X_d @ b
        same = bool(np.abs(X_d @ b - ref).max() < 1e-12)
        print(f"초기값 ({b0[0]:5.1f},{b0[1]:5.1f}) -> 해 ({b[0]:.4f}, {b[1]:.4f})   "
              f"합 {b.sum():.4f}   목적함수 {obj_d(b):.6f}   예측 같음 {same}")
    ```

    출력:

    ```text
    x'y/m = 1.9667,   닫힌 꼴 s* = S(x'y/m, lam) = 1.6667
    초기값 (  0.0,  0.0) -> 해 (1.6667, 0.0000)   합 1.6667   목적함수 0.676845   예측 같음 True
    초기값 (  1.0,  1.0) -> 해 (0.6667, 1.0000)   합 1.6667   목적함수 0.676845   예측 같음 True
    초기값 ( -1.0,  3.0) -> 해 (0.0000, 1.6667)   합 1.6667   목적함수 0.676845   예측 같음 True
    초기값 (  0.6,  0.6) -> 해 (1.0667, 0.6000)   합 1.6667   목적함수 0.676845   예측 같음 True
    ```

    **네 개별 해가 전부 다르다.** $(1.6667,\ 0)$, $(0.6667,\ 1.0)$, $(0,\ 1.6667)$,
    $(1.0667,\ 0.6)$ — 연습문제 15가 예고한 선분
    $\{(\beta_1,\beta_2) : \beta_1 + \beta_2 = s^\ast,\ \beta_1\beta_2 \ge 0\}$의 네 점이고,
    양 끝점 $(s^\ast, 0)$과 $(0, s^\ast)$까지 실제로 나온다. 알고리즘은 틀린 것이 아니다. 선분
    위의 어느 점이든 전역최소점이므로 **초기값이 답을 고른다.**

    왜 그런지는 갱신식 한 줄로 보인다. $\lVert x\rVert_2^2/m = 1$이고 $c_d = x^\top y/m$일 때
    두 갱신은

    $$
    \beta_1 \leftarrow S(c_d - \beta_2,\ \lambda),
    \qquad
    \beta_2 \leftarrow S(c_d - \beta_1,\ \lambda)
    $$

    이다. $\beta_1 = t \in [0, s^\ast]$, $\beta_2 = s^\ast - t$를 넣으면
    $S(c_d - s^\ast + t,\ \lambda) = S(\lambda + t,\ \lambda) = t$로 **그 자리가 고정점**이다.
    선분 전체가 좌표하강의 고정점이니 한 바퀴만 돌아도 멈춘다.

    **유일한 양은 정확히 재현된다.** 합이 네 번 모두 $1.6667$로 닫힌 꼴
    $s^\ast = S(1.9667,\ 0.3) = 1.6667$과 같고, 적합값 $X\hat\beta = x\,(\beta_1+\beta_2)$가 $10^{-12}$
    안에서 일치하고, 목적함수도 $0.676845$로 같다. 목적함수가 $\beta$에 $s = \beta_1 + \beta_2$를
    통해서만 의존하고(적합항), $\lVert\beta\rVert_1 = \lvert s\rvert$ 역시 부호가 같은 한 $s$로만
    정해지기 때문이다.

    그러므로 **개별 계수는 보고할 수 없고 합·적합값·목적함수만 보고할 수 있다.** 개별 계수는
    초기값·좌표를 훑는 순서·수렴 허용오차에 따라 달라지는 양이고, 그런 수를 소수점까지 싣는
    글은 다른 사람의 컴퓨터에서 재현되지 않는다. 설명변수가 거의 중복인 실제 자료에서
    "라쏘가 고른 변수"를 단정해서는 안 되는 까닭도 같다 — 자료를 조금 흔들면 고르는 쪽이
    바뀐다. 엘라스틱넷이 $L_2$ 항을 조금 섞어 목적함수를 강볼록으로 만드는 것이 바로 이
    선분을 한 점 $(s^\ast/2,\ s^\ast/2)$으로 줄이기 위한 장치다. $\square$

---

<div class="drillbox" markdown>

**연습문제 20.** <span class="diff hard" title="어려움"></span> 보기 4(1)은 직교설계에서 라쏘가
최소제곱 해의 연성 문턱임을 보였다. 그 축소가 **언제나 이득인가**를 묻는다. 직교설계
$X^\top X/n = I$에서 좌표마다 $\hat\beta^{\text{OLS}} \sim N(\beta,\ \tau^2)$이고 독립이라 하자
($\tau^2 = \sigma^2/n$). 한 좌표만 떼어 $\beta \ne 0$, $\hat\beta \sim N(\beta, \tau^2)$,
$\delta_\lambda = S(\hat\beta, \lambda)$로 쓰고 $R(\lambda) = E\big[(\delta_\lambda - \beta)^2\big]$라 하자.

**(1)** $\lambda$가 작을 때 $R(\lambda) = \tau^2 - 4\tau\varphi(\beta/\tau)\,\lambda + \lambda^2 + O(\lambda^3)$
임을 보이고, 이로부터

$$
R'(0^+) = -4\tau\,\varphi(\beta/\tau) < 0,
\qquad \varphi(z) = \frac{1}{\sqrt{2\pi}}e^{-z^2/2}
$$

를 유도하라. (힌트: $A = \{\lvert\hat\beta\rvert \le \lambda\}$에서 참 손실과
$\hat\beta - \lambda\operatorname{sign}(\hat\beta)$의 손실이 얼마나 다른지, 그리고 $P(A)$가 얼마나
작은지를 함께 보라. 또 $\hat\beta = \beta + \tau u$로 두고
$E\big[u\operatorname{sign}(u + \beta/\tau)\big]$를 구하라.)

**(2)** (1)이 말하는 것을 한 줄로 적어라. 또 $\varphi(\beta/\tau)$가 들어온 것이 왜 자연스러운지,
신호 대 잡음 $\beta/\tau$가 클 때와 작을 때로 나누어 설명하라.

**(3)** $\tau = 1$, $\beta = 1.5$에서 2차항까지의 근사로 최적 $\lambda$와 그때의 $R$를 구하고,
최소제곱($\lambda = 0$)보다 몇 퍼센트 나은지 적어라.

</div>

??? success "풀이"

    **(1)** $A = \{\lvert\hat\beta\rvert \le \lambda\}$라 하고, 매끄러운 대역 추정량
    $\tilde\delta_\lambda = \hat\beta - \lambda\operatorname{sign}(\hat\beta)$를 둔다. $A^c$에서는
    $\delta_\lambda = \tilde\delta_\lambda$로 **똑같다.** $A$에서는 $\delta_\lambda = 0$이고
    $\lvert\tilde\delta_\lambda\rvert \le 2\lambda$이므로 두 손실의 차이는

    $$
    \left\lvert (\delta_\lambda - \beta)^2 - (\tilde\delta_\lambda - \beta)^2 \right\rvert
    = \left\lvert \tilde\delta_\lambda\left(\tilde\delta_\lambda - 2\beta\right) \right\rvert
    \le 2\lambda\big(2\lambda + 2\lvert\beta\rvert\big) = O(\lambda)
    $$

    다. 한편 $\hat\beta$의 밀도가 유계이고 $\beta \ne 0$이므로 $\lambda$가 작을 때

    $$
    P(A) = \int_{-\lambda}^{\lambda} \frac{1}{\tau}\varphi\!\left(\frac{t - \beta}{\tau}\right)dt
    = \frac{2\lambda}{\tau}\varphi\!\left(\frac{\beta}{\tau}\right) + O(\lambda^3)
    = O(\lambda)
    $$

    이다. 둘을 곱하면 $A$가 기여하는 차이가 $O(\lambda) \cdot O(\lambda) = O(\lambda^2)$이므로
    **1차항까지는 $\delta_\lambda$를 $\tilde\delta_\lambda$로 바꾸어도 된다.**

    더 세게 말할 수도 있다. $A$에서 두 손실의 차이는
    $2\beta\big(\hat\beta - \lambda\operatorname{sign}(\hat\beta)\big) + O(\lambda^2)$인데
    $\hat\beta$와 $\lambda\operatorname{sign}(\hat\beta)$는 모두 $\hat\beta \mapsto -\hat\beta$에 대해
    **홀함수**이고 $A = [-\lambda, \lambda]$는 대칭이다. 밀도를 $\beta$에서 1차까지 펴면
    상수항이 주는 적분이 대칭성으로 지워지고 $O(\lambda^3)$만 남는다. 그래서 아래 전개의
    $\lambda^2$ 계수까지가 **정확**하고, (3)에서 2차식을 최소화하는 것이 뜻을 갖는다.
    실제로 $\tau = 1$, $\beta = 1.5$에서 나머지를 수치적으로 재면 $\lambda^3$에 비례해
    $-0.2806\lambda^3$으로 줄어든다.

    이제 매끄러운 쪽을 전개한다.

    $$
    E\big[(\tilde\delta_\lambda - \beta)^2\big]
    = E\big[(\hat\beta - \beta)^2\big]
      - 2\lambda\,E\big[(\hat\beta - \beta)\operatorname{sign}(\hat\beta)\big]
      + \lambda^2 E\big[\operatorname{sign}(\hat\beta)^2\big]
    $$

    에서 첫 항은 $\tau^2$, 셋째 항은 $\lambda^2$다($P(\hat\beta = 0) = 0$). 가운데를 구한다.
    $\hat\beta = \beta + \tau u$, $u \sim N(0,1)$, $a = \beta/\tau$라 두면

    $$
    E\big[(\hat\beta - \beta)\operatorname{sign}(\hat\beta)\big]
    = \tau\,E\big[u\operatorname{sign}(u + a)\big]
    $$

    이고 $\operatorname{sign}(u + a) = 1 - 2\cdot\mathbf{1}\{u < -a\}$이므로

    $$
    E\big[u\operatorname{sign}(u + a)\big]
    = E[u] - 2E\big[u\,\mathbf{1}\{u < -a\}\big]
    = 0 - 2\big(-\varphi(-a)\big)
    = 2\varphi(a)
    $$

    다(여기서 $E[u\mathbf{1}\{u < c\}] = \int_{-\infty}^{c} u\varphi(u)\,du = -\varphi(c)$를 썼고
    $\varphi$가 짝함수임을 썼다). 따라서
    $E\big[(\hat\beta - \beta)\operatorname{sign}(\hat\beta)\big] = 2\tau\varphi(\beta/\tau)$이고

    $$
    R(\lambda) = \tau^2 - 4\tau\varphi\!\left(\frac{\beta}{\tau}\right)\lambda + \lambda^2 + O(\lambda^3)
    $$

    에서

    $$
    R'(0^+) = -4\tau\,\varphi\!\left(\frac{\beta}{\tau}\right) < 0
    $$

    이다. $\varphi > 0$이므로 **도함수가 반드시 음수다.** $\square$

    **(2)** $R(0) = \tau^2$가 최소제곱의 MSE인데 $R'(0^+) < 0$이므로, **아무리 좋은 자료에서도
    최소제곱보다 MSE가 작은 $\lambda > 0$이 반드시 존재한다.** 편향을 조금 사서 분산을 더 많이
    파는 거래가 $\lambda = 0$ 근방에서는 언제나 남는 장사라는 뜻이다. 6장 정리 1의 분해로 읽으면
    $\lambda$의 1차에서 편향제곱은 $O(\lambda^2)$로 늦게 자라는데 분산은 $O(\lambda)$로 먼저
    줄어드는 것이다.

    $\varphi(\beta/\tau)$가 들어온 것도 자연스럽다. 축소가 이득이 되는 것은 $\hat\beta$의 부호가
    $\beta$의 부호와 **같을** 때뿐인데(그때만 $\lambda$만큼 깎는 것이 참값 쪽으로 가는 걸음이다),
    부호가 뒤집힐 위험의 크기를 재는 양이 바로 $0$에서의 밀도 $\varphi(\beta/\tau)/\tau$다.

    - **$\beta/\tau$가 크면**(신호가 깨끗하면) $\varphi(\beta/\tau)$가 지수적으로 작아져 이득이
      거의 없다. $\beta = 3\tau$면 $\varphi(3) = 0.0044$로 기울기가 $-0.018\tau$에 불과하다.
      이미 잘 추정된 큰 계수를 깎아서 얻을 것은 없다.
    - **$\beta/\tau$가 작으면** $\varphi(0) = 0.3989$가 되어 기울기가 $-1.596\tau$로 가장 급하다.
      잡음에 묻힌 작은 계수일수록 0으로 당기는 이득이 크고, 이것이 희소한 참 계수에서 라쏘가
      이기는 까닭이다. 실제로 $\beta = 0$이면 $A$에서 $\delta_\lambda = 0$이 **정답**이어서
      이득이 더 커진다.

    **(3)** $\tau = 1$, $\beta = 1.5$면 $\varphi(1.5) = 0.129518$이므로

    $$
    R'(0^+) = -4 \times 0.129518 = -0.518070,
    \qquad
    R(\lambda) \approx 1 - 0.518070\,\lambda + \lambda^2
    $$

    이다. 이 2차식의 최소는

    $$
    \lambda^\ast \approx \frac{0.518070}{2} = 0.2590,
    \qquad
    R(\lambda^\ast) \approx 1 - \frac{0.518070^2}{4} = 0.9329
    $$

    에서 일어난다. 최소제곱의 $R(0) = 1$에 비해 $6.7\%$ 작다.

    2차 근사이므로 $\lambda^\ast$ 자체는 참 최적값과 조금 다르다($O(\lambda^3)$ 항을 버렸고,
    그 자리는 이미 $P(A)$가 무시할 만큼 작지 않은 영역이다). 그러나 **이득이 있다는 결론은
    1차항 하나로 이미 결정된다.** $\lambda$를 $0$에서 아주 조금 올리는 것만으로 MSE가 기울기
    $-0.518$로 내려가기 시작하고, 그것을 뒤집을 수 있는 항은 없다. 보기 4의 교차검증이
    $\hat\lambda = 0.2212 > 0$을 고른 것은 우연이 아니라 이 부등식이 보장한 바다. $\square$

---

## 정리하며

라쏘는 **축소와 변수선택을 동시에** 한다.

$$
\hat{\boldsymbol\beta}_{\text{lasso}}=\arg\min\;\|\mathbf y-\mathbf X\boldsymbol\beta\|^2+\lambda\|\boldsymbol\beta\|_1
$$

- **$L_1$ 벌점이 계수를 정확히 $0$ 으로 만든다.** 제약 영역이 마름모라 꼭짓점에서 해가 잡히며, 그 꼭짓점이 축 위에 있다. **기하가 희소성을 설명한다.**
- **닫힌 형태가 없다.** 절댓값이 미분 불가능하므로 좌표하강이나 LARS 같은 알고리즘을 쓴다.
- **상관된 변수 중 하나만 고르는 경향이 있다.** 나머지는 $0$ 이 되며, **어느 것이 뽑히느냐가 자료의 작은 변화에 민감하다.** 엘라스틱넷이 이 문제를 완화한다.
- **선택한 변수의 계수에 대한 추론은 타당하지 않다.** 11장의 선택 후 추론 문제가 그대로 있으며, 사후선택 추론이라는 별도의 이론이 필요하다.
- **$p>n$ 에서도 최대 $n$ 개까지만 고를 수 있다.** 구조적 한계다.

다음 절 **[라쏘 회귀 보기](lasso_housing_regularization_path.md)**로 넘어간다.
