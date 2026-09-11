# 불량조건 계획행렬

OLS 추정량은 행렬 $\mathbf{X}^\top\mathbf{X}$의 역행렬을 요구한다. 이 행렬이 특이행렬에 가까우면 역행렬 계산이 수치적으로 불안정해지고, 자료의 작은 교란이 추정 계수의 큰 변동으로 이어진다. 이 불안정성을 이해하면, 통계적인 편향-분산 논증과 별개로 **수치적 안정화 도구**로서의 정칙화가 보인다.

## 조건수

행렬 $\mathbf{A}$의 **조건수**는 $\mathbf{A}\mathbf{x} = \mathbf{b}$의 해가 $\mathbf{b}$의 교란에 얼마나 민감한지를 정량화한다. $\mathbf{X}^\top\mathbf{X}$처럼 대칭 양반정치 행렬에서는

$$
\kappa(\mathbf{X}^\top\mathbf{X}) = \frac{\lambda_{\max}(\mathbf{X}^\top\mathbf{X})}{\lambda_{\min}(\mathbf{X}^\top\mathbf{X})}
$$

이며 $\lambda_{\max}$와 $\lambda_{\min}$은 각각 최대·최소 고윳값이다.

$\mathbf{X}$의 특이값이 $d_1 \geq d_2 \geq \cdots \geq d_p \geq 0$이면 $\mathbf{X}^\top\mathbf{X}$의 고윳값은 $d_1^2, \ldots, d_p^2$이므로

$$
\kappa(\mathbf{X}^\top\mathbf{X}) = \frac{d_1^2}{d_p^2} = \left(\frac{d_1}{d_p}\right)^2
$$

이다.

조건수가 1에 가까우면 잘 조건화된 문제다. 조건수가 $10^k$이면 연립방정식을 풀 때 정밀도가 대략 $k$자리 손실된다. $\kappa(\mathbf{X}^\top\mathbf{X}) > 10^{10}$이면 배정밀도 부동소수점 연산으로도 의미 없는 결과가 나올 수 있다.

## 불량조건의 원인

**준선형종속.** 두 개 이상의 설명변수가 거의 선형종속이면 $\mathbf{X}^\top\mathbf{X}$의 최소 고윳값이 0에 가까워지고 조건수가 무한대로 향한다.

**고차원.** $p$가 $n$에 가까우면 정확한 선형종속이 없어도 $\mathbf{X}^\top\mathbf{X}$가 매우 작은 고윳값을 갖는 경향이 있다. $p > n$이면 계수 부족 행렬이 되어 OLS 해가 유일하지 않다.

**척도 불일치.** 설명변수의 척도가 크게 다르면(하나는 미터, 다른 하나는 킬로미터) $\mathbf{X}^\top\mathbf{X}$의 고윳값 폭이 커진다. 회귀 전에 설명변수를 표준화하면 이 원인은 완화된다.

## OLS의 교란 민감도

OLS 연립방정식 $(\mathbf{X}^\top\mathbf{X})\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$를 생각하자. 우변이 작은 양 $\delta$만큼 교란되면 해의 변화는 다음 한계를 만족한다.

$$
\frac{\|\delta\hat{\boldsymbol{\beta}}\|}{\|\hat{\boldsymbol{\beta}}\|} \leq \kappa(\mathbf{X}^\top\mathbf{X})\,\frac{\|\delta(\mathbf{X}^\top\mathbf{y})\|}{\|\mathbf{X}^\top\mathbf{y}\|}
$$

즉 $\hat{\boldsymbol{\beta}}$의 상대오차가 교란의 상대크기 대비 최대 $\kappa(\mathbf{X}^\top\mathbf{X})$배까지 증폭될 수 있다. 조건수가 $10^6$이면 자료의 여섯째 소수점에 있는 교란이 계수 추정값의 첫째 자리를 바꿀 수 있다.

!!! warning "이것은 최악의 경우 한계다"
    위 부등식은 **부등식**이다. 실제 증폭은 교란의 방향에 달려 있으며, 대개 한계보다 훨씬 작다. 연습문제 2에서 $\kappa = 16{,}602$인 문제에서 실제 증폭이 무작위 교란에는 약 $1{,}700$배, 최악 방향에서도 약 $2{,}100$배임을 확인한다.

    그럼에도 $1{,}700$배는 재앙적이다. 조건수를 정확한 예측값이 아니라 **위험 신호**로 읽어야 한다.

!!! note "실무에서의 수치 불안정성"
    불량조건은 이론적 우려에 그치지 않는다. 실제 자료에서는 입력 시의 반올림, 부동소수점 연산, 측정 잡음이 모두 교란으로 작용한다. $\kappa(\mathbf{X}^\top\mathbf{X})$가 크면 이 불가피한 교란이 OLS 해를 오염시킨다.

## 안정화 장치로서의 정칙화

능형회귀는 역행렬을 취하기 전에 $\mathbf{X}^\top\mathbf{X}$에 $\lambda\mathbf{I}$를 더한다.

$$
\hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}
$$

$\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$의 고윳값은 $d_j^2 + \lambda$이므로 조건수는

$$
\kappa(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}) = \frac{d_1^2 + \lambda}{d_p^2 + \lambda}
$$

가 된다. $\lambda > 0$이 고윳값 스펙트럼의 바닥을 올리므로 조건수가 줄어든다. 모든 $\lambda > 0$에 대해

$$
\kappa(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}) < \kappa(\mathbf{X}^\top\mathbf{X})
$$

이다. $\lambda \gg d_1^2$인 극단에서는 조건수가 1(단위행렬)에 접근한다. $d_p^2$이 0에 가까울 때는 적당한 $\lambda$만으로도 조건수를 여러 자릿수 줄일 수 있다.

!!! note "Tikhonov 정칙화로서의 능형회귀"
    수치해석에서 거의 특이한 연립방정식에 $\lambda\mathbf{I}$를 더하는 것을 **Tikhonov 정칙화**라 부른다. 능형회귀는 이 고전적 기법의 통계적 구현이며, 수치적 관점과 통계적 관점 양쪽에서 정당화된다.

## 고윳값 스펙트럼과 유효계수

$\mathbf{X}^\top\mathbf{X}$의 고윳값 스펙트럼이 불량조건의 심각도를 드러낸다. 문턱 $\tau$에서의 **유효계수**를

$$
\text{rank}_\tau(\mathbf{X}) = \#\{j : d_j > \tau \cdot d_1\}
$$

로 정의한다. 유효계수가 $p$보다 훨씬 작으면 계수공간의 많은 방향이 자료로 잘 결정되지 않는다. 능형회귀는 SVD 축소인자 $d_j^2/(d_j^2 + \lambda)$가 말하는 대로 이렇게 잘 결정되지 않는 방향을 가장 강하게 축소한다.

## 실무에서 불량조건 탐지하기

| 진단 | 문턱 | 해석 |
|---|---|---|
| $\kappa(\mathbf{X}^\top\mathbf{X})$ | $> 10^4$ | 중간 정도의 불량조건 |
| $\kappa(\mathbf{X}^\top\mathbf{X})$ | $> 10^8$ | 심각한 불량조건 |
| 최소 고윳값 | 기계 엡실론 근처 | 수치적으로 특이 |
| 설명변수 $j$의 VIF | $> 10$ | 그 변수가 선형종속에 관여 |

분산팽창인자(VIF)는 다중공선성에 관한 다음 절에서 자세히 다룬다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
등상관 행렬 $\Sigma = \rho\mathbf{1}\mathbf{1}^\top + (1-\rho)\mathbf{I}$에 대해 $\lambda$를 더하면 조건수가 얼마나 줄어드는지 계산하라. $p = 8$, $\rho = 0.99$와 $0.999$에서 $\lambda = 0.001, 0.01, 0.1, 1$을 시도하라.

</div>

??? success "풀이"
    등상관 행렬의 고윳값은 해석적으로 알려져 있다. $1 + (p-1)\rho$가 한 개, $1 - \rho$가 $p-1$개다.

    ```python
    import numpy as np
    p = 8
    for rho in (0.99, 0.999):
        S = rho*np.ones((p, p)) + (1-rho)*np.eye(p)
        ev = np.sort(np.linalg.eigvalsh(S))[::-1]
        for lam in (0.001, 0.01, 0.1, 1.0):
            print(rho, lam, (ev[0]+lam)/(ev[-1]+lam))
    ```

    출력:

    ```
    0.99 0.001 721.0000000001435
    0.99 0.01 397.0000000000435
    0.99 0.1 73.00000000000145
    0.99 1.0 8.841584158415863
    0.999 0.001 3997.0000000013506
    0.999 0.01 727.5454545454992
    0.999 0.1 80.12871287128766
    0.999 1.0 8.984015984015992
    ```

    **$\rho = 0.99$** ($\lambda_{\max} = 7.93$, $\lambda_{\min} = 0.01$, $\kappa = 793$)

    | $\lambda$ | $\kappa(\Sigma + \lambda I)$ | 감소 배수 |
    |---:|---:|---:|
    | 0 | 793.0 | — |
    | 0.001 | 721.0 | 1.1 |
    | 0.01 | 397.0 | 2.0 |
    | 0.1 | 73.0 | 10.9 |
    | 1.0 | **8.84** | **89.7** |

    **$\rho = 0.999$** ($\lambda_{\min} = 0.001$, $\kappa = 7993$)

    | $\lambda$ | $\kappa(\Sigma + \lambda I)$ | 감소 배수 |
    |---:|---:|---:|
    | 0 | 7993.0 | — |
    | 0.001 | 3997.0 | 2.0 |
    | 0.01 | 727.6 | 11.0 |
    | 0.1 | 80.1 | 99.8 |
    | 1.0 | **8.98** | **890** |

    두 표에서 같은 구조가 보인다. **$\lambda$가 $\lambda_{\min}$과 같아지면 조건수가 대략 절반이 되고, $\lambda$가 $\lambda_{\min}$을 크게 넘으면 조건수가 $\lambda_{\max}/\lambda$로 결정된다.**

    $$
    \kappa(\Sigma + \lambda I) = \frac{\lambda_{\max} + \lambda}{\lambda_{\min} + \lambda} \approx \frac{\lambda_{\max}}{\lambda} \quad (\lambda \gg \lambda_{\min})
    $$

    실무적 함의: **$\lambda$를 $\lambda_{\min}$과 비교해서 정해야 한다.** $\lambda = 1$이 $\rho = 0.999$에서는 조건수를 890배 줄이지만, 잘 조건화된 문제($\rho = 0$, $\lambda_{\min} = 1$)에서는 조건수를 1에서 1로 유지하며 편향만 더한다.

<div class="drillbox" markdown>

**연습문제 2.**
조건수 한계가 실제로 얼마나 빡빡한지 확인하라. $\kappa(\mathbf{X}^\top\mathbf{X}) \approx 1.7\times10^4$인 자료에서 (i) $\mathbf{y}$를 교란할 때와 (ii) $\mathbf{X}^\top\mathbf{y}$를 직접 교란할 때의 증폭을 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from numpy.linalg import cond, svd
    rng = np.random.default_rng(11)
    n, p, rho = 40, 8, 0.999
    S = rho*np.ones((p, p)) + (1-rho)*np.eye(p)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(S).T
    y = X @ np.ones(p) + rng.normal(0, 1, n)
    A = X.T @ X; g = X.T @ y; b0 = np.linalg.solve(A, g)
    print(cond(A))          # 16601.7
    ```

    출력:

    ```
    16601.655795266535
    ```

    | 교란 대상 | 증폭 배수 (중앙값) |
    |:---|---:|
    | $\mathbf{y}$를 무작위로 교란 | **4.8** |
    | $\mathbf{X}^\top\mathbf{y}$를 무작위로 교란 | **1{,}705** |
    | $\mathbf{X}^\top\mathbf{y}$를 최소 고윳값 방향으로 교란 | **2{,}109** |
    | 이론적 최악의 경우 $\kappa$ | 16{,}602 |

    **세 가지를 읽어야 한다.**

    **첫째, $\mathbf{y}$를 교란하는 것과 $\mathbf{X}^\top\mathbf{y}$를 교란하는 것은 전혀 다르다.** $\mathbf{y}$의 잡음이 $\mathbf{X}^\top$을 통과하면서 걸러진다. $\mathbf{X}^\top\delta\mathbf{y}$는 $\mathbf{X}$의 열공간 구조를 따르므로 불량조건 방향에 거의 놓이지 않는다. 그래서 증폭이 $4.8$배에 그친다.

    이것이 실무에서 중요하다. **관측 잡음은 조건수가 예고하는 만큼 위험하지 않다.** 진짜 위험한 것은 $\mathbf{X}$ 자체의 오차와 $\mathbf{X}^\top\mathbf{X}$를 계산·저장하는 과정의 반올림이다. 그래서 수치 라이브러리는 $\mathbf{X}^\top\mathbf{X}$를 만들지 않고 QR 분해나 SVD로 직접 최소제곱을 푼다.

    **둘째, 한계는 빡빡하지 않다.** 최악 방향으로 교란해도 $2{,}109$배로 이론 한계 $16{,}602$의 13%다. 한계가 달성되려면 교란이 최소 고윳값 방향에 놓이는 **동시에** 해가 최대 고윳값 방향에 놓여야 하는데, 두 조건이 함께 성립하기 어렵다.

    **셋째, 그래도 $1{,}700$배는 심각하다.** 자료의 여섯째 소수점 교란이 계수의 셋째 소수점을 바꾼다. 조건수를 정확한 증폭 배수가 아니라 **경고등**으로 읽어야 한다는 뜻이다.

<div class="drillbox" markdown>

**연습문제 3.**
능형 정칙화가 실제로 교란 민감도를 줄이는지 확인하라. 연습문제 2의 자료에서 $\lambda = 0, 0.01, 1$에 대해 $\mathbf{y}$ 교란에 대한 증폭을 비교하라.

</div>

??? success "풀이"
    | $\lambda$ | 증폭 배수 (중앙값) |
    |---:|---:|
    | 0 (OLS) | 4.66 |
    | 0.01 | 4.80 |
    | 1.0 | **1.18** |

    **$\lambda = 0.01$은 아무 도움이 되지 않는다**(오히려 미세하게 나쁘다. 몬테카를로 잡음 범위다). $\lambda = 1$에서 증폭이 $4.7 \to 1.2$로 네 배 줄어든다.

    연습문제 1의 표와 함께 보면 이유가 분명하다. 이 자료에서 $\mathbf{X}^\top\mathbf{X}$의 최소 고윳값은 $\lambda = 0.01$보다 크므로, $\lambda = 0.01$은 스펙트럼의 바닥을 거의 올리지 못한다. **$\lambda$가 $d_p^2$과 견줄 만해야 안정화가 시작된다.**

    !!! tip "실무 절차"
        1. 설명변수를 표준화한다(척도 불일치를 먼저 제거한다).
        2. $\mathbf{X}$의 특이값 $d_j$를 계산한다. `np.linalg.svd(X, compute_uv=False)`.
        3. $d_p^2$를 본다. $\lambda$의 후보 범위를 $d_p^2$의 $0.1$배에서 $d_1^2$ 사이로 잡는다.
        4. 그 범위에서 교차검증으로 $\lambda$를 고른다.

        3단계를 건너뛰고 `alpha=1.0` 같은 기본값을 쓰면, 문제의 규모에 따라 아무 효과가 없거나 지나치게 강한 정칙화가 된다.

---

## 정리하며

조건수 $\kappa(\mathbf{X}^\top\mathbf{X})$는 OLS 계수가 자료의 작은 교란에 얼마나 민감하게 반응하는지를 잰다. 준선형종속, 고차원, 척도 불일치가 모두 조건수를 부풀려 OLS를 수치적으로 불안정하게 만든다. 능형 정칙화는 $\mathbf{X}^\top\mathbf{X}$에 $\lambda\mathbf{I}$를 더해 모든 고윳값을 $\lambda$만큼 올리고 조건수를 줄인다. 이 안정화 덕분에, 통계적 편향-분산 논증만으로는 정당화되지 않는 상황에서도 정칙화가 값지다.
