# 로지스틱 회귀의 가능도


## 설정

$y^{(i)}\in\{0,1\}$이고 $A$가 $n\times(p+1)$ 계획행렬(절편을 위한 1의 열을 포함)일 때, 관측치
$(A[i,:],\; y^{(i)})$ $n$개가 주어졌다고 하자. 관측치 $i$의 예측확률은

$$
\sigma^{(i)} = \sigma\!\bigl(z^{(i)}\bigr),
\qquad
z^{(i)} = A[i,:]\,\boldsymbol{\theta}
$$

이다.

## 베르누이 가능도

각 이름표 $y^{(i)}$는 성공확률이 $\sigma^{(i)}$인 베르누이 확률변수로 모형화된다. 전체 자료의
가능도는

$$
\mathcal{L}(\boldsymbol{\theta})
= \prod_{i=1}^{n}\bigl[\sigma^{(i)}\bigr]^{y^{(i)}}
  \bigl[1-\sigma^{(i)}\bigr]^{1-y^{(i)}}
$$

이다.

## 교차엔트로피 손실(음의 로그가능도)

음의 로그를 취하면 **교차엔트로피 손실**(**로그손실**이라고도 한다)을 얻는다.

$$
\ell = -\sum_{i=1}^{n}
  \Bigl[
    y^{(i)}\log\sigma^{(i)}
    + \bigl(1-y^{(i)}\bigr)\log\bigl(1-\sigma^{(i)}\bigr)
  \Bigr]
$$

$\boldsymbol{\theta}$에 대해 $\ell$을 최소화하는 것은 $\mathcal{L}$을 최대화하는 것과 같다.

### 왜 교차엔트로피인가

교차엔트로피 손실은 분류에서 제곱오차 같은 대안보다 선호되는 두 가지 중요한 성질을 갖는다.

1. **볼록성.** $\ell$은 $\boldsymbol{\theta}$에 대해 볼록이므로 모든 국소최솟값이 전역최솟값이다.
2. **정보이론적 동기.** 참 분포 $y$와 모형 분포 $\hat{y}=\sigma$ 사이의 교차엔트로피는, 참
   분포 대신 모형 분포로 이름표를 부호화할 때 추가로 필요한 비트 수를 잰다.

### 극단에서의 행동

$y^{(i)}=1$인데 $\sigma^{(i)}\to 0$이면 항 $-\log\sigma^{(i)}\to+\infty$가 되어, 확신에 찬
오답에 큰 벌칙을 준다. $y^{(i)}=0$이고 $\sigma^{(i)}\to 1$인 경우도 대칭적으로 마찬가지다.
바로 이 비대칭적 벌칙이 모형을 잘 보정된 확률로 이끈다.

## 정보이론과의 관계

$q$를 모형 분포, $p$를 경험(참) 분포라 하면 교차엔트로피는

$$
H(p, q) = -\sum_x p(x)\log q(x)
$$

이다. 교차엔트로피는 $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$로 분해되며, 여기서 $H(p)$는 참
분포의 엔트로피이고 $D_{\mathrm{KL}}$은 쿨백-라이블러 발산이다. $H(p)$는
$\boldsymbol{\theta}$와 무관한 상수이므로, 교차엔트로피를 최소화하는 것은 KL 발산을 최소화하는
것과 같다.

## 수치적 안정성

실무에서는 $\log 0$을 피하기 위해 로그 안에 작은 상수 $\varepsilon$(예: $10^{-6}$)을 더한다.

$$
\ell \approx -\sum_{i=1}^{n}
  \Bigl[
    y^{(i)}\log\bigl(\sigma^{(i)}+\varepsilon\bigr)
    + \bigl(1-y^{(i)}\bigr)\log\bigl(1-\sigma^{(i)}+\varepsilon\bigr)
  \Bigr]
$$

또는 많은 프레임워크는 수치적으로 안정한 항등식

$$
-\log\sigma(z) = \log(1+e^{-z}) = \operatorname{softplus}(-z)
$$

를 이용해 로짓 $z^{(i)}$에서 곧바로 손실을 계산한다.


## 연습문제

**연습문제 1.**
교차엔트로피 손실의 기울기가
$\nabla_{\boldsymbol{\theta}}\,\ell = A^\top(\boldsymbol{\sigma} - \mathbf{y})$
임을 유도하라. 여기서 $\boldsymbol{\sigma} = (\sigma^{(1)}, \dots, \sigma^{(n)})^\top$이다.

??? success "풀이"

    한 관측치의 기여를 $\ell_i = -[y^{(i)}\log\sigma^{(i)} + (1-y^{(i)})\log(1-\sigma^{(i)})]$
    라 하고 $z^{(i)}$로 미분한다. 연쇄법칙과
    $\dfrac{d\sigma}{dz} = \sigma(1-\sigma)$를 쓰면

    $$
    \frac{\partial \ell_i}{\partial z^{(i)}}
    = -\left[\frac{y^{(i)}}{\sigma^{(i)}} - \frac{1-y^{(i)}}{1-\sigma^{(i)}}\right]\sigma^{(i)}(1-\sigma^{(i)})
    $$

    이고, 괄호를 정리하면

    $$
    = -\left[y^{(i)}(1-\sigma^{(i)}) - (1-y^{(i)})\sigma^{(i)}\right]
    = \sigma^{(i)} - y^{(i)}
    $$

    이다. $z^{(i)} = A[i,:]\boldsymbol{\theta}$이므로
    $\partial z^{(i)}/\partial \boldsymbol{\theta} = A[i,:]^\top$이고, 모든 $i$에 대해 더하면

    $$
    \nabla_{\boldsymbol{\theta}}\,\ell = \sum_{i=1}^n (\sigma^{(i)} - y^{(i)})\,A[i,:]^\top
    = A^\top(\boldsymbol{\sigma} - \mathbf{y})
    $$

    를 얻는다. 최소제곱의 정규방정식 $A^\top(A\hat\beta - y) = 0$과 형태가 같다는 점이
    인상적이다. 차이는 $\boldsymbol{\sigma}$가 $\boldsymbol{\theta}$의 비선형함수라는 것이며,
    바로 그 때문에 닫힌 형태의 해가 없다. $\square$

---

**연습문제 2.**
$\ell$의 헤세행렬이 $H = A^\top W A$($W = \operatorname{diag}(\sigma^{(i)}(1-\sigma^{(i)}))$)
임을 보이고, 이로부터 $\ell$이 볼록임을 결론지어라. 언제 강볼록이 되는가?

??? success "풀이"

    연습문제 1에서 $\partial \ell/\partial z^{(i)} = \sigma^{(i)} - y^{(i)}$이므로 한 번 더
    미분하면 $\partial^2 \ell/\partial (z^{(i)})^2 = \sigma^{(i)}(1-\sigma^{(i)})$이다. 서로
    다른 관측치는 교차항을 만들지 않으므로

    $$
    H = \nabla^2_{\boldsymbol{\theta}}\,\ell = A^\top W A,
    \qquad W = \operatorname{diag}\bigl(\sigma^{(i)}(1-\sigma^{(i)})\bigr)
    $$

    이다. 임의의 $v$에 대해 $v^\top H v = \sum_i w_i (A[i,:]v)^2 \ge 0$이고
    $w_i = \sigma^{(i)}(1-\sigma^{(i)}) > 0$이므로 $H$는 양반정치이고 $\ell$은 볼록이다.

    $A$가 완전열계수이면 $Av = 0 \Rightarrow v = 0$이므로 $v^\top H v > 0$이 되어 $H$가
    양정치, 즉 $\ell$이 강볼록이 된다. 그러나 이는 **주어진 $\boldsymbol{\theta}$에서**의
    이야기다. 자료가 완전히 분리 가능하면 $\|\boldsymbol{\theta}\|\to\infty$에 따라
    $w_i \to 0$이 되어 $H \to 0$이고, 하한은 도달되지 않는다(연습문제 5). $\square$

---

**연습문제 3.**
절편만 있는 모형($z^{(i)} = \theta_0$)을 생각하자. MLE가 $\hat\sigma = \bar{y}$임을 보여라.
$n = 10$이고 그중 3개가 1일 때 $\hat\theta_0$과 최소 교차엔트로피 손실을 계산하라.

??? success "풀이"

    $z^{(i)} = \theta_0$이면 모든 $\sigma^{(i)}$가 같은 값 $\sigma$이다. 연습문제 1의 결과에서
    $A$가 1의 열 하나뿐이므로 기울기는

    $$
    \frac{\partial \ell}{\partial \theta_0} = \sum_{i=1}^n (\sigma - y^{(i)}) = n\sigma - \sum_i y^{(i)}
    $$

    이고, 0으로 놓으면 $\hat\sigma = \frac{1}{n}\sum_i y^{(i)} = \bar{y}$이다. 즉 절편만 있는
    로지스틱 회귀는 표본비율을 되돌려 준다.

    $n = 10$, $\sum y^{(i)} = 3$이면 $\hat\sigma = 0.3$이고

    $$
    \hat\theta_0 = \operatorname{logit}(0.3) = \log\frac{0.3}{0.7} = -0.8473
    $$

    이다. 최소 손실은

    $$
    \ell_{\min} = -10\bigl[0.3\log 0.3 + 0.7\log 0.7\bigr] = 10 \times 0.61086 = 6.1086
    $$

    이다. 대괄호 안의 값 $0.61086$이 바로 베르누이(0.3)의 엔트로피(내트 단위)이므로,
    $\ell_{\min} = n\,H(\bar{y})$이다. 즉 설명변수가 없는 모형의 손실은 결과 자체의 엔트로피이며,
    이것이 모든 로지스틱 모형이 넘어서야 할 기준선(영이탈도)이다. $\square$

---

**연습문제 4.**
위의 $\varepsilon$ 보정이 왜 편향을 만드는지 수치로 보이고, softplus 방식이 왜 더 나은지
설명하라.

??? success "풀이"

    $\varepsilon = 10^{-6}$일 때 항 $-\log(\sigma + \varepsilon)$을 보자.

    | $\sigma$ | $-\log\sigma$ | $-\log(\sigma+10^{-6})$ | 차이 |
    |---|---|---|---|
    | $10^{-3}$ | $6.9078$ | $6.9068$ | $0.0010$ |
    | $10^{-6}$ | $13.8155$ | $13.1224$ | $0.6931$ |

    $\sigma$가 $\varepsilon$ 수준으로 작아지면 보정이 손실을 $\log 2 = 0.6931$만큼 깎아낸다.
    즉 확신에 찬 오답에 대한 벌칙이 인위적으로 $-\log\varepsilon$ 근처에서 잘린다. 이는
    수치적 무한대는 막아 주지만 목적함수 자체를 바꾸는 것이며, 특히 극단적으로 잘못된 예측이
    있을 때 최적해를 이동시킨다.

    softplus 방식은 $\sigma$를 거치지 않고 로짓에서 직접 계산한다.

    $$
    -\log\sigma(z) = \operatorname{softplus}(-z), \qquad
    -\log(1-\sigma(z)) = \operatorname{softplus}(z)
    $$

    예를 들어 $z = -800$이면 $\sigma(z)$는 배정도 부동소수점에서 0으로 반올림되어
    $-\log\sigma$가 `inf`가 되지만, $\operatorname{softplus}(800) = 800.0$은 정확히 계산된다
    (`np.logaddexp(0, 800)`). 근사가 전혀 개입하지 않으므로 목적함수가 왜곡되지 않는다.
    $\square$

---

**연습문제 5.**
자료가 **완전히 분리 가능**하다고 하자. 즉 모든 $i$에 대해
$y^{(i)}(2 A[i,:]\boldsymbol{\theta}^* ) > 0$이 되는 $\boldsymbol{\theta}^*$가 존재한다고 하자
(양성과 음성을 오차 없이 가르는 초평면이 있다는 뜻이다). 이때 MLE가 존재하지 않음을 보여라.

??? success "풀이"

    분리 초평면을 주는 $\boldsymbol{\theta}^*$를 잡고 $c > 0$에 대해
    $\boldsymbol{\theta} = c\,\boldsymbol{\theta}^*$를 생각하자. 분리 가능성에 의해
    $y^{(i)} = 1$인 모든 $i$에서 $z^{(i)} = c\,A[i,:]\boldsymbol{\theta}^* > 0$이고,
    $y^{(i)} = 0$인 모든 $i$에서 $z^{(i)} < 0$이다.

    $c \to \infty$이면 $y^{(i)}=1$인 관측치에서 $\sigma^{(i)} \to 1$, $y^{(i)}=0$인 관측치에서
    $\sigma^{(i)} \to 0$이므로 모든 항이 0으로 가고 $\ell(c\,\boldsymbol{\theta}^*) \to 0$이다.
    그런데 $\ell > 0$은 항상 성립하므로 하한 0은 어떤 유한한
    $\boldsymbol{\theta}$에서도 도달되지 않는다. 따라서 최소점이 존재하지 않고, MLE는 정의되지
    않는다.

    **실무에서 나타나는 증상:** 계수와 그 표준오차가 반복이 진행될수록 함께 발산한다.
    `statsmodels`는 "Perfect separation detected" 경고를 내고, `sklearn`은 기본으로 벌점을 주기
    때문에(기본 `C=1.0`) 유한한 답을 돌려주지만 그 답은 MLE가 아니다. 해결책은 벌점을 명시적으로
    넣거나(L2 정칙화, 19.4절 참조) 파스 보정(Firth's correction)을 쓰거나, 결과를 완벽히
    예측하는 변수를 모형에서 빼는 것이다. $\square$
