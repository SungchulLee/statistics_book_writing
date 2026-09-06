# 다항 로지스틱 회귀


## 이항에서 다범주로

로지스틱 회귀는 이항 반응변수를 모형화한다. 반응변수가 $C>2$개의 범주를 가지면 **다항 로지스틱
회귀**(**소프트맥스 회귀**라고도 한다)로 일반화한다. 하나의 가중벡터 $\boldsymbol{\theta}$
대신, 각 입력을 범주마다 하나씩 $C$개의 실숫값 점수(로짓)로 옮기는 가중행렬 $\mathbf{W}$와
편향벡터 $\mathbf{b}$를 학습한다.

## 모형 구조(단층)

관측치 $n$개와 특성 $p$개를 갖는 자료에 대해 단층 소프트맥스 모형은 다음을 계산한다.

$$
\underset{n \times C}{\mathbf{Z}}
= \underset{n \times p}{\mathbf{X}}\;
  \underset{p \times C}{\mathbf{W}} + \underset{1 \times C}{\mathbf{b}}
$$

$$
\underset{n \times C}{\hat{\mathbf{Y}}}
= \operatorname{softmax}(\mathbf{Z})
$$

여기서 $\hat{\mathbf{Y}}$의 각 행은 $C$개 범주에 대한 확률분포다.

## 이층 모형(은닉층 + 소프트맥스)

로지스틱 활성함수를 갖는 은닉층을 추가하면 얕은 신경망이 된다. 아래 MNIST 예제에서 쓰는 구조다.

$$
\begin{aligned}
\underset{n \times 100}{\mathbf{Z}^h}
  &= \underset{n \times 784}{\mathbf{X}}\;
     \underset{784 \times 100}{\mathbf{W}^h} + \underset{1 \times 100}{\mathbf{b}^h} \\[4pt]
\underset{n \times 100}{\mathbf{H}}
  &= \operatorname{logistic}\!\bigl(\mathbf{Z}^h\bigr) \\[4pt]
\underset{n \times 10}{\mathbf{Z}^o}
  &= \underset{n \times 100}{\mathbf{H}}\;
     \underset{100 \times 10}{\mathbf{W}^o} + \underset{1 \times 10}{\mathbf{b}^o} \\[4pt]
\underset{n \times 10}{\hat{\mathbf{Y}}}
  &= \operatorname{softmax}\!\bigl(\mathbf{Z}^o\bigr)
\end{aligned}
$$

로지스틱(시그모이드) 활성함수는

$$
\operatorname{logistic}(x) = \frac{1}{1+e^{-x}},
\qquad
\operatorname{logistic}'(x) = \operatorname{logistic}(x)\bigl(1-\operatorname{logistic}(x)\bigr)
$$

이다.

## MNIST 자료

MNIST 자료는 이 모형군의 표준 기준자료다.

$$
\mathbf{X} \in \mathbb{R}^{n\times 784},\quad
\mathbf{Y} \in \{0,1\}^{n\times 10}\;\text{(one-hot)},\quad
\mathbf{y}_{\text{cls}} \in \{0,\ldots,9\}^n
$$

각 이미지는 $28\times 28$ 화소이며 784차원 벡터로 펼친다. 화소값은 $[0,1]$로 척도화한다.

## 이항 로지스틱 회귀와의 관계

$C=2$이면 다항 로지스틱 회귀는 보통의 로지스틱 회귀로 환원된다. 두 범주 소프트맥스가 시그모이드
모형과 같은 결정경계를 주는 이유는 범주 확률의 로그비가 특성에 대해 일차식이기 때문이다.

$$
\log\frac{P(Y=1\mid\mathbf{x})}{P(Y=0\mid\mathbf{x})}
= (\mathbf{w}_1-\mathbf{w}_0)^T\mathbf{x} + (b_1-b_0)
$$


## 연습문제

**연습문제 1.**
$C = 2$인 소프트맥스 회귀가 이항 로지스틱 회귀와 동등함을 보이고, 로지스틱 모형의 계수
$\boldsymbol\theta$가 소프트맥스의 $\mathbf{w}_0$, $\mathbf{w}_1$과 어떤 관계인지 밝혀라.

??? success "연습문제 1 풀이"

    $C = 2$인 소프트맥스에서

    $$
    P(Y=1\mid\mathbf{x}) = \frac{e^{z_1}}{e^{z_0}+e^{z_1}}
    = \frac{1}{1+e^{-(z_1-z_0)}} = \sigma(z_1 - z_0)
    $$

    이고 $z_k = \mathbf{w}_k^T\mathbf{x} + b_k$이므로

    $$
    z_1 - z_0 = (\mathbf{w}_1-\mathbf{w}_0)^T\mathbf{x} + (b_1-b_0)
    $$

    이다. 따라서 소프트맥스는 계수

    $$
    \boldsymbol\theta = \mathbf{w}_1-\mathbf{w}_0, \qquad \theta_0 = b_1-b_0
    $$

    을 갖는 로지스틱 회귀와 정확히 같다.

    **중요한 귀결:** 자료가 결정하는 것은 오직 **차** $\mathbf{w}_1-\mathbf{w}_0$뿐이다.
    $\mathbf{w}_0$과 $\mathbf{w}_1$ 각각은 결정되지 않는다. 같은 벡터를 둘 다에 더해도 차가
    변하지 않으므로 예측이 같기 때문이다. 이것이 소프트맥스 모수화의 중복이며, 일반적인 $C$에
    대해서는 자유도가 $C$가 아니라 $C-1$임을 뜻한다. $\square$

---

**연습문제 2.**
MNIST($p = 784$, $C = 10$)에서 단층 소프트맥스 모형과 은닉 노드 100개짜리 이층 모형의 모수
개수를 각각 세어라. 두 모형이 실제로 갖는 자유도는 얼마인가?

??? success "연습문제 2 풀이"

    **단층 모형:**

    | 성분 | 크기 | 개수 |
    |---|---|---|
    | $\mathbf{W}$ | $784 \times 10$ | $7{,}840$ |
    | $\mathbf{b}$ | $10$ | $10$ |
    | **합계** | | $\mathbf{7{,}850}$ |

    **이층 모형:**

    | 성분 | 크기 | 개수 |
    |---|---|---|
    | $\mathbf{W}^h$ | $784 \times 100$ | $78{,}400$ |
    | $\mathbf{b}^h$ | $100$ | $100$ |
    | $\mathbf{W}^o$ | $100 \times 10$ | $1{,}000$ |
    | $\mathbf{b}^o$ | $10$ | $10$ |
    | **합계** | | $\mathbf{79{,}510}$ |

    이층 모형이 10배 이상 많은 모수를 쓴다.

    **실제 자유도.** 연습문제 1에서 보았듯이 소프트맥스 출력층은 중복되어 있다. 모든 로짓에서
    같은 값을 빼도 출력이 같으므로, 출력층에서 $784 + 1 = 785$개(단층) 또는
    $100 + 1 = 101$개(이층)의 자유도가 실제로는 없다. 따라서 유효 자유도는 각각 $7{,}065$개와
    $79{,}409$개다.

    MNIST의 훈련표본이 60,000개이므로, 단층 모형은 모수당 관측치가 약 8개인 반면 이층 모형은
    1개도 되지 않는다. 이것이 이층 모형에 정칙화나 조기 종료가 특히 필요한 이유다. $\square$

---

**연습문제 3.**
단층 소프트맥스 모형에서 교차엔트로피 손실의 $\mathbf{W}$에 대한 기울기가
$\nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^T(\hat{\mathbf{Y}} - \mathbf{Y})$
임을 유도하라.

??? success "연습문제 3 풀이"

    관측치 $i$에 대한 손실은 $\mathcal{L}_i = -\sum_c Y_{ic}\log \hat Y_{ic}$이고
    $\hat Y_{ic} = \operatorname{softmax}(\mathbf{z}_i)_c$이다. 먼저 소프트맥스의 야코비를 구한다.

    $$
    \frac{\partial \hat Y_{ic}}{\partial z_{ik}} = \hat Y_{ic}(\delta_{ck} - \hat Y_{ik})
    $$

    ($\delta_{ck}$는 크로네커 델타). 연쇄법칙으로

    $$
    \frac{\partial \mathcal{L}_i}{\partial z_{ik}}
    = -\sum_c \frac{Y_{ic}}{\hat Y_{ic}}\cdot \hat Y_{ic}(\delta_{ck}-\hat Y_{ik})
    = -\sum_c Y_{ic}(\delta_{ck}-\hat Y_{ik})
    $$

    이고, $\sum_c Y_{ic} = 1$(원-핫)이므로

    $$
    = -Y_{ik} + \hat Y_{ik}\sum_c Y_{ic} = \hat Y_{ik} - Y_{ik}
    $$

    를 얻는다. 즉 **로짓에 대한 기울기가 단순한 잔차다.**

    이제 $z_{ik} = \sum_j X_{ij}W_{jk} + b_k$이므로
    $\partial z_{ik}/\partial W_{jk} = X_{ij}$이고, 모든 관측치에 대해 더하면

    $$
    \frac{\partial \mathcal{L}}{\partial W_{jk}} = \sum_i X_{ij}(\hat Y_{ik} - Y_{ik})
    \quad\Longrightarrow\quad
    \nabla_{\mathbf{W}}\mathcal{L} = \mathbf{X}^T(\hat{\mathbf{Y}} - \mathbf{Y})
    $$

    이다. 편향은 $\nabla_{\mathbf{b}}\mathcal{L} = \mathbf{1}^T(\hat{\mathbf{Y}} - \mathbf{Y})$
    이다.

    이 결과가 우아한 이유는, 소프트맥스 야코비의 복잡한 항들이 교차엔트로피의 $1/\hat Y$와
    정확히 소거되기 때문이다. 이는 우연이 아니라 **로짓이 다항분포의 정준연결**이라는 사실의
    귀결이며, 19장에서 로지스틱 회귀의 기울기가 $A^T(\boldsymbol\sigma - \mathbf{y})$였던
    것과 같은 이유다. $\square$

---

**연습문제 4.**
은닉층의 활성함수를 제거하면(즉 $\mathbf{H} = \mathbf{Z}^h$로 두면) 이층 모형이 단층 모형과
동등해짐을 보여라. 이것이 비선형 활성함수의 필요성에 대해 무엇을 말해 주는가?

??? success "연습문제 4 풀이"

    활성함수 없이 $\mathbf{H} = \mathbf{Z}^h = \mathbf{X}\mathbf{W}^h + \mathbf{b}^h$이면,

    $$
    \mathbf{Z}^o = \mathbf{H}\mathbf{W}^o + \mathbf{b}^o
    = \mathbf{X}\underbrace{\mathbf{W}^h\mathbf{W}^o}_{=: \mathbf{W}^{\text{eff}}}
      + \underbrace{\mathbf{b}^h\mathbf{W}^o + \mathbf{b}^o}_{=: \mathbf{b}^{\text{eff}}}
    $$

    이다. 즉 $\mathbf{Z}^o$는 $\mathbf{X}$의 아핀함수이며, 이는 정확히 단층 모형의 형태다.
    층을 아무리 쌓아도 마찬가지다. 아핀사상의 합성은 아핀사상이다.

    **함의:** 깊이 자체가 표현력을 주는 것이 아니라 **비선형성**이 준다. 활성함수가 없으면
    79,510개의 모수를 학습하고도 $784 \times 10 + 10 = 7{,}850$개짜리 모형의 함수족을 벗어나지
    못한다. 모수만 늘고 표현력은 그대로다.

    한 가지 더 미묘한 점이 있다. $\mathbf{W}^{\text{eff}} = \mathbf{W}^h\mathbf{W}^o$의
    계수(rank)는 $\min(784, 100, 10) = 10$ 이하이므로, 은닉층이 $C$보다 좁으면
    오히려 단층 모형보다 **제약이 생긴다.** 이 경우 활성함수 없는 이층 모형은 저계수
    제약을 건 소프트맥스 회귀와 같다. $\square$

---

**연습문제 5.**
MNIST 화소값을 $[0, 255]$가 아니라 $[0,1]$로 척도화하는 이유는 무엇인가? 척도화하지 않으면
학습에 무슨 일이 일어나는가?

??? success "연습문제 5 풀이"

    화소값을 그대로 두면 입력의 크기가 255배 커지고, 그에 따라 로짓과 기울기의 크기도 커진다.
    구체적으로 세 가지 문제가 생긴다.

    1. **로짓 폭발과 포화.** $\mathbf{z} = \mathbf{X}\mathbf{W}$에서 $\mathbf{X}$가 255배
       커지면 초기 가중치가 같을 때 로짓도 255배가 된다. 소프트맥스가 곧바로 포화되어
       예측확률이 0이나 1이 되고, 기울기 $\hat{\mathbf{Y}} - \mathbf{Y}$는 여전히 유한하지만
       은닉층의 시그모이드는 도함수 $\sigma(1-\sigma) \approx 0$이 되어 학습이 멈춘다.

    2. **학습률 문제.** 19장 연습문제 3에서 보았듯 허용 가능한 학습률의 상한은
       $\lambda_{\max}(\mathbf{X}^T\mathbf{X})$에 반비례한다. 입력을 255배 키우면 이 값이
       $255^2 \approx 65{,}000$배 커지므로 학습률을 그만큼 줄여야 하고, 그러면 수렴이 그만큼
       느려진다.

    3. **정칙화의 의미 변화.** 벌점 $\|\mathbf{W}\|^2$은 가중치의 절대적 크기를 벌하는데,
       입력의 척도가 바뀌면 같은 예측을 내는 가중치의 크기도 바뀐다. 즉 같은 $\lambda$가
       전혀 다른 강도의 정칙화가 된다.

    $[0,1]$ 척도화는 세 문제를 한꺼번에 완화한다. 더 나아가 화소별로 평균 0, 분산 1로 표준화하면
    수렴이 더 빨라지지만, MNIST는 화소값 분포가 이미 비슷해 단순한 $255$로 나누기만 해도
    충분한 경우가 많다. $\square$
