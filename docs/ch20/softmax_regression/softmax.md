# 소프트맥스 함수와 확률단체


## 정의

**소프트맥스** 함수는 $C$개의 실숫값 로짓으로 이루어진 벡터
$\mathbf{z}=(z_1,\ldots,z_C)$를 확률분포로 옮긴다.

$$
\operatorname{softmax}(\mathbf{z})_c
= \frac{e^{z_c}}{\sum_{c'=1}^{C}e^{z_{c'}}},
\qquad c=1,\ldots,C
$$

### 성질

1. **비음성:** 모든 출력이 엄밀히 양수다.
2. **합이 1:** $\sum_c\operatorname{softmax}(\mathbf{z})_c = 1$.
3. **단조성:** 로짓 $z_c$가 클수록 확률도 크다.
4. **평행이동 불변성:** 임의의 스칼라 $\alpha$에 대해
   $\operatorname{softmax}(\mathbf{z}+\alpha\mathbf{1}) = \operatorname{softmax}(\mathbf{z})$.

성질 4는 수치적 안정성에 활용된다. 지수화하기 전에 $\max_c z_c$를 뺀다.

## 확률단체

소프트맥스의 출력은 **확률단체**

$$
\Delta^{C-1} = \Bigl\{\mathbf{p}\in\mathbb{R}^C : p_c\ge 0,\;
\sum_c p_c = 1\Bigr\}
$$

위에 놓인다. $C=3$이면 3차원 공간 안의 삼각형이고, $C=10$(MNIST)이면 9차원 단체다.

## 시그모이드의 일반화로서의 소프트맥스

$C=2$이고 로짓이 $(z_1,z_2)$일 때,

$$
\operatorname{softmax}(\mathbf{z})_1
= \frac{e^{z_1}}{e^{z_1}+e^{z_2}}
= \frac{1}{1+e^{-(z_1-z_2)}}
= \sigma(z_1-z_2)
$$

이다. 즉 이항 소프트맥스는 두 로짓의 차에 시그모이드를 적용한 것과 정확히 같다.

## 온도 척도화

흔히 쓰는 변형으로 온도 모수 $\tau>0$를 도입한다.

$$
\operatorname{softmax}(\mathbf{z}/\tau)_c
= \frac{e^{z_c/\tau}}{\sum_{c'}e^{z_{c'}/\tau}}
$$

$\tau\to 0$이면 분포가 $\arg\max_c z_c$에 집중된 점질량으로 수렴하고(확정적 결정),
$\tau\to\infty$이면 균등분포에 가까워진다. 온도 척도화는 모형 보정과 생성모형(예: 언어모형의
"창의성" 조절)에 쓰인다.

## NumPy 구현

```python
import numpy as np

def softmax(z):
    """Numerically stable softmax."""
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
```

`np.max`를 빼는 것은 평행이동 불변성 덕분에 결과를 바꾸지 않으면서 `np.exp`의 오버플로를 막는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
소프트맥스 함수 계산

로짓 벡터가 $\mathbf{z} = (2, 1, -1)^\top$인 3범주 분류 문제를 생각하자.

**(a)** 다음 식으로 소프트맥스 확률 $\hat{p}_k = \text{softmax}(\mathbf{z})_k$
($k = 1, 2, 3$)를 계산하라.

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^C e^{z_j}}
$$

**(b)** 확률의 합이 1이고 각각이 비음수임을 확인하라.

**(c)** 소프트맥스가 평행이동 불변임을 보여라. 즉 임의의 상수 $c$에 대해
$\text{softmax}(\mathbf{z} + c\mathbf{1}) = \text{softmax}(\mathbf{z})$임을 보이고, 이 성질이
수치적 안정성에 왜 중요한지 설명하라.

</div>

??? success "풀이"

    **(a)** 먼저 지수를 계산한다.

    $$
    e^{z_1} = e^2 \approx 7.389, \quad e^{z_2} = e^1 \approx 2.718, \quad e^{z_3} = e^{-1} \approx 0.368
    $$

    정규화 상수는

    $$
    \sum_{j=1}^3 e^{z_j} = 7.389 + 2.718 + 0.368 = 10.475
    $$

    이고, 소프트맥스 확률은

    $$
    \hat{p}_1 = \frac{7.389}{10.475} \approx 0.705, \quad \hat{p}_2 = \frac{2.718}{10.475} \approx 0.259, \quad \hat{p}_3 = \frac{0.368}{10.475} \approx 0.035
    $$

    이다.

    **(b)** $\hat{p}_1 + \hat{p}_2 + \hat{p}_3 = 0.705 + 0.259 + 0.035 \approx 1.0$이고, 지수는
    항상 양수이므로 각 확률도 양수다. 소프트맥스가 임의의 실숫값 로짓 벡터를 단체 위의 유효한
    확률분포로 옮긴다는 것을 확인해 준다.

    **(c)** 임의의 상수 $c$에 대해

    $$
    \text{softmax}(\mathbf{z} + c\mathbf{1})_k = \frac{e^{z_k + c}}{\sum_j e^{z_j + c}} = \frac{e^c \cdot e^{z_k}}{e^c \cdot \sum_j e^{z_j}} = \frac{e^{z_k}}{\sum_j e^{z_j}} = \text{softmax}(\mathbf{z})_k
    $$

    로 $e^c$ 인자가 소거된다. 이 성질이 **로그-합-지수 기법**의 핵심이다.
    $c = -\max_k z_k$로 두면 가장 큰 지수가 $e^0 = 1$이 되어 로짓이 커도 수치적 오버플로를
    막을 수 있다.

<div class="drillbox" markdown>

**연습문제 2.**
소프트맥스 회귀의 가중행렬

범주 $C = 3$개, 입력 특성 $d = 2$개(그리고 편향)인 소프트맥스 회귀에서 모형은

$$
\mathbf{z} = \mathbf{W}\mathbf{x} + \mathbf{b}
$$

를 계산한다. 여기서 $\mathbf{W} \in \mathbb{R}^{3 \times 2}$이고 $\mathbf{b} \in \mathbb{R}^3$
이다.

**(a)** 학습된 모수가

$$
\mathbf{W} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
$$

일 때, 입력 $\mathbf{x} = (1, 1)^\top$에 대한 로짓과 예측 범주를 구하라.

**(b)** 가중행렬의 기하학적 의미를 설명하라. $\mathbf{W}$의 각 행은 어떻게 선형 분류기를
정의하는가?

**(c)** 이 모형은 모수화가 중복되어 있다. 임의의 벡터 $\mathbf{v}$와 스칼라 $v_b$에 대해
$\mathbf{w}_k' = \mathbf{w}_k - \mathbf{v}$, $b_k' = b_k - v_b$로 둔 모수가 같은 소프트맥스
출력을 낸다. 이를 이용해 마지막 범주의 로짓이 항상 0이 되도록 만들어 자유 모수를 줄여라.

</div>

??? success "풀이"

    **(a)** 로짓을 계산하면

    $$
    \mathbf{z} = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1 \\ 1 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 0 \end{pmatrix} + \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix} = \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}
    $$

    이다. 세 로짓이 모두 같으므로 $\text{softmax}(\mathbf{z}) = (1/3, 1/3, 1/3)^\top$이다. 어느
    범주도 선호되지 않으며 모형이 이 입력에 대해 최대로 불확실하다. 어느 범주를 예측해도
    무방하다(관례상 첫 번째 범주).

    **(b)** $\mathbf{W}$의 각 행 $\mathbf{w}_k^\top$은 선형 점수함수
    $z_k = \mathbf{w}_k^\top \mathbf{x} + b_k$를 정의한다. 기하학적으로 $\mathbf{w}_k$는 특성
    공간에서 범주 $k$의 점수가 증가하는 방향이다. 범주 $i$와 $j$ 사이의 결정경계는 $z_i = z_j$인
    초평면, 즉 $(\mathbf{w}_i - \mathbf{w}_j)^\top \mathbf{x} + (b_i - b_j) = 0$이며, 가중벡터
    $\mathbf{w}_i - \mathbf{w}_j$가 이 초평면의 법선이다.

    **(c)** 마지막 범주를 기준으로 삼는다. $\mathbf{v} = \mathbf{w}_3 = (0, 0)^\top$,
    $v_b = b_3 = 1$로 두면

    $$
    \mathbf{W}' = \begin{pmatrix} 2 & -1 \\ -1 & 2 \\ 0 & 0 \end{pmatrix}, \qquad
    \mathbf{b}' = \begin{pmatrix} 0 - 1 \\ 0 - 1 \\ 1 - 1 \end{pmatrix} = \begin{pmatrix} -1 \\ -1 \\ 0 \end{pmatrix}
    $$

    이 된다. 이제 $\mathbf{w}_3' = \mathbf{0}$, $b_3' = 0$이므로 세 번째 범주의 로짓이 항상 0인
    기준범주가 된다. 평행이동 불변성에 의해 소프트맥스 출력은 전혀 바뀌지 않는다. 자유 모수는
    $3 \times 2 + 3 = 9$개에서 $2 \times 2 + 2 = 6$개로 줄고, 모수화가 식별 가능해진다.

    !!! note "실무에서는 중복 모수화를 그대로 두는 경우가 많다"
        기준범주를 고정하면 식별 가능해지지만, 신경망 구현에서는 대개 $C$개 행을 모두 유지한다.
        이유는 두 가지다. 첫째, 구현이 대칭적이어서 단순하다. 둘째, **정칙화가 있으면 중복이
        사라진다.** $L_2$ 벌점 $\frac{\lambda}{2}\|\mathbf{W}\|_F^2$을 걸면 같은 예측을 주는
        모수 중 노름이 최소인 것이 유일하게 선택되므로 식별성이 회복된다. 벌점 없이 소프트맥스
        회귀를 적합하면 계수 자체는 유일하지 않으며, 계수를 해석할 때 이 점을 반드시 기억해야
        한다. 예측확률은 언제나 유일하다.
