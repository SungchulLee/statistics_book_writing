# 최대가능도 추정


## 교차엔트로피 손실의 기울기

교차엔트로피 손실은

$$
\ell = -\sum_{i=1}^{n}\bigl[y^{(i)}\log\sigma^{(i)} + (1-y^{(i)})\log(1-\sigma^{(i)})\bigr]
$$

이다. 시그모이드의 도함수 $\sigma'(z)=\sigma(z)(1-\sigma(z))$를 쓰면
$\boldsymbol{\theta}$에 대한 기울기가 깔끔한 행렬식으로 정리된다.

### 성분별 유도

$$
\frac{\partial\ell}{\partial\boldsymbol{\theta}}
= -\sum_{i=1}^{n}\left[
  \frac{y^{(i)}}{\sigma^{(i)}}\,\sigma^{(i)}(1-\sigma^{(i)})\,A[i,:]^T
  - \frac{1-y^{(i)}}{1-\sigma^{(i)}}\,\sigma^{(i)}(1-\sigma^{(i)})\,A[i,:]^T
\right]
$$

시그모이드 항이 약분되어

$$
\nabla\ell = \sum_{i=1}^{n}\bigl(\sigma^{(i)}-y^{(i)}\bigr)\,A[i,:]^T
$$

만 남는다.

### 행렬 형태

잔차를 열벡터 $\boldsymbol{\sigma}-\mathbf{y}$로 쓰고 계획행렬의 행을 쌓으면

$$
\nabla\ell = A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

이다. 이는 제곱손실을 쓰는 선형회귀의 기울기와 같은 형태이며, 잔차
$\hat{\mathbf{y}}-\mathbf{y}$가 $\boldsymbol{\sigma}-\mathbf{y}$로 바뀐 것뿐이다.

## 헤세행렬

기울기를 한 번 더 미분하면

$$
\nabla^2\ell
= \sum_{i=1}^{n}\sigma^{(i)}(1-\sigma^{(i)})\;A[i,:]^T\,A[i,:]
= A^T B\, A
$$

이고, 여기서 $B$는 $n\times n$ 대각행렬

$$
B = \operatorname{diag}\!\bigl(\sigma^{(1)}(1-\sigma^{(1)}),\;\ldots,\;\sigma^{(n)}(1-\sigma^{(n)})\bigr)
$$

이다. $B$의 모든 대각원소가 $0<\sigma(1-\sigma)\le\tfrac14$을 만족하므로, 헤세행렬은 임의의
$\boldsymbol{\theta}$에 대해 **양반정치**이고 교차엔트로피 손실이 볼록임을 확인해 준다.

## 경사하강

1차 갱신식은

$$
\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \alpha\,\nabla\ell
= \boldsymbol{\theta} - \alpha\,A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

이며, $\alpha$는 학습률이다.

## 뉴턴법과 IRLS

뉴턴(2차) 갱신은 헤세행렬을 이용한다.

$$
\boldsymbol{\theta}
= \boldsymbol{\theta}_0 - (A^TBA)^{-1}\,A^T(\boldsymbol{\sigma}-\mathbf{y})
$$

이는 **가중최소제곱**의 정규방정식으로 다시 쓸 수 있다. *작업 반응변수*를

$$
\mathbf{z} = A\boldsymbol{\theta}_0 - B^{-1}(\boldsymbol{\sigma}-\mathbf{y})
$$

로 정의하면 갱신식은

$$
\boldsymbol{\theta} = (A^TBA)^{-1}\,A^TB\,\mathbf{z}
$$

가 된다. 이는 가중최소제곱 문제
$\min_{\boldsymbol{\theta}}\|B^{1/2}(\mathbf{z}-A\boldsymbol{\theta})\|^2$의 해와 정확히 같다.

### 반복재가중최소제곱(IRLS)

가중행렬 $B$가 (($\boldsymbol{\sigma}$를 통해) 현재의 모수벡터 $\boldsymbol{\theta}$에
의존하므로, 정규방정식을 반복 적용하며 각 단계에서 $B$와 $\mathbf{z}$를 다시 계산해야 한다.
이 절차를 **반복재가중최소제곱(IRLS)**이라 한다(Rubin, 1983).

!!! info "IRLS 알고리즘"

    1. $\boldsymbol{\theta}_0$을 초기화한다.
    2. $\boldsymbol{\sigma} = \sigma(A\boldsymbol{\theta}_0)$을 계산한다.
    3. $B = \operatorname{diag}(\sigma^{(i)}(1-\sigma^{(i)}))$을 만든다.
    4. 작업 반응변수 $\mathbf{z} = A\boldsymbol{\theta}_0 - B^{-1}(\boldsymbol{\sigma}-\mathbf{y})$
       를 계산한다.
    5. $\boldsymbol{\theta} = (A^TBA)^{-1}A^TB\mathbf{z}$를 푼다.
    6. $\boldsymbol{\theta}_0 \leftarrow \boldsymbol{\theta}$로 놓고 수렴할 때까지 반복한다.

IRLS는 보통 적은 반복으로 수렴하며, 많은 고전적 로지스틱 회귀 해법기의 바탕이 되는
알고리즘이다.

## 구현: 경사하강으로 하는 로지스틱 회귀

다음 NumPy 구현은 1차 경사하강만으로 로지스틱 회귀를 처음부터 학습시킨다.

### 자료 적재

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(seed=1):
    url = ('https://raw.githubusercontent.com/codebasics/py/'
           'master/ML/7_logistic_reg/insurance_data.csv')
    df = pd.read_csv(url)
    x = df[['age']].values.reshape((-1, 1))
    y = df.bought_insurance.values.reshape((-1,))
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.5, random_state=seed)
    return x_train, x_test, y_train, y_test
```

### 모형 클래스

```python
import numpy as np

class LogisticRegression:
    def __init__(self, x, y, lr=2e-4, epochs=100_000, theta=None):
        self.x = x
        self.y = y
        self.lr = lr
        self.epochs = epochs
        self.theta = (theta if theta is not None
                      else np.random.normal(size=(x.shape[1] + 1, 1)))

    @staticmethod
    def design_matrix(x):
        ones = np.ones((x.shape[0], 1))
        return np.concatenate((ones, x), axis=1)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, x):
        A = self.design_matrix(x)
        z = A @ self.theta
        return self.sigmoid(z).reshape((-1,))

    def predict(self, x):
        p = self.predict_proba(x)
        return (p > 0.5).astype(float)

    def loss(self):
        p = self.predict_proba(self.x)
        eps = 1e-6
        return -np.mean(
            self.y * np.log(p + eps) + (1 - self.y) * np.log(1 - p + eps))

    def gradient(self):
        A = self.design_matrix(self.x)
        p = self.predict_proba(self.x).reshape((-1, 1))
        y = self.y.reshape((-1, 1))
        return A.T @ (p - y)

    def train(self):
        for _ in range(self.epochs):
            self.theta -= self.lr * self.gradient()
```

### 학습

```python
x_train, x_test, y_train, y_test = load_data()

model = LogisticRegression(x_train, y_train)
model.train()

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)
```

## 구현: scikit-learn으로 하는 로지스틱 회귀

비교를 위해 같은 작업을 `sklearn`으로 하면 다음과 같다.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]
```

scikit-learn의 `LogisticRegression`은 기본적으로 L-BFGS(준뉴턴법)를 쓰는데, 헤세행렬을 명시적으로
만들거나 역행렬을 구하지 않고 근사한다. 자료가 작을 때는 `solver='newton-cg'` 옵션이 정확한
뉴턴 단계를 밟으며, 이는 IRLS와 동등하다.

## 연습문제

**연습문제 1.**
로지스틱 회귀의 MLE

관측치 4개 $(x_1, y_1) = (1, 0)$, $(x_2, y_2) = (2, 0)$, $(x_3, y_3) = (3, 1)$,
$(x_4, y_4) = (4, 1)$과 모형 $\log\frac{p}{1-p} = \beta_0 + \beta_1 x$가 주어졌다.

**(a)** 로그가능도를 $\beta_0$과 $\beta_1$의 함수로 쓰라.

**(b)** 닫힌 형태의 해가 없고 수치 최적화가 필요한 이유를 설명하라.

**(c)** 파이썬으로 $\hat{\beta}_0$과 $\hat{\beta}_1$을 구하라. 그 결과를 신뢰할 수 있는가?

??? success "풀이"

    **(a)** $p_i = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_i)}}$이고

    $\ell(\beta_0, \beta_1) = \sum_{i=1}^4 \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$

    $= \log(1-p_1) + \log(1-p_2) + \log p_3 + \log p_4$

    **(b)** 점수방정식 $\frac{\partial \ell}{\partial \beta_0} = 0$,
    $\frac{\partial \ell}{\partial \beta_1} = 0$에는 $\beta$의 비선형함수인 $p_i$가 들어 있어
    대수적으로 풀 수 없다.

    **(c)**

    ```python
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])

    model = LogisticRegression(penalty=None, solver='lbfgs')
    model.fit(X, y)
    print(f"beta0 = {model.intercept_[0]:.4f}")
    print(f"beta1 = {model.coef_[0][0]:.4f}")
    ```

    출력은 $\hat\beta_0 = -47.5544$, $\hat\beta_1 = 18.8115$이다.

    **아니다. 이 결과는 신뢰할 수 없다.** 이 자료는 **완전히 분리 가능**하다.
    $x \le 2$이면 $y = 0$, $x \ge 3$이면 $y = 1$이므로 $x = 2.5$를 기준으로 두 범주가 오차 없이
    갈린다. [가능도 절의 연습문제 5](../logistic_regression/likelihood.md)에서 보았듯이,
    분리 가능한 자료에서는 $\|\boldsymbol{\beta}\| \to \infty$일 때 로그가능도가 상한 0에
    한없이 가까워질 뿐 최댓값에 도달하지 않는다. 즉 **MLE가 존재하지 않는다.**

    위에서 나온 $18.8115$라는 값은 어떤 최대점이 아니라, L-BFGS가 기울기 크기가 허용오차 아래로
    떨어졌다고 판단해 16회 반복 만에 멈춘 지점일 뿐이다. `max_iter`를 100에서 100,000으로
    늘려도 값이 그대로인 것은 수렴했기 때문이 아니라 같은 곳에서 같은 이유로 멈추기 때문이다.
    허용오차를 더 조이면 계수는 계속 커진다.

    **올바른 대응:** 벌점을 명시적으로 넣는다. 예컨대 기본 L2 벌점(`C=1.0`)을 쓰면
    $\hat\beta_0 = -2.3955$, $\hat\beta_1 = 0.9582$로 유한하고 안정적인 값을 얻는다. 다만 이는
    MLE가 아니라 사후최빈값(MAP) 추정치이며, 이 자료로부터 "기울기가 $0.96$이다"라고 말할 수는
    없다. 관측치 4개로 분리된 자료가 알려 주는 것은 문턱이 2와 3 사이 어딘가에 있다는 사실뿐이다.
    $\square$

---

**연습문제 2.**
기울기 $\nabla\ell = A^T(\boldsymbol{\sigma}-\mathbf{y})$의 첫 성분(절편에 대응)을 0으로 놓으면
무엇을 얻는가? 이 항등식이 로지스틱 회귀의 보정에 대해 무엇을 말해 주는지 설명하라.

??? success "풀이"

    $A$의 첫 열은 1로만 이루어져 있으므로 기울기의 첫 성분은

    $$
    \sum_{i=1}^n \bigl(\sigma^{(i)} - y^{(i)}\bigr) = 0
    \quad\Longleftrightarrow\quad
    \frac{1}{n}\sum_{i=1}^n \sigma^{(i)} = \bar{y}
    $$

    이다. 즉 **절편을 포함한** 로지스틱 회귀의 MLE에서는 예측확률의 평균이 관측된 사건 비율과
    정확히 같다.

    이것은 무료로 얻는 보정 조건이다. 모형은 적어도 **전체 수준에서는** 항상 완벽히 보정되어
    있다. 100건을 예측했는데 예측확률의 합이 30이라면 실제 사건도 정확히 30건이다.

    다만 이는 전체 평균에 대한 진술일 뿐이며, **부분집단별 보정**은 전혀 보장하지 않는다.
    예측확률이 0.1 근처인 사례들의 실제 발생률이 0.1인지는 별도로 확인해야 하고, 그것이 보정
    곡선과 브라이어 점수가 필요한 이유다(19.3절). 또한 절편을 빼거나 정칙화를 걸면 이 항등식은
    깨진다. 라쏘·능형 로지스틱 회귀에서 예측확률의 평균이 $\bar{y}$와 어긋나는 것은 버그가
    아니라 벌점의 당연한 결과다. $\square$

---

**연습문제 3.**
경사하강의 학습률 상한을 헤세행렬로부터 유도하라. 위 코드가 `lr=2e-4`라는 작은 값을 쓰는
이유를 설명하라.

??? success "풀이"

    $\nabla^2 \ell = A^T B A$이고 $B$의 대각원소가 $\le 1/4$이므로

    $$
    \nabla^2 \ell \preceq \tfrac{1}{4} A^T A
    \quad\Longrightarrow\quad
    L := \lambda_{\max}(\nabla^2 \ell) \le \tfrac{1}{4}\lambda_{\max}(A^T A)
    $$

    이다. 여기서 $L$은 기울기의 립시츠 상수다. 볼록·$L$-평활 함수에 대한 경사하강은
    $\alpha < 2/L$일 때 수렴하고, $\alpha = 1/L$이 표준적인 선택이다.

    위 코드의 설명변수는 나이(age)로, 척도화하지 않은 수십 단위의 값이다. 절편 열을 포함하면
    $A^T A$의 최대 고유값은 $\sum_i x_i^2$ 수준이므로, 나이가 수십이고 $n$이 수십이면
    $10^4$을 훌쩍 넘는다. 따라서

    $$
    \alpha \lesssim \frac{4}{\lambda_{\max}(A^TA)} \sim 10^{-4}
    $$

    가 되어 `lr=2e-4`라는 값이 나온다. 이 때문에 반복도 100,000회나 필요하다.

    **더 나은 해법은 표준화다.** $x$를 표준화하면 $\lambda_{\max}(A^TA) \approx n$이 되어
    허용 학습률이 세 자릿수 커지고, 같은 정확도에 훨씬 적은 반복으로 도달한다. 뉴턴법과 IRLS가
    이 문제를 겪지 않는 이유도 같다. 헤세행렬의 역을 곱하는 것이 곧 좌표계를 자동으로
    재척도화하는 일이기 때문이다. $\square$

---

**연습문제 4.**
`loss` 메서드는 $\varepsilon = 10^{-6}$을 로그 안에 더하지만 `gradient` 메서드는 그런 보정을
하지 않는다. 이 불일치가 문제를 일으키는가? 답을 정당화하라.

??? success "풀이"

    **최적화 자체에는 문제가 없다.** 갱신식은 `gradient`만 사용하고 `loss`는 진행 상황을
    보고하는 데만 쓰이기 때문이다. 그리고 참 기울기 $A^T(\boldsymbol{\sigma}-\mathbf{y})$는
    $\sigma^{(i)}$가 0이나 1에 가까워져도 수치적으로 안전하다. 로그가 등장하지 않으므로 발산할
    항이 없고, 최악의 경우에도 각 잔차의 절댓값은 1 이하다.

    **그러나 보고되는 숫자는 신뢰할 수 없다.** `loss`가 돌려주는 값은 실제 목적함수가 아니라
    $\varepsilon$으로 잘린 근사값이다. 확신에 찬 오답 하나가 참 손실에 $30$ 이상을 기여할 수
    있는 상황에서도 $\varepsilon$ 판본은 $-\log(10^{-6}) = 13.8$에서 멈춘다. 손실 곡선으로
    수렴을 판정하거나 모형을 비교한다면 이 절단이 결론을 바꿀 수 있다.

    이 비대칭 자체가 하나의 교훈이다. 최적화는 기울기만 필요로 하므로 잘 굴러가지만, 사람이
    보는 진단값은 조용히 왜곡된다. 올바른 수정은 `np.logaddexp`를 이용해 로짓에서 직접 손실을
    계산하는 것이다.

    ```python
    def loss(self):
        A = self.design_matrix(self.x)
        z = (A @ self.theta).reshape((-1,))
        # -log sigma(z) = softplus(-z),  -log(1 - sigma(z)) = softplus(z)
        return np.mean(np.logaddexp(0, z) - self.y * z)
    ```

    $\square$
