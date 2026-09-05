# 지도학습

지도학습은 레이블이 붙은 입력–출력 쌍으로 모형을 훈련시켜, 레이블이 없는 새 입력에 대해 출력을 예측하게 한다. 목표 — *$\mathbf{x}$로부터 $y$를 예측한다* — 가 명확하고, 손실을 관측할 수 있으며, 성공 기준이 모호하지 않기 때문에 현대 자료 분석에서 지배적인 패러다임이다. 신용평가, 수요예측, 부정 탐지, 의료 진단, 기계 번역, 그리고 대부분의 이미지 인식 시스템이 내부적으로는 지도학습기다.

## 정의

$(\mathbf{X}, Y)$가 $\mathbf{X} \in \mathcal{X} \subseteq \mathbb{R}^p$, $Y \in \mathcal{Y}$인 결합분포를 갖는 확률변수라고 하자. 이 결합분포에서 i.i.d.로 뽑은 훈련 표본 $\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$을 관측하고, **기대손실**(또는 **위험**)

$$
R(f) = \mathbb{E}_{(\mathbf{X}, Y)}\!\left[L(Y, f(\mathbf{X}))\right]
$$

을 최소화하는 함수 $\hat{f} : \mathcal{X} \to \mathcal{Y}$를 찾는다. 여기서 $L : \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0}$은 문제에 따라 정해지는 손실함수다. 두 가지 표준 과제는 $Y$의 유형에 따라 갈린다.

- **회귀**: $\mathcal{Y} = \mathbb{R}$. 표준 손실은 제곱오차 $L(y, \hat{y}) = (y - \hat{y})^2$이며, 그 위험 최소화 함수는 조건부 평균 $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$이다.
- **분류**: $\mathcal{Y} = \{1, \ldots, K\}$. 표준 손실은 0–1 손실 $L(y, \hat{y}) = \mathbf{1}\{y \ne \hat{y}\}$이며, 그 위험 최소화 함수는 **베이즈 분류기** $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$이다.

$R(f)$는 알려지지 않은 결합분포에 의존하므로, 이를 **경험위험**

$$
\hat{R}_n(f) = \frac{1}{n} \sum_{i=1}^n L(y_i, f(\mathbf{x}_i))
$$

으로 대체하고 선택한 가설 공간 $\mathcal{F}$ 위에서 최소화한다. 이 절차를 **경험위험 최소화**(ERM)라 한다.

## 설명

### 편향–분산 분해

지도학습의 근본적인 긴장은 편향–분산 분해로 포착된다. 고정된 점 $\mathbf{x}_0$에서 제곱오차 회귀에 대해,

$$
\mathbb{E}\!\left[(Y - \hat{f}(\mathbf{x}_0))^2\right] = \underbrace{\sigma^2}_{\text{irreducible}} + \underbrace{\left(\mathbb{E}[\hat{f}(\mathbf{x}_0)] - f^*(\mathbf{x}_0)\right)^2}_{\text{bias}^2} + \underbrace{\mathrm{Var}(\hat{f}(\mathbf{x}_0))}_{\text{variance}}
$$

단순한 모형(선형회귀, 얕은 트리)은 편향이 크고 분산이 작다. 유연한 모형(심층 신경망, 큰 앙상블)은 편향이 작고 분산이 크다. 지도학습의 기예는 주어진 표본 크기에서 편향과 분산이 유리하게 맞바뀌도록 모형 복잡도를 고르는 데 있다.

### 작업 흐름

1. **자료 분할**: 훈련 / 검증 / 시험(예: 60/20/20), 또는 훈련+검증 부분에 $k$-겹 교차검증 적용.
2. **모형 족 $\mathcal{F}$ 선택**: 선형, 트리 기반, 커널 기반, 신경망, 앙상블.
3. **ERM으로 적합**: 경험위험을 최소화하되, 복잡도를 제어하기 위해 흔히 정칙화 벌점 $\lambda \cdot \Omega(f)$를 더한다.
4. **초매개변수 조정**(벌점 강도, 트리 깊이, 학습률): 검증 집합에서 수행.
5. **시험 집합에서 평가** — 모든 모형 결정을 확정한 뒤 정확히 한 번만 사용한다.
6. **배포 후 모니터링**: 분포 표류를 감시한다.

### 지도학습이 사촌들보다 쉬운 이유

비지도학습과 강화학습에 비해 지도학습 문제는 세 가지 구조적 이점을 누린다.

- **관측 가능한 손실**: 모든 예측에 비교할 참값 레이블이 있으므로 $\hat{R}_n(f)$를 계산할 수 있다.
- **독립적인 지도 신호**: 각 $(\mathbf{x}_i, y_i)$가 자체적인 학습 신호를 제공하며, 강화학습에서와 같은 시간적 공로 배분 문제가 없다.
- **정직한 평가**: 훈련 중 정보가 새지 않는 한, 떼어놓은 시험 집합이 일반화 오차의 불편추정값을 준다.

바로 이 이점들 때문에, 시험 분포가 훈련 분포와 다르거나(공변량 이동, 레이블 이동) 레이블 자체가 체계적으로 편향된 경우에는 지도학습이 소리 없이 실패한다.

## 예제

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # logistic function

np.random.seed(42)
n = 1000

# Simulate binary classification: loan default
income = np.random.normal(60, 20, n).clip(10)
dti = np.random.normal(0.3, 0.15, n).clip(0.01, 1.0)
log_odds = -3 + 0.01 * (50 - income) + 5 * (dti - 0.3)
prob = expit(log_odds)
default = np.random.binomial(1, prob)

# Train/test split
train, test = np.arange(700), np.arange(700, n)
X = np.column_stack([np.ones(n), income, dti])

# Fit logistic regression by minimizing negative log-likelihood
def neg_log_lik(beta):
    z = X[train] @ beta
    return -np.sum(default[train] * z - np.log1p(np.exp(z)))

result = minimize(neg_log_lik, np.zeros(3), method="BFGS")
beta_hat = result.x

probs_test = expit(X[test] @ beta_hat)
preds = (probs_test > 0.5).astype(int)
accuracy = np.mean(preds == default[test])
print(f"Test accuracy:        {accuracy:.3f}")
print(f"Default rate (test):  {default[test].mean():.3f}")
print(f"Coefficients:         {beta_hat.round(4)}")
```

클래스가 불균형할 때 정확도만 보면 오도된다. 대출의 10%만 부도가 난다면 "부도 없음"이라고만 답하는 모형도 정확도가 90%가 될 수 있다. 실제 평가에는 정밀도, 재현율, ROC-AUC, 또는 거짓양성과 거짓음성의 비대칭적 비용을 반영한 비용가중 손실이 필요하다.

## 연습문제

**연습문제 1.**
다음 각 과제를 지도학습, 비지도학습, 강화학습으로 분류하고, 지도학습이라면 회귀인지 분류인지 밝혀라.

**(a)** 과거 가격과 거래량 자료로 내일의 주식 종가를 예측하기.
**(b)** 미리 정해진 범주 없이 구매 행동에 따라 고객을 세그먼트로 묶기.
**(c)** 출구에 도달하면 보상하고 충돌하면 벌점을 주어 로봇에게 미로 탐색을 가르치기.
**(d)** 레이블이 붙은 받은편지함으로 이메일을 스팸인지 아닌지 분류하는 모형 훈련하기.
**(e)** 레이블이 붙은 부정 사례가 전혀 없을 때 이상 신용카드 거래 탐지하기.
**(f)** 어떤 환자가 앞으로 1년간 재입원할 횟수 추정하기.

??? success "연습문제 1 풀이"
    (a) 지도학습 — 회귀(연속형 목표).
    (b) 비지도학습 — 군집화, 레이블 없음.
    (c) 강화학습 — 보상에 이끌리는 순차적 의사결정.
    (d) 지도학습 — 이진 분류.
    (e) 비지도학습 — 레이블 없는 이상치 탐지(레이블이 일부 있다면 준지도학습일 수 있다).
    (f) 지도학습 — 계수형 회귀(포아송 회귀가 자연스러운 선택이다).

---

**연습문제 2.**
$L(y, \hat{y}) = (y - \hat{y})^2$을 제곱오차 손실이라 하자. 모든 가측 함수 $f$ 위에서 위험 $R(f) = \mathbb{E}[L(Y, f(\mathbf{X}))]$을 최소화하는 함수 $f^*$가 $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$임을 보여라.

??? success "연습문제 2 풀이"
    $\mathbf{X} = \mathbf{x}$로 조건을 걸고 점별로 최소화한다. 고정된 $\mathbf{x}$에 대해 $\mathbb{E}[(Y - c)^2 \mid \mathbf{X} = \mathbf{x}]$을 최소화하는 $c \in \mathbb{R}$을 찾는다. 전개하면

    $$
    \mathbb{E}[(Y - c)^2 \mid \mathbf{X} = \mathbf{x}] = \mathrm{Var}(Y \mid \mathbf{X} = \mathbf{x}) + (\mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}] - c)^2
    $$

    첫 항은 $c$에 의존하지 않고, 둘째 항은 $c = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$에서 최소가 된다. 이것이 모든 $\mathbf{x}$에 대해 성립하므로 전역 최소화 함수는 $f^*(\mathbf{x}) = \mathbb{E}[Y \mid \mathbf{X} = \mathbf{x}]$이다. $\square$

---

**연습문제 3.**
$K$-클래스 분류에서 0–1 손실에 대해, 베이즈 분류기 $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$가 기대 오분류율을 최소화함을 보여라.

??? success "연습문제 3 풀이"
    임의의 분류기 $f$에 대해

    $$
    \mathbb{E}[\mathbf{1}\{Y \ne f(\mathbf{X})\} \mid \mathbf{X} = \mathbf{x}] = 1 - P(Y = f(\mathbf{x}) \mid \mathbf{X} = \mathbf{x})
    $$

    이다. 이를 점별로 최소화하려면 $f(\mathbf{x}) \in \{1, \ldots, K\}$ 위에서 $P(Y = f(\mathbf{x}) \mid \mathbf{X} = \mathbf{x})$를 최대화해야 하며, 그 결과 $f^*(\mathbf{x}) = \arg\max_k P(Y = k \mid \mathbf{X} = \mathbf{x})$가 선택된다. $\mathbf{X}$에 대해 적분하면 전역 최솟값을 얻는다. $\square$

---

**연습문제 4.**
어떤 은행이 2015–2019년 신청자 자료로 신용평가 모형을 학습시켜 떼어놓은 2019년 표본에서 92%의 시험 정확도를 얻었다. 2024년에 배포하자 정확도가 71%에 그쳤다. 이 하락을 설명할 수 있는 서로 다른 기제 세 가지와 각각에 대한 진단 방법을 제시하라.

??? success "연습문제 4 풀이"
    - **공변량 이동**: $\mathbf{X}$의 분포가 변했다(예: 팬데믹 이후 소득 분포). 진단: KS 검정이나 PSI(모집단 안정성 지수)로 훈련 자료와 2024년 자료의 특성 주변분포를 비교한다.
    - **레이블 이동 / 개념 표류**: 조건부 $P(Y \mid \mathbf{X})$가 변했다(부도 행동이 새로운 경제 여건에 반응한다). 진단: 최근 레이블 자료로 모형을 다시 적합해 계수나 변수 중요도를 비교한다.
    - **훈련 중 자료 누출**: 배포 시점에는 쓸 수 없는 특성이 훈련에 사용되었다(예: 실제로는 부도 이후에 기록된 "현재 잔액"). 진단: 특성 파이프라인을 감사하고 결정 시점 이전의 특성만으로 재학습한다.

    그 밖에 타당한 기제로는 피드백 루프(모형 자신의 결정이 신청자 구성을 바꿈), 원래 훈련 집합의 표집편향, 레이블 품질 변화("부도"의 정의가 개정됨) 등이 있다.

---

**연습문제 5.**
같은 자료에 두 회귀 모형을 적합했다.
- 모형 A: 선형회귀, 훈련 MSE $= 12$, 시험 MSE $= 15$.
- 모형 B: 심층 신경망, 훈련 MSE $= 2$, 시험 MSE $= 25$.

편향–분산 분해를 사용해 각 모형의 특징을 규정하라. 어느 것을 배포하겠으며, 다음으로 무엇을 시도하겠는가?

??? success "연습문제 5 풀이"
    모형 A는 훈련 오차가 크고 시험 오차는 그보다 조금 클 뿐이다. **과소적합**(높은 편향, 낮은 분산) 상태다. 가설 공간이 지나치게 제한적이다.

    모형 B는 시험 집합보다 훈련 집합을 훨씬 잘 적합한다. **과적합**(낮은 편향, 높은 분산) 상태이며 잡음을 외운 것이다.

    모형 A를 배포한다. MSE 15는 일반화되는 값인 반면, 모형 B의 훈련 MSE 2는 허상이다. 다음 단계로는 중간 정도 유연성의 모형(그래디언트 부스팅 트리, 정칙화된 신경망, 커널 능형회귀)을 시도하거나, 모형 B에 정칙화·조기 종료·자료 증강을 적용해 분산을 줄인다.

---

**연습문제 6.**
모형을 자신의 훈련 자료로 평가하면 왜 위험을 지나치게 낙관적으로 추정하게 되는가? $\hat{f}$가 $\hat{R}_n$을 최소화하도록 선택되었을 때 $\hat{R}_n(\hat{f})$와 $R(\hat{f})$의 관계로 답을 형식화하라.

??? success "연습문제 6 풀이"
    훈련 자료는 두 역할을 한다. 추정량 $\hat{f}$를 결정하고($\mathcal{F}$ 위에서 $\hat{R}_n$을 최소화하도록 선택된다), 그다음 $\hat{R}_n(\hat{f})$를 계산하는 데 다시 쓰인다. $\hat{f}$가 $\hat{R}_n$을 가능한 한 작게 만들도록 선택되었으므로 경험위험은 참된 위험을 과소평가한다.

    $$
    \mathbb{E}\!\left[\hat{R}_n(\hat{f})\right] \le \mathbb{E}\!\left[R(\hat{f})\right]
    $$

    등호는 $\mathcal{F}$가 함수 하나만 포함할 때만 성립한다. 이 격차가 훈련오차의 **낙관성**이며, $\mathcal{F}$의 실효 복잡도가 커질수록 커진다. 떼어놓은 시험 집합은 이 의존을 끊는다. 시험 자료는 $\hat{f}$를 고르는 데 쓰이지 않았으므로 $\hat{R}_{\text{test}}(\hat{f})$는 $R(\hat{f})$의 불편추정값이다. 정직한 평가를 위해 훈련/시험 분할(또는 교차검증)이 타협할 수 없는 이유가 이것이다. $\square$
