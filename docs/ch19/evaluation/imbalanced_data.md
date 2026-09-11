# 불균형 자료 다루기


## 문제: 범주 불균형

현실의 많은 분류 문제에서 범주는 **고르게 나타나지 않는다.** 예를 들면,

- **대출 연체:** 연체율 약 5--20%, 대부분은 상환된다
- **이상거래 탐지:** 부정거래는 보통 전체의 1% 미만
- **질병 진단:** 희귀질환의 유병률은 5% 미만
- **스팸 탐지:** 스팸은 보통 전체 메일의 10--20% 미만

불균형 자료로 학습한 분류기는 **다수 범주를 지나치게 자주 예측하는** 경향이 있어, 정확도는
높지만 드문(양성) 범주를 잘 찾아내지 못한다. 예컨대 상환율이 81%인 대출 자료에서 언제나
"상환"이라고 예측하는 모형은 정확도 81%를 달성하지만 연체는 한 건도 잡지 못한다.

### 유병률 비율 문제

불균형 자료로 학습하면 모형은 경험분포로부터 범주 확률을 배운다. 훈련자료의 유병률이 목표
모집단과 다르면 소수 범주에 대한 예측확률의 보정이 무너진다.

## 설정

아래 코드는 모두 같은 자료를 쓴다. 연체율이 약 19%인 대출자료를 만들고
훈련·검증·검정으로 나눈다.

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n = 4000

# 신용점수·부채비율·소득이 연체 위험을 결정한다
score = rng.normal(0, 1, n)
dti = rng.normal(0, 1, n)
income = rng.normal(0, 1, n)
X = np.column_stack([score, dti, income])

# 절편 -2.0이 연체율을 약 19%로 맞춘다
logit = -2.0 - 1.1 * score + 0.9 * dti - 0.5 * income
p_default = 1 / (1 + np.exp(-logit))
y = (rng.random(n) < p_default).astype(int)   # 1 = 연체, 0 = 상환

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=0, stratify=y_tmp)

print(f"n = {n}, 연체율 = {y.mean():.3f}")
print(f"train {len(y_train)}, val {len(y_val)}, test {len(y_test)}")
```

출력:

```
n = 4000, 연체율 = 0.191
train 2250, val 750, test 1000
```

## 전략 1: 가중을 통한 조정

한 가지 접근은 학습 시 **소수 범주 오류의 비용(가중치)을 키우는** 것이다.

$$
\ell_{\text{weighted}} = -\sum_{i=1}^{n} w_i \left[ y^{(i)} \log \hat{p}_i + (1 - y^{(i)}) \log(1 - \hat{p}_i) \right]
$$

여기서 $w_i$는 관측치 $i$에 부여된 가중치다.

### 범주 가중치

흔한 선택은 **빈도의 역수로 가중**하는 것이다.

$$
w_{\text{minority}} = \frac{1}{p_{\text{minority}}}, \quad w_{\text{majority}} = \frac{1}{p_{\text{majority}}}
$$

또는 간단히 $w_{\text{minority}} = 1$, $w_{\text{majority}} = p_{\text{minority}} / p_{\text{majority}}$
로 둔다.

### 예: 대출 자료

연체율 18.9%, 상환율 81.1%인 어떤 대출자료에서 보고된 결과는 다음과 같다.

| 가중 없음 | 가중 적용 |
|---|---|
| 연체로 예측: 0.98% | 연체로 예측: 61.8% |
| 실제 유병률의 20분의 1 수준 | 실제 유병률에 훨씬 가까움 |

가중이 양성 예측 비율을 얼마나 끌어올리는지는 자료마다 다르지만 방향은 같다.
아래 코드는 위 **설정**의 자료(연체율 19.1%)로 같은 현상을 확인한다.

### 구현

scikit-learn에서는 다음과 같다.

```python
from sklearn.linear_model import LogisticRegression

# Option 1: Automatic balance
model = LogisticRegression(class_weight='balanced')

# Option 2: Custom weights (1 = 연체이므로 소수 범주에 5.3배 가중)
weights = [5.3 if yi == 1 else 1.0 for yi in y_train]
model.fit(X_train, y_train, sample_weight=weights)

print("가중 없음 예측 연체율:",
      LogisticRegression().fit(X_train, y_train).predict(X_val).mean().round(4))
print("가중 적용 예측 연체율:", model.predict(X_val).mean().round(4))
```

출력:

```
가중 없음 예측 연체율: 0.096
가중 적용 예측 연체율: 0.776
```

**장점:**

- 구현이 간단하다
- 계산이 효율적이다
- 원래 자료 크기를 보존한다

**단점:**

- 적절한 가중비를 골라야 한다
- **확률 추정의 보정을 무너뜨린다.** 아래 경고와 연습문제 2를 보라

## 전략 2: 재표집

### 과소표집

**과소표집**은 다수 범주의 관측치를 제거해 자료를 균형 있게 만든다.

```
Original:   81,105 paid off + 18,895 default
Undersampled: 18,895 paid off + 18,895 default
```

**장점:**

- 간단하고 범주가 완벽히 균형을 이룬다
- 학습 시간이 줄어든다

**단점:**

- **정보 손실:** 다수 범주 자료를 버린다
- 분산이 커져 추정이 덜 안정적이다
- 확률 추정이 50 대 50 쪽으로 치우친다

### 과대표집

**과대표집**은 소수 범주의 관측치를 복제한다.

```
Original:   81,105 paid off + 18,895 default
Oversampled: 81,105 paid off + 81,105 default (via replication)
```

**단점:**

- 중복된 복사본이 생긴다
- 과적합으로 이어질 수 있다
- 자료 크기가 부풀어 오른다

## 전략 3: 합성 소수범주 과대표집(SMOTE)

**SMOTE**는 이웃한 소수 범주 관측치 사이를 내삽하여 **합성 표본**을 만든다. 정확한 복제 대신
가까운 소수 범주 사례를 잇는 선분 위에 새 점을 만든다.

### SMOTE의 작동 방식

각 소수 범주 표본 $x_i$에 대해,

1. 소수 범주 안에서 $k$개의 최근접 이웃을 찾는다(보통 $k=5$).
2. 이웃 하나 $x_{\text{neighbor}}$를 무작위로 고른다.
3. 합성점을 만든다.

   $$x_{\text{synthetic}} = x_i + \lambda (x_{\text{neighbor}} - x_i)$$

   여기서 $\lambda \in [0, 1]$은 무작위다.

### 예: SMOTE를 적용한 대출 자료

```python
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE(random_state=0).fit_resample(X_train, y_train)
# Result: 50-50 split of defaults and paid-offs (synthetic defaults added)

model = LogisticRegression()
model.fit(X_resampled, y_resampled)

print(f"원자료:   n = {len(y_train)}, 연체율 = {y_train.mean():.3f}")
print(f"SMOTE 후: n = {len(y_resampled)}, 연체율 = {y_resampled.mean():.3f}")
```

출력:

```
원자료:   n = 2250, 연체율 = 0.191
SMOTE 후: n = 3642, 연체율 = 0.500
```

**장점:**

- 내삽으로 그럴듯한 합성 표본을 만든다
- 정확한 복제를 피하므로 순진한 과대표집보다 과적합이 덜하다
- 국소적인 이웃 구조를 보존한다
- 실무에서 널리 쓰인다

**단점:**

- 단순 재표집보다 복잡하다
- 이웃을 찾는 계산 부담이 있다
- 연속형 특성을 전제한다(이산형·범주형은 변형이 필요하다)
- **다른 재표집과 마찬가지로 보정을 무너뜨린다**

### SMOTE의 변형

- **BorderlineSMOTE:** 결정경계 근처의 소수 범주 표본에 집중한다
- **ADASYN:** 학습하기 어려운 소수 범주 사례에 더 많은 표본을 만든다
- **SVMSMOTE:** SVM 결정경계를 이용해 합성 표본 생성을 유도한다

```python
from imblearn.over_sampling import BorderlineSMOTE, ADASYN

# BorderlineSMOTE
X_bl, y_bl = BorderlineSMOTE(random_state=0).fit_resample(X_train, y_train)

# ADASYN
X_ad, y_ad = ADASYN(random_state=0).fit_resample(X_train, y_train)

print(f"BorderlineSMOTE: n = {len(y_bl)}, 연체율 = {y_bl.mean():.3f}")
print(f"ADASYN:          n = {len(y_ad)}, 연체율 = {y_ad.mean():.3f}")
```

출력:

```
BorderlineSMOTE: n = 3642, 연체율 = 0.500
ADASYN:          n = 3635, 연체율 = 0.499
```

## 전략 4: 문턱 조정

자료나 손실함수를 바꾸는 대신, 사후적으로 **결정 문턱**을 조정해 원하는 **작동점**에 맞춘다.

- 불균형 자료에서 기본 문턱 0.5는 최적이 아닌 경우가 많다
- ROC 곡선이나 PR 곡선으로 원하는 정밀도-재현율 절충을 주는 문턱을 고른다
- 업무상의 비용과 제약에 근거해 조정한다

### 예

훈련자료의 연체율은 20%인데 배치 대상의 실제 연체율이 10%라면, 모형은 확률을 체계적으로
과대평가하게 된다. 이때는 문턱을 **높여** 양성 예측을 줄이거나, 더 나은 방법으로 절편을
보정한다(연습문제 1).

!!! danger "재표집과 가중은 보정을 무너뜨린다"
    이 절의 전략 1--3은 모두 모형이 학습하는 유병률을 인위적으로 바꾼다. 그 결과 예측확률이
    더 이상 실제 사건 확률의 추정치가 아니게 된다. 아래 연습문제 3에서 보듯, 균형 조정은
    **AUC를 전혀 개선하지 않으면서** 브라이어 점수를 두 배 이상 악화시킬 수 있다.

    확률 자체가 필요한 응용(신용평가, 의학적 위험 예측, 기대비용 계산)에서는 재표집을 하지
    않거나, 했다면 반드시 절편을 되돌려 보정해야 한다. 이 점은 임상 예측모형 문헌에서 특히
    강조되어 왔다.

## 비교와 권장

| 방법 | 단순함 | 자료 효율 | 계산 | 보정 | 적합한 경우 |
|---|---|---|---|---|---|
| **조정 없음** | 높음 | 높음 | 낮음 | **좋음** | 확률이 필요한 대부분의 경우 |
| **가중** | 높음 | 높음 | 낮음 | 나쁨(절편 보정 필요) | 최적화가 소수 범주를 무시할 때 |
| **과소표집** | 높음 | 낮음 | 낮음 | 나쁨(절편 보정 필요) | 자료가 매우 크고 불균형이 극단적일 때 |
| **과대표집** | 높음 | 중간 | 낮음 | 나쁨 | 과적합 위험이 있어 권장하지 않음 |
| **SMOTE** | 중간 | 높음 | 중간 | 나쁨 | 순위 성능만 필요할 때 |
| **문턱 조율** | 높음 | 높음 | 낮음 | **영향 없음** | 대부분의 경우 첫 번째 선택 |

## 권장 절차

불균형 자료에 대한 견고한 기본 전략은 다음과 같다.

1. **원래 불균형 그대로 모형을 적합한다.** 로지스틱 회귀의 MLE는 불균형 자체 때문에 편향되지
   않는다. 문제는 추정이 아니라 기본 문턱 0.5에 있다.
2. **AUC와 보정을 먼저 확인한다.** 순위가 나쁘면 재표집으로 고쳐지지 않는다. 특성이나 모형을
   바꿔야 한다.
3. **검증자료에서 문턱을 조율한다.** 이때 검증자료는 실제 범주 분포를 그대로 유지해야 한다.
   비용을 알면 $t^* = C_{FP}/(C_{FP}+C_{FN})$을 쓴다.
4. **소수 범주 사례가 절대적으로 너무 적어**(예: 수십 건) 최적화가 불안정할 때에만 가중이나
   재표집을 고려한다. 그 경우에도 예측확률을 쓰기 전에 절편을 보정한다.
5. **원래 불균형을 유지한 검정자료에서** 정밀도-재현율 곡선과 ROC 곡선으로 평가한다.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

# Step 1: fit on the original, imbalanced data
model = LogisticRegression().fit(X_train, y_train)

# Step 2: check ranking and calibration on a validation set
p_val = model.predict_proba(X_val)[:, 1]
print("AUC  :", roc_auc_score(y_val, p_val))
print("Brier:", brier_score_loss(y_val, p_val))

# Step 3: choose the operating point from costs
c_fp, c_fn = 50.0, 1000.0
threshold = c_fp / (c_fp + c_fn)

# Step 4: evaluate once on the test set
y_test_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
```

출력:

```
AUC  : 0.8369258418681812
Brier: 0.111258445223284
```

## 핵심 요약

범주 불균형은 명시적으로 다뤄야 한다. 정확도를 높이는 것만으로는 부족하며, **재현율(드문 사건을
잡아내기)**과 **보정(현실적인 확률 추정)**에 초점을 맞추어야 한다. 그리고 대부분의 경우
불균형에 대한 올바른 대응은 자료를 바꾸는 것이 아니라 **문턱을 바꾸는 것**이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
훈련표본의 양성 비율이 $\tau$이고 모집단의 실제 유병률이 $\pi$일 때, 로지스틱 회귀에서
**기울기 계수는 편향되지 않고 절편만 이동함**을 설명하라. 절편 보정식을 유도하라.

</div>

??? success "풀이"

    범주에 따라 표집률을 달리하는 것을 **결과 기반 표집**(사례-대조 표집)이라 한다.
    양성을 확률 $s_1$로, 음성을 확률 $s_0$으로 뽑았다고 하자. 표본에 포함되었다는 사건을
    $S$라 하면 베이즈 정리에 의해

    $$
    \frac{P(Y=1 \mid \mathbf{x}, S)}{P(Y=0 \mid \mathbf{x}, S)}
    = \frac{P(Y=1\mid\mathbf{x})}{P(Y=0\mid\mathbf{x})} \cdot \frac{s_1}{s_0}
    $$

    이다. 표집이 $\mathbf{x}$가 아니라 $y$에만 의존하기 때문이다. 양변에 로그를 취하면

    $$
    \operatorname{logit} P(Y=1\mid\mathbf{x}, S)
    = \beta_0 + \mathbf{x}^T\boldsymbol{\beta} + \log\frac{s_1}{s_0}
    $$

    을 얻는다. 즉 **$\mathbf{x}$에 붙는 계수는 그대로이고 절편만 $\log(s_1/s_0)$만큼
    이동한다.** 이것이 프렌티스-파이크 결과이며, 로지스틱 회귀가 사례-대조 연구에서 특별히
    유용한 이유다.

    표본 비율 $\tau$와 모집단 유병률 $\pi$로 다시 쓰면
    $\dfrac{s_1}{s_0} = \dfrac{\tau/\pi}{(1-\tau)/(1-\pi)}$이므로, 보정된 절편은

    $$
    \hat\beta_0^{\text{corrected}} = \hat\beta_0 - \log\!\left(\frac{1-\pi}{\pi}\cdot\frac{\tau}{1-\tau}\right)
    $$

    이다. 50 대 50으로 균형을 맞춘 경우 $\tau = 0.5$이므로 보정항은
    $\log\frac{1-\pi}{\pi}$로 단순해진다. $\square$

<div class="drillbox" markdown>

**연습문제 2.**
`class_weight='balanced'`가 절편 이동과 (모형이 옳게 지정되었을 때) 동등함을 설명하라.
따라서 AUC에는 어떤 영향을 주는가?

</div>

??? success "풀이"

    양성에 $w_1$, 음성에 $w_0$의 가중치를 주는 것은 양성을 $w_1$번, 음성을 $w_0$번 복제한
    자료에 가중 없이 적합하는 것과 같다(가중 로그가능도가 정확히 같기 때문이다). 그런데 복제는
    $y$에만 의존하는 표집이므로 연습문제 1이 그대로 적용된다. 절편이
    $\log(w_1/w_0)$만큼 이동하고 기울기는 변하지 않는다.

    `balanced`는 $w_c \propto n/(2 n_c)$로 두므로 $w_1/w_0 = n_0/n_1 = (1-\tau)/\tau$이고,
    절편은 $\log\frac{1-\tau}{\tau}$만큼 올라간다. $\tau = 0.1$이면 $\log 9 = 2.197$이다.

    **AUC에 대한 영향은 없다.** 절편만 바뀌면 모든 관측치의 로짓이 같은 상수만큼 이동하므로
    순위가 완전히 보존된다. 시그모이드는 단조이므로 확률의 순위도 그대로다.

    이 논증은 **모형이 옳게 지정되었을 때** 성립한다는 점이 중요하다. 모형이 잘못 지정되어
    있으면 가중이 적합의 무게중심을 옮기므로 기울기까지 달라지고, AUC도 (조금) 변할 수 있다.
    연습문제 3의 모의실험에서 기울기가 $1.511$에서 $1.508$로 거의 변하지 않는 것은 그곳의 모형이
    참이기 때문이다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
유병률이 약 10%인 자료를 만들어, 조정 없는 적합·`balanced` 가중·과소표집·절편 보정의 네 가지에
대해 AUC와 브라이어 점수를 비교하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss

    rng = np.random.default_rng(11)
    n = 20000
    x = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(-3.0 + 1.5 * x)))   # true beta0 = -3, beta1 = 1.5
    y = (rng.random(n) < p).astype(int)

    ntr = n // 2
    Xtr, ytr = x[:ntr].reshape(-1, 1), y[:ntr]
    Xte, yte = x[ntr:].reshape(-1, 1), y[ntr:]

    def report(name, b0, b1):
        z = b0 + b1 * Xte.ravel()
        ph = 1 / (1 + np.exp(-z))
        print(f"{name:10s} b0={b0:7.4f} b1={b1:.4f} "
              f"AUC={roc_auc_score(yte, ph):.4f} "
              f"Brier={brier_score_loss(yte, ph):.5f} "
              f"mean_p={ph.mean():.4f}")

    m = LogisticRegression(penalty=None).fit(Xtr, ytr)
    report("plain", m.intercept_[0], m.coef_[0][0])

    mb = LogisticRegression(penalty=None, class_weight='balanced').fit(Xtr, ytr)
    report("balanced", mb.intercept_[0], mb.coef_[0][0])

    i1 = np.where(ytr == 1)[0]
    i0 = rng.choice(np.where(ytr == 0)[0], size=len(i1), replace=False)
    ii = np.concatenate([i0, i1])
    mu = LogisticRegression(penalty=None).fit(Xtr[ii], ytr[ii])
    report("under", mu.intercept_[0], mu.coef_[0][0])

    tau, pi = 0.5, ytr.mean()
    b0_corr = mu.intercept_[0] - np.log(((1 - pi) / pi) * (tau / (1 - tau)))
    report("corrected", b0_corr, mu.coef_[0][0])
    ```

    출력:

    ```
    plain      b0=-3.0714 b1=1.5111 AUC=0.8295 Brier=0.07247 mean_p=0.0910
    balanced   b0=-0.8018 b1=1.5079 AUC=0.8295 Brier=0.16354 mean_p=0.3589
    under      b0=-0.7882 b1=1.4686 AUC=0.8295 Brier=0.16282 mean_p=0.3596
    corrected  b0=-3.0598 b1=1.4686 AUC=0.8295 Brier=0.07258 mean_p=0.0892
    ```

    훈련자료의 유병률은 $0.0961$이고 결과는 다음과 같다.

    | 방법 | $\hat\beta_0$ | $\hat\beta_1$ | AUC | 브라이어 | 평균 $\hat p$ |
    |---|---|---|---|---|---|
    | 조정 없음 | $-3.0714$ | $1.5111$ | $0.8295$ | $\mathbf{0.07247}$ | $0.0910$ |
    | balanced 가중 | $-0.8018$ | $1.5079$ | $0.8295$ | $0.16354$ | $0.3589$ |
    | 과소표집 | $-0.7882$ | $1.4686$ | $0.8295$ | $0.16282$ | $0.3596$ |
    | 절편 보정 | $-3.0598$ | $1.4686$ | $0.8295$ | $\mathbf{0.07258}$ | $0.0892$ |

    (검정자료의 실제 양성 비율은 $0.0988$이다.)

    읽어야 할 점이 세 가지다.

    1. **AUC는 네 경우 모두 $0.8295$로 완전히 같다.** 균형 조정은 순위 성능을 조금도 개선하지
       않는다. 연습문제 2에서 예측한 그대로다.
    2. **브라이어 점수는 2.3배 나빠진다**($0.0725 \to 0.1635$). 평균 예측확률이 실제
       유병률 $0.099$에서 $0.359$로 부풀어 오르는 것이 원인이다. 이 모형이 "연체 확률 36%"라고
       말하면 그것은 완전한 허구다.
    3. **절편 보정이 이를 되돌린다.** 보정된 절편 $-3.0598$은 참값 $-3.0$과 조정 없는 추정치
       $-3.0714$ 양쪽에 가깝고, 브라이어 점수도 $0.0726$으로 회복된다. 기울기 $1.4686$은
       과소표집으로 자료의 절반 이상을 버렸기 때문에 조금 덜 정확하다.

    **결론:** 균형 조정으로 얻는 것은 없고 잃는 것은 보정이다. 굳이 해야 한다면 절편을 반드시
    되돌려라. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
훈련·검정 분할 **전에** SMOTE를 적용하면 왜 자료 누설이 되는지 설명하라.

</div>

??? success "풀이"

    SMOTE는 소수 범주 관측치 쌍 사이를 내삽해 새 점을 만든다. 분할 전에 적용하면, 원본 관측치
    $x_i$와 $x_j$로 만든 합성점이 훈련자료에 들어가고 $x_i$ 자체는 검정자료에 들어갈 수 있다.

    그러면 모형은 검정 관측치의 좌표를 **이미 훈련 과정에서 본 셈**이 된다. 합성점
    $x_i + \lambda(x_j - x_i)$는 $\lambda$가 작으면 $x_i$와 거의 같은 위치다. 사실상 검정
    관측치가 훈련자료에 흐릿한 형태로 복제된 것이며, 검정 성능이 낙관적으로 부풀려진다.

    $k$-최근접이웃이나 트리 모형처럼 국소 구조에 민감한 모형에서 효과가 특히 크다. 문헌에는
    이 실수 때문에 정확도가 0.75에서 0.95로 뛴 사례가 흔하다.

    **올바른 순서:**

    1. 먼저 훈련/검증/검정으로 나눈다.
    2. **훈련 부분에만** SMOTE를 적용한다.
    3. 검증자료와 검정자료는 원래 범주 분포를 그대로 둔다.

    교차검증에서는 SMOTE가 반드시 `Pipeline` 안에 들어가야 각 겹의 훈련 부분에서만 적합된다.
    `imblearn.pipeline.Pipeline`이 이를 올바르게 처리한다(scikit-learn의 기본 `Pipeline`은
    표본 수를 바꾸는 단계를 다루지 못한다). $\square$

<div class="drillbox" markdown>

**연습문제 5.**
불균형 자료를 만나면 어떤 순서로 대응해야 하는가? "SMOTE로 균형을 맞추고 나서 학습한다"가
좋은 기본값이 **아닌** 이유를 설명하라.

</div>

??? success "풀이"

    **권장 순서:**

    1. 원래 불균형 그대로 적합한다. 로지스틱 회귀의 MLE는 불균형 때문에 편향되지 않는다.
       $n$이 충분하면 $\hat\beta$는 일치추정량이다.
    2. AUC로 순위 성능을, 보정 곡선과 브라이어 점수로 확률의 질을 확인한다.
    3. 비용에 근거해 문턱을 정한다. 이것이 대부분의 "불균형 문제"의 실제 해법이다.
    4. 그래도 부족하면 특성을 개선하거나 더 유연한 모형을 쓴다.
    5. 소수 범주의 **절대 개수**가 너무 적을 때만(사건당 변수 10개 미만 같은 상황) 벌점가능도나
       파스 보정을 고려한다. 이것은 불균형이 아니라 **표본 부족** 문제다.

    **"먼저 SMOTE"가 나쁜 기본값인 이유:**

    - **문제를 오진한다.** 불균형 자료에서 모형이 나빠 보이는 주된 이유는 추정이 아니라
      문턱 0.5가 부적절하기 때문이다. 문턱을 고치면 재표집 없이 해결된다.
    - **보정을 파괴한다.** 연습문제 3에서 브라이어 점수가 2.3배 나빠졌다. 확률이 필요한
      응용에서는 치명적이다.
    - **순위를 개선하지 않는다.** AUC가 그대로였다. 즉 실제로 얻는 정보가 없다.
    - **새로운 위험을 들여온다.** 합성점이 범주 경계를 넘거나(특히 범주가 겹칠 때) 범주형
      특성에서 의미 없는 값을 만들 수 있다.
    - **누설의 기회를 만든다.** 연습문제 4.

    임상 예측모형 문헌에서는 이 점이 반복해서 확인되었다. 불균형 보정은 판별력을 개선하지 않으면서
    보정만 체계적으로 악화시킨다. **불균형은 자료의 문제가 아니라 결정 규칙의 문제로 다루라.**
    $\square$
