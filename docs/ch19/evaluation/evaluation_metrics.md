# ROC 곡선과 평가지표 구현


## 개요

이 절에서는 이항 분류의 핵심 평가지표를 NumPy만으로 처음부터 구현하고 해석한다. 혼동행렬을
만들고, 정밀도·재현율·$F_1$ 점수를 계산하며, 문턱을 훑어 ROC 곡선을 구성하고, 사다리꼴 공식으로
AUC를 계산한다.

## 모의 분류 문제

양성 120개와 음성 180개로 이루어진 두 범주 자료를 만든다. 예측 점수는 겹치는 정규분포에서
뽑아 두 범주가 완전히 분리되지 않게 한다.

$$
s_i \sim
\begin{cases}
N(0.65,\; 0.25^2) & \text{if } y_i = 1 \\
N(0.35,\; 0.25^2) & \text{if } y_i = 0
\end{cases}
$$

```python
import numpy as np

np.random.seed(42)
n = 300
y_true = np.concatenate([np.ones(120), np.zeros(180)])
scores = np.concatenate([
    np.random.normal(0.65, 0.25, 120),
    np.random.normal(0.35, 0.25, 180)
])
scores = np.clip(scores, 0, 1)
```

## 혼동행렬

주어진 문턱 $\tau$에서 $s_i \geq \tau$이면 양성($\hat{y} = 1$), 아니면 음성으로 분류한다.
혼동행렬의 네 원소는 다음과 같다.

| | 음성으로 예측 | 양성으로 예측 |
|---|---|---|
| **실제 음성** | TN | FP |
| **실제 양성** | FN | TP |

```python
def confusion_matrix(y_true, y_pred):
    """Compute 2x2 confusion matrix [TN, FP; FN, TP]."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

y_pred = (scores >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)
```

$\tau = 0.5$에서 TN $= 129$, FP $= 51$, FN $= 30$, TP $= 90$이다.

## 정밀도, 재현율, F1 점수

혼동행렬에서 다음을 유도한다.

$$
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \qquad
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
$$

$$
F_1 = \frac{2\,\text{Precision}\cdot\text{Recall}}{\text{Precision} + \text{Recall}}
$$

정확도는 전체 예측 중 옳은 것의 비율이다.

$$
\text{Accuracy} = \frac{\text{TP} + \text{TN}}{n}
$$

```python
def precision_recall_f1(y_true, y_pred):
    """Compute precision, recall, and F1 score."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1

prec, rec, f1 = precision_recall_f1(y_true, y_pred)
accuracy = np.mean(y_true == y_pred)
```

결과는 정확도 $0.7300$, 정밀도 $0.6383$, 재현율 $0.7500$, $F_1 = 0.6897$이다. 재현율이
정밀도보다 높은 것은 문턱 $0.5$가 음성 분포의 중심($0.35$)보다 양성 분포의 중심($0.65$)에서
더 멀어, 음성 쪽에서 위양성이 많이 나오기 때문이다.

## ROC 곡선 직접 만들기

ROC 곡선은 문턱 $\tau$를 최대 점수에서 최소 점수까지 낮추며 FPR에 대한 TPR을 그린다.

$$
\text{TPR}(\tau) = \frac{\text{TP}(\tau)}{\text{TP}(\tau) + \text{FN}(\tau)}, \qquad
\text{FPR}(\tau) = \frac{\text{FP}(\tau)}{\text{FP}(\tau) + \text{TN}(\tau)}
$$

```python
def roc_curve(y_true, scores):
    """Compute ROC curve (FPR, TPR) for varying thresholds."""
    thresholds = np.sort(np.unique(scores))[::-1]
    fpr_list, tpr_list = [0.0], [0.0]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    for thresh in thresholds:
        y_pred = (scores >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)

    return np.array(fpr_list), np.array(tpr_list), thresholds

fpr, tpr, thresholds = roc_curve(y_true, scores)
```

!!! warning "반환되는 배열의 길이가 다르다"
    `fpr_list`와 `tpr_list`는 $(0,0)$으로 시작하므로 `thresholds`보다 원소가 하나 많다. 이
    자료에서는 `len(fpr) == 279`인데 `len(thresholds) == 278`이다. 따라서 유든의 J 같은
    관용적인 코드

    ```python
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]   # off by one!
    ```

    는 **한 칸 어긋난 문턱**을 돌려주며, `optimal_idx`가 마지막 값이면 `IndexError`가 난다.
    `thresholds[optimal_idx - 1]`을 쓰거나, 처음부터 앞에 `np.inf`를 붙여 세 배열의 길이를
    맞추어야 한다. `sklearn.metrics.roc_curve`는 후자를 택해 첫 문턱으로
    `max(scores) + 1`을 넣어 길이를 맞춘다.

## 사다리꼴 공식으로 AUC 구하기

ROC 곡선 아래 면적은 이웃한 점들이 만드는 사다리꼴 넓이의 합으로 근사한다.

$$
\text{AUC} \approx \sum_{i=1}^{m}
  \frac{\text{TPR}_i + \text{TPR}_{i-1}}{2}
  \bigl(\text{FPR}_i - \text{FPR}_{i-1}\bigr)
$$

```python
def auc_trapezoid(fpr, tpr):
    """Compute AUC via the trapezoidal rule."""
    order = np.argsort(fpr)
    fpr_sorted = fpr[order]
    tpr_sorted = tpr[order]
    return np.trapz(tpr_sorted, fpr_sorted)

area = auc_trapezoid(fpr, tpr)
print(f"AUC = {area:.4f}")
```

이 구현은 AUC $= 0.7935$를 주는데, `sklearn.metrics.roc_auc_score`는 $0.7931$을 준다.

!!! note "미세한 차이는 동점 때문이다"
    `np.clip(scores, 0, 1)` 때문에 점수 $0$에 13개, $1$에 11개의 관측치가 몰려 **동점**이
    생겼다. 만-휘트니 해석에서 동점은 절반씩 나누어야 하지만, 위 구현은
    `np.unique`로 동점을 하나의 문턱으로 합친 뒤 사다리꼴로 잇는다. 그 결과 동점 구간을
    직선으로 근사하게 되어 값이 조금 달라진다. 동점이 없으면 두 계산은 **정확히** 일치한다.
    (사다리꼴 공식이 AUC의 "근사"라기보다 동점 없는 경우 만-휘트니 U 통계량과 **동일한** 값을
    준다는 점은 [평가지표 절](./metrics.md)의 연습문제 3에서 다룬다.)

## 시각화

이 스크립트는 세 개의 패널을 만든다.

1. 문턱 0.5에서의 **혼동행렬 열지도**
2. 양성과 음성 범주의 **점수 분포**와 세로선으로 표시한 문턱
3. AUC를 표기한 **ROC 곡선**

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Confusion matrix heatmap
im = axes[0].imshow(cm, cmap='Blues', aspect='equal')
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Neg', 'Pos'])
axes[0].set_yticklabels(['Neg', 'Pos'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Score distributions
axes[1].hist(scores[y_true == 0], bins=25, alpha=0.6,
             label='Negative', edgecolor='k')
axes[1].hist(scores[y_true == 1], bins=25, alpha=0.6,
             label='Positive', edgecolor='k')
axes[1].axvline(0.5, color='red', linestyle='--', label='Threshold')
axes[1].set_xlabel('Predicted score'); axes[1].set_ylabel('Frequency')
axes[1].set_title('Score Distributions'); axes[1].legend(fontsize=8)

# ROC curve
axes[2].plot(fpr, tpr, linewidth=2, label=f'AUC = {area:.3f}')
axes[2].plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random')
axes[2].set_xlabel('False Positive Rate')
axes[2].set_ylabel('True Positive Rate')
axes[2].set_title('ROC Curve'); axes[2].legend(fontsize=9)

plt.tight_layout()
plt.show()
```

## 해석

- 두 점수 분포가 얼마나 떨어져 있는지가 분류기의 판별력을 결정한다. 더 많이 떨어져 있을수록
  AUC가 높다.
- 문턱 0.5에서의 혼동행렬은 위양성과 위음성 사이의 구체적인 절충을 보여준다.
- 사다리꼴 공식은 동점이 없을 때 경험적 ROC 점들로부터 AUC를 **정확히** 계산한다. 근사가 아니다.
- AUC는 무작위로 뽑은 양성 사례가 무작위로 뽑은 음성 사례보다 높은 점수를 받을 확률과 같다.

## 연습문제

**연습문제 1.**
어떤 분류기가 문턱 0.5에서 다음 혼동행렬을 냈다.

$$
\begin{pmatrix} 90 & 10 \\ 30 & 70 \end{pmatrix}
$$

정확도, 정밀도, 재현율, $F_1$을 계산하라.

??? success "풀이"

    TN = 90, FP = 10, FN = 30, TP = 70이다.

    $$
    \text{Accuracy} = \frac{90 + 70}{200} = 0.80
    $$

    $$
    \text{Precision} = \frac{70}{70 + 10} = \frac{70}{80} = 0.875
    $$

    $$
    \text{Recall} = \frac{70}{70 + 30} = \frac{70}{100} = 0.70
    $$

    $$
    F_1 = \frac{2 \times 0.875 \times 0.70}{0.875 + 0.70}
        = \frac{1.225}{1.575} \approx 0.778
    $$

    $\square$

---

**연습문제 2.**
ROC 곡선 위의 네 점 $(0, 0)$, $(0.2, 0.7)$, $(0.5, 0.9)$, $(1, 1)$이 주어졌을 때 사다리꼴
공식으로 AUC를 계산하라.

??? success "풀이"

    이웃한 쌍마다 사다리꼴 공식을 적용한다.

    $$
    A_1 = \frac{0 + 0.7}{2}(0.2 - 0) = 0.07
    $$

    $$
    A_2 = \frac{0.7 + 0.9}{2}(0.5 - 0.2) = 0.24
    $$

    $$
    A_3 = \frac{0.9 + 1.0}{2}(1.0 - 0.5) = 0.475
    $$

    $$
    \text{AUC} = 0.07 + 0.24 + 0.475 = 0.785
    $$

    $\square$

---

**연습문제 3.**
무작위 분류기(점수가 이름표와 독립)의 기대 AUC가 0.5임을 증명하라.

??? success "풀이"

    점수가 이름표와 독립이면, 무작위로 뽑은 양성-음성 쌍 $(s^+, s^-)$에 대해 대칭성에 의해

    $$
    P(s^+ > s^-) = P(s^+ < s^-) = \frac{1}{2}
    $$

    이다(연속형 점수에서는 동점의 확률이 0이므로 무시한다). AUC는 무작위 양성이 무작위 음성보다
    높은 점수를 받을 확률이므로 $\text{AUC} = 0.5$이다.

    !!! note "기대값이 0.5이지 항상 0.5인 것은 아니다"
        유한표본에서 실제 계산된 AUC는 $0.5$ 주위에서 요동친다. 양성 $m$개, 음성 $n$개일 때
        만-휘트니 통계량의 귀무분포로부터
        $\operatorname{Var}(\widehat{\text{AUC}}) = \frac{m+n+1}{12mn}$임이 알려져 있다.
        $m = 120$, $n = 180$이면 표준편차가 $0.0345$이므로, 아무 판별력이 없는 분류기도
        AUC $0.57$을 내는 일이 드물지 않다. AUC가 $0.5$보다 크다고 곧바로 판별력이 있다고
        결론지어서는 안 된다. $\square$

---

**연습문제 4.**
위 `roc_curve` 함수와 같은 방식으로 문턱을 훑으며 (재현율, 정밀도) 쌍을 기록하는
**정밀도-재현율 곡선** 함수를 구현하라.

??? success "풀이"

    ```python
    def pr_curve(y_true, scores):
        """Compute precision-recall curve for varying thresholds."""
        thresholds = np.sort(np.unique(scores))[::-1]
        precisions, recalls = [1.0], [0.0]
        n_pos = np.sum(y_true == 1)

        for thresh in thresholds:
            y_pred = (scores >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)

        return np.array(recalls), np.array(precisions), thresholds
    ```

    곡선은 (재현율 = 0, 정밀도 = 1)에서 시작해 문턱이 낮아짐에 따라 절충을 그린다. 평균정밀도
    (AP)는 곡선을 따라 $\sum_k (R_k - R_{k-1}) P_k$를 더해 근사할 수 있다.

    **주의할 점 두 가지.**

    - 시작점 $(0, 1)$은 관례적으로 붙이는 값이지 계산된 값이 아니다. 재현율이 0인 지점에서
      정밀도는 정의되지 않는다(분모 TP + FP가 0이거나 표본이 극히 적다). 곡선의 왼쪽 끝은
      해석하지 말라.
    - AP를 계산할 때 사다리꼴 공식을 쓰면 안 된다. PR 곡선은 톱니 모양이라 이웃 점 사이를
      직선으로 잇는 것이 낙관적으로 편향된다. `sklearn.metrics.average_precision_score`가
      직사각형 합 $\sum_k (R_k - R_{k-1})P_k$를 쓰는 이유가 이것이다.

    $\square$

---

**연습문제 5.**
자료가 $n$개인 이항 분류기에서 정확도를 혼동행렬로

$$
\text{Accuracy} = \frac{\text{trace}(C)}{n}
$$

와 같이 쓸 수 있음을 보여라. 여기서 $C$는 $2\times 2$ 혼동행렬이다.

??? success "풀이"

    혼동행렬은

    $$
    C = \begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix}
    $$

    이고 대각합은 $\text{trace}(C) = \text{TN} + \text{TP}$로 옳게 예측한 총 개수다. 표본 크기는
    $n = \text{TN} + \text{FP} + \text{FN} + \text{TP}$이므로

    $$
    \text{Accuracy}
      = \frac{\text{TP} + \text{TN}}{n}
      = \frac{\text{trace}(C)}{n}
    $$

    이다.

    이는 $K$개 범주로 일반화된다. $K\times K$ 혼동행렬에서도 정확도는 여전히
    $\text{trace}(C)/n$이다. 다만 다범주에서는 이 값의 정보량이 더 떨어진다는 점에 유의하라.
    $\text{trace}(C)$는 어느 범주에서 틀렸는지, 어떤 범주로 잘못 분류했는지를 전혀 구별하지
    않는다. 비대각 원소의 구조 — 어떤 범주 쌍이 서로 혼동되는지 — 가 실제로 모형을 개선하는 데
    필요한 정보다. $\square$
