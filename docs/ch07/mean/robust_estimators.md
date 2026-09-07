# 로버스트 추정량 비교

## 개요

로버스트 추정량은 이상점과 분포 가정으로부터의 이탈에 저항한다. 표본평균과 표준편차는 정규 자료에서 최적이지만 극단 관측값 몇 개만으로도 심하게 왜곡될 수 있다. 이 페이지에서는 위치(절사평균, 가중평균, 중앙값)와 척도(MAD, IQR) 추정에 대한 로버스트한 대안들을 비교하고, 깨끗한 자료와 오염된 자료에서의 거동을 보인다.

## 위치추정량

### 절사평균

**$\alpha$-절사평균**은 정렬한 자료에서 아래위로 $\alpha$ 비율만큼 제거하고 나머지를 평균한다:

$$\bar{X}_\alpha = \frac{1}{n - 2k}\sum_{i=k+1}^{n-k} X_{(i)}$$

여기서 $k = \lfloor n\alpha \rfloor$이고 $X_{(i)}$는 $i$번째 순서통계량이다.

```python
import numpy as np

def trimmed_mean(data, proportion=0.1):
    x = np.sort(data)
    n = len(x)
    k = int(np.floor(n * proportion))
    if k == 0:
        return x.mean()
    return x[k:-k].mean()
```

### 가중평균과 가중중앙값

**가중평균**은 관측값마다 다른 중요도를 부여한다:

$$\bar{X}_w = \frac{\sum_{i=1}^n w_i X_i}{\sum_{i=1}^n w_i}$$

**가중중앙값**은 양쪽의 누적 가중치가 각각 50%를 넘지 않게 하는 값 $m$이다. 가중평균보다 로버스트하다.

```python
def weighted_mean(data, weights):
    return np.sum(data * weights) / np.sum(weights)

def weighted_median(data, weights):
    order = np.argsort(data)
    sorted_data = data[order]
    sorted_w = weights[order]
    cum_w = np.cumsum(sorted_w) / np.sum(sorted_w)
    idx = np.searchsorted(cum_w, 0.5)
    return sorted_data[idx]
```

## 척도추정량

### 중앙값 절대편차

**MAD**(중앙값 절대편차)는 산포의 로버스트한 측도이다:

$$\text{MAD} = \text{median}(|X_i - \text{median}(X)|)$$

정규 자료에서 $\text{MAD} \approx 0.6745\sigma$이므로 $\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{MAD}$가 $\sigma$의 일치추정량이 된다.

```python
def mad(data):
    med = np.median(data)
    return np.median(np.abs(data - med))
```

### 사분위수범위

**IQR**(사분위수범위)은 또 다른 로버스트 척도이다:

$$\text{IQR} = Q_3 - Q_1$$

정규 자료에서 $\text{IQR} \approx 1.349\sigma$이므로 $\hat{\sigma}_{\text{IQR}} = \text{IQR}/1.349$가 $\sigma$를 추정한다.

## 오염 아래에서의 비교

다음 코드는 깨끗한 정규 자료와 극단 이상점으로 오염된 자료에서 위치·척도 추정량을 비교한다.

```python
np.random.seed(42)

# Clean data
clean = np.random.normal(loc=50, scale=10, size=100)

# Contaminated data: add 5 extreme outliers
outliers = np.array([200, 250, 300, -100, -150])
contaminated = np.concatenate([clean, outliers])

for label, data in [("Clean", clean), ("Contaminated", contaminated)]:
    print(f"\n{label} data (n = {len(data)}):")
    print(f"  Mean             = {data.mean():.2f}")
    print(f"  Median           = {np.median(data):.2f}")
    print(f"  Trimmed mean 10% = {trimmed_mean(data, 0.10):.2f}")
    print(f"  Trimmed mean 20% = {trimmed_mean(data, 0.20):.2f}")
    print(f"  Std dev          = {data.std(ddof=1):.2f}")
    print(f"  IQR              = {np.percentile(data, 75) - np.percentile(data, 25):.2f}")
    print(f"  MAD              = {mad(data):.2f}")
```

!!! note "이상점의 영향"
    관측값 105개 중 이상점 5개가 평균을 2 남짓 옮기고(약 50에서 약 52로) 표준편차는 네 배 넘게 키운다. 반면 중앙값, 절사평균, IQR, MAD는 거의 영향을 받지 않는다.

## 점진적 오염 아래에서의 로버스트성

붕괴 과정을 시각화하기 위해, 깨끗한 관측값 100개에 이상점(값 = 300)을 하나씩 늘려 가며 각 추정량을 추적한다.

```python
import matplotlib.pyplot as plt

n_out_range = range(0, 21)
means, medians, trims = [], [], []
stds, iqrs, mads_list = [], [], []

for n_out in n_out_range:
    extra = np.full(n_out, 300.0)
    data = np.concatenate([clean, extra])
    means.append(data.mean())
    medians.append(np.median(data))
    trims.append(trimmed_mean(data, 0.10))
    stds.append(data.std(ddof=1))
    iqrs.append(np.percentile(data, 75) - np.percentile(data, 25))
    mads_list.append(mad(data))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(list(n_out_range), means, 'o-', label='Mean', markersize=4)
ax1.plot(list(n_out_range), medians, 's-', label='Median', markersize=4)
ax1.plot(list(n_out_range), trims, 'D-', label='Trimmed Mean (10%)', markersize=4)
ax1.set_xlabel('Number of outliers added (value = 300)')
ax1.set_ylabel('Estimated location')
ax1.set_title('Location Estimators vs Outlier Count')
ax1.legend()

ax2.plot(list(n_out_range), stds, 'o-', label='Std Dev', markersize=4)
ax2.plot(list(n_out_range), iqrs, 's-', label='IQR', markersize=4)
ax2.plot(list(n_out_range), mads_list, 'D-', label='MAD', markersize=4)
ax2.set_xlabel('Number of outliers added (value = 300)')
ax2.set_ylabel('Estimated scale')
ax2.set_title('Scale Estimators vs Outlier Count')
ax2.legend()

plt.tight_layout()
plt.show()
```

## 붕괴점

추정량의 **붕괴점**은 결과가 얼마든지 나빠지기 전까지 견딜 수 있는 임의 오염의 최대 비율이다.

| 추정량 | 붕괴점 |
|-----------|----------------|
| 평균 | $0\%$ (극단값 하나로 값이 얼마든지 바뀐다) |
| 중앙값 | $50\%$ |
| $\alpha$-절사평균 | $\alpha$ (예: 10% 절사이면 10%) |
| 표준편차 | $0\%$ |
| MAD | $50\%$ |
| IQR | $25\%$ |

!!! info "로버스트성 대 효율성"
    로버스트 추정량은 가정한 모형 아래에서 어느 정도 효율을 희생하는 대신(예: 정규 자료에서 중앙값의 효율은 평균의 63.7%에 불과하다) 모형 위반에 대한 보호를 얻는다. 절사평균은 유용한 중간 지점을 준다: 정규성 아래에서 평균에 거의 맞먹는 효율을 유지하면서 의미 있는 로버스트성을 제공한다.

## 해석

- **평균과 표준편차**는 깨끗한 정규 자료에서 최적이지만 이상점에 얼마든지 민감하다(붕괴점 0).
- **중앙값과 MAD**는 붕괴점이 50%이다 — 자료의 절반 가까이가 오염되어도 여전히 유의미하다.
- **절사평균**은 조절 가능한 절충안을 준다: 절사비율이 작으면 효율을 유지하면서 적당한 로버스트성을 얻는다.
- **가중추정량**은 관측값의 품질이 다를 때 유용하지만, 가중치 자체가 로버스트하게 정해지지 않으면 가중평균은 여전히 이상점에 민감하다.
- 실무에서는 고전적 추정량과 로버스트 추정량을 함께 계산하는 것이 좋다. 둘이 크게 다르면 자료를 더 들여다볼 이유가 된다.

## 연습문제

**연습문제 1.**
자료 $\{1, 2, 3, 4, 5, 6, 7, 8, 9, 100\}$에 대해 평균, 중앙값, 10% 절사평균을 계산하라. 어느 추정량이 "전형적인" 값을 가장 잘 나타내는가?

??? success "풀이"
    **평균:** $\frac{1+2+3+4+5+6+7+8+9+100}{10} = \frac{145}{10} = 14.5$

    **중앙값:** 정렬한 자료가 10개이므로 중앙값은 5번째와 6번째의 평균이다: $(5+6)/2 = 5.5$.

    **10% 절사평균:** $n = 10$, $\alpha = 0.10$이므로 양 끝에서 $k = \lfloor 10 \times 0.10 \rfloor = 1$개씩 절사한다. 남는 것: $\{2, 3, 4, 5, 6, 7, 8, 9\}$. 평균: $44/8 = 5.5$.

    평균(14.5)은 100이라는 이상점 하나 때문에 자료 대부분보다 훨씬 위로 끌려간다. 중앙값(5.5)과 절사평균(5.5)은 이 이상점을 무시하고 전형적인 값을 더 잘 나타낸다. $\square$

---

**연습문제 2.**
표본중앙값의 붕괴점이 $\lfloor(n-1)/2\rfloor / n$이며 큰 $n$에서 50%에 가까워짐을 증명하라.

??? success "풀이"
    관측값 $x_1 \leq x_2 \leq \cdots \leq x_n$을 생각하자. 중앙값은 대략 $x_{(\lceil n/2 \rceil)}$이다.

    중앙값을 얼마든지 크게 만들려면 자료의 절반 이상이 극단이 되도록 충분히 많은 관측값을 바꿔야 한다. 구체적으로 $\lceil n/2 \rceil$개를 $+\infty$로 가는 값으로 바꾸어야 하며, 그러면 새 중앙값이 그 극단값 중 하나가 된다.

    $\lfloor (n-1)/2 \rfloor$개만 바꾸면 원래 관측값이 적어도 $\lceil (n+1)/2 \rceil$개 남는다. 중앙값은 이 원래 값들 사이에 놓이므로 유계로 남는다.

    따라서 붕괴점은 $\lfloor (n-1)/2 \rfloor / n$이다. $n$이 홀수이면 $(n-1)/(2n)$, 짝수이면 $(n-2)/(2n)$이다. 두 경우 모두 $n \to \infty$일 때 붕괴점이 $1/2 = 50\%$에 가까워진다. $\square$

---

**연습문제 3.**
$X \sim N(\mu, \sigma^2)$에서 $\text{MAD} = \mathcal{N}^{-1}(3/4) \cdot \sigma \approx 0.6745\sigma$임을 보이고, 따라서 $1.4826 \times \text{MAD}$가 $\sigma$의 일치추정량임을 보여라.

??? success "풀이"
    $X \sim N(\mu, \sigma^2)$에서 편차 $|X - \mu|$는 반정규분포를 따른다. $|X - \mu|$의 중앙값은 다음을 만족하는 값 $m$이다:

    $$P(|X - \mu| \leq m) = \frac{1}{2}$$

    즉 $P(-m \leq X - \mu \leq m) = 1/2$이므로 $\mathcal{N}(m/\sigma) - \mathcal{N}(-m/\sigma) = 1/2$, 따라서 $2\mathcal{N}(m/\sigma) - 1 = 1/2$이고 $\mathcal{N}(m/\sigma) = 3/4$이다.

    그러므로 $m = \sigma \mathcal{N}^{-1}(3/4) \approx 0.6745\sigma$이다.

    (참 중앙값 $\mu$를 쓴) 모집단 MAD는 $0.6745\sigma$와 같다. 표본중앙값의 일치성과 연속사상정리에 의해 표본 MAD는 모집단 MAD로 수렴한다.

    따라서 $\hat{\sigma} = \text{MAD}/0.6745 = 1.4826 \times \text{MAD}$는 $\sigma$의 일치추정량이다. $\square$

---

**연습문제 4.**
관측값 200개의 자료에서 평균 50, 표준편차 10, 중앙값 49, MAD 6.5를 얻었다. 이상점이나 비정규성의 증거가 있는가? 고전적 추정량과 로버스트 추정량의 비를 써서 답을 정당화하라.

??? success "풀이"
    정규성 아래에서 기대되는 바:

    - 평균 $\approx$ 중앙값: 여기서는 $50 \approx 49$로 잘 맞는다. 오른쪽으로 약간 치우쳤다.
    - $\text{MAD} \approx 0.6745\sigma$: 기대 MAD $= 0.6745 \times 10 = 6.745$. 관측된 MAD $= 6.5$. 비: $6.5/6.745 = 0.964$.
    - $\hat{\sigma}_{\text{MAD}} = 1.4826 \times 6.5 = 9.637$ 대 $s = 10$. 비: $s/\hat{\sigma}_{\text{MAD}} = 10/9.637 = 1.038$.

    고전적 표준편차가 로버스트 추정값보다 약 3.8% 클 뿐이다. 이상점이 상당했다면 $s$가 $\hat{\sigma}_{\text{MAD}}$보다 훨씬 컸을 것이다(비가 1.2 이상이면 우려할 만하다).

    모든 짝(평균/중앙값, $s$/MAD 기반 $\hat{\sigma}$)의 추정값이 거의 일치하므로 자료는 큰 이상점 오염 없이 근사적으로 정규라고 볼 수 있다. 약간의 불일치는 $n = 200$에서의 정상적인 표본변동 범위 안이다. $\square$

---

**연습문제 5.**
$\alpha$-절사평균의 로버스트성–효율성 맞바꿈을 설명하라. $n = 100$인 정규 자료에서 표본평균 대비 10% 절사평균의 점근 상대효율은 얼마인가?

??? success "풀이"
    **맞바꿈:** $\alpha$를 키우면 로버스트성이 좋아지지만(붕괴점이 높아지지만) 정규 모형 아래에서 효율이 떨어진다(좋은 자료를 버리기 때문이다). $\alpha = 0$이면 평균(완전한 효율, 로버스트성 0)이고, $\alpha \to 0.5$이면 중앙값(최대 로버스트성, 정규 자료에서 효율 63.7%)에 가까워진다.

    정규 자료에서 $\alpha$-절사평균의 점근분산은

    $$\text{Var}(\bar{X}_\alpha) \approx \frac{\sigma^2 c(\alpha)}{n}$$

    꼴로 쓸 수 있으며, $\alpha > 0$이면 $c(\alpha) > 1$이다. 표본평균 대비 점근 상대효율(ARE)은

    $$\text{ARE} = \frac{\text{Var}(\bar{X})}{\text{Var}(\bar{X}_\alpha)} = \frac{1}{c(\alpha)}$$

    이다. $\alpha = 0.10$(10% 절사)이면 ARE가 대략 **0.95**이다(절사평균이 자료가 담은 정보의 약 95%를 쓴다).

    즉 정규성 아래에서 효율을 약 5%만 잃으면서 붕괴점 10%를 얻는다(오염을 10%까지 견딘다). 일반적으로 훌륭한 맞바꿈으로 여겨지며, 10% 절사평균이 많은 응용에서 인기 있는 기본값인 이유이다. $\square$
