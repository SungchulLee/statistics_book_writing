# Kolmogorov-Smirnov 검정과 Lilliefors 검정

시각적 방법과 기술통계가 자료 분포에 대한 통찰을 주는 반면, 형식적 통계검정은 정규성을 평가하는 더 엄밀한 방법을 제공한다. 이 검정들은 관측된 자료가 기대되는 정규분포에서 유의하게 벗어나는지 평가하여 판단의 통계적 근거를 제공한다.

## Kolmogorov-Smirnov 검정

**Kolmogorov-Smirnov(K-S) 검정**은 표본자료의 경험적 누적분포함수(ECDF)를 기준분포(여기서는 정규분포)의 누적분포함수(CDF)와 비교하는 비모수 검정이다. 두 분포의 중심위치와 전반적 모양(분산) 양쪽의 차이에 민감하다.

### 가설

- **귀무가설** ($H_0$): 자료가 지정한 분포(정규분포)를 따른다.
- **대립가설** ($H_1$): 자료가 지정한 분포를 따르지 않는다.

### K-S 검정통계량의 계산

Kolmogorov-Smirnov 검정통계량 $D$는 표본의 경험적 CDF와 기준분포의 CDF 사이의 최대 절대차로 계산한다. 절차는 다음과 같다.

1. **자료 정렬**: $X_1, X_2, \dots, X_n$을 오름차순으로 정렬하여 $X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}$을 얻는다.

2. **경험적 CDF 계산**: 정렬된 각 자료점 $X_{(i)}$에 대해 ECDF를 그 이하인 자료점의 비율로 정의한다.

    $$
    F_{\text{emp}}(X_{(i)}) = \frac{i}{n}
    $$

    여기서 $n$은 전체 표본크기이고 $i$는 자료점의 순위이다.

3. **이론적 CDF 계산**: $X_{(i)}$에서 평가한 정규분포의 이론적 CDF $F_{\text{norm}}(X_{(i)})$는

    $$
    F_{\text{norm}}(X_{(i)}) = \Phi\left( \frac{X_{(i)} - \mu}{\sigma} \right)
    $$

    여기서 $\Phi$는 표준정규 누적분포함수이고 $\mu$와 $\sigma$는 각각 표본평균과 표본표준편차이다.

4. **검정통계량 $D$**: K-S 검정통계량 $D$는 임의의 표본점에서 경험적 CDF와 이론적 CDF의 최대 절대차이다.

    $$
    D = \max_i \left| F_{\text{emp}}(X_{(i)}) - F_{\text{norm}}(X_{(i)}) \right|
    $$

    곧 $D$는 자료 범위에 걸쳐 두 CDF 사이의 가장 큰 수직 거리를 잰다.

!!! note "실제 구현은 양쪽 계단을 모두 본다"
    경험적 CDF는 각 자료점에서 뛰어오르는 계단함수이므로, 엄밀한 통계량은 각 점의 왼쪽 극한 $(i-1)/n$과 오른쪽 값 $i/n$ 모두에 대한 차이의 최댓값을 취한다. `scipy.stats.kstest`가 이 방식으로 계산한다.

### 판정 규칙

- $D$가 선택한 유의수준 $\alpha$의 임계값을 넘으면 $H_0$을 기각한다(자료가 정규분포를 따르지 않는다).
- $D$가 임계값보다 작으면 $H_0$을 기각하지 못한다(자료가 정규분포를 따를 수 있다).

K-S 검정은 분포의 중심위치와 모양의 차이를 탐지하는 데 효과적이다. 다만 Anderson-Darling 검정 같은 방법에 비해 꼬리에서의 이탈에는 덜 민감하다.

### Python 구현

```python
import numpy as np
from scipy import stats

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

data_ks = (data - data.mean()) / data.std()

# Perform Kolmogorov-Smirnov test
stat, p_value = stats.kstest(data_ks, 'norm')
print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

출력:

```text
Kolmogorov-Smirnov Test: Statistic=0.01903411267034605, p-value=0.8547733408587939
Fail to reject H_0: The data is normally distributed.
```

!!! warning "이 코드는 사실 타당한 K-S 검정이 아니다"
    위 코드는 **자료에서 추정한** 평균과 표준편차로 자료를 표준화한 뒤 표준 K-S 임계값을 쓴다. 표준화가 경험적 CDF를 이론적 CDF 쪽으로 인위적으로 끌어당기므로 $D$가 작아지고 $p$값이 지나치게 커진다. 곧 검정이 보수적이 되어 검정력을 잃는다. 이것이 바로 다음 절의 Lilliefors 검정이 필요한 이유이다.

---

## Kolmogorov-Smirnov 검정과 Lilliefors 검정

SciPy의 `stats.kstest` 함수는 **Kolmogorov-Smirnov(K-S) 검정**을 수행하며 **Lilliefors 검정**이 아니다. 두 검정의 차이는 다음과 같다.

### Kolmogorov-Smirnov 검정 (`stats.kstest`)

- **목적**: 일반적인 **적합도 검정**으로, 표본을 **모수가 고정된** 알려진 분포와 비교한다.
- **모수 가정**: 분포의 모수(예: 평균, 표준편차)를 **사전에 안다**고 가정한다.
- **용도**: 미리 정해진 모수를 가진 특정 분포에 자료가 맞는지 보고 싶을 때 적절하다.

### Lilliefors 검정 (`statsmodels.stats.diagnostic.lilliefors`)

- **목적**: 모집단 모수(평균과 표준편차)가 **미지이고 표본에서 추정될 때**의 **정규성 검정**을 위해 K-S를 수정한 것이다.
- **모수 가정**: 모수를 표본에서 추정한다는 사실을 조정하여 이 상황에 맞춘 다른 임계값을 제공한다.
- **제공처**: SciPy에는 없고 `statsmodels`가 제공한다.

### 비교

| 항목 | `stats.kstest` (K-S 검정) | Lilliefors 검정 |
|---------------------------|--------------------------------------------------|-------------------------------------------------|
| **목적** | 일반적 적합도(임의의 분포) | 모수가 미지일 때의 정규성 검정 |
| **모수 지식** | 모수를 안다고 가정 | 모수를 모른다고 가정 |
| **모수 추정** | 추정된 모수를 위해 설계되지 않음 | 추정된 모수에 맞게 조정됨 |
| **SciPy 구현** | 있음 (`stats.kstest`) | SciPy에는 직접 구현 없음 |

### Python 구현

```python
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import lilliefors

np.random.seed(0)

# Generate a sample dataset
# data = np.random.normal(0, 1, 1000)
data = np.random.normal(1, 10, 1000)

data_ks = (data - data.mean()) / data.std()

# Perform Kolmogorov-Smirnov test
stat, p_value = stats.kstest(data_ks, 'norm')
print(f"Kolmogorov-Smirnov Test: Statistic={stat}, p-value={p_value}")

stat, p_value = lilliefors(data)
print(f"Lilliefors Test: Statistic={stat}, p-value={p_value}")

# Interpretation
alpha = 0.05
if p_value <= alpha:
    print("Reject H_0: The data is not normally distributed.")
else:
    print("Fail to reject H_0: The data is normally distributed.")
```

출력:

```text
Kolmogorov-Smirnov Test: Statistic=0.01903411267034605, p-value=0.8547733408587939
Lilliefors Test: Statistic=0.019125294462402076, p-value=0.5818164701330186
Fail to reject H_0: The data is normally distributed.
```

두 검정통계량은 거의 같지만($0.01903$ 대 $0.01913$) **$p$값이 다르다**($0.855$ 대 $0.582$). Lilliefors가 모수 추정을 반영한 다른 귀무분포를 쓰기 때문이다. 여기서는 자료가 실제로 정규이므로 두 검정 모두 기각하지 않지만, 자료가 정규가 아니라면 Lilliefors 쪽이 훨씬 먼저 이를 잡아낸다.

---

## 어느 검정을 고를 것인가

### `stats.kstest`(Kolmogorov-Smirnov 검정)를 고를 때

- **용도**: **모수가 고정된** 알려진 이론적 분포와 표본을 비교하는 일반적 적합도 검정.
- **유연성**: 분포의 모수를 미리 지정하기만 하면 어떤 분포(정규, 지수 등)에 대해서도 검정할 수 있다.
- **장점**: 정규성을 넘어 다양한 이론적 분포와 경험적 자료를 비교하는 데 폭넓게 적용된다.
- **한계**: 추정된 모수로 정규성 검정에 쓰면 부정확하다. 모수 추정을 조정하지 않기 때문이다.

### `statsmodels.stats.diagnostic.lilliefors`(Lilliefors 검정)를 고를 때

- **용도**: 분포의 모수가 **미지이고 표본에서 추정될 때**의 **정규성 검정**을 위해 특별히 설계되었다.
- **유연성**: 정규성 검정에 한정되지만 그 맥락에서는 매우 정확하다.
- **장점**: 모수를 추정하는 상황에서 정규성 검정을 더 정확하게 수행한다.
- **한계**: 정규성 검정에 한정되며 다른 분포의 검정에는 쓸 수 없다.

### 권장 사항

- **일반적인 분포 검정**(예: 모수가 고정된 지수분포나 Weibull 분포에 자료가 맞는지 확인)에는 **`stats.kstest`**가 낫다.
- **모수가 미지인 정규성 검정**에는 모수 추정 과정을 반영하여 더 정확하게 평가하는 **`statsmodels`의 `lilliefors`**가 선호된다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
Kolmogorov-Smirnov 검정과 Lilliefors 검정의 차이를 설명하라. KS 대신 Lilliefors를 반드시 써야 하는 경우는 언제인가?

</div>

??? success "풀이"
    **Kolmogorov-Smirnov 검정**은 경험적 CDF를 완전히 지정된 이론적 CDF(모든 모수를 앎)와 비교한다. 예를 들어 자료가 정확히 $N(0, 1)$에서 왔는지 검정하는 경우이다.

    **Lilliefors 검정**은 모수를 자료에서 추정하는 복합가설을 위한 수정이다. 정규성 검정에서는 보통 자료에서 $\mu$와 $\sigma$를 추정한 뒤 $N(\hat{\mu}, \hat{\sigma}^2)$과 비교한다.

    검정 대상과 같은 자료에서 모수를 추정하는 경우에는 반드시 KS가 아니라 Lilliefors를 써야 한다. 추정된 모수로 KS 임계값을 쓰는 것은 타당하지 않다. 모수를 추정하면 경험적 CDF가 이론적 CDF에 더 가까워지므로(적합이 최적화되므로) KS의 $p$값이 지나치게 커지고(보수적이 되고) 검정력을 잃는다.

<div class="drillbox" markdown>

**연습문제 2.**
KS 검정통계량은 $D_n = \sup_x |F_n(x) - F_0(x)|$이다. 이것이 기하학적으로 무엇을 재는지 설명하라.

</div>

??? success "풀이"
    $D_n$은 모든 $x$에 걸쳐 경험적 CDF(계단함수)와 이론적 CDF(매끄러운 곡선)의 최대 수직 거리이다. 기하학적으로는 두 곡선 사이의 가장 큰 틈이다.

    자료가 $F_0$에서 왔다면 Glivenko-Cantelli 정리에 의해 경험적 CDF가 이론적 CDF를 가깝게 따라가므로 $D_n$이 작아야 한다. $D_n$이 크다는 것은 분포의 어느 지점에서 관측 자료가 이론적 분포의 예측에서 크게 벗어난다는 뜻이다. 어떤 구역에 관측값이 너무 많거나 너무 적다는 것이다.

<div class="drillbox" markdown>

**연습문제 3.**
비정규성 탐지에서 KS/Lilliefors 검정이 Shapiro-Wilk나 Anderson-Darling 검정보다 대체로 검정력이 낮은 이유는 무엇인가?

</div>

??? success "풀이"
    KS 검정은 최대 편차 $D_n$만 쓴다. 이는 최악의 불일치를 하나의 수치로 요약한 것이며 두 가지 단점이 있다.

    1. **꼬리 가중이 없다:** KS 검정은 (자료가 빽빽한) 분포 중앙의 이탈과 (정규성에 더 중요한) 꼬리의 이탈을 똑같이 다룬다. Anderson-Darling 검정은 꼬리의 이탈에 더 큰 가중치를 준다.

    2. **한 점에만 초점:** 상한만 쓰므로 KS는 이탈의 패턴을 무시한다. Shapiro-Wilk 검정은 상관 기반 계산에 모든 순서통계량을 써서 자료에서 더 많은 정보를 뽑아낸다.

    KS 검정은 (임의의 분포에 대한) 일반적 적합도 검정으로 설계되었지 정규성 전용이 아니다. Shapiro-Wilk 같은 특화된 검정은 정규분포의 구조를 활용한다.

<div class="drillbox" markdown>

**연습문제 4.**
관측값 $n = 30$개에 대한 Lilliefors 검정에서 $D_n = 0.14$를 얻었다. $\alpha = 0.05$의 임계값은 $0.161$이다. 결론은 무엇인가?

</div>

??? success "풀이"
    $D_n = 0.14 < 0.161$이므로 $\alpha = 0.05$에서 $H_0$을 기각하지 못한다. Lilliefors 검정에 따르면 자료가 정규성과 일관된다.

    (참고로 이 임계값은 잘 알려진 근사식 $0.886/\sqrt{n} = 0.886/\sqrt{30} = 0.1618$과 일치한다.)

    다만 $n = 30$이면 검정력이 제한적이다. 특히 가벼운 두꺼운 꼬리 같은 미묘한 대립가설에 대해 그렇다. 기각하지 못했다는 것이 정규성을 증명하지는 않으며, 표본크기가 부족했음을 반영할 뿐일 수 있다. Q-Q 그림을 함께 보면 이탈의 성격과 정도에 대한 추가적인 시각적 증거를 얻을 수 있다.
