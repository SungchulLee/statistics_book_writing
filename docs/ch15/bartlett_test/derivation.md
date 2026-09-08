# 유도와 카이제곱 근사

Bartlett 검정통계량은 가능도비 원리로 동기를 부여할 수 있다. $k$개 집단이 공통 분산을 갖는다는 귀무가설 아래에서 가능도함수가 단순해지고, 제약된 가능도와 제약 없는 가능도의 비에서 익숙한 Bartlett 공식이 곧바로 나온다. 이 절은 유도를 단계별로 제시하고 정확한 카이제곱 근사를 보장하는 보정인자를 설명한다.

## 설정과 기호

정규 모집단에서 $k$개의 독립 표본을 뽑았다고 하자.

$$
X_{ij} \sim N(\mu_i, \sigma_i^2), \quad i = 1, \ldots, k, \quad j = 1, \ldots, n_i
$$

$N = \sum_{i=1}^{k} n_i$를 전체 표본크기, $S_i^2$을 집단 $i$의 표본분산이라 하고 합동분산을 다음으로 정의한다.

$$
S_p^2 = \frac{\sum_{i=1}^{k} (n_i - 1) S_i^2}{N - k}
$$

이는 자유도 $\nu_i = n_i - 1$에 비례하는 가중치를 갖는 집단분산들의 가중평균이다.

## 가능도비

귀무가설 $H_0\colon \sigma_1^2 = \cdots = \sigma_k^2 = \sigma^2$ 아래에서 공통 분산의 최대가능도추정량은 $\hat{\sigma}^2 = S_p^2$이다. 대립가설 아래에서는 각 집단이 자기 추정량 $\hat{\sigma}_i^2 = S_i^2$을 갖는다.

로그가능도비 통계량은

$$
-2 \ln \Lambda = \sum_{i=1}^{k} \nu_i \ln\!\left(\frac{S_p^2}{S_i^2}\right) = (N-k)\ln S_p^2 - \sum_{i=1}^{k} \nu_i \ln S_i^2
$$

여기서 $\nu_i = n_i - 1$이다. 이 양은 (로그의 오목성과 Jensen 부등식에 의해) 항상 음이 아니며, 모든 표본분산이 같을 때만 0이 된다.

## 보정인자

통계량 $-2\ln\Lambda$는 표본크기가 커지면 $\chi^2_{k-1}$로 수렴하지만, 중간 정도의 표본크기에서는 수렴이 느리다. Bartlett(1937)은 카이제곱 근사를 개선하기 위해 보정인자를 도입했다.

$$
C = 1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k} \frac{1}{\nu_i} - \frac{1}{N - k}\right)
$$

보정된 검정통계량은

$$
T = \frac{-2\ln\Lambda}{C} = \frac{(N-k)\ln S_p^2 - \sum_{i=1}^{k} \nu_i \ln S_i^2}{1 + \frac{1}{3(k-1)}\left(\sum_{i=1}^{k} \frac{1}{\nu_i} - \frac{1}{N-k}\right)}
$$

$H_0$과 정규성 아래에서 $T \stackrel{\text{근사}}{\sim} \chi^2_{k-1}$이다.

!!! note "보정의 목적"
    보정인자 $C$는 항상 1보다 크므로 $C$로 나누면 검정통계량이 줄어든다. 보정하지 않으면 작은 표본에서 $-2\ln\Lambda$가 지나치게 커져 제1종 오류율이 부풀려진다. 보정은 실제 기각률을 명목 수준 $\alpha$에 가깝게 되돌린다.

## 공식의 직관

$T$의 분자는 다음과 같이 해석할 수 있다.

- $(N - k)\ln S_p^2$은 합동분산의 로그에 전체 자유도를 곱한 것이다.
- $\sum \nu_i \ln S_i^2$은 각 집단분산의 로그에 그 집단의 자유도를 곱해 더한 것이다.

모든 집단분산이 같으면 모든 $i$에 대해 $S_i^2 \approx S_p^2$이므로 분자가 0에 가깝다. 하나 이상의 집단이 실질적으로 다른 분산을 가지면 개별 $\ln S_i^2$이 $\ln S_p^2$에서 벗어나고 분자가 커진다.

곧 이 검정통계량은 개별 로그분산이 합동 로그분산에서 얼마나 벗어나는지를 표본크기로 조정하여 재는 것이다.

## 판정규칙

유의수준 $\alpha$에서 다음이면 $H_0$을 기각한다.

$$
T > \chi^2_{1-\alpha,\, k-1}
$$

여기서 $\chi^2_{1-\alpha,\, k-1}$은 자유도 $k-1$인 카이제곱분포의 $(1-\alpha)$ 분위수이다. 등분산에서 어느 방향으로 벗어나든 $T$가 커지므로 이 검정은 항상 단측(오른쪽 꼬리)이다.

## 예제

세 집단의 표본크기와 분산이 다음과 같다.

| 집단 | $n_i$ | $S_i^2$ | $\nu_i = n_i - 1$ |
|---|---|---|---|
| 1 | 10 | 5.2 | 9 |
| 2 | 12 | 8.1 | 11 |
| 3 | 8 | 4.7 | 7 |

**1단계.** 합계: $N = 30$, $N - k = 27$.

**2단계.** 합동분산:

$$
S_p^2 = \frac{9(5.2) + 11(8.1) + 7(4.7)}{27} = \frac{46.8 + 89.1 + 32.9}{27} = \frac{168.8}{27} = 6.2519
$$

**3단계.** 분자:

$$
27 \ln(6.2519) - [9\ln(5.2) + 11\ln(8.1) + 7\ln(4.7)]
$$

$$
= 27(1.8329) - [9(1.6487) + 11(2.0919) + 7(1.5476)]
$$

$$
= 49.488 - [14.838 + 23.011 + 10.833] = 49.488 - 48.682 = 0.806
$$

**4단계.** 보정인자:

$$
C = 1 + \frac{1}{3(2)}\left(\frac{1}{9} + \frac{1}{11} + \frac{1}{7} - \frac{1}{27}\right) = 1 + \frac{1}{6}(0.1111 + 0.0909 + 0.1429 - 0.0370) = 1 + \frac{0.3078}{6} = 1.0513
$$

**5단계.** 검정통계량:

$$
T = \frac{0.806}{1.0513} = 0.767
$$

**6단계.** $\chi^2_{0.95,\, 2} = 5.991$과 비교한다. $0.767 < 5.991$이므로 $H_0$을 기각하지 못한다($p = 0.682$). 집단분산이 다르다고 결론지을 증거가 충분하지 않다.

## Python 검증

```python
import numpy as np
from scipy import stats

# Group data
n = np.array([10, 12, 8])
s2 = np.array([5.2, 8.1, 4.7])
k = len(n)
nu = n - 1
N = n.sum()

# Pooled variance
s2_pooled = np.sum(nu * s2) / np.sum(nu)

# Numerator
numerator = np.sum(nu) * np.log(s2_pooled) - np.sum(nu * np.log(s2))

# Correction factor
C = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / nu) - 1 / np.sum(nu))

# Test statistic
T = numerator / C

# p-value
p_value = stats.chi2.sf(T, k - 1)

print(f"Pooled variance: {s2_pooled:.4f}")
print(f"Numerator: {numerator:.4f}")
print(f"Correction factor C: {C:.4f}")
print(f"Test statistic T: {T:.4f}")
print(f"P-value: {p_value:.4f}")
```

출력:

```text
Pooled variance: 6.2519
Numerator: 0.8063
Correction factor C: 1.0513
Test statistic T: 0.7670
P-value: 0.6815
```


## 연습문제

**연습문제 1.**
$-2\ln\Lambda = (N-k)\ln S_p^2 - \sum_i \nu_i \ln S_i^2$을 정규 로그가능도에서 직접 유도하라.

??? success "풀이"
    집단 $i$의 정규 로그가능도는(평균을 $\bar{X}_i$로 최적화한 뒤)

    $$
    \ell_i(\sigma_i^2) = -\frac{n_i}{2}\ln(2\pi) - \frac{n_i}{2}\ln \sigma_i^2 - \frac{1}{2\sigma_i^2}\sum_j (X_{ij}-\bar{X}_i)^2.
    $$

    $\sum_j (X_{ij}-\bar{X}_i)^2 = \nu_i S_i^2$로 쓰고 $\sigma_i^2$에 대해 미분하여 0으로 두면

    $$
    -\frac{n_i}{2\sigma_i^2} + \frac{\nu_i S_i^2}{2\sigma_i^4} = 0 \implies \hat{\sigma}_i^2 = \frac{\nu_i S_i^2}{n_i}.
    $$

    **대립가설 아래.** 각 집단이 자기 추정량을 쓰므로 최대화된 로그가능도는 상수를 빼고

    $$
    \ell_1 = -\frac{1}{2}\sum_i n_i \ln \hat{\sigma}_i^2 - \frac{N}{2}.
    $$

    **귀무가설 아래.** 공통 분산 하나뿐이므로

    $$
    \hat{\sigma}^2 = \frac{\sum_i \nu_i S_i^2}{N} = \frac{(N-k)S_p^2}{N}, \qquad \ell_0 = -\frac{N}{2}\ln \hat{\sigma}^2 - \frac{N}{2}.
    $$

    따라서

    $$
    -2\ln\Lambda = 2(\ell_1 - \ell_0) = N \ln \hat{\sigma}^2 - \sum_i n_i \ln \hat{\sigma}_i^2.
    $$

    **$\nu_i$ 버전으로의 이행.** 위 식은 MLE(분모 $n_i$)를 쓰지만 Bartlett 통계량은 불편추정량(분모 $\nu_i$)을 쓴다. $\hat{\sigma}_i^2 = (\nu_i/n_i)S_i^2$을 대입하면

    $$
    \sum_i n_i \ln \hat{\sigma}_i^2 = \sum_i n_i \ln S_i^2 + \sum_i n_i \ln(\nu_i/n_i)
    $$

    이고 두 번째 합은 자료에 의존하지 않는 상수이다. 실제로 쓰이는 형태는 $n_i$를 $\nu_i$로 바꾼

    $$
    -2\ln\Lambda = (N-k)\ln S_p^2 - \sum_i \nu_i \ln S_i^2
    $$

    이다. 이 치환은 자유도를 올바르게 반영하여 유한표본 성질을 개선하며, Bartlett 보정과 함께 쓰일 때 $\chi^2_{k-1}$ 근사가 가장 정확해진다. $\square$

---

**연습문제 2.**
$-2\ln\Lambda \geq 0$임을 Jensen 부등식으로 증명하라.

??? success "풀이"
    $w_i = \nu_i/(N-k)$라 하면 $\sum_i w_i = 1$이고 $w_i > 0$이므로 $\{w_i\}$가 확률분포를 이룬다.

    $$
    -2\ln\Lambda = (N-k)\left[\ln S_p^2 - \sum_i w_i \ln S_i^2\right] = (N-k)\left[\ln\left(\sum_i w_i S_i^2\right) - \sum_i w_i \ln S_i^2\right].
    $$

    $\ln$이 **오목**함수이므로 Jensen 부등식에 의해

    $$
    \ln\left(\sum_i w_i S_i^2\right) \geq \sum_i w_i \ln S_i^2.
    $$

    따라서 대괄호 안이 음이 아니고 $(N-k) > 0$이므로 $-2\ln\Lambda \geq 0$이다.

    **등호 조건.** $\ln$이 **엄격히** 오목하므로 Jensen 부등식의 등호는 $S_i^2$이 (확률 1로) 상수일 때, 곧 모든 $i$에 대해 $S_i^2$이 같을 때만 성립한다.

    **일반적 의미.** 이는 가능도비 검정 일반의 성질을 반영한다. $-2\ln\Lambda \geq 0$은 제약 없는 최대가능도가 제약된 최대가능도보다 항상 크거나 같기 때문이며, 이 검정이 항상 오른쪽 꼬리 단측검정인 이유이기도 하다. $\square$

---

**연습문제 3.**
Bartlett 보정이 실제로 유한표본 크기를 개선하는지 모의실험으로 확인하라. 세 집단, 각 $n = 5$인 정규 자료에서 보정 있는 통계량과 없는 통계량의 경험적 크기를 비교하라.

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    k, n, R, alpha = 3, 5, 20000, 0.05
    nu = n - 1
    N = k * n
    C = 1 + (1 / (3 * (k - 1))) * (k / nu - 1 / (N - k))
    crit = stats.chi2.ppf(1 - alpha, k - 1)

    rej_raw = rej_corr = 0
    for _ in range(R):
        s2 = np.array([rng.normal(0, 1, n).var(ddof=1) for _ in range(k)])
        sp2 = s2.mean()                      # equal nu, so simple mean
        raw = (N - k) * np.log(sp2) - nu * np.sum(np.log(s2))
        rej_raw += (raw > crit)
        rej_corr += (raw / C > crit)

    print(f"Correction factor C = {C:.4f}")
    print(f"Uncorrected size: {rej_raw / R:.4f}")
    print(f"Corrected size:   {rej_corr / R:.4f}")
    ```

    출력:

    ```text
    Correction factor C = 1.1111
    Uncorrected size: 0.0672
    Corrected size:   0.0505
    ```

    보정하지 않은 통계량의 크기가 $0.067$로 명목값보다 34% 크다. 보정 후에는 $0.050$으로 명목값과 사실상 일치한다(몬테카를로 표준오차 0.0015).

    $n = 5$에서 $C = 1 + \frac{k+1}{3k(n-1)} = 1 + \frac{4}{36} = 1.1111$이다. 이렇게 작은 표본에서 점근 $\chi^2$ 근사가 부정확하며, Bartlett의 보정이 그 오차를 거의 완전히 제거함을 보여준다.

    **주의.** 이 실험은 자료가 **정규일 때**만 보정이 작동함을 보여줄 뿐이다. 비정규 자료에서는 보정이 있어도 크기가 통제되지 않는다. 보정은 유한표본 근사 오차를 고치는 것이지 모형 오설정을 고치는 것이 아니다. $\square$

---

**연습문제 4.**
본문 예제에서 집단 2의 분산을 $S_2^2 = 8.1$에서 점점 키워가며 $T$가 임계값 $5.991$을 넘는 지점을 찾아라. 이것이 Bartlett 검정의 검정력에 대해 무엇을 말해 주는가?

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    n = np.array([10, 12, 8])
    k, nu, N = 3, n - 1, n.sum()
    C = 1 + (1 / (3 * (k - 1))) * (np.sum(1 / nu) - 1 / np.sum(nu))
    crit = stats.chi2.ppf(0.95, k - 1)

    print(f"{'s2_2':>8} {'T':>8} {'p':>8} {'ratio to others':>18}")
    for s2_2 in [8.1, 15, 21, 25, 40, 60, 80]:
        s2 = np.array([5.2, s2_2, 4.7])
        sp2 = np.sum(nu * s2) / np.sum(nu)
        T = (np.sum(nu) * np.log(sp2) - np.sum(nu * np.log(s2))) / C
        p = stats.chi2.sf(T, k - 1)
        print(f"{s2_2:>8.1f} {T:>8.3f} {p:>8.4f} {s2_2 / 5.0:>18.1f}")
    ```

    출력:

    ```text
        s2_2        T        p    ratio to others
         8.1    0.767   0.6815                1.6
        15.0    3.856   0.1454                3.0
        21.0    6.468   0.0394                4.2
        25.0    8.045   0.0179                5.0
        40.0   12.938   0.0016                8.0
        60.0   17.761   0.0001               12.0
        80.0   21.438   0.0000               16.0
    ```

    이분법으로 정확한 교차점을 찾으면 $S_2^2 = 19.85$에서 $T = 5.9915$가 되어 임계값과 일치한다. 곧 집단 2의 분산이 나머지 두 집단(약 5)의 **네 배**를 넘어야 5% 수준에서 기각한다.

    **검정력에 대한 함의.** 총 $N = 30$개의 관측값으로는 4배의 분산 차이도 겨우 탐지한다. 표준편차로 환산하면 2배이다.

    이는 15.3절 연습문제 3에서 F 검정에 대해 본 것과 같은 결론이다. **분산 검정은 검정력이 근본적으로 낮다.** 실무에서 사전검정이 기각하지 못했다는 사실만으로 등분산성을 확신해서는 안 된다.

    표를 더 보면 $T$가 분산비의 로그에 거의 선형으로 반응한다는 점도 확인할 수 있다. 분산비가 3배에서 16배로 늘어날 때($\ln$ 기준 1.10에서 2.77로 2.5배) $T$는 3.86에서 21.44로 5.6배 늘었다. Bartlett 통계량이 로그 척도에서 작동하고 그 제곱 규모로 커지기 때문이다. $\square$
