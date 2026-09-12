# 회귀계수의 가설검정 (t-검정)

선형회귀에서 핵심 목표 가운데 하나는 각 설명변수가 종속변수에 미치는 영향을 평가하는 것이다. 보통 **t-검정**으로 수행하는 회귀계수의 가설검정은 설명변수가 모형에 유의하게 기여하는지를 판단한다.

---

## t-검정의 설정

선형회귀의 t-검정은 회귀계수가 0이라는 귀무가설, 곧 해당 설명변수가 종속변수에 아무 효과가 없다는 가설을 평가한다.

가설은 다음과 같다.

- **귀무가설 ($H_0$):** $\beta_i = 0$ — 설명변수에 효과가 없다.
- **대립가설 ($H_1$):** $\beta_i \neq 0$ — 설명변수에 유의한 효과가 있다.

각 계수에 대해 **t 통계량**을 계산한다.

$$
t = \frac{\hat{\beta}_i}{SE(\hat{\beta}_i)}
$$

여기서

- $\hat{\beta}_i$는 설명변수 $i$의 추정된 계수,
- $SE(\hat{\beta}_i)$는 추정값의 표준오차로, 추정의 변동을 나타낸다.

t 통계량은 추정된 계수가 0에서 표준오차 몇 개만큼 떨어져 있는지를 잰다. 절댓값이 클수록 귀무가설에 반하는 증거가 강하다.

---

## t-검정의 p값

**p값**은 귀무가설이 참이라고 가정했을 때 계산된 t 통계량만큼 또는 그보다 더 극단적인 값을 관측할 확률이다.

- **작은 p값 (< 0.05):** 계수가 0과 유의하게 다르다 — $H_0$을 기각한다. 그 설명변수는 종속변수에 유의한 영향을 준다.
- **큰 p값 ($\geq$ 0.05):** $H_0$을 기각하지 못한다. 그 설명변수는 종속변수에 유의한 영향을 주지 않을 수 있고, 포함해도 예측 성능이 나아지지 않을 수 있다.

!!! example "연봉 예측"
    경력 연수로 연봉을 예측하는 선형회귀 모형에서 t-검정은 "경력 연수"가 연봉에 유의한 효과를 갖는지 평가한다. 그 계수의 p값이 0.001이라면 경력 연수가 연봉의 유의한 설명변수라고 결론짓는다.

---

## 계수의 신뢰구간

p값이 유의성에 대한 이분법적 판정을 준다면, **신뢰구간**은 계수가 취할 만한 값의 범위를 준다. 95% 신뢰구간은 다음과 같이 계산한다.

$$
CI = \hat{\beta}_i \pm \left( t_{\alpha/2} \times SE(\hat{\beta}_i) \right)
$$

여기서 $t_{\alpha/2}$는 원하는 신뢰수준에서 t 분포의 임계값이다.

**해석:**

- 신뢰구간이 **0을 포함하면** 계수는 0과 유의하게 다르지 않다. 그 설명변수는 유의한 영향을 주지 않을 수 있다.
- 신뢰구간이 **0을 배제하면** 그 설명변수는 유의하며, 모형에 의미 있게 기여한다는 더 강한 증거가 된다.

!!! example "광고 예산"
    매출을 예측하는 회귀모형에서 "광고 예산"이라는 설명변수의 95% 신뢰구간이 $(0.03, 0.12)$라면 0을 포함하지 않으므로 광고 예산이 매출에 유의한 영향을 준다고 결론짓는다.

---

## 유의성의 해석

설명변수의 유의성은 t-검정의 결과와 그에 딸린 p값으로 정해진다.

**통계적으로 유의한 계수** ($p < 0.05$):

- 그 설명변수는 종속변수의 변동 일부를 설명하며 모형을 개선한다.
- 이 변수의 변화는 결과의 변화와 연관된다.
- 예를 들어 직무 성과를 예측하는 데 "교육 수준"이 유의하다면 교육 수준은 중요한 결정 요인이다.

**통계적으로 유의하지 않은 계수** ($p \geq 0.05$):

- 이 설명변수가 종속변수에 영향을 준다고 주장할 증거가 충분하지 않다.
- 전반적인 적합을 개선하지 못한다면 모형에서 뺄 수 있다.

---

## t-검정에서 다중공선성의 역할

**다중공선성**은 둘 이상의 설명변수가 강하게 상관되어 개별 효과를 구분하기 어려워지는 상황이다. 이는 다음을 일으킨다.

- 계수의 표준오차를 부풀려 t 통계량을 줄인다.
- p값을 부풀려 유의성에 대한 잘못된 결론을 낳는다.
- 신뢰구간을 넓혀 유의성을 결론짓기 어렵게 만든다.

**다중공선성에 대처하기:**

- **분산팽창인자(VIF):** VIF가 5나 10을 넘으면 다중공선성을 시사한다. 상관된 설명변수를 빼거나 결합하는 것이 대책이다.
- **주성분분석(PCA):** 설명변수를 서로 무상관인 성분으로 변환한다.
- **릿지 회귀:** 계수를 정칙화하여 다중공선성의 영향을 줄인다.

---

## 연습문제

!!! note "이 절의 연습문제에 대하여"
    아래 네 문제는 두 표본의 평균을 비교하는 $t$ 검정이다. 회귀계수의 $t$ 검정과 통계량의 구조($\hat{\theta}/SE(\hat{\theta})$)와 판정 규칙이 동일하므로, $t$ 검정의 기계적 절차를 익히는 연습으로 삼는다.

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
이탈리아와 프랑스의 교육을 연구하는 한 연구자가 두 나라 남성이 평균적으로 학교를 몇 년 다녔는지 비교하려 한다. 각 나라에서 남성을 무작위로 표본추출하여 다음을 얻었다.

| | 이탈리아 | 프랑스 |
|:---:|:---:|:---:|
| 평균 | 10.7 | 10.4 |
| 표준편차 | 2.3 | 2.5 |
| 표본 수 | 46 | 58 |

유의수준 $\alpha = 0.05$에서 두 나라의 평균 재학 연수에 유의한 차이가 있는지 검정하라.

</div>

??? success "풀이"

    두 표본 $t$ 검정(합동분산):

    $$
    H_0: \mu_A = \mu_B \quad \text{대} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 10.7, 10.4
        s_1, s_2 = 2.3, 2.5
        n_1, n_2 = 46, 58

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

    출력:

    ```
    df = 102.0000
    statistic = 0.6295
    p_value   = 0.5304
    We choose H_0, or using statistician's jargon, fail to reject H_0
    ```

    $p = 0.530$으로 기각하지 못한다. 자유도 102는 합동 $t$-검정의 $n_1 + n_2 - 2$다.

    $p = 0.530 > 0.05$이므로 $H_0$을 기각하지 못한다. 두 나라의 평균 재학 연수가 다르다는 증거가 없다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
노르웨이와 미국의 소득을 연구하는 한 경제학자가 두 나라의 연평균 소득을 비교하려 한다. 각 나라에서 사람들을 무작위로 표본추출하여 천 달러 단위로 소득을 얻었다.

| | 노르웨이 | 미국 |
|:---:|:---:|:---:|
| 평균 | 64.3 | 53.4 |
| 표준편차 | 18.2 | 23.9 |
| 표본 수 | 65 | 75 |

유의수준 $\alpha = 0.05$에서 두 나라의 평균 연소득에 유의한 차이가 있는지 검정하라.

</div>

??? success "풀이"

    두 표본 $t$ 검정(Welch, 비합동분산):

    $$
    H_0: \mu_A = \mu_B \quad \text{대} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 64.3, 53.4
        s_1, s_2 = 18.2, 23.9
        n_1, n_2 = 65, 75

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / (n_1 - 1) + (s_2**2 / n_2)**2 / (n_2 - 1)
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

    출력:

    ```
    df = 135.8395
    statistic = 3.0572
    p_value   = 0.0027
    We choose H_1, or using statistician's jargon, reject H_0
    ```

    $p = 0.0027$로 기각한다. 자유도가 135.8로 정수가 아닌 것은 Welch 자유도이기 때문이다.

    $p = 0.0027 < 0.05$이므로 $H_0$을 기각한다. 두 나라의 평균 연소득에 유의한 차이가 있다.

    !!! warning "Welch 자유도 공식"
        Welch–Satterthwaite 자유도의 분모는

        $$
        \frac{(s_1^2/n_1)^2}{n_1 - 1} + \frac{(s_2^2/n_2)^2}{n_2 - 1}
        $$

        이다. 분모를 $n_1$, $n_2$로 나누는 실수를 자주 보는데, 표본이 크면 차이가 미미하지만(여기서는 135.84 대 137.77) 표본이 작으면 자유도를 크게 부풀려 $p$값을 과소평가하게 된다. 연습문제 4가 그 예이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
미국과 캐나다의 결혼을 연구하는 한 사회학자가 두 나라 여성이 처음 결혼했을 때의 평균 나이를 비교하려 한다. 각 나라에서 기혼 여성을 무작위로 표본추출하여 다음을 얻었다.

| | 미국 | 캐나다 |
|:---:|:---:|:---:|
| 평균 | 25.5 | 26.3 |
| 표준편차 | 3.8 | 3.2 |
| 표본 수 | 108 | 102 |

유의수준 $\alpha = 0.05$에서 두 나라의 초혼 평균 연령에 유의한 차이가 있는지 검정하라.

</div>

??? success "풀이"

    두 표본 $t$ 검정(합동분산):

    $$
    H_0: \mu_A = \mu_B \quad \text{대} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 25.5, 26.3
        s_1, s_2 = 3.8, 3.2
        n_1, n_2 = 108, 102

        s_p_square = ((n_1-1) * s_1**2 + (n_2-1) * s_2**2) / (n_1 + n_2 - 2)
        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_p_square / n_1 + s_p_square / n_2)
        df = n_1 + n_2 - 2
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

    출력:

    ```
    df = 208.0000
    statistic = -1.6454
    p_value   = 0.1014
    We choose H_0, or using statistician's jargon, fail to reject H_0
    ```

    $p = 0.101$로 5% 수준에서는 기각하지 못한다. 경계에 가까운 값이므로 "차이가 없다"가 아니라 "이 표본으로는 판단하기 어렵다"로 읽어야 한다.

    $p = 0.101 > 0.05$이므로 $H_0$을 기각하지 못한다. 표본평균의 차이 0.8년은 이 표본크기에서 우연으로 설명될 수 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
Julie는 새로 나온 전기차 두 모델 A와 B가 완전 충전 후 얼마나 멀리 갈 수 있는지 시험했다. 각 모델의 새 차 5대씩을 표본으로 얻어 완전히 충전한 뒤 통제된 경로에서 최대한 멀리 주행했다.

| | 모델 A | 모델 B |
|:---:|:---:|:---:|
| 평균 | 168 km | 172 km |
| 표준편차 | 5.4 km | 7.5 km |
| 표본 수 | 5 | 5 |

유의수준 $\alpha = 0.05$에서 두 모델의 평균 주행거리에 유의한 차이가 있는지 검정하라.

</div>

??? success "풀이"

    두 표본 $t$ 검정(Welch, 비합동분산):

    $$
    H_0: \mu_A = \mu_B \quad \text{대} \quad H_1: \mu_A \neq \mu_B
    $$

    ```python
    import numpy as np
    from scipy import stats

    def main():
        X_1_bar, X_2_bar = 168, 172
        s_1, s_2 = 5.4, 7.5
        n_1, n_2 = 5, 5

        statistic = (X_1_bar - X_2_bar) / np.sqrt(s_1**2 / n_1 + s_2**2 / n_2)
        top = (s_1**2 / n_1 + s_2**2 / n_2)**2
        bottom = (s_1**2 / n_1)**2 / (n_1 - 1) + (s_2**2 / n_2)**2 / (n_2 - 1)
        df = top / bottom
        p_value = 2 * stats.t(df).cdf(-abs(statistic))
        print(f"{df = :.4f}")
        print(f"{statistic = :.4f}")
        print(f"{p_value   = :.4f}")

        alpha = 0.05
        if p_value <= alpha:
            print("We choose H_1, or using statistician's jargon, reject H_0")
        else:
            print("We choose H_0, or using statistician's jargon, fail to reject H_0")

    if __name__ == "__main__":
        main()
    ```

    출력:

    ```
    df = 7.2688
    statistic = -0.9678
    p_value   = 0.3642
    We choose H_0, or using statistician's jargon, fail to reject H_0
    ```

    $p = 0.364$로 기각하지 못한다. 표본이 작아 자유도가 7.27밖에 안 되는 것이 결정적이다.

    $p = 0.364 > 0.05$이므로 $H_0$을 기각하지 못한다. 각 모델 5대로는 4 km의 차이를 탐지할 검정력이 거의 없다.

    자유도 분모에 $n-1$ 대신 $n$을 쓰면 자유도가 7.27이 아니라 9.09가 되고 $p$값도 0.364가 아닌 0.358이 된다. 결론은 같지만, 표본이 작을수록 이 오류의 영향이 커진다.

---

## 정리하며

회귀계수의 가설검정(t-검정)은 개별 설명변수의 유의성을 판단하는 데 결정적이다. p값과 신뢰구간을 평가하여 어떤 설명변수가 종속변수의 변동을 설명하는 데 유의하게 기여하는지 가려낸다. 설명변수의 참된 유의성을 가릴 수 있는 다중공선성을 확인하는 일도 중요하다.
