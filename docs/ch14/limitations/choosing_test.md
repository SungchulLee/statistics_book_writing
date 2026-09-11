# 올바른 검정 고르기

## 검정 선택이 중요한 이유

정규성 검정에는 여러 가지가 있으며 각각 강점과 민감도가 다르다. Shapiro-Wilk 검정, Anderson-Darling 검정, Kolmogorov-Smirnov 검정, Jarque-Bera 검정은 모두 정규성을 평가하지만, 어떤 유형의 이탈을 가장 효과적으로 탐지하는지, 어떤 표본크기를 감당하는지, 표준 소프트웨어에서 어떻게 구현되어 있는지가 다르다. 올바른 검정의 선택은 표본크기, 의심되는 이탈의 유형, 추론의 맥락에 달려 있다.

## 흔한 정규성 검정 개관

### Shapiro-Wilk 검정

Shapiro-Wilk 검정은 정렬된 표본값이 정규분포의 기대 순서통계량과 얼마나 잘 맞는지를 재는 검정통계량 $W$를 계산하여 귀무가설 $H_0$: "자료가 정규분포에서 왔다"를 평가한다. 통계량은

$$
W = \frac{\left(\sum_{i=1}^{n} a_i X_{(i)}\right)^2}{\sum_{i=1}^{n} (X_i - \bar{X})^2}
$$

여기서 가중치 $a_i$는 정규 순서통계량의 공분산행렬에서 유도된다. $W$가 1에 가까우면 정규성을, 작으면 이탈을 나타낸다.

**강점:** 정규성 이탈을 탐지하는 데 대체로 가장 강력하며, 특히 작거나 중간 크기의 표본($n \leq 50$)에서 그렇다. 치우침과 두꺼운 꼬리 모두에 민감하다.

**한계:** 대부분의 구현이 $n$을 최대 5000으로 제한한다. 이탈의 유형(치우침 대 첨도)을 알려주지 않는다.

### Anderson-Darling 검정

Anderson-Darling 검정은 경험분포함수(EDF)에 기초한다. EDF $F_n(x)$와 가정한 정규 누적분포함수 $\Phi(x)$ 사이의 거리를 가중하여 잰다.

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i - 1}{n} \left[\ln \Phi(Z_{(i)}) + \ln\left(1 - \Phi(Z_{(n+1-i)})\right)\right]
$$

여기서 $Z_{(i)} = (X_{(i)} - \bar{X}) / S$는 표준화된 순서통계량이다. Anderson-Darling 통계량은 Kolmogorov-Smirnov 검정보다 분포의 꼬리에 더 큰 가중치를 준다.

**강점:** KS 검정보다 꼬리 이탈에 민감하다. 더 큰 표본크기에도 쓸 수 있다. 다양한 대립가설에서 좋은 성능을 낸다.

**한계:** 작은 표본에서는 Shapiro-Wilk보다 조금 덜 강력하다. 임계값이 모수를 추정했는지 알고 있는지에 따라 달라진다.

### Kolmogorov-Smirnov 검정

Kolmogorov-Smirnov(KS) 검정은 EDF와 가정한 누적분포함수 사이의 최대 절대거리를 잰다.

$$
D = \sup_x \left| F_n(x) - \Phi(x) \right|
$$

모수를 추정한 정규분포와 비교할 때는 반드시 Lilliefors 보정을 적용해야 한다. 표준 KS 임계값이 모수가 완전히 지정되었다고 가정하기 때문이다.

**강점:** 개념이 단순하다. 임의의 연속분포에 적용할 수 있다. 표본크기 제한이 없다.

**한계:** 정규성 위배 탐지에서 Shapiro-Wilk와 Anderson-Darling보다 대체로 검정력이 낮다. 분포 중앙의 이탈에 가장 민감하고 꼬리의 이탈에는 상대적으로 둔감하다.

### Jarque-Bera 검정

Jarque-Bera 검정은 표본왜도와 표본첨도에 기초한다. 검정통계량은

$$
JB = \frac{n}{6}\left(\hat{\gamma}^2 + \frac{(\hat{\kappa} - 3)^2}{4}\right)
$$

여기서 $\hat{\gamma}$는 표본왜도, $\hat{\kappa}$는 표본첨도이다. $H_0$ 아래에서 $n \to \infty$일 때 $JB \overset{d}{\to} \chi^2_2$이다.

**강점:** 정규성 이탈의 가장 흔한 두 유형인 치우침과 첨도를 직접 겨냥한다. 계산이 단순하다. 계량경제학과 금융에서 널리 쓰인다.

**한계:** 점근적 검정이므로 작은 표본($n < 30$)에서는 믿을 수 없다. 왜도와 첨도에 영향을 주지 않는 이탈(예: 대칭이고 중첨인 성분으로 이루어진 이봉분포)에는 둔감하다.

## 비교표

| 검정 | 적합한 표본크기 | 민감한 대상 | 꼬리 민감도 | 계산 비용 |
|---|---|---|---|---|
| Shapiro-Wilk | $n \leq 5000$ | 일반적 이탈 | 높음 | 중간 |
| Anderson-Darling | 제한 없음 | 꼬리 이탈 | 매우 높음 | 낮음 |
| Kolmogorov-Smirnov (Lilliefors) | 제한 없음 | 중앙의 이탈 | 낮음 | 낮음 |
| Jarque-Bera | $n \geq 30$ | 왜도와 첨도 | 중간 | 매우 낮음 |

## 판단의 틀

다음 지침이 적절한 검정을 고르는 데 도움이 된다.

**1단계: 표본크기를 고려한다.**

- $n < 30$이면 Shapiro-Wilk 검정을 쓴다. 작은 표본에서 검정력이 가장 좋다. Q-Q 그림으로 보완하라.
- $30 \leq n \leq 5000$이면 Shapiro-Wilk가 여전히 강력한 기본 선택이다. 꼬리 거동이 주된 관심사라면 Anderson-Darling 검정이 좋은 대안이다.
- $n > 5000$이면 (표본크기 제한이 없는) Anderson-Darling 검정을 쓴다. 왜도나 첨도가 관심사라면 Jarque-Bera 검정도 적절하다.

**2단계: 의심되는 이탈을 고려한다.**

- 두꺼운 꼬리가 의심되면(예: 금융 자료) Anderson-Darling이나 Jarque-Bera를 선호한다. KS 검정은 꼬리에서 너무 둔감하다.
- 치우침이 주된 관심사면 Jarque-Bera가 그것을 직접 검정한다.
- 이탈의 유형을 모르면 Shapiro-Wilk가 가장 폭넓게 강력한 선택이다.

**3단계: 추론의 맥락을 고려한다.**

- 후속 분석이 $t$ 검정이나 분산분석이면 범용 검정(Shapiro-Wilk)이 적절하다.
- 후속 분석이 분산이나 꼬리 위험과 관련되면 꼬리에 민감한 검정(Anderson-Darling)이 더 적합하다.

??? tip "언제나 시각적 방법으로 보완하라"
    어떤 정규성 검정도 시각적 평가를 대체하지 못한다. Q-Q 그림은 이탈의 유형과 위치(꼬리, 중앙, 치우침)를 드러내고 히스토그램은 전반적 모양을 보여준다. 형식적 검정과 Q-Q 그림의 조합이 가장 유익한 평가를 제공한다.

## Python 예제

```python
import numpy as np
from scipy import stats

# ===================================================================
# Run multiple normality tests on the same dataset and compare results
# ===================================================================

np.random.seed(42)
n = 100

# Generate data from a t-distribution (heavy tails)
data = stats.t.rvs(df=5, size=n)

if __name__ == "__main__":
    # Shapiro-Wilk
    sw_stat, sw_p = stats.shapiro(data)
    print(f"Shapiro-Wilk:      W = {sw_stat:.4f}, p = {sw_p:.4f}")

    # Anderson-Darling
    ad_result = stats.anderson(data, dist="norm")
    print(f"Anderson-Darling:  A2 = {ad_result.statistic:.4f}, "
          f"critical (5%) = {ad_result.critical_values[2]:.4f}")

    # Kolmogorov-Smirnov (Lilliefors via kstest with estimated params)
    ks_stat, ks_p = stats.kstest(data, "norm", args=(np.mean(data), np.std(data)))
    print(f"KS (estimated):    D = {ks_stat:.4f}, p = {ks_p:.4f}")

    # Jarque-Bera
    jb_stat, jb_p = stats.jarque_bera(data)
    print(f"Jarque-Bera:       JB = {jb_stat:.4f}, p = {jb_p:.4f}")
```

출력:

```text
Shapiro-Wilk:      W = 0.6287, p = 0.0000
Anderson-Darling:  A2 = 6.0914, critical (5%) = 0.7590
KS (estimated):    D = 0.1783, p = 0.0030
Jarque-Bera:       JB = 5365.9041, p = 0.0000
```

이 표본은 왜도 $4.83$, 초과첨도 $34.56$으로 매우 극단적인 값이 섞여 있어 네 검정이 모두 기각한다. 다만 **증거의 강도가 크게 다르다**. Shapiro-Wilk와 Jarque-Bera의 $p$값은 사실상 0이고 Anderson-Darling 통계량 $6.09$는 5% 임계값 $0.759$의 여덟 배인 반면, KS 검정의 $p$값은 $0.003$으로 상대적으로 가장 약한 증거를 준다. 같은 자료에서도 KS가 가장 둔감함을 보여준다.

## 요약

보편적으로 최적인 정규성 검정은 없다. Shapiro-Wilk 검정은 폭넓은 검정력 덕분에 작거나 중간 크기의 표본에서 가장 나은 기본 선택이다. Anderson-Darling 검정은 꼬리 이탈 탐지에 뛰어나고 표본크기 제한이 없다. Jarque-Bera 검정은 왜도나 첨도가 의심될 때 효율적이지만 최소한 중간 크기의 표본이 필요하다. Kolmogorov-Smirnov 검정은 널리 알려져 있지만 정규성 검정으로는 대체로 검정력이 가장 낮다. 어떤 경우에도 형식적 검정에는 시각적 진단이 따라야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
어떤 연구자가 $n = 25$인 표본의 정규성을 검정해야 한다. 검정을 추천하고 이유를 설명하라.

</div>

??? success "풀이"
    $n = 25$에서는 **Shapiro-Wilk 검정**을 권장한다. 작거나 중간 크기의 표본에서 정규성 검정 가운데 검정력이 가장 높고 폭넓은 이탈(치우침, 두꺼운 꼬리, 다봉성)을 탐지한다.

    꼬리 이탈이 주된 관심사라면 Anderson-Darling 검정이 합리적인 대안이다. 적률 기반 검정(D'Agostino, Jarque-Bera)은 $n = 25$에서 검정력이 부족하므로 피해야 한다. KS/Lilliefors 검정은 Shapiro-Wilk와 Anderson-Darling 둘 다보다 검정력이 낮다.

    형식적 검정은 언제나 Q-Q 그림으로 보완하여 시각적으로 평가하라.

<div class="drillbox" markdown>

**연습문제 2.**
큰 자료($n = 10{,}000$)에서 형식적 정규성 검정이 도움이 되지 않을 수 있는 이유와 더 나은 접근을 설명하라.

</div>

??? success "풀이"
    $n = 10{,}000$이면 모든 정규성 검정의 검정력이 극도로 높아, 추론에 실질적 영향이 없는 사소한 이탈에도 정규성을 기각한다. 기각은 자료가 *정확히* 정규는 아니라는 사실(실제 자료에서는 언제나 참이다)을 알려줄 뿐 그 이탈이 문제가 되는지는 알려주지 않는다.

    더 나은 접근: (1) Q-Q 그림으로 비정규성의 정도를 시각적으로 평가한다. (2) 왜도와 첨도를 계산해 이탈을 수량화한다. (3) 그 이탈이 의도한 분석에 실질적으로 관련 있는지 평가한다(예: 신뢰구간의 포함확률이나 검정의 크기에 영향을 주는가?). (4) 민감도 확인으로 정규론 방법의 결과를 로버스트한 대안의 결과와 비교한다.

<div class="drillbox" markdown>

**연습문제 3.**
표본크기와 의심되는 이탈 유형에 따라 정규성 검정을 고르는 판단 흐름도를 만들어라.

</div>

??? success "풀이"

    1. **언제나 Q-Q 그림으로 시작한다**(모든 $n$).
    2. $n < 50$이면 **Shapiro-Wilk**를 쓴다(전반적 검정력이 가장 좋다).
    3. $50 \leq n \leq 5000$이면 **Shapiro-Wilk**나 **Anderson-Darling**을 쓴다(꼬리 특유의 이탈에는 AD가 낫다).
    4. $n > 5000$이면 형식적 검정이 지나치게 강력하므로 **Q-Q 그림**, **왜도/첨도** 값, 실질적 유의성 평가에 의존한다.
    5. **치우침이 구체적으로** 의심되면 `skewtest`로 보완한다.
    6. **두꺼운 꼬리가 구체적으로** 의심되면 `kurtosistest`나 (꼬리에 가중치를 주는) Anderson-Darling으로 보완한다.
    7. **특정 대립가설**(예: $t$ 분포)을 검정한다면 그 분포에 대한 Q-Q 그림을 쓴다.

<div class="drillbox" markdown>

**연습문제 4.**
두 검정이 엇갈린다. Shapiro-Wilk는 정규성을 기각하고($p = 0.03$) Anderson-Darling은 기각하지 않는다($p = 0.08$). 어떻게 진행해야 하는가?

</div>

??? success "풀이"
    검정마다 민감도가 다르므로 결과가 엇갈리는 일은 드물지 않다. Shapiro-Wilk는 전반적으로 더 강력하고 Anderson-Darling은 꼬리 이탈에 더 민감하다. 이 엇갈림은 비정규성이 가볍고 꼬리보다는 중앙에 몰려 있을 가능성을 시사한다.

    진행 방법: (1) Q-Q 그림을 살펴 이탈의 성격과 정도를 확인한다. (2) 이탈이 가벼우면(Q-Q 그림이 거의 선형이면) 정규론 방법으로 진행한다. (3) 비모수 대안으로 민감도 분석을 수행한다. (4) 투명성을 위해 두 검정 결과와 시각적 평가를 모두 보고한다.
