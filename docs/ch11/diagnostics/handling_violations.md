# 가정 위반의 처리

## 개요

진단 결과 분산분석의 가정이 하나 이상 어긋난 것으로 드러나면, 타당한 결론을 얻기 위해 시정 조치를 취해야 한다. 적절한 대응은 위반의 성격과 심각성에 따라 달라진다. 이 절은 위반 유형별로 대처하는 체계적인 지침을 제공한다.

## 단계별 접근

1. **원인 파악:** 앞 절들에서 설명한 진단 도구로 어느 가정이 어느 정도로 어긋났는지 판정한다.
2. **심각성 평가:** 표본이 크고 균형 잡혀 있으면 가벼운 위반은 결과에 거의 영향을 주지 않을 수 있다. 심한 위반은 시정 조치가 필요하다.
3. **처방 선택:** 구체적인 위반에 따라 아래 선택지 중에서 고른다.
4. **개선 확인:** 보정을 적용한 뒤 진단을 다시 수행하여 가정이 이제 충족되는지 확인한다.

## 비모수 대안

### Kruskal-Wallis 검정

정규성 가정이 어긋날 때 Kruskal-Wallis 검정은 일원배치 분산분석의 비모수 대안이 된다. 집단 사이에서 평균 대신 중앙값(더 정확히는 평균 순위)을 비교하며 잔차의 정규성을 가정하지 않는다.

```python
from scipy.stats import kruskal

stat, p_value = kruskal(group1, group2, group3)
print(f"Kruskal-Wallis: H = {stat:.4f}, p-value = {p_value:.4f}")
```

Kruskal-Wallis 검정은 이상점이나 치우친 분포에 덜 민감하지만, 분포의 모양이 같고 위치만 다르다고 가정한다. 자세한 내용은 [Kruskal-Wallis 검정](../../ch16/multi_group_nonparametric/kruskal_wallis.md)을 보라.

## 자료 변환

변환은 자료의 척도를 바꾸어 정규성과 등분산성 위반을 한꺼번에 다룰 수 있다.

### 로그 변환

자료가 양의 방향으로 치우쳐 있거나 분산이 평균과 함께 커질 때 쓴다:

$$
Y' = \log(Y) \quad \text{or} \quad Y' = \log(Y + c) \text{ if } Y \text{ contains zeros}
$$

```python
import numpy as np

data['log_response'] = np.log(data['response'])
```

### 제곱근 변환

Poisson 계열 분포를 따르는 도수 자료에 유용하다:

$$
Y' = \sqrt{Y}
$$

```python
data['sqrt_response'] = np.sqrt(data['response'])
```

### Box-Cox 변환

$\lambda$로 모수화된 거듭제곱 변환의 족으로, 정규성에 가장 가까워지도록 $\lambda$를 최적화할 수 있다:

$$
Y'(\lambda) = \begin{cases} \frac{Y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\ \log(Y) & \text{if } \lambda = 0 \end{cases}
$$

```python
from scipy.stats import boxcox

transformed_data, best_lambda = boxcox(data['response'])
print(f"Optimal lambda = {best_lambda:.4f}")
```

!!! note "변환 후의 해석"
    자료를 변환하면 분산분석은 원래 평균이 아니라 변환된 평균에 관한 가설을 검정한다. 결과를 해석하고 보고할 때 주의하라. 가능하면 추정값을 역변환하고, 어떤 척도에서 분석했는지 분명히 밝혀야 한다.

## 로버스트 분산분석 방법

### Welch 분산분석

Welch 분산분석은 집단 사이의 등분산을 가정하지 않는다. Welch-Satterthwaite 근사로 F-검정의 자유도를 조정한다:

```python
from scipy.stats import f_oneway
# Or use pingouin for Welch's ANOVA directly
import pingouin as pg

welch_result = pg.welch_anova(dv='response', between='group', data=data)
print(welch_result)
```

전체 논의는 [Welch의 일원배치 분산분석](../anova_welch/welch_one_way.md)을 보라.

### 로버스트 추정량

Huber나 M-추정량 같은 방법은 이상점에 덜 민감한 분산분석 유사 결과를 준다:

```python
import statsmodels.api as sm

rlm_model = sm.RLM.from_formula('response ~ group', data=data, M=sm.robust.norms.HuberT())
result = rlm_model.fit()
print(result.summary())
```

## 순열검정

순열검정은 분포에 대한 가정을 최소한으로만 둔다. 작동 방식은 다음과 같다:

1. 관측된 F-통계량을 계산한다.
2. 집단 표시를 여러 번 무작위로 섞는다.
3. 각 순열마다 F-통계량을 다시 계산한다.
4. 관측된 F-통계량을 순열분포와 비교한다.

```python
import numpy as np
from scipy.stats import f_oneway

# Observed F-statistic
observed_f, _ = f_oneway(group1, group2, group3)

# Permutation test
all_data = np.concatenate([group1, group2, group3])
group_sizes = [len(group1), len(group2), len(group3)]
n_permutations = 10000
perm_f_stats = []

rng = np.random.default_rng(42)
for _ in range(n_permutations):
    shuffled = rng.permutation(all_data)
    g1 = shuffled[:group_sizes[0]]
    g2 = shuffled[group_sizes[0]:group_sizes[0]+group_sizes[1]]
    g3 = shuffled[group_sizes[0]+group_sizes[1]:]
    f_stat, _ = f_oneway(g1, g2, g3)
    perm_f_stats.append(f_stat)

p_value = np.mean(np.array(perm_f_stats) >= observed_f)
print(f"Permutation test p-value: {p_value:.4f}")
```

자세한 내용은 [순열검정](../../ch17/permutation/foundations.md)을 보라.

## 위반별 처방 요약

| 위반 | 권장 처방 |
|-----------|---------------------|
| 비정규성 | 변환, Kruskal-Wallis, 붓스트랩 |
| 이분산 | Welch 분산분석, 변환, 로버스트 표준오차 |
| 비독립성 | 혼합효과 모형, 반복측정 분산분석, GEE |
| 비선형성 | 다항 항, 변환, GAM |
| 이상점/영향점 | 로버스트 추정량, 민감도 분석, 변환 |

## 연습문제

**연습문제 1.**
집단이 넷인 일원배치 분산분석에서 F-검정이 유의하게 나왔지만($p = 0.008$), Levene 검정이 등분산 귀무가설을 기각하고($p = 0.003$) 잔차에 대한 Shapiro-Wilk 검정은 유의하지 않다($p = 0.34$). 타당한 추론을 얻기 위한 단계별 계획을 제시하라.

??? success "풀이"

    1. **정규성:** Shapiro-Wilk 검정이 유의하지 않으므로 정규성은 문제가 아니다. 조치가 필요 없다.

    2. **등분산성:** Levene 검정이 등분산을 강하게 기각한다. 표준 분산분석 F-검정 결과를 믿을 수 없다.

    3. **권장 조치:** 등분산을 가정하지 않는 **Welch의 일원배치 분산분석**으로 다시 분석한다. Welch 분산분석도 유의하면 Tukey의 HSD 대신 (분산이 다를 때를 위해 설계된) **Games-Howell 사후검정**으로 이어간다.

    4. **선택 사항:** 분산 안정화 변환(예: 로그)을 시도하여 이분산이 해소되는지 확인한다. 해소되면 변환된 자료에 표준 분산분석을 써도 된다.

---

**연습문제 2.**
크기가 $n = 12, 15, 10$인 세 집단의 자료에서 정규성과 등분산성이 모두 어긋났다. 분석 전략을 권하고 각 선택의 근거를 밝혀라.

??? success "풀이"
    두 가정이 모두 어긋났을 때 선호되는 순서대로 선택지는 다음과 같다:

    1. **Kruskal-Wallis 검정.** 일원배치 분산분석의 비모수 대안으로 정규성이나 등분산을 가정하지 않는다. 평균 대신 중앙값 순위를 비교하며 순서형이거나 치우친 자료에 적합하다.

    2. **붓스트랩 분산분석.** 재표본추출로 분포 가정 없이 F-통계량의 귀무분포를 얻는다. 평균 비교의 틀을 유지하면서 가정을 완화한다.

    3. **변환 + Welch 분산분석.** 변환(예: 로그나 Box-Cox)으로 자료를 근사적으로 정규화할 수 있다면, 남은 이분산은 Welch 분산분석이 처리한다.

    표본크기가 서로 다르면 표준 분산분석이 이분산에 특히 민감해지므로 Welch 분산분석이나 비모수 방법을 쓸 근거가 더 강해진다.

---

**연습문제 3.**
Cook의 거리로 탐지한 이상점을 그냥 제거하는 것이 분산분석 진단에서 언제나 최선의 전략은 아닌 이유를 설명하라. 연구자는 대신 무엇을 해야 하는가?

??? success "풀이"
    이상점을 제거하면 **선택 편향**이 생기고 표본크기가 줄어들며, 모집단의 진짜 변동을 나타내는 타당한 관측값을 없앨 수도 있다. 연구자는 대신 다음을 해야 한다:

    1. **이상점을 조사한다.** 자료 입력 오류인지, 측정 장비의 오작동인지, 아니면 정당하게 극단적인 관측인지 판정한다.

    2. **민감도 분석을 수행한다.** 이상점을 포함한 경우와 제외한 경우로 분산분석을 수행하여 결과를 비교한다. 결론이 같으면 그 이상점은 영향점이 아니다.

    3. **로버스트 방법을 쓴다.** 절사평균, 윈저화 분산분석, M-추정량은 극단 관측값을 버리지 않으면서 가중치를 낮춘다.

    4. **두 분석을 모두 보고한다.** 결론이 다르면 이상점을 포함한 결과와 제외한 결과를 함께 보고하고 그 차이를 논의한다.
