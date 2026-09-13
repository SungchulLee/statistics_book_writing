# 기울기의 신뢰구간 (카페인 예제)

## 개요

이 페이지는 단순선형회귀에서 기울기 계수의 신뢰구간을 어떻게 구성하는지 보인다. 공부 시간과 카페인 섭취의 관계를 살핀 학생 20명의 연구를 이용해 $t$ 분포로 오차한계를 계산하고 $\beta_1$의 95% 신뢰구간을 만든다.

## 수학적 배경

단순선형회귀 모형은

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i, \qquad \varepsilon_i \overset{\text{iid}}{\sim} N(0, \sigma^2).
$$

최소제곱 추정량 $\hat{\beta}_1$의 표집분포는

$$
\hat{\beta}_1 \sim N\!\left(\beta_1,\; \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2}\right).
$$

$\sigma^2$을 모르므로 잔차분산 $s^2$으로 대체하고 자유도 $n - 2$의 $t$ 분포를 쓴다. $\beta_1$의 $(1 - \alpha)$ 수준 신뢰구간은

$$
\hat{\beta}_1 \pm t^*_{n-2,\,\alpha/2} \cdot \mathrm{SE}(\hat{\beta}_1),
$$

여기서 $t^*_{n-2,\,\alpha/2}$는 $t$ 분포의 임계값이다.

다음 코드는 카페인 연구에서 기울기의 95% 신뢰구간을 계산한다.

<div class="codebox" markdown>

**예제 1.** 기울기의 신뢰구간

```python
from scipy import stats

# 회귀 출력표에서 그대로 읽은 값이다.
beta_1_hat = 0.164      # 추정된 기울기
standard_error = 0.057  # 기울기의 표준오차

# 자유도는 n-2 다. 절편과 기울기를 자료에서 추정했기 때문이다.
n = 20
df = n - 2

# 표본이 20 으로 작아 t 임계값이 정규의 1.96 보다 눈에 띄게 크다.
confidence_level = 0.95
alpha = 1 - confidence_level
t_star = stats.t(df).ppf(1 - alpha / 2)

margin_of_error = t_star * standard_error

ci_lower = beta_1_hat - margin_of_error
ci_upper = beta_1_hat + margin_of_error

print(f"Slope estimate: {beta_1_hat:.4f}")
print(f"Standard error: {standard_error:.4f}")
print(f"t* (df={df}): {t_star:.4f}")
print(f"Margin of error: {margin_of_error:.4f}")
print(f"\n{confidence_level:.0%} confidence interval of the slope")
print(f"{beta_1_hat:.4f} +/- {margin_of_error:.4f}")
print(f"({ci_lower:.4f}, {ci_upper:.4f})")
```

출력:

```
Slope estimate: 0.1640
Standard error: 0.0570
t* (df=18): 2.1009
Margin of error: 0.1198

95% confidence interval of the slope
0.1640 +/- 0.1198
(0.0442, 0.2838)
```

자유도가 $n - 2 = 18$이므로 $t^* = 2.1009$다. 정규분포의 1.96보다 큰 이 값이 $\sigma$를 추정한 대가다.

</div>

## 해석

- 추정된 기울기 $\hat{\beta}_1 = 0.164$는 카페인 섭취가 한 단위 늘어날 때 공부 시간이 평균 0.164만큼 늘어남과 연관됨을 뜻한다.
- 95% 신뢰구간은, 이 연구를 여러 번 반복한다면 그렇게 얻은 구간의 약 95%가 참 기울기 $\beta_1$을 포함할 것임을 말해 준다.
- 구간이 0을 포함하지 않으므로(양 끝이 모두 양수) 유의수준 5%에서 카페인 섭취와 공부 시간 사이에 통계적으로 유의한 양의 연관이 있다고 결론지을 수 있다.
- 오차한계는 세 가지 양에 의존한다. 임계값 $t^*$(신뢰수준이 높아지거나 $n$이 작아지면 커진다), 기울기의 표준오차, 그리고 암묵적으로 설명변수 값들의 산포이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 같은 기울기 추정값에 대해 99% 신뢰구간을 계산하라. 95% 구간과 폭을 비교하면 어떠한가?

</div>

??? success "풀이"

    $\alpha = 0.05$를 $\alpha = 0.01$로 바꾼다.

    ```python
    alpha_99 = 0.01
    t_star_99 = stats.t(df).ppf(1 - alpha_99 / 2)  # approximately 2.8784
    margin_99 = t_star_99 * standard_error
    ci_lower_99 = beta_1_hat - margin_99
    ci_upper_99 = beta_1_hat + margin_99
    print(f"99% CI: ({ci_lower_99:.4f}, {ci_upper_99:.4f})")
    ```

    출력:

    ```
    99% CI: (-0.0001, 0.3281)
    ```

    99% 구간 $(-0.0001, 0.3281)$은 0을 아슬아슬하게 담는다. 같은 자료의 95% 구간은 0을 담지 않았다. 신뢰수준을 올리면 구간이 넓어지고 결론이 뒤집힐 수 있다.

    $t^*_{18,\,0.005} = 2.8784$이므로 오차한계는 $0.1641$이고 99% 신뢰구간은 $(-0.0001, 0.3281)$이다. 신뢰수준이 높아지면 임계값 $t^*$가 커지므로 99% 구간이 95% 구간보다 넓다.

    여기서 99% 구간은 0을 아슬아슬하게 **포함한다**. 곧 유의수준 1%에서는 $H_0\colon \beta_1 = 0$을 기각하지 못한다. 이는 $p$값이 $2 \times P(T_{18} > 0.164/0.057) = 0.0100$으로 0.01을 간발의 차로 넘는다는 사실과 정확히 일치한다. 회귀 출력표에 반올림되어 적힌 "0.010"이 실제로는 0.01보다 아주 조금 크다는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span> 표본크기가 $n = 20$이 아니라 $n = 50$이라면($\hat{\beta}_1$과 SE는 그대로) 95% 신뢰구간은 어떻게 달라지는가? 이유를 설명하라.

</div>

??? success "풀이"

    $n = 50$이면 자유도가 $df = 48$이 된다. 임계값 $t^*_{48, 0.025}$는 표준정규의 $z^* = 1.96$에 더 가까워지므로 구간이 조금 좁아진다. 더 중요한 것은, 표본이 커지면 표준오차 자체가 줄어들 가능성이 높다는 점이다($\mathrm{SE} \propto 1/\sqrt{\sum(x_i - \bar{x})^2}$이므로). 그러면 구간이 훨씬 더 좁아진다. 표본이 클수록 추정이 정밀해진다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span> 신뢰구간을 이용해 $\alpha = 0.05$에서 $H_0\colon \beta_1 = 0$의 양측 가설검정을 수행하라. 판정을 서술하고 신뢰구간과 가설검정의 쌍대성을 설명하라.

</div>

??? success "풀이"

    95% 신뢰구간은 약 $(0.044, 0.284)$이다. $0$이 이 구간에 들어 있지 않으므로 유의수준 5%에서 $H_0\colon \beta_1 = 0$을 기각한다. 쌍대성이란 유의수준 $\alpha$에서 $H_0$을 기각하는 것이 $(1-\alpha)$ 수준 신뢰구간이 귀무값을 포함하지 않는 것과 동치라는 뜻이다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> $w_i = (x_i - \bar{x}) / \sum_{j=1}^n (x_j - \bar{x})^2$일 때 $\hat{\beta}_1 = \sum_{i=1}^n w_i y_i$에서 출발하여 $\mathrm{SE}(\hat{\beta}_1)$의 공식을 유도하라.

</div>

??? success "풀이"

    $S_{xx} = \sum_j (x_j - \bar{x})^2$일 때 $w_i = (x_i - \bar{x})/S_{xx}$이고 $\hat{\beta}_1 = \sum_i w_i y_i$이며, $y_i$들은 독립이고 분산이 $\sigma^2$이므로

    $$
    \mathrm{Var}(\hat{\beta}_1) = \sum_{i=1}^n w_i^2 \,\sigma^2 = \frac{\sigma^2}{S_{xx}^2}\sum_{i=1}^n (x_i - \bar{x})^2 = \frac{\sigma^2}{S_{xx}}.
    $$

    따라서 $\mathrm{SE}(\hat{\beta}_1) = s / \sqrt{S_{xx}}$이며 $s$는 잔차 표준오차이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> $n \to \infty$일 때 $t$ 기반 신뢰구간이 $z$ 기반 구간으로 수렴함을 증명하라. 실무에서 이 구분이 중요해지는 조건은 무엇인가?

</div>

??? success "풀이"

    자유도 $\nu$의 $t$ 분포는 $\nu \to \infty$일 때 $N(0,1)$로 분포수렴한다. 신뢰구간에서는 $n \to \infty$이면 $df = n - 2 \to \infty$이므로 $t^*_{n-2,\,\alpha/2} \to z_{\alpha/2}$가 되어 $t$ 구간이 $z$ 구간이 된다. 실무에서 이 구분은 $n$이 작을 때(대략 $n < 30$) 중요하다. 그때는 $t$ 분포의 두꺼운 꼬리가 더 넓은 구간을 만들어 $\sigma$를 추정하는 데서 오는 추가 불확실성을 제대로 반영한다. $\square$

---

## 정리하며

기울기 신뢰구간을 **실제 자료로** 계산했다.

- **$n=20$ 이므로 자유도가 $18$ 이다.** $t_{0.025,18}\approx2.101$ 로 $z$ 의 $1.96$ 보다 뚜렷이 크며, **소표본에서 $t$ 를 써야 하는 이유**가 수치로 드러난다.
- **표준오차가 $s/\sqrt{S_{xx}}$ 다.** 잔차의 흩어짐이 클수록, 그리고 $x$ 가 좁게 몰려 있을수록 커진다.
- **구간이 $0$ 을 포함하는지 확인한다.** 포함하지 않으면 $\alpha=0.05$ 양측검정에서 기각한다는 뜻이다.
- **구간의 폭이 효과의 정밀도를 말해 준다.** 유의하더라도 구간이 넓으면 "효과가 있다"는 것 외에 할 말이 별로 없다.
- **단위를 함께 적는다.** 기울기는 단위가 있는 양이므로 "카페인 1mg 당 시간"처럼 읽혀야 한다.

다음 절 **신뢰구간과 예측 띠**로 넘어간다.
