# Anderson-Darling 검정

## 개요

Anderson-Darling(AD) 검정은 EDF에 기반한 정규성 검정으로, Kolmogorov-Smirnov 검정에 비해 분포의 꼬리에 추가 가중을 준다. 이 꼬리 민감성 덕분에 두꺼운 꼬리, 극단부의 치우침, 이상점이 잘 생기는 분포로 나타나는 정규성 이탈을 탐지하는 데 특히 효과적이다. SciPy는 정확한 $p$값 대신 $A^2$ 통계량과 표로 정리된 임계값을 보고한다.

## 검정통계량

순서통계량 $X_{(1)} \leq \cdots \leq X_{(n)}$과 가설 CDF $F_0$이 주어졌을 때 Anderson-Darling 통계량은

$$
A^2 = -n - \sum_{i=1}^{n} \frac{2i - 1}{n} \Bigl[\ln F_0(X_{(i)}) + \ln\bigl(1 - F_0(X_{(n+1-i)})\bigr)\Bigr].
$$

$U_i = F_0(X_{(i)})$(확률적분변환값)로 쓰면 동등하게

$$
A^2 = -n - \frac{1}{n}\sum_{i=1}^{n} (2i - 1)\bigl[\ln U_i + \ln(1 - U_{n+1-i})\bigr].
$$

가중치 $(2i-1)/n$은 꼬리에 있는 관측값($U_i$가 0이나 1에 가까운 곳)에 더 큰 비중을 둔다. $U_i \to 0$ 또는 $U_i \to 1$일 때 $\ln U_i$와 $\ln(1 - U_i)$가 발산하기 때문이다.

## 꼬리 민감성이 중요한 이유

KS 통계량 $D_n = \sup_x |F_n(x) - F_0(x)|$은 분포의 모든 부분을 동등하게 취급한다. 반면 $A^2$은 제곱차 $(F_n - F_0)^2$을 $[F_0(1 - F_0)]^{-1}$로 가중한 적분이다.

$$
A^2 = n \int_{-\infty}^{\infty} \frac{[F_n(x) - F_0(x)]^2}{F_0(x)\,[1 - F_0(x)]}\, dF_0(x).
$$

분모 $F_0(1-F_0)$이 꼬리에서 가장 작으므로 그곳의 불일치가 증폭된다.

## 가설과 임계값

$$
H_0: \text{자료가 정규분포를 따른다}, \qquad H_1: \text{자료가 정규분포를 따르지 않는다}.
$$

SciPy의 `stats.anderson`은 통계량 $A^2$과 함께 유의수준 15%, 10%, 5%, 2.5%, 1%에서의 임계값을 반환한다. $A^2$이 해당 임계값을 넘으면 수준 $\alpha$에서 $H_0$을 기각한다.

<div class="codebox" markdown>

**예제 1.** 섞인 자료에 Anderson-Darling 검정

```python
import numpy as np
from scipy import stats

# 정규 230개에 로그정규 70개를 섞은 자료다.
rng = np.random.default_rng(0)
x = np.concatenate([rng.normal(0, 1, size=230),
                    rng.lognormal(0, 0.6, size=70)])

# A^2 은 경험분포와 이론분포의 차이를 제곱해 적분한 값인데, 꼬리 쪽에
# 더 큰 무게를 싣는다. 그래서 KS 보다 꼬리 이탈을 잘 잡는다.
res = stats.anderson(x, dist="norm")
print(f"Sample size n = {x.size}")
print(f"Anderson-Darling A^2 = {res.statistic:.4f}")
print("Critical values vs significance levels:")
for cv, sl in zip(res.critical_values, res.significance_level):
    flag = " REJECT" if res.statistic > cv else ""
    print(f"  {sl:.1f}% -> {cv:.4f}{flag}")

reject_5 = res.statistic > res.critical_values[
    list(res.significance_level).index(5.0)]
print(f"Decision at 5%: {'Reject' if reject_5 else 'Fail to reject'} normality")
```

출력:

```text
Sample size n = 300
Anderson-Darling A^2 = 1.0034
Critical values vs significance levels:
  15.0% -> 0.5690 REJECT
  10.0% -> 0.6480 REJECT
  5.0% -> 0.7770 REJECT
  2.5% -> 0.9060 REJECT
  1.0% -> 1.0780
Decision at 5%: Reject normality
```

</div>

## 해석

혼합 자료(정규 + 대수정규)에서 $A^2 = 1.0034$는 15%, 10%, 5%, 2.5% 임계값을 넘어 정규성에 반하는 강한 증거를 준다. 대수정규 성분이 오른쪽 꼬리를 부풀리므로 AD 검정이 특히 적합한 상황이다.

다만 $A^2 = 1.0034$가 **1% 임계값 $1.078$은 넘지 못한다**는 점에 주의하라. 곧 "모든 표로 정리된 수준에서 기각"하는 것이 아니라 2.5%까지만 기각한다. 오염된 관측값이 70개(전체의 23%)나 되는데도 그렇다. AD 검정이 강력하기는 하지만 무한히 강력하지는 않다.

AD 검정이 기각하는데 KS 검정이 기각하지 않는 경우는 흔히 이탈이 꼬리에서 일어나기 때문이다. 바로 AD 검정이 탐지하도록 설계된 영역이다. 연습문제 2에서 이를 극적으로 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 표준정규 관측값 $n = 300$개를 생성하라. Anderson-Darling 검정을 수행하고 $A^2$이 15% 임계값보다 작음을 확인하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=300)

    res = stats.anderson(x, dist="norm")
    print(f"A^2 = {res.statistic:.4f}")
    print(f"15% critical value: {res.critical_values[0]:.4f}")
    print("Below 15% critical value:", res.statistic < res.critical_values[0])
    ```

    출력:

    ```text
    A^2 = 0.3329
    15% critical value: 0.5690
    Below 15% critical value: True
    ```

    $A^2 = 0.333$은 가장 느슨한 임계값 $0.569$보다도 훨씬 작다. 정규성 아래에서 기대되는 결과이다.

    참고로 $A^2$의 귀무분포는 표본크기에 거의 의존하지 않는다. 위의 임계값들은 $n$이 커져도 거의 그대로이다(SciPy는 $n$에 따른 작은 보정만 적용한다). $D_n$이 $1/\sqrt{n}$으로 줄어드는 KS 통계량과 다른 점이다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $t_4$ 분포에서 관측값 $n = 300$개를 생성하라. AD 검정과 ($\mathcal{N}(0,1)$에 대한) KS 검정을 수행하라. 어느 검정이 더 강하게 기각하는가?

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.standard_t(df=4, size=300)

    # Anderson-Darling 검정 — 꼬리에 민감하다
    ad = stats.anderson(x, dist="norm")
    print(f"A^2 = {ad.statistic:.4f}")
    for cv, sl in zip(ad.critical_values, ad.significance_level):
        flag = " *" if ad.statistic > cv else ""
        print(f"  {sl:.1f}%: {cv:.4f}{flag}")

    # KS 검정
    D, p_ks = stats.kstest(x, 'norm', args=(0, 1))
    print(f"\nKS: D = {D:.4f}, p = {p_ks:.4f}")
    print(f"Sample variance = {x.var(ddof=1):.4f}")
    ```

    출력:

    ```text
    A^2 = 2.0862
      15.0%: 0.5690 *
      10.0%: 0.6480 *
      5.0%: 0.7770 *
      2.5%: 0.9060 *
      1.0%: 1.0780 *

    KS: D = 0.0496, p = 0.4386
    Sample variance = 1.6985
    ```

    결과가 극명하게 갈린다.

    | 검정 | 결과 | 판정 |
    |---|---|---|
    | Anderson-Darling | $A^2 = 2.086$ vs 1% 임계값 $1.078$ | 모든 수준에서 기각 |
    | Kolmogorov-Smirnov | $D = 0.0496$, $p = 0.4386$ | **기각 못 함** |

    AD 검정은 1% 수준에서도 단호하게 기각하는 반면 **KS 검정은 아예 기각하지 못한다**($p = 0.44$).

    $t_4$의 이론적 분산은 $\nu/(\nu-2) = 4/2 = 2$로 1과 크게 다른데도 왜 KS가 실패할까? 표본분산이 $1.699$로 이미 2보다 작게 나왔다는 점도 있지만, 근본적인 이유는 CDF 사이의 최대 수직 거리가 작다는 것이다. $t_4$와 $\mathcal{N}(0,1)$의 CDF는 여러 번 교차하며(중앙에서는 $t_4$가 더 뾰족하고 꼬리에서는 더 두껍다), 어느 한 점에서의 차이가 크지 않다.

    AD는 꼬리의 불일치를 $[F_0(1-F_0)]^{-1}$로 증폭하므로 같은 이탈을 훨씬 크게 본다. **꼬리 이탈을 다룰 때 AD가 KS보다 결정적으로 낫다는 것을 보여주는 교과서적 예이다.** $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> AD 통계량이 제곱 EDF 차의 가중적분으로 쓰일 수 있음을 보여라. 정의에서 출발하여 적분 형태를 유도하라.

</div>

??? success "풀이"

    가중 Cramér-von Mises 범함수를 정의한다.

    $$
    A^2 = n \int_{-\infty}^{\infty} \frac{[F_n(x) - F_0(x)]^2}{F_0(x)(1 - F_0(x))}\, dF_0(x).
    $$

    $u = F_0(x)$로 치환하면($du = f_0(x)\,dx = dF_0(x)$)

    $$
    A^2 = n \int_0^1 \frac{[F_n(F_0^{-1}(u)) - u]^2}{u(1-u)}\, du.
    $$

    가중치 $1/[u(1-u)]$는 $u \to 0$ 또는 $u \to 1$일 때 발산하여 극단 꼬리의 편차에 무한한 가중을 준다.

    이 가중치가 임의로 선택된 것이 아니라는 점이 중요하다. $H_0$ 아래에서 $\sqrt{n}[F_n(x) - F_0(x)]$는 분산이 $u(1-u)$인 Brown 브리지로 수렴한다. 곧 $1/[u(1-u)]$는 **분산의 역수 가중**이며, 각 위치의 편차를 그 표준편차로 나누어 표준화하는 셈이다. 이 관점에서 AD는 자연스러운 표준화된 적분 검정이고, 가중치가 없는 Cramér-von Mises 통계량이 오히려 특수한 선택이다.

    실제 계산에서는 $F_n$의 계단함수 형태를 이용해 적분을 전개하면 이산합 형태

    $$
    A^2 = -n - \frac{1}{n}\sum_{i=1}^n (2i-1)[\ln U_i + \ln(1 - U_{n+1-i})]
    $$

    를 얻는다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> SciPy의 AD 검정은 지수분포와 로지스틱분포에 대해서도 검정할 수 있다. 같은 자료를 세 분포 모두에 대해 검정하고 결과를 비교하는 코드를 작성하라.

</div>

??? success "풀이"

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    x = rng.exponential(2.0, size=200)

    for dist in ["norm", "expon", "logistic"]:
        res = stats.anderson(x, dist=dist)
        print(f"\n{dist}: A^2 = {res.statistic:.4f}")
        for cv, sl in zip(res.critical_values, res.significance_level):
            flag = " REJECT" if res.statistic > cv else ""
            print(f"  {sl}%: {cv:.4f}{flag}")
    ```

    출력(요약):

    ```text
    norm: A^2 = 7.8017
      15.0%: 0.5650 REJECT
      10.0%: 0.6440 REJECT
      5.0%: 0.7720 REJECT
      2.5%: 0.9010 REJECT
      1.0%: 1.0710 REJECT

    expon: A^2 = 0.5851
      15.0%: 0.9190
      10.0%: 1.0750
      5.0%: 1.3370
      2.5%: 1.6010
      1.0%: 1.9510

    logistic: A^2 = 16.1719
      25.0%: 0.4250 REJECT
      10.0%: 0.5620 REJECT
      5.0%: 0.6590 REJECT
      2.5%: 0.7680 REJECT
      1.0%: 0.9050 REJECT
      0.5%: 1.0090 REJECT
    ```

    지수 자료에 대해 `"expon"` 검정은 기각하지 않고(올바른 귀무가설) `"norm"`과 `"logistic"` 검정은 기각한다(틀린 분포). AD 검정이 여러 분포족에 걸쳐 쓸 수 있음을 보여준다.

    !!! warning "두 가지 실무적 주의사항"
        **(1) 유의수준 배열이 분포마다 다르다.** `norm`과 `expon`은 `[15, 10, 5, 2.5, 1]`의 다섯 수준을 반환하지만 `logistic`은 `[25, 10, 5, 2.5, 1, 0.5]`의 **여섯** 수준을 반환하며 첫 항목이 15%가 아니라 **25%**이다. 따라서 `res.critical_values[2]`가 5% 임계값이라고 가정하는 코드는 세 분포에서 우연히 맞지만, `critical_values[0]`을 15% 임계값으로 해석하면 로지스틱에서 틀린다. 항상 `significance_level`과 함께 짝지어 읽어야 한다.

        **(2) `dist="logistic"`은 경고를 낼 수 있다.** SciPy는 로지스틱 모수를 `fsolve`로 추정하는데, 지수 자료처럼 로지스틱과 크게 다른 자료에서는 `RuntimeWarning: overflow encountered in exp`와 수렴 실패 경고가 나온다. 이 예에서 $A^2 = 16.17$이라는 결론 자체는 타당하지만(모든 수준에서 압도적 기각), 경계선상의 사례에서는 이런 경고를 무시해서는 안 된다.

        $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> SciPy가 Anderson-Darling 검정에 정확한 $p$값을 제공하지 않는 이유를 설명하고, 모의실험으로 $p$값을 얻는 방법을 기술하라.

</div>

??? success "풀이"

    $A^2$의 귀무분포는 비표준적이다. 가설 분포족에 의존하고, 모수를 추정하는지 여부에도 의존한다. 닫힌 형태의 표현은 특정 경우에만 존재하며, SciPy는 정확한 $p$값을 계산하는 대신 표로 정리된 임계값으로 검정을 구현한다.

    모의실험 기반 $p$값을 얻으려면

    1. 자료에서 $A^2_{\text{obs}}$를 계산한다.
    2. $b = 1, \ldots, B$에 대해 같은 크기의 $x^{*(b)} \sim \mathcal{N}(\hat{\mu}, \hat{\sigma}^2)$를 모의생성하고 $A^{2*}_{(b)}$를 계산한다(각 붓스트랩 표본에서 모수를 **다시 추정**해야 한다).
    3. $p$값은 $\hat{p} = \frac{1}{B}\sum_b \mathbf{1}(A^{2*}_{(b)} \geq A^2_{\text{obs}})$이다.

    Lilliefors 붓스트랩과 같은 원리이며 모수 추정을 올바르게 반영한다. $B = 5000$이면 5% 경계 근처에서 $p$값의 표준오차는 $\sqrt{0.05 \times 0.95/5000} \approx 0.003$이다.

    한 가지 유용한 사실을 덧붙인다. 정규성 검정의 경우 $A^2$의 귀무분포가 위치-척도 불변이므로 $\hat{\mu}, \hat{\sigma}$에 의존하지 않는다. 따라서 $\mathcal{N}(0,1)$에서 한 번만 큰 모의실험을 돌려 임계값 표를 만들어 두면 모든 자료에 재사용할 수 있다. SciPy가 하는 일이 바로 그것이다. $\square$

---

## 정리하며

앤더슨–달링의 **출력 형식**이 다른 검정과 다르다.

- **$A^2$ 통계량과 임계값 배열을 돌려준다.** $p$ 값이 없으므로 **통계량을 임계값과 직접 비교**해 판정한다. `result.statistic > result.critical_values[2]` 가 $\alpha=0.05$ 판정이다.
- **유의수준 목록도 함께 온다.** `significance_level` 속성이 $[15,10,5,2.5,1]$ 을 준다.
- **꼬리 가중이 코드에서 확인된다.** 직접 구현해 보면 $\log F_0$ 과 $\log(1-F_0)$ 항이 극단에서 커지는 것을 볼 수 있다.
- **모수 추정이 이미 반영되어 있다.** `dist='norm'` 이면 자료에서 $\mu,\sigma$ 를 추정하는 상황의 임계값을 주므로, KS 처럼 따로 보정할 필요가 없다.
- **꼬리가 관심이면 이 검정을 고른다.** 금융 자료 진단에 특히 맞다.

다음 절 **Shapiro-Wilk 검정 (코드)** 로 넘어간다.
