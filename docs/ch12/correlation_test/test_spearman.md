# Spearman의 rho 검정

Spearman 순위상관 $r_s$는 두 변수 사이 단조 연관의 강도를 잰다. 관측된 $r_s$가 통계적으로 유의한지 판정하려면 모집단 Spearman 상관이 0이라는 귀무가설을 검정한다. 이 절에서는 가설검정, 그 귀무분포, 그리고 순위에 대한 Pearson t-검정과의 연결을 다룬다.

---

## 가설

표준 검정은

$$
H_0\!: \rho_s = 0 \quad \text{vs} \quad H_1\!: \rho_s \neq 0
$$

이며 $\rho_s$는 모집단 Spearman 순위상관이다. 귀무가설은 $X$와 $Y$ 사이에 단조 연관이 없다는 것이다.

단조 관계의 방향을 미리 지정한 경우에는 단측 대립가설($H_1\!: \rho_s > 0$ 또는 $H_1\!: \rho_s < 0$)을 쓴다.

---

## 검정통계량: t 근사

표본이 어느 정도 크면 검정통계량은

$$
t = r_s \sqrt{\frac{n - 2}{1 - r_s^2}}
$$

이다. $H_0$(단조 연관 없음) 아래에서 이 통계량은 근사적으로 **자유도 $n - 2$인 t-분포**를 따른다. Pearson의 $r$을 검정할 때 쓰는 것과 같은 공식을 순위상관에 적용한 것이다.

이 근사는 자료의 순위에 Pearson 상관 t-검정을 수행하는 것과 같다. Spearman의 $r_s$가 순위에 대해 계산한 Pearson의 $r$과 같으므로 두 검정 절차는 대수적으로 동등하다.

---

## 작은 표본에서의 정확분포

표본이 작으면(보통 $n \le 20$) 순위의 가능한 $n!$개 순열을 모두 고려하여 $H_0$ 아래 $r_s$의 정확한 귀무분포를 계산할 수 있다. $H_0$ 아래에서 모든 순열이 똑같이 그럴듯하므로 $r_s$의 분포를 정확히 표로 만들 수 있다.

작은 $n$에 대한 정확분포의 임계값은 표로 제공된다. 대부분의 통계 소프트웨어는 작은 표본에는 정확분포를 쓰고 표본이 커지면 t 근사로 전환한다.

---

## 판정 규칙

유의수준 $\alpha$의 양측검정에서:

- t 근사를 쓰면: $|t| > t_{\alpha/2, \, n-2}$이면 $H_0$을 기각한다
- 정확표를 쓰면: 주어진 $n$과 $\alpha$의 임계값을 $|r_s|$가 넘으면 $H_0$을 기각한다

p-값은 t 근사에서

$$
p = 2 \cdot P(T_{n-2} > |t|)
$$

로 계산하고, 작은 표본에서는 정확 순열분포에서 계산한다.

---

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 체질량과 대사율. 어떤 생물학자가 동물 12마리를 체질량($X$)과 대사율($Y$)로 순위를 매겨 $r_s = 0.72$를 얻었다.

</div>

??? success "풀이"
    $$
    t = 0.72 \sqrt{\frac{12 - 2}{1 - 0.72^2}} = 0.72 \sqrt{\frac{10}{0.4816}} = 0.72 \times 4.557 = 3.281
    $$

    자유도 $n - 2 = 10$에서 $\alpha = 0.05$ 양측검정의 임계값은 $t_{0.025, 10} = 2.228$이다. $|t| = 3.281 > 2.228$이므로 $H_0$을 기각하고 체질량과 대사율 사이에 통계적으로 유의한 단조 연관이 있다고 결론짓는다.

## 가정

Spearman의 $r_s$ 검정은 다음을 요구한다:

1. **독립성**: 관측값이 독립인 쌍이다.
2. **순서형 또는 연속형 자료**: 두 변수가 적어도 순서형이어야 한다(순위가 의미를 가져야 한다).
3. **특정 분포 가정 없음**: Pearson t-검정과 달리 Spearman 검정은 정규성을 가정하지 않는다. **분포에 의존하지 않는**(비모수) 검정이다.

분포에 의존하지 않는다는 성질 덕분에 Spearman 검정은 다음일 때 적절하다:

- 자료가 정규가 아니거나 치우쳐 있을 때.
- 이상점이 있을 때.
- 관계가 단조이지만 선형이 아닐 때.

---

## Spearman 검정과 Pearson 검정 중 무엇을 고를까

| 항목 | Pearson t-검정 | Spearman t-검정 |
|:---|:---|:---|
| 탐지하는 것 | 선형 연관 | 단조 연관 |
| 분포 가정 | 이변량 정규성 | 없음(분포에 의존하지 않음) |
| 이상점에 대한 로버스트성 | 낮음 | 높음 |
| 정규성+선형성 아래의 검정력 | 더 높음 | 약간 낮음 |
| 비정규성 아래의 검정력 | 더 낮음 | 더 높음 |

자료가 이변량 정규이고 관계가 선형이면 Pearson 검정의 검정력이 조금 더 높다. 그 밖의 모든 상황에서는 Spearman 검정이 적어도 같거나 더 강력한 경우가 많다.

---

<div class="codebox" markdown>

### 예제 1. Spearman 상관 검정과 손계산 { .eg }

```python
import numpy as np
from scipy import stats

x = np.array([3, 1, 6, 4, 8, 2, 7, 5, 10, 9, 11, 12])
y = np.array([5, 2, 8, 3, 11, 1, 9, 6, 12, 7, 10, 4])

# 순위만 쓰므로 정규성이 필요 없고 이상치에도 강하다.
r_s, p_value = stats.spearmanr(x, y)
print(f"Spearman r_s = {r_s:.4f}")
print(f"p-value      = {p_value:.4f}")

# 피어슨과 같은 꼴의 t 통계량으로 손계산해 확인한다. 순위에 피어슨을
# 적용한 것이 스피어만이므로 검정 방식도 그대로 따라온다.
n = len(x)
t_stat = r_s * np.sqrt((n - 2) / (1 - r_s**2))
p_manual = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
print(f"t-statistic  = {t_stat:.4f}")
print(f"Manual p     = {p_manual:.4f}")
```

출력:

```
Spearman r_s = 0.6573
p-value      = 0.0202
t-statistic  = 2.7584
Manual p     = 0.0202
```

scipy의 p-값과 $t$ 근사로 손계산한 값이 소수점 넷째 자리까지 같다. Spearman 검정의 p-값이 자유도 $n-2$인 $t$-분포에서 나온다는 것을 확인해 준다.

`scipy.stats.spearmanr` 함수는 표본이 크면 t 근사를 쓰며 동점도 처리한다.

</div>

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$n = 20$인 표본에서 얻은 Spearman의 $r_s = 0.55$가 $\alpha = 0.05$에서 0과 유의하게 다른지 검정하라.

</div>

??? success "풀이"
    검정통계량은

    $$
    t = r_s \sqrt{\frac{n-2}{1-r_s^2}} = 0.55\sqrt{\frac{18}{1 - 0.3025}} = 0.55\sqrt{\frac{18}{0.6975}} = 0.55\sqrt{25.806} = 0.55 \times 5.080 = 2.794
    $$

    이다. $df = n - 2 = 18$에서 임계값은 $t_{0.025, 18} = 2.101$이다. $|t| = 2.794 > 2.101$이므로 $\alpha = 0.05$에서 $H_0$을 기각한다. 단조 연관의 유의한 증거가 있다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
Spearman 검정의 t-분포 근사가 $n$이 클 때 더 정확한 이유를 설명하라. 작은 $n$에 대한 정확검정은 무엇인가?

</div>

??? success "풀이"
    검정통계량 $t = r_s\sqrt{(n-2)/(1-r_s^2)}$은 $H_0$ 아래에서 근사적으로 $t_{n-2}$ 분포를 갖는다. 이 근사는 $n \geq 10$에서 잘 작동하고 극한에서 정확해진다.

    $n$이 작으면 $H_0$ 아래 $r_s$의 정확분포가 (순위가 정수값만 가지므로) 이산이며, 한 순위 벡터의 $n!$개 순열을 모두 열거하여 계산한다. 통계 소프트웨어(예: `scipy.stats.spearmanr`)는 작은 $n$에서 정확 p-값을 계산하고 표본이 커지면 t 근사로 전환한다.

    정확검정은 순열검정이다. 독립을 가정하고 $Y$ 순위의 가능한 모든 순열에서 얻은 $r_s$ 값의 분포와 관측된 $r_s$를 비교한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
50개국 자료에서 1인당 GDP 순위와 기대수명 순위 사이의 Spearman $r_s = 0.72$를 얻었다. 모집단 $\rho_s$에 대한 근사 95% 신뢰구간을 만들어라.

</div>

??? success "풀이"
    (Spearman의 $r_s$에도 근사적으로 쓸 수 있는) Fisher의 z 변환을 적용한다:

    $$
    z_r = \frac{1}{2}\ln\frac{1 + 0.72}{1 - 0.72} = \frac{1}{2}\ln(6.143) = \frac{1}{2}(1.815) = 0.9076
    $$

    표준오차는 근사적으로 $1/\sqrt{n-3} = 1/\sqrt{47} = 0.1459$이다.

    $z_\rho$에 대한 95% 신뢰구간은 $0.9076 \pm 1.96 \times 0.1459 = (0.6216, 1.1936)$이다.

    $r = \frac{e^{2z}-1}{e^{2z}+1}$으로 역변환하면:

    - 하한: $\tanh(0.6216) = 0.553$
    - 상한: $\tanh(1.1936) = 0.832$

    $\rho_s$에 대한 95% 신뢰구간은 약 $(0.55, 0.83)$이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
Spearman 검정이 Pearson 상관 검정보다 선호되는 것은 언제인가? 세 가지 상황을 들어라.

</div>

??? success "풀이"
    Spearman 검정이 선호되는 경우:

    1. **관계가 단조이지만 비선형일 때:** Pearson의 $r$은 선형 연관만 재므로 휘어진 단조 관계의 강도를 과소평가할 수 있다. Spearman의 $r_s$는 어떤 단조 패턴이든 포착한다.

    2. **자료에 이상점이 있을 때:** Spearman의 $r_s$는 순위에 기반하므로 극단값에 로버스트하다. 이상점 하나가 Pearson의 $r$을 크게 바꿀 수 있지만 $r_s$에는 거의 영향을 주지 않는다.

    3. **자료가 순서형일 때:** 측정이 순서 척도(예: 리커트 평정, 순위)이면 수치값에 의미 있는 간격 해석이 없다. Spearman의 $r_s$는 순위 정보만 쓰므로 적절하지만 Pearson의 $r$은 구간 척도를 가정한다.

    또한 Spearman 검정은 Pearson의 $r$로 정확한 추론을 하는 데 필요한 이변량 정규성 가정을 요구하지 않는다.


<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
연습문제 4는 스피어만이 나은 상황 셋을 물었다. 그중 **이상치**를 수치로 확인하라. 참 상관이 $+0.4$인 자료에 이상치 하나를 넣고 세 상관계수가 무엇을 보고하는지 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    n = 30
    rp, rs, rk = [], [], []
    sp = ss = sk = 0
    for _ in range(3000):
        X = rng.normal(0, 1, n)
        Y = 0.4*X + rng.normal(0, 1, n)
        X[0] += 10; Y[0] -= 10          # 이상치 하나
        a = stats.pearsonr(X, Y)
        b = stats.spearmanr(X, Y)
        c = stats.kendalltau(X, Y)
        rp.append(a.statistic); rs.append(b.statistic); rk.append(c.statistic)
        sp += (a.pvalue < 0.05) and a.statistic > 0
        ss += (b.pvalue < 0.05) and b.statistic > 0
        sk += (c.pvalue < 0.05) and c.statistic > 0

    print(f"  평균 r: Pearson {np.mean(rp):+.4f}  "
          f"Spearman {np.mean(rs):+.4f}  Kendall {np.mean(rk):+.4f}")
    print(f"  '양의 상관'을 옳게 잡은 비율:")
    print(f"    Pearson {sp/3000:.4f}  Spearman {ss/3000:.4f}  Kendall {sk/3000:.4f}")
    ```

    출력:

    ```
      평균 r: Pearson -0.6716  Spearman +0.2123  Kendall +0.1575
      '양의 상관'을 옳게 잡은 비율:
        Pearson 0.0000  Spearman 0.1640  Kendall 0.2033
    ```

    **피어슨이 부호를 뒤집는다.** 참 상관이 $+0.4$인데 평균 $r = -0.67$이다. 관측 하나가 $(x+10,\ y-10)$이라 지렛대가 극단적으로 크고, 그 한 점이 전체 기울기를 음수로 끌어내린다.

    **더 나쁜 것은 "유의하게" 뒤집는다는 점이다.** 피어슨이 양의 상관을 옳게 잡은 비율이 **정확히 $0$**이다. 매번 유의하게 기각하지만 **언제나 틀린 방향**이다. 검정력이 높아 보이는 것이 오히려 함정이다.

    **순위 기반 측도는 부호를 지킨다.** 스피어만 $+0.21$, 켄달 $+0.16$으로 참값 $+0.4$보다 작지만 방향은 맞다. 이상치가 순위에서는 "가장 큰 $x$, 가장 작은 $y$" 한 쌍일 뿐이라 영향이 $1/n$ 수준으로 제한된다.

    | | 평균 $r$ | 부호 | 옳게 잡은 비율 |
    |---|---|---|---|
    | 피어슨 | $-0.67$ | **뒤집힘** | $0.000$ |
    | 스피어만 | $+0.21$ | 맞음 | $0.164$ |
    | 켄달 | $+0.16$ | 맞음 | $0.203$ |

    **순위 측도의 검정력도 낮다**($0.16 \sim 0.20$). 이상치가 순위 구조도 어지럽히기 때문이다. **이상치는 어떤 방법으로도 공짜로 처리되지 않는다.**

    **실무 결론.** 상관을 계산하기 전에 **산점도를 반드시 본다.** 이상치가 보이면 (1) 자료 오류인지 확인하고, (2) 오류가 아니면 포함·제외 양쪽 결과를 보고하며, (3) 순위 기반 측도를 병기한다. **피어슨과 스피어만이 크게 다르면 그 자체가 진단 신호**이며, 여기서는 부호까지 달라 경보음이 분명하다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
스피어만 $r_s$는 **순위의 피어슨 상관**이다. 동점이 없을 때 간편식

$$
r_s = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}
$$

이 성립함을 보이고($d_i$는 순위 차), 동점이 있으면 왜 깨지는지 설명하라.

</div>

??? success "풀이"
    **동점이 없으면 순위가 $1,\ldots,n$의 순열이다.** 따라서 두 순위 변수의 평균과 분산이 **같고 이미 알려져 있다.**

    $$
    \bar R = \frac{n+1}{2},
    \qquad
    \sum_i\left(R_i-\bar R\right)^2 = \frac{n(n^2-1)}{12}
    $$

    피어슨 공식에 넣으면

    $$
    r_s = \frac{\sum_i (R_i-\bar R)(S_i-\bar R)}{n(n^2-1)/12}
    $$

    이고, $d_i = R_i - S_i$에 대해 $\sum d_i^2 = \sum(R_i-\bar R)^2 + \sum(S_i-\bar R)^2 - 2\sum(R_i-\bar R)(S_i-\bar R)$이므로 정리하면 간편식이 나온다. $\square$

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 12); y = 0.6*x + rng.normal(0, 1, 12)
    R, S = stats.rankdata(x), stats.rankdata(y)
    n = len(x); d = R - S
    print(f"  간편식      {1 - 6*np.sum(d**2)/(n*(n**2-1)):.6f}")
    print(f"  순위의 피어슨 {stats.pearsonr(R, S).statistic:.6f}")
    print(f"  scipy       {stats.spearmanr(x, y).statistic:.6f}")

    # 동점을 만들어 본다
    xt = np.round(x); yt = np.round(y)
    Rt, St = stats.rankdata(xt), stats.rankdata(yt)
    dt = Rt - St
    print(f"\n  동점 있음:")
    print(f"  간편식      {1 - 6*np.sum(dt**2)/(n*(n**2-1)):.6f}")
    print(f"  순위의 피어슨 {stats.pearsonr(Rt, St).statistic:.6f}")
    ```

    출력:

    ```
      간편식      0.545455
      순위의 피어슨 0.545455
      scipy       0.545455

      동점 있음:
      간편식      0.692308
      순위의 피어슨 0.655729
    ```

    **동점이 없을 때는 셋이 완전히 같다.** 동점을 만들자 간편식이 $0.692$, 올바른 값이 $0.656$으로 갈린다.

    **동점이 있으면 깨지는 이유.** 동점에는 **평균 순위**를 주는데(예: 3위와 4위가 같으면 둘 다 $3.5$), 그러면 순위가 더 이상 $1,\ldots,n$의 순열이 아니다. 따라서

    $$
    \sum_i(R_i-\bar R)^2 < \frac{n(n^2-1)}{12}
    $$

    로 **분산이 줄어든다.** 간편식은 분산이 이론값이라고 가정하므로 어긋난다.

    **올바른 처리는 순위에 피어슨 공식을 그대로 쓰는 것이다.** 이렇게 하면 실제 순위 분산을 쓰므로 동점이 있어도 정확하다. `scipy.stats.spearmanr`가 이 방식이다.

    | | 동점 없음 | 동점 있음 |
    |---|---|---|
    | 간편식 | 정확 | **틀림** |
    | 순위의 피어슨 | 정확 | **정확** |

    **간편식은 손계산 시대의 유물이다.** 제곱합 하나만 구하면 되어 편했지만, 컴퓨터로는 순위를 매겨 피어슨을 부르는 것이 더 간단하고 언제나 옳다. **오래된 교과서의 공식을 그대로 쓰다가 동점 자료에서 틀리는 일**이 실무에서 종종 생긴다.

---

## 정리하며

Spearman의 $r_s$에 대한 가설검정은 Pearson 검정과 같은 t-통계량 공식을 자료의 순위에 적용한다. 분포에 의존하지 않고 정규성을 요구하지 않으므로 이상점에 로버스트하고 순서형 자료에도 적용할 수 있다. 표본이 작으면 순열에 기반한 정확 p-값을 쓸 수 있다. 자료가 정규가 아니거나 이상점이 있거나 관계가 단조이면서 비선형일 때 Pearson 검정보다 Spearman 검정이 선호된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
순위를 쓰면 정보를 버리는 셈이니 검정력이 떨어질 것 같다. **얼마나** 떨어지는가? 여러 모집단에서 피어슨과 스피어만 검정의 검정력을 비교하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(0)
    REP, n = 3000, 30
    print(f"{'모집단':>10}{'Pearson':>10}{'Spearman':>10}")
    for name, gen in [("정규",  lambda m: rng.normal(0, 1, m)),
                      ("t(3)",  lambda m: rng.standard_t(3, m)),
                      ("지수",  lambda m: rng.exponential(1, m) - 1)]:
        rp = rs = 0
        for _ in range(REP):
            z, e = gen(n), gen(n)
            X, Y = z, 0.35*z + e
            rp += stats.pearsonr(X, Y).pvalue < 0.05
            rs += stats.spearmanr(X, Y).pvalue < 0.05
        print(f"{name:>10}{rp/REP:>10.4f}{rs/REP:>10.4f}")
    ```

    출력:

    ```
           모집단   Pearson  Spearman
            정규    0.4390    0.3920
          t(3)    0.4800    0.4720
            지수    0.4503    0.5013
    ```

    **정규에서만 피어슨이 이긴다**($0.439$ 대 $0.392$). 손실이 약 $11\%$다.

    **꼬리가 두꺼워지면 차이가 사라진다.** $t(3)$에서 $0.480$ 대 $0.472$로 거의 같다.

    **지수분포에서는 스피어만이 이긴다**($0.501$ 대 $0.450$). 치우친 분포에서 순위가 오히려 유리하다.

    **이론적 배경.** 이변량 정규에서 스피어만 검정의 점근상대효율은 $9/\pi^2 \approx 0.912$다. 즉 **정규일 때조차 $9\%$만 손해**다. 표본으로 환산하면 피어슨의 $n=100$이 스피어만의 $n=110$에 해당한다.

    | 모집단 | 유리한 쪽 | 차이 |
    |---|---|---|
    | 정규 | 피어슨 | 작다($\approx 9\%$) |
    | 두꺼운 꼬리 | 비슷 | 거의 없음 |
    | 치우침 | **스피어만** | 스피어만이 낫다 |

    **"비모수는 검정력을 크게 잃는다"는 통념이 과장임을 보여 준다.** 최선의 조건(정규)에서도 손실이 $10\%$ 안쪽이고, 조건이 나빠지면 오히려 이득이다. **잃을 것은 적고 지킬 것은 많다**는 것이 순위 방법의 일반적 성격이며, 16장에서 이 논점을 본격적으로 다룬다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
스피어만 $\rho_s$에 대한 **신뢰구간**을 만들어라. 피셔 $z$ 변환을 그대로 쓸 수 있는가? 어떤 보정이 필요한가?

</div>

??? success "풀이"
    **피셔 $z$를 그대로 쓰면 분산이 맞지 않는다.** 피어슨에서는 $\operatorname{Var}(z) \approx 1/(n-3)$인데, 스피어만의 $z$ 변환값은 분산이 조금 더 크다. 널리 쓰이는 보정(보노–웨어)은

    $$
    \operatorname{Var}\big(\operatorname{arctanh} r_s\big) \approx \frac{1+r_s^2/2}{n-3}
    $$

    이다.

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(2)
    n, rho = 40, 0.6
    REP = 20_000
    cov_naive = cov_adj = 0
    rho_s_true = 6/np.pi*np.arcsin(rho/2)

    for _ in range(REP):
        X = rng.multivariate_normal([0,0], [[1,rho],[rho,1]], n)
        rs = stats.spearmanr(X[:,0], X[:,1]).statistic
        z = np.arctanh(rs)
        for se, box in ((1/np.sqrt(n-3), 'naive'),
                        (np.sqrt((1 + rs**2/2)/(n-3)), 'adj')):
            lo, hi = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
            if lo <= rho_s_true <= hi:
                if box == 'naive': cov_naive += 1
                else: cov_adj += 1

    print(f"  참 rho_s = {rho_s_true:.4f}")
    print(f"  보정 없는 피셔 z : 포함률 {cov_naive/REP:.4f}")
    print(f"  보노-웨어 보정   : 포함률 {cov_adj/REP:.4f}")
    ```

    출력:

    ```
      참 rho_s = 0.5819
      보정 없는 피셔 z : 포함률 0.9395
      보노-웨어 보정   : 포함률 0.9573
    ```

    **보정하지 않으면 포함률이 $0.940$으로 명목에 못 미친다.** 분산을 과소평가해 구간이 좁아지기 때문이다. 보정하면 $0.957$로 명목을 조금 넘어선다. 완벽하지는 않지만 보정 없는 쪽보다 $0.95$에 가깝다.

    **주의할 점이 하나 더 있다.** 구간이 담아야 할 참값은 **$\rho$가 아니라 $\rho_s$**다. 이변량 정규에서 $\rho = 0.6$이면 $\rho_s = \frac{6}{\pi}\arcsin(0.3) = 0.582$로 조금 작다(켄달 문서 연습문제 8). **"스피어만 구간이 피어슨 $\rho$를 담는다"고 기대하면 틀린다.**

    **부트스트랩도 좋은 대안이다.** 관측 쌍을 재표본하면 변환이나 보정 없이 구간을 얻을 수 있고, 정규성 가정도 필요 없다. 표본이 작지 않다면 결과가 비슷하다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
스피어만은 **단조** 관계를 잰다. 단조가 아니면 어떻게 되는가? $U$자 관계와 주기적 관계에서 $\rho_s$가 무엇을 보고하는지 확인하고, 진단 방법을 제안하라.

</div>

??? success "풀이"
    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(5)
    n = 300
    x = rng.uniform(-3, 3, n)
    cases = {
        "단조 증가":   0.8*x + rng.normal(0, 0.5, n),
        "U자":        x**2 + rng.normal(0, 0.5, n),
        "주기(sin)":  np.sin(2*x) + rng.normal(0, 0.2, n),
        "단조 포화":   np.tanh(2*x) + rng.normal(0, 0.2, n),
    }
    print(f"{'관계':>12}{'Pearson':>10}{'Spearman':>11}")
    for k, y in cases.items():
        print(f"{k:>12}{stats.pearsonr(x,y).statistic:>10.4f}"
              f"{stats.spearmanr(x,y).statistic:>11.4f}")
    ```

    출력:

    ```
            관계   Pearson   Spearman
        단조 증가    0.9412     0.9485
            U자   -0.0057    -0.0249
      주기(sin)   -0.4010    -0.3729
        단조 포화    0.9037     0.8733
    ```

    **U자에서 둘 다 $0$이다.** $y = x^2$은 완전한 함수관계인데 $\rho_s = -0.025$다. 증가 구간과 감소 구간이 상쇄된다.

    **주기 관계에서는 $-0.40$이라는 오해를 부르는 값이 나온다.** 표본 범위 $[-3,3]$ 안에서 $\sin(2x)$가 전체적으로 감소하는 경향을 보였을 뿐이고, 범위를 바꾸면 값이 완전히 달라진다.

    **단조 포화에서는 오히려 피어슨이 조금 크다**($0.904$ 대 $0.873$). 직관과 어긋난다. $\tanh(2x)$는 대부분의 점이 $\pm1$ 근처에 몰리는데, 순위로 바꾸면 그 몰림이 균등하게 펴지면서 잡음의 영향이 상대적으로 커지기 때문이다.

    **그래서 "$r_s > r$이면 단조 비선형"이라는 어림은 믿을 것이 못 된다.** 비선형의 모양에 따라 어느 쪽이 클지가 달라진다. **두 값이 다르다는 사실은 신호이지만, 어느 쪽이 큰지는 원인을 특정해 주지 않는다.**

    **진단 방법 셋.**

    | 방법 | 무엇을 보나 |
    |---|---|
    | **산점도** | 가장 확실하다 |
    | 피어슨과 스피어만 비교 | 크게 다르면 **무언가 있다**(방향은 특정 못 함) |
    | **거리상관**(피어슨 문서 연습문제 6) | $r_s \approx 0$인데 크면 비단조 |

    **셋을 합치면 표가 된다.**

    | $r$ | $r_s$ | 거리상관 | 해석 |
    |---|---|---|---|
    | 큼 | 큼 | 큼 | 선형 |
    | 작음 | 큼 | 큼 | **단조 비선형**(순위로만 보이는 관계) |
    | 작음 | 작음 | **큼** | **비단조**(U자 등) |
    | 작음 | 작음 | 작음 | 관계 없음 |

    **$\rho_s = 0$을 "관계 없음"으로 읽으면 안 된다.** "단조 관계 없음"이 정확한 진술이며, U자나 주기 구조는 따로 찾아야 한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
$\rho_s$의 정의는 "순위의 피어슨 상관"이다. 그런데 **순위 대신 다른 점수**를 쓸 수도 있다. **정규점수 상관**(반 데르 바르덴)을 소개하고 스피어만과 비교하라.

</div>

??? success "풀이"
    **발상은 순위를 정규분위수로 바꾸는 것이다.** 순위 $R_i$를

    $$
    a_i = \Phi^{-1}\!\left(\frac{R_i}{n+1}\right)
    $$

    로 변환한 뒤 피어슨 상관을 구한다. 순위가 $1,\ldots,n$의 균등한 간격이던 것을 **정규분포 모양의 간격**으로 다시 벌리는 셈이다.

    ```python
    import numpy as np
    from scipy import stats

    def vdw(x, y):
        n = len(x)
        a = stats.norm.ppf(stats.rankdata(x) / (n + 1))
        b = stats.norm.ppf(stats.rankdata(y) / (n + 1))
        return stats.pearsonr(a, b)

    rng = np.random.default_rng(7)
    REP, n = 3000, 30
    print(f"{'모집단':>10}{'Pearson':>10}{'Spearman':>10}{'정규점수':>10}")
    for name, gen in [("정규", lambda m: rng.normal(0,1,m)),
                      ("t(3)", lambda m: rng.standard_t(3,m)),
                      ("지수", lambda m: rng.exponential(1,m)-1)]:
        rp = rs = rv = 0
        for _ in range(REP):
            z, e = gen(n), gen(n)
            X, Y = z, 0.35*z + e
            rp += stats.pearsonr(X,Y).pvalue < 0.05
            rs += stats.spearmanr(X,Y).pvalue < 0.05
            rv += vdw(X,Y).pvalue < 0.05
        print(f"{name:>10}{rp/REP:>10.4f}{rs/REP:>10.4f}{rv/REP:>10.4f}")
    ```

    출력:

    ```
           모집단   Pearson  Spearman      정규점수
            정규    0.4390    0.3920    0.4297
          t(3)    0.4800    0.4720    0.5030
            지수    0.4503    0.5013    0.4733
    ```

    **정규점수가 정규 모집단에서 스피어만보다 낫다**($0.430$ 대 $0.392$). 피어슨의 $0.439$에 거의 근접한다.

    **이것이 이 방법의 매력이다.** 순위만 쓰므로 이상치에 강건한데, **정규 모집단에서의 효율 손실이 거의 없다.** 이론적으로 반 데르 바르덴 검정의 점근상대효율은 정규에서 **정확히 $1$**이고, 다른 많은 분포에서 $1$보다 크다.

    | | 정규에서의 ARE | 이상치 강건성 |
    |---|---|---|
    | 피어슨 | $1$(최적) | **없음** |
    | 스피어만 | $9/\pi^2 = 0.912$ | 있음 |
    | **정규점수** | **$1$** | 있음 |

    **"공짜 점심"에 가까워 보이는데 왜 덜 쓰이는가.**

    - **덜 알려져 있다.** 스피어만·켄달이 교과서와 소프트웨어에 먼저 자리 잡았다.
    - **해석이 어렵다.** 스피어만은 "순위의 상관", 켄달은 "일치확률"로 읽히는데 정규점수 상관은 직관적 해석이 없다.
    - **정규성을 은근히 들여온다.** 점수를 정규분위수로 매기는 선택 자체가 정규분포를 기준으로 삼는다.

    **그럼에도 알아 둘 가치가 있다.** 16장에서 다룰 순위 검정 전반에 같은 발상이 적용되며(순위 대신 정규점수를 쓰는 노멀 스코어 검정), **"순위로 바꾼 뒤 어떤 점수를 매길 것인가"가 비모수 방법 설계의 핵심 자유도**임을 보여 준다. $\square$

---
