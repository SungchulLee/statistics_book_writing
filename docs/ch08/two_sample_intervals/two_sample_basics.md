# 비율구간과 두 표본 비교 — 기본

비율의 구간이 경계에서 무너지는 문제와, 두 집단을 견주는 세 가지 구간 — 독립인 두 평균의 차, 대응표본, 두 비율의 차 — 를 한자리에서 다룬다.

<div class="thmbox" markdown>

### 정리 1. Wilson 점수구간 { .thm }

$$
\frac{
\hat p+\dfrac{(z_{\alpha/2})^2}{2n}
\pm z_{\alpha/2}\sqrt{\dfrac{\hat p(1-\hat p)}n+\dfrac{(z_{\alpha/2})^2}{4n^2}}
}{1+\dfrac{(z_{\alpha/2})^2}{n}}
$$

이 식은 $|\hat p-p|/\sqrt{p(1-p)/n}\le z_{\alpha/2}$ 를 $p$ 에 관하여 푼 것이다.

</div>

??? proof "증명"

    분모의 $p$ 를 바꿔 치우지 않고 그대로 둔 채 부등식을 푼다. 양변을 제곱하면

    $$
    (\hat p-p)^2\le\frac{(z_{\alpha/2})^2}{n}p(1-p)
    $$

    이고, 정리하면 $p$ 에 대한 이차부등식

    $$
    \left(1+\frac{(z_{\alpha/2})^2}{n}\right)p^2
    -\left(2\hat p+\frac{(z_{\alpha/2})^2}{n}\right)p
    +\hat p^{\,2}\le0
    $$

    이 된다. 최고차항의 계수가 양수이므로 해는 두 근 사이의 닫힌구간이고, 근의 공식을 쓰면 위 식이 나온다.

    !!! note "왜 $[0,1]$ 을 벗어나지 않는가"
        두 근은 $p=0$ 과 $p=1$ 에서의 이차식 값이 각각 $\hat p^{\,2}\ge0$, $(1-\hat p)^2\ge0$ 으로 음이 아니라는 사실에서 모두 $[0,1]$ 안에 놓인다. $\hat p=0$ 이어도 상한이 $\dfrac{(z_{\alpha/2})^2/n}{1+(z_{\alpha/2})^2/n}>0$ 이라 폭이 사라지지 않는다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> $n=20$, $x=0$ 에 대하여 $z_{\alpha/2}=1.96$ 인 Wilson 구간을 구하시오.

</div>

??? success "풀이"

    $\hat p=0$ 을 Wilson 공식에 대입하면 하한은 $0$ 이고 상한은

    $$
    \frac{(1.96)^2/20}{1+(1.96)^2/20}
    \approx0.1611
    $$

    이다. 따라서 약 $[0,0.161]$ 이다. Wald 구간 $[0,0]$ 과 달리 불확실성을 반영한다.

<div class="probox" markdown>

**문제 1.** <span class="diff hard" title="어려움"></span> Wald 구간이 $[0,1]$ 밖으로 벗어날 수 있는 이유를 설명하고, Wilson 구간은 왜 이런 문제를 피하는지 정리 4의 유도 관점에서 설명하시오.

</div>

??? success "풀이"

    Wald 구간은 $\hat p$ 를 중심으로 좌우에 같은 길이를 붙이는 대칭구간이므로 $\hat p$ 가 경계 $0$ 또는 $1$ 에 가까우면 모수공간을 벗어날 수 있다. Wilson 구간은 표준오차 안의 미지수 $p$ 를 $\hat p$ 로 바로 바꾸지 않고

    $$
    \frac{|\hat p-p|}{\sqrt{p(1-p)/n}}\le z_{\alpha/2}
    $$

    를 $p$ 에 관하여 풀어 얻는다. 그 결과 경계 부근에서 비대칭이 되며 구간이 $[0,1]$ 안에 머문다.

<div class="defn" markdown>

**정의 1.** [독립인 두 표본]

두 집단에서 얻은 표본이 서로 영향을 주지 않을 때 **독립인 두 표본**이라 한다. 서로 다른 사람으로 구성된 처치집단과 대조집단이 대표적인 예이다. 이때 추정 대상은

$$
\mu_X-\mu_Y
$$

이며 순서를 바꾸면 차의 부호도 바뀐다.

</div>

<div class="thmbox" markdown>

### 정리 2. 독립인 두 평균 차의 평균과 표준오차 { .thm }

두 표본이 서로 독립이면

$$
E[\bar X-\overline Y]=\mu_X-\mu_Y
$$

$$
\operatorname{SE}(\bar X-\overline Y)
=\sqrt{\frac{\sigma_X^2}{n_X}+\frac{\sigma_Y^2}{n_Y}}
$$

</div>

??? proof "증명"

    기댓값의 선형성으로 첫 식을 얻는다. 독립성 때문에 $\operatorname{Cov}(\bar X,\overline Y)=0$ 이므로

    $$
    \operatorname{Var}(\bar X-\overline Y)
    =\operatorname{Var}(\bar X)+\operatorname{Var}(\overline Y)
    =\frac{\sigma_X^2}{n_X}+\frac{\sigma_Y^2}{n_Y}
    $$

    차를 구하지만 두 분산은 더해진다는 점에 주의한다.

<div class="thmbox" markdown>

### 정리 3. 독립인 두 평균 차의 신뢰구간 { .thm }

모표준편차를 알면

$$
(\bar x-\overline y)
\pm z_{\alpha/2}\sqrt{\frac{\sigma_X^2}{n_X}+\frac{\sigma_Y^2}{n_Y}}
$$

이 $\mu_X-\mu_Y$ 의 신뢰구간이다.

모표준편차를 모르면 Welch 구간

$$
(\bar x-\overline y)
\pm t^*_{\nu}\sqrt{\frac{s_X^2}{n_X}+\frac{s_Y^2}{n_Y}}
$$

을 사용한다.

</div>

??? proof "증명"

    정리 2 에서 $\bar X-\bar Y$ 의 평균은 $\mu_X-\mu_Y$ 이고 표준오차는 $\sqrt{\dfrac{\sigma_X^2}{n_X}+\dfrac{\sigma_Y^2}{n_Y}}$ 이다. 두 표본이 독립이므로 그 차도 근사적으로 정규분포를 따르고, 따라서

    $$
    P\!\left(\left|\frac{(\bar X-\overline Y)-(\mu_X-\mu_Y)}{\sqrt{\sigma_X^2/n_X+\sigma_Y^2/n_Y}}\right|\le z_{\alpha/2}\right)\approx1-\alpha
    $$

    이다. 안쪽 부등식을 $\mu_X-\mu_Y$ 에 대하여 풀면 첫째 구간이 나온다.

    모표준편차를 모르면 그 자리에 $s_X,\ s_Y$ 를 넣는데, 분모가 확률변수가 되므로 $z_{\alpha/2}$ 대신 $t^*$ 를 쓴다(「[모표준편차를 모를 때의 t 신뢰구간 — 기본](../one_sample_intervals/t_interval_basics.md)」의 정리 8). 다만 두 표본의 분산이 서로 다르면 자유도가 $n_X-1$ 이나 $n_Y-1$ 로 딱 떨어지지 않아 Welch 의 근사 자유도 $\nu$ 를 쓴다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> $n_A=n_B=100$, $\bar x_A=85$, $\bar x_B=80$, $\sigma_A=6$, $\sigma_B=8$ 일 때 $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    점추정은 $5$, 표준오차는

    $$
    \sqrt{36/100+64/100}=1
    $$

    이므로 구간은 $[3.04,6.96]$ 이다.

<div class="defn" markdown>

**정의 2.** [대응표본]

같은 대상을 전후로 측정했다면 각 쌍의 차 $D_i=X_i-Y_i$ 를 하나의 표본으로 분석한다. 독립인 두 표본 공식에 억지로 넣지 않는다.

</div>

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 왼손과 오른손의 악력을 비교하려고 같은 사람 $30$ 명의 양손 악력을 측정했다. 독립표본과 대응표본 중 어느 방법을 사용해야 하는가?

</div>

??? success "풀이"

    같은 사람에게서 나온 두 측정값은 서로 관련되어 있으므로 대응표본이다. 각 사람의 차 $D_i=$ 왼손 악력 $-$ 오른손 악력을 계산한 뒤 $D_1,\dots,D_{30}$ 을 하나의 표본으로 분석한다.

<div class="thmbox" markdown>

### 정리 4. 대응표본 평균차 { .thm }

$$
\overline d\pm t^*_{n-1}\frac{s_D}{\sqrt n}
$$

이 평균차 $\mu_D$ 의 신뢰구간이다.

</div>

??? proof "증명"

    짝지어진 두 값의 차 $D_i=X_i-Y_i$ 를 하나의 자료로 보면, $D_1,\dots,D_n$ 은 서로 독립인 **한 표본**이다. 따라서 두 표본 문제가 아니라 한 표본의 평균 문제가 되고, 「[모표준편차를 모를 때의 t 신뢰구간 — 기본](../one_sample_intervals/t_interval_basics.md)」의 정리 2 를 $D$ 에 그대로 적용하면 위 구간이 나온다.

    !!! note "짝을 지으면 이득인 까닭"
        같은 사람의 전후를 재면 $X_i$ 와 $Y_i$ 가 강하게 양의 상관을 갖는다. 이때 $\operatorname{Var}(D_i)=\sigma_X^2+\sigma_Y^2-2\operatorname{Cov}(X_i,Y_i)$ 에서 공분산이 크게 빠지므로, 독립인 두 표본으로 다룰 때보다 표준오차가 훨씬 작아진다. 개인차가 큰 자료일수록 짝짓기의 이득이 크다.

<div class="probox" markdown>

**문제 2.** <span class="diff med" title="중간"></span> 같은 학생 $16$ 명의 전후 점수차 평균이 $3.2$, 표준편차가 $4$, $t^*_{15}=2.131$ 일 때 신뢰구간을 구하시오.

</div>

??? success "풀이"

    표준오차는 $1$ 이므로 $[1.069,5.331]$ 이다.

<div class="probox" markdown>

**문제 3.** <span class="diff hard" title="어려움"></span> 한 쌍 $(X_i,Y_i)$ 에 대하여 $\operatorname{Var}(X_i)=\sigma_X^2$, $\operatorname{Var}(Y_i)=\sigma_Y^2$, $\operatorname{Corr}(X_i,Y_i)=\rho$ 이다. 차 $D_i=X_i-Y_i$ 의 분산을 구하고, $\rho>0$ 일 때 대응설계가 유리할 수 있는 이유를 설명하시오.

</div>

??? success "풀이"

    $$
    \operatorname{Var}(D_i)
    =\sigma_X^2+\sigma_Y^2-2\rho\sigma_X\sigma_Y
    $$

    $\rho>0$ 이면 공분산 항이 분산을 줄인다. 같은 사람의 전후 측정처럼 양의 상관이 큰 경우 개인 간 차이를 제거하여 평균차를 더 정밀하게 추정할 수 있다.

<div class="defn" markdown>

**정의 3.** [위험차, 위험비와 오즈비]

두 집단의 성공확률이 $p_1,p_2$ 일 때 다음 양으로 효과를 비교할 수 있다.

| 측도 | 정의 | 기준값 |
|---|---:|---:|
| 위험차 | $p_1-p_2$ | $0$ |
| 위험비 | $p_1/p_2$ | $1$ |
| 오즈비 | $\dfrac{p_1/(1-p_1)}{p_2/(1-p_2)}$ | $1$ |

이 절의 신뢰구간은 주로 해석이 직접적인 위험차 $p_1-p_2$ 를 다룬다. 세 측도는 서로 다른 수이므로 어떤 측도를 보고하는지 밝혀야 한다.

</div>

<div class="thmbox" markdown>

### 정리 5. 두 표본비율 차의 표본분포 { .thm }

독립인 두 표본에서

$$
E[\hat p_1-\hat p_2]=p_1-p_2
$$

이고

$$
\operatorname{SE}(\hat p_1-\hat p_2)
=\sqrt{\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}}
$$

각 집단의 성공·실패 횟수가 충분히 크면 두 비율의 차도 근사적으로 정규분포를 따른다.

</div>

??? proof "증명"

    각 표본에서 $E[\hat p_i]=p_i$ 이므로(「[비율의 표본분포](../../ch05/applications/sample_proportion.md)」) 기댓값의 선형성에서 차의 평균은 $p_1-p_2$ 이다. 두 표본이 서로 독립이므로 분산은 더해지고($\operatorname{Var}(-\hat p_2)=\operatorname{Var}(\hat p_2)$ 이다),

    $$
    \operatorname{Var}(\hat p_1-\hat p_2)
    =\frac{p_1(1-p_1)}{n_1}+\frac{p_2(1-p_2)}{n_2}
    $$

    이다. 제곱근이 표준오차이다. 각 $\hat p_i$ 가 중심극한정리로 정규분포에 가까워지고 독립인 정규변수의 차가 다시 정규분포이므로, 차도 근사적으로 정규분포를 따른다.

<div class="thmbox" markdown>

### 정리 6. 두 비율 차의 신뢰구간 { .thm }

독립인 두 표본에서

$$
(\hat p_1-\hat p_2)
\pm z_{\alpha/2}\sqrt{
\frac{\hat p_1(1-\hat p_1)}{n_1}
+\frac{\hat p_2(1-\hat p_2)}{n_2}
}
$$

이 $p_1-p_2$ 의 근사 신뢰구간이다.

</div>

??? proof "증명"

    정리 5 에서 $\hat p_1-\hat p_2$ 가 근사적으로 평균 $p_1-p_2$, 표준오차 $\sqrt{\dfrac{p_1(1-p_1)}{n_1}+\dfrac{p_2(1-p_2)}{n_2}}$ 인 정규분포를 따르므로, 표준화한 값이 $\pm z_{\alpha/2}$ 안에 들어갈 확률이 약 $1-\alpha$ 이다. 그 부등식을 $p_1-p_2$ 에 대하여 풀고, 모르는 $p_i$ 자리에 $\hat p_i$ 를 넣으면 위 구간이 된다.

    Wald 구간과 같은 이유로 이 구간도 $\hat p_i$ 가 $0$ 이나 $1$ 에 가까우면 폭이 지나치게 좁아진다. 그럴 때는 각 비율에 Wilson 방식을 쓰거나 표본을 늘려야 한다.

<div class="exbox" markdown>

**보기 4.** <span class="diff easy" title="쉬움"></span> A 지역은 $1600$ 명 중 $1280$ 명, B 지역은 $2400$ 명 중 $1440$ 명이 찬성했다. $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    $\hat p_A=0.8$, $\hat p_B=0.6$ 이고 표준오차는 $\sqrt{0.0002}\approx0.0141$ 이다. 따라서

    $$
    0.2\pm1.96(0.0141)\approx[0.1723,0.2277]
    $$

<div class="exbox" markdown>

**보기 5.** <span class="diff easy" title="쉬움"></span> 집단 1의 성공률이 $0.30$, 집단 2의 성공률이 $0.20$ 이다. 위험차, 위험비, 오즈비를 구하시오.

</div>

??? success "풀이"

    $$
    \text{위험차}=0.30-0.20=0.10,
    \qquad
    \text{위험비}=\frac{0.30}{0.20}=1.5
    $$

    $$
    \text{오즈비}
    =\frac{0.30/0.70}{0.20/0.80}
    =\frac{12}{7}\approx1.714
    $$

    같은 자료도 측도에 따라 $10\%\mathrm p$ 증가, $1.5$ 배, 오즈 $1.714$ 배로 표현된다.

<div class="probox" markdown>

**문제 4.** <span class="diff med" title="중간"></span> $p_1-p_2$ 의 $95\%$ 신뢰구간이 $[0.02,0.14]$ 이다. 통계적 결론과 효과의 크기에 대한 결론을 구분하여 설명하시오.

</div>

??? success "풀이"

    구간이 $0$ 을 포함하지 않으므로, 두 비율이 같다고 보기는 어렵다. 또한 $p_1$ 이 $p_2$ 보다 약 $2\%\mathrm p$ 에서 $14\%\mathrm p$ 높다고 추정한다. 이 차이가 실제로 중요한지는 연구 맥락에서 요구되는 최소 중요 차이와 비교해야 한다.

<div class="probox" markdown>

**문제 5.** <span class="diff hard" title="어려움"></span> 평균차의 $95\%$ 신뢰구간이 $[0.01,0.09]$ 이고 실제적으로 의미 있는 최소 차이가 $1$ 이라고 하자. 결과를 해석하시오.

</div>

??? success "풀이"

    구간이 $0$ 을 포함하지 않으므로 통계적으로는 양의 차이가 있다. 그러나 가능한 효과 전체가 $1$ 보다 훨씬 작으므로, 미리 정한 기준에 따르면 실제적으로 중요한 차이라고 보기 어렵다.

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 서로 다른 두 학급의 평균 점수를 비교하는 경우와 같은 학생의 중간·기말 점수를 비교하는 경우를 각각 독립표본 또는 대응표본으로 분류하시오.

</div>

??? success "풀이"

    서로 다른 학생들로 구성된 두 학급은 독립표본으로 보는 것이 일반적이다. 같은 학생의 중간·기말 점수는 한 학생 안에서 관련되어 있으므로 대응표본이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> 두 모평균 차의 신뢰구간이 $[-0.5,3.5]$ 일 때 결론을 설명하시오.

</div>

??? success "풀이"

    $0$ 을 포함하므로 차이 없음을 배제할 수 없다. 그러나 두 평균이 같다고 증명한 것은 아니다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> A 집단은 $500$ 명 중 $300$ 명, B 집단은 $400$ 명 중 $200$ 명이 성공했다. $p_A-p_B$ 의 $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    $\hat p_A=0.6$, $\hat p_B=0.5$ 이므로 점추정값은 $0.1$ 이다. 표준오차는

    $$
    \sqrt{\frac{0.6(0.4)}{500}+\frac{0.5(0.5)}{400}}
    =\sqrt{0.001105}\approx0.03324
    $$

    따라서

    $$
    0.1\pm1.96(0.03324)\approx[0.0348,0.1652]
    $$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 독립인 두 집단에서 $n_X=64$, $n_Y=100$, $\bar x=52$, $\overline y=48$, $\sigma_X=8$, $\sigma_Y=10$ 이다. $\mu_X-\mu_Y$ 의 $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    점추정값은 $4$ 이고

    $$
    \operatorname{SE}=\sqrt{\frac{8^2}{64}+\frac{10^2}{100}}
    =\sqrt2
    $$

    따라서 구간은

    $$
    4\pm1.96\sqrt2\approx[1.228,6.772]
    $$

    이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 두 집단의 표준편차가 $12,6$ 이고 전체 $300$ 명을 배분할 때 평균차의 분산을 최소로 하는 비가 $2:1$ 임을 설명하시오.

</div>

??? success "풀이"

    고정된 총표본에서 $\sigma_X^2/n_X+\sigma_Y^2/n_Y$ 를 최소화하면 $n_X/n_Y=\sigma_X/\sigma_Y=2$ 이다. 따라서 $200명,100명$ 으로 배분한다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span> $n_X+n_Y=N$ 이 고정되어 있을 때

$$
\frac{\sigma_X^2}{n_X}+\frac{\sigma_Y^2}{n_Y}
$$

를 최소화하는 표본배분이 $n_X:n_Y=\sigma_X:\sigma_Y$ 임을 보이시오.

</div>

??? success "풀이"

    코시–슈바르츠 부등식으로

    $$
    \left(\frac{\sigma_X^2}{n_X}+\frac{\sigma_Y^2}{n_Y}\right)(n_X+n_Y)
    \ge(\sigma_X+\sigma_Y)^2
    $$

    등호는 $\sigma_X/n_X=\sigma_Y/n_Y$ 일 때 성립한다. 따라서

    $$
    n_X:n_Y=\sigma_X:\sigma_Y
    $$

    로 배분할 때 평균차의 분산이 최소가 된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span> 대응자료에서 $\sigma_X=\sigma_Y=10$, $\rho=0.8$, $n=25$ 이다. 대응표본 평균차의 표준오차를 구하고, 상관을 무시하여 독립표본처럼 계산한 값과 비교하시오.

</div>

??? success "풀이"

    한 쌍의 차의 분산은

    $$
    \operatorname{Var}(D)=100+100-2(0.8)(10)(10)=40
    $$

    따라서 대응표본 평균차의 표준오차는

    $$
    \sqrt{\frac{40}{25}}\approx1.265
    $$

    독립표본처럼 계산하면

    $$
    \sqrt{\frac{100}{25}+\frac{100}{25}}=\sqrt8\approx2.828
    $$

    이다. 양의 상관을 활용하는 대응분석이 훨씬 정밀하다.
