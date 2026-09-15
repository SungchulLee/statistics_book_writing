# 추정량의 품질 — 기본

추정량을 고를 때 무엇을 보아야 하는지를 편향·분산·평균제곱오차로 정리한다. 뒤의 절들이 쓰는 기본 어휘를 여기서 세운다.

<div class="defn" markdown>

**정의 1.** [점추정]

모수 $\theta$ 를 하나의 통계량 $T$ 로 추정하는 것을 점추정이라 한다. $T$ 는 추정량, 관측된 값 $t$ 는 점추정값이다.

평균제곱오차 $\operatorname{MSE}(T)=E[(T-\theta)^2]$ 는 아래 정의 3 에서 세운다.

</div>

<div class="defn" markdown>

**정의 2.** [일치추정량]

표본크기 $n$ 에 따른 추정량 $T_n$ 이 모든 $\varepsilon>0$ 에 대하여

$$
P(|T_n-\theta|>\varepsilon)\longrightarrow0
\qquad(n\to\infty)
$$

을 만족하면 $T_n$ 을 모수 $\theta$ 의 **일치추정량**이라 한다. 즉 표본이 커질수록 추정량이 참값 가까이에 있을 확률이 $1$ 에 가까워진다.

</div>

<div class="defn" markdown>

**정의 3.** [평균제곱오차]

추정량 $T$ 와 모수 $\theta$ 의 차이를 제곱하여 평균한

$$
\operatorname{MSE}(T)=E[(T-\theta)^2]
$$

를 **평균제곱오차**라 한다. 이는 추정량의 정확도를 편향과 변동성을 함께 고려하여 평가한다.

</div>

<div class="thmbox" markdown>

### 정리 1. 편향–분산 분해 { .thm }

$$
\operatorname{MSE}(T)
=\operatorname{Var}(T)+\operatorname{Bias}(T)^2
$$

</div>

??? proof "증명"

    $T-\theta=(T-E[T])+(E[T]-\theta)$ 로 나누어 제곱한다. 교차항의 기댓값은 $E[T-E[T]]=0$ 이므로 사라진다.

<div class="thmbox" markdown>

### 정리 2. 평균제곱오차의 분해 { .thm }

$$
\operatorname{MSE}(T)
=\operatorname{Var}(T)+\{\operatorname{Bias}(T)\}^2
$$

</div>

??? proof "증명"

    $T-\theta=\{T-E[T]\}+\{E[T]-\theta\}$ 로 쓰고 제곱한다. 교차항의 기댓값은

    $$
    2(E[T]-\theta)E[T-E[T]]=0
    $$

    이므로 결과를 얻는다.

<div class="thmbox" markdown>

### 정리 3. 표본평균의 불편성과 일치성 { .thm }

$$
E[\bar X]=\mu,
\qquad
\operatorname{MSE}(\bar X)=\frac{\sigma^2}{n}\to0
$$

따라서 표본평균은 불편추정량이면서 일치추정량이다.

</div>

??? proof "일치성의 확인"

    체비쇼프 부등식에 의해 모든 $\varepsilon>0$ 에 대하여

    $$
    P(|\bar X-\mu|>\varepsilon)
    \le\frac{\operatorname{Var}(\bar X)}{\varepsilon^2}
    =\frac{\sigma^2}{n\varepsilon^2}\longrightarrow0
    $$

    따라서 표본평균은 확률수렴의 의미에서 $\mu$ 로 수렴한다.

<div class="thmbox" markdown>

### 정리 4. MSE 수렴은 일치성을 보장한다 { .thm }

$$
\operatorname{MSE}(T_n)=E[(T_n-\theta)^2]\longrightarrow0
$$

이면 $T_n$ 은 $\theta$ 의 일치추정량이다.

</div>

??? proof "증명"

    마르코프 부등식을 $(T_n-\theta)^2$ 에 적용하면

    $$
    P(|T_n-\theta|>\varepsilon)
    =P((T_n-\theta)^2>\varepsilon^2)
    \le\frac{E[(T_n-\theta)^2]}{\varepsilon^2}\to0
    $$

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> $E[T_1]=\theta$, $\operatorname{Var}(T_1)=9$ 이고 $E[T_2]=\theta+1$, $\operatorname{Var}(T_2)=4$ 일 때 MSE를 비교하시오.

</div>

??? success "풀이"

    $\operatorname{MSE}(T_1)=9$, $\operatorname{MSE}(T_2)=4+1=5$ 이다. 작은 편향을 허용해 분산을 크게 줄이면 MSE가 더 작을 수 있다.

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> $E[T_n]=\theta$ 이고 $\operatorname{Var}(T_n)=9/n$ 이다. $T_n$ 의 불편성과 일치성을 판단하시오.

</div>

??? success "풀이"

    모든 $n$ 에서 $E[T_n]=\theta$ 이므로 불편추정량이다. 또한

    $$
    \operatorname{MSE}(T_n)=\operatorname{Var}(T_n)=\frac9n\to0
    $$

    이므로 정리 4 에 따라 일치추정량이다.

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 두 추정량 $T_1,T_2$ 에 대하여

$$
E[T_1]=\theta,\quad \operatorname{Var}(T_1)=9,
\qquad
E[T_2]=\theta+3,\quad \operatorname{Var}(T_2)=4
$$

이다. 평균제곱오차가 더 작은 추정량을 고르시오.

</div>

??? success "풀이"

    $\operatorname{MSE}(T_1)=9$ 이고 $\operatorname{MSE}(T_2)=4+3^2=13$ 이다. $T_2$ 는 분산이 더 작지만 편향이 커서 전체 오차는 오히려 크다. 따라서 $T_1$ 이 낫다. 분산만으로도, 편향만으로도 우열을 정할 수 없고 정리 3 처럼 둘을 함께 더해 보아야 한다.

<div class="probox" markdown>

**문제 1.** <span class="diff med" title="중간"></span> $E[T_n]=\theta+1/\sqrt n$, $\operatorname{Var}(T_n)=2/n$ 일 때 편향과 MSE를 구하고 일치성을 판단하시오.

</div>

??? success "풀이"

    편향은 $1/\sqrt n$ 이고

    $$
    \operatorname{MSE}(T_n)=\frac2n+\left(\frac1{\sqrt n}\right)^2=\frac3n\to0
    $$

    따라서 각 $n$ 에서는 편향되어 있지만 일치추정량이다.

<div class="probox" markdown>

**문제 2.** <span class="diff hard" title="어려움"></span> 평균이 $\mu$, 분산이 $\sigma^2$ 인 임의표본에서

$$
T_a=a\bar X+(1-a)c
$$

로 $\mu$ 를 추정한다. 여기서 $c$ 는 알려진 상수이다. $T_a$ 의 편향과 평균제곱오차를 구하고, 이를 최소화하는 $a$ 를 구하시오.

</div>

??? success "풀이"

    $$
    E[T_a]-\mu=(1-a)(c-\mu),
    \qquad
    \operatorname{Var}(T_a)=\frac{a^2\sigma^2}{n}
    $$

    따라서

    $$
    \operatorname{MSE}(T_a)
    =\frac{a^2\sigma^2}{n}+(1-a)^2(c-\mu)^2
    $$

    이를 $a$ 에 관하여 미분하면 최소점은

    $$
    a^*=\frac{(c-\mu)^2}{(c-\mu)^2+\sigma^2/n}
    $$

    이다. 실제로는 $\mu$ 가 미지이므로 이 식을 그대로 사용할 수는 없지만, 약간의 편향을 허용하여 분산을 줄이는 원리를 보여 준다.

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 추정량 $T$ 가 $E[T]=\theta-2$, $\operatorname{Var}(T)=5$ 를 만족한다. 편향과 평균제곱오차를 구하시오.

</div>

??? success "풀이"

    편향은 $-2$ 이고

    $$
    \operatorname{MSE}(T)=5+(-2)^2=9
    $$

    이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span> 크기 $N=500$ 인 모집단의 분산이 $25$ 이다. 비복원 단순임의추출로 $n=80$ 을 뽑을 때 $\operatorname{Var}(\bar X)$ 를 구하고, 복원추출로 계산한 분산과 비교하시오.

</div>

??? success "풀이"

    비복원추출에서는

    $$
    \operatorname{Var}(\bar X)
    =\frac{25}{80}\frac{500-80}{500-1}
    \approx0.2630
    $$

    복원추출에서는 $25/80=0.3125$ 이다. 비복원추출은 이미 뽑은 개체가 다시 나오지 않아 추출값들 사이에 음의 공분산이 생기므로 분산이 더 작다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> $E[T_n]=\theta+1/n$, $\operatorname{Var}(T_n)=4/n$ 일 때 MSE를 구하고 일치성을 설명하시오.

</div>

??? success "풀이"

    $\operatorname{MSE}(T_n)=4/n+1/n^2\to0$ 이므로 $T_n$ 은 일치추정량이다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span> $T_n$ 이 불편추정량이고 $\operatorname{Var}(T_n)=\sigma^2/n$ 이다. $a\ne0$ 인 상수에 대하여 $U_n=aT_n$ 으로 $a\theta$ 를 추정할 때 불편성, MSE와 일치성을 보이시오.

</div>

??? success "풀이"

    $$
    E[U_n]=aE[T_n]=a\theta
    $$

    이므로 불편추정량이다. 또한

    $$
    \operatorname{MSE}(U_n)
    =\operatorname{Var}(U_n)
    =a^2\operatorname{Var}(T_n)
    =\frac{a^2\sigma^2}{n}\to0
    $$

    따라서 $U_n$ 은 $a\theta$ 의 일치추정량이다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 표본평균 $\bar X$ 대신 $T_n=\bar X+1/n$ 으로 $\mu$ 를 추정한다. $T_n$ 의 편향과 MSE를 구하고, 불편성 및 일치성을 판단하시오.

</div>

??? success "풀이"

    $$
    \operatorname{Bias}(T_n)=E[\bar X]+\frac1n-\mu=\frac1n
    $$

    $$
    \operatorname{MSE}(T_n)
    =\operatorname{Var}(\bar X)+\frac1{n^2}
    =\frac{\sigma^2}{n}+\frac1{n^2}\to0
    $$

    따라서 $T_n$ 은 불편추정량은 아니지만 일치추정량이다.
