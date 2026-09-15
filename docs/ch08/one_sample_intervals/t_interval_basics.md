# 모표준편차를 모를 때의 t 신뢰구간 — 기본

모표준편차 $\sigma$ 를 모를 때 표본표준편차 $S$ 로 대신하면 무엇이 달라지는지를 본다. 답은 $z$ 대신 $t$ 를 쓰는 것이다.

<div class="thmbox" markdown>

### 정리 1. 표본분산은 모분산의 불편추정량 { .thm }

평균 $\mu$, 분산 $\sigma^2$ 인 모집단에서 뽑은 크기 $n\ (n\ge2)$ 의 임의표본에 대하여

$$
E\!\left[\sum_{i=1}^n(X_i-\bar X)^2\right]=(n-1)\sigma^2,
\qquad
E[S^2]=\sigma^2
$$

</div>

??? proof "증명"

    **편차를 두 조각으로 나눈다.** 각 편차를 $X_i-\mu=(X_i-\bar X)+(\bar X-\mu)$ 로 쪼개어 제곱해 더하면

    $$
    \sum_{i=1}^n(X_i-\mu)^2
    =\sum_{i=1}^n(X_i-\bar X)^2
    +2(\bar X-\mu)\sum_{i=1}^n(X_i-\bar X)
    +n(\bar X-\mu)^2
    $$

    표본평균의 정의에서 $\displaystyle\sum_{i=1}^n(X_i-\bar X)=\sum X_i-n\bar X=0$ 이므로 가운데 항이 사라지고

    $$
    \sum_{i=1}^n(X_i-\mu)^2=\sum_{i=1}^n(X_i-\bar X)^2+n(\bar X-\mu)^2
    $$

    **양변의 기댓값을 잰다.** 왼쪽은 각 항이 $E[(X_i-\mu)^2]=\sigma^2$ 이므로 $n\sigma^2$ 이다. 오른쪽 끝 항은 $\bar X$ 가 $\mu$ 의 불편추정량이므로(「[편향과 일치성](../../ch07/mean/bias_consistency.md)」) $\operatorname{Var}(\bar X)=\operatorname{MSE}(\bar X)=\dfrac{\sigma^2}{n}$ 이고

    $$
    E\!\left[n(\bar X-\mu)^2\right]=n\operatorname{Var}(\bar X)=\sigma^2
    $$

    따라서

    $$
    E\!\left[\sum_{i=1}^n(X_i-\bar X)^2\right]=n\sigma^2-\sigma^2=(n-1)\sigma^2
    $$

    양변을 $n-1$ 로 나누면 $E[S^2]=\sigma^2$ 이다.

    !!! note "$n-1$ 을 자유도라 부르는 까닭"
        편차 $X_i-\bar X$ 는 $n$ 개이지만 그 합이 언제나 $0$ 이므로, 자유롭게 움직일 수 있는 것은 $n-1$ 개뿐이다. 모평균 $\mu$ 대신 표본에서 계산한 $\bar X$ 를 중심으로 삼은 대가로 정보 한 개를 잃은 셈이고, 그만큼 분모를 줄여 보정한다.

<div class="probox" markdown>

**문제 1.** <span class="diff med" title="중간"></span> 편차제곱의 합을 $n$ 으로 나눈

$$
V=\frac1n\sum_{i=1}^n(X_i-\bar X)^2
$$

을 모분산의 추정량으로 쓸 때 $E[V]$ 와 편향을 구하고, $n$ 이 커지면 어떻게 되는지 말하시오.

</div>

??? success "풀이"

    정리 1 에서 $E\!\left[\sum(X_i-\bar X)^2\right]=(n-1)\sigma^2$ 이므로

    $$
    E[V]=\frac{n-1}{n}\sigma^2
    $$

    편향은

    $$
    E[V]-\sigma^2=\left(\frac{n-1}{n}-1\right)\sigma^2=-\frac{\sigma^2}{n}
    $$

    편향이 음수이므로 $V$ 는 모분산을 **늘 조금 작게** 추정한다. 표본이 스스로 계산한 중심 $\bar X$ 가 그 표본에 가장 가까운 점이기 때문이며, $n$ 이 커지면 $-\dfrac{\sigma^2}{n}\to0$ 이므로 편향은 사라진다. 곧 $V$ 는 편향이 있지만 일치추정량이다.

<div class="thmbox" markdown>

### 정리 2. 정규표본의 t 통계량 { .thm }

정규모집단의 임의표본에서

$$
T=\frac{\bar X-\mu}{S/\sqrt n}\sim t_{n-1}
$$

따라서 $\sigma$ 를 모를 때 모평균의 신뢰구간은

$$
\bar x\pm t^*_{n-1}\frac{s}{\sqrt n}
$$

증명에는 $\dfrac{(n-1)S^2}{\sigma^2}$ 이 자유도 $n-1$ 인 카이제곱분포를 따르고 $\bar X$ 와 서로 독립이라는 사실이 필요하므로 이 책의 범위를 넘는다. 여기서는 **$\sigma$ 를 $s$ 로 바꾸어 쓰면 $z$ 대신 $t$ 를 써야 한다**는 결론만 쓴다.

</div>

<div class="defn" markdown>

**정의 1.** [자유도]

$t_{n-1}$ 의 아래첨자 $n-1$ 을 **자유도**라 한다. $S^2$ 을 계산할 때 $n$ 개 편차의 합이 $0$ 이라는 제약 하나가 생기므로 독립적으로 정할 수 있는 편차가 $n-1$ 개이기 때문이다.

</div>

<div class="thmbox" markdown>

### 정리 3. t 분포와 표준정규분포의 관계 { .thm }

$t$ 분포는 표준정규분포와 같이 $0$ 에 대하여 대칭이지만 꼬리가 더 두껍다. 자유도가 커지면

$$
t_\nu\Longrightarrow\mathrm N(0,1)
\qquad(\nu\to\infty)
$$

이므로 $t$ 임계값은 대응하는 $z$ 임계값에 가까워진다.

</div>

??? proof "증명"

    **왜 꼬리가 두꺼운가.** $Z=\dfrac{\bar X-\mu}{\sigma/\sqrt n}$ 의 분모는 상수이지만, $T$ 의 분모 $S/\sqrt n$ 은 표본마다 달라지는 확률변수이다. $S$ 가 우연히 작게 나온 표본에서는 $T$ 가 크게 튀므로, 분자의 흔들림 위에 분모의 흔들림이 얹혀 극단값이 더 자주 나온다.

    **왜 정규로 수렴하는가.** 큰 수의 법칙에서 $S^2\to\sigma^2$ 이므로 $\nu=n-1$ 이 커지면 분모의 흔들림이 사라지고 $T$ 는 $Z$ 와 같아진다. 그래서 자유도가 커질수록 두 분포가 겹치고 임계값도 가까워진다 — 예를 들어 $t^*_{30}=2.042$, $t^*_{120}=1.980$ 로 $z_{\alpha/2}=1.96$ 에 다가간다.

!!! warning "$z$ 인가 $t$ 인가"
    알려진 $\sigma$ 를 쓰면 $z$, 표본표준편차 $s$ 로 대체하면 $t$ 를 쓴다. 작은 표본에서 $t$ 분포는 추가 불확실성을 반영하여 꼬리가 더 두껍다.

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> $n=16$, $\bar x=52$, $s=8$, $t^*_{15}=2.131$ 일 때 $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    표준오차는 $8/4=2$ 이므로

    $$
    52\pm2.131\cdot2=[47.738,56.262]
    $$

<div class="probox" markdown>

**문제 2.** <span class="diff med" title="중간"></span> 같은 표본에서 $s=8$, $n=16$ 일 때 $z_{\alpha/2}=1.96$ 을 쓴 오차한계와 $t^*_{15}=2.131$ 을 쓴 오차한계를 비교하고, $t$ 구간이 더 넓은 이유를 설명하시오.

</div>

??? success "풀이"

    표준오차는 $8/4=2$ 이다. $z$ 오차한계는 $1.96(2)=3.92$, $t$ 오차한계는 $2.131(2)=4.262$ 이다. $\sigma$ 를 모르기 때문에 $S$ 로 추정하면서 생기는 추가 불확실성을 $t$ 분포의 두꺼운 꼬리가 반영한다.

!!! warning "$t$ 구간의 조건"
    표본이 작을 때 정확한 $t$ 구간을 쓰려면 모집단이 정규분포여야 한다. 표본이 충분히 크면 중심극한정리 때문에 중간 정도의 비정규성에는 비교적 강건하지만, 극단값이 많거나 분포가 매우 비대칭이면 큰 표본에서도 자료를 먼저 확인해야 한다.

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> $n=25$, $\bar x=10$, $s=5$, $t^*_{24}=2.064$ 일 때 $95\%$ 구간을 구하시오.

</div>

??? success "풀이"

    표준오차는 $1$ 이므로 $[7.936,12.064]$ 이다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $n=10$, $\bar x=25$, $s=6$, $t^*_9=2.262$ 일 때 $95\%$ 신뢰구간을 구하시오.

</div>

??? success "풀이"

    표준오차는 $6/\sqrt{10}\approx1.897$ 이고 오차한계는

    $$
    2.262(1.897)\approx4.291
    $$

    따라서 신뢰구간은 약 $[20.709,29.291]$ 이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span> 모표준편차가 알려져 있고 표본크기가 $n$ 일 때 $95\%$ 신뢰구간의 길이를 절반으로 줄이는 두 가지 방법을 논하시오.

</div>

??? success "풀이"

    같은 신뢰수준과 같은 모집단에서는 구간 길이가 $2(1.96)\sigma/\sqrt n$ 이다. 표본크기를 $4n$ 으로 늘리면 길이가 절반이 된다. 또는 표준편차가 절반인 더 균질한 모집단·층을 조사하면 같은 $n$ 에서 길이가 절반이 된다. 후자는 연구 대상을 바꿀 수 있으므로 일반화 범위도 함께 바뀐다는 점에 주의해야 한다.
