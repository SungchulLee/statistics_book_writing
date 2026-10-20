# 3차원 회귀평면

## 개요

이 페이지는 설명변수가 둘인 다중선형회귀를 3차원 공간의 평면으로 시각화한다. 인공 광고자료(TV와 Radio 지출로 Sales를 예측)를 써서 회귀평면을 적합하고, 자료점을 3차원에 흩뿌리고, 잔차선을 그려 다중회귀의 기하학적 해석을 보인다.

---

## 1. 수학적 배경

설명변수가 둘일 때 다중선형회귀 모형은

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \varepsilon_i.
$$

적합값 $\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{i1} + \hat{\beta}_2 x_{i2}$는 $(x_1, x_2, y)$ 공간에서 **평면**을 이룬다. 잔차 $e_i = y_i - \hat{y}_i$는 자료점에서 평면까지의 수직 거리이다.

결정계수는 설명된 분산의 비율을 잰다.

$$
R^2 = 1 - \frac{\mathrm{RSS}}{\mathrm{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}.
$$

각 계수는 **부분적 해석**을 갖는다. $\hat{\beta}_1$은 $x_2$를 고정했을 때 $x_1$이 한 단위 늘어날 때 기대되는 $y$의 변화이다. 기하학적으로 $\hat{\beta}_1$은 $x_1$ 방향으로 잰 평면의 기울기이다.

### 자료 생성과 적합

<div class="exbox" markdown>

**보기 1.** <span class="diff easy" title="쉬움"></span> 설명변수 둘인 자료와 적합. $\text{TV} \sim U(0, 300)$, $\text{Radio} \sim U(0, 50)$ 을 독립으로 뽑고 $\text{Sales} = 5 + 0.04\,\text{TV} + 0.15\,\text{Radio} + \varepsilon$, $\varepsilon \sim N(0, 1.5^2)$ 로 $n = 150$ 개를 만든 뒤 적합한다.

**(1)** 두 설명변수가 **독립**이므로 $\operatorname{Var}(\hat\beta_j) \approx \sigma^2/S_{jj}$ 로 근사할 수 있다. 균등분포의 분산이 $a^2/12$ 임을 써서 $S_{\text{RR}}$, $S_{\text{TT}}$ 와 두 계수의 **표준오차를 자료를 보기 전에** 예측하시오.

**(2)** 적합해 보고 (1)의 예측과 견주시오. 세 참 모수가 $95\%$ 신뢰구간에 들어오는가. 예측한 표준오차와 실제 표준오차가 어긋난다면 그 까닭은 무엇인가.

</div>

??? success "풀이"

    **(1) 해석적으로.** $\operatorname{Var}(\hat{\boldsymbol\beta}) = \sigma^2 (X^TX)^{-1}$ 이고, 두 설명변수가 독립이면 중심화한 뒤 $X^TX$ 가 거의 대각이 되어

    $$
    \operatorname{Var}(\hat\beta_j) \approx \frac{\sigma^2}{S_{jj}}, \qquad S_{jj} = \sum_i (x_{ij} - \bar x_j)^2
    $$

    이다. $S_{jj}$ 의 기댓값은 $(n-1)\operatorname{Var}(x_j)$ 이고 $U(0,a)$ 의 분산이 $a^2/12$ 이므로

    $$
    E[S_{\text{RR}}] = 149 \times \frac{50^2}{12} = 31041.67, \qquad
    E[S_{\text{TT}}] = 149 \times \frac{300^2}{12} = 1117500
    $$

    이다. 표준오차는 그 제곱근으로 나눈다.

    $$
    \operatorname{se}(\hat\beta_{\text{Radio}}) \approx \frac{1.5}{\sqrt{31041.67}} = 0.008514, \qquad
    \operatorname{se}(\hat\beta_{\text{TV}}) \approx \frac{1.5}{\sqrt{1117500}} = 0.001419
    $$

    **TV 쪽 표준오차가 여섯 배 작다.** TV 가 여섯 배 넓은 구간에 퍼져 있어 $S_{\text{TT}}$ 가 $36$ 배 크기 때문이다. 이것이 A.4절의 $\operatorname{Var}(\hat\beta_1) = \sigma^2/S_{xx}$ 가 말하는 바다. **설명변수를 넓게 퍼뜨릴수록 그 계수가 정밀해진다.**

    **(2) 수치적으로.**

    ```python
    import numpy as np
    import statsmodels.api as sm
    from sklearn.linear_model import LinearRegression

    # 설명변수가 둘이면 회귀선이 아니라 회귀평면이 된다. 그것을 눈으로 본다.
    np.random.seed(42)
    n = 150
    TV = np.random.uniform(0, 300, n)
    Radio = np.random.uniform(0, 50, n)
    Sales = 5 + 0.04 * TV + 0.15 * Radio + np.random.normal(0, 1.5, n)

    X = np.column_stack([Radio, TV])
    y = Sales

    model = LinearRegression()
    model.fit(X, y)

    beta_0 = model.intercept_
    beta_1 = model.coef_[0]  # Radio
    beta_2 = model.coef_[1]  # TV

    # 균등분포의 분산은 a^2/12 이므로 S_jj 를 미리 예측할 수 있다.
    S_RR, S_TT = ((Radio - Radio.mean()) ** 2).sum(), ((TV - TV.mean()) ** 2).sum()
    print(f"S_RR 실제 {S_RR:12.2f}   예측 (n-1)*50^2/12  = {(n - 1) * 2500 / 12:12.2f}")
    print(f"S_TT 실제 {S_TT:12.2f}   예측 (n-1)*300^2/12 = {(n - 1) * 90000 / 12:12.2f}")
    print(f"corr(TV, Radio) = {np.corrcoef(TV, Radio)[0, 1]:+.4f}  (독립이므로 0 에 가깝다)")
    print()

    res = sm.OLS(y, sm.add_constant(X)).fit()
    names = ["절편", "Radio", "TV"]
    truth = [5.0, 0.15, 0.04]
    theory_se = [np.nan, 1.5 / np.sqrt((n - 1) * 2500 / 12), 1.5 / np.sqrt((n - 1) * 90000 / 12)]
    print(f"{'항':8s}{'참값':>8s}{'추정값':>11s}{'SE':>10s}{'이론 SE':>11s}"
          f"{'95% CI':>24s}  판정")
    for k, (nm, tv, tse) in enumerate(zip(names, truth, theory_se)):
        lo, hi = res.conf_int()[k]
        print(f"{nm:8s}{tv:8.4f}{res.params[k]:11.6f}{res.bse[k]:10.6f}"
              f"{tse:11.6f}   [{lo:8.5f}, {hi:8.5f}]  {'포함' if lo <= tv <= hi else '벗어남'}")
    print()
    print(f"s = {np.sqrt(res.mse_resid):.4f}   (참 sigma = 1.5,  비 {np.sqrt(res.mse_resid) / 1.5:.4f})")
    print(f"R^2 = {model.score(X, y):.6f}")
    ```

    출력:

    ```
    S_RR 실제     31633.01   예측 (n-1)*50^2/12  =     31041.67
    S_TT 실제   1179148.00   예측 (n-1)*300^2/12 =   1117500.00
    corr(TV, Radio) = +0.0357  (독립이므로 0 에 가깝다)

    항             참값        추정값        SE      이론 SE                  95% CI  판정
    절편        5.0000   5.089605  0.291330        nan   [ 4.51387,  5.66534]  포함
    Radio     0.1500   0.142954  0.007850   0.008514   [ 0.12744,  0.15847]  포함
    TV        0.0400   0.041303  0.001286   0.001419   [ 0.03876,  0.04384]  포함

    s = 1.3953   (참 sigma = 1.5,  비 0.9302)
    R^2 = 0.905404
    ```

    **세 참값이 모두 신뢰구간 안에 있다.** $S_{jj}$ 의 예측도 좋다. $S_{\text{RR}}$ 은 $31042$ 예측에 $31633$ 실제로 $1.9\%$ 차이, $S_{\text{TT}}$ 는 $1117500$ 예측에 $1179148$ 실제로 $5.5\%$ 차이다. 둘 다 같은 방향으로 크게 나왔는데, $S_{jj}$ 의 상대 표준오차가 $\sqrt{2/(n-1)} \approx 11.6\%$ 가 아니라 균등분포에서는 더 작으므로 이 정도 어긋남은 흔하다.

    **표준오차는 예측보다 작게 나왔다.** Radio 는 $0.008514$ 예측에 $0.007850$ 실제로 $-7.8\%$, TV 는 $0.001419$ 에 $0.001286$ 으로 $-9.4\%$ 다. 까닭이 두 가지로 나뉜다. 첫째, $s = 1.3953$ 이 참 $\sigma = 1.5$ 보다 $7.0\%$ 작다. 이 표본의 오차가 우연히 작게 뽑힌 것이다. 둘째, $S_{jj}$ 가 예측보다 커서 분모가 커졌다. 두 효과를 함께 넣으면

    $$
    1.5 \times \frac{0.9302}{\sqrt{31633.01}} = 0.007845
    $$

    로 실제 $0.007850$ 과 소수 다섯째 자리에서 맞는다(남은 $6\times10^{-6}$ 은 $X^TX$ 가 완전히 대각이 아니기 때문이다). **어긋남의 정체가 완전히 설명된다.** 이론식이 틀린 것이 아니라 $\sigma$ 와 $S_{jj}$ 를 참값이 아닌 실현값으로 바꿔야 했던 것이다.

    참값 $0.15$ 대 추정값 $0.142954$ 의 차이는 $-0.007$ 로 표준오차 $0.00785$ 의 $0.90$ 배다. TV 는 $+0.0013$ 으로 표준오차의 $1.01$ 배다. 둘 다 정상 범위다. $\square$

### 회귀평면 격자 만들기

<div class="exbox" markdown>

**보기 2.** <span class="diff easy" title="쉬움"></span> 회귀평면 격자 만들기. `np.arange` 로 두 축의 격자를 만들고 칸마다 적합값을 계산한다.

**(1)** `np.arange(0, 50, 5)` 와 `np.arange(0, 300, 30)` 은 각각 몇 개의 점을 주며 끝점 $50$ 과 $300$ 을 포함하는가. 격자가 덮는 범위를 **자료의 범위**와 견주어, 격자 밖으로 나가는 점이 몇 개인지 세시오.

**(2)** 격자를 $10\times10$ 대신 $2\times2$ 로 줄여도 그려지는 면이 **똑같다**. 왜 그런지 말하고 수치로 확인하시오. 어떤 모형에서는 이것이 성립하지 않는가.

</div>

??? success "풀이"

    **(1) 둘 다 $10$ 개이고 끝점은 포함하지 않는다.** `np.arange(start, stop, step)` 은 **$\text{stop}$ 을 포함하지 않는** 반열린구간을 준다. 그래서 Radio 격자는 $0, 5, \ldots, 45$ 로 끝나고 TV 격자는 $0, 30, \ldots, 270$ 으로 끝난다.

    자료는 그보다 넓다. $\text{Radio} \sim U(0,50)$ 이고 $\text{TV} \sim U(0,300)$ 이니 $45$ 를 넘는 Radio 가 기대값으로 $150 \times 0.1 = 15$ 개, $270$ 을 넘는 TV 가 $150 \times 0.1 = 15$ 개다. 둘 중 하나라도 넘을 확률은 $1 - 0.9^2 = 0.19$ 이므로 **약 $29$ 개 점이 격자 밖**으로 나간다. 끝점을 포함하려면 `np.linspace(0, 50, 11)` 를 쓰거나 `np.arange(0, 50.1, 5)` 처럼 $\text{stop}$ 을 밀어야 한다.

    **(2) 적합된 함수가 $\text{Radio}$ 와 $\text{TV}$ 에 대해 **일차**이기 때문이다.** $\hat y = \hat\beta_0 + \hat\beta_1 r + \hat\beta_2 t$ 는 평면이고, 평면은 네 꼭짓점 — 사실은 일직선 위에 없는 세 점 — 으로 완전히 정해진다. 격자를 촘촘히 하는 것은 같은 평면 위에 점을 더 찍는 일에 지나지 않는다.

    더 정확히 말하면 이 함수는 **이중선형보간과 정확히 일치한다.** 직사각형의 네 꼭짓점 값으로 이중선형보간을 하면 $r$, $t$, $rt$ 항이 나오는데, 평면에는 $rt$ 항의 계수가 $0$ 이라 보간이 평면을 그대로 되살린다.

    성립하지 않는 경우는 **적합된 함수가 비선형일 때**다. 교호작용 $\hat\beta_3 rt$ 를 넣으면 쌍곡포물면이 되어 네 꼭짓점만으로는 휘어짐을 그릴 수 없고, 다항항이나 스플라인을 넣으면 더하다. 그때는 격자가 촘촘해야 곡면의 모양이 보인다.

    ```python
    # 평면을 그리려면 두 축의 격자를 만들고 칸마다 적합값을 계산한다.
    Radio_range = np.arange(0, 50, 5)
    TV_range = np.arange(0, 300, 30)
    Radio_mesh, TV_mesh = np.meshgrid(Radio_range, TV_range)

    Sales_mesh = beta_0 + beta_1 * Radio_mesh + beta_2 * TV_mesh

    print(f"Radio 격자 {Radio_range}   점 {len(Radio_range)}개,  마지막 {Radio_range[-1]}")
    print(f"TV    격자 {TV_range}   점 {len(TV_range)}개,  마지막 {TV_range[-1]}")
    print(f"격자 모양 {Radio_mesh.shape},  Sales_mesh 모양 {Sales_mesh.shape}")
    print()
    print(f"자료 범위:  Radio [{Radio.min():.3f}, {Radio.max():.3f}]   "
          f"TV [{TV.min():.3f}, {TV.max():.3f}]")
    print(f"격자 범위:  Radio [0, {Radio_range[-1]}]   TV [0, {TV_range[-1]}]")
    print(f"격자 밖으로 나가는 점:  Radio>45 {np.sum(Radio > 45)}개,  "
          f"TV>270 {np.sum(TV > 270)}개,  둘 중 하나라도 {np.sum((Radio > 45) | (TV > 270))}개"
          f"  ({np.mean((Radio > 45) | (TV > 270)):.1%})")
    print()
    # 평면은 선형이므로 꼭짓점 네 개면 똑같은 면이 나온다.
    corners_R, corners_T = np.meshgrid([0, 45], [0, 270])
    corners_S = beta_0 + beta_1 * corners_R + beta_2 * corners_T
    print("꼭짓점 네 개만으로 같은 평면이 되는가")
    print(f"  (0,0)     격자 {Sales_mesh[0, 0]:.6f}   꼭짓점 {corners_S[0, 0]:.6f}")
    print(f"  (45,0)    격자 {Sales_mesh[0, -1]:.6f}   꼭짓점 {corners_S[0, 1]:.6f}")
    print(f"  (0,270)   격자 {Sales_mesh[-1, 0]:.6f}   꼭짓점 {corners_S[1, 0]:.6f}")
    print(f"  (45,270)  격자 {Sales_mesh[-1, -1]:.6f}   꼭짓점 {corners_S[1, 1]:.6f}")
    # 격자 한가운데 칸이 네 꼭짓점의 선형보간과 같은가
    mid = beta_0 + beta_1 * 22.5 + beta_2 * 135
    bilinear = corners_S.mean()
    print(f"\n격자 중심 (22.5, 135) 의 평면값 {mid:.6f}")
    print(f"네 꼭짓점의 평균               {bilinear:.6f}   차이 {abs(mid - bilinear):.2e}")
    ```

    출력:

    ```
    Radio 격자 [ 0  5 10 15 20 25 30 35 40 45]   점 10개,  마지막 45
    TV    격자 [  0  30  60  90 120 150 180 210 240 270]   점 10개,  마지막 270
    격자 모양 (10, 10),  Sales_mesh 모양 (10, 10)

    자료 범위:  Radio [0.253, 49.503]   TV [1.657, 296.066]
    격자 범위:  Radio [0, 45]   TV [0, 270]
    격자 밖으로 나가는 점:  Radio>45 15개,  TV>270 14개,  둘 중 하나라도 29개  (19.3%)

    꼭짓점 네 개만으로 같은 평면이 되는가
      (0,0)     격자 5.089605   꼭짓점 5.089605
      (45,0)    격자 11.522515   꼭짓점 11.522515
      (0,270)   격자 16.241330   꼭짓점 16.241330
      (45,270)  격자 22.674240   꼭짓점 22.674240

    격자 중심 (22.5, 135) 의 평면값 13.881923
    네 꼭짓점의 평균               13.881923   차이 1.78e-15
    ```

    **(1)의 예측이 거의 그대로 나왔다.** $\text{Radio} > 45$ 가 $15$ 개(예측 $15$), $\text{TV} > 270$ 이 $14$ 개(예측 $15$), 둘 중 하나라도 넘는 점이 $29$ 개로 전체의 $19.3\%$ 다. 예측한 $19\%$ 와 맞는다. 자료의 최댓값이 Radio $49.503$, TV $296.066$ 이니 **그림에서 그 $29$ 개 점은 평면 바깥에 떠 있게 된다.** 평면이 끝난 자리 위로 점이 매달려 보이는 것은 모형의 문제가 아니라 격자를 잘못 잡은 것이다.

    **(2)도 확실하다.** 네 꼭짓점의 값이 $10\times10$ 격자의 네 구석과 소수 여섯째 자리까지 같고, 격자 중심 $(22.5, 135)$ 의 평면값 $13.881923$ 이 네 꼭짓점의 **단순평균**과 $1.8\times10^{-15}$ 안에서 같다. 평균이 보간값과 같은 것은 중심이 네 꼭짓점의 중점이고 함수가 일차이기 때문이다. **평면에서는 보간과 외삽이 모두 계수 세 개로 끝난다.**

    그러므로 이 격자를 만드는 계산은 **그림을 위한 것이지 모형을 위한 것이 아니다.** 모형은 이미 보기 1에서 계수 세 개로 끝났고, 격자는 그 세 수를 눈에 보이는 면으로 펼친 것에 지나지 않는다. $\square$

### 3차원 시각화

<div class="exbox" markdown>

**보기 3.** <span class="diff easy" title="쉬움"></span> 평면과 잔차를 3차원으로. 반투명 회귀평면 위에 점 $150$ 개를 흩뿌리고 다섯 점마다 하나씩 잔차선을 긋는다.

**(1)** 그림의 빨간 선분은 모두 $y$ 축과 나란하다. 점에서 **평면까지의 최단거리**는 잔차와 어떤 비로 다른지 평면의 법선벡터로 유도하고, 이 자료에서 그 비를 구하시오.

**(2)** 그림에서 읽을 수 있는 것과 **읽을 수 없는 것**을 수치와 함께 적으시오.

</div>

??? success "풀이"

    유도할 식은 (1) 하나뿐이고, (2)는 **그림에서 무엇이 읽히고 무엇이 읽히지 않는가**가 전부다.

    **(1) 해석적으로.** 평면 $y = \hat\beta_0 + \hat\beta_1 r + \hat\beta_2 t$ 를 $(r, t, y)$ 공간의 음함수로 적으면

    $$
    \hat\beta_1 r + \hat\beta_2 t - y + \hat\beta_0 = 0
    $$

    이므로 법선벡터가 $\mathbf n = (\hat\beta_1,\ \hat\beta_2,\ -1)$ 이고 $\|\mathbf n\| = \sqrt{1 + \hat\beta_1^2 + \hat\beta_2^2}$ 다. 점 $(r_i, t_i, y_i)$ 에서 평면까지의 최단거리는 점-평면 거리 공식으로

    $$
    d_i = \frac{\lvert \hat\beta_1 r_i + \hat\beta_2 t_i - y_i + \hat\beta_0 \rvert}{\|\mathbf n\|}
    = \frac{\lvert \hat y_i - y_i \rvert}{\sqrt{1 + \hat\beta_1^2 + \hat\beta_2^2}}
    = \frac{\lvert e_i \rvert}{\sqrt{1 + \hat\beta_1^2 + \hat\beta_2^2}}
    $$

    이다. 곧 **잔차와 최단거리의 비가 $\sqrt{1 + \hat\beta_1^2 + \hat\beta_2^2}$ 로 모든 점에서 같은 상수**다. 비가 상수라는 점이 중요하다. 최소제곱이 수직거리의 제곱합을 최소화하는 일과 최단거리의 제곱합을 최소화하는 일이 **같은 평면을 주지 않는** 까닭은 계수가 바뀌면 이 비도 바뀌기 때문이고, 계수가 고정된 뒤에는 두 거리가 비례한다.

    이 자료에서는 $\hat\beta_1 = 0.142954$, $\hat\beta_2 = 0.041303$ 이 둘 다 작아 비가 $1$ 에 매우 가깝다.

    **(2) 수치적으로.**

    ```python
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 회귀평면. 반투명으로 그려 점이 앞뒤 어디에 있는지 보이게 한다.
    ax.plot_surface(Radio_mesh, TV_mesh, Sales_mesh,
                    alpha=0.3, cmap='coolwarm')

    # 관측점
    ax.scatter(Radio, TV, Sales, c='blue', s=50, alpha=0.6)

    # 점에서 평면까지 수직으로 선을 긋는다. 그 길이가 잔차이고,
    # 최소제곱은 이 길이들의 제곱합을 가장 작게 만드는 평면을 고른 것이다.
    # 다 그리면 지저분하므로 다섯 점마다 하나씩만 그린다.
    y_pred = model.predict(X)
    for i in range(0, n, 5):
        ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]],
                [y[i], y_pred[i]], 'r-', alpha=0.3)

    ax.set_xlabel('Radio')
    ax.set_ylabel('TV')
    ax.set_zlabel('Sales')
    plt.tight_layout()
    plt.show()
    ```

    ![회귀평면](./img/regression_plane_3d_62.png)

    설명변수가 둘이면 회귀직선이 아니라 회귀**평면**이 된다. 점들이 평면 위아래로 흩어진 거리가 잔차다. 이제 그 거리를 수로 잰다.

    ```python
    # 그림에서 읽을 수치를 미리 찍어 둔다.
    residuals = y - model.predict(X)
    factor = np.sqrt(1 + beta_1 ** 2 + beta_2 ** 2)
    print(f"잔차선을 그린 점 {len(range(0, n, 5))}개 / 전체 {n}개")
    print(f"잔차  평균 {residuals.mean():+.2e}   표준편차 {residuals.std(ddof=3):.4f}"
          f"   최대 |e| {np.abs(residuals).max():.4f}")
    print(f"평면 위 {np.sum(residuals > 0)}개,  아래 {np.sum(residuals < 0)}개")
    print()
    print(f"평면의 법선 (-b1, -b2, 1) 의 길이 = sqrt(1 + b1^2 + b2^2) = {factor:.6f}")
    print(f"  수직 잔차 / 평면까지의 최단거리 = {factor:.6f}   (차이 {factor - 1:.2%})")
    print(f"  가장 큰 잔차 {np.abs(residuals).max():.4f} 에 대해 "
          f"최단거리는 {np.abs(residuals).max() / factor:.4f}")
    print()
    print(f"R^2 = {model.score(X, y):.6f},  RSS = {np.sum(residuals ** 2):.4f},"
          f"  TSS = {np.sum((y - y.mean()) ** 2):.4f}")
    ```

    출력:

    ```
    잔차선을 그린 점 30개 / 전체 150개
    잔차  평균 +5.46e-15   표준편차 1.3953   최대 |e| 5.0688
    평면 위 77개,  아래 73개

    평면의 법선 (-b1, -b2, 1) 의 길이 = sqrt(1 + b1^2 + b2^2) = 1.011010
      수직 잔차 / 평면까지의 최단거리 = 1.011010   (차이 1.10%)
      가장 큰 잔차 5.0688 에 대해 최단거리는 5.0136

    R^2 = 0.905404,  RSS = 286.2067,  TSS = 3025.5718
    ```

    **(1)의 비가 $1.011010$ 이다.** 수직 잔차가 최단거리보다 $1.10\%$ 길다. 가장 큰 잔차 $5.0688$ 조차 최단거리로는 $5.0136$ 이니 차이가 $0.055$ 에 지나지 않는다. **이 그림에서는 두 거리가 사실상 같다.**

    까닭을 오해하면 안 된다. 기울기가 **작아서** 그런 것이 아니라 **단위 때문에** 작아 보이는 것이다. TV 를 천 달러 단위에서 백만 달러 단위로 바꾸면 $\hat\beta_2$ 가 $0.0413$ 에서 $41.3$ 으로 커지고 비가 $\sqrt{1 + 41.3^2} \approx 41.3$ 이 된다. **최소제곱이 수직거리를 쓰는 것과 최단거리를 쓰는 것은 근본적으로 다른 방법이며**, 후자는 전직교회귀(또는 주성분회귀)라 불리고 **단위를 바꾸면 답이 바뀐다.** 최소제곱은 $y$ 축 방향만 재므로 설명변수의 단위에 영향받지 않는다. 이 그림에서 두 거리가 비슷해 보이는 것은 **우연한 단위 선택의 결과일 뿐**이다.

    **(2) 그림에서 읽히는 것.** 점들이 평면 위 $77$ 개, 아래 $73$ 개로 거의 반반이고 잔차의 평균이 $5.5\times10^{-15}$ 다. 정규방정식이 $\sum e_i = 0$ 을 강제하므로 이것은 확인이지 발견이 아니다. 잔차의 표준편차가 $1.3953$ 이고 $\text{Sales}$ 의 범위가 대략 $5$ 에서 $23$ 이니 **점구름의 두께가 평면이 덮는 높이의 $8\%$ 쯤**이다. 그래서 그림에서 점들이 평면에 꽤 붙어 보이고, $R^2 = 0.9054$ 가 그것을 수로 적은 값이다.

    **그림이 읽을 수 없는 것은 네 가지다.** 첫째, **잔차선이 $150$ 개 중 $30$ 개만 그려져 있다.** `range(0, n, 5)` 가 다섯 점마다 하나씩 고르므로, 가장 큰 잔차 $5.0688$ 을 가진 점이 그려졌는지 알 수 없다. 극단값을 찾는 도구로 이 그림을 쓸 수 없다. 둘째, **등분산인지 가늠할 수 없다.** 3차원 투영에서는 어느 점이 앞이고 뒤인지 흐려져 "$\hat y$ 가 클 때 흩어짐이 큰가"를 눈으로 판정할 수 없다. 그 판정은 잔차 대 적합값의 2차원 그림이 한다. 셋째, **다중공선성이 보이지 않는다.** 여기서는 $\operatorname{corr}(\text{TV}, \text{Radio}) = 0.0357$ 로 점들이 바닥 전체에 고르게 깔려 있지만, 두 변수가 상관되면 점들이 바닥의 한 직선 근처로 몰려 평면의 기울기가 불안정해진다. 그 쏠림은 바닥을 **위에서 내려다보아야** 보인다. 넷째, **격자가 자료를 다 덮지 못한다**(보기 2). $29$ 개 점이 평면 밖에 떠 있다.

    그러므로 이 그림의 몫은 **"평면이다"와 "잔차는 세로로 잰다"는 두 가지를 눈에 새기는 것**이고, 진단은 다른 그림들이 맡는다. 설명변수가 셋이 되면 이 그림 자체가 불가능해지지만 그 두 사실은 그대로 남는다. $\square$

---

## 2. 해석

- **회귀평면**: 색칠된 곡면은 TV와 Radio 지출의 모든 조합에 대한 모형의 예측을 나타낸다. 평면의 기울기 방향이 계수들의 상대적 크기를 반영한다.
- **자료점**: 평면 주위에 흩어진 파란 점들이다. 평면 위의 점은 양의 잔차를, 아래의 점은 음의 잔차를 갖는다.
- **잔차선**: 자료점과 평면을 잇는 빨간 수직 선분이다. OLS는 이 선분들의 길이 제곱의 합을 최소화한다.
- **계수의 의미**: $\hat{\beta}_{\text{Radio}} = 0.15$, $\hat{\beta}_{\text{TV}} = 0.04$라면, (TV를 고정했을 때) Radio 지출 \$1,000 증가는 Sales 0.15 단위 증가와 연관되고, TV 지출 \$1,000 증가는 0.04 단위 증가와 연관된다.
- **한계**: 설명변수가 3개 이상이면 회귀 곡면이 초평면이 되어 직접 시각화할 수 없다. 3차원 시각화는 설명변수가 둘일 때만 쓸 수 있는 교육용 도구이다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span> 모형에 세 번째 설명변수(예: Newspaper 지출)를 추가하라. 그 결과 회귀 곡면을 3차원에 그릴 수 없는 이유를 설명하고 시각화의 대안을 제시하라.

</div>

??? success "풀이"

    설명변수가 셋이면 회귀 곡면은 4차원 공간($x_1, x_2, x_3, y$)의 초평면이 되어 직접 그릴 수 없다. 대안으로는 (1) 부분회귀 그림(다른 설명변수의 선형 효과를 제거한 뒤 $y$를 $x_j$에 대해 그린다), (2) 단면 그림(설명변수 둘을 평균에 고정하고 $y$를 나머지 하나에 대해 그린다), (3) 추가변수 그림, (4) 점추정값과 신뢰구간을 보여주는 계수 그림이 있다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> RSS와 TSS로부터 $R^2$를 직접 계산하라. `model.score(X, y)`와 일치하는지 확인하라.

</div>

??? success "풀이"

    ```python
    y_pred = model.predict(X)
    RSS = np.sum((y - y_pred) ** 2)
    TSS = np.sum((y - y.mean()) ** 2)
    R2_manual = 1 - RSS / TSS
    R2_sklearn = model.score(X, y)
    print(f"Manual R2: {R2_manual:.4f}")
    print(f"Sklearn R2: {R2_sklearn:.4f}")
    print(f"Match: {np.isclose(R2_manual, R2_sklearn)}")
    ```

    출력:

    ```
    Manual R2: 0.9054
    Sklearn R2: 0.9054
    Match: True
    ```

    직접 계산한 $R^2$와 sklearn의 값이 정확히 같다. $R^2 = 1 - \text{RSS}/\text{TSS}$라는 정의를 확인한 셈이다.

    정의상 두 값은 동일하다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> 3차원 그림을 여러 시점으로 돌려 보라. 어느 각도에서 잔차가 가장 작아 보이는가? 기하학적으로 설명하라.

</div>

??? success "풀이"

    잔차 선분은 모두 **$y$(Sales)축과 나란한 수직선**이다. 따라서 잔차의 겉보기 길이는 평면의 방향이 아니라 오직 시선이 $y$축과 이루는 각도에 달려 있다.

    시선 방향의 단위벡터를 $\mathbf{d}$, $y$축 방향을 $\hat{\mathbf{z}}$라 하면, 길이 $\ell$인 수직 선분이 화면에 투영되는 길이는

    $$
    \ell \left\lVert \hat{\mathbf{z}} - (\hat{\mathbf{z}} \cdot \mathbf{d})\,\mathbf{d} \right\rVert = \ell \sin\theta, \qquad \theta = \angle(\hat{\mathbf{z}}, \mathbf{d})
    $$

    이다. 따라서

    - **바로 위(또는 아래)에서 내려다볼 때**($\mathbf{d} \parallel \hat{\mathbf{z}}$, $\theta = 0$): 잔차가 점으로 축소되어 **가장 작아 보인다**. `matplotlib`에서는 `ax.view_init(elev=90, azim=0)`이다.
    - **수평 시점**($\mathbf{d} \perp \hat{\mathbf{z}}$, $\theta = 90^\circ$): 잔차가 실제 길이로 보여 **가장 크게 보인다**. `ax.view_init(elev=0, ...)`이다.

    평면이 선으로 보이는(측면으로 보이는) 각도는 이와 별개의 이야기이다. 시선이 평면 **안에** 놓일 때, 곧 시선이 평면의 법선벡터와 **직교**할 때 평면이 측면으로 보인다. 법선벡터를 **따라** 보면 평면은 측면이 아니라 정면으로 보여 화면을 가득 채운다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> (절편이 포함될 때) OLS 잔차가 각 설명변수 $j$에 대해 $\sum_{i=1}^n e_i = 0$과 $\sum_{i=1}^n x_{ij} e_i = 0$을 만족함을 증명하라.

</div>

??? success "풀이"

    정규방정식은 $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$, 곧 $\mathbf{X}^\top\mathbf{e} = \mathbf{0}$이다. $\mathbf{X}$의 첫 열이 $\mathbf{1}$(절편 열)이므로 첫 번째 식이 $\mathbf{1}^\top\mathbf{e} = \sum e_i = 0$을 준다. $(j+1)$번째 식은 $\mathbf{x}_j^\top\mathbf{e} = \sum x_{ij}e_i = 0$을 준다. 이 직교성 조건들이 OLS의 근본 성질이다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span> 모형 $\text{Sales} = \beta_0 + \beta_1 \cdot \text{Radio} + \beta_2 \cdot \text{TV} + \varepsilon$에서 $\beta_1$(부분계수)과 Sales를 Radio에만 회귀시켜 얻은 계수의 차이를 설명하라.

</div>

??? success "풀이"

    단순회귀 계수 $\tilde{\beta}_1$은 Radio와 Sales 사이의 전체 연관을 포착하며, 여기에는 TV를 거치는 간접 연관도 들어 있다(예: Radio에 많이 쓰는 기업이 TV에도 많이 쓴다면). 부분계수 $\hat{\beta}_1$은 Radio와 Sales 양쪽에서 TV의 선형 영향을 제거한 뒤 Radio의 직접 효과만 분리한다. 형식적으로 $\hat{\beta}_1$은 Sales를 TV에 회귀시킨 잔차를 Radio를 TV에 회귀시킨 잔차에 회귀시킨 기울기와 같다(Frisch-Waugh-Lovell 정리). 두 계수는 Radio와 TV가 무상관일 때에만 일치한다. $\square$

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span> 보기 3 은 "TV 를 천 달러 단위에서 백만 달러 단위로 바꾸면 $\hat\beta_2$ 가 $0.0413$ 에서 $41.3$ 으로 커지고 비가 $\sqrt{1 + 41.3^2} \approx 41.3$ 이 된다" 고 했다. $\text{TV}' = \text{TV}/1000$ 으로 같은 자료를 다시 적합한다고 하자.

**(1)** $\hat\beta_0$, $\hat\beta_1$, $\hat\beta_2$, 잔차 $e_i$, $R^2$ 가 각각 어떻게 바뀌는지 적합값의 식으로 보이시오.

**(2)** 새 비 $\sqrt{1 + \hat\beta_1^2 + (\hat\beta_2')^2}$ 와, 보기 3 의 가장 큰 잔차 $5.0688$ 에 대응하는 평면까지의 최단거리를 구하시오.

**(3)** TV 를 $10^k$ 로 나누어 단위를 계속 키우면 최단거리는 어디로 가는가. 그 극한이 "최단거리의 제곱합을 최소화한다"는 기준에 대해 무엇을 말해 주는가.

</div>

??? success "풀이"

    **(1) 기울기 하나만 바뀌고 나머지는 전부 그대로다.** 적합값이

    $$
    \hat y_i = \hat\beta_0 + \hat\beta_1 r_i + \hat\beta_2' \frac{t_i}{1000}
    $$

    인데 $\hat\beta_2' = 1000\,\hat\beta_2$ 로 두면 모든 $i$ 에서 원래 적합값과 **완전히 같아진다.** 최소제곱해는 $X$ 가 완전계수이면 유일하므로, 새 문제의 해가 바로 이것이다. 곧

    $$
    \hat\beta_0 = 5.089605, \qquad \hat\beta_1 = 0.142954, \qquad \hat\beta_2' = 1000 \times 0.041303 = 41.302687
    $$

    이고 잔차 $e_i$ 는 하나도 달라지지 않으므로 $\mathrm{RSS} = 286.2067$, $\mathrm{TSS} = 3025.5718$, $R^2 = 0.905404$ 도 그대로다. 일반적으로 설명변수의 열을 $c$ 로 나누면 그 계수가 $c$ 배 되고 적합은 불변이다. **OLS 는 설명변수의 단위에 대해 등변(equivariant)이다.**

    **(2) 비가 $41.3$ 으로 뛴다.**

    $$
    \sqrt{1 + 0.142954^2 + 41.302687^2} = \sqrt{1706.9336} = 41.315038
    $$

    이므로 가장 큰 잔차에 대응하는 최단거리는

    $$
    d_{\max} = \frac{5.0688}{41.315038} = 0.122686
    $$

    이다. 원래 단위에서는 $5.0688/1.011010 = 5.0136$ 이었다. **수직 잔차는 $5.0688$ 에서 꼼짝도 하지 않았는데 "평면까지의 최단거리"만 $40.9$ 배 줄었다.**

    **(3) $0$ 으로 간다.** TV 를 $10^k$ 로 나누면 그 계수가 $10^k \hat\beta_2$ 이므로

    $$
    \sqrt{1 + \hat\beta_1^2 + (10^k \hat\beta_2)^2} \sim 10^k \lvert \hat\beta_2 \rvert
    \quad \Longrightarrow \quad
    d_i \sim \frac{\lvert e_i \rvert}{10^k \times 0.041303} \to 0
    $$

    이다. 적합이 글자 하나 바뀌지 않았는데도 최단거리는 단위 선택만으로 **얼마든지 작게** 만들 수 있다. 그러므로 최단거리의 제곱합은 **두 축의 단위가 서로 견줄 수 있을 때에만 뜻이 있는 양**이다. 수직거리는 $y$ 축의 단위만 쓰므로 설명변수를 어떻게 재든 영향받지 않는다. 보기 3 이 두 거리가 비슷해 보이는 것을 "우연한 단위 선택의 결과"라 한 까닭이 이것이다. $\square$

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span> 보기 3 은 $\mathrm{RSS} = 286.2067$, $\mathrm{TSS} = 3025.5718$, $n = 150$ 을 주었다. 설명변수가 $k$ 개일 때 수정 결정계수는 $\bar R^2 = 1 - \dfrac{\mathrm{RSS}/(n-k-1)}{\mathrm{TSS}/(n-1)}$ 이다.

**(1)** 이 모형($k=2$)의 $\bar R^2$ 를 구하고 $R^2 = 0.905404$ 와 견주시오.

**(2)** 연습문제 1 처럼 Newspaper 를 넣어 $k=3$ 으로 늘렸을 때 $\bar R^2$ 가 **커지려면** RSS 가 적어도 얼마나 줄어야 하는가. 그것은 RSS 의 몇 %이며 어떤 양과 같은가.

**(3)** (2)의 조건을 새 변수의 부분 $F$ 통계량 $F = \dfrac{\mathrm{RSS}_2 - \mathrm{RSS}_3}{\mathrm{RSS}_3/(n-4)}$ 로 다시 적고, 유의수준 $5\%$ 의 문턱 $F_{1,146;\,0.95} = 3.9059$ 와 견주시오. $\bar R^2$ 를 변수 선택의 기준으로 쓸 때 무엇을 조심해야 하는가.

</div>

??? success "풀이"

    **(1)** $\mathrm{RSS}/(n-k-1) = 286.2067/147 = 1.946984$ 이고 $\mathrm{TSS}/(n-1) = 3025.5718/149 = 20.305851$ 이므로

    $$
    \bar R^2 = 1 - \frac{1.946984}{20.305851} = 1 - 0.095883 = 0.904117
    $$

    이다. $R^2 = 0.905404$ 보다 $0.001287$ 작다. 분자 $1.946984$ 는 보기 1 의 $s = 1.3953$ 의 제곱, 곧 **잔차평균제곱** $s^2$ 다.

    **(2) $s^2 = 1.946984$ 보다 많이 줄어야 한다.** $\mathrm{TSS}$ 는 변수를 더해도 그대로이므로 $\bar R^2$ 가 커지는 것은 $\mathrm{RSS}/(n-k-1)$ 이 작아지는 것과 같다. 따라서

    $$
    \frac{\mathrm{RSS}_3}{146} < \frac{286.2067}{147} = 1.946984
    \quad \Longleftrightarrow \quad
    \mathrm{RSS}_3 < 284.259716
    $$

    이고 감소량은 $\Delta = 286.2067 - 284.259716 = 1.946984$ 를 넘어야 한다. 이것이 RSS 의 $1/147 = 0.68\%$ 다. 하한이 하필 $s^2$ 과 같은 것은 우연이 아니다. 자유도 하나를 쓰는 값이 잔차평균제곱 한 몫이기 때문이다. 이 문턱에서 $R^2$ 는 $1 - 284.2597/3025.5718 = 0.906048$ 로 $0.000644$ 만 오른다.

    **(3) 조건은 정확히 $F > 1$ 이다.** $\mathrm{RSS}_2 = \mathrm{RSS}_3 + \Delta$ 를 넣으면

    $$
    \Delta > \frac{\mathrm{RSS}_2}{147} = \frac{\mathrm{RSS}_3 + \Delta}{147}
    \ \Longleftrightarrow\ 146\,\Delta > \mathrm{RSS}_3
    \ \Longleftrightarrow\ \frac{\Delta}{\mathrm{RSS}_3/146} > 1
    \ \Longleftrightarrow\ F > 1
    $$

    이다. 유의성의 문턱은 $F > 3.9059$, 곧 $\lvert t \rvert > 1.9763$ 인데 $\bar R^2$ 의 문턱은 $F > 1$, 곧 $\lvert t \rvert > 1$ 이다. **$\lvert t \rvert$ 가 $1$ 과 $1.9763$ 사이인 변수는 유의하지 않은데도 $\bar R^2$ 를 올린다.** 더구나 쓸모가 전혀 없는 변수($\beta_3 = 0$)라도 $F_{1,146} > 1$ 일 확률이 $0.3190$ 이니, 무작위 변수를 셋 던져 넣으면 하나쯤은 $\bar R^2$ 를 올려 준다.

    그러므로 $\bar R^2$ 는 $R^2$ 처럼 변수를 넣을수록 반드시 오르지는 않는다는 점에서 개선이지만, **벌점이 자유도 하나에 $s^2$ 한 몫뿐이라 매우 느슨하다.** 변수 선택을 하려면 벌점이 더 센 기준($\mathrm{AIC}$ 는 $2$, $\mathrm{BIC}$ 는 $\ln 150 = 5.01$ 에 해당)이나 검정을 써야 한다. $\square$

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span> 보기 3 (2)는 "그림에서 다중공선성이 보이지 않는다" 며 $\operatorname{corr}(\text{TV}, \text{Radio}) = 0.0357$ 을 들었다. 보이지 않는 그 효과를 수로 잰다. 설명변수가 둘일 때 정확한 식은

$$
\operatorname{se}(\hat\beta_1) = \frac{s}{\sqrt{S_{11}(1 - \rho^2)}}, \qquad \rho = \operatorname{corr}(x_1, x_2)
$$

이며, 보기 1 의 근사식은 여기서 $1 - \rho^2$ 를 뺀 것이다.

**(1)** $s = 1.395344$, $S_{\text{RR}} = 31633.01$, $\rho = 0.035691$ 로 보기 1 의 $\operatorname{se}(\hat\beta_{\text{Radio}}) = 0.007850$ 을 되살리고, 공선성이 부풀린 몫이 몇 %인지 말하시오.

**(2)** $s$, $S_{\text{RR}}$, 추정값 $\hat\beta_{\text{Radio}} = 0.142954$ 는 그대로이고 $\rho$ 만 커진다고 하자. 표준오차가 두 배가 되는 $\rho$ 는 얼마인가.

**(3)** $95\%$ 신뢰구간이 $0$ 을 품으려면 $\rho$ 가 얼마나 커야 하는가($t_{0.975,\,147} = 1.97623$). 분산팽창계수 $\mathrm{VIF} = 1/(1-\rho^2)$ 로도 적고, 이 수에서 읽을 것을 한 줄로 쓰시오.

</div>

??? success "풀이"

    **(1) 부풀린 몫이 $0.064\%$ 다.** 먼저 $1 - \rho^2$ 를 뺀 값이

    $$
    \frac{s}{\sqrt{S_{\text{RR}}}} = \frac{1.395344}{177.8567} = 0.0078453
    $$

    이고 팽창인자가 $1/\sqrt{1 - 0.035691^2} = 1.000638$ 이므로

    $$
    \operatorname{se}(\hat\beta_{\text{Radio}}) = 0.0078453 \times 1.000638 = 0.0078503
    $$

    로 보기 1 이 statsmodels 에서 받은 $0.007850$ 과 같다. 공선성이 더한 몫은 $0.064\%$, 곧 소수 여섯째 자리에서 겨우 보인다. **보기 1 의 근사식이 잘 맞은 까닭이 바로 이것이다.**

    **(2) $\rho = \sqrt3/2 = 0.866025$.** 팽창인자가 $2$ 가 되려면

    $$
    \frac{1}{\sqrt{1-\rho^2}} = 2 \ \Longrightarrow\ 1 - \rho^2 = \frac14 \ \Longrightarrow\ \rho = \frac{\sqrt3}{2} = 0.866025
    $$

    이고 이때 $\mathrm{VIF} = 4$, $\operatorname{se} = 0.015691$ 이다. **상관이 $0.866$ 이나 되어야 표준오차가 겨우 두 배다.** 참고로 $\rho = 0.9$ 에서도 $\mathrm{VIF} = 5.263$, 팽창인자 $2.294$ 에 그친다.

    **(3) $\rho = 0.994101$, $\mathrm{VIF} = 85.02$.** 신뢰구간이 $0$ 을 품는 것은 $\lvert t \rvert < 1.97623$ 과 같고

    $$
    \frac{0.142954}{\operatorname{se}} < 1.97623
    \quad \Longleftrightarrow \quad
    \operatorname{se} > \frac{0.142954}{1.97623} = 0.072337
    $$

    이다. 필요한 팽창인자는 $0.072337/0.0078453 = 9.2204$ 이므로

    $$
    \frac{1}{1-\rho^2} = 9.2204^2 = 85.0156
    \ \Longrightarrow\ 1 - \rho^2 = 0.011762
    \ \Longrightarrow\ \rho = 0.994101
    $$

    이다. **공선성은 $t$ 값을 $\sqrt{1-\rho^2}$ 배로 깎을 뿐이다.** $\rho = 0$ 에서 $t = 0.142954/0.0078453 = 18.22$ 인 이 자료는 $9.22$ 배나 깎여야 유의성을 잃으므로 $\rho = 0.99$ 로도 꺾이지 않는다. 거꾸로 $t$ 가 $3$ 쯤인 변수라면 $3/1.97623 = 1.518$ 배만 깎여도 유의성을 잃으니 $\rho \approx 0.752$ 로 충분하다. 곧 **공선성이 얼마나 치명적인지는 $\rho$ 혼자가 아니라 $\rho$ 와 신호 세기가 함께 정한다.** 보기 3 이 "평면의 기울기가 불안정해진다"고 말한 것은 추정값의 **분산**에 관한 말이고, 유의성이 사라진다는 말은 아니다. $\square$

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span> 연습문제 5 는 부분계수 $\hat\beta_1$ 과 Sales 를 Radio 에만 회귀시킨 계수 $\tilde\beta_1$ 이 "Radio 와 TV 가 무상관일 때에만" 일치한다고 했다. 그 어긋남을 식으로 적는다.

**(1)** 연습문제 4 의 직교성 $\sum_i e_i = 0$, $\sum_i x_{ij} e_i = 0$ 만 써서

$$
\tilde\beta_1 = \hat\beta_1 + \hat\beta_2 \delta, \qquad \delta = \frac{S_{\text{RT}}}{S_{\text{RR}}}
$$

를 유도하시오. 여기서 $S_{\text{RT}} = \sum_i (r_i - \bar r)(t_i - \bar t)$ 이고 $\delta$ 는 TV 를 Radio 에 단순회귀한 기울기이다.

**(2)** 이 자료에서 $\delta$ 와 $\tilde\beta_1$ 을 $\rho = 0.035691$, $S_{\text{RR}} = 31633.01$, $S_{\text{TT}} = 1179148.00$ 으로 손으로 계산한 뒤 코드로 확인하시오. 어긋남은 표준오차 $0.007850$ 의 몇 배인가. 상관이 $0.036$ 밖에 안 되는데도 어긋남이 그만큼 나오는 까닭은 무엇인가.

</div>

??? success "풀이"

    **(1) 유도.** 적합의 항등식 $y_i = \hat\beta_0 + \hat\beta_1 r_i + \hat\beta_2 t_i + e_i$ 를 중심화하여 $r$ 과의 교차적을 잡으면

    $$
    S_{\text{R}y} = \sum_i (r_i - \bar r)(y_i - \bar y)
    = \hat\beta_1 S_{\text{RR}} + \hat\beta_2 S_{\text{RT}} + \sum_i (r_i - \bar r) e_i
    $$

    이다. 마지막 항은

    $$
    \sum_i (r_i - \bar r) e_i = \sum_i r_i e_i - \bar r \sum_i e_i = 0 - 0 = 0
    $$

    으로 **연습문제 4 의 두 직교성이 그대로 지워 버린다.** 단순회귀 기울기가 $\tilde\beta_1 = S_{\text{R}y}/S_{\text{RR}}$ 이므로 양변을 $S_{\text{RR}}$ 로 나누면

    $$
    \tilde\beta_1 = \hat\beta_1 + \hat\beta_2 \frac{S_{\text{RT}}}{S_{\text{RR}}} = \hat\beta_1 + \hat\beta_2 \delta
    $$

    를 얻는다. $\delta = 0$, 곧 $S_{\text{RT}} = 0$ 일 때만 두 계수가 같다. $\square$

    **(2) 손으로.** $\delta$ 를 상관으로 적으면 $\delta = \rho \sqrt{S_{\text{TT}}/S_{\text{RR}}}$ 이고

    $$
    \sqrt{\frac{1179148.00}{31633.01}} = \sqrt{37.2756} = 6.105397,
    \qquad \delta = 0.035691 \times 6.105397 = 0.217908
    $$

    이므로

    $$
    \tilde\beta_{\text{Radio}} = 0.142954 + 0.041303 \times 0.217908 = 0.142954 + 0.009000 = 0.151954
    $$

    이다. 어긋남 $0.009000$ 은 표준오차 $0.007850$ 의 $1.1465$ 배다.

    ```python
    # 연습문제 4 의 직교성만 쓰면 단순회귀 기울기가 부분계수에서 바로 나온다.
    Rc, Tc, yc = Radio - Radio.mean(), TV - TV.mean(), y - y.mean()
    S_RR, S_RT, S_TT = (Rc ** 2).sum(), (Rc * Tc).sum(), (Tc ** 2).sum()
    delta = S_RT / S_RR      # TV 를 Radio 에 단순회귀한 기울기
    gamma = S_RT / S_TT      # Radio 를 TV 에 단순회귀한 기울기

    simple_R = (Rc * yc).sum() / S_RR    # Sales 를 Radio 에만 회귀
    simple_T = (Tc * yc).sum() / S_TT    # Sales 를 TV 에만 회귀

    print(f"S_RT = {S_RT:.4f},  delta = S_RT/S_RR = {delta:.6f},  gamma = S_RT/S_TT = {gamma:.8f}")
    print()
    print(f"Radio  부분계수 {beta_1:.6f}   단순회귀 {simple_R:.6f}   "
          f"공식 b1+b2*delta = {beta_1 + beta_2 * delta:.6f}   차이 {simple_R - beta_1:+.6f}")
    print(f"TV     부분계수 {beta_2:.6f}   단순회귀 {simple_T:.6f}   "
          f"공식 b2+b1*gamma = {beta_2 + beta_1 * gamma:.6f}   차이 {simple_T - beta_2:+.6f}")
    print()
    print(f"Radio 쪽 차이는 se(b_Radio) = 0.007850 의 {(simple_R - beta_1) / 0.007850:.4f} 배")
    tol = 1e-6
    print(f"직교성(연습문제 4) 확인:  |sum(e)|, |sum(Radio*e)|, |sum(TV*e)| 모두 {tol:g} 미만인가 "
          f"{bool(max(abs(residuals.sum()), abs((Radio * residuals).sum()), abs((TV * residuals).sum())) < tol)}")
    ```

    출력:

    ```text
    S_RT = 6893.1340,  delta = S_RT/S_RR = 0.217910,  gamma = S_RT/S_TT = 0.00584586

    Radio  부분계수 0.142954   단순회귀 0.151954   공식 b1+b2*delta = 0.151954   차이 +0.009000
    TV     부분계수 0.041303   단순회귀 0.042138   공식 b2+b1*gamma = 0.042138   차이 +0.000836

    Radio 쪽 차이는 se(b_Radio) = 0.007850 의 1.1465 배
    직교성(연습문제 4) 확인:  |sum(e)|, |sum(Radio*e)|, |sum(TV*e)| 모두 1e-06 미만인가 True
    ```

    **공식이 소수 여섯째 자리까지 단순회귀 기울기를 맞힌다.** Radio 는 $0.151954$, TV 는 $0.042138$ 로 양쪽 모두 정확하다.

    까닭은 $\delta = \rho \sqrt{S_{\text{TT}}/S_{\text{RR}}}$ 에 있다. **$\rho = 0.0357$ 이라는 작은 상관이 두 변수의 퍼짐 비 $6.105$ 배로 증폭되어 $\delta = 0.2179$ 가 된다.** 그래서 $\hat\beta_2 \delta = 0.0090$ 이라는, 표준오차의 $1.15$ 배나 되는 치우침이 생긴다. 반대 방향은 $\gamma = \rho\sqrt{S_{\text{RR}}/S_{\text{TT}}} = 0.00585$ 로 $1/6.105$ 배 **축소**되어 TV 쪽 어긋남은 $0.000836$, 표준오차 $0.001286$ 의 $0.65$ 배에 그친다.

    요점은 **무상관성이 "상관계수가 작다"가 아니라 "$S_{\text{RT}} = 0$"이어야 한다**는 것이고, 독립으로 뽑은 유한 표본에서 그것은 거의 일어나지 않는다는 것이다. 보기 1 이 "독립이므로 $0$ 에 가깝다"고 넘긴 $0.0357$ 이 여기서는 표준오차 한 몫만큼의 차이를 만든다. $\square$

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span> 보기 3 (1)은 계수가 고정되면 수직 잔차와 최단거리가 비례하지만 "최소제곱이 수직거리의 제곱합을 최소화하는 일과 최단거리의 제곱합을 최소화하는 일이 같은 평면을 주지 않는다"고 했다. 그것을 미분으로 확인한다. 보기 3 (1)의 거리식이 모든 $\boldsymbol\beta = (\beta_0, \beta_1, \beta_2)$ 에서 성립하므로 최단거리의 제곱합은

$$
D(\boldsymbol\beta) = \sum_i d_i^2 = \frac{\mathrm{RSS}(\boldsymbol\beta)}{1 + \beta_1^2 + \beta_2^2}
$$

이다.

**(1)** OLS 해 $\hat{\boldsymbol\beta}$ 에서 $\partial D/\partial\beta_0 = 0$ 임을 보이고, 그것이 뜻하는 바를 말하시오.

**(2)** $\hat{\boldsymbol\beta}$ 에서 $\partial D/\partial\beta_1$ 과 $\partial D/\partial\beta_2$ 를 구하시오($\mathrm{RSS} = 286.2067$, $\hat\beta_1 = 0.142954$, $\hat\beta_2 = 0.041303$). 그 부호에서 $D$ 를 최소화하는 평면의 기울기가 OLS 보다 **큰지 작은지** 판정하시오.

**(3)** 이 기울기벡터가 $(\hat\beta_1, \hat\beta_2)$ 와 나란함을 보이시오. 그런데 실제로 $D$ 를 최소화하면 $\hat\beta_1$ 은 $0.89\%$, $\hat\beta_2$ 는 $0.006\%$ 밖에 커지지 않는다. 두 기울기가 같은 비율로 움직이지 않는 까닭을 $S_{\text{RR}} = 31633.01$, $S_{\text{TT}} = 1179148.00$ 으로 설명하시오.

</div>

??? success "풀이"

    $f(\boldsymbol\beta) = 1 + \beta_1^2 + \beta_2^2$ 로 줄여 쓰면 $D = \mathrm{RSS}/f$ 이고

    $$
    \frac{\partial D}{\partial \beta_j} = \frac{1}{f}\frac{\partial \mathrm{RSS}}{\partial \beta_j} - \frac{\mathrm{RSS}}{f^2}\frac{\partial f}{\partial \beta_j}
    $$

    이다. OLS 해에서는 정규방정식이 $\partial \mathrm{RSS}/\partial\beta_j = -2\sum_i x_{ij} e_i = 0$ 을 모든 $j$ 에 대해 보장하므로(연습문제 4) **첫 항이 세 성분 모두에서 사라진다.** 남는 것은 둘째 항뿐이다.

    **(1) $\partial f/\partial\beta_0 = 0$ 이므로 $0$ 이다.** 절편은 $f$ 에 들어 있지 않으니 둘째 항도 사라지고, 첫 항은 $\partial\mathrm{RSS}/\partial\beta_0 = -2\sum_i e_i = 0$ 으로 사라진다. 뜻은 분명하다. **두 기준이 다투는 것은 기울기뿐이고 절편은 아니다.** 어느 기준에서든 평면은 자료의 중심 $(\bar r, \bar t, \bar y)$ 를 지나며, 기울기가 정해지면 절편은 그 조건으로 따라 나온다.

    **(2) 둘 다 음수이므로 최단거리 기준의 기울기가 더 크다.** $\partial f/\partial\beta_j = 2\beta_j$ 이므로

    $$
    \left.\frac{\partial D}{\partial \beta_j}\right|_{\hat{\boldsymbol\beta}} = -\frac{2\,\mathrm{RSS}\,\hat\beta_j}{f^2}, \qquad f = 1.0221416, \quad f^2 = 1.0447735
    $$

    이고 수치를 넣으면

    $$
    \frac{\partial D}{\partial \beta_1} = -\frac{2 \times 286.2067 \times 0.142954}{1.0447735} = -78.3218, \qquad
    \frac{\partial D}{\partial \beta_2} = -\frac{2 \times 286.2067 \times 0.041303}{1.0447735} = -22.6290
    $$

    이다. 둘 다 음수이므로 $\beta_1$ 과 $\beta_2$ 를 **키우는** 쪽으로 $D$ 가 줄어든다. 곧 OLS 해는 $D$ 의 정류점이 **아니고**, 최단거리를 최소화하는 평면(전직교회귀)의 기울기는 OLS 보다 절댓값이 크다. 실제로 풀면 $(0.144230,\ 0.041305)$ 로 둘 다 커지고 $D$ 는 $280.0068$ 에서 $279.9569$ 로 줄며, 그 평면의 $\mathrm{RSS}$ 는 $286.2582$ 로 OLS 의 $286.2067$ 보다 크다. **두 기준이 서로 다른 평면을 고른다는 것이 수로 확인된다.**

    **(3) 평행하다.** 위 식이 $\nabla_{(\beta_1,\beta_2)} D = -\dfrac{2\,\mathrm{RSS}}{f^2}(\hat\beta_1, \hat\beta_2)$ 이므로 기울기벡터는 $(\hat\beta_1,\hat\beta_2)$ 의 음의 상수배다. 비를 보아도

    $$
    \frac{78.3218}{22.6290} = 3.4611 = \frac{0.142954}{0.041303}
    $$

    로 같다. 곧 $D$ 를 줄이는 **첫 걸음**은 기울기벡터를 방사 방향으로 늘리는 것이다.

    그런데 걸음의 **크기**는 방향마다 다르다. $\mathrm{RSS}$ 는 $\beta_j$ 를 $\Delta_j$ 만큼 흔들면 약 $S_{jj}\Delta_j^2$ 만큼 늘어나므로, 1차 근사로 $2 S_{jj}\Delta_j \approx 2\,\mathrm{RSS}\,\hat\beta_j / f$ 를 놓으면

    $$
    \Delta_j \approx \frac{\mathrm{RSS}\,\hat\beta_j}{S_{jj}\,f}, \qquad
    \Delta_1 \approx \frac{286.2067 \times 0.142954}{31633.01 \times 1.0221418} = 0.001265
    $$

    로 실제 이동량 $0.144230 - 0.142954 = 0.001276$ 과 맞는다. **TV 쪽은 $S_{\text{TT}}$ 가 $37.3$ 배 크므로 같은 이득을 얻는 데 드는 값이 $37.3$ 배 비싸고**, 그래서 $\hat\beta_2$ 는 거의 움직이지 않는다. 기울기가 방사 방향을 가리켜도 실제 경로는 **싼 방향으로 휜다.**

    마지막으로 이 기준이 왜 쓰이지 않는지가 드러난다. $D$ 는 $\beta_1, \beta_2$ 를 $y$ 의 단위와 섞어 재므로 단위에 매달린다. 연습문제 6 처럼 TV 를 백만 달러 단위로 바꾸어 전직교회귀를 풀면 TV 계수가 원래 단위로 $0.047184$ 가 나와 $0.041305$ 에서 $14\%$ 어긋난다. **OLS 는 어떤 단위에서도 같은 평면을 주지만 최단거리 기준은 단위마다 다른 평면을 준다.** $\square$

---

## 정리하며

설명변수가 둘이면 적합된 것은 **평면**이다.

- **직선 → 평면 → 초평면.** $p=1$ 이면 직선, $p=2$ 면 평면, $p\ge3$ 이면 그릴 수 없는 초평면이다. **$p=2$ 가 눈으로 볼 수 있는 마지막 경우**이므로 직관을 세우기에 좋다.
- **잔차는 수직 거리다.** 점에서 평면까지 **$y$ 축 방향으로** 잰 거리이며, 평면에 대한 최단거리가 아니다. 최소제곱이 최소화하는 것이 바로 이 거리의 제곱합이다.
- **계수가 기울기 둘이다.** $\beta_1$ 은 Radio 를 고정한 채 TV 방향의 기울기, $\beta_2$ 는 그 반대다.
- **교호작용이 없으면 평면이다.** $x_1x_2$ 항을 넣으면 휘어진 곡면이 되며, 그때 "다른 변수를 고정한 채"라는 해석이 수준마다 달라진다.
- **그림이 다중공선성도 드러낸다.** 두 설명변수가 강하게 상관되면 점들이 평면 위의 한 직선 근처에 몰려, 평면의 기울기가 불안정해진다.

다음 절 **회귀 (대마 인구통계)** 에서 실제 자료에 적용한다.
