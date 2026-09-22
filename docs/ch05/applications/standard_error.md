# 표준오차

## 개요

> **참고 자료:** [YouTube — Standard Error](https://www.youtube.com/watch?v=A82brFpdr9g) | [Blog — SD vs SE](https://statisticsbyjim.com/basics/difference-standard-deviation-vs-standard-error/)

어떤 논문의 표에 "평균 50, ±8"이라고 적혀 있다. 이 8은 무엇인가. 응답자 한 사람 한 사람이 평균에서 대략 8만큼 떨어져 있다는 뜻일 수도 있고, 이 연구를 처음부터 다시 했을 때 평균값 자체가 8쯤 흔들린다는 뜻일 수도 있다. 두 진술은 전혀 다른 말이다. 앞의 것이 **표준편차**이고, 뒤의 것이 **표준오차**다.

앞 절에서 통계량이 확률변수임을 보았다. 확률변수에는 분포가 있고, 분포에는 표준편차가 있다. 통계량의 분포, 곧 표본분포의 표준편차에 붙인 이름이 표준오차다. 그러니 표준오차는 새 개념이 아니라 **어느 분포의 표준편차인지**를 밝히는 이름이다. 이 절의 일은 그 "어느 분포"를 표준편차의 것과 헷갈리지 않게 하는 것이다.

## 무엇이 흩어지는가

두 양의 공식을 나란히 적어 보면 차이가 어디에 있는지 보인다. 표준편차는 관측값 하나를 나타내는 확률변수 $X$의 분산에서 나온다.

$$
\text{SD} = \sqrt{\text{Var}(X)}
$$

표준오차는 표본 전체를 받아 수 하나를 내놓는 통계량 $\hat\theta$의 분산에서 나온다.

$$
\text{SE} = \sqrt{\text{Var}(\hat{\theta}(X_1, \dots, X_n))}
$$

제곱근 안에 들어 있는 것이 다르다. 앞의 것은 **개별 자료점**이 평균 주위로 퍼진 정도를 재고, 뒤의 것은 표본을 다시 뽑을 때마다 **계산 결과**가 참 모수 주위로 얼마나 튀는지를 잰다.

이 차이에서 성질의 차이가 따라 나온다. 표준편차는 모집단의 변동성만으로 정해지는 모집단의 성질이라 표본크기와 아무 상관이 없다. 키가 7cm만큼 흩어져 있는 나라에서는 1만 명을 재든 100만 명을 재든 여전히 7cm만큼 흩어져 있다. 반면 표준오차는 모집단의 변동성**과** 표본크기에 함께 의존한다. 표본평균의 경우 표준편차가 $\sigma$인 데 비해 표준오차는 $\sigma/\sqrt n$이므로, $n$이 커지면 앞의 것은 그대로인데 뒤의 것만 줄어든다.

그러므로 "자료를 더 모으면 흩어짐이 줄어든다"는 말은 절반만 맞다. 줄어드는 것은 **추정값의 흔들림**이지 자료 자체의 흩어짐이 아니다. 이 혼동은 그림에 오차막대를 잘못 붙이는 흔한 실수로 이어지며, 연습문제 8에서 그 대가를 수치로 확인한다.

## 표준화의 공통 형태

> **참고 자료:** [Khan Academy — Standard Error of the Mean](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/sampling-distribution-mean/v/standard-error-of-the-mean)

표준오차가 이토록 중요한 것은 그것이 언제나 **분모 자리에** 놓이기 때문이다. 추정값에서 모수를 빼고 표준오차로 나누면 "참값에서 몇 표준오차만큼 떨어져 있는가"라는 눈금 없는 수가 되고, 그 수는 이미 표가 마련되어 있는 분포를 따른다. 추론통계학을 관통하는 공통 형태가 이것이다.

$$
\begin{array}{lllllll}
\displaystyle
\frac{\text{unbiased\_estimator} - \text{parameter}}{\text{standard\_error}}
&=&
\displaystyle
\frac{\bar{X} - \mu}{\frac{\sigma}{\sqrt{n}}}
&\approx&
\displaystyle
\frac{\bar{X} - \mu}{\frac{s}{\sqrt{n}}}
&\approx&
z \;\text{ or }\; t_{n-1} \\[16pt]
\displaystyle
\frac{\text{unbiased\_estimator} - \text{parameter}}{\text{standard\_error}}
&=&
\displaystyle
\frac{\hat{p} - p}{\sqrt{\frac{p(1-p)}{n}}}
&\approx&
\displaystyle
\frac{\hat{p} - p}{\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}}
&\approx&
z
\end{array}
$$

두 줄이 같은 이야기를 한다. 왼쪽 끝의 뼈대는 언제나 "불편추정량 빼기 모수, 나누기 표준오차"다. 그런데 가운데 항은 $\sigma$나 $p$를 알아야 계산되므로 통계량이 아니다. 그래서 오른쪽처럼 $\sigma$를 $s$로, $p$를 $\hat p$로 바꿔 쓰며, 그 대체의 대가가 정규분포 대신 $t_{n-1}$이거나 "표본이 충분히 커야 한다"는 단서다. 이 책에서 만날 검정통계량은 거의 모두 이 틀 안에 들어온다.

## 물 부족

> **참고 자료:** [Khan Academy — Sampling Distribution Example Problem](https://www.khanacademy.org/math/ap-statistics/sampling-distribution-ap/sampling-distribution-mean/v/sampling-distribution-example-problem)

한 사람이 물을 얼마나 마시는지는 사람마다 0.7리터쯤 다르다. 그러나 50명이 마시는 **평균**은 그만큼 흔들리지 않는다. 물을 얼마나 챙겨 가야 하는지를 정하는 것은 개인차가 아니라 평균의 흔들림, 곧 표준오차다.

<div class="probox" markdown>

**문제.** <span class="diff easy" title="쉬움"></span> 남성이 야외 활동을 할 때 평균 2리터의 물을 마시고 표준편차는 0.7리터이다. 남성 50명이 하루 종일 자연 탐방을 가는데 물 110리터를 가져간다. 여행 중 물이 떨어질 확률을 구하라.

</div>

??? success "풀이"
    $X_i$를 $i$번째 사람의 물 소비량이라 하자. 독립을 가정하면 중심극한정리에 의해 표본평균 $\bar{X}$는 근사적으로 평균 2, 표준편차 $0.7/\sqrt{50} \approx 0.0990$인 정규분포를 따른다.

    $$
    \begin{array}{lll}
    \displaystyle
    P\!\left(\bar{X} > \frac{110}{50}\right)
    &=&
    \displaystyle
    P\!\left(\frac{\bar{X} - 2}{0.0990} > \frac{2.2 - 2}{0.0990}\right) \\[12pt]
    &\approx&
    \displaystyle
    P(Z > 2.020) \\[8pt]
    &\approx&
    0.0217
    \end{array}
    $$

## 표집분포를 직접 만들어 보면

표준오차가 표집분포의 표준편차라면, 표집분포를 손에 쥐고 있을 때 표준오차는 그저 그 값들의 표준편차일 뿐이다. 아래 코드가 하는 일이 정확히 그것이다. $\text{Uniform}(0,1)$에서 크기 5짜리 표본을 1만 번 뽑아 표본평균 1만 개를 모으고, 그 1만 개의 평균과 표준편차를 잰다. 평균은 참값 0.5 근처에 앉고, 표준편차가 곧 표준오차다.

물론 현실에서 표본은 한 번밖에 뽑지 못하므로 이렇게 할 수 없다. 그래서 $s/\sqrt n$이라는 공식을 쓰는 것이고, 이 모의실험은 그 공식이 무엇을 대신하고 있는지를 눈으로 보여 주는 장치다.

<div class="codebox" markdown>

#### 예제 1. 표준오차를 한 파일로 구하기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

def main():
    # 크기 5짜리 균등표본을 1만 번 뽑아 표본평균을 모은다.
    X_bar = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        x_bar = x.mean()
        X_bar.append(x_bar)

    # 표준오차의 정의를 그대로 실행한 것이 아래 두 줄이다.
    #   average        = 표집분포의 중심 (참 mu = 0.5 의 좋은 추정)
    #   standard_error = 표집분포의 **표준편차**
    # 즉 표준오차는 새로운 개념이 아니라, 통계량의 분포에 대한 표준편차다.
    # 현실에서는 표본이 하나뿐이라 이렇게 구할 수 없어 공식 s/sqrt(n) 을 쓴다.
    average = np.array(X_bar).mean()
    standard_error = np.array(X_bar).std()

    print(f'(Estimated) Mean of X_bar : {average:.4}')
    print(f'Standard Error   of X_bar : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of X_bar", fontsize=20)

    ax.hist(X_bar, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=5, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')

    # 평균에서 +1 표준오차까지를 양방향 화살표로 표시해
    # "표준오차 = 이만큼의 폭"임을 그림에서 직접 보여 준다.
    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 5),
                xytext=(average + standard_error, 5),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 5.5),
                xytext=(average, 5.5),
                fontsize=15)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```

출력:

```
(Estimated) Mean of X_bar : 0.4981
Standard Error   of X_bar : 0.1287
```

![Sampling Distribution of X_bar](./img/standard_error_106.png)

</div>

## 같은 코드를 파일 둘로 나눈다면

위 예제는 한 덩어리로 실행하는 형태였다. 실제 프로젝트에서는 설정과 본문을 파일로 나누는 편이 낫다. 아래 두 블록은 **두 개의 `.py` 파일**을 각각 적은 것이므로, 문서에서 이어 붙여 실행할 수는 없다. 같은 디렉터리에 저장한 뒤 `python standard_error_of_x_bar.py --seed 7` 처럼 실행한다.

### 모듈 버전: `global_name_space.py`

<div class="codebox" markdown>

#### 예제 2. 공유 설정 모듈 { .eg }

```python
import argparse
import numpy as np

# 이 파일은 여러 스크립트가 공유하는 설정을 한곳에 모아 두는 용도다.
# 다른 모듈에서 `from global_name_space import ARGS` 로 가져다 쓴다.
parser = argparse.ArgumentParser(description='Standard error simulation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
ARGS = parser.parse_args()

# 시드를 여기서 한 번만 고정하면 이 설정을 가져다 쓰는 모든 스크립트가
# 같은 난수열을 쓰게 되어 결과가 재현된다.
np.random.seed(ARGS.seed)
```

</div>

### 모듈 버전: `standard_error_of_x_bar.py`

<div class="codebox" markdown>

#### 예제 3. 표준오차를 그림에 표시하기 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

from global_name_space import ARGS

def main():
    X_bar = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        x_bar = x.mean()
        X_bar.append(x_bar)

    average = np.array(X_bar).mean()
    standard_error = np.array(X_bar).std()

    print(f'(Estimated) Mean of X_bar : {average:.4}')
    print(f'Standard Error   of X_bar : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of X_bar", fontsize=20)

    ax.hist(X_bar, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=5, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=5, alpha=0.7, color='k', ls='--')

    # 평균에서 +1 표준오차까지를 양방향 화살표로 표시해
    # "표준오차 = 이만큼의 폭"임을 그림에서 직접 보여 준다.
    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 5),
                xytext=(average + standard_error, 5),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 5.5),
                xytext=(average, 5.5),
                fontsize=15)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.1, 6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```

출력:

```
(Estimated) Mean of X_bar : 0.4993
Standard Error   of X_bar : 0.1281
```

![Sampling Distribution of X_bar](./img/standard_error_215.png)

예제 1과 계산이 같은데 값이 조금 다르다. 시드가 다르기 때문이다(예제 1은 0, 여기서는 `global_name_space.py`의 기본값 1). 두 결과가 모두 참값 $\mu = 0.5$와 $\sigma/\sqrt5 = 0.1291$ 근처에 있으며, 그 흔들림 자체가 모의실험 오차다.

</div>

## 분산에도 표준오차가 있다

표준오차는 평균만의 것이 아니다. 표본마다 값이 달라지는 양이면 무엇이든 표본분포를 갖고, 따라서 표준오차를 갖는다. 같은 실험을 $S^2$에 대해 되풀이해 보자. 파일 이름은 `standard_error_of_s_square.py`이고 앞의 `global_name_space.py`를 그대로 가져다 쓴다. 바뀌는 것은 표본평균 대신 표본분산을 기록한다는 것 하나뿐인데, 나오는 그림의 모양은 사뭇 다르다.

<div class="codebox" markdown>

#### 예제 4. 표본크기에 따른 표준오차 변화 { .eg }

```python
import matplotlib.pyplot as plt
import numpy as np

from global_name_space import ARGS

def main():
    S_square = []
    for _ in range(10_000):
        x = np.random.uniform(size=(5,))
        # ddof=1 을 반드시 적어야 한다. numpy의 기본값은 ddof=0 이라
        # x.var() 나 x.std()**2 는 n 으로 나눈 **편향추정량**을 준다.
        S_square.append(x.var(ddof=1))

    average = np.array(S_square).mean()
    standard_error = np.array(S_square).std()

    print(f'(Estimated) Mean of S^2 : {average:.4}')
    print(f'Standard Error   of S^2 : {standard_error:.4}')

    fig, ax = plt.subplots(figsize=(12, 3))

    ax.set_title("Sampling Distribution of S^2", fontsize=20)

    ax.hist(S_square, bins=100, density=True, alpha=0.3)
    ax.vlines(average, ymin=0, ymax=8, alpha=1.0, color='k', ls='-', lw=5)
    ax.vlines(average + standard_error, ymin=0, ymax=8, alpha=0.7, color='k', ls='--')
    ax.vlines(average - standard_error, ymin=0, ymax=8, alpha=0.7, color='k', ls='--')

    # 평균에서 +1 표준오차까지를 양방향 화살표로 표시해
    # "표준오차 = 이만큼의 폭"임을 그림에서 직접 보여 준다.
    arrowprops = dict(arrowstyle='<->', color='k', linewidth=3, mutation_scale=20)
    ax.annotate(text='',
                xy=(average, 8),
                xytext=(average + standard_error, 8),
                arrowprops=arrowprops)
    ax.annotate(text='Standard Error',
                xy=(average, 8.8),
                xytext=(average, 8.8),
                fontsize=15)

    ax.set_xlim(0.0, 0.25)
    ax.set_ylim(-0.1, 10)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.show()

if __name__ == "__main__":
    main()
```

출력:

```
(Estimated) Mean of S^2 : 0.08406
Standard Error   of S^2 : 0.04263
```

![Sampling Distribution of S^2](./img/standard_error_277.png)

두 수치를 이론과 견주어 보자. $\text{Uniform}(0,1)$의 분산이 $\sigma^2 = 1/12 = 0.0833$이므로 평균 0.0841은 **불편성**을 확인해 준다. 표준오차 0.0426은 5.6절에서 볼 공식

$$
\text{SE}(S^2) = \sqrt{\frac1n\left(\kappa - \frac{n-3}{n-1}\right)}\,\sigma^2
$$

에 균등분포의 첨도 $\kappa = 1.8$과 $n = 5$를 넣은 값 0.0425와 맞는다.

!!! warning "numpy의 `ddof` 기본값은 0이다"
    `x.var()`와 `x.std()**2`는 $n$으로 나눈 **편향추정량**을 준다. $S^2$을 원하면 `x.var(ddof=1)`이라고 명시해야 한다. 위 코드에서 `ddof=1`을 빼면 평균이 0.0841이 아니라 $\frac{n-1}{n}\sigma^2 = 0.0667$로 나와 불편성이 깨진 것처럼 보인다.

    $n = 5$처럼 작은 표본에서는 이 차이가 20%나 된다. pandas의 `.var()`는 반대로 기본값이 `ddof=1`이므로, 두 라이브러리를 섞어 쓸 때 특히 조심해야 한다.

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$\sigma = 50$이다. (a) $n = 25, 100$일 때 $\mathrm{SE}$를 계산하라. (b) $n = 16$에서 $\mathrm{SE} = 5$일 때 $\sigma$를 구하고 $n = 64$에서의 $\mathrm{SE}$를 계산하라.

</div>

??? success "풀이"
    (a) $\mathrm{SE}_{25} = 50/\sqrt{25} = 10$. $\mathrm{SE}_{100} = 50/\sqrt{100} = 5$. $n$이 네 배가 되면 표준오차가 절반이 된다.

    (b) $\sigma/\sqrt{16} = 5$에서 $\sigma = 20$이다. $n = 64$에서 $\mathrm{SE} = 20/\sqrt{64} = 2.5$이다. $n$을 네 배로 하면 표준오차가 절반이 된다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
**표준오차와 표준편차.** 어떤 연구자가 $n = 100$에 대해 "표본평균 $= 50$, 표준편차 $= 8$"이라고 보고했다. (a) 표본평균의 표준오차는? (b) 비전문가에게 둘의 차이를 설명하라.

</div>

??? success "풀이"
    (a) ($\sigma$ 대신 $s$를 사용한) 추정 표준오차: $\mathrm{SE} = 8/\sqrt{100} = 0.8$.

    (b) **SD = 8:** 자료 집합에서 개별 관측값이 얼마나 달라지는지를 기술한다. 전형적인 개체는 평균에서 약 8단위 떨어져 있다.

    **SE = 0.8:** 표본이 달라질 때 표본평균이 얼마나 달라지는지를 기술한다. 참 모평균은 50에서 약 1.6단위(≈ 표준오차 2개) 이내에 있을 가능성이 높다.

    자료를 더 모아도 표준편차는 달라지지 않지만 표준오차는 $1/\sqrt n$의 비율로 줄어든다. 표준오차를 말해야 할 자리에 표준편차를 보고하거나 그 반대로 하는 것은 과학 논문에서 흔한 오류이다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
**붓스트랩 표준오차.** 모집단이 정규가 아니고 $\sigma$를 모를 때 **붓스트랩**이 표준오차 추정값을 준다. 붓스트랩으로 SE($\bar X$)를 계산하는 절차를 서술하라.

</div>

??? success "풀이"
    i.i.d. 표본 $X_1, \ldots, X_n$이 주어졌을 때:

    1. 원래 표본에서 복원추출하여 붓스트랩 표본 $X_1^*, \ldots, X_n^*$을 뽑는다.
    2. 붓스트랩 표본에서 $\bar X^*$를 계산한다.
    3. 1–2단계를 $B$번 반복하여(보통 $B = 1000$에서 10000) $\bar X^*_1, \ldots, \bar X^*_B$을 얻는다.
    4. 붓스트랩 복제값들의 표본표준편차로 표준오차를 추정한다: $\hat{\mathrm{SE}}_{\text{boot}} = \sqrt{(1/(B-1))\sum(\bar X^*_b - \bar X^*_\cdot)^2}$.

    **왜 작동하는가:** 붓스트랩 분포가 반복추출에서의 표본분포를 근사한다. 점근적으로 $\hat{\mathrm{SE}}_{\text{boot}} \to \sigma/\sqrt n$이지만, 붓스트랩은 정규근사보다 분포의 모양(치우침, 두꺼운 꼬리)을 더 잘 포착한다.

    닫힌 형태의 표준오차가 없는 경우(중앙값, 비, 복잡한 모형의 회귀계수)에 특히 유용하다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
**함수의 표준오차.** $\hat\theta$의 $\mathrm{SE}(\hat\theta)$를 알고 $g$가 미분가능할 때 **델타 방법**으로 $\mathrm{SE}(g(\hat\theta))$를 계산하라.

</div>

??? success "풀이"
    **델타 방법:** $\sqrt n(\hat\theta - \theta) \xrightarrow{d} N(0, \sigma^2)$이면, $g'(\theta) \ne 0$인 미분가능한 $g$에 대해:

    $$
    \sqrt n(g(\hat\theta) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)
    $$

    표준오차 형태로는 $\mathrm{SE}(g(\hat\theta)) \approx |g'(\hat\theta)| \cdot \mathrm{SE}(\hat\theta)$이다.

    **예:** $g(\hat p) = \log(\hat p/(1 - \hat p))$(로짓)에 대해 $g'(\hat p) = 1/(\hat p(1 - \hat p))$이므로 $\mathrm{SE}(\hat\eta) = \mathrm{SE}(\hat p)/(\hat p(1 - \hat p))$이다.

    추정량이 표준오차를 아는 더 단순한 추정량의 변환일 때, 델타 방법은 표준오차 계산의 주된 도구가 된다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
**두 표본의 합동 표준오차.** 평균이 $\mu_1, \mu_2$이고 분산은 미지인 두 모집단에서 독립인 표본을 뽑는다. (a) 등분산 가정(합동), (b) 이분산(Welch) 아래에서 $\bar X_1 - \bar X_2$의 표준오차를 유도하라.

</div>

??? success "풀이"
    독립성에 의해 $\mathrm{Var}(\bar X_1 - \bar X_2) = \sigma_1^2/n_1 + \sigma_2^2/n_2$이다.

    **(a) 합동 ($\sigma_1 = \sigma_2 = \sigma$ 가정):**

    합동 분산: $s_p^2 = ((n_1 - 1)s_1^2 + (n_2 - 1)s_2^2)/(n_1 + n_2 - 2)$.

    $\mathrm{SE}_{\text{pool}} = s_p \sqrt{1/n_1 + 1/n_2}$.

    분산이 같다고 볼 만할 때 표준적인 두 표본 $t$ 검정에서 사용한다. 이 가정이 성립하면 약간 더 효율적이다.

    **(b) Welch (이분산):**

    $\mathrm{SE}_{\text{Welch}} = \sqrt{s_1^2/n_1 + s_2^2/n_2}$.

    합동하지 않고 각 표본이 자신의 분산을 기여한다. $t$ 임계값에는 (정수가 아닌) Welch 자유도를 사용한다.

    현대적 권고: 기본적으로 Welch를 선호하라. 강한 등분산 가정을 요구하지 않으면서 분산이 같을 때에도 거의 같은 성능을 낸다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**비복원추출에서의 표준오차.** 크기 $N$인 모집단에서 크기 $n$인 표본을 비복원으로 뽑는다. SE($\bar X$)를 계산하고 **유한모집단 수정**을 찾아라.

</div>

??? success "풀이"
    유한모집단에서 비복원추출을 하면:

    $$
    \mathrm{Var}(\bar X) = \frac{\sigma^2}{n}\!\left(1 - \frac{n}{N}\right)
    $$

    따라서 $\mathrm{SE}(\bar X) = (\sigma/\sqrt n) \sqrt{1 - n/N}$이다. 인수 $\sqrt{1 - n/N}$이 **유한모집단 수정(FPC)**이다.

    **극한:**

    - $n/N \to 0$ (추출 비율이 매우 작을 때): FPC $\to 1$로 표준적인 $\sigma/\sqrt n$이 복원된다. 전국 조사($n = 1000, N \approx 10^8$)에 해당한다.
    - $n/N \to 1$ (전수조사): FPC $\to 0$으로 표본추출 변동성이 없다. 전수조사는 결정론적 추정값을 준다.

    **FPC가 중요한 경우:** 감사(송장 500건 중 100건 추출, $n/N = 0.2$, FPC $\approx 0.89$). FPC가 신뢰구간을 약 11% 좁히므로 무시할 수 없다.

    입문 통계학의 공식 대부분이 FPC를 무시하는 이유는 일반적인 과학 표본에서 $n/N$이 작기 때문이다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
단순선형회귀 $y_i = \beta_0+\beta_1x_i+\varepsilon_i$에서 기울기 추정량의 표준오차가

$$
\operatorname{SE}(\hat\beta_1) = \frac{\sigma}{\sqrt{\sum_i(x_i-\bar x)^2}} = \frac{\sigma}{s_x\sqrt{n-1}}
$$

임을 확인하고, 이 식이 실험 설계에 주는 조언을 정리하라.

</div>

??? success "풀이"
    $\hat\beta_1 = \sum_i w_iy_i$($w_i = (x_i-\bar x)/S_{xx}$, $S_{xx}=\sum(x_i-\bar x)^2$)로 쓸 수 있으므로 $y_i$가 독립이고 분산이 $\sigma^2$이면

    $$
    \operatorname{Var}(\hat\beta_1) = \sigma^2\sum_i w_i^2 = \sigma^2\cdot\frac{S_{xx}}{S_{xx}^2} = \frac{\sigma^2}{S_{xx}}
    $$

    이다. $S_{xx} = (n-1)s_x^2$이므로 위 식이 나온다. $\square$

    **설계에 주는 조언.** 표준오차를 줄이는 길이 세 가지다.

    1. **$n$을 늘린다.** $\sqrt{n-1}$에 반비례하므로 익숙한 $1/\sqrt n$ 규칙이다.
    2. **오차분산 $\sigma$를 줄인다.** 측정 정밀도를 높이거나, 다른 설명변수를 넣어 잔차를 줄이거나, 실험 조건을 통제한다.
    3. **$x$를 넓게 퍼뜨린다.** $s_x$에 반비례한다. 이것이 관측연구에는 없고 **실험 설계에만 있는 지렛대**다.

    세 번째가 특히 중요하다. 관측이 $n$개로 고정되어 있다면, $x$ 값을 가능한 범위의 **양 끝에 몰아 놓는 것**이 $S_{xx}$를 최대로 하고 따라서 기울기를 가장 정밀하게 추정한다. 약물 용량 반응 실험에서 중간 용량을 생략하고 최저·최고 용량에 표본을 몰아 배치하는 설계가 그 예다.

    다만 이 설계에는 대가가 있다. 양 끝만 보면 **관계가 직선인지 확인할 수 없다.** 굽은 관계를 직선으로 잘못 읽어도 알아챌 방법이 없다. 그래서 실무에서는 중간에도 일부 배치해 모형 적합성을 점검할 여지를 남긴다. **최적 설계와 모형 진단 사이의 맞바꿈**이다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
논문의 그림에 오차막대가 그려져 있는데 그것이 SD인지 SE인지 95% 신뢰구간인지 밝혀져 있지 않다. $n=25$일 때 세 막대의 길이 비를 구하고, 왜 반드시 밝혀야 하는지 설명하라.

</div>

??? success "풀이"
    $n=25$에서 세 막대의 반길이는 다음과 같다.

    | 막대 | 반길이 | $s$ 대비 |
    |---|---|---|
    | SD | $s$ | 1.00 |
    | SE | $s/\sqrt{25} = s/5$ | 0.20 |
    | 95% CI | $t_{0.975,24}\,s/5 = 2.064s/5$ | 0.41 |

    **SD 막대가 SE 막대의 5배**이고, 신뢰구간은 SE의 2.06배다. 같은 자료인데 그림의 인상이 완전히 달라진다.

    **왜 반드시 밝혀야 하는가.** 세 막대가 **서로 다른 질문에 답하기** 때문이다.

    - **SD 막대**: "개별 관측값이 얼마나 흩어져 있는가." $n$이 커져도 줄지 않는다. 자료의 변동성 자체를 보여 주고 싶을 때 쓴다.
    - **SE 막대**: "평균을 얼마나 정확히 알고 있는가." $n$과 함께 줄어든다.
    - **신뢰구간**: "참 평균이 어디에 있을 법한가." 해석이 가장 직접적이다.

    독자가 SE 막대를 SD로 오해하면 자료가 실제보다 훨씬 균일하다고 믿게 되고, 반대로 오해하면 추정이 실제보다 부정확하다고 믿게 된다.

    **더 흔한 함정.** "두 집단의 SE 막대가 겹치니 차이가 유의하지 않다"는 판단은 **틀렸다.** 두 집단의 표본크기와 분산이 같을 때 차이의 표준오차는 $\sqrt2\operatorname{SE}$이므로, 두 SE 막대가 살짝 겹쳐도 $t$가 2를 넘을 수 있다. 대략 두 SE 막대의 끝이 서로 상대의 평균에 닿지 않을 정도로 겹치면 아직 유의할 수 있다.

    반대로 **95% 신뢰구간 막대가 겹치지 않으면** 차이는 거의 확실히 유의하다(보수적인 판단이다). 겹치더라도 유의할 수 있으므로, 결국 **그림으로 유의성을 판정하지 말고 차이에 대한 구간을 따로 보고하는 것**이 옳다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
어떤 조사가 학교 50곳에서 각각 학생 20명씩, 모두 1000명을 조사했다. 학교 내 급내상관계수가 $\text{ICC}=0.05$일 때 설계효과와 유효 표본크기를 구하라. 독립을 가정하면 표준오차를 얼마나 과소평가하는가?

</div>

??? success "풀이"
    군집 크기가 $m=20$이므로 설계효과는

    $$
    \text{DEFF} = 1 + (m-1)\,\text{ICC} = 1 + 19\times0.05 = 1.95
    $$

    이고 유효 표본크기는

    $$
    n_{\text{eff}} = \frac{1000}{1.95} = 513
    $$

    이다. **1000명을 조사했지만 독립 표본 513명만큼의 정보밖에 없다.**

    **과소평가의 크기.** 분산이 1.95배이므로 표준오차는 $\sqrt{1.95} = 1.40$배다. 독립을 가정하면 표준오차를 **29% 과소평가**한다($1 - 1/1.40$). 그 결과

    - 신뢰구간이 실제보다 29% 좁아지고,
    - 검정통계량이 1.40배 부풀려지며,
    - 명목 5% 검정의 실제 오류율이 15% 안팎으로 뛴다.

    **ICC가 작아 보여도 방심할 수 없다.** 0.05는 낮은 값인데도 설계효과가 2에 가깝다. 괄호 안이 $(m-1)\times\text{ICC}$이므로 **군집이 클수록 영향이 커지기** 때문이다. 같은 ICC라도 학급당 5명이면 DEFF가 1.2에 그치고, 100명이면 5.95가 된다.

    **설계에 주는 함의.** 같은 예산이라면 **군집을 크게 하기보다 군집 수를 늘리는 편이 낫다.** 학교 50곳에서 20명씩 뽑는 것보다 학교 100곳에서 10명씩 뽑는 쪽이 유효 표본크기가 크다($\text{DEFF} = 1.45$, $n_{\text{eff}} = 690$). 다만 학교를 추가하는 비용이 학생을 추가하는 비용보다 훨씬 크므로, 실제로는 두 비용의 비와 ICC를 함께 넣어 최적 군집 크기를 계산한다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
시계열 $X_1,\dots,X_n$이 정상이고 자기상관함수가 $\rho_k$일 때 $\operatorname{Var}(\bar X)$를 구하라. AR(1) 과정 $\rho_k = \phi^k$에 대해 $n$이 클 때의 근사식을 얻고, $\phi = 0.8$에서 유효 표본크기를 계산하라.

</div>

??? success "풀이"
    **일반식.**

    $$
    \operatorname{Var}(\bar X) = \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n\operatorname{Cov}(X_i,X_j) = \frac{\sigma^2}{n^2}\sum_{i,j}\rho_{|i-j|}
    $$

    이고, 지연 $k$인 쌍이 $2(n-k)$개($k\ge1$)이므로

    $$
    \operatorname{Var}(\bar X) = \frac{\sigma^2}{n}\left\{1 + 2\sum_{k=1}^{n-1}\left(1-\frac kn\right)\rho_k\right\}
    $$

    이다.

    **$n$이 클 때.** $\sum_k|\rho_k| < \infty$이면 $1-k/n \to 1$이므로

    $$
    \operatorname{Var}(\bar X) \approx \frac{\sigma^2}{n}\left(1+2\sum_{k=1}^\infty\rho_k\right)
    $$

    이다. 괄호 안이 설계효과에 해당하며 **적분시간척도**라 부른다.

    **AR(1).** $\rho_k = \phi^k$이므로 등비급수에서

    $$
    1 + 2\sum_{k=1}^\infty \phi^k = 1 + \frac{2\phi}{1-\phi} = \frac{1+\phi}{1-\phi}
    $$

    이다. 따라서

    $$
    \operatorname{Var}(\bar X) \approx \frac{\sigma^2}{n}\cdot\frac{1+\phi}{1-\phi}, \qquad n_{\text{eff}} = n\cdot\frac{1-\phi}{1+\phi}
    $$

    **$\phi=0.8$이면**

    $$
    \frac{1+0.8}{1-0.8} = \frac{1.8}{0.2} = 9
    $$

    로 분산이 **아홉 배**다. 표준오차는 3배이고, 유효 표본크기가 $n/9$로 줄어든다. 관측 900개가 독립 관측 100개만큼의 정보밖에 없다.

    | $\phi$ | 0.2 | 0.5 | 0.8 | 0.9 | 0.95 |
    |---|---|---|---|---|---|
    | DEFF | 1.5 | 3 | 9 | 19 | 39 |

    3장에서 본 $\phi = 0.9$가 가운데 있다. 관측 만 개가 독립 관측 526개 값어치로 줄어드는 그 경우다.

    **실무적 함의.**

    - 시계열을 독립 표본처럼 다루면 표준오차를 심하게 과소평가한다. $\phi=0.95$라면 세 배가 아니라 여섯 배 과소평가한다.
    - $\phi < 0$(음의 자기상관)이면 반대로 $\operatorname{Var}(\bar X)$가 **줄어든다.** 계통표집이 단순무작위표집보다 나을 수 있는 이유가 이것이다.
    - MCMC 표본이 정확히 이 구조를 갖는다. 그래서 유효표본크기(ESS)를 보고하고, 사슬을 솎아 내거나(thinning) 더 길게 돌리는 것이다.
    - 실무에서는 $\rho_k$를 모르므로 뉴이-웨스트 추정량처럼 표본 자기공분산을 가중합한 **HAC 표준오차**를 쓴다.

<div class="drillbox" markdown>

**연습문제 11.** <span class="diff hard" title="어려움"></span>
분위수마다 추정의 어려움이 다르다. 표본 분위수의 표준오차를 유도하고, 왜 **극단 분위수가 훨씬 부정확한지** 수치로 확인하라.

</div>

??? success "풀이"
    표본 $p$ 분위수 $\hat{x}_p$는 점근적으로

    $$
    \sqrt{n}\left(\hat{x}_p - x_p\right) \xrightarrow{d} N\!\left(0,\ \frac{p(1-p)}{f(x_p)^2}\right)
    $$

    를 따른다. 즉 표준오차가

    $$
    \operatorname{SE}(\hat{x}_p) \approx \frac{1}{f(x_p)}\sqrt{\frac{p(1-p)}{n}}
    $$

    이다. **분자는 $p = 0.5$에서 최대이지만, 분모의 밀도 $f(x_p)$가 꼬리에서 급격히 작아지므로 극단 분위수의 표준오차가 훨씬 크다.**

    ```python
    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(1)
    B, n = 40_000, 500
    d = rng.normal(0, 1, (B, n))
    se_mean = 1 / np.sqrt(n)

    print(f"N(0,1), n={n}")
    print(f"{'p':>6}{'모의 SE':>11}{'이론 SE':>11}{'평균의 SE 대비':>15}")
    for p in (0.5, 0.75, 0.90, 0.95, 0.99):
        q = np.quantile(d, p, axis=1)
        theory = np.sqrt(p * (1 - p) / n) / stats.norm.pdf(stats.norm.ppf(p))
        print(f"{p:>6.2f}{q.std():>11.5f}{theory:>11.5f}{q.std() / se_mean:>15.3f}")
    ```

    출력:

    ```
    N(0,1), n=500
         p      모의 SE      이론 SE      평균의 SE 대비
      0.50    0.05590    0.05605          1.250
      0.75    0.06114    0.06094          1.367
      0.90    0.07609    0.07645          1.701
      0.95    0.09410    0.09450          2.104
      0.99    0.15955    0.16696          3.568
    ```

    이론과 모의가 소수점 셋째 자리까지 맞는다.

    | $p$ | SE | 표본평균의 SE 대비 |
    |---|---|---|
    | $0.50$ | $0.0557$ | $1.25$배 |
    | $0.90$ | $0.0761$ | $1.70$배 |
    | $0.99$ | $\mathbf{0.1601}$ | $\mathbf{3.58}$배 |

    $99$번째 백분위수를 중앙값만큼 정밀하게 추정하려면 **표본이 약 $8$배 필요하다**($3.58^2 \approx 12.8$, 중앙값 대비로는 $(0.160/0.056)^2 \approx 8.3$).

    **실무적 함의.**

    - **위험관리의 VaR**은 보통 $99\%$나 $99.9\%$ 분위수다. 이 계산이 말하는 바는 **그 추정값이 본질적으로 불안정하다**는 것이다. $99.9\%$ 분위수를 안정적으로 추정하려면 관측이 수만 개 필요하다.
    - **관측 범위를 넘는 분위수는 추정할 수 없다.** $n = 500$이면 경험적으로 $99.8$번째 백분위수가 최댓값이다. 그보다 극단적인 분위수를 말하려면 반드시 **모형 가정(극단값 이론 등)** 을 끌어들여야 하며, 그 순간 결과는 가정에 의존하게 된다.
    - **중앙값이 특별히 정밀한 것도 아니다.** 표본평균보다 $1.25$배 나쁘다(정규분포에서 효율 $2/\pi$의 제곱근인 $1/\sqrt{0.637} = 1.25$).

    **밀도가 작은 곳은 추정이 어렵다**는 것이 일반 원리다. 자료가 드문 영역에 대해 정밀한 진술을 하려면 자료를 훨씬 많이 모으거나, 구조에 대한 가정을 빌려 와야 한다. $\square$

---

## 정리하며

표준오차는 통계량의 표본분포의 표준편차다. 표준편차와 이름이 닮았을 뿐 재는 대상이 전혀 다르다. 표준편차는 개별 관측값이 흩어진 정도를 재며, 모집단의 성질이므로 표본을 아무리 늘려도 변하지 않는다. 표준오차는 표본통계량이 흔들리는 정도를 재며, $1/\sqrt n$의 비율로 줄어든다. 이 쪽에서 가져갈 것 하나를 고르라면 그것이다. **표본을 늘려도 자료가 덜 흩어지지는 않는다.** 줄어드는 것은 추정값의 흔들림뿐이다.

표준오차가 이 책 전체에서 되풀이되는 것은 $(\hat\theta-\theta)/\mathrm{SE}(\hat\theta)$라는 표준화 형태 때문이다. $z$ 통계량도, $t$ 통계량도, 회귀계수의 검정통계량도 모두 이 꼴이며 분모에 놓인 것은 언제나 표준오차다. 표본평균에서 그 분모는 $\sigma/\sqrt n$이고, $\sigma$를 모르면 $s/\sqrt n$으로 바꿔 쓴다. 그 대체가 $t$ 분포를 불러온다.

표준오차를 갖는 것은 평균만이 아니다. $S^2$에도 표준오차가 있고, 예제 4에서 보았듯 그 값은 모집단의 4차적률에 의존한다. 분산의 추정은 평균의 추정보다 훨씬 꼬리에 민감하다는 뜻이다.

다음 쪽 **붓스트랩 표준오차**로 넘어간다. 이론 공식이 없거나 가정을 믿기 어려울 때 자료에서 직접 표준오차를 만들어 내는 방법이다. 그다음 5.4절부터는 통계량을 하나씩 잡아 그 표본분포를 이론과 모의실험으로 살펴본다.
