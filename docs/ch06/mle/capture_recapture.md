# 포획–재포획법

> **참고 자료:** [Wikipedia — Mark and Recapture](https://en.wikipedia.org/wiki/Mark_and_recapture)

## 개요

표지–재포획법은 개체를 하나하나 세는 것이 현실적으로 불가능할 때 동물 개체군의 크기를 추정하기 위해 생태학에서 흔히 쓰이는 방법이다. 개체군의 일부를 포획하여 표지를 붙인 뒤 놓아 준다. 나중에 다시 일부를 포획하고 그 표본 안에서 표지된 개체의 수를 센다. 두 번째 표본 안의 표지 개체 수는 개체군 전체의 표지 개체 수에 비례해야 하므로, 표지 개체 수를 두 번째 표본에서의 표지 개체 비율로 나누면 전체 개체군 크기의 추정값을 얻을 수 있다.

이 방법은 포획–재포획, 포획–표지–재포획, 표지–방사–재포획, 다중체계 추정, 밴드 회수, Petersen법, Lincoln법 등으로도 불린다.

## 포획–재포획법의 단계

1. **포획과 표지**: 개체군에서 개체들을 확률적으로 표본추출하여 포획한다. 재포획되었을 때 식별할 수 있도록 표지(꼬리표, 밴드, 무해한 염료 등)를 붙인다. 표지된 개체 수를 $M$이라 하자.

2. **방사**: 표지된 개체를 개체군에 다시 놓아 주고 충분히 섞이도록 한다.

3. **재포획**: 개체군에서 두 번째 확률표본을 얻는다. 이 표본의 일부는 이미 표지되어 있을 것이다. 두 번째 표본의 전체 개체 수를 $n$, 그중 표지된 개체 수를 $m$이라 하자.

4. **추정**: 두 번째 표본의 표지 개체 비율이 첫 단계에서 표지된 개체군 전체의 비율을 대표한다는 가정 아래 개체군 크기 $N$을 추정할 수 있다:

$$
\hat{N} = \frac{M \cdot n}{m}
$$

## 가정

1. **폐쇄 개체군**: 연구 기간 동안 개체군 크기가 일정하다(출생, 사망, 유입, 유출이 없다).
2. **동일한 포획확률**: 각 표본추출에서 모든 개체가 포획될 기회가 같다.
3. **표지가 행동에 영향을 주지 않음**: 표지 과정이 재포획 가능성에 영향을 주지 않는다.
4. **표지가 오래가고 눈에 띔**: 표지가 사라지지 않고 연구 기간 내내 식별 가능하다.

## 간단한 예

연못의 물고기 개체수를 추정한다고 하자:

1. **첫 포획**: 물고기 50마리를 잡아 표지하고 놓아 준다($M = 50$).
2. **두 번째 포획**: 물고기 40마리를 잡았는데($n = 40$) 그중 10마리가 표지되어 있다($m = 10$).
3. **추정**:

$$
\hat{N} = \frac{M \cdot n}{m} = \frac{50 \cdot 40}{10} = 200
$$

연못의 추정 물고기 개체수는 200마리이다.

## MLE 유도

### 초기하분포

개체군 크기가 $N$일 때 크기 $n$인 재포획 표본에서 표지 개체 $m$마리를 관측할 확률은:

$$
P(m \mid N) = \frac{\binom{M}{m} \binom{N-M}{n-m}}{\binom{N}{n}}
$$

여기서:

- $\binom{M}{m}$: $M$마리 중 표지된 $m$마리를 고르는 경우의 수.
- $\binom{N-M}{n-m}$: 남은 $N-M$마리 중 표지되지 않은 $n-m$마리를 고르는 경우의 수.
- $\binom{N}{n}$: $N$마리 중 $n$마리를 고르는 전체 경우의 수.

### 가능도함수

가능도함수 $L(N)$은 이 확률에 비례한다:

$$
L(N) = P(m \mid N) \propto \frac{\binom{M}{m} \binom{N-M}{n-m}}{\binom{N}{n}}
$$

여기서 $M$, $n$, $m$은 실험에서 알려진 값이고 $N$이 추정 대상 모수이다.

### 간단히 한 가능도

$N$에 의존하지 않는 상수를 무시하면 가능도는 다음과 같아진다:

$$
L(N) \propto \frac{(N-M)!\,(N-n)!}{(N-M-n+m)!\;N!}
$$

### 가능도의 최대화

가능도에 자연로그를 취해 $\ell(N) = \log L(N)$을 얻고 $N$에 대해 미분하면 $N$의 MLE를 얻는다:

$$
\hat{N} = \frac{M \cdot n}{m}
$$

### MLE의 직관

추정값 $\hat{N}$은 재포획 표본에서의 표지 개체 비율($m/n$)이 개체군 전체에서의 표지 개체 비율($M/N$)을 반영한다는 착상에 기반한다:

$$
\frac{m}{n} \approx \frac{M}{N}
\quad\Longrightarrow\quad
\hat{N} = \frac{M \cdot n}{m}
$$

## MLE의 성질

### 편향

표본크기가 작으면 $\hat{N}$이 약간 편향될 수 있다. **Chapman 추정량**이 편향을 보정한 대안을 제공한다:

$$
\hat{N}_{\text{Chapman}} = \frac{(M+1)(n+1)}{m+1} - 1
$$

### 분산

MLE의 분산은 다음과 같이 근사할 수 있다:

$$
\text{Var}(\hat{N}) \approx \frac{M^2 \cdot n \cdot (n-m)}{m^3}
$$

## 확장과 변형

- **다중 재포획**: 여러 차례에 걸쳐 포획하고 표지하여 추정값을 정교하게 만든다.
- **개방 개체군**: Jolly-Seber 모형 같은 확장은 개체가 들어오고 나갈 수 있는 개체군을 다룰 수 있다.
- **포획확률이 다른 경우**: Lincoln-Petersen 추정량이나 로지스틱 회귀 같은 모형으로 포획확률의 변동을 보정할 수 있다.

<div class="codebox" markdown>

### 예제 1. 포획-재포획으로 모집단 크기 추정하기 { .eg }

```python
import matplotlib.pyplot as plt
from scipy import special

def prob(n, c, r, t):
    """
    Calculate the probability of capturing 't' tagged birds in a recapture
    sample of size 'r', given that there are 'n' birds in total.

    Parameters:
    - n: Total number of birds in the population
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - Probability of observing 't' tagged birds in the recapture sample.
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)

def capture_recapture(c=10, r=10, t=3):
    """
    Calculate the probability distribution over possible total population sizes
    and determine the MLE (Maximum Likelihood Estimate) for the population size.

    Parameters:
    - c: Number of birds captured and tagged in the first stage
    - r: Number of birds recaptured in the second stage
    - t: Number of tagged birds in the recapture stage

    Returns:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    """
    prob_list = []

    # 가능한 모집단 크기 n마다 확률을 구한다.
    for n in range(c + r - t, 10 * (c + r - t)):
        prob_list.append(prob(n, c, r, t))

    # 확률이 가장 큰 n이 모집단 크기의 MLE다.
    prob_max = max(prob_list)
    idx = prob_list.index(prob_max)
    mle_n = idx + (c + r - t)
    print(f'MLE n: {mle_n}')

    return prob_list, mle_n

def draw(prob_list, mle_n, c=10, r=10, t=3):
    """
    Plot the probability distribution of the total population size
    and highlight the MLE.

    Parameters:
    - prob_list: List of probabilities for each population size
    - mle_n: MLE for the total population size
    - c, r, t: Parameters for the capture-recapture model
    """
    idx = mle_n - (c + r - t)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(range(c + r - t, 10 * (c + r - t)), prob_list, label='Probability')
    ax.plot([mle_n, mle_n], [0, prob_list[idx]], 'o--r', label=f'MLE: {mle_n}')

    # 축 이름과 범례를 다듬는다.
    ax.set_xlabel('Total Population Size (n)')
    ax.set_ylabel('Probability')
    ax.set_title('Capture-Recapture MLE for Population Size')
    ax.legend()
    plt.show()

# 포획-재포획 모형의 값들: 표지 수, 재포획 수, 그중 표지된 수.
c = 5   # Birds captured and tagged in the first stage
r = 6   # Birds recaptured in the second stage
t = 2   # Tagged birds in the recapture stage

# 후보마다 확률을 구하고 가장 큰 것을 고른다.
prob_list, mle_n = capture_recapture(c, r, t)

# 확률을 후보별로 그리고 최댓값 자리를 표시한다.
draw(prob_list, mle_n, c, r, t)
```

출력:

```
MLE n: 14
```

![Capture-Recapture MLE for Population Size](./img/capture_recapture_124.png)

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
어떤 야생동물 생물학자가 호수에서 물고기 $M = 20$마리를 잡아 표지했다. 나중에 $n = 15$마리를 재포획했더니 $m = 5$마리가 표지되어 있었다. 전체 개체수 $\hat{N}$에 대한 Lincoln-Petersen MLE와 Chapman 편향 보정 추정값을 계산하라.

</div>

??? success "풀이"
    Lincoln-Petersen MLE는:

    $$
    \hat{N} = \frac{M \cdot n}{m} = \frac{20 \times 15}{5} = 60
    $$

    Chapman 추정량은:

    $$
    \hat{N}_{\text{Chapman}} = \frac{(M+1)(n+1)}{m+1} - 1 = \frac{21 \times 16}{6} - 1 = 56 - 1 = 55
    $$

    Chapman 추정값(55)이 Lincoln-Petersen 추정값(60)보다 약간 작으며, 이는 소표본에 대한 편향 보정을 반영한다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
포획–재포획법이 깔고 있는 가정들을 설명하고 그것이 위배되면 어떤 일이 생기는지 서술하라.

</div>

??? success "풀이"
    핵심 가정은 다음과 같다:

    1. **폐쇄 개체군**: 포획과 재포획 사이에 출생, 사망, 유입, 유출이 없다. 위배되면 개체군이 늘었는지 줄었는지에 따라 $\hat{N}$이 부풀거나 줄어든다.

    2. **동일한 포획확률**: 모든 개체가 포획될 확률이 같다. 표지된 개체가 "덫을 피하게" 되면(재포획될 가능성이 낮아지면) $m$이 너무 작아져 $\hat{N}$이 지나치게 커진다. 반대로 "덫을 좋아하게" 되면 $\hat{N}$이 지나치게 작아진다.

    3. **표지가 사라지지 않음**: 재포획 시 표지가 여전히 보인다. 표지가 사라지면 재포획된 표지 개체 일부가 비표지로 세어져 $m$이 줄고 $\hat{N}$이 부풀려진다.

    4. **섞임**: 포획 사이에 표지된 개체가 개체군과 무작위로 섞인다. 무리 지어 남아 있으면 재포획 표본이 대표성을 갖지 못한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
어떤 포획–재포획 연구에서 $M = 10$마리에 표지를 붙였다. $n = 10$마리를 재포획했는데 표지된 개체가 하나도 없었다($m = 0$). MLE 공식은 무엇을 주는가? 이 추정값이 합리적인가? 대안을 제시하라.

</div>

??? success "풀이"
    Lincoln-Petersen 공식은 $\hat{N} = Mn/m = 10 \times 10 / 0$을 주는데, 이는 0으로 나누는 것이므로 **정의되지 않는다**. 개체군이 무한히 크다고 추정하는 셈이어서 합리적이지 않다.

    이런 상황은 표지한 수에 비해 개체군이 매우 크거나 가정이 위배될 때 생긴다. Chapman 추정량은 이 경우를 처리할 수 있다: $\hat{N}_{\text{Chapman}} = (11 \times 11)/1 - 1 = 120$. 또는 표지하는 개체 수나 재포획 표본크기를 늘리면 더 믿을 만한 추정값을 얻을 수 있다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
$m$을 확률변수로 보고 $\hat{N} = Mn/m$에 델타 방법을 적용하여 근사 분산 공식 $\text{Var}(\hat{N}) \approx \frac{M^2 n(n-m)}{m^3}$을 유도하라.

</div>

??? success "풀이"
    $\hat{N} = g(m) = Mn/m$에 델타 방법을 적용한다. (초기하 모형 아래에서) $m$의 기댓값은 $E[m] = nM/N$이다. 도함수는:

    $$
    g'(m) = -\frac{Mn}{m^2}
    $$

    (초기하분포에서) $m$의 분산은 근사적으로:

    $$
    \text{Var}(m) \approx n \cdot \frac{M}{N} \cdot \frac{N-M}{N} \cdot \frac{N-n}{N-1}
    $$

    델타 방법에 의해 $\text{Var}(\hat{N}) \approx [g'(m)]^2 \text{Var}(m)$이다. $N \approx \hat{N} = Mn/m$을 대입하고 (큰 $N$에서 유한모집단 수정을 무시하며) 정리하면:

    $$
    \text{Var}(\hat{N}) \approx \frac{M^2 n^2}{m^4} \cdot \frac{n \cdot m \cdot (n-m)}{n^2} = \frac{M^2 n(n-m)}{m^3}
    $$

    $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
연습문제 4의 분산 공식으로 $M=20$, $n=15$, $m=5$일 때 $\hat N$의 표준오차와 95% 신뢰구간을 구하라. 이 구간을 그대로 보고해도 되는가?

</div>

??? success "풀이"
    $\hat N = Mn/m = 60$이고

    $$
    \operatorname{Var}(\hat N) \approx \frac{M^2n(n-m)}{m^3} = \frac{400\times15\times10}{125} = 480
    $$

    이므로 $\operatorname{SE}(\hat N) = \sqrt{480} = 21.9$이다. 왈드 구간은

    $$
    60 \pm 1.96(21.9) = (17.1,\ 102.9)
    $$

    이다.

    **그대로 보고하면 안 된다.** 세 가지 문제가 있다.

    1. **하한이 말이 안 된다.** $M=20$마리에 표지했으므로 $N \ge 20$이어야 하고, 재포획에서 $n=15$마리를 잡았으므로 사실 $N \ge 30$이다. 구간의 하한 17.1은 논리적으로 불가능한 값이다.
    2. **심하게 비대칭이어야 한다.** $\hat N = Mn/m$은 $m$의 **역수**이므로 $m$이 조금만 작아져도 $\hat N$이 크게 뛴다. $m=4$였다면 $\hat N=75$, $m=3$이었다면 100이다. 위쪽으로 훨씬 긴 구간이 나와야 하는데 대칭 구간은 이를 담지 못한다.
    3. **$m=5$는 너무 작다.** 앞서 본 대로 정밀도를 정하는 것은 표본 크기가 아니라 **재포획된 표지 개체 수 $m$** 이다. $m=5$면 상대오차가 $1/\sqrt5 = 45\%$ 수준이다.

    **대안.** $\ln\hat N$ 척도에서 구간을 만들거나($\operatorname{SE}(\ln\hat N)\approx\sqrt{(n-m)/(nm)}$), 초기하 가능도의 프로파일 구간을 쓰거나, $m$의 정확 이항·초기하 구간을 뒤집어 $N$의 구간을 얻는다. 어느 쪽이든 비대칭 구간이 나오고 하한이 $\max(M+n-m, \hat N_{\text{하한}})$ 위에 놓인다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
**포획확률이 개체마다 다르면** 링컨-피터슨 추정량이 어느 방향으로 치우치는가? 이유를 설명하고 대처법을 적어라.

</div>

??? success "풀이"
    **방향: 과소추정한다.**

    **이유.** 잡히기 쉬운 개체는 **첫 번째와 두 번째 포획 모두에서** 잡히기 쉽다. 따라서 표지 개체가 재포획 표본에 과대 대표되고, 관측된 $m$이 "모든 개체가 똑같이 잡힌다면 기대되는 값"보다 커진다.

    $\hat N = Mn/m$에서 $m$이 과대하면 $\hat N$이 과소해진다.

    형식적으로, 개체 $i$의 포획확률을 $p_i$라 하면

    $$
    E[m] \approx N\,E[p^2] \ge N\,(E[p])^2
    $$

    이고(코시-슈바르츠 또는 분산의 비음수성), 등호는 모든 $p_i$가 같을 때만 성립한다. 포획확률의 변동이 클수록 편향이 커진다.

    **다른 가정 위반의 방향도 함께 정리하면.**

    | 위반 | $\hat N$의 방향 |
    |---|---|
    | 포획 이질성(잡히기 쉬운 개체 존재) | 과소추정 |
    | 표지 탈락·사망 | 과대추정 |
    | 표지 개체가 덫을 피함(trap-shy) | 과대추정 |
    | 표지 개체가 덫에 끌림(trap-happy) | 과소추정 |
    | 두 포획 사이에 유입·유출 | 방향 불분명 |

    **대처.**

    - **표본을 여러 번 잡는다.** 3회 이상이면 포획 이력(예: 101, 110)에서 이질성을 추정할 수 있다. 오터리스 모형군의 $M_h$(이질성), $M_b$(행동 반응), $M_t$(시간 변동) 모형이 각각의 위반을 다룬다.
    - **공변량을 쓴다.** 크기, 나이, 성별로 포획확률을 모형화하면 관측된 이질성을 설명할 수 있다.
    - **층화한다.** 서식지나 성별로 나누어 각각 추정한 뒤 합친다.
    - **설계로 줄인다.** 두 포획의 방법을 서로 다르게 하면(덫과 그물) 포획확률의 상관이 줄어 편향이 완화된다.

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
두 개의 불완전한 명단(예: 병원 기록과 보건소 신고)으로 어떤 질병의 실제 환자 수를 추정하려 한다. 포획-재포획의 구조와 어떻게 대응되는지 밝히고, 이 응용에서 특히 위험한 가정을 지적하라.

</div>

??? success "풀이"
    **대응.**

    | 포획-재포획 | 다중 명단 추정 |
    |---|---|
    | 1차 포획해 표지한 개체 $M$ | 명단 A에 오른 환자 수 |
    | 2차 포획 표본 $n$ | 명단 B에 오른 환자 수 |
    | 재포획된 표지 개체 $m$ | 두 명단 모두에 오른 수 |
    | 전체 개체수 $N$ | 실제 전체 환자 수 |

    추정값은 그대로 $\hat N = Mn/m$이며, 명단 어디에도 없는 "숨은" 환자 수가 $\hat N - (M+n-m)$이다.

    **특히 위험한 가정: 두 명단의 독립성.**

    야생동물에서는 두 번의 포획이 시간적으로 분리되어 있어 독립을 어느 정도 기대할 수 있다. 그러나 두 명단은 **같은 이유로 사람을 놓친다.**

    - 중증 환자는 병원에도 가고 신고도 되지만, 경증이거나 의료 접근성이 낮은 사람은 **양쪽 모두에서 빠진다.**
    - 이는 양의 의존을 만들고, 앞 연습문제와 같은 논리로 $m$이 과대해져 **실제 환자 수를 과소추정**한다.

    반대 방향의 의존도 가능하다. 병원에서 신고를 대신 해 주는 제도가 있으면 A에 오른 사람이 B에도 오를 가능성이 높아져 역시 양의 의존이 된다.

    **그 밖의 위험.**

    - **명단 연결 오류.** 이름·생년월일로 두 명단을 맞추는데, 놓치면($m$ 과소) 과대추정하고 잘못 연결하면($m$ 과대) 과소추정한다.
    - **모집단의 정의.** "환자"의 정의가 두 명단에서 다르면 애초에 같은 $N$을 추정하는 것이 아니다.
    - **폐쇄성.** 관찰 기간 동안 유입·유출이 없어야 한다.

    **대처.** 명단을 셋 이상 확보하면 로그선형모형으로 **의존구조를 모형화**할 수 있다. 두 명단만으로는 독립 가정을 검정할 방법이 전혀 없다는 점이 핵심이며, 그래서 실무에서는 세 개 이상을 권한다. 그럼에도 추정값의 불확실성은 통계적 오차보다 모형 가정에서 오는 몫이 크므로, 여러 모형의 결과 범위를 함께 보고하는 것이 정직하다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
연구를 설계하는 단계에서 $\hat N$의 상대오차를 20% 이내로 하려면 재포획 표본 $n$을 얼마나 잡아야 하는가? $N \approx 500$, $M = 50$으로 가정하라.

</div>

??? success "풀이"
    **정밀도는 $m$이 정한다.** 델타 방법에서

    $$
    \frac{\operatorname{SE}(\hat N)}{\hat N} \approx \sqrt{\frac{n-m}{nm}} \approx \frac{1}{\sqrt m}\quad (n \gg m)
    $$

    이므로 상대오차 20%를 얻으려면

    $$
    \frac{1}{\sqrt m} \le 0.20 \implies m \ge 25
    $$

    **필요한 $n$.** $E[m] = nM/N = n(50/500) = 0.1n$이므로

    $$
    0.1n \ge 25 \implies n \ge 250
    $$

    이다. 개체 500마리 중 250마리를 다시 잡아야 한다. 전체의 절반이다.

    **읽어 둘 점.**

    - **$M$을 늘리는 것이 훨씬 효율적이다.** $M=100$이면 $E[m] = 0.2n$이 되어 $n=125$면 충분하다. $M$과 $n$을 함께 늘리는 것이 좋고, 대칭적인 역할이므로 비용이 비슷하면 $M \approx n$으로 두는 것이 최적이다.
    - **$M=n=\sqrt{25N}$이 어림이다.** $E[m] = Mn/N$이므로 $m \ge 25$를 얻으려면 $Mn \ge 25N$이면 되고, 총 포획 $M+n$을 최소로 하려면 둘을 같게 둔다. $N=500$이면 $M=n=112$다.
    - **유한모집단 보정.** 위 식은 $n \ll N$을 전제한다. $n$이 $N$에 가까우면 실제 분산이 더 작아지므로 필요한 $n$이 조금 줄어든다.

    **현실적 함의.** 개체수가 많을수록 필요한 포획량도 비례해 늘어난다. 희귀종이나 대규모 모집단에서 포획-재포획이 어려운 이유이며, 그래서 표지 없이 추정하는 방법(거리표본조사, 유전자 표지, 카메라 트랩)이 개발되었다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff hard" title="어려움"></span>
$N$은 정수이므로 미분으로 최대화할 수 없다. 가능도비 $L(N)/L(N-1)$을 계산해 MLE가 $\hat N = \lfloor Mn/m \rfloor$임을 보여라.

</div>

??? success "풀이"
    초기하 가능도는

    $$
    L(N) = \frac{\binom{M}{m}\binom{N-M}{n-m}}{\binom{N}{n}}
    $$

    이다. $\binom{M}{m}$은 $N$과 무관하므로 비를 계산하면

    $$
    \frac{L(N)}{L(N-1)} = \frac{\binom{N-M}{n-m}}{\binom{N-1-M}{n-m}}\cdot\frac{\binom{N-1}{n}}{\binom{N}{n}}
    $$

    이다. 두 인수를 각각 정리하면

    $$
    \frac{\binom{N-M}{n-m}}{\binom{N-1-M}{n-m}} = \frac{N-M}{N-M-n+m}, \qquad \frac{\binom{N-1}{n}}{\binom{N}{n}} = \frac{N-n}{N}
    $$

    이므로

    $$
    \frac{L(N)}{L(N-1)} = \frac{(N-M)(N-n)}{N(N-M-n+m)}
    $$

    이다.

    **1과 비교.** 분모가 양수일 때 $L(N) > L(N-1)$은

    $$
    (N-M)(N-n) > N(N-M-n+m)
    $$

    와 동치다. 좌변을 전개하면 $N^2 - N(M+n) + Mn$이고 우변은 $N^2 - N(M+n) + Nm$이므로, 조건이

    $$
    Mn > Nm \iff N < \frac{Mn}{m}
    $$

    로 정리된다. $\square$

    **결론.** $N < Mn/m$이면 가능도가 증가하고 $N > Mn/m$이면 감소한다. 따라서 최대는 $Mn/m$ 이하의 가장 큰 정수, 즉

    $$
    \hat N = \left\lfloor \frac{Mn}{m}\right\rfloor
    $$

    이다. $Mn/m$이 정확히 정수이면 $L(\hat N) = L(\hat N - 1)$로 최댓값이 두 곳에서 달성된다.

    **요령의 일반성.** 이산 모수의 MLE를 찾을 때 **이웃한 가능도의 비를 1과 비교하는 것**이 표준적인 방법이다. 앞서 이항분포의 최빈값을 찾을 때 쓴 것과 같은 기법이며, 미분이 불가능한 상황에서 "증가하다가 감소한다"는 구조를 직접 확인해 준다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff med" title="중간"></span>
채프먼 추정량 $\hat N_C = \dfrac{(M+1)(n+1)}{m+1}-1$이 링컨-피터슨보다 나은 이유를 두 가지 들어라. $M=20$, $n=15$, $m=0$인 경우에 두 값을 비교하라.

</div>

??? success "풀이"
    **이유 1 — $m=0$에서 정의된다.** 링컨-피터슨은 $Mn/0 = \infty$로 아무 답도 주지 못한다. 채프먼은

    $$
    \hat N_C = \frac{21\times16}{1}-1 = 335
    $$

    를 준다. 유한한 값이며, "표지 개체를 하나도 못 잡았으니 모집단이 꽤 크다"는 정보를 합리적으로 반영한다.

    **이유 2 — 편향이 훨씬 작다.** 링컨-피터슨은 $1/m$이 볼록함수라 옌센 부등식에 따라 $N$을 **과대추정**한다. $m$이 작을수록 심하다. 채프먼은 분자·분모에 1을 더해 이 편향을 상쇄하며, $n+M \ge N$이면 **정확히 불편**임이 알려져 있다. 그렇지 않은 경우에도 편향이 링컨-피터슨보다 자릿수로 작다.

    **$m=5$인 경우 비교.**

    $$
    \hat N_{\text{LP}} = \frac{20\times15}{5} = 60, \qquad \hat N_C = \frac{21\times16}{6}-1 = 55
    $$

    채프먼 쪽이 5 작다. 편향 보정이 아래 방향으로 작용한 결과다.

    **덧붙임.** 채프먼 추정량의 분산 추정값으로는

    $$
    \widehat{\operatorname{Var}}(\hat N_C) = \frac{(M+1)(n+1)(M-m)(n-m)}{(m+1)^2(m+2)}
    $$

    가 쓰인다. 역시 $m=0$에서도 유한하다. 현대의 포획-재포획 실무에서는 링컨-피터슨보다 채프먼을 기본으로 삼는다.

---

## 정리하며

포획–재포획은 **세지 않고 크기를 재는** 방법이다.

$$
\hat N = \frac{M n}{m}
$$

- **비례식 하나가 전부다.** 두 번째 표본의 표지 비율 $m/n$ 이 개체군 전체의 표지 비율 $M/N$ 을 대표한다고 놓으면 곧바로 나온다. 이것이 최대가능도추정량이기도 하다(링컨–피터슨 추정량).
- **$m$ 이 분모에 있다는 점이 위험하다.** 재포획에서 표지 개체가 하나도 없으면 $\hat N=\infty$ 이고, 적게 나오면 추정값이 폭발한다. 그래서 실무에서는 편향을 줄인 채프먼 추정량 $\hat N=\frac{(M+1)(n+1)}{m+1}-1$ 을 쓴다.
- **가정이 강하다.** 개체군이 닫혀 있어야 하고(출생·사망·이동 없음), 표지가 떨어지지 않아야 하며, 표지 개체와 비표지 개체의 포획 확률이 같아야 한다. **마지막 조건이 가장 자주 깨진다** — 한 번 잡힌 개체가 덫을 피하거나 반대로 더 잘 잡히는 일이 흔하다.
- **생태학 밖에서도 쓰인다.** 소프트웨어 결함 수 추정, 인구 미계수 보정, 문서 중복 탐지가 같은 구조다.

다음 절 **최대가능도의 점근적 성질**로 넘어간다. 지금까지 개별 분포에서 확인한 성질들이 **일반적으로** 왜 성립하는지를 본다.
