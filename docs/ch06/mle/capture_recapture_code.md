# 포획–재포획 최대가능도

## 개요

**포획–재포획법**은 직접 셀 수 없는 개체군의 크기를 추정하는 고전적인 기법이다. 일부 개체를 포획해 표지를 붙여 놓아 준 뒤 다시 일부를 포획하여 표지된 개체가 몇 마리인지 세면, 전체 개체군 크기의 최대가능도추정값을 유도할 수 있다. 이 페이지에서는 포획–재포획 모형의 초기하 가능도와 MLE를 전개한다.

## 포획–재포획의 설정

이 방법은 두 단계로 진행된다:

1. **포획 단계:** 크기가 미지인 $N$의 개체군에서 $c$마리를 포획해 표지를 붙이고 놓아 준다.
2. **재포획 단계:** $r$마리를 포획한다. 그중 $t$마리가 표지되어 있다.

핵심 물음은 $N$이 얼마인가이다.

!!! info "가정"

    - 개체군이 폐쇄되어 있다(두 단계 사이에 출생, 사망, 유입, 유출이 없다).
    - 모든 개체가 포획될 확률이 같다.
    - 표지가 사라지지 않고 올바르게 식별된다.
    - 두 번째 단계의 포획이 첫 단계의 포획과 독립이다.

## 초기하 모형

전체 $N$마리 중 $c$마리가 표지되어 있을 때, 크기 $r$인 재포획 표본에서 표지된 개체 수 $T$는 **초기하분포**를 따른다:

$$
P(T = t \mid N) = \frac{\binom{c}{t}\binom{N - c}{r - t}}{\binom{N}{r}}
$$

이는 $\max(0, r + c - N) \leq t \leq \min(r, c)$에서 성립한다.

관심 모수는 $N$이고, 가능도함수는 자료 $(c, r, t)$를 고정한 채 $L(N) = P(T = t \mid N)$을 $N$의 함수로 본 것이다.

## 최대가능도추정량

$N$의 MLE는 $L(N)$을 최대화하는 값이다. $N$이 (양의 정수인) 이산 모수이므로 $N \geq c + r - t$인 정수에서 탐색한다.

MLE는 잘 알려진 닫힌 형태를 갖는다:

$$
\hat{N}_{\text{MLE}} = \left\lfloor \frac{cr}{t} \right\rfloor
$$

이것이 (가장 가까운 정수로 내림한) **Lincoln-Petersen 추정값**이다.

### 직관

MLE는 비례 논증에서 나온다. 재포획이 대표성을 가지면 재포획 표본에서 표지된 개체의 비율이 개체군에서의 비율을 근사해야 한다:

$$
\frac{t}{r} \approx \frac{c}{N} \quad \Rightarrow \quad N \approx \frac{cr}{t}
$$

## 구현

<div class="codebox" markdown>

**예제 1.** 포획-재포획 MLE 구현

```python
from scipy import special

def prob(n, c, r, t):
    """
    Hypergeometric probability: P(T = t | N = n).
    
    Parameters
    ----------
    n : Total population size
    c : Number tagged in capture phase
    r : Number in recapture sample
    t : Number of tagged in recapture
    """
    return special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)


def capture_recapture_mle(c, r, t):
    """
    Compute the MLE of population size N via exhaustive search.
    """
    n_min = c + r - t  # minimum possible N
    n_max = 10 * n_min  # search range

    prob_list = [prob(n, c, r, t) for n in range(n_min, n_max)]
    mle_idx = max(range(len(prob_list)), key=lambda i: prob_list[i])
    mle_n = mle_idx + n_min

    return mle_n, prob_list


# 보기: 10마리에 표지, 10마리를 다시 잡았고 그중 3마리가 표지된 개체였다.
c, r, t = 10, 10, 3
mle_n, probs = capture_recapture_mle(c, r, t)
print(f"Capture: {c} tagged, Recapture: {r} caught, {t} tagged")
print(f"MLE of N: {mle_n}")
print(f"Lincoln-Petersen estimate: {c * r // t}")
```

출력:

```
Capture: 10 tagged, Recapture: 10 caught, 3 tagged
MLE of N: 33
Lincoln-Petersen estimate: 33
```

</div>

<div class="codebox" markdown>

### 예제 2. 포획–재포획 최대가능도 { .eg }

어떤 야생동물 생물학자가 새 $c = 5$마리를 잡아 표지하고 놓아 준 뒤, 나중에 $r = 6$마리를 재포획했더니 그중 $t = 2$마리가 표지되어 있었다고 하자.

```python
c, r, t = 5, 6, 2
mle_n, probs = capture_recapture_mle(c, r, t)
print(f"MLE of N: {mle_n}")
print(f"Lincoln-Petersen: {c * r // t}")
```

출력:

```
MLE of N: 14
Lincoln-Petersen: 15
```

Lincoln-Petersen 추정값은 $\hat{N} = \lfloor 5 \times 6 / 2 \rfloor = 15$이다.

가능도함수는 $N = 15$에서 뚜렷한 봉우리를 보이며, $N$이 그보다 작거나 크면 확률이 줄어든다.

</div>

## 추정량의 성질

!!! note "Lincoln-Petersen 추정량의 편향"
    기본 Lincoln-Petersen 추정량 $cr/t$는 편향되어 있으며, 특히 $t$가 작을 때 $N$을 과대추정하는 경향이 있다. Chapman의 보정 추정량이 이 편향을 줄여 준다:

    $$
    \hat{N}_{\text{Chapman}} = \frac{(c+1)(r+1)}{t+1} - 1
    $$

이 문제의 가능도함수 $L(N)$은 **단봉**이므로(봉우리가 하나이므로) MLE가 유일하고 격자탐색을 믿을 수 있다.

## 민감도 분석

추정의 품질은 재포획된 표지 개체 수 $t$에 크게 의존한다:

- ($r$과 $c$에 비해) $t$가 크면 추정이 정밀하다.
- $t$가 작으면(예: $t = 1$) 추정을 신뢰할 수 없고 가능도함수가 평평하다.
- $t = 0$이면 MLE가 정의되지 않는다(개체군이 얼마든지 클 수 있다).

<div class="codebox" markdown>

**예제 3.** 재포획 결과에 따른 민감도

```python
from scipy import special

def sensitivity_analysis():
    """재포획된 표지 개체 수 t 를 바꿔 가며 MLE가 어떻게 변하는지 본다.

    t가 작을수록(표지가 거의 안 잡힐수록) 추정 개체수가 커진다.
    t = 1 처럼 극단적인 경우 추정값이 100을 넘고 매우 불안정해지는데,
    포획-재포획 조사에서 재포획 표본을 충분히 크게 잡아야 하는 이유다.
    """
    c, r = 10, 10
    print(f"c = {c}, r = {r}")
    print(f"{'t':>4} {'MLE':>6} {'cr/t':>8}")
    print("-" * 20)
    for t in range(1, min(c, r) + 1):
        n_min = c + r - t
        n_max = 10 * n_min
        probs = [special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)
                 for n in range(n_min, n_max)]
        mle_idx = max(range(len(probs)), key=lambda i: probs[i])
        mle_n = mle_idx + n_min
        print(f"{t:>4} {mle_n:>6} {c*r/t:>8.1f}")

sensitivity_analysis()
```

출력:

```
c = 10, r = 10
   t    MLE     cr/t
--------------------
   1     99    100.0
   2     49     50.0
   3     33     33.3
   4     24     25.0
   5     19     20.0
   6     16     16.7
   7     14     14.3
   8     12     12.5
   9     11     11.1
  10     10     10.0
```

</div>

## 해석

- 포획–재포획 MLE는 표지–재포획 자료로부터 개체군 크기를 추정하는 원리 있는 방법을 제공한다.
- 이 방법은 유한모집단에서의 비복원추출을 모형화하는 초기하분포에 의존한다.
- Lincoln-Petersen 공식 $\hat{N} = cr/t$는 우아한 비례 해석을 갖지만 $t$가 작으면 편향될 수 있다.
- 실제 생태학 응용에서는 폐쇄 개체군 가정과 동일 포획확률 가정이 깨지는 경우를 고려해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span> 어떤 해양생물학자가 물고기 $c = 20$마리에 표지를 붙여 놓아 주었다. 나중에 $r = 25$마리를 표본으로 잡았더니 $t = 5$마리가 표지되어 있었다. 격자탐색과 Lincoln-Petersen 공식 두 가지로 전체 개체수의 MLE를 계산하라.

</div>

??? success "풀이"
    Lincoln-Petersen: $\hat{N} = \lfloor cr/t \rfloor = \lfloor 20 \times 25/5 \rfloor = 100$.

    격자탐색:
    ```python
    from scipy import special
    c, r, t = 20, 25, 5
    n_min = c + r - t  # = 40
    probs = [special.comb(n - c, r - t) * special.comb(c, t) / special.comb(n, r)
             for n in range(n_min, 500)]
    mle_idx = max(range(len(probs)), key=lambda i: probs[i])
    print(f"MLE: N = {mle_idx + n_min}")
    ```

    출력:

    ```
    MLE: N = 100
    ```

    두 방법 모두 $\hat{N} = 100$을 준다. $\square$

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span> $N$을 연속으로 다룰 때 Lincoln-Petersen 추정량 $\hat{N} = cr/t$가 초기하 가능도를 최대화하는 값임을 보여라. (힌트: $L(N)/L(N-1) > 1$일 필요충분조건이 $N < cr/t$임을 보여라.)

</div>

??? success "풀이"
    가능도비는:

    $$
    \frac{L(N)}{L(N-1)} = \frac{\binom{N-c}{r-t}}{\binom{N-1-c}{r-t}} \cdot \frac{\binom{N-1}{r}}{\binom{N}{r}}
    $$

    $\binom{n}{k}/\binom{n-1}{k} = n/(n-k)$를 사용하면:

    $$
    \frac{L(N)}{L(N-1)} = \frac{N - c}{N - c - (r-t)} \cdot \frac{N - r}{N} = \frac{(N-c)(N-r)}{N(N-c-r+t)}
    $$

    이 비가 1을 넘을 조건은 $(N-c)(N-r) > N(N-c-r+t)$, 즉 $N^2 - (c+r)N + cr > N^2 - (c+r-t)N$이며, 정리하면 $cr > tN$, 즉 $N < cr/t$이다.

    따라서 $L(N)$은 $N < cr/t$에서 증가하고 $N > cr/t$에서 감소하므로, ($N$이 정수여야 하므로) 최댓값이 $N = \lfloor cr/t \rfloor$에 있음이 확인된다. $\square$

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span> Chapman의 보정 추정량은 $\hat{N}_C = (c+1)(r+1)/(t+1) - 1$이다. $c = 10, r = 10, t = 3$에 대해 $\hat{N}_C$를 계산하고 MLE와 비교하라. 이 보정이 유용한 이유는 무엇인가?

</div>

??? success "풀이"
    Chapman 추정값: $\hat{N}_C = (11)(11)/4 - 1 = 121/4 - 1 = 30.25 - 1 = 29.25$.

    MLE(Lincoln-Petersen)는 $\hat{N} = \lfloor 100/3 \rfloor = 33$을 준다.

    Chapman 추정량이 더 작은 이유는 Lincoln-Petersen 추정량의 양의 편향을 보정하기 때문이다. 이 편향은 $1/T$가 볼록하므로 Jensen 부등식에 의해 $E[cr/T] > cr/E[T]$이기 때문에 생긴다. Chapman의 보정은 이 편향을 대략 제거하며, $r$과 $c$에 비해 $t$가 작을 때 특히 유용하다. $\square$

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span> 재포획에서 표지된 개체가 하나도 없으면($t = 0$) MLE가 존재하지 않는 이유를 설명하라. 이는 포획–재포획 연구의 설계에 무엇을 함의하는가?

</div>

??? success "풀이"
    $t = 0$일 때 가능도함수는:

    $$
    L(N) = \frac{\binom{N-c}{r}}{\binom{N}{r}}
    $$

    $t = 0$이면 $L(N)$은 $N$에 대해 증가한다. 개체군이 클수록 재포획된 개체 중에 표지된 것이 하나도 없을 가능성이 커지기 때문이다. $N \to \infty$일 때 $L(N) \to 1$이다. 유한한 최대점이 없으므로 MLE가 존재하지 않는다.

    **설계에 대한 함의:** $t > 0$이 될 가능성이 높도록 연구를 설계해야 한다. 이를 위해서는:

    - 충분히 많은 수 $c$에 표지를 붙인다.
    - 충분히 많은 수 $r$을 재포획한다.
    - $P(T > 0)$이 높아지도록 곱 $cr/N$이 충분히 커야 한다. 경험 법칙으로 $cr \gg N$이거나 적어도 $cr/N > 5$여야 표지된 개체를 재포획할 확률이 웬만큼 확보된다. $\square$

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span> 델타 방법으로 Lincoln-Petersen 추정량 $\hat{N} = cr/T$의 분산을 유도하라. 초기하분포의 분산은 $\text{Var}(T) = r \cdot \frac{c}{N} \cdot \frac{N-c}{N} \cdot \frac{N-r}{N-1}$이다.

</div>

??? success "풀이"
    $g(T) = cr/T$로 두어 $\hat{N} = g(T)$라 하자. 델타 방법에 의해:

    $$
    \text{Var}(\hat{N}) \approx [g'(E[T])]^2 \, \text{Var}(T)
    $$

    $g'(T) = -cr/T^2$이고 $E[T] = rc/N$이므로:

    $$
    g'(E[T]) = \frac{-cr}{(rc/N)^2} = \frac{-N^2}{cr}
    $$

    대입하면:

    $$
    \text{Var}(\hat{N}) \approx \frac{N^4}{c^2 r^2} \cdot r \cdot \frac{c}{N} \cdot \frac{N-c}{N} \cdot \frac{N-r}{N-1}
    $$

    $$
    = \frac{N^2(N-c)(N-r)}{cr(N-1)}
    $$

    이 식은 $c$와 $r$이 커질수록 분산이 줄고 $N$이 커질수록 분산이 늘어남을 보여 준다. 포획 비율이 작은 큰 개체군에서는 분산이 매우 커질 수 있어 상당한 포획 노력이 필요함을 말해 준다. $\square$

---

## 정리하며

포획–재포획을 **초기하 가능도**에서 정식으로 유도했다.

- **가능도가 $N$ 의 함수다.** 표지 $c$ 마리가 있는 개체군 $N$ 에서 $r$ 마리를 뽑아 $t$ 마리가 표지일 확률이 초기하분포로 주어지고, 이를 $N$ 의 함수로 본 것이 가능도다.
- **$N$ 이 정수라 미분을 쓸 수 없다.** 대신 연속한 두 값의 가능도비 $L(N)/L(N-1)$ 이 $1$ 을 넘는지 따져 최댓값을 찾는다. 이산 모수에서 쓰는 표준 수법이다.
- **결과는 앞 절의 비례식과 같다.** $\hat N=\lfloor cr/t\rfloor$ 이며, 직관적 추정량이 최대가능도추정량이기도 하다는 것이 확인된다.
- **가능도가 $N$ 에 대해 매우 평평하다.** 그래서 점추정값은 얻기 쉬워도 신뢰구간이 넓으며, **$t$ 가 작을수록 오른쪽으로 심하게 늘어진다.** 점추정값만 보고하는 것이 위험한 대표적인 예다.

다음 절 **로그가능도 시각화**로 넘어간다. 가능도의 봉우리가 실제로 어떻게 생겼는지 그려 보면, 추정값과 정밀도가 한 그림에서 읽힌다.
