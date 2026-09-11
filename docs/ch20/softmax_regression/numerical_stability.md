# 수치적 안정성(로그-합-지수 기법)

## 오버플로 문제

소프트맥스를 계산하려면 로짓 $z_1, \ldots, z_C$를 지수화해야 한다. 64비트 부동소수점에서
$e^{z}$는 $z \gtrsim 709$이면 무한대로 넘치고 $z \lesssim -745$이면 0으로 사라진다. 신경망이나
큰 선형모형이 내는 원시 로짓은 이 한계를 쉽게 넘어서므로, 소프트맥스를 순진하게 구현하면
`inf`나 `0/0`이 나온다.

이 절에서는 수학적 결과를 바꾸지 않으면서 오버플로를 없애고 언더플로를 크게 줄이는 간단한
대수적 항등식, **로그-합-지수(LSE) 기법**을 전개한다.

## 소프트맥스의 평행이동 불변성

[소프트맥스 절](softmax.md)에서 보았듯이 소프트맥스는 평행이동 불변이다. 임의의 스칼라 $c$에
대해

$$
\operatorname{softmax}(\mathbf{z})_k
= \frac{e^{z_k}}{\sum_{j=1}^{C}e^{z_j}}
= \frac{e^{z_k - c}}{\sum_{j=1}^{C}e^{z_j - c}}
$$

가 성립한다. 분자와 분모에 $e^{-c}$를 곱하면 소거되기 때문이다. $c$를 잘 고르면 모든 지수를
안전한 범위에 둘 수 있다.

## 로그-합-지수 항등식

벡터 $\mathbf{z}$의 **로그-합-지수**는

$$
\operatorname{LSE}(\mathbf{z})
= \log\sum_{j=1}^{C}e^{z_j}
$$

이다. $c = \max_j z_j$로 평행이동을 적용하면

$$
\operatorname{LSE}(\mathbf{z})
= c + \log\sum_{j=1}^{C}e^{z_j - c}
$$

를 얻는다. 이동 후에는 모든 지수가 $z_j - c \le 0$이므로 $e^{z_j - c} \le 1$이다. 따라서
다음이 보장된다.

- **오버플로 없음:** 가장 큰 지수가 $e^0 = 1$이다.
- **언더플로 감소:** 최댓값보다 훨씬 작은 로짓만 0으로 사라지는데, 그런 항은 어차피 합에
  기여하는 바가 무시할 만하다.

## LSE를 이용한 안정적인 소프트맥스

두 아이디어를 결합하면 수치적으로 안정한 소프트맥스는

$$
\log\operatorname{softmax}(\mathbf{z})_k
= z_k - \operatorname{LSE}(\mathbf{z})
= (z_k - c) - \log\sum_{j=1}^{C}e^{z_j - c}
$$

이다. 확률 자체가 필요하면 지수화한다.
$\operatorname{softmax}(\mathbf{z})_k = e^{\log\operatorname{softmax}(\mathbf{z})_k}$.
가능한 한 오래 로그 공간에서 작업할수록 안정성이 좋아진다.

## 안정적인 교차엔트로피 손실

참 범주가 $y$인 관측치 하나의 교차엔트로피 손실은

$$
\mathcal{L} = -\log\operatorname{softmax}(\mathbf{z})_y
= -z_y + \operatorname{LSE}(\mathbf{z})
$$

이고, 안정적인 LSE를 쓰면

$$
\mathcal{L} = -(z_y - c) + \log\sum_{j=1}^{C}e^{z_j - c}
$$

가 된다. 이 형태는 큰 $z_j$에 대해 $e^{z_j}$를 직접 계산하는 일이 아예 없다. PyTorch와
TensorFlow 같은 프레임워크는 확률이 아니라 원시 로짓을 받는 `cross_entropy_with_logits` 같은
융합 연산으로 이를 구현한다.

!!! warning "손실함수에 확률을 넘기지 말 것"
    `softmax`를 먼저 계산한 뒤 `log`를 취하면 수치 문제가 되살아난다. 확률이 0으로 언더플로되면
    `log(0)`이 `-inf`가 되기 때문이다. 프레임워크가 제공하는 융합 log-softmax나
    로짓 기반 교차엔트로피 함수를 항상 사용하라.

## 단계별 알고리즘

| 단계 | 연산 | 목적 |
|---|---|---|
| 1 | $c \leftarrow \max_j z_j$ | 이동 상수 찾기 |
| 2 | 모든 $j$에 대해 $s_j \leftarrow z_j - c$ | 로짓을 비양수 범위로 이동 |
| 3 | 모든 $j$에 대해 $e_j \leftarrow e^{s_j}$ | 안전한 지수화($e_j \le 1$) |
| 4 | $S \leftarrow \sum_j e_j$ | 소프트맥스의 분모 |
| 5 | $p_j \leftarrow e_j / S$ | 소프트맥스 확률 |
| 6 | $\operatorname{LSE} \leftarrow c + \log S$ | 손실함수용 |

??? example "예제"
    $C = 3$이고 로짓이 $\mathbf{z} = (1000,\; 1001,\; 999)$라 하자.

    **순진한 계산.** float64에서 $e^{1000}$이 `inf`로 넘쳐 `inf / inf = NaN`이 된다.

    **안정적인 계산.**

    1. $c = 1001$.
    2. 이동된 로짓: $(-1, 0, -2)$.
    3. 지수: $(e^{-1}, e^0, e^{-2}) = (0.3679, 1.0, 0.1353)$.
    4. 합: $S = 1.5032$.
    5. 확률: $(0.2447, 0.6652, 0.0900)$.
    6. $\operatorname{LSE} = 1001 + \log(1.5032) = 1001.4076$.

    모든 연산이 안전한 범위에 머물고, 확률은 정확한 수학적 값과 일치한다.

## 두 항의 LSE(시그모이드)

이항 로지스틱 회귀($C = 2$)에서 LSE는

$$
\log(1 + e^z) = \max(0, z) + \log(1 + e^{-|z|})
$$

로 환원된다. 이 항등식은 지수 $-|z| \le 0$을 유지하여 오버플로를 막는다. 수치 라이브러리에서
**softplus** 함수의 표준 구현이다.

## 요약

| 문제 | 순진한 접근 | 안정적인 접근 |
|---|---|---|
| 소프트맥스 오버플로 | 큰 $z_j$에 대한 $e^{z_j}$ | 먼저 $c = \max z_j$를 뺀다 |
| log-softmax 언더플로 | $p_k \approx 0$일 때 $\log(0)$ | LSE로 로그 공간에서 계산 |
| 교차엔트로피 손실 | 소프트맥스 후 로그 | 융합 `cross_entropy_with_logits` 사용 |
| softplus($C = 2$) | $\log(1 + e^z)$가 넘침 | $\max(0,z) + \log(1 + e^{-|z|})$ |

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.**
수치적 안정성과 로그-합-지수

로짓 $\mathbf{z} = (1000, 1001, 999)^\top$을 생각하자.

**(a)** $e^{1000}$, $e^{1001}$, $e^{999}$를 순진하게 계산하면 표준 64비트 부동소수점에서 왜
오버플로가 나는지 설명하라.

**(b)** 로그-합-지수 기법을 적용하라. $m = \max_k z_k = 1001$을 빼고
$\text{softmax}(\mathbf{z} - m\mathbf{1})$을 계산하라.

**(c)** 로그-합-지수 공식

$$
\log\sum_{k=1}^C e^{z_k} = m + \log\sum_{k=1}^C e^{z_k - m}
$$

을 유도하고, 우변이 왜 수치적으로 안정한지 설명하라.

</div>

??? success "풀이"

    **(a)** 64비트 부동소수점(배정도)에서 표현 가능한 최댓값은 약 $1.8 \times 10^{308}$이며
    이는 $e^{709.8}$에 해당한다. $e^{999} \approx 10^{434}$부터 이미 이 한계를 훨씬 넘으므로
    $e^{1000}$이나 $e^{1001}$을 직접 계산하면 `inf`가 되고, 소프트맥스
    $\hat{p}_k = e^{z_k}/\sum e^{z_j}$는 `inf/inf`가 되어 `nan`을 낸다.

    **(b)** $m = 1001$만큼 이동하면

    $$
    \mathbf{z} - m\mathbf{1} = (-1, 0, -2)^\top
    $$

    $$
    e^{-1} \approx 0.368, \quad e^{0} = 1, \quad e^{-2} \approx 0.135
    $$

    $$
    S = 0.368 + 1 + 0.135 = 1.503
    $$

    $$
    \hat{p}_1 = \frac{0.368}{1.503} \approx 0.245, \quad \hat{p}_2 = \frac{1}{1.503} \approx 0.665, \quad \hat{p}_3 = \frac{0.135}{1.503} \approx 0.090
    $$

    모든 계산이 작은 수의 지수만 다루므로 오버플로가 없다.

    **(c)** 정의에서 출발하면

    $$
    \log\sum_{k=1}^C e^{z_k} = \log\sum_{k=1}^C e^{(z_k - m) + m} = \log\left(e^m \sum_{k=1}^C e^{z_k - m}\right)
    $$

    $$
    = m + \log\sum_{k=1}^C e^{z_k - m}
    $$

    이다. 우변이 안정적인 이유는 다음과 같다. (1) 합에서 가장 큰 항이
    $e^{z_{\max} - m} = e^0 = 1$이므로 어떤 항도 넘치지 않는다. (2) 나머지 항은
    $e^{\text{음수}} < 1$이므로 0으로 언더플로되어도 무해하다(그 항이 무시할 만하다는 뜻일
    뿐이다). (3) 따라서 로그 안의 합이 최소 1이므로 $\log$ 값이 0 이상이고, 최종 결과
    $m + (\text{0 이상})$이 잘 정의된다.

<div class="drillbox" markdown>

**연습문제 2.**
$\operatorname{LSE}(\mathbf{z})$의 기울기가 $\operatorname{softmax}(\mathbf{z})$임을 보여라.
이 사실이 왜 유용한가?

</div>

??? success "풀이"

    $\operatorname{LSE}(\mathbf{z}) = \log\sum_j e^{z_j}$를 $z_k$로 미분하면

    $$
    \frac{\partial}{\partial z_k}\operatorname{LSE}(\mathbf{z})
    = \frac{e^{z_k}}{\sum_j e^{z_j}} = \operatorname{softmax}(\mathbf{z})_k
    $$

    이다. 즉 $\nabla \operatorname{LSE}(\mathbf{z}) = \operatorname{softmax}(\mathbf{z})$다.

    **유용한 이유 세 가지.**

    1. **기울기 유도가 즉시 나온다.** 교차엔트로피 손실이
       $\mathcal{L} = -z_y + \operatorname{LSE}(\mathbf{z})$이므로
       $\nabla_{\mathbf{z}}\mathcal{L} = -\mathbf{e}_y + \operatorname{softmax}(\mathbf{z})
       = \hat{\mathbf{y}} - \mathbf{y}$가 한 줄로 나온다. 소프트맥스 야코비를 전개할 필요가
       없다.
    2. **볼록성이 따라온다.** LSE의 헤세행렬은
       $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$인데, 임의의 $v$에 대해
       $v^T H v = \mathbb{E}_p[v^2] - (\mathbb{E}_p[v])^2 = \operatorname{Var}_p(v) \ge 0$
       이므로 양반정치다. 따라서 LSE는 볼록이고 교차엔트로피 손실도 볼록이다.
    3. **LSE는 최댓값의 매끄러운 근사다.** 실제로
       $\max_j z_j \le \operatorname{LSE}(\mathbf{z}) \le \max_j z_j + \log C$이므로
       "soft max"라는 이름이 여기에서 온다. 엄밀히 말해 소프트맥스 함수는 최댓값이 아니라
       **arg max의 매끄러운 근사**이고, LSE가 최댓값의 매끄러운 근사다. $\square$

<div class="drillbox" markdown>

**연습문제 3.**
$\max(0, z) + \log(1 + e^{-|z|}) = \log(1 + e^z)$임을 $z > 0$과 $z \le 0$ 두 경우로 나누어
증명하라. $z = 1000$에서 두 식을 각각 계산해 비교하라.

</div>

??? success "풀이"

    **경우 1: $z > 0$.** $\max(0,z) = z$이고 $|z| = z$이므로 좌변은

    $$
    z + \log(1 + e^{-z}) = \log\bigl(e^z(1 + e^{-z})\bigr) = \log(e^z + 1)
    $$

    **경우 2: $z \le 0$.** $\max(0,z) = 0$이고 $|z| = -z$이므로 좌변은

    $$
    0 + \log(1 + e^{z}) = \log(1 + e^z)
    $$

    두 경우 모두 우변과 같다.

    **$z = 1000$에서:**

    - 순진한 식 $\log(1 + e^{1000})$: `np.exp(1000)`이 `inf`가 되고
      `np.log(inf)` $= \infty$가 되어 손실이 `inf`가 된다.
    - 안정적인 식: $\max(0, 1000) + \log(1 + e^{-1000}) = 1000 + \log(1 + 0) = 1000.0$.

    참값은 $\log(1+e^{1000}) = 1000 + \log(1+e^{-1000}) \approx 1000$이므로 안정적인 식이
    정확하다. NumPy에서는 `np.logaddexp(0, z)`가 정확히 이 계산을 수행한다. $\square$

<div class="drillbox" markdown>

**연습문제 4.**
로짓이 $\mathbf{z} = (-1000, -1001, -999)$처럼 **모두 매우 작은** 경우에도 최댓값을 빼는 것이
올바르게 작동하는지 확인하라. 최솟값을 빼면 어떻게 되는가?

</div>

??? success "풀이"

    **최댓값을 뺄 때.** $c = \max_j z_j = -999$이므로 이동된 로짓은 $(-1, -2, 0)$이고 지수는
    $(0.368, 0.135, 1.0)$, 합은 $1.503$, 확률은 $(0.245, 0.090, 0.665)$다. 문제없다.

    최댓값을 빼는 규칙은 로짓의 절대적 크기와 무관하게 작동한다. 이동 후 최댓값이 항상 정확히
    0이 되도록 만들기 때문이다. 로짓이 모두 크든 모두 작든 결과는 같다.

    **최솟값을 빼면.** $c = \min_j z_j = -1001$이면 이동된 로짓이 $(1, 0, 2)$가 되어 이 예에서는
    안전하다. 그러나 일반적으로는 위험하다. 로짓의 범위가 넓으면(예:
    $\mathbf{z} = (0, 800)$) 최솟값을 뺀 결과가 $(0, 800)$ 그대로여서 $e^{800}$이 넘친다.

    **핵심:** 최댓값을 빼는 것은 이동 후 최대 지수를 $e^0 = 1$로 **고정**하므로 언제나 안전하다.
    다른 어떤 선택도 이 보장을 주지 못한다. 언더플로는 여전히 생길 수 있지만, 언더플로되는 항은
    정의상 최댓값보다 $700$ 이상 작아 상대 기여가 $e^{-700} \approx 10^{-304}$이므로 무해하다.
    $\square$

<div class="drillbox" markdown>

**연습문제 5.**
어떤 구현이 `p = softmax(z)`를 계산한 뒤 `loss = -np.log(p[y])`로 손실을 구한다. 이 구현이
실패하는 구체적인 상황을 제시하고, 왜 융합 연산이 필요한지 설명하라.

</div>

??? success "풀이"

    **실패 상황.** $\mathbf{z} = (0, 800)$이고 참 범주가 $y = 0$이라 하자. 안정적인 소프트맥스는

    $$
    p_0 = \frac{e^{0-800}}{e^{-800} + e^{0}} = \frac{e^{-800}}{1 + e^{-800}}
    $$

    을 계산하는데, $e^{-800}$이 배정도에서 정확히 0으로 언더플로되므로 $p_0 = 0.0$이 된다.
    그러면 `-np.log(0.0)` $= \infty$가 되어 손실이 `inf`, 기울기가 `nan`이 된다.

    **참값은 유한하다.**

    $$
    \mathcal{L} = -z_0 + \operatorname{LSE}(\mathbf{z}) = 0 + \bigl(800 + \log(1 + e^{-800})\bigr) \approx 800.0
    $$

    $800$은 매우 큰 손실이지만 완벽히 유한하고, 기울기 $\hat{\mathbf{y}} - \mathbf{y}$도
    $(0-1, 1-0) = (-1, 1)$로 잘 정의된다. 즉 학습이 계속될 수 있다.

    **융합 연산이 필요한 이유:** 문제의 근원은 $\log$와 $\exp$를 **따로** 수행하는 데 있다.
    소프트맥스가 확률을 만들 때 지수화하면서 정보가 소실되고($e^{-800} \to 0$), 그다음 로그가
    그 소실된 값을 복원할 수 없다. 융합 연산은 두 단계를 대수적으로 합쳐
    $\mathcal{L} = -z_y + \operatorname{LSE}(\mathbf{z})$를 직접 계산하므로 중간에 확률을
    거치지 않는다.

    PyTorch의 `F.cross_entropy`, TensorFlow의
    `softmax_cross_entropy_with_logits`, scikit-learn 내부의 `log_logistic`이 모두 이렇게
    구현되어 있다. **모형의 출력층에 소프트맥스를 넣고 손실에 다시 로그를 취하는 것은 흔하지만
    잘못된 패턴이다.** 모형은 로짓을 내보내고, 손실함수가 로짓을 받도록 하라. $\square$
