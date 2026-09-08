# 수열, 극한, 점근

수열과 그 극한 행동은 통계적 추론의 뼈대다. 큰수의 법칙, 중심극한정리, 추정량의 일치성, 그리고 이 책에 나오는 거의 모든 점근 결과가 극한에 관한 진술이다. 이 개념들을 확률적 수열로 끌어올리기 전에, 확률 기계장치가 매달릴 해석학적 골격인 결정론적 수열에 익숙해져야 한다.

## 정의

### 수열의 수렴

$\mathbb{R}$ 안의 수열 $(a_n)_{n \ge 1}$이 $L \in \mathbb{R}$로 **수렴한다**는 것은($a_n \to L$ 또는 $\lim_{n \to \infty} a_n = L$로 쓴다)

$$
\forall\, \varepsilon > 0,\;\; \exists\, N \in \mathbb{N} \text{ such that } n > N \implies |a_n - L| < \varepsilon
$$

이 성립한다는 뜻이다.

수열이 **코시(Cauchy)** 라는 것은 $\forall\, \varepsilon > 0,\, \exists\, N$이 있어 $m, n > N \implies |a_m - a_n| < \varepsilon$이 성립한다는 뜻이다. $\mathbb{R}$에서는 코시임과 수렴함이 동치다(완비성).

### 급수의 수렴

급수 $\sum_{n=1}^\infty a_n$이 $S$로 수렴한다는 것은 부분합 $S_N = \sum_{n=1}^N a_n$이 $S$로 수렴한다는 뜻이다. 판정법으로는 비교, 비율, 근, 적분, 교대급수 판정법이 있다.

### 큰-O와 작은-o

수열(및 $n$의 함수)에 대해,

$$
f(n) = O(g(n)) \;\Longleftrightarrow\; \exists\, C > 0,\, N \text{ s.t. } |f(n)| \le C |g(n)| \text{ for } n > N
$$

$$
f(n) = o(g(n)) \;\Longleftrightarrow\; \lim_{n \to \infty} \frac{f(n)}{g(n)} = 0
$$

$f(n) \sim g(n)$은 $f(n)/g(n) \to 1$을 뜻한다(점근적 동치). 확률적 대응물인 $O_p$와 $o_p$는 제3장에 나온다.

## 설명

### 극한 법칙

$a_n \to L$이고 $b_n \to M$이면 $a_n + b_n \to L + M$, $a_n b_n \to LM$이며, $M \ne 0$일 때 $a_n / b_n \to L/M$이다. 조임 정리: 결국 $a_n \le b_n \le c_n$이고 $a_n, c_n \to L$이면 $b_n \to L$이다. 연속함수는 극한을 보존한다. $g$가 $L$에서 연속이면 $a_n \to L \Rightarrow g(a_n) \to g(L)$이다.

### 책 전체에서 쓰이는 급수

| 급수 | 합 | 등장하는 곳 |
|---|---|---|
| $\sum_{n=0}^\infty r^n = \dfrac{1}{1-r}$, $|r|<1$ | 기하급수 | 기하 / 음이항 확률질량함수 |
| $\sum_{n=0}^\infty \dfrac{x^n}{n!} = e^x$ | 지수급수 | 포아송 확률질량함수, 적률생성함수 |
| $-\sum_{n=1}^\infty \dfrac{(-1)^n x^n}{n} = \ln(1+x)$, $|x|<1$ | 로그급수 | 로그가능도 전개 |
| $\sum_{n=1}^\infty \dfrac{1}{n^s}$은 $s > 1$일 때에 한해 수렴 | $p$-급수 | 꼬리 한계 진단 |

이와 별개로 늘 쓰이는 적분이 가우스 적분 $\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$이며, 표준정규밀도를 정규화한다.

### 테일러 전개

$f$가 $a$에서 충분히 매끄러우면,

$$
f(x) = f(a) + f'(a)(x - a) + \tfrac{1}{2} f''(a)(x - a)^2 + \cdots + \tfrac{1}{k!} f^{(k)}(a)(x - a)^k + R_k(x)
$$

이고 나머지는 $x \to a$일 때 $R_k(x) = o((x - a)^k)$이다. 다음 두 귀결이 반복해서 쓰인다.

- **델타 방법**: $\sqrt{n}(\hat{\theta}_n - \theta) \xrightarrow{d} N(0, \sigma^2)$이고 $g$가 $\theta$에서 미분가능하면 $\sqrt{n}(g(\hat{\theta}_n) - g(\theta)) \xrightarrow{d} N(0, [g'(\theta)]^2 \sigma^2)$이다.
- **적률생성함수를 이용한 중심극한정리 유도**: $M_X(t/\sqrt{n})$을 $0$ 주위로 2차까지 전개하면, 살아남는 항이 분산의 기여분이다.

### 통계에서의 점근 표기

"$\hat{\theta}_n - \theta = O_p(n^{-1/2})$" 같은 진술은 추정오차가 표준적인 $\sqrt{n}$ 속도로 줄어든다는 뜻이며, 이는 일치성 있는 정칙 추정량의 전형적인 속도다. 더 빠른 속도($n^{-1}$)는 초효율이나 경계 문제에서 나타나고, 더 느린 속도($n^{-1/4}$, $\log n$)는 비모수 추정에서 나타난다. 두 추정량이 모두 일치성을 가질 때 이들을 비교하는 방법이 바로 속도를 따지는 것이다.

### 수렴의 여러 모드 (예고)

확률적 수열은 서로 동치가 아닌 여러 의미로 수렴할 수 있으며, 제3장에서 다룬다.

1. **거의 확실한 수렴**: $P(\lim_n X_n = X) = 1$.
2. **확률수렴**: $\forall \varepsilon, P(|X_n - X| > \varepsilon) \to 0$.
3. **분포수렴**: $F_X$의 연속점에서 $F_{X_n}(x) \to F_X(x)$.
4. **$L^p$ 수렴**: $\mathbb{E}|X_n - X|^p \to 0$.

위계는 (1) $\Rightarrow$ (2) $\Rightarrow$ (3)이고 (4) $\Rightarrow$ (2)이다. 역방향 함의는 일반적으로 어느 것도 성립하지 않는다.

## 예제

```python
import math
import numpy as np

# === (1 + 1/n)^n → e ===
ns = [10, 100, 1_000, 10_000, 100_000]
for n in ns:
    approx = (1 + 1/n)**n
    print(f"n={n:>7d}: (1+1/n)^n = {approx:.8f}, error = {abs(approx - math.e):.2e}")

# === Geometric series partial sums ===
r = 0.5
partial = np.cumsum(r ** np.arange(20))
exact = 1 / (1 - r)
print(f"\nGeometric r=0.5: S_19 = {partial[-1]:.10f}, exact = {exact}")

# === Taylor approximation of e^x at x = 0.3 ===
x = 0.3
for k in range(1, 7):
    taylor = sum(x**n / math.factorial(n) for n in range(k + 1))
    print(f"order {k}: {taylor:.8f}, exact: {math.exp(x):.8f}")

# === Demonstrating o(1/n) vs O(1/n) ===
n = np.arange(1, 50)
print("\nlog(n) / n   (o(1)? yes):", (np.log(n) / n)[-1])
print("sin(n) / n^2 (O(1/n^2)):", (np.sin(n) / n**2)[-1])
```

출력:

```
n=     10: (1+1/n)^n = 2.59374246, error = 1.25e-01
n=    100: (1+1/n)^n = 2.70481383, error = 1.35e-02
n=   1000: (1+1/n)^n = 2.71692393, error = 1.36e-03
n=  10000: (1+1/n)^n = 2.71814593, error = 1.36e-04
n= 100000: (1+1/n)^n = 2.71826824, error = 1.36e-05

Geometric r=0.5: S_19 = 1.9999980927, exact = 2.0
order 1: 1.30000000, exact: 1.34985881
order 2: 1.34500000, exact: 1.34985881
order 3: 1.34950000, exact: 1.34985881
order 4: 1.34983750, exact: 1.34985881
order 5: 1.34985775, exact: 1.34985881
order 6: 1.34985876, exact: 1.34985881

log(n) / n   (o(1)? yes): 0.07942490404307401
sin(n) / n^2 (O(1/n^2)): -0.00039723142555579834
```

## 연습문제

**연습문제 1.**
$\varepsilon$–$N$ 정의로부터 직접

$$
\lim_{n \to \infty} \frac{3n + 1}{n + 2} = 3
$$

임을 증명하라.

??? success "풀이"
    계산하면

    $$
    \left| \frac{3n+1}{n+2} - 3 \right| = \left| \frac{3n+1 - 3(n+2)}{n+2} \right| = \frac{5}{n+2}
    $$

    이다. $\varepsilon > 0$이 주어지면 $N = \lceil 5/\varepsilon - 2 \rceil$로 두자. 그러면 모든 $n > N$에 대해

    $$
    \frac{5}{n+2} < \frac{5}{N+2} \le \varepsilon
    $$

    이다. 따라서 $|a_n - 3| < \varepsilon$이다. $\square$

---

**연습문제 2.**
**(a)** $\sum_{k=0}^\infty r^k$이 $|r| < 1$일 때에 한해 수렴함을 보이고 그 합을 구하라.
**(b)** (a)를 이용해 $\displaystyle\sum_{k=1}^\infty \frac{3}{4^k}$을 계산하라.

??? success "풀이"
    (a) 부분합은

    $$
    S_n = \sum_{k=0}^n r^k = \frac{1 - r^{n+1}}{1 - r} \qquad (r \ne 1)
    $$

    이다. $|r| < 1$이면 $r^{n+1} \to 0$이므로 $S_n \to 1/(1 - r)$이다. $|r| \ge 1$이면 $|r^k|$가 0으로 가지 않아 발산 판정법에 걸리므로 급수가 발산한다.

    (b)

    $$
    \sum_{k=1}^\infty \frac{3}{4^k} = 3 \sum_{k=1}^\infty \left(\tfrac{1}{4}\right)^{\!k} = 3 \cdot \frac{1/4}{1 - 1/4} = 3 \cdot \tfrac{1}{3} = 1
    $$

---

**연습문제 3.**
고정된 모든 $x \in \mathbb{R}$에 대해 $n \to \infty$일 때 $(1 + x/n)^n \to e^x$임을 보여라.

??? success "풀이"
    로그를 취한다. 고정된 $x$와, $x/n$이 $\ln(1 + \cdot)$의 정의역에 들어갈 만큼 큰 $n$에 대해,

    $$
    n \ln\!\left(1 + \frac{x}{n}\right) = n \left[\frac{x}{n} - \frac{1}{2}\!\left(\frac{x}{n}\right)^{\!2} + O\!\left(\tfrac{1}{n^3}\right) \right] = x - \frac{x^2}{2n} + O\!\left(\tfrac{1}{n^2}\right)
    $$

    이다. 여기서 $u = 0$ 주위의 $\ln(1 + u)$의 테일러 전개를 썼다. 우변은 $x$로 수렴하고 $\exp$는 연속이므로

    $$
    (1 + x/n)^n = \exp\!\left(n \ln(1 + x/n)\right) \to e^x
    $$

    이다. $\square$

---

**연습문제 4.**
조임 정리를 증명하라: 충분히 큰 모든 $n$에 대해 $a_n \le b_n \le c_n$이고 $a_n, c_n \to L$이면 $b_n \to L$이다.

??? success "풀이"
    $\varepsilon > 0$이라 하자. $n > N_1 \Rightarrow |a_n - L| < \varepsilon$인 $N_1$과 $n > N_2 \Rightarrow |c_n - L| < \varepsilon$인 $N_2$를 고른다. $a_n \le b_n \le c_n$이 성립하기 시작하는 경계를 $N_3$이라 하고 $N = \max(N_1, N_2, N_3)$으로 두자.

    $n > N$에 대해

    $$
    L - \varepsilon < a_n \le b_n \le c_n < L + \varepsilon
    $$

    이므로 $|b_n - L| < \varepsilon$이다. $\square$

---

**연습문제 5.**
테일러 전개를 이용해, $X \sim \mathrm{Bernoulli}(p)$이고 표본평균이 $\bar{X}_n$일 때 분산안정화 변환 $g(p) = 2\arcsin(\sqrt{p})$가

$$
\sqrt{n}\!\left(g(\bar{X}_n) - g(p)\right) \xrightarrow{d} N(0, 1)
$$

을 만족함을 보여라. 따라서 $g(\bar{X}_n)$은 $p$와 무관하게 근사적으로 일정한 분산 $1/n$을 갖는다.

??? success "풀이"
    중심극한정리에 의해 $\sqrt{n}(\bar{X}_n - p) \xrightarrow{d} N(0, p(1 - p))$이다. 델타 방법을 쓰면

    $$
    \sqrt{n}\!\left(g(\bar{X}_n) - g(p)\right) \xrightarrow{d} N\!\left(0, [g'(p)]^2 \, p(1 - p)\right)
    $$

    이다. $g(p) = 2 \arcsin(\sqrt{p})$를 미분하면

    $$
    g'(p) = 2 \cdot \frac{1}{\sqrt{1 - p}} \cdot \frac{1}{2\sqrt{p}} = \frac{1}{\sqrt{p(1-p)}}
    $$

    이다. 따라서 모든 $p \in (0, 1)$에 대해 $[g'(p)]^2 \cdot p(1-p) = 1$이므로 극한은 $N(0, 1)$이다. $\square$

---

**연습문제 6.**
큰-O가 대칭이 아님을 보여라. 즉 $f(n) = O(g(n))$이지만 $g(n) \ne O(f(n))$인 수열을 제시하라. 그다음 "$f \asymp g$"($f = O(g)$ **그리고** $g = O(f)$를 뜻함)로 정의되는 자연스러운 동치관계를 진술하고, $f \asymp g$이지만 $f \not\sim g$인 두 수열의 예를 들어라.

??? success "풀이"
    **비대칭 예:** $f(n) = 1$, $g(n) = n$. 그러면 $f(n) = O(g(n))$이지만($C = 1$로 두면 된다) $g(n)/f(n) = n \to \infty$이므로 $g(n) \ne O(f(n))$이다.

    **동치관계:** $f \asymp g$일 필요충분조건은 어떤 $0 < c_1 \le c_2 < \infty$와 $N$이 있어 $n > N$에 대해 $c_1 |g(n)| \le |f(n)| \le c_2 |g(n)|$이 성립하는 것이다. 이 관계는 반사적이고 (정의상) 대칭이며 추이적이다. "같은 증가 차수"를 포착한다.

    **$f \asymp g$이지만 $f \not\sim g$:** $f(n) = n$, $g(n) = 2n$을 잡자. $c_1 = 1/2$, $c_2 = 2$가 통하므로 $f \asymp g$이다. 그러나 $f(n)/g(n) = 1/2 \ne 1$이므로 점근적으로 동치는 아니다. $\square$
