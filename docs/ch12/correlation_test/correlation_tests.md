# 상관 검정 개관

이 절에서는 두 변수 사이 관계의 유의성을 평가하는 데 널리 쓰이는 세 가지 통계 검정을 다룬다: Pearson 상관, Spearman 순위상관, Kendall의 타우.

---

## 준비: 공통 자료 생성

다음 모듈들은 서로 다른 상관 상황을 보이기 위해 세 가지 유형의 자료를 생성한다.

### `global_name_space.py`

<div class="codebox" markdown>

#### 예제 1. 설정 모듈 { .eg }

```python
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Correlation Test Examples')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
# parse_args()가 아니라 parse_known_args()를 쓴다.
# 노트북이나 REPL에서는 sys.argv에 다른 인자가 들어 있어 parse_args()가
# SystemExit을 던지며 커널을 멈춰 세운다.
ARGS, _ = parser.parse_known_args()

np.random.seed(ARGS.seed)
ARGS.size = 1000
```

</div>

### `load_data.py`

<div class="codebox" markdown>

#### 예제 2. 자료 적재 모듈 { .eg }

```python
import numpy as np

# 위의 global_name_space.py를 파일로 저장했다면 다음 한 줄로 대신할 수 있다.
#   from global_name_space import ARGS

def load_data(data_type=0):
    data_dict = {}

    x = np.random.rand(ARGS.size) * 20
    eps = np.random.rand(ARGS.size) * 10

    # 자료 0: 관계 없음
    y = np.random.rand(ARGS.size) * 20
    data_dict[0] = (x, y)

    # 자료 1: 단조이지만 곡선인 관계(삼차)
    y = (x + eps) ** 3
    data_dict[1] = (x, y)

    # 자료 2: 단조가 아닌 관계(사인)
    y = np.sin(x + eps)
    data_dict[2] = (x, y)

    return data_dict
```

세 자료가 나타내는 것:

- **자료 0**: 관계 없음 — 무작위 산포. Pearson과 순위 기반 상관 모두 0에 가까워야 한다.
- **자료 1**: 단조 비선형 — Pearson은 선형성을 재므로 관계를 과소평가할 수 있지만 (단조성을 재는) Spearman과 Kendall은 이를 탐지해야 한다.
- **자료 2**: 비단조(사인) — 관계가 주기적이고 단조도 선형도 아니므로 모든 상관 측도가 약해야 한다.

</div>

---

## Pearson 상관 검정

[문서: `scipy.stats.pearsonr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)

Pearson의 $r$은 두 변수 사이의 **선형** 관계를 잰다. 귀무가설 $H_0: \rho = 0$ 아래에서 검정통계량은 자유도 $n-2$인 $t$-분포를 따른다.

<div class="codebox" markdown>

### 예제 3. Pearson 상관 검정 { .eg }

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# 위의 load_data.py를 파일로 저장했다면 다음 한 줄로 대신할 수 있다.
#   from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.pearsonr(x, y)
        ax.set_title(f"Pearson's r: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

![Pearson 상관: 세 자료](./img/correlation_tests_72.png)

왼쪽부터 무관계, 단조 관계, 사인 관계다. Pearson은 가운데에서만 큰 값을 준다. 오른쪽 사인 자료는 눈으로는 뚜렷한 구조가 있지만 $r$이 0 근처다.

**언제 쓰는가**: 두 변수가 모두 연속형이고 **선형** 관계를 예상할 때. Pearson의 $r$은 이상점에 민감하며 p-값이 정확하려면 이변량 정규성을 가정한다.

</div>

---

## Spearman 순위상관 검정

[영상: Spearman's Rank Correlation](https://www.youtube.com/watch?v=YpG2MlulP_o) |
[문서: `scipy.stats.spearmanr`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)

Spearman의 $\rho_s$는 두 변수 사이의 **단조** 관계를 잰다. 원자료 값이 아니라 순위에 Pearson의 $r$을 적용하여 계산한다. 그래서 이상점에 로버스트하고 비선형이지만 단조인 관계에도 적용할 수 있다.

<div class="codebox" markdown>

### 예제 4. Spearman 순위상관 검정 { .eg }

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# 위의 load_data.py를 파일로 저장했다면 다음 한 줄로 대신할 수 있다.
#   from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.spearmanr(x, y)
        ax.set_title(f"Spearman rho: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

![Spearman 순위상관: 세 자료](./img/correlation_tests_105.png)

단조 자료에서 Spearman이 Pearson보다 높은 값을 준다. 사인 자료에서는 둘 다 0 근처인데, 관계가 단조가 아니어서 순위로 바꾸는 것도 도움이 되지 않기 때문이다.

**언제 쓰는가**: 관계가 단조일 수 있으나 반드시 선형은 아닐 때, 또는 자료에 이상점이 있거나 순서형일 때.

</div>

---

## Kendall의 타우

[영상 1: Kendall's Tau Explained](https://www.youtube.com/watch?v=oXVxaSoY94k) |
[영상 2: Kendall's Tau Calculation](https://www.youtube.com/watch?v=V4MgE43SrgM) |
[문서: `scipy.stats.kendalltau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)

Kendall의 $\tau$도 **단조** 관계의 강도를 재지만 순위가 아니라 일치쌍과 불일치쌍의 수에 기반한다. 표본이 작을 때 더 로버스트한 편이고 가설검정에 좋은 통계적 성질을 갖는다.

$$
\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\binom{n}{2}}
$$

<div class="codebox" markdown>

### 예제 5. Kendall의 타우 검정 { .eg }

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# 위의 load_data.py를 파일로 저장했다면 다음 한 줄로 대신할 수 있다.
#   from load_data import load_data

def main():
    data_dict = load_data()
    _, axes = plt.subplots(1, len(data_dict), figsize=(12, 3))

    for ax, (x, y) in zip(axes, data_dict.values()):
        ax.plot(x, y, ".k")
        coef, p_val = stats.kendalltau(x, y)
        ax.set_title(f"Kendall's tau: {coef:.4f}\np-value: {p_val:.4f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
```

![Kendall의 타우: 세 자료](./img/correlation_tests_143.png)

Kendall의 $\tau$는 세 자료 모두에서 Spearman과 같은 방향을 가리키되 절댓값이 작다. 척도가 다르기 때문이며, 두 계수를 직접 비교하면 안 된다.

**언제 쓰는가**: Spearman의 $\rho_s$와 비슷한 상황이지만, 표본이 작거나 쌍별 일치에 기반한 더 해석하기 쉬운 측도를 원할 때 선호된다.

</div>

---

## 세 검정의 비교

| 항목 | Pearson의 $r$ | Spearman의 $\rho_s$ | Kendall의 $\tau$ |
|---------|--------------|-------------------|----------------|
| 재는 것 | 선형 연관 | 단조 연관 | 단조 연관 |
| 자료 유형 | 연속형 | 연속형 또는 순서형 | 연속형 또는 순서형 |
| 이상점에 대한 민감성 | 높음 | 낮음 | 낮음 |
| 가정 | 이변량 정규성(정확한 p-값을 위해) | 없음(순위 기반) | 없음(순위 기반) |
| 범위 | $[-1, 1]$ | $[-1, 1]$ | $[-1, 1]$ |
| 작은 표본 | 덜 믿을 만함 | 보통 | 더 믿을 만함 |

---

<div class="codebox" markdown>

### 예제 6. 나이와 소득 { .eg }

**문제**: $\alpha = 0.05$에서 나이와 소득이 관련되어 있는지 검정하라.

```
age    = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
income = [15000, 29000, 68000, 52000, 32000, 80000, 41000, 45000, 26000, 33000]
```

#### 풀이

세 검정 모두 $p \approx 0.0000$을 주어 나이와 소득 사이에 강하고 통계적으로 유의한 관계가 있음을 나타낸다.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    x = [18, 25, 57, 45, 26, 64, 37, 40, 24, 33]
    y = [15_000, 29_000, 68_000, 52_000, 32_000, 80_000, 41_000, 45_000, 26_000, 33_000]

    coef, p_val = stats.pearsonr(x, y)
    print(f"Pearson's r:   coef = {coef:.4f},  p-value = {p_val:.4f}")

    coef, p_val = stats.spearmanr(x, y)
    print(f"Spearman rho:  coef = {coef:.4f},  p-value = {p_val:.4f}")

    coef, p_val = stats.kendalltau(x, y)
    print(f"Kendall's tau: coef = {coef:.4f},  p-value = {p_val:.4f}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(x, y, "ok")
    ax.set_xlabel("Age")
    ax.set_ylabel("Income")
    plt.show()

if __name__ == "__main__":
    main()
```

출력:

```
Pearson's r:   coef = 0.9923,  p-value = 0.0000
Spearman rho:  coef = 1.0000,  p-value = 0.0000
Kendall's tau: coef = 1.0000,  p-value = 0.0000
```

![세 검정의 비교](./img/correlation_tests_195.png)

지수 관계라 단조이지만 선형은 아니다. 순위만 보는 Spearman과 Kendall이 정확히 1.0을 주는 반면 Pearson은 0.9923에 그친다.

**해석**: 모든 p-값이 $\alpha = 0.05$보다 훨씬 작으므로 $H_0: \rho = 0$을 기각하고 이 표본에서 나이와 소득 사이에 통계적으로 유의한 양의 관계가 있다고 결론짓는다. 다만 이것이 인과관계를 확립하지는 않는다. 경력, 학력, 업종 같은 교란요인이 두 변수 모두에 영향을 줄 수 있다.

</div>

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
$n = 25$인 표본에서 Pearson 상관이 $r = 0.42$이다. 상관에 대한 $t$-검정으로 $\alpha = 0.05$에서 귀무가설 $H_0: \rho = 0$을 검정하라.

</div>

??? success "풀이"
    검정통계량은

    $$
    t = \frac{r\sqrt{n-2}}{\sqrt{1-r^2}} = \frac{0.42\sqrt{23}}{\sqrt{1-0.1764}} = \frac{0.42 \times 4.796}{\sqrt{0.8236}} = \frac{2.014}{0.9075} = 2.219
    $$

    이다. $df = n - 2 = 23$에서 $\alpha = 0.05$ 양측검정의 임계값은 $t_{0.025, 23} \approx 2.069$이다.

    $|t| = 2.219 > 2.069$이므로 $H_0$을 기각하고 5% 수준에서 두 변수 사이에 통계적으로 유의한 선형관계가 있다고 결론짓는다.

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$H_0: \rho = 0$을 검정하는 것과 $\rho_0 \neq 0$인 $H_0: \rho = \rho_0$을 검정하는 것의 차이를 설명하라. 두 번째 검정에 왜 Fisher의 $z$ 변환이 필요한가?

</div>

??? success "풀이"
    $\rho = 0$이면 $r$의 표본분포가 대칭이고 $t$-통계량 $r\sqrt{(n-2)/(1-r^2)}$이 자유도 $n-2$인 $t$-분포를 정확히 따른다.

    $\rho \neq 0$이면 $r$의 표본분포가 (특히 $|\rho|$가 1에 가까울 때) **치우쳐** 있으므로 $t$-검정이 더 이상 타당하지 않다. Fisher의 $z$ 변환

    $$
    z = \frac{1}{2}\ln\frac{1+r}{1-r} = \text{arctanh}(r)
    $$

    은 $r$을 평균이 $\text{arctanh}(\rho)$이고 표준오차가 $1/\sqrt{n-3}$인 근사적 정규분포로 바꾸어, 임의의 $\rho_0$에 대한 타당한 가설검정과 신뢰구간을 가능하게 한다.

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
크기가 $n_1 = 40$, $n_2 = 35$인 두 독립 표본에서 Pearson 상관 $r_1 = 0.55$, $r_2 = 0.30$을 얻었다. $\alpha = 0.05$에서 두 모상관이 같은지 검정하라.

</div>

??? success "풀이"
    각각에 Fisher의 $z$ 변환을 적용한다:

    $$
    z_1 = \text{arctanh}(0.55) = 0.6184, \quad z_2 = \text{arctanh}(0.30) = 0.3095
    $$

    독립인 두 상관을 비교하는 검정통계량은

    $$
    Z = \frac{z_1 - z_2}{\sqrt{\frac{1}{n_1-3} + \frac{1}{n_2-3}}} = \frac{0.6184 - 0.3095}{\sqrt{\frac{1}{37} + \frac{1}{32}}} = \frac{0.3089}{\sqrt{0.02703 + 0.03125}} = \frac{0.3089}{\sqrt{0.05828}} = \frac{0.3089}{0.2414} = 1.28
    $$

    이다. $\alpha = 0.05$ 양측검정의 임계값은 $z_{0.025} = 1.96$이다. $|Z| = 1.28 < 1.96$이므로 $H_0$을 기각하지 못한다. 두 모상관이 다르다고 결론지을 증거가 충분하지 않다.

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
$H_0:\rho=0$의 $t$ 검정은 **정규성을 요구하는가?** 여러 분포에서 1종 오류를 확인하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(30001)
    B = 20_000
    gens = {
        "정규": lambda n: rng.standard_normal(n),
        "균등": lambda n: rng.uniform(-1, 1, n),
        "지수 (왜도 2)": lambda n: rng.exponential(1, n),
        "로그정규 (왜도 6)": lambda n: np.exp(rng.standard_normal(n)),
        "t(3) 두꺼운 꼬리": lambda n: rng.standard_t(3, n),
        "코시": lambda n: rng.standard_cauchy(n),
    }
    print("두 변수가 독립일 때의 1종 오류 (명목 0.05)")
    print(f"{'분포':>22s} {'n=10':>8s} {'n=30':>8s} {'n=100':>8s}")
    for lab, g in gens.items():
        row = []
        for n in [10, 30, 100]:
            a = sum(stats.pearsonr(g(n), g(n)).pvalue < 0.05 for _ in range(B))
            row.append(a / B)
        print(f"{lab:>22s} {row[0]:8.4f} {row[1]:8.4f} {row[2]:8.4f}")
    ```

    ```text
    두 변수가 독립일 때의 1종 오류 (명목 0.05)
                        분포     n=10     n=30    n=100
                        정규   0.0498   0.0483   0.0500
                        균등   0.0498   0.0500   0.0534
                 지수 (왜도 2)   0.0483   0.0530   0.0522
               로그정규 (왜도 6)   0.0578   0.0530   0.0451
               t(3) 두꺼운 꼬리   0.0474   0.0525   0.0505
                        코시   0.0771   0.0641   0.0470
    ```

    **놀랍도록 견고하다.** 왜도 6의 로그정규, 분산이 무한한 코시에서도 오류율이 거의 0.05다.

    | 분포 | $n=10$ | $n=100$ |
    |---|---|---|
    | 정규 | 0.050 | 0.050 |
    | 로그정규(왜도 6) | 0.058 | **0.045** |
    | **코시**(평균도 없음) | **0.077** | **0.047** |

    **코시의 $n=10$에서 0.077**이 유일하게 눈에 띄는 이탈이고, $n=100$이면 0.047로 회복된다.

    **왜 견고한가.** $H_0:\rho=0$ **그리고 두 변수가 독립**이면, $Y$를 어떻게 뒤섞어도 분포가 같다. 즉 **교환가능성**이 성립한다.

    ```text
    독립 + H0 → (x_i, y_σ(i)) 의 분포가 모든 순열 σ 에서 동일
             → r 의 귀무분포가 주변분포에 (거의) 의존하지 않는다
             → t 근사가 잘 맞는다
    ```

    **중요한 단서 — "독립"과 "$\rho=0$"은 다르다.**

    | 귀무가설 | $t$ 검정이 타당한가 |
    |---|---|
    | $X\perp Y$(완전 독립) | **그렇다**(분포 무관) |
    | $\rho=0$**이지만 종속** | **아니다** |

    **두 번째의 예.** $Y$의 **분산**이 $X$에 의존하면(이분산) $\rho=0$이어도 $r$의 분산이 공식과 다르다.

    **그러면 "정규성 가정"은 무엇에 필요한가.**

    | 목적 | 정규성 필요 |
    |---|---|
    | $H_0:\rho=0$ 검정 | **거의 불필요** |
    | $H_0:\rho=\rho_0$($\rho_0\neq0$) | **필요** |
    | **신뢰구간** | **필요**(다음 문제) |
    | $r$의 해석 | 필요(선형성 가정) |

    **이 구분이 실무에서 자주 흐려진다.** "자료가 정규가 아니니 스피어만을 쓰자"는 결정은, **$H_0:\rho=0$만 검정한다면 근거가 약하다.** 스피어만을 쓸 이유는 **이상점과 비선형**이지 비정규성 자체가 아니다.

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff hard" title="어려움"></span>
그렇다면 **신뢰구간**도 견고한가? $\rho=0.6$에서 확인하라.

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(30002)

    def gen(kind, n, rho=0.6):
        """가우스 코퓰러로 주변분포만 바꾼다."""
        z = rng.standard_normal((n, 2))
        x = z[:, 0]
        y = rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]
        if kind == "정규":
            return x, y
        if kind == "로그정규":
            return np.exp(x), np.exp(y)
        df = 5 if kind.startswith("t(5)") else 3
        return (stats.t.ppf(stats.norm.cdf(x), df),
                stats.t.ppf(stats.norm.cdf(y), df))

    B = 10_000
    print("참 ρ=0.6 의 피셔 z 구간, 명목 95%")
    print(f"{'분포':>16s} {'n=30':>9s} {'n=100':>9s} {'n=500':>9s} {'참 r':>9s}")
    for kind in ["정규", "로그정규", "t(5) 주변화", "t(3) 주변화"]:
        bx, by = gen(kind, 400_000)
        true = np.corrcoef(bx, by)[0, 1]
        row = []
        for n in [30, 100, 500]:
            a = 0
            for _ in range(B):
                x, y = gen(kind, n)
                r = np.corrcoef(x, y)[0, 1]
                q = 1.96 / np.sqrt(n - 3)
                a += (np.tanh(np.arctanh(r) - q) <= true
                      <= np.tanh(np.arctanh(r) + q))
            row.append(a / B)
        print(f"{kind:>16s} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {true:9.4f}")
    ```

    ```text
    참 ρ=0.6 의 피셔 z 구간, 명목 95%
                  분포      n=30     n=100     n=500       참 r
                  정규    0.9469    0.9486    0.9510    0.5988
                로그정규    0.7965    0.7332    0.6482    0.4765
            t(5) 주변화    0.9394    0.9380    0.9295    0.5862
            t(3) 주변화    0.9097    0.8697    0.7772    0.5372
    ```

    **구간은 전혀 견고하지 않다.** 검정과 정반대다.

    | 분포 | $n=30$ | $n=500$ |
    |---|---|---|
    | 정규 | 0.947 | 0.951 ✓ |
    | **로그정규** | 0.797 | **0.648** |
    | t(5) | 0.939 | 0.930 |
    | **t(3)** | 0.910 | **0.777** |

    **표본이 커질수록 나빠진다.** 로그정규에서 $0.797\to0.648$이다.

    **이것이 요점이다.** 표본이 커지면 $r$의 분산이 줄어드는데, **피셔 $z$의 $\operatorname{SE}=1/\sqrt{n-3}$이 실제 분산을 계속 과소평가**하므로 구간이 참값을 점점 덜 덮는다.

    **왜 검정은 괜찮고 구간은 아닌가.**

    | | $H_0:\rho=0$ | 구간($\rho\neq0$) |
    |---|---|---|
    | 귀무분포 | **주변분포와 무관** | 4차 적률에 의존 |
    | 필요 가정 | 독립(교환가능성) | **이변량 정규성** |

    **정확한 점근 분산**(비정규 포함)은

    $$
    \operatorname{Var}(r)\approx\frac{1}{n}\left[(1-\rho^2)^2+\rho^2\cdot(\text{초과첨도 항})\right]
    $$

    로, **꼬리가 두꺼우면 훨씬 커진다.**

    **부트스트랩이 고치는가.**

    ```python
    rng = np.random.default_rng(30003)

    def lognorm_pair(n, rho=0.6):
        z = rng.standard_normal((n, 2))
        return (np.exp(z[:, 0]),
                np.exp(rho * z[:, 0] + np.sqrt(1 - rho**2) * z[:, 1]))

    bx, by = lognorm_pair(400_000)
    true = np.corrcoef(bx, by)[0, 1]
    print(f"\n로그정규, 참 상관 = {true:.4f}")
    B, R = 2_000, 999
    for n in [30, 100]:
        a = b = 0
        for _ in range(B):
            x, y = lognorm_pair(n)
            r = np.corrcoef(x, y)[0, 1]
            q = 1.96 / np.sqrt(n - 3)
            a += (np.tanh(np.arctanh(r) - q) <= true
                  <= np.tanh(np.arctanh(r) + q))
            idx = rng.integers(0, n, (R, n))
            bs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            b += lo <= true <= hi
        print(f"  n={n:3d}: 피셔 z {a / B:.4f},  부트스트랩 백분위 {b / B:.4f}")
    ```

    ```text

    로그정규, 참 상관 = 0.4764
      n= 30: 피셔 z 0.7870,  부트스트랩 백분위 0.9110
      n=100: 피셔 z 0.7290,  부트스트랩 백분위 0.9135
    ```

    **부트스트랩이 0.73에서 0.91로 크게 개선**한다. 완벽하지는 않지만 실용적이다.

    **BCa 부트스트랩**을 쓰면 더 나아진다. 편향과 왜도를 보정하기 때문이다.

    **권고 셋.**

    1. **$\rho=0$ 검정만 한다면** 피셔 $z$나 $t$ 검정으로 충분하다.
    2. **구간을 보고한다면** 이변량 정규성을 확인하거나 **부트스트랩**을 쓴다.
    3. **꼬리가 두꺼우면 순위상관**을 함께 보고한다. 변환에 불변이라 이 문제가 없다.

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
**"상관이 없다"**를 주장하려면 어떻게 하는가? 등가성 검정을 설계하라.

</div>

??? success "풀이"
    **$p>0.05$는 "상관이 없다"의 근거가 아니다.** 무증거일 뿐이다.

    **등가성 검정(TOST).** 등가 한계 $\pm\delta$를 정하고 **두 개의 단측 검정**을 한다.

    $$
    H_{01}:\rho\leq-\delta
    \quad\text{와}\quad
    H_{02}:\rho\geq+\delta
    $$

    **둘 다 기각되면** $\lvert\rho\rvert<\delta$라고 주장할 수 있다.

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    def tost(r, n, delta=0.2):
        """상관의 두 단측 검정. 반환값이 작으면 등가."""
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        p1 = stats.norm.sf((z - np.arctanh(-delta)) / se)
        p2 = stats.norm.cdf((z - np.arctanh(delta)) / se)
        return max(p1, p2)

    rng = np.random.default_rng(30003)
    B = 10_000
    print("등가 한계 ±0.2, 명목 0.05. 표는 '등가'라고 결론 내릴 확률")
    print(f"{'참 ρ':>7s} {'n=50':>9s} {'n=200':>9s} {'n=500':>9s} {'n=1000':>9s}")
    for rho in [0.0, 0.1, 0.2, 0.3]:
        row = []
        for n in [50, 200, 500, 1000]:
            a = 0
            for _ in range(B):
                z = rng.standard_normal((n, 2))
                r = np.corrcoef(z[:, 0], rho * z[:, 0]
                                + np.sqrt(1 - rho**2) * z[:, 1])[0, 1]
                a += tost(r, n) < 0.05
            row.append(a / B)
        print(f"{rho:7.1f} {row[0]:9.4f} {row[1]:9.4f} {row[2]:9.4f} {row[3]:9.4f}")
    ```

    ```text
    등가 한계 ±0.2, 명목 0.05. 표는 '등가'라고 결론 내릴 확률
        참 ρ      n=50     n=200     n=500    n=1000
        0.0    0.0000    0.7707    0.9966    1.0000
        0.1    0.0000    0.4097    0.7422    0.9411
        0.2    0.0000    0.0454    0.0507    0.0470
        0.3    0.0000    0.0007    0.0000    0.0000
    ```

    **$n=50$에서는 무슨 수를 써도 등가를 주장할 수 없다.** 참 $\rho$가 정확히 0이어도 확률이 0.000이다.

    | $n$ | $\rho=0$에서 등가 결론 확률 |
    |---|---|
    | 50 | **0.000** |
    | 200 | 0.771 |
    | 500 | 0.997 |
    | 1000 | 1.000 |

    **$\rho=0.2$(경계)에서 0.045~0.051**이다. 1종 오류가 올바르게 통제된다.

    **$n=50$이 왜 불가능한가.** 구간의 폭이

    $$
    \pm\frac{1.645}{\sqrt{47}}=\pm0.24
    $$

    ($z$ 척도)로, **등가 한계 $\operatorname{arctanh}(0.2)=0.203$보다 넓다.** 구간이 $[-\delta,\delta]$ 안에 들어갈 수 없다.

    **필요 표본의 대략적 조건.**

    $$
    \frac{z_\alpha}{\sqrt{n-3}}<\operatorname{arctanh}\delta
    \quad\Longrightarrow\quad
    n>3+\left(\frac{z_\alpha}{\operatorname{arctanh}\delta}\right)^2
    $$

    | $\delta$ | 최소 $n$ | 검정력 0.8에 필요한 $n$($\rho=0$) |
    |---|---|---|
    | 0.1 | 269 | 약 780 |
    | **0.2** | **69** | **약 210** |
    | 0.3 | 32 | 약 95 |

    **최소 $n$과 실용적 $n$이 3배 차이**난다. 최소치는 "$\hat r=0$이 나왔을 때만" 가능한 값이다.

    **동등한 방법 — 신뢰구간 포함 관계.**

    ```text
    90% 신뢰구간이 [-δ, +δ] 안에 완전히 들어가면 등가
      (양측 90% = 단측 5% 두 개)

    이것이 TOST 와 정확히 같은 결정을 준다
    ```

    **실무 지침 넷.**

    1. **$\delta$를 자료를 보기 전에** 정한다. 분야의 관례나 실질적 무시 가능 수준으로.
    2. **"유의하지 않음"과 "등가"를 구분**해 보고한다.
    3. **셋 중 하나의 결론**이 나온다: 유의한 상관 / 등가 / **결론 유보**.
    4. **결론 유보가 가장 흔하다.** 표본이 작으면 언제나 그렇다.

    **보고 형식.**

    ```text
    r = 0.04,  95% CI [-0.10, 0.18],  n = 200
    등가 검정 (δ = 0.2): p = 0.021  → 등가

    "무시 가능한 수준(|ρ| < 0.2) 이라고 결론지을 수 있다"

    ── 반면 n = 50 이었다면 ──
    r = 0.04,  95% CI [-0.24, 0.31]
    등가 검정: p = 0.14  → 결론 유보
    "상관이 없다고 말할 근거가 없다"
    ```

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
상관 검정의 **검정력과 표본 크기**를 정리하라.

</div>

??? success "풀이"
    **필요 표본 크기**($H_0:\rho=0$, 양측 $\alpha$, 검정력 $1-\beta$).

    $$
    n=\left(\frac{z_{\alpha/2}+z_\beta}{\operatorname{arctanh}\rho}\right)^2+3
    $$

    ```python
    import numpy as np
    from scipy import stats

    za = stats.norm.ppf(0.975)
    print("H0: ρ=0, 양측 0.05")
    print(f"{'ρ':>6s} {'검정력 0.80':>11s} {'0.90':>8s} {'0.95':>8s} {'0.99':>8s}")
    for rho in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        row = []
        for pw in [0.80, 0.90, 0.95, 0.99]:
            zb = stats.norm.ppf(pw)
            row.append(int(np.ceil(((za + zb) / np.arctanh(rho))**2 + 3)))
        print(f"{rho:6.1f} {row[0]:11d} {row[1]:8d} {row[2]:8d} {row[3]:8d}")

    print("\n주어진 n 에서 검출 가능한 최소 상관 (검정력 0.80)")
    zb = stats.norm.ppf(0.80)
    print(f"{'n':>6s} {'최소 ρ':>9s}")
    for n in [20, 30, 50, 100, 200, 500, 1000]:
        print(f"{n:6d} {np.tanh((za + zb) / np.sqrt(n - 3)):9.4f}")
    ```

    ```text
    H0: ρ=0, 양측 0.05
         ρ    검정력 0.80     0.90     0.95     0.99
       0.1         783     1047     1294     1828
       0.2         194      259      320      451
       0.3          85      113      139      195
       0.5          30       38       47       64
       0.7          14       17       21       28
       0.9           7        8        9       12

    주어진 n 에서 검출 가능한 최소 상관 (검정력 0.80)
         n      최소 ρ
        20    0.5912
        30    0.4924
        50    0.3873
       100    0.2770
       200    0.1970
       500    0.1250
      1000    0.0885
    ```


    **$n=20$이면 $\rho=0.59$ 이상만 안정적으로 검출**된다.

    | $n$ | 검출 가능한 최소 $\rho$ |
    |---|---|
    | 20 | **0.59** |
    | 50 | 0.39 |
    | **100** | **0.28** |
    | 500 | 0.13 |

    **심리·사회과학의 전형적 효과($\rho\approx0.2$)를 잡으려면 194명**이 필요하다.

    **실제 연구의 표본이 이보다 훨씬 작은 경우가 많다.** 그 결과가 **재현성 위기**의 통계적 측면이다.

    **승자의 저주를 수치로 확인한다.**

    ```python
    rng = np.random.default_rng(31001)
    B = 20_000
    print("유의한 연구만 보면 r 이 얼마나 부풀려지나")
    print(f"{'참 ρ':>6s} {'n':>5s} {'유의 확률':>9s} "
          f"{'유의한 경우 평균 |r|':>19s} {'부풀림 배율':>11s}")
    for rho, n in [(0.2, 40), (0.2, 100), (0.2, 200), (0.1, 40), (0.5, 20)]:
        rs, sig = [], 0
        for _ in range(B):
            z = rng.standard_normal((n, 2))
            r = np.corrcoef(z[:, 0], rho * z[:, 0]
                            + np.sqrt(1 - rho**2) * z[:, 1])[0, 1]
            t = r * np.sqrt((n - 2) / (1 - r**2))
            if 2 * stats.t.sf(abs(t), n - 2) < 0.05:
                sig += 1
                rs.append(abs(r))
        print(f"{rho:6.1f} {n:5d} {sig / B:9.4f} {np.mean(rs):19.4f} "
              f"{np.mean(rs) / rho:11.2f}")
    ```

    ```text
    유의한 연구만 보면 r 이 얼마나 부풀려지나
       참 ρ     n     유의 확률       유의한 경우 평균 |r|      부풀림 배율
       0.2    40    0.2351              0.3964        1.98
       0.2   100    0.5155              0.2736        1.37
       0.2   200    0.8139              0.2222        1.11
       0.1    40    0.0927              0.3783        3.78
       0.5    20    0.6382              0.5982        1.20
    ```

    **참 $\rho=0.2$, $n=40$인 연구가 유의했다면 평균 $r=0.396$**으로 **참값의 2배**다.

    | 참 $\rho$ | $n$ | 유의 확률 | 유의할 때 평균 $\lvert r\rvert$ | 부풀림 |
    |---|---|---|---|---|
    | 0.2 | **40** | 0.235 | **0.396** | **2.0배** |
    | 0.2 | 100 | 0.516 | 0.274 | 1.4배 |
    | 0.2 | 200 | 0.814 | 0.222 | 1.1배 |
    | **0.1** | **40** | 0.093 | **0.378** | **3.8배** |

    **참 상관이 작을수록 부풀림이 심하다.** $\rho=0.1$, $n=40$이면 **3.8배**다.

    **검정력이 높으면 부풀림이 사라진다.** $n=200$에서 1.1배다. **검정력 확보가 편향 제거와 같은 일**이라는 뜻이다.

    **설계 지침 넷.**

    1. **선행 연구의 효과크기를 그대로 쓰지 않는다.** 출판된 값은 부풀려져 있다.
    2. **가장 작은 관심 효과**(SESOI)로 설계한다.
    3. **검정력 0.80은 최소선**이다. 확증 연구는 0.90~0.95를 목표로.
    4. **검출 가능한 최소 $\rho$를 논문에 명시**한다. 독자가 판단할 수 있다.

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
자료가 **군집화**되어 있으면(학급·병원·가구) 상관 검정이 어떻게 되는가?

</div>

??? success "풀이"
    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(30004)
    B = 5_000
    print("두 변수가 독립인데 군집 효과가 있는 자료 (전체 n=200)")
    print(f"{'군집 수':>7s} {'군집 크기':>9s} {'ICC':>6s} {'1종 오류':>9s} "
          f"{'설계효과':>9s} {'유효 n':>8s}")
    for k, m in [(200, 1), (50, 4), (20, 10), (10, 20), (5, 40)]:
        for icc in ([0.0] if m == 1 else [0.3]):
            a = 0
            for _ in range(B):
                gx = rng.standard_normal(k) * np.sqrt(icc)
                gy = rng.standard_normal(k) * np.sqrt(icc)
                x = np.repeat(gx, m) + rng.standard_normal(k * m) * np.sqrt(1 - icc)
                y = np.repeat(gy, m) + rng.standard_normal(k * m) * np.sqrt(1 - icc)
                a += stats.pearsonr(x, y).pvalue < 0.05
            deff = 1 + (m - 1) * icc
            print(f"{k:7d} {m:9d} {icc:6.1f} {a / B:9.4f} {deff:9.2f} "
                  f"{200 / deff:8.1f}")
    ```

    ```text
    두 변수가 독립인데 군집 효과가 있는 자료 (전체 n=200)
       군집 수     군집 크기    ICC     1종 오류      설계효과     유효 n
        200         1    0.0    0.0478      1.00    200.0
         50         4    0.3    0.0820      1.90    105.3
         20        10    0.3    0.1510      3.70     54.1
         10        20    0.3    0.2262      6.70     29.9
          5        40    0.3    0.3024     12.70     15.7
    ```

    **오류율이 0.05에서 0.30까지 간다.**

    | 군집 크기 | 설계효과 | 1종 오류 |
    |---|---|---|
    | 1(군집 없음) | 1.0 | 0.048 ✓ |
    | 4 | 1.9 | 0.082 |
    | 10 | 3.7 | **0.151** |
    | **40** | **12.7** | **0.302** |

    **설계효과 $1+(m-1)\rho_I$가 그대로 문제의 크기**다. 군집 40명, ICC 0.3이면 **표본 200명이 사실상 16명 값어치**다.

    **왜 이렇게 심한가.** $X$와 $Y$ 모두 군집 수준의 성분을 가지면, **군집 평균끼리 우연히 상관**되었을 때 개체 수준에서 강한 상관으로 나타난다.

    ```text
    학급 20개, 학급당 10명

      학급 평균 키 와 학급 평균 성적이 우연히 상관되면
      → 학생 200명 수준에서 "유의한 상관" 이 나온다
      → 실제로는 학급 20개의 우연일 뿐
    ```

    **이것이 생태학적 상관·다수준 문제와 같은 뿌리**다.

    **해결책 넷.**

    | 방법 | 내용 |
    |---|---|
    | **군집 평균으로 집계** | $n=k$(군집 수)로 분석 |
    | **다수준 모형** | 군집 무선효과 포함 |
    | **군집 부트스트랩** | 군집 단위로 재표집 |
    | 군집 강건 표준오차 | 샌드위치 추정량 |

    **집계는 정보를 버리지만 항상 타당**하다. 다수준 모형이 표준이다.

    **군집내상관(ICC)을 먼저 추정**한다.

    $$
    \rho_I=\frac{\sigma^2_{\text{군집간}}}{\sigma^2_{\text{군집간}}+\sigma^2_{\text{군집내}}}
    $$

    **ICC가 0.01만 되어도** 군집이 100명이면 설계효과가 $1+99\times0.01=1.99$로 2배다. **작은 ICC를 무시하면 안 된다.**

    **군집의 예 일곱.**

    ```text
    학급 · 학교 · 병원 · 의사 · 가구 · 마을 · 회사
    반복측정(같은 사람의 여러 관측)
    실험의 배치(batch) · 조사원 · 실험일
    ```

    **마지막 줄이 자주 잊힌다.** 같은 날 측정한 관측들이 서로 닮았다면 그것도 군집이다.

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
$H_0:\rho=\rho_0$($\rho_0\neq0$) 검정과 $H_0:\rho=0$ 검정이 왜 다른 통계량을 쓰는지 수치로 보여라.

</div>

??? success "풀이"
    **두 검정.**

    | 가설 | 통계량 | 분포 |
    |---|---|---|
    | $\rho=0$ | $t=\dfrac{r\sqrt{n-2}}{\sqrt{1-r^2}}$ | $t(n-2)$ **정확** |
    | $\rho=\rho_0$ | $z=\dfrac{\operatorname{arctanh}r-\operatorname{arctanh}\rho_0}{1/\sqrt{n-3}}$ | $N(0,1)$ **근사** |

    ```python
    import warnings
    warnings.filterwarnings("ignore")

    import numpy as np
    from scipy import stats

    rng = np.random.default_rng(30005)
    B = 20_000
    print("1종 오류 (명목 0.05, 이변량 정규)")
    print(f"{'참 ρ0':>7s} {'n':>5s} {'피셔 z':>9s} {'t 공식을 오용':>13s}")
    for rho0 in [0.0, 0.5, 0.9]:
        for n in [20, 100]:
            a = b = 0
            for _ in range(B):
                z = rng.standard_normal((n, 2))
                r = np.corrcoef(z[:, 0], rho0 * z[:, 0]
                                + np.sqrt(1 - rho0**2) * z[:, 1])[0, 1]
                zz = (np.arctanh(r) - np.arctanh(rho0)) * np.sqrt(n - 3)
                a += 2 * stats.norm.sf(abs(zz)) < 0.05
                t = (r - rho0) * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                b += 2 * stats.t.sf(abs(t), n - 2) < 0.05
            print(f"{rho0:7.1f} {n:5d} {a / B:9.4f} {b / B:13.4f}")
    ```

    ```text
    1종 오류 (명목 0.05, 이변량 정규)
       참 ρ0     n      피셔 z      t 공식을 오용
        0.0    20    0.0485        0.0477
        0.0   100    0.0515        0.0513
        0.5    20    0.0502        0.0260
        0.5   100    0.0526        0.0244
        0.9    20    0.0517        0.0001
        0.9   100    0.0498        0.0000
    ```

    **$\rho_0=0$에서는 둘이 정확히 같다.** 그럴 수밖에 없다.

    **$\rho_0\neq0$이면 $t$ 공식이 무너진다.**

    | $\rho_0$ | $n$ | 피셔 $z$ | $t$ 오용 |
    |---|---|---|---|
    | 0.0 | 20 | 0.049 | 0.048 |
    | 0.5 | 20 | 0.050 | **0.026** |
    | **0.9** | **20** | 0.052 | **0.0001** |
    | 0.9 | 100 | 0.050 | **0.0000** |

    **$\rho_0=0.9$에서 오류율이 사실상 0**이다. 검정이 **무엇도 기각하지 못한다.**

    **방향에 주의한다.** $t$ 공식의 오용은 오류율을 **부풀리는 것이 아니라 지워 버린다.** 겉으로는 "보수적이니 안전하다"로 보이지만, **검정력도 함께 사라지므로** 실제로는 쓸모없는 검정이 된다.

    **왜 그런가.** 오용된 통계량

    $$
    t=\frac{(r-\rho_0)\sqrt{n-2}}{\sqrt{1-r^2}}
    $$

    에서 분모 $\sqrt{1-r^2}$는 $\rho_0$가 1에 가까우면 매우 작아진다. 그런데 **$r$의 실제 표준오차 $(1-\rho^2)/\sqrt{n}$는 그보다 훨씬 더 빨리 작아진다.** 결과적으로 분모가 상대적으로 과대해져 $t$가 0 쪽으로 눌린다.

    **정확한 표준오차와의 비교**($\rho_0=0.9$, $n=20$).

    | 양 | 값 |
    |---|---|
    | 실제 $\operatorname{SD}(r)$ | 약 $(1-0.81)/\sqrt{20}=0.042$ |
    | $t$ 공식이 쓰는 분모 | $\sqrt{1-0.81}/\sqrt{18}=0.103$ |

    **2.4배 과대**하므로 $t$가 그만큼 작아진다.

    **또 하나의 이유는 $r$의 분포가 비대칭**이라는 것이다. $\rho=0$일 때만 대칭이다.

    ```text
    ρ = 0   : r 의 분포가 0 을 중심으로 대칭  →  t 근사가 정확
    ρ = 0.9 : r 의 분포가 왼쪽으로 크게 치우침 →  대칭 근사가 실패
              (r 이 1 을 넘을 수 없으므로 위쪽이 눌린다)
    ```

    **피셔 $z$ 변환이 하는 일이 정확히 이 비대칭의 제거**다.

    $$
    \operatorname{arctanh}r\approx N\!\left(\operatorname{arctanh}\rho+\frac{\rho}{2(n-1)},\ \frac{1}{n-3}\right)
    $$

    **분산이 $\rho$에 의존하지 않는다**는 것이 핵심이다.

    **피셔 $z$는 $\rho_0$와 $n$에 관계없이 0.049~0.053**으로 정확하다. 작은 표본에서 더 정밀하게 하려면 편향 항 $\rho/(2(n-1))$까지 보정한다.

    **언제 $\rho_0\neq0$을 검정하나.**

    | 상황 | $\rho_0$ |
    |---|---|
    | **신뢰도 검증** | 0.7(허용 가능 최소) |
    | 측정 도구의 동등성 | 0.9 |
    | **선행 연구와의 비교** | 보고된 값 |
    | 등가성 검정 | $\pm\delta$ |

    **신뢰도 검증이 가장 흔하다.** "재검사 신뢰도가 0.7을 넘는가"를 물으면 $H_0:\rho\leq0.7$의 단측 검정이다.

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
상관 검정의 **선택과 보고 지침**을 정리하라.

</div>

??? success "풀이"
    **검정의 선택.**

    | 목적 | 방법 |
    |---|---|
    | $H_0:\rho=0$ | $t=r\sqrt{(n-2)/(1-r^2)}$, df $=n-2$ |
    | $H_0:\rho=\rho_0$ | **피셔 $z$** |
    | 신뢰구간 | 피셔 $z$(정규) 또는 **부트스트랩**(비정규) |
    | **"상관 없음" 주장** | **TOST 등가성 검정** |
    | $n<15$ · 이상점 | **순열검정** |
    | 군집 자료 | **다수준 모형**이나 군집 부트스트랩 |

    **핵심 수치 일곱.**

    | 사실 | 값 |
    |---|---|
    | $H_0:\rho=0$의 오류율(코시, $n=100$) | **0.047**(견고) |
    | 피셔 $z$ 구간의 피복(로그정규, $n=500$) | **0.648** |
    | 부트스트랩으로 개선 | 0.729 → **0.914** |
    | $\rho=0.2$ 검출에 필요한 $n$(검정력 0.8) | **194** |
    | $n=100$에서 검출 가능한 최소 $\rho$ | 0.277 |
    | 군집 40명·ICC 0.3의 1종 오류 | **0.302** |
    | $t$ 공식을 $\rho_0=0.9$에 오용 | 오류율 **0.0001** |

    **가장 중요한 구분.**

    ```text
    H0: ρ = 0 의 검정       →  분포 가정에 매우 견고
    구간 · ρ0 ≠ 0 의 검정   →  이변량 정규성이 필요

    "비정규라서 검정을 못 한다"는 대개 과장이고
    "비정규라서 구간을 믿을 수 없다"는 대개 맞다
    ```

    **흔한 실수 여섯.**

    | 실수 | 대가 |
    |---|---|
    | $p>0.05$를 "상관 없음"으로 | **등가성 검정**이 필요 |
    | $t$ 공식을 $\rho_0\neq0$에 사용 | 검정력이 사라진다 |
    | 군집 구조 무시 | 오류율 0.30 |
    | 비정규 자료에 피셔 $z$ **구간** | 피복 0.65 |
    | 검정력 고려 없이 소표본 | 승자의 저주 |
    | 여러 쌍 검정 후 보정 없음 | 다중검정 |

    **보고 체크리스트 일곱.**

    ```text
    □ 어느 상관계수인가 (피어슨/스피어만/켄들)
    □ n (결측 처리 후)
    □ 신뢰구간 (방법 명시: 피셔 z / 부트스트랩)
    □ 관측의 독립성 (군집·시계열 여부)
    □ 검정력 또는 검출 가능한 최소 ρ
    □ 여러 검정을 했다면 보정
    □ 산점도
    ```

    **보고 형식.**

    ```text
    운동 빈도와 우울 점수 (n = 312)

      피어슨 r = -0.28,  95% CI [-0.38, -0.17],  t(310) = -5.14, p < 0.001
      스피어만 r_s = -0.26  (두 값이 가까워 이상점 영향은 작다)

      구간은 피셔 z 변환으로 구했다. 이변량 정규성은 QQ 그림으로
      점검했으며 뚜렷한 이탈은 없었다.

      이 표본 크기는 |ρ| ≥ 0.16 을 검정력 0.80 으로 검출한다.

      참가자는 12개 지역센터에서 모집되었으나 센터 수준의
      군집내상관은 0.01 미만이었다 (설계효과 1.2).
    ```

    **마지막 두 문단이 좋은 보고를 만든다.** 검출 가능한 최소 효과와 군집 구조를 밝히면, 독자가 **결과를 스스로 평가**할 수 있다.

    **한 문장.** 상관 검정은 **귀무가설이 $\rho=0$일 때만 가정에 관대**하며, 구간·$\rho_0\neq0$·군집 자료로 넘어가는 순간 각각 다른 도구가 필요하다.

---

## 정리하며

상관의 유의성을 검정하는 **세 가지 방법**을 정리했다.

- **피어슨 $r$ 의 검정은 $t$ 통계량을 쓴다.** $t=r\sqrt{n-2}/\sqrt{1-r^2}$ 이 자유도 $n-2$ 의 $t$ 분포를 따르며, **이변량 정규성을 가정한다.**
- **스피어만과 켄달은 순위 기반이라 분포 가정이 약하다.** 이상치가 있거나 관계가 비선형이면 이쪽이 안전하다.
- **$H_0$ 은 대개 $\rho=0$ 이다.** $\rho=\rho_0\ne0$ 을 검정하려면 피셔 $z$ 변환이 필요하다.
- **$n$ 이 크면 작은 $r$ 도 유의해진다.** $n=1000$ 이면 $r=0.07$ 도 유의하지만 실질적으로는 무의미하다. **$r$ 값 자체와 신뢰구간을 함께 보고해야 한다.**
- **유의성은 인과를 말하지 않는다.** 이 장 전체의 결론이며, 검정이 기각했다는 사실은 연관이 우연이 아니라는 것까지만 말한다.

**이것으로 12장이 끝난다.** 상관의 세 측도와 그 시각화에서 시작해, 교란과 심슨의 역설로 상관이 인과가 아닌 이유를 보았고, 부분상관과 유의성 검정까지 다뤘다.

다음 장 **선형회귀**로 넘어간다. 두 변수의 관계를 하나의 수로 요약하는 대신 **함수로 모형화**하며, 설명변수가 여럿인 경우로 확장한다.
