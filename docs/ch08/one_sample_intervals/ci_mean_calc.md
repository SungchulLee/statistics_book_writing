# 일표본 평균 신뢰구간의 계산

## 개요

이 페이지에서는 모평균 $\mu$에 대한 일표본 신뢰구간을 실제로 계산하는 방법을 다룬다. 두 가지 방법을 다룬다: 모표준편차 $\sigma$를 아는 경우의 $z$-구간과, $\sigma$를 모르고 표본표준편차 $s$로 추정하는 경우의 $t$-구간. Python 구현은 원자료와 요약통계량 어느 쪽이든 받는다.

## z-구간 (분산을 아는 경우)

$\sigma$를 알 때 $\mu$의 $(1-\alpha)100\%$ 신뢰구간은

$$
\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}
$$

표준오차는 $\text{SE} = \sigma / \sqrt{n}$이고 오차한계는 $\text{MOE} = z_{\alpha/2} \cdot \text{SE}$이다.

## t-구간 (분산을 모르는 경우)

$\sigma$를 모르면 $s$로 바꾸고 자유도 $\text{df} = n - 1$인 $t$-분포를 쓴다:

$$
\bar{x} \pm t_{\alpha/2,\,n-1} \cdot \frac{s}{\sqrt{n}}
$$

$t$-분포는 표준정규보다 꼬리가 두꺼우므로 모든 유한한 $n$에서 $t_{\alpha/2,\,n-1} > z_{\alpha/2}$이고, 그 결과 $\sigma$ 추정의 불확실성을 반영한 더 넓은 구간이 나온다.

## Python 코드

### CSV에서 자료 읽기

```python
import csv
import numpy as np

def load_data(csv_path):
    """Read a single column of numeric values from a CSV file."""
    arr = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    arr.append(float(item))
    if len(arr) == 0:
        raise ValueError("No numeric values found in CSV.")
    return np.array(arr, dtype=float)
```

### 신뢰구간의 계산

```python
import math
from scipy.stats import norm, t

def ci_mean(n, xbar, s=None, known_sigma=None, method="t", cl=0.95):
    """
    Compute a one-sample CI for the population mean.

    Parameters
    ----------
    n : int            - sample size
    xbar : float       - sample mean
    s : float or None  - sample standard deviation (required for t-interval)
    known_sigma : float or None - known population sigma (required for z-interval)
    method : str       - 't' or 'z'
    cl : float         - confidence level (default 0.95)
    """
    alpha = 1 - cl

    if method == "z":
        se = known_sigma / math.sqrt(n)
        z_star = norm.ppf(1 - alpha / 2)
        moe = z_star * se
    else:
        se = s / math.sqrt(n)
        df = n - 1
        t_star = t.ppf(1 - alpha / 2, df=df)
        moe = t_star * se

    return xbar - moe, xbar + moe

# Example: t-interval from summary statistics
lo, hi = ci_mean(n=25, xbar=3.2, s=1.1, method="t", cl=0.95)
print(f"95% t-interval: ({lo:.4f}, {hi:.4f})")

# Example: z-interval with known sigma
lo, hi = ci_mean(n=25, xbar=3.2, known_sigma=1.0, method="z", cl=0.95)
print(f"95% z-interval: ({lo:.4f}, {hi:.4f})")
```

### 명령줄 사용법

함께 제공되는 스크립트 `ci_mean_calc.py`는 명령줄 인자를 지원한다:

```bash
# t-interval from a CSV file
python ci_mean_calc.py --csv data.csv

# z-interval from summary statistics
python ci_mean_calc.py --n 25 --mean 3.2 --sd 1.1 --known-sigma 1.0 --method z

# 99% confidence level
python ci_mean_calc.py --csv data.csv --cl 0.99
```

## 해석

- **신뢰수준** $1 - \alpha$는 절차의 장기적 성공률을 기술한다: 연구를 여러 번 반복하며 매번 신뢰구간을 만들면 그중 약 $(1-\alpha)100\%$가 $\mu$를 담는다.
- **구간이 넓다는 것**은 불확실성이 크다는 뜻이다. $n$이 줄거나, $s$(또는 $\sigma$)가 커지거나, 신뢰수준이 올라가면 너비가 늘어난다.
- $t$-구간은 언제나 대응하는 $z$-구간 이상으로 넓다. $n \ge 30$이면 $t_{\alpha/2,\,n-1} \approx z_{\alpha/2}$이므로 차이가 작다.
- 자료가 강하게 비정규이고 $n$이 작으면 $t$-구간이 명목 포함확률을 달성하지 못할 수 있다. 이럴 때는 비모수 붓스트랩 신뢰구간을 고려하라.

## 연습문제

**연습문제 1.** 정규모집단에서 뽑은 관측값 $n = 16$개에서 $\bar{x} = 50$, $s = 8$을 얻었다. $\mu$의 95%와 99% $t$-신뢰구간을 계산하라. 너비는 어떻게 변하는가?

??? success "풀이"

    $\text{df} = 15$일 때:

    **95% 신뢰구간:** $t_{0.025,15} = 2.131$. $\text{MOE} = 2.131 \times 8/\sqrt{16} = 2.131 \times 2 = 4.262$. 신뢰구간: $(45.74, 54.26)$. 너비 = 8.524.

    **99% 신뢰구간:** $t_{0.005,15} = 2.947$. $\text{MOE} = 2.947 \times 2 = 5.894$. 신뢰구간: $(44.11, 55.89)$. 너비 = 11.788.

    99% 구간은 95% 구간의 $11.788/8.524 = 1.38$배 넓다. 더 높은 확신을 위해서는 더 넓은 범위의 그럴듯한 값을 받아들여야 한다. $\square$

---

**연습문제 2.** $z$-구간의 오차한계가 $n$의 감소함수임을 보이고 감소 속도를 구하라.

??? success "풀이"

    오차한계는 $\text{MOE} = z_{\alpha/2} \cdot \sigma / \sqrt{n}$이다. $z_{\alpha/2}$와 $\sigma$가 상수이므로:

    $$
    \frac{d(\text{MOE})}{dn} = z_{\alpha/2} \cdot \sigma \cdot \left(-\frac{1}{2}\right) n^{-3/2} < 0
    $$

    따라서 오차한계는 $n$에 대해 순감소한다. 감소 속도는 $O(n^{-1/2})$이다: $n$을 두 배로 하면 오차한계가 $1/\sqrt{2} \approx 0.707$배가 되어 약 29% 줄어든다. 오차한계를 절반으로 줄이려면 표본크기를 네 배로 해야 한다. $\square$

---

**연습문제 3.** 어떤 연구자가 $n = 100$, $s = 10$일 때 $\mu$의 95% $t$-구간과 $z$-구간이 "사실상 같다"고 주장한다. 수치로 확인하라.

??? success "풀이"

    $n = 100$이면 $\text{df} = 99$이고:

    - $z_{0.025} = 1.960$
    - $t_{0.025,99} = 1.984$

    표준오차는 $s/\sqrt{n} = 10/10 = 1$이다.

    - $z$-오차한계 $= 1.960 \times 1 = 1.960$
    - $t$-오차한계 $= 1.984 \times 1 = 1.984$

    차이는 $1.984 - 1.960 = 0.024$로 $z$-오차한계의 1.3% 미만이다. 표본평균이 예컨대 50이라면 $z$-구간은 $(48.040, 51.960)$, $t$-구간은 $(48.016, 51.984)$이다. 연구자의 주장이 옳다: 양쪽 끝에서 0.024 단위만 다르다. $\square$

---

**연습문제 4.** $\nu \to \infty$일 때 $t_{\alpha/2,\,\nu} \to z_{\alpha/2}$임을 증명하라.

??? success "풀이"

    $T \sim t_\nu$이면 $T = Z / \sqrt{V/\nu}$이며 여기서 $Z \sim N(0,1)$과 $V \sim \chi^2_\nu$는 독립이다. 대수의법칙에 의해 $\nu \to \infty$일 때 $V/\nu \xrightarrow{P} 1$이다. 따라서 Slutsky 정리에 의해

    $$
    T = \frac{Z}{\sqrt{V/\nu}} \xrightarrow{d} \frac{Z}{1} = Z \sim N(0,1)
    $$

    이다. $t_\nu$의 누적분포함수가 표준정규 누적분포함수로 점별 수렴하므로 분위수도 수렴한다:

    $$
    t_{\alpha/2,\,\nu} \to z_{\alpha/2} \quad (\nu \to \infty)
    $$

    $\square$

---

**연습문제 5.** 원자료 $\{12, 15, 14, 10, 13, 16, 11, 14, 13, 12\}$에 대해 $\mu$의 90% $t$-구간을 손으로 계산하고 코드로 확인하라.

??? success "풀이"

    **손으로:** $n = 10$, $\bar{x} = (12+15+14+10+13+16+11+14+13+12)/10 = 130/10 = 13.0$.

    $$
    s^2 = \frac{1}{9}\sum(x_i - 13)^2 = \frac{1}{9}(1+4+1+9+0+9+4+1+0+1) = \frac{30}{9} = 3.333
    $$

    $$
    s = \sqrt{3.333} = 1.826
    $$

    $\text{df} = 9$, $\alpha = 0.10$, $t_{0.05,9} = 1.833$이므로:

    $$
    13.0 \pm 1.833 \times \frac{1.826}{\sqrt{10}} = 13.0 \pm 1.833 \times 0.5774 = 13.0 \pm 1.059
    $$

    90% 신뢰구간은 $(11.94, 14.06)$이다.

    **확인:**

    ```python
    import numpy as np
    from scipy.stats import t
    data = np.array([12, 15, 14, 10, 13, 16, 11, 14, 13, 12])
    n = len(data)
    xbar, s = data.mean(), data.std(ddof=1)
    t_crit = t.ppf(0.95, df=n-1)
    moe = t_crit * s / np.sqrt(n)
    print(f"({xbar - moe:.2f}, {xbar + moe:.2f})")
    # Output: (11.94, 14.06)
    ```

    $\square$
