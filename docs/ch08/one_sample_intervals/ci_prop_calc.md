# 일표본 비율 신뢰구간의 계산

## 개요

이 페이지에서는 단일 모비율 $p$의 신뢰구간을 실제로 계산하는 방법을 다룬다. 네 가지 방법 — Wald 구간, Wilson score 구간, Agresti–Coull 구간, Clopper–Pearson 정확 구간 — 을 다룬다. Python 구현은 성공/실패 개수나 0/1 값이 담긴 CSV 파일 어느 쪽이든 받으며 임의의 신뢰수준을 지원한다.

## Wald 구간

독립인 Bernoulli 시행 $n$번에서 성공이 $k$번이면 표본비율은 $\hat{p} = k/n$이다. Wald $(1-\alpha)100\%$ 신뢰구간은

$$
\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

가장 단순한 방법이지만 $n$이 작거나 $\hat{p}$가 0 또는 1에 가까우면 이항분포에 대한 정규근사가 무너지므로 포함확률이 크게 부족할 수 있다.

## Wilson score 구간

Wilson 구간은 score 검정을 뒤집어 얻는다. $z = z_{\alpha/2}$라 하면 구간의 끝점은

$$
\frac{\hat{p} + \dfrac{z^2}{2n}}{1 + \dfrac{z^2}{n}}
\;\pm\;
\frac{z}{1 + \dfrac{z^2}{n}}
\sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}
$$

분모 $1 + z^2/n$이 구간을 $1/2$ 쪽으로 축소하여 $p$의 전 범위에서 포함확률을 개선한다. Wilson 구간은 대부분의 실무에서 권장되는 기본값이다.

## Agresti–Coull 구간

가상의 성공 $z^2/2$개와 가상의 실패 $z^2/2$개를 더해 보정된 양을 만든다:

$$
\tilde{n} = n + z^2, \quad \tilde{p} = \frac{k + z^2/2}{\tilde{n}}
$$

그다음 $\tilde{p}$와 $\tilde{n}$으로 Wald 공식을 적용한다:

$$
\tilde{p} \pm z_{\alpha/2} \sqrt{\frac{\tilde{p}(1-\tilde{p})}{\tilde{n}}}
$$

95% 수준에서는 성공과 실패를 각각 약 2개씩 더하는 셈이다("plus-four" 규칙). 계산이 더 간단하면서도 포함확률은 Wilson과 매우 가깝다.

## Clopper–Pearson (정확) 구간

Clopper–Pearson 구간은 Beta 분위수로 두 개의 단측 이항검정을 뒤집는다:

$$
\left(\text{Beta}\!\left(\frac{\alpha}{2};\; k,\; n-k+1\right),\;\;
      \text{Beta}\!\left(1-\frac{\alpha}{2};\; k+1,\; n-k\right)\right)
$$

$k = 0$이면 하한을 0으로, $k = n$이면 상한을 1로 두는 관례를 따른다. 이 구간은 모든 $p$에서 적어도 $(1-\alpha)100\%$의 포함확률을 보장하지만 (필요보다 넓게) 보수적이다.

## Python 코드

### CSV에서 자료 읽기

```python
import csv
import numpy as np

def load_data(csv_path):
    """비율 추정을 위해 CSV에서 0/1 값을 읽는다."""
    arr = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    v = float(item)
                    # 0과 1 외의 값이 섞이면 여기서 끊는다.
                    # 그냥 두면 sum()이 성공 횟수가 아니게 되어
                    # p_hat이 1을 넘는 식으로 조용히 망가진다.
                    if v not in (0, 1):
                        raise ValueError("CSV must contain only 0/1 values.")
                    arr.append(v)
    if len(arr) == 0:
        raise ValueError("No values found in CSV.")
    return np.array(arr, dtype=float)


# 임시 파일로 확인한다. 성공 횟수 k와 표본크기 n만 있으면 구간을 만들 수 있다.
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "bernoulli.csv")
    with open(path, "w") as f:
        f.write("1,0,0,1,0\n1,0,0,0,0\n")
    y = load_data(path)
print(f"n = {len(y)}, k = {int(y.sum())}, p_hat = {y.mean()}")
```

출력:

```
n = 10, k = 3, p_hat = 0.3
```

### 신뢰구간의 계산

```python
import math
from scipy.stats import norm, beta

def ci_proportion(k, n, method="wilson", cl=0.95):
    """모비율에 대한 일표본 신뢰구간. 네 가지 방법을 한 함수에 모았다.

    기본값이 wald가 아니라 wilson인 것에 주의하라.
    Wald는 교과서에 먼저 나오지만 실무 기본값으로 삼을 만한 방법이 아니다.

    k : 성공 횟수
    n : 표본크기
    method : 'wald', 'wilson', 'ac', 'cp'
    cl : 신뢰수준 (기본 0.95)
    """
    alpha = 1 - cl
    z = norm.ppf(1 - alpha / 2)
    phat = k / n

    if method == "wald":
        se = math.sqrt(phat * (1 - phat) / n)
        lo = phat - z * se
        hi = phat + z * se
    elif method == "wilson":
        denom = 1 + z * z / n
        center = (phat + z * z / (2 * n)) / denom
        half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
        lo, hi = center - half, center + half
    elif method == "ac":
        n_tilde = n + z * z
        p_tilde = (k + 0.5 * z * z) / n_tilde
        se_tilde = math.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
        lo = p_tilde - z * se_tilde
        hi = p_tilde + z * se_tilde
    else:  # cp (Clopper-Pearson)
        # 정규근사를 아예 쓰지 않고 이항분포를 직접 뒤집는다.
        # Beta가 나오는 것은 이항 꼬리확률과 Beta 누적분포가 같은 식이기 때문이다.
        # k=0이나 k=n이면 한쪽 Beta의 모수가 0이 되어 정의되지 않으므로 관례를 따른다.
        lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)

    # Wald는 끝점이 [0,1]을 벗어날 수 있다. 나머지 셋은 그럴 일이 없다.
    lo = max(0.0, lo)
    hi = min(1.0, hi)
    return lo, hi

# 50번 중 12번 성공에 대한 95% Wilson 구간
lo, hi = ci_proportion(k=12, n=50, method="wilson", cl=0.95)
print(f"95% Wilson CI: ({lo:.4f}, {hi:.4f})")

# 같은 자료의 99% Clopper-Pearson 구간
lo, hi = ci_proportion(k=12, n=50, method="cp", cl=0.99)
print(f"99% Clopper-Pearson CI: ({lo:.4f}, {hi:.4f})")
```

출력:

```
95% Wilson CI: (0.1430, 0.3741)
99% Clopper-Pearson CI: (0.1056, 0.4255)
```

두 구간의 신뢰수준이 다르므로 너비를 곧바로 비교할 수는 없다. 같은 95%로 맞추면 Wilson이 $(0.1430, 0.3741)$, Clopper–Pearson이 $(0.1306, 0.3817)$로 후자가 약 9% 넓다. 이것이 "모든 $p$에서 95% 아래로 내려가지 않는다"는 보장의 값이다.

### 명령줄 사용법

함께 제공되는 스크립트 `ci_prop_calc.py`는 명령줄 인자를 지원한다:

```bash
# Wilson interval from counts
python ci_prop_calc.py --k 12 --n 50 --method wilson

# Clopper-Pearson interval from a CSV of 0/1 values
python ci_prop_calc.py --csv bernoulli.csv --method cp

# 99% confidence level
python ci_prop_calc.py --k 12 --n 50 --method wilson --cl 0.99
```

## 해석

- **Wald 구간**은 계산하기 쉽지만 $\hat{p}$가 0이나 1에 가까우면 말이 안 되는 결과(음수인 끝점이나 1을 넘는 끝점)를 낼 수 있다. 끝점을 $[0, 1]$로 자르더라도 이런 경우 포함확률은 여전히 나쁘다.
- **Wilson score 구간**은 중심을 $1/2$ 쪽으로 조정하며 일반적인 용도로 권장된다. 표본크기가 중간이거나 비율이 극단적이어도 좋은 포함확률을 유지한다.
- **Agresti–Coull 구간**은 표본크기를 $z^2$만큼 부풀리고 $\hat{p}$를 다시 중심화하는 더 간단한 장치로 Wilson에 필적하는 포함확률을 얻는다. 손으로 계산하기 쉬워야 할 때 실용적인 선택이다.
- **Clopper–Pearson 구간**은 모든 $p$에서 적어도 $(1-\alpha)100\%$의 포함확률을 보장하는 유일한 방법이지만 그 대가로 너비가 커진다. $n$이 크면 보수성이 약하지만 $n$이 작으면 상당할 수 있다.

## 연습문제

**연습문제 1.** 품질관리 표본에서 200개 중 8개가 불량이다. 불량률 $p$의 95% Wald와 Wilson 신뢰구간을 계산하라. 차이를 논하라.

??? success "풀이"

    여기서 $k = 8$, $n = 200$, $\hat{p} = 0.04$, $z = 1.96$이다.

    **Wald:** $\text{SE} = \sqrt{0.04 \times 0.96 / 200} = \sqrt{0.000192} = 0.01386$. 신뢰구간: $0.04 \pm 1.96 \times 0.01386 = 0.04 \pm 0.02716 = (0.0128, 0.0672)$.

    **Wilson:** 분모: $1 + 1.96^2/200 = 1 + 0.01921 = 1.01921$. 중심: $(0.04 + 3.8416/400)/1.01921 = 0.04960/1.01921 = 0.04868$. 반너비: $1.96 \times \sqrt{0.04 \times 0.96/200 + 3.8416/160000}/1.01921 = 1.96 \times \sqrt{0.000192 + 0.000024}/1.01921 = 1.96 \times 0.01470/1.01921 = 0.02826$. 신뢰구간: $(0.0204, 0.0770)$.

    Wilson 구간은 오른쪽으로 옮겨져 있고(중심 0.049 대 0.040) 약간 더 넓다. $n\hat{p} = 8 < 10$이므로 Wald 구간의 바탕인 정규근사가 아슬아슬하며, Wilson 구간이 더 믿을 만한 포함확률을 준다. $\square$

---

**연습문제 2.** $\hat{p} = 0$이거나 $\hat{p} = 1$일 때 Wald 구간의 너비가 0임을 보이고 왜 문제인지 설명하라.

??? success "풀이"

    $\hat{p} = 0$(즉 $k = 0$)이면 표준오차는

    $$
    \text{SE} = \sqrt{\frac{0 \cdot 1}{n}} = 0
    $$

    이므로 Wald 구간이 한 점 $[0, 0]$으로 무너진다. $\hat{p} = 1$일 때도 마찬가지로 구간이 $[1, 1]$이 된다.

    이것이 문제인 이유는 $n$번 시행에서 성공을 $k = 0$번 관측했다고 해서 $p = 0$이 확실한 것은 아니기 때문이다. 예를 들어 $p = 0.01$이고 $n = 50$이면 $k = 0$을 관측할 확률이 $(1 - 0.01)^{50} \approx 0.605$로 결코 무시할 수 없다. 0에서 퇴화한 구간은 $p$의 어떤 양수 값도 담지 못하므로 포함확률이 명목 수준보다 크게 떨어진다. Wilson과 Clopper–Pearson 방법은 $k = 0$이나 $k = n$일 때도 너비가 양수인 구간을 만들어 이 퇴화를 피한다. $\square$

---

**연습문제 3.** 어떤 조사에서 응답자 1000명 중 540명이 어떤 정책을 지지한다. 네 가지 방법(Wald, Wilson, Agresti–Coull, Clopper–Pearson)으로 95% 신뢰구간을 계산하고 너비를 비교하라.

??? success "풀이"

    여기서 $k = 540$, $n = 1000$, $\hat{p} = 0.54$, $z = 1.96$이다.

    **Wald:** $\text{SE} = \sqrt{0.54 \times 0.46/1000} = \sqrt{0.000248} = 0.01576$. 신뢰구간: $0.54 \pm 0.03089 = (0.5091, 0.5709)$. 너비 = 0.0618.

    **Wilson:** 분모 $= 1 + 3.8416/1000 = 1.003842$. 중심 $= (0.54 + 0.001921)/1.003842 = 0.5398$. 반너비 $= 1.96 \times \sqrt{0.000248 + 0.00000096}/1.003842 = 1.96 \times 0.01578/1.003842 = 0.03082$. 신뢰구간: $(0.5090, 0.5706)$. 너비 = 0.0616.

    **Agresti–Coull:** $\tilde{n} = 1003.84$, $\tilde{p} = (540 + 1.9208)/1003.84 = 0.5398$. 표준오차 $= \sqrt{0.5398 \times 0.4602/1003.84} = 0.01574$. 신뢰구간: $0.5398 \pm 0.03085 = (0.5090, 0.5707)$. 너비 = 0.0617.

    **Clopper–Pearson:** $L = \text{Beta}(0.025;\,540,\,461) = 0.5087$. $U = \text{Beta}(0.975;\,541,\,460) = 0.5712$. 너비 = 0.0625.

    $n = 1000$이고 $\hat{p}$가 $0.5$ 근처이므로 네 방법이 거의 같은 구간을 준다. Clopper–Pearson이 조금 더 넓다(0.0625 대 나머지 약 0.0617). $n$이 크고 $p$가 경계에서 멀면 방법의 선택이 거의 중요하지 않다. $\square$

---

**연습문제 4.** 어떤 임상시험에서 환자 30명 중 중대한 이상반응이 0건이다. Clopper–Pearson 방법으로 $p$의 단측 95% 상한을 계산하고 "3의 법칙" 근사를 진술하라.

??? success "풀이"

    $k = 0$, $n = 30$이면 Clopper–Pearson 양측 95% 구간의 하한은 0이다. 상한은:

    $$
    U = 1 - (\alpha/2)^{1/n}
    $$

    더 정확히는 $U = \text{Beta}(0.975;\, 1,\, 30) = 1 - 0.025^{1/30}$을 쓴다. 계산하면 $\ln(0.025)/30 = -3.6889/30 = -0.12296$이므로 $U = 1 - e^{-0.12296} = 1 - 0.8843 = 0.1157$이다.

    단측 95% 상한(위쪽 꼬리에만 $\alpha = 0.05$를 두는 경우)은:

    $$
    U = 1 - (0.05)^{1/30} = 1 - e^{\ln(0.05)/30} = 1 - e^{-0.0999} \approx 0.0951
    $$

    **3의 법칙**은 빠른 근사를 준다: $k = 0$일 때 95% 단측 상한은 대략 $3/n$이다. 여기서는 $3/30 = 0.10$으로 정확한 값 0.0951에 가깝다. 3의 법칙은 근사 $1 - \alpha^{1/n} \approx -\ln(\alpha)/n$과 $-\ln(0.05) \approx 3$이라는 사실에서 나온다. $\square$

---

**연습문제 5.** 95% 수준에서 $k = 7$, $n = 25$일 때 Wilson과 Agresti–Coull 구간이 거의 같음을 수치로 확인하고, 왜 가깝지만 정확히 같지는 않은지 대수적으로 설명하라.

??? success "풀이"

    $k = 7$, $n = 25$, $\hat{p} = 0.28$, $z = 1.96$일 때:

    **Wilson:** 분모 $= 1 + 3.8416/25 = 1.15366$. 중심 $= (0.28 + 0.07683)/1.15366 = 0.3093$. 반너비 $= 1.96 \times \sqrt{0.28 \times 0.72/25 + 3.8416/2500}/1.15366 = 1.96 \times \sqrt{0.008064 + 0.001537}/1.15366 = 1.96 \times 0.09798/1.15366 = 0.1664$. 신뢰구간: $(0.1429, 0.4757)$.

    **Agresti–Coull:** $\tilde{n} = 25 + 3.8416 = 28.8416$. $\tilde{p} = (7 + 1.9208)/28.8416 = 0.3093$. 표준오차 $= \sqrt{0.3093 \times 0.6907/28.8416} = \sqrt{0.007408} = 0.08607$. 신뢰구간: $0.3093 \pm 1.96 \times 0.08607 = 0.3093 \pm 0.1687 = (0.1406, 0.4780)$.

    중심이 사실상 같고(둘 다 0.3093) 반너비는 0.002 정도만 다르다.

    **대수적 설명.** 두 방법 모두 중심을 $\tilde{p} = (k + z^2/2)/(n + z^2)$으로 조정한다. Wilson 구간은 정확한 표준오차 $\sqrt{\hat{p}(1-\hat{p})/n + z^2/(4n^2)}$을 $1 + z^2/n$으로 나누어 쓰는 반면, Agresti–Coull은 $\sqrt{\tilde{p}(1-\tilde{p})/\tilde{n}}$을 쓴다. 차이는 Wilson의 반너비가 근호 안에 (보정하지 않은) $\hat{p}$를 쓰고 Agresti–Coull은 $\tilde{p}$를 쓰는 데서 온다. $n$이 적당하면 $\hat{p}$와 $\tilde{p}$가 가까우므로 두 구간이 거의 일치한다. $n$이 아주 작거나 $\hat{p}$가 극단적이면 차이가 더 눈에 띈다. $\square$
