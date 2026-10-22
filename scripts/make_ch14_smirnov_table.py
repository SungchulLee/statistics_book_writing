r"""콜모고로프의 급수와, 스미르노프가 열다섯 해 뒤에 내놓은 표.

콜모고로프($1933$)가 준 것은 분포함수의 **급수**다.

    K(t) = 1 - 2 * sum_{k>=1} (-1)^{k-1} exp(-2 k^2 t^2)

스미르노프가 $1948$ 년 『Annals of Mathematical Statistics』 $19$ 권 $279$–$281$ 쪽에
실은 것은 그 급수의 **표**다.

① 그 한 장이 덮는 곡선과 임계점 둘.
② 그 표 — 급수를 자리 여섯까지 계산해 채운 칸들.

곁들여, 자리 여섯까지 맞추는 데 항이 몇 개 드는지도 찍는다. 오른쪽 꼬리는 한두 항이면
되지만 왼쪽 끝은 열 항이 넘고, 거기서는 세타함수 꼴의 다른 급수가 훨씬 싸다.

만드는 파일: docs/ch14/formal_tests/img/smirnov_table.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

TOL = 5e-7  # 소수 여섯째 자리까지


def K_alt(t, k=200):
    """콜모고로프의 교대급수."""
    j = np.arange(1, k + 1)
    return 1 - 2 * np.sum((-1) ** (j - 1) * np.exp(-2 * j**2 * t**2))


def K_theta(t, k=200):
    """작은 t 쪽에서 빠른 세타함수 꼴."""
    j = np.arange(1, k + 1)
    return (np.sqrt(2 * np.pi) / t) * np.sum(
        np.exp(-((2 * j - 1) ** 2) * np.pi**2 / (8 * t**2))
    )


def K(t):
    return K_theta(t) if t < 0.92 else K_alt(t)


def terms_alt(t, kmax=400):
    for k in range(1, kmax + 1):
        if 2 * np.exp(-2 * (k + 1) ** 2 * t**2) < TOL:
            return k
    return np.nan


def terms_theta(t, kmax=400):
    c = np.sqrt(2 * np.pi) / t
    for k in range(1, kmax + 1):
        if c * np.exp(-((2 * (k + 1) - 1) ** 2) * np.pi**2 / (8 * t**2)) < TOL:
            return k
    return np.nan


# === 먼저 우리 계산이 맞는지 확인한다 =================================
print("① 두 급수와 scipy 의 콜모고로프 분포가 같은 값을 주는가")
print("     t      교대급수       세타급수        scipy        최대 차")
for t in [0.3, 0.5, 0.8, 1.0, 1.3581, 1.6276, 2.0]:
    a, b, c = K_alt(t), K_theta(t), stats.kstwobign.cdf(t)
    print(f"  {t:.4f}   {a:.8f}   {b:.8f}   {c:.8f}   {max(abs(a-b), abs(a-c)):.1e}")

print("\n   자리 여섯까지 맞추는 데 드는 항의 수")
print("      t     교대급수   세타급수")
for t in [0.25, 0.40, 0.60, 0.92, 1.3581, 2.00]:
    print(f"   {t:>6.4f}      {terms_alt(t):>3}       {terms_theta(t):>3}")

t05 = brentq(lambda t: K_alt(t) - 0.95, 1.0, 2.0)
t01 = brentq(lambda t: K_alt(t) - 0.99, 1.0, 2.5)
print(f"\n   K(t) = 0.95 인 t = {t05:.6f}   (scipy {stats.kstwobign.ppf(0.95):.6f})")
print(f"   K(t) = 0.99 인 t = {t01:.6f}   (scipy {stats.kstwobign.ppf(0.99):.6f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.4),
                               gridspec_kw={"width_ratios": [1.45, 1.0]})

# === ① 한 장이 덮는 곡선 ============================================
tt = np.linspace(0.25, 2.40, 600)
kk = np.array([K(t) for t in tt])
ax1.plot(tt, kk, color="#1f77b4", lw=2.8)
for t, q, c, lab in [(t05, 0.95, "#d62728", "$5\\%$"),
                     (t01, 0.99, "#8c564b", "$1\\%$")]:
    ax1.plot([t, t], [0, q], color=c, lw=1.2, ls=":")
    ax1.plot([tt[0], t], [q, q], color=c, lw=1.2, ls=":")
    ax1.plot([t], [q], "o", color=c, ms=9)
    ax1.text(t + 0.04, q - 0.055, f"{lab} 점   $t = {t:.4f}$", color=c, fontsize=11.5)
ax1.set_xlim(tt[0], tt[-1])
ax1.set_ylim(0, 1.04)
ax1.set_xlabel("$t = \\sqrt{n}\\,D_n$")
ax1.set_ylabel("$K(t)$")
ax1.set_title("① 분포가 무엇이든, $n$ 이 얼마든 이 곡선 하나", fontsize=12)
ax1.grid(alpha=0.25)
ax1.text(
    0.30, 0.72,
    "$K(t) = 1 - 2\\sum_{k \\geq 1} (-1)^{k-1} e^{-2k^2t^2}$\n"
    "콜모고로프 $1933$",
    fontsize=11.5, va="top",
    bbox=dict(boxstyle="round,pad=0.45", fc="#eef4fa", ec="#1f77b4", lw=1.0),
)

# === ② 그 표 =========================================================
ROWS = np.round(np.arange(0.30, 2.05, 0.10), 2)
ax2.axis("off")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
top = 0.94
ax2.plot([0.06, 0.94], [top, top], color="#333333", lw=1.6)
ax2.text(0.22, top - 0.045, "$t$", ha="center", fontsize=11.5)
ax2.text(0.68, top - 0.045, "$K(t)$", ha="center", fontsize=11.5)
ax2.plot([0.06, 0.94], [top - 0.065, top - 0.065], color="#333333", lw=1.0)
dy = (top - 0.10) / len(ROWS)
for i, t in enumerate(ROWS):
    y = top - 0.095 - (i + 0.5) * dy
    col = "#222222"
    ax2.text(0.22, y, f"{t:.2f}", ha="center", fontsize=10.8, color=col,
             family="monospace")
    ax2.text(0.68, y, f"{K(t):.6f}", ha="center", fontsize=10.8, color=col,
             family="monospace")
ax2.plot([0.06, 0.94], [top - 0.095 - len(ROWS) * dy, top - 0.095 - len(ROWS) * dy],
         color="#333333", lw=1.6)
ax2.set_title("② 스미르노프 $1948$ 이 채운 칸 — 자리 여섯까지", fontsize=12)

fig.tight_layout()
out = "docs/ch14/formal_tests/img/smirnov_table.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

print("\n② 표의 값")
for t in ROWS:
    print(f"   K({t:.2f}) = {K(t):.6f}")
