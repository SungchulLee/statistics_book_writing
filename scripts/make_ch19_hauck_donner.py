"""하우크-도너 현상을 그들이 쓴 2x2 자료에서 그대로 그린다.

ch19/estimation_inference/img/hauck_donner.png

Hauck 과 Donner(1977)가 효과를 보이려고 쓴 2x2 표는 두 집단이 각각 100 이고
기준 집단의 성공이 25 다. 둘째 집단의 성공 R 을 26 부터 99 까지 밀면서
왈드 통계량과 가능도비 통계량을 **해석적으로** 잰다(모의실험이 아니다).

  로짓 모형  logit p_i = b1 + b2 x_i,  x = 0 또는 1
  b2hat     = logit(R/N) - logit(R0/N)              곧 로그오즈비
  Var(b2hat)= 1/(N p0 q0) + 1/(N p1 q1)             두 칸의 분산 합
  W^2       = b2hat^2 / Var(b2hat)
  Lambda    = 2 [ l(두 비율 따로) - l(합친 비율) ]

실행:  python3 scripts/make_ch19_hauck_donner.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/estimation_inference/img/"

INK = "#37474F"
MUTED = "#90A4AE"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"

N, R0 = 100, 25


def loglik(r, n, p):
    return r * np.log(p) + (n - r) * np.log(1 - p)


def stats(R):
    """로그오즈비, 왈드 W^2, 가능도비 Lambda 를 돌려준다."""
    p0, p1 = R0 / N, R / N
    b2 = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
    w2 = b2 ** 2 / (1 / (N * p0 * (1 - p0)) + 1 / (N * p1 * (1 - p1)))
    pp = (R0 + R) / (2 * N)
    lam = 2 * ((loglik(R0, N, p0) + loglik(R, N, p1))
               - (loglik(R0, N, pp) + loglik(R, N, pp)))
    return b2, w2, lam


def chi2_sf(x):
    """자유도 1 의 꼬리확률. erfc 로 적는다 — scipy 없이."""
    from math import erfc, sqrt
    return np.array([erfc(sqrt(v / 2)) for v in np.atleast_1d(x)])


Rs = np.arange(26, 100)
B2, W2, LAM = map(np.array, zip(*[stats(R) for R in Rs]))
peak = int(Rs[W2.argmax()])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.9))
fig.suptitle("효과가 강해지는데 왈드만 돌아선다", fontsize=15, color=INK, y=0.99)

# ── 왼쪽: 두 통계량
ax1.plot(Rs, LAM, color=TEAL, lw=2.4, label="가능도비 $\\Lambda$ — 계속 오른다")
ax1.plot(Rs, W2, color=RED, lw=2.4, ls="--", label="왈드 $W^2$ — 돌아선다")
ax1.plot([peak], [W2.max()], "o", color=RED, ms=9, zorder=5)
ax1.annotate(f"R = {peak} 에서 꼭짓점\n$W^2$ = {W2.max():.1f}",
             xy=(peak, W2.max()), xytext=(62, 96), color=RED, fontsize=11,
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
# 같은 구간(꼭짓점 -> 끝)에서 두 통계량이 어디로 가는지 — 기준을 맞춘다
ax1.annotate(f"꼭짓점에서 끝까지\n증거는 {LAM[-1]/W2.max():.1f} 배 더 쌓였는데\n"
             f"왈드는 {W2[-1]/W2.max():.2f} 배로 내려간다",
             xy=(98.4, W2[-1] - 1.5), xytext=(60, 2), color=INK, fontsize=10.5,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.4))
ax1.set_xlabel("둘째 집단의 성공 수 $R$  (기준 집단은 $100$ 중 $25$)")
ax1.set_ylabel("검정통계량")
ax1.set_title("두 자가 같은 자료를 다르게 읽는다", fontsize=12, color=INK)
ax1.legend(loc="upper left", fontsize=10.5, framealpha=0.95)
ax1.grid(alpha=0.25)

# ── 오른쪽: p-값. 로그 눈금의 마이너스는 mathtext 라 포매터로 직접 쓴다
ax2.semilogy(Rs, chi2_sf(LAM), color=TEAL, lw=2.4, label="가능도비")
ax2.semilogy(Rs, chi2_sf(W2), color=RED, lw=2.4, ls="--", label="왈드")
ax2.axvline(peak, color=MUTED, lw=1.2, ls=":")
ax2.annotate("왈드 p-값이 바닥을 치고\n거꾸로 올라간다",
             xy=(97, chi2_sf(W2[Rs == 97])[0]), xytext=(34, 1e-20),
             color=AMBER, fontsize=11,
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
ax2.yaxis.set_major_formatter(FuncFormatter(
    lambda v, _: "1" if v == 1 else f"1e{int(round(np.log10(v)))}"))
ax2.set_xlabel("둘째 집단의 성공 수 $R$")
ax2.set_ylabel("p-값 (로그 눈금)")
ax2.set_title("증거가 쌓일수록 왈드의 p-값이 나빠진다", fontsize=12, color=INK)
ax2.legend(loc="lower left", fontsize=10.5, framealpha=0.95)
ax2.grid(alpha=0.25, which="both")

fig.text(0.5, 0.015,
         f"두 집단 각 $N = {N}$, 기준 집단 성공 $R_0 = {R0}$ 으로 고정하고 $R$ 만 민다. "
         "모의실험이 아니라 2x2 표에서 바로 계산한 값이다.",
         ha="center", fontsize=10.5, color=MUTED)

fig.tight_layout(rect=(0, 0.045, 1, 0.955))
path = OUT + "hauck_donner.png"
fig.savefig(path, dpi=150)
print("saved:", path)

print(f"\n  W^2 꼭짓점: R = {peak}  (W^2 = {W2.max():.2f})")
for R in (70, 91, 99):
    b2, w2, lam = stats(R)
    print(f"  R = {R:3d}  로그오즈비 {b2:.4f}  오즈비 {np.exp(b2):6.2f}  "
          f"W^2 {w2:6.2f} (p {chi2_sf(w2)[0]:.2e})  Lambda {lam:6.2f} (p {chi2_sf(lam)[0]:.2e})")
