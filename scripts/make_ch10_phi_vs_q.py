r"""12.1 「점이연 상관과 파이 계수」 쪽의 상자에 들어가는 그림을 만든다.

연관의 세기를 오즈비로 고정해 놓고 주변비율만 움직이면, 파이 계수는 주저앉고
율의 Q 는 꼼짝하지 않는다. 1913 년에 피어슨과 율이 150 쪽에 걸쳐 싸운 것이 이것이다.

만드는 파일:
  ch12/correlation/img/phi_vs_q.png

실행:  python3 scripts/make_ch12_phi_vs_q.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch12/correlation/img/"
os.makedirs(OUT, exist_ok=True)


def cell_a(p, q, theta):
    """주변비율 p, q 와 오즈비 theta 를 모두 맞추는 P(X=1, Y=1).

    (theta-1)a^2 - a[(theta-1)(p+q) + 1] + theta*p*q = 0 의 알맞은 근."""
    if abs(theta - 1) < 1e-12:
        return p * q
    A = theta - 1.0
    B = -((theta - 1.0) * (p + q) + 1.0)
    C = theta * p * q
    disc = B * B - 4 * A * C
    r1 = (-B - np.sqrt(disc)) / (2 * A)
    r2 = (-B + np.sqrt(disc)) / (2 * A)
    lo, hi = max(0.0, p + q - 1.0), min(p, q)
    for r in (r1, r2):
        if lo - 1e-12 <= r <= hi + 1e-12:
            return min(max(r, lo), hi)
    raise ValueError("근을 찾지 못했다")


def measures(p, q, theta):
    a = cell_a(p, q, theta)
    b, c = p - a, q - a
    d = 1 - p - q + a
    phi = (a * d - b * c) / np.sqrt(p * (1 - p) * q * (1 - q))
    Q = (a * d - b * c) / (a * d + b * c)
    return phi, Q, a


Q_FIX = 0.5
THETAS = [3.0, 9.0, 25.0]
ps = np.linspace(0.02, 0.98, 481)

print(f"P(Y=1) = {Q_FIX} 로 고정하고 P(X=1) 을 움직인다")
print(f"{'theta':>6s} {'Q':>7s} | {'phi(p=.5)':>10s} {'phi(p=.1)':>10s} "
      f"{'phi(p=.02)':>11s}")
curves = {}
for th in THETAS:
    phis = np.array([measures(p, Q_FIX, th)[0] for p in ps])
    curves[th] = phis
    Q = (th - 1) / (th + 1)
    g = lambda v: measures(v, Q_FIX, th)[0]
    print(f"{th:6.0f} {Q:7.3f} | {g(0.5):10.3f} {g(0.1):10.3f} {g(0.02):11.3f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ------------------------------------------------ 왼쪽: 세 오즈비에서 phi 와 Q
for th, c in zip(THETAS, [BLUE, ORANGE, GREEN]):
    Q = (th - 1) / (th + 1)
    axL.plot(ps, curves[th], color=c, lw=2.4, zorder=3,
             label=f"$\\phi$  (오즈비 ${th:.0f}$)")
    axL.axhline(Q, color=c, lw=1.5, ls="--", zorder=2)
    axL.text(0.985, Q + 0.012, f"율의 $Q = {Q:.2f}$", color=c, fontsize=9,
             ha="right")
axL.axvline(Q_FIX, color=MUTED, lw=1.0, ls=":", zorder=1)
axL.text(0.515, 0.035, "주변이 서로 같은 자리", color=INK, fontsize=9,
         ha="left")
axL.set_xlim(0, 1)
axL.set_ylim(0, 1.0)
axL.set_xlabel(f"$P(X=1)$   ($P(Y=1) = {Q_FIX}$ 로 고정)")
axL.set_ylabel("연관의 세기")
axL.set_title("연관을 고정해 놓고 주변만 움직였다", color=INK, fontsize=12, pad=9)
axL.legend(loc="upper left", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------ 오른쪽: 표 셋을 직접 보여 준다
TH = 9.0
SHOW = [0.50, 0.20, 0.05]
axR.axis("off")
axR.set_title(f"오즈비가 모두 ${TH:.0f}$ 인 세 표", color=INK, fontsize=12, pad=9)
y0 = 0.90
for i, p in enumerate(SHOW):
    phi, Q, a = measures(p, Q_FIX, TH)
    b, c, d = p - a, Q_FIX - a, 1 - p - Q_FIX + a
    n = 1000
    cells = np.round(np.array([[a, b], [c, d]]) * n).astype(int)
    y = y0 - i * 0.325
    axR.text(0.00, y, f"$P(X=1) = {p:.2f}$", fontsize=10, color=INK,
             va="center", weight="bold")
    tbl = (f"  {cells[0,0]:4d}   {cells[0,1]:4d}\n"
           f"  {cells[1,0]:4d}   {cells[1,1]:4d}")
    axR.text(0.33, y, tbl, fontsize=10, color=INK, va="center",
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.35", fc="#F5F5F5", ec=MUTED))
    axR.text(0.66, y, f"$\\phi = {phi:.3f}$", fontsize=11, color=RED,
             va="center")
    axR.text(0.86, y, f"$Q = {Q:.3f}$", fontsize=11, color=GREEN, va="center")
axR.text(0.0, -0.06, "세 표에서 오즈비도 $Q$ 도 같다. 바뀐 것은 주변비율뿐인데\n"
                     "$\\phi$ 만 절반 아래로 주저앉는다.",
         fontsize=9.5, color=INK, va="top")

fig.tight_layout()
path = OUT + "phi_vs_q.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
