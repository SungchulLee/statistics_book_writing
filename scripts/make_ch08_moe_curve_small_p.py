"""8.2절 p의 신뢰구간 — 발표문의 +-3.1 은 꼭대기 값이다.

들머리가 든 조사(전국 1,000 명)를 그대로 쓴다. 발표문 끝줄의 오차한계는
지지율이 50% 일 때의 값이고, 오차는 바로 거기서 가장 크다. 그래서 그 한
줄을 표의 모든 수에 갖다 대면 작은 수에서 터무니없이 커지고, 3% 짜리
후보에서는 아래끝이 0 아래로 넘어간다.

오른쪽은 그 자리에 제대로 된 구간 넷을 세운 것이다. 쪽의 2 절 그림
four_methods_tradeoff 는 n = 50, k = 12 라는 다른 자료를 쓰므로 겹치지
않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/one_sample_intervals/img/"

N = 1000
ANN = 3.1          # 발표문 끝줄의 오차한계 (퍼센트포인트)
Z = stats.norm.ppf(0.975)


def moe(p, n=N):
    return 100 * Z * np.sqrt(p * (1 - p) / n)


def four(k, n=N):
    """같은 자료에 네 방법. 모두 퍼센트로 돌려준다."""
    ph = k / n
    h = Z * np.sqrt(ph * (1 - ph) / n)
    wald = (ph - h, ph + h)
    d = Z * Z / n
    c = (ph + d / 2) / (1 + d)
    hw = (Z / (1 + d)) * np.sqrt(ph * (1 - ph) / n + d / (4 * n))
    wilson = (c - hw, c + hw)
    nt = n + Z * Z
    pt = (k + Z * Z / 2) / nt
    ht = Z * np.sqrt(pt * (1 - pt) / nt)
    ac = (max(pt - ht, 0.0), pt + ht)
    cp = (stats.beta.ppf(0.025, k, n - k + 1),
          stats.beta.ppf(0.975, k + 1, n - k))
    return [("왈드 z-구간", wald), ("윌슨", wilson),
            ("아그레스티–코울", ac), ("클로퍼–피어슨", cp)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# ── (좌) 오차한계는 50% 에서 가장 크다 ──────────────────────────────
g = np.linspace(0.001, 0.999, 999)
ax1.plot(100 * g, moe(g), color="tab:blue", lw=2.4)
ax1.axhline(ANN, color="tab:red", ls="--", lw=1.8)
ax1.text(99, ANN + 0.07, f"발표문의 ±{ANN} 퍼센트포인트", ha="right",
         fontsize=11, color="tab:red", fontweight="bold")
SPOT = ((0.50, "50% 자리", "0.3", (16, -0.45), "left"),
        (0.44, "44% 후보", "tab:green", (-17, -0.75), "right"),
        (0.03, "3% 후보", "tab:orange", (7, -0.55), "left"))
for p, name, c, (dx, dy), ha in SPOT:
    m = moe(p)
    ax1.plot([100 * p], [m], "o", ms=9, color=c)
    ax1.annotate(f"{name}\n±{m:.2f}", xy=(100 * p, m),
                 xytext=(100 * p + dx, m + dy),
                 ha=ha, fontsize=10.5, color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 3.75)
ax1.set_xlabel("지지율 (%)", fontsize=11)
ax1.set_ylabel("오차한계 (퍼센트포인트)", fontsize=11)
ax1.set_title(f"오차한계는 50% 에서 가장 크다 (n = {N:,})", fontsize=12.5)
ax1.grid(alpha=0.25)

# ── (우) 3% 후보에 다섯 구간 ───────────────────────────────────────
k = 30
rows = [("발표문의 ±3.1 을 그대로", (0.03 - ANN / 100, 0.03 + ANN / 100))]
rows += four(k)
for i, (name, (lo, hi)) in enumerate(rows):
    y = len(rows) - 1 - i
    bad = lo < 0
    col = "tab:red" if bad else "tab:blue"
    ax2.plot([100 * lo, 100 * hi], [y, y], "-", lw=5.0, color=col,
             solid_capstyle="butt")
    for x in (lo, hi):
        ax2.plot([100 * x, 100 * x], [y - 0.14, y + 0.14], "-", lw=3.0,
                 color=col)
    ax2.text(100 * hi + 0.12, y, f"({100*lo:.2f}, {100*hi:.2f})", ha="left",
             va="center", fontsize=10, color=col)
    ax2.text(100 * lo, y + 0.20, name, ha="left", va="bottom", fontsize=10.5,
             color=col, fontweight="bold" if bad else "normal")
ax2.axvline(0, color="0.15", lw=2.0)
ax2.text(0.12, len(rows) - 0.15, "득표율 0", fontsize=10.5, color="0.15")
ax2.plot([100 * rows[0][1][0]], [len(rows) - 1], "X", ms=13, mew=3,
         color="tab:red", zorder=5)
ax2.text(100 * rows[0][1][0] - 0.18, len(rows) - 1.0, "음수다", ha="right",
         va="center", fontsize=10.5, color="tab:red", fontweight="bold")
ax2.set_xlim(-1.5, 7.6)
ax2.set_ylim(-0.6, len(rows) + 0.15)
ax2.set_yticks([])
ax2.set_xlabel("득표율 (%)", fontsize=11)
ax2.set_title(f"3% 후보에 구간을 붙이면 (n = {N:,}, k = {k})", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("발표문 끝줄의 한 수를 표의 모든 수에 갖다 쓸 수 없다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "moe_curve_small_p.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N}, 95% 오차한계")
for p in (0.50, 0.44, 0.03):
    print(f"  p = {p:.2f} 에서 ±{moe(p):.4f} 퍼센트포인트")
print(f"\nk = {k} (3%) 에 다섯 구간")
for name, (lo, hi) in rows:
    print(f"  {name:<22s} ({100*lo:7.4f}, {100*hi:7.4f}) %"
          f"{'   <- 음수' if lo < 0 else ''}")
