"""9.1절 유의수준과 판정 규칙 — 0.049 와 0.051 을 가르는 벽의 두께.

들머리가 묻는 것을 그린다. p = 0.049 로 끝난 연구와 0.051 로 끝난 연구의
자료는 얼마나 다른가. 표준오차를 1 로 두면 추정값이 1.9686 과 1.9515 로
0.0171 표준오차만큼 다르다. 구간은 하나가 0.0086 만큼 0 을 비키고 다른
하나가 0.0085 만큼 0 을 담는다.

쪽의 p005 와 alpha_debate 는 베이즈 인자와 사후확률을 다루고
rejection_region_demo_51 은 기각역을 그린다. 여기서는 문턱 양쪽에 선 두
연구의 자료만 맞댄다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
Z = stats.norm.ppf(0.975)
ROWS = [(0.049, "논문에 실린다", "tab:blue"),
        (0.051, "서랍에 들어간다", "tab:red")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})
for i, (p, fate, col) in enumerate(ROWS):
    est = stats.norm.isf(p / 2)
    lo, hi = est - Z, est + Z
    y = 1 - i
    ax1.plot([lo, hi], [y, y], "-", lw=7.0, color=col, solid_capstyle="butt")
    for x in (lo, hi):
        ax1.plot([x, x], [y - 0.1, y + 0.1], "-", lw=3.5, color=col)
    ax1.plot([est], [y], "o", ms=10, color="white", mec=col, mew=2.5)
    ax1.text(est, y + 0.19, f"추정값 {est:.4f}", ha="center", fontsize=11,
             color=col, fontweight="bold")
    ax1.text(hi + 0.08, y, f"$p$ = {p}\n{fate}", ha="left", va="center",
             fontsize=11, color=col, fontweight="bold")
    ax1.text(lo - 0.08, y, f"아래끝 {lo:+.4f}", ha="right", va="center",
             fontsize=10.5, color=col)
ax1.axvline(0, color="0.15", lw=2.2)
ax1.text(0.03, 1.42, "효과 없음 = 0", fontsize=11, color="0.15",
         fontweight="bold")
ax1.set_xlim(-0.55, 5.1)
ax1.set_ylim(-0.6, 1.62)
ax1.set_yticks([])
ax1.set_xlabel("효과 (표준오차 단위)", fontsize=11)
ax1.set_title("문턱 양쪽에 선 두 연구", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)

est1, est2 = stats.norm.isf(0.049 / 2), stats.norm.isf(0.051 / 2)
lo1, lo2 = est1 - Z, est2 - Z
ax2.axvspan(lo2, lo1, color="0.85")
for x, c, lab in ((lo2, "tab:red", f"{lo2:+.4f}"), (lo1, "tab:blue", f"{lo1:+.4f}")):
    ax2.axvline(x, color=c, lw=2.6)
    ax2.text(x, 0.72, lab, ha="center", fontsize=12, color=c,
             fontweight="bold")
ax2.axvline(0, color="0.15", lw=2.0, ls="--")
ax2.annotate("", xy=(lo1, 0.3), xytext=(lo2, 0.3),
             arrowprops=dict(arrowstyle="<->", color="0.2", lw=2.0))
ax2.text((lo1 + lo2) / 2, 0.38, f"벽의 두께\n{lo1 - lo2:.4f}", ha="center",
         va="bottom", fontsize=12, color="0.15", fontweight="bold")
ax2.text(0.5, 0.05, "구간의 아래끝을 확대한 것이다",
         transform=ax2.transAxes, ha="center", fontsize=10.5, color="0.45")
ax2.set_xlim(-0.045, 0.045)
ax2.set_ylim(-0.75, 1.0)
ax2.set_yticks([])
ax2.set_xlabel("효과 (표준오차 단위)", fontsize=11)
ax2.set_title("한쪽은 0 을 비키고 한쪽은 0 을 담는다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("운명을 가른 것은 자료가 아니라 거기 그어 둔 문턱이다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "threshold_wall.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for p, fate, _ in ROWS:
    est = stats.norm.isf(p / 2)
    print(f"  p={p}: 추정값 {est:.4f} · 구간 ({est-Z:+.4f}, {est+Z:.4f}) · {fate}")
print(f"  두 추정값의 차 {est1-est2:.4f} 표준오차 · 아래끝의 차 {lo1-lo2:.4f}")
