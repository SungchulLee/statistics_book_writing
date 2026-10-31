"""8.2절 μ의 신뢰구간 — 교과서의 37.0 은 구간 안에 있는가.

들머리가 적은 네 구간을 그대로 그린다. 마코위악 외(1992)의 초록에서 평균
36.8 과 정상 상한 37.7 을 읽고, 상한을 "평균 + 1.96 x 산포" 로 되짚어
산포 0.46 을 얻은 뒤(이 되짚기는 이 책의 가정이다), 네 가지로 구간을
만든다 --- n = 148 과 n = 5, 각각 sigma 를 모를 때와 알 때.

오른쪽은 그 다음 물음이다. 다섯 명으로는 37.0 을 배제하지 못했는데,
같은 산포라면 몇 명을 재야 배제할 수 있는가. 쪽의 1 절 그림
t_vs_z_cost 가 임계값의 비를 n 에 따라 그리므로 여기서는 그리지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/one_sample_intervals/img/"

XBAR = 36.8        # 초록의 평균
SIG = 0.46         # 상한 37.7 을 1.96 배로 되짚은 산포
BOOK = 37.0        # 백이십 년 동안 교과서가 적어 온 수
Z = stats.norm.ppf(0.975)

ROWS = []
for n in (148, 5):
    se = SIG / np.sqrt(n)
    for known, crit in (("σ 를 모를 때 (t 구간)", stats.t.ppf(0.975, n - 1)),
                        ("σ 를 알 때 (z 구간)", Z)):
        h = crit * se
        ROWS.append((n, known, crit, XBAR - h, XBAR + h, 2 * h))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                               gridspec_kw={"width_ratios": [1.35, 1]})

# ── (좌) 네 구간과 37.0 ────────────────────────────────────────────
ys = [3, 2.3, 1.0, 0.3]
for y, (n, lab, crit, lo, hi, w) in zip(ys, ROWS):
    col = "tab:blue" if n == 148 else "tab:orange"
    ax1.plot([lo, hi], [y, y], "-", lw=5.0, color=col, solid_capstyle="butt")
    for x in (lo, hi):
        ax1.plot([x, x], [y - 0.09, y + 0.09], "-", lw=3.0, color=col)
    ax1.plot([XBAR], [y], "o", ms=7, color="white", mec=col, mew=2.0)
    ax1.text((lo + hi) / 2, y + 0.13, f"({lo:.4f}, {hi:.4f})", ha="center",
             va="bottom", fontsize=10, color=col)
    ax1.text(hi + 0.02, y, f"폭 {w:.4f}", ha="left", va="center",
             fontsize=10, color="0.35")
ax1.text(36.15, 3.60, "건강한 성인 148 명", fontsize=11.5, fontweight="bold",
         color="tab:blue")
ax1.text(36.15, 1.58, "같은 산포로 5 명만 재었다면", fontsize=11.5,
         fontweight="bold", color="tab:orange")
for y, (n, lab, *_rest) in zip(ys, ROWS):
    ax1.text(36.15, y - 0.24, lab, fontsize=9.5, color="0.4")
ax1.axvline(BOOK, color="tab:red", lw=2.2)
ax1.text(BOOK + 0.02, 3.90, f"교과서의 {BOOK:.1f} °C", fontsize=11.5,
         color="tab:red", fontweight="bold")
ax1.axvline(XBAR, color="0.45", ls="--", lw=1.4)
ax1.text(XBAR, -0.42, f"표본평균 {XBAR}", ha="center", va="top", fontsize=10,
         color="0.35")
ax1.annotate("", xy=(BOOK, 2.72), xytext=(ROWS[0][4], 2.72),
             arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.5))
ax1.text(BOOK + 0.03, 2.72, "37.0 이 밖에 있다", ha="left", va="center",
         fontsize=10.5, color="tab:red", fontweight="bold")
ax1.text(BOOK + 0.03, 0.66, "둘 다 37.0 을 담는다", ha="left",
         fontsize=10.5, color="tab:orange", fontweight="bold")
ax1.set_xlim(36.12, 37.74)
ax1.set_ylim(-0.75, 4.15)
ax1.set_yticks([])
ax1.set_xlabel("체온 (°C)", fontsize=11)
ax1.set_title("같은 산포, 다른 표본크기 — 네 구간", fontsize=12.5)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 몇 명을 재야 37.0 을 배제하나 ─────────────────────────────
gap = BOOK - XBAR
ns = np.arange(3, 201)
half = stats.t.ppf(0.975, ns - 1) * SIG / np.sqrt(ns)
need = int(ns[half < gap][0])
ax2.plot(ns, half, color="tab:purple", lw=2.2)
ax2.axhline(gap, color="tab:red", lw=1.8, ls="--")
ax2.text(200, gap + 0.012, f"교과서까지의 거리 {gap:.1f} °C", ha="right",
         fontsize=10.5, color="tab:red", fontweight="bold")
ax2.axvline(need, color="0.3", lw=1.4, ls=":")
ax2.plot([need], [stats.t.ppf(0.975, need - 1) * SIG / np.sqrt(need)], "o",
         ms=9, color="tab:purple")
ax2.annotate(f"{need} 명부터 37.0 이 구간 밖으로 나간다",
             xy=(need, gap), xytext=(need + 14, 0.40), fontsize=11,
             fontweight="bold", color="tab:purple",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.6))
for n, c in ((5, "tab:orange"), (148, "tab:blue")):
    h = stats.t.ppf(0.975, n - 1) * SIG / np.sqrt(n)
    ax2.plot([n], [h], "o", ms=8, color=c)
    ax2.annotate(f"n = {n}\n반폭 {h:.4f}", xy=(n, h),
                 xytext=(n + (10 if n == 5 else -8), h + 0.055),
                 ha="left" if n == 5 else "right", fontsize=10, color=c,
                 fontweight="bold")
ax2.set_xlim(0, 205)
ax2.set_ylim(0, 0.70)
ax2.set_xlabel("잰 사람 수 n", fontsize=11)
ax2.set_ylabel("구간의 반폭 (°C)", fontsize=11)
ax2.set_title("몇 명을 재야 교과서를 반박할 수 있나", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("같은 산포인데 표본크기가 결론을 가른다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "body_temp_intervals.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n평균 {XBAR}, 되짚은 산포 {SIG}, 교과서 {BOOK}")
for n, lab, crit, lo, hi, w in ROWS:
    print(f"  n={n:3d}  {lab:<22s} 임계값 {crit:.4f}  ({lo:.4f}, {hi:.4f})  "
          f"폭 {w:.4f}  37.0 을 {'담는다' if lo <= BOOK <= hi else '담지 않는다'}")
w148 = [r[5] for r in ROWS if r[0] == 148]
w5 = [r[5] for r in ROWS if r[0] == 5]
print(f"  폭의 차: n=148 에서 {100*(w148[0]/w148[1]-1):.2f}% · "
      f"n=5 에서 {100*(w5[0]/w5[1]-1):.1f}%")
print(f"  37.0 을 배제하려면 n >= {need} (반폭 "
      f"{stats.t.ppf(0.975, need-1)*SIG/np.sqrt(need):.4f} < {gap:.1f})")
