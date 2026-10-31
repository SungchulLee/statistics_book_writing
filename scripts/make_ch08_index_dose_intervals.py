#!/usr/bin/env python3
"""8장 개요의 그림 — 네 용량의 신뢰구간을 세로로 맞대면 결정이 읽힌다.

왼쪽: SURMOUNT-1 이 보고한 네 구간. 5 mg 과 10 mg 은 닿지도 않고
      10 mg 과 15 mg 은 겹친다.
오른쪽: **겹침으로 판단하면 안 된다.** 겹치는지가 아니라 차이 자체의 구간을
      보아야 한다. 10 mg -> 15 mg 은 구간이 겹치는데도 차이의 구간이 0 을
      비켜 간다 — 겹침만 보고 「더 준 값이 없었다」고 읽으면 반대로 답하게
      된다. 8.1절 오차막대 그림이 그 습관을 따로 다룬다.

자료: Jastreboff 외, 『New England Journal of Medicine』 387(2022) 205 쪽 초록.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글이 두부로 깨지지 않게, 그리고 축의 음수 기호가 제대로 나오게
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch08/img/"

# (이름, 점추정값, 아래끝, 위끝) — 논문 초록이 보고한 값
ARMS = [
    ("위약",    -3.1,  -4.3,  -1.9),
    ("5 mg",   -15.0, -15.9, -14.2),
    ("10 mg",  -19.5, -20.4, -18.5),
    ("15 mg",  -20.9, -21.8, -19.9),
]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                             gridspec_kw={"width_ratios": [1.45, 1]})

# ── 왼쪽: 네 구간을 세로로 맞댄다
# 두 띠를 먼저 깔아 자료 아래에 둔다
gap_lo, gap_hi = -18.5, -15.9          # 10 mg 위끝과 5 mg 아래끝 — 비어 있다
ov_lo, ov_hi = -20.4, -19.9            # 10 mg 과 15 mg 이 겹치는 구간
ax.axvspan(gap_lo, gap_hi, color="#2e7d32", alpha=0.13, zorder=1)
ax.axvspan(ov_lo, ov_hi, color="#b8860b", alpha=0.28, zorder=1)

LABEL_X = 0.8                           # 수를 적는 칸을 따로 둔다
ys = range(len(ARMS))
for y, (name, est, lo, hi) in zip(ys, ARMS):
    color = "#c0392b" if name == "위약" else "#1f4e79"
    ax.plot([lo, hi], [y, y], color=color, lw=3.2, solid_capstyle="butt", zorder=3)
    ax.plot([est], [y], "o", color=color, ms=9, zorder=4)
    for end in (lo, hi):
        ax.plot([end, end], [y - 0.14, y + 0.14], color=color, lw=2.0, zorder=3)
    ax.text(LABEL_X, y, f"{est}   ({lo}, {hi})", va="center", ha="left",
            fontsize=10.5, color=color)
ax.text(LABEL_X, 3.45, "점추정값과 95% 구간", va="center", ha="left",
        fontsize=9.5, color="#555555")

# 띠의 뜻은 지시선 없이 바로 위아래에 적는다 — 사선이 자료를 가로지르지 않게
ax.text((gap_lo + gap_hi) / 2, 3.42, "닿지도 않는다\n이쪽은 겹침만으로 말할 수 있다",
        ha="center", va="center", fontsize=10, color="#2e7d32", linespacing=1.35)
ax.text(-23.0, -0.42, "겹친다 — 그래서 오른쪽을 보아야 한다",
        ha="left", va="center", fontsize=10, color="#8a6508")

ax.set_yticks(list(ys))
ax.set_yticklabels([a[0] for a in ARMS], fontsize=11.5)
ax.set_xlim(-23.4, 9.6)
ax.set_ylim(-0.95, 3.85)
ax.set_xticks([-20, -15, -10, -5, 0])
ax.set_xlabel("72 주 뒤 체중 변화 (퍼센트)", fontsize=11)
ax.set_title("(가) 네 구간을 세로로 읽으면 결정이 읽힌다", fontsize=12.5, pad=10)
ax.grid(axis="x", alpha=0.25)
ax.set_axisbelow(True)

# ── 오른쪽: 겹침이 아니라 차이 자체의 구간을 본다
# 보고된 구간에서 표준오차를 되짚고(폭 = 2 x 1.96 x SE) 이웃한 두 팔의 차와
# 그 구간을 만든다. 두 팔은 서로 다른 사람이므로 분산이 더해진다.
import numpy as np
from scipy import stats

Z = stats.norm.ppf(0.975)
se = {n: (hi - lo) / 2 / Z for n, _, lo, hi in ARMS}
pt = {n: e for n, e, _, _ in ARMS}

STEPS = [("위약 → 5 mg", "위약", "5 mg"),
         ("5 → 10 mg", "5 mg", "10 mg"),
         ("10 → 15 mg", "10 mg", "15 mg")]
rows = []
for label, a_, b_ in STEPS:
    d = pt[a_] - pt[b_]                      # 한 단계 올려 더 줄어든 퍼센트포인트
    sd = np.hypot(se[a_], se[b_])
    rows.append((label, d, d - Z * sd, d + Z * sd))

yy = np.arange(len(rows))[::-1]
for y, (label, d, lo, hi) in zip(yy, rows):
    c = "#1f4e79" if lo > 0 else "#c0392b"
    bx.plot([lo, hi], [y, y], color=c, lw=3.2, solid_capstyle="butt", zorder=3)
    bx.plot([d], [y], "o", color=c, ms=9, zorder=4)
    bx.text(hi + 0.4, y, f"{d:.1f}   ({lo:.2f}, {hi:.2f})", va="center",
            ha="left", fontsize=10.5, color=c)

bx.axvline(0, color="#c0392b", ls="--", lw=1.8, zorder=2)
bx.text(0.3, len(rows) - 0.42, "0 — 더 준 값이 없었다는 뜻", ha="left",
        va="bottom", fontsize=10, color="#c0392b")
last = rows[-1]
bx.annotate(f"구간은 겹치는데\n차이의 구간은 0 을 비켜 간다 ({last[2]:.2f})",
            (last[2], yy[-1]), textcoords="offset points", xytext=(26, -34),
            ha="left", fontsize=10.5, color="#1f4e79", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#1f4e79", lw=1.3))

bx.set_yticks(list(yy))
bx.set_yticklabels([r[0] for r in rows], fontsize=11)
bx.set_xlim(-1.4, 17.5)
bx.set_ylim(-0.85, len(rows) - 0.25)
bx.set_xlabel("한 단계 올려 더 줄어든 양 (퍼센트포인트)", fontsize=11)
bx.set_title("(나) 겹침이 아니라 차이의 구간을 본다", fontsize=12.5, pad=10)
bx.grid(axis="x", alpha=0.25)
bx.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT + "dose_intervals.png", dpi=170, bbox_inches="tight", facecolor="white")

print("saved", OUT + "dose_intervals.png")
print(f"5 mg 위끝 {ARMS[1][3]} 와 10 mg 아래끝 {ARMS[2][2]} 사이 빈 구간 폭 "
      f"{abs(ARMS[2][2] - ARMS[1][3]):.1f}")
print(f"10 mg 과 15 mg 이 겹치는 구간 ({ov_lo}, {ov_hi}) 폭 {ov_hi - ov_lo:.1f}")
for label, d, lo, hi in rows:
    print(f"  {label:12s} {d:5.2f}  ({lo:6.3f}, {hi:6.3f})  "
          f"0 을 {'비켜 간다' if lo > 0 else '담는다'}")
