#!/usr/bin/env python3
"""8장 개요의 그림 — 네 용량의 신뢰구간을 세로로 맞대면 결정이 읽힌다.

왼쪽: SURMOUNT-1 이 보고한 네 구간. 5 mg 과 10 mg 은 겹치는 데가 없고
      10 mg 과 15 mg 은 겹친다. 용량을 올릴지를 이 그림이 답한다.
오른쪽: 보고된 폭을 1/sqrt(n) 으로 환산해 본 것. 환자가 적었다면
      괄호가 넓어지고, 넓은 괄호는 「아직 모른다」는 뜻이다.

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
ax.text((gap_lo + gap_hi) / 2, 3.42, "겹치지 않는다\n용량을 올린 값이 있었다",
        ha="center", va="center", fontsize=10, color="#2e7d32", linespacing=1.35)
ax.text(-23.0, -0.42, "겹친다 — 더 올린 값은 자료가 받쳐 주지 않는다",
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

# ── 오른쪽: 폭은 표본이 정한다
EST, HALF = -20.9, (-19.9 - (-21.8)) / 2      # 보고된 반폭 0.95
SCALES = [("환자가 4분의 1이면", 0.5, 2.0),    # 폭은 1/sqrt(n) 이므로 두 배
          ("보고된 시험", 1.0, 1.0),
          ("환자가 네 배면", 2.0, 0.5)]
for y, (label, _, mult) in zip(range(len(SCALES)), SCALES):
    h = HALF * mult
    on = label == "보고된 시험"
    color = "#1f4e79" if on else "#8c8c8c"
    bx.plot([EST - h, EST + h], [y, y], color=color,
            lw=4.0 if on else 3.0, solid_capstyle="butt", zorder=3)
    bx.plot([EST], [y], "o", color=color, ms=8, zorder=4)
    bx.text(EST, y + 0.26, f"폭 {2 * h:.2f} 퍼센트포인트", ha="center",
            fontsize=10, color=color)

bx.axvline(EST, color="#1f4e79", lw=0.9, ls=":", zorder=2)
bx.set_yticks(range(len(SCALES)))
bx.set_yticklabels([s[0] for s in SCALES], fontsize=11)
bx.set_xlim(EST - 2.6, EST + 2.6)
bx.set_ylim(-0.55, 2.62)
bx.set_xlabel("체중 변화 (퍼센트)", fontsize=11)
bx.set_title("(나) 넓은 괄호는 「아직 모른다」는 뜻이다", fontsize=12.5, pad=10)
bx.grid(axis="x", alpha=0.25)
bx.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT + "dose_intervals.png", dpi=150, bbox_inches="tight")
print("saved", OUT + "dose_intervals.png")

# 쪽에 적은 수가 그림과 맞는지 확인
print(f"5 mg 위끝 {ARMS[1][3]} 와 10 mg 아래끝 {ARMS[2][2]} 사이 빈 구간 폭 "
      f"{gap_hi - gap_lo:.1f}")
print(f"10 mg 과 15 mg 이 겹치는 구간 ({ov_lo}, {ov_hi}) 폭 {ov_hi - ov_lo:.1f}")
print(f"보고된 반폭 {HALF:.2f} -> 폭 {2 * HALF:.2f}")
