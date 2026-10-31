"""8.4절 대응 비율의 구간 — 같은 2 퍼센트포인트, 다른 속.

들머리의 요점을 그린다. 47% 에서 49% 로 올랐다는 사실만으로는 그 안에서
무슨 일이 있었는지 알 수 없다. 같은 가장자리 합을 내는 세상이 여럿이고,
매번 새 사람을 뽑는 조사는 그 가장자리 합만 본다.

왼쪽은 같은 47/49 를 내는 두 세상의 2x2 표다. 오른쪽은 그 둘에 구간을
붙인 것인데, **패널이 늘 이기는 것이 아니다** --- 마음을 바꾼 사람이 적은
세상에서는 훨씬 좁고, 절반 넘게 바꾼 세상에서는 오히려 넓다.

쪽의 mcnemar_ci_width 는 표 하나에서 불일치 칸만 강조하고,
panel_discordant 는 바꾼 비율에 따른 반폭 곡선을 그린다. 여기서는 그
곡선 위의 두 점을 표로 세워 **가장자리 합이 같다**는 것을 보인다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.patches import Rectangle
from scipy import stats

OUT = "docs/ch08/paired_sample_intervals/img/"

N = 600
Z = stats.norm.ppf(0.975)

# b = 지난달 반대 -> 이번달 찬성, c = 그 반대. 둘 다 순변화 b - c = 12 를 낸다.
WORLDS = [("거의 움직이지 않은 세상", 25, 13),
          ("많이 뒤섞인 세상", 180, 168)]


def table(b, c):
    """지난달 47%(282 명), 이번달 49%(294 명) 에 맞춘 2x2 표."""
    a = 282 - c                 # 두 달 다 찬성
    d = N - a - b - c           # 두 달 다 반대
    return np.array([[a, c], [b, d]], float)


def paired_ci(b, c):
    d = (b - c) / N
    se = np.sqrt(b + c - (b - c) ** 2 / N) / N
    return 100 * d, 100 * (d - Z * se), 100 * (d + Z * se)


p_ind = 282 / N
se_ind = np.sqrt(2 * p_ind * (1 - p_ind) / N)
ind = (2.0, 2.0 - 100 * Z * se_ind, 2.0 + 100 * Z * se_ind)

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5),
                         gridspec_kw={"width_ratios": [1, 1, 1.5]})

# ── (좌·중) 같은 가장자리 합을 내는 두 표 ──────────────────────────
for ax, (name, b, c) in zip(axes[:2], WORLDS):
    O = table(b, c)
    cells = [((0, 1), O[0, 0], "0.80"), ((1, 1), O[0, 1], "tab:orange"),
             ((0, 0), O[1, 0], "tab:orange"), ((1, 0), O[1, 1], "0.80")]
    for (x, y), v, col in cells:
        ax.add_patch(Rectangle((x, y), 0.96, 0.96, facecolor=col, alpha=0.75,
                               edgecolor="white", lw=2))
        ax.text(x + 0.48, y + 0.48, f"{v:.0f}", ha="center", va="center",
                fontsize=15, fontweight="bold", color="0.1")
    ax.text(0.48, 2.12, "이번 달 찬성", ha="center", fontsize=10.5,
            color="0.35")
    ax.text(1.48, 2.12, "이번 달 반대", ha="center", fontsize=10.5,
            color="0.35")
    ax.text(-0.09, 1.48, "지난달\n찬성", ha="right", va="center",
            fontsize=10.5, color="0.35")
    ax.text(-0.09, 0.48, "지난달\n반대", ha="right", va="center",
            fontsize=10.5, color="0.35")
    ax.text(0.98, -0.30, f"마음을 바꾼 사람 {b + c:.0f} 명 (주황 두 칸)",
            ha="center", fontsize=10.5, color="tab:orange",
            fontweight="bold")
    ax.set_xlim(-0.95, 2.05)
    ax.set_ylim(-0.62, 2.45)
    ax.axis("off")
    ax.set_title(name, fontsize=12.5)

axes[0].text(0.98, -0.56, "지난달 찬성 282 명 · 이번 달 찬성 294 명",
             ha="center", fontsize=10, color="0.45")
axes[1].text(0.98, -0.56, "가장자리 합이 왼쪽과 똑같다", ha="center",
             fontsize=10, color="0.45", fontweight="bold")

# ── (우) 세 가지 구간 ──────────────────────────────────────────────
ax = axes[2]
rows = [("매번 새 600 명씩", ind, "tab:red")]
rows += [(f"같은 600 명 — {n}", paired_ci(b, c), "tab:blue")
         for n, b, c in WORLDS]
for i, (name, (m, lo, hi), col) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([lo, hi], [y, y], "-", lw=6.0, color=col, solid_capstyle="butt")
    for x in (lo, hi):
        ax.plot([x, x], [y - 0.13, y + 0.13], "-", lw=3.0, color=col)
    ax.plot([m], [y], "o", ms=8, color="white", mec=col, mew=2.2)
    ax.text(hi + 0.25, y, f"({lo:.2f}, {hi:.2f})", ha="left", va="center",
            fontsize=10.5, color=col)
    ax.text(lo - 0.25, y, name, ha="right", va="center", fontsize=10.5,
            color=col, fontweight="bold")
ax.axvline(0, color="0.15", lw=2.0)
ax.text(0.15, len(rows) - 0.45, "변화 없음", fontsize=10.5, color="0.15")
ax.set_xlim(-9.5, 13.5)
ax.set_ylim(-0.7, len(rows) - 0.25)
ax.set_yticks([])
ax.set_xlabel("찬성 비율의 변화 (퍼센트포인트)", fontsize=11)
ax.set_title("세 설계가 같은 2 퍼센트포인트에 붙이는 구간", fontsize=12.5)
ax.grid(axis="x", alpha=0.25)

fig.suptitle("순변화 2 퍼센트포인트는 같은데 그 안에서 일어난 일이 다르다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "same_margin_two_worlds.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n매번 새 {N} 명씩: {ind[0]:.2f} ({ind[1]:.4f}, {ind[2]:.4f}) 퍼센트포인트")
for name, b, c in WORLDS:
    m, lo, hi = paired_ci(b, c)
    O = table(b, c)
    print(f"{name}: b={b} c={c} 불일치 {b+c}  표 {O.astype(int).tolist()}")
    print(f"   {m:.2f} ({lo:.4f}, {hi:.4f}) 퍼센트포인트 · 반폭 "
          f"{(hi-lo)/2:.4f}  (새 표본의 {(hi-lo)/(ind[2]-ind[1]):.2f} 배)")
