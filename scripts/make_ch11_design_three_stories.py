r"""11.2절 동질성 검정 §1 의 그림 — 같은 표, 세 가지 표집 설계.

만드는 파일:
  ch11/test/img/same_table_three_designs.png
      돌과 힐(1950) 남성 표를 세 번 그리고, 설계마다 연구자가 고정한 합을 칠한다.
      아래 줄은 그 설계에서 말해도 되는 비율과 말할 수 없는 비율이다.

실행:  python3 scripts/make_ch11_design_three_stories.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
"""
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
FIX, FIX_F = "#E65100", "#FFE0B2"
OK, NO = "#2E7D32", "#C62828"
MUTED = "#90A4AE"

OUT = "docs/ch11/test/img/"

# 돌과 힐(1950) Table IV, 남성. 행 = 폐암 / 대조, 열 = 흡연자 / 비흡연자
T = np.array([[647, 2], [622, 27]])
R, C, N = T.sum(1), T.sum(0), T.sum()
chi2 = stats.chi2_contingency(T, correction=False)[0]
OR = T[0, 0] * T[1, 1] / (T[0, 1] * T[1, 0])

DESIGNS = [
    ("① 남성 1,298 명을 무작위로", "총합만 고정 → 독립성 검정", "total",
     [("흡연자 중 폐암  647/1269 = 0.51", True),
      ("폐암 중 흡연자  647/649 = 0.997", True)]),
    ("② 환자 649 명 + 대조 649 명 (실제)", "행 합 고정 → 동질성 검정", "rows",
     [("폐암 중 흡연자  0.997 대 대조 0.958", True),
      ("흡연자 중 폐암  0.51 — 연구자가 정한 수", False)]),
    ("③ 흡연자 1,269 명 · 비흡연자 29 명 추적", "열 합 고정 → 동질성 검정", "cols",
     [("흡연자 중 폐암  0.51 대 비흡연자 0.069", True),
      ("폐암 중 흡연자  0.997 — 연구자가 정한 수", False)]),
]


def draw(ax, title, sub, fixed, lines):
    ax.set_xlim(-1.6, 3.2)
    ax.set_ylim(-2.6, 3.4)
    ax.axis("off")
    ax.text(0.8, 3.15, title, ha="center", va="center", fontsize=12.5,
            color=INK, weight="bold")
    ax.text(0.8, 2.65, sub, ha="center", va="center", fontsize=11.5, color=FIX)
    # 칸: x 0..2 (두 열), 합 열 x 2..3 ; y 0..2 (두 행), 합 행 y -1..0
    cols = ["흡연자", "비흡연자", "합"]
    rows = ["폐암", "대조", "합"]
    for j, c in enumerate(cols):
        ax.text(j + 0.5, 2.2, c, ha="center", va="center", fontsize=11, color=INK)
    for i, r in enumerate(rows):
        ax.text(-0.15, 1.5 - i, r, ha="right", va="center", fontsize=11, color=INK)
    vals = [[T[0, 0], T[0, 1], R[0]], [T[1, 0], T[1, 1], R[1]],
            [C[0], C[1], N]]
    for i in range(3):
        for j in range(3):
            x, y = j, 1 - i
            is_fix = ((fixed == "rows" and j == 2 and i < 2)
                      or (fixed == "cols" and i == 2 and j < 2)
                      or (i == 2 and j == 2))
            fc = FIX_F if is_fix else ("#F5F7F8" if (i == 2 or j == 2) else "white")
            ec = FIX if is_fix else MUTED
            lw = 2.2 if is_fix else 1.0
            ax.add_patch(Rectangle((x, y), 1, 1, fc=fc, ec=ec, lw=lw,
                                   zorder=3 if is_fix else 2))
            ax.text(x + 0.5, y + 0.5, f"{vals[i][j]:,}", ha="center",
                    va="center", fontsize=12.5, zorder=4,
                    color=FIX if is_fix else INK,
                    weight="bold" if is_fix else "normal")
    for k, (txt, good) in enumerate(lines):
        y = -1.55 - 0.62 * k
        ax.text(-1.45, y, "○" if good else "×", ha="left", va="center",
                fontsize=13, color=OK if good else NO, weight="bold")
        ax.text(-1.05, y, txt, ha="left", va="center", fontsize=10.5,
                color=OK if good else NO)


fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
for ax, d in zip(axes, DESIGNS):
    draw(ax, *d)
fig.text(0.5, 0.015,
         f"세 설계 모두  X² = {chi2:.2f} (자유도 1),  오즈비 = {OR:.1f}  —  "
         "검정은 같은 답을 내고, 말해도 되는 비율이 다르다",
         ha="center", fontsize=12, color=INK)
fig.subplots_adjust(left=0.02, right=0.99, top=0.97, bottom=0.08, wspace=0.12)
fig.savefig(OUT + "same_table_three_designs.png", dpi=170,
            facecolor="white", bbox_inches="tight")
print("saved", OUT + "same_table_three_designs.png", f"X2={chi2:.4f} OR={OR:.4f}")
