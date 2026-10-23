r"""2.5절 점그림·줄기잎그림 쪽의 「그림으로 보는 통계학」 그림 — 시각표가 곧 줄기잎이다.

만드는 파일:
  docs/ch02/visualization/img/timetable_stemleaf.png

시각표의 수는 직접 만든 예다(아침·저녁 혼잡시간을 둔 통근선 한 방향).
실제 운행 시각이 아니다.

실행:  python3 scripts/make_ch02_timetable_stemleaf.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
MUTED, MUTED_F = "#90A4AE", "#ECEFF1"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

# 직접 만든 통근선 시각표 — 시(줄기)와 분(잎)
TT = {
    5: [12, 32, 52],
    6: [8, 20, 32, 44, 56],
    7: [2, 8, 14, 20, 26, 31, 36, 41, 46, 51, 56],
    8: [1, 6, 11, 16, 21, 26, 31, 37, 43, 49, 55],
    9: [3, 13, 25, 37, 49],
    10: [1, 21, 41],
    11: [1, 21, 41],
    12: [1, 21, 41],
    17: [2, 14, 26, 38, 50],
    18: [0, 6, 12, 18, 24, 30, 36, 42, 48, 54],
    19: [2, 12, 22, 34, 46, 58],
    20: [12, 32, 52],
    21: [12, 42],
    22: [12, 52],
}

hours = sorted(TT)
counts = np.array([len(TT[h]) for h in hours])

fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), width_ratios=[1.45, 1])

# ── ① 시각표 한 칸 ──────────────────────────────────────────────────────
ax = axes[0]
for i, h in enumerate(hours):
    y = len(hours) - i
    ax.add_patch(plt.Rectangle((-0.06, y - 0.36), 0.055, 0.72,
                               facecolor=MUTED_F, edgecolor=MUTED, lw=0.8))
    ax.text(-0.033, y, f"{h}", ha="center", va="center", fontsize=11,
            color=INK, fontweight="bold")
    # 잎의 수만큼 뒤에 옅은 막대를 깔아 둔다 — 줄의 길이가 곧 도수다
    ax.add_patch(plt.Rectangle((0.0, y - 0.3), len(TT[h]) / 11 * 0.98, 0.6,
                               facecolor=BLUE_F, edgecolor="none", zorder=0))
    for j, m in enumerate(TT[h]):
        ax.text(0.012 + j / 11 * 0.98, y, f"{m:02d}", ha="left", va="center",
                fontsize=10, color=BLUE, family="monospace")
ax.text(-0.033, len(hours) + 0.9, "시", ha="center", fontsize=10.5, color=INK)
ax.text(0.03, len(hours) + 0.9, "분", ha="left", fontsize=10.5, color=BLUE)
ax.set_xlim(-0.1, 1.02)
ax.set_ylim(0.3, len(hours) + 1.3)
ax.axis("off")
ax.set_title("① 시각표 — 시가 줄기, 분이 잎", fontsize=11, color=INK)

# ── ② 같은 표를 도수로 ──────────────────────────────────────────────────
ax = axes[1]
y = np.arange(len(hours), 0, -1)
ax.barh(y, counts, height=0.6, color=ORANGE_F, edgecolor=ORANGE, lw=1.0)
for yy, c in zip(y, counts):
    ax.text(c + 0.15, yy, f"{c}", va="center", fontsize=9.5, color=ORANGE)
ax.set_yticks(y)
ax.set_yticklabels([f"{h} 시" for h in hours], fontsize=9.5, color=INK)
ax.set_xlabel("그 시간대의 열차 수", fontsize=9.5, color=INK)
ax.set_xlim(0, 13)
ax.set_title("② 줄의 길이를 옮겨 적으면 히스토그램", fontsize=11, color=INK)
ax.tick_params(labelsize=9, colors=INK, length=3)
ax.spines[["top", "right"]].set_visible(False)
for s in ax.spines.values():
    s.set_color(MUTED)

fig.tight_layout()
path = OUT + "timetable_stemleaf.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
print(f"시각표의 열차 {counts.sum()} 편, 시간대 {len(hours)} 개, "
      f"가장 붐비는 시간 {hours[int(counts.argmax())]} 시 {counts.max()} 편")
