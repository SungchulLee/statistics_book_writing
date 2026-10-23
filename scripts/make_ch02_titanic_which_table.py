r"""2.2절 타이타닉 사례 연구의 「그림으로 보는 통계학」 그림 — 세 표가 담은 사람이 다르다.

만드는 파일:
  docs/ch02/exploration/img/titanic_which_table.png

수는 모두 공개된 세 파일에서 직접 세었다(2026-10-23 에 내려받아 확인).

  A 영국 상무부 집계 2,201 명 (승객 1,316 + 승무원 885)
    https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/Titanic.csv
  B 승객 명부 titanic3, 1,309 명 (승무원 없음)
    https://hbiostat.org/data/repo/titanic3.csv
  C 이 쪽이 쓰는 891 행 (B 의 부분집합 — 891 개 이름이 모두 B 에 있다)
    https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

실행:  python3 scripts/make_ch02_titanic_which_table.py   (저장소 최상위에서)
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
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
MUTED, MUTED_F = "#90A4AE", "#ECEFF1"

OUT = "docs/ch02/exploration/img/"
os.makedirs(OUT, exist_ok=True)

# ── 직접 센 수 ──────────────────────────────────────────────────────────
# 표 A: 2,201 = 승객 1,316 + 승무원 885,  생존 711
# 표 B: 승객 1,309,  생존 500
# 표 C: 891,  생존 342
rows = [
    # 이름,        승객,  승무원, 전체 생존율, 여성 생존율, 남성 생존율, 여성 수, 남성 수
    ("A 상무부 집계\n2,201 명", 1316, 885, 0.3230, 0.7319, 0.2120, 470, 1731),
    ("B 승객 명부\n1,309 명", 1309, 0, 0.3820, 0.7275, 0.1910, 466, 843),
    ("C 이 절의 자료\n891 명", 891, 0, 0.3838, 0.7420, 0.1889, 314, 577),
]

fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), width_ratios=[1, 1.25])


def tidy(ax, left=True):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


# ── ① 누가 표 안에 있는가 ────────────────────────────────────────────────
ax = axes[0]
y = np.arange(3)[::-1]
pax = [r[1] for r in rows]
crew = [r[2] for r in rows]
ax.barh(y, pax, height=0.55, color=BLUE_F, edgecolor=BLUE, lw=1.0, label="승객")
# 승무원이 0 인 표에 빈 막대를 그리면 테두리만 선으로 남는다
mask = np.array(crew) > 0
ax.barh(y[mask], np.array(crew)[mask], left=np.array(pax)[mask], height=0.55,
        color=ORANGE_F, edgecolor=ORANGE, lw=1.0, label="승무원")
for k, (yy, p, c) in enumerate(zip(y, pax, crew)):
    ax.text(p / 2, yy, f"{p:,}", ha="center", va="center", fontsize=9, color=BLUE)
    if c:
        ax.text(p + c / 2, yy, f"{c:,}", ha="center", va="center", fontsize=9,
                color=ORANGE)
ax.annotate("", xy=(895, y[2] + 0.30), xytext=(1315, y[1] - 0.30),
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2,
                            connectionstyle="arc3,rad=-0.3"))
ax.text(1420, y[1] - 0.55, "891 개 이름이 모두 B 안에 있다", fontsize=8.5,
        color=MUTED, ha="left", va="center")
ax.set_yticks(y)
ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
ax.set_xlabel("사람 수", fontsize=9.5, color=INK)
ax.set_title("① 세 표가 담은 사람이 다르다", fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_xlim(0, 2450)
tidy(ax)

# ── ② 전체는 흔들리고 격차는 남는다 ──────────────────────────────────────
ax = axes[1]
x = np.arange(3)
w = 0.26
allr = [r[3] for r in rows]
fem = [r[4] for r in rows]
male = [r[5] for r in rows]
for off, vals, col, fill, name in [(-w, allr, MUTED, MUTED_F, "전체"),
                                   (0.0, fem, GREEN, GREEN_F, "여성"),
                                   (w, male, BLUE, BLUE_F, "남성")]:
    b = ax.bar(x + off, vals, width=w, color=fill, edgecolor=col, lw=1.1,
               label=name)
    for xi, v in zip(x + off, vals):
        ax.text(xi, v + 0.015, f"{100*v:.1f}", ha="center", fontsize=8.5,
                color=col)
ax.set_xticks(x)
ax.set_xticklabels([r[0].split("\n")[0] for r in rows], fontsize=9.5, color=INK)
ax.set_ylabel("생존율", fontsize=9.5, color=INK)
ax.set_ylim(0, 0.88)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8"])
ax.set_title("② 전체 생존율은 표마다 다르고, 남녀 격차는 남는다",
             fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False, ncols=3, loc="upper center")
ax.text(1.0, -0.17, "전체 생존율은 32.3 에서 38.4 로 6 포인트 움직인다 — 승무원을 빼면 올라간다",
        fontsize=9, color=MUTED, ha="center", transform=ax.get_xaxis_transform())
tidy(ax)

fig.tight_layout()
path = OUT + "titanic_which_table.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
