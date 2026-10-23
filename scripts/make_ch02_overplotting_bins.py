r"""2.5절 육각형 구간 그림 쪽의 「AI로 보는 통계학」 그림 — 점을 찍는 대신 칸을 센다.

만드는 파일:
  docs/ch02/visualization/img/overplotting_bins.png

실행:  python3 scripts/make_ch02_overplotting_bins.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다. 로그 눈금의 마이너스도 mathtext 라
       한글 글꼴에서 두부가 되므로 눈금을 직접 적는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE = "#1565C0"
ORANGE = "#E65100"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(2)
N = 2_000_000

# 봉우리 둘과, 전체의 0.02 % 밖에 안 되는 작은 덩어리 하나
n1 = int(N * 0.60)
n2 = int(N * 0.3998)
n3 = N - n1 - n2
xy = np.vstack([
    rng.multivariate_normal([-1.0, -0.4], [[1.0, 0.55], [0.55, 0.8]], n1),
    rng.multivariate_normal([1.6, 0.9], [[0.7, -0.3], [-0.3, 0.6]], n2),
    rng.multivariate_normal([3.6, -2.6], [[0.02, 0], [0, 0.02]], n3),
])
x, y = xy[:, 0], xy[:, 1]
print(f"N = {N:,},  작은 덩어리 {n3:,} 개 ({100*n3/N:.3f} %)")

fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


LIM = dict(xlim=(-4.6, 5.0), ylim=(-3.6, 3.6))

# ── ① 점을 찍으면 ───────────────────────────────────────────────────────
ax = axes[0]
sub = rng.choice(N, 200_000, replace=False)
ax.scatter(x[sub], y[sub], s=1, alpha=0.05, color=BLUE, linewidths=0)
ax.set(**LIM)
ax.set_title("① 산점도 — 200만 점 가운데 20만 점", fontsize=10.5, color=INK)
ax.set_xlabel("축 1", fontsize=9.5, color=INK)
ax.set_ylabel("축 2", fontsize=9.5, color=INK)
tidy(ax)

# ── ② 칸을 세면 ─────────────────────────────────────────────────────────
ax = axes[1]
hb = ax.hexbin(x, y, gridsize=70, cmap="viridis", norm=LogNorm(),
               extent=(-4.6, 5.0, -3.6, 3.6), linewidths=0)
cb = fig.colorbar(hb, ax=ax, pad=0.015)
cb.set_label("칸 안의 점 수 (로그)", fontsize=9, color=INK)
cb.ax.tick_params(labelsize=8, colors=INK)
cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
cb.ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set(**LIM)
ax.set_title("② 육각 칸으로 센 같은 자료 (200만 점 전부)", fontsize=10.5, color=INK)
ax.set_xlabel("축 1", fontsize=9.5, color=INK)
tidy(ax)

counts = hb.get_array().data
counts = counts[counts > 0]
print(f"칸 {len(counts):,} 개,  가장 붐비는 칸 {counts.max():,.0f} 점,  "
      f"가운뎃값 {np.median(counts):,.0f} 점")

# ── ③ 눈에 보이는 검기는 금세 1 에 붙는다 ────────────────────────────────
ax = axes[2]
k = np.arange(1, 301)
for a, col, ls in [(0.05, BLUE, "-"), (0.01, ORANGE, "--")]:
    ax.plot(k, 1 - (1 - a) ** k, color=col, ls=ls, lw=1.8,
            label=f"알파 {a}")
    k_sat = int(np.ceil(np.log(1 - 0.95) / np.log(1 - a)))
    ax.plot([k_sat], [0.95], "o", ms=5, color=col)
    ax.annotate(f"{k_sat} 점이면 95 퍼센트", xy=(k_sat, 0.95),
                xytext=(k_sat + 18, 0.95 - (0.12 if a == 0.05 else 0.0)),
                fontsize=8.5, color=col, va="center")
frac = (counts >= 60).sum() / len(counts)
share = counts[counts >= 60].sum() / counts.sum()
ax.axhline(1.0, color=MUTED, lw=0.8, ls=":")
ax.set_ylim(0, 1.08)
ax.set_xlabel("한 자리에 겹친 점의 수", fontsize=9.5, color=INK)
ax.set_ylabel("그 자리의 불투명도", fontsize=9.5, color=INK)
ax.set_title("③ 포화 — 그 너머의 밀도 차이는 보이지 않는다",
             fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.text(0.30, 0.10,
        f"②의 칸 가운데 {100*frac:.0f} 퍼센트가 60 점 이상이고\n"
        f"자료의 {100*share:.0f} 퍼센트가 그 칸들에 들어 있다",
        transform=ax.transAxes, fontsize=9, color=RED)
# 작은 덩어리가 선형 눈금에서 묻히는지 — 그 칸들의 점 수를 가장 붐비는 칸과 견준다
sel = (hb.get_offsets()[:, 0] > 3.0) & (hb.get_offsets()[:, 1] < -2.0)
cl = hb.get_array().data[sel]
print(f"작은 덩어리 칸 {int((cl > 0).sum())} 개, 가장 많은 칸 {cl.max():,.0f} 점 "
      f"= 최대 칸의 {100 * cl.max() / counts.max():.1f} 퍼센트")
tidy(ax)
print(f"60 점 이상인 칸의 비율 {frac:.4f},  그 칸들이 담은 자료의 비율 {share:.4f}")

fig.tight_layout()
path = OUT + "overplotting_bins.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
