r"""2.5절 쌍그림 쪽의 「AI로 보는 통계학」 그림 — 쌍이 많아지면 가장 센 쌍은 언제나 세다.

만드는 파일:
  docs/ch02/visualization/img/pairs_explode.png

실행:  python3 scripts/make_ch02_pairs_explode.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
주의:  한글은 수식 $...$ 바깥에만 쓴다. 로그 눈금은 평문으로 적는다.
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE = "#E65100"
GREEN = "#33691E"
PURPLE = "#6A1B9A"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/visualization/img/"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(0)
n = 100
PS = [(20, GREEN), (200, BLUE), (2000, PURPLE)]
REP = 200


def max_abs_corr(X):
    Z = (X - X.mean(0)) / X.std(0)
    R = (Z.T @ Z) / Z.shape[0]
    iu = np.triu_indices(R.shape[0], 1)
    return R, iu, np.abs(R[iu])


def tidy(ax):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2))

# ── ① 쌍의 수 ───────────────────────────────────────────────────────────
ax = axes[0]
p = np.arange(2, 3001)
ax.plot(p, p * (p - 1) / 2, color=BLUE, lw=1.8)
for pp, col in PS:
    ax.plot([pp], [pp * (pp - 1) / 2], "o", ms=6, color=col)
    ax.annotate(f"변수 {pp} 개 -> 쌍 {pp*(pp-1)//2:,} 개",
                xy=(pp, pp * (pp - 1) / 2), xytext=(4, -12),
                textcoords="offset points", fontsize=8.5, color=col)
ax.set_xscale("log")
ax.set_yscale("log")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("변수의 수", fontsize=9.5, color=INK)
ax.set_ylabel("그려야 할 칸의 수", fontsize=9.5, color=INK)
ax.set_title("① 칸은 변수의 제곱으로 는다", fontsize=10.5, color=INK)
tidy(ax)

# ── ② 모두 독립인데 가장 센 쌍은 ────────────────────────────────────────
ax = axes[1]
summary = []
keep = None
for pp, col in PS:
    mx = np.empty(REP)
    for r in range(REP):
        X = rng.normal(size=(n, pp))
        R, iu, a = max_abs_corr(X)
        mx[r] = a.max()
        if pp == 200 and r == 0:
            j = int(a.argmax())
            keep = (X[:, iu[0][j]], X[:, iu[1][j]], R[iu[0][j], iu[1][j]])
    ax.hist(mx, bins=28, color=col, histtype="step", lw=1.8,
            label=f"변수 {pp} 개")
    summary.append((pp, np.median(mx), np.quantile(mx, 0.05),
                    np.quantile(mx, 0.95), (mx > 0.3).mean()))
ax.axvline(0.3, color=RED, ls="--", lw=1.2)
ax.text(0.297, ax.get_ylim()[1] * 0.45, "눈길이 가는 세기 0.3", color=RED,
        fontsize=9, rotation=90, ha="right", va="center")
ax.set_xlabel("가장 센 쌍의 상관 (절댓값)", fontsize=9.5, color=INK)
ax.set_ylabel("되풀이 횟수", fontsize=9.5, color=INK)
ax.set_title(f"② 서로 독립인 변수들, 표본 {n} 개", fontsize=10.5, color=INK)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
tidy(ax)

# ── ③ 그 쌍을 그려 보면 ─────────────────────────────────────────────────
ax = axes[2]
xs, ys, r = keep
ax.scatter(xs, ys, s=14, color=BLUE, alpha=0.75, linewidths=0)
b = np.polyfit(xs, ys, 1)
g = np.linspace(xs.min(), xs.max(), 2)
ax.plot(g, np.polyval(b, g), color=ORANGE, lw=1.6)
ax.set_xlabel("변수 하나", fontsize=9.5, color=INK)
ax.set_ylabel("다른 변수 하나", fontsize=9.5, color=INK)
ax.set_title(f"③ 19,900 쌍 가운데 가장 센 것 (상관 {r:+.2f})",
             fontsize=10.5, color=INK)
ax.text(0.03, 0.03, "둘은 독립인 난수다", transform=ax.transAxes,
        fontsize=9.5, color=RED)
tidy(ax)

fig.tight_layout()
path = OUT + "pairs_explode.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
print("변수수    쌍의 수   max|r| 가운뎃값   5~95 퍼센타일      0.3 넘을 확률")
for pp, med, lo, hi, over in summary:
    print(f"{pp:6d}{pp*(pp-1)//2:11,d}{med:14.3f}   [{lo:.3f}, {hi:.3f}]{over:16.2f}")
print(f"③ 의 쌍: 상관 {r:+.4f}")
