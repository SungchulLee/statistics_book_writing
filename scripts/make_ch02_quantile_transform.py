r"""2.2절 ECDF 쪽의 「AI로 보는 통계학」 그림 — ECDF 를 그대로 계산해 특징을 펴는 일.

만드는 파일:
  docs/ch02/exploration/img/quantile_transform.png

실행:  python3 scripts/make_ch02_quantile_transform.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
RED = "#D32F2F"
MUTED = "#90A4AE"

OUT = "docs/ch02/exploration/img/"
os.makedirs(OUT, exist_ok=True)


def tidy(ax, left=True):
    ax.tick_params(labelsize=9, colors=INK, length=3)
    ax.spines[["top", "right"]].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    for s in ax.spines.values():
        s.set_color(MUTED)


rng = np.random.default_rng(0)
n_tr = 2000
train = rng.lognormal(0, 1, n_tr)
new = rng.lognormal(0, 1, 20000)
grid = np.sort(train)


def F_hat(t):
    return np.searchsorted(grid, t, side="right") / n_tr


u = F_hat(new)
eps = 1 / (2 * n_tr)
z = stats.norm.ppf(np.clip(u, eps, 1 - eps))

fig = plt.figure(figsize=(13.2, 4.2))
gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1], hspace=0.55, wspace=0.28)
ax_left = fig.add_subplot(gs[:, 0])
ax_mid = [fig.add_subplot(gs[i, 1]) for i in range(3)]
ax_right = fig.add_subplot(gs[:, 2])

# ── ① 계단 하나가 변환이다 ────────────────────────────────────────────────
ax = ax_left
small = np.sort(rng.choice(train, 25, replace=False))
xs = np.r_[0, np.repeat(small, 2), small[-1] * 1.15]
ys = np.r_[0, 0, np.repeat(np.arange(1, 25) / 25, 2), 1, 1]
ax.plot(xs, ys, color=BLUE, lw=1.8, solid_joinstyle="miter")
ax.plot(small, np.arange(1, 26) / 25, "o", ms=3.5, color=BLUE)
for x0 in (small[6], small[18]):
    y0 = F_hat(x0) if False else np.searchsorted(small, x0, side="right") / 25
    ax.annotate("", xy=(0, y0), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.annotate("", xy=(x0, y0), xytext=(x0, 0),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.0, ls=":"))
    ax.text(-0.02 * small[-1], y0, f"{y0:.2f}", color=ORANGE, fontsize=8.5,
            ha="right", va="center")
ax.set_title("① 변환값은 계단의 높이다", fontsize=10.5, color=INK)
ax.set_xlabel("원래 값 x", fontsize=9.5, color=INK)
ax.set_ylabel("ECDF 값", fontsize=9.5, color=INK)
ax.set_ylim(-0.03, 1.08)
ax.text(0.97, 0.12, f"눈금 하나 = 1/n\n(여기서는 n = 25)", transform=ax.transAxes,
        ha="right", fontsize=8.5, color=MUTED)
tidy(ax)

# ── ② 치우침이 사라진다 ──────────────────────────────────────────────────
panels = [
    (np.clip(new, 0, 8), 60, ORANGE_F, ORANGE,
     f"원자료 — 왜도 {stats.skew(new):.2f} (8 에서 잘라 그렸다)"),
    (u, 40, BLUE_F, BLUE, f"ECDF 변환 — 왜도 {stats.skew(u):.2f}, 고르다"),
    (z, 60, GREEN_F, GREEN, f"정규 변환 — 왜도 {stats.skew(z):.2f}"),
]
for k, (v, bins, fill, edge, label) in enumerate(panels):
    a = ax_mid[k]
    a.hist(v, bins=bins, density=True, color=fill, edgecolor=edge, lw=0.5)
    a.set_title(label, fontsize=9, color=edge, pad=3)
    a.set_yticks([])
    tidy(a, left=False)
    if k == 1:
        a.set_ylim(0, 1.6)
ax_mid[0].text(0.5, 1.42, "② 같은 자료, 세 가지 눈금", transform=ax_mid[0].transAxes,
               ha="center", fontsize=10.5, color=INK)

# ── ③ 끝에서 뭉친다 ─────────────────────────────────────────────────────
ax = ax_right
order = np.argsort(new)
xx, uu = new[order], u[order]
inside = xx <= train.max()
ax.plot(xx[inside], uu[inside], color=BLUE, lw=1.6)
ax.plot(xx[~inside], uu[~inside], "o", ms=4, color=RED)
ax.axvline(train.max(), color=MUTED, ls="--", lw=1.0)
ax.text(train.max(), 0.45, " 훈련자료의 최댓값\n 21.46", fontsize=8.5,
        color=MUTED, va="center")
ax.annotate(f"{int((~inside).sum())} 개가 ECDF = 1 한 점에",
            xy=(xx.max(), 1.0), xytext=(0.04, 0.80), textcoords="axes fraction",
            fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
ax.set_xscale("log")
# 로그 눈금의 마이너스는 mathtext 라 한글 글꼴에서 두부가 된다 — 평문으로 적는다
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax.set_xlabel("새 자료의 값 (로그 눈금)", fontsize=9.5, color=INK)
ax.set_ylabel("ECDF 값", fontsize=9.5, color=INK)
ax.set_title("③ 훈련 범위를 넘으면 구별이 사라진다", fontsize=10.5, color=INK)
ax.set_ylim(-0.03, 1.06)
tidy(ax)

path = OUT + "quantile_transform.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
print(f"원자료   왜도 {stats.skew(new):.3f}  첨도 {stats.kurtosis(new):.3f}")
print(f"ECDF 변환 왜도 {stats.skew(u):.3f}  첨도 {stats.kurtosis(u):.3f}")
print(f"정규 변환 왜도 {stats.skew(z):.3f}  첨도 {stats.kurtosis(z):.3f}")
print(f"최대 |F-U| {np.abs(np.sort(u) - np.arange(1, len(u)+1)/len(u)).max():.4f}")
print(f"훈련 최대 {train.max():.2f}, 넘은 개수 {int((new > train.max()).sum())}")
