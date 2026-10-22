r"""16.2 Wilcoxon 부호순위검정 쪽의 상자에 들어가는 그림을 만든다.

「D_i = 0 인 관측값을 버린다」는 한 줄이 무엇을 망가뜨리는지 보인다. 자료를 대립가설
쪽으로 좋게 바꾸었는데 p 값이 커지는 자료 한 쌍을 찾아, 버리는 관례와 Pratt(1959)의
방식이 그 쌍에서 어떻게 갈리는지 정확 분포로 그린다.

만드는 파일:
  ch16/one_sample_nonparametric/img/pratt_zeros.png

실행:  python3 scripts/make_ch16_pratt_zeros.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy.stats import rankdata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/one_sample_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05

# -1 이 0 으로 바뀌는 것 말고는 같은 자료 — 대립가설(양의 이동) 쪽으로 좋아진 것이다
D_BEFORE = np.array([-1.0, 0, 3, 4, 5, 6, 7, -8, 9, 10])
D_AFTER = np.array([0.0, 0, 3, 4, 5, 6, 7, -8, 9, 10])


def subset_pmf(ranks):
    """주어진 순위들의 부분집합합 분포 — 부호 배정 2^n 가지가 동등할 때의 W+ 분포."""
    grid = np.array([0.0])
    pmf = np.array([1.0])
    for r in ranks:
        g2 = np.union1d(grid, grid + r)
        p2 = np.zeros(len(g2))
        for g, p in zip(grid, pmf):
            p2[np.searchsorted(g2, g)] += p / 2
            p2[np.searchsorted(g2, g + r)] += p / 2
        grid, pmf = g2, p2
    return grid, pmf


def discard(D):
    """교과서의 관례 — 0 을 버리고 남은 것에 순위를 매긴다."""
    d = D[D != 0]
    r = rankdata(np.abs(d))
    return r[d > 0].sum(), r


def pratt(D):
    """Pratt(1959) — 0 까지 넣어 순위를 매기고, 귀무분포는 0 이 아닌 것들의 순위로 만든다."""
    r = rankdata(np.abs(D))
    return r[D > 0].sum(), r[D != 0]


def tail(Wp, ranks):
    grid, pmf = subset_pmf(ranks)
    return pmf[grid >= Wp - 1e-9].sum(), grid, pmf


rows = []
print("                       n   W+     정확 p (오른쪽 꼬리)")
for label, D in [("그대로", D_BEFORE), ("-1 이 0 으로", D_AFTER)]:
    for name, fn in [("버리기", discard), ("Pratt ", pratt)]:
        Wp, ranks = fn(D)
        p, grid, pmf = tail(Wp, ranks)
        rows.append((label, name, len(ranks), Wp, p, grid, pmf))
        print(f"  {label:12s} {name}  {len(ranks):2d}  {Wp:5.1f}      {p:.4f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ---------------------------------------------------- 왼쪽: 버리기의 두 귀무분포
for (label, name, n, Wp, p, grid, pmf), color, fill, ls in [
        (rows[0], BLUE, BLUE_F, "-"), (rows[2], ORANGE, ORANGE_F, "--")]:
    axL.step(grid, pmf, where="mid", color=color, lw=1.8, ls=ls, zorder=3,
             label=f"{label} — $n = {n}$, $W^+ = {Wp:.0f}$, $p = {p:.4f}$")
    m = grid >= Wp - 1e-9
    axL.fill_between(grid[m], 0, pmf[m], step="mid", color=color, alpha=0.35,
                     zorder=2)
axL.set_xlim(0, 56)
axL.set_xlabel("$W^+$")
axL.set_ylabel("확률")
axL.set_title("영을 버리면 귀무분포 자체가 바뀐다", color=INK, fontsize=12, pad=9)
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ---------------------------------------------------- 오른쪽: 네 개의 p 값
labels = [f"{r[1].strip()}\n{r[0]}" for r in rows]
ps = [r[4] for r in rows]
colors = [BLUE, GREEN, ORANGE, GREEN]
xs = np.array([0, 1, 2.6, 3.6])
axR.bar(xs, ps, width=0.72, color=[BLUE, GREEN, ORANGE, GREEN],
        edgecolor=INK, linewidth=0.5, zorder=3)
for x, p in zip(xs, ps):
    axR.text(x, p - 0.0035, f"{p:.4f}", ha="center", va="top", fontsize=9.5,
             color="white", fontweight="bold", zorder=5)
axR.axhline(ALPHA, color=RED, lw=1.4, ls="--", zorder=4)
axR.text(-0.52, 0.0515, "겨눈 수준 $0.05$", color=RED, fontsize=9, va="bottom")
axR.set_xlim(-0.6, 4.2)
for x0, x1, txt, col in [(0, 2, "버리기 — 자료가 좋아졌는데 올라간다", ORANGE),
                         (2.6, 3.6, "Pratt — 내려간다", GREEN)]:
    axR.annotate("", xy=(x1, 0.0625), xytext=(x0, 0.0625),
                 arrowprops=dict(arrowstyle="->", color=col, lw=1.6))
    axR.text((x0 + x1) / 2, 0.0638, txt, ha="center", fontsize=9, color=col)
axR.set_xticks(xs)
axR.set_xticklabels(["버리기\n그대로", "Pratt\n그대로",
                     "버리기\n$-1 \\to 0$", "Pratt\n$-1 \\to 0$"], fontsize=9)
axR.set_ylim(0, 0.072)
axR.set_ylabel("정확 $p$ 값 (오른쪽 꼬리)")
axR.set_title("같은 자료 한 쌍, 두 가지 관례", color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "pratt_zeros.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
