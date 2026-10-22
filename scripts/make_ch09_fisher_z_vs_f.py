r"""9.4 「두 분산의 F 검정」 쪽의 상자에 들어가는 그림을 만든다.

피셔가 표로 만든 것은 분산비 F 가 아니라 z = (1/2) ln F 였다. 같은 귀무분포를
두 눈금에 올려 보면 까닭이 보인다 — F 는 오른쪽으로 길게 끌리고 z 는 거의 대칭이다.

만드는 파일:
  ch09/two_sample_tests/img/fisher_z_vs_f.png

실행:  python3 scripts/make_ch09_fisher_z_vs_f.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

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

OUT = "docs/ch09/two_sample_tests/img/"
os.makedirs(OUT, exist_ok=True)

DFS = [(10, 10), (5, 20)]
ALPHA = 0.05


def skew_of(d1, d2, B=400000, seed=1934):
    """F 와 z 의 치우침을 표본으로 잰다 — 같은 난수에서 둘 다."""
    r = np.random.default_rng(seed)
    f = r.chisquare(d1, B) / d1 / (r.chisquare(d2, B) / d2)
    return stats.skew(f), stats.skew(0.5 * np.log(f))


fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.6))

for row, (d1, d2) in enumerate(DFS):
    lo, hi = stats.f.ppf(ALPHA / 2, d1, d2), stats.f.isf(ALPHA / 2, d1, d2)
    sf, sz = skew_of(d1, d2)
    print(f"df = ({d1}, {d2}):  F 의 양측 임계값 = [{lo:.4f}, {hi:.4f}]  "
          f"(오른쪽/왼쪽 = {hi/lo:.1f} 배)")
    print(f"    치우침  F = {sf:+.3f},   z = (1/2)lnF = {sz:+.3f}")

    # ---- 왼쪽: F 눈금
    ax = axes[row][0]
    x = np.linspace(0.01, stats.f.isf(0.001, d1, d2), 1400)
    y = stats.f.pdf(x, d1, d2)
    ax.plot(x, y, color=ORANGE, lw=2.3, zorder=3)
    for a, b in [(x[0], lo), (hi, x[-1])]:
        m = (x >= a) & (x <= b)
        ax.fill_between(x[m], 0, y[m], color=ORANGE_F, zorder=2)
    ax.axvline(1, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.set_xlim(0, min(x[-1], 6.5))
    ax.set_ylim(0, y.max() * 1.18)
    ax.set_xlabel("$F = s_1^2 / s_2^2$")
    ax.set_ylabel("밀도")
    ax.set_title(f"$F$ 눈금 — $d_1={d1}$, $d_2={d2}$,  치우침 ${sf:+.2f}$",
                 color=INK, fontsize=11, pad=7)
    ax.text(0.99, 0.93, f"임계값 ${lo:.3f}$ 와 ${hi:.3f}$\n"
                        f"$1$ 에서의 거리가 ${1-lo:.2f}$ 대 ${hi-1:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=ORANGE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- 오른쪽: z 눈금
    ax = axes[row][1]
    zlo, zhi = 0.5 * np.log(lo), 0.5 * np.log(hi)
    zx = np.linspace(0.5 * np.log(stats.f.ppf(0.0005, d1, d2)),
                     0.5 * np.log(stats.f.isf(0.0005, d1, d2)), 1400)
    # z 의 밀도는 야코비안 dF/dz = 2 e^{2z}
    zy = stats.f.pdf(np.exp(2 * zx), d1, d2) * 2 * np.exp(2 * zx)
    ax.plot(zx, zy, color=BLUE, lw=2.3, zorder=3)
    for a, b in [(zx[0], zlo), (zhi, zx[-1])]:
        m = (zx >= a) & (zx <= b)
        ax.fill_between(zx[m], 0, zy[m], color=BLUE_F, zorder=2)
    ax.axvline(0, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.set_ylim(0, zy.max() * 1.18)
    ax.set_xlabel("$z = \\frac{1}{2}\\ln F$")
    ax.set_ylabel("밀도")
    ax.set_title(f"$z$ 눈금 — 같은 분포,  치우침 ${sz:+.2f}$",
                 color=INK, fontsize=11, pad=7)
    ax.text(0.99, 0.93, f"임계값 ${zlo:+.3f}$ 와 ${zhi:+.3f}$\n"
                        f"$0$ 에서의 거리가 ${-zlo:.2f}$ 대 ${zhi:.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color=BLUE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "fisher_z_vs_f.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
