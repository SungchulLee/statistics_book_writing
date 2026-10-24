r"""12.2 「두 상관의 비교」 쪽의 상자에 들어가는 그림을 만든다.

피셔의 z 변환이 하는 일 — r 의 표준편차는 rho 에 따라 무너지는데 z 의 것은
rho 와 무관하게 1/sqrt(n-3) 에 머문다 — 을 모의실험으로 보인다.

만드는 파일:
  ch10/correlation_test/img/fisher_z_stabilize.png

실행:  python3 scripts/make_ch12_fisher_z_stabilize.py   (저장소 최상위에서)
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

OUT = "docs/ch10/correlation_test/img/"
os.makedirs(OUT, exist_ok=True)

N = 12
B = 200000
rng = np.random.default_rng(1921)


def sample_r(rho, n=N, B=B):
    """이변량정규에서 표본상관 B 개를 뽑는다."""
    x = rng.standard_normal((B, n))
    e = rng.standard_normal((B, n))
    y = rho * x + np.sqrt(1 - rho ** 2) * e
    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))
    return num / den


RHOS = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
sd_r, sd_z, sk_r, sk_z = [], [], [], []
print(f"n = {N},  모의실험 {B:,} 회")
print(f"{'rho':>6s} {'sd(r)':>8s} {'sd(z)':>8s} {'1/sqrt(n-3)':>12s} "
      f"{'치우침 r':>10s} {'치우침 z':>10s}")
for rho in RHOS:
    r = sample_r(rho)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    sd_r.append(r.std()); sd_z.append(z.std())
    sk_r.append(stats.skew(r)); sk_z.append(stats.skew(z))
    print(f"{rho:6.2f} {r.std():8.4f} {z.std():8.4f} "
          f"{1/np.sqrt(N-3):12.4f} {stats.skew(r):10.3f} {stats.skew(z):10.3f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ------------------------------------------------ 왼쪽: 표준편차가 rho 를 따라가는가
axL.plot(RHOS, sd_r, "o-", color=RED, lw=2.4, ms=5, zorder=4,
         label="$\\mathrm{sd}(r)$ — $\\rho$ 를 따라 무너진다")
axL.plot(RHOS, sd_z, "s-", color=BLUE, lw=2.4, ms=5, zorder=4,
         label="$\\mathrm{sd}(z)$ — 꼼짝하지 않는다")
axL.axhline(1 / np.sqrt(N - 3), color=MUTED, lw=1.4, ls="--", zorder=2)
axL.text(0.02, 1 / np.sqrt(N - 3) + 0.012,
         f"$1/\\sqrt{{n-3}} = {1/np.sqrt(N-3):.4f}$", fontsize=9.5, color=INK)
axL.set_xlim(-0.03, 1.0)
axL.set_ylim(0, 0.45)
axL.set_xlabel("참 상관 $\\rho$")
axL.set_ylabel("표집분포의 표준편차")
axL.set_title(f"$n = {N}$ — 왜 바로 견줄 수 없는가", color=INK,
              fontsize=12, pad=9)
axL.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------ 오른쪽: rho 셋에서 r 의 분포
for rho, col, fill in [(0.0, GREEN, "#DCEDC8"), (0.5, ORANGE, ORANGE_F),
                       (0.9, RED, "#FFCDD2")]:
    rr = sample_r(rho)
    axR.hist(rr, bins=110, density=True, color=fill, edgecolor=col,
             linewidth=0.3, alpha=0.85, zorder=3,
             label=f"$\\rho = {rho}$   (치우침 ${stats.skew(rr):+.2f}$, "
                   f"$\\mathrm{{sd}} = {rr.std():.3f}$)")
    axR.axvline(rho, color=col, lw=1.2, ls=":", zorder=4)
axR.set_xlim(-1.0, 1.02)
axR.set_xlabel("표본상관 $r$")
axR.set_ylabel("밀도")
axR.set_title("$\\rho$ 가 커지면 좁아지면서 한쪽으로 쏠린다", color=INK,
              fontsize=12, pad=9)
axR.legend(loc="upper left", fontsize=9, framealpha=0.95)
for s_ in ("top", "right"):
    axR.spines[s_].set_visible(False)

fig.tight_layout()
path = OUT + "fisher_z_stabilize.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
