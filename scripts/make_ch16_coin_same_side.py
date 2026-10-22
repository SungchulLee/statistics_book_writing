r"""16.2 이항검정 쪽의 상자에 들어가는 그림을 만든다.

사람이 던진 동전 350,757 번에서 나온 「같은 면」 비율 0.508 을 이항검정의 눈으로
읽는다. 기각에 필요한 n 과, 「얼마나 치우쳤나」를 말하는 데 필요한 n 이 다르다.

만드는 파일:
  ch16/one_sample_nonparametric/img/coin_same_side.png

실행:  python3 scripts/make_ch16_coin_same_side.py   (저장소 최상위에서)
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
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/one_sample_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)

N_OBS = 350757          # 보고된 던진 횟수
P_HAT = 0.508           # 보고된 「같은 면」 비율
P0 = 0.5
ALPHA = 0.05

# ----------------------------------------------- 보고된 비율을 그대로 검정한다
s_obs = int(round(P_HAT * N_OBS))
bt = stats.binomtest(s_obs, N_OBS, P0, alternative="two-sided")
z = (s_obs - N_OBS * P0) / np.sqrt(N_OBS * P0 * (1 - P0))
print(f"같은 면 {s_obs:,} / {N_OBS:,}  ->  정확 이항검정 p = {bt.pvalue:.3e}, z = {z:.2f}")
ci = bt.proportion_ci(confidence_level=0.95, method="exact")
print(f"정확 95% 구간 = [{ci.low:.5f}, {ci.high:.5f}]")


def power_at(n, p_true=P_HAT):
    """p0 = 0.5 를 겨눈 양측 정확 이항검정의 검정력."""
    lo = stats.binom.ppf(ALPHA / 2, n, P0) - 1
    hi = stats.binom.isf(ALPHA / 2, n, P0)
    return stats.binom.cdf(lo, n, p_true) + stats.binom.sf(hi, n, p_true)


ns = np.unique(np.round(np.logspace(3, 6.1, 160)).astype(int))
pw = np.array([power_at(n) for n in ns])
n80 = int(ns[np.searchsorted(pw, 0.80)])
print(f"검정력 0.80 에 닿는 n = {n80:,}   (그때 검정력 {power_at(n80):.3f})")
print(f"n = {N_OBS:,} 에서의 검정력 = {power_at(N_OBS):.4f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4))

# ----------------------------------------------- 왼쪽: 검정력과 n
axL.plot(ns, pw, color=BLUE, lw=2.4, zorder=3)
axL.axhline(0.80, color=MUTED, lw=0.9, ls=":", zorder=1)
axL.axvline(n80, color=GREEN, lw=1.2, ls="--", zorder=2)
axL.axvline(N_OBS, color=ORANGE, lw=1.2, ls="--", zorder=2)
axL.set_xscale("log")
# 로그 눈금의 지수는 mathtext 라 평문으로 박는다
ticks = [1000, 10000, 100000, 1000000]
axL.set_xticks(ticks)
axL.set_xticklabels(["1천", "1만", "10만", "100만"], fontsize=9)
axL.minorticks_off()
axL.set_xlim(1000, 1_300_000)
axL.set_ylim(0, 1.03)
axL.set_xlabel("던진 횟수 $n$ (가로축은 로그눈금)")
axL.set_ylabel("$p = 0.5$ 를 기각할 확률")
axL.set_title("$0.008$ 을 잡는 데 드는 횟수", color=INK, fontsize=12, pad=9)
axL.text(n80 * 1.15, 0.30, f"검정력 $0.80$ 까지\n{n80:,} 번", fontsize=9, color=GREEN)
axL.text(N_OBS * 1.12, 0.52, f"실제로 던진\n{N_OBS:,} 번", fontsize=9, color=ORANGE)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ----------------------------------------------- 오른쪽: 구간이 0.5 를 비켜나는 자리
show_ns = [1000, 5000, 10000, 30000, 100000, N_OBS]
ys = np.arange(len(show_ns))[::-1]
axR.axvline(P0, color=RED, lw=1.4, zorder=1)
axR.axvline(P_HAT, color=MUTED, lw=0.9, ls=":", zorder=1)
for y, n in zip(ys, show_ns):
    s = int(round(P_HAT * n))
    c = stats.binomtest(s, n, P0).proportion_ci(confidence_level=0.95,
                                                method="exact")
    clears = c.low > P0
    col = BLUE if clears else MUTED
    axR.plot([c.low, c.high], [y, y], color=col, lw=3.2,
             solid_capstyle="round", zorder=3)
    axR.plot([P_HAT], [y], "o", color=col, ms=6, zorder=4)
    axR.text(0.5415, y, "$0.5$ 를 벗어났다" if clears else "$0.5$ 를 아직 품고 있다",
             fontsize=9, color=col, va="center")
    print(f"n = {n:>7,}  95% 구간 [{c.low:.4f}, {c.high:.4f}]"
          f"  {'0.5 밖' if clears else '0.5 포함'}")

axR.set_yticks(ys)
axR.set_yticklabels([f"{n:,}" for n in show_ns], fontsize=9)
axR.set_xlim(0.470, 0.585)
axR.set_ylim(-0.7, len(show_ns) - 0.3)
axR.set_xticks([0.48, 0.50, 0.52, 0.54])
axR.set_xticklabels(["0.48", "0.50", "0.52", "0.54"], fontsize=9)
axR.set_xlabel("「같은 면」 비율")
axR.set_ylabel("던진 횟수")
axR.set_title("$\\hat p = 0.508$ 을 고정했을 때의 $95\\%$ 구간",
              color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "coin_same_side.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
