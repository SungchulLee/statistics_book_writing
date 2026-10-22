r"""16.3 대응 부호검정 쪽의 상자에 들어가는 그림을 만든다.

0 인 쌍을 버리면 부호검정이 답하는 물음이 「중앙값이 0 인가」에서 「0 이 아닌 쌍
가운데 양수가 절반인가」로 바뀐다. 두 물음은 0 에 확률덩어리가 있으면 갈라지고,
그때 중앙값이 참으로 0 인데도 기각확률이 1 로 간다. 그것을 정확히 계산해 그린다.

만드는 파일:
  ch16/paired_sample_nonparametric/img/sign_zeros_hypothesis.png

실행:  python3 scripts/make_ch16_sign_zeros_hypothesis.py   (저장소 최상위에서)
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
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch16/paired_sample_nonparametric/img/"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05
RATIO = 0.6          # 0 이 아닌 쌍 가운데 양수의 비율


def masses(q):
    """P(D<0), P(D=0), P(D>0). q = 0 이면 0.5 / 0 / 0.5 로 둔다."""
    if q == 0.0:
        return 0.5, 0.0, 0.5
    return (1 - q) * (1 - RATIO), q, (1 - q) * RATIO


def median_is_zero(pm, q, pp):
    return pm <= 0.5 + 1e-12 and pp <= 0.5 + 1e-12


def reject_prob(n, q):
    """0 을 버리는 양측 부호검정의 기각확률 — 정확 계산."""
    pm, _, pp = masses(q)
    cond = pp / (pp + pm)
    tot = 0.0
    for m in range(1, n + 1):
        w = stats.binom.pmf(m, n, 1 - q)
        k = np.arange(m + 1)
        p2 = 2 * np.minimum(stats.binom.cdf(k, m, 0.5),
                            stats.binom.sf(k - 1, m, 0.5))
        tot += w * stats.binom.pmf(k, m, cond)[p2 <= ALPHA].sum()
    return tot


QS = [0.0, 0.2, 0.4, 0.6]
NS = [10, 20, 35, 50, 75, 100, 140, 200, 280, 400]

print("  q    P(D<0)  P(D=0)  P(D>0)  중앙값 0?   기각확률")
curves = {}
for q in QS:
    pm, p0, pp = masses(q)
    r = [reject_prob(n, q) for n in NS]
    curves[q] = r
    print(f" {q:.1f}    {pm:.2f}    {p0:.2f}    {pp:.2f}   "
          f"{'예' if median_is_zero(pm, q, pp) else '아니오'}      "
          + " ".join(f"{v:.3f}" for v in r))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4))

# --------------------------------------------- 왼쪽: 분포의 생김새
q_show = 0.4
pm, p0, pp = masses(q_show)
axL.bar([-1, 0, 1], [pm, p0, pp], width=0.52,
        color=[ORANGE, MUTED, BLUE], edgecolor=INK, linewidth=0.6, zorder=3)
for x, h, lab in [(-1, pm, f"$P(D<0) = {pm:.2f}$"), (0, p0, f"$P(D=0) = {p0:.2f}$"),
                  (1, pp, f"$P(D>0) = {pp:.2f}$")]:
    axL.text(x, h + 0.012, lab, ha="center", fontsize=9.5, color=INK)
axL.axvline(0, color=GREEN, lw=1.6, ls="--", zorder=2)
# mathtext 은 \le 를 모르고 마크다운 굵게도 모른다 — \leq 를 쓰고 강조는 빼 둔다
axL.text(-1.42, 0.625, "중앙값은 정확히 $0$ 이다 — "
                       "$P(D<0)=0.24 \\leq 0.5$ 이고 $P(D>0)=0.36 \\leq 0.5$",
         fontsize=9.5, color=GREEN, va="top")
axL.text(-1.42, 0.560, "그런데 $0$ 을 버리고 나면 양수 비율이 "
                       "$0.36 / 0.60 = 0.6$ 이 된다",
         fontsize=9.5, color=RED, va="top")
axL.set_xticks([-1, 0, 1])
axL.set_xticklabels(["$D < 0$", "$D = 0$", "$D > 0$"], fontsize=10)
axL.set_xlim(-1.5, 1.5)
axL.set_ylim(0, 0.64)
axL.set_ylabel("확률")
axL.set_title("귀무가설 「중앙값 $=0$」 이 참인 분포", color=INK, fontsize=12, pad=9)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# --------------------------------------------- 오른쪽: 기각확률이 1 로 간다
styles = {0.0: (MUTED, ":", "o"), 0.2: (RED, "-", "o"),
          0.4: (ORANGE, "-", "s"), 0.6: (BLUE, "-", "^")}
for q in QS:
    c, ls, mk = styles[q]
    lab = ("$P(D=0) = 0$ — 덩어리가 없으면 수준을 지킨다" if q == 0
           else f"$P(D=0) = {q:.1f}$")
    axR.plot(NS, curves[q], ls, marker=mk, color=c, lw=2.0, ms=4.5,
             zorder=3, label=lab)
axR.axhline(ALPHA, color=GREEN, lw=1.2, ls="--", zorder=1)
axR.text(11, 0.075, "겨눈 수준 $0.05$", fontsize=9, color=GREEN)
axR.set_xscale("log")
axR.set_xticks(NS)
axR.set_xticklabels([str(v) for v in NS], fontsize=8.5)
axR.minorticks_off()
axR.set_xlim(10, 400)
axR.set_ylim(0, 1.0)
axR.set_xlabel("쌍의 수 $n$ (가로축은 로그눈금)")
axR.set_ylabel("「중앙값 $=0$」을 기각할 확률")
axR.set_title("참인 귀무가설을 기각할 확률이 $1$ 로 간다",
              color=INK, fontsize=12, pad=9)
axR.legend(loc="upper left", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "sign_zeros_hypothesis.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
