r"""9.5 거짓발견율 쪽의 상자에 들어가는 그림을 만든다.

검정 수 m 이 커질 때 Bonferroni/Holm 의 검정력은 무너지고 BH 는 제자리를 지킨다.
m 이 열일 때는 셋이 거의 같다 — FDR 이 1995 년에 곧바로 받아들여지지 않은 까닭이
그 그림 안에 있다.

만드는 파일:
  ch09/multiple_testing/img/fdr_scaling.png

실행:  python3 scripts/make_ch09_fdr_scaling.py   (저장소 최상위에서)
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

OUT = "docs/ch09/multiple_testing/img/"
os.makedirs(OUT, exist_ok=True)

ALPHA = Q = 0.05
PI1, DELTA = 0.10, 3.0          # 참인 효과 10 %, 한쪽검정 이동폭 3
MS = [10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000]
REPS = {10: 4000, 30: 3000, 100: 2000, 300: 1200, 1000: 600,
        3000: 300, 10000: 120, 30000: 50, 100000: 25}

rng = np.random.default_rng(1995)


def run(m, B):
    m1 = int(round(PI1 * m))
    m0 = m - m1
    acc = np.zeros((3, 3))                      # [방법][찾은 비율, FDR, FWER]
    for _ in range(B):
        z = np.concatenate([rng.normal(0, 1, m0), rng.normal(DELTA, 1, m1)])
        truth = np.r_[np.zeros(m0, bool), np.ones(m1, bool)]
        p = stats.norm.sf(z)
        order = np.argsort(p)
        ps, tr = p[order], truth[order]

        rej = {}
        rej["bonf"] = ps <= ALPHA / m
        h = ps <= ALPHA / (m - np.arange(m))     # Holm — 단계하강
        k = np.argmax(~h) if (~h).any() else m
        rej["holm"] = np.r_[np.ones(k, bool), np.zeros(m - k, bool)]
        below = ps <= (np.arange(1, m + 1) / m) * Q
        kk = np.max(np.flatnonzero(below)) + 1 if below.any() else 0
        rej["bh"] = np.r_[np.ones(kk, bool), np.zeros(m - kk, bool)]

        for i, name in enumerate(["bonf", "holm", "bh"]):
            R = rej[name]
            V = int((R & ~tr).sum())
            S = int((R & tr).sum())
            acc[i, 0] += S / m1
            acc[i, 1] += V / max(int(R.sum()), 1)
            acc[i, 2] += V > 0
    return acc / B


print(f"참인 효과 {PI1:.0%}, 이동폭 {DELTA}, 수준 {ALPHA}")
print(f"{'m':>7s} {'본페로니':>9s} {'홀름':>9s} {'BH':>9s} | "
      f"{'BH 의 FDR':>10s} {'BH 의 FWER':>11s} {'본페로니 FWER':>13s}")
rows = []
for m in MS:
    r = run(m, REPS[m])
    rows.append(r)
    print(f"{m:7d} {r[0,0]:9.3f} {r[1,0]:9.3f} {r[2,0]:9.3f} | "
          f"{r[2,1]:10.4f} {r[2,2]:11.3f} {r[0,2]:13.3f}")

pw = np.array([[r[i, 0] for r in rows] for i in range(3)])
fdr = np.array([[r[i, 1] for r in rows] for i in range(3)])
fwer = np.array([[r[i, 2] for r in rows] for i in range(3)])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ------------------------------------------------- 왼쪽: 찾아내는 비율
axL.semilogx(MS, pw[2], "o-", color=BLUE, lw=2.4, ms=5, zorder=4,
             label="Benjamini-Hochberg (FDR)")
axL.semilogx(MS, pw[1], "s--", color=GREEN, lw=2.0, ms=4.5, zorder=3,
             label="Holm (FWER)")
axL.semilogx(MS, pw[0], "^:", color=ORANGE, lw=2.0, ms=4.5, zorder=3,
             label="Bonferroni (FWER)")
axL.axvspan(8, 64, color=MUTED, alpha=0.16, zorder=0)   # 1995 년 논문의 모의실험 범위 (4~64)
axL.annotate("$1995$ 년 논문의 모의실험이 다룬 범위\n(가설 $4$~$64$ 개 — 셋이 거의 같다)",
             xy=(30, 0.55), xytext=(700, 0.30), fontsize=9, color=INK,
             arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
axL.set_xticks(MS)
axL.set_xticklabels(["10", "30", "100", "300", "1000", "3000",
                     "1만", "3만", "10만"], fontsize=8.5)
axL.minorticks_off()
axL.set_xlim(8, 130000)
axL.set_ylim(0, 0.78)
axL.set_xlabel("검정 수 $m$ (가로축은 로그눈금)")
axL.set_ylabel("참인 효과 가운데 찾아낸 비율")
axL.set_title("$m$ 이 커지면 갈린다", color=INK, fontsize=12, pad=9)
axL.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------- 오른쪽: 무엇을 지키고 무엇을 내주나
axR.semilogx(MS, fdr[2], "o-", color=BLUE, lw=2.4, ms=5, zorder=4,
             label="BH 가 지키는 것 — 거짓발견의 비율")
axR.semilogx(MS, fwer[2], "o--", color=RED, lw=2.0, ms=4.5, zorder=3,
             label="BH 가 내주는 것 — 하나라도 틀릴 확률")
axR.semilogx(MS, fwer[0], "^:", color=ORANGE, lw=2.0, ms=4.5, zorder=3,
             label="Bonferroni 의 FWER")
axR.axhline(ALPHA, color=MUTED, lw=1.2, ls="--", zorder=1)
axR.text(9.5, 0.075, f"겨눈 수준 ${ALPHA}$", fontsize=9, color=INK)
axR.set_xticks(MS)
axR.set_xticklabels(["10", "30", "100", "300", "1000", "3000",
                     "1만", "3만", "10만"], fontsize=8.5)
axR.minorticks_off()
axR.set_xlim(8, 130000)
axR.set_ylim(0, 1.05)
axR.set_xlabel("검정 수 $m$ (가로축은 로그눈금)")
axR.set_ylabel("비율")
axR.set_title("값은 치르고 있다", color=INK, fontsize=12, pad=9)
axR.legend(loc="center right", fontsize=8.5, framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "fdr_scaling.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
