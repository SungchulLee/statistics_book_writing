r"""봉우리를 세는 일이 왜 어려운가.

이 쪽의 $\S 5$ 는 "대역폭을 달리한 커널밀도추정"과 "단봉성에 대한 형식적
검정"을 이름만 대고 지나간다. 그 둘을 수로 만든다.

① 같은 표본인데 띠폭 하나로 봉우리가 $9$ 개도 되고 $1$ 개도 된다.
② 실버만의 임계 띠폭 검정 — 「봉우리가 하나뿐인가」에 $p$ 값을 붙인다.
   그리고 두 봉우리를 알아보려면 얼마나 떨어져 있어야 하는지 잰다.
③ 평균·표준편차·왜도가 **모두 같은데** 봉우리가 다섯과 셋인 두 분포.

만드는 파일: docs/ch02/shape/img/modality_counting.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(7)
N1, SIG = 600, 0.5
HS = [0.08, 0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5]
H_SHOW = [0.1, 0.45, 1.5]
DS = [1.0, 1.5, 2.0, 2.5, 3.0]          # 두 봉우리 사이 거리 (sigma 단위)
N2, B_BOOT, REPS = 200, 80, 80
NBIG = 200000
GRID = 1200


def nmodes(x, h, grid=GRID):
    """띠폭 h 의 가우스 KDE 가 갖는 국소최대의 수."""
    g = np.linspace(x.min() - 3 * h, x.max() + 3 * h, grid)
    d = stats.gaussian_kde(x, bw_method=h / x.std(ddof=1))(g)
    return int(((d[1:-1] > d[:-2]) & (d[1:-1] > d[2:])).sum())


def h_crit(x, k, steps=40):
    """봉우리가 k 개 이하가 되는 가장 작은 띠폭 (이분법)."""
    lo, hi = 1e-3, 3 * x.std(ddof=1)
    for _ in range(steps):
        m = 0.5 * (lo + hi)
        if nmodes(x, m) > k:
            lo = m
        else:
            hi = m
    return hi


def silverman_p(x, k, B=B_BOOT, seed=0):
    """실버만의 임계 띠폭 검정. 매끄럽게 한 분포에서 재표집한다."""
    r = np.random.default_rng(seed)
    h = h_crit(x, k)
    n, s, mu = len(x), x.std(ddof=1), x.mean()
    hit = 0
    for _ in range(B):
        i = r.integers(0, n, n)
        xb = mu + (x[i] + h * r.normal(0, 1, n) - mu) / np.sqrt(1 + h * h / s ** 2)
        hit += nmodes(xb, h) > k
    return h, hit / B


# === ① 띠폭이 봉우리 수를 정한다 =====================================
x3 = np.concatenate([RNG.normal(m, SIG, N1 // 3) for m in (-3, 0, 3)])
RNG.shuffle(x3)
print(f"① 세 봉우리에서 뽑은 표본 {N1} 개 — 띠폭만 바꾼다")
MODES = {h: nmodes(x3, h) for h in HS}
for h in HS:
    print(f"   h = {h:<5} 봉우리 {MODES[h]} 개")
print(f"   같은 표본인데 {max(MODES.values())} 개에서 "
      f"{min(MODES.values())} 개까지 간다.")

# === ② 실버만 검정 ==================================================
print(f"\n② 「봉우리가 하나뿐이다」에 대한 실버만 검정 (재표집 {B_BOOT} 회)")
uni = RNG.normal(0, 1, N1)
for nm, v in (("단봉 (표준정규)", uni), ("세 봉우리", x3)):
    h, p = silverman_p(v, 1)
    print(f"   {nm:<16} 임계 띠폭 {h:.3f},  p = {p:.3f}")

print(f"\n   두 봉우리를 알아보려면 얼마나 떨어져야 하는가 "
      f"(각 봉우리 {N2 // 2} 개, {REPS} 번)")
POW = []
for d in DS:
    rej = 0
    for r in range(REPS):
        z = np.concatenate([RNG.normal(0, 1, N2 // 2),
                            RNG.normal(d, 1, N2 // 2)])
        RNG.shuffle(z)
        rej += silverman_p(z, 1, B=B_BOOT, seed=r)[1] < 0.05
    POW.append(rej / REPS)
    print(f"      떨어진 거리 {d}sigma : 기각률 {POW[-1]:.3f}")

# === ③ 요약통계량은 봉우리를 모른다 ==================================
true = np.concatenate([RNG.normal(m, SIG, NBIG // 5)
                       for m in (-4, -2, 0, 2, 4)])
RNG.shuffle(true)
W = 0.25                      # 분산이 같아지도록 고른 무게
gen = np.concatenate([RNG.normal(-4, SIG, int(NBIG * W)),
                      RNG.normal(0, SIG, int(NBIG * (1 - 2 * W))),
                      RNG.normal(4, SIG, int(NBIG * W))])
RNG.shuffle(gen)
print(f"\n③ 참 분포(봉우리 다섯) 와 모드 둘을 잃은 생성기(봉우리 셋)")
print(f"{'':<8}{'평균':>9}{'표준편차':>10}{'왜도':>9}{'첨도':>9}{'봉우리':>8}")
STAT = {}
for nm, v in (("참", true), ("생성기", gen)):
    sub = v[:6000]
    STAT[nm] = (v.mean(), v.std(ddof=1), stats.skew(v),
                stats.kurtosis(v, fisher=False), nmodes(sub, 0.35))
    print(f"{nm:<8}{STAT[nm][0]:>+9.3f}{STAT[nm][1]:>10.3f}"
          f"{STAT[nm][2]:>+9.3f}{STAT[nm][3]:>9.3f}{STAT[nm][4]:>8d}")
print("   앞의 셋은 사실상 같고, 넷째(첨도)만 조금 다르다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.1))
C1, C2 = "#d62728", "#1f77b4"

ax1.step(HS, [MODES[h] for h in HS], where="mid", color=C1, lw=2.8)
ax1.plot(HS, [MODES[h] for h in HS], "o", color=C1, ms=7)
ax1.axhline(3, color="black", lw=1.6, ls="--")
ax1.text(2.5, 3.3, "참 봉우리 수 $3$", fontsize=10.5, ha="right")
ax1.set_xscale("log")
ax1.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0])
ax1.set_xticklabels(["$0.1$", "$0.2$", "$0.5$", "$1$", "$2$"], fontsize=10)
ax1.minorticks_off()
ax1.set_xlabel("커널의 띠폭 $h$")
ax1.set_ylabel("밀도그림에서 세어지는 봉우리 수")
ax1.set_ylim(0, max(MODES.values()) + 1.5)
ax1.set_title(f"① 같은 표본, {max(MODES.values())} 개에서 {min(MODES.values())} 개까지", fontsize=12)
ax1.grid(alpha=0.25)
g = np.linspace(-5.5, 5.5, 500)
for k, h in enumerate(H_SHOW):
    ins = ax1.inset_axes([0.06 + 0.31 * k, 0.70, 0.27, 0.25])
    ins.plot(g, stats.gaussian_kde(x3, bw_method=h / x3.std(ddof=1))(g),
             color=C1, lw=1.4)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title(f"$h = {h}$ → ${MODES[h]}$", fontsize=8.5, pad=2)

ax2.plot(DS, POW, "o-", color=C2, lw=2.8, ms=8)
ax2.axhline(0.8, color="black", lw=1.4, ls="--")
ax2.text(3.0, 0.82, "검정력 $0.8$", fontsize=10.5, ha="right")
ax2.axhline(0.05, color="#888888", lw=1.2, ls=":")
ax2.set_xticks(DS)
ax2.set_xticklabels([f"${v}$" for v in DS], fontsize=10.5)
ax2.set_xlabel("두 봉우리 사이의 거리 ($\\sigma$ 단위)")
ax2.set_ylabel("「봉우리가 둘 이상」이라 말할 확률")
ax2.set_ylim(0, 1.03)
ax2.set_title(f"② 표본 ${N2}$ 개로 알아볼 수 있는 간격", fontsize=12)
ax2.grid(alpha=0.25)

gg = np.linspace(-6.5, 6.5, 800)
for nm, v, c, ls in (("참 (봉우리 $5$)", true, "#2ca02c", "-"),
                     ("생성기 (봉우리 $3$)", gen, C1, "--")):
    ax3.plot(gg, stats.gaussian_kde(v[:20000], bw_method=0.06)(gg),
             color=c, lw=2.6, ls=ls, label=nm)
# 고정폭 글꼴을 지정하면 한글이 두부가 된다 — 줄을 나눠 적는다
ax3.text(0.025, 0.975,
         f"평균     ${STAT['참'][0]:+.3f}$  대  ${STAT['생성기'][0]:+.3f}$\n"
         f"표준편차 ${STAT['참'][1]:.3f}$  대  ${STAT['생성기'][1]:.3f}$\n"
         f"왜도     ${STAT['참'][2]:+.3f}$  대  ${STAT['생성기'][2]:+.3f}$\n"
         f"첨도     ${STAT['참'][3]:.3f}$  대  ${STAT['생성기'][3]:.3f}$",
         transform=ax3.transAxes, va="top", ha="left", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax3.set_xlabel("값")
ax3.set_ylabel("밀도")
ax3.set_ylim(0, 0.46)
ax3.set_title("③ 요약통계량은 봉우리를 세지 않는다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch02/shape/img/modality_counting.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
