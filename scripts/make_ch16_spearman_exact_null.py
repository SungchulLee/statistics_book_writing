r"""16.6 Spearman 순위상관(재론) 쪽의 상자에 들어가는 그림을 만든다.

쪽이 「n 이 작으면 정확 임계값을 표에서 얻는다」고만 적어 둔 자리에, 그 표가
왜 필요했는지를 붙인다. n! 가지를 모두 세어 정확 귀무분포를 구하고 t 근사와 견준다.

만드는 파일:
  ch16/correlation/img/spearman_exact_null.png

실행:  python3 scripts/make_ch16_spearman_exact_null.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import itertools
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

OUT = "docs/ch16/correlation/img/"
os.makedirs(OUT, exist_ok=True)

ALPHA = 0.05


def exact_rs(n):
    """n! 가지 순열을 모두 세어 r_s 의 정확 귀무분포를 돌려준다."""
    base = np.arange(1, n + 1)
    perms = np.array(list(itertools.permutations(base)), dtype=np.int16)
    d2 = ((perms - base) ** 2).sum(axis=1)
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def t_critical_rs(n):
    """t 근사가 쓰는 양측 5% 임계값을 r_s 눈금으로 돌려준다."""
    tc = stats.t.ppf(1 - ALPHA / 2, n - 2)
    return tc / np.sqrt(tc ** 2 + n - 2)


def exact_critical(rs):
    """P(r_s >= c) <= 0.025 를 만족하는 가장 작은 c. 없으면 nan — 양측 5% 가 불가능하다."""
    vals = np.unique(rs)
    for c in vals:
        if (rs >= c).mean() <= ALPHA / 2:
            return c
    return np.nan


# ------------------------------------------------- 수를 뽑는다
print("  n       n!    t 근사 임계값   정확 임계값   t 근사의 실제 크기")
rows = []
dists = {}
for n in range(4, 11):
    rs = exact_rs(n)
    dists[n] = rs
    rc_t = t_critical_rs(n)
    rc_e = exact_critical(rs)
    size = (np.abs(rs) >= rc_t).mean()          # t 임계값을 그대로 쓸 때의 실제 크기
    rows.append((n, len(rs), rc_t, rc_e, size))
    ce = "  불가능" if np.isnan(rc_e) else f"{rc_e:7.4f}"
    print(f"{n:3d} {len(rs):>9,}      {rc_t:7.4f}      {ce}        {size:7.4f}")

ns = np.array([r[0] for r in rows])
rc_t_v = np.array([r[2] for r in rows])
rc_e_v = np.array([r[3] for r in rows])
size_v = np.array([r[4] for r in rows])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4))

# ------------------------------------------------- 왼쪽: n = 8 의 정확 귀무분포
N_SHOW = 8
rs8 = dists[N_SHOW]
vals, cnt = np.unique(rs8, return_counts=True)
prob = cnt / cnt.sum()
# r_s 가 놓이는 격자는 sum d^2 이 짝수 정수인 데서 온다 — 간격을 직접 적는다
step = 12.0 / (N_SHOW * (N_SHOW ** 2 - 1))
axL.bar(vals, prob / step, width=step * 0.9, color=BLUE_F, edgecolor=BLUE,
        linewidth=0.4, zorder=2, label=f"정확 귀무분포 ($8! = {len(rs8):,}$ 가지)")

grid = np.linspace(-0.9995, 0.9995, 1200)
tg = grid * np.sqrt((N_SHOW - 2) / (1 - grid ** 2))
jac = np.sqrt(N_SHOW - 2) / (1 - grid ** 2) ** 1.5      # dt/dr
dens = stats.t.pdf(tg, N_SHOW - 2) * jac
print(f"  (확인) t 근사 밀도의 적분 = {np.trapz(dens, grid):.6f}")
print(f"  (확인) 정확분포 확률의 합 = {prob.sum():.6f}")
axL.plot(grid, dens, color=ORANGE, lw=2.2, zorder=3,
         label="$t$ 근사가 말하는 밀도")

rc_t8 = t_critical_rs(N_SHOW)
rc_e8 = exact_critical(rs8)
gap = (np.abs(rs8) >= rc_t8).mean() - (np.abs(rs8) >= rc_e8).mean()
for sign in (1, -1):
    axL.axvspan(sign * rc_t8, sign * rc_e8, color=RED, alpha=0.22, zorder=1)
axL.axvline(rc_t8, color=ORANGE, lw=1.4, ls="--", zorder=4,
            label=f"$t$ 근사가 긋는 선 $\\pm{rc_t8:.3f}$")
axL.axvline(-rc_t8, color=ORANGE, lw=1.4, ls="--", zorder=4)
axL.axvline(rc_e8, color=GREEN, lw=1.4, ls="-.", zorder=4,
            label=f"정확 임계값 $\\pm{rc_e8:.3f}$")
axL.axvline(-rc_e8, color=GREEN, lw=1.4, ls="-.", zorder=4)
axL.text(1.02, 1.145, f"붉은 두 띠가 $t$ 근사만 기각하는 구역 — 확률 {gap:.4f}",
         fontsize=9, color=RED, ha="right", va="top")
axL.set_xlim(-1.05, 1.05)
axL.set_ylim(0, 1.22)
axL.set_xlabel("$r_s$")
axL.set_ylabel("밀도")
axL.set_title(f"$n = {N_SHOW}$ — 계단인 분포에 매끄러운 곡선을 씌우면",
              color=INK, fontsize=12, pad=9)
axL.legend(loc="upper left", fontsize=8.5, framealpha=0.95)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------- 오른쪽: t 근사의 실제 크기
axR.axhline(ALPHA, color=RED, lw=1.3, ls="--", zorder=1, label="겨눈 수준 $0.05$")
axR.plot(ns, size_v, "o-", color=BLUE, lw=2.2, ms=6, zorder=3,
         label="$t$ 근사를 그대로 썼을 때의 실제 크기")
for n, s in zip(ns, size_v):
    axR.annotate(f"{s:.4f}", xy=(n, s), xytext=(0, 9 if s > ALPHA else -16),
                 textcoords="offset points", ha="center", fontsize=8.5,
                 color=BLUE if s > ALPHA else GREEN)
axR.set_xticks(ns)
axR.set_xticklabels([str(v) for v in ns], fontsize=9)
axR.set_xlabel("$n$")
axR.set_ylabel("귀무가설에서의 실제 기각률")
axR.set_ylim(0, 0.105)
axR.set_title("$n! $ 가지를 다 세어 잰 값", color=INK, fontsize=12, pad=9)
axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "spearman_exact_null.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
