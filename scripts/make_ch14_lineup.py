r"""「그림을 보는 일」을 정식 검정으로 바꾸는 규약 — 라인업.

진짜 자료의 그림 하나를 귀무가설에서 만든 그림 $m-1$ 개 사이에 숨겨 놓고 고르게 한다.
귀무가 참이면 맞힐 확률이 정확히 $1/m$ 이므로, **고르는 행위 자체가 유의수준 $1/m$ 의
검정**이 된다.

여기서는 사람 대신 **기계 심사자**를 세웠다 — $m$ 개 그림 가운데 샤피로–윌크 통계량이
가장 작은(가장 비정규해 보이는) 것을 고르게 한다. 사람의 성적이 아니라 **규약의 천장**을
재는 셈이다.

① 라인업 한 벌. $20$ 개 가운데 하나만 정규가 아니다.
② 그 천장이 정식 검정에 얼마나 못 미치는가.
③ 라인업 크기 $m$ 을 키우면 유의수준이 $1/m$ 로 내려가고 검정력도 따라 내려간다.

만드는 파일: docs/ch14/graphical_methods/img/lineup.png
"""

import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

warnings.filterwarnings("ignore")
from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

M, REPS = 20, 4000
NS = (20, 50, 100)
MS = np.array([5, 10, 20, 50, 100])
ALTS = {
    "정규 (귀무)": lambda n: RNG.standard_normal(n),
    "$t_5$": lambda n: RNG.standard_t(5, n),
    "지수": lambda n: RNG.exponential(1.0, n),
    "대수정규 $(0.5)$": lambda n: RNG.lognormal(0.0, 0.5, n),
}


def wstat(x):
    return stats.shapiro(x).statistic


def lineup_hit(n, draw, m=M, reps=REPS):
    """가장 비정규해 보이는 그림을 고르는 심사자의 적중률과, 정식 검정의 기각률."""
    hit = sw = 0
    for _ in range(reps):
        real = draw(n)
        ws = [wstat(real)] + [wstat(RNG.standard_normal(n)) for _ in range(m - 1)]
        hit += int(np.argmin(ws) == 0)
        sw += int(stats.shapiro(real).pvalue < 0.05)
    return hit / reps, sw / reps


print(f"라인업 크기 {M}, 반복 {REPS:,}")
print(f"{'분포':>16}{'n':>6}{'라인업 적중':>12}{'샤피로 기각':>12}")
pts = []
for name, d in ALTS.items():
    for n in NS:
        h, s = lineup_hit(n, d)
        pts.append((name, n, h, s))
        print(f"{name.replace('$',''):>16}{n:>6}{h:>12.3f}{s:>12.3f}")

null = [h for nm, _, h, _ in pts if "귀무" in nm]
print(f"\n귀무 아래 적중률 {np.mean(null):.4f}  (이론값 {1 / M:.4f})")

# === ③ 라인업 크기에 따라 ============================================
print(f"\n③ 라인업 크기 m 에 따라 (대수정규 0.5, n = 50)")
print(f"{'m':>5}{'귀무 적중':>11}{'대립 적중':>11}{'1/m':>8}")
null_m, pow_m = [], []
for m in MS:
    h0, _ = lineup_hit(50, ALTS["정규 (귀무)"], m=int(m), reps=2000)
    h1, _ = lineup_hit(50, ALTS["대수정규 $(0.5)$"], m=int(m), reps=2000)
    null_m.append(h0); pow_m.append(h1)
    print(f"{m:>5}{h0:>11.4f}{h1:>11.4f}{1 / m:>8.4f}")
null_m, pow_m = np.array(null_m), np.array(pow_m)

# === 그림 ============================================================
fig = plt.figure(figsize=(15.4, 5.0))
gs = fig.add_gridspec(1, 3, wspace=0.26)
sub = gs[0, 0].subgridspec(4, 5, hspace=0.12, wspace=0.12)

# ① 라인업 한 벌 — 20 개 가운데 하나만 대수정규
N1 = 50
true_pos = int(RNG.integers(M))
qs = stats.norm.ppf((np.arange(1, N1 + 1) - 0.375) / (N1 + 0.25))
for k in range(M):
    ax = fig.add_subplot(sub[k // 5, k % 5])
    v = (RNG.lognormal(0, 0.5, N1) if k == true_pos
         else RNG.standard_normal(N1))
    v = (v - v.mean()) / v.std()
    ax.plot(qs, np.sort(v), ".", ms=2.4, color="#1f77b4")
    ax.plot([-2.6, 2.6], [-2.6, 2.6], color="#bbbbbb", lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
    ax.text(0.04, 0.84, f"{k + 1}", transform=ax.transAxes, fontsize=7.5,
            color="#888888")
    for sp in ax.spines.values():
        sp.set_linewidth(0.6); sp.set_color("#999999")
fig.text(0.175, 0.945, "① 스물 중 하나만 정규가 아니다", ha="center",
         fontsize=12)
fig.text(0.175, 0.045, f"정답은 {true_pos + 1}번  (각 $n = {N1}$)",
         ha="center", fontsize=10.5, color="#555555")

ax2 = fig.add_subplot(gs[0, 1])
COL = {"정규 (귀무)": "#999999", "$t_5$": "#1f77b4", "지수": "#d62728",
       "대수정규 $(0.5)$": "#2ca02c"}
MK = {20: "o", 50: "s", 100: "^"}
for nm, n, h, s in pts:
    ax2.plot(s, h, MK[n], color=COL[nm], ms=11, alpha=0.9)
ax2.plot([0, 1], [0, 1], color="#bbbbbb", lw=1.4, ls=":")
for nm, c in COL.items():
    ax2.plot([], [], "o", color=c, ms=9, label=nm)
for n, mk in MK.items():
    ax2.plot([], [], mk, color="black", ms=8, label=f"$n = {n}$")
ax2.set_xlabel("샤피로–윌크 검정의 기각률")
ax2.set_ylabel("라인업 심사자의 적중률")
ax2.set_xlim(-0.03, 1.05); ax2.set_ylim(-0.03, 1.05)
ax2.set_title("② 규약이 잃는 것은 크지 않다", fontsize=12)
ax2.legend(loc="upper left", fontsize=8.5, ncol=2)
ax2.grid(alpha=0.25)

ax3 = fig.add_subplot(gs[0, 2])
ax3.semilogx(MS, pow_m, "o-", color="#2ca02c", lw=2.8, ms=9,
             label="대수정규 $(0.5)$ 를 맞힐 확률")
ax3.semilogx(MS, null_m, "s--", color="#999999", lw=2.2, ms=7,
             label="귀무 아래 적중률")
ax3.semilogx(MS, 1 / MS, color="black", lw=1.4, ls=":", label="$1/m$")
ax3.set_xticks(MS)
ax3.set_xticklabels([f"${v}$" for v in MS])
ax3.minorticks_off()
ax3.set_xlabel("라인업 크기 $m$ (그림 수)")
ax3.set_ylabel("고른 그림이 진짜일 확률")
ax3.set_ylim(0, 1.0)
ax3.set_title(f"③ 유의수준이 $1/m$ 이다 ($n = 50$)", fontsize=12)
ax3.legend(loc="center right", fontsize=9.5)
ax3.grid(alpha=0.25)

out = "docs/ch14/graphical_methods/img/lineup.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
