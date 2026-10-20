r"""네이만이 $1934$ 년에 그린 「신뢰띠」와, 그날 보울리가 물은 것.

띠를 **가로로** 자르면 $\theta$ 가 주어졌을 때 받아들일 $x$ 의 범위이고,
**세로로** 자르면 $x$ 가 관측되었을 때의 신뢰구간이다. 네이만은 띠를 먼저
그린다 — 「우리는 표본을 뽑기 전에 추론을 시작한다」.

① 이항 $n = 20$ 의 신뢰띠와 두 방향의 절단.
② 사전분포가 무엇이든 장기 포함률은 $95\%$ 다 (네이만의 보장).
③ 그러나 $\hat p$ 를 알고 나서 묻는 물음의 답은 사전분포가 정한다 (보울리의 결핍).

만드는 파일: docs/ch08/one_sample_intervals/img/neyman_belt.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
Z = stats.norm.isf(0.025)


def wilson(k, n, z=Z):
    """z 검정을 뒤집어 얻는 구간 — 곧 네이만의 띠를 세로로 자른 것."""
    ph = k / n
    d = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def accept(theta, n, z=Z):
    """띠를 가로로 자른 것 — theta 가 참일 때 받아들일 k 의 범위."""
    se = np.sqrt(theta * (1 - theta) / n)
    return n * (theta - z * se), n * (theta + z * se)


# === ② ③ 모의실험 ==================================================
N = 1000
M = 400_000
PRIORS = {
    "평평한 사전분포": lambda m, r: r.uniform(0.05, 0.20, m),
    "치우친 사전분포": lambda m, r: np.clip(0.135 + 0.012 * r.standard_normal(m),
                                      0.001, 0.999),
}
sims = {}
for name, draw in PRIORS.items():
    th = draw(M, RNG)
    k = RNG.binomial(N, th)
    lo, hi = wilson(k, N)
    sims[name] = (th, k / N, (lo <= th) & (th <= hi))
    print(f"{name}: 주변 포함률 {sims[name][2].mean() * 100:.2f}%")

th_fix = np.full(M, 0.10)
k_fix = RNG.binomial(N, th_fix)
lo_f, hi_f = wilson(k_fix, N)
cov_fix = (lo_f <= th_fix) & (th_fix <= hi_f)
print(f"고정 theta = 0.10: 주변 포함률 {cov_fix.mean() * 100:.2f}%")

# 보울리가 그날 든 수
k_bow = 100
lo_b, hi_b = wilson(k_bow, N)
print(f"\n보울리의 보기 — n = {N}, k = {k_bow} 이면 95% 구간 "
      f"[{lo_b * 1000:.1f}, {hi_b * 1000:.1f}] (천분율)")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

# ---- ① 띠 --------------------------------------------------------
NB = 20
ks = np.arange(NB + 1)
blo, bhi = wilson(ks, NB)
ax1.fill_betweenx(np.linspace(0, 1, 2), 0, 0, color="none")  # 눈금 잡기용
ax1.fill_between(ks, blo, bhi, color="#1f77b4", alpha=0.22, step="mid", lw=0)
ax1.step(ks, blo, where="mid", color="#1f77b4", lw=2.0)
ax1.step(ks, bhi, where="mid", color="#1f77b4", lw=2.0)
ax1.text(10, 0.5, "신 뢰 띠", ha="center", va="center", fontsize=15,
         rotation=38, color="#1f77b4", alpha=0.85)

KS = 6
l6, h6 = wilson(KS, NB)
ax1.vlines(KS, l6, h6, color="#d62728", lw=3.4, zorder=5)
ax1.plot([KS], [KS / NB], "o", color="#d62728", ms=7, zorder=6)
ax1.annotate(f"세로로 자르면 **신뢰구간**\n$x = {KS}$ → "
             f"$[{l6:.2f},\\ {h6:.2f}]$".replace("**", ""),
             xy=(KS, h6), xytext=(0.3, 0.90), fontsize=9.5, color="#d62728",
             ha="left", va="top",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))

TH = 0.30
a_lo, a_hi = accept(TH, NB)
ax1.hlines(TH, a_lo, a_hi, color="#2ca02c", lw=3.4, zorder=5)
ax1.annotate(f"가로로 자르면 **받아들임 구간**\n$\\theta = {TH}$ → "
             f"$x \\in [{a_lo:.1f},\\ {a_hi:.1f}]$".replace("**", ""),
             xy=(a_hi, TH), xytext=(9.6, 0.13), fontsize=9.5, color="#2ca02c",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
ax1.set_xlim(-0.5, NB + 0.5)
ax1.set_ylim(0, 1)
ax1.set_xlabel(f"관측한 성공 횟수 $x$  ($n = {NB}$)")
ax1.set_ylabel("모비율 $\\theta$")
ax1.set_title("① 네이만의 신뢰띠", fontsize=12)
ax1.grid(alpha=0.25)

# ---- ② 주변 포함률 -------------------------------------------------
step = 400
idx = np.arange(step, M + 1, step)
for (name, c) in (("평평한 사전분포", "#1f77b4"), ("치우친 사전분포", "#d62728")):
    cum = np.cumsum(sims[name][2])[idx - 1] / idx
    ax2.plot(idx, cum * 100, color=c, lw=2.0, label=name)
ax2.plot(idx, (np.cumsum(cov_fix)[idx - 1] / idx) * 100, color="#2ca02c",
         lw=2.0, label="$\\theta$ 를 $0.10$ 으로 고정")
ax2.axhline(95, color="black", lw=1.5, ls="--")
ax2.text(M * 0.98, 95.25, "명목 $95\\%$", ha="right", fontsize=10)
ax2.set_xscale("log")
ax2.set_xlabel("되풀이한 실험 수")
ax2.set_ylabel("누적 포함률 ($\\%$)")
ax2.set_title("② 사전분포가 무엇이든 — 네이만의 보장", fontsize=12)
ax2.set_ylim(90, 99)
ax2.set_xticks([1e3, 1e4, 1e5, 4e5])
ax2.set_xticklabels(["$10^3$", "$10^4$", "$10^5$", "$4{\\times}10^5$"])
ax2.minorticks_off()
ax2.legend(loc="upper right", fontsize=9.5)
ax2.grid(alpha=0.25, which="major")

# ---- ③ 조건부 포함률 -----------------------------------------------
EDGES = np.array([0.07, 0.09, 0.105, 0.12, 0.135, 0.15, 0.18])
mids = np.arange(len(EDGES) - 1)
w = 0.36
for j, (name, c) in enumerate((("평평한 사전분포", "#1f77b4"),
                               ("치우친 사전분포", "#d62728"))):
    _, ph, cov = sims[name]
    vals, ns = [], []
    for i in range(len(EDGES) - 1):
        m = (ph >= EDGES[i]) & (ph < EDGES[i + 1])
        vals.append(cov[m].mean() * 100 if m.sum() > 300 else np.nan)
        ns.append(m.sum())
    ax3.bar(mids + (j - 0.5) * w, vals, w, color=c, label=name)
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax3.text(i + (j - 0.5) * w, v + 1.5, f"{v:.0f}", ha="center",
                     fontsize=9, color=c)
ax3.axhline(95, color="black", lw=1.5, ls="--")
ax3.text(len(mids) - 0.55, 96.5, "명목 $95\\%$", ha="right", fontsize=10)
ax3.set_xticks(mids)
ax3.set_xticklabels([f"${EDGES[i]:.3f}$–\n${EDGES[i + 1]:.3f}$"
                     for i in range(len(EDGES) - 1)], fontsize=8.5)
ax3.set_xlabel("관측한 $\\hat p$ 가 이 구간일 때")
ax3.set_ylabel("그 경우들만 모은 포함률 ($\\%$)")
ax3.set_title("③ $\\hat p$ 를 알고 나서 물으면 — 보울리의 물음", fontsize=12)
ax3.set_ylim(0, 112)
ax3.legend(loc="lower right", fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch08/one_sample_intervals/img/neyman_belt.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① n = {NB} 의 띠")
print(f"   x = {KS} 를 세로로 자르면 [{l6:.4f}, {h6:.4f}]")
print(f"   theta = {TH} 를 가로로 자르면 x in [{a_lo:.2f}, {a_hi:.2f}]")

print(f"\n③ 관측 구간별 포함률")
hdr = f"{'p̂ 구간':>16}"
for name in PRIORS:
    hdr += f"{name:>14}"
print(hdr + f"{'사례 수(평평/치우침)':>22}")
for i in range(len(EDGES) - 1):
    row = f"{f'{EDGES[i]:.3f}–{EDGES[i + 1]:.3f}':>16}"
    cnt = []
    for name in PRIORS:
        _, ph, cov = sims[name]
        m = (ph >= EDGES[i]) & (ph < EDGES[i + 1])
        row += f"{cov[m].mean() * 100:>13.1f}%" if m.sum() > 300 else f"{'—':>14}"
        cnt.append(m.sum())
    print(row + f"{f'{cnt[0]:,} / {cnt[1]:,}':>22}")
