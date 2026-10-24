r"""배포된 모형의 「분포가 움직였는가」는 이 쪽의 동질성 검정과 같은 물음이다.

업계에서 쓰는 PSI(모집단 안정성 지수)는

$$\mathrm{PSI} = \sum_i (p_i - q_i)\ln\frac{p_i}{q_i} = D_{\mathrm{KL}}(p\|q) + D_{\mathrm{KL}}(q\|p)$$

이고, 이 쪽의 가능도비 통계량은 $G^2 = 2n\,D_{\mathrm{KL}}(p\|q)$ 다. **둘의 유일한
차이가 $n$ 이다** — PSI 에는 표본 수가 들어 있지 않다.

① 같은 이동이라도 $p$ 값은 $n$ 과 함께 $0$ 으로 가고 PSI 는 그대로다.
② PSI 와 $G^2/n$ 은 거의 같은 수다.
③ 그래서 PSI 문턱 하나가 $n$ 마다 전혀 다른 엄격함이 된다.

만드는 파일: docs/ch12/test/img/psi_vs_g2.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
Q = np.array([0.10, 0.20, 0.40, 0.20, 0.10])      # 기준 분포
K = len(Q)
DF = K - 1
NS = np.array([100, 300, 1000, 3000, 10000, 30000])
REPS = 2000
PSI_T = (0.10, 0.25)                               # 업계에서 흔히 쓰는 문턱


def shifted(q, delta):
    """가운데 칸의 질량을 바깥으로 delta 만큼 옮긴다."""
    p = q.copy()
    p[2] -= delta
    p[0] += delta / 2
    p[4] += delta / 2
    return p


def psi(p, q):
    m = (p > 0) & (q > 0)
    return float(np.sum((p[m] - q[m]) * np.log(p[m] / q[m])))


def g2(counts, q):
    n = counts.sum()
    p = counts / n
    m = p > 0
    return float(2 * n * np.sum(p[m] * np.log(p[m] / q[m])))


DELTA = 0.06
P = shifted(Q, DELTA)
psi_true = psi(P, Q)
print(f"기준 {Q}")
print(f"이동 {np.round(P, 3)}   참 PSI = {psi_true:.4f}")
print(f"  KL(p||q) = {np.sum(P * np.log(P / Q)):.5f}, "
      f"KL(q||p) = {np.sum(Q * np.log(Q / P)):.5f}  (합이 PSI)")

# === ① n 에 따라 ====================================================
med_p, med_psi = [], []
for n in NS:
    c = RNG.multinomial(n, P, size=REPS)
    gs = np.array([g2(row, Q) for row in c])
    ps = stats.chi2.sf(gs, DF)
    pp = np.array([psi(row / n, Q) for row in c])
    med_p.append(np.median(ps))
    med_psi.append(np.median(pp))
    print(f"  n = {n:>6}  중앙값 p = {med_p[-1]:.2e}   중앙값 PSI = "
          f"{med_psi[-1]:.4f}")
med_p, med_psi = np.array(med_p), np.array(med_psi)

# === ② PSI 와 G^2/n ================================================
pairs = []
for _ in range(600):
    d = RNG.uniform(0.005, 0.18)
    Pd = shifted(Q, d)
    n = 5000
    c = RNG.multinomial(n, Pd)
    pairs.append((psi(c / n, Q), g2(c, Q) / n))
pairs = np.array(pairs)
print(f"\n② PSI 와 G^2/n 의 상관 {np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]:.5f}, "
      f"기울기 {np.polyfit(pairs[:, 1], pairs[:, 0], 1)[0]:.4f}")

# === ③ PSI 문턱이 뜻하는 p 값 =======================================
print(f"\n③ PSI 문턱이 n 마다 뜻하는 p 값 (G^2 ≈ n·PSI, 자유도 {DF})")
print(f"{'n':>8}" + "".join(f"{f'PSI={t}':>16}" for t in PSI_T))
tbl = {t: [] for t in PSI_T}
for n in NS:
    row = f"{n:>8}"
    for t in PSI_T:
        pv = stats.chi2.sf(n * t, DF)
        tbl[t].append(pv)
        row += f"{pv:>16.2e}"
    print(row)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.loglog(NS, med_p, "o-", color="#d62728", lw=2.6, ms=7,
           label="$G^2$ 검정의 $p$ 값 (중앙값)")
ax1.axhline(0.05, color="black", lw=1.4, ls="--")
ax1.text(NS[0] * 1.1, 0.075, "$\\alpha = 0.05$", fontsize=10)
ax1.set_xlabel("표본 수 $n$")
ax1.set_ylabel("$p$ 값")
ax1.set_ylim(1e-20, 3)
ax1.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1])
ax1.set_yticklabels(["$10^{-20}$", "$10^{-15}$", "$10^{-10}$", "$10^{-5}$",
                     "$1$"])
ax1b = ax1.twinx()
ax1b.plot(NS, med_psi, "s-", color="#1f77b4", lw=2.6, ms=7,
          label="PSI (중앙값)")
ax1b.axhline(psi_true, color="#1f77b4", lw=1.2, ls=":")
ax1b.set_ylabel("PSI", color="#1f77b4")
ax1b.tick_params(axis="y", labelcolor="#1f77b4")
ax1b.set_ylim(0, psi_true * 2.4)
ax1.set_xscale("log")                 # twinx 가 되돌린 것을 다시 건다
ax1.set_xticks(NS)
ax1.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax1.minorticks_off()
ax1b.minorticks_off()
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax1b.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9.5)
ax1.set_title("① 같은 이동, 다른 답", fontsize=12)
ax1.grid(alpha=0.25, which="major")

ax2.scatter(pairs[:, 1], pairs[:, 0], s=10, alpha=0.35, color="#1f77b4",
            edgecolors="none")
lim = pairs.max() * 1.05
ax2.plot([0, lim], [0, lim], color="#d62728", lw=1.8, ls="--",
         label="$\\mathrm{PSI} = G^2/n$")
ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
ax2.set_xlabel("$G^2 / n$")
ax2.set_ylabel("PSI")
ax2.set_title("② 둘은 거의 같은 수다", fontsize=12)
ax2.legend(loc="upper left", fontsize=11)
ax2.text(0.97, 0.06,
         f"상관 ${np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]:.4f}$",
         transform=ax2.transAxes, ha="right", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
ax2.grid(alpha=0.25)

for t, c in zip(PSI_T, ("#ff7f0e", "#d62728")):
    ax3.loglog(NS, tbl[t], "o-", color=c, lw=2.6, ms=7,
               label=f"PSI $= {t}$ 일 때의 $p$ 값")
ax3.axhline(0.05, color="black", lw=1.4, ls="--")
ax3.text(NS[0] * 1.1, 0.09, "$\\alpha = 0.05$", fontsize=10)
ax3.set_xlabel("표본 수 $n$")
ax3.set_ylabel("그 PSI 가 뜻하는 $p$ 값")
ax3.set_title("③ 같은 문턱, 다른 엄격함", fontsize=12)
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS], fontsize=9)
ax3.set_ylim(1e-30, 3)
ax3.set_yticks([1e-30, 1e-20, 1e-10, 1])
ax3.set_yticklabels(["$10^{-30}$", "$10^{-20}$", "$10^{-10}$", "$1$"])
ax3.minorticks_off()
ax3.legend(loc="lower left", fontsize=10)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch12/test/img/psi_vs_g2.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
