r"""정규곡선을 검정하는 대신 **강제한** 제도 — 상대평가.

마이어($1908$)가 제안한 눈금은 정규곡선에서 바로 나온다. 극단점을 $3$ 으로 잡고
상위 집단의 경계를 $0.68$ 로 둔 뒤 그 둘의 중점 $1.84$ 를 다시 그으면
$3 / 22 / 50 / 22 / 3$ 퍼센트가 된다.

① 그 눈금을 확인한다 — $\Phi(-1.84) = 0.033$, $\Phi(-0.68) = 0.248$.
② 그가 스스로 던진 물음에 답한다 — 학생 다섯이나 여덟 명짜리 반에서 $3\%$ 는 몇 명인가.
③ 반마다 능력 수준이 다르면, 같은 능력의 학생이 전혀 다른 학점을 받는다.

만드는 파일: docs/ch14/introduction/img/grading_curve.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261022)

CUT = np.array([-1.84, -0.68, 0.68, 1.84])      # 마이어가 그은 네 금
GRADES = ["낙제", "열등", "보통", "우수", "최우수"]
P = np.array([0.03, 0.22, 0.50, 0.22, 0.03])    # 그가 적은 비율
NS = np.array([5, 8, 12, 20, 30, 45, 60, 90, 120, 200])
TAUS = np.array([0.0, 0.3, 0.6, 1.0])
Z_STAR = stats.norm.isf(0.10)                    # 전체에서 상위 10% 인 학생
NFIX, REPS = 24, 40000

print("① 마이어의 눈금")
edges = np.concatenate([[-np.inf], CUT, [np.inf]])
share = np.diff(stats.norm.cdf(edges))
for g, lo, hi, s in zip(GRADES, edges[:-1], edges[1:], share):
    print(f"   {g:>4} : ({lo:>6.2f}, {hi:>6.2f}]  {100 * s:>5.1f}%")
print(f"   $1.84$ 은 $0.68$ 과 $3$ 의 중점 : {(0.68 + 3) / 2}")

# === ② 작은 반에서 ===================================================
print(f"\n② 반 크기에 따라 「최우수」로 뽑히는 사람")
top = []
for n in NS:
    k = int(round(P[-1] * n))
    top.append(k / n)
    print(f"   n = {n:>4} : 기대 {P[-1] * n:>5.2f} 명 -> {k} 명  "
          f"(실제 {100 * k / n:>4.1f}%)")
top = np.array(top)
n_zero = int(NS[np.argmax(top > 0)])
print(f"   「최우수」가 한 명이라도 나오려면 n >= {n_zero}")

# === ③ 같은 능력, 다른 학점 ==========================================
cum = np.cumsum(P[::-1])[:-1]          # 위에서부터 자른 누적 비율
print(f"\n③ 전체 상위 10%(z = {Z_STAR:.2f}) 인 학생이 받는 학점 "
      f"(반 크기 {NFIX}, {REPS:,} 번)")
dist = {}
for tau in TAUS:
    mu = RNG.normal(0.0, tau, REPS)
    others = RNG.normal(mu[:, None], 1.0, (REPS, NFIX - 1))
    better = (others > Z_STAR).sum(axis=1)          # 나보다 나은 급우 수
    frac = (better + 0.5) / NFIX                     # 반 안의 상위 비율
    gi = 4 - np.searchsorted(cum, frac, side="left")  # 0=낙제 … 4=최우수
    gi = np.clip(gi, 0, 4)
    dist[tau] = np.array([(gi == k).mean() for k in range(5)])
    print(f"   tau = {tau} : " + ",  ".join(
        f"{g} {100 * v:.1f}%" for g, v in zip(GRADES[::-1], dist[tau][::-1])))

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = ["#8c564b", "#ff7f0e", "#cccccc", "#1f77b4", "#2ca02c"]

xs = np.linspace(-3.4, 3.4, 600)
pdf = stats.norm.pdf(xs)
ax1.plot(xs, pdf, color="black", lw=1.8)
for k in range(5):
    m = (xs > edges[k]) & (xs <= edges[k + 1])
    ax1.fill_between(xs[m], pdf[m], color=COL[k], alpha=0.75)
    c = 0.5 * (max(edges[k], -3.4) + min(edges[k + 1], 3.4))
    ax1.text(c, stats.norm.pdf(c) * 0.45 + 0.012,
             f"{GRADES[k]}\n${100 * share[k]:.0f}\\%$", ha="center",
             fontsize=10.5)
for z in CUT:
    ax1.axvline(z, color="#555555", lw=1.1, ls=":")
ax1.set_xticks(CUT)
ax1.set_xticklabels([f"${v}$" for v in CUT], fontsize=10)
ax1.set_xlabel("능력 (표준편차 단위)")
ax1.set_ylabel("밀도")
ax1.set_ylim(0, 0.47)
ax1.set_title("① 금 네 개로 다섯 등급", fontsize=12)
ax1.grid(alpha=0.2)

ax2.step(NS, 100 * top, where="mid", color="#2ca02c", lw=2.8)
ax2.plot(NS, 100 * top, "o", color="#2ca02c", ms=8)
ax2.axhline(3.0, color="#d62728", lw=1.8, ls="--", label="제도가 적은 $3\\%$")
ax2.axvspan(4.5, 8.5, color="#d62728", alpha=0.13)
ax2.annotate("마이어가 물은 자리\n「다섯이나 여덟 명짜리 반」",
             xy=(8, 0.4), xytext=(16, 6.0), fontsize=10.5, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xscale("log")
ax2.set_xticks(NS)
ax2.set_xticklabels([f"${v}$" for v in NS], fontsize=8.5)
ax2.minorticks_off()
ax2.set_xlabel("반의 학생 수")
ax2.set_ylabel("실제로 「최우수」가 되는 비율 ($\\%$)")
ax2.set_ylim(-0.4, 8.5)
ax2.set_title("② 작은 반에서는 아무도 없다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25)

bottom = np.zeros(len(TAUS))
xpos = np.arange(len(TAUS))
for k in range(4, -1, -1):
    vals = np.array([100 * dist[t][k] for t in TAUS])
    ax3.bar(xpos, vals, 0.6, bottom=bottom, color=COL[k], label=GRADES[k])
    for x, (b, v) in enumerate(zip(bottom, vals)):
        if v > 6:
            ax3.text(x, b + v / 2, f"${v:.0f}$", ha="center", va="center",
                     fontsize=10)
    bottom += vals
ax3.set_xticks(xpos)
ax3.set_xticklabels([f"$\\tau = {t}$" for t in TAUS], fontsize=10.5)
ax3.set_xlabel("반 사이 평균 능력의 흩어짐")
ax3.set_ylabel("받는 학점의 분포 ($\\%$)")
ax3.set_ylim(0, 100)
ax3.set_title(f"③ 전체 상위 $10\\%$ 인 한 학생 (반 {NFIX} 명)", fontsize=12)
ax3.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5)
ax3.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch14/introduction/img/grading_curve.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
