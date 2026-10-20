r"""「가장 많이 나온 번호」는 얼마나 많이 나와야 이상한가.

로또 $6/45$ 는 매 회 $45$ 개에서 $6$ 개를 **중복 없이** 뽑는다. 공이 완전히 공평해도
$45$ 개의 출현 횟수는 흩어지고, 그 가운데 **가장 큰 것**은 평균보다 한참 위에 있다.

① 회차가 $1{,}200$ 일 때 각 번호의 출현 횟수와, 최댓값·최솟값의 귀무분포.
② 회차가 쌓이면 **횟수 차이는 벌어지고 비율 차이는 좁아진다.**
③ $45$ 칸을 한꺼번에 보는 카이제곱과, 가장 큰 칸 하나만 보는 일의 차이.

만드는 파일: docs/ch10/test/img/lotto_uniform.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
K, PICK = 45, 6
N_SHOW = 1200
REP = 20000


def counts(n_draws, rep, rng):
    """회차마다 45 개에서 6 개를 중복 없이 뽑아 번호별 출현 횟수를 센다."""
    out = np.empty((rep, K), dtype=np.int32)
    for r in range(rep):
        # argsort 로 매 회차 6 개를 뽑는다 (복원 없음)
        picks = np.argsort(rng.random((n_draws, K)), axis=1)[:, :PICK]
        out[r] = np.bincount(picks.ravel(), minlength=K)
    return out


C = counts(N_SHOW, 1200, RNG)  # 그림 ①·③ 용
mx, mn = C.max(axis=1), C.min(axis=1)
E = N_SHOW * PICK / K
chi2 = ((C - E) ** 2 / E).sum(axis=1)

# === ② 회차가 쌓이면 ===============================================
NS = np.array([100, 200, 400, 800, 1200, 2000, 4000, 8000])
gap_cnt, gap_prop = [], []
for n in NS:
    c = counts(int(n), 300, RNG)
    g = c.max(axis=1) - c.min(axis=1)
    gap_cnt.append(g.mean())
    gap_prop.append((g / n).mean() * 100)
gap_cnt = np.array(gap_cnt)
gap_prop = np.array(gap_prop)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

one = C[0]
order = np.argsort(-one)
ax1.bar(range(K), one[order], color="#9e9e9e", width=0.86)
ax1.bar([0], [one[order][0]], color="#d62728", width=0.86)
ax1.bar([K - 1], [one[order][-1]], color="#1f77b4", width=0.86)
ax1.axhline(E, color="black", lw=1.6, ls="--")
ax1.text(K - 0.5, E + 2.5, f"기대 {E:.1f} 회", ha="right", fontsize=10)
ax1.annotate(
    f"가장 많이 나온 번호\n{one.max()} 회",
    xy=(0, one.max()),
    xytext=(9, one.max() + 18),
    fontsize=10,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
)
ax1.annotate(
    f"가장 적게 나온 번호\n{one.min()} 회",
    xy=(K - 1, one.min()),
    xytext=(K - 10, one.min() - 30),
    ha="center",
    fontsize=10,
    color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
)
ax1.set_xlabel(f"번호 {K} 개를 출현 횟수 순으로 ({N_SHOW:,} 회 추첨)")
ax1.set_ylabel("출현 횟수")
ax1.set_title("① 공이 공평해도 이만큼 흩어진다", fontsize=12)
ax1.set_ylim(0, one.max() + 42)
ax1.set_xticks([])
ax1.grid(alpha=0.25, axis="y")

ax2.loglog(NS, gap_cnt, "o-", color="#d62728", lw=2.4, ms=6, label="횟수의 차이")
ax2.set_xlabel("쌓인 추첨 회차")
ax2.set_ylabel("최다 $-$ 최소 (회)", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.set_xticks([100, 1000, 8000])
ax2.set_xticklabels(["$100$", "$1{,}000$", "$8{,}000$"])
ax2.set_yticks([20, 50, 100, 200])
ax2.set_yticklabels(["$20$", "$50$", "$100$", "$200$"])
ax2.minorticks_off()
ax2.grid(alpha=0.25, which="major")
ax2b = ax2.twinx()
ax2b.loglog(NS, gap_prop, "s-", color="#1f77b4", lw=2.4, ms=6, label="비율의 차이")
ax2b.set_ylabel("최다 $-$ 최소 (회차의 $\\%$)", color="#1f77b4")
ax2b.tick_params(axis="y", labelcolor="#1f77b4")
ax2b.set_yticks([2, 5, 10, 25])
ax2b.set_yticklabels(["$2$", "$5$", "$10$", "$25$"])
ax2b.minorticks_off()
ax2.set_title("② 차이는 벌어지고 비율은 좁아진다", fontsize=12)
# twinx 가 가로 눈금을 되돌리므로 마지막에 다시 건다
ax2.set_xticks([100, 1000, 8000])
ax2.set_xticklabels(["$100$", "$1{,}000$", "$8{,}000$"])
ax2.minorticks_off()
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax2b.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, loc="lower center", fontsize=10)

ax3.hist(chi2, bins=48, density=True, color="#9e9e9e", alpha=0.85, edgecolor="none",
         label="실제 추첨 방식")
xs = np.linspace(15, 85, 400)
ax3.plot(xs, stats.chi2.pdf(xs, K - 1), color="#d62728", lw=2.4,
         label=f"$\\chi^2_{{{K - 1}}}$")
crit = stats.chi2.isf(0.05, K - 1)
ax3.axvline(crit, color="#555555", lw=1.4, ls="--")
ax3.text(crit + 1.2, ax3.get_ylim()[1] * 0.82,
         f"$5\\%$ 임계값 ${crit:.1f}$", fontsize=10, color="#555555")
ax3.set_xlabel("$45$ 칸 전체로 잰 카이제곱")
ax3.set_ylabel("밀도")
ax3.set_title("③ 한꺼번에 보면 아무 일도 없다", fontsize=12)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch10/test/img/lotto_uniform.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
sd = np.sqrt(N_SHOW * (PICK / K) * (1 - PICK / K))
print(f"\n① 추첨 {N_SHOW:,} 회 — 번호 하나의 기대 출현 {E:.1f} 회, 표준편차 {sd:.2f} 회")
print(f"  최댓값: 평균 {mx.mean():.1f} 회 (중앙 95% {np.quantile(mx, .025):.0f}–{np.quantile(mx, .975):.0f})")
print(f"  최솟값: 평균 {mn.mean():.1f} 회 (중앙 95% {np.quantile(mn, .025):.0f}–{np.quantile(mn, .975):.0f})")
print(f"  최다 - 최소: 평균 {(mx - mn).mean():.1f} 회")
print(f"  최댓값을 z 로 옮기면 평균 {((mx.mean() - E) / sd):.2f}")
print(f"  그림에 쓴 한 판: 최다 {one.max()} 회, 최소 {one.min()} 회")

print("\n② 회차가 쌓이면")
print(f"{'회차':>8}{'최다-최소 (회)':>16}{'회차의 %':>12}")
for n, a, b in zip(NS, gap_cnt, gap_prop):
    print(f"{n:>8,}{a:>15.1f}{b:>11.2f}%")

print("\n③ 45 칸 전체로 잰 카이제곱 (자유도 44)")
print(f"  평균 {chi2.mean():.2f} (χ²_44 의 평균은 {K - 1})")
print(f"  5% 임계값 {crit:.2f} 를 넘는 비율: {np.mean(chi2 > crit) * 100:.2f}%")
print(f"  1 등 당첨 확률 1/C(45,6) = 1/{int(round(1 / stats.hypergeom.pmf(6, 45, 6, 6))):,}")
