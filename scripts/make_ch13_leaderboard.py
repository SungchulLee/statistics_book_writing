r"""공개 순위표는 떼어 둔 자료를 몇 번이나 쓰는가.

① 한 번의 대회: $1{,}000$ 팀의 공개 점수와 비공개 점수. 공개 상위권이 비공개에서 내려앉는다.
② 공개 $1$ 위가 비공개 $1$ 위이기도 할 확률 — 실력의 폭과 잡음의 비로 정해진다.

만드는 파일: docs/ch13/model_selection/img/leaderboard.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)

K = 1000  # 참가 팀
TAU = 1.0  # 실력의 표준편차
SIGMA = 0.5  # 한 자료집합에서 재었을 때의 잡음
TOP = 10

# === ① 한 번의 대회 =================================================
# 실력 theta 는 팀마다 다르고, 공개/비공개 점수는 같은 실력에 서로 다른
# 잡음이 얹힌 값이다. 둘은 「떼어 둔 자료」를 반씩 나눠 가진 셈이다.
theta = RNG.normal(0, TAU, K)
pub = theta + RNG.normal(0, SIGMA, K)
pri = theta + RNG.normal(0, SIGMA, K)

order = np.argsort(-pub)
top = order[:TOP]
fall = pub[top] - pri[top]


# === ② 공개 1 위가 비공개 1 위이기도 할 확률 ========================
def agree(k, ratio, rep=4000, rng=None):
    """ratio = tau/sigma. 공개 1 위와 비공개 1 위가 같을 확률."""
    rng = rng or np.random.default_rng(7)
    th = rng.normal(0, ratio, (rep, k))
    a = (th + rng.normal(0, 1, (rep, k))).argmax(axis=1)
    b = (th + rng.normal(0, 1, (rep, k))).argmax(axis=1)
    return float(np.mean(a == b))


ratios = np.linspace(0.1, 6.0, 24)

# === 그림 ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

ax1.scatter(pub, pri, s=10, color="#9e9e9e", alpha=0.45, edgecolors="none")
ax1.scatter(pub[top], pri[top], s=46, color="#d62728", zorder=5, label=f"공개 상위 {TOP} 팀")
for i in top:
    ax1.plot([pub[i], pub[i]], [pub[i], pri[i]], color="#d62728", lw=1.0, alpha=0.6)
lim = [-3.6, 4.6]
ax1.plot(lim, lim, color="black", lw=1.2, ls="--", label="공개 $=$ 비공개")
ax1.set_xlim(*lim)
ax1.set_ylim(*lim)
ax1.set_aspect("equal")
ax1.set_xlabel("공개 순위표 점수")
ax1.set_ylabel("비공개 점수")
ax1.set_title(f"① 한 번의 대회 — {K:,} 팀", fontsize=12)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25)
ax1.text(
    0.97,
    0.04,
    f"공개 상위 {TOP} 팀이 비공개에서\n평균 {fall.mean():.2f} 내려앉는다",
    transform=ax1.transAxes,
    ha="right",
    va="bottom",
    fontsize=10,
    color="#d62728",
)

for k, c in [(10, "#2ca02c"), (100, "#ff7f0e"), (1000, "#d62728")]:
    ys = [agree(k, r) for r in ratios]
    ax2.plot(ratios, ys, "o-", color=c, lw=2.2, ms=4, label=f"{k:,} 팀")
ax2.axhline(1.0, color="#555555", lw=1.0, ls=":")
ax2.set_xlabel("실력의 폭 $\\div$ 잡음의 크기  ($\\tau/\\sigma$)")
ax2.set_ylabel("공개 $1$ 위가 비공개 $1$ 위이기도 할 확률")
ax2.set_title("② 순위가 뒤집히는 까닭", fontsize=12)
ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, 6.2)
ax2.legend(loc="lower right", fontsize=10)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch13/model_selection/img/leaderboard.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① {K} 팀, tau = {TAU}, sigma = {SIGMA}  (tau/sigma = {TAU / SIGMA:g})")
print(f"  공개 상위 {TOP} 팀의 공개 점수 평균 {pub[top].mean():+.3f}")
print(f"  같은 팀들의 비공개 점수 평균     {pri[top].mean():+.3f}")
print(f"  평균 내려앉음                   {fall.mean():.3f}")
print(f"  같은 팀들의 참 실력 평균         {theta[top].mean():+.3f}")
print(f"  공개 1 위 팀의 비공개 등수: {int(np.sum(pri > pri[order[0]])) + 1} 위")
print(f"  비공개 1 위 팀의 공개 등수: {int(np.sum(pub > pub[np.argmax(pri)])) + 1} 위")
print(f"  공개 상위 {TOP} 안에 비공개 상위 {TOP} 가 몇 팀: "
      f"{len(set(top) & set(np.argsort(-pri)[:TOP]))} 팀")

print("\n② 공개 1 위 = 비공개 1 위일 확률")
print(f"{'tau/sigma':>10}{'10 팀':>10}{'100 팀':>10}{'1000 팀':>10}")
for r in (0.5, 1.0, 2.0, 3.0, 5.0):
    print(f"{r:>10.1f}{agree(10, r):>10.3f}{agree(100, r):>10.3f}{agree(1000, r):>10.3f}")
