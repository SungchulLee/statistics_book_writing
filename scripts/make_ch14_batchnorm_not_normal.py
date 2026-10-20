r"""정규화(normalization)는 정규성(normality)이 아니다.

배치정규화는 평균과 분산 **둘만** 맞춘다. 한 층을 통과한 활성값에 그것을 걸면
평균 $0$, 분산 $1$ 이 정확히 되지만 모양은 정규와 멀다. 모양을 정하는 것은
정규화가 아니라 **활성함수**다.

만드는 파일: docs/ch14/introduction/img/batchnorm_not_normal.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
BATCH = 2048
D_IN = 64

# === 한 층을 흉내 낸다 ==============================================
# 입력 -> 선형결합 -> ReLU 가 신경망의 한 칸이다. 배치정규화는 그 뒤에 붙어
# 미니배치 안에서 평균을 빼고 표준편차로 나눈다.

x = RNG.standard_normal((BATCH, D_IN))
W = RNG.standard_normal((D_IN, 1)) / np.sqrt(D_IN)
b = -0.35
pre = (x @ W).ravel() + b
ACTS = {
    "항등 (선형)": lambda t: t,
    "$\\tanh$": np.tanh,
    "ReLU": lambda t: np.maximum(t, 0.0),
}


def batchnorm(v):
    """배치정규화가 하는 일 전부 — 미니배치의 평균을 빼고 표준편차로 나눈다."""
    return (v - v.mean()) / v.std(ddof=0)


act = ACTS["ReLU"](pre)
bn = batchnorm(act)


def moments(v):
    return v.mean(), v.std(ddof=0), stats.skew(v), stats.kurtosis(v)


def sw(v, m=500):
    """샤피로-윌크는 n<=5000 이므로 그대로 쓴다."""
    s = stats.shapiro(v[:m])
    return s.statistic, s.pvalue


# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.7))

grid = np.linspace(-3.2, 4.2, 400)

ax1.hist(act, bins=60, color="#8c8c8c", edgecolor="none", alpha=0.85)
ax1.set_xlabel("활성값 $\\mathrm{ReLU}(w^\\top x + b)$")
ax1.set_ylabel("빈도")
ax1.set_title("① 층을 지난 값 — 정규화 전", fontsize=12)
ax1.grid(alpha=0.25)
ax1.annotate(
    f"$0$ 에 쌓인 몫 {np.mean(act == 0) * 100:.0f}$\\%$",
    xy=(0.0, np.mean(act == 0) * BATCH * 0.55),
    xytext=(0.28, BATCH * 0.30),
    fontsize=10,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

ax2.hist(
    bn,
    bins=60,
    density=True,
    color="#1f77b4",
    edgecolor="none",
    alpha=0.8,
    label="배치정규화 뒤",
)
ax2.plot(grid, stats.norm.pdf(grid), color="black", lw=1.6, ls="--", label="$N(0,1)$")
ax2.set_xlim(-1.5, 4.0)
ax2.set_xlabel("표준화한 활성값")
ax2.set_ylabel("밀도")
ax2.set_title("② 평균 $0$ · 분산 $1$ 이 되었는데", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25)

pp = (np.arange(1, BATCH + 1) - 0.375) / (BATCH + 0.25)
zq = stats.norm.ppf(pp)
for (lab, f), c in zip(ACTS.items(), ["#2ca02c", "#ff7f0e", "#1f77b4"]):
    ax3.plot(zq, np.sort(batchnorm(f(pre))), color=c, lw=2.4, label=lab)
ax3.plot([-4, 4], [-4, 4], color="black", lw=1.2, ls="--", label="정규라면")
ax3.annotate(
    "평평한 토막 = $0$ 에 쌓인 원자\n어떤 단조변환도 이것을 못 편다",
    xy=(-1.2, -0.525),
    xytext=(0.0, -3.3),
    ha="center",
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)
ax3.set_xlim(-3.8, 3.8)
ax3.set_ylim(-3.8, 4.4)
ax3.set_xlabel("정규 분위수")
ax3.set_ylabel("표본 분위수")
ax3.set_title("③ 모양을 정하는 것은 활성함수다", fontsize=12)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch14/introduction/img/batchnorm_not_normal.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\nReLU 가 0 으로 만든 몫: {np.mean(act == 0) * 100:.1f}%")
print(f"{'':<14}{'평균':>9}{'표준편차':>9}{'왜도':>9}{'초과첨도':>10}")
for lab, v in (("정규화 전", act), ("배치정규화 뒤", bn)):
    m, s, sk, ku = moments(v)
    print(f"{lab:<14}{m:>9.4f}{s:>9.4f}{sk:>9.3f}{ku:>10.3f}")

print("\n활성함수별 — 배치정규화를 건 뒤")
print(f"{'':<14}{'평균':>9}{'표준편차':>9}{'왜도':>9}{'초과첨도':>10}{'W':>9}{'p':>12}")
for lab, f in ACTS.items():
    v = batchnorm(f(pre))
    m, sd_, sk, ku = moments(v)
    w, pv = sw(v)
    print(f"{lab:<14}{m:>9.4f}{sd_:>9.4f}{sk:>9.3f}{ku:>10.3f}{w:>9.4f}{pv:>12.3g}")

print("\n꼬리가 어긋나는 폭 — 분위수를 바로 견준다")
for q in (0.01, 0.05, 0.95, 0.99):
    print(
        f"  {q * 100:>5.1f}%:  배치정규화 {np.quantile(bn, q):+.3f}   "
        f"정규 {stats.norm.ppf(q):+.3f}"
    )
