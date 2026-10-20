r"""샤피로-윌크의 표가 왜 $n$ 마다 따로였고, 어떻게 한 식으로 접혔는가.

① 귀무분포가 $n$ 마다 다른 모양이라 콜모고로프식 「한 장짜리 표」가 불가능하다.
② $\ln(1-W)$ 의 평균과 표준편차는 $n$ 의 매끄러운 함수다.
③ 그 둘로 표준화하면 모든 $n$ 이 하나의 정규 직선 위로 포개진다 — 로이스턴의 길.

만드는 파일: docs/ch14/formal_tests/img/shapiro_table_wall.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261020)
REP = 4000
SHOW = [10, 20, 50, 200, 1000]
ALL_N = [10, 15, 20, 30, 50, 80, 120, 200, 350, 600, 1000, 2000]


def w_null(n, rep=REP):
    """표준정규에서 뽑은 표본 rep 개의 W 값."""
    x = RNG.standard_normal((rep, n))
    return np.array([stats.shapiro(row).statistic for row in x])


table = {n: w_null(n) for n in ALL_N}

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.7))

COL = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# ① n 마다 다른 귀무분포
for n, c in zip(SHOW, COL):
    v = np.log10(1 - table[n])
    xs = np.linspace(v.min(), v.max(), 400)
    ax1.plot(10**xs, stats.gaussian_kde(v)(xs), color=c, lw=2.2, label=f"$n = {n}$")
    ax1.plot([1 - np.quantile(table[n], 0.05)], [0], marker="v", color=c,
             ms=8, clip_on=False)
ax1.set_xscale("log")
ax1.set_xlim(2e-4, 0.6)
ax1.set_ylim(bottom=0)
ax1.set_xticks([1e-3, 1e-2, 1e-1])
ax1.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
ax1.minorticks_off()
ax1.set_xlabel("$1 - W$ (로그 눈금)   삼각형 = 그 $n$ 의 $5\\%$ 임계값")
ax1.set_ylabel("밀도 (로그 눈금 위에서)")
ax1.set_title("① 귀무분포가 $n$ 마다 다르다", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25)

# ② ln(1-W) 의 두 모멘트는 n 의 매끄러운 함수
ns = np.array(ALL_N, dtype=float)
mu = np.array([np.log1p(-table[n]).mean() for n in ALL_N])
sd = np.array([np.log1p(-table[n]).std(ddof=1) for n in ALL_N])

ax2.semilogx(ns, mu, "o-", color="#1f77b4", lw=2.0, ms=6, label="평균 $\\mu(n)$")
ax2.semilogx(ns, sd, "s-", color="#d62728", lw=2.0, ms=6, label="표준편차 $\\sigma(n)$")
ax2.set_xlabel("표본 크기 $n$")
ax2.set_ylabel("$\\ln(1 - W)$ 의 모멘트")
ax2.set_title("② 두 곡선만 적어 두면 된다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25, which="both")
ax2.set_xticks([10, 100, 1000])
ax2.set_xticklabels(["$10$", "$100$", "$1000$"])
ax2.minorticks_off()
ax2.axvline(50, color="#777777", lw=1.0, ls=":")
ax2.annotate(
    "$1965$ 년 표가 멈춘 자리",
    xy=(50, -2.2),
    xytext=(130, -1.35),
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

# ③ 표준화하면 하나로 포개진다
pp = (np.arange(1, REP + 1) - 0.375) / (REP + 0.25)
zq = stats.norm.ppf(pp)
for n, c in zip(SHOW, COL):
    v = np.log1p(-table[n])
    z = np.sort((v - v.mean()) / v.std(ddof=1))
    ax3.plot(zq, z, color=c, lw=2.0, alpha=0.8, label=f"$n = {n}$")
ax3.plot([-4, 4], [-4, 4], color="black", lw=1.2, ls="--", label="$N(0,1)$")
ax3.set_xlim(-3.6, 3.6)
ax3.set_ylim(-3.6, 4.2)
ax3.set_xlabel("정규 분위수")
ax3.set_ylabel("표준화한 $\\ln(1-W)$ 의 분위수")
ax3.set_title("③ 표준화하면 한 직선 위로", fontsize=12)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch14/formal_tests/img/shapiro_table_wall.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① n 마다 다른 5% 임계값 (모의실험 4000 회)")
for n in [10, 20, 50, 200, 1000]:
    w = table[n]
    print(
        f"  n = {n:>4}:  5% 임계값 {np.quantile(w, 0.05):.4f}   "
        f"중앙값 {np.median(w):.4f}"
    )

print("\n② ln(1-W) 의 두 모멘트")
for n, m, s in zip(ALL_N, mu, sd):
    print(f"  n = {n:>4}:  mu = {m:+.4f}   sigma = {s:.4f}")
c = np.polyfit(np.log(ns), mu, 1)
print(f"  mu 를 ln n 의 1 차로 맞추면 기울기 {c[0]:+.4f}, 절편 {c[1]:+.4f}")

print("\n③ 표준화한 ln(1-W) 의 왜도·첨도 (정규면 0, 0)")
for n in SHOW:
    v = np.log1p(-table[n])
    z = (v - v.mean()) / v.std(ddof=1)
    print(f"  n = {n:>4}:  왜도 {stats.skew(z):+.3f}   초과첨도 {stats.kurtosis(z):+.3f}")
print("\n   견줌 — 표준화하기 전 W 자신의 왜도")
for n in SHOW:
    print(f"  n = {n:>4}:  왜도 {stats.skew(table[n]):+.3f}")
