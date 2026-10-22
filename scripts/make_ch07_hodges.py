r"""호지스의 초효율 추정량 — 공짜로 보이는 것의 값은 어디에 적혀 있는가.

$X_1,\dots,X_n \sim \mathcal{N}(\theta, 1)$ 에서 $a_n = n^{-1/4}$ 로 두고

    theta_hat = Xbar  if |Xbar| >= a_n,   0  otherwise

이면 $\theta = 0$ 에서 점근분산이 $0$ 이고 다른 모든 $\theta$ 에서는 MLE 와 같다.
「어디서도 지지 않고 한 점에서 이긴다」로 보이지만, 위험함수를 그려 보면 값이 어디에
적혀 있는지 드러난다.

위험 $R_n(\theta) = n\,E[(\hat\theta - \theta)^2]$ 는 닫힌 꼴로 적힌다.
$L = \sqrt n(-a_n-\theta)$, $U = \sqrt n(a_n-\theta)$, $p = \Phi(U)-\Phi(L)$ 일 때

    R_n(theta) = 1 - p + (U phi(U) - L phi(L)) + n theta^2 p

① $n$ 마다 위험함수. $0$ 에서 파이는 만큼 옆에서 솟는다.
② 봉우리의 높이와 자리를 $n$ 을 따라 쫓는다.

만드는 파일: docs/ch07/theory/img/hodges_risk.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

RNG = np.random.default_rng(19510101)


def risk(theta, n):
    """n * E[(theta_hat - theta)^2] — 닫힌 꼴."""
    theta = np.asarray(theta, dtype=float)
    a = n ** (-0.25)
    rt = np.sqrt(n)
    L, U = rt * (-a - theta), rt * (a - theta)
    p = stats.norm.cdf(U) - stats.norm.cdf(L)
    trunc = p - (U * stats.norm.pdf(U) - L * stats.norm.pdf(L))  # E[Z^2 1{L<Z<U}]
    return 1 - trunc + n * theta**2 * p


def risk_sim(theta, n, rep=400_000):
    """같은 값을 모의실험으로 — 식이 맞는지 확인한다."""
    a = n ** (-0.25)
    xbar = theta + RNG.standard_normal(rep) / np.sqrt(n)
    est = np.where(np.abs(xbar) >= a, xbar, 0.0)
    return n * np.mean((est - theta) ** 2)


NS = [10, 100, 1_000, 10_000]
print("① 식과 모의실험이 맞는가 (위험 = n E[(theta_hat - theta)^2])")
print("      n    theta      식        모의실험")
for n in NS:
    for th in [0.0, n ** (-0.25) * 0.8, 0.5]:
        print(f"  {n:>6}   {th:6.4f}   {risk(th, n):8.4f}   {risk_sim(th, n):8.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

# === ① 위험함수 ======================================================
COL = ["#1f77b4", "#2ca02c", "#d9a441", "#d62728"]
th = np.linspace(-0.75, 0.75, 4001)
for n, c in zip(NS, COL):
    ax1.plot(th, risk(th, n), color=c, lw=2.1, label=f"$n = {n:,}$")
ax1.axhline(1.0, color="#333333", lw=1.8, ls="--", label="MLE ($\\bar X$) 의 위험")
ax1.set_yscale("log")
ax1.set_xlim(-0.75, 0.75)
ax1.set_ylim(0.02, 40)
ax1.set_xlabel("참값 $\\theta$")
ax1.set_ylabel("위험 $n\\,E[(\\hat\\theta - \\theta)^2]$ (로그 눈금)")
ax1.set_title("① $\\theta = 0$ 에서 파인 만큼 바로 옆에서 솟는다", fontsize=12)
ax1.legend(loc="upper right", fontsize=9.5, ncol=2)
ax1.grid(alpha=0.25, which="both")
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
ax1.set_yticks([0.1, 1, 10])

# === ② 봉우리의 높이와 자리 ==========================================
NN = np.unique(np.round(np.logspace(1, 6, 40)).astype(int))
peak_h, peak_t = [], []
for n in NN:
    a = n ** (-0.25)
    r = minimize_scalar(lambda t: -risk(t, n), bounds=(1e-6, 3 * a), method="bounded")
    peak_h.append(-r.fun)
    peak_t.append(r.x)
peak_h, peak_t = np.array(peak_h), np.array(peak_t)

ax2.loglog(NN, peak_h, "o-", color="#d62728", lw=2.2, ms=5, label="봉우리의 높이")
ax2.loglog(NN, peak_h[0] * (NN / NN[0]) ** 0.5, "--", color="#999999", lw=1.4,
           label="$\\sqrt{n}$ 에 비례하는 선 (견줌)")
ax2.axhline(1.0, color="#333333", lw=1.8, ls="--")
ax2.text(12, 1.18, "MLE 의 위험 $= 1$", fontsize=10.5, ha="left")
ax2.set_xlabel("표본 크기 $n$")
ax2.set_ylabel("$\\max_\\theta$ 위험")
ax2.set_title("② 표본을 키울수록 봉우리가 높아진다", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, which="both")
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

axin = ax2.inset_axes([0.60, 0.14, 0.36, 0.30])
axin.loglog(NN, peak_t, color="#1f77b4", lw=2.0)
axin.loglog(NN, NN ** (-0.25), "--", color="#999999", lw=1.2)
axin.set_title("봉우리의 자리 $\\theta$", fontsize=9)
axin.tick_params(labelsize=7.5)
axin.grid(alpha=0.25, which="both")
axin.set_xticks([10, 1_000, 100_000])
axin.set_yticks([0.1, 0.5])
axin.minorticks_off()
axin.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
axin.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

fig.tight_layout()
out = "docs/ch07/theory/img/hodges_risk.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

# === 본문에 적을 수 =================================================
print("\n② 고정된 theta 에서는 어떻게 되는가 (n 을 키우며)")
print("        n   theta=0   theta=0.1   theta=0.5   봉우리 높이   봉우리 자리")
for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    a = n ** (-0.25)
    r = minimize_scalar(lambda t: -risk(t, n), bounds=(1e-6, 3 * a), method="bounded")
    print(f"  {n:>7,}  {risk(0.0, n):8.4f}  {risk(0.1, n):10.4f}  "
          f"{risk(0.5, n):10.4f}  {-r.fun:11.2f}  {r.x:11.5f}")
print("\n   견줌 — 문턱 a_n = n^{-1/4}")
for n in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
    print(f"  {n:>7,}   a_n = {n ** -0.25:.5f}")
