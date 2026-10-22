r"""푸앵카레가 $1900$ 년 심사보고서에 단 단서 — 「편차가 너무 크지 않다면」.

바슐리에의 가정은 **가격의 변동 확률이 가격 수준에 기대지 않는다**는 것이고,
거기서 가격 자체가 정규인 모형(산술 브라운 운동)이 따라 나온다. 오늘 쓰는 모형은
**로그가격**이 정규인 모형(기하 브라운 운동)이다. 둘은 작은 편차에서 겹치고
큰 편차에서 갈린다 — 푸앵카레가 적은 단서가 바로 그 자리다.

① 두 모형이 주는 만기 가격의 분포. 바슐리에 쪽은 $0$ 아래로 새어 나간다.
② $P(S_T \le 0)$ — 그 샘이 기간에 따라 얼마나 커지는가.

만드는 파일: docs/ch14/introduction/img/bachelier_1900.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

S0 = 100.0
SIGMAS = [0.15, 0.20, 0.30]  # 연 변동성
SIG = 0.20
HORIZONS = [0.25, 1.0, 5.0]  # 해 단위


def bachelier(T, sigma=SIG):
    """가격 자체가 정규: S_T ~ N(S0, (S0 sigma)^2 T)."""
    return stats.norm(loc=S0, scale=S0 * sigma * np.sqrt(T))


def lognormal(T, sigma=SIG):
    """로그가격이 정규: ln S_T ~ N(ln S0 - sigma^2 T/2, sigma^2 T)."""
    return stats.lognorm(s=sigma * np.sqrt(T), scale=S0 * np.exp(-(sigma**2) * T / 2))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

# === ① 만기 가격의 분포 ==============================================
COL = ["#1f77b4", "#2ca02c", "#d62728"]
xs = np.linspace(-80, 300, 1200)
print(f"① S0 = {S0:.0f}, 연 변동성 {SIG:.0%} 에서 만기 가격의 분포")
for T, c in zip(HORIZONS, COL):
    b, g = bachelier(T), lognormal(T)
    ax1.plot(xs, b.pdf(xs), color=c, lw=2.2, ls="--")
    ax1.plot(xs, g.pdf(xs), color=c, lw=2.2,
             label=f"$T = {T:g}$ 해")
    print(f"   T = {T:>4g} 해:  바슐리에 표준편차 {b.std():6.2f}   "
          f"P(S<=0) = {b.cdf(0):.6f}   "
          f"두 모형의 1% 분위수 {b.ppf(0.01):7.2f} 대 {g.ppf(0.01):6.2f}")
ax1.axvspan(-80, 0, color="#d62728", alpha=0.10)
ax1.axvline(0, color="#d62728", lw=1.2)
ax1.set_xlim(-80, 300)
ax1.set_ylim(bottom=0)
ax1.set_xlabel("만기 가격 (시작값 $100$)")
ax1.set_ylabel("밀도")
ax1.set_title("① 실선은 로그가 정규, 파선은 가격이 정규", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25)
ax1.text(
    0.012, 0.97, "가격이 음수인 구역",
    transform=ax1.transAxes, ha="left", va="top", fontsize=10, color="#d62728",
    rotation=90,
)

# === ② P(S_T <= 0) ===================================================
Ts = np.logspace(np.log10(1 / 52), np.log10(30), 300)
print("\n② 가격이 정규인 모형이 음수에 주는 확률")
print(f"{'기간':>10}" + "".join(f"{s:>12.0%}" for s in SIGMAS))
for T, lab in [(1 / 52, "한 주"), (1 / 12, "한 달"), (1.0, "한 해"),
               (5.0, "다섯 해"), (30.0, "서른 해")]:
    row = "".join(f"{bachelier(T, s).cdf(0):>12.2e}" for s in SIGMAS)
    print(f"{lab:>10}{row}")

for s, c in zip(SIGMAS, COL):
    ax2.loglog(Ts, [bachelier(t, s).cdf(0) for t in Ts], color=c, lw=2.3,
               label=f"연 변동성 ${s * 100:.0f}\\%$")
ax2.axhline(0.01, color="#777777", lw=1.1, ls=":")
ax2.text(0.013, 0.0125, "$1\\%$", fontsize=10, color="#555555")
ax2.set_xlim(Ts[0], Ts[-1])
ax2.set_ylim(1e-12, 0.5)
ax2.set_xlabel("기간 (해)")
ax2.set_ylabel("$P(S_T \\leq 0)$")
ax2.set_title("② 「편차가 너무 크지 않다면」이 깨지는 속도", fontsize=12)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, which="both")
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_major_formatter(
    FuncFormatter(lambda v, _: f"{v:g}" if v >= 1e-3 else f"10^{int(round(np.log10(v)))}")
)
ax2.set_xticks([0.1, 1, 10])
ax2.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1e-1])
ax2.set_yticklabels(["$10^{-12}$", "$10^{-9}$", "$10^{-6}$", "$10^{-3}$", "$10^{-1}$"])
ax2.minorticks_off()
for T, lab in [(1 / 52, "한 주"), (1.0, "한 해"), (10.0, "열 해")]:
    ax2.axvline(T, color="#aaaaaa", lw=0.8, ls="--")
    ax2.text(T, 1.6e-12, lab, fontsize=9.5, color="#666666", ha="center",
             bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))

fig.tight_layout()
out = "docs/ch14/introduction/img/bachelier_1900.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

# === 본문에 적을 수 =================================================
print("\n③ 두 모형이 갈리는 자리 — 편차가 커질수록")
# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print("  편차        바슐리에        로그정규         비")
b, g = bachelier(1.0), lognormal(1.0)
for k in [1, 2, 3, 4]:
    lo = S0 - k * S0 * SIG
    pb, pg = b.cdf(lo), g.cdf(lo)
    print(f"{-k:>4}σ  {pb:>13.3e}  {pg:>13.3e}  {pb / pg:>12.3g}")
