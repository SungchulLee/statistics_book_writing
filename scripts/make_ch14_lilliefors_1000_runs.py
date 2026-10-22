r"""$1967$ 년에 릴리포스가 돌린 「표본 크기마다 $1000$ 회」를 그대로 되풀이한다.

① $K = 1000$ 으로 만든 $5\%$ 임계값 표는 $n$ 을 따라 **울퉁불퉁하다.**
   같은 절차를 세 번 되풀이하면 세 개의 다른 울퉁불퉁한 표가 나온다.
   매끄러운 기준선은 몰랭–압디($1998$)의 $K = 100{,}000$ 표다.
② 그렇게 만든 표를 실제로 쓰면 유의수준이 $0.05$ 에서 얼마나 흔들리는가.

만드는 파일: docs/ch14/formal_tests/img/lilliefors_1000_runs.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(19670101)
K_OLD = 1_000  # 릴리포스와 판수스트가 n 마다 쓴 표본 수
K_NEW = 100_000  # 몰랭–압디 1998

# 몰랭–압디(1998) 표 1 의 alpha = 0.05 열 — K = 100,000 으로 다시 구한 값
MA_05 = {
    4: 0.3754, 5: 0.3427, 6: 0.3245, 7: 0.3041, 8: 0.2875, 9: 0.2744,
    10: 0.2616, 11: 0.2506, 12: 0.2426, 13: 0.2337, 14: 0.2257, 15: 0.2196,
    16: 0.2128, 17: 0.2071, 18: 0.2018, 19: 0.1965, 20: 0.1920, 25: 0.1726,
    30: 0.1590, 31: 0.1559, 32: 0.1542, 33: 0.1518, 34: 0.1497, 35: 0.1478,
    36: 0.1454, 37: 0.1436, 38: 0.1421, 39: 0.1402, 40: 0.1386, 41: 0.1373,
    42: 0.1353, 43: 0.1339, 44: 0.1322, 45: 0.1309, 46: 0.1293, 47: 0.1282,
    48: 0.1269, 49: 0.1256, 50: 0.1246,
}
NS = sorted(MA_05)


def dl(x):
    """릴리포스 통계량 D_L. x 는 (rep, n) 꼴."""
    n = x.shape[1]
    z = (x - x.mean(1, keepdims=True)) / x.std(1, ddof=1, keepdims=True)
    f = stats.norm.cdf(np.sort(z, axis=1))
    up = np.arange(1, n + 1) / n - f
    lo = f - np.arange(n) / n
    return np.maximum(up.max(1), lo.max(1))


def crit(n, k, alpha=0.05):
    """표본 k 개로 만든 임계값 — 릴리포스가 한 그대로."""
    return np.quantile(dl(RNG.standard_normal((k, n))), 1 - alpha)


# === ① K = 1000 짜리 표 세 장 ========================================
tables = [np.array([crit(n, K_OLD) for n in NS]) for _ in range(3)]
ma = np.array([MA_05[n] for n in NS])

# 우리 구현이 몰랭–압디 표를 재현하는지 먼저 확인한다
check_ns = [5, 10, 20, 30, 50]
print(f"① 구현 확인 — K = {K_NEW:,} 로 다시 구한 값과 몰랭–압디 표 1")
print("     n     우리 값   몰랭–압디     차")
mine_big = {}
for n in check_ns:
    c = crit(n, K_NEW)
    mine_big[n] = c
    print(f"  {n:>4}    {c:.4f}      {MA_05[n]:.4f}   {c - MA_05[n]:+.4f}")

print(f"\n   K = {K_OLD:,} 짜리 표 세 장이 몰랭–압디 표에서 벗어난 정도")
for j, t in enumerate(tables, 1):
    d = t - ma
    print(f"   {j} 번째:  평균 {d.mean():+.5f}   최대 편차 {np.abs(d).max():.5f}"
          f"   상대오차 최대 {np.abs(d / ma).max() * 100:.2f}%")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.2))

COL = ["#1f77b4", "#2ca02c", "#d62728"]
for j, (t, c) in enumerate(zip(tables, COL), 1):
    ax1.plot(NS, t, "-", color=c, lw=1.5, alpha=0.9,
             label=f"$K = 1{{,}}000$ 으로 만든 표 ({j} 번째)")
ax1.plot(NS, ma, "k-", lw=2.6, label=f"$K = 100{{,}}000$ (몰랭–압디 $1998$)")
ax1.set_xlabel("표본 크기 $n$")
ax1.set_ylabel("$5\\%$ 임계값")
ax1.set_title("① 표 한 장이 아니라 표 한 장의 「한 번 뽑은 값」이다", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25)
ax1.set_xlim(3, 51)

# 울퉁불퉁한 구간을 확대해 보인다
axin = ax1.inset_axes([0.40, 0.40, 0.56, 0.32])
for t, c in zip(tables, COL):
    axin.plot(NS, t, "-", color=c, lw=1.5)
axin.plot(NS, ma, "k-", lw=2.4)
axin.set_xlim(29.5, 50.5)
axin.set_ylim(0.118, 0.168)
axin.tick_params(labelsize=8)
axin.grid(alpha=0.25)
ax1.indicate_inset_zoom(axin, edgecolor="#666666")

# === ② 그 표를 쓰면 유의수준이 얼마가 되는가 ==========================
N2 = 20
REF = dl(RNG.standard_normal((400_000, N2)))  # 참값을 재는 자
real = []
for _ in range(400):
    c = np.quantile(dl(RNG.standard_normal((K_OLD, N2))), 0.95)
    real.append(float((REF > c).mean()))
real = np.array(real)
c_ma = MA_05[N2]
a_ma = float((REF > c_ma).mean())
print(f"\n② n = {N2} 에서 K = {K_OLD:,} 짜리 표를 400 번 새로 만들어 쓴 결과")
print(f"   실제 유의수준:  평균 {real.mean():.4f}   표준편차 {real.std(ddof=1):.4f}")
print(f"                  5 백분위 {np.quantile(real, 0.05):.4f}"
      f"   95 백분위 {np.quantile(real, 0.95):.4f}")
print(f"   몰랭–압디 표의 {c_ma} 을 쓰면 {a_ma:.4f}")

ax2.hist(real, bins=34, color="#1f77b4", alpha=0.78, edgecolor="white")
ax2.axvline(0.05, color="black", lw=2.0, ls="--", label="약속한 $0.05$")
ax2.axvline(real.mean(), color="#d62728", lw=1.8,
            label=f"$K = 1{{,}}000$ 표들의 평균 ${real.mean():.4f}$")
ax2.set_xlabel(f"$n = {N2}$ 에서 실제로 나오는 유의수준")
ax2.set_ylabel("표 몇 장")
ax2.set_title("② 같은 절차로 만든 표 $400$ 장이 내는 유의수준", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25, axis="y")
ax2.text(
    0.03, 0.96,
    f"표준편차 ${real.std(ddof=1):.4f}$\n"
    f"$90\\%$ 가 ${np.quantile(real, 0.05):.3f}$–${np.quantile(real, 0.95):.3f}$ 사이",
    transform=ax2.transAxes, ha="left", va="top", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#cc9900", lw=1.0),
)

fig.tight_layout()
out = "docs/ch14/formal_tests/img/lilliefors_1000_runs.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)
