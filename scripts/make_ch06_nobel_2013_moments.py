r"""반대되는 두 주장이 같은 해 상을 나눠 가졌다 — $2013$ 년과 적률 조건.

파마는 주가가 짧은 앞날에는 예측되지 않는다고 했고 실러는 긴 앞날에는 된다고
했다. 한센은 **가능도를 적지 않고 적률 조건만으로** 그런 주장을 검정하는 방법을
세웠다. 셋이 $2013$ 년 상을 함께 받았다. 실러의 자료로 그 셋을 한 그림에 놓는다.

① 한 달 앞 — $1/\mathrm{CAPE}$ 로 다음 달을 맞히려 하면 구름뿐이다.
② 십 년 앞 — 같은 설명변수, 같은 자료인데 기울기가 또렷하다.
③ 겹치는 표본에서 순진한 $t$ 는 부풀어 있다 — 한센의 $S$ 가 하는 일.

자료: 로버트 실러의 『비이성적 과열』 자료집에서 날짜·실질 총수익지수·CAPE 를
뽑아 data/shiller_ie/ie_data_extract.csv 로 두었다 (원본은 ie_data.xls).

만드는 파일: docs/ch06/estimation_methods/img/nobel_2013_moments.png
"""

import csv

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

SRC = "data/shiller_ie/ie_data_extract.csv"

rows = list(csv.DictReader(open(SRC)))
DATE = np.array([float(r["date"]) for r in rows])
TR = np.array([float(r["real_tr_price"]) for r in rows])
CAPE = np.array([float(r["cape"]) if r["cape"] else np.nan for r in rows])
HAS = np.isfinite(CAPE)
print(f"자료: {SRC}")
print(f"  CAPE 가 있는 구간 {DATE[HAS].min():.0f}–{DATE[HAS].max():.0f}, "
      f"{HAS.sum():,} 개월")


def nw_cov(X, u, lag):
    """뉴이–웨스트. 겹치는 표본이 만드는 적률의 자기상관을 S 에 넣는다."""
    n = len(u)
    g = X * u[:, None]
    S = g.T @ g / n
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        A = g[l:].T @ g[:-l] / n
        S += w * (A + A.T)
    XXi = np.linalg.inv(X.T @ X / n)
    return XXi @ S @ XXi / n


def horizon_fit(months):
    """연율 로그 실질 총수익을 1/CAPE 에 회귀한다."""
    n = len(TR) - months
    i = np.arange(n)
    y = np.log(TR[i + months] / TR[i]) / (months / 12.0)
    x = 1.0 / CAPE[i]
    g = HAS[i] & np.isfinite(y) & np.isfinite(x)
    X = np.c_[np.ones(g.sum()), x[g]]
    yy = y[g]
    b = np.linalg.lstsq(X, yy, rcond=None)[0]
    u = yy - X @ b
    r2 = 1.0 - u.var() / yy.var()
    se_n = np.sqrt(np.sum(u ** 2) / (len(yy) - 2)
                   / np.sum((x[g] - x[g].mean()) ** 2))
    se_h = np.sqrt(nw_cov(X, u, max(months, 1))[1, 1])
    return dict(x=x[g], y=yy, t=DATE[i][g], b=b, r2=r2,
                t_naive=b[1] / se_n, t_hac=b[1] / se_h, n=g.sum())


HOR = [(1, "1 개월"), (3, "3 개월"), (12, "1 년"), (24, "2 년"), (36, "3 년"),
       (60, "5 년"), (84, "7 년"), (120, "10 년"), (180, "15 년"),
       (240, "20 년")]
print(f"\n{'지평':>8}{'n':>7}{'기울기':>9}{'R^2':>9}{'t 순진':>9}{'t HAC':>8}")
FIT = {}
for m, lab in HOR:
    f = horizon_fit(m)
    FIT[m] = f
    print(f"{lab:>8}{f['n']:>7,}{f['b'][1]:>9.3f}{f['r2']:>9.4f}"
          f"{f['t_naive']:>9.1f}{f['t_hac']:>8.1f}")
A, B = FIT[1], FIT[120]
print(f"\n  한 달 앞: R^2 = {A['r2']:.4f}, HAC t = {A['t_naive']:.1f} -> "
      f"{A['t_hac']:.1f}  (1.96 을 넘지 못한다)")
print(f"  십 년 앞: R^2 = {B['r2']:.4f}, HAC t = {B['t_naive']:.1f} -> "
      f"{B['t_hac']:.1f}")
print(f"  십 년 지평에서 순진한 t 는 HAC 의 {B['t_naive'] / B['t_hac']:.1f} 배다 "
      f"— 겹치는 표본을 독립으로 센 몫이다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_F, C_S, C_H = "#1f77b4", "#d62728", "#2ca02c"


def scatter_panel(ax, f, color, title, note, at=(0.035, 0.035),
                  ha="left", va="bottom"):
    sc = ax.scatter(f["x"], f["y"], c=f["t"], cmap="viridis", s=9,
                    alpha=0.65, linewidths=0)
    xs = np.linspace(f["x"].min(), f["x"].max(), 50)
    ax.plot(xs, f["b"][0] + f["b"][1] * xs, color=color, lw=3.0, zorder=5)
    ax.axhline(0, color="#999999", lw=1.0)
    ax.set_xlabel("$1/\\mathrm{CAPE}$ (그때의 수익수익률)")
    ax.set_title(title, fontsize=12)
    ax.annotate(note, xy=at, xycoords="axes fraction", fontsize=11,
                color=color, ha=ha, va=va)
    ax.grid(alpha=0.25)
    return sc


sc1 = scatter_panel(
    ax1, A, C_F, "① 파마 — 다음 한 달",
    f"$R^2 = {A['r2']:.4f}$\nHAC $t = {A['t_hac']:.1f}$  (유의하지 않다)")
ax1.set_ylabel("이후 실질 총수익 (연율 로그)")
ax1.set_ylim(-1.6, 1.6)

sc2 = scatter_panel(
    ax2, B, C_S, "② 실러 — 이후 십 년",
    f"$R^2 = {B['r2']:.4f}$\nHAC $t = {B['t_hac']:.1f}$",
    at=(0.035, 0.96), ha="left", va="top")
ax2.set_ylabel("이후 실질 총수익 (연율 로그)")
cb = fig.colorbar(sc2, ax=ax2, pad=0.02)
cb.set_label("관측한 해", fontsize=10)

ms = [m for m, _ in HOR]
yrs = np.array(ms) / 12.0
ax3.semilogx(yrs, [FIT[m]["t_naive"] for m in ms], "s--", color="#999999",
             lw=2.2, ms=8, label="순진한 $t$ (겹침을 무시)")
ax3.semilogx(yrs, [FIT[m]["t_hac"] for m in ms], "o-", color=C_H, lw=2.8,
             ms=9, label="뉴이–웨스트 $t$ (한센의 $S$)")
ax3.axhline(1.96, color="#d62728", lw=1.5, ls=":")
ax3.text(0.45, 0.6, "$1.96$", color="#d62728", fontsize=10.5)
ax3.set_xticks([1 / 12, 0.25, 1, 3, 10, 20])
ax3.set_xticklabels(["$1$ 개월", "$3$ 개월", "$1$ 년", "$3$ 년", "$10$ 년",
                     "$20$ 년"], fontsize=9.5)
ax3.minorticks_off()
ax3.set_xlabel("내다보는 지평")
ax3.set_ylabel("기울기의 $t$ 값")
ax3.set_ylim(0, 33)
ax3.set_title("③ 한센 — 적률의 공분산을 제대로 잡으면", fontsize=12)
ax3.legend(loc="upper left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch06/estimation_methods/img/nobel_2013_moments.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
