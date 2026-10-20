r"""「얼마나 더 모아야 하는가」 — 같은 물음, 다른 지수.

고전적 표본크기 공식 $n = (z\sigma/E)^2$ 는 오차가 $n^{-1/2}$ 로 준다는 말이다.
자료를 네 배 모으면 오차 한계가 절반이 된다. 이 수 $1/2$ 는 정리에서 나온다.

언어모형의 손실은 그렇게 줄지 않는다. 카플란 등(2020)이 잰 지수는 $0.095$ 다.
절반으로 줄이려면 자료를 $2^{1/0.095} \approx 1{,}450$ 배 모아야 한다.

왜 이렇게 다른가. $n^{-1/2}$ 는 **차원이 고정된** 모수를 추정할 때의 속도다.
차원이 올라가면 비모수 추정의 속도가 느려진다. 최근접이웃 회귀로 직접 재면 지수가
$d = 1$ 에서 $0.48$, $d = 16$ 에서 $0.077$ 까지 내려가고, 카플란의 $0.095$ 는 그
사이에 있다. (여기 쓴 표적은 $\sin$ 이라 리프시츠보다 매끄럽고, 그래서 실측 지수는
리프시츠 최악의 경우인 $1/(2+d)$ 보다 크다.)

① 두 법칙을 같은 로그-로그 축에.
② 오차를 절반으로 줄이려면 자료가 몇 배 드는가.
③ 차원이 오르면 지수가 어떻게 무너지는가 — 최근접이웃 회귀로 실측.

만드는 파일: docs/ch08/sample_size/img/scaling_exponent.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from sklearn.neighbors import KNeighborsRegressor  # noqa: E402

RNG = np.random.default_rng(20261020)
ALPHA_D = 0.095          # 카플란 등(2020) 식 (1.2)
DIMS = (1, 2, 4, 8, 16)
NS = np.array([500, 1500, 4500, 13500, 40500])
REPS = 6
SIGMA = 0.25


def target(x):
    """진짜로 d 차원인 매끄러운 함수 (리프시츠)."""
    d = x.shape[1]
    return np.sin(2 * np.pi * x).sum(axis=1) / np.sqrt(d)


def excess_rmse(d, n, rng):
    """최근접이웃 회귀의 잡음을 뺀 오차. k 는 이론이 주는 n^{2/(2+d)} 로 둔다."""
    xtr = rng.random((n, d))
    ytr = target(xtr) + SIGMA * rng.standard_normal(n)
    k = max(1, int(round(n ** (2 / (2 + d)))))
    f = KNeighborsRegressor(n_neighbors=k).fit(xtr, ytr)
    xte = rng.random((4000, d))
    err = f.predict(xte) - target(xte)
    return np.sqrt(np.mean(err**2))


# === ③ 실측 =======================================================
slopes, curves = {}, {}
for d in DIMS:
    vals = []
    for n in NS:
        vals.append(np.mean([excess_rmse(d, int(n), RNG) for _ in range(REPS)]))
    vals = np.array(vals)
    curves[d] = vals
    slopes[d] = np.polyfit(np.log(NS), np.log(vals), 1)[0]
    print(f"  d = {d}: 실측 기울기 {slopes[d]:+.3f}  (이론 {-1 / (2 + d):+.3f})")

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

# ---- ① 두 법칙 ----------------------------------------------------
r = np.logspace(0, 6, 200)          # 자료를 몇 배로 늘렸나
ax1.loglog(r, r ** (-0.5), color="#1f77b4", lw=2.8,
           label="고전: 오차 한계 $\\propto n^{-1/2}$")
ax1.loglog(r, r ** (-ALPHA_D), color="#d62728", lw=2.8,
           label=f"언어모형: 손실 $\\propto D^{{-{ALPHA_D}}}$")
ax1.axhline(0.5, color="black", lw=1.4, ls="--")
ax1.text(1.3, 0.54, "절반으로", fontsize=10)
for e, c in ((0.5, "#1f77b4"), (ALPHA_D, "#d62728")):
    x = 2 ** (1 / e)
    ax1.plot([x], [0.5], "o", color=c, ms=8, zorder=5)
    ax1.vlines(x, 1e-3, 0.5, color=c, lw=1.1, ls=":")
    ax1.text(x * 1.4, 0.30, f"${x:,.0f}$ 배".replace(",", "{,}"), ha="left",
             fontsize=10.5, color=c)
ax1.set_xlim(1, 1e6)
ax1.set_ylim(1e-3, 1.5)
ax1.set_xlabel("자료를 몇 배로 늘렸는가")
ax1.set_ylabel("처음 대비 남은 오차")
ax1.set_title("① 같은 로그-로그 축에 놓으면", fontsize=12)
ax1.set_xticks([1, 10, 100, 1e3, 1e4, 1e5, 1e6])
ax1.set_xticklabels(["$1$", "$10$", "$10^2$", "$10^3$", "$10^4$", "$10^5$",
                     "$10^6$"])
ax1.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax1.set_yticklabels(["$0.001$", "$0.01$", "$0.1$", "$1$"])
ax1.minorticks_off()
ax1.legend(loc="lower left", fontsize=10)
ax1.grid(alpha=0.25, which="major")

# ---- ② 두 배 늘리면 ------------------------------------------------
labs = ["고전\n$n^{-1/2}$", "언어모형\n$D^{-0.095}$"]
exps = [0.5, ALPHA_D]
cols = ["#1f77b4", "#d62728"]
gain = [(1 - 2 ** (-e)) * 100 for e in exps]
ax2.bar(np.arange(2), gain, 0.5, color=cols)
for i, g in enumerate(gain):
    ax2.text(i, g + 0.8, f"${g:.1f}\\%$", ha="center", fontsize=12)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(labs, fontsize=11)
ax2.set_ylabel("자료를 두 배로 늘릴 때 줄어드는 오차 ($\\%$)")
ax2.set_title("② 자료를 두 배 모으면", fontsize=12)
ax2.set_ylim(0, 36)
ax2.grid(alpha=0.25, axis="y")
ax2.text(0.5, 24,
         "절반으로 줄이려면\n"
         f"고전 ${2 ** (1 / 0.5):.0f}$ 배 · "
         f"언어모형 ${2 ** (1 / ALPHA_D):,.0f}$ 배".replace(",", "{,}"),
         ha="center", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#cccccc"))

# ---- ③ 차원이 오르면 -----------------------------------------------
for d, c in zip(DIMS, ("#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd")):
    v = curves[d] / curves[d][0]
    ax3.loglog(NS / NS[0], v, "o-", color=c, lw=2.2, ms=5,
               label=f"$d = {d}$  (기울기 {slopes[d]:.3f})")
ax3.loglog(NS / NS[0], (NS / NS[0]) ** (-0.5), color="black", lw=1.4, ls="--",
           label="$n^{-1/2}$")
ax3.loglog(NS / NS[0], (NS / NS[0]) ** (-ALPHA_D), color="#999999", lw=1.4,
           ls=":", label=f"$n^{{-{ALPHA_D}}}$ (카플란)")
ax3.set_xlabel("표본을 몇 배로 늘렸는가")
ax3.set_ylabel("처음 대비 남은 오차")
ax3.set_title("③ 차원이 오르면 지수가 무너진다", fontsize=12)
ax3.set_xticks([1, 3, 9, 27, 81])
ax3.set_xticklabels(["$1$", "$3$", "$9$", "$27$", "$81$"])
ax3.set_yticks([0.1, 0.2, 0.5, 1.0])
ax3.set_yticklabels(["$0.1$", "$0.2$", "$0.5$", "$1$"])
ax3.set_ylim(0.08, 1.25)
ax3.minorticks_off()
ax3.legend(loc="lower left", fontsize=9)
ax3.grid(alpha=0.25, which="major")

fig.tight_layout()
out = "docs/ch08/sample_size/img/scaling_exponent.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① 자료를 두 배로 늘릴 때")
print(f"   고전 (지수 0.5)   : 오차가 {2 ** -0.5:.4f} 배 "
      f"({(1 - 2 ** -0.5) * 100:.1f}% 감소)")
print(f"   카플란 (지수 {ALPHA_D}): 손실이 {2 ** -ALPHA_D:.4f} 배 "
      f"({(1 - 2 ** -ALPHA_D) * 100:.1f}% 감소)")
print(f"\n   절반으로 줄이는 데 드는 배수: 고전 {2 ** (1 / 0.5):.0f} 배, "
      f"카플란 {2 ** (1 / ALPHA_D):,.0f} 배")
print(f"   (모수 쪽 지수 0.076 이면 {2 ** (1 / 0.076):,.0f} 배)")

print(f"\n③ 최근접이웃 회귀 — 잡음을 뺀 오차의 기울기")
print(f"{'d':>4}{'실측':>10}{'이론 -1/(2+d)':>16}")
for d in DIMS:
    print(f"{d:>4}{slopes[d]:>10.3f}{-1 / (2 + d):>16.3f}")
