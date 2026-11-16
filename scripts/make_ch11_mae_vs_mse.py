r"""13.7절 「그림으로 보는 통계학」 — 척도를 고르는 일이 답을 고르는 일이다.

MAE 와 RMSE 는 「같은 것을 다른 단위로 잰 것」이 아니다. **다른 것을
맞히라고 시키는 명령**이다.

  상수 c 로 y 를 맞힌다고 할 때
    E|y - c| 를 최소로 하는 c  =  중앙값
    E(y - c)^2 를 최소로 하는 c =  평균

회귀로 올리면 그대로 이어진다. 제곱손실로 적합하면 조건부 **평균**의
직선이, 절댓값손실로 적합하면 조건부 **중앙값**의 직선이 나온다. 오차가
치우쳐 있으면 두 직선이 눈에 띄게 어긋난다.

오차를 Exp(1) - 1 로 두었다(평균 0, 중앙값 ln2 - 1 = -0.3069).

만드는 파일:

  ch13/performance/img/mae_vs_mse.png

실행:  python3 scripts/make_ch13_mae_vs_mse.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/performance/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N, SEED = 300, 13
A0, B0 = 1.0, 2.0
MED = np.log(2) - 1.0          # 중심화한 지수분포의 중앙값


def fit_lad(x, y):
    """절댓값손실로 적합 — 조건부 중앙값의 직선."""
    f = lambda p: np.abs(y - (p[0] + p[1] * x)).sum()
    start = np.polyfit(x, y, 1)[::-1]
    return optimize.minimize(f, start, method="Nelder-Mead",
                             options=dict(xatol=1e-10, fatol=1e-10,
                                          maxiter=20000)).x


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-2, 2, N)
    eps = rng.exponential(1.0, N) - 1.0            # 평균 0, 오른쪽으로 긴 꼬리
    y = A0 + B0 * x + eps

    b_ols = np.polyfit(x, y, 1)[::-1]
    b_lad = fit_lad(x, y)
    print(f"    참 평균 직선    절편 {A0:.4f}  기울기 {B0:.4f}")
    print(f"    참 중앙값 직선   절편 {A0+MED:.4f}  기울기 {B0:.4f}")
    print(f"    제곱손실 적합    절편 {b_ols[0]:+.4f}  기울기 {b_ols[1]:+.4f}")
    print(f"    절댓값손실 적합  절편 {b_lad[0]:+.4f}  기울기 {b_lad[1]:+.4f}")

    # 새 자료에서 두 적합을 두 척도로 채점한다
    xt = rng.uniform(-2, 2, 200_000)
    yt = A0 + B0 * xt + (rng.exponential(1.0, 200_000) - 1.0)
    rows = []
    for nm, b in (("제곱손실 적합", b_ols), ("절댓값손실 적합", b_lad)):
        r = yt - (b[0] + b[1] * xt)
        rows.append((nm, float(np.abs(r).mean()), float(np.sqrt((r ** 2).mean()))))
        print(f"    {nm:14} MAE {rows[-1][1]:.4f}   RMSE {rows[-1][2]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6), constrained_layout=True)

    # ── (1) 상수 하나로 맞힐 때의 두 손실 ──
    ax = axes[0]
    e = rng.exponential(1.0, 400_000) - 1.0
    cs = np.linspace(-1.0, 1.4, 400)
    mae = np.array([np.abs(e - c).mean() for c in cs])
    mse = np.array([((e - c) ** 2).mean() for c in cs])
    ax.plot(cs, mae / mae.min(), lw=2.8, color=TEAL, label="절댓값손실")
    ax.plot(cs, mse / mse.min(), lw=2.8, color=AMBER, label="제곱손실")
    ax.axvline(MED, color=TEAL, lw=1.6, ls=":")
    ax.axvline(0.0, color=AMBER, lw=1.6, ls=":")
    ax.text(MED - 0.05, 1.42, f"중앙값\n{MED:.3f}", fontsize=10.2, color=TEAL,
            ha="right")
    ax.text(0.06, 1.42, "평균\n0.000", fontsize=10.2, color=AMBER)
    ax.set_xlabel("맞히려는 상수 $c$", fontsize=11.5)
    ax.set_ylabel("손실 (최솟값을 1 로)", fontsize=11.5)
    ax.set_title("최소가 되는 자리가 다르다", fontsize=12.6, color=INK, pad=8)
    ax.set_ylim(0.97, 1.6)
    ax.legend(fontsize=10.4, framealpha=0.93)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── (2) 회귀로 올리면 ──
    ax = axes[1]
    ax.scatter(x, y, s=16, color=MUTED, alpha=0.65, edgecolors="none")
    g = np.linspace(-2, 2, 50)
    ax.plot(g, b_ols[0] + b_ols[1] * g, lw=3.0, color=AMBER,
            label="제곱손실 — 조건부 평균")
    ax.plot(g, b_lad[0] + b_lad[1] * g, lw=3.0, color=TEAL,
            label="절댓값손실 — 조건부 중앙값")
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title(f"두 직선이 {abs(b_ols[0]-b_lad[0]):.2f} 만큼 떨어져 있다",
                 fontsize=12.6, color=INK, pad=8)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── (3) 새 자료에서 채점 ──
    ax = axes[2]
    w = 0.34
    idx = np.arange(2)
    ax.bar(idx - w / 2, [rows[0][1], rows[1][1]], width=w, color=TEAL,
           alpha=0.85, label="MAE 로 채점")
    ax.bar(idx + w / 2, [rows[0][2], rows[1][2]], width=w, color=AMBER,
           alpha=0.85, label="RMSE 로 채점")
    for i in range(2):
        ax.text(i - w / 2, rows[i][1] + 0.012, f"{rows[i][1]:.4f}", ha="center",
                fontsize=10, color=INK)
        ax.text(i + w / 2, rows[i][2] + 0.012, f"{rows[i][2]:.4f}", ha="center",
                fontsize=10, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(["제곱손실로\n적합한 직선", "절댓값손실로\n적합한 직선"],
                       fontsize=10.6)
    ax.set_ylabel("새 자료에서의 점수", fontsize=11.5)
    ax.set_title("각자 제 척도에서 이긴다", fontsize=12.6, color=INK, pad=8)
    ax.set_ylim(0.6, 1.15)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper center")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("MAE 와 RMSE 는 같은 것을 다른 단위로 잰 것이 아니다", fontsize=14.5)
    fig.text(0.5, -0.055,
             "오차를 $\\mathrm{Exp}(1) - 1$ 로 두어 오른쪽으로 치우치게 했다. "
             "평균은 $0$, 중앙값은 $\\ln 2 - 1 = -0.307$ 이고 그 차이가 두 직선의 "
             "간격으로 그대로 나타난다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "mae_vs_mse.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
