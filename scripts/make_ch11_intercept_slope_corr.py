r"""11.2절 「그림으로 보는 통계학」 — 절편과 기울기는 같이 흔들린다.

단순회귀의 두 추정량은 따로 노는 수가 아니다. 공분산이 닫힌 꼴로 나온다.

$$
\operatorname{Cov}(\hat\beta_0, \hat\beta_1) = -\,\frac{\bar x\,\sigma^2}{S_{xx}},
\qquad
\operatorname{Corr}(\hat\beta_0, \hat\beta_1) = \frac{-\bar x}{\sqrt{S_{xx}/n + \bar x^2}}
$$

$\bar x = 0$ 이면 $0$ 이고, $x$ 의 평균이 퍼짐에 비해 커질수록 $-1$ 로 간다.
$x$ 를 통째로 옮기는 것만으로 두 추정량의 관계가 바뀐다는 뜻이다. 기울기의
표준오차는 그동안 한 치도 움직이지 않는다.

이 스크립트가 재는 값($n = 25$, $\sigma = 2$, 되풀이 $20{,}000$ 번):

  x 평균   모의 상관    식      SD(b0)   SD(b1)
     0      +0.010    0.000     0.398    0.133
     2      -0.554   -0.554     0.481    0.134
     5      -0.856   -0.857     0.776    0.133
    50      -0.998   -0.998     6.729    0.134

만드는 파일:

  ch11/estimation_inference/img/intercept_slope_corr.png

실행:  python3 scripts/make_ch13_intercept_slope_corr.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 25
SIGMA = 2.0
B0, B1 = 2.0, 1.5
REPS = 20_000
SHIFTS = (0, 1, 2, 3, 5, 8, 12, 20, 50)
SEED = 5


def simulate(shift, rng, reps=REPS):
    """x 의 평균을 shift 에 두고 OLS 를 reps 번 되풀이한다."""
    x = np.linspace(0, 10, N) - 5 + shift
    X = np.column_stack([np.ones(N), x])
    XtXi = np.linalg.inv(X.T @ X)
    mu = B0 + B1 * x
    eps = rng.normal(0, SIGMA, (reps, N))
    Y = mu + eps
    b = (XtXi @ X.T @ Y.T).T              # (reps, 2)
    xb = x.mean()
    sxx = ((x - xb) ** 2).sum()
    theo = -xb / np.sqrt(sxx / N + xb ** 2)
    return b, float(np.corrcoef(b.T)[0, 1]), float(theo)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("x 평균   모의 상관       식     SD(b0)   SD(b1)")
    store = {}
    for s in SHIFTS:
        b, emp, theo = simulate(s, rng)
        store[s] = b
        rows.append((s, emp, theo, b[:, 0].std(), b[:, 1].std()))
        print(f"{s:6d} {emp:+11.4f} {theo:+9.4f} {b[:, 0].std():9.3f} "
              f"{b[:, 1].std():8.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    # ── ① 두 구름 ──
    ax = axes[0]
    for s, col, lab in ((0, TEAL, "$\\bar x = 0$ — 중심화했다"),
                        (5, RED, "$\\bar x = 5$ — 옮기지 않았다")):
        b = store[s]
        emp = [r[1] for r in rows if r[0] == s][0]
        ax.plot(b[::6, 0], b[::6, 1], ".", ms=2.6, color=col, alpha=0.45,
                label=f"{lab}   상관 ${emp:+.3f}$")
    ax.plot([B0], [B1], "*", ms=18, color=INK, zorder=5)
    ax.annotate("참값 $(\\beta_0, \\beta_1) = (2,\\ 1.5)$", xy=(B0, B1),
                xytext=(B0 + 0.55, B1 + 0.33), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel("$\\hat\\beta_0$ (절편)", fontsize=11.5)
    ax.set_ylabel("$\\hat\\beta_1$ (기울기)", fontsize=11.5)
    ax.set_title("① 같은 자료, 같은 기울기, 다른 구름", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    leg = ax.legend(fontsize=10, framealpha=0.95, loc="upper right",
                    markerscale=5)
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 상관과 두 표준오차 ──
    ax = axes[1]
    xs = np.array([r[0] for r in rows], dtype=float)
    grid = np.linspace(0, 52, 400)
    sxx = ((np.linspace(0, 10, N) - 5) ** 2).sum()
    curve = -grid / np.sqrt(sxx / N + grid ** 2)
    ax.plot(grid, curve, lw=2.4, color=INK, label="식이 말하는 상관")
    ax.plot(xs, [r[1] for r in rows], "o", ms=9, color=RED, mec="white",
            mew=1.2, label="모의실험으로 센 상관")
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlabel("$x$ 의 평균 $\\bar x$  (퍼짐은 그대로다)", fontsize=11.5)
    ax.set_ylabel("$\\operatorname{Corr}(\\hat\\beta_0, \\hat\\beta_1)$",
                  fontsize=11.5)
    ax.set_ylim(-1.08, 0.15)
    ax.set_title("② 중심을 옮기면 상관이 $-1$ 로 간다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(xs, [r[3] for r in rows], "s--", ms=6, lw=1.8, color=AMBER,
             label="$\\mathrm{SD}(\\hat\\beta_0)$")
    ax2.plot(xs, [r[4] for r in rows], "D--", ms=6, lw=1.8, color=TEAL,
             label="$\\mathrm{SD}(\\hat\\beta_1)$")
    ax2.set_ylabel("표준오차", fontsize=11.5)
    ax2.set_ylim(0, 7.4)
    ax2.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax2.spines[["top"]].set_visible(False)
    ax2.text(30, 0.62, "기울기 쪽은 꿈쩍도 하지 않는다", ha="center",
             fontsize=10.8, color=TEAL)

    fig.suptitle("절편과 기울기는 따로 흔들리지 않는다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"$n = {N}$, $\\sigma = {SIGMA:g}$, 참값 $(2,\\ 1.5)$ 로 "
             f"${REPS // 1000}$ 천 번씩 되풀이했다. $x$ 의 퍼짐은 모든 경우에 같고 "
             f"평균만 옮겼다. 모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "intercept_slope_corr.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
