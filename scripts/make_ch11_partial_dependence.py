r"""11.9절 「AI로 보는 통계학」 — 부분의존 곡선은 자료가 정하지 않는다.

$1$ 절의 **부분의존**은 가법모형의 항 $f_j(x_j)$ 였다. 오늘 블랙박스 모형을 읽을 때
쓰는 **부분의존 그림**은 그 정의를 가법성이 없는 모형에까지 밀어붙인 것이다.

$$
\mathrm{PD}_j(v) \;=\; \frac1n \sum_{i=1}^{n} \hat f\bigl(v,\; \mathbf x_{i,-j}\bigr)
$$

곧 **모든 사람의 나머지 변수는 그대로 두고 그 변수만 $v$ 로 바꿔 평균**한다.

이 스크립트가 재는 값:

  ① 설명변수가 서로 독립인 가법모형에서
       PD_1 과 f_1 의 최대 차이 = 2.7e-15    (상수항을 맞춘 뒤)
       PD_2 과 f_2 의 최대 차이 = 3.1e-15
     곧 가법모형에서 부분의존은 그 항 자체다.

  ② 상관 0.98 인 두 변수에서 — 참 모형 y = x1 + x2, 관측 100 개를 40 번 새로 뽑아
     그때마다 적합하고 x2 의 부분의존 곡선을 그렸다.
       개별 기울기 b2         범위 +0.23 ~ +2.02,  SD 0.4335
       둘을 함께 움직인 b1+b2  범위 +1.79 ~ +2.15,  SD 0.0816
       비 = 5.31,  식이 말하는 1/sqrt(1-rho^2) = 5.03
       참값에 견준 흩어짐 — 개별 43%, 합 4%
     곧 **예측은 안정적인데 「그 변수 하나의 효과」는 자료가 정해 주지 않는다.**

만드는 파일:

  ch11/splines_gams/img/partial_dependence.png

실행:  python3 scripts/make_ch13_partial_dependence.py   (저장소 최상위에서)
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

OUT = "docs/ch11/splines_gams/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 700            # ① 가법모형에 쓰는 관측 수
N2 = 100           # ② 다시 뽑는 자료의 관측 수
REPS = 40          # ② 를 몇 번 다시 뽑는가
RHO = 0.98
SEED = 9


def pd_curve(predict, X, j, grid):
    """부분의존 — 그 열만 v 로 바꿔 모든 행에 대해 평균한다."""
    out = np.empty(len(grid))
    for k, v in enumerate(grid):
        Z = X.copy()
        Z[:, j] = v
        out[k] = predict(Z).mean()
    return out


def main():
    rng = np.random.default_rng(SEED)

    # ── ① 독립인 설명변수, 가법모형 ──
    X = rng.uniform(-3, 3, (N, 2))          # 두 설명변수는 서로 독립이다
    f1 = lambda v: 2.0 * np.sin(1.2 * v)
    f2 = lambda v: 0.4 * v ** 2 - 1.2
    add = lambda Z: 10 + f1(Z[:, 0]) + f2(Z[:, 1])     # 참 가법모형

    g = np.linspace(-3, 3, 120)
    pd1 = pd_curve(add, X, 0, g)
    pd2 = pd_curve(add, X, 1, g)
    d1 = np.abs((pd1 - pd1.mean()) - (f1(g) - f1(g).mean())).max()
    d2 = np.abs((pd2 - pd2.mean()) - (f2(g) - f2(g).mean())).max()
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print("① 설명변수가 독립인 가법모형")
    print(f"  PD_1 과 f_1 의 최대 차이 = {d1:.1e}   (상수항을 맞춘 뒤)")
    print(f"  PD_2 과 f_2 의 최대 차이 = {d2:.1e}")

    # ── ② 상관 0.98 인 두 변수 ──
    print(f"\n② 상관 {RHO} 인 두 변수 — 참 모형 y = x1 + x2, 관측 {N2} 개")
    gc = np.linspace(-2.6, 2.6, 120)
    only2, both = [], []
    for r in range(REPS):
        rr = np.random.default_rng([SEED, r])
        z1 = rr.normal(size=N2)
        z2 = RHO * z1 + np.sqrt(1 - RHO ** 2) * rr.normal(size=N2)
        y = z1 + z2 + rr.normal(0, 1.0, N2)
        M = np.column_stack([np.ones(N2), z1, z2])
        b = np.linalg.lstsq(M, y, rcond=None)[0]
        only2.append(b[2])
        both.append(b[1] + b[2])
    only2, both = np.array(only2), np.array(both)
    print(f"  개별 기울기의 흩어짐   SD(b2)    = {only2.std():.4f}")
    print(f"  둘을 함께 움직인 기울기 SD(b1+b2) = {both.std():.4f}")
    print(f"  비 = {only2.std() / both.std():.2f},  "
          f"식이 말하는 1/sqrt(1-rho^2) = {1 / np.sqrt(1 - RHO ** 2):.2f}")
    print(f"  b2 의 범위 {only2.min():+.2f} ~ {only2.max():+.2f}   "
          f"b1+b2 의 범위 {both.min():+.2f} ~ {both.max():+.2f}")
    print(f"  참값에 견준 흩어짐 — 개별 {100 * only2.std() / 1.0:.0f}%  "
          f"합 {100 * both.std() / 2.0:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.9), constrained_layout=True)

    # ── ① ──
    ax = axes[0]
    ax.plot(g, f1(g) - f1(g).mean(), lw=5.0, color=MUTED, alpha=0.75,
            label="참 항 $f_1$, $f_2$")
    ax.plot(g, f2(g) - f2(g).mean(), lw=5.0, color=MUTED, alpha=0.75)
    ax.plot(g[::4], pd1[::4] - pd1.mean(), "o", ms=6, color=TEAL,
            label="부분의존 $\\mathrm{PD}_1$")
    ax.plot(g[::4], pd2[::4] - pd2.mean(), "s", ms=6, color=AMBER,
            label="부분의존 $\\mathrm{PD}_2$")
    # mathtext 는 1e-15 의 e 를 기울여 찍으므로 10 의 거듭제곱으로 적는다
    def sci(v):
        e = int(np.floor(np.log10(v)))
        return f"{v / 10 ** e:.1f} \\times 10^{{{e}}}"
    ax.text(-2.9, 2.45, f"최대 어긋남 ${sci(d1)}$ 과 ${sci(d2)}$",
            fontsize=11, color=INK)
    ax.set_xlabel("$x_j$", fontsize=11.5)
    ax.set_ylabel("중심을 맞춘 값", fontsize=11.5)
    ax.set_ylim(-2.6, 3.0)
    ax.set_title("① 가법모형에서는 부분의존이 곧 그 항이다", fontsize=12.6,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② ──
    ax = axes[1]
    for k, s in enumerate(only2):
        ax.plot(gc, s * gc, lw=1.6, color=RED, alpha=0.38,
                label="한 번 적합할 때마다의 $x_2$ 부분의존" if k == 0 else None)
    for k, s in enumerate(both):
        ax.plot(gc, s * gc, lw=1.6, color=TEAL, alpha=0.38,
                label="둘을 함께 움직였을 때" if k == 0 else None)
    ax.plot(gc, 1.0 * gc, lw=3.4, color="white", zorder=5)
    ax.plot(gc, 1.0 * gc, lw=2.4, color=INK, ls="--", zorder=6,
            label="참값 — $x_2$ 의 기울기 $1$")
    ax.text(-2.5, 5.2, f"$x_2$ 쪽 기울기 ${only2.min():+.2f} \\sim "
            f"{only2.max():+.2f}$  (SD ${only2.std():.2f}$)",
            fontsize=10.8, color=RED, fontweight="bold")
    ax.text(-2.5, 4.3, f"둘을 함께 ${both.min():+.2f} \\sim {both.max():+.2f}$  "
            f"(SD ${both.std():.2f}$)",
            fontsize=10.8, color=TEAL, fontweight="bold")
    ax.set_xlabel("$x_2$", fontsize=11.5)
    ax.set_ylabel("중심을 맞춘 부분의존", fontsize=11.5)
    ax.set_ylim(-6.4, 6.4)
    ax.set_title(f"② 상관 ${RHO}$ — 자료를 다시 뽑을 때마다 곡선이 춤춘다",
                 fontsize=12.6, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("「이 변수의 효과」라고 읽기 전에", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"① 은 관측 ${N}$ 개, ② 는 관측 ${N2}$ 개짜리 자료를 ${REPS}$ 번 "
             f"새로 뽑아 그때마다 적합한 것이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "partial_dependence.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
