r"""13.6절 「그림으로 보는 통계학」 — 룽게가 $1901$ 년에 보인 것과, 그것의 회귀판.

쪽의 본문이 「특히 경계 근처에서 크게 흔들린다(Runge 현상)」고 괄호로만 적고
지나간다. 그 괄호를 펴 본다.

  ① 룽게의 보기 — $f(x) = 1/(1+25x^2)$ 를 $[-1, 1]$ 의 **같은 간격** 점들로
     보간하면 차수를 올릴수록 **끝에서** 오차가 커진다. 최대오차를 차수마다 잰다
  ② 그 일이 최소제곱에서도 일어난다 — 같은 $21$ 개 점에 $d$ 차 다항식을
     최소제곱으로 맞출 때, 적합값의 분산은 $\sigma^2 h(x)$ 이고 $h$ 는 끝에서
     가장 크다. 차수를 올리면 끝의 $h$ 가 빠르게 $1$ 에 다가간다 — $1$ 이면
     그 자리의 적합값이 **관측값 자신**이 되어 평활이 사라진다

위키백과가 적는 어림 $N < 2\sqrt{m}$ ($m$ 은 점의 수) 도 함께 표시한다.
$m = 21$ 이면 $2\sqrt{21} = 9.17$ 이다.

만드는 파일:

  ch13/interaction_polynomial/img/runge_edges.png

실행:  python3 scripts/make_ch13_runge_edges.py   (저장소 최상위에서)
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

OUT = "docs/ch13/interaction_polynomial/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

M = 21                      # ② 가 쓰는 점의 수
DEGS_INTERP = (5, 9, 15)
DEGS_LS = (1, 3, 5, 9, 15)


def runge(x):
    return 1.0 / (1.0 + 25.0 * x * x)


def interp(deg):
    """같은 간격 deg+1 개 점을 지나는 deg 차 다항식."""
    xs = np.linspace(-1.0, 1.0, deg + 1)
    return np.polynomial.Polynomial.fit(xs, runge(xs), deg), xs


def leverage(x, deg):
    """최소제곱 다항적합의 h(x) = 대각(H). QR 로 안정하게 구한다."""
    V = np.polynomial.legendre.legvander(x, deg)   # 직교기저라 조건수가 낫다
    Q, _ = np.linalg.qr(V)
    return np.einsum("ij,ij->i", Q, Q)


def main():
    grid = np.linspace(-1.0, 1.0, 2001)

    print("① 같은 간격 보간의 최대오차")
    for d in (5, 9, 15, 21):
        p, _ = interp(d)
        err = np.abs(p(grid) - runge(grid))
        print(f"   차수 {d:2d}   최대오차 {err.max():10.4f}   "
              f"그 자리 x = {grid[err.argmax()]:+.4f}")

    xs = np.linspace(-1.0, 1.0, M)
    print(f"\n② 점 {M} 개에 최소제곱으로 맞출 때의 지렛대")
    print("   차수   h(가운데)   h(끝)    끝/가운데")
    for d in DEGS_LS:
        h = leverage(xs, d)
        print(f"   {d:3d}    {h[M // 2]:8.4f}  {h[-1]:7.4f}   "
              f"{h[-1] / h[M // 2]:8.2f}")
    print(f"   위키백과가 적는 어림 N < 2 sqrt(m) = {2 * np.sqrt(M):.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 룽게의 보기 ──
    ax = axes[0]
    ax.plot(grid, runge(grid), color=INK, lw=3.0, label="$f(x)=1/(1+25x^2)$")
    for d, col in zip(DEGS_INTERP, (BLUE, TEAL, RED)):
        p, nodes = interp(d)
        err = np.abs(p(grid) - runge(grid)).max()
        ax.plot(grid, p(grid), color=col, lw=1.9, ls="--",
                label=f"차수 {d} — 최대오차 {err:.2f}")
        ax.plot(nodes, runge(nodes), "o", ms=4.5, color=col, alpha=0.7)
    ax.axhline(0, color=MUTED, lw=0.9, ls=":")
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.3, 2.0)
    ax.set_xlabel("$x$ — 점은 같은 간격으로 찍는다", fontsize=11.5)
    ax.set_title("① 차수를 올릴수록 끝이 터진다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 최소제곱에서도 끝이 비싸다 ──
    ax = axes[1]
    for d, col in zip(DEGS_LS, (MUTED, BLUE, TEAL, AMBER, RED)):
        h = leverage(xs, d)
        ax.plot(xs, h, "o-", ms=5, lw=1.9, color=col,
                label=f"차수 {d} — 끝에서 $h = {h[-1]:.2f}$")
    ax.axhline(1.0, color=INK, lw=1.4, ls="--")
    ax.text(0.0, 0.845, "$h = 1$ — 적합값이 관측값 자신이 된다", fontsize=10.6,
            color=INK, ha="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, alpha=0.92))
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(0.0, 1.18)
    ax.set_xlabel(f"$x$ — 같은 간격의 점 ${M}$ 개", fontsize=11.5)
    ax.set_ylabel("지렛대 $h(x)$   ($\\operatorname{Var}\\hat y = \\sigma^2 h$)",
                  fontsize=11.5)
    ax.set_title("② 최소제곱으로 맞춰도 끝이 비싸다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.4, framealpha=0.93, loc="upper center", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("끝에서 일어나는 일은 보간만의 사정이 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽은 룽게가 $1901$ 년에 든 함수와 같은 간격 보간이다. "
             "오른쪽의 $h$ 는 르장드르 기저로 QR 분해해 직접 계산했다. "
             "두 판의 수는 모두 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "runge_edges.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
