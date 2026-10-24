r"""11.1절 「그림으로 보는 통계학」 — 「다른 것을 고정하고」를 수로 하면.

단순회귀의 기울기와 다중회귀의 같은 계수는 다른 양이다. 뒤엣것은
**다른 설명변수를 고정했을 때**의 기울기이고, 둘은 부호까지 갈릴 수 있다.

그림은 그 자리를 보인다.

  왼쪽   y 를 x1 에만 회귀하면 기울기가 +0.30 이다.
  오른쪽 x2 를 양쪽에서 걷어 낸 잔차끼리 그리면 기울기가 -0.50 이다.
         이것이 다중회귀가 내놓는 x1 의 계수와 **정확히** 같다
         (프리슈-워-로벨 정리).

자료는 x1 = 0.8 x2 + 잡음, y = 1.0 x2 - 0.5 x1 + 잡음 으로 만들었다.
x1 이 x2 를 대신 말해 주고 있어서, x2 를 안 넣으면 x1 의 계수가 그
몫까지 받아 부호가 뒤집힌다.

만드는 파일:

  ch11/linear_regression/img/partial_slope.png

실행:  python3 scripts/make_ch13_partial_slope.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/linear_regression/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N = 120
SEED = 1899           # 율의 논문이 나온 해


def slope(a, b):
    """b 를 a 에 단순회귀했을 때의 기울기 (둘 다 중심화되어 있다고 본다)."""
    return float((a - a.mean()) @ (b - b.mean()) / ((a - a.mean()) ** 2).sum())


def main():
    rng = np.random.default_rng(SEED)
    x2 = rng.normal(0, 1, N)                      # 숨은 쪽 — 이를테면 노령 인구 비율
    x1 = 0.8 * x2 + rng.normal(0, 0.6, N)         # 관심 변수 — x2 와 얽혀 있다
    y = 1.0 * x2 - 0.5 * x1 + rng.normal(0, 0.5, N)

    b_simple = slope(x1, y)
    A = np.column_stack([np.ones(N), x1, x2])
    b_mult = np.linalg.lstsq(A, y, rcond=None)[0]

    # 양쪽에서 x2 를 걷어 낸 잔차
    ry = y - (y.mean() + slope(x2, y) * (x2 - x2.mean()))
    rx = x1 - (x1.mean() + slope(x2, x1) * (x2 - x2.mean()))
    b_fwl = slope(rx, ry)

    print(f"    단순회귀  x1 의 기울기        {b_simple:+.4f}")
    print(f"    다중회귀  x1 의 계수          {b_mult[1]:+.4f}   (x2 의 계수 {b_mult[2]:+.4f})")
    print(f"    잔차끼리 그린 기울기          {b_fwl:+.4f}")
    print(f"    다중회귀와의 차              {abs(b_fwl - b_mult[1]):.2e}")
    print(f"    참값은 -0.5 이고 x1 과 x2 의 상관은 {np.corrcoef(x1, x2)[0,1]:.3f} 이다")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    ax = axes[0]
    ax.scatter(x1, y, s=26, color=MUTED, alpha=0.75, edgecolors="none")
    g = np.linspace(x1.min(), x1.max(), 50)
    ax.plot(g, y.mean() + b_simple * (g - x1.mean()), lw=3.0, color=AMBER)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title(f"$x_1$ 만 넣으면 기울기 {b_simple:+.2f}",
                 fontsize=13, color=AMBER, pad=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.scatter(rx, ry, s=26, color=TEAL, alpha=0.75, edgecolors="none")
    g = np.linspace(rx.min(), rx.max(), 50)
    ax.plot(g, b_fwl * g, lw=3.0, color=TEAL)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("$x_1$ 에서 $x_2$ 를 걷어 낸 나머지", fontsize=11.5)
    ax.set_ylabel("$y$ 에서 $x_2$ 를 걷어 낸 나머지", fontsize=11.5)
    ax.set_title(f"$x_2$ 를 고정하면 기울기 {b_fwl:+.2f}",
                 fontsize=13, color=TEAL, pad=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.05, f"다중회귀가 내놓는 $x_1$ 의 계수 {b_mult[1]:+.4f}\n"
                        f"이 그림의 기울기 {b_fwl:+.4f} — 같은 수다",
            transform=ax.transAxes, fontsize=10.4, color=INK)

    fig.suptitle("같은 자료, 같은 변수인데 부호가 뒤집힌다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "$x_1$ 이 $x_2$ 와 얽혀 있어서(상관 $0.81$), $x_2$ 를 빼놓고 보면 "
             "$x_1$ 의 계수가 $x_2$ 의 몫까지 받는다. 참값은 $-0.5$ 다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "partial_slope.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
