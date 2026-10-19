r"""18.2절 「그림으로 보는 통계학」 — 편향을 일부러 들여온 사람들.

가우스-마르코프 정리는 최소제곱이 BLUE 라고 말한다 — Best Linear
**Unbiased** Estimator. 「최선」이 불편추정량 **안에서의** 최선이라는
것이 백오십 년 동안 거의 읽히지 않았다. 불편성은 고를 수 있는 성질이
아니라 갖추어야 할 조건으로 여겨졌다.

1970 년에 호얼과 케너드가 그것을 버리자고 했다. 일부러 치우친 추정량을
쓰자는 것이다. 받아들여지기 쉬운 제안이 아니었다.

그들이 함께 내놓은 것이 이 정리다.

  참 beta 가 무엇이든, ridge 의 MSE 를 최소제곱보다 낮추는 lambda > 0 이
  **반드시 존재한다.**

곧 불편성은 어느 경우에도 최적이 아니다.

확인해 보면 (n=50, p=10, 설명변수 상관 0.7, sigma=1)

   ||beta||   OLS MSE   최적 lambda   최적 MSE   줄어든 몫
      0.2      0.834      203.36       0.036      95.7%
      1.0      0.834        9.74       0.421      49.5%
      3.0      0.834        0.69       0.772       7.4%
     10.0      0.834        0.31       0.804       3.6%

beta 가 커질수록 이득이 줄지만 **0 이 되지는 않는다.** 최적 lambda 도
작아질 뿐 양수로 남는다.

다만 그 최적 lambda 는 모르는 beta 에 달려 있다. 그래서 교차검증이
있는 것이다 — 정리는 존재를 보장할 뿐 값을 알려 주지 않는다.

만드는 파일:

  ch18/ridge/img/ridge_existence.png

실행:  python3 scripts/make_ch18_ridge_existence.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/ridge/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

N, P, RHO, SIG, SEED = 50, 10, 0.7, 1.0, 1970
SHOW = ((0.2, BLUE), (1.0, TEAL), (3.0, AMBER), (10.0, RED))
LAMS = np.concatenate([[0.0], np.logspace(-2, 2.8, 500)])


def main():
    rng = np.random.default_rng(SEED)
    S = RHO * np.ones((P, P))
    np.fill_diagonal(S, 1.0)
    X = rng.normal(size=(N, P)) @ np.linalg.cholesky(S).T
    X -= X.mean(0)
    G = X.T @ X
    I = np.eye(P)
    direction = rng.normal(size=P)
    direction /= np.linalg.norm(direction)

    def mse(lam, beta):
        A = np.linalg.inv(G + lam * I)
        bias = (A @ G - I) @ beta
        return float(bias @ bias + SIG ** 2 * np.trace(A @ G @ A))

    print(f"  n={N}, p={P}, 설명변수 상관 {RHO}, sigma={SIG}")
    print(f"{'||beta||':>10}{'OLS MSE':>11}{'최적 lambda':>13}"
          f"{'최적 MSE':>11}{'줄어든 몫':>11}")
    curves, marks = [], []
    for s, col in SHOW:
        b = s * direction
        v = np.array([mse(l, b) for l in LAMS])
        i = int(np.argmin(v))
        curves.append((s, col, v / v[0]))
        marks.append((LAMS[i], v[i] / v[0], col, s))
        print(f"{s:>10.1f}{v[0]:>11.4f}{LAMS[i]:>13.4f}{v[i]:>11.4f}"
              f"{100*(1-v[i]/v[0]):>10.1f}%")

    scales = np.logspace(-1, 1.6, 40)
    best_l, best_gain = [], []
    for s in scales:
        b = s * direction
        v = np.array([mse(l, b) for l in LAMS])
        i = int(np.argmin(v))
        best_l.append(LAMS[i]); best_gain.append(1 - v[i] / v[0])
    print(f"\n  최적 lambda 가 0 인 경우: "
          f"{sum(1 for l in best_l if l == 0)} / {len(best_l)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: MSE 곡선 ──
    ax = axes[0]
    ax.axhline(1.0, color=INK, lw=2.2, ls="--")
    ax.text(0.012, 1.025, "최소제곱 — 불편추정량", fontsize=11, color=INK)
    for s, col, v in curves:
        ax.plot(LAMS[1:], v[1:], lw=3.0, color=col,
                label=f"$\\|\\beta\\| = {s:g}$")
    for lam, val, col, s in marks:
        if lam > 0:
            ax.plot([lam], [val], marker="o", ms=11, color=col, zorder=5,
                    mec="white", mew=1.6)
    ax.set_xscale("log")
    ax.set_xticks([0.01, 0.1, 1, 10, 100])
    ax.set_xticklabels(["0.01", "0.1", "1", "10", "100"])
    ax.minorticks_off()
    ax.set_xlabel("정칙화 세기 $\\lambda$", fontsize=11.5)
    ax.set_ylabel("MSE (최소제곱 $= 1$)", fontsize=11.5)
    ax.set_title("어느 $\\beta$ 에서도 $1$ 아래로 내려간다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 1.45)
    ax.annotate(f"$\\|\\beta\\| = 10$ 이어도\n최적 $\\lambda$ 가 "
                f"${marks[3][0]:.2f}$ 로 양수다",
                xy=(marks[3][0], marks[3][1]), xytext=(2.2, 0.42),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 이득은 줄어도 사라지지 않는다 ──
    ax = axes[1]
    ax.plot(scales, np.array(best_gain) * 100, lw=3.2, color=TEAL,
            marker="o", ms=5)
    ax.axhline(0, color=INK, lw=2.0, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
    ax.set_yscale("log")
    ax.set_yticks([1, 3, 10, 30, 100])
    ax.set_yticklabels(["1%", "3%", "10%", "30%", "100%"])
    ax.minorticks_off()
    for s, col in SHOW:
        i = int(np.argmin(abs(scales - s)))
        ax.plot([scales[i]], [best_gain[i] * 100], marker="o", ms=11,
                color=col, zorder=5, mec="white", mew=1.6)
    ax.annotate(f"$\\|\\beta\\|$ 가 커지면 이득이 줄지만\n"
                f"$0$ 이 되지 않는다 — $\\|\\beta\\| = 40$ 에서도 "
                f"${best_gain[-1]*100:.1f}\\%$",
                xy=(scales[-1], best_gain[-1] * 100), xytext=(0.9, 2.0),
                fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("참 계수의 크기 $\\|\\beta\\|$", fontsize=11.5)
    ax.set_ylabel("최적 $\\lambda$ 가 줄여 주는 MSE", fontsize=11.5)
    ax.set_title("이득은 줄어도 사라지지 않는다", fontsize=12.8, color=INK,
                 pad=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("불편성은 어느 경우에도 최적이 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개, 설명변수 ${P}$ 개, 쌍별 상관 ${RHO}$, "
             f"오차 표준편차 ${SIG}$. MSE 는 모의가 아니라 닫힌 식으로 "
             "계산했다 — 치우침의 제곱과 분산을 그대로 더한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "ridge_existence.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
