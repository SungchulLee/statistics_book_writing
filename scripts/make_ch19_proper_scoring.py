r"""19.4절 「그림으로 보는 통계학」 — 거짓말하면 손해 보게 만든 점수.

브라이어 점수는 1950 년에 미국 기상국의 글렌 브라이어가 **일기예보관을
채점하려고** 만들었다. 예보관이 「내일 비 올 확률 70%」라고 말하면 그
말을 어떻게 채점할 것인가.

언뜻 절대오차가 자연스러워 보인다 — 비가 오면 |0.7 - 1| = 0.3, 안 오면
|0.7 - 0| = 0.7. 그런데 이 규칙에는 치명적인 구멍이 있다.

참으로 믿는 값이 p 일 때 q 라고 보고하면 기대 점수가

  절대오차   p(1-q) + (1-p)q = q(1-2p) + p      <- q 에 대해 1 차식
  브라이어   p(1-q)^2 + (1-p)q^2                 <- q = p 에서 최소
  로그       -[p ln q + (1-p) ln(1-q)]           <- q = p 에서 최소

절대오차는 q 에 대해 **직선**이므로 최소가 끝점에서 난다. p > 0.5 면
q = 1 이라고 말하는 것이 이득이다. 70% 라고 믿으면서 100% 라고 외쳐야
점수가 좋아지는 규칙이다.

브라이어와 로그는 그렇지 않다. 참으로 믿는 값을 그대로 말하는 것이
**정확히** 최적이다. 이런 규칙을 적정(proper) 점수규칙이라 한다.

일기예보관이 세상에서 가장 잘 보정된 직업군으로 알려진 까닭이 여기
있다 — 1950 년 이래 날마다, 공개적으로, 거짓말하면 손해 보는 규칙으로
채점받아 왔다. 빠르고 잦고 모호하지 않은 되먹임이다.

만드는 파일:

  ch19/evaluation/img/proper_scoring.png

실행:  python3 scripts/make_ch19_proper_scoring.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/evaluation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

P_TRUE = 0.70
EPS = 1e-9


def brier(p, q):
    return p * (1 - q) ** 2 + (1 - p) * q ** 2


def absol(p, q):
    return p * (1 - q) + (1 - p) * q


def logsc(p, q):
    q = np.clip(q, EPS, 1 - EPS)
    return -(p * np.log(q) + (1 - p) * np.log(1 - q))


def main():
    qs = np.linspace(0, 1, 1001)
    print(f"    참으로 믿는 값 p = {P_TRUE}")
    for nm, f in (("브라이어", brier), ("로그", logsc), ("절대오차", absol)):
        v = f(P_TRUE, qs)
        print(f"      {nm:<6} 최소가 되는 보고값 q = {qs[int(np.argmin(v))]:.3f}")
    print("\n    보고하는 것이 이득인 값 (참 믿음별)")
    ps = np.array([0.1, 0.3, 0.45, 0.55, 0.7, 0.9])
    for p in ps:
        best = [qs[int(np.argmin(f(p, qs)))] for f in (brier, logsc, absol)]
        print(f"      p = {p:.2f}   브라이어 {best[0]:.2f}   로그 {best[1]:.2f}   "
              f"절대오차 {best[2]:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 70% 를 믿는 예보관이 무엇을 말해야 하나 ──
    ax = axes[0]
    for f, col, lab, sc in ((brier, TEAL, "브라이어 점수", 1.0),
                            (logsc, BLUE, "로그 점수 (척도 조정)", 0.45),
                            (absol, RED, "절대오차", 1.0)):
        ax.plot(qs, sc * f(P_TRUE, qs), lw=3.0, color=col, label=lab)
    for f, col, sc in ((brier, TEAL, 1.0), (logsc, BLUE, 0.45)):
        ax.plot([P_TRUE], [sc * f(P_TRUE, P_TRUE)], marker="o", ms=12,
                color=col, zorder=5, mec="white", mew=1.6)
    ax.plot([1.0], [absol(P_TRUE, 1.0)], marker="o", ms=12, color=RED,
            zorder=5, mec="white", mew=1.6)
    ax.axvline(P_TRUE, color=INK, lw=1.8, ls="--")
    ax.text(P_TRUE - 0.018, 0.175, f"참으로 믿는 값 ${P_TRUE}$", fontsize=10.6,
            color=INK, ha="right", rotation=90, va="top")
    ax.annotate("브라이어와 로그는\n믿는 값을 그대로 말할 때 최소다",
                xy=(P_TRUE, brier(P_TRUE, P_TRUE)), xytext=(0.30, 0.055),
                fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate("절대오차는 $q = 1$ 에서 최소다 —\n"
                "$70\\%$ 를 믿으면서 $100\\%$ 라 외쳐야 이득이다",
                xy=(1.0, absol(P_TRUE, 1.0)), xytext=(0.40, 0.48),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("예보관이 **보고하는** 확률 $q$".replace("**", ""),
                  fontsize=11.5)
    ax.set_ylabel("기대 점수 (작을수록 좋다)", fontsize=11.5)
    ax.set_title(f"비 올 확률을 ${P_TRUE}$ 로 믿는 예보관", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.66)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 참 믿음마다 무엇을 말하는 것이 이득인가 ──
    ax = axes[1]
    pp = np.linspace(0.001, 0.999, 400)
    ax.plot(pp, pp, lw=3.4, color=TEAL, label="브라이어 · 로그 — 믿는 대로 말한다")
    step = np.where(pp > 0.5, 1.0, 0.0)
    ax.plot(pp[pp < 0.5], step[pp < 0.5], lw=3.0, color=RED,
            label="절대오차 — $0$ 또는 $1$ 만 말한다")
    ax.plot(pp[pp > 0.5], step[pp > 0.5], lw=3.0, color=RED)
    ax.plot([0.5, 0.5], [0, 1], lw=1.6, color=RED, ls=":")
    ax.annotate("믿음이 조금만 $0.5$ 를 넘어도\n「확실하다」고 말하게 된다",
                xy=(0.52, 1.0), xytext=(0.72, 0.52), fontsize=10.8, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("적정 점수규칙 —\n정직이 유일한 최적 전략이다",
                xy=(0.33, 0.33), xytext=(0.30, 0.78), fontsize=10.8,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("예보관이 참으로 믿는 확률 $p$", fontsize=11.5)
    ax.set_ylabel("점수가 가장 좋아지는 보고값", fontsize=11.5)
    ax.set_title("규칙이 무엇을 말하게 만드는가", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.04, 1.08)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("거짓말하면 손해 보게 만든 점수", fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽에서 로그 점수는 다른 둘과 같은 축에 놓으려고 $0.45$ 를 "
             "곱했다. 최소가 나는 자리는 척도와 무관하다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "proper_scoring.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
