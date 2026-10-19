r"""1.4절 「그림으로 보는 통계학」 — 걷고, 말하고, 보고, 쓰고.

1958 년 7 월 8 일, 『뉴욕 타임스』에 이런 기사가 실렸다.
「새 해군 장치는 해 보면서 배운다 — 심리학자가 컴퓨터의 배아를 보이다」.
해군은 이 장치가 「걷고, 말하고, 보고, 쓰고, 스스로를 복제하고, 자기
존재를 의식하게 될 것으로 기대한다」고 밝혔다.

프랭크 로젠블랫의 퍼셉트론이었다. 눈에 해당하는 광전지 400 개가 보낸
신호를 받아 1000 개 남짓한 연결의 무게를 고쳐 가며 배우는 기계였다.

1969 년에 마빈 민스키와 시모어 페퍼트가 『퍼셉트론』을 냈다. 그 안에
한 줄짜리 증명이 있다. 한 층짜리 퍼셉트론은 배타적 논리합(XOR)을
나타낼 수 없다. 선 하나로 가를 수 없는 문제이기 때문이다.

중요한 것은 이것이 **배움의 문제가 아니라 표현의 문제**라는 점이다.
자료를 백 배 주어도, 천 배 주어도 달라지지 않는다. 편향-분산 분해에서
분산은 자료로 줄일 수 있지만 편향은 그럴 수 없다.

xy > 0 을 1 로 두고 이름표에 5% 잡음을 섞어 재면

       자료 수      x1, x2 만      x1*x2 를 더하면
          50          0.518           0.849
         500          0.511           0.909
       5,000          0.499           0.935
     100,000          0.498           0.946

왼쪽 줄은 꼼짝하지 않고 오른쪽 줄은 천장(0.95)까지 올라간다. 항 하나의
차이다.

만드는 파일:

  ch01/paradigms/img/perceptron_xor.png

실행:  python3 scripts/make_ch01_perceptron_xor.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/paradigms/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

FLIP, SEED, NTEST, REPS = 0.05, 1958, 40_000, 30
NS = (50, 100, 200, 500, 1000, 2000, 5000, 20000, 100000)
NSHOW = 400


def draw(rng, n):
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    flip = rng.random(n) < FLIP
    return X, y ^ flip


def feats(X, cross):
    return np.c_[X, X[:, 0] * X[:, 1]] if cross else X


def acc(rng, n, cross):
    Xtr, ytr = draw(rng, n)
    Xte, yte = draw(rng, NTEST)
    if len(np.unique(ytr)) < 2:
        return 0.5
    m = LogisticRegression(C=1e4, max_iter=2000).fit(feats(Xtr, cross), ytr)
    return float(m.score(feats(Xte, cross), yte))


def main():
    rng = np.random.default_rng(SEED)
    plain, crossed = [], []
    print(f"{'자료 수':>10}{'x1, x2 만':>14}{'x1*x2 를 더하면':>18}")
    for n in NS:
        a = np.mean([acc(rng, n, False) for _ in range(REPS)])
        b = np.mean([acc(rng, n, True) for _ in range(REPS)])
        plain.append(a); crossed.append(b)
        print(f"{n:>10,}{a:>14.3f}{b:>18.3f}")
    print(f"\n  이름표 잡음이 {FLIP:.0%} 이므로 천장은 {1-FLIP:.2f} 다.")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 선 하나로는 가를 수 없다 ──
    ax = axes[0]
    rs = np.random.default_rng(3)
    X, y = draw(rs, NSHOW)
    m = LogisticRegression(C=1e4, max_iter=2000).fit(X, y)
    ax.plot(X[y == 1, 0], X[y == 1, 1], lw=0, marker="o", ms=6.5, color=TEAL,
            alpha=0.85, zorder=3)
    ax.plot(X[y == 0, 0], X[y == 0, 1], lw=0, marker="X", ms=7.5, color=RED,
            alpha=0.85, zorder=3)
    w, b0 = m.coef_[0], m.intercept_[0]
    xs = np.linspace(-3.6, 3.6, 10)
    ax.plot(xs, -(b0 + w[0] * xs) / w[1], lw=3.2, color=INK, zorder=4)
    ax.axhline(0, color=MUTED, lw=1.2, zorder=1)
    ax.axvline(0, color=MUTED, lw=1.2, zorder=1)
    ax.annotate(f"한 층짜리 퍼셉트론이 그을 수 있는 선\n"
                f"어디에 그어도 맞히는 비율이 ${plain[-1]:.2f}$ 근처다",
                xy=(1.1, -(b0 + w[0] * 1.1) / w[1]), xytext=(0.0, -3.05),
                fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlabel("$x_1$", fontsize=11.5)
    ax.set_ylabel("$x_2$", fontsize=11.5)
    ax.set_title("선 하나로는 가를 수 없다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-3.6, 3.6)
    ax.legend(handles=[
        Line2D([], [], lw=0, marker="o", ms=7, color=TEAL,
               label="$x_1 x_2 > 0$"),
        Line2D([], [], lw=0, marker="X", ms=8, color=RED,
               label="$x_1 x_2 < 0$")],
        fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 자료를 더 준다고 되는 일이 아니다 ──
    ax = axes[1]
    ax.axhline(1 - FLIP, color=GREY, lw=2.2, ls=":")
    ax.text(62, 1 - FLIP + 0.013, f"이름표 잡음이 정하는 천장 ${1-FLIP:.2f}$",
            fontsize=10.6, color=GREY)
    ax.plot(NS, crossed, lw=3.6, color=TEAL, marker="o", ms=7,
            label="$x_1 x_2$ 항을 하나 더한 모형")
    ax.plot(NS, plain, lw=3.4, color=RED, marker="D", ms=7, ls="--",
            label="$x_1, x_2$ 만 쓰는 모형")
    ax.annotate(f"자료를 ${NS[-1]//NS[0]}$ 배 늘려도\n"
                f"${plain[-1]:.3f}$ 에서 꼼짝하지 않는다",
                xy=(NS[-1], plain[-1]), xytext=(2600, 0.63), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks([50, 500, 5000, 50000])
    ax.set_xticklabels(["$50$", "$500$", "$5{,}000$", "$50{,}000$"])
    ax.minorticks_off()
    ax.set_xlabel("학습에 쓴 자료의 수", fontsize=11.5)
    ax.set_ylabel("새 자료에서 맞힌 비율", fontsize=11.5)
    ax.set_title("자료를 더 준다고 되는 일이 아니다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.44, 1.0)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="center right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("걷고, 말하고, 보고, 쓰고", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"참 규칙은 $x_1 x_2 > 0$ 이고 이름표의 ${FLIP*100:.0f}\\%$ 를 "
             "뒤집어 잡음을 넣었다. 두 모형 모두 로지스틱 회귀이며 "
             "다른 것은 설명변수 목록뿐이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "perceptron_xor.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
