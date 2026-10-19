r"""5.5절 「그림으로 보는 통계학」 — 한 번도 일어나지 않았을 때.

n 번 보고 한 번도 일어나지 않았으면 p-hat = 0 이고 왈드 구간은 폭이 0 이다.
「확률이 0 이다」가 아니라 「아직 못 보았다」인데도 그렇다.

정확한 95% 상한은 1 - 0.05^(1/n) 이고, 이것이 n 이 조금만 커져도 3/n 에
거의 붙는다. 임상시험에서 쓰는 「삼의 법칙」이 이것이다.

만드는 파일:

  ch05/applications/img/rule_of_three.png
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/applications/img/"
INK, RED, TEAL, MUTED = "#37474F", "#D32F2F", "#00838F", "#90A4AE"


def main():
    n = np.arange(10, 1001)
    exact = 1 - 0.05 ** (1 / n)
    approx = 3 / n

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.plot(n, exact * 100, lw=3.0, color=TEAL, label="정확한 95% 상한")
    ax.plot(n, approx * 100, lw=2.0, color=RED, ls="--", label="삼의 법칙  3/n")
    ax.axhline(0, color=MUTED, lw=1.4)
    ax.text(520, 1.6, "왈드 구간은 여기 — 폭이 0 이다", fontsize=10.5, color=MUTED)
    ax.plot([10, 1000], [0, 0], lw=3.0, color=MUTED, alpha=0.5)
    ax.set_xlabel("관찰 횟수 n (한 번도 일어나지 않음)", fontsize=11.5)
    ax.set_ylabel("p 의 95% 상한 (%)", fontsize=11.5)
    ax.set_title("한 번도 못 보았다고 0 은 아니다", fontsize=12.5, color=INK, pad=8)
    ax.set_xlim(10, 1000); ax.set_ylim(-0.6, 20)
    ax.legend(fontsize=10.5); ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(n, (approx - exact) / exact * 100, lw=3.0, color=RED)
    ax.axhline(0, color=MUTED, lw=1.4, ls="--")
    for k in (30, 100, 300):
        e = 1 - 0.05 ** (1 / k)
        ax.plot([k], [(3 / k - e) / e * 100], marker="o", ms=9, color=RED)
        ax.annotate(f"n={k}\n{(3/k-e)/e*100:+.1f}%", xy=(k, (3/k-e)/e*100),
                    xytext=(k + 55, (3/k-e)/e*100 + 0.9), fontsize=10, color=RED)
    ax.set_xlabel("관찰 횟수 n", fontsize=11.5)
    ax.set_ylabel("3/n 이 정확한 값보다 몇 % 큰가", fontsize=11.5)
    ax.set_title("조금 넉넉한 쪽으로만 틀린다", fontsize=12.5, color=INK, pad=8)
    ax.set_xlim(10, 1000); ax.grid(alpha=0.25)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("삼의 법칙 — 분자가 0 일 때 쓰는 한 줄", fontsize=14.5)
    fig.text(0.5, -0.04,
             "3/n 은 언제나 정확한 값보다 조금 크다. 안전한 쪽으로 틀리므로 어림으로 쓰기에 알맞다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "rule_of_three.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    for k in (30, 100, 300, 1000):
        e = 1 - 0.05 ** (1 / k)
        print(f"    n={k:5d}  정확 {e:.5f}  3/n {3/k:.5f}  {(3/k-e)/e*100:+.1f}%")


if __name__ == "__main__":
    main()
