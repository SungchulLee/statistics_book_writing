r"""3.1절 「그림으로 보는 통계학」 — 공리를 어기면 돈을 잃는다.

이 쪽은 콜모고로프의 세 공리를 적고 거기서 무엇이 따라 나오는지를 보인다.
그런데 「왜 그 셋을 받아들여야 하는가」는 다른 물음이다. 램지(1926)와
드 피네티(1931)가 따로 내놓은 답이 더치북이다.

어떤 사건 A 에 대해 「A 가 일어나면 1 을 주는 표」를 p 에 팔겠다고 하자.
p 가 당신이 매긴 믿음의 값이다. 그러면

  p(A) + p(A^c) = s

일 때, 상대가 두 표를 다 사면 그는 s 를 내고 **반드시** 1 을 받는다.
s < 1 이면 그가 확실히 벌고, s > 1 이면 방향을 뒤집어 확실히 번다.
어느 쪽이든 당신은 |s - 1| 만큼 **확실히** 잃는다.

가법성도 같다. A 와 B 가 배반일 때 p(A∪B) 가 p(A)+p(B) 와 다르면
그 차액만큼 확실히 잃는다.

만드는 파일:

  ch03/probability/img/dutch_book.png

실행:  python3 scripts/make_ch03_dutch_book.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch03/probability/img/"

INK = "#37474F"
MUTED = "#90A4AE"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"

PA, PAC = 0.6, 0.6          # 당신이 매긴 두 값 — 합이 1.2 다


def main():
    s = PA + PAC
    loss = abs(s - 1)
    print(f"    매긴 값  p(A) = {PA},  p(A^c) = {PAC},  합 = {s}")
    print(f"    상대가 두 표를 모두 당신에게 팔면 그는 {s} 를 받고 1 을 돌려준다")
    print(f"    A 가 일어나든 안 일어나든 당신의 손익 = {1 - s:+.2f}")
    print(f"    확실한 손실 = |s - 1| = {loss:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 두 갈래 모두 손해 ──
    ax = axes[0]
    rows = [("$A$ 가 일어난다", PA, PAC, 1.0), ("$A$ 가 안 일어난다", PA, PAC, 1.0)]
    labels = ["$A$ 가 일어난다", "$A^c$ 가 일어난다"]
    recv = [s, s]                       # 두 표를 팔아 받은 돈
    pay = [1.0, 1.0]                    # 어느 쪽이든 한 표가 당첨되어 1 을 준다
    x = np.arange(2)
    ax.bar(x - 0.17, recv, width=0.34, color=TEAL, label="받은 돈 (표 두 장 값)")
    ax.bar(x + 0.17, pay, width=0.34, color=AMBER, label="치러야 할 돈")
    for i in range(2):
        ax.text(i - 0.17, recv[i] + 0.03, f"{recv[i]:.2f}", ha="center",
                fontsize=11.5, color=INK)
        ax.text(i + 0.17, pay[i] + 0.03, f"{pay[i]:.2f}", ha="center",
                fontsize=11.5, color=INK)
        ax.annotate("", xy=(i + 0.17, 1.0), xytext=(i - 0.17, s),
                    arrowprops=dict(arrowstyle="<->", color=RED, lw=1.8))
        ax.text(i, (s + 1) / 2 + 0.02, f"{1-s:+.2f}", ha="center",
                fontsize=12, color=RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("금액", fontsize=11.5)
    ax.set_title(f"$p(A) = {PA}$, $p(A^c) = {PAC}$ 라고 말하면\n"
                 "두 갈래가 모두 손해다", fontsize=12.6, color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 합이 1 에서 벗어난 만큼 ──
    ax = axes[1]
    ss = np.linspace(0.4, 1.6, 400)
    ax.plot(ss, np.abs(ss - 1), lw=3.0, color=RED)
    ax.axvline(1, color=TEAL, lw=2.4, ls="--")
    ax.text(1.02, 0.52, "합이 $1$ 일 때만\n확실한 손실이 $0$ 이다",
            fontsize=11, color=TEAL)
    ax.plot([s], [loss], marker="o", ms=10, color=INK, zorder=5)
    ax.annotate(f"왼쪽 그림의 자리\n합 {s}, 손실 {loss:.1f}",
                xy=(s, loss), xytext=(1.36, 0.30), fontsize=10.6, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.fill_between(ss, 0, np.abs(ss - 1), color=RED, alpha=0.12)
    ax.set_xlabel("$p(A) + p(A^c)$ — 당신이 매긴 두 값의 합", fontsize=11.5)
    ax.set_ylabel("상대가 확실히 가져가는 금액", fontsize=11.5)
    ax.set_title("공리를 어긴 만큼이 그대로 값이다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 0.64)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("믿음의 값이 공리를 어기면 확실히 돈을 잃는다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "표 한 장은 「$A$ 가 일어나면 $1$ 을 준다」는 약속이고, 당신이 매긴 값이 "
             "그 값이다. 상대는 당신의 값만 보고 베팅을 짜며, $A$ 가 일어날지는 "
             "알 필요가 없다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "dutch_book.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
