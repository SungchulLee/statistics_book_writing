r"""6.4절 「그림으로 보는 통계학」 — 왜 「확률」이 아니라 「가능도」인가.

이 쪽은 가능도함수를 세우고 그 봉우리를 찾는 일이 MLE 라고 적는다.
그런데 왜 굳이 「가능도」라는 다른 말을 쓰는가. 자료를 보고 나서 모수에
대해 말하는 것이니 그냥 확률이라 부르면 안 되는가.

피셔가 1912 년에 이 방법을 처음 적었을 때 그도 그 구분이 흐릿했다.
케임브리지 3 학년 때 쓴 그 논문에서 그는 이 양을 적분하기도 했다.
1920 년대 초에 그는 **가능도**라는 말을 따로 만들고, 그것이 확률이
아니며 확률의 법칙을 따르지 않는다고 못 박았다.

까닭은 눈금을 바꾸어 보면 드러난다. n=10, k=3 에서

  p 의 MLE                     0.300000
  오즈로 옮긴 값 g(p̂)           0.428572
  오즈 눈금에서 직접 최대화한 값   0.428572   <- 같다

**MLE 는 눈금을 바꾸어도 따라간다.** 봉우리는 수가 아니라 자리이기
때문이다.

그런데 가능도를 모수의 밀도로 다루면 달라진다. 밀도를 옮기려면
야코비안을 곱해야 하고, 그러면 봉우리가 움직인다.

  p 밀도의 최빈값                0.300000
  오즈 밀도의 최빈값              0.333301
  그것을 p 로 되돌리면            0.249982   <- 0.300 이 아니다

그리고 가능도는 애초에 p 에 대해 적분하면 1 이 아니다
(이 자료에서 0.000758).

만드는 파일:

  ch06/mle/img/likelihood_not_probability.png

실행:  python3 scripts/make_ch06_likelihood_not_probability.py  (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/mle/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, K = 10, 3
TICKS_P = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def main():
    p = np.linspace(1e-6, 1 - 1e-6, 400_001)
    L = p ** K * (1 - p) ** (N - K)
    phat = float(p[int(np.argmax(L))])
    om_hat = phat / (1 - phat)

    w = np.linspace(1e-6, 60, 600_001)
    pw = w / (1 + w)
    Lw = pw ** K * (1 - pw) ** (N - K)
    w_direct = float(w[int(np.argmax(Lw))])

    dens_p = L / np.trapz(L, p)
    dens_w = Lw / (1 + w) ** 2
    dens_w /= np.trapz(dens_w, w)
    mode_w = float(w[int(np.argmax(dens_w))])
    mode_w_as_p = mode_w / (1 + mode_w)

    print(f"  n={N}, k={K}")
    print(f"    p 의 MLE                 {phat:.6f}")
    print(f"    오즈로 옮긴 값 g(p̂)       {om_hat:.6f}")
    print(f"    오즈 눈금에서 직접 최대화   {w_direct:.6f}")
    print(f"\n    가능도를 밀도로 다루면")
    print(f"      p 밀도의 최빈값          {p[int(np.argmax(dens_p))]:.6f}")
    print(f"      오즈 밀도의 최빈값        {mode_w:.6f}")
    print(f"      그것을 p 로 되돌리면      {mode_w_as_p:.6f}")
    print(f"\n    가능도의 p 에 대한 적분 = {np.trapz(L, p):.8f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    # ── 왼쪽: 봉우리는 수가 아니라 자리다 ──
    ax = axes[0]
    ax.plot(p, L / L.max(), lw=3.4, color=TEAL)
    ax.axvline(phat, color=INK, lw=2.0, ls="--")
    ax.plot([phat], [1.0], marker="o", ms=13, color=INK, zorder=5)
    ax.annotate(f"같은 봉우리를\n$p$ 로 읽으면 ${phat:.3f}$\n"
                f"오즈로 읽으면 ${om_hat:.4f}$",
                xy=(phat, 1.0), xytext=(0.63, 0.72), fontsize=11, color=INK,
                ha="center", arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(0.63, 0.40,
            "오즈 눈금에서 가능도를\n직접 최대화해도 "
            f"${w_direct:.4f}$ 다.\n**눈금을 바꾸어도 따라간다.**".replace("**", ""),
            fontsize=10.6, color=TEAL, ha="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GREY, alpha=0.95))
    ax.set_xlabel("$p$", fontsize=11.5)
    ax.set_ylabel("가능도 (봉우리를 $1$ 로)", fontsize=11.5)
    ax.set_title("봉우리는 수가 아니라 자리다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    top = ax.secondary_xaxis("top")
    top.set_xticks(list(TICKS_P))
    top.set_xticklabels([f"{t/(1-t):.2f}" for t in TICKS_P], fontsize=10)
    top.set_xlabel("같은 자리를 오즈 $\\omega = p/(1-p)$ 로 읽은 것",
                   fontsize=11, labelpad=6)

    # ── 오른쪽: 밀도로 다루면 봉우리가 움직인다 ──
    ax = axes[1]
    ax.plot(p, dens_p / dens_p.max(), lw=3.4, color=TEAL,
            label="가능도를 $p$ 의 밀도로 본 것")
    # 오즈 밀도의 "모양"을 p 눈금에 그대로 옮겨 그린다 (야코비안을 되곱하지 않는다)
    back = np.interp(p, pw, dens_w)
    back = back / np.nanmax(back)
    ax.plot(p, back, lw=3.0, color=RED, ls="--",
            label="가능도를 오즈의 밀도로 본 것 (되돌림)")
    ax.axvline(phat, color=TEAL, lw=1.8, ls=":")
    ax.axvline(mode_w_as_p, color=RED, lw=1.8, ls=":")
    ax.annotate(f"${phat:.3f}$", xy=(phat, 0.06), xytext=(phat + 0.06, 0.16),
                fontsize=11.5, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
    ax.annotate(f"${mode_w_as_p:.3f}$", xy=(mode_w_as_p, 0.06),
                xytext=(mode_w_as_p - 0.07, 0.16), fontsize=11.5, color=RED,
                ha="right", arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.annotate("밀도로 다루면 눈금을 바꿀 때\n야코비안이 곱해지고,\n"
                "봉우리가 옮겨 간다",
                xy=(0.42, 0.55), xytext=(0.70, 0.62), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("$p$", fontsize=11.5)
    ax.set_ylabel("밀도 (봉우리를 $1$ 로)", fontsize=11.5)
    ax.set_title("밀도라고 부르는 순간 달라진다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("왜 「확률」이 아니라 「가능도」인가", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"베르누이 시행 ${N}$ 번에 성공 ${K}$ 번이다. 두 곡선 모두 "
             "봉우리를 $1$ 로 맞추어 그렸으므로 높이는 견주지 말고 "
             "봉우리의 **자리**만 보면 된다.".replace("**", ""),
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "likelihood_not_probability.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
