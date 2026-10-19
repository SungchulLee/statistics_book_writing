r"""20.3절 「그림으로 보는 통계학」 — 온도라는 이름이 붙은 까닭.

소프트맥스는 1870 년대 볼츠만이 기체를 설명하려고 쓴 식과 같다.
열평형에 있는 계가 에너지 E_i 인 상태에 있을 확률이

  p_i = exp(-E_i / kT) / sum_j exp(-E_j / kT)

이다. z_i = -E_i/k 로 두면 글자까지 같은 식이 된다. 분모에 깁스가
「분배함수」라는 이름을 붙였고, 지금도 그렇게 부른다.

그래서 언어모형에서 손잡이의 이름이 「온도」다. 비유가 아니라 같은 식의
같은 자리다.

  T -> 0    얼어붙는다. 가장 큰 로짓 하나만 남는다 (argmax)
  T  = 1    모형이 내놓은 그대로
  T -> oo   녹는다. 모든 상태가 똑같이 그럴듯해진다 (균등분포)

로짓 10 개를 놓고 재면

  T     엔트로피/log10     으뜸이 뽑힐 확률
   0.2      0.169              0.899
   0.5      0.478              0.594
   1.0      0.720              0.380
   2.0      0.897              0.237
   5.0      0.981              0.150
  20.0      0.999              0.112

만드는 파일:

  ch20/optimization/img/boltzmann_temperature.png

실행:  python3 scripts/make_ch20_boltzmann_temperature.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch20/optimization/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

Z = np.array([3.1, 2.6, 2.4, 1.5, 1.1, 0.6, 0.2, -0.4, -1.0, -1.8])
SHOW = ((0.25, BLUE, "$T = 0.25$ — 얼어붙었다"),
        (1.0, INK, "$T = 1$ — 모형이 내놓은 그대로"),
        (4.0, RED, "$T = 4$ — 녹았다"))


def soft(z, T):
    e = np.exp((z - z.max()) / T)
    return e / e.sum()


def main():
    C = len(Z)
    Ts = np.logspace(-1, 1.5, 400)
    ent = np.array([-(p := soft(Z, T)) @ np.log(p) for T in Ts]) / np.log(C)
    top = np.array([soft(Z, T)[Z.argmax()] for T in Ts])

    print(f"  로짓 {C} 개: {', '.join(f'{v:g}' for v in Z)}")
    print(f"\n{'T':>8}{'엔트로피/log C':>16}{'으뜸이 뽑힐 확률':>20}")
    for T in (0.2, 0.5, 1.0, 2.0, 5.0, 20.0):
        p = soft(Z, T)
        print(f"{T:>8.1f}{-(p @ np.log(p)) / np.log(C):>16.3f}"
              f"{p[Z.argmax()]:>20.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 같은 로짓, 다른 온도 ──
    ax = axes[0]
    xs = np.arange(C)
    w = 0.26
    for k, (T, col, lab) in enumerate(SHOW):
        ax.bar(xs + (k - 1) * w, soft(Z, T), width=w, color=col, alpha=0.88,
               label=lab, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"${v:g}$" for v in Z], fontsize=9.6)
    ax.set_xlabel("범주별 로짓 $z_i$", fontsize=11.5)
    ax.set_ylabel("뽑힐 확률", fontsize=11.5)
    ax.set_title("같은 로짓인데 온도가 다르면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 0.95)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 얼었다가 녹는다 ──
    ax = axes[1]
    ax.axvspan(Ts[0], 0.45, color=BLUE, alpha=0.08)
    ax.axvspan(3.2, Ts[-1], color=RED, alpha=0.07)
    ax.plot(Ts, ent, lw=3.4, color=AMBER)
    ax.plot(Ts, top, lw=3.2, color=TEAL, ls="--")
    ax.annotate("흩어진 정도 — 엔트로피 $\\div \\log C$",
                xy=(1.35, 0.81), xytext=(3.6, 0.76), fontsize=10.6,
                color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.annotate("가장 큰 로짓이 실제로 뽑힐 확률",
                xy=(1.55, 0.29), xytext=(4.6, 0.47), fontsize=10.6,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.axvline(1.0, color=INK, lw=1.8, ls=":")
    ax.plot([1.0], [ent[np.argmin(abs(Ts - 1))]], marker="o", ms=12,
            color=AMBER, zorder=5, mec="white", mew=1.6)
    ax.plot([1.0], [top[np.argmin(abs(Ts - 1))]], marker="D", ms=11,
            color=TEAL, zorder=5, mec="white", mew=1.6)
    ax.text(0.165, 0.24, "얼어붙는다\n$\\arg\\max$ 하나만", fontsize=10.8,
            color=BLUE, ha="center")
    ax.text(13.0, 0.30, "녹는다\n모두 똑같아진다", fontsize=10.8,
            color=RED, ha="center")
    ax.text(1.06, 0.022, "$T = 1$", fontsize=11, color=INK, ha="left")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2", "5", "10", "30"])
    ax.minorticks_off()
    ax.set_xlabel("온도 $T$", fontsize=11.5)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_title("한쪽 끝에서 얼고 반대쪽 끝에서 녹는다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("온도라는 이름이 붙은 까닭", fontsize=14.5)
    fig.text(0.5, -0.06,
             "로짓 $10$ 개를 고정해 놓고 $T$ 만 바꾸었다. 볼츠만의 식에서 "
             "$T$ 가 올라가면 기체의 상태들이 고르게 퍼지고 내려가면 가장 "
             "낮은 에너지 상태로 몰린다 — 여기서도 똑같다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "boltzmann_temperature.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
