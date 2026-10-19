r"""9.1절 「그림으로 보는 통계학」 — 「편리하므로」.

이 쪽의 표는 0.05 를 「대부분의 과학 연구(관례적 기본값)」로 적는다.
그 관례가 어디서 왔는지는 적혀 있지 않다.

1925 년 피셔의 『연구자를 위한 통계적 방법』에 이런 문장이 있다.
P = .05, 곧 스무 번에 한 번에 해당하는 값이 1.96 으로 거의 2 이니,
유의한지 아닌지를 판단하는 한계로 이 점을 잡는 것이 **편리하다**.

편리하다. 그것이 이유의 전부였다. 그리고 실제로 편리했다 — 그는 정확한
p 값 대신 몇 개의 유의수준에서의 임계값만 표로 실었다.

피셔 자신은 뒤에 이 읽기를 부정했다. 1956 년에 그는 과학자가 해마다
어떤 상황에서나 쓰는 고정된 유의수준 같은 것은 없으며 사안마다 증거를
보고 판단한다고 적었다. 오늘날 우리가 쓰는 고정 알파는 네이만-피어슨의
결정이론에서 온 것이고, 두 틀이 교과서에서 뒤섞였다.

그렇다면 p = 0.05 는 증거로서 얼마나 센가. 셀키·바야리·버거(2001)가
정밀한 귀무가설에 대해 최소 베이즈 인자의 하한을 주었다.

  B(p) = -e p ln p        (p < 1/e)

사전확률이 반반일 때 귀무가설의 사후확률이 B/(1+B) 이상이다.

      p        최대 BF      H0 의 사후확률 하한
   0.05         2.46            0.289
   0.01         7.99            0.111
   0.005       13.89            0.067
   0.001       53.26            0.018

p = 0.05 는 기껏해야 2.5 대 1 이다. 「스무 번에 한 번」이라는 말이
주는 인상과 전혀 다르다. 2018 년에 유의수준을 0.005 로 낮추자는 제안이
나온 근거가 이 표다.

만드는 파일:

  ch09/foundations/img/p005.png

실행:  python3 scripts/make_ch09_p005.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/foundations/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

MARKS = ((0.05, RED, "$p = 0.05$", 0.45, 2.6),
         (0.005, AMBER, "$p = 0.005$", 0.50, 2.4),
         (0.001, TEAL, "$p = 0.001$", 0.30, 4.6))
BANDS = ((1, 3.2, "말할 거리가 못 된다"), (3.2, 10, "보통"),
         (10, 100, "강하다"), (100, 400, "매우 강하다"))


def bmin(p):
    return -np.e * p * np.log(p)


def main():
    print(f"{'p':>9}{'최대 BF':>11}{'H0 의 사후확률 하한':>22}")
    for p in (0.1, 0.05, 0.01, 0.005, 0.001, 0.0001):
        b = bmin(p)
        print(f"{p:>9.4f}{1/b:>11.2f}{b/(1+b):>22.4f}")

    ps = np.logspace(np.log10(2e-5), np.log10(0.2), 500)
    bf = 1 / bmin(ps)
    post = bmin(ps) / (1 + bmin(ps))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: p 값이 사 주는 최대 증거 ──
    ax = axes[0]
    for lo, hi, lab in BANDS:
        ax.axhspan(lo, hi, color=GREY, alpha=0.08)
        ax.text(1.7e-5, np.sqrt(lo * hi), lab, fontsize=10.4, color=GREY,
                va="center")
    ax.plot(ps, bf, lw=3.4, color=INK)
    for p, col, lab, fx, fy in MARKS:
        ax.plot([p], [1 / bmin(p)], marker="o", ms=13, color=col, zorder=5,
                mec="white", mew=1.6)
        ax.annotate(f"{lab}\n최대 ${1/bmin(p):.1f}$ 대 $1$",
                    xy=(p, 1 / bmin(p)),
                    xytext=(p * fx, 1 / bmin(p) * fy), fontsize=10.6,
                    color=col, ha="center",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1"])
    ax.set_yticks([1, 3, 10, 30, 100, 300])
    ax.set_yticklabels(["1", "3", "10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_xlabel("$p$ 값", fontsize=11.5)
    ax.set_ylabel("$H_1$ 쪽으로 살 수 있는 최대 베이즈 인자", fontsize=11.5)
    ax.set_title("$p$ 값이 사 줄 수 있는 증거의 상한", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(1.6e-5, 0.21)
    ax.set_ylim(1, 400)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 귀무가설의 사후확률 ──
    ax = axes[1]
    ax.fill_between(ps, 0, post, color=RED, alpha=0.14)
    ax.plot(ps, post, lw=3.4, color=RED)
    for p, col, lab, _fx, _fy in MARKS:
        b = bmin(p)
        ax.plot([p], [b / (1 + b)], marker="o", ms=13, color=col, zorder=5,
                mec="white", mew=1.6)
    b05 = bmin(0.05)
    ax.annotate(f"$p = 0.05$ 로 기각해도\n귀무가설의 사후확률이 "
                f"적어도 ${b05/(1+b05):.2f}$ 다",
                xy=(0.05, b05 / (1 + b05)), xytext=(0.0024, 0.40),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    b005 = bmin(0.005)
    ax.annotate(f"$0.005$ 로 낮추면 ${b005/(1+b005):.3f}$ —\n"
                "$2018$ 년 제안의 근거다",
                xy=(0.005, b005 / (1 + b005)), xytext=(0.0011, 0.17),
                fontsize=10.6, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_xticklabels(["0.0001", "0.001", "0.01", "0.1"])
    ax.minorticks_off()
    ax.set_xlabel("$p$ 값", fontsize=11.5)
    ax.set_ylabel("$H_0$ 의 사후확률 하한 (사전확률 반반)", fontsize=11.5)
    ax.set_title("기각한 뒤에도 남는 몫", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(1.6e-5, 0.21)
    ax.set_ylim(0, 0.52)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$p = 0.05$ 는 얼마나 센 증거인가", fontsize=14.5)
    fig.text(0.5, -0.05,
             "셀키·바야리·버거(2001)의 하한 $B(p) = -e\\,p\\ln p$ 다. "
             "정밀한 귀무가설을 검정할 때, 어떤 사전분포를 골라도 이보다 "
             "더 $H_1$ 쪽으로 갈 수 없다는 뜻이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "p005.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
