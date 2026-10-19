r"""9.2절 「그림으로 보는 통계학」 — 문턱은 누가 정했는가.

1983 년 9 월 26 일 새벽, 모스크바 남쪽 세르푸호프-15 의 조기경보
지휘소에서 당직을 서던 스타니슬라프 페트로프 중령 앞에서 경보가
울렸다. 미국에서 대륙간탄도탄 한 발이 발사되었다는 것이었고, 곧
넷이 더해져 다섯이 되었다.

그는 설비 고장이라고 판단해 위로 그렇게 보고했다. 까닭 하나는
이것이었다 — 선제공격이라면 다섯 발일 리가 없다. 지상 레이더도
아무것도 잡지 못했다. 원인은 추분 무렵의 높은 구름에 반사된 햇빛이
위성 궤도와 겹친 것이었다. 이 일은 1998 년까지 비밀이었다.

경보기의 문턱은 설계자가 정했다. 그리고 그 문턱에는 그날 밤 그가
치러야 할 대가가 들어 있지 않았다.

대가와 기저율이 정하는 문턱을 적어 보면 닫힌 꼴이 나온다.
H0: N(0,1), H1: N(delta,1), 문턱 c 일 때 기대손실

  L(c) = (1-pi) C1 (1-Phi(c)) + pi C2 Phi(c-delta)

를 최소로 하는 c 는

  c* = delta/2 + (1/delta) log[ (1-pi) C1 / (pi C2) ]

이다. delta=3, pi=1e-4 로 두면

  C1/C2        c*        최적 alpha
  1/1000      2.27       1.2e-02
  1           4.57       2.4e-06
  1000        6.87       3.2e-12
  1e6         9.18       2.3e-20

알파를 0.05 로 두는 것은 이 두 축 위의 한 점을 고르는 일이고,
그 점이 옳은 자리는 거의 없다.

만드는 파일:

  ch09/errors_and_power/img/petrov_threshold.png

실행:  python3 scripts/make_ch09_petrov_threshold.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/errors_and_power/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

DELTA = 3.0
PI = 1e-4                       # 「오늘 밤 선제공격이 있다」의 사전확률
RATIOS = ((1e-3, TEAL, "놓치는 쪽이 $1000$ 배 비싸다"),
          (1.0, AMBER, "둘의 대가가 같다"),
          (1e6, RED, "헛경보가 $10^{6}$ 배 비싸다"))
PIS = ((1e-2, BLUE, "$10^{-2}$"), (1e-4, INK, "$10^{-4}$"),
       (1e-6, RED, "$10^{-6}$"))


def cstar(ratio, pi, d=DELTA):
    """기대손실을 최소로 하는 문턱. C1/C2 = ratio."""
    return d / 2 + np.log((1 - pi) * ratio / pi) / d


def loss(c, ratio, pi, d=DELTA):
    """헛경보 한 번의 대가를 1 로 둔 기대손실 (놓침은 1/ratio)."""
    return (1 - pi) * norm.sf(c) + pi / ratio * norm.cdf(c - d)


def main():
    print(f"  delta = {DELTA}, pi = {PI:g}")
    print(f"{'C1/C2':>10}{'c*':>10}{'최적 alpha':>16}")
    for r in (1e-3, 1e0, 1e3, 1e6):
        c = cstar(r, PI)
        print(f"{r:>10.0e}{c:>10.2f}{norm.sf(c):>16.2e}")
    print(f"\n  alpha = 0.05 는 c = {norm.isf(0.05):.3f} 이다.")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 대가가 바뀌면 최소점이 옮겨 간다 ──
    ax = axes[0]
    cs = np.linspace(0.0, 11.0, 800)
    for ratio, col, lab in RATIOS:
        L = loss(cs, ratio, PI)
        ax.plot(cs, L / L.min(), lw=3.2, color=col, label=lab)
        c = cstar(ratio, PI)
        ax.plot([c], [1.0], marker="o", ms=12, color=col, zorder=5,
                mec="white", mew=1.7)
    ax.axvline(norm.isf(0.05), color=GREY, lw=2.0, ls=":")
    ax.text(1.80, 0.795, "$\\alpha = 0.05$ 의 자리", fontsize=11, color=GREY)
    ax.set_yscale("log")
    ax.set_xlabel("문턱 $c$ — 오른쪽일수록 경보를 아낀다", fontsize=11.5)
    ax.set_ylabel("기대손실 $\\div$ 그 곡선의 최솟값", fontsize=11.5)
    ax.set_title("대가가 바뀌면 최소점이 옮겨 간다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.75, 60)
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(["$1$", "$3$", "$10$", "$30$"])
    ax.minorticks_off()
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 최적 알파는 수십 자릿수를 오간다 ──
    ax = axes[1]
    rs = np.logspace(-6, 9, 400)
    for pi, col, lab in PIS:
        ax.plot(rs, norm.sf(cstar(rs, pi)), lw=3.2, color=col,
                label=f"사전확률 $\\pi = ${lab}")
    ax.axhline(0.05, color=GREY, lw=2.4, ls="--",
               label="관행의 $\\alpha = 0.05$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1e-6, 1e-3, 1, 1e3, 1e6, 1e9])
    ax.set_xticklabels(["$10^{-6}$", "$10^{-3}$", "$1$", "$10^{3}$",
                        "$10^{6}$", "$10^{9}$"])
    ax.set_yticks([1e-20, 1e-15, 1e-10, 1e-5, 1])
    ax.set_yticklabels(["$10^{-20}$", "$10^{-15}$", "$10^{-10}$",
                        "$10^{-5}$", "$1$"])
    ax.minorticks_off()
    ax.set_xlabel("헛경보의 대가 $\\div$ 놓침의 대가", fontsize=11.5)
    ax.set_ylabel("기대손실을 최소로 하는 $\\alpha$", fontsize=11.5)
    ax.set_title("최적 $\\alpha$ 는 스무 자릿수를 오간다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(1e-22, 3)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("문턱은 누가 정했는가", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"$H_0: N(0,1)$, $H_1: N({DELTA:g},1)$ 로 두었다. "
             f"왼쪽은 사전확률 $\\pi = 10^{{-4}}$ 에서 대가의 비만 바꾼 "
             "것이고, 동그라미가 각 곡선의 최소점이다. 세로눈금은 곡선마다 "
             "제 최솟값으로 나누어 모양을 견줄 수 있게 했다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "petrov_threshold.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
