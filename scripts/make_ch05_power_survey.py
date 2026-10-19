r"""5.7절 「그림으로 보는 통계학」 — 집단당 스무 명으로 무엇을 할 수 있는가.

이 쪽은 합동 t 통계량의 표집분포를 정확하게 세운다. 그 분포가 실제
연구에서 어떻게 쓰였는지를 두 번의 조사가 말해 준다.

1962 년에 제이컵 코언이 『이상·사회심리학회지』 한 권에 실린 논문을
전부 모아, 각 연구가 **중간 크기의 효과를 잡아낼 확률**을 계산했다.
중앙값이 0.48 이었다. 동전 던지기다.

1989 년에 제들마이어와 기거렌처가 25 년 뒤의 같은 학회지로 같은 일을
했다. 코언의 논문은 그 사이에 수없이 인용되었다. 그런데 검정력은
나아지지 않았다.

까닭은 관행이다. 심리학에서 「집단당 스무 명」이 오래 기본값이었다.
합동 t 검정으로 집단당 n 명이 가지는 검정력은 이렇다.

      n     d=0.2    d=0.5    d=0.8
     20     0.095    0.338    0.693
     30     0.119    0.478    0.861
     64     0.202    0.801    0.994
    394     0.801    1.000    1.000

검정력 0.80 에 필요한 집단당 인원은 d=0.8 이면 26, d=0.5 면 64,
d=0.2 면 394 다.

더 나쁜 것이 있다. 검정력이 낮으면 **유의하게 나온 연구가 효과를
부풀린다.** 참 d = 0.5 일 때 유의했던 연구들만 모아 평균을 내면

      n=10   d_hat 평균 1.222   참값의 2.44 배
      n=20   d_hat 평균 0.867   참값의 1.73 배
      n=64   d_hat 평균 0.565   참값의 1.13 배

작은 표본에서 유의가 나오려면 운이 좋아야 하고, 운이 좋았다는 것은
효과가 커 보였다는 뜻이다. 문헌에 남는 것은 그 운 좋은 쪽뿐이다.

만드는 파일:

  ch05/applications/img/power_survey.png

실행:  python3 scripts/make_ch05_power_survey.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

ALPHA, SEED, REPS = 0.05, 1962, 200_000
DS = ((0.8, TEAL, "큰 효과  $d = 0.8$"),
      (0.5, AMBER, "중간 효과  $d = 0.5$"),
      (0.2, BLUE, "작은 효과  $d = 0.2$"))
NS_SIM = (8, 10, 15, 20, 30, 45, 64, 100, 150)
D_TRUE = 0.5


def power(n, d, a=ALPHA):
    df = 2 * n - 2
    nc = d * np.sqrt(n / 2)
    t = stats.t.ppf(1 - a / 2, df)
    return float(stats.nct.sf(t, df, nc) + stats.nct.cdf(-t, df, nc))


def main():
    ns = np.arange(5, 401)
    print(f"{'n':>6}" + "".join(f"{f'd={d}':>9}" for d, _, _ in DS))
    for n in (10, 20, 30, 50, 64, 100, 394):
        print(f"{n:>6}" + "".join(f"{power(n, d):>9.3f}" for d, _, _ in DS))
    need = {}
    for d, _, _ in DS:
        need[d] = next(n for n in range(4, 4000) if power(n, d) >= 0.80)
        print(f"    d = {d}: 검정력 $0.80$ 에 필요한 집단당 n = {need[d]}")

    rng = np.random.default_rng(SEED)
    infl, pw = [], []
    print(f"\n    참 d = {D_TRUE} 일 때 유의한 연구가 보고하는 효과크기")
    print(f"{'n':>6}{'검정력':>9}{'유의한 것의 평균':>18}{'부풀림':>9}")
    for n in NS_SIM:
        a = rng.normal(D_TRUE, 1, (REPS, n))
        b = rng.normal(0, 1, (REPS, n))
        m = a.mean(1) - b.mean(1)
        sp = np.sqrt((a.var(1, ddof=1) + b.var(1, ddof=1)) / 2)
        t = m / (sp * np.sqrt(2 / n))
        dh = m / sp
        sig = np.abs(t) > stats.t.ppf(1 - ALPHA / 2, 2 * n - 2)
        infl.append(dh[sig].mean() / D_TRUE)
        pw.append(sig.mean())
        print(f"{n:>6}{sig.mean():>9.3f}{dh[sig].mean():>18.3f}{infl[-1]:>9.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 집단당 몇 명이면 되는가 ──
    ax = axes[0]
    for d, col, lab in DS:
        ax.plot(ns, [power(n, d) for n in ns], lw=2.9, color=col, label=lab)
    ax.axhline(0.80, color=INK, lw=1.8, ls="--")
    ax.text(300, 0.815, "흔히 목표로 하는 $0.80$", fontsize=10.8, color=INK,
            ha="right")
    ax.axhline(0.48, color=RED, lw=1.8, ls=":")
    ax.text(392, 0.497, "코언이 $1962$ 년에 센 중앙값 $0.48$", fontsize=10.8,
            color=RED, ha="right")
    ax.axvline(20, color=GREY, lw=1.8, ls=":")
    ax.annotate(f"집단당 $20$ 명 —\n"
                f"중간 효과도 {power(20, 0.5)*100:.0f}% 로만 잡는다",
                xy=(20, power(20, 0.5)), xytext=(95, 0.21),
                fontsize=10.8, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    for d, col, _ in DS:
        ax.plot([need[d]], [0.80], marker="o", ms=8, color=col, zorder=5)
        ax.text(need[d] + 4, 0.845, f"{need[d]}", ha="left", fontsize=10.4,
                color=col)
    ax.set_xlabel("집단당 인원 $n$", fontsize=11.5)
    ax.set_ylabel("효과를 잡아낼 확률 (검정력)", fontsize=11.5)
    ax.set_title("검정력 $0.80$ 에 드는 인원", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(5, 400)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 유의한 것만 남으면 효과가 부풀려진다 ──
    ax = axes[1]
    ax.plot(NS_SIM, infl, lw=3.0, color=RED, marker="o", ms=7)
    ax.axhline(1.0, color=INK, lw=2.0, ls="--")
    ax.text(150, 1.045, "참 효과크기", fontsize=11, color=INK, ha="right")
    ax.fill_between(NS_SIM, 1.0, infl, color=RED, alpha=0.14)
    i20 = NS_SIM.index(20)
    ax.plot([20], [infl[i20]], marker="o", ms=13, color=RED, zorder=5)
    ax.annotate(f"집단당 $20$ 명에서\n유의했던 연구들의 평균 효과는\n"
                f"참값의 {infl[i20]:.2f} 배",
                xy=(20, infl[i20]), xytext=(74, 2.10), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    i8 = NS_SIM.index(8)
    ax.annotate(f"$n = 8$ 이면 {infl[i8]:.1f} 배", xy=(8, infl[i8]),
                xytext=(26, infl[i8] + 0.04), fontsize=10.6, color=RED,
                ha="left", arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    for n, p in zip(NS_SIM, pw):
        if n in (20, 64):
            ax.text(n, 0.955, f"검정력 {p:.2f}", ha="center", va="top",
                    fontsize=10.2, color=GREY)
    ax.set_xlabel("집단당 인원 $n$", fontsize=11.5)
    ax.set_ylabel("보고되는 효과크기 ÷ 참 효과크기", fontsize=11.5)
    ax.set_title(f"참 효과는 $d = {D_TRUE}$ 하나뿐이다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(0.80, 2.95)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("집단당 스무 명으로 무엇을 할 수 있는가", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"오른쪽은 참 효과를 $d = {D_TRUE}$ 하나로 고정해 두고 유의하게 "
             f"나온 연구만 모아 평균 낸 것이다(각 {REPS//10000} 만 번). "
             "효과가 커진 것이 아니라, 작은 표본에서 유의가 나오려면 "
             "운이 좋아야 하기 때문이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "power_survey.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
