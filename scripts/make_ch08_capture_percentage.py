r"""8.1절 「그림으로 보는 통계학」 — 95 퍼센트가 가리키는 것.

신뢰구간을 읽는 흔한 오독 하나가 「이 구간이 다음 실험의 결과도 95%
담을 것」이라는 것이다. 담지 않는다. 얼마나 담는지는 닫힌 꼴로 나온다.

실험을 한 번 해서 x-bar1 을 얻고 구간 x-bar1 +- 1.96 sigma/sqrt(n) 을
그린다. 같은 모집단에서 같은 크기로 한 번 더 해서 x-bar2 를 얻는다.
두 평균의 차는 분산이 두 배이므로

  P(x-bar2 가 구간 안) = P(|Z| <= 1.96/sqrt(2)) = 2 Phi(1.3859) - 1
                       = 0.8341

모수는 95% 담지만 다음 실험의 평균은 83.4% 담는다.

그리고 이 83.4% 는 평균일 뿐이다. 첫 실험의 평균이 참값에서 d 표준오차만큼
떨어져 있었다면 그 구간의 포착률은

  Phi(d + 1.96) - Phi(d - 1.96)

이고, d = 0 이면 0.950, d = 1 이면 0.834, d = 1.96 이면 0.500,
d = 2.5 이면 0.295 다. 내가 그린 구간이 어느 쪽인지는 알 수 없다.

만드는 파일:

  ch08/foundations/img/capture_percentage.png

실행:  python3 scripts/make_ch08_capture_percentage.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch08/foundations/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

Z, NREP, SEED = 1.959964, 25, 1937


def capture(d):
    """첫 평균이 참값에서 d 표준오차 떨어졌을 때 다음 평균을 담을 확률."""
    return norm.cdf(d + Z) - norm.cdf(d - Z)


def main():
    avg = 2 * norm.cdf(Z / np.sqrt(2)) - 1
    print(f"  평균 포착률 = 2 Phi({Z:.4f}/sqrt2) - 1 = {avg:.4f}")
    print(f"\n{'d':>6}{'그 구간의 포착률':>18}")
    for d in (0.0, 0.5, 1.0, 1.5, Z, 2.5, 3.0):
        print(f"{d:>6.2f}{capture(d):>18.4f}")
    print(f"\n  모의로 확인: ", end="")
    rng = np.random.default_rng(SEED)
    m1 = rng.normal(size=400_000)
    m2 = rng.normal(size=400_000)
    print(f"{np.mean(np.abs(m2 - m1) <= Z):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 어디에 떨어졌느냐에 달렸다 ──
    ax = axes[0]
    ds = np.linspace(-3.6, 3.6, 600)
    ax.fill_between(ds, 0, norm.pdf(ds) * 1.9, color=MUTED, alpha=0.22,
                    zorder=1)
    ax.plot(ds, capture(ds), lw=3.6, color=TEAL, zorder=3)
    ax.axhline(avg, color=RED, lw=2.4, ls="--", zorder=2)
    ax.axhline(0.95, color=INK, lw=1.8, ls=":", zorder=2)
    for d, col in ((0.0, INK), (Z, AMBER), (2.5, RED)):
        ax.plot([d], [capture(d)], marker="o", ms=12, color=col, zorder=5,
                mec="white", mew=1.6)
    ax.text(-3.5, 0.962, "약속한 $0.95$", fontsize=10.6, color=INK)
    ax.text(-3.5, avg - 0.052, f"평균 포착률 ${avg:.3f}$", fontsize=10.8,
            color=RED)
    ax.annotate(f"참값 위에 떨어진 구간은\n${capture(0):.3f}$ 을 담는다",
                xy=(0.0, capture(0)), xytext=(-2.45, 0.50), fontsize=10.6,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate(f"${Z:.2f}$ 표준오차 벗어난 구간은\n${capture(Z):.3f}$ 뿐이다",
                xy=(Z, capture(Z)), xytext=(2.62, 0.655), fontsize=10.6,
                color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.text(0.0, 0.055, "첫 실험의 평균이 떨어지는 자리", fontsize=10.2,
            color=GREY, ha="center")
    ax.set_xlabel("첫 실험의 평균이 참값에서 떨어진 거리 (표준오차)",
                  fontsize=11.5)
    ax.set_ylabel("다음 실험의 평균을 담을 확률", fontsize=11.5)
    ax.set_title("어디에 떨어졌느냐에 달렸다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(0, 1.04)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 스물다섯 번 되풀이하면 ──
    ax = axes[1]
    rs = np.random.default_rng(4)   # 포착 수가 이론값 근처로 나오는 판
    ms = rs.normal(size=NREP)
    lo, hi = ms[0] - Z, ms[0] + Z
    inside = (ms[1:] >= lo) & (ms[1:] <= hi)
    ax.axvspan(lo, hi, color=TEAL, alpha=0.14, zorder=1)
    ax.axvline(0, color=INK, lw=2.2, ls="--", zorder=2)
    ys = np.arange(NREP)
    ax.plot([lo, hi], [0, 0], lw=3.6, color=TEAL, zorder=4)
    ax.plot([ms[0]], [0], marker="o", ms=11, color=TEAL, zorder=5,
            mec="white", mew=1.5)
    for i in range(1, NREP):
        col = GREY if inside[i - 1] else RED
        ax.plot([ms[i] - Z, ms[i] + Z], [ys[i], ys[i]], lw=2.0, color=MUTED,
                zorder=3)
        ax.plot([ms[i]], [ys[i]], marker="o", ms=8, color=col, zorder=5,
                mec="white", mew=1.2)
    ax.text(0.12, NREP - 0.6, "참값", fontsize=10.8, color=INK)
    ax.annotate(f"처음 한 번으로 그린 구간\n"
                f"뒤따른 ${NREP-1}$ 번 가운데 ${inside.sum()}$ 번의 평균을 담았다",
                xy=(hi, 3.2), xytext=(hi + 0.80, 8.8), fontsize=10.8,
                color=TEAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("평균 (참값을 $0$, 표준오차를 $1$ 로 잰 눈금)", fontsize=11.5)
    ax.set_ylabel("실험 번호", fontsize=11.5)
    ax.set_title(f"같은 실험을 ${NREP}$ 번 되풀이하면", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(-4.6, 5.4)
    ax.set_ylim(-1.2, NREP)
    ax.legend(handles=[
        Line2D([], [], color=TEAL, lw=3.2, label="처음 그린 구간"),
        Line2D([], [], lw=0, marker="o", ms=8, color=GREY,
               label="그 구간 안에 든 평균"),
        Line2D([], [], lw=0, marker="o", ms=8, color=RED,
               label="벗어난 평균")],
        fontsize=10.0, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$95$ 퍼센트가 가리키는 것", fontsize=14.5)
    fig.text(0.5, -0.06,
             "모든 눈금은 표준오차 단위다. 왼쪽 회색 산은 첫 실험의 평균이 "
             "어디에 떨어지는지의 분포이고, 이 분포로 포착률 곡선을 평균 내면 "
             f"${avg:.4f}$ 이 된다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "capture_percentage.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
