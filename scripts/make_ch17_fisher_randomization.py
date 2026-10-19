r"""17.4절 「그림으로 보는 통계학」 — 피셔가 손으로 센 32,768 가지.

이 쪽은 순열검정을 「가정이 없고 작은 표본에서 정확한」 방법으로 소개한다.
맞는 말인데, 그 정확한 방법이 **t 검정보다 먼저** 나왔다는 사실은 적혀 있지
않다.

다윈은 1876 년에 옥수수 열다섯 쌍을 길렀다. 한 쌍은 같은 화분에 심은
타가수분 포기와 자가수분 포기이고, 재어 둔 것은 키의 차이다(1/8 인치 단위).

  49, -67, 8, 16, 6, 23, 28, 41, 14, 29, 56, 24, 75, 60, -48

피셔는 1935 년 『실험의 설계』에서 이 자료를 두 번 분석했다. 한 번은 t 검정
으로, 한 번은 **부호를 뒤집는 모든 경우**로. 쌍이 열다섯이니 2^15 = 32,768
가지이고, 그는 그것을 다 세었다. 컴퓨터 없이.

두 p 값이 거의 같다. 피셔가 t 검정을 쓸 수 있다고 본 근거가 바로 이것이다 —
t 검정은 무작위화 검정의 **계산할 수 있는 근사**였다.

그런데 셀 수 있는 쪽이 교과서가 되고, 정확한 쪽은 컴퓨터를 사십 년 기다렸다.

오른쪽 그림은 그 근사가 n 이 커질수록 조여든다는 것을 잰다. n = 15 는 이미
두 p 값이 거의 붙는 자리다.

만드는 파일:

  ch17/permutation/img/fisher_randomization.png

실행:  python3 scripts/make_ch17_fisher_randomization.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/permutation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

# 다윈(1876) 옥수수 열다섯 쌍, 타가수분 - 자가수분, 1/8 인치
DARWIN = np.array([49, -67, 8, 16, 6, 23, 28, 41, 14, 29, 56, 24, 75, 60, -48],
                  dtype=float)

NS = (5, 8, 11, 15)          # 오른쪽 그림에서 전수조사할 쌍의 수
REPS, SEED = 400, 1935


def all_signs(n):
    """(2^n, n) 부호 행렬."""
    return np.array(list(itertools.product((-1.0, 1.0), repeat=n)))


def exact_p(d, signs):
    """부호 뒤집기 전수조사로 얻은 양쪽 p 값 — 통계량은 평균."""
    obs = abs(d.mean())
    means = signs @ d / len(d)
    return float((np.abs(means) >= obs - 1e-12).mean())


def t_p(d):
    return float(stats.ttest_1samp(d, 0.0).pvalue)


def main():
    n = len(DARWIN)
    signs15 = all_signs(n)
    means = signs15 @ DARWIN / n
    obs = DARWIN.mean()

    # 부호를 뒤집어도 sum(d^2) 은 변하지 않는다. 그래서 t 는 |평균| 의 증가함수이고
    # 평균으로 센 p 값과 t 로 센 p 값이 정확히 같다 — 척도만 맞추어 그린다.
    ss = (DARWIN ** 2).sum()
    s2 = (ss - n * means ** 2) / (n - 1)
    ts = means / np.sqrt(s2 / n)

    p_perm = exact_p(DARWIN, signs15)
    p_t = t_p(DARWIN)
    tstat = float(stats.ttest_1samp(DARWIN, 0.0).statistic)
    se = DARWIN.std(ddof=1) / np.sqrt(n)

    print(f"    다윈의 열다섯 쌍 — 평균차 {obs:.3f} (1/8 인치), 표준오차 {se:.3f}")
    print(f"    경우의 수 2^{n} = {2**n:,}")
    print(f"    무작위화 검정 p = {p_perm:.4f}")
    print(f"    t 검정        t = {tstat:.3f},  p = {p_t:.4f}")
    print(f"    두 p 값의 차이 = {abs(p_perm - p_t):.4f}")

    # ── n 에 따른 근사의 조여듦 ──
    rng = np.random.default_rng(SEED)
    gaps = {}
    for m in NS:
        sg = all_signs(m)
        g = []
        for _ in range(REPS):
            d = rng.normal(0.6, 1.0, m)      # 효과가 조금 있는 자료
            g.append(abs(exact_p(d, sg) - t_p(d)))
        gaps[m] = np.array(g)
        print(f"    n = {m:2}  |p_순열 - p_t| 중앙값 {np.median(gaps[m]):.4f}   "
              f"90 분위 {np.quantile(gaps[m], 0.9):.4f}   (전수 {2**m:,} 가지)")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)

    # ── 왼쪽: 다윈 자료의 무작위화 분포 ──
    ax = axes[0]
    lim = 4.6
    cnt, edges = np.histogram(np.clip(ts, -lim, lim), bins=110,
                              range=(-lim, lim), density=True)
    mid = (edges[:-1] + edges[1:]) / 2
    cols = [RED if abs(m) >= abs(tstat) - 1e-9 else MUTED for m in mid]
    ax.bar(mid, cnt, width=edges[1] - edges[0], color=cols, alpha=0.88,
           linewidth=0)
    xs = np.linspace(-lim, lim, 400)
    ax.plot(xs, stats.t.pdf(xs, n - 1), lw=2.6, color=BLUE)
    handles = [
        Line2D([], [], color=BLUE, lw=2.6, label=f"$t_{{{n-1}}}$ 밀도 — 피셔가 쓴 근사"),
        Patch(facecolor=MUTED, alpha=0.88, label=f"부호를 뒤집은 {2**n:,} 가지 전부"),
        Patch(facecolor=RED, alpha=0.88,
              label=f"관측값만큼 극단인 쪽 — $p = {p_perm:.4f}$"),
    ]
    for s in (-1, 1):
        ax.axvline(s * tstat, color=RED, lw=1.8, ls="--")
    ax.annotate(f"관측된 $t = {tstat:.3f}$\n(평균차 ${obs:.2f}$, $1/8$ 인치)",
                xy=(tstat + 0.25, 0.016), xytext=(3.35, 0.255),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(0.02, 0.97,
            f"무작위화 검정   $p = {p_perm:.4f}$\n"
            f"$t$ 검정            $p = {p_t:.4f}$",
            transform=ax.transAxes, va="top", fontsize=12.2, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GREY, alpha=0.95))
    ax.set_xlabel("부호를 뒤집어 다시 계산한 $t$ 통계량", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_ylim(0, 0.46)
    ax.set_title("다윈의 옥수수 열다섯 쌍, 피셔가 센 것 그대로",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(handles=handles, fontsize=9.8, framealpha=0.93,
              loc="upper left", bbox_to_anchor=(0.0, 0.80))
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 근사는 n 과 함께 조여든다 ──
    ax = axes[1]
    data = [gaps[m] for m in NS]
    bp = ax.boxplot(data, positions=range(len(NS)), widths=0.52,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color=RED, lw=2.4))
    for patch, col in zip(bp["boxes"], (AMBER, AMBER, TEAL, TEAL)):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    for i, m in enumerate(NS):
        ax.text(i, 0.142, f"중앙값\n{np.median(gaps[m]):.3f}",
                ha="center", va="top", fontsize=10.4, color=INK)
    ax.annotate("피셔가 센 자리", xy=(3, 0.022), xytext=(3, 0.082),
                fontsize=11.2, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels([f"$n = {m}$\n$2^{{{m}}} = {2**m:,}$".replace(",", "{,}")
                        for m in NS], fontsize=10.6)
    ax.set_ylim(0, 0.155)
    ax.set_xlabel("쌍의 수", fontsize=11.5)
    ax.set_ylabel("두 $p$ 값의 차이 (순열검정 대 $t$ 검정, 절댓값)", fontsize=11.5)
    ax.set_title("쌍이 늘면 근사가 조여든다", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("정확한 검정이 먼저 나왔고, 근사가 교과서가 되었다", fontsize=14.5)
    fig.text(0.5, -0.045,
             f"오른쪽은 자료를 {REPS} 번 새로 뽑아 쌍의 수마다 전수조사한 "
             "결과다. 상자는 사분범위, 가운데 선은 중앙값이다. "
             "순열검정의 $p$ 값은 어느 $n$ 에서나 정확하고, $t$ 검정의 $p$ 값은 "
             "그것에 가까워질 뿐이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "fisher_randomization.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
