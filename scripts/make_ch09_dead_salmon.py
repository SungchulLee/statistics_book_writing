r"""9.4절 「그림으로 보는 통계학」 — 죽은 연어의 뇌.

베넷과 동료들은 fMRI 장비를 점검하려고 호박과 콘월산 닭을 넣어 보다가
대서양 연어 한 마리를 넣었다. 죽은 연어다. 사람들이 찍힌 사진을 보여
주고 「이 사람이 어떤 감정인지 판단하라」고 지시했다.

보정 없이 복셀마다 p < 0.001 로 자르자 연어의 뇌에서 세 점이 켜졌다.
다중비교 보정을 넣으면 하나도 남지 않았다. 2009 년에 발표했고 2012 년에
이그노벨 신경과학상을 받았다. 발표 당시 fMRI 논문의 25~40% 가 다중비교
보정을 보고하지 않고 있었다.

이 쪽 1 절의 FWER 곡선은 검정이 **독립**일 때의 그림이다. 뇌영상 자료는
독립이 아니다. 이웃한 복셀끼리 닮아 있고, 분석 과정에서 일부러 평활을
넣기도 한다. 그래서 「이웃한 복셀 셋 이상이 함께 켜져야 인정한다」는
흔한 안전장치가 있는데, 이것이 얼마나 지켜 주는지 재 본다.

64 x 64 격자에 신호를 하나도 넣지 않고 p < 0.001 로 자르면

                   넘은 복셀   P(최대 덩어리 >= 3)   P(>= 5)
  독립                 4.1           0.001           0.000
  평활 sigma=1.2       4.1           0.456           0.158

넘은 복셀 수는 같다. 뭉치느냐 흩어지느냐만 다르다. 그리고 덩어리 셋을
요구하는 안전장치는 평활이 들어간 순간 열 번에 네 번 그냥 통과한다.

만드는 파일:

  ch09/multiple_testing/img/dead_salmon.png

실행:  python3 scripts/make_ch09_dead_salmon.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/multiple_testing/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N = 64                      # 64 x 64 = 4096 복셀
ZCUT = 3.0902               # 한쪽꼬리 p = 0.001
SMOOTH = 1.2
REPS = 2000
KS = np.arange(1, 9)


def field(rng, sm):
    x = rng.normal(size=(N, N))
    if sm:
        x = gaussian_filter(x, sm, mode="wrap")
        x /= x.std()
    return x


def biggest(mask):
    lab, k = label(mask)
    return 0 if k == 0 else int(np.bincount(lab.ravel())[1:].max())


def sweep(seed, sm):
    rng = np.random.default_rng(seed)
    big = np.empty(REPS, int)
    cnt = np.empty(REPS, int)
    for i in range(REPS):
        m = field(rng, sm) > ZCUT
        cnt[i] = m.sum()
        big[i] = biggest(m)
    return cnt, big


def main():
    cI, bI = sweep(2009, 0)
    cS, bS = sweep(2012, SMOOTH)
    pI = np.array([(bI >= k).mean() for k in KS])
    pS = np.array([(bS >= k).mean() for k in KS])

    print(f"  {N}x{N} = {N*N} 복셀, 신호 없음, p < 0.001 로 자름, {REPS} 번")
    print(f"{'':>14}{'넘은 복셀':>12}{'P(>=3)':>10}{'P(>=5)':>10}"
          f"{'최대 덩어리':>14}")
    for nm, c, b in (("독립", cI, bI), (f"평활 {SMOOTH}", cS, bS)):
        print(f"{nm:>14}{c.mean():>12.1f}{(b>=3).mean():>10.3f}"
              f"{(b>=5).mean():>10.3f}{b.mean():>14.2f}")

    # 보일 두 장을 고른다 — 평활 쪽은 덩어리가 생긴 판을 쓴다
    r1 = np.random.default_rng(31)
    fI = field(r1, 0)
    r2 = np.random.default_rng(7)
    while True:
        fS = field(r2, SMOOTH)
        if biggest(fS > ZCUT) >= 4:
            break

    fig = plt.figure(figsize=(13.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.45])

    for row, (f, nm) in enumerate(((fI, "이웃끼리 독립일 때"),
                                   (fS, f"평활을 넣었을 때 ($\\sigma = {SMOOTH}$)"))):
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(f, cmap="Greys", vmin=-3.6, vmax=3.6, interpolation="nearest")
        m = f > ZCUT
        ys, xs = np.nonzero(m)
        ax.scatter(xs, ys, s=26, marker="s", color=RED, zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{nm} — 넘은 복셀 {m.sum()} 개, "
                     f"가장 큰 덩어리 {biggest(m)}", fontsize=11.4, color=INK,
                     pad=5)
        for sp in ax.spines.values():
            sp.set_color(MUTED)

    ax = fig.add_subplot(gs[:, 1])
    ax.axvline(3, color=GREY, lw=1.8, ls=":")
    ax.plot(KS, pS, lw=3.4, color=RED, marker="o", ms=8,
            label=f"평활을 넣었을 때 ($\\sigma = {SMOOTH}$)")
    ax.plot(KS, pI, lw=3.2, color=TEAL, marker="D", ms=7, ls="--",
            label="이웃끼리 독립일 때")
    ax.plot([3], [pS[2]], marker="o", ms=14, color=RED, zorder=5,
            mec="white", mew=1.8)
    ax.annotate(f"「이웃 셋이 함께 켜져야 인정」이라는\n"
                f"안전장치가 ${pS[2]*100:.0f}\\%$ 그냥 통과한다",
                xy=(3, pS[2]), xytext=(5.1, 0.62), fontsize=11,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("독립이면 이 자리에서 이미 $0$ 이다",
                xy=(3.1, 0.012), xytext=(6.3, 0.19), fontsize=11,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xticks(KS)
    ax.set_xlabel("요구하는 덩어리 크기 (붙어 있는 복셀 수)", fontsize=11.5)
    ax.set_ylabel("신호가 하나도 없는데 통과할 확률", fontsize=11.5)
    ax.set_title("넘은 복셀 수는 같다. 뭉치느냐만 다르다.", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(-0.03, 1.04)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["$0$", "$0.25$", "$0.5$", "$0.75$", "$1$"])
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("죽은 연어의 뇌", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"왼쪽 두 장은 ${N} \\times {N}$ 격자에 신호를 하나도 넣지 "
             f"않고 만든 잡음이다. 빨간 칸이 $p < 0.001$ 을 넘은 복셀이다. "
             f"오른쪽은 같은 실험을 ${REPS}$ 번 되풀이한 결과다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "dead_salmon.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
