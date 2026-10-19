r"""21.1절 「그림으로 보는 통계학」 — 상을 받으려면 먼저 살아 있어야 한다.

생존분석에서 가장 흔하고 가장 치명적인 실수가 불멸시간 편향이다.
이 쪽이 세우는 틀 — 「시작 시점」과 「사건 시점」을 분명히 한다 — 이
왜 형식이 아닌지를 보인다.

2001 년에 아카데미상 수상 배우가 후보에 그친 배우보다 평균 4 년쯤 더
오래 살았다는 연구가 의학 학술지에 실렸다. 명예가 건강에 좋다는
설명이 붙었다.

그런데 상을 받으려면 **시상식 날까지 살아 있어야 한다.** 서른에 죽은
배우는 예순에 상을 받을 수 없다. 수상자 집단은 「수상까지의 기간을
살아남은 사람들」로 미리 걸러져 있고, 소박한 비교는 그 기간을 통째로
수상의 공로로 돌린다. 이것을 불멸시간이라 부른다 — 수상자에게는
수상 전까지가 죽을 수 없는 시간이기 때문이다.

2006 년에 재분석이 나왔다. 수상을 시간의존 공변량으로 다루자 효과가
사실상 사라졌다.

모의실험으로 그 크기를 잰다. 수상이 수명에 **아무 영향도 주지 않도록**
만들어 두고 소박하게 비교하면

  수상자 평균 수명 75.9 년,  비수상자 48.2 년   차이 +27.7 년

수상 시점부터 재면(랜드마크) 차이가 사라진다.

만드는 파일:

  ch21/introduction/img/immortal_time.png

실행:  python3 scripts/make_ch21_immortal_time.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/introduction/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, MEAN, Q, SEED = 40_000, 50.0, 0.20, 2001
WINDOWS = (10, 20, 30, 40, 50, 60)
SHOW_W = 60


def trial(rng, window):
    """수상은 수명에 아무 영향도 주지 않는다. 살아 있어야 받을 수 있을 뿐."""
    T = rng.exponential(MEAN, N)
    A = rng.uniform(0, window, N)
    won = (T > A) & (rng.random(N) < Q)
    return T, A, won


def surv(v):
    v = np.sort(v)
    return v, 1 - np.arange(1, len(v) + 1) / len(v)


def main():
    rng = np.random.default_rng(SEED)
    print(f"{'수상 창 (년)':>12}{'수상자':>9}{'소박한 차이':>13}"
          f"{'수상 시점부터 잰 차이':>22}")
    naive, land = [], []
    for w in WINDOWS:
        T, A, won = trial(rng, w)
        g1 = T[won].mean() - T[~won].mean()
        # 랜드마크 — 수상 시점부터 남은 수명 대 같은 시점에 살아 있던 사람의 남은 수명
        rest_w = (T - A)[won]
        A2 = rng.uniform(0, w, N)
        alive = T > A2          # 그 시점에 살아 있던 사람 전부
        rest_c = (T - A2)[alive]
        g2 = rest_w.mean() - rest_c.mean()
        naive.append(g1); land.append(g2)
        print(f"{w:>12}{int(won.sum()):>9,}{g1:>13.2f}{g2:>22.2f}")
        if w == SHOW_W:
            keep = (T, A, won, rest_w, rest_c)

    T, A, won, rest_w, rest_c = keep
    print(f"\n    수상 창 {SHOW_W} 년에서")
    print(f"      수상자 평균 수명 {T[won].mean():.2f} 년, "
          f"비수상자 {T[~won].mean():.2f} 년")
    print(f"      수상 시점부터: 수상자 {rest_w.mean():.2f} 년, "
          f"대조 {rest_c.mean():.2f} 년")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 소박하게 그린 두 곡선 ──
    ax = axes[0]
    xw, yw = surv(T[won])
    xn, yn = surv(T[~won])
    xs = np.linspace(0, 200, 400)
    ax.plot(xs, np.exp(-xs / MEAN), lw=3.0, color=INK, ls="--",
            label="참 생존함수 — 두 집단이 같다")
    ax.plot(xw, yw, lw=3.0, color=RED, label=f"「수상자」 {int(won.sum()):,} 명")
    ax.plot(xn, yn, lw=3.0, color=TEAL, label=f"「비수상자」 {int((~won).sum()):,} 명")
    ax.axhline(0.5, color=GREY, lw=1.3, ls=":")
    mw = np.median(T[won]); mn = np.median(T[~won])
    ax.annotate(f"중앙 수명 {mw:.0f} 년", xy=(mw, 0.5), xytext=(mw + 14, 0.66),
                fontsize=10.8, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(f"{mn:.0f} 년", xy=(mn, 0.5), xytext=(mn + 3, 0.80),
                fontsize=10.8, color=TEAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("경력 시작 뒤 지난 햇수", fontsize=11.5)
    ax.set_ylabel("생존확률", fontsize=11.5)
    ax.set_title("수상이 수명에 전혀 영향을 주지 않는 자료다",
                 fontsize=12.6, color=INK, pad=8)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 언제부터 재느냐가 전부다 ──
    ax = axes[1]
    ax.axhline(0, color=INK, lw=2.4, ls="--")
    ax.text(10.5, 1.2, "참 효과 $0$", fontsize=11, color=INK)
    ax.plot(WINDOWS, naive, lw=3.0, color=RED, marker="o", ms=8,
            label="경력 시작부터 잰 평균 수명의 차이")
    ax.plot(WINDOWS, land, lw=3.0, color=TEAL, marker="D", ms=7,
            label="수상 시점부터 잰 남은 수명의 차이")
    ax.fill_between(WINDOWS, 0, naive, color=RED, alpha=0.13)
    i = WINDOWS.index(SHOW_W)
    ax.annotate(f"수상이 늦게까지 일어날 수 있을수록\n"
                f"불멸시간이 길어지고 거짓 효과가 커진다\n"
                f"— 여기서 $+{naive[i]:.1f}$ 년",
                xy=(SHOW_W, naive[i]), xytext=(38, 7.5), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("수상이 일어날 수 있는 기간 (년)", fontsize=11.5)
    ax.set_ylabel("수상자가 더 오래 산 햇수", fontsize=11.5)
    ax.set_title("언제부터 재는가가 답을 정한다", fontsize=12.6, color=INK,
                 pad=8)
    ax.set_ylim(-3.5, 33)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("상을 받으려면 먼저 살아 있어야 한다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"사람 {N:,} 명의 수명을 평균 {MEAN:.0f} 년인 지수분포로 만들고, "
             f"살아 있는 사람에게 확률 {Q:.0%} 로 수상을 주었다. "
             "수상은 수명에 아무 영향도 주지 않는다 — 그런데도 소박한 비교는 "
             "큰 차이를 만들어 낸다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "immortal_time.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
