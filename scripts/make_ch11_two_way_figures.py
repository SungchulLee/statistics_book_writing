r"""11장 이원배치 분산분석 개요 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch11/anova_two_way/img/twoway_blocking.png   둘째 요인을 모형에 넣으면 오차가 줄어든다

실행:  python3 scripts/make_ch11_two_way_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch11/anova_two_way/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def fig_blocking():
    # 연습문제 1 의 자료: 교육 방식(온라인/대면) x 경력(주니어/시니어), 칸당 n = 4
    cell = np.array([[12.5, 18.5],     # 온라인: 주니어, 시니어
                     [15.0, 22.5]])    # 대면
    n = 4
    SSA, SSB, SSAB, SSW = 42.25, 182.25, 2.25, 20.0
    SS_tot = SSA + SSB + SSAB + SSW

    MSW = SSW / 12
    F_two = SSA / MSW
    p_two = stats.f.sf(F_two, 1, 12)

    SSE_one = SSB + SSAB + SSW          # 경력을 무시하면 모두 오차가 된다
    MSE_one = SSE_one / 14
    F_one = SSA / MSE_one
    p_one = stats.f.sf(F_one, 1, 14)
    crit_two = stats.f.ppf(0.95, 1, 12)
    crit_one = stats.f.ppf(0.95, 1, 14)
    print(f"  이원배치: MSW={MSW:.4f} F={F_two:.4f} p={p_two:.6f} crit={crit_two:.3f}")
    print(f"  일원배치: MSE={MSE_one:.4f} F={F_one:.4f} p={p_one:.6f} crit={crit_one:.3f}")

    fig = plt.figure(figsize=(12.6, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1], wspace=0.38)

    # --- (a) 칸 평균
    ax = fig.add_subplot(gs[0, 0])
    xs = [0, 1]
    ax.plot(xs, cell[0], "o-", color=BLUE, lw=2.2, ms=9, label="온라인")
    ax.plot(xs, cell[1], "s-", color=ORANGE, lw=2.2, ms=9, label="대면")
    for i, row in enumerate(cell):
        for j, v in enumerate(row):
            ax.annotate(f"{v}", (xs[j], v), textcoords="offset points",
                        xytext=(0, 10 if i == 1 else -17), ha="center",
                        fontsize=10, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["주니어", "시니어"], fontsize=10.5)
    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(9, 26)
    ax.set_ylabel("생산성 (칸 평균)", fontsize=10.5, color=INK)
    ax.set_title("칸 평균 — 두 선이 거의 평행하다", fontsize=11.5, color=INK)
    ax.legend(fontsize=10, loc="upper left")
    clean(ax)

    # --- (b) 제곱합 246.75 를 어디에 배분하는가
    ax = fig.add_subplot(gs[0, 1])
    labels = ["교육 방식 (A)", "경력 (B)", "교호작용", "오차"]
    two = [SSA, SSB, SSAB, SSW]
    one = [SSA, 0, 0, SSE_one]
    colors = [BLUE_F, GREEN_F, "#EDE7F6", ORANGE_F]
    edges = [BLUE, GREEN, PURPLE, ORANGE]
    for x, vals in [(0, two), (1, one)]:
        bottom = 0.0
        for v, c, e, lb in zip(vals, colors, edges, labels):
            if v == 0:
                continue
            ax.bar(x, v, bottom=bottom, width=0.55, color=c,
                   edgecolor=e, lw=1.5,
                   label=lb if x == 0 else None)
            if v < 30:
                ax.plot([x + 0.28, x + 0.36], [bottom + v / 2] * 2,
                        color=e, lw=1)
                ax.text(x + 0.39, bottom + v / 2, f"{v:.2f}", ha="left",
                        va="center", fontsize=9.5, color=INK)
            else:
                ax.text(x, bottom + v / 2, f"{v:.2f}", ha="center",
                        va="center", fontsize=10, color=INK)
            bottom += v
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["이원배치\n(경력을 모형에)", "일원배치\n(경력을 무시)"],
                       fontsize=10)
    ax.set_xlim(-0.6, 1.95)
    ax.set_ylim(0, 345)
    ax.set_ylabel("제곱합", fontsize=10.5, color=INK)
    ax.set_title("전체 제곱합 246.75 의 배분", fontsize=11.5, color=INK)
    ax.legend(fontsize=9, loc="upper right", frameon=True)
    ax.annotate("", xy=(0.95, 262), xytext=(0.05, 262),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.text(0.5, 270, "오차로 흘러든다", ha="center", fontsize=10, color=RED)
    clean(ax)

    # --- (c) 같은 SSA 인데 F 가 달라진다
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar([0, 1], [F_two, F_one], width=0.55,
                  color=[BLUE_F, ORANGE_F], edgecolor=[BLUE, ORANGE], lw=1.6)
    for b, f_, p_, c in zip(bars, [F_two, F_one], [p_two, p_one], [BLUE, ORANGE]):
        ax.text(b.get_x() + b.get_width() / 2, max(f_ + 0.8, 6.3),
                f"$F$ = {f_:.2f}\n$p$ = {p_:.4f}", ha="center",
                fontsize=10.5, color=c)
    ax.hlines(crit_two, -0.28, 0.28, color=RED, lw=2)
    ax.hlines(crit_one, 0.72, 1.28, color=RED, lw=2)
    ax.text(1.33, crit_one - 0.3, "5 % 임계값", color=RED, fontsize=9.5,
            va="top", ha="left")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["이원배치", "일원배치"], fontsize=10.5)
    ax.set_xlim(-0.55, 2.3)
    ax.set_ylim(0, 31)
    ax.set_ylabel("교육 방식의 $F$ 통계량", fontsize=10.5, color=INK)
    ax.set_title("SSA 는 42.25 로 같다", fontsize=11.5, color=INK)
    clean(ax)

    save(fig, "twoway_blocking.png")


if __name__ == "__main__":
    fig_blocking()
