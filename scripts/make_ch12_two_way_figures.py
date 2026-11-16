r"""10장 이원배치 분산분석 개요 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch10/anova_two_way/img/twoway_blocking.png       둘째 요인을 모형에 넣으면 오차가 줄어든다
  ch10/anova_two_way/img/twoway_signal_noise.png   칸평균은 그대로 두고 칸 안의 산포만 키운다

실행:  python3 scripts/make_ch12_two_way_figures.py [blocking|signal_noise]   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib (signal_noise 는 pandas·statsmodels 와 망 연결도)
       — 문서 빌드에는 필요하지 않다.
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

OUT = "docs/ch10/anova_two_way/img/"
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


TG_URL = ("https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/"
          "1dcc2bf5f955cc1224a3e1307256e1fe86b68dae/csv/datasets/ToothGrowth.csv")


def fig_signal_noise():
    """10.2절 교호작용 효과 쪽 8단계 — 쪽의 코드와 같은 seed 42, 같은 잡음."""
    import pandas as pd
    from statsmodels.formula.api import ols
    from statsmodels.stats.anova import anova_lm

    df = pd.read_csv(TG_URL, usecols=["len", "supp", "dose"])
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, len(df))
    noise -= (pd.Series(noise).groupby([df["supp"], df["dose"]])
              .transform("mean").to_numpy())          # 칸마다 평균 0

    doses = [0.5, 1.0, 2.0]
    style = {"OJ": (ORANGE, -0.09), "VC": (BLUE, +0.09)}
    jit = np.random.default_rng(7).uniform(-0.05, 0.05, len(df))
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ylims = [None, None]
    for j, (ax, sigma) in enumerate(zip(axes, [0.0, 60.0])):
        d = df.assign(len=df["len"] + sigma * noise)
        t = anova_lm(ols("len ~ C(supp) * C(dose)", data=d).fit(), typ=2)
        x = d["dose"].map({v: i for i, v in enumerate(doses)}).to_numpy(float)
        for supp, (col, off) in style.items():
            m = (d["supp"] == supp).to_numpy()
            ax.scatter(x[m] + off + jit[m], d["len"][m], s=16, color=col,
                       alpha=0.45, edgecolor="none", zorder=2)
            cm = d[m].groupby("dose")["len"].mean().reindex(doses)
            ax.plot(np.arange(3) + off, cm.to_numpy(), color=col, lw=2.4,
                    marker="o", ms=6, zorder=4, label=supp)
        rows = [("보충제", "C(supp)"), ("용량", "C(dose)"),
                ("교호작용", "C(supp):C(dose)")]
        def plab(pv):
            return "$p < 10^{-4}$" if pv < 1e-4 else f"$p$ = {pv:.4f}"
        txt = "\n".join(f"{k}  $F$ = {t.loc[r, 'F']:.3f},  " + plab(t.loc[r, "PR(>F)"])
                        for k, r in rows)
        ax.text(0.03, 0.97, txt, transform=ax.transAxes, fontsize=9.5,
                color=RED, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MUTED, lw=0.8))
        ax.set_xticks(range(3))
        ax.set_xticklabels([f"{v:g} mg" for v in doses], fontsize=10)
        ax.set_xlim(-0.45, 2.45)
        ax.set_xlabel("용량", fontsize=10.5, color=INK)
        ax.set_title(f"잡음 표준편차 {sigma:g}", fontsize=11.5, color=INK)
        clean(ax)
        ylims[j] = ax.get_ylim()
        print(f"  sigma={sigma:g}: SSE={t.loc['Residual', 'sum_sq']:.3f}")

    lo, hi = 0, 37                                   # 왼쪽 그림의 세로 범위
    axes[0].set_ylim(lo, hi)
    axes[0].set_ylabel("치아 길이", fontsize=10.5, color=INK)
    axes[0].legend(fontsize=10, loc="lower right")
    axes[1].axhspan(lo, hi, color="#ECEFF1", zorder=0)
    axes[1].text(2.42, axes[1].get_ylim()[0] + 3, "음영 = 왼쪽 그림의 세로 범위 (0~37)",
                 fontsize=9, color=INK, ha="right", va="bottom")
    fig.suptitle("칸평균 여섯 개는 두 그림이 똑같다 — 칸 안의 산포만 다르다",
                 fontsize=12.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "twoway_signal_noise.png")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["blocking", "signal_noise"]
    if "blocking" in which:
        fig_blocking()
    if "signal_noise" in which:
        fig_signal_noise()
