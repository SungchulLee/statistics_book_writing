r"""16.5절 「그림으로 보는 통계학」 — 스물다섯 살의 프리드먼.

이 검정을 만든 사람은 밀턴 프리드먼이다. 1976 년에 노벨 경제학상을 받은
그 프리드먼이고, 이 논문을 쓸 때는 스물다섯 살이었다.

  Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of
  Normality Implicit in the Analysis of Variance." JASA 32, 675-701.

제목이 하는 일을 그대로 말한다. 분산분석에 깔린 정규성 가정을 피하려고
순위를 쓴다는 것이다. 그런데 이 검정의 진짜 힘은 정규성을 피하는 데
있지 않다. **블록 안에서만 순위를 매긴다**는 데 있다.

k = 4 처리, b = 12 블록, 처리 간격 0.45, 오차 sd 1 로 두고 블록 사이의
들쭉날쭉함 tau 를 키우며 검정력을 재면

    tau     프리드먼   반복측정 ANOVA   크러스컬-월리스   일원 ANOVA
    0.0      0.660        0.793           0.774          0.805
    1.0      0.673        0.806           0.350          0.382
    2.0      0.656        0.799           0.038          0.034
    4.0      0.653        0.789           0.001          0.001
    8.0      0.645        0.788           0.000          0.000

블록을 쓰는 두 검정은 꼼짝도 하지 않는다. 블록을 무시하는 두 검정은
0.05 아래로까지 내려앉는다 — 블록이 만든 들쭉날쭉함을 오차로 세느라
오차 추정이 부풀어서 아무것도 유의해지지 않는 것이다. 같은 자료다.

만드는 파일:

  ch16/multi_group_nonparametric/img/friedman_blocks.png

실행:  python3 scripts/make_ch16_friedman_blocks.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import f as fdist, friedmanchisquare, kruskal, f_oneway

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch16/multi_group_nonparametric/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

K, B, DELTA, SIG = 4, 12, 0.45, 1.0
TAUS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
REPS, ALPHA, SEED = 4000, 0.05, 1937
TAUSHOW = 4.0


def sample(rng, tau):
    """행이 블록, 열이 처리. 블록 효과가 통째로 한 행을 들었다 놓는다."""
    eff = DELTA * np.arange(K)
    return (eff[None, :] + rng.normal(0, tau, (B, 1))
            + rng.normal(0, SIG, (B, K)))


def rm_anova_p(Y):
    """반복측정 분산분석 — 블록 제곱합을 걷어 낸다."""
    g, rowm, colm = Y.mean(), Y.mean(1), Y.mean(0)
    ss_t = B * ((colm - g) ** 2).sum()
    ss_e = ((Y - rowm[:, None] - colm[None, :] + g) ** 2).sum()
    df_e = (B - 1) * (K - 1)
    return float(fdist.sf((ss_t / (K - 1)) / (ss_e / df_e), K - 1, df_e))


def power(rng, tau):
    hits = np.zeros(4)
    for _ in range(REPS):
        Y = sample(rng, tau)
        cols = [Y[:, j] for j in range(K)]
        hits += (friedmanchisquare(*cols).pvalue < ALPHA,
                 rm_anova_p(Y) < ALPHA,
                 kruskal(*cols).pvalue < ALPHA,
                 f_oneway(*cols).pvalue < ALPHA)
    return hits / REPS


def main():
    rng = np.random.default_rng(SEED)
    names = ("프리드먼", "반복측정 ANOVA", "크러스컬–월리스", "일원 ANOVA")
    rows = []
    print(f"{'tau':>6}" + "".join(f"{n:>16}" for n in names))
    for tau in TAUS:
        p = power(rng, tau)
        rows.append(p)
        print(f"{tau:>6.1f}" + "".join(f"{v:>16.3f}" for v in p))
    rows = np.array(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 블록이 커지면 ──
    ax = axes[0]
    Y = sample(np.random.default_rng(11), TAUSHOW)
    xs = np.arange(1, K + 1)
    for i in range(B):
        ax.plot(xs, Y[i], lw=1.6, color=MUTED, marker="o", ms=5, alpha=0.9,
                zorder=2)
    ax.plot(xs, Y.mean(0), lw=3.8, color=RED, marker="D", ms=10, zorder=4)
    ax.annotate("블록 안에서 보면 오른쪽이 위인 쪽이 많다\n"
                "— 프리드먼은 이것만 본다",
                xy=(2.5, Y[np.argmax(Y.mean(1))][1] + 0.5),
                xytext=(2.5, 11.7), fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("처리별로 모아 놓으면 완전히 겹친다\n"
                "— 블록을 무시하는 검정이 보는 것",
                xy=(1.08, Y[:, 0].min() + 0.3), xytext=(2.7, -10.8),
                fontsize=10.8, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"처리 {j}" for j in xs])
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(-12.6, 13.6)
    ax.set_ylabel("관측값", fontsize=11.5)
    ax.set_title(f"블록 사이의 들쭉날쭉함이 ${TAUSHOW:.0f}$ 일 때", fontsize=12.8,
                 color=INK, pad=8)
    ax.legend(handles=[
        Line2D([], [], color=MUTED, lw=1.8, marker="o", ms=5,
               label="블록 하나(사람 하나)"),
        Line2D([], [], color=RED, lw=3.2, marker="D", ms=8,
               label="처리별 평균")],
        fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 블록을 쓰는 쪽과 무시하는 쪽 ──
    ax = axes[1]
    style = ((0, TEAL, "-", "o", "프리드먼 — 블록 안에서 순위"),
             (1, BLUE, "-", "s", "반복측정 ANOVA — 블록을 걷어 냄"),
             (2, RED, "--", "D", "크러스컬–월리스 — 블록 무시"),
             (3, AMBER, "--", "^", "일원 ANOVA — 블록 무시"))
    for j, col, ls, mk, lab in style:
        ax.plot(TAUS, rows[:, j], lw=3.2, color=col, ls=ls, marker=mk, ms=7,
                label=lab)
    ax.axhline(ALPHA, color=GREY, lw=2.0, ls=":")
    ax.text(0.12, ALPHA + 0.022, "찍기 수준 $0.05$", fontsize=10.6,
            color=GREY, ha="left")
    ax.axvline(TAUSHOW, color=GREY, lw=1.6, ls=":")
    ax.annotate("같은 자료인데\n여기서 이만큼 갈린다",
                xy=(TAUSHOW, (rows[TAUS.index(TAUSHOW), 0]
                              + rows[TAUS.index(TAUSHOW), 2]) / 2),
                xytext=(6.3, 0.46), fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlabel("블록 사이의 들쭉날쭉함 $\\tau$ (오차 표준편차 $=1$)",
                  fontsize=11.5)
    ax.set_ylabel(f"처리 차이를 찾아낼 확률", fontsize=11.5)
    ax.set_title("블록을 쓰는 쪽은 꼼짝도 하지 않는다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(0, 0.92)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("스물다섯 살의 프리드먼", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"처리 ${K}$ 가지를 블록 ${B}$ 개에서 재었고, 처리 사이의 참 "
             f"간격은 ${DELTA}$, 오차 표준편차는 ${SIG:.0f}$ 이다. "
             f"네 검정 모두 같은 자료를 받는다. 각 점은 ${REPS//1000}{{,}}000$ "
             "번의 결과다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "friedman_blocks.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
