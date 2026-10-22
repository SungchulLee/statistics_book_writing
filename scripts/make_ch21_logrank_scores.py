r"""21.3절 「그림으로 보는 통계학」 — 로그순위라는 이름 속의 로그.

절 4 의 통계량에는 로그도 순위도 없다. $O_1 - E_1$ 과 그 분산뿐이다. 그런데 피토와
피토(1972)가 붙인 이름은 「logrank」다.

로그는 **점수** 안에 있다. 관측 하나에 다음 점수를 준다.

    a(t) = (사건이면 1, 절단이면 0)  -  Lambda_hat(t),   Lambda_hat = -ln S_hat

$\hat\Lambda$ 는 넬슨–알렌 누적위험, 곧 생존함수의 로그에 음수를 붙인 것이다. 이
점수를 한 집단에서 더하면 **$O_1 - E_1$ 과 정확히 같은 수**가 나온다. 쪽의 보기 1
자료(집단 A: 2, 4+, 6, 8+, 10 / 집단 B: 1, 3, 5+, 7, 9)에서

    O_A - E_A = 3 - 4.1556 = -1.1556,   집단 A 의 점수합 = -1.1556

이고 전체 점수합은 0 이다. 순위는 점수가 **시간의 순서에만** 의존한다는 데서 온다 —
시간축을 어떻게 늘이고 줄여도 점수가 변하지 않는다.

만드는 파일:

  ch21/nonparametric/img/logrank_scores.png

실행:  python3 scripts/make_ch21_logrank_scores.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/nonparametric/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# 쪽의 보기 1 — 집단 A(처리): 2, 4+, 6, 8+, 10 · 집단 B(대조): 1, 3, 5+, 7, 9
T = np.array([2, 4, 6, 8, 10, 1, 3, 5, 7, 9], float)
D = np.array([1, 0, 1, 0, 1, 1, 1, 0, 1, 1])       # 1 = 사건
G = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])       # 0 = A, 1 = B


def main():
    o = np.argsort(T)
    t, d, g = T[o], D[o], G[o]

    # 넬슨–알렌 누적위험 = -ln S_hat
    cum, run = {}, 0.0
    for tt in np.unique(t):
        dj = ((t == tt) & (d == 1)).sum()
        if dj:
            run += dj / (t >= tt).sum()
        cum[tt] = run
    lam = np.array([cum[tt] for tt in t])
    score = d - lam

    # O - E 와 분산 (쪽의 표)
    oa = ea = va = 0.0
    for tt in np.unique(t[d == 1]):
        at = t >= tt
        n, na = at.sum(), (at & (g == 0)).sum()
        dj = ((t == tt) & (d == 1)).sum()
        da = ((t == tt) & (d == 1) & (g == 0)).sum()
        oa += da
        ea += dj * na / n
        if n > 1:
            va += na * (n - na) * dj * (n - dj) / (n ** 2 * (n - 1))

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("  시간  사건  집단   -ln S(t)      점수")
    for ti, di, gi, li, si in zip(t, d, g, lam, score):
        print(f"{ti:>6.0f}{di:>6d}{'AB'[gi]:>6}{li:>11.4f}{si:>11.4f}")
    print(f"\nO_A - E_A = {oa:.4f} - {ea:.4f} = {oa-ea:.4f}")
    print(f"집단 A 의 점수합 = {score[g==0].sum():.4f}   "
          f"집단 B = {score[g==1].sum():.4f}   전체 = {score.sum():+.1e}")
    print(f"V_A = {va:.4f}   chi2 = {(oa-ea)**2/va:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── ① -ln S(t) 계단과 관측 ──
    ax = axes[0]
    xs = np.concatenate([[0], np.repeat(np.unique(t), 2), [11]])
    ys = [0.0]
    cur = 0.0
    for tt in np.unique(t):
        ys.append(cur)
        cur = cum[tt]
        ys.append(cur)
    ys.append(cur)
    ax.step(xs, ys, where="post", lw=2.8, color=TEAL)
    for ti, di, gi, li in zip(t, d, g, lam):
        col = AMBER if gi == 0 else BLUE
        ax.plot([ti], [li], marker="o" if di else "P", ms=11,
                color=col, zorder=5,
                markerfacecolor=col if di else "white",
                markeredgecolor=col, markeredgewidth=2)
    ax.plot([], [], "o", color=AMBER, ms=9, label="집단 A — 사건")
    ax.plot([], [], "P", color=AMBER, ms=10, markerfacecolor="white",
            markeredgewidth=2, label="집단 A — 절단")
    ax.plot([], [], "o", color=BLUE, ms=9, label="집단 B — 사건")
    ax.plot([], [], "P", color=BLUE, ms=10, markerfacecolor="white",
            markeredgewidth=2, label="집단 B — 절단")
    ax.set_xlim(0, 11)
    ax.set_xticks(range(0, 11))
    ax.set_xlabel("시간", fontsize=11.5)
    ax.set_ylabel("$-\\ln \\hat S(t)$  (넬슨–알렌 누적위험)", fontsize=11.5)
    ax.set_title("① 이름 속의 로그가 있는 자리", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.6, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 점수 막대 ──
    ax = axes[1]
    cols = [AMBER if gi == 0 else BLUE for gi in g]
    ax.bar(t, score, width=0.62, color=cols, edgecolor="white")
    ax.axhline(0, color=INK, lw=1.2)
    sa, sb = score[g == 0].sum(), score[g == 1].sum()
    ax.text(0.5, 1.18, f"파란 막대(집단 B)의 합  ${sb:+.4f}$",
            fontsize=11, color=BLUE)
    ax.text(0.5, -1.18, f"주황 막대(집단 A)의 합  ${sa:.4f}$  $= O_A - E_A$",
            fontsize=11, color=AMBER)
    ax.set_xlim(0, 11)
    ax.set_xticks(range(0, 11))
    ax.set_ylim(-1.45, 1.35)
    ax.set_xlabel("시간", fontsize=11.5)
    ax.set_ylabel("로그순위 점수  $d_i - (-\\ln \\hat S(t_i))$", fontsize=11.5)
    ax.set_title("② 더하면 $O - E$ 가 된다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("로그도 순위도 식에는 없고 점수에 있다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "쪽의 보기 $1$ 자료 그대로다. 열 사람의 점수를 모두 더하면 $0$ 이고, "
             f"집단 A 에서만 더하면 ${sa:.4f}$ 로 $O_A - E_A$ 와 같다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "logrank_scores.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
