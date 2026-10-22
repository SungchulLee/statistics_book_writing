r"""8.4절 「그림으로 보는 통계학」 — 처음으로 t 를 쓴 자료.

$1905$ 년 쿠시니와 피블스가 캘러머주 정신병원에서 잰 「대조보다 늘어난 수면 시간」을
$1908$ 년 Student(고셋)가 새 분포의 첫 보기로 삼았다. 그 열 사람의 수가 이 그림의
자료다(두 약 모두 같은 사람에게서 쟀다).

  1 번 약   0.7  -1.6  -0.2  -1.2  -0.1   3.4   3.7   0.8   0.0   2.0
  2 번 약   1.9   0.8   1.1   0.1  -0.1   4.4   5.5   1.6   4.6   3.4

짝을 지으면 95% 구간이 $(0.70,\ 2.46)$ 으로 $0$ 을 담지 않고, 같은 수를 두 독립
표본으로 다루면 $(-0.20,\ 3.36)$ 으로 $0$ 을 담는다. 두 측정의 상관이 $r = 0.795$ 라
사람마다의 차이가 짝을 지으면 통째로 상쇄되기 때문이다.

만드는 파일:

  ch08/paired_sample_intervals/img/first_t_data.png

실행:  python3 scripts/make_ch08_first_t_data.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch08/paired_sample_intervals/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

# Student(1908) 가 실은 표 — 대조보다 늘어난 수면 시간(시간)
DRUG1 = np.array([0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0])
DRUG2 = np.array([1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4])


def main():
    d = DRUG2 - DRUG1
    n = len(d)
    m, s = d.mean(), d.std(ddof=1)
    se = s / np.sqrt(n)
    t1 = stats.t.ppf(0.975, n - 1)
    lo1, hi1 = m - t1 * se, m + t1 * se

    sp = np.sqrt(((n - 1) * DRUG1.var(ddof=1) + (n - 1) * DRUG2.var(ddof=1))
                 / (2 * n - 2))
    se2 = sp * np.sqrt(2 / n)
    t2 = stats.t.ppf(0.975, 2 * n - 2)
    lo2, hi2 = m - t2 * se2, m + t2 * se2
    r = float(np.corrcoef(DRUG1, DRUG2)[0, 1])

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print(f"차이 평균 {m:.3f}  표준편차 {s:.4f}  SE {se:.4f}")
    print(f"대응 95% 구간 ({lo1:.4f}, {hi1:.4f})   t = {m/se:.4f}  "
          f"p = {stats.ttest_rel(DRUG2, DRUG1).pvalue:.6f}")
    print(f"독립 95% 구간 ({lo2:.4f}, {hi2:.4f})   "
          f"p = {stats.ttest_ind(DRUG2, DRUG1).pvalue:.4f}")
    print(f"두 측정의 상관 r = {r:.4f}")
    print(f"구간 폭 — 대응 {hi1-lo1:.4f}  독립 {hi2-lo2:.4f}  "
          f"비 {(hi2-lo2)/(hi1-lo1):.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    # ── ① 열 사람의 짝 ──
    ax = axes[0]
    for i, (p1, p2) in enumerate(zip(DRUG1, DRUG2)):
        ax.plot([1, 2], [p1, p2], color=MUTED, lw=1.6, zorder=1)
    ax.plot(np.full(n, 1.0), DRUG1, "o", ms=10, color=AMBER, zorder=3,
            label="$1$ 번 약")
    ax.plot(np.full(n, 2.0), DRUG2, "o", ms=10, color=TEAL, zorder=3,
            label="$2$ 번 약")
    ax.plot([1, 2], [DRUG1.mean(), DRUG2.mean()], color=INK, lw=3.2, ls="--",
            marker="D", ms=9, zorder=4, label="평균")
    ax.axhline(0, color=MUTED, lw=1.2, ls=":")
    ax.set_xlim(0.72, 2.28)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["$1$ 번 약", "$2$ 번 약"], fontsize=12)
    ax.set_ylabel("대조보다 늘어난 수면 시간 (시간)", fontsize=11.5)
    ax.set_title(f"① 열 사람에게서 둘 다 쟀다  ($r = {r:.3f}$)",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 두 구간 ──
    ax = axes[1]
    for y, (lo, hi), col, lab in [(1.0, (lo1, hi1), TEAL, "짝으로 본 구간"),
                                  (0.4, (lo2, hi2), RED, "두 독립 표본으로 본 구간")]:
        ax.plot([lo, hi], [y, y], lw=7, color=col, solid_capstyle="round",
                alpha=0.75)
        ax.plot([m], [y], marker="D", ms=11, color=col)
        ax.text((lo + hi) / 2, y + 0.13, lab, ha="center", fontsize=11.5,
                color=col)
        ax.text(lo - 0.08, y, f"${lo:.2f}$", ha="right", va="center",
                fontsize=10.8, color=col)
        ax.text(hi + 0.08, y, f"${hi:.2f}$", ha="left", va="center",
                fontsize=10.8, color=col)
    ax.axvline(0, color=INK, lw=2.0, ls="--")
    ax.text(0.05, 1.42, "$0$ — 「차이 없음」", fontsize=11, color=INK)
    ax.annotate("같은 자료인데\n한쪽만 $0$ 을 담는다", xy=(-0.1, 0.4),
                xytext=(0.55, 0.08), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlim(-0.75, 3.9)
    ax.set_ylim(-0.05, 1.6)
    ax.set_yticks([])
    ax.set_xlabel("두 약의 평균 차이 (시간)", fontsize=11.5)
    ax.set_title("② 같은 열 사람, 두 가지 $95\\%$ 구간", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("짝을 풀면 구간이 두 배 넓어진다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "$1905$ 년 쿠시니와 피블스의 표를 $1908$ 년 Student 가 옮겨 실은 "
             "그 열 사람이다. 가운데 마름모는 두 경우 모두 같은 $1.58$ 이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "first_t_data.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
