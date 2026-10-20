r"""5.3절 「그림으로 보는 통계학」 — 어려울수록 올라가는 점수.

수능 성적표에 찍히는 표준점수가 z 점수다. 국어·수학·영어는

  표준점수 = 20 x (원점수 - 평균) / 표준편차 + 100

이고 탐구는 10z + 50 이다. 해마다 시험의 난이도가 다르니 원점수를
그대로 견줄 수 없고, 그래서 표준화해서 내보낸다.

그런데 해마다 신문에 나는 수가 있다 — 표준점수 최고점.

  2022 학년도  국어 149  수학 147
  2023 학년도  국어 134  수학 145
  2024 학년도  국어 150  수학 148
  2025 학년도  국어 139  수학 140

열여섯 점 차이가 난다. 어려운 해일수록 올라간다. 까닭은 식에 있다.
원점수의 천장이 100 으로 고정되어 있으므로 만점의 표준점수는

  20 x (100 - 평균) / 표준편차 + 100

이고, 시험이 어려워 평균이 내려가면 이 수가 올라간다.

표준화가 하는 일과 못 하는 일이 여기서 갈린다. 가운데는 잘 맞춰 준다 —
같은 실력이면 쉬운 해든 어려운 해든 거의 같은 표준점수가 나온다.
천장은 못 맞춰 준다. 쉬운 해에는 잘하는 학생들이 100 점에 몰려 서로
구별되지 않는다.

만드는 파일:

  ch05/distributions/img/suneung_standard_score.png

실행:  python3 scripts/make_ch05_suneung_standard_score.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/distributions/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

SPREAD = 18.0                    # 실력이 원점수로 퍼지는 폭
EASY, HARD = 72.0, 54.0          # 쉬운 해와 어려운 해의 바탕 평균
N, SEED = 600_000, 2024
REAL = ((2022, 149, 147), (2023, 134, 145),
        (2024, 150, 148), (2025, 139, 140))


def exam(base, a):
    """실력 a 를 원점수로 바꾼다. 천장 100, 바닥 0."""
    return np.clip(base + SPREAD * a, 0, 100)


def standard(raw):
    return 20 * (raw - raw.mean()) / raw.std(ddof=0) + 100


def top_score(base):
    """만점(원점수 100)의 표준점수와 만점자 비율."""
    a = np.linspace(-5, 5, 20001)
    w = norm.pdf(a)
    r = exam(base, a)
    m = np.average(r, weights=w)
    s = np.sqrt(np.average((r - m) ** 2, weights=w))
    return 20 * (100 - m) / s + 100, float(norm.sf((100 - base) / SPREAD))


def main():
    rng = np.random.default_rng(SEED)
    a = rng.normal(size=N)
    pct = norm.cdf(a) * 100
    res = {}
    for nm, base in (("쉬운 해", EASY), ("어려운 해", HARD)):
        raw = exam(base, a)
        z = standard(raw)
        res[nm] = (raw, z)
        full = raw >= 100
        print(f"  {nm}: 원점수 평균 {raw.mean():.1f}, 표준편차 "
              f"{raw.std(ddof=0):.1f},  만점자 {full.mean()*100:.2f}%,  "
              f"만점의 표준점수 {z[full].mean():.1f}")

    print(f"\n  같은 실력에서 두 해의 표준점수 차이")
    for q in (50, 70, 84, 93, 97.7):
        i = int(np.argmin(np.abs(pct - q)))
        print(f"    백분위 {q:>5.1f}   쉬운 해 {res['쉬운 해'][1][i]:6.1f}   "
              f"어려운 해 {res['어려운 해'][1][i]:6.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 가운데는 맞고 꼭대기는 안 맞는다 ──
    ax = axes[0]
    grid = np.linspace(1, 99.9, 600)
    for nm, col in (("쉬운 해", TEAL), ("어려운 해", RED)):
        raw, z = res[nm]
        o = np.argsort(pct)
        ax.plot(grid, np.interp(grid, pct[o], z[o]), lw=3.6, color=col,
                label=f"{nm} (원점수 평균 {raw.mean():.0f})")
    ax.axvline(84.1, color=MUTED, lw=1.6, ls=":")
    i84 = int(np.argmin(np.abs(pct - 84.1)))
    ax.annotate(f"백분위 ${84.1}$ 에서 "
                f"${res['쉬운 해'][1][i84]:.0f}$ 대 "
                f"${res['어려운 해'][1][i84]:.0f}$\n"
                f"— 가운데는 거의 맞춰 준다",
                xy=(84.1, res["쉬운 해"][1][i84]), xytext=(40, 150),
                fontsize=10.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.annotate("쉬운 해는 꼭대기가 눌린다\n— 잘하는 학생이 구별되지 않는다",
                xy=(98.6, np.interp(98.6, np.sort(pct),
                                    res["쉬운 해"][1][np.argsort(pct)])),
                xytext=(42, 119), fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("백분위 (실력의 자리)", fontsize=11.5)
    ax.set_ylabel("표준점수", fontsize=11.5)
    ax.set_title("가운데는 맞고 꼭대기는 안 맞는다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(1, 100)
    ax.set_ylim(55, 160)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 만점의 표준점수는 난이도가 정한다 ──
    ax = axes[1]
    bases = np.linspace(40, 86, 400)
    tops = np.array([top_score(b)[0] for b in bases])
    ax.plot(bases, tops, lw=3.8, color=INK)
    for yr, kor, mat in REAL:
        ax.axhline(kor, color=MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(41.0, 126.0, "가로 점선 — 실제 발표된 국어 표준점수 최고점\n"
            "$2023$ 년 $134$ · $2025$ 년 $139$ · "
            "$2022$ 년 $149$ · $2024$ 년 $150$",
            fontsize=9.8, color=GREY, va="center", ha="left")
    for b, col, nm in ((EASY, TEAL, "쉬운 해"), (HARD, RED, "어려운 해")):
        t, f = top_score(b)
        ax.plot([b], [t], marker="o", ms=13, color=col, zorder=5,
                mec="white", mew=1.7)
        print(f"  {nm} 모형: 바탕 평균 {b:.0f} -> 만점 표준점수 {t:.1f}, "
              f"만점자 {f*100:.2f}%")
    ax.annotate("시험이 어려울수록\n만점의 표준점수가 올라간다",
                xy=(60, top_score(60.0)[0]), xytext=(74, 152),
                fontsize=10.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlabel("원점수의 바탕 평균 — 왼쪽일수록 어렵다", fontsize=11.5)
    ax.set_ylabel("만점(원점수 $100$)의 표준점수", fontsize=11.5)
    ax.set_title("실제로 찍힌 수들이 이 선 위에 있다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(40, 86)
    ax.set_ylim(118, 162)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("어려울수록 올라가는 점수", fontsize=14.5)
    fig.text(0.5, -0.06,
             "실력이 정규분포를 따르고 원점수가 실력의 일차함수라고 둔 "
             f"모형이다(퍼짐 ${SPREAD:.0f}$ 점, 천장 $100$). 오른쪽 가로 "
             "점선은 실제 수능에서 발표된 국어 표준점수 최고점이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "suneung_standard_score.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
