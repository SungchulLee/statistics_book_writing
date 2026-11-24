r"""11.2절 독립성 검정 §4 의 그림 — 같은 자료의 도수를 k 배 하면 통계량과 p 값이 어떻게 되는가.

만드는 파일:
  ch11/test/img/multiply_counts.png   (가) X² 대 k, (나) p 값 대 k — 자료 네 벌

실행:  python3 scripts/make_ch11_multiply_counts.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
"""
import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
MUTED = "#90A4AE"
COLORS = ["#1565C0", "#E65100", "#33691E", "#6A1B9A"]

OUT = "docs/ch11/test/img/"


def gof(obs, exp):
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    return ((obs - exp) ** 2 / exp).sum(), len(obs) - 1, exp.min()


def indep(table):
    O = np.asarray(table, float)
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return ((O - E) ** 2 / E).sum(), (O.shape[0] - 1) * (O.shape[1] - 1), E.min()


DATA = [
    ("연령대 적합도 (df 2)", lambda k: gof(k * np.array([95, 115, 90]), k * np.array([100, 100, 100]))),
    ("성별×선호 2×2 (df 1)", lambda k: indep(k * np.array([[30, 20], [25, 25]]))),
    ("헤드라인 3×2 (df 2)", lambda k: indep(k * np.array([[14, 986], [8, 992], [12, 988]]))),
    ("타이타닉 2×2 (df 1)", lambda k: indep(k * np.array([[233, 81], [109, 468]]))),
]
KS = np.array([0.01, 0.1, 1, 10, 100, 1000])
GRID = np.logspace(-2, 3, 400)


def fmt_k(v, _):
    return f"{v:g}" if v >= 1 else f"1/{1 / v:g}"


def main():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for (name, f), c in zip(DATA, COLORS):
        res = np.array([f(k) for k in GRID])
        x, df, emin = res[:, 0], res[0, 1], res[:, 2]
        p = stats.chi2.sf(x, df)
        ok = emin >= 5
        for ax, y in ((a1, x), (a2, p)):
            ax.plot(GRID[~ok], y[~ok], color=c, lw=1.6, ls=":")
            ax.plot(GRID[ok], y[ok], color=c, lw=2.2, label=name)
            pts = np.array([f(k) for k in KS])
            yy = pts[:, 0] if ax is a1 else stats.chi2.sf(pts[:, 0], df)
            ax.plot(KS, yy, "o", color=c, ms=5)

    a1.axhline(3.8415, color=MUTED, lw=1, ls="--")
    a1.axhline(5.9915, color=MUTED, lw=1, ls="--")
    a1.text(900, 3.3, "임계값 3.84 (df 1)", color=INK, fontsize=9, va="top", ha="right")
    a1.text(900, 6.8, "임계값 5.99 (df 2)", color=INK, fontsize=9, va="bottom", ha="right")
    a1.set_xscale("log")
    a1.set_yscale("log")
    a1.set_ylim(5e-3, 1e6)
    a1.set_ylabel("카이제곱 통계량 X²")
    a1.set_title("(가) 통계량은 k 에 정비례한다 — 기울기 1 의 직선", fontsize=11, loc="left")

    a2.axhline(0.05, color=INK, lw=1, ls="--")
    a2.text(900, 0.06, "α = 0.05", color=INK, fontsize=9, ha="right")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_ylim(1e-12, 1.5)
    a2.set_ylabel("p 값")
    a2.set_title("(나) p 값은 벼랑처럼 떨어진다 (아래로 잘린 선은 계속 내려간다)", fontsize=11, loc="left")
    a2.legend(loc="lower left", fontsize=9, frameon=False)

    for ax in (a1, a2):
        ax.set_xlabel("도수에 곱한 배수 k  (k = 1 이 실제 자료)")
        ax.set_xticks(KS)
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_k))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
        # 로그 눈금의 마이너스가 mathtext 로 그려지면 한글 글꼴에서 두부가 된다
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("e+0", "e").replace("e-0", "e-")))
        ax.axvline(1, color=MUTED, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK)

    fig.text(0.5, -0.03, "점선 구간은 기대도수 최솟값이 5 아래라 카이제곱 근사가 서지 않는 곳이다",
             ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    path = OUT + "multiply_counts.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
