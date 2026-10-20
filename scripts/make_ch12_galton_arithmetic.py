r"""12.2절 「그림으로 보는 통계학」 — 상관 0.46 이 할 수 있는 일.

이 쪽이 쓰는 말 셋의 출처가 한 연구 계획이다.

  회귀   골턴 「유전되는 키의 평범으로의 회귀」(1886)
  상관   골턴이 같은 자료에서 꺼내고 피어슨이 식으로 다듬은 것
  분산   피셔 「멘델 유전을 전제한 혈연 간 상관」(1918) 에서 처음 쓰인 말

셋 다 사람의 형질이 얼마나 유전되는지를 재려고 만든 도구이고,
그 재는 일의 목적이 우생학이었다.

여기서 재는 것은 그 계획의 산술이다. 골턴 자신의 자료에서 중간부모와
자식의 키 상관이 0.46 쯤이다. 그 값이 무엇을 할 수 있는가.

  설명되는 몫 R^2 = 0.46^2 = 0.212

부모의 키로 자식의 키를 설명하면 다섯 가운데 하나다.

위쪽 10% 의 부모만 골라도

  자식 평균의 이동   0.46 x 1.755 = 0.807 표준편차
  그 자식들 가운데 전체 중앙값 아래인 비율   0.187

열에 둘 가까이가 평범한 사람들의 절반 아래에 떨어진다.

만드는 파일:

  ch12/correlation/img/galton_arithmetic.png

실행:  python3 scripts/make_ch12_galton_arithmetic.py   (저장소 최상위에서)
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

OUT = "docs/ch12/correlation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

R_GALTON = 0.46
TOP = 0.10                      # 위쪽 10% 의 부모
NSHOW, SEED = 2500, 1886


def selected(r, top=TOP):
    """위쪽 top 비율의 부모를 골랐을 때 자식 분포의 평균·표준편차·중앙값 아래 비율."""
    a = norm.isf(top)
    lam = norm.pdf(a) / top                       # 잘린 정규의 평균
    v = 1 + a * lam - lam ** 2                    # 잘린 정규의 분산
    mu = r * lam
    sd = np.sqrt(r ** 2 * v + (1 - r ** 2))
    return mu, sd, float(norm.cdf(-mu / sd))


def main():
    mu, sd, below = selected(R_GALTON)
    print(f"  상관 {R_GALTON},  위쪽 {TOP:.0%} 의 부모를 고를 때")
    print(f"    설명되는 몫 R^2 = {R_GALTON**2:.3f}")
    print(f"    자식 평균의 이동 {mu:.3f} 표준편차   (자식 표준편차 {sd:.3f})")
    print(f"    그 자식들 가운데 전체 중앙값 아래인 비율 {below:.3f}")
    print(f"\n{'상관 r':>8}{'평균 이동':>12}{'중앙값 아래':>14}{'R^2':>10}")
    for r in (0.2, 0.3, 0.46, 0.6, 0.8, 0.95):
        m, s, b = selected(r)
        print(f"{r:>8.2f}{m:>12.3f}{b:>14.3f}{r*r:>10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 상관 0.46 이 그리는 그림 ──
    ax = axes[0]
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=NSHOW)
    y = R_GALTON * x + np.sqrt(1 - R_GALTON ** 2) * rng.normal(size=NSHOW)
    a = norm.isf(TOP)
    hit = x > a
    ax.axvspan(a, 4.2, color=AMBER, alpha=0.13, zorder=1)
    ax.plot(x[~hit], y[~hit], lw=0, marker="o", ms=3.0, color=MUTED,
            alpha=0.7, zorder=2)
    ax.plot(x[hit], y[hit], lw=0, marker="o", ms=4.2, color=AMBER, zorder=3)
    gx = np.array([-4.0, 4.0])
    ax.plot(gx, R_GALTON * gx, lw=3.4, color=INK, zorder=4)
    ax.plot(gx, gx, lw=2.0, color=GREY, ls=":", zorder=4)
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.annotate(f"기울기 ${R_GALTON}$ — 자식은 부모만큼\n극단으로 가지 않는다",
                xy=(2.4, R_GALTON * 2.4), xytext=(-1.5, 2.95), fontsize=10.6,
                color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.text(3.95, 3.55, "$y = x$", fontsize=10.6, color=GREY, ha="right")
    ax.text(a + 0.12, -3.5, f"위쪽 ${TOP*100:.0f}\\%$ 의 부모", fontsize=10.6,
            color=AMBER)
    ax.set_xlabel("중간부모의 키 (표준편차 단위)", fontsize=11.5)
    ax.set_ylabel("자식의 키 (표준편차 단위)", fontsize=11.5)
    ax.set_title(f"상관 ${R_GALTON}$ — 설명되는 몫이 ${R_GALTON**2:.2f}$",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-4.0, 4.2); ax.set_ylim(-4.0, 4.0)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 골라도 이만큼이다 ──
    ax = axes[1]
    rs = np.linspace(0.02, 0.99, 400)
    vals = np.array([selected(r) for r in rs])
    ax.plot(rs, vals[:, 2], lw=3.8, color=RED,
            label=f"그 자식들 가운데 전체 중앙값 아래인 비율")
    ax.plot(rs, vals[:, 0] / 3.0, lw=3.4, color=TEAL, ls="--",
            label="자식 평균의 이동 $\\div\\,3$ (표준편차 단위)")
    ax.axvline(R_GALTON, color=INK, lw=2.0, ls=":")
    ax.plot([R_GALTON], [below], marker="o", ms=13, color=RED, zorder=5,
            mec="white", mew=1.7)
    ax.plot([R_GALTON], [mu / 3.0], marker="D", ms=11, color=TEAL, zorder=5,
            mec="white", mew=1.6)
    ax.annotate(f"골턴의 자료에서 ${R_GALTON}$\n"
                f"고른 부모의 자식 가운데 ${below*100:.0f}\\%$ 가\n"
                f"평범한 사람들의 절반 아래에 있다",
                xy=(R_GALTON, below), xytext=(0.785, 0.245), fontsize=10.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"평균은 ${mu:.2f}$ 표준편차만 옮겨 간다",
                xy=(R_GALTON, mu / 3.0), xytext=(0.99, 0.045), fontsize=10.6,
                color=TEAL, ha="right",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("부모와 자식의 상관 $r$", fontsize=11.5)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_title(f"위쪽 ${TOP*100:.0f}\\%$ 를 골라도", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 0.56)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"상관 ${R_GALTON}$ 이 할 수 있는 일", fontsize=14.5)
    fig.text(0.5, -0.06,
             "모두 표준편차 단위이며 이변량정규를 가정하고 식으로 구한 "
             f"값이다. 왼쪽 점구름은 상관 ${R_GALTON}$ 짜리 자료 "
             f"${NSHOW:,}$ 개다.".replace(",", "{,}"),
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "galton_arithmetic.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
