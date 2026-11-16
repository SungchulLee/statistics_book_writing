r"""13.3절 「그림으로 보는 통계학」 — 계수는 멀쩡한데 표준오차가 거짓말한다.

이분산이 OLS 의 계수를 **치우치게 하지는 않는다.** 깨지는 것은 그 계수의
표준오차이고, 표준오차가 틀리면 신뢰구간과 p 값이 통째로 틀린다.

  고전 표준오차  sigma^2 (X'X)^-1      — 분산이 모두 같다고 믿는다
  샌드위치       (X'X)^-1 X' diag(e^2) X (X'X)^-1  — 자료가 말하게 둔다

그림은 이분산의 세기를 0 에서 2 까지 올려 가며 명목 95% 구간의 실제
포함률을 센다. 고전 쪽은 내려앉고 HC0·HC3 은 버틴다. 기울기 추정값의
평균은 어느 쪽에서나 참값 그대로다 — 깨진 것은 자이지 눈금이 아니다.

Var(eps | x) = (1 + a|x|)^2 로 두었다. a = 0 이 등분산이다.

만드는 파일:

  ch13/assumptions/img/sandwich_coverage.png

실행:  python3 scripts/make_ch13_sandwich_coverage.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/assumptions/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N, REPS, SEED = 60, 20_000, 80
BETA = 2.0
STRENGTH = np.linspace(0, 2, 9)


def one_strength(a, rng):
    """세기 a 에서 기울기의 포함률 셋과 추정값 평균을 돌려준다."""
    x = rng.normal(0, 1, (REPS, N))
    sd = 1.0 + a * np.abs(x)
    y = 1.0 + BETA * x + rng.normal(0, 1, (REPS, N)) * sd

    xc = x - x.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    sxx = (xc ** 2).sum(axis=1)
    b = (xc * yc).sum(axis=1) / sxx                  # 기울기 추정값
    e = yc - b[:, None] * xc                         # 잔차
    h = 1.0 / N + xc ** 2 / sxx[:, None]             # 레버리지

    v_cls = (e ** 2).sum(axis=1) / (N - 2) / sxx                     # 고전
    v_hc0 = ((xc ** 2) * (e ** 2)).sum(axis=1) / sxx ** 2            # 샌드위치
    v_hc3 = ((xc ** 2) * (e / (1 - h)) ** 2).sum(axis=1) / sxx ** 2  # 작은 표본 보정

    tc = stats.t.ppf(0.975, N - 2)
    cov = [float(np.mean(np.abs(b - BETA) <= tc * np.sqrt(v)))
           for v in (v_cls, v_hc0, v_hc3)]
    return cov, float(b.mean())


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print("    세기 a   고전     HC0      HC3     기울기 평균")
    for a in STRENGTH:
        cov, bm = one_strength(a, rng)
        rows.append(cov)
        print(f"    {a:5.2f}   {cov[0]:.4f}   {cov[1]:.4f}   {cov[2]:.4f}   {bm:+.4f}")
    rows = np.array(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 자료가 어떻게 생겼나 ──
    ax = axes[0]
    r2 = np.random.default_rng(3)
    x = r2.normal(0, 1, 300)
    y = 1 + BETA * x + r2.normal(0, 1, 300) * (1 + 2 * np.abs(x))
    ax.scatter(x, y, s=18, color=MUTED, alpha=0.7, edgecolors="none")
    g = np.linspace(-3.2, 3.2, 50)
    ax.plot(g, 1 + BETA * g, lw=2.8, color=TEAL)
    ax.fill_between(g, 1 + BETA * g - 2 * (1 + 2 * np.abs(g)),
                    1 + BETA * g + 2 * (1 + 2 * np.abs(g)),
                    color=AMBER, alpha=0.16)
    ax.set_xlabel("$x$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("세기 $a = 2$ 에서의 자료 — 나팔 모양",
                 fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 포함률 ──
    ax = axes[1]
    ax.axhline(95, color=MUTED, lw=1.8, ls="--")
    ax.text(1.98, 95.3, "약속한 95%", ha="right", fontsize=10.4, color=MUTED)
    for j, (nm, col, mk) in enumerate([("고전 표준오차", RED, "o"),
                                       ("샌드위치 HC0", AMBER, "s"),
                                       ("샌드위치 HC3", TEAL, "D")]):
        ax.plot(STRENGTH, rows[:, j] * 100, marker=mk, ms=7, lw=2.6,
                color=col, label=nm)
    ax.annotate(f"{rows[-1,0]*100:.1f}%\n스무 번에 네 번 빗나간다",
                xy=(STRENGTH[-1], rows[-1, 0] * 100),
                xytext=(1.42, rows[-1, 0] * 100 + 2.6), fontsize=10.6, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("이분산의 세기 $a$   ($a = 0$ 이 등분산)", fontsize=11.5)
    ax.set_ylabel("명목 95% 구간의 실제 포함률 (%)", fontsize=11.5)
    ax.set_title("계수는 멀쩡한데 자가 틀렸다", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("이분산이 깨뜨리는 것은 추정값이 아니라 표준오차다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"$n = {N}$, {REPS//1000} 천 번. 기울기 추정값의 평균은 어느 세기에서나 "
             f"참값 ${BETA:g}$ 그대로다. 달라지는 것은 그 추정값 둘레에 그리는 "
             "구간의 폭뿐이다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "sandwich_coverage.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
