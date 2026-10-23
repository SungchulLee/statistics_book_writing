r"""13.5절 「그림으로 보는 통계학」 — 순서가 없는 종속.

이 쪽이 가르치는 검정은 모두 **순서**를 쓴다. 더빈–왓슨도 브로이슈–고드프리도
「이전 관측」이 있어야 한다. 그런데 $1$ 절이 독립성 위배의 하나로 든 **군집 자료**
(학교 안의 학생, 병원 안의 환자)에는 이전이 없다. 그러면 얼마나 해로운가.

학교 $50$ 개에 학생이 $30$ 명씩, 모두 $1500$ 명이다. 처치는 **학교 단위**로 배정하고
참 효과는 $0$ 으로 둔다. 급내상관 $\rho$ 를 바꿔 가며 $t$ 검정이 $5\%$ 수준에서
얼마나 기각하는지 센다($2000$ 번).

이 스크립트가 재는 값:

  ICC    OLS 기각률   군집 기각률   SE 비(군집/OLS)   멀턴 sqrt(1+(m-1)rho)
  0.00      0.042        0.049           0.99              1.00
  0.02      0.117        0.051           1.24              1.26
  0.05      0.218        0.054           1.55              1.57
  0.10      0.322        0.052           1.95              1.97
  0.20      0.464        0.051           2.58              2.61

곧 급내상관이 $0.10$ 이면 **스무 번에 한 번 틀리기로 한 검정이 세 번에 한 번 틀린다.**
표준오차의 비는 멀턴의 어림 $\sqrt{1 + (m-1)\rho}$ 와 소수 둘째 자리까지 맞는다.

만드는 파일:

  ch13/assumptions/img/clustered_se.png

실행:  python3 scripts/make_ch13_clustered_se.py   (저장소 최상위에서)
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

OUT = "docs/ch13/assumptions/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

G = 50             # 학교 수
M = 30             # 학교마다의 학생 수
REPS = 2000
RHOS = (0.0, 0.02, 0.05, 0.10, 0.20)
ALPHA = 0.05
SEED = 5


def main():
    n = G * M
    g = np.repeat(np.arange(G), M)
    T = np.repeat((np.arange(G) < G // 2).astype(float), M)   # 처치는 학교 단위
    X = np.column_stack([np.ones(n), T])
    XtXi = np.linalg.inv(X.T @ X)
    tc_ols = stats.t.ppf(1 - ALPHA / 2, n - 2)
    tc_cl = stats.t.ppf(1 - ALPHA / 2, G - 1)                 # 자유도는 군집 수

    def one(rho, r):
        rr = np.random.default_rng([SEED, r])
        y = (np.sqrt(rho) * np.repeat(rr.normal(size=G), M)
             + np.sqrt(1 - rho) * rr.normal(size=n))           # 참 효과 0
        b = XtXi @ X.T @ y
        e = y - X @ b
        se_ols = np.sqrt((e ** 2).sum() / (n - 2) * XtXi[1, 1])
        meat = np.zeros((2, 2))                                # 군집마다 묶는다
        for gg in range(G):
            u = X[g == gg].T @ e[g == gg]
            meat += np.outer(u, u)
        c = G / (G - 1) * (n - 1) / (n - 2)
        se_cl = np.sqrt((XtXi @ meat @ XtXi * c)[1, 1])
        return abs(b[1] / se_ols), abs(b[1] / se_cl), se_ols, se_cl

    rows = []
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"학교 {G} 개 x 학생 {M} 명 = {n}, 처치는 학교 단위, 참 효과 0, "
          f"{REPS} 번")
    print("  ICC   OLS 기각률   군집 기각률   SE 비(군집/OLS)   멀턴 어림")
    for rho in RHOS:
        o = np.array([one(rho, r) for r in range(REPS)])
        rej_o = float(np.mean(o[:, 0] > tc_ols))
        rej_c = float(np.mean(o[:, 1] > tc_cl))
        ratio = float(np.mean(o[:, 3] / o[:, 2]))
        moul = float(np.sqrt(1 + (M - 1) * rho))
        rows.append((rho, rej_o, rej_c, ratio, moul))
        print(f" {rho:5.2f} {rej_o:11.3f} {rej_c:12.3f} {ratio:16.2f} "
              f"{moul:12.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)
    xs = [r[0] for r in rows]

    # ── ① 기각률 ──
    ax = axes[0]
    ax.plot(xs, [r[1] for r in rows], "o-", ms=10, lw=2.6, color=RED,
            label="보통 표준오차로 잰 $t$ 검정")
    ax.plot(xs, [r[2] for r in rows], "s-", ms=9, lw=2.6, color=TEAL,
            label="군집 표준오차로 잰 $t$ 검정")
    ax.axhline(ALPHA, color=INK, lw=1.8, ls="--")
    ax.text(0.125, ALPHA + 0.022, f"약속한 수준 ${ALPHA}$", fontsize=11,
            color=INK)
    r10 = [r for r in rows if r[0] == 0.10][0]
    ax.annotate(f"급내상관 ${r10[0]}$ 에서 ${r10[1]:.3f}$\n"
                f"— 세 번에 한 번 틀린다",
                xy=(0.10, r10[1]), xytext=(0.021, 0.40), fontsize=11.2,
                color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("급내상관 $\\rho$ (같은 학교 학생끼리의 닮음)", fontsize=11.5)
    ax.set_ylabel(f"참 효과가 $0$ 인데 기각한 비율", fontsize=11.5)
    ax.set_ylim(0, 0.55)
    ax.set_title("① 약속은 $5\\%$, 실제는 $46\\%$ 까지", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 멀턴의 어림 ──
    ax = axes[1]
    gr = np.linspace(0, 0.21, 200)
    ax.plot(gr, np.sqrt(1 + (M - 1) * gr), lw=2.6, color=INK,
            label=f"멀턴의 어림 $\\sqrt{{1 + ({M}-1)\\rho}}$")
    ax.plot(xs, [r[3] for r in rows], "o", ms=12, color=AMBER, mec="white",
            mew=1.5, label="모의실험으로 잰 표준오차의 비")
    for x, _, _, ratio, moul in rows:
        if x > 0:
            ax.text(x, ratio - 0.14, f"${ratio:.2f}$", ha="center",
                    fontsize=10.8, color=AMBER, fontweight="bold")
    ax.set_xlabel("급내상관 $\\rho$", fontsize=11.5)
    ax.set_ylabel("군집 표준오차 $\\div$ 보통 표준오차", fontsize=11.5)
    ax.set_ylim(0.85, 3.0)
    ax.set_title("② 얼마나 틀렸는지는 식으로 적힌다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("순서가 없는 종속 — 더빈–왓슨이 잡지 못하는 것", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"학교 ${G}$ 개에 학생 ${M}$ 명씩, 처치를 학교 단위로 배정하고 "
             f"참 효과를 ${0}$ 으로 둔 자료를 ${REPS}$ 번 만들었다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "clustered_se.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
