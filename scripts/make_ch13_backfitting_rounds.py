r"""13.9절 「그림으로 보는 통계학」 — 돌아가며 푸는 셈법.

이 쪽은 가법모형을 **역적합**(backfitting)으로 적합한다고 적는다. 그 셈법이 실제로
무엇을 하는지, 그리고 언제 느려지는지를 재어 본다.

역적합은 한 번에 하나씩 푼다. $j$ 번째 항을 고칠 때 나머지를 빼고 남은
**부분잔차** $y - \sum_{k \ne j} f_k(x_k)$ 에 $x_j$ 만으로 맞춘 뒤, 다음 항으로 넘어간다.
한 바퀴를 다 돌면 처음으로 돌아가 다시 돈다.

항이 모두 일차식이면 수렴한 자리가 **최소제곱해와 정확히 같다.** 그러므로 「몇 바퀴
도는가」만 재면 된다. 관측 $400$ 개, 항 셋, 설명변수끼리의 상관을 $\rho$ 로 두었다.

이 스크립트가 재는 값(최대 차이가 $10^{-10}$ 아래로 내려갈 때까지):

   rho   통과 횟수   처음 세 바퀴의 최대 차이
  0.00        7      1.38e-01  4.16e-03  1.20e-04
  0.30       14      2.56e-01  8.79e-02  1.68e-02
  0.60       32      5.48e-01  2.64e-01  1.60e-01
  0.90      151      9.19e-01  8.01e-01  6.92e-01
  0.98      787      1.08e+00  1.05e+00  1.03e+00

곧 **설명변수가 서로 무관하면 일곱 바퀴면 끝나고, 상관이 $0.98$ 이면 칠백팔십일곱
바퀴가 든다.** 쪼개서 푸는 값이 거기서 나온다.

만드는 파일:

  ch13/splines_gams/img/backfitting_rounds.png

실행:  python3 scripts/make_ch13_backfitting_rounds.py   (저장소 최상위에서)
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
# 로그 축의 음수 지수는 mathtext 를 거치므로 unicode_minus 로 막지 못한다
plt.rcParams["mathtext.fontset"] = "dejavusans"

OUT = "docs/ch13/splines_gams/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 400
P = 3
BETA = np.array([2.0, -1.0, 0.5])
RHOS = (0.0, 0.30, 0.60, 0.90, 0.98)
COLS = (TEAL, BLUE, AMBER, RED, INK)
TOL = 1e-10
MAXIT = 2000
SEED = 5


def data(rho):
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=(N, P))
    c = rng.normal(size=N)
    X = np.sqrt(1 - rho) * z + np.sqrt(rho) * c[:, None]
    X = X - X.mean(0)
    y = X @ BETA + rng.normal(0, 1, N)
    return X, y - y.mean()


def backfit(X, y):
    """한 번에 한 항씩 부분잔차에 맞추기를 되풀이한다."""
    b = np.zeros(P)
    bols = np.linalg.lstsq(X, y, rcond=None)[0]
    hist = []
    for it in range(1, MAXIT + 1):
        for j in range(P):
            r = y - X @ b + X[:, j] * b[j]          # 부분잔차
            b[j] = (X[:, j] @ r) / (X[:, j] @ X[:, j])
        hist.append(float(np.abs(b - bols).max()))
        if hist[-1] < TOL:
            break
    return b, bols, it, hist


def main():
    res = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"관측 {N} 개, 항 {P} 개, 최대 차이가 {TOL:g} 아래로 갈 때까지")
    print("   rho   통과 횟수   처음 세 바퀴의 최대 차이")
    for rho in RHOS:
        X, y = data(rho)
        b, bols, it, hist = backfit(X, y)
        res[rho] = (it, hist, b, bols)
        head = "  ".join(f"{h:.2e}" for h in hist[:3])
        print(f"  {rho:4.2f} {it:9d}      {head}")
    X0, y0 = data(0.0)
    b0, bols0, _, _ = backfit(X0, y0)
    print(f"\n  rho = 0 에서 최소제곱해 {np.round(bols0, 6)}")
    print(f"              역적합의 답 {np.round(b0, 6)}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 바퀴 수 ──
    ax = axes[0]
    its = [res[r][0] for r in RHOS]
    ax.bar(np.arange(len(RHOS)), its, width=0.55, color=COLS, alpha=0.9)
    for i, v in enumerate(its):
        ax.text(i, v * 1.18, f"${v}$", ha="center", fontsize=12.4,
                color=COLS[i], fontweight="bold")
    ax.set_yscale("log")
    ax.set_yticks([1, 10, 100, 1000])
    ax.set_yticklabels(["$1$", "$10$", "$100$", "$1000$"])
    ax.minorticks_off()
    ax.set_xticks(np.arange(len(RHOS)))
    ax.set_xticklabels([f"$\\rho = {r:g}$" for r in RHOS], fontsize=11.4)
    ax.set_ylim(1, 3000)
    ax.set_ylabel("최소제곱해에 닿기까지의 바퀴 수 (로그 눈금)", fontsize=11.2)
    ax.set_title("① 설명변수가 닮을수록 오래 돈다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 수렴 곡선 ──
    ax = axes[1]
    for rho, col in zip(RHOS, COLS):
        hist = res[rho][1]
        ax.plot(np.arange(1, len(hist) + 1), hist, lw=2.4, color=col,
                label=f"$\\rho = {rho:g}$  (${res[rho][0]}$ 바퀴)")
    ax.axhline(TOL, color=MUTED, lw=1.4, ls="--")
    ax.text(1000, TOL * 2.6, "멈춤 문턱 $10^{-10}$", ha="right",
            fontsize=10.8, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["$1$", "$10$", "$100$", "$1000$"])
    ax.minorticks_off()
    ax.set_xlim(1, 1200)
    ax.set_ylim(1e-11, 3)
    ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    ax.set_yticklabels([f"$10^{{{e}}}$" for e in (-10, -8, -6, -4, -2, 0)])
    ax.set_xlabel("바퀴 수 (로그 눈금)", fontsize=11.5)
    ax.set_ylabel("최소제곱해와의 최대 차이 (로그 눈금)", fontsize=11.2)
    ax.set_title("② 한 바퀴마다 같은 비율로 줄어든다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, which="major")
    ax.legend(fontsize=9.8, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("큰 문제를 쪼개 돌아가며 푸는 값", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개에 일차항 ${P}$ 개짜리 가법모형을 역적합으로 "
             f"적합한 것이다. 항이 모두 일차식이므로 수렴한 자리는 "
             f"최소제곱해와 같고, 재는 것은 거기까지의 바퀴 수뿐이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "backfitting_rounds.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
