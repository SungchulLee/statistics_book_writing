r"""18.3절 「그림으로 보는 통계학」 — 버그인 줄 알았다.

2004 년 무렵 캘리포니아 공과대학의 에마뉘엘 캉데스가 자기공명영상의
표준 시험 영상을 가지고 실험하고 있었다. 주파수 자료를 아주 조금만
떼어 쓰고 L1 노름을 최소로 하는 영상을 찾는 것이었다. 대충 비슷한
그림이 나오리라 기대했는데, 화소 하나까지 **정확히** 같은 영상이
나왔다. 그는 코드에 버그가 있다고 생각했다.

버그가 아니었다. 캉데스와 롬버그와 테런스 타오가 2006 년에, 그리고
데이비드 도노호가 같은 해에 따로, 이것을 정리로 적었다. 신호에 0 이
아닌 성분이 k 개뿐이면, 길이 n 을 다 재지 않고 무작위 측정 m 개만으로
**정확히** 복원할 수 있다. 그 m 이 n 보다 훨씬 작아도 된다.

이것이 압축센싱이다. 그리고 그 기계는 이 쪽이 그리는 L1 공의 모서리다.
모서리가 축 위에 있어서, 제약면과 처음 닿는 자리가 0 이 많은 점이 된다.

n = 120 에서 재면

            m=20   m=30   m=40   m=50   m=60   m=70
  k =  5    0.40   0.98   1.00   1.00   1.00   1.00
  k = 10    0.00   0.20   0.87   1.00   1.00   1.00
  k = 20    0.00   0.00   0.00   0.15   0.82   1.00

문턱이 날카롭다. 조금 모자라면 전혀 안 되고, 조금 넘으면 거의 언제나
된다. L2 로 같은 문제를 풀면 어떤 m 에서도 정확복원이 나오지 않는다.

만드는 파일:

  ch18/lasso/img/compressed_sensing.png

실행:  python3 scripts/make_ch18_compressed_sensing.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.optimize import linprog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch18/lasso/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, SEED = 120, 2006
KSHOW, MSHOW = 10, 40
MS = np.arange(10, 116, 5)
KS = ((5, TEAL, "$0$ 이 아닌 성분 $5$ 개"),
      (10, AMBER, "$10$ 개"),
      (20, RED, "$20$ 개"))
TRIALS, TOL = 60, 1e-6


def sci(v):
    """0.0000001 같은 수를 $10^{-7}$ 꼴로 적는다."""
    return f"10^{{{int(np.floor(np.log10(v)))}}}"


def basis_pursuit(A, b):
    """min ||z||_1  s.t.  Az = b  를 선형계획으로 푼다 (z = u - v, u,v >= 0)."""
    n = A.shape[1]
    r = linprog(np.ones(2 * n), A_eq=np.hstack([A, -A]), b_eq=b,
                bounds=(0, None), method="highs")
    return r.x[:n] - r.x[n:]


def draw(rng, k, m):
    A = rng.normal(size=(m, N)) / np.sqrt(m)
    x = np.zeros(N)
    x[rng.choice(N, k, replace=False)] = rng.normal(0, 1, k)
    return A, x


def main():
    rng = np.random.default_rng(SEED)

    A0, x0 = draw(rng, KSHOW, MSHOW)               # 왼쪽 그림에 쓸 한 판
    b0 = A0 @ x0
    z1 = basis_pursuit(A0, b0)
    z2 = np.linalg.pinv(A0) @ b0                   # 노름이 가장 작은 L2 해
    err1 = float(np.abs(z1 - x0).max())
    print(f"  n = {N}, 성분 {KSHOW} 개, 측정 {MSHOW} 개")
    print(f"    L1   최대오차 {err1:.2e}   "
          f"0 이 아닌 성분 {(np.abs(z1)>1e-8).sum()} 개")
    print(f"    L2   최대오차 {np.abs(z2-x0).max():.2e}   "
          f"0 이 아닌 성분 {(np.abs(z2)>1e-8).sum()} 개")

    print(f"\n  정확복원 비율 ({TRIALS} 번 중)")
    print(f"{'m':>6}" + "".join(f"{'k='+str(k):>9}" for k, _, _ in KS))
    rates = {k: [] for k, _, _ in KS}
    for m in MS:
        row = []
        for k, _, _ in KS:
            ok = 0
            for _ in range(TRIALS):
                Am, xm = draw(rng, k, m)
                ok += np.abs(basis_pursuit(Am, Am @ xm) - xm).max() < TOL
            rates[k].append(ok / TRIALS)
            row.append(ok / TRIALS)
        print(f"{m:>6}" + "".join(f"{v:>9.2f}" for v in row))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 대충이 아니라 정확히 ──
    ax = axes[0]
    idx = np.arange(N)
    ax.plot(idx, z2, lw=0, marker="o", ms=4.5, color=MUTED, zorder=2)
    nz = np.abs(x0) > 0
    ax.vlines(idx[nz], 0, x0[nz], lw=2.4, color=INK, zorder=3)
    ax.plot(idx[nz], x0[nz], lw=0, marker="o", ms=8, color=INK, zorder=4)
    sup = np.abs(z1) > 1e-8                        # L1 이 0 이 아니라고 한 자리
    ax.plot(idx[sup], z1[sup], lw=0, marker="o", ms=13, mfc="none", mec=RED,
            mew=2.2, zorder=5)
    ax.axhline(0, color=INK, lw=1.2)
    jmax = int(idx[nz][np.argmax(x0[nz])])
    ax.annotate(f"$L_1$ 은 ${sup.sum()}$ 자리를 맞히고 나머지 "
                f"${N - int(sup.sum())}$ 자리에 정확히 $0$ 을 넣었다\n"
                f"맞힌 값의 최대오차가 ${sci(err1)}$ 쯤이다",
                xy=(jmax, x0[jmax]), xytext=(jmax + 2, 2.26), fontsize=10.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"노름이 가장 작은 $L_2$ 해는\n${N}$ 자리 전부에 값이 있다",
                xy=(70, z2[70]), xytext=(84, -1.98), fontsize=10.8,
                color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.set_xlabel(f"성분의 자리 ($n = {N}$)", fontsize=11.5)
    ax.set_ylabel("성분의 값", fontsize=11.5)
    ax.set_title(f"측정 ${MSHOW}$ 개로 ${N}$ 자리를 되살린다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(-2.5, 2.7)
    ax.legend(handles=[
        Line2D([], [], lw=0, marker="o", ms=8, color=INK, label="참 신호"),
        Line2D([], [], lw=0, marker="o", ms=11, mfc="none", mec=RED, mew=2.0,
               label="$L_1$ 로 복원"),
        Line2D([], [], lw=0, marker="o", ms=6, color=MUTED,
               label="$L_2$ 로 복원")],
        fontsize=10.2, framealpha=0.93, loc="lower left", ncol=3)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 문턱이 날카롭다 ──
    ax = axes[1]
    for k, col, lab in KS:
        ax.plot(MS, rates[k], lw=3.4, color=col, marker="o", ms=6, label=lab)
    ax.axvline(N, color=INK, lw=2.0, ls="--")
    ax.text(N - 4, 0.52, "전부 재는 지점", fontsize=10.8, color=INK,
            ha="right", rotation=90, va="center")
    ax.annotate("조금 모자라면 전혀 안 되고\n조금 넘으면 거의 언제나 된다",
                xy=(62, 0.5), xytext=(88, 0.70), fontsize=10.8, color=INK,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel(f"무작위 측정 횟수 $m$ (성분은 ${N}$ 자리)", fontsize=11.5)
    ax.set_ylabel("정확히 복원할 확률", fontsize=11.5)
    ax.set_title("문턱이 날카롭다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(8, 123)
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="center left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("버그인 줄 알았다", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"「정확히」는 비유가 아니다. 최대오차가 ${sci(TOL)}$ 보다 "
             f"작을 때만 복원으로 세었다. 각 점은 ${TRIALS}$ 번의 결과다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "compressed_sensing.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
