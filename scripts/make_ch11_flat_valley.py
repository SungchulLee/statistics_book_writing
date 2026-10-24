r"""11.2절 「AI로 보는 통계학」 — 바닥이 점이 아니라 골짜기일 때.

이 쪽의 $2$ 절은 「RSS 곡면은 전역 최솟값이 하나뿐인 그릇 모양이므로 어디서 출발해도
OLS 해로 수렴한다」고 적었다. 설계행렬의 열이 선형종속이면 그 말이 깨진다. **최솟값은
여전히 유일한데 그 값을 내는 점이 무한히 많다** — 바닥이 점이 아니라 골짜기다.

같은 열을 두 번 넣어 만들었다. $X = [\,\mathbf 1 \mid x \mid x\,]$, 관측 $100$ 개.
계수는 셋인데 계수행렬의 계수(rank)는 $2$ 다.

이 스크립트가 재는 값:

  출발점 (b1, b2)   도착점 (b1, b2)         b1 + b2     RSS
  ( 0,  0)          ( 1.528364,  1.528364)  3.056729   207.8212922306
  ( 5, -5)          ( 6.528364, -3.471636)  3.056729   207.8212922306
  (-3,  6)          (-2.971636,  6.028364)  3.056729   207.8212922306
  ( 8,  2)          ( 4.528364, -1.471636)  3.056729   207.8212922306

  네 해의 적합값 최대 차이 = 1.6e-14
  np.linalg.lstsq 가 주는 해 = (1.528364, 1.528364)  — 원점에서 출발한 것과 같다

곧 **자료는 $\hat\beta_1 + \hat\beta_2$ 만 정하고, 그 둘을 어떻게 나눌지는 알고리듬이
정한다.** 경사하강은 출발점에서 골짜기로 내린 수선의 발에 멈추고, `lstsq` 는 노름이
가장 작은 점을 준다.

만드는 파일:

  ch11/estimation_inference/img/flat_valley.png

실행:  python3 scripts/make_ch13_flat_valley.py   (저장소 최상위에서)
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

OUT = "docs/ch11/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 100
ETA = 2e-4
ITERS = 200_000
STARTS = ((0.0, 0.0), (5.0, -5.0), (-3.0, 6.0), (8.0, 2.0))
COLS = (TEAL, RED, AMBER, BLUE)
SEED = 4


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, 10, N)
    x = x - x.mean()
    y = 2 + 3 * x + rng.normal(0, 1.5, N)
    X = np.column_stack([np.ones(N), x, x])      # 같은 열을 두 번 넣었다
    rank = int(np.linalg.matrix_rank(X))

    b_ls = np.linalg.lstsq(X, y, rcond=None)[0]

    def gd(w0):
        w = np.array(w0, float)
        for _ in range(ITERS):
            w -= ETA * (X.T @ (X @ w - y))
        return w

    ends = [gd((0.0, s[0], s[1])) for s in STARTS]
    rss = [float(((y - X @ w) ** 2).sum()) for w in ends]
    maxdiff = max(float(np.abs(X @ w - X @ b_ls).max()) for w in ends)

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"설계행렬 {X.shape} 의 계수(rank) = {rank}  — 열 셋에 rank 가 둘이다")
    print("  출발점 (b1, b2)    도착점 (b1, b2)           b1 + b2            RSS")
    for s, w, r in zip(STARTS, ends, rss):
        print(f"  ({s[0]:5.1f}, {s[1]:5.1f})   "
              f"({w[1]:9.6f}, {w[2]:9.6f})   {w[1] + w[2]:.6f}  {r:.10f}")
    print(f"\n  네 해의 적합값 최대 차이 = {maxdiff:.1e}")
    print(f"  np.linalg.lstsq 가 주는 해 = "
          f"({b_ls[1]:.6f}, {b_ls[2]:.6f})   노름 "
          f"{np.linalg.norm(b_ls[1:]):.6f}")
    print(f"  (골짜기 위의 점 가운데 노름이 가장 작은 점이다)")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    tot = ends[0][1] + ends[0][2]

    # ── ① 골짜기 ──
    ax = axes[0]
    g = np.linspace(-6, 10, 220)
    B1, B2 = np.meshgrid(g, g)
    b0 = b_ls[0]
    R = np.empty_like(B1)
    for i in range(B1.shape[0]):
        for j in range(B1.shape[1]):
            e = y - (b0 + (B1[i, j] + B2[i, j]) * x)
            R[i, j] = (e ** 2).sum()
    cs = ax.contour(B1, B2, R, levels=np.logspace(np.log10(rss[0] + 1), 5.6, 14),
                    colors=MUTED, linewidths=1.0)
    ax.plot(g, tot - g, lw=3.0, color=INK,
            label=f"바닥 — $\\hat\\beta_1 + \\hat\\beta_2 = {tot:.4f}$ 인 직선")
    for s, w, col in zip(STARTS, ends, COLS):
        ax.annotate("", xy=(w[1], w[2]), xytext=s,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0))
        ax.plot([s[0]], [s[1]], "o", ms=9, color=col, mec="white", mew=1.3,
                zorder=5)
        ax.plot([w[1]], [w[2]], "X", ms=14, color=col, mec="white", mew=1.4,
                zorder=6)
    ax.plot([b_ls[1]], [b_ls[2]], "o", ms=18, mfc="none", mec=INK, mew=2.0,
            zorder=7)
    ax.annotate("lstsq 가 주는 점\n(노름이 가장 작다)",
                xy=(b_ls[1], b_ls[2]), xytext=(-5.4, -4.2), fontsize=10.8,
                color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlim(-6, 10)
    ax.set_ylim(-6, 10)
    ax.set_aspect("equal")
    ax.set_xlabel("$\\hat\\beta_1$", fontsize=12)
    ax.set_ylabel("$\\hat\\beta_2$", fontsize=12)
    ax.set_title("① 바닥이 점이 아니라 직선이다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 계수는 다르고 적합은 같다 ──
    ax = axes[1]
    xs = np.arange(len(STARTS))
    w = 0.36
    ax.bar(xs - w / 2, [e[1] for e in ends], width=w * 0.92, color=INK,
           alpha=0.85, label="$\\hat\\beta_1$")
    ax.bar(xs + w / 2, [e[2] for e in ends], width=w * 0.92, color=MUTED,
           alpha=0.95, label="$\\hat\\beta_2$")
    for i, e in enumerate(ends):
        for off, v in ((-w / 2, e[1]), (w / 2, e[2])):
            ax.text(i + off, v + (0.35 if v > 0 else -0.75), f"${v:.2f}$",
                    ha="center", fontsize=10.6, color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"출발점\n$({s[0]:.0f},\\ {s[1]:.0f})$" for s in STARTS],
                       fontsize=10.8)
    ax.set_ylim(-6.2, 9.4)
    ax.set_ylabel("도착한 계수", fontsize=11.5)
    ax.text(-0.46, 8.4,
            f"네 해의 RSS 가 모두 ${rss[0]:.4f}$ 이고\n"
            f"적합값의 최대 차이가 ${maxdiff / 1e-14:.1f} \\times 10^{{-14}}$ 다",
            fontsize=11.4, color=TEAL, fontweight="bold")
    ax.set_title("② 같은 적합, 네 가지 계수", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.6, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("자료가 정하지 않는 것을 알고리듬이 정한다", fontsize=14.5)
    fig.text(0.5, -0.04,
             f"관측 ${N}$ 개에 같은 열을 두 번 넣어 만든 설계행렬이다"
             f"(열 셋, rank ${rank}$). 경사하강은 학습률 ${ETA:g}$ 로 "
             f"${ITERS // 1000}$ 천 걸음 돌렸다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "flat_valley.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
