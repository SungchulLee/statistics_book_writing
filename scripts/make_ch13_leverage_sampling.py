r"""13.2절 「AI로 보는 통계학」 — 사영행렬의 대각합이 표집 확률이 될 때.

이 쪽의 사영행렬 $\mathbf P = \mathbf X(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top$ 는
$\operatorname{tr}(\mathbf P) = k$ 를 만족한다. 대각원소 $h_{ii}$ (지렛값)가 모두
음이 아니고 합이 $k$ 이므로, $h_{ii}/k$ 는 그 자체로 **행 위의 확률분포**다.

행이 수억 개인 회귀에서는 $\mathbf X^\top\mathbf X$ 를 통째로 만들 수 없어 **행을 뽑아**
푼다. 그때 어느 행을 뽑느냐가 답을 가른다.

설계: 관측 $200{,}000$ 개, 계수 $5$ 개. 그 가운데 하나는 $200{,}000$ 행 중 $30$ 행만
$1$ 인 **드문 범주**다. 표본으로 $1000$ 행을 뽑아 가중최소제곱으로 계수를 추정하고,
전체자료로 푼 값과 견준다($400$ 번 되풀이).

이 스크립트가 재는 값:

  지렛값 합 = 5.0000 (= k)
  드문 범주 한 행의 지렛값 0.03334,  보통 행의 중앙값 1.686e-05
  드문 30 행이 지렛값 전체의 20.0% 를 가진다

  뽑는 법    상대오차 중앙값   드문 범주 계수의 중앙값   그 범주가 한 행도
                                 (전체자료 값 2.745)      없던 표본
  균등          0.7080                +0.000                87.5%
  지렛값 비례    0.0246                +2.749                 0.0%

만드는 파일:

  ch13/estimation_inference/img/leverage_sampling.png

실행:  python3 scripts/make_ch13_leverage_sampling.py   (저장소 최상위에서)
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

OUT = "docs/ch13/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 200_000        # 전체 행 수
S = 1000           # 뽑는 행 수
REPS = 400         # 되풀이
RARE = 30          # 드문 범주에 속한 행 수
BETA = np.array([1.0, 2.0, -1.5, 0.5, 3.0])
SEED = 3


def main():
    rng = np.random.default_rng(SEED)
    X = np.column_stack([np.ones(N), rng.normal(0, 1, (N, 3)), np.zeros(N)])
    rare = rng.choice(N, RARE, replace=False)
    X[rare, 4] = 1.0                       # 20 만 행 가운데 30 행만 1 이다
    y = X @ BETA + rng.normal(0, 1, N)

    XtXi = np.linalg.inv(X.T @ X)
    full = XtXi @ X.T @ y                  # 전체자료로 푼 값
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    k = X.shape[1]

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"지렛값 합 = {h.sum():.4f} (= k = {k})")
    print(f"드문 범주 한 행의 지렛값 {h[rare].mean():.5f},  "
          f"보통 행의 중앙값 {np.median(np.delete(h, rare)):.3e}")
    print(f"드문 {RARE} 행이 지렛값 전체의 {100 * h[rare].sum() / h.sum():.1f}% 를 가진다")
    print(f"전체자료 추정값 {np.round(full, 4)}")

    def draw(p, r):
        rr = np.random.default_rng([11, r])
        i = rr.choice(N, size=S, replace=True, p=p)
        w = 1.0 / (S * p[i])               # 치우쳐 뽑았으니 되돌리는 가중값
        b = np.linalg.lstsq(X[i] * np.sqrt(w)[:, None], y[i] * np.sqrt(w),
                            rcond=None)[0]
        return b, int((X[i, 4] > 0).sum())

    schemes = (("균등", np.full(N, 1.0 / N), MUTED),
               ("지렛값 비례", h / h.sum(), TEAL))
    res = {}
    print("\n  뽑는 법    상대오차 중앙값   드문 범주 계수의 중앙값   없던 표본")
    for nm, p, col in schemes:
        bs, cnt = [], []
        for r in range(REPS):
            b, c = draw(p, r)
            bs.append(b)
            cnt.append(c)
        bs, cnt = np.array(bs), np.array(cnt)
        err = np.linalg.norm(bs - full, axis=1) / np.linalg.norm(full)
        res[nm] = (bs, cnt, err, col)
        print(f"  {nm:10s} {np.median(err):10.4f} {np.median(bs[:, 4]):20.3f} "
              f"{100 * np.mean(cnt == 0):14.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 지렛값은 고르지 않다 ──
    ax = axes[0]
    vals = [np.median(np.delete(h, rare)), h[rare].mean()]
    cols = [MUTED, TEAL]
    ax.bar([0, 1], vals, width=0.5, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v * 1.6, f"${v:.5g}$".replace("e-05", "\\times 10^{-5}"),
                ha="center", fontsize=11.8, color=cols[i], fontweight="bold")
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"보통 행\n(${N - RARE:,}$ 개)".replace(",", "{,}"),
                        f"드문 범주의 행\n(${RARE}$ 개)"], fontsize=11.2)
    ax.set_ylim(5e-6, 2.0)
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0])
    ax.set_yticklabels([f"$10^{{{e}}}$" for e in (-5, -4, -3, -2, -1, 0)])
    ax.minorticks_off()
    ax.set_ylabel("지렛값 $h_{ii}$ (로그 눈금)", fontsize=11.5)
    ax.set_title(f"① 행 ${RARE}$ 개가 $\\operatorname{{tr}}(\\mathbf{{P}})$ 의 "
                 f"${100 * h[rare].sum() / h.sum():.0f}\\%$ 를 가진다",
                 fontsize=12.6, color=INK)
    ax.grid(alpha=0.22, axis="y", which="both")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 뽑는 법이 답을 가른다 ──
    ax = axes[1]
    bins = np.linspace(-0.6, 4.2, 70)
    for nm in ("균등", "지렛값 비례"):
        bs, cnt, err, col = res[nm]
        ax.hist(bs[:, 4], bins=bins, color=col, alpha=0.8,
                label=f"{nm} — 상대오차 중앙값 ${np.median(err):.4f}$")
    ax.axvline(full[4], color=RED, lw=2.4)
    ax.text(full[4] + 0.08, ax.get_ylim()[1] * 0.80,
            f"전체자료로 푼 값 ${full[4]:.3f}$", fontsize=11.2, color=RED,
            fontweight="bold")
    bs0, cnt0, _, _ = res["균등"]
    ax.annotate(f"표본의 ${100 * np.mean(cnt0 == 0):.1f}\\%$ 에는\n"
                f"그 범주가 한 행도 없다",
                xy=(0.0, REPS * 0.42), xytext=(0.72, REPS * 0.62),
                fontsize=11.2, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.set_xlabel("드문 범주 계수의 추정값", fontsize=11.5)
    ax.set_ylabel("되풀이 횟수", fontsize=11.5)
    ax.set_title(f"② ${S}$ 행만 뽑아 푼 ${REPS}$ 번", fontsize=12.6, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("어느 행을 뽑을 것인가 — 대각합이 답을 준다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"행 ${N // 1000}$ 천 개짜리 회귀에서 ${S}$ 행씩 뽑아 "
             f"가중최소제곱으로 푼 것을 ${REPS}$ 번 되풀이했다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "leverage_sampling.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
