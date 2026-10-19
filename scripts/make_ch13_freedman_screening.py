r"""13.6절 「그림으로 보는 통계학」 — 잡음만 넣어도 유의하게 나온다.

프리드먼(1983)이 보인 것을 그대로 다시 돌린다. 설명변수 50 개와 반응
변수를 **모두 독립인 잡음**으로 만들어 놓고,

  1 단계   50 개를 다 넣고 적합해 p < 0.25 인 것만 남긴다
  2 단계   남은 것만으로 다시 적합하고 R^2 와 F 검정을 본다

그러면 아무 관계도 없는 자료에서 R^2 이 0.3 을 넘고, 명목 5% 인 F 검정이
열 번에 아홉 번 넘게 기각한다. 같은 자료로 변수를 고르고 같은 자료로
검정했기 때문이고, 2 단계의 F 분포는 더 이상 F 가 아니다.

만드는 파일:

  ch13/model_selection/img/freedman_screening.png

실행:  python3 scripts/make_ch13_freedman_screening.py   (저장소 최상위에서)
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

OUT = "docs/ch13/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N, P, REPS, CUT = 100, 50, 4000, 0.25
SEED = 1983


def fit(X, y):
    """절편을 붙여 OLS. R^2 · 전체 F · 자유도 · t 값을 돌려준다."""
    A = np.column_stack([np.ones(len(y)), X])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ b
    rss = float(resid @ resid)
    k = A.shape[1] - 1
    df = len(y) - k - 1
    s2 = rss / df
    se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * s2)
    tss = float(((y - y.mean()) ** 2).sum())
    F = (tss - rss) / k / s2
    return 1 - rss / tss, F, k, df, b[1:] / se[1:]


def main():
    rng = np.random.default_rng(SEED)
    r2_all, r2_sel, p_sel, nkept = [], [], [], []
    for _ in range(REPS):
        X = rng.normal(size=(N, P))
        y = rng.normal(size=N)                       # X 와 아무 관계가 없다
        r2, _, _, df, t = fit(X, y)
        r2_all.append(r2)

        keep = np.where(2 * stats.t.sf(np.abs(t), df) < CUT)[0]
        if keep.size == 0:
            keep = np.array([int(np.argmax(np.abs(t)))])
        r22, F2, k2, df2, _ = fit(X[:, keep], y)
        r2_sel.append(r22)
        nkept.append(keep.size)
        p_sel.append(stats.f.sf(F2, k2, df2))

    r2_all = np.array(r2_all); r2_sel = np.array(r2_sel)
    p_sel = np.array(p_sel); nkept = np.array(nkept)
    rej = (p_sel < 0.05).mean()

    print(f"    1 단계 50 개 전부   R^2 평균 {r2_all.mean():.4f}  (이론 {P/(N-1):.4f})")
    print(f"    걸러진 개수        평균 {nkept.mean():.2f}  (기대 {P*CUT:.1f})")
    print(f"    2 단계 고른 것만    R^2 평균 {r2_sel.mean():.4f}")
    print(f"    2 단계 F 의 p 값   중앙값 {np.median(p_sel):.4f}")
    print(f"    p < 0.05 인 비율   {rej:.4f}")
    print(f"    p < 0.01 인 비율   {(p_sel < 0.01).mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    ax = axes[0]
    ax.hist(r2_sel, bins=60, range=(0, 0.75), color=AMBER, alpha=0.85)
    ax.axvline(r2_sel.mean(), color=RED, lw=2.4)
    ax.text(r2_sel.mean() + 0.012, ax.get_ylim()[1] * 0.92,
            f"평균 {r2_sel.mean():.3f}", fontsize=11, color=RED)
    ax.set_xlabel("고른 변수만으로 다시 적합한 $R^2$", fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title("참 관계가 하나도 없는 자료다", fontsize=12.8, color=INK, pad=8)
    ax.text(0.97, 0.72, f"걸러 낸 변수\n평균 {nkept.mean():.1f} 개",
            transform=ax.transAxes, ha="right", fontsize=10.6, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(p_sel, bins=50, range=(0, 1), color=TEAL, alpha=0.85,
            label="고르고 나서 잰 $p$ 값")
    ax.axhline(REPS / 50, color=MUTED, lw=2.4, ls="--",
               label="아무 일도 없었다면 — 균등")
    ax.axvline(0.05, color=RED, lw=2.0, ls=":")
    ax.annotate(f"$p < 0.05$ 가 {rej*100:.1f}%\n약속은 5% 였다",
                xy=(0.05, REPS / 50 * 2.2), xytext=(0.30, REPS / 50 * 5.0),
                fontsize=11.2, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax.set_xlabel("2 단계 전체 $F$ 검정의 $p$ 값", fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title("균등해야 할 것이 $0$ 에 쌓인다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("같은 자료로 고르고 같은 자료로 검정하면", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"$n = {N}$ 에 설명변수 {P} 개를 전부 난수로 만들고 반응변수도 난수로 "
             f"만들었다. $p < {CUT}$ 인 것만 남겨 다시 적합하기를 {REPS//1000} 천 번 "
             "되풀이한 것이다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "freedman_screening.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
