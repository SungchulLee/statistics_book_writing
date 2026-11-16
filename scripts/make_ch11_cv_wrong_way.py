r"""13.6절 「그림으로 보는 통계학」 — 폴드 밖에서 고르면 교차검증이 거짓말한다.

교차검증은 「자료를 나눠서 고르는 데 쓴 자료와 채점하는 데 쓰는 자료를
떼어 놓는다」는 생각이다. 그런데 **변수를 고르는 일**을 나누기 **전에**
해 버리면 그 생각이 통째로 무너진다.

  잘못된 길   전체 자료로 y 와 상관이 큰 변수 열 개를 먼저 고른 뒤
              그 열 개로 교차검증한다
  옳은 길     폴드마다 훈련 쪽만 보고 다시 고른다

관계가 **하나도 없는** 자료에서 재면, 잘못된 길은 설명력 R^2 = 0.43 을
내놓고 옳은 길은 음수를 내놓는다(평균만 쓰느니만 못하다는 뜻이고 그것이
맞는 답이다).

만드는 파일:

  ch13/model_selection/img/cv_wrong_way.png

실행:  python3 scripts/make_ch13_cv_wrong_way.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

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

N, K, TOP, SEED = 50, 5, 10, 1974
PS = (20, 100, 500, 2000)
REPS_MAIN, REPS_SWEEP = 400, 150
P_MAIN = 1000


def cv_r2(X, y, preselect, rng):
    """K 겹 교차검증의 R^2. preselect 면 전체 자료로 변수를 먼저 고른다."""
    n = len(y)
    idx = rng.permutation(n)
    folds = np.array_split(idx, K)
    keep_all = None
    if preselect:
        c = np.abs((X - X.mean(0)).T @ (y - y.mean()))
        keep_all = np.argsort(c)[::-1][:TOP]
    sse = 0.0
    for f in folds:
        tr = np.setdiff1d(idx, f)
        if preselect:
            keep = keep_all
        else:
            c = np.abs((X[tr] - X[tr].mean(0)).T @ (y[tr] - y[tr].mean()))
            keep = np.argsort(c)[::-1][:TOP]
        A = np.column_stack([np.ones(len(tr)), X[np.ix_(tr, keep)]])
        b = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        At = np.column_stack([np.ones(len(f)), X[np.ix_(f, keep)]])
        sse += float(((y[f] - At @ b) ** 2).sum())
    return 1.0 - sse / float(((y - y.mean()) ** 2).sum())


def main():
    rng = np.random.default_rng(SEED)

    wrong, right = [], []
    for _ in range(REPS_MAIN):
        X = rng.normal(size=(N, P_MAIN))
        y = rng.normal(size=N)                  # X 와 아무 관계가 없다
        wrong.append(cv_r2(X, y, True, rng))
        right.append(cv_r2(X, y, False, rng))
    wrong = np.array(wrong)
    right = np.array(right)
    print(f"    후보 {P_MAIN} 개 · {REPS_MAIN} 번")
    print(f"      잘못된 길  CV R^2 평균 {wrong.mean():+.4f}   "
          f"양수인 비율 {(wrong > 0).mean():.3f}")
    print(f"      옳은 길    CV R^2 평균 {right.mean():+.4f}   "
          f"양수인 비율 {(right > 0).mean():.3f}")

    sweep_w, sweep_r = [], []
    for p in PS:
        w = [cv_r2(rng.normal(size=(N, p)), rng.normal(size=N), True, rng)
             for _ in range(REPS_SWEEP)]
        r = [cv_r2(rng.normal(size=(N, p)), rng.normal(size=N), False, rng)
             for _ in range(REPS_SWEEP)]
        sweep_w.append(np.mean(w))
        sweep_r.append(np.mean(r))
        print(f"      후보 {p:5} 개 — 잘못 {np.mean(w):+.4f}   옳음 {np.mean(r):+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(-1.2, 0.8, 60)
    ax.hist(wrong, bins=bins, color=RED, alpha=0.78, label="전체 자료로 고르고 나서 CV")
    ax.hist(right, bins=bins, color=TEAL, alpha=0.70, label="폴드 안에서 고르고 CV")
    ax.axvline(0, color=INK, lw=2.0, ls="--")
    ax.text(0.02, ax.get_ylim()[1] * 0.93, "참값은 $0$ 이다", fontsize=10.6, color=INK)
    ax.annotate(f"평균 {wrong.mean():+.2f}", xy=(wrong.mean(), 10),
                xytext=(wrong.mean() - 0.05, ax.get_ylim()[1] * 0.60),
                fontsize=11, color=RED, ha="right")
    ax.set_xlabel("교차검증이 말하는 $R^2$", fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title(f"관계가 하나도 없는 자료  (후보 {P_MAIN} 개)",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.axhline(0, color=INK, lw=2.0, ls="--")
    ax.plot(PS, sweep_w, marker="o", ms=8, lw=2.8, color=RED,
            label="전체 자료로 고르고 나서 CV")
    ax.plot(PS, sweep_r, marker="s", ms=8, lw=2.8, color=TEAL,
            label="폴드 안에서 고르고 CV")
    for p, v in zip(PS, sweep_w):
        ax.text(p, v + 0.035, f"{v:+.2f}", ha="center", fontsize=10.2, color=RED)
    ax.set_xscale("log")
    ax.set_xticks(PS)
    ax.set_xticklabels([str(p) for p in PS])
    ax.minorticks_off()
    ax.set_xlabel("고를 수 있는 변수의 개수", fontsize=11.5)
    ax.set_ylabel("교차검증이 말하는 $R^2$ (평균)", fontsize=11.5)
    ax.set_title("많이 훑어볼수록 더 크게 속는다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="center left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("고르는 데 쓴 자료로 채점하면 교차검증이 아니다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"$n = {N}$, 상관이 큰 {TOP} 개를 골라 {K} 겹 교차검증한 것이다. "
             "옳은 길이 음수를 내놓는 것은 고장이 아니라 맞는 답이다 — "
             "쓸모없는 모형은 평균만 쓰느니만 못하다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "cv_wrong_way.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
