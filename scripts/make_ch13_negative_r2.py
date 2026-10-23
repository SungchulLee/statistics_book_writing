r"""13.9절 「그림으로 보는 통계학」 — `score()` 가 음수를 돌려줄 때.

$R^2 = 1 - \mathrm{SS}_{\text{res}}/\mathrm{SS}_{\text{tot}}$ 에서 분모는 **그 자료의
평균으로만 맞췄을 때의 오차**다. 그러니 모형이 그 평균보다 못하면 $R^2$ 이 음수가
된다. 훈련자료에 절편을 넣고 적합하면 그런 일이 생길 수 없어서 $[0, 1]$ 안의 수로
오해하기 쉽지만, 두 자리에서 쉽게 깨진다.

  ① 검정자료 — 훈련 범위 **밖**에서 재면 외삽 오차가 분모를 넘어선다
  ② 절편을 빼고 적합하면 — `fit_intercept=False` 는 훈련자료에서도 음수를 만든다

이 스크립트가 재는 값(씨앗 고정):

  훈련 $R^2$ $0.916$ · 범위 밖 검정 $R^2$ $-1.190$
  절편 있음 $0.075$ · `fit_intercept=False` $-778.501$

만드는 파일:

  ch13/package_usage/img/negative_r2.png

실행:  python3 scripts/make_ch13_negative_r2.py   (저장소 최상위에서)
필요:  numpy, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/package_usage/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 7


def main():
    rng = np.random.default_rng(SEED)

    # ── ① 범위 밖에서 재면 ──
    n = 40
    x = rng.uniform(0, 10, n)
    y = 2 + 1.5 * x + rng.normal(0, 2, n)
    model = LinearRegression().fit(x[:, None], y)
    xt = rng.uniform(10, 14, 15)
    yt = 2 + 1.5 * xt + rng.normal(0, 2, 15)
    r2_train = model.score(x[:, None], y)
    r2_test = model.score(xt[:, None], yt)

    # ── ② 절편을 빼면 ──
    y2 = 50 + 0.1 * x + rng.normal(0, 1, n)
    m_no = LinearRegression(fit_intercept=False).fit(x[:, None], y2)
    m_yes = LinearRegression().fit(x[:, None], y2)
    r2_no = m_no.score(x[:, None], y2)
    r2_yes = m_yes.score(x[:, None], y2)

    folds = cross_val_score(LinearRegression(), x[:, None], y,
                            cv=KFold(5, shuffle=True, random_state=0),
                            scoring="neg_mean_squared_error")

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"① 훈련 R^2 {r2_train:7.3f}   범위 밖 검정 R^2 {r2_test:8.3f}")
    print(f"   검정자료의 평균만으로 맞췄을 때의 오차합 "
          f"{((yt - yt.mean()) ** 2).sum():8.2f}")
    print(f"   모형으로 맞췄을 때의 오차합              "
          f"{((yt - model.predict(xt[:, None])) ** 2).sum():8.2f}")
    print(f"② 절편 있음 R^2 {r2_yes:7.3f}   fit_intercept=False {r2_no:10.3f}")
    print(f"③ neg_mean_squared_error 폴드별 {np.round(folds, 2)}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True)

    # ── ① ──
    ax = axes[0]
    grid = np.linspace(0, 14.5, 100)
    ax.plot(x, y, "o", ms=7, color=BLUE, alpha=0.75, label="훈련자료")
    ax.plot(xt, yt, "s", ms=8, color=RED, alpha=0.85, label="검정자료 (범위 밖)")
    ax.plot(grid, model.predict(grid[:, None]), lw=2.6, color=INK,
            label="적합한 직선")
    ax.axhline(yt.mean(), color=RED, lw=1.8, ls="--")
    ax.text(6.0, yt.mean() + 1.2, "검정자료의 평균만으로 맞춘 선", ha="center",
            fontsize=10.6, color=RED)
    ax.axvspan(10, 14.5, color=RED, alpha=0.07)
    ax.text(12.2, 6.5, "훈련자료가\n없던 구간", ha="center", fontsize=10.8,
            color=RED)
    ax.annotate(f"훈련 $R^2 = {r2_train:.3f}$\n검정 $R^2 = {r2_test:.3f}$",
                xy=(0.3, 25.0), fontsize=12, color=INK, fontweight="bold")
    ax.set_xlim(-0.5, 14.8)
    ax.set_xlabel("$x$", fontsize=11.5)
    ax.set_ylabel("$y$", fontsize=11.5)
    ax.set_title("① 범위 밖에서 재면 평균보다 못하다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② ──
    ax = axes[1]
    g2 = np.linspace(0, 10.5, 100)
    ax.plot(x, y2, "o", ms=7, color=BLUE, alpha=0.75, label="자료")
    ax.plot(g2, m_yes.predict(g2[:, None]), lw=2.6, color=TEAL,
            label=f"절편 있음 — $R^2 = {r2_yes:.3f}$")
    ax.plot(g2, m_no.predict(g2[:, None]), lw=2.6, ls="--", color=RED,
            label=f"fit_intercept=False — $R^2 = {r2_no:.1f}$")
    ax.plot([0], [0], "x", ms=12, mew=3, color=RED)
    ax.annotate("원점을 지나라고 시키면\n자료 근처를 지날 수 없다",
                xy=(0.6, 3.0), xytext=(2.6, 14),
                fontsize=11.2, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlim(-0.4, 10.6)
    ax.set_ylim(-3, 57)
    ax.set_xlabel("$x$", fontsize=11.5)
    ax.set_ylabel("$y$", fontsize=11.5)
    ax.set_title("② 절편을 빼면 훈련자료에서도 음수", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$R^2$ 의 분모는 「그 자료의 평균」이다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "두 판 모두 scikit-learn 의 LinearRegression 이 돌려준 "
             "`score()` 값을 그대로 적었다. 모든 수는 이 스크립트가 계산한 값이다."
             .replace("`", ""),
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "negative_r2.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
