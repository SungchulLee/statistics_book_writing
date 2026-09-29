r"""13장 '패키지 사용법' 세 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/package_usage/img/no_intercept_trap.png  add_constant 를 빠뜨리면
  ch13/package_usage/img/train_vs_cv.png        훈련 점수와 교차검증 점수
  ch13/package_usage/img/two_libraries.png      두 라이브러리가 주는 것

실행:  python3 scripts/make_ch13_package_figures.py   (저장소 최상위에서)
필요:  numpy, statsmodels, scikit-learn, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch13/package_usage/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 1. add_constant 를 빠뜨리면 ===
def fig_no_intercept():
    rng = np.random.default_rng(2)
    n = 80
    x = rng.uniform(8, 14, n)
    y = 20 + 1.4 * x + rng.normal(0, 1.2, n)
    r1 = sm.OLS(y, sm.add_constant(x)).fit()
    r0 = sm.OLS(y, x).fit()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    g = np.linspace(0, 15.5, 200)
    ax.scatter(x, y, s=26, color=MUTED, alpha=0.9, edgecolor="none",
               label="관측값", zorder=4)
    ax.plot(g, r1.params[0] + r1.params[1] * g, color=BLUE, lw=2.6,
            label=f"절편 있음  기울기 {r1.params[1]:.3f}")
    ax.plot(g, r0.params[0] * g, color=RED, lw=2.6,
            label=f"절편 없음  기울기 {r0.params[0]:.3f}")
    ax.scatter([0], [0], s=70, color=RED, marker="X", zorder=6)
    ax.annotate("원점을 지나도록 강요된다", xy=(0.35, 1.1), xytext=(2.3, 8.6),
                fontsize=9.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.axvspan(x.min(), x.max(), color=MUTED, alpha=0.12)
    ax.text((x.min() + x.max()) / 2, 4, "자료가 있는 구간", fontsize=9.4,
            color=INK, ha="center")
    ax.set_xlim(-0.6, 15.5)
    ax.set_ylim(0, 48)
    ax.set_xlabel(r"$x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$y$", fontsize=11, color=INK)
    ax.set_title(r"$x$ 가 0 에서 멀면 두 직선이 완전히 갈라진다",
                 fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="upper left", frameon=False)
    clean(ax)

    ax = axes[1]
    labels = ["기울기 추정값", r"보고되는 $R^2$", "잔차의 합"]
    v1 = [r1.params[1], r1.rsquared, r1.resid.sum()]
    v0 = [r0.params[0], r0.rsquared, r0.resid.sum()]
    rows = [2.15, 1.55, 0.95]
    for yy, i in zip(rows, range(3)):
        ax.text(0.02, yy, labels[i], fontsize=11.5, color=INK, ha="left",
                va="center")
        ax.text(0.58, yy, f"{v1[i]:,.4f}", fontsize=13, color=BLUE,
                ha="center", va="center")
        ax.text(0.87, yy, f"{v0[i]:,.4f}", fontsize=13, color=RED,
                ha="center", va="center")
    ax.text(0.58, 2.62, "절편 있음", fontsize=12.5, color=BLUE, ha="center")
    ax.text(0.87, 2.62, "절편 없음", fontsize=12.5, color=RED, ha="center")
    ax.plot([0.02, 0.99], [2.46, 2.46], color=MUTED, lw=1.2)
    ax.annotate("참 기울기는 1.4 다", xy=(0.58, 2.0), xytext=(0.30, 1.86),
                fontsize=9.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.text(0.02, 0.36,
            "절편이 없으면 statsmodels 는 중심화하지 않은 " + r"$R^2$ 을 찍는다."
            "\n적합은 더 나빠졌는데 " + r"$R^2$ 은 오히려 올라간다."
            "\n잔차의 합도 더 이상 0 이 아니다.",
            fontsize=10, color=RED, ha="left", va="top", linespacing=1.8)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.1, 2.95)
    ax.axis("off")
    ax.set_title("출력표의 세 줄이 이렇게 달라진다", fontsize=11.5,
                 color=INK, pad=6)

    fig.tight_layout()
    save(fig, "no_intercept_trap.png")
    print(f"  절편 있음: 절편={r1.params[0]:.4f} 기울기={r1.params[1]:.4f} "
          f"R2={r1.rsquared:.4f} 잔차합={r1.resid.sum():.6f}")
    print(f"  절편 없음: 기울기={r0.params[0]:.4f} R2={r0.rsquared:.4f} "
          f"잔차합={r0.resid.sum():.4f}")
    print(f"  참값: 절편 20, 기울기 1.4")


# === 2. 훈련 점수와 교차검증 점수 ===
def fig_train_vs_cv():
    rng = np.random.default_rng(7)
    n, p, reps, m_test = 200, 8, 800, 20000
    beta = np.array([1.2, -0.8, 0.5] + [0.0] * (p - 3))
    sig = 1.0
    Xt = rng.normal(0, 1, (m_test, p))
    At = np.column_stack([np.ones(m_test), Xt])

    def r2(y, f):
        return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    tr, ho, cv, tgt = [], [], [], []
    for _ in range(reps):
        X = rng.normal(0, 1, (n, p))
        y = X @ beta + rng.normal(0, sig, n)
        A = np.column_stack([np.ones(n), X])
        b = np.linalg.lstsq(A, y, rcond=None)[0]
        tr.append(r2(y, A @ b))
        yt = Xt @ beta + rng.normal(0, sig, m_test)
        tgt.append(r2(yt, At @ b))            # 실제 표본밖 성능

        idx = rng.permutation(n)
        cut = int(n * 0.75)
        i1, i2 = idx[:cut], idx[cut:]
        b1 = np.linalg.lstsq(A[i1], y[i1], rcond=None)[0]
        ho.append(r2(y[i2], A[i2] @ b1))

        folds = np.array_split(rng.permutation(n), 5)
        sc = []
        for f in folds:
            mk = np.ones(n, bool)
            mk[f] = False
            bf = np.linalg.lstsq(A[mk], y[mk], rcond=None)[0]
            sc.append(r2(y[f], A[f] @ bf))
        cv.append(np.mean(sc))

    tr, ho, cv, tgt = map(np.array, (tr, ho, cv, tgt))
    pop_r2 = tgt.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    bins = np.linspace(0.42, 0.88, 95)
    ax.hist(tr, bins=bins, density=True, color=BLUE_F, edgecolor=BLUE,
            lw=1.2, label=f"훈련 $R^2$    평균 {tr.mean():.3f}")
    ax.hist(ho, bins=bins, density=True, histtype="step", color=RED, lw=2.0,
            label=f"한 번 분할   평균 {ho.mean():.3f}")
    ax.hist(cv, bins=bins, density=True, histtype="step", color=GREEN, lw=2.2,
            label=f"5겹 교차검증  평균 {cv.mean():.3f}")
    ax.axvline(pop_r2, color=INK, lw=1.8, ls="--")
    ax.annotate(f"실제 표본밖 성능 {pop_r2:.3f}", xy=(pop_r2, 5.5),
                xytext=(0.455, 8.6), fontsize=9.8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlim(0.42, 0.88)
    ax.set_ylim(0, 16)
    ax.set_xlabel(r"$R^2$", fontsize=11, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title(f"같은 실험을 {reps}번 되풀이하면", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.2, loc="upper right", frameon=False)
    clean(ax)

    ax = axes[1]
    names = ["훈련", "한 번 분할", "5겹 교차검증"]
    arrs = [tr, ho, cv]
    cols = [BLUE, RED, GREEN]
    idx = np.arange(3)
    bias = [(a - tgt).mean() for a in arrs]
    sd = [(a - tgt).std() for a in arrs]
    w = 0.34
    ax.bar(idx - w / 2, bias, w, color=cols, alpha=0.95,
           label="평균 오차 (추정 빼기 실제)")
    ax.bar(idx + w / 2, sd, w, color=MUTED, label="오차의 표준편차")
    ax.axhline(0, color=INK, lw=1.2)
    for i in range(3):
        off = 0.004 if bias[i] > 0 else -0.010
        ax.text(i - w / 2, bias[i] + off, f"{bias[i]:+.3f}", ha="center",
                fontsize=10, color=INK)
        ax.text(i + w / 2, sd[i] + 0.004, f"{sd[i]:.3f}", ha="center",
                fontsize=10, color=INK)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(-0.04, 0.14)
    ax.set_ylabel(r"$R^2$ 단위", fontsize=10.5, color=INK)
    ax.set_title("낙관 편향과 흔들림을 함께 보면", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper left", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "train_vs_cv.png")
    print(f"  n={n}, p={p}, 실제 표본밖 R2 평균={pop_r2:.4f}")
    for nm, a in zip(names, arrs):
        print(f"  {nm:12s} 평균={a.mean():+.4f} 평균오차={(a-tgt).mean():+.4f} "
              f"오차표준편차={(a-tgt).std():.4f}")


# === 3. 두 라이브러리가 주는 것 ===
def fig_two_libraries():
    fig, ax = plt.subplots(figsize=(10.6, 5.2))
    ax.add_patch(Ellipse((0.36, 0.5), 0.60, 0.72, facecolor=BLUE_F,
                         edgecolor=BLUE, lw=2.4, alpha=0.75))
    ax.add_patch(Ellipse((0.64, 0.5), 0.60, 0.72, facecolor=ORANGE_F,
                         edgecolor=ORANGE, lw=2.4, alpha=0.6))
    ax.text(0.14, 0.92, "statsmodels", fontsize=15, color=BLUE, ha="center")
    ax.text(0.86, 0.92, "scikit-learn", fontsize=15, color=ORANGE,
            ha="center")
    ax.text(0.14, 0.86, "추론이 목적일 때", fontsize=10.5, color=BLUE,
            ha="center")
    ax.text(0.86, 0.86, "예측이 목적일 때", fontsize=10.5, color=ORANGE,
            ha="center")

    left = ["표준오차와 $t$ 값", "$p$ 값과 신뢰구간", "$F$ 검정과 AIC · BIC",
            "잔차 진단과 VIF", "로버스트 · 가중 최소제곱"]
    mid = ["계수 " + r"$\hat\beta$", "적합값과 잔차", r"훈련 $R^2$",
           "같은 정규방정식의 해"]
    right = ["훈련 · 검정 분할", "교차검증", "파이프라인", "능형 · 라쏘 정칙화",
             "격자탐색"]
    for i, t in enumerate(left):
        ax.text(0.205, 0.70 - i * 0.085, t, fontsize=10.6, color=INK,
                ha="center")
    for i, t in enumerate(mid):
        ax.text(0.50, 0.665 - i * 0.085, t, fontsize=10.8, color=INK,
                ha="center", fontweight="bold")
    for i, t in enumerate(right):
        ax.text(0.795, 0.70 - i * 0.085, t, fontsize=10.6, color=INK,
                ha="center")
    ax.text(0.50, 0.295, "겹치는 부분은 두 라이브러리가 똑같이 준다",
            fontsize=10.0, color=INK, ha="center")
    ax.text(0.50, 0.055,
            "실무의 순서 :  statsmodels 로 살펴보고 진단한 뒤,  "
            "scikit-learn 으로 검증하고 배포한다",
            fontsize=11, color=INK, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "two_libraries.png")


if __name__ == "__main__":
    fig_no_intercept()
    fig_train_vs_cv()
    fig_two_libraries()
