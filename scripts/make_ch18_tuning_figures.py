r"""18.6 조정 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/tuning/img/leakage.png       y 를 보는 전처리가 일으키는 누설
  ch18/tuning/img/fold_curves.png   겹마다 다른 교차검증 곡선
  ch18/tuning/img/aic_bic_path.png  라쏘 경로 위의 AIC 와 BIC

실행:  python3 scripts/make_ch18_tuning_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from sklearn.linear_model import Lasso, LinearRegression, lasso_path
from sklearn.model_selection import KFold

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/tuning/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def logx_plain(ax, ticks, labels):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())


# ==================================================================
# 1. 자료 누설   (cross_validation.md)
# ==================================================================
def leakage():
    def one(seed, k=10):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(60, 500))
        y = rng.normal(size=60)          # y 는 X 와 완전히 무관하다
        kf = KFold(10, shuffle=True, random_state=0)
        out = []
        for leak in (True, False):
            errs = []
            s = np.argsort(-np.abs(X.T @ y))[:k] if leak else None
            for tr, te in kf.split(X):
                idx = s if leak else np.argsort(-np.abs(X[tr].T @ y[tr]))[:k]
                m = LinearRegression().fit(X[tr][:, idx], y[tr])
                errs.append(((y[te] - m.predict(X[te][:, idx])) ** 2).mean())
            out.append(np.mean(errs))
        return out

    print("seed=1:", np.round(one(1), 3))
    R = 200
    res = np.array([one(s) for s in range(R)])
    lk, ok = res[:, 0], res[:, 1]
    print(f"[leak] 누설 중앙값 {np.median(lk):.3f}  (최소 {lk.min():.3f}, "
          f"최대 {lk.max():.3f})")
    print(f"       올바름 중앙값 {np.median(ok):.3f}  "
          f"(최소 {ok.min():.3f}, 최대 {ok.max():.3f})")
    print(f"       누설이 1보다 작은 비율 {(lk < 1).mean():.0%}")

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    bins = np.linspace(0.6, 3.6, 46)
    ax.hist(lk, bins=bins, color=RED, alpha=0.7, label="전체 자료로 변수선택 (누설)")
    ax.hist(ok, bins=bins, color=BLUE, alpha=0.7, label="겹 안에서 변수선택 (올바름)")
    ax.axvline(1.0, color=INK, ls="--", lw=2.0, zorder=5)
    ax.text(1.03, ax.get_ylim()[1] * 0.93,
            "예측 불가능한 자료의\n참 오차 = 1.0", fontsize=10.5, color=INK,
            va="top")
    ax.set_xlabel(f"10-겹 교차검증 오차   (순수 잡음 자료 {R}개, "
                  r"$n=60$, $p=500$)", fontsize=11, color=INK)
    ax.set_ylabel("빈도", fontsize=11, color=INK)
    ax.set_title("신호가 전혀 없는 자료에서 보고되는 교차검증 오차",
                 fontsize=12.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax.text(3.55, ax.get_ylim()[1] * 0.72,
            f"중앙값\n누설 {np.median(lk):.2f}\n올바름 {np.median(ok):.2f}",
            fontsize=10.5, color=INK, va="top", ha="right")
    clean(ax)
    fig.tight_layout()
    save(fig, "leakage.png")


# ==================================================================
# 2. 겹마다 다른 곡선   (cv_tuning.md)
# ==================================================================
def fold_curves():
    rng = np.random.default_rng(42)
    n, p = 100, 20
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [3.0, -2.0, 1.5]
    y = X @ beta + rng.normal(0, 1, n)
    lams = np.logspace(0.5, -2.5, 25)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    fold = np.zeros((5, len(lams)))
    for k, (tr, va) in enumerate(kf.split(X)):
        for i, lam in enumerate(lams):
            m = Lasso(alpha=lam, max_iter=10000).fit(X[tr], y[tr])
            fold[k, i] = np.mean((y[va] - m.predict(X[va])) ** 2)
    mu = fold.mean(0)
    se = fold.std(0, ddof=1) / np.sqrt(5)
    i_min = int(np.argmin(mu))
    thr = mu[i_min] + se[i_min]
    i_1se = int(np.where(mu <= thr)[0][0])
    print(f"[cv] lam_min={lams[i_min]:.4f} CV={mu[i_min]:.3f} "
          f"SE={se[i_min]:.3f}")
    print(f"     lam_1se={lams[i_1se]:.4f} CV={mu[i_1se]:.3f}")
    for k in range(5):
        j = int(np.argmin(fold[k]))
        nz = int((np.abs(Lasso(alpha=lams[j], max_iter=10000).fit(X, y).coef_)
                  > 1e-8).sum())
        print(f"     겹 {k+1}: 최적 lam {lams[j]:.4f}  (오차 {fold[k, j]:.3f}) "
              f"변수 {nz}개")
    for i, nm in [(i_min, "min"), (i_1se, "1se")]:
        b = Lasso(alpha=lams[i], max_iter=10000).fit(X, y).coef_
        print(f"     {nm}: 변수 {int((np.abs(b) > 1e-8).sum())}개")

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    for k in range(5):
        ax.plot(lams, fold[k], lw=1.4, color=MUTED, zorder=2,
                label="겹 하나하나의 곡선" if k == 0 else None)
        j = int(np.argmin(fold[k]))
        ax.plot([lams[j]], [fold[k, j]], "v", ms=8, color=MUTED, mec="white",
                mew=0.8, zorder=3)
    ax.fill_between(lams, mu - se, mu + se, color=BLUE_F, alpha=0.9, zorder=1)
    ax.plot(lams, mu, lw=2.8, color=BLUE, zorder=4, label="5겹 평균")
    ax.axhline(thr, color=MUTED, ls=":", lw=1.4, zorder=2)
    ax.axvline(lams[i_min], color=GREEN, ls="--", lw=1.8, zorder=2)
    ax.axvline(lams[i_1se], color=ORANGE, ls="--", lw=1.8, zorder=2)
    ax.plot([lams[i_min]], [mu[i_min]], "o", ms=10, color=GREEN, mec="white",
            mew=1.4, zorder=6)
    ax.plot([lams[i_1se]], [mu[i_1se]], "o", ms=10, color=ORANGE, mec="white",
            mew=1.4, zorder=6)
    logx_plain(ax, [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax.set_ylim(0.6, 3.2)
    ax.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax.set_ylabel("검증 겹의 MSE", fontsize=11, color=INK)
    ax.set_title(r'같은 자료, 같은 격자, 다섯 개의 다른 곡선 ($n=100$, $p=20$)',
                 fontsize=12.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.text(0.0042, 2.55,
            f"$\\lambda_{{\\min}}={lams[i_min]:.3f}$   CV {mu[i_min]:.3f}\n"
            f"$\\lambda_{{1SE}}={lams[i_1se]:.3f}$   CV {mu[i_1se]:.3f}\n"
            f"표준오차 {se[i_min]:.3f}", fontsize=10.5, color=INK, va="top")
    ax.text(0.0042, 0.75, "역삼각형: 겹마다의 최적 $\\lambda$", fontsize=10,
            color=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "fold_curves.png")


# ==================================================================
# 3. 경로 위의 AIC 와 BIC   (information_criteria.md)
# ==================================================================
def aic_bic_path():
    rng = np.random.default_rng(55)
    n, p = 100, 20
    X = rng.normal(size=(n, p))
    X -= X.mean(0)
    X /= X.std(0)
    beta = np.zeros(p)
    beta[:5] = [3, -2, 1.5, 1, -1]
    y = X @ beta + rng.normal(0, 1, n)
    y = y - y.mean()

    b_ols = LinearRegression(fit_intercept=False).fit(X, y).coef_
    s2 = ((y - X @ b_ols) ** 2).sum() / (n - p)
    alphas, coefs, _ = lasso_path(X, y, n_alphas=300, eps=1e-4)
    rss = ((y[:, None] - X @ coefs) ** 2).sum(0)
    df = (np.abs(coefs) > 1e-10).sum(0)
    aic = rss / s2 + 2 * df
    bic = rss / s2 + df * np.log(n)
    ia, ib = int(np.argmin(aic)), int(np.argmin(bic))
    print(f"sigma^2 추정 {s2:.4f}")
    print(f"AIC 최소: lam {alphas[ia]:.4f}  df {df[ia]}  값 {aic[ia]:.1f}")
    print(f"BIC 최소: lam {alphas[ib]:.4f}  df {df[ib]}  값 {bic[ib]:.1f}")
    print(f"df=5 에서 AIC {aic[df == 5].min():.1f}  BIC {bic[df == 5].min():.1f}")
    print(f"log n = {np.log(n):.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    ax1.plot(alphas, aic, lw=2.6, color=BLUE, label="AIC")
    ax1.plot(alphas, bic, lw=2.6, color=ORANGE, label="BIC")
    ax1.plot(alphas, rss / s2, lw=2.0, color=MUTED, ls="--",
             label=r'적합항 $\mathrm{RSS}/\hat\sigma^2$')
    ax1.plot([alphas[ia]], [aic[ia]], "o", ms=9, color=BLUE, mec="white",
             mew=1.3, zorder=5)
    ax1.plot([alphas[ib]], [bic[ib]], "o", ms=9, color=ORANGE, mec="white",
             mew=1.3, zorder=5)
    ax1.axvline(alphas[ia], color=INK, ls=":", lw=1.4)
    logx_plain(ax1, [0.001, 0.01, 0.1, 1], ["0.001", "0.01", "0.1", "1"])
    ax1.set_xlim(alphas.min(), alphas.max())
    ax1.set_ylim(80, 210)
    ax1.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel("기준값 (작을수록 좋다)", fontsize=11, color=INK)
    ax1.set_title("(가) 라쏘 경로를 따라 계산한 AIC 와 BIC", fontsize=12,
                  color=INK)
    ax1.legend(fontsize=10, loc="upper left", frameon=False)
    ax1.annotate(f"둘 다 $\\lambda={alphas[ia]:.4f}$",
                 xy=(alphas[ia], aic[ia]), xytext=(0.0016, 132),
                 fontsize=10.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    clean(ax1)

    ax2.plot(df, aic, "o", ms=4, color=BLUE, alpha=0.6, mew=0, label="AIC")
    ax2.plot(df, bic, "o", ms=4, color=ORANGE, alpha=0.6, mew=0, label="BIC")
    ax2.axvline(5, color=RED, ls="--", lw=1.8)
    ax2.axvline(df[ia], color=INK, ls=":", lw=1.4)
    ax2.text(5.2, 205, "참 변수 5개", fontsize=10.5, color=RED, va="top")
    ax2.text(df[ia] + 0.25, 190, f"고른 모형 df = {df[ia]}", fontsize=10.5,
             color=INK, va="top")
    ax2.set_xlim(0, 21)
    ax2.set_ylim(80, 210)
    ax2.set_xticks(range(0, 21, 5))
    ax2.set_xlabel(r'유효자유도 df $=$ 0이 아닌 계수의 개수', fontsize=11,
                   color=INK)
    ax2.set_ylabel("기준값", fontsize=11, color=INK)
    ax2.set_title("(나) 같은 값을 자유도에 대해 다시 그리면", fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10, loc="upper right", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "aic_bic_path.png")


if __name__ == "__main__":
    leakage()
    fold_curves()
    aic_bic_path()
