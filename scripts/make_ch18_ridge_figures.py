r"""18.2 능형회귀 여섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/ridge/img/rss_valley.png         벌점이 평평한 골짜기를 그릇으로 되살린다
  ch18/ridge/img/svd_shrinkage.png      축소인자와 유효자유도
  ch18/ridge/img/ridge_tangency.png     타원과 원의 접점, 그리고 능형 자취
  ch18/ridge/img/bayes_prior_post.png   사전분포 x 가능도 = 사후분포, 신용구간
  ch18/ridge/img/cv_1se.png             CV 곡선의 1-SE 규칙과 능형 자취
  ch18/ridge/img/shrink_vs_accuracy.png 축소는 단조, 정확도는 U자

실행:  python3 scripts/make_ch18_ridge_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.patches import Circle
from sklearn.linear_model import Ridge, LinearRegression
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

OUT = "docs/ch18/ridge/img/"
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


def collinear_data(n=50, rho=0.99, seed=3, beta=(1.0, 1.0), sd=1.0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x1 = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n)
    x2 = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n)
    X = np.c_[x1, x2]
    X -= X.mean(0)
    y = X @ np.array(beta) + rng.normal(0, sd, n)
    y -= y.mean()
    return X, y


# ==================================================================
# 1. 벌점이 평평한 골짜기를 그릇으로 되살린다  (ridge.md)
# ==================================================================
def rss_valley():
    X, y = collinear_data()
    n = len(y)
    A = X.T @ X
    b = X.T @ y
    ev = np.linalg.eigvalsh(A)
    ols = np.linalg.solve(A, b)
    lam = 20.0
    rid = np.linalg.solve(A + lam * np.eye(2), b)
    evr = np.linalg.eigvalsh(A + lam * np.eye(2))
    print(f"[rss_valley] X'X 고윳값 {ev[::-1].round(2)}  조건수 {ev[1]/ev[0]:.0f}")
    print(f"  OLS {ols.round(2)}  능형(lam=20) {rid.round(3)}  "
          f"벌점 후 고윳값 {evr[::-1].round(2)}  조건수 {evr[1]/evr[0]:.1f}")

    g = np.linspace(-5, 6, 400)
    B1, B2 = np.meshgrid(g, g)
    flat = np.c_[B1.ravel(), B2.ravel()]
    rss = ((y[:, None] - X @ flat.T) ** 2).sum(axis=0).reshape(B1.shape)
    pen = rss + lam * (B1 ** 2 + B2 ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    for ax, Zv, ctr, ttl, col in [
            (axes[0], rss, ols, r'(가) 최소제곱 목적함수 $\|y-X\beta\|^2$', BLUE),
            (axes[1], pen, rid,
             r'(나) 벌점 목적함수 $\|y-X\beta\|^2+20\|\beta\|^2$', ORANGE)]:
        lo = Zv.min()
        levels = lo + np.array([0.5, 2, 5, 12, 25, 45, 75, 120, 180, 260, 360])
        ax.contourf(B1, B2, Zv, levels=levels, cmap="Blues_r", alpha=0.65,
                    extend="min")
        ax.contour(B1, B2, Zv, levels=levels, colors=MUTED, linewidths=0.7)
        ax.plot([ctr[0]], [ctr[1]], "*", ms=18, color=col, mec="white",
                mew=1.2, zorder=5)
        ax.axhline(0, color=INK, lw=0.8)
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-5, 6)
        ax.set_ylim(-5, 6)
        ax.set_aspect("equal")
        ax.set_xlabel(r'$\beta_1$', fontsize=12, color=INK)
        ax.set_title(ttl, fontsize=12, color=INK)
        clean(ax)
    axes[0].set_ylabel(r'$\beta_2$', fontsize=12, color=INK)
    axes[0].annotate(f"OLS 해\n({ols[0]:.2f}, {ols[1]:.2f})", xy=ols,
                     xytext=(2.0, 4.4), fontsize=10.5, color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    axes[0].text(-4.6, -4.4, "골짜기가 평평하다\n어디가 바닥인지 자료가 모른다",
                 fontsize=10.5, color=INK)
    axes[1].annotate(f"능형 해\n({rid[0]:.2f}, {rid[1]:.2f})", xy=rid,
                     xytext=(2.0, 4.4), fontsize=10.5, color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    axes[1].text(-4.6, -4.4, "그릇 모양이 되살아난다\n바닥이 한 점으로 정해진다",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "rss_valley.png")


# ==================================================================
# 2. 축소인자와 유효자유도  (formulation.md)
# ==================================================================
def svd_shrinkage():
    rng = np.random.default_rng(21)
    n, p, rho = 50, 4, 0.95
    S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(S).T
    X -= X.mean(0)
    d2 = np.linalg.eigvalsh(X.T @ X)[::-1]
    print("[svd] 고윳값", d2.round(3))

    lams = np.logspace(-2, 4, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    cols = [BLUE, ORANGE, GREEN, PURPLE]
    for j in range(p):
        f = d2[j] / (d2[j] + lams)
        ax1.plot(lams, f, lw=2.2, color=cols[j],
                 label=f"$d_{{{j+1}}}^2={d2[j]:.1f}$")
        ax1.plot([d2[j]], [0.5], "o", ms=7, color=cols[j], mec="white",
                 mew=1.2, zorder=5)
    ax1.axhline(0.5, color=MUTED, ls=":", lw=1.2)
    logx_plain(ax1, [0.01, 0.1, 1, 10, 100, 1000, 10000],
               ["0.01", "0.1", "1", "10", "100", "1000", "10000"])
    ax1.set_ylim(-0.03, 1.08)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel(r'축소인자 $d_j^2/(d_j^2+\lambda)$', fontsize=11, color=INK)
    ax1.set_title("(가) 성분마다 축소의 강도가 다르다", fontsize=12, color=INK)
    ax1.legend(fontsize=10, loc="lower left", frameon=False)
    ax1.text(12000, 0.78, r'점: $\lambda=d_j^2$ 에서 정확히 절반이 남는다',
             fontsize=10, color=INK, ha="right")
    clean(ax1)

    df = (d2[None, :] / (d2[None, :] + lams[:, None])).sum(axis=1)
    ax2.plot(lams, df, lw=2.6, color=PURPLE)
    for lam in (1.0, 10.0, 100.0):
        v = (d2 / (d2 + lam)).sum()
        ax2.plot([lam], [v], "o", ms=8, color=PURPLE, mec="white", mew=1.4,
                 zorder=5)
        ax2.annotate(f"$\\lambda={lam:g}$\ndf = {v:.2f}", xy=(lam, v),
                     xytext=(lam * 2.2, v + 0.55), fontsize=10, color=INK,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.0))
    ax2.axhline(p, color=MUTED, ls="--", lw=1.2)
    ax2.text(9000, p + 0.08, r'OLS: df $=p=4$', fontsize=10.5, color=INK,
             ha="right", va="bottom")
    logx_plain(ax2, [0.01, 0.1, 1, 10, 100, 1000, 10000],
               ["0.01", "0.1", "1", "10", "100", "1000", "10000"])
    ax2.set_ylim(-0.12, 4.6)
    ax2.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel(r'유효자유도 $\mathrm{df}(\lambda)$', fontsize=11, color=INK)
    ax2.set_title("(나) 살아남은 방향의 개수", fontsize=12, color=INK)
    clean(ax2)

    fig.tight_layout()
    save(fig, "svd_shrinkage.png")


# ==================================================================
# 3. 타원과 원의 접점  (geometry.md)
# ==================================================================
def ridge_tangency():
    X, y = collinear_data(n=50, rho=0.9, seed=7, beta=(1.0, 1.0), sd=1.0)
    A = X.T @ X
    b = X.T @ y
    ols = np.linalg.solve(A, b)
    lams = np.concatenate([[0], np.logspace(-2, 3.6, 400)])
    path = np.array([np.linalg.solve(A + l * np.eye(2), b) for l in lams])
    print("[tangency] OLS", ols.round(3))
    for l in (20.0, 120.0):
        s = np.linalg.solve(A + l * np.eye(2), b)
        print(f"  lam={l:g} -> {s.round(3)}  반지름 {np.linalg.norm(s):.3f}")

    g = np.linspace(-1.2, 3.4, 400)
    B1, B2 = np.meshgrid(g, g)
    flat = np.c_[B1.ravel(), B2.ravel()]
    rss = ((y[:, None] - X @ flat.T) ** 2).sum(axis=0).reshape(B1.shape)

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    base = rss.min()
    for l, col in [(20.0, ORANGE), (120.0, RED)]:
        s = np.linalg.solve(A + l * np.eye(2), b)
        r = np.linalg.norm(s)
        lvl = ((y - X @ s) ** 2).sum()
        ax.contour(B1, B2, rss, levels=[lvl], colors=[col], linewidths=1.8)
        ax.add_patch(Circle((0, 0), r, fill=False, ec=col, lw=1.6, ls="--"))
        ax.plot([s[0]], [s[1]], "o", ms=10, color=col, mec="white", mew=1.4,
                zorder=6)
    ax.contour(B1, B2, rss, levels=base + np.array([1.5, 6.0]),
               colors=[MUTED], linewidths=0.7)
    ax.plot(path[:, 0], path[:, 1], lw=2.2, color=BLUE, zorder=4)
    ax.plot([ols[0]], [ols[1]], "*", ms=18, color=BLUE, mec="white", mew=1.2,
            zorder=6)
    ax.plot([0], [0], "o", ms=6, color=INK, zorder=6)
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlim(-1.45, 3.1)
    ax.set_ylim(-1.45, 3.1)
    ax.set_aspect("equal")
    ax.set_xlabel(r'$\beta_1$', fontsize=12, color=INK)
    ax.set_ylabel(r'$\beta_2$', fontsize=12, color=INK)
    ax.set_title(r'RSS 타원이 L2 공에 처음 닿는 점', fontsize=12.5, color=INK)
    ax.text(ols[0] + 0.09, ols[1] + 0.10, "OLS", fontsize=11, color=BLUE)
    s5 = np.linalg.solve(A + 20 * np.eye(2), b)
    s40 = np.linalg.solve(A + 120 * np.eye(2), b)
    ax.annotate(r'$\lambda=20$', xy=(s5[0], s5[1]), textcoords="offset points",
                xytext=(34, -30), fontsize=11, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate(r'$\lambda=120$', xy=(s40[0], s40[1]),
                textcoords="offset points", xytext=(-80, 20), fontsize=11,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(-1.38, 3.02, "파란 곡선: 능형 자취\n회색 타원: RSS 등고선\n"
                         "점선 원: 제약 " + r'$\|\beta\|^2\leq t$',
            fontsize=10.5, color=INK, va="top")
    ax.text(3.02, -1.33, "접점이 어느 축에도 닿지 않는다", fontsize=10.5,
            color=INK, ha="right")
    clean(ax)
    fig.tight_layout()
    save(fig, "ridge_tangency.png")


# ==================================================================
# 4. 사전분포와 사후분포  (bayesian.md)
# ==================================================================
def bayes_prior_post():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # --- (가) 1차원 그림
    bhat, s, tau = 2.0, 0.8, 1.0
    g = np.linspace(-1.6, 4.2, 600)
    lik = np.exp(-0.5 * ((g - bhat) / s) ** 2)
    pri = np.exp(-0.5 * (g / tau) ** 2)
    w = (1 / s ** 2) / (1 / s ** 2 + 1 / tau ** 2)
    mpost = w * bhat
    spost = np.sqrt(1 / (1 / s ** 2 + 1 / tau ** 2))
    post = np.exp(-0.5 * ((g - mpost) / spost) ** 2)
    print(f"[bayes] 사후평균 {mpost:.3f}  사후표준편차 {spost:.3f}  "
          f"lambda=sig^2/tau^2={s**2/tau**2 * 0 + 1:.0f}")

    ax1.fill_between(g, pri, color=GREEN_F, alpha=0.7, zorder=1)
    ax1.plot(g, pri, lw=2, color=GREEN, zorder=2, label=r'사전분포 $N(0,\tau^2)$')
    ax1.plot(g, lik, lw=2, color=BLUE, zorder=5, label="가능도 (자료)")
    ax1.fill_between(g, post, color=ORANGE_F, alpha=0.8, zorder=3)
    ax1.plot(g, post, lw=2.4, color=ORANGE, zorder=4, label="사후분포")
    for xv, col, txt, dy in [(0, GREEN, "0", 0.0),
                             (mpost, ORANGE, f"MAP = {mpost:.2f}", 0.0),
                             (bhat, BLUE, f"OLS = {bhat:.2f}", 0.0)]:
        ax1.plot([xv, xv], [0, 1.02], ls=":", lw=1.2, color=col, zorder=1)
    ax1.annotate("", xy=(mpost, 1.12), xytext=(bhat, 1.12),
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
    ax1.text((mpost + bhat) / 2, 1.16, "축소", fontsize=10.5, color=RED,
             ha="center")
    ax1.text(bhat, -0.09, f"OLS {bhat:.1f}", fontsize=10, color=BLUE,
             ha="center")
    ax1.text(mpost, -0.09, f"MAP {mpost:.2f}", fontsize=10, color=ORANGE,
             ha="center")
    ax1.set_ylim(-0.14, 1.3)
    ax1.set_xlim(-1.6, 4.2)
    ax1.set_yticks([])
    ax1.set_xlabel(r'계수 $\beta$', fontsize=11, color=INK)
    ax1.set_title(r'(가) 사전분포 $\times$ 가능도 $=$ 사후분포', fontsize=12,
                  color=INK)
    ax1.legend(fontsize=10, loc="upper left", frameon=False)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.spines["bottom"].set_color(MUTED)
    ax1.tick_params(labelsize=10, colors=INK)

    # --- (나) 신용구간은 좁아지는데 중심이 0으로 간다
    rng = np.random.default_rng(0)
    n, p, rho, sig = 50, 4, 0.95, 1.0
    S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    Xb = rng.multivariate_normal(np.zeros(p), S, n)
    yb = Xb @ np.ones(p) + rng.normal(0, sig, n)
    A = Xb.T @ Xb
    bb = Xb.T @ yb
    lams = np.logspace(-2, 3, 200)
    ctr, sd = [], []
    for l in lams:
        M = np.linalg.inv(A + l * np.eye(p))
        ctr.append((M @ bb)[0])
        sd.append(np.sqrt(M[0, 0]))
    ctr, sd = np.array(ctr), np.array(sd)
    for l in (0.0, 1.0, 10.0):
        M = np.linalg.inv(A + l * np.eye(p))
        print(f"  lam={l:g}: b1={(M @ bb)[0]:.3f}  sd={np.sqrt(M[0,0]):.3f}")

    ax2.fill_between(lams, ctr - 1.96 * sd, ctr + 1.96 * sd, color=BLUE_F,
                     alpha=0.9, zorder=1, label="95% 신용구간")
    ax2.plot(lams, ctr, lw=2.4, color=BLUE, zorder=3, label=r'사후평균 $\hat\beta_1$')
    ax2.axhline(1.0, color=RED, ls="--", lw=1.6, zorder=2)
    ax2.text(0.012, 1.08, r'참값 $\beta_1=1$', fontsize=10.5, color=RED)
    ax2.axhline(0, color=INK, lw=0.8, zorder=2)
    logx_plain(ax2, [0.01, 0.1, 1, 10, 100, 1000],
               ["0.01", "0.1", "1", "10", "100", "1000"])
    ax2.set_ylim(-0.75, 2.6)
    ax2.set_xlabel(r'$\lambda=\sigma^2/\tau^2$ (로그 눈금)', fontsize=11,
                   color=INK)
    ax2.set_ylabel(r'$\beta_1$', fontsize=11, color=INK)
    ax2.set_title("(나) 구간은 좁아지고 중심은 0으로 간다", fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10, loc="lower left", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "bayes_prior_post.png")


# ==================================================================
# 5. CV 곡선과 1-SE 규칙  (lambda_selection.md)
# ==================================================================
def cv_1se():
    rng = np.random.default_rng(31)
    n, p, rho = 80, 15, 0.9
    S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(S).T
    X -= X.mean(0)
    X /= X.std(0)
    beta = np.zeros(p)
    beta[:4] = [3, -2, 1.5, 1]
    y = X @ beta + rng.normal(0, 1, n)
    y -= y.mean()

    d2 = np.linalg.eigvalsh(X.T @ X)[::-1]
    lams = np.logspace(-2, 3, 60)
    kf = KFold(10, shuffle=True, random_state=0)
    mu, se = [], []
    for lam in lams:
        errs = [((y[te] - X[te] @ Ridge(alpha=lam, fit_intercept=False)
                  .fit(X[tr], y[tr]).coef_) ** 2).mean()
                for tr, te in kf.split(X)]
        mu.append(np.mean(errs))
        se.append(np.std(errs, ddof=1) / np.sqrt(10))
    mu, se = np.array(mu), np.array(se)
    i = int(mu.argmin())
    j = int(np.max(np.where(mu <= mu[i] + se[i])[0]))
    dfs = np.array([(d2 / (d2 + l)).sum() for l in lams])
    print(f"[cv] lam_min={lams[i]:.3f} CV={mu[i]:.3f} SE={se[i]:.3f} "
          f"df={dfs[i]:.2f}")
    print(f"     lam_1se={lams[j]:.3f} CV={mu[j]:.3f} df={dfs[j]:.2f}")

    coefs = np.array([Ridge(alpha=l, fit_intercept=False).fit(X, y).coef_
                      for l in lams])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    ax1.fill_between(lams, mu - se, mu + se, color=BLUE_F, alpha=0.9, zorder=1)
    ax1.plot(lams, mu, lw=2.4, color=BLUE, zorder=3)
    ax1.axhline(mu[i] + se[i], color=MUTED, ls=":", lw=1.4, zorder=2)
    ax1.axvline(lams[i], color=GREEN, ls="--", lw=1.6, zorder=2)
    ax1.axvline(lams[j], color=ORANGE, ls="--", lw=1.6, zorder=2)
    ax1.plot([lams[i]], [mu[i]], "o", ms=9, color=GREEN, mec="white", mew=1.4,
             zorder=5)
    ax1.plot([lams[j]], [mu[j]], "o", ms=9, color=ORANGE, mec="white", mew=1.4,
             zorder=5)
    logx_plain(ax1, [0.01, 0.1, 1, 10, 100, 1000],
               ["0.01", "0.1", "1", "10", "100", "1000"])
    ax1.set_ylim(1.0, 2.9)
    ax1.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel("10-겹 교차검증 오차", fontsize=11, color=INK)
    ax1.set_title("(가) 최솟값과 1-표준오차 문턱", fontsize=12, color=INK)
    ax1.text(0.012, 2.82,
             f"$\\lambda_{{\\min}}={lams[i]:.2f}$\n"
             f"CV {mu[i]:.3f}  df {dfs[i]:.2f}", fontsize=10.5, color=GREEN,
             va="top")
    ax1.text(0.012, 2.42,
             f"$\\lambda_{{1SE}}={lams[j]:.2f}$\n"
             f"CV {mu[j]:.3f}  df {dfs[j]:.2f}", fontsize=10.5, color=ORANGE,
             va="top")
    ax1.text(120, mu[i] + se[i] + 0.05, "1-SE 문턱", fontsize=10, color=INK,
             ha="center")
    clean(ax1)

    for k in range(p):
        c = [BLUE, ORANGE, GREEN, PURPLE][k] if k < 4 else MUTED
        ax2.plot(lams, coefs[:, k], lw=2.0 if k < 4 else 1.2, color=c)
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.axvline(lams[i], color=GREEN, ls="--", lw=1.6)
    ax2.axvline(lams[j], color=ORANGE, ls="--", lw=1.6)
    logx_plain(ax2, [0.01, 0.1, 1, 10, 100, 1000],
               ["0.01", "0.1", "1", "10", "100", "1000"])
    ax2.set_ylim(-2.6, 3.4)
    ax2.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel("계수 추정값", fontsize=11, color=INK)
    ax2.set_title("(나) 같은 두 눈금을 능형 자취 위에 얹으면", fontsize=12,
                  color=INK)
    ax2.text(lams[i] * 1.25, 3.15, r'$\lambda_{\min}$', fontsize=11,
             color=GREEN)
    ax2.text(lams[j] * 1.25, 3.15, r'$\lambda_{1SE}$', fontsize=11,
             color=ORANGE)
    ax2.text(0.012, -1.95, "굵은 선 4개 = 참 변수\n가는 회색 11개 = 잡음변수",
             fontsize=10, color=INK, va="top")
    clean(ax2)

    fig.tight_layout()
    save(fig, "cv_1se.png")


# ==================================================================
# 6. 축소는 단조, 정확도는 U자  (ridge_examples.md)
# ==================================================================
def shrink_vs_accuracy():
    rng = np.random.default_rng(42)
    n, p = 40, 20
    X = rng.normal(size=(n, p))
    beta = np.zeros(p)
    beta[:3] = [3.0, -2.0, 1.5]
    y = X @ beta + rng.normal(0, 1, n)

    ols = LinearRegression().fit(X, y).coef_
    lams = np.logspace(-2, 2.7, 160)
    norms, dists = [], []
    for l in lams:
        c = Ridge(alpha=l).fit(X, y).coef_
        norms.append(np.linalg.norm(c))
        dists.append(np.linalg.norm(c - beta))
    norms, dists = np.array(norms), np.array(dists)
    k = int(np.argmin(dists))
    print(f"[shrink] OLS 노름 {np.linalg.norm(ols):.3f}  "
          f"거리 {np.linalg.norm(ols - beta):.3f}")
    for l in (0.1, 1.0, 10.0, 100.0):
        c = Ridge(alpha=l).fit(X, y).coef_
        print(f"  lam={l:g}: 노름 {np.linalg.norm(c):.3f}  "
              f"거리 {np.linalg.norm(c - beta):.3f}")
    print(f"  최적 lam={lams[k]:.2f} 거리 {dists[k]:.3f}")

    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(lams, norms, lw=2.6, color=BLUE, label=r'계수의 크기 $\|\hat\beta\|_2$')
    ax.plot(lams, dists, lw=2.6, color=ORANGE,
            label=r'참값과의 거리 $\|\hat\beta-\beta\|_2$')
    ax.plot([lams[k]], [dists[k]], "o", ms=10, color=ORANGE, mec="white",
            mew=1.5, zorder=5)
    ax.axhline(np.linalg.norm(ols - beta), color=MUTED, ls=":", lw=1.4)
    logx_plain(ax, [0.01, 0.1, 1, 10, 100],
               ["0.01", "0.1", "1", "10", "100"])
    ax.set_ylim(0, 5.0)
    ax.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax.set_ylabel("L2 노름", fontsize=11, color=INK)
    ax.set_title(r'$n=40$, $p=20$, 참 변수 3개: 축소는 단조, 정확도는 U자',
                 fontsize=12.5, color=INK)
    ax.annotate(f"최적 $\\lambda={lams[k]:.2f}$\n거리 {dists[k]:.3f}",
                xy=(lams[k], dists[k]), xytext=(lams[k] * 0.09, 1.9),
                fontsize=10.5, color=ORANGE,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.text(0.011, np.linalg.norm(ols - beta) + 0.1,
            f"OLS 거리 {np.linalg.norm(ols - beta):.3f}", fontsize=10.5,
            color=INK)
    ax.legend(fontsize=10.5, loc="upper right", frameon=False)
    clean(ax)
    fig.tight_layout()
    save(fig, "shrink_vs_accuracy.png")


if __name__ == "__main__":
    rss_valley()
    svd_shrinkage()
    ridge_tangency()
    bayes_prior_post()
    cv_1se()
    shrink_vs_accuracy()
