r"""18.3 라쏘 여덟 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/lasso/img/l1_vs_l2_tangency.png   마름모의 꼭짓점과 원의 옆구리
  ch18/lasso/img/soft_threshold_min.png  1차원 목적함수의 최솟값이 0에 걸린다
  ch18/lasso/img/laplace_prior.png       라플라스 사전분포와 최빈값-평균의 괴리
  ch18/lasso/img/coord_descent.png       좌표하강의 지그재그와 수렴 속도
  ch18/lasso/img/post_lasso_bias.png     라쏘의 편향은 정확히 lambda 다
  ch18/lasso/img/shrinkage_operators.png 세 축소 연산자와 실제 계수의 이동
  ch18/lasso/img/path_cv.png             경로와 교차검증 곡선
  ch18/lasso/img/housing_path.png        주택자료 라쏘 경로와 활성 변수 개수

실행:  python3 scripts/make_ch18_lasso_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       마지막 housing_path 만 pandas 와 인터넷 연결이 필요하다(자료는 gitignore
       되어 있어 내려받는다). PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from matplotlib.patches import Circle, Polygon
from sklearn.linear_model import Lasso, Ridge, LinearRegression

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/lasso/img/"
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


def soft(z, l):
    return np.sign(z) * np.maximum(np.abs(z) - l, 0)


# ==================================================================
# 1. 마름모의 꼭짓점 대 원의 옆구리   (geometry.md)
# ==================================================================
def l1_vs_l2_tangency():
    rng = np.random.default_rng(12)
    n = 60
    z = rng.normal(size=n)
    x1 = 0.75 * z + 0.66 * rng.normal(size=n)
    x2 = 0.75 * z + 0.66 * rng.normal(size=n)
    X = np.c_[x1, x2]
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([1.5, 0.35]) + rng.normal(0, 1.0, n)
    y = y - y.mean()

    A = X.T @ X
    b = X.T @ y
    ols = np.linalg.solve(A, b)
    lam = 0.45
    bl = Lasso(alpha=lam, fit_intercept=False, max_iter=200000).fit(X, y).coef_
    t1 = np.abs(bl).sum()
    lam2 = 55.0
    br = np.linalg.solve(A + lam2 * np.eye(2), b)
    t2 = np.linalg.norm(br)
    print(f"[tangency] OLS {ols.round(3)}  라쏘 {bl.round(3)} (t={t1:.3f})  "
          f"능형 {br.round(3)} (r={t2:.3f})")

    g = np.linspace(-0.55, 2.3, 400)
    B1, B2 = np.meshgrid(g, g)
    flat = np.c_[B1.ravel(), B2.ravel()]
    rss = ((y[:, None] - X @ flat.T) ** 2).sum(axis=0).reshape(B1.shape)
    base = rss.min()

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    for ax, sol, shape, ttl, col in [
            (axes[0], bl, "l1", r'(가) 라쏘: $|\beta_1|+|\beta_2|\leq t$', ORANGE),
            (axes[1], br, "l2", r'(나) 능형: $\beta_1^2+\beta_2^2\leq t$', BLUE)]:
        lvl = ((y - X @ sol) ** 2).sum()
        ax.contour(B1, B2, rss, levels=base + np.array([3, 14]), colors=[MUTED],
                   linewidths=0.7)
        ax.contour(B1, B2, rss, levels=[lvl], colors=[col], linewidths=2.0)
        if shape == "l1":
            t = np.abs(sol).sum()
            ax.add_patch(Polygon([[t, 0], [0, t], [-t, 0], [0, -t]],
                                 closed=True, fc=ORANGE_F, ec=ORANGE, lw=1.8,
                                 alpha=0.85, zorder=2))
        else:
            r = np.linalg.norm(sol)
            ax.add_patch(Circle((0, 0), r, fc=BLUE_F, ec=BLUE, lw=1.8,
                                alpha=0.85, zorder=2))
        ax.plot([ols[0]], [ols[1]], "*", ms=18, color=INK, mec="white", mew=1.2,
                zorder=6)
        ax.plot([sol[0]], [sol[1]], "o", ms=11, color=col, mec="white", mew=1.4,
                zorder=6)
        ax.axhline(0, color=INK, lw=0.9)
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_xlim(-0.55, 2.3)
        ax.set_ylim(-0.55, 2.3)
        ax.set_aspect("equal")
        ax.set_xlabel(r'$\beta_1$', fontsize=12, color=INK)
        ax.set_title(ttl, fontsize=12.5, color=INK)
        ax.text(ols[0] + 0.07, ols[1] + 0.08, "OLS", fontsize=11, color=INK)
        clean(ax)
    axes[0].set_ylabel(r'$\beta_2$', fontsize=12, color=INK)
    axes[0].annotate(f"꼭짓점에서 만난다\n$\\hat\\beta=({bl[0]:.2f},\\ 0)$",
                     xy=(bl[0], 0.0), xytext=(1.15, 1.55), fontsize=11,
                     color=ORANGE,
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    axes[1].annotate(f"옆구리에서 만난다\n$\\hat\\beta=({br[0]:.2f},\\ {br[1]:.2f})$",
                     xy=(br[0], br[1]), xytext=(1.15, 1.55), fontsize=11,
                     color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    fig.tight_layout()
    save(fig, "l1_vs_l2_tangency.png")


# ==================================================================
# 2. 1차원 목적함수의 최솟값   (formulation.md)
# ==================================================================
def soft_threshold_min():
    lam = 0.6
    g = np.linspace(-0.9, 2.4, 700)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), sharey=True)
    for ax, kind, ttl in [
            (axes[0], "l1", r'(가) L1 벌점  $\frac{1}{2}(\beta-z)^2+\lambda|\beta|$'),
            (axes[1], "l2",
             r'(나) L2 벌점  $\frac{1}{2}(\beta-z)^2+\frac{\lambda}{2}\beta^2$')]:
        for zv, col in [(0.35, BLUE), (1.5, ORANGE)]:
            pen = lam * np.abs(g) if kind == "l1" else 0.5 * lam * g ** 2
            f = 0.5 * (g - zv) ** 2 + pen
            star = soft(zv, lam) if kind == "l1" else zv / (1 + lam)
            fstar = (0.5 * (star - zv) ** 2
                     + (lam * abs(star) if kind == "l1" else 0.5 * lam * star ** 2))
            ax.plot(g, f, lw=2.4, color=col, zorder=3,
                    label=f"자료가 주는 값 $z={zv}$")
            ax.plot([star], [fstar], "o", ms=10, color=col, mec="white",
                    mew=1.4, zorder=5)
            ax.plot([zv, zv], [0, 0.06], lw=2, color=col, zorder=3)
            ax.text(star, fstar - 0.12, f"{star:.2f}", fontsize=10.5, color=col,
                    ha="center", va="top")
            print(f"  {kind} z={zv} -> {star:.3f}")
        ax.axvline(0, color=INK, lw=0.9)
        ax.set_ylim(-0.25, 2.6)
        ax.set_xlim(-0.9, 2.4)
        ax.set_xlabel(r'$\beta$', fontsize=12, color=INK)
        ax.set_title(ttl, fontsize=12, color=INK)
        ax.legend(fontsize=10.5, loc="lower right", frameon=False)
        clean(ax)
    axes[0].set_ylabel("목적함수 값", fontsize=11, color=INK)
    axes[0].annotate(r'$0$ 에서 꺾인다',
                     xy=(0.0, 0.5 * 0.35 ** 2), xytext=(0.55, 1.75),
                     fontsize=11, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[1].annotate(r'$0$ 에서 매끄럽다',
                     xy=(0.0, 0.5 * 0.35 ** 2), xytext=(0.55, 1.75),
                     fontsize=11, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[0].text(1.25, 2.45, r'$\lambda=0.6$', fontsize=11, color=INK,
                 va="top")
    axes[1].text(2.35, 2.45, "짧은 세로 막대가 $z$ 의 위치", fontsize=10,
                 color=INK, ha="right", va="top")
    fig.tight_layout()
    save(fig, "soft_threshold_min.png")


# ==================================================================
# 3. 라플라스 사전분포   (bayesian.md)
# ==================================================================
def laplace_prior():
    g = np.linspace(-4.2, 4.2, 900)
    gau = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    bb = 1 / np.sqrt(2)                       # 분산을 1로 맞춘다
    lap = np.exp(-np.abs(g) / bb) / (2 * bb)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    ax1.fill_between(g, lap, color=ORANGE_F, alpha=0.6)
    ax1.plot(g, lap, lw=2.4, color=ORANGE, label="라플라스 (라쏘)")
    ax1.plot(g, gau, lw=2.4, color=BLUE, label="가우스 (능형)")
    ax1.set_xlim(-4.2, 4.2)
    ax1.set_ylim(0, 0.78)
    ax1.set_xlabel(r'$\beta_j$', fontsize=12, color=INK)
    ax1.set_ylabel("사전분포 밀도", fontsize=11, color=INK)
    ax1.set_title("(가) 분산을 1로 맞춘 두 사전분포", fontsize=12, color=INK)
    ax1.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax1.annotate("0에서 뾰족하다", xy=(0, 0.707), xytext=(-4.0, 0.60),
                 fontsize=10.5, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax1.annotate("꼬리가 두껍다", xy=(3.2, lap[np.argmin(np.abs(g - 3.2))]),
                 xytext=(1.6, 0.30), fontsize=10.5, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    print(f"[prior] 밀도비: 0.5에서 {np.exp(-0.5/bb)/(2*bb)/(np.exp(-0.125)/np.sqrt(2*np.pi)):.2f}배")
    for xv in (2.0, 3.0, 4.0):
        i = np.argmin(np.abs(g - xv))
        print(f"  beta={xv}: 라플라스 {lap[i]:.4f}  가우스 {gau[i]:.4f}  "
              f"비 {lap[i]/gau[i]:.1f}")
    clean(ax1)

    # --- (나) 최빈값은 0, 평균은 0이 아니다
    s, bpar, zv = 0.5, 0.5, 0.2
    gg = np.linspace(-1.6, 1.8, 4000)
    post = np.exp(-0.5 * ((gg - zv) / s) ** 2 - np.abs(gg) / bpar)
    post /= np.trapz(post, gg)
    mean = np.trapz(gg * post, gg)
    mode = gg[int(np.argmax(post))]
    lik = np.exp(-0.5 * ((gg - zv) / s) ** 2)
    lik = lik / np.trapz(lik, gg)
    print(f"[posterior] 문턱 s^2/b={s**2/bpar:.2f}  z={zv}  "
          f"최빈값 {mode:.3f}  사후평균 {mean:.4f}")

    ax2.fill_between(gg, post, color=GREEN_F, alpha=0.6)
    ax2.plot(gg, post, lw=2.4, color=GREEN, label="사후밀도")
    ax2.plot(gg, lik, lw=1.8, color=MUTED, ls="--", label="가능도")
    ax2.axvline(0, color=RED, lw=1.8, ls=":")
    ax2.axvline(mean, color=PURPLE, lw=1.8, ls="-.")
    ax2.set_xlim(-1.6, 1.8)
    ax2.set_ylim(0, 1.35)
    ax2.set_xlabel(r'$\beta_j$', fontsize=12, color=INK)
    ax2.set_ylabel("밀도", fontsize=11, color=INK)
    ax2.set_title(r'(나) 약한 신호 ($z=0.2$) 의 사후분포', fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax2.annotate(f"최빈값 = 0\n(라쏘가 주는 값)", xy=(0, 1.02),
                 xytext=(-1.52, 1.12), fontsize=10.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax2.annotate(f"사후평균 = {mean:.3f}\n(0이 아니다)", xy=(mean, 0.45),
                 xytext=(0.55, 0.72), fontsize=10.5, color=PURPLE,
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.2))
    clean(ax2)

    fig.tight_layout()
    save(fig, "laplace_prior.png")


# ==================================================================
# 4. 좌표하강   (coordinate_descent.md)
# ==================================================================
def coord_descent():
    # --- (가) 2차원 지그재그
    rng = np.random.default_rng(3)
    n = 60
    z = rng.normal(size=n)
    X = np.c_[0.8 * z + 0.6 * rng.normal(size=n),
              0.8 * z + 0.6 * rng.normal(size=n)]
    X = (X - X.mean(0)) / X.std(0)
    y = X @ np.array([1.2, 1.0]) + rng.normal(0, 1, n)
    y = y - y.mean()
    lam = 0.25

    def obj(B1, B2):
        r = y[:, None] - X @ np.c_[B1.ravel(), B2.ravel()].T
        return ((r ** 2).sum(axis=0) / (2 * n)
                + lam * (np.abs(B1.ravel()) + np.abs(B2.ravel()))
                ).reshape(B1.shape)

    g = np.linspace(-0.4, 2.6, 300)
    B1, B2 = np.meshgrid(g, g)
    Z = obj(B1, B2)

    beta = np.array([2.4, -0.25])
    pts = [beta.copy()]
    r = y - X @ beta
    for it in range(9):
        for j in range(2):
            r = r + X[:, j] * beta[j]
            zj = X[:, j] @ r / n
            beta[j] = soft(zj, lam)
            r = r - X[:, j] * beta[j]
            pts.append(beta.copy())
    pts = np.array(pts)
    sol = Lasso(alpha=lam, fit_intercept=False, max_iter=200000).fit(X, y).coef_
    print(f"[cd] 해 {sol.round(3)}  내 구현 {pts[-1].round(3)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    lo = Z.min()
    ax1.contourf(B1, B2, Z, levels=lo + np.array([0, .02, .06, .14, .28, .5,
                                                  .85, 1.4, 2.2, 3.4]),
                 cmap="Blues_r", alpha=0.6, extend="min")
    ax1.contour(B1, B2, Z, levels=lo + np.array([.02, .06, .14, .28, .5, .85,
                                                 1.4, 2.2]),
                colors=[MUTED], linewidths=0.6)
    ax1.plot(pts[:, 0], pts[:, 1], "-o", lw=2.0, ms=5, color=ORANGE,
             mec="white", mew=0.8, zorder=4)
    ax1.plot([pts[0, 0]], [pts[0, 1]], "s", ms=10, color=RED, mec="white",
             mew=1.2, zorder=5)
    ax1.plot([sol[0]], [sol[1]], "*", ms=18, color=GREEN, mec="white", mew=1.2,
             zorder=6)
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.axvline(0, color=INK, lw=0.9)
    ax1.set_xlim(-0.4, 2.6)
    ax1.set_ylim(-0.4, 2.6)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r'$\beta_1$', fontsize=12, color=INK)
    ax1.set_ylabel(r'$\beta_2$', fontsize=12, color=INK)
    ax1.set_title(r'(가) 한 번에 한 좌표씩 ($\lambda=0.25$)', fontsize=12,
                  color=INK)
    ax1.text(pts[0, 0] - 0.1, pts[0, 1] - 0.22, "출발점", fontsize=10.5,
             color=RED, ha="center")
    ax1.annotate(f"해 ({sol[0]:.2f}, {sol[1]:.2f})", xy=(sol[0], sol[1]),
                 xytext=(1.35, 2.25), fontsize=10.5, color=GREEN,
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))
    clean(ax1)

    # --- (나) 수렴 속도
    rng2 = np.random.default_rng(4)
    n2, p2 = 80, 10
    X2 = rng2.normal(size=(n2, p2))
    X2 -= X2.mean(0)
    X2 /= X2.std(0)
    beta2 = np.zeros(p2)
    beta2[:3] = [3, -2, 1.5]
    y2 = X2 @ beta2 + rng2.normal(0, 1, n2)
    y2 = y2 - y2.mean()

    for lm, col in [(0.05, BLUE), (0.20, ORANGE), (0.50, GREEN)]:
        b = np.zeros(p2)
        r = y2.copy()
        hist = []
        for it in range(40):
            mx = 0.0
            for j in range(p2):
                r = r + X2[:, j] * b[j]
                zj = X2[:, j] @ r / n2
                nb = soft(zj, lm)
                mx = max(mx, abs(nb - b[j]))
                b[j] = nb
                r = r - X2[:, j] * b[j]
            hist.append(max(mx, 1e-17))
            if mx < 1e-10:
                break
        nz = int((np.abs(b) > 1e-9).sum())
        ax2.plot(np.arange(1, len(hist) + 1), hist, "-o", lw=2.0, ms=5,
                 color=col, mec="white", mew=0.8,
                 label=f"$\\lambda={lm}$  ({len(hist)}회, 활성 {nz}개)")
        print(f"  lam={lm}: {len(hist)}회 순회, 활성 {nz}개")
    ax2.axhline(1e-10, color=RED, ls="--", lw=1.4)
    ax2.text(18.6, 3e-9, "허용오차 1e-10", fontsize=10, color=RED, ha="right")
    ax2.set_yscale("log")
    ax2.set_yticks([1e0, 1e-3, 1e-6, 1e-9, 1e-12])
    ax2.set_yticklabels(["1", "1e-3", "1e-6", "1e-9", "1e-12"])
    ax2.yaxis.set_minor_locator(NullLocator())
    ax2.set_ylim(1e-13, 20)
    ax2.set_xlim(0, 19)
    ax2.set_xticks([1, 5, 10, 15])
    ax2.set_xlabel("순회 횟수", fontsize=11, color=INK)
    ax2.set_ylabel(r'그 순회에서의 최대 변화량 (로그 눈금)', fontsize=11,
                   color=INK)
    ax2.set_title(r'(나) $n=80$, $p=10$ 에서의 수렴', fontsize=12, color=INK)
    ax2.legend(fontsize=10, loc="upper right", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "coord_descent.png")


# ==================================================================
# 5. 라쏘의 편향과 사후 라쏘   (feature_selection.md)
# ==================================================================
def post_lasso_bias():
    rng = np.random.default_rng(0)
    n, p, lam, R = 80, 10, 0.2, 400
    beta = np.array([3.0, -2.0, 1.5] + [0.0] * 7)
    las, post = [], []
    for _ in range(R):
        X = rng.normal(0, 1, (n, p))
        y = X @ beta + rng.normal(0, 1, n)
        b = Lasso(alpha=lam, fit_intercept=False, max_iter=50000).fit(X, y).coef_
        S = np.abs(b) > 1e-9
        bp = np.zeros(p)
        bp[S] = LinearRegression(fit_intercept=False).fit(X[:, S], y).coef_
        las.append(b)
        post.append(bp)
    las, post = np.array(las), np.array(post)
    ml, mp = las[:, 0].mean(), post[:, 0].mean()
    rl = np.sqrt(((las[:, :3] - beta[:3]) ** 2).mean(0))
    rp = np.sqrt(((post[:, :3] - beta[:3]) ** 2).mean(0))
    print(f"[post] 라쏘 평균 {las[:, :3].mean(0).round(3)}  "
          f"사후 {post[:, :3].mean(0).round(3)}")
    print(f"  RMSE 라쏘 {rl.round(3)}  사후 {rp.round(3)}")

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    bins = np.linspace(2.2, 3.6, 46)
    ax.hist(las[:, 0], bins=bins, color=BLUE, alpha=0.65, label="라쏘")
    ax.hist(post[:, 0], bins=bins, color=ORANGE, alpha=0.65,
            label="사후 라쏘 OLS")
    ax.axvline(3.0, color=RED, lw=2.0, ls="--", zorder=5)
    ax.axvline(ml, color=BLUE, lw=1.8, zorder=5)
    ax.axvline(mp, color=ORANGE, lw=1.8, zorder=5)
    ax.annotate("", xy=(ml, 50), xytext=(3.0, 50),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text((ml + 3.0) / 2, 52, r'편향 $=-\lambda=-0.2$', fontsize=11,
            color=INK, ha="center")
    ax.text(3.04, 57, r'참값 $\beta_1=3.0$', fontsize=11, color=RED)
    ax.set_ylim(0, 64)
    ax.set_xlim(2.2, 3.6)
    ax.set_xlabel(r'$\hat\beta_1$   (400개 자료, $n=80$, $p=10$, $\lambda=0.2$)',
                  fontsize=11, color=INK)
    ax.set_ylabel("빈도", fontsize=11, color=INK)
    ax.set_title("라쏘의 편향은 우연이 아니라 설계다", fontsize=12.5, color=INK)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax.text(2.24, 40, f"라쏘 평균 {ml:.3f}   RMSE {rl[0]:.3f}\n"
                      f"사후 평균 {mp:.3f}   RMSE {rp[0]:.3f}",
            fontsize=10.5, color=INK, va="top")
    clean(ax)
    fig.tight_layout()
    save(fig, "post_lasso_bias.png")


# ==================================================================
# 6. 세 축소 연산자   (lasso.md)
# ==================================================================
def shrinkage_operators():
    lam = 1.0
    z = np.linspace(-3.2, 3.2, 800)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8))

    ax1.plot(z, z, lw=1.4, color=MUTED, ls=":", zorder=2)
    ax1.plot(z, z / (1 + lam), lw=2.4, color=BLUE, zorder=3, label="능형 (비례)")
    ax1.plot(z, soft(z, lam), lw=2.4, color=ORANGE, zorder=4,
             label="라쏘 (연성 문턱)")
    hard = z * (np.abs(z) > lam)
    hard_m = np.where(np.abs(np.abs(z) - lam) < 0.012, np.nan, hard)
    ax1.plot(z, hard_m, lw=2.0, color=GREEN, zorder=3,
             label="최량 부분집합 (경성 문턱)")
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.axvline(0, color=INK, lw=0.9)
    ax1.fill_betweenx([-3.2, 3.2], -lam, lam, color=ORANGE_F, alpha=0.45,
                      zorder=1)
    ax1.text(0, -2.95, r'$|z|\leq\lambda$ 면 0', fontsize=10.5, color=ORANGE,
             ha="center")
    ax1.set_xlim(-3.2, 3.2)
    ax1.set_ylim(-3.2, 3.2)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r'OLS 계수 $\hat\beta_j^{\,\mathrm{OLS}}$', fontsize=11,
                   color=INK)
    ax1.set_ylabel("벌점 추정값", fontsize=11, color=INK)
    ax1.set_title(r'(가) 세 축소 연산자 ($\lambda=1$)', fontsize=12, color=INK)
    ax1.legend(fontsize=10, loc="upper left", frameon=False)
    clean(ax1)

    rng = np.random.default_rng(6)
    p = 12
    ols = np.sort(rng.normal(0, 1.4, p))[::-1]
    ols[0], ols[1] = 2.9, -2.4
    las = soft(ols, lam)
    rid = ols / (1 + lam)
    yy = np.arange(p)[::-1]
    for i in range(p):
        ax2.annotate("", xy=(las[i], yy[i]), xytext=(ols[i], yy[i]),
                     arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.6,
                                     shrinkA=3, shrinkB=3))
    ax2.plot(ols, yy, "o", ms=8, color=INK, mec="white", mew=1.0, zorder=5,
             label="OLS")
    ax2.plot(rid, yy, "s", ms=7, color=BLUE, mec="white", mew=1.0, zorder=5,
             label="능형")
    ax2.plot(las, yy, "D", ms=7, color=ORANGE, mec="white", mew=1.0, zorder=6,
             label="라쏘")
    ax2.axvline(0, color=INK, lw=1.0)
    ax2.axvspan(-lam, lam, color=ORANGE_F, alpha=0.4, zorder=0)
    nz = int((np.abs(las) > 1e-12).sum())
    ax2.set_yticks([])
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-1.2, p - 0.2)
    ax2.set_xlabel("계수값", fontsize=11, color=INK)
    ax2.set_title(f"(나) 계수 12개에 적용: {p - nz}개가 정확히 0", fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10, loc="lower right", frameon=False, ncol=3)
    print(f"[op] OLS {np.round(ols, 2)}")
    print(f"     라쏘 {np.round(las, 2)}  0의 개수 {p - nz}")
    clean(ax2)
    ax2.spines["left"].set_visible(False)

    fig.tight_layout()
    save(fig, "shrinkage_operators.png")


# ==================================================================
# 7. 경로와 교차검증   (lasso_examples.md)
# ==================================================================
def path_cv():
    np.random.seed(42)
    n, p = 150, 10
    Xr = np.random.randn(n, p)
    bt = np.array([4.0, -3.0, 2.0] + [0.0] * 7)
    y = Xr @ bt + np.random.randn(n) * 2
    X = (Xr - Xr.mean(0)) / Xr.std(0)
    yc = y - y.mean()

    lams = np.logspace(1, -2, 120)
    coefs = np.array([Lasso(alpha=l, fit_intercept=False, max_iter=100000)
                      .fit(X, yc).coef_ for l in lams])

    from sklearn.model_selection import KFold
    lams_cv = np.logspace(1, -2, 30)
    kf = KFold(5, shuffle=True, random_state=0)
    mse = []
    for l in lams_cv:
        e = []
        for tr, te in kf.split(X):
            b = Lasso(alpha=l, fit_intercept=False, max_iter=100000) \
                .fit(X[tr], yc[tr]).coef_
            e.append(((yc[te] - X[te] @ b) ** 2).mean())
        mse.append(np.mean(e))
    mse = np.array(mse)
    k = int(np.argmin(mse))
    best = lams_cv[k]
    bb = Lasso(alpha=best, fit_intercept=False, max_iter=100000).fit(X, yc).coef_
    print(f"[path_cv] 최적 lambda {best:.4f}  CV MSE {mse[k]:.3f}  "
          f"계수 {np.round(bb[:3], 3)}  활성 {(np.abs(bb) > 1e-8).sum()}")
    b_ref = Lasso(alpha=0.2212, fit_intercept=False, max_iter=100000) \
        .fit(X, yc).coef_
    print(f"  lambda=0.2212 에서 {np.round(b_ref[:3], 3)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    cols = [BLUE, ORANGE, GREEN] + [MUTED] * 7
    for j in range(p):
        ax1.plot(lams, coefs[:, j], lw=2.2 if j < 3 else 1.2, color=cols[j])
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.axvline(best, color=PURPLE, ls="--", lw=1.8)
    for j, nm in enumerate([r'$x_1$ (참 $4.0$)', r'$x_2$ (참 $-3.0$)',
                            r'$x_3$ (참 $2.0$)']):
        dy = 0.36 if coefs[-1, j] >= 0 else -0.36
        ax1.text(0.0105, coefs[-1, j] + dy, nm, fontsize=10.5, color=cols[j],
                 va="bottom" if dy > 0 else "top")
    logx_plain(ax1, [0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax1.set_xlim(0.01, 10)
    ax1.set_ylim(-3.6, 4.6)
    ax1.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel("계수 추정값", fontsize=11, color=INK)
    ax1.set_title("(가) 정칙화 경로", fontsize=12, color=INK)
    ax1.text(best * 1.25, 4.2, f"$\\hat\\lambda={best:.4f}$", fontsize=10.5,
             color=PURPLE)
    ax1.text(1.1, -3.3, "잡음변수 7개는\n끝까지 0 근처", fontsize=10,
             color=MUTED)
    clean(ax1)

    ax2.plot(lams_cv, mse, "-o", lw=2.2, ms=5, color=BLUE, mec="white",
             mew=0.8)
    ax2.plot([best], [mse[k]], "o", ms=11, color=PURPLE, mec="white", mew=1.4,
             zorder=5)
    ax2.axvline(best, color=PURPLE, ls="--", lw=1.8)
    ax2.axhline(4.0, color=RED, ls=":", lw=1.4)
    ax2.text(9.0, 4.3, r'잡음 바닥 $\sigma^2=4$', fontsize=10.5, color=RED,
             ha="right")
    logx_plain(ax2, [0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax2.set_xlim(0.01, 10)
    ax2.set_ylim(3.5, 30)
    ax2.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel("5-겹 교차검증 MSE", fontsize=11, color=INK)
    ax2.set_title("(나) 교차검증 곡선", fontsize=12, color=INK)
    ax2.annotate(f"최소 CV MSE {mse[k]:.3f}", xy=(best, mse[k]),
                 xytext=(best * 1.3, 15.0), fontsize=10.5, color=PURPLE,
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3))
    clean(ax2)

    fig.tight_layout()
    save(fig, "path_cv.png")


# ==================================================================
# 8. 주택자료 경로   (lasso_housing_regularization_path.md)
# ==================================================================
def housing_path():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LassoCV

    url = ("https://raw.githubusercontent.com/gedeck/"
           "practical-statistics-for-data-scientists/master/data/"
           "house_sales.csv")
    house = pd.read_csv(url, sep="\t")
    predictors = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms',
                  'BldgGrade', 'PropertyType', 'NbrLivingUnits',
                  'SqFtFinBasement', 'YrBuilt', 'YrRenovated',
                  'NewConstruction']
    X = pd.get_dummies(house[predictors], drop_first=True)
    X['NewConstruction'] = X['NewConstruction'].astype(int)
    y = house['AdjSalePrice'].to_numpy(float)
    names = list(X.columns)
    Xs = StandardScaler().fit_transform(X.to_numpy(float))

    alphas = np.logspace(5, 1, 70)
    coefs = np.array([Lasso(alpha=a, max_iter=20000).fit(Xs, y).coef_
                      for a in alphas])
    nact = (np.abs(coefs) > 1e-8).sum(axis=1)
    cv = LassoCV(alphas=np.logspace(5, 1, 60), cv=5, random_state=42,
                 max_iter=20000).fit(Xs, y)
    best = cv.alpha_
    nz = int((np.abs(cv.coef_) > 1e-8).sum())
    order = np.argsort(-np.abs(cv.coef_))
    print(f"[housing] p={len(names)}  최적 alpha {best:.1f}  활성 {nz}개")
    for i in order[:6]:
        print(f"   {names[i]:24s} {cv.coef_[i]:12.1f}")
    for i in order[-4:]:
        print(f"   {names[i]:24s} {cv.coef_[i]:12.1f}")
    first = {names[j]: alphas[np.abs(coefs[:, j]) > 1e-8][0]
             for j in range(len(names)) if (np.abs(coefs[:, j]) > 1e-8).any()}
    print("  진입 alpha 큰 순:",
          sorted(((round(v), k) for k, v in first.items()), reverse=True)[:5])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))
    big = order[:4]
    palette = {int(big[0]): BLUE, int(big[1]): ORANGE, int(big[2]): GREEN,
               int(big[3]): PURPLE}
    for j in range(len(names)):
        c = palette.get(j, MUTED)
        ax1.plot(alphas, coefs[:, j], lw=2.2 if j in big else 1.1, color=c,
                 label=names[j] if j in big else None)
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.axvline(best, color=RED, ls="--", lw=1.8)
    ax1.set_ylim(-135000, 215000)
    ax1.legend(fontsize=10, loc="center left", frameon=False,
               bbox_to_anchor=(0.02, 0.62))
    logx_plain(ax1, [10, 100, 1000, 10000, 100000],
               ["10", "100", "1000", "10000", "100000"])
    ax1.set_xlim(10, 1e5)
    ax1.set_xlabel(r'벌점 $\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel("표준화 계수 (달러)", fontsize=11, color=INK)
    ax1.set_title("(가) 주택가격 자료의 라쏘 경로", fontsize=12, color=INK)
    ax1.text(best * 1.35, 120000, f"교차검증 $\\hat\\lambda={best:.0f}$",
             fontsize=10.5, color=RED)
    clean(ax1)

    ax2.step(alphas, nact, where="post", lw=2.4, color=BLUE)
    ax2.axvline(best, color=RED, ls="--", lw=1.8)
    ax2.plot([best], [nz], "o", ms=10, color=RED, mec="white", mew=1.4,
             zorder=5)
    logx_plain(ax2, [10, 100, 1000, 10000, 100000],
               ["10", "100", "1000", "10000", "100000"])
    ax2.set_xlim(10, 1e5)
    ax2.set_ylim(-0.5, len(names) + 0.8)
    ax2.set_yticks(range(0, len(names) + 1, 2))
    ax2.set_xlabel(r'벌점 $\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel("0이 아닌 계수의 개수", fontsize=11, color=INK)
    ax2.set_title("(나) 살아남은 변수의 개수", fontsize=12, color=INK)
    ax2.annotate(f"{nz}개 선택", xy=(best, nz), xytext=(45, nz - 4.2),
                 fontsize=10.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax2.text(1.4e4, len(names) - 0.4, f"전체 {len(names)}개", fontsize=10.5,
             color=INK, ha="center")
    clean(ax2)

    fig.tight_layout()
    save(fig, "housing_path.png")


if __name__ == "__main__":
    l1_vs_l2_tangency()
    soft_threshold_min()
    laplace_prior()
    coord_descent()
    post_lasso_bias()
    shrinkage_operators()
    path_cv()
    try:
        housing_path()
    except Exception as exc:          # 인터넷이 없으면 건너뛴다
        print("housing_path 건너뜀:", exc)
