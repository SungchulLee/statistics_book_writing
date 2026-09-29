r"""18.4 엘라스틱넷 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/elastic_net/img/en_geometry.png    둥근 마름모와 갱신 연산자
  ch18/elastic_net/img/group_paths.png    상관 집단에서 갈라지는 경로와 겹치는 경로
  ch18/elastic_net/img/saturation.png     p>n 에서 라쏘의 n 개 벽
  ch18/elastic_net/img/alpha_lambda_cv.png  두 초모수의 교차검증 지형
  ch18/elastic_net/img/group_split.png    집단을 어떻게 나누는가

실행:  python3 scripts/make_ch18_elastic_net_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from sklearn.linear_model import Lasso, ElasticNet

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/elastic_net/img/"
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


def two_groups(n=100, seed=0, rho=0.95, p=20):
    r = np.random.default_rng(seed)
    z1, z2 = r.normal(size=n), r.normal(size=n)
    X = r.normal(size=(n, p))
    for j in (0, 1, 2):
        X[:, j] = np.sqrt(rho) * z1 + np.sqrt(1 - rho) * X[:, j]
    for j in (3, 4, 5):
        X[:, j] = np.sqrt(rho) * z2 + np.sqrt(1 - rho) * X[:, j]
    X -= X.mean(0)
    X /= X.std(0)
    beta = np.zeros(p)
    beta[:6] = 1.5
    y = X @ beta + r.normal(0, 1, n)
    return X, y - y.mean()


# ==================================================================
# 1. 둥근 마름모와 갱신 연산자   (formulation.md)
# ==================================================================
def en_geometry():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.2))

    th = np.linspace(0, 2 * np.pi, 1200)
    for a, col, lab in [(1.0, ORANGE, r'$\alpha=1$ (라쏘)'),
                        (0.6, PURPLE, r'$\alpha=0.6$'),
                        (0.3, GREEN, r'$\alpha=0.3$'),
                        (0.0, BLUE, r'$\alpha=0$ (능형)')]:
        t = a + (1 - a) / 2                      # (1,0) 을 지나도록 맞춘다
        u, v = np.cos(th), np.sin(th)
        s = np.empty_like(th)
        for i in range(len(th)):
            A = (1 - a) / 2 * (u[i] ** 2 + v[i] ** 2)
            B = a * (abs(u[i]) + abs(v[i]))
            s[i] = (t / B if A == 0 else (-B + np.sqrt(B ** 2 + 4 * A * t))
                    / (2 * A))
        ax1.plot(s * u, s * v, lw=2.4, color=col, label=lab)
    ax1.plot([1, 0, -1, 0, 1], [0, 1, 0, -1, 0], "o", ms=5, color=INK,
             zorder=5)
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.axvline(0, color=INK, lw=0.8)
    ax1.set_xlim(-1.45, 1.45)
    ax1.set_ylim(-1.45, 1.45)
    ax1.set_aspect("equal")
    ax1.set_xlabel(r'$\beta_1$', fontsize=12, color=INK)
    ax1.set_ylabel(r'$\beta_2$', fontsize=12, color=INK)
    ax1.set_title("(가) 같은 꼭짓점을 지나도록 맞춘 제약영역", fontsize=12,
                  color=INK)
    ax1.legend(fontsize=10, loc="upper left", frameon=False,
               bbox_to_anchor=(0.0, 1.0))
    ax1.annotate("꼭짓점은 남고\n변만 둥글어진다", xy=(0.62, 0.62),
                 xytext=(0.72, -1.2), fontsize=10.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    clean(ax1)

    lam = 1.0
    z = np.linspace(-3.2, 3.2, 800)
    ax2.plot(z, z, lw=1.3, color=MUTED, ls=":", zorder=2)
    for a, col, lab in [(1.0, ORANGE, r'$\alpha=1$ (라쏘)'),
                        (0.6, PURPLE, r'$\alpha=0.6$'),
                        (0.3, GREEN, r'$\alpha=0.3$'),
                        (0.0, BLUE, r'$\alpha=0$ (능형)')]:
        ax2.plot(z, soft(z, a * lam) / (1 + (1 - a) * lam), lw=2.4, color=col,
                 label=lab, zorder=3)
        print(f"  alpha={a}: z=2 -> {soft(2, a*lam)/(1+(1-a)*lam):.3f}  "
              f"문턱 {a*lam:.2f}  기울기 {1/(1+(1-a)*lam):.3f}")
    ax2.axhline(0, color=INK, lw=0.8)
    ax2.axvline(0, color=INK, lw=0.8)
    ax2.set_xlim(-3.2, 3.2)
    ax2.set_ylim(-3.2, 3.2)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r'$z_j$ (부분잔차 회귀계수)', fontsize=11, color=INK)
    ax2.set_ylabel(r'갱신된 $\hat\beta_j$', fontsize=11, color=INK)
    ax2.set_title(r'(나) 좌표하강 갱신 $S_{\alpha\lambda}(z)/[1+(1-\alpha)\lambda]$'
                  "\n" r'($\lambda=1$)', fontsize=12, color=INK)
    ax2.legend(fontsize=10, loc="upper left", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "en_geometry.png")


# ==================================================================
# 2. 상관 집단의 경로   (grouping_effect.md)
# ==================================================================
def group_paths():
    X, y = two_groups(seed=0)
    lams = np.logspace(0.2, -2, 120)
    P = {}
    P["lasso"] = np.array([Lasso(alpha=l, fit_intercept=False, max_iter=200000)
                           .fit(X, y).coef_ for l in lams])
    P["enet"] = np.array([ElasticNet(alpha=l, l1_ratio=0.5,
                                     fit_intercept=False, max_iter=200000)
                          .fit(X, y).coef_ for l in lams])
    for k in ("lasso", "enet"):
        C = P[k]
        i = np.argmin(np.abs(lams - 0.3))
        print(f"[{k}] lam=0.3 에서 1집단 {C[i, :3].round(3)}  "
              f"2집단 {C[i, 3:6].round(3)}")
        print(f"      활성 {int((np.abs(C[i]) > 1e-8).sum())}개, "
              f"집단내 표준편차 {C[i, :3].std():.3f} / {C[i, 3:6].std():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    for ax, key, ttl in [(axes[0], "lasso", r'(가) 라쏘 ($\alpha=1$)'),
                         (axes[1], "enet",
                          r'(나) 엘라스틱넷 ($\alpha=0.5$)')]:
        C = P[key]
        for j in range(3):
            ax.plot(lams, C[:, j], lw=2.2, color=BLUE, zorder=4)
        for j in range(3, 6):
            ax.plot(lams, C[:, j], lw=2.2, color=ORANGE, zorder=4)
        for j in range(6, 20):
            ax.plot(lams, C[:, j], lw=1.0, color=MUTED, zorder=2)
        ax.axhline(0, color=INK, lw=0.9)
        logx_plain(ax, [0.01, 0.1, 1], ["0.01", "0.1", "1"])
        ax.set_xlim(lams[-1], lams[0])
        ax.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
        ax.set_title(ttl, fontsize=12, color=INK)
        clean(ax)
    axes[0].set_ylabel("계수 추정값", fontsize=11, color=INK)
    axes[0].set_ylim(-0.3, 2.35)
    axes[0].text(0.011, 2.30, "파랑: 1집단 ($x_1,x_2,x_3$)   "
                              "주황: 2집단 ($x_4,x_5,x_6$)   "
                              "회색: 잡음변수 14개", fontsize=10, color=INK,
                 va="top")
    axes[0].annotate("집단 안에서 갈라진다", xy=(0.3, 1.2), xytext=(0.013, 0.42),
                     fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[1].annotate("집단이 겹쳐서 간다", xy=(0.3, 1.34), xytext=(0.013, 0.42),
                     fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    fig.tight_layout()
    save(fig, "group_paths.png")


# ==================================================================
# 3. p > n 에서의 포화   (advantages.md)
# ==================================================================
def saturation():
    rng = np.random.default_rng(77)
    n, p = 40, 200
    X = rng.normal(size=(n, p))
    X -= X.mean(0)
    X /= X.std(0)
    beta = np.zeros(p)
    beta[:60] = 1.0
    y = X @ beta + rng.normal(0, 1, n)
    y = y - y.mean()

    alphas = np.array([1.0, 0.9, 0.7, 0.5, 0.3, 0.1, 0.05])
    counts, hits = [], []
    for a in alphas:
        if a == 1.0:
            b = Lasso(alpha=0.05, fit_intercept=False, max_iter=500000,
                      tol=1e-12).fit(X, y).coef_
        else:
            b = ElasticNet(alpha=0.05, l1_ratio=a, fit_intercept=False,
                           max_iter=500000, tol=1e-12).fit(X, y).coef_
        m = np.abs(b) > 1e-8
        counts.append(int(m.sum()))
        hits.append(int(m[:60].sum()))
        print(f"  alpha={a}: 선택 {m.sum()}개, 참 변수 적중 {m[:60].sum()}/60")

    grid = np.logspace(-0.3, -2.3, 20)
    mx = max(int((np.abs(Lasso(alpha=l, fit_intercept=False, max_iter=500000,
                               tol=1e-12).fit(X, y).coef_) > 1e-8).sum())
             for l in grid)
    print(f"  라쏘가 격자 전체에서 고른 최대 변수 수: {mx}")

    x = np.arange(len(alphas))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.bar(x - 0.2, counts, 0.4, color=BLUE, label="선택된 변수 수", zorder=3)
    ax.bar(x + 0.2, hits, 0.4, color=GREEN, label="그중 참 변수", zorder=3)
    ax.axhline(n - 1, color=RED, ls="--", lw=1.8, zorder=4)
    ax.axhline(60, color=MUTED, ls=":", lw=1.6, zorder=2)
    ax.text(-0.45, n + 1.5, r'라쏘의 벽 $n-1=39$', fontsize=10.5,
            color=RED, ha="left")
    ax.text(-0.45, 62, "참 변수 60개", fontsize=10.5, color=INK, ha="left")
    for i, (c, h) in enumerate(zip(counts, hits)):
        ax.text(x[i] - 0.2, c + 2, str(c), fontsize=10, color=BLUE,
                ha="center")
        ax.text(x[i] + 0.2, h + 2, str(h), fontsize=10, color=GREEN,
                ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:g}" for a in alphas])
    ax.set_xlabel(r'혼합모수 $\alpha$   (1이 라쏘, 0이 능형)', fontsize=11,
                  color=INK)
    ax.set_ylabel("변수 개수", fontsize=11, color=INK)
    ax.set_title(r'$n=40$, $p=200$, 참 변수 60개에서의 선택 개수',
                 fontsize=12.5, color=INK)
    ax.set_ylim(0, max(counts) * 1.22)
    ax.legend(fontsize=10.5, loc="upper left", frameon=False)
    clean(ax)
    fig.tight_layout()
    save(fig, "saturation.png")


# ==================================================================
# 4. 두 초모수의 교차검증 지형   (elastic_net.md)
# ==================================================================
def alpha_lambda_cv():
    from sklearn.model_selection import KFold
    X, y = two_groups(n=80, seed=5)
    lams = np.logspace(0.1, -2.2, 22)
    alphas = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 1.0])
    kf = KFold(5, shuffle=True, random_state=0)
    M = np.zeros((len(alphas), len(lams)))
    for i, a in enumerate(alphas):
        for j, l in enumerate(lams):
            e = []
            for tr, te in kf.split(X):
                if a == 1.0:
                    b = Lasso(alpha=l, fit_intercept=False, max_iter=100000) \
                        .fit(X[tr], y[tr]).coef_
                else:
                    b = ElasticNet(alpha=l, l1_ratio=a, fit_intercept=False,
                                   max_iter=100000).fit(X[tr], y[tr]).coef_
                e.append(((y[te] - X[te] @ b) ** 2).mean())
            M[i, j] = np.mean(e)
    ia, il = np.unravel_index(np.argmin(M), M.shape)
    print(f"[grid] 최적 alpha={alphas[ia]}  lambda={lams[il]:.4f}  "
          f"CV={M[ia, il]:.3f}")
    print(f"  라쏘 최선 {M[-1].min():.3f} (lam {lams[M[-1].argmin()]:.3f})  "
          f"alpha=0.05 최선 {M[0].min():.3f}")
    print(f"  전체 최악 {M.max():.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    Lg, Ag = np.meshgrid(lams, alphas)
    pc = ax1.pcolormesh(Lg, Ag, M, cmap="YlGnBu", shading="gouraud",
                        vmin=M.min(), vmax=np.percentile(M, 80))
    cb = fig.colorbar(pc, ax=ax1)
    cb.set_label("5-겹 교차검증 MSE", fontsize=10.5, color=INK)
    cb.ax.tick_params(labelsize=9.5, colors=INK)
    ax1.plot([lams[il]], [alphas[ia]], "*", ms=20, color=RED, mec="white",
             mew=1.2)
    logx_plain(ax1, [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax1.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_ylabel(r'혼합모수 $\alpha$', fontsize=11, color=INK)
    ax1.set_title("(가) 두 초모수 위의 CV 오차 지형", fontsize=12, color=INK)
    ax1.annotate(f"최적 ({lams[il]:.3f}, {alphas[ia]:g})",
                 xy=(lams[il], alphas[ia]), xytext=(0.012, 0.86),
                 fontsize=10.5, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    clean(ax1)

    for i, (a, col) in enumerate([(1.0, ORANGE), (0.5, PURPLE),
                                  (0.1, BLUE)]):
        k = int(np.argmin(np.abs(alphas - a)))
        ax2.plot(lams, M[k], lw=2.4, color=col,
                 label=f"$\\alpha={alphas[k]:g}$  (최선 {M[k].min():.3f})")
        ax2.plot([lams[M[k].argmin()]], [M[k].min()], "o", ms=8, color=col,
                 mec="white", mew=1.2, zorder=5)
    logx_plain(ax2, [0.01, 0.1, 1], ["0.01", "0.1", "1"])
    ax2.set_xlabel(r'$\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel("5-겹 교차검증 MSE", fontsize=11, color=INK)
    ax2.set_title(r'(나) $\alpha$ 를 고정하고 자른 단면', fontsize=12,
                  color=INK)
    ax2.set_ylim(M.min() * 0.95, M.min() * 3.2)
    ax2.legend(fontsize=10, loc="upper left", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "alpha_lambda_cv.png")


# ==================================================================
# 5. 집단을 어떻게 나누는가   (elastic_net_examples.md)
# ==================================================================
def group_split():
    rng = np.random.default_rng(42)
    n = 100
    z = rng.normal(size=n)
    X = np.column_stack([z + rng.normal(0, 0.05, n),
                         z + rng.normal(0, 0.05, n),
                         z + rng.normal(0, 0.05, n),
                         rng.normal(size=(n, 3))])
    y = 3 * z + rng.normal(0, 1, n)
    bl = Lasso(alpha=0.5).fit(X, y).coef_
    be = ElasticNet(alpha=0.5, l1_ratio=0.5).fit(X, y).coef_
    print(f"[one] 라쏘 {bl.round(3)}  합 {bl[:3].sum():.3f}  sd {bl[:3].std():.3f}")
    print(f"      EN   {be.round(3)}  합 {be[:3].sum():.3f}  sd {be[:3].std():.3f}")

    R = 200
    sl, se = [], []
    for s in range(R):
        r = np.random.default_rng(1000 + s)
        zz = r.normal(size=n)
        XX = np.column_stack([zz + r.normal(0, 0.05, n),
                              zz + r.normal(0, 0.05, n),
                              zz + r.normal(0, 0.05, n),
                              r.normal(size=(n, 3))])
        yy = 3 * zz + r.normal(0, 1, n)
        sl.append(Lasso(alpha=0.5).fit(XX, yy).coef_[:3].std())
        se.append(ElasticNet(alpha=0.5, l1_ratio=0.5).fit(XX, yy)
                  .coef_[:3].std())
    sl, se = np.array(sl), np.array(se)
    print(f"[rep] 집단내 표준편차 중앙값: 라쏘 {np.median(sl):.3f}  "
          f"EN {np.median(se):.3f}  최대 라쏘 {sl.max():.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    idx = np.arange(6)
    ax1.bar(idx - 0.2, bl, 0.4, color=BLUE, label="라쏘", zorder=3)
    ax1.bar(idx + 0.2, be, 0.4, color=ORANGE, label="엘라스틱넷", zorder=3)
    ax1.axvspan(-0.6, 2.6, color=GREEN_F, alpha=0.5, zorder=0)
    ax1.axhline(0, color=INK, lw=0.9)
    ax1.set_xticks(idx)
    ax1.set_xticklabels([f"$x_{i}$" for i in range(6)])
    ax1.set_ylim(-0.05, 1.18)
    ax1.set_xlabel("설명변수", fontsize=11, color=INK)
    ax1.set_ylabel("계수 추정값", fontsize=11, color=INK)
    ax1.set_title(r'(가) 한 자료에서 ($\lambda=0.5$)', fontsize=12, color=INK)
    ax1.text(1.0, 1.12, "거의 같은 변수 셋", fontsize=10.5, color=GREEN,
             ha="center")
    ax1.text(4.5, 0.10, "잡음변수 (모두 0)", fontsize=10.5, color=MUTED,
             ha="center")
    ax1.text(3.0, 0.72, f"라쏘  합 {bl[:3].sum():.2f}, 표준편차 "
                        f"{bl[:3].std():.3f}", fontsize=10, color=BLUE)
    ax1.text(3.0, 0.58, f"EN    합 {be[:3].sum():.2f}, 표준편차 "
                        f"{be[:3].std():.3f}", fontsize=10, color=ORANGE)
    ax1.legend(fontsize=10.5, loc="upper right", frameon=False,
               bbox_to_anchor=(1.0, 1.02))
    clean(ax1)

    bins = np.linspace(0, 1.4, 45)
    ax2.hist(sl, bins=bins, color=BLUE, alpha=0.7, label="라쏘")
    ax2.hist(se, bins=bins, color=ORANGE, alpha=0.85, label="엘라스틱넷")
    ax2.set_xlabel(r'$\hat\beta_0,\hat\beta_1,\hat\beta_2$ 의 표준편차',
                   fontsize=11, color=INK)
    ax2.set_ylabel("빈도", fontsize=11, color=INK)
    ax2.set_title(f"(나) 자료를 {R}번 다시 뽑으면", fontsize=12, color=INK)
    ax2.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax2.text(1.38, 0.62 * ax2.get_ylim()[1],
             f"중앙값\n라쏘 {np.median(sl):.3f}\n엘라스틱넷 {np.median(se):.3f}",
             fontsize=10.5, color=INK, ha="right", va="top")
    clean(ax2)

    fig.tight_layout()
    save(fig, "group_split.png")


if __name__ == "__main__":
    en_geometry()
    group_paths()
    saturation()
    alpha_lambda_cv()
    group_split()
