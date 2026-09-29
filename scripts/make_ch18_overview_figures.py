r"""18.0 개관 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/overview/img/three_paths.png        세 벌점의 계수 경로 모양
  ch18/overview/img/selection_stability.png 상관 집단에서 라쏘의 선택이 흔들린다
  ch18/overview/img/conditional_gain.png   정칙화의 이득은 조건부다

실행:  python3 scripts/make_ch18_overview_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/overview/img/"
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
# 1. 세 벌점의 계수 경로
# ==================================================================
def three_paths():
    rng = np.random.default_rng(4)
    n, p = 80, 8
    Z = rng.normal(size=n)
    X = rng.normal(size=(n, p))
    for j in (0, 1):                       # 두 변수를 강하게 상관시킨다
        X[:, j] = np.sqrt(0.95) * Z + np.sqrt(0.05) * X[:, j]
    X = (X - X.mean(0)) / X.std(0)
    beta = np.zeros(p)
    beta[:4] = [2.5, 2.0, -1.6, 0.8]
    y = X @ beta + rng.normal(0, 1.0, n)
    y = y - y.mean()

    lams = np.logspace(-3, 1, 160)
    paths = {}
    paths["ridge"] = np.array([Ridge(alpha=n * l, fit_intercept=False)
                               .fit(X, y).coef_ for l in lams])
    paths["lasso"] = np.array([Lasso(alpha=l, fit_intercept=False,
                                     max_iter=200000).fit(X, y).coef_
                               for l in lams])
    paths["enet"] = np.array([ElasticNet(alpha=l, l1_ratio=0.3,
                                         fit_intercept=False,
                                         max_iter=200000).fit(X, y).coef_
                              for l in lams])

    cols = [BLUE, ORANGE, GREEN, PURPLE] + [MUTED] * 4
    lws = [2.2, 2.2, 2.2, 2.2] + [1.3] * 4
    titles = [("(가) 능형 (L2)", "ridge"),
              ("(나) 라쏘 (L1)", "lasso"),
              (r'(다) 엘라스틱넷 ($\alpha=0.3$)', "enet")]

    for key in ("lasso", "enet"):
        nz = (np.abs(paths[key]) > 1e-8).sum(axis=1)
        print(key, "0이 아닌 계수:", nz[0], "->", nz[-1])
        first = {j: lams[np.abs(paths[key][:, j]) > 1e-8].max()
                 for j in range(p) if (np.abs(paths[key][:, j]) > 1e-8).any()}
        print("  변수별 진입 lambda:",
              {j: round(v, 3) for j, v in sorted(first.items())})
    print("ridge 최소 |계수| at lam=4:",
          np.abs(paths["ridge"][-1]).min())

    lam_zero = lams[(np.abs(paths["lasso"]) > 1e-8).sum(axis=1) == 0].min()
    print("라쏘가 모든 계수를 0으로 만드는 lambda:", round(lam_zero, 3))

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.8), sharey=True)
    handles = []
    for ax, (title, key) in zip(axes, titles):
        P = paths[key]
        for j in range(p):
            ln, = ax.plot(lams, P[:, j], lw=lws[j], color=cols[j], zorder=3)
            if ax is axes[0] and j <= 4:
                handles.append(ln)
        ax.axhline(0, color=INK, lw=0.9, zorder=2)
        logx_plain(ax, [0.001, 0.01, 0.1, 1, 10],
                   ["0.001", "0.01", "0.1", "1", "10"])
        ax.set_xlim(lams[0], lams[-1])
        ax.set_xlabel(r'벌점 $\lambda$ (로그 눈금)', fontsize=11, color=INK)
        ax.set_title(title, fontsize=12, color=INK)
        clean(ax)
    axes[0].set_ylabel("계수 추정값", fontsize=11, color=INK)
    axes[0].set_ylim(-2.3, 3.0)

    axes[1].annotate(f"$\\lambda={lam_zero:.1f}$ 에서 모두 0",
                     xy=(lam_zero, -0.08), xytext=(0.5, -1.7), fontsize=10.5,
                     color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[0].annotate("0에 닿지 않는다", xy=(6.0, 0.10), xytext=(0.06, 1.5),
                     fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[2].annotate("상관된 둘이 함께 간다", xy=(2.2, 1.28),
                     xytext=(0.004, 2.6), fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    fig.legend(handles,
               [r'$x_1$ (상관)', r'$x_2$ (상관)', r'$x_3$', r'$x_4$',
                r'$x_5\!-\!x_8$ (잡음변수)'],
               fontsize=10.5, ncol=5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    save(fig, "three_paths.png")


# ==================================================================
# 2. 상관 집단에서의 선택 안정성
# ==================================================================
def selection_stability():
    p, n, rho, R = 20, 60, 0.99, 400
    lam = 0.6
    sel_l = np.zeros(p)
    sel_e = np.zeros(p)
    all3_l = all3_e = 0
    noise_l, noise_e = [], []
    for s in range(R):
        r = np.random.default_rng(1000 + s)
        Z = r.normal(size=n)
        X = r.normal(size=(n, p))
        for j in range(3):
            X[:, j] = np.sqrt(rho) * Z + np.sqrt(1 - rho) * X[:, j]
        beta = np.zeros(p)
        beta[:3] = 2.0
        y = X @ beta + r.normal(0, 1, n)
        bl = Lasso(alpha=lam, max_iter=50000).fit(X, y).coef_
        be = ElasticNet(alpha=lam, l1_ratio=0.2, max_iter=50000).fit(X, y).coef_
        ml, me = np.abs(bl) > 1e-8, np.abs(be) > 1e-8
        sel_l += ml
        sel_e += me
        all3_l += ml[:3].all()
        all3_e += me[:3].all()
        noise_l.append(ml[3:].sum())
        noise_e.append(me[3:].sum())
    sel_l /= R
    sel_e /= R
    print("라쏘 선택빈도(참변수):", np.round(sel_l[:3], 3))
    print("EN  선택빈도(참변수):", np.round(sel_e[:3], 3))
    print(f"세 변수 모두: 라쏘 {all3_l/R:.0%}  EN {all3_e/R:.0%}")
    print(f"평균 잡음변수 수: 라쏘 {np.mean(noise_l):.1f}  "
          f"EN {np.mean(noise_e):.1f}")

    idx = np.arange(1, p + 1)
    w = 0.4
    fig, ax = plt.subplots(figsize=(11.4, 4.6))
    ax.axvspan(0.4, 3.6, color=GREEN_F, alpha=0.5, zorder=0)
    ax.bar(idx - w / 2, sel_l, w, color=BLUE, zorder=3,
           label=r'라쏘 ($\lambda=0.6$)')
    ax.bar(idx + w / 2, sel_e, w, color=ORANGE, zorder=3,
           label=r'엘라스틱넷 ($\lambda=0.6,\ \alpha=0.2$)')
    ax.set_xticks(idx)
    ax.set_xlim(0.3, p + 0.7)
    ax.set_ylim(0, 1.18)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel(f"설명변수 번호   ({R}개 자료, $n=60$, $p=20$)",
                  fontsize=11, color=INK)
    ax.set_ylabel("선택된 비율", fontsize=11, color=INK)
    ax.set_title(r'상관 $0.99$ 인 참 변수 3개와 잡음변수 17개', fontsize=12.5,
                 color=INK)
    ax.text(2.0, 1.12, "참 변수", fontsize=11, color=GREEN, ha="center",
            va="center")
    ax.text(12.0, 1.12, "잡음변수 (참값 0)", fontsize=11, color=MUTED,
            ha="center", va="center")
    ax.legend(fontsize=10.5, loc="upper right", frameon=False,
              bbox_to_anchor=(1.0, 0.98))
    ax.text(13.0, 0.80, "라쏘의 잡음변수 막대는 0에 가까워 보이지 않는다",
            fontsize=10, color=BLUE, ha="center")
    ax.text(4.2, 0.72,
            f"세 참 변수를 모두 살린 비율\n"
            f"라쏘 {all3_l/R:.0%}   엘라스틱넷 {all3_e/R:.0%}\n"
            f"평균 잡음변수 개수\n"
            f"라쏘 {np.mean(noise_l):.1f}   엘라스틱넷 {np.mean(noise_e):.1f}",
            fontsize=10.5, color=INK, va="top")
    clean(ax)
    fig.tight_layout()
    save(fig, "selection_stability.png")


# ==================================================================
# 3. 정칙화의 이득은 조건부다
# ==================================================================
def conditional_gain():
    rhos = [0.0, 0.5, 0.9, 0.99, 0.999]
    n, p, M, lam = 50, 10, 400, 1.0
    beta = np.zeros(p)
    beta[:3] = [3, -2, 1.5]
    eo, er, conds = [], [], []
    for rho in rhos:
        rng = np.random.default_rng(18)
        S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
        L = np.linalg.cholesky(S)
        conds.append(np.linalg.cond(S))
        a, b = [], []
        for _ in range(M):
            X = rng.normal(size=(n, p)) @ L.T
            y = X @ beta + rng.normal(0, 1, n)
            bo = LinearRegression(fit_intercept=False).fit(X, y).coef_
            br = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
            a.append(((bo - beta) ** 2).sum())
            b.append(((br - beta) ** 2).sum())
        eo.append(np.mean(a))
        er.append(np.mean(b))
    eo, er = np.array(eo), np.array(er)
    for rho, a, b, c in zip(rhos, eo, er, conds):
        print(f"rho={rho}: OLS {a:.3f}  ridge {b:.3f}  "
              f"개선 {a/b:.2f}배  cond {c:.1f}")

    x = np.arange(len(rhos))
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x, eo, "o-", ms=9, lw=2.4, color=BLUE, mec="white", mew=1.4,
            label="OLS", zorder=3)
    ax.plot(x, er, "s-", ms=9, lw=2.4, color=ORANGE, mec="white", mew=1.4,
            label=r'능형 $\lambda=1$', zorder=3)
    for i in range(len(rhos)):
        ax.annotate("", xy=(x[i], er[i]), xytext=(x[i], eo[i]),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2,
                                    ls=":"))
        if eo[i] / er[i] > 1.3:
            ax.text(x[i] + 0.08, np.sqrt(eo[i] * er[i]), f"{eo[i]/er[i]:.1f}배",
                    fontsize=10.5, color=RED, va="center")
        else:
            ax.text(x[i], eo[i] * 1.9, f"{eo[i]/er[i]:.2f}배", fontsize=10.5,
                    color=RED, va="center", ha="center")
    ax.set_yscale("log")
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(0.1, 400)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlim(-0.35, len(rhos) - 0.35)
    ax.set_xlabel(r'설명변수의 등상관 $\rho$   ($n=50$, $p=10$)',
                  fontsize=11, color=INK)
    ax.set_ylabel(r'$E\|\hat{\beta}-\beta\|^2$ (로그 눈금)', fontsize=11,
                  color=INK)
    ax.set_title("정칙화의 이득은 문제가 나쁠수록 커진다", fontsize=12.5,
                 color=INK)
    ax.legend(fontsize=11, loc="upper left", frameon=False)
    for i, c in enumerate(conds):
        ax.text(x[i], 0.135, f"$\\kappa={c:.0f}$", fontsize=9.5, color=MUTED,
                ha="center")
    ax.text(2.0, 0.21, "상관행렬의 조건수", fontsize=10,
            color=MUTED, ha="center")
    clean(ax)
    fig.tight_layout()
    save(fig, "conditional_gain.png")


if __name__ == "__main__":
    three_paths()
    selection_stability()
    conditional_gain()
