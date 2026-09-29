r"""18.5 PCR·PLS 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/pcr_pls/img/pcr_vs_pls_components.png  성분 수에 따른 교차검증 R^2
  ch18/pcr_pls/img/pcr_shrinkage.png          능형의 연속 축소와 PCR의 계단
  ch18/pcr_pls/img/pls_direction.png          PCA 방향과 PLS 방향
  ch18/pcr_pls/img/housing_components.png     주택자료의 성분 수 대 CV RMSE

실행:  python3 scripts/make_ch18_pcr_pls_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       마지막 housing_components 만 pandas 와 인터넷 연결이 필요하다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/pcr_pls/img/"
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


# ==================================================================
# 1. 성분 수에 따른 교차검증 R^2   (index.md)
# ==================================================================
def pcr_vs_pls_components():
    rng = np.random.default_rng(0)
    n, p = 100, 30
    Z = rng.normal(size=(n, 3))
    A = rng.normal(size=(3, p))
    X = Z @ A + rng.normal(0, 0.5, (n, p))
    y = Z[:, 0] * 2 - Z[:, 2] * 1.5 + rng.normal(0, 1, n)
    Xc, yc = X - X.mean(0), y - y.mean()
    kf = KFold(5, shuffle=True, random_state=0)

    ks = np.arange(1, 9)
    r_pcr, r_pls = [], []
    for k in ks:
        pc = PCA(k).fit_transform(Xc)
        r_pcr.append(cross_val_score(LinearRegression(), pc, yc, cv=kf,
                                     scoring="r2").mean())
        r_pls.append(cross_val_score(PLSRegression(k), Xc, yc, cv=kf,
                                     scoring="r2").mean())
    r_pcr, r_pls = np.array(r_pcr), np.array(r_pls)
    print("PCR", r_pcr.round(3))
    print("PLS", r_pls.round(3))

    ev = PCA().fit(Xc).explained_variance_ratio_
    print("설명분산비", ev[:5].round(3))
    # 각 주성분과 y 의 상관
    P = PCA(5).fit_transform(Xc)
    cors = [abs(np.corrcoef(P[:, j], yc)[0, 1]) for j in range(5)]
    print("주성분-y 상관", np.round(cors, 3))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    ax1.plot(ks, r_pls, "o-", lw=2.6, ms=8, color=ORANGE, mec="white", mew=1.2,
             label="PLS (지도)")
    ax1.plot(ks, r_pcr, "s-", lw=2.6, ms=8, color=BLUE, mec="white", mew=1.2,
             label="PCR (비지도)")
    i1, i2 = int(np.argmax(r_pls)), int(np.argmax(r_pcr))
    ax1.plot([ks[i1]], [r_pls[i1]], "*", ms=18, color=RED, mec="white",
             mew=1.0, zorder=5)
    ax1.plot([ks[i2]], [r_pcr[i2]], "*", ms=18, color=RED, mec="white",
             mew=1.0, zorder=5)
    ax1.set_xticks(ks)
    ax1.set_ylim(0.35, 0.95)
    ax1.set_xlabel("성분 수", fontsize=11, color=INK)
    ax1.set_ylabel(r'5-겹 교차검증 $R^2$', fontsize=11, color=INK)
    ax1.set_title(r'(가) $n=100$, $p=30$, 잠재요인 3개 중 2개만 $y$ 에 관여',
                  fontsize=12, color=INK)
    ax1.legend(fontsize=10.5, loc="lower right", frameon=False)
    ax1.annotate(f"성분 2개로 {r_pls[1]:.3f}", xy=(2, r_pls[1]),
                 xytext=(2.6, 0.93), fontsize=10.5, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax1.annotate(f"같은 수준에 3개 필요", xy=(3, r_pcr[2]),
                 xytext=(4.2, 0.60), fontsize=10.5, color=BLUE,
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    clean(ax1)

    idx = np.arange(1, 6)
    w = 0.4
    ax2.bar(idx - w / 2, ev[:5], w, color=BLUE, label=r'$X$ 의 설명분산 비율')
    ax2.bar(idx + w / 2, cors, w, color=ORANGE,
            label=r'그 성분과 $y$ 의 상관 (절댓값)')
    for i in range(5):
        ax2.text(idx[i] - w / 2, ev[i] + 0.015, f"{ev[i]:.2f}", fontsize=9.5,
                 color=BLUE, ha="center")
        ax2.text(idx[i] + w / 2, cors[i] + 0.015, f"{cors[i]:.2f}",
                 fontsize=9.5, color=ORANGE, ha="center")
    ax2.set_xticks(idx)
    ax2.set_ylim(0, 0.76)
    ax2.set_xlabel("주성분 번호", fontsize=11, color=INK)
    ax2.set_title("(나) 분산이 큰 성분이 $y$ 를 잘 맞히지는 않는다",
                  fontsize=12, color=INK)
    ax2.legend(fontsize=10, loc="upper right", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "pcr_vs_pls_components.png")


# ==================================================================
# 2. 능형의 연속 축소와 PCR 의 계단   (principal_components_regression.md)
# ==================================================================
def pcr_shrinkage():
    rng = np.random.default_rng(21)
    n, p, rho = 60, 10, 0.9
    S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    X = rng.normal(size=(n, p)) @ np.linalg.cholesky(S).T
    X -= X.mean(0)
    d2 = np.linalg.eigvalsh(X.T @ X)[::-1]
    print("고윳값", d2.round(2))

    j = np.arange(1, p + 1)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for lam, col in [(1.0, BLUE), (10.0, PURPLE), (100.0, GREEN)]:
        ax.plot(j, d2 / (d2 + lam), "o-", lw=2.2, ms=7, color=col, mec="white",
                mew=1.0, label=f"능형 $\\lambda={lam:g}$")
        print(f"  lam={lam:g}: {np.round(d2/(d2+lam), 3)}  "
              f"df={np.sum(d2/(d2+lam)):.2f}")
    for M, col in [(1, ORANGE), (3, RED)]:
        step = (j <= M).astype(float)
        ax.step(np.r_[j, p + 1] - 0.5, np.r_[step, step[-1]], where="post",
                lw=2.2, color=col, ls="--", label=f"PCR $M={M}$")
        print(f"  PCR M={M}: df={M}")
    ax.set_xticks(j)
    ax.set_xlim(0.4, p + 0.6)
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlabel(r'주성분 번호 $j$   (고윳값이 큰 것부터)', fontsize=11,
                  color=INK)
    ax.set_ylabel(r'축소인자 $c_j$', fontsize=11, color=INK)
    ax.set_title(r'$n=60$, $p=10$, 등상관 $0.9$ 에서의 축소인자',
                 fontsize=12.5, color=INK)
    ax.legend(fontsize=10.5, loc="center right", frameon=False)
    ax.text(1.2, 1.06, "1 = 그대로 둔다", fontsize=10, color=INK)
    ax.text(6.0, 0.15, "0 = 버린다", fontsize=10, color=INK)
    clean(ax)
    fig.tight_layout()
    save(fig, "pcr_shrinkage.png")


# ==================================================================
# 3. PCA 방향과 PLS 방향   (partial_least_squares.md)
# ==================================================================
def pls_direction():
    rng = np.random.default_rng(11)
    n = 300
    u = rng.normal(0, 3.0, n)          # 분산이 큰 방향 (y 와 무관)
    v = rng.normal(0, 1.0, n)          # 분산이 작은 방향 (y 의 신호)
    X = np.c_[u + 0.2 * v, 0.35 * u + v]
    X = X - X.mean(0)
    y = 2.0 * v + rng.normal(0, 0.6, n)
    y = y - y.mean()

    v1 = PCA(2).fit(X).components_[0]
    w1 = X.T @ y
    w1 = w1 / np.linalg.norm(w1)
    if v1[0] < 0:
        v1 = -v1
    if w1[0] < 0:
        w1 = -w1
    print("PCA 방향", v1.round(3), " PLS 방향", w1.round(3))
    ang = np.degrees(np.arccos(abs(v1 @ w1)))
    print(f"두 방향의 각도 {ang:.1f}도")
    zp = X @ v1
    zw = X @ w1
    print(f"분산: PCA 성분 {zp.var():.2f}  PLS 성분 {zw.var():.2f}")
    print(f"y 와의 상관: PCA {abs(np.corrcoef(zp, y)[0,1]):.3f}  "
          f"PLS {abs(np.corrcoef(zw, y)[0,1]):.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 5.2))
    sc = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=22,
                     edgecolors="none", zorder=3)
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label(r'반응변수 $y$', fontsize=10.5, color=INK)
    cb.ax.tick_params(labelsize=9.5, colors=INK)
    L = 7.5
    ax1.annotate("", xy=(L * v1[0], L * v1[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6,
                                 mutation_scale=22), zorder=5)
    ax1.annotate("", xy=(L * w1[0], L * w1[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.6,
                                 mutation_scale=22), zorder=5)
    ax1.text(9.0, 4.6, "PCA 첫 방향\n(분산 최대)", fontsize=10.5, color=BLUE,
             ha="right", va="bottom")
    ax1.text(-0.8, 6.2, "PLS 첫 방향\n($y$ 와의 공분산 최대)", fontsize=10.5,
             color=GREEN, ha="right", va="center")
    ax1.axhline(0, color=INK, lw=0.7)
    ax1.axvline(0, color=INK, lw=0.7)
    ax1.set_xlim(-11, 11)
    ax1.set_ylim(-7.5, 9.5)
    ax1.set_xlabel(r'$x_1$', fontsize=12, color=INK)
    ax1.set_ylabel(r'$x_2$', fontsize=12, color=INK)
    ax1.set_title("(가) 같은 자료, 다른 첫 방향", fontsize=12, color=INK)
    clean(ax1)

    for z, col, lab in [(zp, BLUE, "PCA 첫 성분"), (zw, GREEN, "PLS 첫 성분")]:
        ax2.plot(z, y, "o", ms=4, color=col, alpha=0.45, mew=0, label=lab)
        b = np.polyfit(z, y, 1)
        g = np.linspace(z.min(), z.max(), 10)
        ax2.plot(g, np.polyval(b, g), lw=2.2, color=col)
    ax2.set_xlabel("성분 점수", fontsize=11, color=INK)
    ax2.set_ylabel(r'$y$', fontsize=12, color=INK)
    ax2.set_title("(나) 그 성분으로 $y$ 를 얼마나 설명하는가", fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10.5, loc="upper left", frameon=False)
    ax2.text(0.98, 0.04,
             f"$y$ 와의 상관\nPCA {abs(np.corrcoef(zp, y)[0,1]):.3f}   "
             f"PLS {abs(np.corrcoef(zw, y)[0,1]):.3f}",
             transform=ax2.transAxes, fontsize=10.5, color=INK, ha="right",
             va="bottom")
    clean(ax2)

    fig.tight_layout()
    save(fig, "pls_direction.png")


# ==================================================================
# 4. 주택자료의 성분 수   (pcr_pls_examples.md)
# ==================================================================
def housing_components():
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import RidgeCV

    url = ("https://raw.githubusercontent.com/gedeck/"
           "practical-statistics-for-data-scientists/master/data/"
           "house_sales.csv")
    house = pd.read_csv(url, sep="\t")
    feats = ['SqFtTotLiving', 'SqFtLot', 'Bathrooms', 'Bedrooms', 'BldgGrade',
             'NbrLivingUnits', 'SqFtFinBasement', 'YrBuilt', 'YrRenovated']
    X = house[feats].to_numpy(float)
    y = house['AdjSalePrice'].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(10, shuffle=True, random_state=42)
    p = Xs.shape[1]

    rmse_pcr, rmse_pls = [], []
    for M in range(1, p + 1):
        pipe = Pipeline([("pca", PCA(M)), ("lr", LinearRegression())])
        s = -cross_val_score(pipe, Xs, y, cv=kf,
                             scoring="neg_mean_squared_error").mean()
        rmse_pcr.append(np.sqrt(s))
        s = -cross_val_score(PLSRegression(M), Xs, y, cv=kf,
                             scoring="neg_mean_squared_error").mean()
        rmse_pls.append(np.sqrt(s))
    rmse_pcr, rmse_pls = np.array(rmse_pcr), np.array(rmse_pls)
    ols = np.sqrt(-cross_val_score(LinearRegression(), Xs, y, cv=kf,
                                   scoring="neg_mean_squared_error").mean())
    ridge = np.sqrt(-cross_val_score(
        RidgeCV(alphas=np.logspace(-2, 5, 60)), Xs, y, cv=kf,
        scoring="neg_mean_squared_error").mean())
    ev = PCA().fit(Xs).explained_variance_ratio_
    print("PCR RMSE", rmse_pcr.round(0))
    print("PLS RMSE", rmse_pls.round(0))
    print(f"OLS {ols:.0f}  능형 {ridge:.0f}")
    print("설명분산 누적", np.cumsum(ev).round(3))

    M = np.arange(1, p + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7))
    ax1.plot(M, rmse_pls, "o-", lw=2.4, ms=7, color=ORANGE, mec="white",
             mew=1.0, label="PLS")
    ax1.plot(M, rmse_pcr, "s-", lw=2.4, ms=7, color=BLUE, mec="white",
             mew=1.0, label="PCR")
    ax1.axhline(ols, color=INK, ls="--", lw=1.6)
    ax1.axhline(ridge, color=GREEN, ls=":", lw=1.8)
    ax1.text(4.4, 253000,
             f"OLS {ols:,.0f} 와 능형 {ridge:,.0f} 은\n두 선이 겹칠 만큼 같다",
             fontsize=10, color=INK, va="bottom")
    ax1.set_xticks(M)
    ax1.set_xlim(0.6, p + 0.4)
    ax1.set_xlabel("성분 수 $M$", fontsize=11, color=INK)
    ax1.set_ylabel("10-겹 교차검증 RMSE (달러)", fontsize=11, color=INK)
    ax1.set_title("(가) 성분을 늘릴 때의 예측오차", fontsize=12, color=INK)
    ax1.legend(fontsize=10.5, loc="upper right", frameon=False)
    clean(ax1)

    ax2.bar(M, ev, 0.55, color=BLUE, zorder=3)
    ax2.plot(M, np.cumsum(ev), "o-", lw=2.2, ms=6, color=ORANGE, mec="white",
             mew=1.0, zorder=4, label="누적 설명분산")
    ax2.axhline(0.9, color=RED, ls=":", lw=1.5)
    ax2.text(p + 0.3, 0.915, "90%", fontsize=10, color=RED, ha="right")
    ax2.set_xticks(M)
    ax2.set_ylim(0, 1.12)
    ax2.set_xlabel("주성분 번호", fontsize=11, color=INK)
    ax2.set_ylabel(r'$X$ 의 설명분산 비율', fontsize=11, color=INK)
    ax2.set_title("(나) 스크리 그림과 누적 설명분산", fontsize=12, color=INK)
    ax2.legend(fontsize=10.5, loc="center right", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "housing_components.png")


if __name__ == "__main__":
    pcr_vs_pls_components()
    pcr_shrinkage()
    pls_direction()
    try:
        housing_components()
    except Exception as exc:
        print("housing_components 건너뜀:", exc)
