r"""18.1 정칙화의 동기 세 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch18/motivation/img/train_test_gap.png        복잡도가 커지면 두 오차가 갈라진다
  ch18/motivation/img/collinear_seesaw.png      상관된 두 계수의 시소와 능형의 수축
  ch18/motivation/img/condition_number_floor.png  lambda 가 스펙트럼 바닥을 올린다

실행:  python3 scripts/make_ch18_motivation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch18/motivation/img/"
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
    """로그 x축의 눈금 이름을 평문으로 박는다 (mathtext 두부 방지)."""
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.xaxis.set_minor_locator(NullLocator())


def logy_plain(ax, ticks, labels):
    ax.set_yscale("log")
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.yaxis.set_minor_locator(NullLocator())


# ==================================================================
# 1. 훈련오차와 검정오차의 갈라짐 + 편향-분산 절충
# ==================================================================
def train_test_gap():
    rng = np.random.default_rng(7)
    n, s2 = 60, 4.0

    # --- (가) 모수 개수를 늘릴 때
    ps = np.array([3, 6, 10, 15, 20, 28, 34, 40, 46, 52])
    tr_sim, te_sim = [], []
    for p in ps:
        X = rng.normal(size=(n, p))
        b = np.ones(p)
        t1, t2 = [], []
        for _ in range(600):
            y = X @ b + rng.normal(0, np.sqrt(s2), n)
            bh = np.linalg.lstsq(X, y, rcond=None)[0]
            t1.append(((y - X @ bh) ** 2).mean())
            y2 = X @ b + rng.normal(0, np.sqrt(s2), n)
            t2.append(((y2 - X @ bh) ** 2).mean())
        tr_sim.append(np.mean(t1))
        te_sim.append(np.mean(t2))
    tr_sim, te_sim = np.array(tr_sim), np.array(te_sim)

    grid = np.linspace(1, 55, 200)
    tr_th = s2 * (1 - grid / n)
    te_th = s2 * (1 + grid / n)

    # --- (나) 능형 벌점을 키울 때의 편향-분산 분해
    p, nn, rho, c = 8, 40, 0.9, 2.0
    S = rho * np.ones((p, p)) + (1 - rho) * np.eye(p)
    L = np.linalg.cholesky(S)
    beta = c * np.ones(p)
    lams = np.logspace(-2, 3, 26)
    M = 500
    rng2 = np.random.default_rng(11)
    Xs = [rng2.normal(size=(nn, p)) @ L.T for _ in range(M)]
    ys = [X @ beta + rng2.normal(0, 1, nn) for X in Xs]
    bias2, var, tot = [], [], []
    for lam in lams:
        B = np.empty((M, p))
        for m in range(M):
            X, y = Xs[m], ys[m]
            A = X.T @ X + lam * np.eye(p)
            B[m] = np.linalg.solve(A, X.T @ y)
        mb = B.mean(axis=0)
        bias2.append(((mb - beta) ** 2).sum())
        var.append(B.var(axis=0).sum())
        tot.append(((B - beta) ** 2).sum(axis=1).mean())
    bias2, var, tot = map(np.array, (bias2, var, tot))
    k = int(np.argmin(tot))
    # OLS 기준값
    B0 = np.empty((M, p))
    for m in range(M):
        B0[m] = np.linalg.lstsq(Xs[m], ys[m], rcond=None)[0]
    ols_mse = ((B0 - beta) ** 2).sum(axis=1).mean()

    print(f"[가] p=40: 훈련 {tr_sim[ps == 40][0]:.3f}  검정 {te_sim[ps == 40][0]:.3f}")
    print(f"[나] OLS MSE {ols_mse:.3f}  최적 lam {lams[k]:.1f}  MSE {tot[k]:.3f}")
    print(f"     최적점 편향^2 {bias2[k]:.3f}  분산 {var[k]:.3f}")
    print(f"     lam 최소값에서 분산 {var[0]:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    ax1.fill_between(grid, tr_th, te_th, color=ORANGE_F, alpha=0.55, zorder=0)
    ax1.plot(grid, te_th, color=ORANGE, lw=2, zorder=3)
    ax1.plot(grid, tr_th, color=BLUE, lw=2, zorder=3)
    ax1.plot(ps, te_sim, "o", ms=6, color=ORANGE, mec="white", mew=1.2, zorder=4)
    ax1.plot(ps, tr_sim, "o", ms=6, color=BLUE, mec="white", mew=1.2, zorder=4)
    ax1.axhline(s2, color=MUTED, ls=":", lw=1.4, zorder=1)
    ax1.text(14, s2 + 0.15, r'잡음 바닥 $\sigma^2=4$', fontsize=10, color=INK)
    ax1.text(40, 6.9, r'검정 $\sigma^2(1+p/n)$', fontsize=11, color=ORANGE,
             ha="right", va="bottom")
    ax1.text(40, 0.75, r'훈련 $\sigma^2(1-p/n)$', fontsize=11, color=BLUE,
             ha="right", va="top")
    ax1.annotate("", xy=(46, s2 * (1 + 46 / n)), xytext=(46, s2 * (1 - 46 / n)),
                 arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax1.text(44.2, s2, r'격차 $2\sigma^2 p/n$', fontsize=10.5, color=RED,
             ha="right", va="center")
    ax1.set_xlim(0, 56)
    ax1.set_ylim(0, 7.8)
    ax1.set_xlabel(r'모수 개수 $p$   ($n=60$ 고정)', fontsize=11, color=INK)
    ax1.set_ylabel("평균제곱 예측오차", fontsize=11, color=INK)
    ax1.set_title("(가) 복잡도를 늘리면 두 오차가 갈라진다", fontsize=12, color=INK)
    clean(ax1)

    ax2.plot(lams, tot, color=PURPLE, lw=2.4, zorder=4, label="총 MSE")
    ax2.plot(lams, var, color=BLUE, lw=2, zorder=3, label="분산")
    ax2.plot(lams, bias2, color=ORANGE, lw=2, zorder=3, label=r'편향$^2$')
    ax2.axhline(ols_mse, color=MUTED, ls="--", lw=1.4, zorder=1)
    ax2.plot([lams[k]], [tot[k]], "o", ms=9, color=PURPLE, mec="white", mew=1.6,
             zorder=5)
    logx_plain(ax2, [0.01, 0.1, 1, 10, 100, 1000],
               ["0.01", "0.1", "1", "10", "100", "1000"])
    logy_plain(ax2, [0.01, 0.1, 1, 10], ["0.01", "0.1", "1", "10"])
    ax2.set_ylim(0.004, 30)
    ax2.text(0.013, ols_mse * 1.18, f"OLS MSE = {ols_mse:.2f}", fontsize=10.5,
             color=INK)
    ax2.annotate(f"최적 $\\lambda={lams[k]:.0f}$\nMSE = {tot[k]:.2f}",
                 xy=(lams[k], tot[k]), xytext=(lams[k] * 3.2, tot[k] * 0.12),
                 fontsize=10.5, color=PURPLE, ha="left",
                 arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.4))
    ax2.set_xlabel(r'능형 벌점 $\lambda$   (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel(r'$\|\hat{\beta}-\beta\|^2$ 의 기댓값 (로그 눈금)',
                   fontsize=11, color=INK)
    ax2.set_title("(나) 편향을 조금 들이면 분산이 훨씬 줄어든다", fontsize=12,
                  color=INK)
    ax2.legend(fontsize=10, loc="center left", frameon=False,
               bbox_to_anchor=(0.01, 0.40))
    clean(ax2)

    fig.tight_layout()
    save(fig, "train_test_gap.png")


# ==================================================================
# 2. 상관된 두 계수의 시소
# ==================================================================
def collinear_seesaw():
    rng = np.random.default_rng(5)
    n, r, R = 50, 0.999, 400
    bo = np.empty((R, 2))
    br = np.empty((R, 2))
    for i in range(R):
        z = rng.normal(size=n)
        x1 = np.sqrt(r) * z + np.sqrt(1 - r) * rng.normal(size=n)
        x2 = np.sqrt(r) * z + np.sqrt(1 - r) * rng.normal(size=n)
        X = np.c_[x1, x2]
        Xc = X - X.mean(axis=0)
        y = 1.0 * x1 + 1.0 * x2 + rng.normal(0, 1, n)
        yc = y - y.mean()
        bo[i] = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        br[i] = np.linalg.solve(Xc.T @ Xc + 1.0 * np.eye(2), Xc.T @ yc)

    so, sr = bo.sum(axis=1), br.sum(axis=1)
    do, dr = bo[:, 0] - bo[:, 1], br[:, 0] - br[:, 1]
    print(f"OLS : sd(b1)={bo[:,0].std():.2f}  sd(합)={so.std():.3f}  "
          f"sd(차)={do.std():.2f}  합 평균={so.mean():.3f}")
    print(f"능형: sd(b1)={br[:,0].std():.3f} sd(합)={sr.std():.3f}  "
          f"sd(차)={dr.std():.3f} 합 평균={sr.mean():.3f}")
    print(f"OLS 계수 상관 {np.corrcoef(bo.T)[0,1]:.4f}")
    print(f"OLS b1 범위 [{bo[:,0].min():.1f}, {bo[:,0].max():.1f}]")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.6))

    lim = 12
    t = np.linspace(-lim - 2, lim + 2, 10)
    ax.plot(t, 2 - t, color=GREEN, lw=1.8, ls="--", zorder=2)
    ax.plot(bo[:, 0], bo[:, 1], "o", ms=4.5, color=BLUE, alpha=0.5, mew=0,
            zorder=3, label=f"OLS ({R}개 자료)")
    ax.plot(br[:, 0], br[:, 1], "o", ms=4.0, color=ORANGE, alpha=0.85, mew=0,
            zorder=4, label=r'능형 $\lambda=1$')
    ax.plot([1], [1], "*", ms=18, color=RED, mec="white", mew=1.0, zorder=6)
    ax.annotate("참값 (1, 1)", xy=(1.7, 0.6), xytext=(5.0, 3.2), fontsize=11,
                color=RED, arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.text(-lim + 0.6, -lim + 0.6, r'점선: $\beta_1+\beta_2=2$',
            fontsize=10.5, color=GREEN)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r'$\hat{\beta}_1$', fontsize=12, color=INK)
    ax.set_ylabel(r'$\hat{\beta}_2$', fontsize=12, color=INK)
    ax.set_title(r'(가) 상관 $r=0.999$: 합은 안정, 차는 폭주', fontsize=12,
                 color=INK)
    ax.legend(fontsize=10.5, loc="upper right", frameon=False)
    clean(ax)

    ax2.plot(t, 2 - t, color=GREEN, lw=1.8, ls="--", zorder=2)
    ax2.plot(bo[:, 0], bo[:, 1], "o", ms=5, color=BLUE, alpha=0.55, mew=0,
             zorder=3)
    ax2.plot(br[:, 0], br[:, 1], "o", ms=5, color=ORANGE, alpha=0.85, mew=0,
             zorder=4)
    ax2.plot([1], [1], "*", ms=18, color=RED, mec="white", mew=1.0, zorder=6)
    ax2.set_xlim(0.2, 1.8)
    ax2.set_ylim(0.2, 1.8)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r'$\hat{\beta}_1$', fontsize=12, color=INK)
    ax2.set_ylabel(r'$\hat{\beta}_2$', fontsize=12, color=INK)
    ax2.set_title("(나) 참값 둘레를 16배 확대", fontsize=12, color=INK)
    ax2.text(0.26, 0.26, f"$\\hat{{\\beta}}_1$ 의 표준편차\n"
                         f"능형 {br[:,0].std():.2f}  /  OLS {bo[:,0].std():.2f}",
             fontsize=10.5, color=INK, va="bottom")
    clean(ax2)

    fig.tight_layout()
    save(fig, "collinear_seesaw.png")


# ==================================================================
# 3. lambda 가 고윳값 스펙트럼의 바닥을 올린다
# ==================================================================
def condition_number_floor():
    p = 8
    lam_grid = np.logspace(-4, 1, 240)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.5))

    # --- (가) 스펙트럼
    rho = 0.999
    ev = np.sort(np.array([1 + (p - 1) * rho] + [1 - rho] * (p - 1)))[::-1]
    idx = np.arange(1, p + 1)
    for lam, col, mk, lab in [(0.0, MUTED, "o", r'$\lambda=0$'),
                              (0.01, BLUE, "s", r'$\lambda=0.01$'),
                              (1.0, ORANGE, "D", r'$\lambda=1$')]:
        ax1.plot(idx, ev + lam, mk + "-", ms=7, lw=1.8, color=col, mec="white",
                 mew=1.0, label=lab)
    logy_plain(ax1, [0.001, 0.01, 0.1, 1, 10],
               ["0.001", "0.01", "0.1", "1", "10"])
    ax1.set_ylim(4e-4, 40)
    ax1.set_xticks(idx)
    ax1.set_xlabel(r'고윳값 번호 $j$', fontsize=11, color=INK)
    ax1.set_ylabel(r'$d_j^2+\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax1.set_title(r'(가) 등상관 $\rho=0.999$, $p=8$ 의 스펙트럼', fontsize=12,
                  color=INK)
    ax1.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax1.annotate("바닥이 0.001 에서 1 로", xy=(5.0, 1.0), xytext=(3.1, 0.07),
                 fontsize=10.5, color=ORANGE,
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    clean(ax1)

    # --- (나) 조건수
    for rho, col, lab in [(0.99, BLUE, r'$\rho=0.99$'),
                          (0.999, ORANGE, r'$\rho=0.999$')]:
        ev = np.array([1 + (p - 1) * rho] + [1 - rho] * (p - 1))
        hi, lo = ev.max(), ev.min()
        kap = (hi + lam_grid) / (lo + lam_grid)
        ax2.plot(lam_grid, kap, lw=2.4, color=col, label=lab)
        ax2.axhline(hi / lo, color=col, ls=":", lw=1.2)
        ax2.axvline(lo, color=col, ls="--", lw=1.2, alpha=0.7)
        print(f"rho={rho}: kappa(0)={hi/lo:.1f}  lam_min={lo:g}  "
              f"kappa(lam=1)={(hi+1)/(lo+1):.2f}")
    logx_plain(ax2, [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
               ["0.0001", "0.001", "0.01", "0.1", "1", "10"])
    logy_plain(ax2, [1, 10, 100, 1000, 10000],
               ["1", "10", "100", "1000", "10000"])
    ax2.set_ylim(0.8, 6e4)
    ax2.set_xlim(1e-4, 10)
    ax2.text(1.15e-4, 900, "793", fontsize=10, color=BLUE, va="bottom")
    ax2.text(1.15e-4, 9200, "7993", fontsize=10, color=ORANGE, va="bottom")
    ax2.text(0.0115, 1.15, r'$\lambda_{\min}=0.01$', fontsize=10, color=BLUE,
             rotation=90, va="bottom")
    ax2.text(0.00115, 1.15, r'$\lambda_{\min}=0.001$', fontsize=10,
             color=ORANGE, rotation=90, va="bottom")
    ax2.set_xlabel(r'더해 준 $\lambda$ (로그 눈금)', fontsize=11, color=INK)
    ax2.set_ylabel(r'조건수 $\kappa(\Sigma+\lambda I)$ (로그 눈금)',
                   fontsize=11, color=INK)
    ax2.set_title(r'(나) $\lambda$ 가 $\lambda_{\min}$ 을 넘어야 떨어진다',
                  fontsize=12, color=INK)
    ax2.legend(fontsize=10.5, loc="lower left", frameon=False)
    clean(ax2)

    fig.tight_layout()
    save(fig, "condition_number_floor.png")


if __name__ == "__main__":
    train_test_gap()
    collinear_seesaw()
    condition_number_floor()
