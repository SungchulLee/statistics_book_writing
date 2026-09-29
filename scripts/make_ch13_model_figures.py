r"""13장의 회귀 모형화 다섯 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/linear_regression/img/categorical_coding.png       부호화가 모형을 바꾼다
  ch13/linear_regression/img/coef_scale_vs_effect.png     계수 크기는 단위의 문제다
  ch13/interaction_polynomial/img/interaction_slopes.png  기울기가 상대에 따라 달라진다
  ch13/interaction_polynomial/img/polynomial_degree.png   차수를 올리면 어디까지
  ch13/diagnostics/img/leverage_residual_cook.png         지렛대와 잔차와 영향력

실행:  python3 scripts/make_ch13_model_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

LIN = "docs/ch13/linear_regression/img/"
INT = "docs/ch13/interaction_polynomial/img/"
DIA = "docs/ch13/diagnostics/img/"
for d in (LIN, INT, DIA):
    os.makedirs(d, exist_ok=True)


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y, f):
    return 1 - ((y - f) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# === 1. 범주형 부호화 ===
def fig_categorical():
    rng = np.random.default_rng(0)
    n = 900
    city = rng.integers(0, 3, n)
    CITY = ["서울", "부산", "대구"]
    true = np.array([10.0, 16.0, 12.0])
    y = true[city] + rng.normal(0, 2, n)

    Xi = np.column_stack([np.ones(n), city.astype(float)])
    bi = ols(Xi, y)
    fi = Xi @ bi
    D = np.column_stack([np.ones(n), np.eye(3)[city][:, 1:]])
    bd = ols(D, y)
    fd = D @ bd
    means = np.array([y[city == k].mean() for k in range(3)])

    # 순서형 예 (연습문제 1)
    rng2 = np.random.default_rng(1)
    m = 6000
    lev = rng2.integers(0, 5, m)
    tedu = np.array([0.0, 0.2, 0.5, 3.0, 3.4])
    ye = tedu[lev] + rng2.normal(0, 1, m)
    Xe = np.column_stack([np.ones(m), lev.astype(float)])
    be = ols(Xe, ye)
    Xh = np.column_stack([np.ones(m), np.eye(5)[lev][:, 1:]])
    bh = ols(Xh, ye)
    emeans = np.array([ye[lev == k].mean() for k in range(5)])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    xs = np.arange(3)
    for k in range(3):
        jj = xs[k] + rng.normal(0, 0.055, (city == k).sum())
        ax.scatter(jj, y[city == k], s=8, color=MUTED, alpha=0.25,
                   edgecolor="none")
    ax.plot([-0.35, 2.35], bi[0] + bi[1] * np.array([-0.35, 2.35]),
            color=RED, lw=2.4, zorder=4)
    ax.scatter(xs, bi[0] + bi[1] * xs, s=120, color=RED, marker="s", zorder=6,
               label=f"정수 코딩  ($R^2$ = {r2(y, fi):.3f})")
    ax.scatter(xs, means, s=150, color=BLUE, marker="D", zorder=6,
               label=f"더미 부호화  ($R^2$ = {r2(y, fd):.3f})")
    for k in range(3):
        ax.annotate("", xy=(xs[k], means[k]), xytext=(xs[k], bi[0] + bi[1] * xs[k]),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2))
    ax.text(0.02, 19.9, "화살표 = 정수 코딩이 놓친 몫", fontsize=9.6,
            color=INK, ha="left")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n(코드 {k})" for k, c in enumerate(CITY)],
                       fontsize=10.5)
    ax.set_ylabel("임금", fontsize=10.5, color=INK)
    ax.set_ylim(2, 21)
    ax.set_title("명목형 — 순서도 등간격도 자료에 없다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="lower right", frameon=False)
    clean(ax)

    ax = axes[1]
    ls = np.arange(5)
    ax.plot([-0.3, 4.3], be[0] + be[1] * np.array([-0.3, 4.3]), color=RED,
            lw=2.4, zorder=4,
            label=f"정수 코딩  기울기 {be[1]:.3f}  ($R^2$ = {r2(ye, Xe@be):.3f})")
    ax.scatter(ls, be[0] + be[1] * ls, s=110, color=RED, marker="s", zorder=6)
    ax.plot(ls, emeans, color=BLUE, lw=2.4, marker="D", ms=10, zorder=5,
            label=f"원-핫  ($R^2$ = {r2(ye, Xh@bh):.3f})")
    ax.annotate("2단계와 3단계 사이의 도약", xy=(2.5, 1.75), xytext=(2.3, 0.12),
                fontsize=9.8, color=BLUE, ha="left",
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    ax.set_xticks(ls)
    ax.set_xticklabels([str(v) for v in ls], fontsize=10.5)
    ax.set_xlabel("교육 수준 (순서형)", fontsize=10.5, color=INK)
    ax.set_ylabel("임금 효과", fontsize=10.5, color=INK)
    ax.set_ylim(-0.6, 4.1)
    ax.set_title("순서형 — 순서는 있지만 등간격은 가정이다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="upper left", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, LIN + "categorical_coding.png")
    print(f"  집단평균 {means.round(4)}, 정수코딩 예측 "
          f"{(bi[0]+bi[1]*xs).round(4)}")
    print(f"  R2: 정수 {r2(y, fi):.4f}, 더미 {r2(y, fd):.4f}")
    print(f"  순서형: 정수 기울기 {be[1]:.4f}, R2 {r2(ye, Xe@be):.4f} / "
          f"원-핫 R2 {r2(ye, Xh@bh):.4f}")
    print(f"  원-핫 계수 {bh[1:].round(3)}  참값 {(tedu - tedu[0]).round(3)}")


# === 2. 계수 크기 대 1표준편차 효과 ===
def fig_coef_scale():
    np.random.seed(42)
    n = 50
    pop = np.random.lognormal(15.0, 0.9, n)
    income = np.random.normal(52_000, 9_000, n)
    pw = np.clip(np.random.normal(0.72, 0.13, n), 0.2, 0.95)
    _pb = np.clip(np.random.normal(0.12, 0.08, n), 0.01, 0.40)
    hq = (250 + 0.0009 * (income - 52_000) - 1.2e-6 * pop
          + 18 * (pw - 0.72) + np.random.normal(0, 12, n))
    TEST = {3, 8, 15, 22, 29, 34, 41, 47}
    mask = np.array([i not in TEST for i in range(n)])
    Xtr = np.column_stack([np.ones(mask.sum()), pop[mask], income[mask], pw[mask]])
    b = ols(Xtr, hq[mask])
    e = hq[mask] - Xtr @ b
    s2 = (e ** 2).sum() / (mask.sum() - 4)
    se = np.sqrt(s2 * np.diag(np.linalg.inv(Xtr.T @ Xtr)))
    tv = b / se
    sds = np.array([pop[mask].std(ddof=1), income[mask].std(ddof=1),
                    pw[mask].std(ddof=1)])
    per_sd = b[1:] * sds
    names = ["인구", "1인당 소득", "백인 비율"]
    cols = [MUTED, BLUE, ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    vals = np.abs(b[1:])
    ax.barh(np.arange(3), vals, color=cols, height=0.55)
    for i, v in enumerate(vals):
        ax.text(v * 1.35, i, f"{b[i+1]:.3g}", va="center", fontsize=10.5,
                color=INK)
    ax.set_xscale("log")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(1e-8, 1e4)
    ax.set_xticks([1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2])
    ax.set_xticklabels(["1e-8", "1e-6", "1e-4", "0.01", "1", "100"],
                       fontsize=9)
    ax.minorticks_off()
    ax.set_xlabel("계수의 절댓값 (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_title("날것의 계수는 9차수에 걸쳐 흩어진다", fontsize=11.5,
                 color=INK, pad=6)
    ax.text(1e-4, 1.52, "크기를 견줄 수 없다", fontsize=9.8, color=RED,
            ha="center")
    clean(ax)

    ax = axes[1]
    ax.barh(np.arange(3), per_sd, color=cols, height=0.55)
    ax.axvline(0, color=INK, lw=1.2)
    for i, v in enumerate(per_sd):
        off = 0.22 if v > 0 else -0.22
        ax.text(v + off, i, f"{v:+.2f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10.5, color=INK)
        ax.text(5.9, i + 0.30, f"t = {tv[i+1]:+.2f}", va="center", ha="right",
                fontsize=9.6, color=cols[i])
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlim(-1.6, 6.0)
    ax.set_xlabel("설명변수가 1 표준편차 오를 때 HighQ 의 변화 (달러)",
                  fontsize=10, color=INK)
    ax.set_title("표준편차로 재면 비로소 견줄 수 있다", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    fig.tight_layout()
    save(fig, LIN + "coef_scale_vs_effect.png")
    for i, nm in enumerate(names):
        print(f"  {nm:8s} coef={b[i+1]:>12.4g}  SD={sds[i]:>12.4g}  "
              f"1SD효과={per_sd[i]:+.3f}  t={tv[i+1]:+.3f}")


# === 3. 교호작용: 기울기가 상대에 따라 달라진다 ===
def fig_interaction():
    rng = np.random.default_rng(11)
    n = 400
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    y = 2 + 1.5 * x1 + 0.8 * x2 - 0.18 * x1 * x2 + rng.normal(0, 3, n)

    Xa = np.column_stack([np.ones(n), x1, x2])              # 가법
    ba = ols(Xa, y)
    Xi = np.column_stack([np.ones(n), x1, x2, x1 * x2])     # 교호작용
    bi = ols(Xi, y)
    ei = y - Xi @ bi
    s2 = (ei ** 2).sum() / (n - 4)
    V = s2 * np.linalg.inv(Xi.T @ Xi)
    tc = stats.t.ppf(0.975, n - 4)

    g = np.linspace(0, 10, 100)
    levels = [1.0, 5.0, 9.0]
    lcols = [GREEN, BLUE, PURPLE]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    for ax, bb, X, ttl in [
            (axes[0], ba, Xa, "교호작용 없는 모형 — 평행선"),
            (axes[1], bi, Xi, "교호작용 있는 모형 — 부채꼴")]:
        for lv, c in zip(levels, lcols):
            if X is Xa:
                yy = bb[0] + bb[1] * g + bb[2] * lv
                slope = bb[1]
            else:
                yy = bb[0] + bb[1] * g + bb[2] * lv + bb[3] * g * lv
                slope = bb[1] + bb[3] * lv
            ax.plot(g, yy, color=c, lw=2.6,
                    label=f"$x_2$ = {lv:.0f}   기울기 {slope:+.2f}")
        ax.set_xlabel(r"$x_1$", fontsize=11, color=INK)
        ax.set_ylabel(r"$y$", fontsize=11, color=INK)
        ax.set_ylim(-2, 22)
        ax.set_title(ttl, fontsize=11.5, color=INK, pad=6)
        ax.legend(fontsize=9.2, loc="upper left", frameon=False)
        clean(ax)

    ax = axes[2]
    gg = np.linspace(0, 10, 200)
    eff = bi[1] + bi[3] * gg
    sd = np.sqrt(V[1, 1] + gg ** 2 * V[3, 3] + 2 * gg * V[1, 3])
    ax.fill_between(gg, eff - tc * sd, eff + tc * sd, color=ORANGE_F,
                    alpha=0.85, label="95% 신뢰띠")
    ax.plot(gg, eff, color=ORANGE, lw=2.8,
            label=r"$\partial y/\partial x_1 = \hat\beta_1 + \hat\beta_3 x_2$")
    ax.axhline(0, color=RED, lw=1.6, ls="--")
    zero = -bi[1] / bi[3]
    ax.axvline(zero, color=MUTED, lw=1.4, ls=":")
    ax.annotate(f"$x_2$ = {zero:.1f} 을 넘으면\n" + r"$x_1$ 의 효과가 음수가 된다",
                xy=(zero, 0), xytext=(2.3, -0.95), fontsize=9.8, color=RED,
                linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel(r"$x_2$", fontsize=11, color=INK)
    ax.set_ylabel(r"$x_1$ 의 부분효과", fontsize=10.5, color=INK)
    ax.set_ylim(-1.5, 2.1)
    ax.set_title("주효과는 이제 수 하나가 아니다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.4, loc="upper right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, INT + "interaction_slopes.png")
    print(f"  가법 모형   : {ba.round(4)}")
    print(f"  교호작용 모형: {bi.round(4)}  (참값 2, 1.5, 0.8, -0.18)")
    for lv in levels:
        print(f"    x2={lv:.0f} 에서 x1 의 기울기 = {bi[1]+bi[3]*lv:+.3f}")
    print(f"  부분효과가 0 이 되는 x2 = {zero:.3f}")
    print(f"  R2: 가법 {r2(y, Xa@ba):.4f}, 교호작용 {r2(y, Xi@bi):.4f}")


# === 4. 다항 차수 ===
def fig_polynomial():
    rng = np.random.default_rng(4)
    n = 40
    x = np.sort(rng.uniform(-3, 3, n))
    f = lambda t: 2 + 1.2 * t - 0.9 * t ** 2
    y = f(x) + rng.normal(0, 2.0, n)
    xt = np.sort(rng.uniform(-3, 3, 400))
    yt = f(xt) + rng.normal(0, 2.0, 400)

    degs = list(range(1, 16))
    tr_rmse, te_rmse = [], []
    fits = {}
    for d in degs:
        X = np.vander(x, d + 1, increasing=True)
        b = ols(X, y)
        tr_rmse.append(np.sqrt(((y - X @ b) ** 2).mean()))
        Xt = np.vander(xt, d + 1, increasing=True)
        te_rmse.append(np.sqrt(((yt - Xt @ b) ** 2).mean()))
        fits[d] = b

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    g = np.linspace(-3.05, 3.05, 400)
    ax.scatter(x, y, s=34, color=MUTED, alpha=0.95, edgecolor="none",
               zorder=3, label="관측값")
    ax.plot(g, f(g), color=INK, lw=2.2, ls="--", label="참 곡선 (2차)")
    for d, c in [(1, GREEN), (2, BLUE), (15, RED)]:
        yy = np.vander(g, d + 1, increasing=True) @ fits[d]
        ax.plot(g, yy, color=c, lw=2.4, label=f"차수 {d}")
    ax.set_ylim(-14, 10)
    ax.set_xlabel(r"$x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$y$", fontsize=11, color=INK)
    ax.set_title(f"관측값 {n}개에 차수를 달리해 맞춘다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.2, loc="lower center", frameon=False, ncol=2)
    clean(ax)

    ax = axes[1]
    ax.plot(degs, tr_rmse, color=BLUE, lw=2.4, marker="o", ms=5,
            label="훈련 RMSE")
    ax.plot(degs, te_rmse, color=RED, lw=2.4, marker="s", ms=5,
            label="검정 RMSE")
    ax.axhline(2.0, color=MUTED, lw=1.4, ls="--")
    ax.text(15.3, 2.12, r"잡음 $\sigma = 2$", fontsize=9.6, color=INK,
            ha="right")
    best = degs[int(np.argmin(te_rmse))]
    ax.axvline(best, color=GREEN, lw=1.6, ls=":")
    ax.annotate(f"검정 RMSE 최소: 차수 {best}  ({min(te_rmse):.2f})",
                xy=(best, min(te_rmse)), xytext=(3.4, 0.9), fontsize=9.8,
                color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.annotate(f"차수 15 에서 {te_rmse[-1]:.0f}", xy=(15, te_rmse[-1]),
                xytext=(11.3, 120), fontsize=9.8, color=RED, ha="right",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 100, 1000])
    ax.set_yticklabels(["1", "2", "5", "10", "100", "1000"], fontsize=9)
    ax.minorticks_off()
    ax.set_xticks(degs)
    ax.set_xticklabels([str(d) for d in degs], fontsize=8.6)
    ax.set_ylim(0.7, 3000)
    ax.set_xlabel("다항식의 차수", fontsize=10.5, color=INK)
    ax.set_ylabel("RMSE (로그 눈금)", fontsize=10.5, color=INK)
    ax.set_title("훈련은 계속 좋아지고 검정은 돌아선다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper left", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, INT + "polynomial_degree.png")
    for d in (1, 2, 3, 5, 10, 15):
        i = degs.index(d)
        print(f"  차수 {d:2d}: 훈련 RMSE {tr_rmse[i]:.3f}  "
              f"검정 RMSE {te_rmse[i]:.3f}")
    print(f"  최적 차수 = {best}")


# === 5. 지렛대와 잔차와 영향력 ===
def fig_influence():
    rng = np.random.default_rng(6)
    n = 28
    x0 = rng.uniform(1, 7, n)
    y0 = 2 + 1.1 * x0 + rng.normal(0, 0.8, n)

    cases = [
        ("지렛대 낮고 잔차 큼", 4.0, 2 + 1.1 * 4.0 + 4.2, ORANGE),
        ("지렛대 높고 잔차 작음", 14.0, 2 + 1.1 * 14.0 + 0.4, GREEN),
        ("지렛대 높고 잔차 큼", 14.0, 2 + 1.1 * 14.0 - 6.5, RED),
    ]

    fig = plt.figure(figsize=(12.0, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], hspace=0.42,
                          wspace=0.26)

    diag = []
    for j, (ttl, px, py, col) in enumerate(cases):
        xx = np.append(x0, px)
        yy = np.append(y0, py)
        X = np.column_stack([np.ones(n + 1), xx])
        b = ols(X, yy)
        e = yy - X @ b
        s2 = (e ** 2).sum() / (n + 1 - 2)
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        h = np.diag(H)
        rstd = e / np.sqrt(s2 * (1 - h))
        cook = rstd ** 2 / 2 * h / (1 - h)
        b0 = ols(np.column_stack([np.ones(n), x0]), y0)
        diag.append((h[-1], rstd[-1], cook[-1], col, ttl))

        ax = fig.add_subplot(gs[0, j])
        gx = np.array([0, 16])
        ax.scatter(x0, y0, s=26, color=MUTED, alpha=0.9, edgecolor="none")
        ax.scatter([px], [py], s=110, color=col, zorder=6, edgecolor="white",
                   lw=1.2)
        ax.plot(gx, b0[0] + b0[1] * gx, color=MUTED, lw=2.0, ls="--",
                label="그 점을 뺀 적합")
        ax.plot(gx, b[0] + b[1] * gx, color=col, lw=2.4, label="전부 넣은 적합")
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 22)
        ax.set_title(ttl, fontsize=11, color=col, pad=6)
        ax.set_xlabel(r"$x$", fontsize=10, color=INK)
        if j == 0:
            ax.set_ylabel(r"$y$", fontsize=10, color=INK)
        ax.text(0.5, 0.94,
                f"h = {h[-1]:.2f},  " + r"$r$ = " + f"{rstd[-1]:+.2f}\n"
                + r"Cook $D$ = " + f"{cook[-1]:.3f}",
                transform=ax.transAxes, fontsize=9.2, color=INK, ha="center",
                va="top", linespacing=1.5)
        ax.legend(fontsize=8.4, loc="lower right", frameon=False)
        clean(ax)

    ax = fig.add_subplot(gs[1, :])
    hs = np.linspace(0.02, 0.75, 400)
    for D, ls in [(0.5, "--"), (1.0, "-")]:
        for sgn in (1, -1):
            rr = sgn * np.sqrt(2 * D * (1 - hs) / hs)
            ax.plot(hs, rr, color=MUTED, lw=1.4, ls=ls)
    ax.annotate(r"Cook $D = 0.5$", xy=(0.15, -2.38), xytext=(0.215, -1.45),
                fontsize=9.6, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.annotate(r"Cook $D = 1$", xy=(0.15, -3.37), xytext=(0.245, -4.5),
                fontsize=9.6, color=MUTED, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    ax.axhline(0, color=MUTED, lw=1.0)
    for hh, rr, cc, col, ttl in diag:
        ax.scatter([hh], [rr], s=80 + 300 * np.sqrt(cc), color=col, alpha=0.6,
                   edgecolor=col, lw=1.6)
        ax.annotate(ttl, xy=(hh, rr), xytext=(hh + 0.045, rr + 1.15),
                    fontsize=9.8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
    ax.set_xlim(0, 0.78)
    ax.set_ylim(-6.2, 6.2)
    ax.set_xlabel(r"지렛값 $h_i$", fontsize=11, color=INK)
    ax.set_ylabel("스튜던트화 잔차", fontsize=11, color=INK)
    ax.set_title(r"Cook 거리 = 지렛값과 잔차의 곱  "
                 r"$D_i = \dfrac{r_i^2}{p}\cdot\dfrac{h_i}{1-h_i}$",
                 fontsize=12, color=INK, pad=8)
    ax.text(0.02, -5.6, "원이 클수록 Cook 거리가 크다", fontsize=9.6,
            color=INK)
    clean(ax)

    save(fig, DIA + "leverage_residual_cook.png")
    for hh, rr, cc, col, ttl in diag:
        print(f"  {ttl:22s} h={hh:.3f}  r={rr:+.3f}  Cook D={cc:.4f}")


if __name__ == "__main__":
    fig_categorical()
    fig_coef_scale()
    fig_interaction()
    fig_polynomial()
    fig_influence()
