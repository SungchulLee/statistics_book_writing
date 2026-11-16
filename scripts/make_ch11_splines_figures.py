r"""13장 '스플라인과 GAM' 네 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/splines_gams/img/smoothing_tradeoff.png    벌점과 유효자유도의 절충
  ch13/splines_gams/img/bspline_vs_natural.png    국소 기저와 경계 거동
  ch13/splines_gams/img/step_function_bins.png    계단함수와 구간 개수
  ch13/splines_gams/img/gam_partial_functions.png GAM 이 되찾는 부분함수

실행:  python3 scripts/make_ch13_splines_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, patsy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from patsy import dmatrix, build_design_matrices
from scipy.interpolate import BSpline

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

OUT = "docs/ch13/splines_gams/img/"
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


# --- P-스플라인 (B-스플라인 기저 + 2차 차분 벌점) ---
def pspline(x, y, lo, hi, n_inner=22, k=3):
    t_in = np.linspace(lo, hi, n_inner + 2)[1:-1]
    t = np.concatenate([[lo] * (k + 1), t_in, [hi] * (k + 1)])
    B = np.asarray(BSpline.design_matrix(np.clip(x, lo, hi), t, k).todense())
    m = B.shape[1]
    D = np.diff(np.eye(m), 2, axis=0)
    BtB, DtD = B.T @ B, D.T @ D

    def fit(lam):
        A = np.linalg.solve(BtB + lam * DtD, B.T)
        beta = A @ y
        edf = np.trace(B @ A)
        return beta, edf

    def curve(beta, g):
        Bg = np.asarray(BSpline.design_matrix(np.clip(g, lo, hi), t, k).todense())
        return Bg @ beta

    return fit, curve, B, t, k


# === 1. 벌점과 유효자유도의 절충 ===
def fig_smoothing():
    rng = np.random.default_rng(3)
    n = 120
    x = np.sort(rng.uniform(0, 10, n))
    truth = lambda t: np.sin(t * 0.9) * 3 + 0.35 * t
    y = truth(x) + rng.normal(0, 1.1, n)
    fit, curve, B, _, _ = pspline(x, y, 0, 10)
    g = np.linspace(0, 10, 500)

    lams = [1e-8, 5.0, 1e6]
    cols = [RED, BLUE, GREEN]
    labs = [r"$\lambda \to 0$", r"$\lambda = 5$", r"$\lambda \to \infty$"]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.scatter(x, y, s=16, color=MUTED, alpha=0.8, edgecolor="none")
    ax.plot(g, truth(g), color=INK, lw=2.0, ls="--", label="참 곡선")
    edfs = []
    for lam, c, lb in zip(lams, cols, labs):
        beta, edf = fit(lam)
        edfs.append(edf)
        ax.plot(g, curve(beta, g), color=c, lw=2.3,
                label=f"{lb}   edf = {edf:.1f}")
    ax.set_xlabel(r"$x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$y$", fontsize=11, color=INK)
    ax.set_ylim(-7, 12)
    ax.set_title("같은 기저, 벌점만 다르게", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.0, loc="upper left", frameon=False, ncol=2)
    clean(ax)

    ax = axes[1]
    lg = np.logspace(-8, 7, 120)
    ed, tr_r, te_r = [], [], []
    xt = np.sort(rng.uniform(0, 10, 400))
    yt = truth(xt) + rng.normal(0, 1.1, 400)
    for lam in lg:
        beta, edf = fit(lam)
        ed.append(edf)
        tr_r.append(np.sqrt(((y - curve(beta, x)) ** 2).mean()))
        te_r.append(np.sqrt(((yt - curve(beta, xt)) ** 2).mean()))
    ed = np.array(ed)
    ax.plot(ed, te_r, color=RED, lw=2.6, label="검정 RMSE")
    ax.plot(ed, tr_r, color=BLUE, lw=2.6, label="훈련 RMSE")
    ax.axhline(1.1, color=MUTED, lw=1.4, ls="--")
    ax.annotate(r"잡음 $\sigma = 1.1$", xy=(17, 1.1), xytext=(17, 0.74),
                fontsize=9.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    i = int(np.argmin(te_r))
    ax.scatter([ed[i]], [te_r[i]], s=70, color=GREEN, zorder=6)
    ax.annotate(f"최적 edf = {ed[i]:.1f}\n검정 RMSE {te_r[i]:.3f}",
                xy=(ed[i], te_r[i]), xytext=(11.5, 2.35), fontsize=9.8,
                color=GREEN, linespacing=1.6, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.set_xlim(1, 25.5)
    ax.set_ylim(0.6, 3.4)
    ax.set_xlabel("유효 자유도 edf  (왼쪽이 큰 벌점, 오른쪽이 작은 벌점)",
                  fontsize=10, color=INK)
    ax.set_ylabel("RMSE", fontsize=10.5, color=INK)
    ax.set_title("복잡도를 재는 자는 기저 개수가 아니라 edf 다",
                 fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "smoothing_tradeoff.png")
    print(f"  기저함수 {B.shape[1]}개")
    for lam, e in zip(lams, edfs):
        print(f"  lambda={lam:<10.0e} edf={e:.3f}")
    print(f"  최적 edf={ed[i]:.2f}, 검정 RMSE={te_r[i]:.4f}, "
          f"훈련 RMSE={tr_r[i]:.4f}")


# === 2. 국소 기저와 경계 거동 ===
def fig_bspline_natural():
    rng = np.random.default_rng(9)
    n = 90
    x = np.sort(rng.uniform(1, 9, n))
    truth = lambda t: 5 + 2.2 * np.sin(0.8 * t)
    y = truth(x) + rng.normal(0, 0.7, n)
    g = np.linspace(-1.5, 11.5, 600)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    # scipy 로 매듭을 직접 지정한 3차 B-스플라인 기저 (밖으로도 계산된다)
    lo, hi = 1.0, 9.0
    inner = np.linspace(lo, hi, 5)[1:-1]
    tk = np.concatenate([[lo] * 4, inner, [hi] * 4])

    def bs_basis(v):
        return np.asarray(
            BSpline.design_matrix(v, tk, 3, extrapolate=True).todense())

    ax = axes[0]
    gg = np.linspace(lo, hi, 600)
    Bg = bs_basis(gg)
    for j in range(Bg.shape[1]):
        ax.plot(gg, Bg[:, j], lw=2.2,
                color=plt.cm.viridis(j / (Bg.shape[1] - 1)))
    ax.plot(gg, Bg.sum(1), color=INK, lw=1.4, ls="--")
    ax.text(5.0, 1.06, "기저함수를 다 더하면 1 이다", fontsize=9.6, color=INK,
            ha="center")
    ax.text(5.0, 0.80, "각 기저는 좁은 구간에서만 0 이 아니다", fontsize=9.8,
            color=INK, ha="center")
    ax.set_ylim(0, 1.22)
    ax.set_xlabel(r"$x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$B_j(x)$", fontsize=10.5, color=INK)
    ax.set_title(f"B-스플라인 기저 {Bg.shape[1]}개 — 국소적이다", fontsize=11.5,
                 color=INK, pad=6)
    clean(ax)

    ax = axes[1]
    ax.axvspan(1, 9, color=MUTED, alpha=0.12)
    ax.scatter(x, y, s=18, color=MUTED, alpha=0.9, edgecolor="none",
               label="관측값 (음영 구간)")
    ax.plot(g, truth(g), color=INK, lw=1.8, ls="--", label="참 곡선")
    Xb = bs_basis(x)
    bb = np.linalg.lstsq(Xb, y, rcond=None)[0]
    ax.plot(g, bs_basis(g) @ bb, color=ORANGE, lw=2.5,
            label=f"B-스플라인 (기저 {Xb.shape[1]}개)")
    dm = dmatrix("cr(x, df=7)", {"x": x}, return_type="matrix")
    Xc = np.asarray(dm)
    bc = np.linalg.lstsq(Xc, y, rcond=None)[0]
    Xcg = np.asarray(build_design_matrices([dm.design_info], {"x": g})[0])
    ax.plot(g, Xcg @ bc, color=GREEN, lw=2.5, label="자연 스플라인 df=7")
    ax.axvline(1, color=MUTED, lw=1.2, ls=":")
    ax.axvline(9, color=MUTED, lw=1.2, ls=":")
    ax.annotate("경계 밖에서 자연 스플라인은\n직선을 유지한다",
                xy=(10.7, 8.9), xytext=(6.4, 12.4), fontsize=9.6, color=GREEN,
                ha="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.annotate("B-스플라인은 마지막 3차식이\n그대로 이어져 휘어 내린다",
                xy=(11.1, 3.9), xytext=(5.0, -2.4), fontsize=9.6, color=ORANGE,
                ha="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    ax.set_ylim(-5.5, 14.5)
    ax.set_xlabel(r"$x$", fontsize=11, color=INK)
    ax.set_ylabel(r"$y$", fontsize=11, color=INK)
    ax.set_title("자료 범위 밖으로 나가면", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.0, loc="lower right", frameon=False, ncol=1)
    clean(ax)

    fig.tight_layout()
    save(fig, "bspline_vs_natural.png")
    ends = np.array([-1.5, 11.5])
    pb, pc = bs_basis(ends) @ bb, np.asarray(
        build_design_matrices([dm.design_info], {"x": ends})[0]) @ bc
    print(f"  B-스플라인   훈련 RMSE={np.sqrt(((y - Xb @ bb)**2).mean()):.4f}  "
          f"x=-1.5 예측={pb[0]:9.2f}  x=11.5 예측={pb[1]:9.2f}")
    print(f"  자연 스플라인 훈련 RMSE={np.sqrt(((y - Xc @ bc)**2).mean()):.4f}  "
          f"x=-1.5 예측={pc[0]:9.2f}  x=11.5 예측={pc[1]:9.2f}")
    print(f"  참값 x=-1.5 {truth(-1.5):.3f},  x=11.5 {truth(11.5):.3f}, "
          f"자료 범위 [{x.min():.2f}, {x.max():.2f}]")


# === 3. 계단함수와 구간 개수 ===
def fig_steps():
    rng = np.random.default_rng(21)
    n = 600
    age = rng.uniform(0, 120, n)
    truth = lambda a: 640 - 6.0 * a + 0.048 * a ** 2
    price = truth(age) + rng.normal(0, 70, n)
    at = rng.uniform(0, 120, 3000)
    pt = truth(at) + rng.normal(0, 70, 3000)

    def step_fit(K, a, p, a2):
        edges = np.linspace(0, 120, K + 1)
        idx = np.clip(np.digitize(a, edges[1:-1]), 0, K - 1)
        means = np.array([p[idx == k].mean() if (idx == k).any() else p.mean()
                          for k in range(K)])
        i2 = np.clip(np.digitize(a2, edges[1:-1]), 0, K - 1)
        return edges, means, means[idx], means[i2]

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    ax = axes[0]
    ax.scatter(age, price, s=12, color=MUTED, alpha=0.4, edgecolor="none")
    g = np.linspace(0, 120, 300)
    ax.plot(g, truth(g), color=INK, lw=2.0, ls="--", label="참 곡선 (U자)")
    A = np.column_stack([np.ones(n), age])
    bl = np.linalg.lstsq(A, price, rcond=None)[0]
    ax.plot(g, bl[0] + bl[1] * g, color=RED, lw=2.4,
            label=f"직선  (기울기 {bl[1]:+.2f})")
    edges, means, _, _ = step_fit(5, age, price, at)
    for k in range(5):
        ax.plot([edges[k], edges[k + 1]], [means[k]] * 2, color=BLUE, lw=3.0,
                solid_capstyle="butt",
                label="계단함수 (구간 5개)" if k == 0 else None)
        if k < 4:
            ax.plot([edges[k + 1]] * 2, [means[k], means[k + 1]], color=BLUE,
                    lw=1.2, ls=":")
    ax.set_xlabel("주택 나이 (년)", fontsize=10.5, color=INK)
    ax.set_ylabel("가격 (천 달러)", fontsize=10.5, color=INK)
    ax.set_ylim(280, 880)
    ax.set_title("직선은 U자를 못 담고 계단은 담는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.2, loc="lower center", frameon=False, ncol=2)
    clean(ax)

    ax = axes[1]
    Ks = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 80, 120]
    tr, te = [], []
    for K in Ks:
        _, _, f_tr, f_te = step_fit(K, age, price, at)
        tr.append(np.sqrt(((price - f_tr) ** 2).mean()))
        te.append(np.sqrt(((pt - f_te) ** 2).mean()))
    ax.plot(Ks, te, color=RED, lw=2.4, marker="s", ms=5, label="검정 RMSE")
    ax.plot(Ks, tr, color=BLUE, lw=2.4, marker="o", ms=5, label="훈련 RMSE")
    ax.axhline(70, color=MUTED, lw=1.4, ls="--")
    ax.annotate(r"잡음 $\sigma = 70$", xy=(30, 70), xytext=(30, 58),
                fontsize=9.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    best = Ks[int(np.argmin(te))]
    ax.axvline(best, color=GREEN, lw=1.6, ls=":")
    ax.annotate(f"검정 RMSE 최소: 구간 {best}개  ({min(te):.1f})",
                xy=(best, min(te)), xytext=(13, 108), fontsize=9.8,
                color=GREEN, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 120])
    ax.set_xticklabels(["2", "5", "10", "20", "50", "120"], fontsize=9)
    ax.minorticks_off()
    ax.set_ylim(50, 140)
    ax.set_xlabel("구간의 개수 K  (관측 600개)", fontsize=10.5, color=INK)
    ax.set_ylabel("RMSE", fontsize=10.5, color=INK)
    ax.set_title("구간을 잘게 나눌수록 좋아지지 않는다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper left", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "step_function_bins.png")
    print(f"  직선 기울기 {bl[1]:+.3f}, 구간 5개 평균 {means.round(1)}")
    for K, a, b in zip(Ks, tr, te):
        if K in (2, 5, 10, 20, 50, 120):
            print(f"  K={K:4d}  훈련 RMSE={a:7.2f}  검정 RMSE={b:7.2f}")
    print(f"  최적 K = {best}")


# === 4. GAM 이 되찾는 부분함수 ===
def fig_gam():
    rng = np.random.default_rng(15)
    n = 700
    x1 = rng.uniform(0, 10, n)
    x2 = rng.uniform(0, 10, n)
    f1 = lambda t: 3 * np.sin(0.65 * t)
    f2 = lambda t: 0.09 * (t - 5) ** 2 - 0.75
    y = 10 + f1(x1) + f2(x2) + rng.normal(0, 1.2, n)
    x1t = rng.uniform(0, 10, 3000)
    x2t = rng.uniform(0, 10, 3000)
    yt = 10 + f1(x1t) + f2(x2t) + rng.normal(0, 1.2, 3000)

    def basis(desc, xa, xb):
        d1 = dmatrix(desc, {"x": xa}, return_type="matrix")
        return d1, np.asarray(d1)

    models = {}
    # 선형
    X = np.column_stack([np.ones(n), x1, x2])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    Xt = np.column_stack([np.ones(3000), x1t, x2t])
    models["선형"] = (np.sqrt(((y - X @ b) ** 2).mean()),
                     np.sqrt(((yt - Xt @ b) ** 2).mean()))
    # 이차 다항
    X = np.column_stack([np.ones(n), x1, x1 ** 2, x2, x2 ** 2])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    Xt = np.column_stack([np.ones(3000), x1t, x1t ** 2, x2t, x2t ** 2])
    models["이차 다항"] = (np.sqrt(((y - X @ b) ** 2).mean()),
                       np.sqrt(((yt - Xt @ b) ** 2).mean()))
    # GAM (자연 스플라인 가법모형)
    d1 = dmatrix("cr(x, df=6) - 1", {"x": x1}, return_type="matrix")
    d2 = dmatrix("cr(x, df=6) - 1", {"x": x2}, return_type="matrix")
    A1, A2 = np.asarray(d1), np.asarray(d2)
    X = np.column_stack([np.ones(n), A1, A2])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    B1t = np.asarray(build_design_matrices([d1.design_info], {"x": x1t})[0])
    B2t = np.asarray(build_design_matrices([d2.design_info], {"x": x2t})[0])
    Xt = np.column_stack([np.ones(3000), B1t, B2t])
    models["GAM"] = (np.sqrt(((y - X @ b) ** 2).mean()),
                     np.sqrt(((yt - Xt @ b) ** 2).mean()))

    g = np.linspace(0, 10, 400)
    G1 = np.asarray(build_design_matrices([d1.design_info], {"x": g})[0])
    G2 = np.asarray(build_design_matrices([d2.design_info], {"x": g})[0])
    k1 = A1.shape[1]
    hat1 = G1 @ b[1:1 + k1]
    hat2 = G2 @ b[1 + k1:]
    hat1 = hat1 - hat1.mean() + f1(g).mean()
    hat2 = hat2 - hat2.mean() + f2(g).mean()

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.1))

    for ax, ff, hh, nm in [(axes[0], f1, hat1, r"$x_1$"),
                           (axes[1], f2, hat2, r"$x_2$")]:
        ax.plot(g, ff(g), color=INK, lw=2.4, ls="--", label="참 부분함수")
        ax.plot(g, hh, color=BLUE, lw=2.6, label="GAM 이 추정한 함수")
        ax.axhline(0, color=MUTED, lw=1.0)
        ax.set_xlabel(nm, fontsize=11, color=INK)
        ax.set_ylabel(f"부분효과", fontsize=10.5, color=INK)
        ax.set_ylim(-4.2, 4.2)
        ax.set_title(nm + " 의 부분함수", fontsize=11.5, color=INK, pad=6)
        ax.legend(fontsize=9.4, loc="upper right", frameon=False)
        clean(ax)

    ax = axes[2]
    names = list(models)
    w = 0.34
    idx = np.arange(3)
    tr = [models[k][0] for k in names]
    te = [models[k][1] for k in names]
    ax.bar(idx - w / 2, tr, w, color=BLUE, label="훈련 RMSE")
    ax.bar(idx + w / 2, te, w, color=RED, label="검정 RMSE")
    for i in range(3):
        ax.text(i - w / 2, tr[i] + 0.05, f"{tr[i]:.2f}", ha="center",
                fontsize=10, color=INK)
        ax.text(i + w / 2, te[i] + 0.05, f"{te[i]:.2f}", ha="center",
                fontsize=10, color=INK)
    ax.axhline(1.2, color=MUTED, lw=1.4, ls="--")
    ax.annotate(r"잡음 $\sigma = 1.2$", xy=(1.05, 1.2), xytext=(1.05, 2.45),
                fontsize=9.8, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.1))
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 3.2)
    ax.set_ylabel("RMSE", fontsize=10.5, color=INK)
    ax.set_title("세 모형의 예측 정확도", fontsize=11.5, color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="upper right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "gam_partial_functions.png")
    for k in names:
        print(f"  {k:8s} 훈련 RMSE={models[k][0]:.4f}  "
              f"검정 RMSE={models[k][1]:.4f}")
    print(f"  GAM 부분함수 최대 오차: f1 {np.abs(hat1-f1(g)).max():.3f}, "
          f"f2 {np.abs(hat2-f2(g)).max():.3f}")


if __name__ == "__main__":
    fig_smoothing()
    fig_bspline_natural()
    fig_steps()
    fig_gam()
