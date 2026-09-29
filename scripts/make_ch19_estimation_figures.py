r"""19.2 추정과 추론 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch19/estimation_inference/img/information_weights.png   정보는 p=0.5 근처에 몰린다
  ch19/estimation_inference/img/newton_vs_gd.png          곡률을 쓰면 몇 걸음이면 끝난다
  ch19/estimation_inference/img/deviance_ladder.png       이탈도의 사다리와 분해
  ch19/estimation_inference/img/wald_vs_lrt.png           포물선 근사 대 실제 로그가능도

실행:  python3 scripts/make_ch19_estimation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np
from scipy import optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch19/estimation_inference/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def study_data():
    """19.1 절과 같은 공부 시간 자료 300건."""
    np.random.seed(42)
    n = 300
    hours = np.random.uniform(1, 10, n)
    noise = np.random.normal(0, 1, n)
    y = np.random.binomial(
        1, sigmoid(-3 + 0.7 * hours + 0.3 * noise)).astype(float)
    X = np.column_stack([np.ones(n), hours])
    return hours, y, X


def newton_fit(X, y, iters=12):
    th = np.zeros(X.shape[1])
    for _ in range(iters):
        p = sigmoid(X @ th)
        W = p * (1 - p)
        th = th + np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
    return th


def nll(X, y, th):
    z = X @ th
    return float(np.sum(np.logaddexp(0, z) - y * z))


# =====================================================================
# 1. 정보는 p = 0.5 근처에 몰린다  (mle.md)
# =====================================================================
def fig_information_weights():
    hours, y, X = study_data()
    th = newton_fit(X, y)
    p = sigmoid(X @ th)
    w = p * (1 - p)
    xcut = -th[0] / th[1]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # --- 가중치 함수 ---
    ax = axes[0]
    z = np.linspace(-6, 6, 500)
    wz = sigmoid(z) * (1 - sigmoid(z))
    ax.fill_between(z, 0, wz, color=BLUE_F, alpha=0.9)
    ax.plot(z, wz, color=BLUE, lw=2.4)
    ax.axhline(0.25, color=MUTED, lw=1.0, ls=":")
    ax.plot([0], [0.25], "o", color=RED, ms=7)
    ax.annotate("최대 $0.25$", xy=(0, 0.25), xytext=(1.0, 0.232),
                color=RED, fontsize=9.8, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    for zv, lab in ((2.0, r"$\pm 2$ 에서 $0.105$"), (4.0, r"$\pm 4$ 에서 $0.018$")):
        wv = sigmoid(zv) * (1 - sigmoid(zv))
        ax.plot([zv, -zv], [wv, wv], "o", color=PURPLE, ms=5.5)
    ax.annotate(r"$z=\pm 2$ 에서 $0.105$", xy=(2.0, 0.1050),
                xytext=(2.6, 0.155), color=PURPLE, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.annotate(r"$z=\pm 4$ 에서 $0.018$", xy=(4.0, 0.0177),
                xytext=(2.6, 0.062), color=PURPLE, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.30)
    ax.set_xlabel(r"로짓 $z=A[i,:]\,\theta$", color=INK, fontsize=10)
    ax.set_ylabel(r"$\sigma(z)\,(1-\sigma(z))$", color=INK, fontsize=10)
    ax.set_title("헤세행렬의 가중치는 가운데에 몰려 있다",
                 color=INK, fontsize=11)
    tidy(ax)

    # --- 같은 가중치를 자료 위에 ---
    ax = axes[1]
    grid = np.linspace(0, 11, 400)
    ax.plot(grid, sigmoid(th[0] + th[1] * grid), color=MUTED, lw=1.6,
            ls="-", label="적합된 확률 곡선")
    ax.vlines(hours, 0, w, color=BLUE, lw=1.0, alpha=0.55)
    ax.plot(hours, w, "o", color=BLUE, ms=3.2, alpha=0.8,
            label=r"관측치별 가중치 $\sigma^{(i)}(1-\sigma^{(i)})$")
    ax.axvline(xcut, color=RED, lw=1.3, ls="--")
    ax.text(xcut - 0.18, 0.86, "$p=0.5$ 인 자리\n$x=4.29$", color=RED,
            fontsize=9.5, va="top", ha="right")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("공부 시간", color=INK, fontsize=10)
    ax.set_ylabel("가중치 / 확률", color=INK, fontsize=10)
    ax.set_title("경계에서 먼 관측치는 추정에 거의 기여하지 않는다",
                 color=INK, fontsize=11)
    ax.legend(loc="upper right", fontsize=8.8, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "information_weights.png")


# =====================================================================
# 2. 뉴턴법 대 경사하강, 그리고 분리에서의 발산  (algorithms.md)
# =====================================================================
def fig_newton_vs_gd():
    hours, y, X = study_data()
    th_star = newton_fit(X, y, iters=40)
    star = nll(X, y, th_star)
    floor = 1e-14

    def mask(a):
        a = np.asarray(a, dtype=float).copy()
        hit = np.flatnonzero(a <= floor)
        if hit.size:
            a[hit[0]:] = np.nan
        return a

    # 뉴턴
    th = np.zeros(2)
    nw = [max(nll(X, y, th) - star, floor)]
    for _ in range(8):
        p = sigmoid(X @ th)
        W = p * (1 - p)
        th = th + np.linalg.solve(X.T @ (X * W[:, None]), X.T @ (y - p))
        nw.append(max(nll(X, y, th) - star, floor))

    # 경사하강 두 가지 학습률
    gd = {}
    for lr in (1e-3, 1e-4):
        t = np.zeros(2)
        rec = [max(nll(X, y, t) - star, floor)]
        for _ in range(20000):
            p = sigmoid(X @ t)
            t = t + lr * (X.T @ (y - p))
            rec.append(max(nll(X, y, t) - star, floor))
        gd[lr] = np.array(rec)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    it = np.arange(1, 20002)
    ax.plot(it, mask(gd[1e-3][:20001]), color=ORANGE, lw=2.0,
            label="경사하강 $\\alpha=0.001$")
    ax.plot(it, mask(gd[1e-4][:20001]), color=GREEN, lw=2.0,
            label="경사하강 $\\alpha=0.0001$")
    ax.plot(np.arange(1, len(nw) + 1), mask(nw), "o-", color=BLUE, lw=2.2, ms=5,
            label="뉴턴-랩슨 (IRLS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 22000)
    ax.set_ylim(3e-15, 300)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(["1", "10", "100", "1000", "10000"])
    ax.set_yticks([1e-14, 1e-11, 1e-8, 1e-5, 1e-2, 10])
    ax.set_yticklabels(["1e-14", "1e-11", "1e-8", "1e-5", "0.01", "10"])
    ax.set_xlabel("반복 횟수", color=INK, fontsize=10)
    ax.set_ylabel("최적값까지 남은 음의 로그가능도", color=INK, fontsize=10)
    ax.set_title("곡률을 쓰면 여섯 걸음, 안 쓰면 수천 걸음",
                 color=INK, fontsize=11)
    ax.legend(loc="lower left", fontsize=8.8, frameon=False)
    tidy(ax)

    # --- 분리 자료에서의 발산 ---
    ax = axes[1]
    xs = np.array([1.0, 2.0, 3.0, 4.0])
    ys = np.array([1.0, 1.0, 0.0, 0.0])
    Xs = np.column_stack([np.ones(4), xs])
    th = np.zeros(2)
    grid = np.linspace(0.3, 4.7, 600)
    shown = {1: (GREEN, "$t=1$"), 2: ("#7CB342", "$t=2$"),
             5: (ORANGE, "$t=5$"), 10: ("#C1440E", "$t=10$"),
             20: (RED, "$t=20$")}
    for t in range(1, 21):
        p = sigmoid(Xs @ th)
        W = p * (1 - p)
        th = th + np.linalg.solve(Xs.T @ (Xs * W[:, None]), Xs.T @ (ys - p))
        if t in shown:
            col, lab = shown[t]
            ax.plot(grid, sigmoid(th[0] + th[1] * grid), color=col, lw=1.9,
                    label=r"%s:  $\theta_1=%.2f$" % (lab, th[1]))

    ax.plot(xs[ys == 1], ys[ys == 1], "o", color=INK, ms=9, zorder=5)
    ax.plot(xs[ys == 0], ys[ys == 0], "o", color=INK, ms=9, zorder=5)
    ax.axvline(2.5, color=MUTED, lw=1.2, ls=":")
    ax.text(2.58, 0.52, "완전히 갈라지는 자리\n$x=2.5$", color=MUTED,
            fontsize=9.3, va="center")
    ax.set_xlim(0.3, 4.7)
    ax.set_ylim(-0.09, 1.24)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("$x$", color=INK, fontsize=10)
    ax.set_ylabel(r"$\hat p$", color=INK, fontsize=10)
    ax.set_title("분리된 자료: 반복이 멈추지 않는다", color=INK, fontsize=11)
    ax.legend(loc="lower left", fontsize=8.2, frameon=False,
              handlelength=1.4, labelspacing=0.25)
    tidy(ax)

    fig.tight_layout()
    save(fig, "newton_vs_gd.png")


# =====================================================================
# 3. 이탈도의 사다리와 분해  (deviance.md)
# =====================================================================
def fig_deviance_ladder():
    hours, y, X = study_data()
    th = newton_fit(X, y)
    p = sigmoid(X @ th)
    D = -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ybar = y.mean()
    D0 = -2 * (y.sum() * np.log(ybar) + (len(y) - y.sum()) * np.log(1 - ybar))
    d = np.sign(y - p) * np.sqrt(
        -2 * (y * np.log(p) + (1 - y) * np.log(1 - p)))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # --- 사다리 ---
    ax = axes[0]
    ax.barh([0], [D0], height=0.42, color="#ECEFF1", edgecolor=MUTED)
    ax.barh([0], [D], height=0.42, color=BLUE_F, edgecolor=BLUE)
    ax.barh([0], [D0 - D], left=[D], height=0.42, color=GREEN_F,
            edgecolor=GREEN)

    ax.text(D / 2, 0, "남은 이탈도\n$D=246.84$", ha="center", va="center",
            color=BLUE, fontsize=10)
    ax.text(D + (D0 - D) / 2, 0, "설명된 부분\n$D_0-D=152.57$",
            ha="center", va="center", color=GREEN, fontsize=10)

    for xv, lab, col in ((0.0, "포화모형\n$\\ell=0$", INK),
                         (D, "적합모형\n$D=246.84$", BLUE),
                         (D0, "영모형\n$D_0=399.40$", MUTED)):
        ax.plot([xv, xv], [-0.30, 0.30], color=col, lw=1.6)
        ax.text(xv, 0.40, lab, ha="center", va="bottom", color=col,
                fontsize=9.5)

    ax.annotate(r"$R^2_{\mathrm{dev}}=1-\dfrac{246.84}{399.40}=0.382$",
                xy=(D0 / 2, -0.55), ha="center", color=INK, fontsize=11)
    ax.set_xlim(-18, D0 + 30)
    ax.set_ylim(-0.95, 0.95)
    ax.set_yticks([])
    ax.set_xlabel("이탈도", color=INK, fontsize=10)
    ax.set_title("이탈도는 포화모형까지의 거리다", color=INK, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)

    # --- 관측치별 기여 ---
    ax = axes[1]
    big = d ** 2 > 4
    ax.plot(p[~big], (d ** 2)[~big], "o", color=BLUE, ms=4.5, alpha=0.6,
            label=r"$d_i^2 \leq 4$ (290 건, $D$ 의 $78.8\%$)")
    ax.plot(p[big], (d ** 2)[big], "o", color=RED, ms=7, alpha=0.9,
            label=r"$d_i^2 > 4$ (10 건, $D$ 의 $21.2\%$)")
    ax.axhline(4.0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 9.0)
    ax.set_xlabel(r"예측확률 $\hat p_i$", color=INK, fontsize=10)
    ax.set_ylabel(r"이탈도 기여 $d_i^2$", color=INK, fontsize=10)
    ax.set_title(r"$D=\sum_i d_i^2$ 를 관측치별로 쪼개 보면",
                 color=INK, fontsize=11)
    ax.legend(loc="upper center", fontsize=9, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "deviance_ladder.png")


# =====================================================================
# 4. 왈드의 포물선 대 실제 로그가능도  (tests.md)
# =====================================================================
def fig_wald_vs_lrt():
    # tests.md 연습문제 2 의 c = 6 자료를 그대로 만든다
    rng = np.random.default_rng(7)
    n = 100
    x = rng.normal(size=n)
    for c in (1.0, 3.0, 6.0):
        p = sigmoid(c * x)
        y = (rng.random(n) < p).astype(float)
    X = np.column_stack([np.ones(n), x])

    th_hat = newton_fit(X, y, iters=30)
    ll_hat = -nll(X, y, th_hat)
    pp = sigmoid(X @ th_hat)
    W = pp * (1 - pp)
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    se1 = np.sqrt(cov[1, 1])
    info = 1.0 / se1 ** 2

    def profile(b1):
        f = lambda b0: nll(X, y, np.array([b0[0], b1]))
        r = optimize.minimize(f, np.array([th_hat[0]]), method="Nelder-Mead",
                              options=dict(xatol=1e-8, fatol=1e-10))
        return -r.fun

    b1g = np.linspace(-1.0, 16.0, 230)
    prof = np.array([profile(b) for b in b1g])
    quad = ll_hat - 0.5 * info * (b1g - th_hat[1]) ** 2

    lam = -2 * (profile(0.0) - ll_hat)
    wald2 = (th_hat[1] / se1) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    ax = axes[0]
    ax.plot(b1g, prof, color=BLUE, lw=2.4, label="프로파일 로그가능도")
    ax.plot(b1g, quad, color=ORANGE, lw=2.0, ls="--",
            label=r"왈드의 포물선 근사")
    ax.axvline(0, color=MUTED, lw=1.0, ls=":")
    ax.plot([th_hat[1]], [ll_hat], "o", color=INK, ms=7)
    ax.text(th_hat[1] + 0.4, ll_hat + 1.5,
            r"$\hat\beta_1=6.78$", color=INK, fontsize=10)

    ll0 = profile(0.0)
    ax.annotate("", xy=(0, ll_hat), xytext=(0, ll0),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.8))
    ax.annotate("실제로 떨어지는 높이\n$\\Lambda/2=45.11$",
                xy=(-0.12, (ll_hat + ll0) / 2), xytext=(-0.5, (ll_hat + ll0) / 2),
                color=BLUE, fontsize=9.8, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9))
    q0 = ll_hat - 0.5 * info * th_hat[1] ** 2
    xo = -2.0
    ax.plot([xo, 0], [q0, q0], color=ORANGE, lw=0.9, ls=":")
    ax.plot([xo, 0], [ll_hat, ll_hat], color=ORANGE, lw=0.9, ls=":")
    ax.annotate("", xy=(xo, ll_hat), xytext=(xo, q0),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.8))
    ax.annotate("포물선이 예상한 높이\n$W^2/2=8.41$",
                xy=(xo - 0.12, (ll_hat + q0) / 2),
                xytext=(xo - 0.45, (ll_hat + q0) / 2),
                color=ORANGE, fontsize=9.8, va="center", ha="right",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.9))

    ax.set_xlim(-7.9, 16)
    ax.set_ylim(ll0 - 14, ll_hat + 9)
    ax.set_xlabel(r"$\beta_1$", color=INK, fontsize=10)
    ax.set_ylabel("로그가능도", color=INK, fontsize=10)
    ax.set_title(r"$\Lambda=%.2f$ 대 $W^2=%.2f$" % (lam, wald2),
                 color=INK, fontsize=11)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    tidy(ax)

    # --- 하우크-도너: 신호를 키우면 ---
    ax = axes[1]
    cs = np.arange(0.5, 8.01, 0.5)
    rng2 = np.random.default_rng(11)
    med_w, med_l = [], []
    for c in cs:
        ws, ls_ = [], []
        for _ in range(300):
            xx = rng2.normal(size=100)
            yy = (rng2.random(100) < sigmoid(c * xx)).astype(float)
            if yy.sum() in (0, 100):
                continue
            XX = np.column_stack([np.ones(100), xx])
            t = newton_fit(XX, yy, iters=30)
            q = sigmoid(XX @ t)
            Wd = q * (1 - q)
            try:
                cv = np.linalg.inv(XX.T @ (XX * Wd[:, None]))
            except np.linalg.LinAlgError:
                continue
            s = np.sqrt(cv[1, 1])
            if not np.isfinite(s) or s <= 0:
                continue
            pb = yy.mean()
            ll0c = 100 * (pb * np.log(pb) + (1 - pb) * np.log(1 - pb))
            ws.append((t[1] / s) ** 2)
            ls_.append(-2 * (ll0c + nll(XX, yy, t)))
        med_w.append(np.median(ws))
        med_l.append(np.median(ls_))

    ax.plot(cs, med_l, "o-", color=BLUE, lw=2.2, ms=4.5,
            label=r"가능도비 $\Lambda$")
    ax.plot(cs, med_w, "o-", color=RED, lw=2.2, ms=4.5,
            label=r"왈드 $W^2$")
    imax = int(np.argmax(med_w))
    ax.plot([cs[imax]], [med_w[imax]], "o", color=RED, ms=9,
            markeredgecolor="white")
    ax.annotate("여기서 꼭짓점 $%.1f$\n이후로는 줄어든다" % med_w[imax],
                xy=(cs[imax], med_w[imax]), xytext=(3.2, 44),
                color=RED, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.text(6.6, 52, "신호가 강해질수록\n계속 커진다", color=BLUE,
            fontsize=9.5, ha="center")
    ax.set_xlim(0.2, 8.4)
    ax.set_ylim(0, 125)
    ax.set_xlabel("참 계수의 크기 $c$", color=INK, fontsize=10)
    ax.set_ylabel("검정통계량 (300회 중앙값)", color=INK, fontsize=10)
    ax.set_title("하우크-도너: 왈드는 신호가 커지면 힘을 잃는다",
                 color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    tidy(ax)

    fig.tight_layout()
    save(fig, "wald_vs_lrt.png")
    print("  lambda=%.3f  W2=%.3f  b1=%.4f se=%.4f" %
          (lam, wald2, th_hat[1], se1))
    print("  wald peak at c=%.1f value %.2f ; last W2=%.2f LRT=%.2f" %
          (cs[imax], med_w[imax], med_w[-1], med_l[-1]))


if __name__ == "__main__":
    fig_information_weights()
    fig_newton_vs_gd()
    fig_deviance_ladder()
    fig_wald_vs_lrt()
