r"""19.1 로지스틱 회귀 기초 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch19/logistic_regression/img/logit_two_scales.png       같은 모형, 두 척도
  ch19/logistic_regression/img/odds_ratio_prob_scale.png  오즈비는 확률비가 아니다
  ch19/logistic_regression/img/crossentropy_shape.png     로그손실의 모양과 절편만 모형
  ch19/logistic_regression/img/lab_one_model_two_views.png 한 모형, 문턱에 딸린 숫자들

실행:  python3 scripts/make_ch19_logistic_regression_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np

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

OUT = "docs/ch19/logistic_regression/img/"
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


# =====================================================================
# 1. 같은 모형을 확률 척도와 로그오즈 척도에서 본다  (logit.md)
# =====================================================================
def fig_two_scales():
    b0, b1 = -3.0, 0.7
    x = np.linspace(0, 12, 400)
    z = b0 + b1 * x
    p = sigmoid(z)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # --- 확률 척도 ---
    ax = axes[0]
    ax.plot(x, p, color=BLUE, lw=2.4)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":")
    ax.axvline(-b0 / b1, color=MUTED, lw=1.0, ls=":")
    ax.set_ylim(-0.05, 1.12)
    ax.set_xlim(0, 12)

    for (xa, xb), col, txy in (((4, 5), ORANGE, (5.9, 0.26)),
                               ((9, 10), GREEN, (10.6, 0.72))):
        pa, pb = sigmoid(b0 + b1 * xa), sigmoid(b0 + b1 * xb)
        ax.plot([xa, xa], [0, pa], color=col, lw=1.0, ls="--")
        ax.plot([xb, xb], [0, pb], color=col, lw=1.0, ls="--")
        ax.plot([xb + 0.22, xb + 0.22], [pa, pb], color=col, lw=3.2,
                solid_capstyle="butt")
        for pv in (pa, pb):
            ax.plot([xb + 0.10, xb + 0.34], [pv, pv], color=col, lw=1.2)
        ax.annotate("확률 증가\n$%.3f$" % (pb - pa),
                    xy=(xb + 0.34, (pa + pb) / 2), xytext=txy,
                    color=col, fontsize=9.5, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.9))

    ax.text(0.35, 1.03, "확률 척도: 같은 1시간이 자리마다 다른 폭",
            color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("공부 시간 $x$", color=INK, fontsize=10)
    ax.set_ylabel(r"$p=P(Y=1\mid x)$", color=INK, fontsize=10)
    tidy(ax)

    # --- 로그오즈 척도 ---
    ax = axes[1]
    ax.plot(x, z, color=BLUE, lw=2.4)
    ax.axhline(0.0, color=MUTED, lw=1.0, ls=":")
    ax.set_xlim(0, 12)
    ax.set_ylim(-3.6, 6.9)

    for (xa, xb), col, dy in (((4, 5), ORANGE, -1.25), ((9, 10), GREEN, -1.25)):
        za, zb = b0 + b1 * xa, b0 + b1 * xb
        ax.plot([xa, xb], [za, za], color=col, lw=1.6)
        ax.annotate("", xy=(xb, zb), xytext=(xb, za),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.8))
        ax.text(xb + 0.25, (za + zb) / 2 + dy,
                "로그오즈 증가\n$0.700$", color=col, fontsize=9.5,
                va="center", ha="left")

    ax.text(0.35, 6.1, "로그오즈 척도: 어디서나 같은 폭 $\\beta_1=0.7$",
            color=INK, fontsize=10.5, fontweight="bold")
    ax.set_xlabel("공부 시간 $x$", color=INK, fontsize=10)
    ax.set_ylabel(r"$\log\dfrac{p}{1-p}$", color=INK, fontsize=10)
    tidy(ax)

    fig.suptitle("한 모형을 두 척도에서 보기:  "
                 r"$\log\frac{p}{1-p}=-3+0.7x$",
                 color=INK, fontsize=12.5, y=1.04)
    fig.tight_layout()
    save(fig, "logit_two_scales.png")


# =====================================================================
# 2. 오즈비 2는 확률을 두 배로 만들지 않는다  (odds_ratios.md)
# =====================================================================
def fig_or_vs_rr():
    p0 = np.linspace(0.001, 0.999, 600)
    odds0 = p0 / (1 - p0)
    p_or = 2 * odds0 / (1 + 2 * odds0)
    p_rr = np.minimum(2 * p0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    ax.fill_between(p0, p0, p_or, color=BLUE_F, alpha=0.85, zorder=1)
    ax.plot(p0, p_or, color=BLUE, lw=2.4, label="오즈비 $2$ 를 적용한 확률",
            zorder=3)
    ax.plot(p0, p_rr, color=ORANGE, lw=2.0, ls="--",
            label="확률을 두 배로 (상대위험도 $2$)", zorder=3)
    ax.plot(p0, p0, color=MUTED, lw=1.2, ls=":", label="변화 없음", zorder=2)

    for base, txy in ((0.10, (0.17, 0.06)), (0.30, (0.40, 0.25)),
                      (0.80, (0.60, 0.52))):
        new = 2 * (base / (1 - base)) / (1 + 2 * (base / (1 - base)))
        ax.plot([base], [new], "o", color=PURPLE, ms=6, zorder=4)
        ax.annotate("%.2f $\\to$ %.3f" % (base, new),
                    xy=(base, new), xytext=txy,
                    color=PURPLE, fontsize=9.5, zorder=5,
                    arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("기준 확률 $p_0$", color=INK, fontsize=10)
    ax.set_ylabel("바뀐 확률", color=INK, fontsize=10)
    ax.set_title("오즈가 두 배 될 때 확률은 얼마나 오르나",
                 color=INK, fontsize=11)
    ax.legend(loc="lower right", fontsize=8.8, frameon=False)
    tidy(ax)

    ax = axes[1]
    rr = p_or / p0
    ax.plot(p0, rr, color=GREEN, lw=2.4)
    ax.axhline(2.0, color=ORANGE, lw=1.4, ls="--")
    ax.text(0.52, 2.03, "오즈비가 뜻하는 값 $2$", color=ORANGE, fontsize=9.5)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax.axvspan(0, 0.05, color=GREEN_F, alpha=0.6)
    ax.text(0.055, 1.05, "드문 사건 영역 $p_0<0.05$", color=GREEN, fontsize=9)

    for base in (0.01, 0.30, 0.50):
        new = 2 * (base / (1 - base)) / (1 + 2 * (base / (1 - base)))
        ax.plot([base], [new / base], "o", color=PURPLE, ms=6)
        ax.annotate("$p_0=%.2f$ 이면 $%.3f$ 배" % (base, new / base),
                    xy=(base, new / base),
                    xytext=(base + 0.06, new / base + 0.06),
                    color=PURPLE, fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.9))

    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 2.25)
    ax.set_xlabel("기준 확률 $p_0$", color=INK, fontsize=10)
    ax.set_ylabel("실제 확률 배수", color=INK, fontsize=10)
    ax.set_title("드문 사건에서만 오즈비가 위험도비에 가깝다",
                 color=INK, fontsize=11)
    tidy(ax)

    fig.tight_layout()
    save(fig, "odds_ratio_prob_scale.png")


# =====================================================================
# 3. 교차엔트로피의 모양, 그리고 절편만 있는 모형의 최솟값  (likelihood.md)
# =====================================================================
def fig_crossentropy():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # --- 한 관측치의 손실 (y = 1) ---
    ax = axes[0]
    s = np.linspace(0.002, 1.0, 800)
    ax.plot(s, -np.log(s), color=BLUE, lw=2.4, label=r"로그손실 $-\log\sigma$")
    ax.plot(s, (1 - s) ** 2, color=ORANGE, lw=2.2, ls="--",
            label=r"제곱오차 $(1-\sigma)^2$")
    ax.set_ylim(0, 6.6)
    ax.set_xlim(0, 1)

    for sv, dy in ((0.01, 0.0), (0.2, 0.0)):
        ax.plot([sv], [-np.log(sv)], "o", color=PURPLE, ms=6)
    ax.annotate(r"$\sigma=0.01$ 에서 $4.605$ 대 $0.980$",
                xy=(0.02, -np.log(0.01)), xytext=(0.19, 4.60),
                color=PURPLE, fontsize=9.5, va="center",
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.annotate(r"$\sigma=0.20$ 에서 $1.609$ 대 $0.640$",
                xy=(0.21, -np.log(0.2)), xytext=(0.33, 2.60),
                color=PURPLE, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))
    ax.set_xlabel(r"예측확률 $\sigma$ (참값은 $y=1$)", color=INK, fontsize=10)
    ax.set_ylabel("한 관측치의 손실", color=INK, fontsize=10)
    ax.set_title("확신에 찬 오답의 값: 한쪽은 발산, 한쪽은 $1$ 에서 멈춘다",
                 color=INK, fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    tidy(ax)

    # --- 절편만 있는 모형 (n = 10, 성공 3) ---
    ax = axes[1]
    n, k = 10, 3
    th = np.linspace(-4.0, 2.0, 800)
    sg = sigmoid(th)
    loss = -(k * np.log(sg) + (n - k) * np.log(1 - sg))
    ax.plot(th, loss, color=BLUE, lw=2.4)

    th_hat = np.log(0.3 / 0.7)
    l_min = -(k * np.log(0.3) + (n - k) * np.log(0.7))
    ax.plot([th_hat], [l_min], "o", color=RED, ms=7, zorder=5)
    ax.plot([th_hat, th_hat], [0, l_min], color=RED, lw=1.0, ls="--")
    ax.axhline(l_min, color=RED, lw=1.0, ls="--")
    ax.set_xlim(-4, 2)
    ax.set_ylim(5.5, 14.5)
    ax.annotate(r"$\hat\theta_0=\operatorname{logit}(0.3)=-0.847$",
                xy=(th_hat - 0.05, l_min + 0.08), xytext=(-3.9, 6.75),
                color=RED, fontsize=9.8, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.text(0.42, l_min + 0.32,
            "최솟값 $6.109 = 10\\times H(0.3)$", color=RED, fontsize=9.8)
    ax.text(-3.9, 13.9, "볼록하므로 홈이 하나뿐이다\n(국소최솟값이 곧 전역최솟값)",
            color=INK, fontsize=9.8, va="top")
    ax.set_xlabel(r"절편 $\theta_0$", color=INK, fontsize=10)
    ax.set_ylabel("교차엔트로피 손실 $\\ell$", color=INK, fontsize=10)
    ax.set_title("절편만 있는 모형: 홈의 바닥이 표본비율", color=INK, fontsize=11)
    tidy(ax)

    fig.tight_layout()
    save(fig, "crossentropy_shape.png")


# =====================================================================
# 4. 한 모형, 여러 읽기  (logistic_regression.md)
# =====================================================================
def fig_lab():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, roc_auc_score

    np.random.seed(42)
    n = 300
    hours = np.random.uniform(1, 10, n)
    noise = np.random.normal(0, 1, n)
    prob = sigmoid(-3 + 0.7 * hours + 0.3 * noise)
    y = np.random.binomial(1, prob)
    X = hours.reshape(-1, 1)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(random_state=42).fit(X_tr, y_tr)
    b0 = model.intercept_[0]
    b1 = model.coef_[0][0]
    y_prob = model.predict_proba(X_te)[:, 1]
    fpr, tpr, thr = roc_curve(y_te, y_prob)
    auc = roc_auc_score(y_te, y_prob)
    j = int(np.argmax(tpr - fpr))
    tau_j = thr[j]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    # --- 적합된 곡선과 두 문턱 ---
    ax = axes[0]
    rng = np.random.default_rng(7)
    jit = rng.uniform(-0.035, 0.035, len(y_te))
    xt = X_te.ravel()
    ax.plot(xt[y_te == 0], y_te[y_te == 0] + jit[y_te == 0], "o",
            color=ORANGE, ms=4.5, alpha=0.65, label="불합격 (검정자료)")
    ax.plot(xt[y_te == 1], y_te[y_te == 1] + jit[y_te == 1], "o",
            color=GREEN, ms=4.5, alpha=0.65, label="합격 (검정자료)")

    grid = np.linspace(0, 11, 400)
    ax.plot(grid, sigmoid(b0 + b1 * grid), color=BLUE, lw=2.4,
            label="적합된 곡선")

    for tau, col, name, ha, off in ((0.5, "#546E7A", r"$\tau=0.5$", "right", -0.15),
                                    (tau_j, PURPLE, r"$\tau=0.716$", "left", 0.15)):
        xcut = (np.log(tau / (1 - tau)) - b0) / b1
        ax.axvline(xcut, color=col, lw=1.4, ls="--")
        ax.text(xcut + off, 0.24,
                "%s\n$x=%.2f$" % (name, xcut), color=col, fontsize=9.5,
                ha=ha, va="center")

    ax.set_xlim(0, 11)
    ax.set_ylim(-0.15, 1.42)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("공부 시간", color=INK, fontsize=10)
    ax.set_ylabel("합격 확률", color=INK, fontsize=10)
    ax.set_title(r"모형은 하나: $\hat\beta_0=-2.999,\ \hat\beta_1=0.710$",
                 color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)
    tidy(ax)

    # --- ROC 와 두 문턱의 위치 ---
    ax = axes[1]
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.0, ls=":")
    ax.fill_between(fpr, 0, tpr, color=BLUE_F, alpha=0.75, step="post")
    ax.step(fpr, tpr, where="post", color=BLUE, lw=2.2)

    i05 = int(np.argmin(np.abs(thr - 0.5)))
    ax.plot([fpr[i05]], [tpr[i05]], "o", color=MUTED, ms=8,
            markeredgecolor=INK)
    ax.annotate(r"$\tau=0.5$: TPR $0.849$, FPR $0.189$",
                xy=(fpr[i05] + 0.01, tpr[i05] - 0.02), xytext=(0.40, 0.44),
                color=INK, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.plot([fpr[j]], [tpr[j]], "o", color=PURPLE, ms=8,
            markeredgecolor="white")
    ax.annotate(r"$\tau=0.716$: TPR $0.811$, FPR $0.054$",
                xy=(fpr[j] + 0.01, tpr[j] - 0.01), xytext=(0.38, 0.62),
                color=PURPLE, fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.0))

    ax.text(0.53, 0.20, "AUC $= %.4f$" % auc, color=BLUE,
            fontsize=12, fontweight="bold")
    ax.text(0.53, 0.11, "문턱과 무관한 숫자", color=BLUE, fontsize=9.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("거짓양성률 FPR", color=INK, fontsize=10)
    ax.set_ylabel("참양성률 TPR", color=INK, fontsize=10)
    ax.set_title("정확도·정밀도·재현율은 이 곡선 위의 한 점일 뿐",
                 color=INK, fontsize=11)
    tidy(ax)

    fig.tight_layout()
    save(fig, "lab_one_model_two_views.png")


if __name__ == "__main__":
    fig_two_scales()
    fig_or_vs_rr()
    fig_crossentropy()
    fig_lab()
