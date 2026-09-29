r"""19.4 벌점 로지스틱 회귀 두 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch19/regularized/img/l1_l2_paths.png        같은 자료, 두 벌점의 경로
  ch19/regularized/img/cv_and_stability.png   교차검증과 안정성 선택

공통 자료: n=200, p=10, 참으로 0 이 아닌 계수는 x1, x3, x7 셋뿐이다.

실행:  python3 scripts/make_ch19_regularized_figures.py   (저장소 최상위에서)
필요:  numpy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

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

OUT = "docs/ch19/regularized/img/"
os.makedirs(OUT, exist_ok=True)

TRUE_IDX = [0, 2, 6]          # x1, x3, x7
TRUE_COEF = {0: 1.6, 2: -1.3, 6: 0.9}
LAMS = np.logspace(0.7, -2.3, 60)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def tidy(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def make_data(seed=3, n=200, p=10):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    b = np.zeros(p)
    for j, v in TRUE_COEF.items():
        b[j] = v
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ b)))).astype(int)
    return X, y


def fit(X, y, lam, penalty):
    return LogisticRegression(penalty=penalty, C=1.0 / (lam * len(y)),
                              solver="liblinear", max_iter=8000).fit(X, y)


def path(X, y, penalty, lams=LAMS):
    return np.array([fit(X, y, lam, penalty).coef_[0] for lam in lams])


# =====================================================================
# 1. 같은 자료, 두 벌점의 경로  (regularization.md)
# =====================================================================
def fig_paths():
    X, y = make_data()
    P2 = path(X, y, "l2")
    P1 = path(X, y, "l1")

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)

    for ax, P, title in ((axes[0], P2, "L2 (능형): 모두 줄지만 아무도 사라지지 않는다"),
                         (axes[1], P1, "L1 (라쏘): 하나씩 정확히 $0$ 이 된다")):
        for j in range(P.shape[1]):
            if j in TRUE_IDX:
                col = {0: BLUE, 2: ORANGE, 6: GREEN}[j]
                ax.plot(LAMS, P[:, j], color=col, lw=2.4,
                        label=r"$x_{%d}$ (참값 $%.1f$)" % (j + 1, TRUE_COEF[j]))
            else:
                ax.plot(LAMS, P[:, j], color=MUTED, lw=1.1, alpha=0.75)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks([1e-2, 1e-1, 1, 5])
        ax.set_xticklabels(["0.01", "0.1", "1", "5"])
        ax.set_xlabel(r"벌점 강도 $\lambda$ (왼쪽이 강한 벌점)", color=INK,
                      fontsize=10)
        ax.set_title(title, color=INK, fontsize=11)
        tidy(ax)

    axes[0].set_ylabel("계수 추정치", color=INK, fontsize=10)
    axes[0].set_ylim(-1.55, 1.55)
    axes[0].text(3.5, -1.32, "회색 곡선 일곱 개는 잡음변수", color=MUTED,
                 fontsize=9.3, ha="left")
    axes[0].legend(loc="upper left", fontsize=8.8, frameon=False)

    # 라쏘에서 각 변수가 들어오는 지점
    ax = axes[1]
    for j in TRUE_IDX:
        nz = np.flatnonzero(np.abs(P1[:, j]) > 1e-8)
        if nz.size:
            lam_in = LAMS[nz[0]]
            ax.plot([lam_in], [0], "v", color={0: BLUE, 2: ORANGE, 6: GREEN}[j],
                    ms=8, zorder=5)
            print("  x%d enters lasso path at lambda = %.4f" % (j + 1, lam_in))
    ax.annotate("여기서부터 변수가\n하나씩 들어온다", xy=(0.15, 0.06),
                xytext=(1.6, 0.62), color=INK, fontsize=9.3, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))

    fig.tight_layout()
    save(fig, "l1_l2_paths.png")
    return P1


# =====================================================================
# 2. 교차검증과 안정성 선택  (feature_selection.md)
# =====================================================================
def fig_cv_stability():
    X, y = make_data()
    n = len(y)

    # --- K-겹 교차검증 이탈도 ---
    K = 5
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=0)
    folds = list(skf.split(X, y))
    dev = np.zeros((len(LAMS), K))
    for t, lam in enumerate(LAMS):
        for k, (tr, te) in enumerate(folds):
            m = fit(X[tr], y[tr], lam, "l1")
            q = np.clip(m.predict_proba(X[te])[:, 1], 1e-12, 1 - 1e-12)
            dev[t, k] = -2 * np.sum(y[te] * np.log(q)
                                    + (1 - y[te]) * np.log(1 - q))
    total = dev.sum(axis=1)
    mean_k = dev.mean(axis=1)
    se_k = dev.std(axis=1, ddof=1) / np.sqrt(K)
    se_tot = se_k * K

    imin = int(np.argmin(total))
    thr = total[imin] + se_tot[imin]
    cand = np.flatnonzero(total <= thr)
    i1se = int(cand.min())          # LAMS 는 큰 값에서 작은 값 순

    nz = np.array([int((np.abs(fit(X, y, lam, "l1").coef_[0]) > 1e-8).sum())
                   for lam in LAMS])

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))

    ax = axes[0]
    ax.fill_between(LAMS, total - se_tot, total + se_tot, color=BLUE_F,
                    alpha=0.9, label=r"$\pm 1$ 표준오차")
    ax.plot(LAMS, total, color=BLUE, lw=2.4, label="교차검증 이탈도")
    ax.axvline(LAMS[imin], color=RED, lw=1.4, ls="--")
    ax.axvline(LAMS[i1se], color=GREEN, lw=1.4, ls="--")
    ax.plot([LAMS[imin]], [total[imin]], "o", color=RED, ms=7, zorder=5)
    ax.plot([LAMS[i1se]], [total[i1se]], "o", color=GREEN, ms=7, zorder=5)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([1e-2, 1e-1, 1, 5])
    ax.set_xticklabels(["0.01", "0.1", "1", "5"])
    ax.set_xlabel(r"벌점 강도 $\lambda$ (왼쪽이 강한 벌점)", color=INK,
                  fontsize=10)
    ax.set_ylabel("5-겹 교차검증 이탈도", color=INK, fontsize=10)
    ax.set_title("최솟값과 1-표준오차 규칙", color=INK, fontsize=11)

    ymin, ymax = total.min(), total.max()
    pad = (ymax - ymin) * 0.22
    ax.set_ylim(ymin - pad * 0.6, ymax + pad)
    ax.annotate("최솟값 $\\lambda=%.3f$\n변수 %d 개, 이탈도 $%.1f$"
                % (LAMS[imin], nz[imin], total[imin]),
                xy=(LAMS[imin], total[imin]),
                xytext=(0.013, ymin + pad * 1.15),
                color=RED, fontsize=9.3, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.annotate("1-표준오차 $\\lambda=%.3f$\n변수 %d 개, 이탈도 $%.1f$"
                % (LAMS[i1se], nz[i1se], total[i1se]),
                xy=(LAMS[i1se], total[i1se]),
                xytext=(4.0, ymin + pad * 0.9),
                color=GREEN, fontsize=9.3, ha="left",
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    tidy(ax)

    # --- 안정성 선택 ---
    ax = axes[1]
    B = 100
    rng = np.random.default_rng(11)
    sel = np.zeros((len(LAMS), X.shape[1]))
    half = n // 2
    for _ in range(B):
        idx = rng.choice(n, size=half, replace=False)
        if y[idx].sum() in (0, half):
            continue
        for t, lam in enumerate(LAMS):
            c = fit(X[idx], y[idx], lam, "l1").coef_[0]
            sel[t] += (np.abs(c) > 1e-8)
    sel /= B

    for j in range(X.shape[1]):
        if j in TRUE_IDX:
            col = {0: BLUE, 2: ORANGE, 6: GREEN}[j]
            ax.plot(LAMS, sel[:, j], color=col, lw=2.4,
                    label=r"$x_{%d}$ (참 변수)" % (j + 1))
        else:
            ax.plot(LAMS, sel[:, j], color=MUTED, lw=1.1, alpha=0.75)
    ax.axhline(0.6, color=PURPLE, lw=1.3, ls="--")
    ax.text(0.011, 0.63, r"문턱 $\pi_{\mathrm{thr}}=0.6$", color=PURPLE,
            fontsize=9.3)
    ax.axvline(0.10, color=INK, lw=1.2, ls=":")
    ax.annotate("$\\lambda=0.10$ 근처가\n가장 깨끗하게 갈린다",
                xy=(0.105, 0.50), xytext=(1.1, 0.45), color=INK,
                fontsize=9.3, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks([1e-2, 1e-1, 1, 5])
    ax.set_xticklabels(["0.01", "0.1", "1", "5"])
    ax.set_ylim(-0.04, 1.14)
    ax.set_xlabel(r"벌점 강도 $\lambda$", color=INK, fontsize=10)
    ax.set_ylabel(r"선택 확률 $\hat\Pi_j(\lambda)$", color=INK, fontsize=10)
    ax.set_title("절반 표본 100회: 누가 계속 뽑히는가", color=INK, fontsize=11)
    ax.legend(loc="upper left", fontsize=8.8, frameon=False)
    ax.text(0.0135, 0.10, "잡음변수 일곱 개", color=MUTED, fontsize=9.3)
    tidy(ax)

    fig.tight_layout()
    save(fig, "cv_and_stability.png")

    print("  CV min  lam=%.4f dev=%.1f nz=%d" % (LAMS[imin], total[imin], nz[imin]))
    print("  CV 1se  lam=%.4f dev=%.1f nz=%d (thr=%.1f)"
          % (LAMS[i1se], total[i1se], nz[i1se], thr))
    print("  null dev (lam large) = %.1f" % total[0])
    mx = sel.max(axis=0)
    print("  max selection prob:", np.round(mx, 2))
    noise = [j for j in range(X.shape[1]) if j not in TRUE_IDX]
    print("  true:", np.round(mx[TRUE_IDX], 2),
          " noise max:", round(float(mx[noise].max()), 2))
    for j in TRUE_IDX:
        print("   x%d at CV-min lambda: pi=%.2f" % (j + 1, sel[imin, j]))


if __name__ == "__main__":
    fig_paths()
    fig_cv_stability()
