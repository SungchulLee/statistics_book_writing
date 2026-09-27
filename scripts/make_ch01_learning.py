r"""1장의 학습 패러다임 그림들을 생성한다.

세 쪽에 들어가는 그림 세 장을 만든다. 모두 작은 모의실험이며, 그림에 적힌
수치는 실행할 때마다 같은 난수 시드에서 다시 계산된다(본문에 옮겨 적은 값과
일치해야 한다).

만드는 파일:

  ch01/paradigms/img/supervised_reg_clf.png    회귀와 분류 — 레이블의 정체만 다르다
  ch01/paradigms/img/unsupervised_k.png        같은 점구름, 군집 개수를 달리하면
  ch01/modern/img/pred_vs_infer_collinear.png  강하게 상관된 두 변수: 예측 대 계수

실행:  python3 scripts/make_ch01_learning.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.

주의:  한글은 수식 $...$ 바깥에만 쓴다. mathtext 에는 한글 글리프가 없다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === 공통 설정 ===
plt.rcParams["font.family"] = "Apple SD Gothic Neo"   # 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

PARADIGMS = "docs/ch01/paradigms/img/"
MODERN = "docs/ch01/modern/img/"


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ===================================================================
# 그림 1. 지도학습 — 회귀와 분류는 레이블의 정체만 다르다
# ===================================================================
def fig_supervised():
    rng = np.random.default_rng(7)

    # --- 왼쪽: 회귀. 레이블 y 는 실수이고 점의 높이로 나타난다 ---
    f_true = lambda t: 1.0 + 0.9 * t - 0.25 * t ** 2
    n_tr, n_te, sigma = 40, 2000, 0.6
    x_tr = rng.uniform(-3, 3, n_tr)
    y_tr = f_true(x_tr) + rng.normal(0, sigma, n_tr)
    x_te = rng.uniform(-3, 3, n_te)
    y_te = f_true(x_te) + rng.normal(0, sigma, n_te)

    # 2차 다항회귀를 최소제곱으로 적합한다(닫힌 해).
    D = lambda t: np.column_stack([np.ones_like(t), t, t ** 2])
    coef = np.linalg.lstsq(D(x_tr), y_tr, rcond=None)[0]
    mse_te = np.mean((y_te - D(x_te) @ coef) ** 2)

    x_star = 1.8                         # 새 입력
    y_star = D(np.array([x_star])) @ coef

    # --- 오른쪽: 분류. 레이블 y 는 범주이고 점의 색으로 나타난다 ---
    m_tr, m_te = 60, 4000
    mu0, mu1 = np.array([-1.0, -0.6]), np.array([1.1, 0.9])
    cov = np.array([[1.0, 0.35], [0.35, 1.0]])
    L = np.linalg.cholesky(cov)

    def draw(m):
        lab = rng.integers(0, 2, m)
        z = rng.normal(0, 1, (m, 2)) @ L.T
        return z + np.where(lab[:, None] == 1, mu1, mu0), lab

    Z_tr, c_tr = draw(m_tr)
    Z_te, c_te = draw(m_te)

    # 로지스틱 회귀를 뉴턴법으로 적합한다.
    A_tr = np.column_stack([np.ones(m_tr), Z_tr])
    beta = np.zeros(3)
    for _ in range(60):
        p = 1 / (1 + np.exp(-A_tr @ beta))
        W = p * (1 - p)
        H = A_tr.T @ (A_tr * W[:, None]) + 1e-6 * np.eye(3)
        beta += np.linalg.solve(H, A_tr.T @ (c_tr - p))
    A_te = np.column_stack([np.ones(m_te), Z_te])
    acc_te = np.mean(((A_te @ beta) > 0).astype(int) == c_te)

    z_star = np.array([1.4, -1.5])       # 새 입력
    p_star = 1 / (1 + np.exp(-(np.r_[1.0, z_star] @ beta)))

    # --- 그리기 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.9))

    # 회귀
    grid = np.linspace(-3.2, 3.2, 300)
    ax1.plot(grid, D(grid) @ coef, color=BLUE, lw=2.4, zorder=4,
             label="배운 함수 $\\hat{f}$")
    ax1.scatter(x_tr, y_tr, s=34, color=BLUE_F, edgecolor=BLUE, lw=1.0,
                zorder=3, label="훈련자료 $(x_i,\\,y_i)$")
    ax1.plot([x_star, x_star], [-4.0, y_star[0]], color=RED, lw=1.2, ls=":",
             zorder=5)
    ax1.plot([-3.4, x_star], [y_star[0], y_star[0]], color=RED, lw=1.2, ls=":",
             zorder=5)
    ax1.scatter([x_star], y_star, s=90, marker="*", color=RED, zorder=6)
    ax1.text(x_star + 0.15, -3.85, "새 입력 $x^{*}$",
             color=RED, fontsize=10, ha="left", va="bottom")
    ax1.text(-3.25, y_star[0] + 0.22, f"예측 $\\hat{{y}}={y_star[0]:.2f}$",
             color=RED, fontsize=10, ha="left", va="bottom")
    ax1.set_xlim(-3.4, 3.4)
    ax1.set_ylim(-4.0, 4.0)
    ax1.set_xlabel("입력 $x$")
    ax1.set_ylabel("레이블 $y$  (실수)")
    ax1.set_title("회귀 — 레이블은 점의 높이", color=INK, fontsize=12, pad=9)
    ax1.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax1.text(0.97, 0.96, f"시험 MSE = {mse_te:.3f}", transform=ax1.transAxes,
             ha="right", va="top", fontsize=10, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    # 분류
    gx, gy = np.meshgrid(np.linspace(-4.6, 4.6, 400), np.linspace(-4.6, 4.6, 400))
    score = beta[0] + beta[1] * gx + beta[2] * gy
    ax2.contourf(gx, gy, score, levels=[-1e9, 0, 1e9],
                 colors=[BLUE_F, ORANGE_F], alpha=0.55, zorder=0)
    ax2.contour(gx, gy, score, levels=[0], colors=[INK], linewidths=2.2,
                zorder=4)
    ax2.scatter(Z_tr[c_tr == 0, 0], Z_tr[c_tr == 0, 1], s=34, color=BLUE_F,
                edgecolor=BLUE, lw=1.0, zorder=3, label="훈련자료 — $y=0$")
    ax2.scatter(Z_tr[c_tr == 1, 0], Z_tr[c_tr == 1, 1], s=34, color=ORANGE_F,
                edgecolor=ORANGE, lw=1.0, zorder=3, label="훈련자료 — $y=1$")
    ax2.scatter([z_star[0]], [z_star[1]], s=110, marker="*", color=RED, zorder=6)
    ax2.annotate(f"새 입력 $x^{{*}}$\n예측 $\\hat{{y}}=0$  ($\\hat{{p}}={p_star:.2f}$)",
                 xy=(z_star[0], z_star[1]), xytext=(z_star[0] - 0.5, z_star[1] - 1.6),
                 color=RED, fontsize=10, ha="center", va="top",
                 arrowprops=dict(arrowstyle="-", color=RED, lw=1.0))
    ax2.text(-4.35, 4.4, "결정경계  $\\hat{p}=0.5$", color=INK, fontsize=10,
             ha="left", va="top", zorder=7,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=MUTED, lw=0.8, alpha=0.9))
    ax2.set_xlim(-4.6, 4.6)
    ax2.set_ylim(-4.6, 4.6)
    ax2.set_xlabel("입력 $x_1$")
    ax2.set_ylabel("입력 $x_2$")
    ax2.set_title("분류 — 레이블은 점의 색", color=INK, fontsize=12, pad=9)
    ax2.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax2.text(0.035, 0.66, f"시험 정확도 = {acc_te:.3f}", transform=ax2.transAxes,
             ha="left", va="top", fontsize=10, color=INK,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)

    fig.suptitle("두 과제, 하나의 틀:  레이블이 붙은 자료로 $\\hat{f}$ 를 배워 새 입력에 쓴다",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, PARADIGMS + "supervised_reg_clf.png")

    print(f"  [supervised] 회귀 계수 = {np.round(coef, 4)}  (참 1, 0.9, -0.25)")
    print(f"  [supervised] 회귀 시험 MSE = {mse_te:.4f}  (잡음 하한 {sigma ** 2:.2f})")
    print(f"  [supervised] x*={x_star} 에서 예측 = {y_star[0]:.4f}, "
          f"참값 평균 = {f_true(x_star):.4f}")
    print(f"  [supervised] 분류 시험 정확도 = {acc_te:.4f}, "
          f"새 점 p_hat = {p_star:.4f}")


# ===================================================================
# 그림 2. 비지도학습 — 같은 점구름, 군집 개수를 달리하면
# ===================================================================
def kmeans(X, k, rng, n_init=30, iters=200):
    """numpy 로 구현한 k-평균. k-평균++ 초기화로 여러 번 돌려 최선을 고른다."""
    best = None
    for _ in range(n_init):
        # k-평균++ 초기화
        C = [X[rng.integers(len(X))]]
        for _ in range(k - 1):
            d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(2), axis=1)
            C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        C = np.array(C, dtype=float)
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
            newC = np.array([X[lab == j].mean(0) if np.any(lab == j)
                             else X[rng.integers(len(X))] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        wcss = float(((X - C[lab]) ** 2).sum())
        if best is None or wcss < best[0]:
            best = (wcss, lab, C)
    return best


def silhouette(X, lab):
    """평균 실루엣 계수. 군집이 하나면 정의되지 않는다."""
    Dm = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(2))
    ks = np.unique(lab)
    s = np.empty(len(X))
    for i in range(len(X)):
        own = lab == i
        inside = lab == lab[i]
        n_in = inside.sum()
        a = Dm[i, inside].sum() / (n_in - 1) if n_in > 1 else 0.0
        b = min(Dm[i, lab == j].mean() for j in ks if j != lab[i])
        s[i] = 0.0 if max(a, b) == 0 else (b - a) / max(a, b)
        del own
    return float(s.mean())


def fig_unsupervised():
    rng = np.random.default_rng(3)

    # 두 덩어리가 멀찍이 떨어져 있고, 각 덩어리는 다시 두 개의 작은 뭉치다.
    # k=2 도, k=4 도 "맞는" 답이 될 수 있는 구조다.
    centers = np.array([[0.0, 0.0], [2.9, 1.0],
                        [10.0, 4.2], [12.6, 3.0]])
    X = np.vstack([rng.normal(c, 0.85, (150, 2)) for c in centers])

    ks = [2, 3, 4]
    palette = [BLUE, ORANGE, GREEN, PURPLE, MUTED, RED]
    faces = {BLUE: BLUE_F, ORANGE: ORANGE_F, GREEN: GREEN_F,
             PURPLE: "#E1BEE7", MUTED: "#ECEFF1", RED: "#FFCDD2"}

    print("  [unsupervised] k 별 WCSS 와 평균 실루엣")
    stats = {}
    for k in range(2, 9):
        wcss, lab, C = kmeans(X, k, np.random.default_rng(100 + k))
        sil = silhouette(X, lab)
        stats[k] = (wcss, lab, C, sil)
        print(f"    k={k}:  WCSS {wcss:8.1f}   실루엣 {sil:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4))
    for ax, k in zip(axes, ks):
        wcss, lab, C, sil = stats[k]
        for j in range(k):
            col = palette[j % len(palette)]
            ax.scatter(X[lab == j, 0], X[lab == j, 1], s=16,
                       color=faces[col], edgecolor=col, lw=0.7, zorder=2)
            ax.scatter(C[j, 0], C[j, 1], s=150, marker="X", color=col,
                       edgecolor="white", lw=1.4, zorder=5)
        ax.set_title(f"$k={k}$", color=INK, fontsize=13, pad=8)
        ax.text(0.5, -0.19, f"WCSS {wcss:.0f}     평균 실루엣 {sil:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, color=INK)
        ax.set_xlim(-3.4, 16.0)
        ax.set_ylim(-3.6, 7.6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(MUTED)

    fig.suptitle("같은 점 600개, 같은 알고리즘 — 군집 개수만 바꾸었다",
                 fontsize=13, color=INK, y=1.03)
    fig.tight_layout()
    save(fig, PARADIGMS + "unsupervised_k.png")


# ===================================================================
# 그림 3. 예측 대 추론 — 강하게 상관된 두 변수
# ===================================================================
def fig_pred_vs_infer():
    rng = np.random.default_rng(11)

    n, B, rho, sigma = 120, 600, 0.99, 1.0
    beta_true = np.array([1.0, 1.0])
    Lc = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))

    def sample(m):
        Z = rng.normal(0, 1, (m, 2)) @ Lc.T
        return Z, Z @ beta_true + rng.normal(0, sigma, m)

    # 시험 자료는 한 번만 크게 뽑아 고정한다.
    Z_te, y_te = sample(20_000)
    A_te = np.column_stack([np.ones(len(Z_te)), Z_te])

    B1 = np.empty(B)
    B2 = np.empty(B)
    MSE = np.empty(B)
    for b in range(B):
        Z, y = sample(n)
        A = np.column_stack([np.ones(n), Z])
        bh = np.linalg.lstsq(A, y, rcond=None)[0]
        B1[b], B2[b] = bh[1], bh[2]
        MSE[b] = np.mean((y_te - A_te @ bh) ** 2)

    s1, s2 = B1.std(ddof=1), B2.std(ddof=1)
    ssum = (B1 + B2).std(ddof=1)
    corr_b = np.corrcoef(B1, B2)[0, 1]
    corr_m = np.corrcoef(B1, MSE)[0, 1]

    print("  [pred vs infer]")
    print(f"    beta1: 평균 {B1.mean():.4f}  표준편차 {s1:.4f}  "
          f"범위 [{B1.min():.3f}, {B1.max():.3f}]")
    print(f"    beta2: 평균 {B2.mean():.4f}  표준편차 {s2:.4f}  "
          f"범위 [{B2.min():.3f}, {B2.max():.3f}]")
    print(f"    beta1+beta2: 평균 {(B1 + B2).mean():.4f}  표준편차 {ssum:.4f}")
    print(f"    corr(beta1, beta2) = {corr_b:.4f}")
    print(f"    시험 MSE: 평균 {MSE.mean():.4f}  표준편차 {MSE.std(ddof=1):.4f}  "
          f"범위 [{MSE.min():.4f}, {MSE.max():.4f}]  (잡음 하한 {sigma ** 2:.2f})")
    print(f"    corr(beta1, 시험 MSE) = {corr_m:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9))

    # --- 왼쪽: 계수는 제각각 ---
    ax1.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax1.axvline(0, color=MUTED, lw=0.8, zorder=1)
    line = np.linspace(-1.6, 3.6, 10)
    ax1.plot(line, 2 - line, color=GREEN, lw=2.0, ls="--", zorder=3)
    ax1.scatter(B1, B2, s=14, color=BLUE, alpha=0.45, edgecolor="none", zorder=4)
    ax1.scatter([1.0], [1.0], s=180, marker="*", color=RED, zorder=6,
                edgecolor="white", lw=0.8)
    ax1.annotate("참값 $(1,\\,1)$", xy=(1.0, 1.0), xytext=(1.9, 2.5),
                 color=RED, fontsize=10,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
    ax1.text(-1.42, 3.45, "$\\beta_1+\\beta_2=2$", color=GREEN, fontsize=11,
             ha="left", va="top", zorder=7,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                       edgecolor="none"))
    ax1.set_xlim(-1.6, 3.6)
    ax1.set_ylim(-1.6, 3.6)
    ax1.set_xlabel("$\\hat{\\beta}_1$")
    ax1.set_ylabel("$\\hat{\\beta}_2$")
    ax1.set_title("추론의 눈: 계수는 제각각이다", color=INK, fontsize=12, pad=9)
    ax1.text(0.03, 0.04,
             f"$\\hat{{\\beta}}_1$ 표준편차 {s1:.2f}\n"
             f"$\\hat{{\\beta}}_2$ 표준편차 {s2:.2f}\n"
             f"둘의 상관 {corr_b:.2f}\n"
             f"$\\hat{{\\beta}}_1+\\hat{{\\beta}}_2$ 표준편차 {ssum:.2f}",
             transform=ax1.transAxes, ha="left", va="bottom", fontsize=10,
             color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    # --- 오른쪽: 예측은 거의 같다 ---
    ax2.scatter(B1, MSE, s=14, color=ORANGE, alpha=0.5, edgecolor="none",
                zorder=4)
    ax2.axhline(sigma ** 2, color=GREEN, lw=2.0, ls="--", zorder=3)
    ax2.axvline(1.0, color=RED, lw=1.2, ls=":", zorder=3)
    top = MSE.max() + 0.012
    ax2.text(3.5, sigma ** 2 - 0.004, "줄일 수 없는 잡음  $\\sigma^2=1$",
             color=GREEN, fontsize=10, ha="right", va="top")
    ax2.text(1.08, top - 0.004, "참값 $\\beta_1=1$", color=RED, fontsize=10,
             ha="left", va="top")
    ax2.set_xlim(-1.6, 3.6)
    ax2.set_ylim(0.982, top)
    ax2.set_xlabel("$\\hat{\\beta}_1$")
    ax2.set_ylabel("시험 MSE")
    ax2.set_title("예측의 눈: 어느 적합이든 똑같이 잘 맞힌다",
                  color=INK, fontsize=12, pad=9)
    ax2.text(0.03, 0.96,
             f"시험 MSE 범위 {MSE.min():.3f} ~ {MSE.max():.3f}\n"
             f"$\\hat{{\\beta}}_1$ 과의 상관 {corr_m:+.2f}",
             transform=ax2.transAxes, ha="left", va="top", fontsize=10,
             color=INK, linespacing=1.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=9)

    fig.suptitle(f"$n={n}$, 상관 ${rho}$ 인 두 설명변수로 회귀를 {B}번 반복했다",
                 fontsize=13, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, MODERN + "pred_vs_infer_collinear.png")


if __name__ == "__main__":
    fig_supervised()
    fig_unsupervised()
    fig_pred_vs_infer()
