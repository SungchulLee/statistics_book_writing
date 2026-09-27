r"""0장 부록 A(회귀를 위한 선형대수) 중 "선형대수와 통계가 만나는" 세 쪽의 그림.

세 그림은 각각 다음 주장을 떠받친다.

  1. 이차형식 $z^T A z$ 가 카이제곱이 되는지는 오직 $A$ 의 고윳값이 결정한다.
     대칭 멱등행렬이면 고윳값이 0 과 1 뿐이고, 1 인 고윳값의 개수(= 계수
     = 대각합 = 사영하는 부분공간의 차원)가 그대로 자유도가 된다.
     대각합이 같아도 고윳값이 0/1 이 아니면 카이제곱이 아니다.
  2. 단순회귀에서 기울기 추정량의 분산은 $\sigma^2 / S_{xx}$ 다. 같은 $n$,
     같은 $\sigma$ 라도 설계행렬 둘째 열의 퍼짐 $S_{xx}$ 가 작으면 적합직선이
     $(\bar x, \bar Y)$ 를 축으로 심하게 흔들린다.
  3. 다중회귀에서 $\hat\beta$ 의 공분산은 $\sigma^2 (X^T X)^{-1}$ 이다. 설계행렬의
     두 열을 상관 $\rho$ 로 만들면 두 계수 추정량의 상관은 정확히 $-\rho$ 가
     되고 분산은 $1/(1-\rho^2)$ 배로 부풀어, 표본분포 타원이 기울며 커진다.

만드는 파일:

  docs/ch00/linalg_regression/linalg_statistics/img/quadratic_form_chi2.png
  docs/ch00/linalg_regression/linalg_statistics/img/slope_variance_spread.png
  docs/ch00/linalg_regression/linalg_statistics/img/beta_covariance_ellipse.png

실행:  python3 scripts/make_ch00_olsdist.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG 로 커밋되므로 CI 에서 다시 그리지 않는다.

본문에 인용할 수치는 모두 stdout 으로 함께 인쇄한다.
"""

import os
from math import cos, exp, gamma, sin, radians, sqrt

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

OUT = "docs/ch00/linalg_regression/linalg_statistics/img/"


def style(ax):
    """공통 축 장식: 위/오른쪽 테두리를 없애고 색을 통일한다."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)


def finish(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {path}")


def chi2_pdf(x, k):
    """자유도 k 인 카이제곱 밀도 (scipy 없이)."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = x > 0
    out[m] = (x[m] ** (k / 2.0 - 1.0) * np.exp(-x[m] / 2.0)
              / (2.0 ** (k / 2.0) * gamma(k / 2.0)))
    return out


def density(sample, lo, hi, bins=140):
    """히스토그램으로 추정한 밀도를 (중심, 높이) 로 돌려준다."""
    h, edges = np.histogram(sample, bins=bins, range=(lo, hi), density=True)
    return 0.5 * (edges[:-1] + edges[1:]), h


def ellipse_95(cov, center, n_pts=400):
    """공분산 cov 의 95% 등고선 타원 (카이제곱 2 자유도 분위수 5.9915)."""
    vals, vecs = np.linalg.eigh(cov)
    t = np.linspace(0.0, 2.0 * np.pi, n_pts)
    circle = np.vstack([np.cos(t), np.sin(t)])
    pts = vecs @ (np.sqrt(vals * 5.9915)[:, None] * circle)
    return center[0] + pts[0], center[1] + pts[1]


# ===========================================================================
# 그림 1. 이차형식이 카이제곱이 되는 조건 — 고윳값과 부분공간의 차원
# ===========================================================================

def fig_quadratic_form_chi2():
    """이차형식 z^T A z 의 분포는 A 의 고윳값만으로 정해진다.

    (a) 기하. R^2 에서 표준정규 구름을 1차원 부분공간 L 과 그 직교여공간으로
        쪼갠다. 사영행렬 P 의 고윳값은 L 방향에서 1, 수직 방향에서 0 이고,
        각 성분의 제곱이 자유도 1 인 카이제곱이 된다.
    (b) 고윳값 스펙트럼. 대각합이 모두 3 인 세 대칭행렬을 나란히 놓는다.
        멱등행렬만 고윳값이 0 과 1 뿐이다.
    (c) 모의실험 밀도. 대각합(=평균)이 같아도 멱등이 아니면 카이제곱이 아니다.
    """
    rng = np.random.default_rng(7)
    n, B = 6, 400_000

    # --- 세 대칭행렬: 대각합이 모두 3 이다 -------------------------------
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))       # 무작위 직교행렬
    A1 = Q[:, :3] @ Q[:, :3].T                         # 계수 3 인 사영행렬
    A2 = 0.5 * np.eye(n)                               # 고윳값이 모두 0.5
    lam3 = np.array([1.8, 0.8, 0.4, 0.0, 0.0, 0.0])
    A3 = Q @ np.diag(lam3) @ Q.T                       # 대칭이지만 멱등은 아님

    mats = [
        ("$\\mathbf{A}_1$", "멱등, 계수 3", A1, BLUE),
        ("$\\mathbf{A}_2$", "고윳값 전부 0.5", A2, ORANGE),
        ("$\\mathbf{A}_3$", "고윳값 1.8, 0.8, 0.4", A3, PURPLE),
    ]

    Z = rng.normal(size=(B, n))
    stats = []
    for name, note, A, col in mats:
        q = np.einsum("bi,ij,bj->b", Z, A, Z)
        tr, tr2 = np.trace(A), np.trace(A @ A)
        stats.append((name, note, A, col, q, tr, tr2))
        print(f"    {name[1:-1]:>14}  대각합 {tr:.4f}  2tr(A^2) {2*tr2:.4f}"
              f"   모의 평균 {q.mean():.4f}  모의 분산 {q.var():.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5))

    # --- (a) 기하 ---------------------------------------------------------
    ax = axes[0]
    ang = radians(33.0)
    u = np.array([cos(ang), sin(ang)])          # L 의 방향 (고윳값 1)
    v = np.array([-sin(ang), cos(ang)])         # L 의 직교여공간 (고윳값 0)

    cloud = rng.normal(size=(500, 2))
    ax.scatter(cloud[:, 0], cloud[:, 1], s=7, color=MUTED, alpha=0.30,
               linewidths=0)

    R = 3.4
    ax.plot([-R * u[0], R * u[0]], [-R * u[1], R * u[1]], color=BLUE, lw=2.0)
    ax.plot([-R * v[0], R * v[0]], [-R * v[1], R * v[1]], color=ORANGE, lw=2.0)

    z = np.array([0.55, 2.75])
    pz = (z @ u) * u
    mz = (z @ v) * v
    ax.plot([0, z[0]], [0, z[1]], color=INK, lw=2.2,
            solid_capstyle="round", zorder=5)
    ax.plot([z[0], pz[0]], [z[1], pz[1]], color=BLUE, lw=1.2, ls=":")
    ax.plot([z[0], mz[0]], [z[1], mz[1]], color=ORANGE, lw=1.2, ls=":")
    ax.plot([0, pz[0]], [0, pz[1]], color=BLUE, lw=3.2, zorder=6)
    ax.plot([0, mz[0]], [0, mz[1]], color=ORANGE, lw=3.2, zorder=6)
    ax.scatter([z[0]], [z[1]], s=46, color=INK, zorder=7)

    ax.annotate("$\\mathbf{z}$", (z[0], z[1]), xytext=(10, 4),
                textcoords="offset points", color=INK, fontsize=12)
    ax.annotate("$\\mathbf{Pz}$", (pz[0], pz[1]), xytext=(10, -20),
                textcoords="offset points", color=BLUE, fontsize=12)
    ax.annotate("$(\\mathbf{I}-\\mathbf{P})\\mathbf{z}$", (mz[0], mz[1]),
                xytext=(-3.30, 0.95), textcoords="data",
                color=ORANGE, fontsize=11, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=0.8,
                                shrinkA=2, shrinkB=6))

    ax.text(-2.95, -2.25, "고윳값 1 방향", color=BLUE, fontsize=9.5,
            ha="left", va="top")
    ax.text(1.15, -1.25, "고윳값 0 방향", color=ORANGE, fontsize=9.5,
            ha="left", va="top")

    ax.text(-3.45, -3.35,
            "$\\|\\mathbf{Pz}\\|^2 \\sim \\chi^2_1$,"
            "   $\\|(\\mathbf{I}-\\mathbf{P})\\mathbf{z}\\|^2 \\sim \\chi^2_1$",
            fontsize=11, color=INK, ha="left", va="bottom")

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.set_aspect("equal")
    ax.set_xticks([-2, 0, 2])
    ax.set_yticks([-2, 0, 2])
    ax.axhline(0, color=MUTED, lw=0.6, zorder=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=0)
    style(ax)
    ax.set_title("(a) 자유도는 사영하는 부분공간의 차원",
                 color=INK, fontsize=11.5, pad=9)

    # --- (b) 고윳값 스펙트럼 ----------------------------------------------
    ax = axes[1]
    for k, (name, note, A, col, q, tr, tr2) in enumerate(stats):
        y = 2 - k
        lam = np.sort(np.linalg.eigvalsh(A))[::-1]
        lam[np.abs(lam) < 1e-10] = 0.0
        ax.axhline(y, color=MUTED, lw=0.6, alpha=0.5, zorder=0)
        ax.scatter(lam, np.full_like(lam, y), s=95, color=col,
                   zorder=3, clip_on=False)
        ax.text(-0.22, y + 0.30, f"{name}  {note},  대각합 {tr:.0f}",
                color=col, fontsize=10.5, ha="left", va="bottom", zorder=4)

    ax.axvline(0.0, color=INK, lw=1.0, ls="--", alpha=0.55, zorder=1)
    ax.axvline(1.0, color=INK, lw=1.0, ls="--", alpha=0.55, zorder=1)
    ax.text(0.0, 2.80, "0", color=INK, fontsize=10, ha="center", va="bottom")
    ax.text(1.0, 2.80, "1", color=INK, fontsize=10, ha="center", va="bottom")
    ax.text(-0.22, -0.55, "멱등행렬의 고윳값은 0 과 1 뿐이다",
            color=INK, fontsize=9.5, ha="left", va="center")

    ax.set_xlim(-0.25, 2.1)
    ax.set_ylim(-0.85, 3.05)
    ax.set_yticks([])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("고윳값", color=INK, fontsize=10)
    ax.spines["left"].set_visible(False)
    style(ax)
    ax.set_title("(b) 고윳값만이 분포를 결정한다",
                 color=INK, fontsize=11.5, pad=18)

    # --- (c) 모의실험 밀도 ------------------------------------------------
    ax = axes[2]
    grid = np.linspace(0.0, 13.0, 600)
    ax.fill_between(grid, chi2_pdf(grid, 3), color=BLUE_F, zorder=0)
    ax.plot(grid, chi2_pdf(grid, 3), color=BLUE, lw=1.4, ls="--", zorder=2,
            label="$\\chi^2_3$ 밀도")
    peak = chi2_pdf(grid, 3).max()
    for name, note, A, col, q, tr, tr2 in stats:
        c, h = density(q, 0.0, 13.0)
        peak = max(peak, h.max())
        ax.plot(c, h, color=col, lw=1.9, zorder=3,
                label=f"{name}: 분산 {q.var():.2f}")
    print(f"    (c) 밀도 최댓값 {peak:.4f}")

    ax.axvline(3.0, color=INK, lw=0.9, ls=":", alpha=0.7)
    ax.annotate("세 이차형식의 평균이 모두 3 이다", (3.0, 0.145),
                xytext=(4.3, 0.185), textcoords="data",
                color=INK, fontsize=9.5, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.8, alpha=0.7,
                                shrinkA=4, shrinkB=2))

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 0.33)
    ax.set_xlabel("$\\mathbf{z}^T\\mathbf{A}\\mathbf{z}$", color=INK,
                  fontsize=11)
    ax.set_ylabel("밀도", color=INK, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    style(ax)
    ax.set_title("(c) 대각합이 같아도 멱등이 아니면 카이제곱이 아니다",
                 color=INK, fontsize=11.5, pad=9)

    fig.tight_layout(w_pad=2.2)
    finish(fig, "quadratic_form_chi2.png")


# ===========================================================================
# 그림 2. 단순회귀 — 기울기의 분산은 왜 sigma^2 / Sxx 인가
# ===========================================================================

def fig_slope_variance_spread():
    """설계행렬 둘째 열의 퍼짐이 기울기 추정량의 분산을 결정한다.

    (a),(b) 같은 n, 같은 sigma, 같은 참 직선. x 를 넓게 퍼뜨린 설계와
            좁게 몰아둔 설계에서 각각 60 번 적합한 직선을 겹쳐 그린다.
            모든 직선이 (xbar, Ybar) 를 지나므로 부채꼴이 된다.
    (c)     기울기 추정량의 모의 밀도와 이론 정규밀도.
    (d)     여러 설계에서 측정한 분산과 곡선 sigma^2 / Sxx.
    """
    rng = np.random.default_rng(11)
    n, b0, b1, sig = 10, 1.0, 2.0, 1.0
    B = 200_000

    designs = [
        ("넓게 퍼뜨린 설계", np.linspace(0.0, 10.0, n), BLUE, BLUE_F),
        ("좁게 몰아둔 설계", np.linspace(4.0, 6.0, n), ORANGE, ORANGE_F),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8))

    # --- (a), (b) 적합직선의 부채꼴 ---------------------------------------
    grid_x = np.linspace(-0.6, 10.6, 60)
    sim = []
    for k, (label, x, col, fill) in enumerate(designs):
        ax = axes[0, k]
        Sxx = ((x - x.mean()) ** 2).sum()
        X = np.column_stack([np.ones(n), x])
        XtXi = np.linalg.inv(X.T @ X)

        Y = b0 + b1 * x + rng.normal(0.0, sig, size=(B, n))
        beta = Y @ X @ XtXi
        sim.append((label, x, col, fill, Sxx, beta))
        print(f"    {label}  Sxx {Sxx:8.4f}"
              f"   모의 분산 {beta[:, 1].var():.5f} 이론 {sig**2/Sxx:.5f}"
              f"   모의 표준편차 {beta[:, 1].std():.4f}"
              f" 이론 {sig/sqrt(Sxx):.4f}")

        for j in range(60):
            ax.plot(grid_x, beta[j, 0] + beta[j, 1] * grid_x,
                    color=col, lw=0.8, alpha=0.30, zorder=2)
        ax.plot(grid_x, b0 + b1 * grid_x, color=INK, lw=2.0, zorder=4,
                label="참 직선")
        ax.scatter(x, Y[0], s=34, color=col, edgecolor="white",
                   linewidths=0.8, zorder=5, label="한 표본")
        ax.scatter([x.mean()], [b0 + b1 * x.mean()], s=90, marker="X",
                   color=RED, zorder=6)
        ax.annotate("$(\\bar{x}, \\bar{Y})$", (x.mean(), b0 + b1 * x.mean()),
                    xytext=(10, -20), textcoords="offset points",
                    color=RED, fontsize=11)

        ax.set_xlim(-0.8, 10.8)
        ax.set_ylim(-9.0, 31.0)
        ax.set_xlabel("$x$", color=INK, fontsize=11)
        ax.set_ylabel("$y$", color=INK, fontsize=11)
        leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
        for t in leg.get_texts():
            t.set_color(INK)
        style(ax)
        tag = "(a)" if k == 0 else "(b)"
        ax.set_title(f"{tag} {label}:  $S_{{xx}} = {Sxx:.2f}$",
                     color=INK, fontsize=11.5, pad=9)

    # --- (c) 기울기의 표본분포 --------------------------------------------
    ax = axes[1, 0]
    grid_b = np.linspace(0.2, 3.8, 600)
    for label, x, col, fill, Sxx, beta in sim:
        sd = sig / sqrt(Sxx)
        theo = np.exp(-0.5 * ((grid_b - b1) / sd) ** 2) / (sd * sqrt(2 * np.pi))
        ax.fill_between(grid_b, theo, color=fill, alpha=0.75, zorder=1)
        c, h = density(beta[:, 1], 0.2, 3.8, bins=180)
        ax.plot(c, h, color=col, lw=1.9, zorder=3,
                label=f"{label}: 표준편차 {beta[:, 1].std():.3f}")
    ax.axvline(b1, color=INK, lw=0.9, ls=":", alpha=0.8)
    ax.text(b1 + 0.14, 2.85, "참 기울기 2", color=INK, fontsize=9.5,
            ha="left", va="top")

    ax.set_xlim(0.2, 3.8)
    ax.set_ylim(0, 4.1)
    ax.set_xlabel("$\\hat{\\beta}_1$", color=INK, fontsize=11)
    ax.set_ylabel("밀도", color=INK, fontsize=10)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    style(ax)
    ax.set_title("(c) 두 설계에서 기울기 추정량의 표본분포",
                 color=INK, fontsize=11.5, pad=9)

    # --- (d) 분산과 1/Sxx -------------------------------------------------
    ax = axes[1, 1]
    halfwidths = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    ms, ts, ss = [], [], []
    B2 = 120_000
    for w in halfwidths:
        x = np.linspace(5.0 - w, 5.0 + w, n)
        Sxx = ((x - x.mean()) ** 2).sum()
        X = np.column_stack([np.ones(n), x])
        XtXi = np.linalg.inv(X.T @ X)
        Y = b0 + b1 * x + rng.normal(0.0, sig, size=(B2, n))
        bb = (Y @ X @ XtXi)[:, 1]
        ms.append(bb.var())
        ts.append(sig ** 2 / Sxx)
        ss.append(Sxx)
        print(f"    반폭 {w:>4.1f}  Sxx {Sxx:8.4f}   모의 분산 {bb.var():.5f}"
              f"   이론 {sig**2/Sxx:.5f}")

    grid_s = np.linspace(3.0, 110.0, 400)
    ax.plot(grid_s, sig ** 2 / grid_s, color=GREEN, lw=2.0, zorder=2,
            label="이론 $\\sigma^2 / S_{xx}$")
    ax.scatter(ss, ms, s=58, color=GREEN, edgecolor="white", linewidths=0.9,
               zorder=4, label="모의실험 분산")
    for j, (label, x, col, fill, Sxx, beta) in enumerate(sim):
        ax.scatter([Sxx], [beta[:, 1].var()], s=140, marker="X", color=col,
                   zorder=5)
        # j = 0 은 오른쪽 끝(넓은 설계)이므로 라벨을 왼쪽으로 붙인다
        off = (-12, 16) if j == 0 else (16, 10)
        ax.annotate(label, (Sxx, beta[:, 1].var()),
                    xytext=off, textcoords="offset points",
                    color=col, fontsize=10,
                    ha="right" if j == 0 else "left")

    ax.set_xlim(0, 112)
    ax.set_ylim(0, 0.30)
    ax.set_xlabel("$S_{xx} = \\sum_i (x_i - \\bar{x})^2$", color=INK,
                  fontsize=11)
    ax.set_ylabel("$\\operatorname{Var}(\\hat{\\beta}_1)$", color=INK,
                  fontsize=11)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    for t in leg.get_texts():
        t.set_color(INK)
    style(ax)
    ax.set_title("(d) 분산은 $S_{xx}$ 에 반비례한다",
                 color=INK, fontsize=11.5, pad=9)

    fig.tight_layout(w_pad=2.4, h_pad=2.6)
    finish(fig, "slope_variance_spread.png")

    return sim


# ===========================================================================
# 그림 3. 다중회귀 — beta_hat 의 공분산 구조와 설계행렬 열의 상관
# ===========================================================================

def fig_beta_covariance_ellipse():
    """설계행렬의 두 열이 상관될수록 두 계수의 추정이 얽힌다.

    열을 직교화해 만들면 X^T X 가 정확히 [[n,0,0],[0,n,n rho],[0,n rho,n]] 이
    되므로 이론값이 닫힌 형태로 나온다.

      Var(beta_j) = sigma^2 / (n (1 - rho^2)),   Corr(beta_1, beta_2) = -rho

    (a),(b) rho = 0 과 rho = 0.9 에서 beta_hat 의 산포와 이론 95% 타원.
    (c)     분산팽창 1/(1 - rho^2) 곡선과 측정값.
    """
    rng = np.random.default_rng(23)
    n, sig = 40, 1.0
    beta = np.array([1.0, 2.0, -1.0])
    B = 60_000

    # 1 에 직교하는 정규직교 두 벡터 (설계행렬의 열을 만드는 재료)
    M = rng.normal(size=(n, 3))
    M[:, 0] = 1.0
    Qn, _ = np.linalg.qr(M)
    u1, u2 = Qn[:, 1], Qn[:, 2]

    def build(rho):
        c = sqrt(n)
        x1 = c * u1
        x2 = c * (rho * u1 + sqrt(1.0 - rho ** 2) * u2)
        return np.column_stack([np.ones(n), x1, x2])

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4))

    panels = [(0.0, BLUE, BLUE_F, "(a)"), (0.9, ORANGE, ORANGE_F, "(b)")]
    for k, (rho, col, fill, tag) in enumerate(panels):
        ax = axes[k]
        X = build(rho)
        XtXi = np.linalg.inv(X.T @ X)
        cov_t = sig ** 2 * XtXi[1:, 1:]

        Y = X @ beta + rng.normal(0.0, sig, size=(B, n))
        bh = Y @ X @ XtXi
        cov_m = np.cov(bh[:, 1:].T)
        r_col = np.corrcoef(X[:, 1], X[:, 2])[0, 1]
        r_beta = cov_m[0, 1] / sqrt(cov_m[0, 0] * cov_m[1, 1])
        print(f"    열 상관 {r_col:+.4f}"
              f"   Var(b1) 모의 {cov_m[0,0]:.5f} 이론 {cov_t[0,0]:.5f}"
              f"   corr(b1,b2) 모의 {r_beta:+.4f} 이론 {-rho:+.4f}")

        ax.scatter(bh[:4000, 1], bh[:4000, 2], s=5, color=col, alpha=0.20,
                   linewidths=0, zorder=2)
        ex, ey = ellipse_95(cov_t, beta[1:])
        ax.plot(ex, ey, color=INK, lw=2.0, zorder=5,
                label="이론 95% 타원")
        ax.scatter([beta[1]], [beta[2]], s=80, marker="X", color=RED, zorder=6,
                   label="참값 $(2,\\,-1)$")

        # 타원의 주축 방향
        vals, vecs = np.linalg.eigh(cov_t)
        for i in (0, 1):
            d = vecs[:, i] * sqrt(vals[i] * 5.9915)
            ax.plot([beta[1] - d[0], beta[1] + d[0]],
                    [beta[2] - d[1], beta[2] + d[1]],
                    color=MUTED, lw=1.0, ls="--", zorder=4)

        ax.text(1.00, -2.10,
                f"두 계수 추정량의 상관 ${r_beta:+.2f}$",
                color=col, fontsize=10.5, ha="left", va="bottom")

        ax.set_xlim(0.95, 3.05)
        ax.set_ylim(-2.15, 0.05)
        ax.set_aspect("equal")
        ax.set_anchor("N")          # 제목 높이를 (c) 와 맞춘다
        ax.set_xlabel("$\\hat{\\beta}_1$", color=INK, fontsize=11)
        ax.set_ylabel("$\\hat{\\beta}_2$", color=INK, fontsize=11)
        leg = ax.legend(frameon=False, fontsize=9.5, loc="upper right")
        for t in leg.get_texts():
            t.set_color(INK)
        style(ax)
        ax.set_title(f"{tag} 설계행렬 두 열의 상관 $\\rho = {rho:.1f}$",
                     color=INK, fontsize=11.5, pad=9)

    # --- (c) 분산팽창 -----------------------------------------------------
    ax = axes[2]
    rhos = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
    meas, theo = [], []
    B3 = 60_000
    for rho in rhos:
        X = build(rho)
        XtXi = np.linalg.inv(X.T @ X)
        Y = X @ beta + rng.normal(0.0, sig, size=(B3, n))
        bh = Y @ X @ XtXi
        base = sig ** 2 / n
        meas.append(bh[:, 1].var() / base)
        theo.append(1.0 / (1.0 - rho ** 2))
        print(f"    rho {rho:.2f}   분산팽창 모의 {meas[-1]:7.4f}"
              f"   이론 {theo[-1]:7.4f}")

    grid_r = np.linspace(0.0, 0.965, 400)
    ax.plot(grid_r, 1.0 / (1.0 - grid_r ** 2), color=GREEN, lw=2.0, zorder=2,
            label="이론 $1/(1-\\rho^2)$")
    ax.scatter(rhos, meas, s=58, color=GREEN, edgecolor="white",
               linewidths=0.9, zorder=4, label="모의실험")
    for rho, col in ((0.0, BLUE), (0.9, ORANGE)):
        i = rhos.index(rho)
        ax.scatter([rho], [meas[i]], s=140, marker="X", color=col, zorder=5)
    ax.annotate("(a)", (0.0, meas[0]), xytext=(10, 8),
                textcoords="offset points", color=BLUE, fontsize=11)
    ax.annotate("(b)", (0.9, meas[rhos.index(0.9)]), xytext=(-24, 2),
                textcoords="offset points", color=ORANGE, fontsize=11)

    ax.set_xlim(-0.03, 1.0)
    ax.set_ylim(0, 12.5)
    ax.set_xlabel("설계행렬 두 열의 상관 $\\rho$", color=INK, fontsize=11)
    ax.set_ylabel("$\\operatorname{Var}(\\hat{\\beta}_1) \\,/\\, (\\sigma^2/n)$",
                  color=INK, fontsize=11)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    style(ax)
    ax.set_title("(c) 열이 닮을수록 분산이 부푼다",
                 color=INK, fontsize=11.5, pad=9)

    fig.tight_layout(w_pad=2.6)
    finish(fig, "beta_covariance_ellipse.png")


# ===========================================================================

if __name__ == "__main__":
    print("그림 1 — 이차형식과 카이제곱")
    fig_quadratic_form_chi2()
    print("그림 2 — 단순회귀 기울기의 분산")
    fig_slope_variance_spread()
    print("그림 3 — beta_hat 의 공분산 구조")
    fig_beta_covariance_ellipse()
    print("완료")
