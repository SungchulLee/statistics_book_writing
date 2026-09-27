r"""0장 부록 A(회귀를 위한 선형대수) 정사각행렬 절의 그림 다섯 장을 생성한다.

선형대수는 기하이므로 다섯 개념을 각각 눈으로 보이는 한 장면으로 환원한다.
다섯 그림의 주제가 서로 겹치지 않도록 다음과 같이 분담했다.

  symmetric_eigen_orthogonal.png   대칭이면 고유벡터가 직교한다 (스펙트럼 정리),
                                   비대칭 행렬과 나란히 놓아 대비
  diagonalize_change_of_coords.png 좌표를 바꾸면 비스듬한 늘이기가
                                   축 방향 늘이기로 보인다
  similar_same_map_two_bases.png   같은 변환·같은 벡터인데 자(=기저)만 바꾸면
                                   성분 표기가 달라진다 (닮은 행렬)
  trace_det_unit_circle.png        면적이 det 배로 늘어난다 + 대각합은
                                   같은 총량을 다르게 쪼갠 것이다
  quadratic_form_shapes.png        이차형식의 그릇 모양: 타원 / 골짜기 / 안장

모든 수치(고윳값, 고유벡터, 대각합, 행렬식, 각도, 면적비)는 numpy 로 계산해
검산 결과를 표준출력에 찍는다. 본문에 적는 수치는 이 출력을 근거로 한다.

실행:  python3 scripts/make_ch00_eigen.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3d projection 등록)

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

OUT = "docs/ch00/linalg_regression/square_matrices/img/"


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def frame(ax, xlim, ylim):
    """원점을 지나는 축만 남긴 깔끔한 좌표평면."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)


def arrow(ax, vec, color, lw=2.2, ls="-", z=5, start=(0.0, 0.0)):
    ax.annotate("", xy=(start[0] + vec[0], start[1] + vec[1]), xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                mutation_scale=15), zorder=z)


def line_through(ax, direction, half_len, color, lw=1.0, ls=(0, (5, 4))):
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    ax.plot([-half_len * d[0], half_len * d[0]],
            [-half_len * d[1], half_len * d[1]],
            color=color, lw=lw, ls=ls, zorder=2)


def lattice(ax, b1, b2, krange, half, color=MUTED, lw=0.7, alpha=0.55):
    """b1, b2 가 만드는 격자선(각 방향의 평행선 다발)을 흐리게 깐다."""
    b1 = np.asarray(b1, float)
    b2 = np.asarray(b2, float)
    for k in krange:
        for u, v in ((b1, b2), (b2, b1)):
            p0 = k * v - half * u
            p1 = k * v + half * u
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=color, lw=lw, alpha=alpha, zorder=1)


# ==================================================================
# 1. symmetric.md — 대칭이면 고유벡터가 직교한다
# ==================================================================
def symmetric_eigen_orthogonal():
    A = np.array([[5.0, 2.0], [2.0, 2.0]])      # 본문의 예
    B = np.array([[5.0, 2.0], [0.0, 2.0]])      # 왼쪽 아래만 0 으로 바꾼 비대칭

    lamA, QA = np.linalg.eigh(A)
    order = np.argsort(lamA)[::-1]
    lamA, QA = lamA[order], QA[:, order]
    # 보기 좋은 부호로 고정
    if QA[0, 0] < 0:
        QA[:, 0] *= -1
    if QA[1, 1] < 0:
        QA[:, 1] *= -1

    lamB, VB = np.linalg.eig(B)
    order = np.argsort(lamB)[::-1]
    lamB, VB = lamB[order], VB[:, order]
    if VB[0, 0] < 0:
        VB[:, 0] *= -1
    if VB[1, 1] > 0:
        VB[:, 1] *= -1                           # 아래를 향하도록

    dotA = float(QA[:, 0] @ QA[:, 1])
    dotB = float(VB[:, 0] @ VB[:, 1])
    angA = np.degrees(np.arccos(np.clip(dotA, -1, 1)))
    angB = np.degrees(np.arccos(np.clip(dotB, -1, 1)))

    print("\n[1] symmetric")
    print("  A =", A.tolist(), " 고윳값", lamA.round(6),
          " 고유벡터(열)\n", QA.round(6))
    print("  q1.q2 =", round(dotA, 12), " 사잇각", round(angA, 4), "도")
    print("  B =", B.tolist(), " 고윳값", lamB.round(6),
          " 고유벡터(열)\n", VB.round(6))
    print("  v1.v2 =", round(dotB, 6), " 사잇각", round(angB, 4), "도")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.6))
    lim = ((-3.2, 7.0), (-3.4, 4.4))

    panels = [
        (axes[0], QA, lamA, "$\\mathbf{A}=\\mathbf{A}^{T}$",
         "$\\mathbf{A}=\\binom{5\\ \\ \\ 2}{2\\ \\ \\ 2}$",
         "고유벡터가 직교한다", angA, dotA, GREEN),
        (axes[1], VB, lamB, "$\\mathbf{B}\\neq\\mathbf{B}^{T}$",
         "$\\mathbf{B}=\\binom{5\\ \\ \\ 2}{0\\ \\ \\ 2}$",
         "고유벡터가 직교하지 않는다", angB, dotB, RED),
    ]

    for ax, V, lam, tag, mat, msg, ang, dot, accent in panels:
        frame(ax, *lim)
        # 단위원 (길이의 눈금)
        t = np.linspace(0, 2 * np.pi, 400)
        ax.plot(np.cos(t), np.sin(t), color=MUTED, lw=0.9, ls=":", zorder=2)

        for j, (col, colf) in enumerate(((BLUE, BLUE_F), (ORANGE, ORANGE_F))):
            v = V[:, j]
            line_through(ax, v, 6.6, col, lw=0.9)
            arrow(ax, lam[j] * v, col, lw=1.6, ls=(0, (4, 3)), z=4)
            arrow(ax, v, col, lw=2.6, z=6)

        # 사잇각 표시
        a0 = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        a1 = np.degrees(np.arctan2(V[1, 1], V[0, 1]))
        lo, hi = sorted((a0, a1))
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        ax.add_patch(Arc((0, 0), 1.5, 1.5, theta1=lo, theta2=hi,
                         color=accent, lw=1.8, zorder=7))
        mid = np.radians((lo + hi) / 2)
        ax.text(1.12 * np.cos(mid), 1.12 * np.sin(mid),
                f"{ang:.1f}°", color=accent, fontsize=12, fontweight="bold",
                ha="center", va="center", zorder=8)

        # 라벨 — 상대편 고유벡터의 반대쪽으로 밀어 겹침을 막는다
        for j, col in enumerate((BLUE, ORANGE)):
            v = V[:, j]
            other = V[:, 1 - j]
            perp = np.array([-v[1], v[0]])
            if perp @ other > 0:
                perp = -perp
            pv = 0.58 * v + 0.42 * perp
            ax.text(pv[0], pv[1], f"$\\mathbf{{v}}_{j + 1}$", color=col,
                    fontsize=12.5, ha="center", va="center", zorder=8)
            pl = (lam[j] + 0.35) * v + 0.6 * perp
            ax.text(pl[0], pl[1], f"$\\lambda_{j + 1}={lam[j]:.0f}$",
                    color=col, fontsize=12.5, fontweight="bold",
                    ha="center", va="center", zorder=8)

        ax.set_title(f"{tag}  —  {msg}", fontsize=13, color=INK, pad=12)
        ax.text(lim[0][0] + 0.2, lim[1][0] + 0.25, mat, fontsize=12,
                color=INK, va="bottom")
        ax.text(lim[0][1] - 0.2, lim[1][0] + 0.25,
                f"$\\mathbf{{v}}_1^{{T}}\\mathbf{{v}}_2 = {dot:.4f}$",
                fontsize=12, color=accent, va="bottom", ha="right")

    fig.text(0.5, -0.015,
             "굵은 화살표는 단위 고유벡터, 점선 화살표는 그것에 행렬을 곱한 결과다. "
             "방향은 그대로이고 길이만 $\\lambda$ 배로 바뀐다.",
             ha="center", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "symmetric_eigen_orthogonal.png")


# ==================================================================
# 2. diagonalizable.md — 좌표를 바꾸면 대각이 된다
# ==================================================================
def diagonalize_change_of_coords():
    A = np.array([[2.0, 1.0], [0.0, 3.0]])       # 본문의 예
    v1 = np.array([1.0, 0.0])                    # lambda = 2
    v2 = np.array([1.0, 1.0])                    # lambda = 3
    P = np.column_stack([v1, v2])
    Lam = np.linalg.inv(P) @ A @ P

    x = v1 + v2                                  # (2, 1)
    Ax = A @ x                                   # (5, 3)
    c = np.linalg.inv(P) @ x                     # (1, 1)
    Lc = Lam @ c                                 # (2, 3)

    print("\n[2] diagonalizable")
    print("  A =", A.tolist(), " 고윳값", np.linalg.eigvals(A).round(6))
    print("  P =", P.tolist(), " P^-1 A P =\n", Lam.round(12))
    print("  x =", x, "-> Ax =", Ax, " / c =", c, "-> Lam c =", Lc)
    print("  P (Lam c) =", P @ Lc, "  (= Ax 이어야 한다)")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.4))
    lim = ((-1.3, 6.3), (-1.3, 4.1))

    def draw(ax, b1, b2, vec, img, names, title, sub):
        frame(ax, *lim)
        lattice(ax, b1, b2, range(-3, 7), 8)
        arrow(ax, b1, MUTED, lw=2.0, z=4)
        arrow(ax, b2, MUTED, lw=2.0, z=4)
        ax.text(b1[0] * 0.55, b1[1] * 0.55 - 0.32, names[0], color=INK,
                fontsize=12, ha="center")
        ax.text(b2[0] * 0.5 - 0.3, b2[1] * 0.5 + 0.06, names[1], color=INK,
                fontsize=12, ha="center")

        # 성분 분해를 점선으로
        for v, col in ((vec, BLUE), (img, ORANGE)):
            k1, k2 = np.linalg.solve(np.column_stack([b1, b2]), v)
            mid = k1 * b1
            ax.plot([0, mid[0]], [0, mid[1]], color=col, lw=1.2,
                    ls=(0, (3, 3)), alpha=0.75, zorder=3)
            ax.plot([mid[0], v[0]], [mid[1], v[1]], color=col, lw=1.2,
                    ls=(0, (3, 3)), alpha=0.75, zorder=3)

        arrow(ax, vec, BLUE, lw=2.8, z=6)
        arrow(ax, img, ORANGE, lw=2.8, z=6)
        ax.set_title(title, fontsize=13, color=INK, pad=12)
        ax.text(lim[0][0] + 0.15, lim[1][1] - 0.15, sub, fontsize=12,
                color=INK, va="top")

    draw(axes[0], v1, v2, x, Ax, ("$\\mathbf{v}_1$", "$\\mathbf{v}_2$"),
         "표준좌표 — 늘이는 방향이 비스듬하다",
         "$\\mathbf{A}=\\binom{2\\ \\ \\ 1}{0\\ \\ \\ 3}$")
    axes[0].text(x[0] + 0.12, x[1] - 0.32,
                 "$\\mathbf{x}=\\mathbf{v}_1+\\mathbf{v}_2$",
                 color=BLUE, fontsize=12)
    axes[0].text(Ax[0] - 1.0, Ax[1] + 0.3,
                 "$\\mathbf{Ax}=2\\mathbf{v}_1+3\\mathbf{v}_2$",
                 color=ORANGE, fontsize=12)

    draw(axes[1], np.array([1.0, 0.0]), np.array([0.0, 1.0]), c, Lc,
         ("$\\mathbf{e}_1$", "$\\mathbf{e}_2$"),
         "고유벡터 좌표 — 축 방향으로만 늘인다",
         "$\\boldsymbol{\\Lambda}=\\mathbf{P}^{-1}\\mathbf{AP}"
         "=\\binom{2\\ \\ \\ 0}{0\\ \\ \\ 3}$")
    axes[1].text(c[0] + 0.15, c[1] - 0.3, "$\\mathbf{c}=(1,1)$",
                 color=BLUE, fontsize=12)
    axes[1].text(Lc[0] + 0.18, Lc[1] - 0.05,
                 "$\\boldsymbol{\\Lambda}\\mathbf{c}=(2,3)$",
                 color=ORANGE, fontsize=12)

    fig.text(0.5, -0.02,
             "두 그림은 같은 사건을 서로 다른 격자에서 본 것이다. 왼쪽에서 비스듬히 "
             "벌어져 보이던 격자가 오른쪽에서는 정사각 격자가 되고, "
             "변환은 가로 2배·세로 3배의 단순한 늘이기로 보인다.",
             ha="center", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "diagonalize_change_of_coords.png")


# ==================================================================
# 3. similar_matrices.md — 같은 변환을 다른 좌표로 적은 것
# ==================================================================
def similar_same_map_two_bases():
    A = np.array([[4.0, 1.0], [2.0, 3.0]])       # 본문의 예
    b1 = np.array([1.0, 3.0])
    b2 = np.array([2.0, 1.0])
    P = np.column_stack([b1, b2])
    B = np.linalg.inv(P) @ A @ P                 # = [[3,1],[2,4]]

    x = b1 + b2                                  # (3, 4)
    Tx = A @ x                                   # (16, 18)
    c = np.linalg.inv(P) @ x                     # (1, 1)
    Bc = B @ c                                   # (4, 6)

    print("\n[3] similar_matrices")
    print("  A =", A.tolist(), "  P =", P.tolist())
    print("  B = P^-1 A P =\n", B.round(12))
    print("  tr:", np.trace(A), np.trace(B),
          "  det:", round(np.linalg.det(A), 6), round(np.linalg.det(B), 6))
    print("  고윳값 A:", np.sort(np.linalg.eigvals(A).real).round(6),
          " B:", np.sort(np.linalg.eigvals(B).real).round(6))
    print("  x =", x, "-> Tx =", Tx)
    print("  같은 벡터의 새 좌표 c =", c, "-> Bc =", Bc,
          " / P(Bc) =", P @ Bc, "(= Tx)")

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.8))
    lim = ((-2.5, 20.5), (-2.5, 20.5))

    for ax in axes:
        frame(ax, *lim)
        arrow(ax, x, BLUE, lw=2.8, z=6)
        arrow(ax, Tx, ORANGE, lw=2.8, z=6)

    # --- 왼쪽: 표준좌표 ---
    ax = axes[0]
    lattice(ax, np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            range(-2, 21), 24, lw=0.5, alpha=0.45)
    for v, col, lab in ((x, BLUE, "$\\mathbf{x}=(3,4)$"),
                        (Tx, ORANGE, "$T\\mathbf{x}=(16,18)$")):
        ax.plot([v[0], v[0]], [0, v[1]], color=col, lw=1.1, ls=(0, (3, 3)),
                alpha=0.8, zorder=3)
        ax.plot([0, v[0]], [v[1], v[1]], color=col, lw=1.1, ls=(0, (3, 3)),
                alpha=0.8, zorder=3)
    ax.text(x[0] + 0.6, x[1] - 1.4, "$\\mathbf{x}=(3,4)$", color=BLUE,
            fontsize=12)
    ax.text(Tx[0] - 0.7, Tx[1] + 1.0, "$T\\mathbf{x}=(16,18)$",
            color=ORANGE, fontsize=12.5, ha="right")
    ax.set_title("표준기저로 읽으면", fontsize=13, color=INK, pad=12)
    ax.text(10.4, 0.6,
            "$\\mathbf{A}=\\binom{4\\ \\ \\ 1}{2\\ \\ \\ 3}$\n"
            "$\\mathbf{A}\\binom{3}{4}=\\binom{16}{18}$",
            fontsize=13, color=INK, va="bottom", linespacing=2.0)

    # --- 오른쪽: 새 기저 ---
    ax = axes[1]
    lattice(ax, b1, b2, range(-4, 13), 14, lw=0.6, alpha=0.6)
    arrow(ax, b1, GREEN, lw=2.2, z=4)
    arrow(ax, b2, GREEN, lw=2.2, z=4)
    ax.text(b1[0] - 1.1, b1[1] + 0.2, "$\\mathbf{b}_1$", color=GREEN,
            fontsize=12)
    ax.text(b2[0] + 0.3, b2[1] - 1.1, "$\\mathbf{b}_2$", color=GREEN,
            fontsize=12)
    for v, col, k1k2 in ((x, BLUE, (1, 1)), (Tx, ORANGE, (4, 6))):
        mid = k1k2[0] * b1
        ax.plot([0, mid[0]], [0, mid[1]], color=col, lw=1.1, ls=(0, (3, 3)),
                alpha=0.8, zorder=3)
        ax.plot([mid[0], v[0]], [mid[1], v[1]], color=col, lw=1.1,
                ls=(0, (3, 3)), alpha=0.8, zorder=3)
    ax.text(x[0] + 0.6, x[1] - 1.4,
            "$\\mathbf{x}=\\mathbf{b}_1+\\mathbf{b}_2$", color=BLUE,
            fontsize=12)
    ax.text(Tx[0] - 0.7, Tx[1] + 1.0,
            "$T\\mathbf{x}=4\\mathbf{b}_1+6\\mathbf{b}_2$", color=ORANGE,
            fontsize=12.5, ha="right")
    ax.set_title("같은 화살표를 새 기저로 읽으면", fontsize=13, color=INK,
                 pad=12)
    ax.text(10.4, 0.6,
            "$\\mathbf{B}=\\mathbf{P}^{-1}\\mathbf{AP}"
            "=\\binom{3\\ \\ \\ 1}{2\\ \\ \\ 4}$\n"
            "$\\mathbf{B}\\binom{1}{1}=\\binom{4}{6}$",
            fontsize=13, color=INK, va="bottom", linespacing=2.0)

    fig.text(0.5, -0.015,
             "두 판의 화살표는 위치·길이·방향이 완전히 같다. 바뀐 것은 자뿐이다. "
             "그래서 성분은 달라지고 $\\operatorname{tr}=7$, $\\det=10$, "
             "고윳값 $\\{5,2\\}$ 는 그대로다.",
             ha="center", fontsize=11, color=INK)
    fig.tight_layout()
    save(fig, "similar_same_map_two_bases.png")


# ==================================================================
# 4. trace_eigenvalues.md — 대각합은 합, 행렬식은 곱
# ==================================================================
def trace_det_unit_circle():
    A = np.array([[4.0, 2.0], [1.0, 3.0]])       # 본문의 예
    lam, V = np.linalg.eig(A)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order].real, V[:, order].real
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1
    v1, v2 = V[:, 0], V[:, 1]

    area0 = abs(np.linalg.det(np.column_stack([v1, v2])))
    area1 = abs(np.linalg.det(np.column_stack([lam[0] * v1, lam[1] * v2])))

    print("\n[4] trace_eigenvalues")
    print("  A =", A.tolist(), " 고윳값", lam.round(6))
    print("  tr(A) =", np.trace(A), " = 4 + 3,  고윳값의 합 =", lam.sum())
    print("  det(A) =", round(np.linalg.det(A), 6),
          " 고윳값의 곱 =", round(float(np.prod(lam)), 6))
    print("  평행사변형 면적", round(area0, 6), "->", round(area1, 6),
          " 비 =", round(area1 / area0, 6))
    print("  단위원 면적 pi -> 타원 면적", round(np.pi * abs(np.linalg.det(A)), 6))

    fig = plt.figure(figsize=(12.0, 5.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.22,
                          left=0.04, right=0.97, top=0.90, bottom=0.11)

    # --- 왼쪽: 면적이 det 배 ---
    ax = fig.add_subplot(gs[0, 0])
    frame(ax, (-5.2, 7.2), (-3.5, 4.0))
    ax.set_anchor("N")
    t = np.linspace(0, 2 * np.pi, 500)
    circ = np.vstack([np.cos(t), np.sin(t)])
    ell = A @ circ
    ax.fill(circ[0], circ[1], color=BLUE_F, alpha=0.9, zorder=2)
    ax.plot(circ[0], circ[1], color=BLUE, lw=1.6, zorder=3)
    ax.plot(ell[0], ell[1], color=ORANGE, lw=1.8, zorder=3)

    quad0 = np.array([[0, 0], v1, v1 + v2, v2])
    quad1 = np.array([[0, 0], lam[0] * v1, lam[0] * v1 + lam[1] * v2,
                      lam[1] * v2])
    ax.add_patch(Polygon(quad1, closed=True, facecolor=ORANGE_F,
                         edgecolor=ORANGE, lw=1.6, alpha=0.55, zorder=2))
    ax.add_patch(Polygon(quad0, closed=True, facecolor=GREEN_F,
                         edgecolor=GREEN, lw=1.6, alpha=0.95, zorder=4))
    arrow(ax, v1, GREEN, lw=2.4, z=6)
    arrow(ax, v2, GREEN, lw=2.4, z=6)
    for v, lab, lam_j in ((v1, 1, lam[0]), (v2, 2, lam[1])):
        other = v2 if lab == 1 else v1
        perp = np.array([-v[1], v[0]])
        if perp @ other > 0:
            perp = -perp
        pv = 0.6 * v + 0.45 * perp
        ax.text(pv[0], pv[1], f"$\\mathbf{{v}}_{lab}$", color=GREEN,
                fontsize=12.5, ha="center", va="center", zorder=8)
        pl = lam_j * v + 0.55 * perp
        ax.text(pl[0], pl[1],
                f"$\\lambda_{lab}\\mathbf{{v}}_{lab}$", color=ORANGE,
                fontsize=12.5, ha="center", va="center", zorder=8)
    ax.text(-5.1, 3.9,
            "$\\lambda_1=5$,  $\\lambda_2=2$\n"
            "$\\det\\mathbf{A}=\\lambda_1\\lambda_2=10$",
            fontsize=12.5, color=INK, va="top", linespacing=1.9)
    ax.text(7.1, -1.5,
            "평행사변형도 단위원도\n면적이 꼭 10배가 된다\n"
            "($\\pi \\to 10\\pi$)",
            fontsize=12, color=ORANGE, ha="right", va="top",
            linespacing=1.7)
    ax.set_title("행렬식은 면적을 몇 배로 늘리는가", fontsize=13, color=INK,
                 pad=10)

    # --- 오른쪽: 대각합은 같은 총량의 다른 배분 ---
    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(-0.4, 8.6)
    ax.set_ylim(-0.62, 1.75)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    rows = [
        (0.92, [(4.0, "$a_{11}=4$", BLUE, BLUE_F),
                (3.0, "$a_{22}=3$", BLUE, BLUE_F)],
         "대각 성분으로 쪼개면"),
        (0.02, [(5.0, "$\\lambda_1=5$", ORANGE, ORANGE_F),
                (2.0, "$\\lambda_2=2$", ORANGE, ORANGE_F)],
         "고윳값으로 쪼개면"),
    ]
    h = 0.44
    for y, parts, label in rows:
        left = 0.0
        for w, txt, edge, face in parts:
            ax.add_patch(plt.Rectangle((left, y), w, h, facecolor=face,
                                       edgecolor=edge, lw=1.6, zorder=3))
            ax.text(left + w / 2, y + h / 2, txt, ha="center", va="center",
                    fontsize=12.5, color=edge, fontweight="bold", zorder=4)
            left += w
        ax.text(0.0, y + h + 0.12, label, fontsize=12, color=INK,
                va="bottom")
    ax.plot([7.0, 7.0], [-0.28, 1.62], color=MUTED, lw=1.2, ls=(0, (4, 3)),
            zorder=2)
    ax.text(7.15, 1.62, "$\\operatorname{tr}\\mathbf{A}=7$", fontsize=13,
            color=INK, va="top")
    ax.text(3.5, -0.4, "쪼개는 방법은 달라도 총량은 7 로 같다", fontsize=12,
            color=MUTED, ha="center", va="top")
    ax.set_title("대각합은 총량을 두 가지로 쪼갠다", fontsize=13, color=INK,
                 pad=10)

    fig.text(0.5, 0.025,
             "$\\mathbf{A}=\\binom{4\\ \\ \\ 2}{1\\ \\ \\ 3}$ 의 "
             "고윳값은 5 와 2 다. 합 7 은 대각합, 곱 10 은 행렬식이고, "
             "곱은 면적이 늘어나는 비율로 눈에 보인다.",
             ha="center", fontsize=11, color=INK)
    save(fig, "trace_det_unit_circle.png")


# ==================================================================
# 5. positive_definite.md — 이차형식의 그릇 모양
# ==================================================================
def quadratic_form_shapes():
    cases = [
        (np.array([[4.0, 2.0], [2.0, 5.0]]), "양정치",
         "$\\mathbf{A}=\\binom{4\\ \\ \\ 2}{2\\ \\ \\ 5}$",
         "그릇 — 등고선은 타원"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "양반정치",
         "$\\mathbf{A}=\\binom{1\\ \\ \\ 1}{1\\ \\ \\ 1}$",
         "골짜기 — 바닥이 직선"),
        (np.array([[1.0, 2.0], [2.0, 1.0]]), "부정치",
         "$\\mathbf{A}=\\binom{1\\ \\ \\ 2}{2\\ \\ \\ 1}$",
         "안장 — 오르내림이 갈린다"),
    ]

    print("\n[5] positive_definite")
    for A, name, _, _ in cases:
        print(f"  {name}: A = {A.tolist()}  고윳값 "
              f"{np.linalg.eigvalsh(A).round(6)}  "
              f"det = {round(np.linalg.det(A), 6)}")

    g = np.linspace(-2, 2, 81)
    X1, X2 = np.meshgrid(g, g)
    rgba_pos = np.array(matplotlib.colors.to_rgba(BLUE_F))
    rgba_neg = np.array(matplotlib.colors.to_rgba(ORANGE_F))

    fig = plt.figure(figsize=(12.4, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.12, 1.0], hspace=0.12,
                          wspace=0.2)

    for j, (A, name, mat, shape) in enumerate(cases):
        Z = (A[0, 0] * X1 ** 2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2 ** 2)
        lam = np.linalg.eigvalsh(A)[::-1]

        # 이차형식이 0 이 되는 방향 (s, 1):  a11 s^2 + 2 a12 s + a22 = 0
        disc = 4 * A[0, 1] ** 2 - 4 * A[0, 0] * A[1, 1]
        nulls = []
        if disc >= 0:
            for s in np.roots([A[0, 0], 2 * A[0, 1], A[1, 1]]):
                nulls.append(np.array([float(np.real(s)), 1.0]))

        # --- 위: 곡면 (음수 부분은 주황으로) ---
        ax = fig.add_subplot(gs[0, j], projection="3d")
        fc = np.where((Z >= -1e-9)[..., None], rgba_pos, rgba_neg)
        ax.plot_surface(X1, X2, Z, rstride=4, cstride=4, facecolors=fc,
                        edgecolor=MUTED, linewidth=0.3, shade=False)
        zmin, zmax = Z.min(), Z.max()
        ax.set_zlim(zmin - 0.04 * (zmax - zmin), zmax)
        ax.view_init(elev=22, azim=45)
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        ax.set_zticks([])
        ax.tick_params(labelsize=8, colors=INK, pad=-1)
        ax.set_xlabel("$x_1$", fontsize=10, color=INK, labelpad=-6)
        ax.set_ylabel("$x_2$", fontsize=10, color=INK, labelpad=-6)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(False)
        ax.set_title(f"{name} — {shape}", fontsize=12.5, color=INK, pad=-2)

        # --- 아래: 등고선 ---
        ax = fig.add_subplot(gs[1, j])
        ax.set_aspect("equal")
        pos = [l for l in (0.5, 2, 5, 10, 18, 28) if l <= Z.max()]
        neg = [-l for l in reversed(pos) if -l >= Z.min()]
        if pos:
            ax.contour(X1, X2, Z, levels=pos, colors=BLUE, linewidths=1.3)
        if neg:
            ax.contour(X1, X2, Z, levels=neg, colors=ORANGE, linewidths=1.3)
        for d in nulls:
            d = d / np.linalg.norm(d)
            ax.plot([-3 * d[0], 3 * d[0]], [-3 * d[1], 3 * d[1]],
                    color=INK, lw=1.7, ls=(0, (5, 3)), zorder=4)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.axvline(0, color=MUTED, lw=0.7)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.tick_params(labelsize=9, colors=INK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.set_xlabel("$x_1$", fontsize=11, color=INK)
        if j == 0:
            ax.set_ylabel("$x_2$", fontsize=11, color=INK)
        ax.set_title(
            f"{mat},   $\\lambda = {lam[0]:.3g},\\ {lam[1]:.3g}$",
            fontsize=11.5, color=INK, pad=8)

    fig.text(0.5, 0.015,
             "파란 곳은 $\\mathbf{x}^{T}\\mathbf{Ax}>0$, 주황 곳은 $<0$, "
             "검은 점선은 $=0$ 인 방향이다. 고윳값의 부호가 그릇 모양을 정한다.",
             ha="center", fontsize=11, color=INK)
    save(fig, "quadratic_form_shapes.png")


# ==================================================================
if __name__ == "__main__":
    symmetric_eigen_orthogonal()
    diagonalize_change_of_coords()
    similar_same_map_two_bases()
    trace_det_unit_circle()
    quadratic_form_shapes()
