r"""0장 부록 A(회귀를 위한 선형대수) 정사각행렬 절의 네 쪽에 들어갈 그림을 만든다.

선형대수는 기하이므로 네 쪽은 서로 다른 기하를 하나씩 맡는다.

  square_matrices/img/oblique_vs_orthogonal.png
      projection.md — 같은 부분공간 위로 사영해도 어느 방향을 따라
      누르느냐에 따라 상이 달라진다(빗각 사영 대 직교사영).

  square_matrices/img/least_squares_pythagoras.png
      orthogonal_projection.md — 잔차가 부분공간에 직교한다는 것과
      그것이 "가장 가까운 점"이라는 것이 같은 말임을, 피타고라스 분해와
      거리함수의 포물선으로 보인다.

  square_matrices/img/idempotent_action.png
      idempotent.md — P^2 = P 의 뜻(두 번 눌러도 제자리)과 고윳값이
      1 과 0 뿐이라는 것, 그래서 공간이 상공간과 영공간으로 쪼개진다는 것.

  square_matrices/img/gram_collinearity.png
      gram_matrices.md — X^T X 가 열들의 내적표라는 것과, 열이 가까워지면
      평행사변형 넓이·행렬식·최소고윳값이 함께 0 으로 간다는 것.

실행:  python3 scripts/make_ch00_projection.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Circle

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


def arrow(ax, tail, head, color, lw=2.0, ls="-", z=4, mut=15):
    ax.add_patch(FancyArrowPatch(tuple(tail), tuple(head),
                                 arrowstyle="-|>", mutation_scale=mut,
                                 lw=lw, ls=ls, color=color,
                                 shrinkA=0, shrinkB=0, zorder=z))


def right_angle(ax, corner, d1, d2, size=0.26, color=INK, lw=1.1):
    """corner 에서 방향 d1, d2 사이의 직각 표시."""
    c = np.asarray(corner, float)
    u = np.asarray(d1, float); u = u / np.linalg.norm(u)
    v = np.asarray(d2, float); v = v / np.linalg.norm(v)
    p = np.array([c + u * size, c + u * size + v * size, c + v * size])
    ax.plot(p[:, 0], p[:, 1], color=color, lw=lw, zorder=6)


def bare(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def axes_cross(ax, xlim, ylim):
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    bare(ax)


# ==================================================================
# 1. projection.md — 빗각 사영과 직교사영
# ==================================================================
def oblique_vs_orthogonal():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0))

    x = np.array([3.0, 2.0])
    xlim, ylim = (-1.7, 5.0), (-1.3, 3.5)

    specs = [
        dict(ax=axes[0], title="비스듬한 사영",
             P=np.array([[1.0, -1.0], [0.0, 0.0]]),
             w=np.array([1.0, 1.0]),
             mat=r"$\mathbf{P}\mathbf{x}=(x_{1}-x_{2},\;0)^{T}$",
             wlab=r"$\mathcal{W}=\mathrm{span}\{(1,1)^{T}\}$",
             dx=0.22, ha="left", col=ORANGE, fill=ORANGE_F),
        dict(ax=axes[1], title="직교사영",
             P=np.array([[1.0, 0.0], [0.0, 0.0]]),
             w=np.array([0.0, 1.0]),
             mat=r"$\mathbf{P}\mathbf{x}=(x_{1},\;0)^{T}$",
             wlab=r"$\mathcal{W}=\mathcal{V}^{\perp}=\mathrm{span}\{(0,1)^{T}\}$",
             dx=-0.22, ha="right", col=BLUE, fill=BLUE_F),
    ]

    for s in specs:
        ax = s["ax"]
        P, w, col = s["P"], s["w"], s["col"]
        px = P @ x
        d = np.linalg.norm(x - px)
        print(f"  [{s['title']}] Px = {px}, 잔차 길이 = {d:.4f}")

        axes_cross(ax, xlim, ylim)

        # 사영이 눌러 없애는 방향의 다발(섬유)
        for t in np.arange(-2.0, 5.01, 1.0):
            base = np.array([t, 0.0])
            p0, p1 = base - 3.4 * w / np.linalg.norm(w), base + 3.4 * w / np.linalg.norm(w)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                    color=MUTED, lw=0.7, ls=(0, (3, 4)), zorder=1)

        # 목표 부분공간 V = x축
        ax.plot([xlim[0] + 0.25, xlim[1] - 0.25], [0, 0], color=GREEN, lw=3.2, zorder=3)
        ax.text(xlim[0] + 0.20, 0.42, r"$\mathcal{V}=\mathrm{span}\{(1,0)^{T}\}$",
                color=GREEN, fontsize=11, ha="left", zorder=7,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

        # x 와 사영된 상
        arrow(ax, (0, 0), x, INK, lw=2.2)
        ax.text(x[0] + 0.12, x[1] + 0.16, r"$\mathbf{x}=(3,2)^{T}$",
                color=INK, fontsize=12)

        arrow(ax, (0, 0), px, col, lw=2.6)
        ax.plot(*px, "o", color=col, ms=7, zorder=5)
        ax.text(px[0] + 0.14, -0.52, r"$\mathbf{P}\mathbf{x}$", color=col, fontsize=13)

        # 사영이 지나간 길 = 잔차
        arrow(ax, x, px, col, lw=1.8, ls=(0, (5, 3)), z=4, mut=13)
        mid = (x + px) / 2
        ax.text(mid[0] + s["dx"], mid[1] + 0.10, f"길이 {d:.3f}",
                color=col, fontsize=11, ha=s["ha"], zorder=7,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

        ax.set_title(s["title"], color=INK, fontsize=14, pad=10)
        ax.text(xlim[0] + 0.15, ylim[1] - 0.25, s["mat"], color=INK,
                fontsize=12, va="top")
        ax.text(xlim[0] + 0.15, ylim[0] + 0.18, s["wlab"], color=MUTED,
                fontsize=10.5, va="bottom", zorder=7,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

    # 직교사영 쪽에만 직각 표시
    right_angle(axes[1], (3.0, 0.0), (-1, 0), (0, 1), size=0.24, color=BLUE)

    fig.suptitle("같은 부분공간 위로 사영해도 누르는 방향이 다르면 상이 다르다",
                 color=INK, fontsize=15, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "oblique_vs_orthogonal.png")


# ==================================================================
# 2. orthogonal_projection.md — 직교성 = 최단거리
# ==================================================================
def least_squares_pythagoras():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2),
                             gridspec_kw=dict(width_ratios=[1.22, 1.0]))

    # --- 왼쪽: 평면과 직각삼각형 (길이·각도가 실제 값) ---
    ax = axes[0]
    nyh = np.sqrt(13.0)      # ||y_hat||
    ne = 2.5                 # ||e||
    ny = np.sqrt(13.0 + 6.25)
    tv = 1.8                 # 부분공간 안의 다른 점 v 의 위치
    nyv = np.sqrt(6.25 + (nyh - tv) ** 2)
    print(f"  ||y_hat|| = {nyh:.4f}, ||e|| = {ne:.4f}, ||y|| = {ny:.4f}, "
          f"||y - v|| = {nyv:.4f}")

    plane = Polygon([(-1.35, -1.05), (5.05, -1.05), (6.35, 0.95), (-0.05, 0.95)],
                    closed=True, facecolor=BLUE_F, edgecolor=BLUE,
                    lw=1.3, alpha=0.9, zorder=1)
    ax.add_patch(plane)
    ax.text(6.15, -0.82, r"$\mathrm{col}(\mathbf{X})$", color=BLUE,
            fontsize=13, ha="right")

    ax.plot([-1.0, 5.8], [0, 0], color=BLUE, lw=0.9, ls=(0, (4, 3)), zorder=2)

    O = np.array([0.0, 0.0])
    yh = np.array([nyh, 0.0])
    y = np.array([nyh, ne])
    v = np.array([tv, 0.0])

    arrow(ax, O, y, INK, lw=2.3)
    arrow(ax, O, yh, BLUE, lw=2.6)
    arrow(ax, yh, y, ORANGE, lw=2.4)
    ax.plot([v[0], y[0]], [v[1], y[1]], color=MUTED, lw=1.5, ls=(0, (5, 3)), zorder=3)
    ax.plot(*v, "o", color=MUTED, ms=6, zorder=4)
    ax.plot(*yh, "o", color=BLUE, ms=7, zorder=5)
    ax.plot(*y, "o", color=INK, ms=7, zorder=5)

    right_angle(ax, yh, (-1, 0), (0, 1), size=0.3, color=ORANGE, lw=1.4)

    ax.text(y[0] + 0.15, y[1] + 0.12, r"$\mathbf{y}$", color=INK, fontsize=15)
    ax.text(1.55, 0.30, r"$\hat{\mathbf{y}}=\mathbf{P}\mathbf{y}$",
            color=BLUE, fontsize=14)
    ax.text(yh[0] + 0.62, 1.92, r"$\mathbf{e}=\mathbf{y}-\hat{\mathbf{y}}$",
            color=ORANGE, fontsize=14)
    ax.text(v[0], -0.34, r"$\mathbf{v}$", color=MUTED, fontsize=13,
            ha="center", va="top")

    ax.text(1.15, 1.72, f"{ny:.3f}", color=INK, fontsize=11, rotation=34)
    ax.text(2.70, -0.34, f"{nyh:.3f}", color=BLUE, fontsize=11,
            ha="center", va="top")
    ax.text(yh[0] + 0.16, 1.05, f"{ne:.3f}", color=ORANGE, fontsize=11)
    ax.text(2.72, 0.98, f"{nyv:.3f}", color=MUTED, fontsize=11, rotation=54)

    ax.text(-1.30, 3.30,
            r"$\|\mathbf{y}\|^{2}=\|\hat{\mathbf{y}}\|^{2}+\|\mathbf{e}\|^{2}$",
            color=INK, fontsize=14, va="top")
    ax.text(-1.30, 2.72,
            f"{ny:.3f}" + r"$^{2}=$" + f"{nyh:.3f}" + r"$^{2}+$" + f"{ne:.3f}"
            + r"$^{2}$",
            color=MUTED, fontsize=11, va="top")

    ax.set_xlim(-1.6, 6.6); ax.set_ylim(-1.35, 3.45)
    ax.set_aspect("equal")
    bare(ax)
    ax.set_title("잔차는 부분공간에 수직이다", color=INK, fontsize=14, pad=8)

    # --- 오른쪽: 거리의 제곱은 포물선 ---
    ax = axes[1]
    t = np.linspace(-3.0, 3.0, 400)
    f = 6.25 + t ** 2
    ax.plot(t, f, color=BLUE, lw=2.4, zorder=3)
    ax.axhline(6.25, color=ORANGE, lw=1.3, ls=(0, (5, 3)), zorder=2)
    ax.fill_between(t, 6.25, f, color=BLUE_F, alpha=0.75, zorder=1)

    ax.plot(0, 6.25, "o", color=ORANGE, ms=8, zorder=4)
    ax.text(0.15, 6.05, r"$\mathbf{v}=\hat{\mathbf{y}}$ 에서 최소",
            color=ORANGE, fontsize=11.5, va="top")
    ax.text(-2.95, 6.55, r"$\|\mathbf{e}\|^{2}=6.25$", color=ORANGE, fontsize=11.5)

    t0 = nyh - tv
    f0 = 6.25 + t0 ** 2
    ax.plot(t0, f0, "o", color=MUTED, ms=7, zorder=5)
    ax.annotate("", xy=(t0, f0), xytext=(t0, 6.25),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
    ax.text(t0 + 0.14, (6.25 + f0) / 2,
            r"$\|\hat{\mathbf{y}}-\mathbf{v}\|^{2}$", color=INK,
            fontsize=12, va="center")
    ax.text(t0, f0 + 0.35, r"$\mathbf{v}$", color=MUTED, fontsize=13,
            ha="center")

    ax.set_xlabel(r"부분공간 안에서 $\hat{\mathbf{y}}$ 로부터 떨어진 거리",
                  color=INK, fontsize=11.5)
    ax.set_ylabel(r"$\|\mathbf{y}-\mathbf{v}\|^{2}$", color=INK, fontsize=13)
    ax.set_xlim(-3.05, 3.05); ax.set_ylim(5.2, 16.2)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.set_title("그래서 가장 가까운 점이다", color=INK, fontsize=14, pad=8)
    ax.text(-2.95, 15.4,
            r"$\|\mathbf{y}-\mathbf{v}\|^{2}=\|\mathbf{e}\|^{2}"
            r"+\|\hat{\mathbf{y}}-\mathbf{v}\|^{2}$",
            color=INK, fontsize=12.5, va="top")

    fig.tight_layout()
    save(fig, "least_squares_pythagoras.png")


# ==================================================================
# 3. idempotent.md — P^2 = P 와 고윳값 1, 0
# ==================================================================
def idempotent_action():
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.3))

    u = np.array([2.0, 1.0]) / np.sqrt(5.0)       # col(P) 방향
    k = np.array([-1.0, 2.0]) / np.sqrt(5.0)      # ker(P) 방향
    P = np.outer(u, u)
    print(f"  P =\n{np.round(P, 4)}")
    print(f"  tr(P) = {np.trace(P):.4f}, rank(P) = {np.linalg.matrix_rank(P)}, "
          f"P^2 = P : {np.allclose(P @ P, P)}, u.k = {u @ k:.1e}")

    # --- 왼쪽: 두 번 눌러도 제자리 ---
    ax = axes[0]
    xlim, ylim = (-2.4, 4.6), (-1.5, 3.9)
    axes_cross(ax, xlim, ylim)

    L = 4.3
    ax.plot([-L * u[0], L * u[0]], [-L * u[1], L * u[1]],
            color=GREEN, lw=3.0, zorder=2)
    ax.text(3.60, 1.38, r"$\mathrm{col}(\mathbf{P})$", color=GREEN, fontsize=12.5)
    ax.plot([-1.6 * k[0], 2.6 * k[0]], [-1.6 * k[1], 2.6 * k[1]],
            color=PURPLE, lw=1.8, ls=(0, (5, 3)), zorder=2)
    ax.text(-1.30, 2.50, r"$\mathrm{ker}(\mathbf{P})$", color=PURPLE, fontsize=12.5,
            ha="right")

    x = np.array([1.0, 3.0])
    px = P @ x
    p2x = P @ px
    print(f"  x = {x}, Px = {np.round(px, 6)}, P(Px) = {np.round(p2x, 6)}")

    arrow(ax, (0, 0), x, INK, lw=2.2)
    ax.text(x[0] - 0.15, x[1] + 0.18, r"$\mathbf{x}=(1,3)^{T}$", color=INK,
            fontsize=12.5, ha="right")

    arrow(ax, x, px, ORANGE, lw=1.9, ls=(0, (5, 3)), mut=14)
    ax.text(1.25, 2.62, r"$(\mathbf{I}-\mathbf{P})\mathbf{x}$", color=ORANGE,
            fontsize=12, ha="left")

    arrow(ax, (0, 0), px, BLUE, lw=2.6)
    ax.plot(*px, "o", color=BLUE, ms=8, zorder=6)
    ax.text(px[0] + 0.28, px[1] - 0.72, r"$\mathbf{P}\mathbf{x}=(2,1)^{T}$",
            color=BLUE, fontsize=12.5)

    # 한 번 더 적용해도 제자리
    ax.annotate("다시 눌러도\n"
                r"$\mathbf{P}(\mathbf{P}\mathbf{x})=\mathbf{P}\mathbf{x}$",
                xy=px, xytext=(3.05, 2.42), color=RED, fontsize=11.5,
                ha="left", va="center", linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4,
                                shrinkB=8,
                                connectionstyle="arc3,rad=-0.30"))

    right_angle(ax, px, -u, k, size=0.24, color=MUTED)
    ax.text(xlim[0] + 0.12, ylim[1] - 0.10,
            r"$\mathbf{P}=\mathbf{u}\mathbf{u}^{T},\;\;$"
            r"$\mathbf{u}=(2,1)^{T}/\sqrt{5}$",
            color=INK, fontsize=12, va="top")
    ax.set_title("두 번 눌러도 더 움직이지 않는다", color=INK, fontsize=14, pad=10)

    # --- 오른쪽: 고윳값은 1 과 0 뿐 ---
    ax = axes[1]
    xlim, ylim = (-1.75, 1.95), (-1.45, 1.95)
    axes_cross(ax, xlim, ylim)

    ax.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=MUTED,
                        lw=1.0, ls=(0, (3, 4)), zorder=2))
    ax.plot([-1.35 * u[0], 1.35 * u[0]], [-1.35 * u[1], 1.35 * u[1]],
            color=GREEN, lw=2.2, zorder=3)
    ax.plot([-1.25 * k[0], 1.25 * k[0]], [-1.25 * k[1], 1.25 * k[1]],
            color=PURPLE, lw=1.6, ls=(0, (5, 3)), zorder=3)

    for th in np.arange(0.0, 360.0, 15.0):
        q = np.array([np.cos(np.radians(th)), np.sin(np.radians(th))])
        pq = P @ q
        ax.plot([q[0], pq[0]], [q[1], pq[1]], color=MUTED, lw=0.6,
                ls=(0, (2, 3)), zorder=3)
        ax.plot(*q, "o", color=MUTED, ms=2.6, zorder=4)
        ax.plot(*pq, "o", color=BLUE, ms=4.0, zorder=5)

    arrow(ax, (0, 0), u, GREEN, lw=2.4, z=7)
    ax.plot(*u, "o", color=GREEN, ms=8, zorder=8)
    ax.text(u[0] + 0.12, u[1] - 0.22,
            r"$\mathbf{P}\mathbf{u}=\mathbf{u}$", color=GREEN, fontsize=12)
    ax.text(1.14, 0.86, "고윳값 1", color=GREEN, fontsize=11.5)

    arrow(ax, (0, 0), k, PURPLE, lw=2.0, ls=(0, (4, 3)), z=7)
    ax.plot(*k, "o", color=PURPLE, ms=7, zorder=8)
    ax.text(k[0] - 0.09, k[1] + 0.08, r"$\mathbf{k}$", color=PURPLE,
            fontsize=13, ha="right")
    ax.text(-0.68, 1.16, "고윳값 0", color=PURPLE, fontsize=11.5, ha="right")

    ax.plot(0, 0, "o", color=RED, ms=7, zorder=8)
    ax.annotate(r"$\mathbf{P}\mathbf{k}=\mathbf{0}$", xy=(-0.05, -0.06),
                xytext=(-0.24, -0.48), color=PURPLE, fontsize=12,
                ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.9,
                                shrinkA=3, shrinkB=5))

    ax.text(xlim[0] + 0.06, ylim[0] + 0.06,
            r"$\mathrm{tr}(\mathbf{P})=\mathrm{rank}(\mathbf{P})=1$",
            color=INK, fontsize=12, va="bottom")
    ax.set_title("원 위의 모든 점이 한 직선으로 눌린다", color=INK,
                 fontsize=14, pad=10)

    fig.tight_layout()
    save(fig, "idempotent_action.png")


# ==================================================================
# 4. gram_matrices.md — 내적표와 공선성
# ==================================================================
def gram_collinearity():
    L1, L2 = 1.4, 1.0
    angles = [90.0, 60.0, 15.0]

    fig = plt.figure(figsize=(11.6, 10.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.02, 0.86, 1.02],
                          hspace=0.62, wspace=0.32)

    def gram(theta_deg):
        th = np.radians(theta_deg)
        v1 = np.array([L1, 0.0])
        v2 = np.array([L2 * np.cos(th), L2 * np.sin(th)])
        X = np.column_stack([v1, v2])          # 2 x 2, 열이 v1, v2
        G = X.T @ X
        return v1, v2, G

    print("  각(도)  넓이      det(G)    lam_min   lam_max   kappa")
    for j, th in enumerate(angles):
        v1, v2, G = gram(th)
        ev = np.linalg.eigvalsh(G)
        det = np.linalg.det(G)
        area = abs(np.linalg.det(np.column_stack([v1, v2])))
        print(f"  {th:5.1f}  {area:8.4f}  {det:8.4f}  {ev[0]:8.4f}  "
              f"{ev[1]:8.4f}  {ev[1]/ev[0]:8.2f}")

        # --- 위: 두 열과 그들이 만드는 평행사변형 ---
        ax = fig.add_subplot(gs[0, j])
        ax.set_xlim(-0.30, 2.66); ax.set_ylim(-0.45, 1.30)
        ax.set_aspect("equal")
        bare(ax)
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.axvline(0, color=MUTED, lw=0.7)

        ax.add_patch(Polygon([(0, 0), v1, v1 + v2, v2], closed=True,
                             facecolor=GREEN_F, edgecolor=GREEN, lw=1.0,
                             alpha=0.65, zorder=2))
        ax.plot([v1[0], (v1 + v2)[0]], [v1[1], (v1 + v2)[1]],
                color=GREEN, lw=0.8, ls=(0, (3, 3)), zorder=3)
        ax.plot([v2[0], (v1 + v2)[0]], [v2[1], (v1 + v2)[1]],
                color=GREEN, lw=0.8, ls=(0, (3, 3)), zorder=3)

        arrow(ax, (0, 0), v1, BLUE, lw=2.4, z=5)
        arrow(ax, (0, 0), v2, ORANGE, lw=2.4, z=5)
        ax.text(v1[0] + 0.05, v1[1] - 0.16, r"$\mathbf{v}_{1}$", color=BLUE,
                fontsize=12.5)
        ax.text(v2[0] + 0.06, v2[1] + 0.06, r"$\mathbf{v}_{2}$", color=ORANGE,
                fontsize=12.5)

        arc_r = 0.34
        ta = np.linspace(0, np.radians(th), 60)
        ax.plot(arc_r * np.cos(ta), arc_r * np.sin(ta), color=INK, lw=1.0,
                zorder=6)
        tm = np.radians(th / 2.0)
        ax.text((arc_r + 0.14) * np.cos(tm), (arc_r + 0.13) * np.sin(tm),
                f"{int(th)}" + r"$^{\circ}$", color=INK, fontsize=11,
                ha="center", va="center")

        ax.set_title(f"두 열 사이의 각 {int(th)}" + r"$^{\circ}$",
                     color=INK, fontsize=13, pad=8)
        ax.text(-0.24, -0.40, f"넓이 = {area:.3f}", color=GREEN, fontsize=11,
                va="bottom")

        # --- 가운데: 내적표 ---
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(G, cmap="Blues", vmin=0.0, vmax=2.2)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels([r"$\mathbf{v}_{1}$", r"$\mathbf{v}_{2}$"],
                           fontsize=12, color=INK)
        ax.set_yticklabels([r"$\mathbf{v}_{1}$", r"$\mathbf{v}_{2}$"],
                           fontsize=12, color=INK)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_color(MUTED)
        for a in range(2):
            for b in range(2):
                val = G[a, b]
                ax.text(b, a, f"{val:.3f}", ha="center", va="center",
                        fontsize=12.5,
                        color="white" if val > 1.25 else INK)
        ax.set_title(r"$\mathbf{G}=\mathbf{X}^{T}\mathbf{X}$", color=INK,
                     fontsize=13, pad=8)
        ax.set_xlabel(f"det = {det:.3f}" + "\n" + r"$\lambda_{\min}$ = "
                      + f"{ev[0]:.3f},  " + r"$\kappa$ = "
                      + f"{ev[1]/ev[0]:.1f}",
                      color=INK, fontsize=11, labelpad=8)

    # --- 아래: 각이 줄면 행렬식과 최소고윳값이 함께 0 으로 ---
    ax = fig.add_subplot(gs[2, :])
    ths = np.linspace(1.0, 90.0, 400)
    dets, lmins = [], []
    for t in ths:
        _, _, G = gram(t)
        dets.append(np.linalg.det(G))
        lmins.append(np.linalg.eigvalsh(G)[0])
    dets = np.array(dets); lmins = np.array(lmins)

    ax.plot(ths, dets, color=GREEN, lw=2.4, label=r"$\det(\mathbf{G})$")
    ax.plot(ths, lmins, color=PURPLE, lw=2.4, ls=(0, (5, 3)),
            label=r"$\lambda_{\min}(\mathbf{G})$")

    for t in angles:
        _, _, G = gram(t)
        ax.plot([t, t], [0, max(np.linalg.det(G), np.linalg.eigvalsh(G)[0])],
                color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
        ax.plot(t, np.linalg.det(G), "o", color=GREEN, ms=7, zorder=4)
        ax.plot(t, np.linalg.eigvalsh(G)[0], "o", color=PURPLE, ms=7, zorder=4)

    ax.set_xlim(0, 93); ax.set_ylim(-0.05, 2.15)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_xlabel("두 열 사이의 각 (도)", color=INK, fontsize=11.5, labelpad=8)
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.legend(fontsize=12, frameon=False, loc="upper left")
    ax.set_title("열이 가까워지면 그람 행렬이 특이해진다", color=INK,
                 fontsize=13.5, pad=8)
    ax.annotate("공선성", xy=(9.5, 0.08), xytext=(25.0, 0.62),
                color=RED, fontsize=12, va="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))

    fig.suptitle(r"$\mathbf{X}^{T}\mathbf{X}$ 는 열들의 내적표다",
                 color=INK, fontsize=15.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    save(fig, "gram_collinearity.png")


if __name__ == "__main__":
    print("1. projection.md")
    oblique_vs_orthogonal()
    print("2. orthogonal_projection.md")
    least_squares_pythagoras()
    print("3. idempotent.md")
    idempotent_action()
    print("4. gram_matrices.md")
    gram_collinearity()
