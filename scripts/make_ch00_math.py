r"""0장 수학 예비지식 세 쪽의 그림을 생성한다.

세 쪽에 각각 한 장씩, 모두 세 장을 만든다. 정의를 눈으로 보여 주는 도식이며,
그림에 적힌 수치는 실행할 때 그 자리에서 계산해 출력한다(본문에 옮겨 적은
값과 일치해야 한다).

만드는 파일:

  ch00/math/img/function_types.png       단사·전사·전단사 — 화살표 그림 하나로 비교
  ch00/math/img/growth_and_asymptotic.png  증가 속도의 서열과 O, o, ~ 가 주장하는 것
  ch00/math/img/matvec_columns.png       행렬-벡터 곱은 열의 선형결합이다

실행:  python3 scripts/make_ch00_math.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.

주의:  한글은 수식 $...$ 바깥에만 쓴다. mathtext 에는 한글 글리프가 없다.
       로그축 눈금 라벨도 mathtext 를 거치므로 음의 지수는 쓰지 않는다.
"""

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon

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

OUT = "docs/ch00/math/img/"


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ===================================================================
# 그림 1. 단사 · 전사 · 전단사
# ===================================================================
def _blob(ax, cx, label, n_dots, color, face):
    """정의역/공역을 나타내는 타원과 그 안의 점들을 그린다. y 좌표를 돌려준다."""
    ax.add_patch(Ellipse((cx, 0.50), 0.22, 0.86, facecolor=face,
                         edgecolor=color, lw=1.4, zorder=1))
    ys = np.linspace(0.78, 0.22, n_dots)
    ax.scatter(np.full(n_dots, cx), ys, s=34, color=color, zorder=4)
    ax.text(cx, 0.965, label, ha="center", va="bottom", fontsize=13,
            color=color, zorder=5)
    return ys


def _arrow(ax, p, q, color, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=7, shrinkB=8, zorder=3))


def fig_function_types():
    cases = [
        # (제목, |A|, |B|, 대응, 개수 관계, 무엇이 모자라는가)
        ("단사이지만 전사는 아니다", 3, 4, [(0, 0), (1, 1), (2, 2)],
         r"개수: $|A| \leq |B|$",
         "화살표가 서로 다른 점으로 가지만\n공역의 한 점이 상 밖에 남는다"),
        ("전사이지만 단사는 아니다", 4, 3, [(0, 0), (1, 0), (2, 1), (3, 2)],
         r"개수: $|A| \geq |B|$",
         "공역을 다 덮지만\n두 점이 한 점으로 간다"),
        ("전단사", 3, 3, [(0, 0), (1, 1), (2, 2)],
         r"개수: $|A| = |B|$",
         "완전한 짝짓기이므로\n역함수 $f^{-1}$ 가 존재한다"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.8))

    for ax, (title, na, nb, pairs, count, note) in zip(axes, cases):
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.30, 1.14)
        ax.axis("off")
        ax.set_title(title, fontsize=13, color=INK, pad=14)

        ya = _blob(ax, 0.27, "$A$", na, BLUE, BLUE_F)
        yb = _blob(ax, 0.73, "$B$", nb, ORANGE, ORANGE_F)

        for i, y in enumerate(ya):
            ax.text(0.27 - 0.15, y, f"$a_{i + 1}$", ha="right", va="center",
                    fontsize=11, color=BLUE)
        for j, y in enumerate(yb):
            ax.text(0.73 + 0.15, y, f"$b_{j + 1}$", ha="left", va="center",
                    fontsize=11, color=ORANGE)

        # 정의에 어긋나는 부분을 붉게 강조한다
        hit = [j for _, j in pairs]
        doubled = {j for j in hit if hit.count(j) > 1}
        for i, j in pairs:
            bad = j in doubled
            _arrow(ax, (0.27, ya[i]), (0.73, yb[j]),
                   RED if bad else INK, lw=1.8 if bad else 1.3)
        for j, y in enumerate(yb):
            if j not in hit:
                ax.scatter([0.73], [y], s=190, facecolor="none",
                           edgecolor=RED, lw=1.8, zorder=5)

        ax.text(0.50, -0.02, count, ha="center", va="top", fontsize=12,
                color=INK)
        ax.text(0.50, -0.13, note, ha="center", va="top", fontsize=10.5,
                color=MUTED, linespacing=1.5)

    fig.suptitle("함수 $f: A \\to B$ — 화살표가 놓이는 방식이 세 성질을 가른다",
                 fontsize=14, color=INK, y=1.02)
    fig.tight_layout()
    save(fig, OUT + "function_types.png")


# ===================================================================
# 그림 2. 증가 속도의 서열과 O, o, ~
# ===================================================================
def fig_growth_and_asymptotic():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))

    # --- 왼쪽: 다섯 가지 증가 속도 ---
    n = np.arange(2, 51, dtype=float)
    curves = [
        (np.log(n), r"$\log n$", MUTED, "--"),
        (n, r"$n$", BLUE, "-"),
        (n * np.log(n), r"$n\log n$", GREEN, "-"),
        (n ** 2, r"$n^2$", ORANGE, "-"),
        (2.0 ** n, r"$2^n$", PURPLE, "-"),
    ]
    for y, lab, c, ls in curves:
        ax1.plot(n, y, color=c, lw=2.0, ls=ls)
        ax1.text(51.5, y[-1], lab, color=c, fontsize=12,
                 ha="left", va="center")

    ax1.set_yscale("log")
    ax1.set_xlim(2, 68)
    ax1.set_ylim(0.5, 5e15)
    ax1.set_yticks([1, 1e3, 1e6, 1e9, 1e12, 1e15])
    ax1.set_yticklabels(["$1$", "$10^{3}$", "$10^{6}$", "$10^{9}$",
                         "$10^{12}$", "$10^{15}$"])
    ax1.set_xticks([10, 20, 30, 40, 50])
    ax1.set_xlabel("$n$", fontsize=12, color=INK)
    ax1.set_title("증가 속도의 서열 (세로축은 로그)", fontsize=13, color=INK)
    ax1.text(0.04, 0.955,
             "곡선의 서열이 곧 점근 서열이다\n"
             r"아래 곡선은 위 곡선에 대해 $o(\cdot)$ 이다",
             transform=ax1.transAxes, ha="left", va="top", fontsize=10.5,
             color=INK, linespacing=1.5, zorder=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    # --- 오른쪽: 하나의 f 에 대해 세 표기가 주장하는 것 ---
    m = np.logspace(1, 9, 400)
    f = 3 * m + 2 * np.sqrt(m) + 5

    ax2.axhline(3.0, color=MUTED, lw=1.0, ls=":")
    ax2.axhline(1.0, color=MUTED, lw=1.0, ls=":")
    ax2.axhline(0.0, color=MUTED, lw=1.0, ls=":")

    ax2.plot(m, f / m, color=BLUE, lw=2.2)
    ax2.plot(m, f / (3 * m), color=ORANGE, lw=2.2)
    ax2.plot(m, f / (m * np.log(m)), color=GREEN, lw=2.2)

    ax2.text(1.3e9, f[-1] / m[-1], r"$f(n)/n$", color=BLUE, fontsize=11.5,
             ha="left", va="center")
    ax2.text(1.3e9, f[-1] / (3 * m[-1]), r"$f(n)/3n$", color=ORANGE,
             fontsize=11.5, ha="left", va="center")
    ax2.text(1.3e9, f[-1] / (m[-1] * np.log(m[-1])), r"$f(n)/(n\log n)$",
             color=GREEN, fontsize=11.5, ha="left", va="center")

    ax2.set_xscale("log")
    ax2.set_xlim(10, 3e11)
    ax2.set_ylim(-0.25, 4.7)
    ax2.set_xticks([1e1, 1e3, 1e5, 1e7, 1e9])
    ax2.set_xticklabels(["$10$", "$10^{3}$", "$10^{5}$", "$10^{7}$",
                         "$10^{9}$"])
    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_xlabel("$n$", fontsize=12, color=INK)
    ax2.set_ylabel("$f(n)/g(n)$", fontsize=12, color=INK)
    ax2.set_title(r"$f(n) = 3n + 2\sqrt{n} + 5$ 에 대한 세 가지 주장",
                  fontsize=13, color=INK)
    ax2.text(0.035, 0.955,
             r"$f = O(n)$   비가 유계다" "\n"
             r"$f \sim 3n$   비가 $1$ 로 간다" "\n"
             r"$f = o(n\log n)$   비가 $0$ 으로 간다",
             transform=ax2.transAxes, ha="left", va="top", fontsize=10.5,
             color=INK, linespacing=1.6, zorder=8,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))

    for ax in (ax1, ax2):
        for s in ax.spines.values():
            s.set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=10)

    fig.tight_layout()
    save(fig, OUT + "growth_and_asymptotic.png")

    # --- 본문에 옮겨 적을 값 ---
    print("  [그림 2] 증가 속도 (n = 50):")
    for y, lab, _, _ in curves:
        print(f"      {lab:>12s} = {y[-1]:.4g}")
    for nn in (10.0, 1e3, 1e6, 1e9):
        ff = 3 * nn + 2 * np.sqrt(nn) + 5
        print(f"  [그림 2] n = {nn:.0e}:  f/n = {ff / nn:.6f}   "
              f"f/3n = {ff / (3 * nn):.6f}   "
              f"f/(n log n) = {ff / (nn * np.log(nn)):.6f}")


# ===================================================================
# 그림 3. 행렬-벡터 곱은 열의 선형결합이다
# ===================================================================
def _vec(ax, tail, head, color, lw=2.4, ls="-", z=5):
    ax.add_patch(FancyArrowPatch(tail, head, arrowstyle="-|>",
                                 mutation_scale=16, lw=lw, color=color,
                                 linestyle=ls, shrinkA=0, shrinkB=0, zorder=z))


def _manual_ticks(ax, xs, ys, dx=0.32, dy=0.30):
    """스파인을 숨긴 대신 축선 위에 직접 눈금 라벨을 찍는다."""
    for v in xs:
        ax.plot([v, v], [-0.09, 0.09], color=MUTED, lw=0.9, zorder=3)
        ax.text(v, -dy, f"{v:g}", ha="center", va="top", fontsize=9,
                color=MUTED, zorder=3)
    for v in ys:
        ax.plot([-0.09, 0.09], [v, v], color=MUTED, lw=0.9, zorder=3)
        ax.text(-dx, v, f"{v:g}", ha="right", va="center", fontsize=9,
                color=MUTED, zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_matvec_columns():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.8))
    WHITE = dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2)

    # --- 왼쪽: 열이 독립이면 도달 범위가 평면 전체 ---
    a1 = np.array([2.0, 1.0])
    a2 = np.array([1.0, 3.0])
    x = np.array([1.5, 1.0])
    Ax = x[0] * a1 + x[1] * a2

    for t in np.arange(-2, 6):         # 두 열이 만드는 격자
        p0, p1 = t * a1 - 2 * a2, t * a1 + 4 * a2
        ax1.plot([p0[0], p1[0]], [p0[1], p1[1]], color=MUTED, lw=0.6,
                 alpha=0.40, zorder=1)
        q0, q1 = t * a2 - 2 * a1, t * a2 + 4 * a1
        ax1.plot([q0[0], q1[0]], [q0[1], q1[1]], color=MUTED, lw=0.6,
                 alpha=0.40, zorder=1)

    # 꼬리-머리로 이어 붙이기: 1.5 a1 그다음 1 a2
    ax1.add_patch(Polygon([[0, 0], (x[0] * a1), Ax, (x[1] * a2)],
                          closed=True, facecolor=PURPLE, alpha=0.07,
                          edgecolor="none", zorder=1))
    _vec(ax1, (0, 0), x[0] * a1, BLUE, lw=3.0, z=4)
    _vec(ax1, x[0] * a1, Ax, ORANGE, lw=3.0, z=4)
    _vec(ax1, (0, 0), a1, BLUE, lw=2.0, z=6)
    _vec(ax1, (0, 0), a2, ORANGE, lw=2.0, z=6)
    _vec(ax1, (0, 0), Ax, PURPLE, lw=2.6, z=7)

    ax1.text(a1[0] + 0.10, a1[1] - 0.26, r"$\mathbf{a}_1$", color=BLUE,
             fontsize=13, ha="center", va="top", zorder=9, bbox=WHITE)
    ax1.text(a2[0] - 0.30, a2[1] + 0.02, r"$\mathbf{a}_2$", color=ORANGE,
             fontsize=13, ha="right", va="center", zorder=9, bbox=WHITE)
    ax1.text(3.18, 1.18, r"$1.5\,\mathbf{a}_1$", color=BLUE, fontsize=12,
             ha="left", va="top", zorder=9, bbox=WHITE)
    ax1.text(3.66, 3.05, r"$1\,\mathbf{a}_2$", color=ORANGE, fontsize=12,
             ha="left", va="center", zorder=9, bbox=WHITE)
    ax1.text(Ax[0] - 0.12, Ax[1] + 0.25, r"$\mathbf{A}\mathbf{x} = (4,\,4.5)$",
             color=PURPLE, fontsize=12, ha="right", va="bottom", zorder=9,
             bbox=WHITE)

    ax1.set_xlim(-1.8, 5.9)
    ax1.set_ylim(-1.6, 6.1)
    ax1.set_title("열이 독립이면 도달 범위는 평면 전체", fontsize=13, color=INK)
    ax1.text(0.025, 0.975,
             r"$\mathbf{A} = [\,\mathbf{a}_1\;\mathbf{a}_2\,]$, "
             r"$\mathbf{x} = (1.5,\,1)$" "\n"
             r"$\mathbf{A}\mathbf{x} = 1.5\,\mathbf{a}_1 + 1\,\mathbf{a}_2$"
             "\n" r"$\det \mathbf{A} = 5 \neq 0$",
             transform=ax1.transAxes, ha="left", va="top", fontsize=10.5,
             color=INK, linespacing=1.6, zorder=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))
    _manual_ticks(ax1, [-1, 1, 2, 3, 4, 5], [-1, 1, 2, 3, 4])

    # --- 오른쪽: 열이 종속이면 도달 범위는 직선 하나 ---
    c1 = np.array([2.0, 1.0])
    c2 = 2.0 * c1
    y = np.array([5.5, 1.5])
    yhat = (y @ c1) / (c1 @ c1) * c1
    resid = y - yhat

    line = np.outer(np.array([-0.8, 3.4]), c1)
    ax2.plot(line[:, 0], line[:, 1], color=GREEN, lw=2.0, zorder=2)
    ax2.text(6.55, 3.45, r"$\mathrm{Col}(\mathbf{A})$", color=GREEN,
             fontsize=12, ha="center", va="bottom", zorder=9, bbox=WHITE)

    _vec(ax2, (0, 0), c2, ORANGE, lw=3.0, z=4)
    _vec(ax2, (0, 0), c1, BLUE, lw=2.2, z=6)
    ax2.text(2.0, 0.62, r"$\mathbf{a}_1$", color=BLUE, fontsize=13,
             ha="center", va="top", zorder=9, bbox=WHITE)
    ax2.text(4.0, 1.52, r"$\mathbf{a}_2 = 2\,\mathbf{a}_1$", color=ORANGE,
             fontsize=12, ha="center", va="top", zorder=9, bbox=WHITE)

    ax2.plot([y[0], yhat[0]], [y[1], yhat[1]], color=RED, lw=1.6, ls="--",
             zorder=5)
    ax2.plot([y[0]], [y[1]], "o", ms=9, color=RED, zorder=7)
    ax2.text(y[0] + 0.18, y[1] - 0.04, r"$\mathbf{y}$", color=RED,
             fontsize=13, ha="left", va="center", zorder=9)
    ax2.plot([yhat[0]], [yhat[1]], "o", ms=8, color=PURPLE, zorder=7)
    ax2.text(5.0, 2.95, r"$\hat{\mathbf{y}} = (5,\,2.5)$", color=PURPLE,
             fontsize=12, ha="center", va="bottom", zorder=9, bbox=WHITE)
    ax2.text(5.45, 2.05, "잔차", color=RED, fontsize=11, ha="left",
             va="center", zorder=9, bbox=WHITE)

    # 직각 표시
    u = c1 / np.linalg.norm(c1)
    v = resid / np.linalg.norm(resid)
    s = 0.32
    ax2.add_patch(Polygon([yhat, yhat + s * u, yhat + s * u + s * v,
                           yhat + s * v], closed=True, facecolor="none",
                          edgecolor=INK, lw=1.1, zorder=6))

    ax2.set_xlim(-2.0, 7.6)
    ax2.set_ylim(-1.9, 4.7)
    ax2.set_title("열이 종속이면 도달 범위는 직선 하나", fontsize=13, color=INK)
    ax2.text(0.975, 0.03,
             r"모든 $\mathbf{A}\mathbf{x}$ 가 한 직선 위에 있다" "\n"
             r"$\mathbf{y}$ 는 그 직선 밖에 있으므로" "\n"
             r"가장 가까운 점 $\hat{\mathbf{y}}$ 로 사영한다",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=10.5,
             color=INK, linespacing=1.6, zorder=10,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#F4F6F8",
                       edgecolor=MUTED, lw=0.8))
    _manual_ticks(ax2, [1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4])

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.axhline(0, color=MUTED, lw=0.9, zorder=2)
        ax.axvline(0, color=MUTED, lw=0.9, zorder=2)
        for s_ in ax.spines.values():
            s_.set_visible(False)

    fig.tight_layout()
    save(fig, OUT + "matvec_columns.png")

    # --- 본문에 옮겨 적을 값 ---
    A = np.column_stack([a1, a2])
    print(f"  [그림 3] A x = {A @ x}   det A = {np.linalg.det(A):.4f}")
    print(f"  [그림 3] y = {y}  yhat = {yhat}  잔차 = {resid}")
    print(f"  [그림 3] 잔차 노름 = {np.linalg.norm(resid):.6f}   "
          f"잔차와 a1 의 내적 = {resid @ c1:.2e}")
    B = np.column_stack([c1, c2])
    print(f"  [그림 3] 종속인 경우 rank = {np.linalg.matrix_rank(B)}   "
          f"det = {np.linalg.det(B):.2f}   "
          f"y 까지의 거리 = {np.linalg.norm(resid):.6f}")


if __name__ == "__main__":
    fig_function_types()
    fig_growth_and_asymptotic()
    fig_matvec_columns()
