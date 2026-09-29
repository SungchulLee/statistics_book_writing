r"""12장 상관 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch12/correlation/img/pearson_pitfalls.png      r 하나로는 모자란 세 가지 경우
  ch12/correlation/img/rank_invariance.png       순위는 단조변환과 이상점에 흔들리지 않는다
  ch12/correlation/img/tau_pairs_and_scale.png   쌍 세기와 타우의 눈금
  ch12/correlation/img/partial_map.png           통제하면 줄어들기만 하는 것이 아니다
  ch12/correlation/img/binary_ceilings.png       이진 변수의 상관에는 천장이 있다

실행:  python3 scripts/make_ch12_correlation_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
from itertools import combinations

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

OUT = "docs/ch12/correlation/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


def headroom(ax, frac=0.30):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


def bivariate(gen, rho, n):
    z1 = gen.normal(0, 1, n)
    z2 = gen.normal(0, 1, n)
    return z1, rho * z1 + np.sqrt(1 - rho ** 2) * z2


# === 그림 1. Pearson 의 함정 ===
def fig_pearson_pitfalls():
    rng = np.random.default_rng(17)

    # (a) 완전한 비선형 관계
    xa = np.linspace(-1, 1, 200)
    ya = xa ** 2
    ra = stats.pearsonr(xa, ya)[0]

    # (b) 이상점 하나가 +1 을 음수로
    xb0 = np.array([1.0, 2, 3, 4, 5])
    yb0 = np.array([1.0, 2, 3, 4, 5])
    xb = np.append(xb0, 10.0)
    yb = np.append(yb0, -5.0)
    rb0 = stats.pearsonr(xb0, yb0)[0]
    rb = stats.pearsonr(xb, yb)[0]

    # (c) 이상점 하나가 없던 상관을 만든다
    xc0 = rng.normal(0, 1, 40)
    yc0 = rng.normal(0, 1, 40)
    rc0 = stats.pearsonr(xc0, yc0)[0]
    xc = np.append(xc0, 9.0)
    yc = np.append(yc0, 9.0)
    rc = stats.pearsonr(xc, yc)[0]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    ax = axes[0]
    clean(ax)
    ax.scatter(xa, ya, s=16, color=BLUE, alpha=0.8, edgecolor="none")
    ax.axhline(ya.mean(), color=ORANGE, lw=2.0)
    headroom(ax, 0.30)
    ax.text(0.03, 0.97, f"r = {ra:+.3f}", transform=ax.transAxes,
            fontsize=11.5, color=RED, va="top")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title(r"$Y = X^2$" + " — 관계는 완벽한데", fontsize=12.5,
                 color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.scatter(xb0, yb0, s=70, color=BLUE, alpha=0.9, edgecolor="none",
               label="원래 다섯 점")
    ax.scatter([10], [-5], s=110, color=RED, marker="X", zorder=3,
               label="추가한 점")
    t = np.linspace(0.5, 10.5, 10)
    s, b = np.polyfit(xb0, yb0, 1)
    ax.plot(t, b + s * t, color=BLUE, lw=1.8, ls="--")
    s, b = np.polyfit(xb, yb, 1)
    ax.plot(t, b + s * t, color=RED, lw=2.2)
    headroom(ax, 0.34)
    ax.text(0.03, 0.97, f"다섯 점만:  r = {rb0:+.2f}\n"
                        f"점 하나 추가: r = {rb:+.2f}",
            transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
            linespacing=1.5)
    ax.legend(fontsize=9.5, frameon=False, loc="lower left")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("점 하나가 부호를 뒤집는다", fontsize=12.5, color=INK, pad=8)

    ax = axes[2]
    clean(ax)
    ax.scatter(xc0, yc0, s=26, color=BLUE, alpha=0.75, edgecolor="none",
               label="원래 40개 점")
    ax.scatter([9], [9], s=110, color=RED, marker="X", zorder=3,
               label="추가한 점")
    t = np.linspace(-3, 9.6, 10)
    s, b = np.polyfit(xc, yc, 1)
    ax.plot(t, b + s * t, color=RED, lw=2.2)
    headroom(ax, 0.34)
    ax.text(0.03, 0.97, f"40개만:    r = {rc0:+.2f}\n"
                        f"점 하나 추가: r = {rc:+.2f}",
            transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
            linespacing=1.5)
    ax.legend(fontsize=9.5, frameon=False, loc="lower right")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("점 하나가 없던 상관을 만든다", fontsize=12.5, color=INK,
                 pad=8)

    save(fig, "pearson_pitfalls.png")
    print(f"  (a) r={ra:.4f}  (b) {rb0:.3f} -> {rb:.3f}  "
          f"(c) {rc0:.3f} -> {rc:.3f}")


# === 그림 2. 순위의 불변성 ===
def fig_rank_invariance():
    rng = np.random.default_rng(101)
    n = 40
    x, y = bivariate(rng, 0.85, n)

    sets = [("원자료", x, y),
            ("Y 에 지수변환", x, np.exp(2.4 * y)),
            ("이상점 하나 추가",
             np.append(x, 3.4), np.append(y, -4.2))]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for ax, (name, xv, yv) in zip(axes, sets):
        clean(ax)
        ax.scatter(xv[:n], yv[:n], s=26, color=BLUE, alpha=0.8,
                   edgecolor="none")
        if len(xv) > n:
            ax.scatter(xv[n:], yv[n:], s=110, color=RED, marker="X",
                       zorder=3)
        r = stats.pearsonr(xv, yv)[0]
        rs = stats.spearmanr(xv, yv)[0]
        s, b = np.polyfit(xv, yv, 1)
        t = np.linspace(xv.min(), xv.max(), 10)
        ax.plot(t, b + s * t, color=ORANGE, lw=2.0)
        headroom(ax, 0.34)
        ax.text(0.03, 0.97,
                f"Pearson  r = {r:+.3f}\nSpearman r = {rs:+.3f}",
                transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
                linespacing=1.5)
        ax.set_xlabel("X", fontsize=10.5, color=INK)
        ax.set_ylabel("Y", fontsize=10.5, color=INK)
        ax.set_title(name, fontsize=12.5, color=INK, pad=8)
        print(f"  {name}: r={r:.4f}, r_s={rs:.4f}")

    save(fig, "rank_invariance.png")


# === 그림 3. 쌍 세기와 타우의 눈금 ===
def fig_tau():
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([3, 5, 4, 2, 1])
    C = D = 0
    segs = []
    for i, j in combinations(range(5), 2):
        sgn = np.sign(x[j] - x[i]) * np.sign(y[j] - y[i])
        if sgn > 0:
            C += 1
        else:
            D += 1
        segs.append(((x[i], y[i]), (x[j], y[j]), sgn > 0))
    tau = (C - D) / 10

    rng = np.random.default_rng(55)
    m = 2000
    taus = np.empty(m)
    rss = np.empty(m)
    for i in range(m):
        rho = rng.uniform(-0.98, 0.98)
        a, b = bivariate(rng, rho, 50)
        taus[i] = stats.kendalltau(a, b)[0]
        rss[i] = stats.spearmanr(a, b)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))

    ax = axes[0]
    clean(ax)
    for (p, q, conc) in segs:
        ax.plot([p[0], q[0]], [p[1], q[1]],
                color=GREEN if conc else RED, lw=2.0,
                alpha=0.75 if conc else 0.45, zorder=1)
    ax.scatter(x, y, s=120, color=INK, zorder=3)
    for xi, yi in zip(x, y):
        ax.text(xi, yi, "", ha="center")
    headroom(ax, 0.34)
    ax.text(0.03, 0.97,
            f"일치쌍 C = {C}   (초록)\n불일치쌍 D = {D}  (빨강)\n"
            + r"$\tau$" + f" = ({C} - {D}) / 10 = {tau:+.1f}",
            transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
            linespacing=1.55)
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_title("타우는 선분 열 개의 기울기 부호를 센 것이다",
                 fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.scatter(rss, taus, s=6, color=BLUE, alpha=0.3, edgecolor="none")
    g = np.linspace(-1, 1, 300)
    ax.plot(g, g, color=MUTED, lw=1.6, ls="--", label=r"$\tau = r_s$")
    ax.plot(g, 2 / np.pi * np.arcsin(g), color=RED, lw=2.2,
            label=r"$\tau = (2/\pi)\arcsin(r_s)$")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_xlabel("Spearman " + r"$r_s$", fontsize=10.5, color=INK)
    ax.set_ylabel("Kendall " + r"$\tau$", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper left")
    ax.set_title("같은 자료에서 타우가 늘 0 에 더 가깝다", fontsize=12.5,
                 color=INK, pad=8)

    ratio = float(np.mean(np.abs(taus) / np.maximum(np.abs(rss), 1e-9)))
    save(fig, "tau_pairs_and_scale.png")
    print(f"  C={C}, D={D}, tau={tau:.2f}")
    print(f"  |tau|/|r_s| 평균 = {ratio:.4f}")
    for v in (0.3, 0.5, 0.7, 0.9):
        print(f"  r_s={v}: 2/pi*arcsin = {2 / np.pi * np.arcsin(v):.4f}")


# === 그림 4. 부분상관의 지도 ===
def partial_from(r_xy, r_xz, r_yz):
    return (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))


def fig_partial_map():
    R_XY = 0.40
    g = np.linspace(-0.97, 0.97, 400)
    A, B = np.meshgrid(g, g)          # A = r_XZ, B = r_YZ
    det = 1 + 2 * R_XY * A * B - R_XY ** 2 - A ** 2 - B ** 2
    P = partial_from(R_XY, A, B)
    P = np.where(det > 0, P, np.nan)

    rng = np.random.default_rng(7)
    n = 300
    Z = rng.normal(0, 1, n)
    X = 0.7 * Z + rng.normal(0, np.sqrt(1 - 0.49), n)
    Y = -0.7 * Z + rng.normal(0, np.sqrt(1 - 0.49), n)
    Y = Y + 0.62 * X                                  # 억제 구조
    r_xy = stats.pearsonr(X, Y)[0]
    eX = X - np.polyval(np.polyfit(Z, X, 1), Z)
    eY = Y - np.polyval(np.polyfit(Z, Y, 1), Z)
    r_part = stats.pearsonr(eX, eY)[0]

    fig = plt.figure(figsize=(13.4, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    clean(ax)
    im = ax.pcolormesh(A, B, P, cmap="RdBu_r", vmin=-1, vmax=1,
                       shading="auto")
    ax.contour(A, B, P, levels=[0.0], colors=[INK], linewidths=1.8)
    ax.contour(A, B, P, levels=[R_XY], colors=[GREEN], linewidths=1.8,
               linestyles="--")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("부분상관", fontsize=10, color=INK, labelpad=10)
    cb.ax.tick_params(labelsize=9, colors=INK)
    pts = [(0.80, 0.50, "사라진다", (-0.92, 0.30)),
           (0.80, 0.80, "뒤집힌다", (-0.55, 0.90)),
           (0.50, -0.40, "커진다", (-0.92, -0.80))]
    for a, b, lab, tp in pts:
        ax.scatter([a], [b], s=60, color="white", edgecolor=INK, lw=1.8,
                   zorder=4)
        ax.annotate(f"{lab} {partial_from(R_XY, a, b):+.2f}", xy=(a, b),
                    xytext=tp, fontsize=10, color=INK, zorder=5,
                    bbox=dict(facecolor="white", alpha=0.88,
                              edgecolor="none", pad=1.6),
                    arrowprops=dict(arrowstyle="-", color=INK, lw=1.0))
    handles = [plt.Line2D([], [], color=INK, lw=1.8, label="부분상관 = 0"),
               plt.Line2D([], [], color=GREEN, lw=1.8, ls="--",
                          label="부분상관 = " + r"$r_{XY}$")]
    ax.legend(handles=handles, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="none")
    ax.set_xlabel(r"$r_{XZ}$", fontsize=11, color=INK)
    ax.set_ylabel(r"$r_{YZ}$", fontsize=11, color=INK)
    ax.set_title(r"$r_{XY} = 0.40$" + " 을 고정했을 때", fontsize=12.5,
                 color=INK, pad=8)

    ax = fig.add_subplot(gs[0, 1])
    clean(ax)
    ax.scatter(X, Y, s=16, color=MUTED, alpha=0.75, edgecolor="none")
    s, b = np.polyfit(X, Y, 1)
    t = np.linspace(X.min(), X.max(), 10)
    ax.plot(t, b + s * t, color=RED, lw=2.2)
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, f"r(X, Y) = {r_xy:+.3f}", transform=ax.transAxes,
            fontsize=11, color=RED, va="top")
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("통제하기 전", fontsize=12.5, color=INK, pad=8)

    ax = fig.add_subplot(gs[0, 2])
    clean(ax)
    ax.scatter(eX, eY, s=16, color=GREEN, alpha=0.75, edgecolor="none")
    s, b = np.polyfit(eX, eY, 1)
    t = np.linspace(eX.min(), eX.max(), 10)
    ax.plot(t, b + s * t, color=GREEN, lw=2.2)
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, f"r(X, Y | Z) = {r_part:+.3f}",
            transform=ax.transAxes, fontsize=11, color=GREEN, va="top")
    ax.set_xlabel("X 의 잔차", fontsize=10.5, color=INK)
    ax.set_ylabel("Y 의 잔차", fontsize=10.5, color=INK)
    ax.set_title("Z 를 통제한 뒤", fontsize=12.5, color=INK, pad=8)

    save(fig, "partial_map.png")
    print(f"  r_xy={r_xy:.4f} -> 부분상관 {r_part:.4f}")
    for a, b in [(0.80, 0.50), (0.80, 0.80), (0.50, -0.40)]:
        print(f"  r_XZ={a}, r_YZ={b}: 부분 {partial_from(R_XY, a, b):+.4f}")
    print(f"  아이스크림 예: {partial_from(0.85, 0.92, 0.88):+.4f}")


# === 그림 5. 이진 변수의 천장 ===
def fig_binary_ceilings():
    ps = np.linspace(0.01, 0.99, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.5))

    ax = axes[0]
    clean(ax)
    for d, col in [(0.5, GREEN), (1.0, BLUE), (2.0, PURPLE)]:
        num = d * np.sqrt(ps * (1 - ps))
        rpb = num / np.sqrt(num ** 2 + 1)
        ax.plot(ps, rpb, color=col, lw=2.3, label=f"d = {d}")
        k = np.argmax(rpb)
        ax.scatter([ps[k]], [rpb[k]], s=40, color=col, zorder=3)
        ax.text(0.5, rpb[k] + 0.028, f"{rpb[k]:.3f}", fontsize=10,
                color=col, ha="center")
    ax.axvline(0.5, color=MUTED, lw=1.0, ls=":")
    ax.axvline(0.1, color=MUTED, lw=1.0, ls=":")
    num1 = 1.0 * np.sqrt(0.1 * 0.9)
    r10 = num1 / np.sqrt(num1 ** 2 + 1)
    ax.scatter([0.1], [r10], s=40, color=BLUE, zorder=3)
    ax.annotate(f"10 대 90 으로 나뉘면\nd = 1 이어도 {r10:.3f}",
                xy=(0.1, r10), xytext=(0.27, 0.045), fontsize=10,
                color=BLUE,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("집단 1 의 비율 " + r"$n_1 / n$", fontsize=10.5, color=INK)
    ax.set_ylabel("점이연 상관 " + r"$r_{pb}$", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper right",
              title="표준화 평균차", title_fontsize=9.5)
    ax.set_title("평균차가 같아도 집단이 치우치면 작아진다", fontsize=12.5,
                 color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    q = np.linspace(0.02, 0.98, 300)
    P, Q = np.meshgrid(q, q)
    lo = np.minimum(P, Q)
    hi = np.maximum(P, Q)
    phimax = np.sqrt(lo * (1 - hi) / (hi * (1 - lo)))
    im = ax.pcolormesh(P, Q, phimax, cmap="viridis", vmin=0, vmax=1,
                       shading="auto")
    cs = ax.contour(P, Q, phimax, levels=[0.3, 0.5, 0.7, 0.9],
                    colors="white", linewidths=1.2)
    ax.clabel(cs, fontsize=9, fmt="%.1f")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("도달 가능한 최대 " + r"$|\phi|$", fontsize=10, color=INK,
                 labelpad=10)
    cb.ax.tick_params(labelsize=9, colors=INK)
    ax.scatter([0.45], [0.5], s=70, color="white", edgecolor=INK, lw=1.6,
               zorder=4)
    ax.annotate("보기 2\n최대 0.905", xy=(0.45, 0.5), xytext=(0.54, 0.26),
                fontsize=10, color="white",
                bbox=dict(facecolor=INK, alpha=0.5, edgecolor="none",
                          pad=2.0))
    ax.scatter([0.05], [0.5], s=70, color="white", edgecolor=RED, lw=1.6,
               zorder=4)
    ax.annotate("드문 사건\n최대 0.229", xy=(0.05, 0.5), xytext=(0.10, 0.70),
                fontsize=10, color="white",
                bbox=dict(facecolor=INK, alpha=0.5, edgecolor="none",
                          pad=2.0))
    ax.set_xlabel(r"$P(X = 1)$", fontsize=11, color=INK)
    ax.set_ylabel(r"$P(Y = 1)$", fontsize=11, color=INK)
    ax.set_title("주변분포가 다르면 파이는 1 에 닿지 못한다", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "binary_ceilings.png")
    for d in (0.5, 1.0, 2.0):
        num = d * 0.5
        print(f"  d={d}: 최대 r_pb = {num / np.sqrt(num ** 2 + 1):.4f}")
    for p in (0.5, 0.2, 0.1, 0.05):
        num = 1.0 * np.sqrt(p * (1 - p))
        print(f"  d=1, p={p}: r_pb = {num / np.sqrt(num ** 2 + 1):.4f}")
    for p, qq in [(0.45, 0.5), (0.05, 0.5), (0.1, 0.9)]:
        a, b = min(p, qq), max(p, qq)
        print(f"  p={p}, q={qq}: phi_max = "
              f"{np.sqrt(a * (1 - b) / (b * (1 - a))):.4f}")


if __name__ == "__main__":
    fig_pearson_pitfalls()
    fig_rank_invariance()
    fig_tau()
    fig_partial_map()
    fig_binary_ceilings()
