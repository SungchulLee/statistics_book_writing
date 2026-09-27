r"""1장 고전적 자료 수집 두 쪽의 그림을 생성한다.

  ch01/classical/img/frame_coverage.png   목표 모집단 · 표집틀 · 표본의 세 겹과 커버리지 오차
  ch01/classical/img/four_designs.png     같은 모집단 위에 그린 네 가지 확률표집 설계

앞 그림은 226명의 가상 모집단을 실제로 만들어 표집틀 안과 목표 모집단 전체의
찬성률을 계산하고, 그 차이(커버리지 편향)를 오른쪽 막대에 적는다.
뒤 그림은 240개 단위의 고정된 모집단에서 네 설계의 표준오차를 재어 제목에 적는다.
단순무작위 · 층화 · 집락은 모의실험(반복 40,000회)이고, 계통표집은 가능한
시작점 10개를 모두 계산하므로 정확한 값이다.

실행:  python3 scripts/make_ch01_sampling.py   (저장소 최상위)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       그림은 PNG로 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

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

OUT = "docs/ch01/classical/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ==================================================================
# 1. 목표 모집단 · 표집틀 · 표본의 세 겹
#    직사각형 = 목표 모집단, 타원 = 표집틀.
#    겹치는 부분이 연구 모집단이고, 어긋난 두 부분이 커버리지 오차다.
#    각 사람에게 찬성/반대를 붙여 두 집단의 찬성률을 실제로 계산한다.
# ==================================================================

# 도형 좌표
RX0, RX1, RY0, RY1 = 0.5, 7.0, 1.0, 6.0        # 목표 모집단 사각형
EX, EY, EA, EB = 5.7, 3.5, 4.4, 2.70           # 표집틀 타원 (중심, 반지름)


def in_frame(x, y):
    """표집틀 타원 안인가."""
    return ((x - EX) / EA) ** 2 + ((y - EY) / EB) ** 2 <= 1.0


def build_population(seed=7, n_target=200, n_over=26, n_sample=20):
    """목표 모집단과 과대포괄 단위를 만들고 찬성 여부를 붙인다.

    찬성 확률은 x 좌표가 작을수록(그림에서 왼쪽일수록) 높다.
    표집틀에서 빠지는 사람이 바로 왼쪽에 몰려 있으므로,
    이 구조가 곧 '빠진 사람들이 체계적으로 다르다'는 상황이 된다.
    """
    rng = np.random.default_rng(seed)

    # 목표 모집단: 사각형 안에 고르게. 테두리에 붙지 않게 여백을 둔다.
    tx = rng.uniform(RX0 + 0.28, RX1 - 0.28, n_target)
    ty = rng.uniform(RY0 + 0.28, RY1 - 0.28, n_target)

    # 과대포괄: 타원 안이면서 사각형 오른쪽 바깥
    ox, oy = [], []
    while len(ox) < n_over:
        x = rng.uniform(RX1 + 0.3, EX + EA - 0.2)
        y = rng.uniform(EY - EB + 0.2, EY + EB - 0.2)
        if in_frame(x, y):
            ox.append(x)
            oy.append(y)
    ox, oy = np.array(ox), np.array(oy)

    covered = in_frame(tx, ty)                 # 목표 모집단 중 표집틀에 잡힌 사람
    p_yes = 0.26 + 0.075 * (RX1 - tx)          # 왼쪽일수록 찬성이 많다
    yes = rng.random(n_target) < p_yes

    idx_cov = np.flatnonzero(covered)
    sample = rng.choice(idx_cov, n_sample, replace=False)

    return dict(tx=tx, ty=ty, ox=ox, oy=oy, covered=covered, yes=yes,
                sample=sample, n_target=n_target, n_over=n_over)


def fig_frame_coverage(path):
    P = build_population()
    tx, ty, covered, yes = P["tx"], P["ty"], P["covered"], P["yes"]
    n_cov = int(covered.sum())
    n_out = int((~covered).sum())

    rate_target = yes.mean()
    rate_frame = yes[covered].mean()
    rate_out = yes[~covered].mean()
    bias = rate_frame - rate_target

    print(f"  목표 모집단 {P['n_target']}명, 찬성률 {rate_target:.4f}")
    print(f"  표집틀에 포함(연구 모집단) {n_cov}명, 찬성률 {rate_frame:.4f}")
    print(f"  표집틀에서 빠진 사람 {n_out}명, 찬성률 {rate_out:.4f}")
    print(f"  과대포괄 {P['n_over']}명")
    print(f"  커버리지 편향 = {bias:+.4f}")

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(12.6, 6.4), gridspec_kw=dict(width_ratios=[2.55, 1.0], wspace=0.30))

    # ----- 왼쪽: 세 겹 모식도 -----
    ax.add_patch(Ellipse((EX, EY), 2 * EA, 2 * EB, facecolor=BLUE_F,
                         edgecolor=BLUE, lw=2.0, alpha=0.95, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (RX0, RY0), RX1 - RX0, RY1 - RY0,
        boxstyle="round,pad=0,rounding_size=0.35",
        facecolor="none", edgecolor=ORANGE, lw=2.4, zorder=3))
    # 목표 모집단 내부를 아주 옅게 깐다 (타원 아래에 둔다)
    ax.add_patch(FancyBboxPatch(
        (RX0, RY0), RX1 - RX0, RY1 - RY0,
        boxstyle="round,pad=0,rounding_size=0.35",
        facecolor=ORANGE_F, edgecolor="none", alpha=0.55, zorder=0))

    ax.scatter(tx[~covered], ty[~covered], s=26, facecolor=ORANGE,
               edgecolor="white", lw=0.5, zorder=4)
    ax.scatter(tx[covered], ty[covered], s=22, facecolor=MUTED,
               edgecolor="white", lw=0.5, zorder=4)
    ax.scatter(P["ox"], P["oy"], s=26, marker="x", color=PURPLE, lw=1.5, zorder=4)
    s = P["sample"]
    ax.scatter(tx[s], ty[s], s=78, facecolor=BLUE, edgecolor="white",
               lw=1.3, zorder=5)

    ax.text(RX0 + 0.05, RY1 + 0.38,
            f"목표 모집단\n알고 싶은 대상 {P['n_target']}명",
            color=ORANGE, fontsize=12.5, fontweight="bold",
            ha="left", va="bottom", linespacing=1.45)
    ax.text(EX + EA + 0.2, EY + EB + 0.45,
            f"표집틀\n실제로 손이 닿는 목록 {n_cov + P['n_over']}명",
            color=BLUE, fontsize=12.5, fontweight="bold",
            ha="right", va="bottom", linespacing=1.45)
    ax.text(5.35, 5.02, f"연구 모집단 {n_cov}명", color=INK, fontsize=10.5,
            ha="center", va="center", zorder=6,
            bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                      edgecolor=MUTED, lw=0.8, alpha=0.92))

    ax.annotate(f"과소포괄 {n_out}명\n목표에는 있으나 목록에 없다",
                xy=(1.05, 1.75), xytext=(0.0, -0.60),
                color=ORANGE, fontsize=10.5, ha="left", va="top",
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4,
                                shrinkA=2, shrinkB=4,
                                connectionstyle="arc3,rad=0.20"))
    ax.annotate(f"과대포괄 {P['n_over']}명\n목록에만 있는 자격 없는 단위",
                xy=(8.8, 2.4), xytext=(11.1, -0.60),
                color=PURPLE, fontsize=10.5, ha="right", va="top",
                arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.4,
                                shrinkA=2, shrinkB=4,
                                connectionstyle="arc3,rad=-0.22"))
    low = s[np.argmin(ty[s])]
    ax.annotate("표본 20명 — 연구 모집단에서만 뽑힌다",
                xy=(tx[low], ty[low]), xytext=(5.5, -2.15),
                color=BLUE, fontsize=10.5, ha="center", va="top",
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.4,
                                shrinkA=2, shrinkB=6,
                                connectionstyle="arc3,rad=0.12"))

    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-3.1, 8.1)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- 오른쪽: 두 집단의 찬성률 -----
    ypos = [1.0, 0.0]
    vals = [rate_target, rate_frame]
    cols = [ORANGE, BLUE]
    fills = [ORANGE_F, BLUE_F]
    bx.axvspan(vals[1], vals[0], color=RED, alpha=0.13, zorder=0)
    for y, v, c, f in zip(ypos, vals, cols, fills):
        bx.barh(y, v, height=0.46, facecolor=f, edgecolor=c, lw=2.0, zorder=2)
        bx.text(v - 0.018, y, f"{v:.3f}", color=c, fontsize=12,
                fontweight="bold", va="center", ha="right", zorder=3)

    bx.set_yticks(ypos)
    bx.set_yticklabels([f"목표 모집단\n{P['n_target']}명 전체",
                        f"표집틀\n{n_cov}명"], fontsize=11, color=INK)
    bx.set_xlim(0.0, 0.62)
    bx.set_ylim(-1.35, 1.62)
    bx.set_xlabel("어떤 정책에 찬성하는 비율", fontsize=11, color=INK)
    bx.set_title("표본이 겨냥하는 값은 위가 아니라 아래다",
                 fontsize=12.5, color=INK, pad=12)
    bx.tick_params(labelsize=10, colors=INK, length=0)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.spines["bottom"].set_color(MUTED)

    for v, c in zip(vals, cols):
        bx.plot([v, v], [-0.85, 1.28], ls=":", color=c, lw=1.2, zorder=1)
    bx.text((vals[0] + vals[1]) / 2, -0.62,
            f"커버리지 편향 {bias:+.3f}", color=RED, fontsize=11,
            fontweight="bold", ha="center", va="top", zorder=4,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white",
                      edgecolor="none", alpha=0.85))
    bx.text(0.0, -1.25,
            "표본을 20명에서 2만 명으로 늘려도\n표본비율은 아래 값으로 모인다.",
            color=INK, fontsize=10.2, ha="left", va="bottom")

    fig.suptitle("표집틀이 목표 모집단과 어긋난 만큼이 커버리지 오차다",
                 fontsize=14.5, color=INK, y=1.01)
    save(fig, path)
    return dict(n_cov=n_cov, n_out=n_out, rate_target=rate_target,
                rate_frame=rate_frame, rate_out=rate_out, bias=bias)


# ==================================================================
# 2. 같은 모집단 위에 그린 네 가지 확률표집 설계
#    240개 단위를 16열 x 15행 목록으로 늘어놓는다.
#      층    = 5행씩 묶은 세 덩어리 (층 평균이 다르다)
#      집락  = 한 행의 왼쪽 8칸 / 오른쪽 8칸 (총 30개 집락)
#      목록 순서 = 행 우선. 계통표집은 이 순서에서 10번째마다 뽑는다.
# ==================================================================

NCOL, NROW = 16, 15
NPOP = NCOL * NROW          # 240
NSAMP = 24                  # 표본 크기 (모집단의 10%)
MSIZE = 8                   # 집락 크기
KSTEP = NPOP // NSAMP       # 계통표집 간격 = 10
MU_H = np.array([40.0, 55.0, 75.0])   # 층 평균
TAU, SIGMA = 5.0, 6.0                 # 집락 간 / 집락 내 표준편차


def build_grid(seed=11):
    rng = np.random.default_rng(seed)
    idx = np.arange(NPOP)
    row, col = idx // NCOL, idx % NCOL
    band = row // 5                              # 0, 1, 2
    cluster = row * 2 + (col // MSIZE)           # 0 .. 29
    ceff = rng.normal(0.0, TAU, NROW * 2)
    y = MU_H[band] + ceff[cluster] + rng.normal(0.0, SIGMA, NPOP)
    return row, col, band, cluster, y


def design_ses(band, cluster, y, B=40_000, seed=12):
    """네 설계의 표준오차를 재어 돌려준다."""
    rng = np.random.default_rng(seed)
    out = {}

    # --- 단순무작위: 240개에서 24개 ---
    keys = rng.random((B, NPOP))
    pick = np.argpartition(keys, NSAMP, axis=1)[:, :NSAMP]
    out["srs"] = y[pick].mean(axis=1).std(ddof=1)

    # --- 층화(비례 배분): 세 층에서 각각 8개 ---
    est = np.zeros(B)
    for h in range(3):
        pool = np.flatnonzero(band == h)
        k = NSAMP // 3
        kk = rng.random((B, len(pool)))
        sel = pool[np.argpartition(kk, k, axis=1)[:, :k]]
        est += y[sel].mean(axis=1) / 3.0         # 층 가중치가 모두 1/3
    out["strat"] = est.std(ddof=1)

    # --- 집락: 30개 집락 중 3개를 뽑아 그 안을 모두 조사 ---
    ncl = cluster.max() + 1
    members = np.array([np.flatnonzero(cluster == c) for c in range(ncl)])
    kk = rng.random((B, ncl))
    csel = np.argpartition(kk, 3, axis=1)[:, :3]
    out["clust"] = y[members[csel].reshape(B, -1)].mean(axis=1).std(ddof=1)

    # --- 계통: 시작점이 10개뿐이므로 전부 계산한다 (정확값) ---
    sys_means = np.array([y[np.arange(s, NPOP, KSTEP)].mean() for s in range(KSTEP)])
    out["syst"] = sys_means.std(ddof=0)          # 시작점 10개에 균등확률
    out["sys_means"] = sys_means
    return out


def draw_panel(ax, row, col, band, sel, color, title, note):
    # 층 배경
    for h, f in enumerate([ORANGE_F, BLUE_F, GREEN_F]):
        r0 = h * 5
        ax.add_patch(Rectangle((-0.62, -(r0 + 4) - 0.44), NCOL - 1 + 1.24, 4 + 0.88,
                               facecolor=f, edgecolor="none", alpha=0.55, zorder=0))
    for h in (1, 2):
        ax.plot([-0.62, NCOL - 0.38], [-(h * 5) + 0.5] * 2,
                color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)

    mask = np.zeros(NPOP, dtype=bool)
    mask[sel] = True
    ax.scatter(col[~mask], -row[~mask], s=15, facecolor=MUTED,
               edgecolor="none", alpha=0.5, zorder=2)
    ax.scatter(col[mask], -row[mask], s=56, facecolor=color,
               edgecolor="white", lw=0.9, zorder=3)

    ax.set_title(title, fontsize=12.5, color=INK, pad=7)
    ax.text(NCOL - 0.4, 1.05, note, fontsize=9.8, color=INK, ha="right", va="bottom")
    ax.set_xlim(-1.9, NCOL - 0.1)
    ax.set_ylim(-(NROW - 1) - 0.9, 2.1)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_four_designs(path):
    row, col, band, cluster, y = build_grid()
    se = design_ses(band, cluster, y)
    mu = y.mean()
    rng = np.random.default_rng(5)

    band_means = [y[band == h].mean() for h in range(3)]

    # 집락 내 상관은 실제 모집단에서 잰다.
    # 집락이 층 안에 통째로 들어 있으므로 층 간 차이도 집락 간 변동의 일부다.
    cmeans = np.array([y[cluster == c].mean() for c in range(cluster.max() + 1)])
    var_b = cmeans.var()
    var_w = np.mean([y[cluster == c].var() for c in range(cluster.max() + 1)])
    rho = var_b / (var_b + var_w)
    deff_theory = 1 + (MSIZE - 1) * rho
    deff_obs = (se["clust"] / se["srs"]) ** 2

    print(f"  모집단 {NPOP}개, 모평균 {mu:.3f}, 표본 크기 {NSAMP}")
    print(f"  층 평균 {band_means[0]:.2f} / {band_means[1]:.2f} / {band_means[2]:.2f}")
    print(f"  집락 내 상관 rho = {rho:.3f}  (집락 간 분산 {var_b:.2f}, 집락 내 분산 {var_w:.2f})")
    print(f"  설계효과: 이론 1+(m-1)rho = {deff_theory:.2f},  실측 = {deff_obs:.2f}")
    for k in ("srs", "strat", "clust", "syst"):
        print(f"  {k:>6}: 표준오차 {se[k]:.4f}  (SRS 의 {se[k] / se['srs']:.2f}배)")
    print(f"  계통 시작점별 표본평균 {np.round(se['sys_means'], 2)}")
    print(f"  분산비: SRS / 층화 = {(se['srs'] / se['strat']) ** 2:.2f},"
          f"  SRS / 계통 = {(se['srs'] / se['syst']) ** 2:.2f}")
    print(f"  집락 유효표본크기 = {NSAMP / deff_obs:.2f}")
    # 계통표집은 시작점이 무엇이든 각 층에서 8개씩, 집락은 겹치지 않는가
    for s in range(KSTEP):
        ii = np.arange(s, NPOP, KSTEP)
        cnt = np.bincount(band[ii], minlength=3)
        dup = len(ii) - len(np.unique(cluster[ii]))
        print(f"    시작점 {s}: 층별 {cnt}, 같은 집락 중복 {dup}개")

    # 그림에 그릴 대표 표본 하나씩
    sel_srs = rng.choice(NPOP, NSAMP, replace=False)
    sel_str = np.concatenate([rng.choice(np.flatnonzero(band == h), 8, replace=False)
                              for h in range(3)])
    cl = rng.choice(cluster.max() + 1, 3, replace=False)
    sel_cl = np.flatnonzero(np.isin(cluster, cl))
    start = 3
    sel_sy = np.arange(start, NPOP, KSTEP)
    print(f"  그림에 그린 단순무작위 표본의 층별 개수 "
          f"{np.bincount(band[sel_srs], minlength=3)}")

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.0))
    fig.subplots_adjust(hspace=0.15, wspace=0.04)

    panels = [
        (axes[0, 0], sel_srs, BLUE,
         f"단순무작위 · 표준오차 {se['srs']:.2f}",
         "240개에서 24개를 그냥 뽑는다"),
        (axes[0, 1], sel_str, GREEN,
         f"층화(비례 배분) · 표준오차 {se['strat']:.2f}",
         "세 층에서 각각 8개씩"),
        (axes[1, 0], sel_cl, ORANGE,
         f"집락 · 표준오차 {se['clust']:.2f}",
         "8호짜리 집락 세 덩어리를 통째로"),
        (axes[1, 1], sel_sy, PURPLE,
         f"계통 · 표준오차 {se['syst']:.2f}",
         f"목록 순서로 {KSTEP}번째마다"),
    ]
    for ax, sel, c, t, note in panels:
        draw_panel(ax, row, col, band, sel, c, t, note)

    # 층 이름과 층 평균은 왼쪽 두 판에만 적는다
    for ax in (axes[0, 0], axes[1, 0]):
        for h in range(3):
            ax.text(-1.35, -(h * 5 + 2), f"층 {h + 1} · 평균 {band_means[h]:.0f}",
                    fontsize=9.2, color=INK, rotation=90, ha="center", va="center")

    # 집락 판에는 뽑힌 집락을 테두리로 감싼다
    axc = axes[1, 0]
    for c in cl:
        m = np.flatnonzero(cluster == c)
        x0, x1 = col[m].min(), col[m].max()
        r = row[m][0]
        axc.add_patch(Rectangle((x0 - 0.45, -r - 0.42), x1 - x0 + 0.9, 0.84,
                                facecolor="none", edgecolor=ORANGE, lw=1.7,
                                zorder=4))

    fig.suptitle(
        f"같은 모집단 240개에서 24개를 뽑는 네 가지 방법 — 뽑는 규칙이 정밀도를 바꾼다\n"
        f"층 평균이 크게 다르고 한 집락 안은 서로 닮은 모집단 "
        f"(집락 내 상관 {rho:.2f}, 모평균 {mu:.1f})",
        fontsize=14.0, color=INK, y=0.975, linespacing=1.6)

    leg = [Line2D([], [], marker="o", ls="", markersize=8, markerfacecolor=INK,
                  markeredgecolor="white", label="뽑힌 단위"),
           Line2D([], [], marker="o", ls="", markersize=5.2, color=MUTED,
                  label="뽑히지 않은 단위")]
    fig.legend(handles=leg, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, 0.055))
    save(fig, path)
    return se, band_means, rho, mu


# ==================================================================
if __name__ == "__main__":
    print("== frame_coverage ==")
    fig_frame_coverage(OUT + "frame_coverage.png")
    print("== four_designs ==")
    fig_four_designs(OUT + "four_designs.png")
