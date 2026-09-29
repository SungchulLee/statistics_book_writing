r"""12장 생태학적 상관 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch12/ecological_correlation/img/berkeley_levels.png     학과별과 전체가 어긋난다
  ch12/ecological_correlation/img/robinson_reversal.png   개인은 내려가고 평균은 올라간다
  ch12/ecological_correlation/img/aggregation_inflation.png 집계는 상관을 부풀린다
  ch12/ecological_correlation/img/simpson_weights.png     가중평균이 부등호를 뒤집는다

실행:  python3 scripts/make_ch12_ecological_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE, BLUE_F = "#1565C0", "#DCEBFB"
ORANGE, ORANGE_F = "#E65100", "#FFE0B2"
GREEN, GREEN_F = "#33691E", "#C5E1A5"
PURPLE = "#6A1B9A"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch12/ecological_correlation/img/"
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


def headroom(ax, frac=0.28):
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + frac * (hi - lo))


# === 그림 1. 버클리 입학 ===
def fig_berkeley():
    major = ["A", "B", "C", "D", "E", "F"]
    nm = np.array([825, 560, 325, 417, 191, 373], float)
    pm = np.array([0.62, 0.63, 0.37, 0.33, 0.28, 0.06])
    nf = np.array([108, 25, 593, 375, 393, 341], float)
    pf = np.array([0.82, 0.68, 0.34, 0.35, 0.24, 0.07])

    all_m = (nm * pm).sum() / nm.sum()
    all_f = (nf * pf).sum() / nf.sum()
    dept_rate = (nm * pm + nf * pf) / (nm + nf)
    share_f = nf / (nm + nf)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7))

    ax = axes[0]
    clean(ax)
    ax.plot([0, 0.9], [0, 0.9], color=MUTED, lw=1.4, ls="--")
    ax.text(0.03, 0.96, "대각선 위쪽 = 여성 합격률이 더 높은 학과",
            transform=ax.transAxes, fontsize=10, color=MUTED, va="top")
    sizes = 24 + (nm + nf) / 933 * 420
    ax.scatter(pm, pf, s=sizes, color=BLUE, alpha=0.55, edgecolor=BLUE,
               lw=1.2, zorder=3)
    offs = [(0.032, 0.030), (0.032, 0.022), (0.040, -0.012),
            (-0.055, 0.030), (-0.058, 0.008), (0.030, 0.022)]
    for k, mj in enumerate(major):
        ax.annotate(mj, xy=(pm[k], pf[k]),
                    xytext=(pm[k] + offs[k][0], pf[k] + offs[k][1]),
                    fontsize=11, color=INK)
    ax.scatter([all_m], [all_f], s=170, color=RED, marker="X", zorder=4)
    ax.annotate(f"여섯 학과를 합치면\n{100 * all_m:.1f}%  대  {100 * all_f:.1f}%",
                xy=(all_m, all_f), xytext=(0.30, 0.13), fontsize=10.5,
                color=RED,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3))
    ax.set_xlim(0, 0.92)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("남성 합격률", fontsize=10.5, color=INK)
    ax.set_ylabel("여성 합격률", fontsize=10.5, color=INK)
    ax.set_title("학과 여섯 개 중 네 개가 대각선 위쪽에 있다", fontsize=12.5,
                 color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    xs = np.arange(6)
    ax.bar(xs, 100 * share_f, width=0.55, color=PURPLE, alpha=0.75,
           label="여성 지원자 비율")
    ax.plot(xs, 100 * dept_rate, "o-", color=ORANGE, lw=2.2, ms=7,
            label="학과 전체 합격률")
    for k in range(6):
        ax.text(xs[k], 100 * share_f[k] + 2.0, f"{100 * share_f[k]:.0f}%",
                ha="center", fontsize=9.5, color=PURPLE)
    ax.set_xticks(xs)
    ax.set_xticklabels(major, fontsize=11, color=INK)
    ax.set_ylim(0, 100)
    ax.set_xlabel("학과", fontsize=10.5, color=INK)
    ax.set_ylabel("퍼센트", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper center")
    ax.set_title("여성은 합격률이 낮은 학과에 몰려 있었다", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "berkeley_levels.png")
    print(f"  전체 남성 {all_m:.4f}, 여성 {all_f:.4f}")
    print("  학과 합격률:", np.round(dept_rate, 3))
    print("  여성 지원자 비율:", np.round(share_f, 3))
    print(f"  여성이 더 높은 학과 수: {(pf > pm).sum()}")


# === 그림 2. 로빈슨의 역설 ===
def fig_robinson():
    # (a) 본문의 열 명짜리 예
    xA = np.array([10.0, 12, 14, 16, 18])
    yA = np.array([8.0, 6, 4, 2, 0])
    xB = np.array([20.0, 22, 24, 26, 28])
    yB = np.array([18.0, 16, 14, 12, 10])
    x_all = np.concatenate([xA, xB])
    y_all = np.concatenate([yA, yB])
    r_pool = stats.pearsonr(x_all, y_all)[0]
    r_eco = stats.pearsonr([xA.mean(), xB.mean()], [yA.mean(), yB.mean()])[0]
    r_in = stats.pearsonr(xA, yA)[0]

    # (b) 로빈슨의 두 숫자를 재현한 모의자료
    rng = np.random.default_rng(24)
    G, m = 48, 120
    # s 와 rho_B 는 개인 수준 r = -0.11, 집단 수준 r = +0.53 이 나오도록 푼 값.
    s, rho_B, rho_W = 0.6644, 0.5469, -0.40

    def exact_pair(u, v, rho):
        """상관이 정확히 rho 이고 표준편차가 1 인 두 벡터를 만든다."""
        u = (u - u.mean()) / u.std()
        v = v - (v @ u) / (u @ u) * u
        v = (v - v.mean()) / v.std()
        return u, rho * u + np.sqrt(1 - rho ** 2) * v

    a0, b0 = exact_pair(rng.normal(size=G), rng.normal(size=G), rho_B)
    A, Bg = s * a0, s * b0
    e, f = exact_pair(rng.normal(size=G * m), rng.normal(size=G * m), rho_W)
    X = np.repeat(A, m) + e
    Y = np.repeat(Bg, m) + f
    gx = X.reshape(G, m).mean(axis=1)
    gy = Y.reshape(G, m).mean(axis=1)
    r_ind = stats.pearsonr(X, Y)[0]
    r_state = stats.pearsonr(gx, gy)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    clean(ax)
    for xg, yg, col, nm in [(xA, yA, BLUE, "집단 A"), (xB, yB, GREEN, "집단 B")]:
        ax.scatter(xg, yg, s=55, color=col, zorder=3)
        sl, bb = np.polyfit(xg, yg, 1)
        t = np.linspace(xg.min(), xg.max(), 8)
        ax.plot(t, bb + sl * t, color=col, lw=2.2)
        ax.text(xg.mean(), yg.max() + 1.3, nm, fontsize=10.5, color=col,
                ha="center")
    mA = (xA.mean(), yA.mean())
    mB = (xB.mean(), yB.mean())
    ax.plot([mA[0], mB[0]], [mA[1], mB[1]], color=RED, lw=2.6, ls="--",
            zorder=2)
    ax.scatter([mA[0], mB[0]], [mA[1], mB[1]], s=180, color=RED, marker="D",
               zorder=4)
    ax.text(15.0, 12.6, "집단 평균을 이은 선", fontsize=10.5,
            color=RED, ha="left")
    headroom(ax, 0.26)
    ax.text(0.03, 0.97,
            f"집단 안의 r = {r_in:+.2f}\n"
            f"열 명을 합친 r = {r_pool:+.2f}\n"
            f"집단 평균의 r = {r_eco:+.2f}",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.55)
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("본문의 열 명짜리 예", fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    ax.scatter(X, Y, s=3, color=MUTED, alpha=0.13, edgecolor="none")
    sl, bb = np.polyfit(X, Y, 1)
    t = np.linspace(np.quantile(X, 0.002), np.quantile(X, 0.998), 10)
    ax.plot(t, bb + sl * t, color=RED, lw=2.6)
    ax.scatter(gx, gy, s=60, color=BLUE, zorder=4, edgecolor="white", lw=0.9)
    sl2, bb2 = np.polyfit(gx, gy, 1)
    ax.plot(t, bb2 + sl2 * t, color=BLUE, lw=2.6, zorder=3)
    headroom(ax, 0.30)
    ax.text(0.03, 0.97,
            f"개인 {G * m:,}명   r = {r_ind:+.2f}   (빨강)\n"
            f"집단 {G}개 평균   r = {r_state:+.2f}   (파랑)",
            transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
            linespacing=1.55)
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("로빈슨의 두 숫자를 재현한 모의자료", fontsize=12.5,
                 color=INK, pad=8)

    save(fig, "robinson_reversal.png")
    print(f"  toy: 집단 내 {r_in:+.3f}, 합친 {r_pool:+.3f}, 평균 {r_eco:+.3f}")
    print(f"  robinson: 개인 {r_ind:+.4f}, 집단 {r_state:+.4f}")


# === 그림 3. 집계가 상관을 부풀린다 ===
def fig_aggregation():
    sA = sB = 1.0
    se = sf = 2.0
    rho_B, rho_W = 0.9, 0.15

    def eco_r(m):
        num = rho_B * sA * sB + rho_W * se * sf / m
        den = np.sqrt((sA ** 2 + se ** 2 / m) * (sB ** 2 + sf ** 2 / m))
        return num / den

    ms = np.unique(np.round(np.logspace(0, 3, 70)).astype(int))
    curve = np.array([eco_r(m) for m in ms])

    rng = np.random.default_rng(404)
    G, m_big = 60, 30
    A = rng.normal(0, sA, G)
    Bg = rho_B * A + np.sqrt(1 - rho_B ** 2) * rng.normal(0, sB, G)
    e = rng.normal(0, se, (G, m_big))
    f = rho_W * e + np.sqrt(1 - rho_W ** 2) * rng.normal(0, sf, (G, m_big))
    Xi = A[:, None] + e
    Yi = Bg[:, None] + f
    r_ind = stats.pearsonr(Xi.ravel(), Yi.ravel())[0]
    gx, gy = Xi.mean(axis=1), Yi.mean(axis=1)
    r_grp = stats.pearsonr(gx, gy)[0]

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    ax = axes[0]
    clean(ax)
    ax.scatter(Xi.ravel(), Yi.ravel(), s=5, color=MUTED, alpha=0.35,
               edgecolor="none")
    sl, bb = np.polyfit(Xi.ravel(), Yi.ravel(), 1)
    t = np.linspace(Xi.min(), Xi.max(), 10)
    ax.plot(t, bb + sl * t, color=RED, lw=2.2)
    headroom(ax, 0.26)
    ax.text(0.03, 0.97, f"개인 {G * m_big:,}명\nr = {r_ind:+.2f}",
            transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
            linespacing=1.5)
    ax.set_xlabel("X", fontsize=10.5, color=INK)
    ax.set_ylabel("Y", fontsize=10.5, color=INK)
    ax.set_title("개인 수준", fontsize=12.5, color=INK, pad=8)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    ax = axes[1]
    clean(ax)
    ax.scatter(gx, gy, s=42, color=BLUE, alpha=0.85, edgecolor="none")
    sl, bb = np.polyfit(gx, gy, 1)
    t = np.linspace(gx.min(), gx.max(), 10)
    ax.plot(t, bb + sl * t, color=BLUE, lw=2.4)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.text(0.03, 0.97, f"집단 {G}개의 평균\n(집단마다 {m_big}명)\n"
                        f"r = {r_grp:+.2f}",
            transform=ax.transAxes, fontsize=10.8, color=INK, va="top",
            linespacing=1.5)
    ax.set_xlabel("평균 X", fontsize=10.5, color=INK)
    ax.set_ylabel("평균 Y", fontsize=10.5, color=INK)
    ax.set_title("같은 자료를 30명씩 평균 내면", fontsize=12.5, color=INK,
                 pad=8)

    ax = axes[2]
    clean(ax)
    ax.plot(ms, curve, color=PURPLE, lw=2.4)
    ax.axhline(rho_B, color=MUTED, lw=1.4, ls=":")
    ax.text(1.12, 0.945, "집단 크기가 무한하면 도달하는 집단 간 상관 0.90",
            fontsize=10, color=MUTED, ha="left")
    ax.set_xscale("log")
    ticks = [1, 3, 10, 30, 100, 300, 1000]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=9.5, color=INK)
    for mm in (1, 10, 100):
        ax.scatter([mm], [eco_r(mm)], s=44, color=PURPLE, zorder=3)
        ax.annotate(f"{eco_r(mm):.2f}", xy=(mm, eco_r(mm)),
                    xytext=(mm * 1.25, eco_r(mm) - 0.075), fontsize=10,
                    color=PURPLE)
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("집단 크기 m", fontsize=10.5, color=INK)
    ax.set_ylabel("생태학적 상관", fontsize=10.5, color=INK)
    ax.set_title("평균 낼 사람이 많을수록 커진다", fontsize=12.5, color=INK,
                 pad=8)

    save(fig, "aggregation_inflation.png")
    print(f"  개인 r={r_ind:.4f}, 집단평균 r={r_grp:.4f}")
    for mm in (1, 2, 5, 10, 30, 100, 1000):
        print(f"  m={mm:5d}: 생태학적 r = {eco_r(mm):.4f}")


# === 그림 4. 가중평균이 부등호를 뒤집는다 ===
def fig_simpson_weights():
    # 신장결석 자료
    small = {"A": (81, 87), "B": (234, 270)}
    large = {"A": (192, 263), "B": (55, 80)}
    tot = {t: small[t][1] + large[t][1] for t in "AB"}
    rate_s = {t: small[t][0] / small[t][1] for t in "AB"}
    rate_l = {t: large[t][0] / large[t][1] for t in "AB"}
    crude = {t: (small[t][0] + large[t][0]) / tot[t] for t in "AB"}

    n_small = small["A"][1] + small["B"][1]
    n_large = large["A"][1] + large["B"][1]
    w_s = n_small / (n_small + n_large)
    std = {t: w_s * rate_s[t] + (1 - w_s) * rate_l[t] for t in "AB"}

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6))

    ax = axes[0]
    clean(ax)
    base = {"A": 0.0, "B": 1.30}
    for t, col in [("A", BLUE), ("B", ORANGE)]:
        ws = small[t][1] / tot[t]
        for x0, w, h, al, n in [(base[t], ws, rate_s[t], 0.85, small[t][1]),
                                (base[t] + ws, 1 - ws, rate_l[t], 0.38,
                                 large[t][1])]:
            ax.add_patch(Rectangle((x0, 0), w, h, facecolor=col, alpha=al,
                                   edgecolor="white", lw=1.4))
            ax.text(x0 + w / 2, 0.45, f"{100 * h:.0f}%", ha="center",
                    va="center", fontsize=12,
                    color="white" if al > 0.6 else INK)
            ax.text(x0 + w / 2, -0.045, f"{n}명", ha="center", va="center",
                    fontsize=9.5, color=INK)
        ax.plot([base[t], base[t] + 1], [crude[t], crude[t]], color=RED,
                lw=2.4, ls="--", zorder=3)
        ax.text(base[t] + 1.02, crude[t], f"{100 * crude[t]:.0f}%",
                fontsize=11, color=RED, va="center")
        ax.text(base[t] + 0.5, -0.115, f"치료 {t}", ha="center",
                fontsize=12.5, color=col)
    handles = [Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.85,
                         label="작은 결석"),
               Rectangle((0, 0), 1, 1, facecolor=MUTED, alpha=0.38,
                         label="큰 결석")]
    ax.legend(handles=handles, fontsize=10, frameon=False,
              loc="upper right", ncol=1)
    ax.set_xlim(-0.12, 2.62)
    ax.set_ylim(-0.16, 1.14)
    ax.set_xticks([])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("성공률", fontsize=10.5, color=INK)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_bounds(0, 1.0)
    ax.set_title("막대의 폭이 환자 비율, 붉은 파선이 합친 성공률",
                 fontsize=12.5, color=INK, pad=8)

    ax = axes[1]
    clean(ax)
    xs = np.arange(2)
    w = 0.34
    vals_c = [crude["A"], crude["B"]]
    vals_s = [std["A"], std["B"]]
    ax.bar(xs - w / 2, vals_c, width=w, color=MUTED, alpha=0.9,
           label="합친 성공률")
    ax.bar(xs + w / 2, vals_s, width=w, color=GREEN, alpha=0.9,
           label="결석 크기를 표준화한 성공률")
    for k in range(2):
        ax.text(xs[k] - w / 2, vals_c[k] + 0.012, f"{100 * vals_c[k]:.1f}%",
                ha="center", fontsize=10.5, color=INK)
        ax.text(xs[k] + w / 2, vals_s[k] + 0.012, f"{100 * vals_s[k]:.1f}%",
                ha="center", fontsize=10.5, color=GREEN)
    ax.set_xticks(xs)
    ax.set_xticklabels(["치료 A", "치료 B"], fontsize=11.5, color=INK)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("성공률", fontsize=10.5, color=INK)
    ax.legend(fontsize=10, frameon=False, loc="upper center")
    ax.set_title("가중치를 맞추면 부등호가 제자리로 돌아온다",
                 fontsize=12.5, color=INK, pad=8)

    save(fig, "simpson_weights.png")
    for t in "AB":
        print(f"  치료 {t}: 작은 {rate_s[t]:.3f} ({small[t][1]}명), "
              f"큰 {rate_l[t]:.3f} ({large[t][1]}명), "
              f"합친 {crude[t]:.4f}, 표준화 {std[t]:.4f}")
    print(f"  작은 결석 비중 w = {w_s:.4f}")


if __name__ == "__main__":
    fig_berkeley()
    fig_robinson()
    fig_aggregation()
    fig_simpson_weights()
