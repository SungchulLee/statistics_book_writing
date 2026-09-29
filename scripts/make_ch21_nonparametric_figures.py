r"""21.2 비모수 생존 추정 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch21/nonparametric/img/km_risk_set.png        절단은 위험집합을 줄여 다음 하강을 키운다
  ch21/nonparametric/img/na_vs_km.png           넬슨-알렌 곡선은 언제나 카플란-마이어 위에
  ch21/nonparametric/img/logrank_crossing.png   곡선이 교차하면 부호가 상쇄된다
  ch21/nonparametric/img/km_ci_transforms.png   선형 구간은 [0,1] 을 벗어난다

실행:  python3 scripts/make_ch21_nonparametric_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
       카플란-마이어와 로그순위는 numpy 로 직접 구현한다(lifelines 불필요).
"""
import os

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

OUT = "docs/ch21/nonparametric/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean_axis(ax):
    ax.tick_params(labelsize=10, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 비모수 추정량 (numpy 만으로) ===
def km_table(t, d):
    """사건시간별 (t_j, n_j, d_j, S_j, H_j, greenwood 누적합) 을 돌려준다."""
    t = np.asarray(t, float)
    d = np.asarray(d, int)
    ts = np.sort(t)
    ev = np.unique(t[d == 1])
    n_j = len(t) - np.searchsorted(ts, ev, side="left")
    es = np.sort(t[d == 1])
    d_j = (np.searchsorted(es, ev, side="right")
           - np.searchsorted(es, ev, side="left"))
    S = np.cumprod(1.0 - d_j / n_j)
    H = np.cumsum(d_j / n_j)
    with np.errstate(divide="ignore"):
        gw = np.cumsum(d_j / (n_j * (n_j - d_j)))
    return ev, n_j, d_j, S, H, gw


def step_xy(ev, vals, start, tmax):
    """계단 그리기용 (x, y). drawstyle='steps-post' 와 함께 쓴다."""
    x = np.concatenate([[0.0], ev, [tmax]])
    y = np.concatenate([[start], vals, [vals[-1]]])
    return x, y


def logrank(t1, d1, t2, d2):
    """두 집단 로그순위 검정. (O1, E1, V1, chi2, p, 사건시간, 누적 O-E) 를 돌려준다."""
    t = np.concatenate([t1, t2])
    d = np.concatenate([d1, d2])
    ev = np.unique(t[d == 1])
    n1 = np.array([(t1 >= u).sum() for u in ev], float)
    n2 = np.array([(t2 >= u).sum() for u in ev], float)
    nj = n1 + n2
    d1j = np.array([((t1 == u) & (d1 == 1)).sum() for u in ev], float)
    dj = np.array([((t == u) & (d == 1)).sum() for u in ev], float)
    e1j = dj * n1 / nj
    with np.errstate(divide="ignore", invalid="ignore"):
        v1j = n1 * n2 * dj * (nj - dj) / (nj ** 2 * (nj - 1))
    v1j = np.where(nj > 1, v1j, 0.0)
    O1, E1, V1 = d1j.sum(), e1j.sum(), v1j.sum()
    chi2 = (O1 - E1) ** 2 / V1
    p = stats.chi2.sf(chi2, 1)
    return O1, E1, V1, chi2, p, ev, np.cumsum(d1j - e1j)


# === 그림 1. 절단은 위험집합을 줄여 다음 하강을 키운다 ===
def fig_km_risk_set():
    t = np.array([1, 2, 3, 4, 5, 5, 7, 9], float)
    d = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    ev, n_j, d_j, S, H, gw = km_table(t, d)
    print("KM 보기:", list(zip(ev, n_j, d_j, np.round(S, 4))))

    fig, (axT, axB) = plt.subplots(
        2, 1, figsize=(10.4, 7.6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15], "hspace": 0.16})

    ys = np.arange(8)[::-1]
    for y, ti, di in zip(ys, t, d):
        axT.plot([0, ti], [y, y], color=INK, lw=2.2, solid_capstyle="butt", zorder=3)
        if di == 1:
            axT.plot(ti, y, "o", color=ORANGE, ms=9, mec="white", mew=1.2, zorder=4)
        else:
            axT.plot([ti, ti], [y - 0.28, y + 0.28], color=BLUE, lw=3.0, zorder=4)
    for u in ev:
        axT.axvline(u, color="#CFD8DC", lw=1.0, zorder=0)
    axT.set_ylim(-0.7, 7.5)
    axT.set_yticks(ys)
    axT.set_yticklabels([f"대상 {i}" for i in range(1, 9)], fontsize=10)
    axT.tick_params(axis="y", length=0)
    axT.set_title("주황은 사건, 파랑은 절단. 절단은 곡선을 떨어뜨리지 않는다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axT)
    axT.spines["left"].set_visible(False)
    axT.spines["bottom"].set_visible(False)

    # 아래: 카플란-마이어 계단
    x, y = step_xy(ev, S, 1.0, 10.0)
    axB.plot(x, y, drawstyle="steps-post", color=BLUE, lw=2.6, zorder=4)
    axB.fill_between(x, y, step="post", color=BLUE_F, alpha=0.75, zorder=1)
    for u in ev:
        axB.axvline(u, color="#CFD8DC", lw=1.0, zorder=0)

    prev = 1.0
    for u, nn, dd, s in zip(ev, n_j, d_j, S):
        axB.annotate("", xy=(u, s), xytext=(u, prev),
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
        axB.text(u + 0.16, (prev + s) / 2, f"{prev - s:.3f} 하강",
                 fontsize=9.5, color=RED, va="center")
        axB.text(u, 1.055, f"$n_j$={int(nn)}\n$d_j$={int(dd)}", ha="center",
                 va="bottom", fontsize=9.5, color=INK, linespacing=1.35)
        prev = s

    # 절단 시점 표시
    for c in [2.0, 4.0, 7.0]:
        axB.plot(c, -0.03, marker="^", color=BLUE, ms=10, clip_on=False, zorder=5)
    axB.text(5.45, 0.11, "파란 삼각형이 절단 시점이다.\n하강은 없고 위험집합만 줄어든다",
             ha="left", va="bottom", fontsize=10.5, color=BLUE, linespacing=1.4,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.92, pad=3))

    axB.set_xlim(-0.2, 10.4)
    axB.set_ylim(0, 1.22)
    axB.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xticks([0, 1, 2, 3, 4, 5, 7, 9])
    axB.set_xlabel("시간", fontsize=11, color=INK)
    axB.set_ylabel(r"$\hat S(t)$", fontsize=11, color=INK)
    clean_axis(axB)

    save(fig, "km_risk_set.png")
    return dict(ev=ev, n_j=n_j, S=S)


# === 그림 2. 넬슨-알렌 대 카플란-마이어 ===
def fig_na_vs_km():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.8))

    # --- 왼쪽: 같은 8명 자료 ---
    t = np.array([1, 2, 3, 4, 5, 5, 7, 9], float)
    d = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    ev, n_j, d_j, S, H, gw = km_table(t, d)
    S_na = np.exp(-H)
    print("KM :", np.round(S, 4))
    print("NA :", np.round(S_na, 4), " H:", np.round(H, 4))

    xk, yk = step_xy(ev, S, 1.0, 10.5)
    xn, yn = step_xy(ev, S_na, 1.0, 10.5)
    axL.fill_between(xk, yk, yn, step="post", color=ORANGE_F, alpha=0.65, zorder=1)
    axL.plot(xk, yk, drawstyle="steps-post", color=BLUE, lw=2.6, zorder=3,
             label=r"카플란-마이어 $\hat S_{KM}$")
    axL.plot(xn, yn, drawstyle="steps-post", color=GREEN, lw=2.6, zorder=3,
             ls=(0, (5, 2)), label=r"넬슨-알렌 $e^{-\hat H}$")
    for u, a, b in [(5.0, S[2], S_na[2]), (9.0, S[3], S_na[3])]:
        axL.annotate("", xy=(u + 0.55, a), xytext=(u + 0.55, b),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
        axL.text(u + 0.75, (a + b) / 2, f"{b - a:.3f}", fontsize=10, color=RED,
                 va="center")
    axL.text(5.0, 0.82, r"$d_j/n_j$ 가 커지는 꼬리에서" "\n" "두 곡선이 벌어진다",
             fontsize=10, color=INK, ha="center")
    axL.set_xlim(-0.2, 10.5)
    axL.set_ylim(-0.02, 1.06)
    axL.set_xlabel("시간", fontsize=11, color=INK)
    axL.set_ylabel("생존 추정치", fontsize=11, color=INK)
    axL.set_title("넬슨-알렌 곡선은 언제나 위에 있다", fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10, frameon=False, loc="lower left")
    clean_axis(axL)

    # --- 오른쪽: 누적위험 그림이 모양을 드러낸다 ---
    rng = np.random.default_rng(21)
    n = 400
    lam = 0.10
    T_exp = rng.exponential(1 / lam, n)
    k, sc = 2.4, 11.0
    T_wb = sc * rng.weibull(k, n)
    slopes = {}
    for name, Tv, col, ls in [("지수 (상수 위험)", T_exp, BLUE, "-"),
                              ("와이불 $k=2.4$ (증가 위험)", T_wb, ORANGE, "-")]:
        C = rng.uniform(4.0, 30.0, n)
        tt = np.minimum(Tv, C)
        dd = (Tv <= C).astype(int)
        ev2, n2, d2, S2, H2, _ = km_table(tt, dd)
        m = ev2 <= 18.0
        axR.plot(np.concatenate([[0.0], ev2[m]]), np.concatenate([[0.0], H2[m]]),
                 drawstyle="steps-post", color=col, lw=2.4, ls=ls, label=name)
        fit = np.polyfit(ev2[m], H2[m], 1)
        slopes[name] = fit[0]
    axR.plot([0, 18], [0, 18 * lam], color=MUTED, lw=1.4, ls=(0, (5, 3)), zorder=0,
             label=r"기울기 $\lambda = 0.10$ 인 직선")
    axR.text(18.2, 0.30, "주황은 위로 굽는다\n— 위험이 커지는 중이다", fontsize=10.5,
             color=ORANGE, ha="right", va="bottom", linespacing=1.4)
    print("누적위험 기울기:", {kk: round(v, 4) for kk, v in slopes.items()})
    axR.set_xlim(0, 18.6)
    axR.set_ylim(0, 2.5)
    axR.set_xlabel("시간 $t$", fontsize=11, color=INK)
    axR.set_ylabel(r"넬슨-알렌 $\hat H(t)$", fontsize=11, color=INK)
    axR.set_title("누적위험 그림은 직선인지만 보면 된다",
                  fontsize=12.5, color=INK, pad=10)
    axR.legend(fontsize=10, frameon=False, loc="upper left")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "na_vs_km.png")
    return dict(S=S, S_na=S_na, slopes=slopes)


# === 그림 3. 교차하는 곡선에서 로그순위가 무너진다 ===
def fig_logrank_crossing():
    rng = np.random.default_rng(2024)
    n = 160
    tau = 40.0

    # 수술군: 초기 위험이 크고 그 시기를 넘기면 매우 안전하다
    early = rng.random(n) < 0.54
    T1 = np.where(early, rng.exponential(2.2, n), 18.0 + rng.exponential(45.0, n))
    # 약물군: 위험이 꾸준하다
    T2 = rng.exponential(15.0, n)
    C1 = np.minimum(rng.uniform(8.0, 60.0, n), tau)
    C2 = np.minimum(rng.uniform(8.0, 60.0, n), tau)
    t1, d1 = np.minimum(T1, C1), (T1 <= C1).astype(int)
    t2, d2 = np.minimum(T2, C2), (T2 <= C2).astype(int)

    O1, E1, V1, chi2, p, ev, cum = logrank(t1, d1, t2, d2)
    print(f"교차: O1={O1:.0f} E1={E1:.2f} V1={V1:.2f} chi2={chi2:.3f} p={p:.3f}")
    print(f"   누적 O-E 최고 {cum.max():.2f} (t={ev[cum.argmax()]:.1f}), 최종 {cum[-1]:.2f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    for tt, dd, col, name in [(t1, d1, ORANGE, "수술군"), (t2, d2, BLUE, "약물군")]:
        ev_g, _, _, S_g, _, _ = km_table(tt, dd)
        x, y = step_xy(ev_g, S_g, 1.0, tau)
        axL.plot(x, y, drawstyle="steps-post", color=col, lw=2.6, label=name)
    # 교차 지점 찾기
    evA, _, _, SA, _, _ = km_table(t1, d1)
    evB, _, _, SB, _, _ = km_table(t2, d2)
    grid = np.linspace(0.2, tau, 800)
    fA = np.interp(grid, np.concatenate([[0], evA]), np.concatenate([[1], SA]))
    fB = np.interp(grid, np.concatenate([[0], evB]), np.concatenate([[1], SB]))
    diff = fA - fB
    cross = grid[np.where(np.sign(diff[:-1]) != np.sign(diff[1:]))[0]]
    xc = cross[0] if len(cross) else np.nan
    print(f"   교차 시점 t={xc:.1f}")
    axL.axvline(xc, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=0)
    axL.text(xc + 0.7, 0.93, f"교차 $t\\approx{xc:.0f}$", fontsize=10, color=INK,
             ha="left", va="top")
    axL.set_xlim(0, tau)
    axL.set_ylim(0, 1.02)
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"$\hat S(t)$", fontsize=11, color=INK)
    axL.set_title("두 곡선은 눈에 띄게 다르다", fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10.5, frameon=False, loc="upper right")
    clean_axis(axL)

    axR.axhline(0.0, color=MUTED, lw=1.2, zorder=0)
    axR.plot(np.concatenate([[0.0], ev]), np.concatenate([[0.0], cum]),
             drawstyle="steps-post", color=PURPLE, lw=2.6, zorder=3)
    axR.fill_between(np.concatenate([[0.0], ev]), np.concatenate([[0.0], cum]),
                     0.0, step="post", color=PURPLE, alpha=0.14, zorder=1)
    jmax = cum.argmax()
    axR.plot(ev[jmax], cum[jmax], "o", color=RED, ms=8, zorder=5)
    axR.annotate(f"앞구간에서 $+{cum[jmax]:.1f}$ 까지 쌓였다가",
                 xy=(ev[jmax], cum[jmax]), xytext=(ev[jmax] + 2.6, cum[jmax] + 5.0),
                 fontsize=10.5, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    axR.plot(ev[-1], cum[-1], "o", color=GREEN, ms=8, zorder=5)
    axR.annotate(f"뒷구간이 되받아 ${cum[-1]:+.1f}$ 로 돌아온다",
                 xy=(ev[-1], cum[-1]), xytext=(ev[-1] - 1.5, 13.0),
                 fontsize=10.5, color=GREEN, ha="right", va="center",
                 arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.3))
    axR.text(0.8, -8.0,
             f"로그순위:  $O_1-E_1$ = {O1 - E1:+.2f},   "
             f"$\\chi^2$ = {chi2:.3f},   $p$ = {p:.2f}",
             fontsize=11.5, color=INK, ha="left", va="bottom")
    axR.set_xlim(0, tau)
    axR.set_ylim(-10.0, 48.0)
    axR.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axR.set_ylabel(r"누적 $\sum (d_{1j}-e_{1j})$", fontsize=11, color=INK)
    axR.set_title("부호가 있는 편차라서 서로 상쇄된다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "logrank_crossing.png")
    return dict(O1=O1, E1=E1, chi2=chi2, p=p, cross=xc,
                peak=cum[jmax], peak_t=ev[jmax], final=cum[-1])


# === 그림 4. 세 신뢰구간의 비교 ===
def fig_km_ci_transforms():
    z = 1.959964

    def bands(t, d, tmax):
        ev, n_j, d_j, S, H, gw = km_table(t, d)
        ok = np.isfinite(gw) & (S > 0)
        se_theta = np.sqrt(np.where(ok, gw, 0.0))
        lin_lo = S - z * S * se_theta
        lin_hi = S + z * S * se_theta
        with np.errstate(divide="ignore", invalid="ignore"):
            phi = np.log(-np.log(S))
            se_phi = se_theta / np.abs(np.log(S))
            ll_lo = np.exp(-np.exp(phi + z * se_phi))
            ll_hi = np.exp(-np.exp(phi - z * se_phi))
        return ev, n_j, S, lin_lo, lin_hi, ll_lo, ll_hi, ok

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 5.0))

    # --- 왼쪽: 환자 10명 자료 ---
    t = np.array([3, 5, 7, 8, 10, 12, 12, 15, 18, 20], float)
    d = np.array([1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    ev, n_j, S, lo, hi, llo, lhi, ok = bands(t, d, 22.0)
    m = ok
    for j in range(len(ev)):
        if not m[j]:
            continue
        print(f"t={ev[j]:.0f} n={n_j[j]} S={S[j]:.3f} "
              f"선형=({lo[j]:.3f},{hi[j]:.3f}) 로그로그=({llo[j]:.3f},{lhi[j]:.3f})")

    xk, yk = step_xy(ev[m], S[m], 1.0, 21.0)
    axL.plot(xk, yk, drawstyle="steps-post", color=INK, lw=2.6, zorder=5)
    xa, y_lo = step_xy(ev[m], lo[m], 1.0, 21.0)
    _, y_hi = step_xy(ev[m], hi[m], 1.0, 21.0)
    _, y_llo = step_xy(ev[m], llo[m], 1.0, 21.0)
    _, y_lhi = step_xy(ev[m], lhi[m], 1.0, 21.0)
    axL.fill_between(xa, y_llo, y_lhi, step="post", color=GREEN_F, alpha=0.75,
                     zorder=1)
    axL.plot(xa, y_llo, drawstyle="steps-post", color=GREEN, lw=1.8, zorder=3)
    axL.plot(xa, y_lhi, drawstyle="steps-post", color=GREEN, lw=1.8, zorder=3)
    axL.plot(xa, y_lo, drawstyle="steps-post", color=RED, lw=1.6, ls=(0, (4, 2)),
             zorder=4)
    axL.plot(xa, y_hi, drawstyle="steps-post", color=RED, lw=1.6, ls=(0, (4, 2)),
             zorder=4)
    ci_handles = [
        plt.Line2D([], [], color=INK, lw=2.6, label=r"$\hat S(t)$"),
        plt.Line2D([], [], color=RED, lw=1.6, ls=(0, (4, 2)), label="선형 95% 구간"),
        plt.Line2D([], [], color=GREEN, lw=1.8, label="로그-로그 95% 구간"),
    ]

    axL.axhline(1.0, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=0)
    axL.axhline(0.0, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=0)
    axL.annotate("상한이 1 을 넘는다", xy=(4.0, 1.086), xytext=(6.4, 1.20),
                 fontsize=10, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axL.annotate("하한이 0 아래로 내려간다", xy=(16.4, -0.006), xytext=(11.0, -0.20),
                 fontsize=10, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    axL.set_xlim(0, 21.0)
    axL.set_ylim(-0.30, 1.34)
    axL.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"$\hat S(t)$", fontsize=11, color=INK)
    axL.set_title(r"$n=10$: 선형 구간은 $[0,1]$ 을 벗어난다",
                  fontsize=12.5, color=INK, pad=10)
    axL.legend(handles=ci_handles, fontsize=9.5, frameon=True, framealpha=0.93,
               edgecolor=MUTED, loc="lower left", ncol=1)
    clean_axis(axL)

    # --- 오른쪽: n=200, 꼬리에서 넓어지는 띠 ---
    rng = np.random.default_rng(7)
    n = 200
    T = rng.exponential(14.0, n)
    C = np.minimum(rng.uniform(3.0, 55.0, n), 36.0)
    tt, dd = np.minimum(T, C), (T <= C).astype(int)
    ev2, n2, S2, lo2, hi2, llo2, lhi2, ok2 = bands(tt, dd, 36.0)
    m2 = ok2 & (ev2 <= 34.0)
    xb, yb = step_xy(ev2[m2], S2[m2], 1.0, 34.0)
    _, y2lo = step_xy(ev2[m2], llo2[m2], 1.0, 34.0)
    _, y2hi = step_xy(ev2[m2], lhi2[m2], 1.0, 34.0)
    axR.fill_between(xb, y2lo, y2hi, step="post", color=BLUE_F, alpha=0.95, zorder=1)
    axR.plot(xb, y2lo, drawstyle="steps-post", color=BLUE, lw=1.3, zorder=2)
    axR.plot(xb, y2hi, drawstyle="steps-post", color=BLUE, lw=1.3, zorder=2)
    axR.plot(xb, yb, drawstyle="steps-post", color=INK, lw=2.4, zorder=4)

    for tq, dx, dy, ha in ((8.0, 0.7, 0.035, "left"), (20.0, 0.7, 0.115, "left"),
                           (32.0, -0.7, 0.035, "right")):
        j = np.argmin(np.abs(ev2[m2] - tq))
        u = ev2[m2][j]
        w = lhi2[m2][j] - llo2[m2][j]
        rel = w / S2[m2][j]
        nn = n2[m2][j]
        print(f"n=200  t={u:.1f}  위험집합={int(nn)}  S={S2[m2][j]:.3f}  "
              f"폭={w:.3f}  상대폭={rel:.2f}")
        axR.annotate("", xy=(u, llo2[m2][j]), xytext=(u, lhi2[m2][j]),
                     arrowprops=dict(arrowstyle="<->", color=RED, lw=1.5))
        axR.text(u + dx, lhi2[m2][j] + dy,
                 f"위험집합 {int(nn)}명\n폭 = $\\hat S$ 의 {rel * 100:.0f}%",
                 fontsize=9.5, color=RED, ha=ha, va="bottom", linespacing=1.35)

    axR.set_xlim(0, 35.5)
    axR.set_ylim(0, 1.16)
    axR.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axR.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axR.set_ylabel(r"$\hat S(t)$", fontsize=11, color=INK)
    axR.set_title(r"$n=200$: 꼬리로 갈수록 상대 불확실성이 커진다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "km_ci_transforms.png")


if __name__ == "__main__":
    fig_km_risk_set()
    fig_na_vs_km()
    fig_logrank_crossing()
    fig_km_ci_transforms()
