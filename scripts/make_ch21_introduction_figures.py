r"""21.1 생존 모형 입문 네 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch21/introduction/img/censoring_min_and_bias.png   t=min(T,C) 와 절단 무시의 편향
  ch21/introduction/img/censoring_vs_truncation.png  세 절단 유형과 절단(truncation)
  ch21/introduction/img/hazard_shapes_same_median.png  같은 중앙값, 다른 위험 모양
  ch21/introduction/img/credit_default_hazard.png    신용 부도의 봉우리형 위험

실행:  python3 scripts/make_ch21_introduction_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np

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

OUT = "docs/ch21/introduction/img/"
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


# === 카플란-마이어 (numpy 만으로) ===
def km_curve(t, d):
    """관측시간 t, 사건지시자 d 로부터 (시점, S) 계단을 돌려준다."""
    t = np.asarray(t, float)
    d = np.asarray(d, int)
    ts = np.sort(t)
    ev = np.unique(t[d == 1])
    n_risk = len(t) - np.searchsorted(ts, ev, side="left")
    ev_sorted = np.sort(t[d == 1])
    d_u = (np.searchsorted(ev_sorted, ev, side="right")
           - np.searchsorted(ev_sorted, ev, side="left"))
    S = np.cumprod(1.0 - d_u / n_risk)
    return np.concatenate([[0.0], ev]), np.concatenate([[1.0], S])


def km_area(t, d, tmax):
    """카플란-마이어 곡선 아래 넓이 (tmax 까지의 제한 평균)."""
    ts, Ss = km_curve(t, d)
    keep = ts <= tmax
    ts, Ss = ts[keep], Ss[keep]
    widths = np.diff(np.append(ts, tmax))
    return float(np.sum(Ss * widths))


# === 그림 1. t = min(T, C) 와 절단을 무시할 때의 편향 ===
def fig_censoring_min_and_bias():
    rng = np.random.default_rng(42)
    n = 100_000
    T = rng.exponential(10.0, n)
    C = rng.exponential(10.0, n)
    t = np.minimum(T, C)
    d = (T <= C).astype(int)

    true_mean = T.mean()
    naive_event = t.mean()
    drop_cens = t[d == 1].mean()
    km_mean = km_area(t, d, t.max())               # 곡선 아래 넓이 = 평균의 추정
    print(f"true={true_mean:.4f} naive={naive_event:.4f} "
          f"drop={drop_cens:.4f} km={km_mean:.4f} "
          f"cens_rate={1 - d.mean():.3f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.2, 4.8), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- 왼쪽: 개인별 추적선 ---
    T_show = np.array([4.2, 11.0, 2.6, 18.5, 7.4, 13.2, 9.1, 16.0])
    C_show = np.array([14.0, 6.3, 14.0, 14.0, 14.0, 9.8, 14.0, 3.5])
    t_show = np.minimum(T_show, C_show)
    d_show = (T_show <= C_show).astype(int)
    ys = np.arange(len(T_show))[::-1]

    for y, Ti, ti, di in zip(ys, T_show, t_show, d_show):
        if di == 0:
            axL.plot([ti, Ti], [y, y], color=MUTED, lw=1.4, ls=(0, (3, 2)), zorder=1)
            axL.plot(Ti, y, marker="x", color=MUTED, ms=7, mew=1.8, zorder=2)
        axL.plot([0, ti], [y, y], color=INK, lw=2.4, solid_capstyle="butt", zorder=3)
        if di == 1:
            axL.plot(ti, y, "o", color=ORANGE, ms=9, mec="white", mew=1.2, zorder=4)
        else:
            axL.plot([ti, ti], [y - 0.28, y + 0.28], color=BLUE, lw=3.0, zorder=4)

    axL.axvline(14.0, color=RED, lw=1.4, ls=(0, (5, 3)), zorder=0)
    axL.text(14.35, len(ys) - 0.5, "연구 종료", color=RED, fontsize=10,
             ha="left", va="bottom")
    axL.set_xlim(-0.6, 20.0)
    axL.set_ylim(-0.75, len(ys) - 0.15)
    axL.set_yticks(ys)
    axL.set_yticklabels([f"대상 {i}" for i in range(1, len(ys) + 1)], fontsize=10)
    axL.set_xlabel("추적 시간 (개월)", fontsize=11, color=INK)
    axL.set_title(r"관측되는 것은 $t_i=\min(T_i,C_i)$ 뿐이다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axL)
    axL.spines["left"].set_visible(False)
    axL.tick_params(axis="y", length=0)

    handles = [
        plt.Line2D([], [], color=INK, lw=2.4, label="관측된 추적 기간"),
        plt.Line2D([], [], color=ORANGE, marker="o", ls="none", ms=9,
                   label=r"사건 발생  $\delta_i=1$"),
        plt.Line2D([], [], color=BLUE, marker="|", ls="none", ms=13, mew=3,
                   label=r"절단  $\delta_i=0$"),
        plt.Line2D([], [], color=MUTED, lw=1.4, ls=(0, (3, 2)), marker="x", ms=7,
                   label=r"관측 못 한 참 $T_i$"),
    ]
    axL.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncol=4, fontsize=9.5, frameon=False, columnspacing=1.6,
               handlelength=2.0)

    # --- 오른쪽: 평균 생존시간 추정치 네 개 ---
    labels = ["절단을\n사건으로", "절단된 대상을\n버림", "카플란-\n마이어", "참값"]
    vals = [naive_event, drop_cens, km_mean, true_mean]
    cols = [RED, RED, GREEN, MUTED]
    xs = np.arange(4)
    axR.bar(xs, vals, width=0.58, color=[BLUE_F, BLUE_F, GREEN_F, "#ECEFF1"],
            edgecolor=cols, lw=1.8)
    for x, v, c in zip(xs, vals, cols):
        axR.text(x, v + 0.28, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=11, color=c, fontweight="bold")
    axR.axhline(true_mean, color=MUTED, lw=1.3, ls=(0, (5, 3)), zorder=0)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, fontsize=10, color=INK)
    axR.set_ylim(0, 12.4)
    axR.set_ylabel("추정된 평균 생존시간", fontsize=11, color=INK)
    axR.set_title("절단을 잘못 다루면 절반으로 내려앉는다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.4)
    save(fig, "censoring_min_and_bias.png")
    return dict(true=true_mean, naive=naive_event, drop=drop_cens,
                km=km_mean, rate=1 - d.mean())


# === 그림 2. 세 절단 유형과 절단(truncation) ===
def fig_censoring_vs_truncation():
    rng = np.random.default_rng(3)
    n = 200_000
    T = rng.exponential(20.0, n)
    a = 10.0
    obs = T[T > a]
    print(f"all mean={T.mean():.3f} observed mean={obs.mean():.3f} "
          f"excluded={(T <= a).mean():.3f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.4, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]})

    # --- 왼쪽: 세 절단 유형이 남기는 정보 ---
    rows = [
        ("우측 중도절단", 6.0, 13.0, BLUE, BLUE_F, r"$T>6$"),
        ("좌측 중도절단", 0.0, 6.0, GREEN, GREEN_F, r"$T<6$"),
        ("구간 중도절단", 4.0, 8.0, ORANGE, ORANGE_F, r"$4<T\leq 8$"),
    ]
    ys = [3, 2, 1]
    for y, (name, lo, hi, col, fill, lab) in zip(ys, rows):
        axL.plot([0, 13.0], [y, y], color="#CFD8DC", lw=1.1, zorder=0)
        axL.add_patch(plt.Rectangle((lo, y - 0.22), hi - lo, 0.44,
                                    facecolor=fill, edgecolor=col, lw=1.6, zorder=2))
        axL.text((lo + hi) / 2, y, lab, ha="center", va="center",
                 fontsize=11, color=col, zorder=3)
        axL.text(-0.5, y, name, ha="right", va="center", fontsize=10.5, color=INK)

    # 좌측 절단(truncation): 대상이 표본에 아예 없다
    y = 0
    axL.plot([0, 13.0], [y, y], color="#CFD8DC", lw=1.1, zorder=0)
    axL.add_patch(plt.Rectangle((0, y - 0.22), 4.0, 0.44,
                                facecolor="white", edgecolor=MUTED, lw=1.4,
                                ls=(0, (3, 2)), zorder=2))
    axL.text(2.0, y, "표본에 없음", ha="center", va="center",
             fontsize=10, color=MUTED, zorder=3)
    axL.plot([4.0, 13.0], [y, y], color=PURPLE, lw=2.6, zorder=2)
    axL.plot(4.0, y, marker=">", color=PURPLE, ms=8, zorder=3)
    axL.text(8.5, y + 0.3, r"$T>4$ 인 대상만 등록된다", ha="center", va="bottom",
             fontsize=10, color=PURPLE)
    axL.text(-0.5, y, "좌측 절단(truncation)", ha="right", va="center",
             fontsize=10.5, color=INK)

    axL.set_xlim(-6.2, 13.6)
    axL.set_ylim(-0.75, 3.75)
    axL.set_yticks([])
    axL.set_xticks([0, 4, 6, 8, 12])
    axL.set_xlabel("시간", fontsize=11, color=INK)
    axL.set_title(r"음영은 참 $T$ 가 있을 수 있는 구간이다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axL)
    axL.spines["left"].set_visible(False)

    # --- 오른쪽: 절단(truncation) 은 표본 자체를 바꾼다 ---
    grid = np.linspace(0, 80, 400)
    dens = np.exp(-grid / 20.0) / 20.0
    axR.fill_between(grid, dens, color=BLUE_F, alpha=0.85, zorder=1)
    axR.plot(grid, dens, color=BLUE, lw=1.8, zorder=2)
    mask = grid <= a
    axR.fill_between(grid[mask], dens[mask], color=RED, alpha=0.30, zorder=3,
                     hatch="///", edgecolor=RED, lw=0.0)
    axR.axvline(a, color=RED, lw=1.5, ls=(0, (5, 3)), zorder=4)

    axR.text(5.0, 0.0255, "표본에\n들어오지\n못한 39.2%",
             fontsize=10, color=RED, ha="center", va="center", linespacing=1.45)
    axR.plot([T.mean()] * 2, [0, 0.0255], color=MUTED, lw=1.6, zorder=5)
    axR.plot(T.mean(), 0.0255, marker="^", color=MUTED, ms=8, zorder=5)
    axR.text(T.mean(), 0.0275, f"전체 평균\n{T.mean():.1f}", ha="center",
             va="bottom", fontsize=9.5, color=INK)
    axR.plot([obs.mean()] * 2, [0, 0.0145], color=PURPLE, lw=1.6, zorder=5)
    axR.plot(obs.mean(), 0.0145, marker="^", color=PURPLE, ms=8, zorder=5)
    axR.text(obs.mean() + 1.2, 0.0165, f"관측 표본 평균\n{obs.mean():.1f}", ha="left",
             va="bottom", fontsize=9.5, color=PURPLE)

    axR.set_xlim(0, 72)
    axR.set_ylim(0, 0.053)
    axR.set_yticks([])
    axR.set_xlabel("수명 (참 사건시간)", fontsize=11, color=INK)
    axR.set_title("절단(truncation)은 정보가 아니라 표본을 잃는다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axR)
    axR.spines["left"].set_visible(False)

    fig.tight_layout(w_pad=2.6)
    save(fig, "censoring_vs_truncation.png")
    return dict(all_mean=T.mean(), obs_mean=obs.mean(),
                excluded=(T <= a).mean())


# === 그림 3. 같은 중앙값, 전혀 다른 위험 모양 ===
def fig_hazard_shapes_same_median():
    tg = np.linspace(1e-4, 24.0, 2400)
    med = 10.0
    ln2 = np.log(2.0)

    def scale_to_median(h_raw):
        """H(10) = ln 2 가 되도록 위험함수를 상수배 한다."""
        H_raw = np.concatenate([[0.0], np.cumsum(np.diff(tg) * h_raw[:-1])])
        H_at_med = np.interp(med, tg, H_raw)
        c = ln2 / H_at_med
        h = c * h_raw
        H = c * H_raw
        return h, H

    shapes = []
    # 상수 위험
    shapes.append(("상수", BLUE, np.ones_like(tg)))
    # 증가 위험 (와이불 k = 2.5)
    shapes.append(("증가", ORANGE, tg ** 1.5))
    # 감소 위험 (와이불 k = 0.55)
    shapes.append(("감소", GREEN, tg ** (-0.45)))
    # 욕조 곡선
    shapes.append(("욕조", PURPLE, 0.9 * np.exp(-tg / 1.1) + 0.10 + 0.0022 * tg ** 2))
    # 봉우리형
    shapes.append(("봉우리", RED, tg * np.exp(-tg / 3.0) + 0.02))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7))
    axH, axS = axes

    at4, at10 = [], []
    for name, col, raw in shapes:
        h, H = scale_to_median(raw)
        S = np.exp(-H)
        axH.plot(tg, h, color=col, lw=2.2, label=name)
        axS.plot(tg, S, color=col, lw=2.2, label=name)
        at4.append((name, float(np.interp(4.0, tg, h)), float(np.interp(4.0, tg, S))))
        at10.append((name, float(np.interp(10.0, tg, S))))

    for name, h4, S4 in at4:
        print(f"{name}: h(4)={h4:.4f}  S(4)={S4:.4f}")
    for name, S10 in at10:
        print(f"{name}: S(10)={S10:.4f}")

    axH.set_xlim(0, 24)
    axH.set_ylim(0, 0.46)
    axH.set_xlabel("시간 $t$", fontsize=11, color=INK)
    axH.set_ylabel(r"위험함수 $h(t)$", fontsize=11, color=INK)
    axH.set_title("위험함수로 보면 다섯이 전혀 다르다",
                  fontsize=12.5, color=INK, pad=10)
    axH.legend(fontsize=10, frameon=False, ncol=2, loc="upper left",
               bbox_to_anchor=(0.02, 1.0))
    clean_axis(axH)

    axS.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    axS.axvline(10.0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    axS.plot(10.0, 0.5, "o", color=INK, ms=7, zorder=5)
    axS.annotate("다섯 곡선이 모두\n이 한 점을 지난다\n(중앙 생존시간 10)",
                 xy=(10.0, 0.5), xytext=(13.0, 0.72), fontsize=10, color=INK,
                 ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2,
                                 shrinkA=2, shrinkB=4))
    axS.set_xlim(0, 24)
    axS.set_ylim(0, 1.02)
    axS.set_xlabel("시간 $t$", fontsize=11, color=INK)
    axS.set_ylabel(r"생존함수 $S(t)$", fontsize=11, color=INK)
    axS.set_title("중앙 생존시간은 다섯이 모두 같다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axS)

    fig.tight_layout(w_pad=2.6)
    save(fig, "hazard_shapes_same_median.png")
    return dict(at4=at4)


# === 그림 4. 신용 부도의 봉우리형 위험 ===
def fig_credit_default_hazard():
    S_pts = {0: 1.00, 12: 0.97, 24: 0.93, 36: 0.90}
    edges = [0, 12, 24, 36]
    haz = []
    for a, b in zip(edges[:-1], edges[1:]):
        haz.append(-np.log(S_pts[b] / S_pts[a]) / ((b - a) / 12.0))
    cond = [1 - S_pts[b] / S_pts[a] for a, b in zip(edges[:-1], edges[1:])]
    print("interval hazard (per year):", [f"{h:.4f}" for h in haz])
    print("conditional default prob  :", [f"{c:.4f}" for c in cond])

    # 로지스틱 회귀의 편향 (본문 연습문제와 같은 설정)
    rng = np.random.default_rng(11)
    n = 200_000
    lam = -np.log(0.90) / 12
    T = rng.exponential(1 / lam, n)
    followup = np.where(rng.random(n) < 0.5, 12.0, 6.0)
    true_p12 = 1 - np.exp(-lam * 12)
    naive = (T <= followup).astype(int).mean()
    complete = (T[followup == 12] <= 12).mean()
    print(f"true={true_p12:.4f} naive={naive:.4f} complete={complete:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.3))
    axS, axH, axB = axes

    # --- (A) 생존곡선: 거의 평평해 보인다 ---
    tt = np.linspace(0, 36, 400)
    SS = np.empty_like(tt)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        m = (tt >= a) & (tt <= b)
        SS[m] = S_pts[a] * np.exp(-haz[k] * (tt[m] - a) / 12.0)
    axS.fill_between(tt, SS, 0.86, color=BLUE_F, alpha=0.8, zorder=1)
    axS.plot(tt, SS, color=BLUE, lw=2.4, zorder=2)
    for m in (12, 24, 36):
        axS.plot(m, S_pts[m], "o", color=BLUE, ms=7, mec="white", mew=1.2, zorder=3)
        axS.text(m, S_pts[m] + 0.004, f"{S_pts[m]:.2f}", ha="center", va="bottom",
                 fontsize=10.5, color=BLUE)
    axS.set_xlim(0, 38)
    axS.set_ylim(0.86, 1.008)
    axS.set_xticks([0, 12, 24, 36])
    axS.set_xlabel("실행 후 경과 개월", fontsize=11, color=INK)
    axS.set_ylabel(r"$\hat S(t)$", fontsize=11, color=INK)
    axS.set_title("생존곡선은 완만한 내림처럼 보인다", fontsize=12, color=INK, pad=10)
    clean_axis(axS)

    # --- (B) 구간별 위험: 봉우리가 드러난다 ---
    centers = [6, 18, 30]
    axH.bar(centers, haz, width=10.5, color=ORANGE_F, edgecolor=ORANGE, lw=1.8)
    for c, h in zip(centers, haz):
        axH.text(c, h + 0.0012, f"{h:.4f}", ha="center", va="bottom",
                 fontsize=10.5, color=ORANGE, fontweight="bold")
    axH.plot(centers, haz, color=RED, lw=1.8, marker="o", ms=6, zorder=5)
    axH.set_xlim(0, 36)
    axH.set_ylim(0, 0.052)
    axH.set_xticks([0, 12, 24, 36])
    axH.set_xlabel("실행 후 경과 개월", fontsize=11, color=INK)
    axH.set_ylabel("구간 평균 위험 (연 단위)", fontsize=11, color=INK)
    axH.set_title("같은 세 수가 봉우리를 그린다", fontsize=12, color=INK, pad=10)
    clean_axis(axH)

    # --- (C) 로지스틱 회귀의 편향 ---
    labels = ["절단을\n부도 없음으로", "12개월\n완전 관측만", "참값"]
    vals = [naive, complete, true_p12]
    edge = [RED, GREEN, MUTED]
    face = [BLUE_F, GREEN_F, "#ECEFF1"]
    xs = np.arange(3)
    axB.bar(xs, vals, width=0.58, color=face, edgecolor=edge, lw=1.8)
    for x, v, c in zip(xs, vals, edge):
        axB.text(x, v + 0.0025, f"{v:.4f}", ha="center", va="bottom",
                 fontsize=10.5, color=c, fontweight="bold")
    axB.axhline(true_p12, color=MUTED, lw=1.3, ls=(0, (5, 3)), zorder=0)
    axB.set_xticks(xs)
    axB.set_xticklabels(labels, fontsize=10, color=INK)
    axB.set_ylim(0, 0.128)
    axB.set_ylabel("추정된 12개월 부도확률", fontsize=11, color=INK)
    axB.set_title("절단을 0으로 묻으면 아래로 편향된다", fontsize=12, color=INK, pad=10)
    clean_axis(axB)

    fig.tight_layout(w_pad=2.8)
    save(fig, "credit_default_hazard.png")
    return dict(haz=haz, cond=cond, true=true_p12, naive=naive, complete=complete)


if __name__ == "__main__":
    fig_censoring_min_and_bias()
    fig_censoring_vs_truncation()
    fig_hazard_shapes_same_median()
    fig_credit_default_hazard()
