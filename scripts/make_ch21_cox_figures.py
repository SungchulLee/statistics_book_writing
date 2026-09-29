r"""21.4 콕스 비례위험 모형 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch21/cox/img/partial_likelihood_riskset.png  위험집합 안의 몫이 기저위험을 지운다
  ch21/cox/img/hr_not_time_ratio.png           같은 HR 이라도 중앙값은 다르게 줄어든다
  ch21/cox/img/loglog_parallel.png             평행이면 비례위험, 모이면 아니다
  ch21/cox/img/schoenfeld_residuals.png        척도화 잔차의 추세가 곧 beta(t)
  ch21/cox/img/cox_baseline_and_fit.png        브레슬로가 사후에 기저위험을 되찾는다

실행:  python3 scripts/make_ch21_cox_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
       콕스 적합·쇤펠트 잔차·일치도 지수는 numpy 로 직접 구현한다(lifelines 불필요).
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

OUT = "docs/ch21/cox/img/"
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


# === 비모수·준모수 도구 (numpy 만으로) ===
def km_table(t, d):
    t = np.asarray(t, float)
    d = np.asarray(d, int)
    ts = np.sort(t)
    ev = np.unique(t[d == 1])
    n_j = len(t) - np.searchsorted(ts, ev, side="left")
    es = np.sort(t[d == 1])
    d_j = (np.searchsorted(es, ev, side="right")
           - np.searchsorted(es, ev, side="left"))
    S = np.cumprod(1.0 - d_j / n_j)
    return ev, n_j, d_j, S


def cox_fit(X, t, d, tol=1e-10, maxit=60):
    """브레슬로 동점 처리를 쓰는 콕스 부분가능도의 뉴턴-랩슨 적합."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(t):
        X = X.T
    n, p = X.shape
    ev = np.unique(t[d == 1])
    beta = np.zeros(p)
    for _ in range(maxit):
        U = np.zeros(p)
        I = np.zeros((p, p))
        w = np.exp(X @ beta)
        for u in ev:
            R = t >= u
            Dj = (t == u) & (d == 1)
            wr = w[R]
            Sw = wr.sum()
            xbar = (X[R] * wr[:, None]).sum(0) / Sw
            dj = int(Dj.sum())
            U += X[Dj].sum(0) - dj * xbar
            Xc = X[R] - xbar
            I += dj * (Xc * wr[:, None]).T @ Xc / Sw
        step = np.linalg.solve(I, U)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    se = np.sqrt(np.diag(np.linalg.inv(I)))
    return beta, se, I


def schoenfeld(X, t, d, beta, I):
    """사건마다의 쇤펠트 잔차와 척도화 잔차, 그리고 GT 검정 통계량."""
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(t):
        X = X.T
    p = X.shape[1]
    w = np.exp(X @ beta)
    times, res, var = [], [], []
    for u in np.unique(t[d == 1]):
        R = t >= u
        wr = w[R]
        Sw = wr.sum()
        xbar = (X[R] * wr[:, None]).sum(0) / Sw
        Xc = X[R] - xbar
        Vj = (Xc * wr[:, None]).T @ Xc / Sw
        for i in np.where((t == u) & (d == 1))[0]:
            times.append(u)
            res.append(X[i] - xbar)
            var.append(np.diag(Vj))
    times = np.array(times)
    res = np.array(res)
    var = np.array(var)
    dtot = len(times)
    Iinv = np.linalg.inv(I)
    scaled = beta + dtot * (res @ Iinv)
    # Grambsch-Therneau: g(t) = t 에 대한 추세 검정
    g = times - times.mean()
    chi2 = np.empty(p)
    for k in range(p):
        num = (g * res[:, k]).sum() ** 2
        den = (g ** 2 * var[:, k]).sum()
        chi2[k] = num / den
    return times, scaled, chi2, stats.chi2.sf(chi2, 1)


def breslow_H0(X, t, d, beta):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(t):
        X = X.T
    w = np.exp(X @ beta)
    ev = np.unique(t[d == 1])
    inc = []
    for u in ev:
        dj = int(((t == u) & (d == 1)).sum())
        inc.append(dj / w[t >= u].sum())
    return ev, np.cumsum(inc)


def c_index(risk, t, d):
    """일치도 지수. risk 가 클수록 사건이 빨라야 일치다."""
    conc = perm = 0.0
    for i in np.where(d == 1)[0]:
        comp = t > t[i]
        conc += (risk[comp] < risk[i]).sum() + 0.5 * (risk[comp] == risk[i]).sum()
        perm += comp.sum()
    return conc / perm


def sample_from_H(t_grid, H_grid, n, rng):
    """누적위험 H 를 역변환해 사건시간을 뽑는다."""
    E = rng.exponential(1.0, n)
    return np.interp(E, H_grid, t_grid)


# === 그림 1. 위험집합 안의 몫 ===
def fig_partial_likelihood_riskset():
    t = np.array([2, 3, 5, 6, 8, 9, 11, 13], float)
    d = np.array([1, 0, 1, 1, 0, 1, 0, 1])
    x = np.array([1, 0, 1, 0, 1, 0, 1, 0], float)
    beta = 0.8
    u = 5.0
    R = np.where(t >= u)[0]
    w = np.exp(beta * x)
    share = w[2] / w[R].sum()
    print(f"위험집합 크기 {len(R)},  분모 {w[R].sum():.4f},  기여 {share:.4f},  "
          f"beta=0 이면 {1/len(R):.4f}")

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.6, 5.0), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ys = np.arange(8)[::-1]
    axL.add_patch(plt.Rectangle((u - 0.35, -0.6), 14.4 - u + 0.35, 6.2,
                                facecolor=BLUE_F, edgecolor=BLUE, lw=1.4,
                                ls=(0, (4, 3)), zorder=0))
    axL.text(9.6, 5.75, r"$t=5$ 의 위험집합  $\mathcal{R}_j$", fontsize=11,
             color=BLUE, ha="center", va="bottom")
    for y, ti, di, xi in zip(ys, t, d, x):
        c = ORANGE if xi == 1 else MUTED
        axL.plot([0, ti], [y, y], color=INK, lw=2.2, zorder=3)
        if di == 1:
            axL.plot(ti, y, "o", color=RED, ms=9, mec="white", mew=1.2, zorder=4)
        else:
            axL.plot([ti, ti], [y - 0.26, y + 0.26], color=BLUE, lw=3.0, zorder=4)
        axL.text(-0.6, y, f"대상 {8 - y},  $x={int(xi)}$", ha="right", va="center",
                 fontsize=10, color=c)
    axL.axvline(u, color=BLUE, lw=1.6, zorder=1)
    axL.plot(u, 5, "o", color=RED, ms=14, mfc="none", mew=2.2, zorder=6)
    axL.annotate("이 사람이 사건을 겪었다", xy=(u, 5), xytext=(6.6, 6.55),
                 fontsize=10.5, color=RED, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    axL.set_xlim(-6.2, 14.4)
    axL.set_ylim(-1.0, 7.3)
    axL.set_yticks([])
    axL.set_xticks([0, 2, 5, 6, 9, 13])
    axL.set_xlabel("시간", fontsize=11, color=INK)
    axL.set_title("부분가능도는 한 사건시점만 본다", fontsize=12.5, color=INK, pad=10)
    clean_axis(axL)
    axL.spines["left"].set_visible(False)

    labels = [f"대상 {i+1}\n$x={int(x[i])}$" for i in R]
    vals = w[R]
    cols = [RED if i == 2 else MUTED for i in R]
    faces = [ORANGE_F if i == 2 else "#ECEFF1" for i in R]
    xs = np.arange(len(R))
    axR.bar(xs, vals, width=0.62, color=faces, edgecolor=cols, lw=1.8)
    for xi, v, c in zip(xs, vals, cols):
        axR.text(xi, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=10,
                 color=c)
    axR.set_xticks(xs)
    axR.set_xticklabels(labels, fontsize=9.5, color=INK, linespacing=1.3)
    axR.set_ylim(0, 4.1)
    axR.set_ylabel(r"$\exp(\beta x_l)$   ($\beta = 0.8$)", fontsize=11, color=INK)
    axR.set_title("기여 = 주황 막대 ÷ 여섯 막대의 합", fontsize=12.5, color=INK,
                  pad=10)
    axR.text(2.5, 4.0,
             f"$\\dfrac{{{w[2]:.3f}}}{{{w[R].sum():.3f}}} = {share:.3f}$",
             fontsize=14, color=INK, ha="center", va="top")
    axR.text(2.5, 3.05,
             r"막대 전체에 $h_0(5)$ 를 곱해도" "\n" "몫은 그대로다",
             fontsize=10.5, color=BLUE, ha="center", va="top", linespacing=1.45)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "partial_likelihood_riskset.png")


# === 그림 2. 같은 HR, 다른 중앙값 변화 ===
def fig_hr_not_time_ratio():
    med0 = 40.0
    HR = 2.0
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    tg = np.linspace(0, 90, 900)
    for k, col, name in [(1.0, BLUE, r"지수 기저 ($k=1$)"),
                         (2.0, ORANGE, r"와이불 기저 ($k=2$)")]:
        sc = med0 / np.log(2) ** (1 / k)
        S0 = np.exp(-(tg / sc) ** k)
        S1 = S0 ** HR
        med1 = med0 * HR ** (-1 / k)
        print(f"k={k}: 중앙값 {med0:.1f} -> {med1:.2f}  (비 {HR**(-1/k):.3f})")
        axL.plot(tg, S0, color=col, lw=2.4, label=f"{name} · 대조군")
        axL.plot(tg, S1, color=col, lw=2.4, ls=(0, (5, 3)),
                 label=f"{name} · $\\mathrm{{HR}}=2$")
        axL.plot(med1, 0.5, "o", color=col, ms=8, mec="white", mew=1.2, zorder=5)
        axL.text(med1 - 1.2, 0.44, f"{med1:.1f}", fontsize=10.5, color=col,
                 ha="right", va="top")
    axL.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    axL.plot(med0, 0.5, "o", color=INK, ms=8, zorder=5)
    axL.text(42.5, 0.60, "대조군 중앙값 40", fontsize=10.5, color=INK,
             ha="left", va="bottom")
    axL.axvline(20.0, color=RED, lw=1.5, ls=(0, (3, 2)), zorder=0,
                label="\"절반이 된다\" 는 오해:  $t = 20$")
    axL.set_xlim(0, 90)
    axL.set_ylim(0, 1.02)
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"$S(t)$", fontsize=11, color=INK)
    axL.set_title(r"같은 $\mathrm{HR}=2$ 인데 중앙값은 다르게 줄어든다",
                  fontsize=12, color=INK, pad=10)
    axL.legend(fontsize=9.5, frameon=True, framealpha=0.94, edgecolor=MUTED,
               loc="upper right")
    clean_axis(axL)

    kk = np.linspace(0.35, 4.0, 400)
    ratio = HR ** (-1 / kk)
    axR.plot(kk, ratio, color=PURPLE, lw=2.8)
    axR.axhline(0.5, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    for k, col in [(0.6, GREEN), (1.0, BLUE), (2.0, ORANGE), (3.0, MUTED)]:
        r = HR ** (-1 / k)
        axR.plot(k, r, "o", color=col, ms=9, mec="white", mew=1.2, zorder=5)
        axR.text(k - 0.06, r + 0.025, f"{r:.3f}", fontsize=10.5, color=col,
                 ha="right", va="bottom")
        print(f"k={k}: 중앙값 비 {r:.4f}")
    axR.text(2.55, 0.34, "위험이 빠르게 커지는 과정일수록\n"
                         "같은 위험비가 시간을 덜 깎는다",
             fontsize=10.5, color=INK, ha="center", va="top", linespacing=1.45)
    axR.set_xlim(0.35, 4.0)
    axR.set_ylim(0.15, 0.95)
    axR.set_xlabel(r"기저 와이불의 형상모수 $k$", fontsize=11, color=INK)
    axR.set_ylabel("중앙 생존시간의 비", fontsize=11, color=INK)
    axR.set_title(r"중앙값의 비는 $2^{-1/k}$ 로 $k$ 에 달려 있다",
                  fontsize=12, color=INK, pad=10)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "hr_not_time_ratio.png")


# === 두 개의 모의자료 (비례위험 성립 / 위배) ===
def make_two_datasets(seed=99, n=450, tau=40.0):
    rng = np.random.default_rng(seed)
    tg = np.linspace(1e-6, 260.0, 40000)
    k0, sc0 = 1.5, 34.0
    h0 = (k0 / sc0) * (tg / sc0) ** (k0 - 1)
    H0 = np.concatenate([[0.0], np.cumsum(np.diff(tg) * h0[:-1])])

    out = {}
    # (a) 비례위험 성립: beta = -0.7 로 일정
    b_const = -0.7
    T_c = sample_from_H(tg, H0, n, rng)
    T_t = sample_from_H(tg, H0 * np.exp(b_const), n, rng)
    # (b) 비례위험 위배: beta(t) = -1.6 + 0.11 t
    bt = -1.6 + 0.11 * tg
    Hv = np.concatenate([[0.0], np.cumsum(np.diff(tg) * (h0 * np.exp(bt))[:-1])])
    T_v = sample_from_H(tg, Hv, n, rng)

    for name, Ttr in (("ph", T_t), ("nph", T_v)):
        T = np.concatenate([T_c, Ttr])
        z = np.concatenate([np.zeros(n), np.ones(n)])
        C = np.minimum(rng.uniform(4.0, 90.0, 2 * n), tau)
        t = np.minimum(T, C)
        d = (T <= C).astype(int)
        out[name] = (t, d, z)
    out["truth"] = dict(b_const=b_const, k0=k0, sc0=sc0, tg=tg, H0=H0)
    return out


def loglog_curve(t, d, z, g, lo=0.03, hi=0.96):
    """사건이 충분히 쌓인 구간만 남긴다. 양 끝은 변동이 커서 평행 여부를 판단할 수 없다."""
    ev, n_j, d_j, S = km_table(t[z == g], d[z == g])
    m = (S > lo) & (S < hi)
    return np.log(ev[m]), np.log(-np.log(S[m]))


# === 그림 3. 로그-로그 그림의 평행성 ===
def fig_loglog_parallel():
    data = make_two_datasets()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)

    titles = ["비례위험이 성립하는 자료", "처리 효과가 사라지는 자료"]
    for ax, key, title in zip(axes, ("ph", "nph"), titles):
        t, d, z = data[key]
        beta, se, I = cox_fit(z, t, d)
        times, scaled, chi2, pv = schoenfeld(z, t, d, beta, I)
        print(f"[{key}] beta={beta[0]:+.4f} (se {se[0]:.4f})  "
              f"HR={np.exp(beta[0]):.3f}  쇤펠트 chi2={chi2[0]:.2f} p={pv[0]:.4f}")
        for g, col, nm in ((0, BLUE, "대조군"), (1, ORANGE, "처리군")):
            lx, ly = loglog_curve(t, d, z, g)
            ax.plot(lx, ly, color=col, lw=2.4, label=nm)
        lx0, ly0 = loglog_curve(t, d, z, 0)
        lx1, ly1 = loglog_curve(t, d, z, 1)
        qlo = max(lx0.min(), lx1.min()) + 0.12
        qhi = min(lx0.max(), lx1.max()) - 0.12
        gaps = []
        for q in np.linspace(qlo, qhi, 3):
            a = np.interp(q, lx0, ly0)
            b = np.interp(q, lx1, ly1)
            gaps.append(b - a)
            ax.annotate("", xy=(q, a), xytext=(q, b),
                        arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
            ax.text(q + 0.07, (a + b) / 2, f"{b - a:+.2f}", fontsize=10.5,
                    color=RED, ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=1.5))
        print(f"    간격 {['%+.2f' % g for g in gaps]}  "
              f"(ln t = {qlo:.2f} ~ {qhi:.2f})")
        ax.text(1.42, 0.78,
                f"쇤펠트 검정:  $\\chi^2$ = {chi2[0]:.2f},   $p$ = {pv[0]:.3f}",
                fontsize=10.5, color=INK, ha="left", va="top")
        ax.set_xlim(1.35, 4.02)
        ax.set_ylim(-4.2, 1.05)
        ax.set_xlabel(r"$\ln t$", fontsize=11, color=INK)
        ax.set_title(title, fontsize=12.5, color=INK, pad=10)
        ax.legend(fontsize=10.5, frameon=False, loc="lower right")
        clean_axis(ax)
    axes[0].set_ylabel(r"$\ln(-\ln \hat S(t))$", fontsize=11, color=INK)
    axes[0].text(1.42, 0.38, "간격이 일정하다 — 평행",
                 fontsize=10.5, color=GREEN, ha="left", va="top")
    axes[1].text(1.42, 0.38, "간격이 줄어들다 부호가 뒤집힌다",
                 fontsize=10.5, color=RED, ha="left", va="top")

    fig.tight_layout(w_pad=2.4)
    save(fig, "loglog_parallel.png")


# === 그림 4. 척도화 쇤펠트 잔차 ===
def fig_schoenfeld_residuals():
    data = make_two_datasets()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)

    titles = ["비례위험이 성립하는 자료", "처리 효과가 사라지는 자료"]
    fits = {}
    for key in ("ph", "nph"):
        t, d, z = data[key]
        beta, se, I = cox_fit(z, t, d)
        fits[key] = (beta, se, I) + schoenfeld(z, t, d, beta, I)
    ylo = min(f[4][:, 0].min() for f in fits.values()) - 0.6
    yhi = max(f[4][:, 0].max() for f in fits.values()) + 2.2
    print(f"쇤펠트 y 범위 {ylo:.2f} ~ {yhi:.2f}")

    for ax, key, title in zip(axes, ("ph", "nph"), titles):
        t, d, z = data[key]
        beta, se, I, times, scaled, chi2, pv = fits[key]
        y = scaled[:, 0]
        ax.plot(times, y, "o", color=MUTED, ms=3.4, alpha=0.55, zorder=2)
        # 이동평균 평활
        order = np.argsort(times)
        ts, ys = times[order], y[order]
        win = max(9, len(ts) // 9)
        ker = np.ones(win) / win
        sm = np.convolve(ys, ker, mode="valid")
        sx = np.convolve(ts, ker, mode="valid")
        ax.plot(sx, sm, color=PURPLE, lw=2.8, zorder=4, label="이동평균")
        ax.axhline(beta[0], color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3,
                   label=f"$\\hat\\beta$ = {beta[0]:+.2f}")
        ax.axhline(0.0, color=MUTED, lw=1.1, zorder=1)
        print(f"[{key}] 척도화 잔차 평활 범위 {sm.min():+.2f} ~ {sm.max():+.2f}")
        ax.text(1.0, yhi - 0.25,
                f"쇤펠트 검정:  $\\chi^2$ = {chi2[0]:.2f},   $p$ = {pv[0]:.3f}",
                fontsize=10.5, color=INK, ha="left", va="top")
        ax.set_xlim(0, 40)
        ax.set_ylim(ylo, yhi)
        ax.set_xlabel("시간 (개월)", fontsize=11, color=INK)
        ax.set_title(title, fontsize=12.5, color=INK, pad=10)
        ax.legend(fontsize=10, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.55, 0.88), ncol=2, columnspacing=1.8)
        clean_axis(ax)
    axes[0].set_ylabel(r"척도화 쇤펠트 잔차 (그 시점의 $\beta$)", fontsize=11,
                       color=INK)
    axes[0].text(1.0, yhi - 1.35, "보라색이 수평이다 —\n효과가 시간에 따라 변하지 않는다",
                 fontsize=10.5, color=GREEN, ha="left", va="top", linespacing=1.45)
    axes[1].text(1.0, yhi - 1.35, "보라색이 올라간다 —\n보호 효과가 사라지고 뒤집힌다",
                 fontsize=10.5, color=RED, ha="left", va="top", linespacing=1.45)

    fig.tight_layout(w_pad=2.4)
    save(fig, "schoenfeld_residuals.png")


# === 그림 5. 브레슬로 기저위험과 예측 생존곡선 ===
def fig_cox_baseline_and_fit():
    # 기저위험의 회수를 보이는 그림이라 앞의 두 그림보다 표본을 크게 잡는다.
    data = make_two_datasets(seed=11, n=1000)
    t, d, z = data["ph"]
    tr = data["truth"]
    beta, se, I = cox_fit(z, t, d)
    lo, hi = np.exp(beta[0] - 1.96 * se[0]), np.exp(beta[0] + 1.96 * se[0])
    ci = c_index(np.exp(beta[0] * z), t, d)
    print(f"beta={beta[0]:+.4f}  se={se[0]:.4f}  HR={np.exp(beta[0]):.3f} "
          f"({lo:.3f}, {hi:.3f})  C-index={ci:.4f}  참 HR={np.exp(tr['b_const']):.3f}")

    ev, H0hat = breslow_H0(z, t, d, beta)
    for q in (10.0, 20.0, 30.0, 40.0):
        hq = np.interp(q, np.concatenate([[0.0], ev]),
                       np.concatenate([[0.0], H0hat]))
        print(f"  t={q:.0f}:  참 H0={((q / tr['sc0']) ** tr['k0']):.4f}  "
              f"브레슬로={hq:.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    tg = np.linspace(0, 40, 400)
    H0true = (tg / tr["sc0"]) ** tr["k0"]
    axL.plot(tg, H0true, color=MUTED, lw=2.4, ls=(0, (5, 3)),
             label=r"참 $H_0(t)$ (와이불 $k=1.5$)")
    axL.plot(np.concatenate([[0.0], ev]), np.concatenate([[0.0], H0hat]),
             drawstyle="steps-post", color=PURPLE, lw=2.4,
             label=r"브레슬로 $\hat H_0(t)$")
    axL.set_xlim(0, 40)
    axL.set_ylim(0, 1.25)
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"기저 누적위험 $H_0(t)$", fontsize=11, color=INK)
    axL.set_title("부분가능도가 버린 것을 사후에 되찾는다",
                  fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10.5, frameon=False, loc="upper left")
    axL.text(38.5, 0.10, "부분가능도는 이 곡선을\n한 번도 쓰지 않았다",
             fontsize=10.5, color=PURPLE, ha="right", va="bottom",
             linespacing=1.45)
    clean_axis(axL)

    S0 = np.exp(-np.interp(tg, np.concatenate([[0.0], ev]),
                           np.concatenate([[0.0], H0hat])))
    for g, col, nm in ((0, BLUE, "대조군"), (1, ORANGE, "처리군")):
        ev_g, n_g, d_g, S_g = km_table(t[z == g], d[z == g])
        x = np.concatenate([[0.0], ev_g])
        y = np.concatenate([[1.0], S_g])
        axR.plot(x, y, drawstyle="steps-post", color=col, lw=1.8, alpha=0.55,
                 label=f"{nm} 카플란-마이어")
        axR.plot(tg, S0 ** np.exp(beta[0] * g), color=col, lw=2.8,
                 label=f"{nm} 콕스 예측")
    axR.set_xlim(0, 40)
    axR.set_ylim(0, 1.02)
    axR.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axR.set_ylabel(r"$\hat S(t \mid x)$", fontsize=11, color=INK)
    axR.set_title(r"$\hat S_0(t)^{\exp(\hat\beta x)}$ 가 계단을 따라간다",
                  fontsize=12.5, color=INK, pad=10)
    axR.legend(fontsize=9.5, frameon=False, loc="lower left")
    axR.text(39, 0.93,
             f"$\\mathrm{{HR}}$ = {np.exp(beta[0]):.3f}  "
             f"({lo:.3f}, {hi:.3f})\n일치도 지수 = {ci:.3f}",
             fontsize=10.5, color=INK, ha="right", va="top", linespacing=1.5)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "cox_baseline_and_fit.png")


if __name__ == "__main__":
    fig_partial_likelihood_riskset()
    fig_hr_not_time_ratio()
    fig_loglog_parallel()
    fig_schoenfeld_residuals()
    fig_cox_baseline_and_fit()
