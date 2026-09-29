r"""21.3 모수적 생존 모형 다섯 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch21/parametric/img/memoryless_vs_aging.png       무기억성은 조건부 곡선이 겹친다는 뜻
  ch21/parametric/img/weibull_shape_slope.png       로그-로그 그림의 기울기가 k 다
  ch21/parametric/img/hump_hazard_tails.png         봉우리는 닮았는데 꼬리가 갈린다
  ch21/parametric/img/censored_likelihood.png       절단 관측은 봉우리를 만들지 못한다
  ch21/parametric/img/three_models_fit.png          생존곡선은 닮아도 위험함수는 다르다

실행:  python3 scripts/make_ch21_parametric_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats, optimize

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

OUT = "docs/ch21/parametric/img/"
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
    H = np.cumsum(d_j / n_j)
    return ev, n_j, d_j, S, H


# === 그림 1. 무기억성 대 노화 ===
def fig_memoryless_vs_aging():
    lam = 0.02
    med = np.log(2) / lam                 # 34.66
    k_w, sc_w = 2.0, med / np.log(2) ** 0.5

    s = np.linspace(0, 70, 700)
    anchors = [0.0, 20.0, 40.0]
    cols = [BLUE, ORANGE, GREEN]
    # 겹쳐 그려도 셋이 다 보이도록 대시 위상을 어긋나게 둔다.
    styles = ["-", (0, (5, 5)), (5, (5, 5))]
    lws = [3.4, 2.2, 2.2]
    s_mark = 12.0

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)

    print(f"지수: 중앙값 {med:.2f},  와이불 척도 {sc_w:.2f}")
    exp_vals, wb_vals = [], []
    for a, c, st, lw in zip(anchors, cols, styles, lws):
        cond_e = np.exp(-lam * (a + s)) / np.exp(-lam * a)
        axL.plot(s, cond_e, color=c, lw=lw, ls=st,
                 label=f"이미 {int(a)}개월 생존" if a else "새 대상 ($t=0$)")
        v = np.exp(-lam * s_mark)
        exp_vals.append(v)

        Sw = np.exp(-((a + s) / sc_w) ** k_w)
        cond_w = Sw / np.exp(-(a / sc_w) ** k_w)
        axR.plot(s, cond_w, color=c, lw=lw, ls=st,
                 label=f"이미 {int(a)}개월 생존" if a else "새 대상 ($t=0$)")
        vw = (np.exp(-((a + s_mark) / sc_w) ** k_w)
              / np.exp(-(a / sc_w) ** k_w))
        wb_vals.append(vw)
        axR.plot(s_mark, vw, "o", color=c, ms=8, mec="white", mew=1.2, zorder=5)
    print("지수 조건부 12개월 생존:", [f"{v:.4f}" for v in exp_vals])
    print("와이불 조건부 12개월 생존:", [f"{v:.4f}" for v in wb_vals])

    for ax, txt in ((axL, f"세 곡선이 완전히 겹친다\n" + r"$P(T>t+12\mid T>t)$" +
                     f" = {exp_vals[0]:.3f} 로 언제나 같다"),
                    (axR, "세 곡선이 아래로 내려앉는다\n오래 버틸수록 다음 12개월이 위험해진다")):
        ax.axvline(s_mark, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=0)
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("앞으로 더 살 기간 $s$ (개월)", fontsize=11, color=INK)
        clean_axis(ax)
        ax.text(68, 0.90, txt, fontsize=10.5, color=INK, ha="right", va="top",
                linespacing=1.45)
    axL.plot(s_mark, exp_vals[0], "o", color=INK, ms=8, mec="white", mew=1.2,
             zorder=6)
    axL.set_ylabel(r"조건부 생존확률 $P(T>t+s \mid T>t)$", fontsize=11, color=INK)
    axL.set_title("지수 모형 — 나이를 기억하지 않는다", fontsize=12.5, color=INK, pad=10)
    axR.set_title(r"와이불 $k=2$ — 나이를 기억한다", fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10, frameon=False, loc="lower left")
    axR.legend(fontsize=10, frameon=False, loc="lower left")

    for ax, vals in ((axR, wb_vals),):
        for a, c, v in zip(anchors, cols, vals):
            ax.text(s_mark + 1.3, v, f"{v:.3f}", fontsize=10, color=c,
                    ha="left", va="center")

    fig.tight_layout(w_pad=2.4)
    save(fig, "memoryless_vs_aging.png")
    return dict(exp=exp_vals, wb=wb_vals)


# === 그림 2. 형상모수와 로그-로그 기울기 ===
def fig_weibull_shape_slope():
    med = 40.0
    ks = [0.6, 1.0, 1.8, 3.0]
    cols = [GREEN, BLUE, ORANGE, PURPLE]
    scales = [med / np.log(2) ** (1 / k) for k in ks]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    tg = np.linspace(0.6, 90, 900)
    for k, sc, c in zip(ks, scales, cols):
        h = (k / sc) * (tg / sc) ** (k - 1)
        axL.plot(tg, h, color=c, lw=2.4, label=f"$k = {k}$")
    axL.axvline(med, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    axL.text(med + 1.6, 0.052, "네 분포 모두\n중앙값 40", fontsize=10, color=INK,
             ha="left", va="top", linespacing=1.4)
    axL.set_xlim(0, 90)
    axL.set_ylim(0, 0.058)
    axL.set_xlabel("시간 $t$ (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"위험함수 $h(t)$", fontsize=11, color=INK)
    axL.set_title(r"형상모수 $k$ 하나가 방향을 정한다", fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10, frameon=False, loc="upper left", ncol=2)
    clean_axis(axL)

    # 오른쪽: 절단된 모의자료의 로그-로그 그림
    rng = np.random.default_rng(314)
    n = 500
    fits = {}
    for k, sc, c in zip(ks, scales, cols):
        T = sc * rng.weibull(k, n)
        C = rng.uniform(5.0, 170.0, n)
        t = np.minimum(T, C)
        d = (T <= C).astype(int)
        ev, n_j, d_j, S, H = km_table(t, d)
        m = (H > 0) & (ev > 2.0) & (n_j >= 25)
        x, y = np.log(ev[m]), np.log(H[m])
        axR.plot(x, y, color=c, lw=0, marker="o", ms=2.6, alpha=0.55)
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 30)
        axR.plot(xs, a * xs + b, color=c, lw=2.2,
                 label=f"$k = {k}$  →  기울기 {a:.2f}")
        fits[k] = a
    print("로그-로그 기울기:", {k: round(v, 3) for k, v in fits.items()})
    axR.set_xlabel(r"$\ln t$", fontsize=11, color=INK)
    axR.set_ylabel(r"$\ln \hat H(t)$", fontsize=11, color=INK)
    axR.set_title("기울기가 곧 형상모수의 추정치다", fontsize=12.5, color=INK, pad=10)
    axR.legend(fontsize=10, frameon=False, loc="lower right")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "weibull_shape_slope.png")
    return fits


# === 그림 3. 봉우리는 닮았는데 꼬리가 갈린다 ===
def fig_hump_hazard_tails():
    mu, sigma = 3.2, 0.8
    med = np.exp(mu)
    k_ll = np.pi / (sigma * np.sqrt(3))     # 로그정규와 초반이 맞도록
    lam_ll = med
    k_wb = 1.6
    sc_wb = med / np.log(2) ** (1 / k_wb)
    print(f"중앙값 {med:.2f},  로그로지스틱 k={k_ll:.3f}")

    def S_ln(t):
        return stats.norm.sf((np.log(t) - mu) / sigma)

    def h_ln(t):
        z = (np.log(t) - mu) / sigma
        return stats.norm.pdf(z) / (t * sigma * stats.norm.sf(z))

    def S_ll(t):
        return 1.0 / (1.0 + (t / lam_ll) ** k_ll)

    def h_ll(t):
        return ((k_ll / lam_ll) * (t / lam_ll) ** (k_ll - 1)
                / (1.0 + (t / lam_ll) ** k_ll))

    def h_wb(t):
        return (k_wb / sc_wb) * (t / sc_wb) ** (k_wb - 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    tg = np.linspace(0.5, 110, 1200)
    axL.plot(tg, h_ln(tg), color=BLUE, lw=2.6, label="로그정규")
    axL.plot(tg, h_ll(tg), color=ORANGE, lw=2.6, ls=(0, (6, 3)), label="로그로지스틱")
    axL.plot(tg, h_wb(tg), color=MUTED, lw=2.0, ls=(0, (2, 2)),
             label=r"와이불 $k=1.6$ (비교용)")
    p_ln = tg[np.argmax(h_ln(tg))]
    p_ll = tg[np.argmax(h_ll(tg))]
    print(f"위험 정점:  로그정규 {p_ln:.1f},  로그로지스틱 {p_ll:.1f}")
    axL.plot(p_ln, h_ln(p_ln), "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=5)
    axL.plot(p_ll, h_ll(p_ll), "o", color=ORANGE, ms=8, mec="white", mew=1.2, zorder=5)
    axL.axvline(med, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=0)
    axL.text(med + 1.8, 0.0062, f"중앙값 {med:.1f}", fontsize=10, color=INK,
             ha="left", va="top")
    axL.annotate(f"정점이 {p_ln:.0f} 개월과 {p_ll:.0f} 개월",
                 xy=((p_ln + p_ll) / 2, max(h_ln(p_ln), h_ll(p_ll))),
                 xytext=(62, 0.049), fontsize=10.5, color=INK, ha="left",
                 va="center",
                 arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axL.set_xlim(0, 110)
    axL.set_ylim(0, 0.056)
    axL.set_xlabel("시간 $t$ (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"위험함수 $h(t)$", fontsize=11, color=INK)
    axL.set_title("봉우리의 위치와 높이가 거의 같다",
                  fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10, frameon=False, loc="lower right")
    clean_axis(axL)

    tt = np.linspace(1.0, 360, 1400)
    ratio = S_ll(tt) / S_ln(tt)
    axR.fill_between(tt, 1.0, ratio, color=ORANGE_F, alpha=0.7, zorder=1)
    axR.plot(tt, ratio, color=ORANGE, lw=2.6, zorder=3)
    axR.axhline(1.0, color=MUTED, lw=1.3, zorder=0)
    for q in (60.0, 120.0, 240.0, 360.0):
        r = float(S_ll(q) / S_ln(q))
        print(f"t={q:.0f}:  S_LL={float(S_ll(q)):.5f}  S_LN={float(S_ln(q)):.5f}  "
              f"비 {r:.2f}")
        axR.plot(q, r, "o", color=ORANGE, ms=7, mec="white", mew=1.2, zorder=5)
        axR.text(q, r + 0.22, f"{r:.1f}배", fontsize=10.5, color=ORANGE,
                 ha="center", va="bottom")
    axR.text(30, 5.4, "관측 구간 안(왼쪽)에서는 둘이 같지만\n"
                      "외삽할수록 로그로지스틱이 훨씬 낙관적이다",
             fontsize=10.5, color=INK, ha="left", va="top", linespacing=1.45)
    axR.set_xlim(0, 375)
    axR.set_ylim(0, 6.2)
    axR.set_xlabel("시간 $t$ (개월)", fontsize=11, color=INK)
    axR.set_ylabel("로그로지스틱 $S(t)$ ÷ 로그정규 $S(t)$", fontsize=11, color=INK)
    axR.set_title("꼬리로 갈수록 두 모형의 답이 갈린다",
                  fontsize=12.5, color=INK, pad=10)
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "hump_hazard_tails.png")


# === 그림 4. 절단 관측은 봉우리를 만들지 못한다 ===
def weibull_negll(p, t, d):
    k, lam = np.exp(p)
    return -np.sum(d * (np.log(k) - k * np.log(lam) + (k - 1) * np.log(t))
                   - (t / lam) ** k)


def fig_censored_likelihood():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.2, 4.9))

    # --- 왼쪽: 관측치 하나의 가능도 기여 (지수 모형) ---
    t0 = 10.0
    lg = np.linspace(0.001, 0.45, 600)
    L_event = lg * np.exp(-lg * t0)
    L_cens = np.exp(-lg * t0)
    axL.plot(lg, L_event / L_event.max(), color=ORANGE, lw=2.8,
             label=r"사건 관측:  $f(t)=\lambda e^{-\lambda t}$")
    axL.plot(lg, L_cens, color=BLUE, lw=2.8, ls=(0, (6, 3)),
             label=r"절단 관측:  $S(t)=e^{-\lambda t}$")
    axL.fill_between(lg, L_event / L_event.max(), color=ORANGE_F, alpha=0.55,
                     zorder=0)
    axL.plot(1 / t0, 1.0, "o", color=ORANGE, ms=9, mec="white", mew=1.2, zorder=5)
    axL.annotate(r"$\hat\lambda = 1/t = 0.10$ 에 봉우리",
                 xy=(1 / t0, 1.0), xytext=(0.175, 0.94), fontsize=10.5,
                 color=ORANGE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
    axL.text(0.225, 0.20, "봉우리가 없다.\n작은 $\\lambda$ 를 선호할 뿐이다",
             fontsize=10.5, color=BLUE, ha="left", va="bottom", linespacing=1.45)
    axL.set_xlim(0, 0.45)
    axL.set_ylim(0, 1.12)
    axL.set_yticks([])
    axL.set_xlabel(r"위험률 $\lambda$", fontsize=11, color=INK)
    axL.set_ylabel("가능도 기여 (최댓값 1로 맞춤)", fontsize=11, color=INK)
    axL.set_title(r"$t=10$ 인 대상 한 명의 기여", fontsize=12.5, color=INK, pad=10)
    axL.legend(fontsize=10.5, frameon=False, loc="upper right")
    clean_axis(axL)
    axL.spines["left"].set_visible(False)

    # --- 오른쪽: 절단율에 따른 프로파일 가능도 ---
    rng = np.random.default_rng(77)
    n = 300
    k_true, sc_true = 1.7, 50.0
    T = sc_true * rng.weibull(k_true, n)
    U = rng.random(n)
    settings = [(400.0, GREEN), (52.0, BLUE), (14.0, RED)]
    kg = np.linspace(0.9, 3.4, 260)
    for cmax, col in settings:
        C = cmax * U
        t = np.minimum(T, C)
        d = (T <= C).astype(int)
        rate = 1 - d.mean()
        prof = []
        for k in kg:
            lam_k = (np.sum(t ** k) / max(d.sum(), 1)) ** (1 / k)
            prof.append(-weibull_negll([np.log(k), np.log(lam_k)], t, d))
        prof = np.array(prof)
        prof -= prof.max()
        khat = kg[prof.argmax()]
        inside = kg[prof >= -1.92]
        lo, hi = inside.min(), inside.max()
        print(f"절단율 {rate:5.1%}  사건 {int(d.sum()):3d}건  "
              f"k_hat={khat:.3f}  프로파일 95% CI=({lo:.2f}, {hi:.2f})  "
              f"폭 {hi - lo:.2f}")
        axR.plot(kg, prof, color=col, lw=2.5,
                 label=f"절단율 {rate*100:.0f}%  ·  사건 {int(d.sum())}건  ·  "
                       f"$k$ 구간 폭 {hi - lo:.2f}")
    axR.axhline(-1.92, color=MUTED, lw=1.3, ls=(0, (4, 3)), zorder=0)
    axR.text(0.95, -1.35, r"문턱 $-1.92$", fontsize=10, color=MUTED,
             ha="left", va="bottom")
    axR.axvline(k_true, color=INK, lw=1.2, ls=(0, (2, 2)), zorder=0)
    axR.text(k_true + 0.05, -9.3, f"참값 $k = {k_true}$", fontsize=10, color=INK,
             ha="left", va="bottom")
    axR.set_xlim(0.9, 3.4)
    axR.set_ylim(-10, 0.6)
    axR.set_xlabel(r"형상모수 $k$", fontsize=11, color=INK)
    axR.set_ylabel(r"프로파일 로그가능도 $\ell(k)-\ell(\hat k)$", fontsize=11,
                   color=INK)
    axR.set_title("절단이 심해지면 곡면이 평평해진다", fontsize=12.5, color=INK, pad=10)
    axR.legend(fontsize=10, frameon=False, loc="lower right")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "censored_likelihood.png")


# === 그림 5. 생존곡선은 닮아도 위험함수는 다르다 ===
def fig_three_models_fit():
    rng = np.random.default_rng(1234)
    n = 400
    mu_t, sg_t = 3.2, 0.8                  # 참값: 로그정규 (봉우리형 위험)
    T = np.exp(mu_t + sg_t * rng.standard_normal(n))
    C = rng.uniform(2.0, 90.0, n)
    t = np.minimum(T, C)
    d = (T <= C).astype(int)
    print(f"절단율 {1 - d.mean():.1%}, 사건 {int(d.sum())}건")

    # 지수
    lam_e = d.sum() / t.sum()
    ll_e = float(np.sum(d * np.log(lam_e) - lam_e * t))

    # 와이불
    r = optimize.minimize(weibull_negll, [0.0, np.log(t.mean())], args=(t, d),
                          method="Nelder-Mead",
                          options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000))
    k_w, lam_w = np.exp(r.x)
    ll_w = -r.fun

    # 로그정규
    def ln_negll(p):
        m, s = p[0], np.exp(p[1])
        z = (np.log(t) - m) / s
        return -np.sum(d * (stats.norm.logpdf(z) - np.log(s * t))
                       + (1 - d) * stats.norm.logsf(z))
    r2 = optimize.minimize(ln_negll, [np.log(t).mean(), np.log(np.log(t).std())],
                           method="Nelder-Mead",
                           options=dict(xatol=1e-8, fatol=1e-8, maxiter=4000))
    mu_h, sg_h = r2.x[0], np.exp(r2.x[1])
    ll_l = -r2.fun

    aic = {"지수": -2 * ll_e + 2 * 1, "와이불": -2 * ll_w + 2 * 2,
           "로그정규": -2 * ll_l + 2 * 2}
    print(f"지수    lam={lam_e:.5f}  ll={ll_e:.2f}  AIC={aic['지수']:.1f}")
    print(f"와이불  k={k_w:.3f} lam={lam_w:.2f}  ll={ll_w:.2f}  AIC={aic['와이불']:.1f}")
    print(f"로그정규 mu={mu_h:.3f} sg={sg_h:.3f}  ll={ll_l:.2f}  AIC={aic['로그정규']:.1f}")

    tg = np.linspace(0.5, 88, 900)
    S_e = np.exp(-lam_e * tg)
    S_w = np.exp(-(tg / lam_w) ** k_w)
    zz = (np.log(tg) - mu_h) / sg_h
    S_l = stats.norm.sf(zz)
    h_e = np.full_like(tg, lam_e)
    h_w = (k_w / lam_w) * (tg / lam_w) ** (k_w - 1)
    h_l = stats.norm.pdf(zz) / (tg * sg_h * stats.norm.sf(zz))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ev, n_j, d_j, S_km, H_km = km_table(t, d)
    m = n_j >= 15
    x = np.concatenate([[0.0], ev[m]])
    y = np.concatenate([[1.0], S_km[m]])
    axL.plot(x, y, drawstyle="steps-post", color=INK, lw=2.6, zorder=5,
             label="카플란-마이어")
    axL.plot(tg, S_e, color=RED, lw=2.2, ls=(0, (5, 2)),
             label=f"지수  (AIC {aic['지수']:.0f})")
    axL.plot(tg, S_w, color=ORANGE, lw=2.2, ls=(0, (2, 2)),
             label=f"와이불  (AIC {aic['와이불']:.0f})")
    axL.plot(tg, S_l, color=BLUE, lw=2.2,
             label=f"로그정규  (AIC {aic['로그정규']:.0f})")
    axL.set_xlim(0, 88)
    axL.set_ylim(0, 1.02)
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"$S(t)$", fontsize=11, color=INK)
    axL.set_title("생존곡선으로는 와이불과 로그정규를 가르기 어렵다",
                  fontsize=12, color=INK, pad=10)
    axL.legend(fontsize=10, frameon=False, loc="upper right")
    clean_axis(axL)

    axR.plot(tg, h_e, color=RED, lw=2.4, ls=(0, (5, 2)), label="지수 — 일정")
    axR.plot(tg, h_w, color=ORANGE, lw=2.4, ls=(0, (2, 2)), label="와이불 — 계속 증가")
    axR.plot(tg, h_l, color=BLUE, lw=2.6, label="로그정규 — 올랐다 내려감")
    pk = tg[np.argmax(h_l)]
    print(f"로그정규 적합 위험의 정점 t={pk:.1f}, h={float(h_l.max()):.4f}")
    axR.plot(pk, h_l.max(), "o", color=BLUE, ms=8, mec="white", mew=1.2, zorder=5)
    axR.annotate(f"정점 $t \\approx {pk:.0f}$ 개월",
                 xy=(pk, h_l.max()), xytext=(pk + 9, h_l.max() + 0.008),
                 fontsize=10.5, color=BLUE, ha="left", va="center",
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3))
    axR.set_xlim(0, 88)
    axR.set_ylim(0, 0.072)
    axR.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axR.set_ylabel(r"$h(t)$", fontsize=11, color=INK)
    axR.set_title("같은 자료인데 위험함수는 서로 딴판이다",
                  fontsize=12, color=INK, pad=10)
    axR.legend(fontsize=10, frameon=False, loc="lower right")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "three_models_fit.png")


if __name__ == "__main__":
    fig_memoryless_vs_aging()
    fig_weibull_shape_slope()
    fig_hump_hazard_tails()
    fig_censored_likelihood()
    fig_three_models_fit()
