r"""21.5 생존 모형 비교 두 쪽의 개념 그림을 생성한다.

만드는 파일:
  ch21/comparison/img/bias_variance_paradigms.png  잘못 지정된 모수 모형은 n 이 커져도 남는다
  ch21/comparison/img/aic_and_calibration.png      AIC 와 일치도 지수는 다른 것을 잰다

실행:  python3 scripts/make_ch21_comparison_figures.py   (저장소 최상위에서)
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

OUT = "docs/ch21/comparison/img/"
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


def km_curve(t, d):
    t = np.asarray(t, float)
    d = np.asarray(d, int)
    ts = np.sort(t)
    ev = np.unique(t[d == 1])
    if len(ev) == 0:
        return np.array([0.0]), np.array([1.0])
    n_j = len(t) - np.searchsorted(ts, ev, side="left")
    es = np.sort(t[d == 1])
    d_j = (np.searchsorted(es, ev, side="right")
           - np.searchsorted(es, ev, side="left"))
    S = np.cumprod(1.0 - d_j / n_j)
    return np.concatenate([[0.0], ev]), np.concatenate([[1.0], S])


def km_at(t, d, q):
    ev, S = km_curve(t, d)
    j = np.searchsorted(ev, q, side="right") - 1
    return float(S[max(j, 0)])


def weibull_mle(t, d):
    """프로파일 가능도로 (k, lambda) 를 구한다. lambda 는 k 에 대해 닫힌 형태다."""
    dd = d.sum()
    if dd == 0:
        return 1.0, np.inf
    slog = np.sum(d * np.log(t))

    def neg(k):
        lam = (np.sum(t ** k) / dd) ** (1.0 / k)
        return -(dd * np.log(k) - dd * k * np.log(lam) + (k - 1) * slog - dd)

    r = optimize.minimize_scalar(neg, bounds=(0.12, 12.0), method="bounded")
    k = float(r.x)
    lam = float((np.sum(t ** k) / dd) ** (1.0 / k))
    return k, lam


# === 그림 1. 세 갈래의 편향-분산 ===
def fig_bias_variance_paradigms():
    k_true, sc_true = 2.2, 40.0
    q = 30.0
    S_true = float(np.exp(-(q / sc_true) ** k_true))
    print(f"참 S(30) = {S_true:.4f}")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    # --- 왼쪽: 한 표본에서의 세 곡선 ---
    rng = np.random.default_rng(5)
    n = 400
    T = sc_true * rng.weibull(k_true, n)
    C = rng.uniform(5.0, 90.0, n)
    t, d = np.minimum(T, C), (T <= C).astype(int)
    lam_e = d.sum() / t.sum()
    k_w, lam_w = weibull_mle(t, d)
    print(f"한 표본(n=400, 절단율 {1-d.mean():.1%}):  지수 lam={lam_e:.5f}, "
          f"와이불 k={k_w:.3f} lam={lam_w:.2f}")

    tg = np.linspace(0, 70, 600)
    ev, S = km_curve(t, d)
    m = ev <= 70
    axL.plot(ev[m], S[m], drawstyle="steps-post", color=INK, lw=2.6, zorder=5,
             label="카플란-마이어 (비모수)")
    axL.plot(tg, np.exp(-(tg / sc_true) ** k_true), color=MUTED, lw=2.2,
             ls=(0, (5, 3)), label="참 곡선")
    axL.plot(tg, np.exp(-lam_e * tg), color=RED, lw=2.2, ls=(0, (2, 2)),
             label="지수 적합 (잘못 지정)")
    axL.plot(tg, np.exp(-(tg / lam_w) ** k_w), color=GREEN, lw=2.2,
             label="와이불 적합 (옳게 지정)")
    axL.axvline(q, color=MUTED, lw=1.1, ls=(0, (3, 3)), zorder=0)
    axL.text(q + 1.2, 0.95, f"$t = 30$ 에서\n참값 $S = {S_true:.3f}$", fontsize=10.5,
             color=INK, ha="left", va="top", linespacing=1.45)
    axL.set_xlim(0, 70)
    axL.set_ylim(0, 1.02)
    axL.set_xlabel("시간 (개월)", fontsize=11, color=INK)
    axL.set_ylabel(r"$S(t)$", fontsize=11, color=INK)
    axL.set_title(r"$n=400$ 짜리 한 표본에서의 세 추정치", fontsize=12.5,
                  color=INK, pad=10)
    axL.legend(fontsize=9.5, frameon=False, loc="lower left")
    clean_axis(axL)

    # --- 오른쪽: 표본이 커질 때의 RMSE ---
    ns = [30, 60, 120, 250, 500, 1000, 2000]
    reps = 220
    rng = np.random.default_rng(2024)
    out = {"km": [], "exp": [], "wb": []}
    bias = {"km": [], "exp": [], "wb": []}
    for nn in ns:
        est = {"km": [], "exp": [], "wb": []}
        for _ in range(reps):
            T = sc_true * rng.weibull(k_true, nn)
            C = rng.uniform(5.0, 90.0, nn)
            t, d = np.minimum(T, C), (T <= C).astype(int)
            est["km"].append(km_at(t, d, q))
            le = d.sum() / t.sum()
            est["exp"].append(np.exp(-le * q))
            kk, ll = weibull_mle(t, d)
            est["wb"].append(np.exp(-(q / ll) ** kk))
        for key in out:
            a = np.array(est[key])
            out[key].append(np.sqrt(np.mean((a - S_true) ** 2)))
            bias[key].append(a.mean() - S_true)
        print(f"n={nn:5d}  RMSE  KM {out['km'][-1]:.4f}  "
              f"지수 {out['exp'][-1]:.4f}  와이불 {out['wb'][-1]:.4f}   "
              f"|편향| 지수 {abs(bias['exp'][-1]):.4f}")

    for key, col, nm in (("km", INK, "카플란-마이어 (비모수)"),
                         ("exp", RED, "지수 (잘못 지정된 모수)"),
                         ("wb", GREEN, "와이불 (옳게 지정된 모수)")):
        axR.plot(ns, out[key], color=col, lw=2.6, marker="o", ms=6,
                 label=nm)
    axR.axhline(abs(bias["exp"][-1]), color=RED, lw=1.3, ls=(0, (4, 3)), zorder=0)
    axR.text(2050, abs(bias["exp"][-1]) + 0.002, "편향의 바닥", fontsize=10,
             color=RED, ha="right", va="bottom")
    axR.set_xscale("log")
    axR.set_xticks(ns)
    # 로그축의 눈금표는 수식으로 그려지므로 평문으로 직접 지정한다.
    axR.set_xticklabels([str(v) for v in ns], fontsize=10, color=INK)
    axR.minorticks_off()
    axR.set_ylim(0, 0.165)
    axR.set_xlabel("표본 크기 $n$ (로그 눈금)", fontsize=11, color=INK)
    axR.set_ylabel(r"$\hat S(30)$ 의 RMSE", fontsize=11, color=INK)
    axR.set_title("표본을 키워도 오지정의 편향은 남는다", fontsize=12.5,
                  color=INK, pad=10)
    axR.legend(fontsize=10, frameon=False, loc="upper right")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.6)
    save(fig, "bias_variance_paradigms.png")


# === 그림 2. AIC 와 일치도 지수는 다른 것을 잰다 ===
def c_index(risk, t, d):
    conc = perm = 0.0
    for i in np.where(d == 1)[0]:
        comp = t > t[i]
        conc += (risk[comp] < risk[i]).sum() + 0.5 * (risk[comp] == risk[i]).sum()
        perm += comp.sum()
    return conc / perm


def fig_aic_and_calibration():
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13.6, 4.9), gridspec_kw={"width_ratios": [1.0, 1.05]})

    # --- 왼쪽: 네 모수 모형의 AIC ---
    rng = np.random.default_rng(404)
    n = 500
    T = np.exp(3.4 + 0.75 * rng.standard_normal(n))      # 참값: 로그정규
    C = rng.uniform(3.0, 140.0, n)
    t, d = np.minimum(T, C), (T <= C).astype(int)
    print(f"AIC 용 자료: n={n}, 절단율 {1-d.mean():.1%}, 사건 {int(d.sum())}건")

    lam_e = d.sum() / t.sum()
    ll_e = float(np.sum(d * np.log(lam_e) - lam_e * t))
    k_w, lam_w = weibull_mle(t, d)
    ll_w = float(np.sum(d * (np.log(k_w) - k_w * np.log(lam_w)
                             + (k_w - 1) * np.log(t)) - (t / lam_w) ** k_w))

    def ln_neg(p):
        m, s = p[0], np.exp(p[1])
        z = (np.log(t) - m) / s
        return -np.sum(d * (stats.norm.logpdf(z) - np.log(s * t))
                       + (1 - d) * stats.norm.logsf(z))
    r = optimize.minimize(ln_neg, [np.log(t).mean(), 0.0], method="Nelder-Mead",
                          options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
    ll_l = -r.fun

    def ll_neg(p):
        kk, lm = np.exp(p)
        u = (t / lm) ** kk
        # 사건이면 밀도 log f = log(k/lam) + (k-1) log(t/lam) - 2 log(1+u),
        # 절단이면 생존함수 log S = -log(1+u).
        return -np.sum(d * (np.log(kk / lm) + (kk - 1) * np.log(t / lm))
                       - (1.0 + d) * np.log1p(u))
    r2 = optimize.minimize(ll_neg, [0.0, np.log(np.median(t))],
                           method="Nelder-Mead",
                           options=dict(xatol=1e-9, fatol=1e-9, maxiter=4000))
    ll_g = -r2.fun

    models = [("지수", ll_e, 1, RED), ("와이불", ll_w, 2, ORANGE),
              ("로그로지스틱", ll_g, 2, PURPLE), ("로그정규", ll_l, 2, BLUE)]
    aics = np.array([-2 * ll + 2 * p for _, ll, p, _ in models])
    best = aics.min()
    dA = aics - best
    for (nm, ll, p, _), a, da in zip(models, aics, dA):
        print(f"  {nm:8s} p={p} loglik={ll:9.2f}  AIC={a:8.2f}  dAIC={da:6.2f}")

    ys = np.arange(len(models))[::-1]
    axL.axvspan(0, 2, color=GREEN_F, alpha=0.7, zorder=0)
    axL.text(1.0, 3.62, "$\\Delta$AIC < 2\n구별되지 않음", fontsize=10,
             color=GREEN, ha="center", va="top", linespacing=1.4)
    for y, (nm, ll, p, col), da in zip(ys, models, dA):
        axL.barh(y, da, height=0.5, color=col, alpha=0.85, zorder=3)
        lab = f"{da:.1f}" + ("   ← 자료를 만든 참 모형" if nm == "로그정규" else "")
        axL.text(da + 2.5, y, lab, fontsize=10.5, color=col,
                 ha="left", va="center")
    axL.set_yticks(ys)
    axL.set_yticklabels([m[0] for m in models], fontsize=11, color=INK)
    axL.tick_params(axis="y", length=0)
    axL.set_xlim(0, max(dA) * 1.45)
    axL.set_ylim(-0.6, 3.9)
    axL.set_xlabel(r"최선 모형과의 $\Delta$AIC", fontsize=11, color=INK)
    axL.set_title("AIC 가 참 모형을 고르지 못한 예", fontsize=12.5, color=INK,
                  pad=10)
    clean_axis(axL)
    axL.spines["left"].set_visible(False)

    # --- 오른쪽: 순위가 같아도 보정은 다르다 ---
    rng2 = np.random.default_rng(77)
    m = 3000
    x = rng2.standard_normal(m)
    beta = 0.9
    k0, sc0 = 1.6, 38.0
    U = rng2.random(m)
    T2 = sc0 * (-np.log(U) / np.exp(beta * x)) ** (1 / k0)
    C2 = rng2.uniform(4.0, 120.0, m)
    t2, d2 = np.minimum(T2, C2), (T2 <= C2).astype(int)
    t0 = 25.0
    S_A = np.exp(-np.exp(beta * x) * (t0 / sc0) ** k0)     # 옳은 모형
    S_B = S_A ** 0.45                                       # 순위는 같고 보정만 틀림
    cA = c_index(-S_A, t2, d2)
    cB = c_index(-S_B, t2, d2)
    print(f"C-지수:  옳은 모형 {cA:.4f},  기저를 잘못 잡은 모형 {cB:.4f}")

    qs = np.quantile(S_A, np.linspace(0, 1, 11))
    for pred, col, nm, mk in ((S_A, GREEN, "기저를 옳게 잡은 모형", "o"),
                              (S_B, RED, "기저를 잘못 잡은 모형", "s")):
        px, py = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            sel = (S_A >= a) & (S_A <= b)
            px.append(pred[sel].mean())
            py.append(km_at(t2[sel], d2[sel], t0))
        axR.plot(px, py, marker=mk, ms=7, lw=2.2, color=col, label=nm)
        print(f"  {nm}: 예측 {np.round(px, 3)}")
        print(f"  {nm}: 관측 {np.round(py, 3)}")
    axR.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls=(0, (5, 3)), zorder=0,
             label="완벽한 보정")
    axR.set_xlim(0, 1.0)
    axR.set_ylim(0, 1.0)
    axR.set_xlabel(r"예측한 $S(25)$ (십분위 평균)", fontsize=11, color=INK)
    axR.set_ylabel(r"관측된 $S(25)$ (카플란-마이어)", fontsize=11, color=INK)
    axR.set_title("두 모형의 일치도 지수는 소수점까지 같다",
                  fontsize=12.5, color=INK, pad=10)
    axR.text(0.97, 0.16,
             f"$C$ = {cA:.4f}  (옳은 모형)\n$C$ = {cB:.4f}  (잘못 잡은 모형)",
             fontsize=11, color=INK, ha="right", va="bottom", linespacing=1.55)
    axR.legend(fontsize=10, frameon=False, loc="upper left")
    clean_axis(axR)

    fig.tight_layout(w_pad=2.8)
    save(fig, "aic_and_calibration.png")


if __name__ == "__main__":
    fig_bias_variance_paradigms()
    fig_aic_and_calibration()
