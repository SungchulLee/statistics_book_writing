"""12장 들머리 그림 여섯 장 (모형 선택 여섯 쪽).

만드는 파일:
  ch12/model_selection/img/penalty_is_the_difference.png  벌점의 차이가 답의 차이다
  ch12/model_selection/img/only_differences_matter.png    값이 아니라 차이만 뜻이 있다
  ch12/model_selection/img/noise_still_gets_in.png        잡음 하나가 들어올 확률
  ch12/model_selection/img/ruler_wobbles_more.png         자가 차이보다 더 흔들린다
  ch12/model_selection/img/which_degree_wins.png          어느 차수가 이기는가
  ch12/model_selection/img/when_global_wins.png           전역 최적이 이기는 자리

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch12/model_selection/img/"


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def rss_of(X, y, cols):
    A = np.column_stack([np.ones(len(y))] + [X[:, c] for c in cols])
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return ((y - A @ b) ** 2).sum()


def forward_path(X, y):
    """전진선택 — 단계마다 잔차제곱합을 가장 많이 줄이는 변수를 넣는다."""
    p = X.shape[1]
    chosen, path = [], [(0, rss_of(X, y, []))]
    rest = list(range(p))
    while rest:
        best = min(rest, key=lambda c: rss_of(X, y, chosen + [c]))
        chosen.append(best)
        rest.remove(best)
        path.append((len(chosen), rss_of(X, y, chosen)))
    return chosen, path


# ════════════════════════════════════════════════════════════════════
def fig_penalty_is_the_difference():
    """11.9 모형 선택 — 같은 걸음에 두 기준이 매기는 값이 다르다.

    쪽의 forward_selection_path 는 전진선택의 자취를, three_criteria_agree
    상자는 셋이 한자리를 가리키는 일이 얼마나 흔한지를 다룬다. 여기서는
    들머리의 셋째 문단 --- **같은 걸음에 AIC 는 0.61 오르고 BIC 는 3.91
    오른다** --- 를 모든 걸음에서 그리고, 그 여섯 배가 어디서 오는지 본다.
    """
    n, p, SIG = 200, 8, 2.0
    rng = np.random.default_rng(1190)
    X = rng.normal(0, 1, (n, p))
    beta = np.array([1.2, -0.9, 0.7] + [0.0] * (p - 3))
    y = 3 + X @ beta + rng.normal(0, SIG, n)
    chosen, path = forward_path(X, y)
    ks = np.array([k for k, _ in path])
    rss = np.array([r for _, r in path])
    fit = n * np.log(rss / n)
    aic = fit + 2 * (ks + 1)
    bic = fit + np.log(n) * (ks + 1)

    d_fit = np.diff(fit)
    d_aic = np.diff(aic)
    d_bic = np.diff(bic)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(1, len(ks))
    w = 0.26
    ax1.bar(xs - w, d_fit, width=w * 0.92, color="0.6", alpha=0.9,
            label="적합항의 변화")
    ax1.bar(xs, d_aic, width=w * 0.92, color="tab:blue", alpha=0.9,
            label="AIC 의 변화 (벌점 $2$)")
    ax1.bar(xs + w, d_bic, width=w * 0.92, color="tab:red", alpha=0.9,
            label=f"BIC 의 변화 (벌점 $\\ln {n} = {np.log(n):.2f}$)")
    ax1.axhline(0, color="0.15", lw=1.8)
    j = 3                                   # 크기 3 에서 4 로 가는 걸음
    ax1.annotate(f"크기 {ks[j]} → {ks[j+1]} 의 걸음\n"
                 f"적합 {d_fit[j]:+.2f} · AIC {d_aic[j]:+.2f} · "
                 f"BIC {d_bic[j]:+.2f}", xy=(xs[j] + w, d_bic[j]),
                 xytext=(4.3, -26), ha="left", fontsize=11, color="0.15",
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="0.35", lw=1.6))
    ax1.set_xticks(xs, [f"{a}→{b}" for a, b in zip(ks[:-1], ks[1:])],
                   fontsize=10.5)
    ax1.set_xlabel("전진선택의 걸음 (모형 크기)", fontsize=11)
    ax1.set_ylabel("값의 변화 — 음수면 그 걸음이 이득이다", fontsize=11)
    ax1.set_title(f"$n = {n}$ · 참 변수 셋 · 잡음 변수 다섯", fontsize=12.5)
    ax1.legend(fontsize=10, loc="lower right", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)

    ns = np.logspace(np.log10(3), np.log10(1e5), 300)
    ax2.plot(ns, np.log(ns) / 2, "-", lw=2.8, color="tab:red")
    ax2.axhline(1, color="tab:blue", ls="--", lw=2.0)
    ax2.text(1e5, 1.08, "둘이 같은 자리", ha="right", va="bottom",
             fontsize=10.5, color="tab:blue", fontweight="bold")
    for nn in (200, 40_000):
        ax2.plot([nn], [np.log(nn) / 2], "o", ms=12, color="tab:red",
                 zorder=5)
        ax2.text(nn * 0.75, np.log(nn) / 2 + 0.35,
                 f"n = {nn:,}\n{np.log(nn) / 2:.2f} 배", ha="right",
                 va="bottom", fontsize=10.5, color="tab:red",
                 fontweight="bold")
    ax2.axvline(np.e ** 2, color="0.35", ls=":", lw=1.8)
    ax2.text(np.e ** 2 * 1.2, 0.18, f"$n = e^2 = {np.e ** 2:.2f}$\n"
             "여기서부터 BIC 가 더 엄하다", ha="left", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlim(3, 1e5)
    ax2.set_ylim(0, 6.5)
    ax2.set_xlabel("표본 크기 n (로그 척도)", fontsize=11)
    ax2.set_ylabel("BIC 의 벌점 ÷ AIC 의 벌점", fontsize=11)
    ax2.set_title("모수 하나에 매기는 값의 비 $\\ln n / 2$", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("세 기준이 갈리는 자리는 벌점이 갈리는 자리다", fontsize=14,
                 y=1.0)
    plt.tight_layout()
    save(fig, "penalty_is_the_difference.png")

    print(f"\n[모형 선택] n = {n} · 전진선택이 넣은 순서 {chosen}")
    for k, r, a, b in zip(ks, rss, aic, bic):
        print(f"  크기 {k}  RSS {r:9.4f}  AIC {a:9.4f}  BIC {b:9.4f}")
    print(f"  크기 {ks[j]} → {ks[j+1]}: 적합 {d_fit[j]:+.4f} · "
          f"AIC {d_aic[j]:+.4f} · BIC {d_bic[j]:+.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_only_differences_matter():
    """11.9 AIC — 값 자체에는 뜻이 없고 차이만 뜻이 있다.

    쪽의 aic_decomposition 은 적합항과 벌점항의 분해를,
    akaike_weight_calibration 상자는 가중치를 다룬다. 여기서는 들머리의
    마지막 문단 --- **39.4 나 42.5 라는 값 자체에는 뜻이 없다** --- 만
    본다. 자료의 단위를 바꾸면 세 후보의 AIC 가 통째로 움직이는데 차이는
    한 자리도 안 바뀐다.
    """
    n = 50
    CAND = [("A", 1, 120.0), ("B", 3, 90.0), ("C", 6, 85.0)]
    SCALES = [(1.0, "그대로"), (10.0, "단위를 $10$ 배로"),
              (0.1, "단위를 $1/10$ 로")]

    def aic(rss, p_):
        # 쪽의 7 절 표와 같은 규약 — k 는 절편과 sigma^2 까지 세므로 k = p + 2 다
        return n * np.log(rss / n) + 2 * (p_ + 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(len(CAND))
    w = 0.26
    cols = ["0.6", "tab:blue", "tab:red"]
    for i, (c, lab) in enumerate(SCALES):
        v = [aic(r * c ** 2, k) for _, k, r in CAND]
        ax1.bar(xs + (i - 1) * w, v, width=w * 0.92, color=cols[i],
                alpha=0.9, label=lab)
        for x_, vv in zip(xs + (i - 1) * w, v):
            ax1.text(x_, vv + 3, f"{vv:.1f}", ha="center", va="bottom",
                     fontsize=9.5, color="0.2", fontweight="bold")
    ax1.axhline(0, color="0.15", lw=1.4)
    ax1.set_xticks(xs, [f"{nm}\n설명변수 {k} 개" for nm, k, _ in CAND],
                   fontsize=11)
    ax1.set_ylim(-330, 430)
    ax1.set_ylabel("AIC", fontsize=11)
    ax1.set_title(f"$n = {n}$ — 같은 세 후보, 단위만 바꿨다", fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(axis="y", alpha=0.25)
    base = [aic(r, k) for _, k, r in CAND]
    ax1.text(0.5, 0.04, f"세 막대가 통째로 오르내려도\n"
             f"B 와 C 의 차이는 늘 {base[2] - base[1]:.2f} 다",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ns = np.arange(10, 501)
    ax2.plot(ns, 100 * (1 - np.exp(-2 / ns)), "-", lw=2.8, color="tab:purple")
    for nn in (20, 50, 200, 500):
        v = 100 * (1 - np.exp(-2 / nn))
        ax2.plot([nn], [v], "o", ms=10, color="tab:purple", zorder=5)
        side = "right" if nn > 400 else "left"
        ax2.text(nn + (-8 if side == "right" else 8), v + 0.25,
                 f"n = {nn}\n{v:.2f}%", ha=side, va="bottom", fontsize=10.5,
                 color="tab:purple", fontweight="bold")
    ax2.set_xlim(10, 520)
    ax2.set_ylim(0, 22)
    ax2.set_xlabel("표본 크기 n", fontsize=11)
    ax2.set_ylabel("모수 하나가 본전이 되려면\n잔차제곱합을 줄여야 하는 비율 (%)",
                   fontsize=11)
    ax2.set_title("모수 하나는 적합항 $2$ 만큼의 값어치가 있어야 한다",
                  fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("AIC 의 값에는 뜻이 없고 같은 자료 안에서의 차이에만 뜻이 있다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "only_differences_matter.png")

    print(f"\n[AIC] n = {n}")
    for (nm, k, r), v in zip(CAND, base):
        print(f"  {nm}  설명변수 {k} 개  RSS {r:6.1f}  AIC {v:7.2f}")
    print(f"  B 와 C 의 차이 {base[2] - base[1]:+.2f} "
          f"(적합 {n * np.log(85 / 90):+.2f} + 벌점 {2 * 3:+.0f})")
    for c, lab in SCALES:
        v = [aic(r * c ** 2, k) for _, k, r in CAND]
        print(f"  단위 x{c:<5g} AIC {v[0]:8.2f} {v[1]:8.2f} {v[2]:8.2f}  "
              f"차이 {v[2] - v[1]:+.2f}")


# ════════════════════════════════════════════════════════════════════
def fig_noise_still_gets_in():
    """11.9 BIC — 잡음 변수 하나가 들어올 확률.

    쪽의 bic_consistency 는 표본을 키울 때의 집어내는 비율을,
    aic_vs_bic 상자는 둘이 다른 물음에 답한다는 것을 다룬다. 여기서는
    그 둘이 수로 말하지 않는 **까닭** --- 잡음 변수 하나가 모형에 들어올
    확률이 AIC 에서는 n 과 무관하게 15.7% 로 고정되고 BIC 에서는 0 으로
    간다 --- 만 본다. 변수 하나를 더 넣을 때 적합항이 줄어드는 양이
    근사적으로 카이제곱(1) 이므로 AIC 는 그것이 2 를 넘을 때,
    BIC 는 ln n 을 넘을 때 받아들인다.
    """
    ns = np.logspace(np.log10(10), np.log10(1e6), 400)
    p_aic = stats.chi2.sf(2, 1) * np.ones_like(ns)
    p_bic = stats.chi2.sf(np.log(ns), 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(ns, 100 * p_aic, "-", lw=2.8, color="tab:blue",
             label="AIC — 임계값이 늘 $2$")
    ax1.plot(ns, 100 * p_bic, "-", lw=2.8, color="tab:red",
             label="BIC — 임계값이 $\\ln n$")
    for nn, col in ((200, "0.35"), (40_000, "0.35")):
        ax1.axvline(nn, color=col, ls=":", lw=1.6)
        ax1.text(nn * 1.2, 14.0, f"n = {nn:,}\nBIC "
                 f"{100 * stats.chi2.sf(np.log(nn), 1):.2f}%", ha="left",
                 va="top", fontsize=10.5, color="tab:red", fontweight="bold")
    ax1.text(1.2e6, 100 * p_aic[0] + 0.6,
             f"{100 * p_aic[0]:.1f}% 에 고정된다", ha="right", va="bottom",
             fontsize=11, color="tab:blue", fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlim(10, 1.2e6)
    ax1.set_ylim(0, 22)
    ax1.set_xlabel("표본 크기 n (로그 척도)", fontsize=11)
    ax1.set_ylabel("잡음 변수 하나가 들어올 확률 (%)", fontsize=11)
    ax1.set_title("자료를 아무리 쌓아도 AIC 는 문을 닫지 않는다",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    MS = np.arange(0, 11)
    ax2.plot(MS, MS * stats.chi2.sf(2, 1), "o-", ms=8, lw=2.6,
             color="tab:blue", label="AIC (n 과 무관하다)")
    for nn, col in ((200, "tab:orange"), (40_000, "tab:red")):
        ax2.plot(MS, MS * stats.chi2.sf(np.log(nn), 1), "s-", ms=8, lw=2.6,
                 color=col, label=f"BIC · n = {nn:,}")
    ax2.set_xticks(MS)
    ax2.set_xlabel("후보에 들어 있는 잡음 변수의 수", fontsize=11)
    ax2.set_ylabel("모형에 섞여 들어오는 잡음 변수의 기대 개수", fontsize=11)
    ax2.set_title("잡음 후보가 많을수록 그대로 비례해 늘어난다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("모수 하나의 벌점을 $2$ 에서 $\\ln n$ 으로 바꾼 것뿐인데",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "noise_still_gets_in.png")

    print(f"\n[BIC] 잡음 변수 하나가 들어올 확률")
    print(f"  AIC — n 과 무관하게 {100 * stats.chi2.sf(2, 1):.2f}%")
    for nn in (50, 200, 1000, 40_000, 10 ** 6):
        print(f"  BIC · n = {nn:>9,}  임계값 ln n = {np.log(nn):5.2f}  "
              f"{100 * stats.chi2.sf(np.log(nn), 1):6.3f}%")


# ════════════════════════════════════════════════════════════════════
def fig_ruler_wobbles_more():
    """11.9 교차검증 — 재는 자가 고르려는 차이보다 더 흔들린다.

    쪽의 cv_flat_curve 는 평평한 곡선을, cv_k_bias_variance 는 K 의
    절충을 그린다. 여기서는 들머리의 둘째·셋째 문단 --- **분할의 난수만
    바꾸어도 고른 차수가 3 과 10 사이를 오간다** --- 를 겹쳐 그리고,
    그 흔들림을 줄이려면 몇 번을 평균해야 하는지 센다.
    """
    n, DEG = 200, np.arange(1, 11)
    SPLITS = 10
    # 쪽의 보기 1 과 **같은 자료**를 쓴다 (generate_data(n=200, seed=42)).
    rng = np.random.default_rng(42)
    x = rng.uniform(-3, 3, n)
    y = np.sin(x) + 0.3 * x + rng.normal(0, 0.5, n)

    def holdout(g):
        idx = g.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        out = []
        for d in DEG:
            c = np.polyfit(x[tr], y[tr], d)
            out.append(np.mean((y[te] - np.polyval(c, x[te])) ** 2))
        return np.array(out)

    # 쪽의 보기 3 과 **같은 난수 흐름** — 씨앗 하나에서 열 번 뽑는다
    g10 = np.random.default_rng(7)
    curves = np.array([holdout(g10) for _ in range(SPLITS)])
    picked = DEG[curves.argmin(axis=1)]

    MANY = 400
    gm = np.random.default_rng(1000)
    all_curves = np.array([holdout(gm) for _ in range(MANY)])
    reps = [1, 2, 5, 10, 20, 50, 100]
    sd_pick = []
    for m in reps:
        picks = [DEG[all_curves[i:i + m].mean(axis=0).argmin()]
                 for i in range(0, MANY - m + 1, max(m, 1))]
        sd_pick.append(np.std(picks, ddof=1) if len(picks) > 1 else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    for c in curves:
        ax1.plot(DEG, c, "-", lw=1.4, color="0.68", alpha=0.9)
    ax1.plot(DEG, curves.mean(axis=0), "o-", ms=8, lw=3.0, color="tab:blue",
             label=f"{SPLITS} 번의 평균")
    for c, pk in zip(curves, picked):
        ax1.plot([pk], [c[pk - 1]], "o", ms=8, color="tab:red", zorder=5)
    ax1.set_xticks(DEG)
    ax1.set_xlabel("다항식의 차수", fontsize=11)
    ax1.set_ylabel("떼어 둔 절반에서 잰 MSE", fontsize=11)
    ax1.set_title(f"$n = {n}$ 을 절반으로 가르는 난수만 {SPLITS} 번 바꿨다",
                  fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.97, 0.05, f"붉은 점이 그 분할이 고른 차수다\n"
             f"{picked.tolist()} — 표준편차 "
             f"{picked.std():.4f}", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=10.5, fontweight="bold",
             color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    ax2.plot(reps, sd_pick, "o-", ms=9, lw=2.6, color="tab:purple")
    ax2.axhline(0.5, color="0.3", ls="--", lw=1.8)
    ax2.text(100, 0.56, "차수 한 칸의 절반", ha="right", va="bottom",
             fontsize=10.5, color="0.3", fontweight="bold")
    for m, v in zip(reps, sd_pick):
        ax2.text(m, v + 0.08, f"{v:.2f}", ha="center", va="bottom",
                 fontsize=10.5, color="tab:purple", fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xticks(reps, [str(m) for m in reps])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, max(sd_pick) * 1.25)
    ax2.set_xlabel("몇 번의 분할을 평균했는가 (로그 척도)", fontsize=11)
    ax2.set_ylabel("고른 차수의 표준편차", fontsize=11)
    ax2.set_title("여러 번 재서 평균하면 가라앉는다", fontsize=12.5)
    ax2.grid(alpha=0.25)

    fig.suptitle("자료도 방법도 그대로인데 어디서 끊었는지만 다르다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "ruler_wobbles_more.png")

    print(f"\n[교차검증] n = {n} · 분할 {SPLITS} 번")
    print(f"  고른 차수 {picked.tolist()} · 표준편차 "
          f"{picked.std():.4f}")
    for m, v in zip(reps, sd_pick):
        print(f"  {m:>4d} 번 평균  고른 차수의 표준편차 {v:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_which_degree_wins():
    """11.9 교차검증으로 차수 선택하기 — 어느 차수가 이기는가.

    쪽의 cv_method_comparison 은 세 방법을 견주고 galileo_degree 상자는
    다른 이야기를 따라간다. 여기서는 들머리의 둘째·셋째 문단 --- **참 답이
    후보 안에 있는데도 네 번에 세 번은 못 맞힌다**, 그리고 **평균은 맞고
    한 번이 흔들린다** --- 만 본다.
    """
    n, DEG = 200, np.arange(1, 11)
    SIG = 2.0
    TRUE = 2
    REPS = 2000
    rng = np.random.default_rng(1192)
    x = rng.uniform(-3, 3, n)
    y = 5 + 2 * x - 0.3 * x ** 2 + rng.normal(0, SIG, n)

    scores = np.empty((REPS, len(DEG)))
    for i in range(REPS):
        idx = rng.permutation(n)
        tr, te = idx[:n // 2], idx[n // 2:]
        for j, d in enumerate(DEG):
            c = np.polyfit(x[tr], y[tr], d)
            scores[i, j] = np.mean((y[te] - np.polyval(c, x[te])) ** 2)
    win = DEG[scores.argmin(axis=1)]
    hit = (win == TRUE).mean()
    s2 = scores[:, TRUE - 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    cnt = np.array([(win == d).mean() for d in DEG])
    cols = ["tab:red" if d == TRUE else "0.65" for d in DEG]
    ax1.bar(DEG, 100 * cnt, width=0.6, color=cols, alpha=0.9)
    for d, v in zip(DEG, cnt):
        if v > 0.004:
            ax1.text(d, 100 * v + 0.8, f"{100 * v:.1f}", ha="center",
                     va="bottom", fontsize=10.5, fontweight="bold",
                     color="0.2")
    ax1.set_xticks(DEG)
    ax1.set_xlabel("검증집합이 고른 차수", fontsize=11)
    ax1.set_ylabel("고른 비율 (%)", fontsize=11)
    ax1.set_title(f"참 차수는 {TRUE} 다 — 분할을 {REPS:,} 번 다르게 했다",
                  fontsize=12.5)
    ax1.set_ylim(0, 100 * cnt.max() * 1.3)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.97, 0.94, f"참값을 맞힌 비율 {100 * hit:.1f}%\n"
             f"이긴 적이 있는 차수가 {int((cnt > 0).sum())} 가지",
             transform=ax1.transAxes, ha="right", va="top", fontsize=11.5,
             fontweight="bold", color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    ax2.hist(s2, bins=60, color="0.68", alpha=0.9)
    ax2.axvline(SIG ** 2, color="tab:red", lw=2.6)
    ax2.text(SIG ** 2 + 0.07, ax2.get_ylim()[1] * 0.95,
             f"참 잡음분산 {SIG ** 2:.0f}", fontsize=11, color="tab:red",
             fontweight="bold", va="top")
    ax2.axvline(s2.mean(), color="tab:blue", lw=2.6, ls="--")
    ax2.text(s2.mean() - 0.07, ax2.get_ylim()[1] * 0.80,
             f"평균 {s2.mean():.4f}", fontsize=11, color="tab:blue",
             fontweight="bold", va="top", ha="right")
    ax2.set_xlabel(f"차수 {TRUE} 의 검증 MSE", fontsize=11)
    ax2.set_ylabel("횟수", fontsize=11)
    ax2.set_title("평균은 맞고 한 번의 값이 흔들린다", fontsize=12.5)
    ax2.grid(axis="y", alpha=0.25)
    ax2.text(0.97, 0.55, f"가장 작은 값 {s2.min():.4f}\n"
             f"가장 큰 값 {s2.max():.4f}\n— {s2.max() / s2.min():.2f} 배 차이다",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("편향이 없고 분산만 큰 자는 한 번으로 쓰면 안 된다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "which_degree_wins.png")

    print(f"\n[차수 선택하기] n = {n} · 분할 {REPS:,} 번 · 참 차수 {TRUE}")
    print(f"  맞힌 비율 {100 * hit:.2f}% · 이긴 적 있는 차수 "
          f"{int((cnt > 0).sum())} 가지")
    print(f"  차수 {TRUE} 의 검증 MSE: 평균 {s2.mean():.4f} "
          f"(참 분산 {SIG ** 2:.1f}) · 최소 {s2.min():.4f} · "
          f"최대 {s2.max():.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_when_global_wins():
    """11.9 부분집합·단계적 선택 — 전역 최적이 이기는 자리.

    쪽의 subset_cost 는 2^p 와 p(p+1)/2 의 값을, three_methods_agree 는
    한 설정에서 셋이 같은 답을 내는 비율을 그린다. 여기서는 들머리의
    셋째 문단이 말만 하고 넘어간 것 --- **변수들이 서로 얽히면 그 96.8%
    가 내려간다** --- 을 상관을 움직여 가며 잰다.
    """
    p, k_true, n, REPS = 8, 4, 60, 300
    RHOS = [0.0, 0.3, 0.6, 0.8, 0.9, 0.95]
    rng = np.random.default_rng(1193)
    beta = np.array([0.55] * k_true + [0.0] * (p - k_true))

    def best_subset(X, y, size):
        from itertools import combinations
        return min(combinations(range(p), size),
                   key=lambda c: rss_of(X, y, list(c)))

    agree, acc_best, acc_fwd = [], [], []
    for rho in RHOS:
        L = np.linalg.cholesky(rho * np.ones((p, p))
                               + (1 - rho) * np.eye(p))
        a = b = c = 0
        for _ in range(REPS):
            X = rng.normal(0, 1, (n, p)) @ L.T
            y = X @ beta + rng.normal(0, 1.0, n)
            bs = set(best_subset(X, y, k_true))
            fw = set(forward_path(X, y)[0][:k_true])
            a += bs == fw
            b += bs == set(range(k_true))
            c += fw == set(range(k_true))
        agree.append(a / REPS)
        acc_best.append(b / REPS)
        acc_fwd.append(c / REPS)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(RHOS, [100 * v for v in agree], "o-", ms=9, lw=2.8,
             color="tab:purple")
    for r, v in zip(RHOS, agree):
        ax1.text(r, 100 * v + 1.6, f"{100 * v:.1f}", ha="center",
                 va="bottom", fontsize=10.5, color="tab:purple",
                 fontweight="bold")
    ax1.set_xlim(-0.04, 1.0)
    ax1.set_ylim(0, 108)
    ax1.set_xlabel("설명변수들 사이의 상관", fontsize=11)
    ax1.set_ylabel("두 방법이 같은 답을 낸 비율 (%)", fontsize=11)
    ax1.set_title(f"최량 부분집합과 전진선택 ($n = {n}$, 변수 {p} 개, "
                  f"각 {REPS} 번)", fontsize=12)
    ax1.grid(alpha=0.25)
    ax1.text(0.5, 0.06, "변수들이 얽힐수록 둘이 갈린다\n"
             "— 들어오는 순서가 결과를 바꾸기 때문이다",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(RHOS, [100 * v for v in acc_best], "o-", ms=9, lw=2.8,
             color="tab:blue", label="최량 부분집합")
    ax2.plot(RHOS, [100 * v for v in acc_fwd], "s-", ms=8, lw=2.8,
             color="tab:red", label="전진선택")
    ax2.set_xlim(-0.04, 1.0)
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("설명변수들 사이의 상관", fontsize=11)
    ax2.set_ylabel("참 변수 넷을 정확히 집어낸 비율 (%)", fontsize=11)
    ax2.set_title("갈려도 정답률은 거의 같이 떨어진다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("$2^p$ 를 치르는 일이 값을 하는 자리는 변수들이 얽힌 "
                 "자리다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, "when_global_wins.png")

    print(f"\n[부분집합] n = {n}, 변수 {p} 개 (참 {k_true}), 각 {REPS} 번")
    print("   상관   같은 답   최량 정답률  전진 정답률")
    for r, a, b, c in zip(RHOS, agree, acc_best, acc_fwd):
        print(f"  {r:.2f}   {100*a:6.1f}%   {100*b:8.1f}%  {100*c:8.1f}%")


fig_penalty_is_the_difference()
fig_only_differences_matter()
fig_noise_still_gets_in()
fig_ruler_wobbles_more()
fig_which_degree_wins()
fig_when_global_wins()
