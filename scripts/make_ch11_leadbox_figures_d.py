"""11장 들머리 그림 여섯 장 (성능 척도 넷 · 스플라인 둘).

만드는 파일:
  ch11/performance/img/three_rulers_three_models.png  세 자가 고르는 모형이 다르다
  ch11/performance/img/two_residuals_carry_it.png     둘이 절반을 들고 있다
  ch11/performance/img/ratio_tells_the_shape.png      비가 모양을 말해 준다
  ch11/performance/img/mape_picks_low.png             MAPE 가 고르는 상수
  ch11/splines_gams/img/knots_beat_degree.png         차수 대신 매듭
  ch11/splines_gams/img/effect_spread_over_basis.png  효과가 열에 흩어진다

본문 그림이 들머리의 수를 이미 그리고 있는 쪽에서는 그 그림이 건드리지
않는 각만 골랐다. 까닭은 함수마다 적어 두었다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

PERF = "docs/ch11/performance/img/"
SPL = "docs/ch11/splines_gams/img/"


def save(fig, path):
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


# ════════════════════════════════════════════════════════════════════
def fig_three_rulers():
    """11.7 결정계수 — 세 자가 서로 다른 모형을 고른다.

    쪽의 r2_inflation 은 잡음 변수를 더할 때 R^2 가 오르는 것을,
    r2_depends_on_range 상자는 x 의 범위가 R^2 를 바꾸는 것을 다룬다.
    여기서는 들머리의 마지막 문단 --- **수정 R^2 는 덜어 줄 뿐 고치지
    못한다** --- 만 본다. 세 자에게 각각 모형을 고르게 하고 그 선택이
    어떻게 갈리는지 센다.
    """
    N, REPS = 50, 400
    KS = np.arange(0, 46)
    rng = np.random.default_rng(1170)

    tr, adj, te = (np.zeros(len(KS)) for _ in range(3))
    pick = {"훈련 $R^2$": [], "수정 $R^2$": [], "검정 $R^2$": []}
    for _ in range(REPS):
        x = rng.normal(0, 1, N)
        noise = rng.normal(0, 1, (N, KS.max()))
        y = 2 * x + rng.normal(0, 1, N)
        xt = rng.normal(0, 1, 500)
        nt = rng.normal(0, 1, (500, KS.max()))
        yt = 2 * xt + rng.normal(0, 1, 500)
        row_tr, row_adj, row_te = [], [], []
        for k in KS:
            A = np.column_stack([np.ones(N), x, noise[:, :k]])
            b = np.linalg.lstsq(A, y, rcond=None)[0]
            r2 = 1 - ((y - A @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            p = k + 1
            a2 = 1 - (1 - r2) * (N - 1) / max(N - p - 1, 1)
            At = np.column_stack([np.ones(500), xt, nt[:, :k]])
            r2t = 1 - ((yt - At @ b) ** 2).sum() / \
                ((yt - yt.mean()) ** 2).sum()
            row_tr.append(r2)
            row_adj.append(a2)
            row_te.append(r2t)
        tr += np.array(row_tr) / REPS
        adj += np.array(row_adj) / REPS
        te += np.array(row_te) / REPS
        pick["훈련 $R^2$"].append(KS[int(np.argmax(row_tr))])
        pick["수정 $R^2$"].append(KS[int(np.argmax(row_adj))])
        pick["검정 $R^2$"].append(KS[int(np.argmax(row_te))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(KS, tr, "-", lw=2.8, color="tab:blue", label="훈련 $R^2$")
    ax1.plot(KS, adj, "-", lw=2.8, color="tab:orange", label="수정 $R^2$")
    ax1.plot(KS, te, "-", lw=2.8, color="tab:red", label="검정 $R^2$")
    ax1.axhline(0, color="0.2", lw=1.6)
    ax1.set_ylim(-3.2, 1.15)
    ax1.set_xlabel("더해 넣은 잡음 변수의 수", fontsize=11)
    ax1.set_ylabel("$R^2$", fontsize=11)
    ax1.set_title(f"$n = {N}$ · {REPS} 번 평균", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="lower left", framealpha=0.95)
    ax1.grid(alpha=0.25)
    ax1.text(0.97, 0.92, "검정 $R^2$ 가 음수라는 것은\n"
             "평균 하나를 찍는 것보다 나쁘다는 뜻이다",
             transform=ax1.transAxes, ha="right", va="top", fontsize=10.5,
             fontweight="bold", color="tab:red",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0))

    COLS = ["tab:blue", "tab:orange", "tab:red"]
    for (nm, v), col in zip(pick.items(), COLS):
        ax2.hist(v, bins=np.arange(-0.5, KS.max() + 1.5, 1), histtype="step",
                 lw=2.4, color=col, label=f"{nm} — 중앙값 {np.median(v):.0f} 개")
    ax2.set_xlim(-1, KS.max() + 1)
    ax2.set_xlabel("그 자가 고른 잡음 변수의 수", fontsize=11)
    ax2.set_ylabel("횟수", fontsize=11)
    ax2.set_title("세 자에게 모형을 고르게 하면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper center", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("수정 결정계수는 병을 덜어 줄 뿐 고치지 못한다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PERF + "three_rulers_three_models.png")

    print(f"\n[결정계수] n = {N} · {REPS} 번")
    for k in (0, 10, 20, 40, 45):
        i = int(np.where(KS == k)[0][0])
        print(f"  잡음 {k:>2d} 개  훈련 {tr[i]:+.4f}  수정 {adj[i]:+.4f}  "
              f"검정 {te[i]:+.4f}")
    for nm, v in pick.items():
        print(f"  {nm.replace('$','')} 가 고른 수: 중앙값 {np.median(v):.0f} · "
              f"0 개를 고른 비율 {100 * np.mean(np.array(v) == 0):.1f}%")


# ════════════════════════════════════════════════════════════════════
def fig_two_residuals():
    """11.7 성능 척도 — 큰 잔차 둘이 MSE 의 절반을 들고 있다.

    쪽의 metric_disagreement 는 네 자가 엇갈리는 표를 그리고,
    aggregate_vs_person 상자는 분포와 개인을 가른다. 여기서는 들머리의
    둘째 문단 --- **엇갈림이 큰 잔차 둘에서 온다** --- 만 본다.
    자료는 이 책이 만든 것이다.
    """
    rng = np.random.default_rng(1171)
    n_tr, n_te = 120, 120
    x1 = rng.normal(0, 1, n_tr)
    e1 = rng.normal(0, 0.64, n_tr)
    e1[[7, 53]] = [5.0, -4.8]                  # 큰 잔차 둘을 심는다
    y1 = 1 + 2 * x1 + e1
    x2 = rng.normal(0, 1, n_te)
    y2 = 1 + 2 * x2 + rng.normal(0, 0.8, n_te)

    r = stats.linregress(x1, y1)
    rtr = y1 - (r.intercept + r.slope * x1)
    rte = y2 - (r.intercept + r.slope * x2)

    def mse(v):
        return (v ** 2).mean()

    def mae(v):
        return np.abs(v).mean()

    order = np.argsort(np.abs(rtr))[::-1]
    share_mse = np.cumsum(rtr[order] ** 2) / (rtr ** 2).sum()
    share_mae = np.cumsum(np.abs(rtr[order])) / np.abs(rtr).sum()
    keep = np.setdiff1d(np.arange(n_tr), order[:2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    kk = np.arange(1, 21)
    ax1.plot(kk, 100 * share_mse[:20], "o-", ms=7, lw=2.6, color="tab:red",
             label="MSE 에서 차지하는 몫")
    ax1.plot(kk, 100 * share_mae[:20], "s-", ms=7, lw=2.6, color="tab:blue",
             label="MAE 에서 차지하는 몫")
    ax1.plot([2], [100 * share_mse[1]], "o", ms=14, mfc="none",
             mec="tab:red", mew=2.4)
    ax1.annotate(f"가장 큰 둘이 MSE 의 {100 * share_mse[1]:.1f}%\n"
                 f"— MAE 에서는 {100 * share_mae[1]:.1f}% 뿐이다",
                 xy=(2, 100 * share_mse[1]), xytext=(6.5, 86), ha="left",
                 va="top", fontsize=11, color="tab:red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_ylim(0, 100)
    ax1.set_xlabel(f"큰 잔차부터 센 개수 (훈련 관측 {n_tr} 개 가운데)",
                   fontsize=11)
    ax1.set_ylabel("누적 몫 (%)", fontsize=11)
    ax1.set_title("제곱은 큰 것을 제 크기의 제곱만큼 센다", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="lower right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    GROUPS = [("MSE", mse(rtr), mse(rte), mse(rtr[keep])),
              ("MAE", mae(rtr), mae(rte), mae(rtr[keep]))]
    xs = np.arange(2)
    w = 0.26
    ax2.bar(xs - w, [g[1] for g in GROUPS], width=w * 0.92, color="tab:red",
            alpha=0.88, label="훈련")
    ax2.bar(xs, [g[2] for g in GROUPS], width=w * 0.92, color="tab:blue",
            alpha=0.88, label="시험")
    ax2.bar(xs + w, [g[3] for g in GROUPS], width=w * 0.92,
            color="0.62", alpha=0.9, label="훈련 — 큰 둘을 빼고")
    for i, g in enumerate(GROUPS):
        for j, v in enumerate(g[1:]):
            ax2.text(i + (j - 1) * w, v + 0.03, f"{v:.3f}", ha="center",
                     va="bottom", fontsize=10.5, fontweight="bold",
                     color="0.15")
    ax2.set_xticks(xs, ["MSE", "MAE"], fontsize=13)
    ax2.set_ylim(0, 1.55)
    ax2.set_ylabel("값", fontsize=11)
    ax2.set_title("MSE 만 「시험이 더 좋다」고 말한다", fontsize=12.5)
    ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("자가 엇갈린 자리를 들여다보면 큰 잔차 둘이 들어 있다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PERF + "two_residuals_carry_it.png")

    print(f"\n[성능 척도] 이 책이 만든 자료 (훈련 {n_tr} · 시험 {n_te})")
    print(f"  MSE 훈련 {mse(rtr):.6f} · 시험 {mse(rte):.6f}  "
          f"({'시험이 더 좋다' if mse(rte) < mse(rtr) else '훈련이 더 좋다'})")
    print(f"  MAE 훈련 {mae(rtr):.6f} · 시험 {mae(rte):.6f}  "
          f"({'시험이 더 좋다' if mae(rte) < mae(rtr) else '훈련이 더 좋다'})")
    print(f"  가장 큰 잔차 둘이 MSE 의 {100 * share_mse[1]:.2f}% · "
          f"MAE 의 {100 * share_mae[1]:.2f}%")
    print(f"  그 둘을 빼면 훈련 MSE {mse(rtr[keep]):.6f} → 역전이 사라진다: "
          f"{mse(rte) > mse(rtr[keep])}")


# ════════════════════════════════════════════════════════════════════
def fig_ratio_tells_the_shape():
    """11.7 MAE·MSE·RMSE — 둘의 비가 잔차의 모양을 말해 준다.

    쪽의 mae_vs_mse·mae_vs_rmse 는 두 자의 모양과 최솟값을 그린다.
    여기서는 들머리의 마지막 문단 --- **RMSE/MAE 비가 잔차가 고른지
    들쭉날쭉한지를 말해 준다** --- 만 본다. 비는 1 에서 시작해
    잔차 하나가 전부를 차지하면 sqrt(n) 까지 간다.
    """
    E = np.array([-1.0, 1.0, 2.0, -1.0, 6.0])     # 쪽이 드는 다섯
    mae0 = np.abs(E).mean()
    rmse0 = np.sqrt((E ** 2).mean())

    SHAPES = [("모두 같다", np.ones(200)),
              ("고르게 퍼졌다", np.linspace(0.05, 1, 200)),
              ("정규", None), ("두꺼운 꼬리", None), ("하나가 다 한다", None)]
    rng = np.random.default_rng(1172)
    vals = []
    for nm, v in SHAPES:
        if nm == "정규":
            v = np.abs(rng.normal(0, 1, 200_000))
        elif nm == "두꺼운 꼬리":
            v = np.abs(rng.standard_t(2.5, 200_000))
        elif nm == "하나가 다 한다":
            v = np.zeros(200)
            v[0] = 1.0
            v[1:] = 1e-6
        vals.append((nm, np.sqrt((v ** 2).mean()) / np.abs(v).mean()))

    ns = np.arange(2, 201)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    xs = np.arange(len(vals))
    cols = ["tab:green", "tab:cyan", "tab:blue", "tab:orange", "tab:red"]
    ax1.bar(xs, [v for _, v in vals], width=0.56, color=cols, alpha=0.88)
    for x_, (nm, v) in zip(xs, vals):
        ax1.text(x_, v + 0.1, f"{v:.3f}", ha="center", va="bottom",
                 fontsize=11.5, fontweight="bold", color="0.15")
    ax1.axhline(1, color="0.2", lw=1.8)
    ax1.axhline(rmse0 / mae0, color="tab:purple", ls="--", lw=2.0)
    ax1.text(4.44, rmse0 / mae0 + 0.08,
             f"쪽의 다섯 잔차 {rmse0 / mae0:.3f}", ha="right", va="bottom",
             fontsize=10.5, color="tab:purple", fontweight="bold")
    ax1.set_xticks(xs, [nm for nm, _ in vals], fontsize=10.5)
    ax1.set_ylim(0, 15.5)
    ax1.set_ylabel("RMSE ÷ MAE", fontsize=11)
    ax1.set_title("잔차의 모양마다 비가 다르다 (관측 200 개 기준)",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)

    ax2.plot(ns, np.sqrt(ns), "-", lw=2.8, color="tab:red",
             label="하나가 다 할 때의 상한  $\\sqrt{n}$")
    ax2.axhline(np.sqrt(np.pi / 2), color="tab:blue", ls="--", lw=2.2,
                label=f"정규 잔차  $\\sqrt{{\\pi/2}} = "
                      f"{np.sqrt(np.pi / 2):.4f}$")
    ax2.axhline(1, color="0.2", lw=2.0, label="모두 같을 때  $1$")
    ax2.plot([5], [rmse0 / mae0], "o", ms=13, color="tab:purple", zorder=5)
    ax2.annotate(f"쪽의 다섯 잔차\n$-1,\\ 1,\\ 2,\\ -1,\\ 6$\n"
                 f"{rmse0:.2f} / {mae0:.1f} = {rmse0 / mae0:.3f}",
                 xy=(5, rmse0 / mae0), xytext=(16, 4.6), fontsize=11,
                 color="tab:purple", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="tab:purple",
                                 lw=1.8))
    ax2.set_xscale("log")
    ax2.set_xticks([2, 5, 10, 20, 50, 100, 200],
                   ["2", "5", "10", "20", "50", "100", "200"])
    ax2.set_xticks([], minor=True)
    ax2.set_ylim(0, 15.5)
    ax2.set_xlabel("관측 수 n (로그 눈금)", fontsize=11)
    ax2.set_ylabel("RMSE ÷ MAE", fontsize=11)
    ax2.set_title("비가 놓일 수 있는 자리는 $1$ 과 $\\sqrt{n}$ 사이다",
                  fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("RMSE 는 언제나 MAE 보다 크거나 같고, 그 비가 모양을 "
                 "말해 준다", fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PERF + "ratio_tells_the_shape.png")

    print(f"\n[MAE·RMSE] 쪽의 다섯 잔차 MAE {mae0:.4f} · RMSE {rmse0:.4f} · "
          f"비 {rmse0 / mae0:.4f}")
    for nm, v in vals:
        print(f"  {nm:<12s} {v:.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_mape_picks_low():
    """11.7 상대 척도 — MAPE 가 고르는 상수는 왜 낮은가.

    쪽의 mape_asymmetry 는 과소예측과 과대예측의 벌점이 어긋나는 것을,
    metric_decides_winner 상자는 순위가 뒤집히는 자리를 다룬다. 여기서는
    들머리의 셋째 문단 --- **MAPE 를 가장 작게 하는 상수가 평균도
    중앙값도 아닌 0.37 이라는 것** --- 만 본다.
    """
    SIG = 1.0                        # 대수정규 LN(0, SIG^2)
    rng = np.random.default_rng(1173)
    y = rng.lognormal(0, SIG, 400_000)
    c = np.linspace(0.05, 4.0, 500)

    mse = np.array([((y - v) ** 2).mean() for v in c])
    mae = np.array([np.abs(y - v).mean() for v in c])
    mape = np.array([(np.abs(y - v) / y).mean() for v in c])
    best = [c[int(np.argmin(m))] for m in (mse, mae, mape)]

    sigs = np.linspace(0.2, 1.6, 40)
    opt = []
    for s in sigs:
        yy = rng.lognormal(0, s, 120_000)
        cc = np.linspace(0.02, 2.0, 300)
        mp = np.array([(np.abs(yy - v) / yy).mean() for v in cc])
        opt.append(cc[int(np.argmin(mp))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    for m, nm, col in ((mse / mse.min(), "MSE", "tab:blue"),
                       (mae / mae.min(), "MAE", "tab:green"),
                       (mape / mape.min(), "MAPE", "tab:red")):
        ax1.plot(c, m, "-", lw=2.8, color=col, label=nm)
    for v, nm, col in zip(best, ("평균", "중앙값", "그 아래"),
                          ("tab:blue", "tab:green", "tab:red")):
        ax1.axvline(v, color=col, ls="--", lw=1.8)
        ax1.text(v, 2.47, f"{v:.2f}\n{nm}", ha="center", va="top",
                 fontsize=11, color=col, fontweight="bold")
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0.95, 2.6)
    ax1.set_xlabel("상수 하나로 예측한다면 그 상수 c", fontsize=11)
    ax1.set_ylabel("각 자의 값 (최솟값을 1 로 맞췄다)", fontsize=11)
    ax1.set_title(f"반응이 대수정규 LN(0, {SIG:.0f}) 일 때", fontsize=12.5)
    ax1.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(sigs, opt, "o-", ms=6, lw=2.4, color="tab:red",
             label="MAPE 가 고르는 상수")
    ax2.plot(sigs, np.ones_like(sigs), "-", lw=2.4, color="tab:green",
             label="중앙값 (MAE 가 고르는 값)")
    ax2.plot(sigs, np.exp(sigs ** 2 / 2), "-", lw=2.4, color="tab:blue",
             label="평균 (MSE 가 고르는 값)")
    ax2.plot([SIG], [best[2]], "o", ms=13, mfc="none", mec="tab:red",
             mew=2.4)
    ax2.set_xlim(0.2, 1.6)
    ax2.set_ylim(0, 3.6)
    ax2.set_xlabel("대수정규의 σ — 오른쪽으로 갈수록 치우친 자료", fontsize=11)
    ax2.set_ylabel("자가 고르는 상수", fontsize=11)
    ax2.set_title("치우칠수록 MAPE 는 더 낮게 찍는다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("낮게 찍을수록 벌점이 작아지니 낮게 찍는 쪽이 이긴다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, PERF + "mape_picks_low.png")

    print(f"\n[상대 척도] LN(0, {SIG}) 에서 상수 하나로 예측하면")
    for v, nm in zip(best, ("MSE", "MAE", "MAPE")):
        print(f"  {nm:<5s} 가 고르는 상수 {v:.4f}")
    print(f"  이론값: 평균 {np.exp(SIG ** 2 / 2):.4f} · 중앙값 1.0000")


# ════════════════════════════════════════════════════════════════════
def fig_knots_beat_degree():
    """11.8 스플라인 들머리 — 차수를 올리는 대신 매듭을 쓴다.

    쪽의 smoothing_tradeoff 는 매끄러움의 절충을 그린다. 여기서는
    들머리의 첫 문단 --- **차수를 높이면 다른 곳에서 진동이 생기는데
    매듭으로 끊으면 그렇지 않다** --- 만 본다.
    """
    f = lambda v: 1 / (1 + 25 * v ** 2)          # 룽게 함수
    xs = np.linspace(-1, 1, 21)
    ys = f(xs)
    g = np.linspace(-1, 1, 801)

    def spline_fit(x, y, knots, gg):
        def basis(v):
            cols = [np.ones_like(v), v, v ** 2, v ** 3]
            cols += [np.clip(v - k, 0, None) ** 3 for k in knots]
            return np.column_stack(cols)
        b = np.linalg.lstsq(basis(x), y, rcond=None)[0]
        return basis(gg) @ b

    KNOTS = np.linspace(-0.8, 0.8, 9)
    poly15 = np.polyval(np.polyfit(xs, ys, 15), g)
    spl = spline_fit(xs, ys, KNOTS, g)

    degs = np.arange(3, 19)
    err_poly = [np.max(np.abs(np.polyval(np.polyfit(xs, ys, d), g) - f(g)))
                for d in degs]
    ks = np.arange(1, 17)
    err_spl = [np.max(np.abs(spline_fit(xs, ys, np.linspace(-0.85, 0.85, k),
                                        g) - f(g))) for k in ks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    ax1.plot(g, f(g), "-", lw=2.6, color="0.25", label="참 곡선")
    ax1.plot(g, poly15, "-", lw=2.4, color="tab:red",
             label="전 범위에 $15$ 차 다항식 하나")
    ax1.plot(g, spl, "-", lw=2.4, color="tab:blue",
             label=f"매듭 {len(KNOTS)} 개로 끊은 $3$ 차 스플라인")
    ax1.plot(xs, ys, "o", ms=7, color="0.35", zorder=5)
    for k in KNOTS:
        ax1.axvline(k, color="tab:blue", lw=0.8, alpha=0.3)
    ax1.set_ylim(-0.6, 1.6)
    ax1.set_xlabel("x", fontsize=11)
    ax1.set_ylabel("y", fontsize=11)
    ax1.set_title("점 스물한 개를 두 가지로 맞췄다", fontsize=12.5)
    ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    ax2.plot(degs, err_poly, "o-", ms=7, lw=2.6, color="tab:red",
             label="차수를 올린다 (전 범위 하나)")
    ax2.plot(ks + 3, err_spl, "s-", ms=7, lw=2.6, color="tab:blue",
             label="매듭을 늘린다 ($3$ 차 스플라인)")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-3, 60)
    ax2.set_yticks([1e-3, 1e-2, 1e-1, 1, 10],
                   ["0.001", "0.01", "0.1", "1", "10"])
    ax2.set_yticks([], minor=True)
    ax2.set_xlabel("쓴 모수의 수", fontsize=11)
    ax2.set_ylabel("참 곡선과의 최대 오차 (로그 눈금)", fontsize=11)
    ax2.set_title("모수를 같은 수만큼 써도 결과가 갈린다", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
    ax2.grid(alpha=0.25)

    fig.suptitle("한 구역을 맞추려고 차수를 올리면 다른 곳이 출렁인다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, SPL + "knots_beat_degree.png")

    print(f"\n[스플라인] 룽게 함수 · 점 {len(xs)} 개")
    print(f"  15 차 다항식의 최대 오차 "
          f"{np.max(np.abs(poly15 - f(g))):.4f}")
    print(f"  매듭 {len(KNOTS)} 개 스플라인의 최대 오차 "
          f"{np.max(np.abs(spl - f(g))):.4f}")


# ════════════════════════════════════════════════════════════════════
def fig_effect_spread_over_basis():
    """11.8 GAM — 효과가 기저 여러 열에 흩어진다.

    쪽의 backfitting_rounds 상자는 셈법을, 본문 그림 둘은 부분함수를
    그린다. 여기서는 들머리의 셋째 문단 --- **표의 어느 행도 효과를
    혼자 담지 못한다** --- 만 본다. 기저 계수 하나하나의 p 값과, 그것을
    다시 모은 곡선의 오르내림, 그리고 기저 전체를 한꺼번에 검정한 p 값을
    나란히 놓는다.
    """
    rng = np.random.default_rng(1174)
    n, K = 300, 10
    x1 = rng.uniform(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    y = 2.0 * x1 + 1.05 * np.sin(2 * np.pi * x2) + rng.normal(0, 0.5, n)

    knots = np.linspace(0, 1, K - 2)[1:-1]

    def basis(v):
        cols = [v, v ** 2, v ** 3]
        cols += [np.clip(v - k, 0, None) ** 3 for k in knots]
        return np.column_stack(cols)

    B = basis(x2)
    B = B - B.mean(axis=0)
    A = np.column_stack([np.ones(n), x1, B])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    dof = n - A.shape[1]
    s2 = (res ** 2).sum() / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(cov))
    tval = beta / se
    pval = 2 * stats.t.sf(np.abs(tval), dof)

    gg = np.linspace(0, 1, 400)
    Bg = basis(gg) - basis(x2).mean(axis=0)
    fhat = Bg @ beta[2:]
    rng_fhat = fhat.max() - fhat.min()

    A0 = np.column_stack([np.ones(n), x1])
    b0 = np.linalg.lstsq(A0, y, rcond=None)[0]
    rss0 = ((y - A0 @ b0) ** 2).sum()
    rss1 = (res ** 2).sum()
    q = A.shape[1] - A0.shape[1]
    F = ((rss0 - rss1) / q) / (rss1 / dof)
    p_joint = stats.f.sf(F, q, dof)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9))

    idx = np.arange(2, len(beta))
    ax1.bar(np.arange(len(idx)), pval[idx], width=0.56, color="tab:orange",
            alpha=0.88)
    ax1.axhline(0.05, color="tab:red", ls="--", lw=1.8)
    ax1.text(len(idx) - 0.5, 0.07, "$0.05$", ha="right", va="bottom",
             fontsize=11, color="tab:red", fontweight="bold")
    ax1.set_xticks(np.arange(len(idx)),
                   [f"기저\n{i + 1}" for i in range(len(idx))], fontsize=9.5)
    ax1.set_ylim(0, 1.30)
    ax1.set_ylabel("그 행 하나의 $p$ 값", fontsize=11)
    ax1.set_title(f"계수표의 기저 {len(idx)} 행 — 혼자서는 아무것도 아니다",
                  fontsize=12.5)
    ax1.grid(axis="y", alpha=0.25)
    ax1.text(0.5, 0.94, f"{(pval[idx] > 0.05).sum()} 행이 $0.05$ 위다",
             transform=ax1.transAxes, ha="center", va="top", fontsize=11.5,
             fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    ax2.plot(gg, 1.05 * np.sin(2 * np.pi * gg), "--", lw=2.4, color="0.35",
             label="참 곡선")
    ax2.plot(gg, fhat, "-", lw=3.0, color="tab:blue",
             label="기저 계수를 다시 모은 곡선")
    ax2.annotate("", xy=(gg[int(np.argmax(fhat))], fhat.max()),
                 xytext=(gg[int(np.argmax(fhat))], fhat.min()),
                 arrowprops=dict(arrowstyle="<->", color="tab:red", lw=2.0))
    ax2.text(gg[int(np.argmax(fhat))] + 0.03, 0,
             f"오르내림 폭\n{rng_fhat:.4f}", ha="left", va="center",
             fontsize=11.5, color="tab:red", fontweight="bold")
    ax2.set_xlabel("$x_2$", fontsize=11)
    ax2.set_ylabel("부분함수", fontsize=11)
    ax2.set_title("흩어진 것을 다시 모으면", fontsize=12.5)
    ax2.legend(fontsize=10.5, loc="lower right", framealpha=0.95)
    ax2.grid(alpha=0.25)
    ax2.text(0.03, 0.04, f"기저 {len(idx)} 열을 한꺼번에 검정하면\n"
             f"$F = {F:.1f}$, $p$ 값이 $10^{{{int(np.floor(np.log10(p_joint)))}}}$ "
             f"보다 작다", transform=ax2.transAxes, ha="left", va="bottom",
             fontsize=11, fontweight="bold", color="0.15",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.5", lw=1.0))

    fig.suptitle("매끄러운 항의 효과는 표가 아니라 그림에서 읽는다",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    save(fig, SPL + "effect_spread_over_basis.png")

    print(f"\n[GAM] n = {n} · 기저 {len(idx)} 개")
    print(f"  기저 행의 p 값 최솟값 {pval[idx].min():.4f} · "
          f"0.05 위인 행 {(pval[idx] > 0.05).sum()} 개")
    print(f"  되찾은 부분함수의 오르내림 폭 {rng_fhat:.4f}")
    print(f"  기저 전체를 한꺼번에 검정: F = {F:.2f}, p = {p_joint:.3e}")


fig_three_rulers()
fig_two_residuals()
fig_ratio_tells_the_shape()
fig_mape_picks_low()
fig_knots_beat_degree()
fig_effect_spread_over_basis()
