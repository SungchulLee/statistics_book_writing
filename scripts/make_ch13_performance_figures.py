r"""13장 '성능 척도' 네 쪽에 들어가는 개념 그림을 만든다.

만드는 파일:
  ch13/performance/img/r2_inflation.png        잡음 변수를 넣으면 R^2 가 오른다
  ch13/performance/img/mae_vs_rmse.png         이상점 하나가 MSE 를 지배한다
  ch13/performance/img/metric_disagreement.png MAE 와 RMSE 가 다른 모형을 고른다
  ch13/performance/img/mape_asymmetry.png      MAPE 는 과소예측을 편든다

실행:  python3 scripts/make_ch13_performance_figures.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
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

OUT = "docs/ch13/performance/img/"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    path = OUT + name
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def clean(ax):
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)


# === 1. 잡음 변수를 넣으면 R^2 가 오른다 ===
def fig_r2_inflation():
    rng = np.random.default_rng(5)
    n = 50
    kmax = 48                                 # p = 1 + kmax = n - 1 이면 완전보간
    x_tr = rng.normal(0, 1, n)
    x_te = rng.normal(0, 1, n)
    y_tr = 1 + 2 * x_tr + rng.normal(0, 1, n)
    y_te = 1 + 2 * x_te + rng.normal(0, 1, n)
    Z_tr = rng.normal(0, 1, (n, kmax))       # 순수한 잡음 설명변수
    Z_te = rng.normal(0, 1, (n, kmax))

    ks = np.arange(0, kmax + 1, 2)
    r2_tr, r2_adj, r2_te = [], [], []
    sst_tr = ((y_tr - y_tr.mean()) ** 2).sum()
    sst_te = ((y_te - y_te.mean()) ** 2).sum()
    fits = {}
    for k in ks:
        X_tr = np.column_stack([np.ones(n), x_tr, Z_tr[:, :k]])
        X_te = np.column_stack([np.ones(n), x_te, Z_te[:, :k]])
        b = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
        f_tr, f_te = X_tr @ b, X_te @ b
        p = 1 + k
        R2 = 1 - ((y_tr - f_tr) ** 2).sum() / sst_tr
        r2_tr.append(R2)
        r2_adj.append(1 - (1 - R2) * (n - 1) / (n - p - 1)
                      if n - p - 1 > 0 else np.nan)
        r2_te.append(1 - ((y_te - f_te) ** 2).sum() / sst_te)
        fits[k] = (f_tr, f_te)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))

    ax = axes[0]
    ps = 1 + ks
    ax.axhline(0, color=MUTED, lw=1.2)
    ax.plot(ps, r2_tr, color=BLUE, lw=2.4, marker="o", ms=5,
            label=r"훈련 $R^2$")
    ax.plot(ps, r2_adj, color=GREEN, lw=2.4, marker="s", ms=5,
            label=r"수정 $R^2$")
    ax.plot(ps, r2_te, color=RED, lw=2.4, marker="^", ms=5,
            label=r"검정 $R^2$")
    ax.set_ylim(-3.0, 1.3)
    ax.set_xlabel("설명변수의 개수 p  (첫 하나만 진짜, 나머지는 잡음)",
                  fontsize=10, color=INK)
    ax.set_ylabel(r"$R^2$", fontsize=11, color=INK)
    ax.set_title(r"쓸모없는 변수를 더해도 훈련 $R^2$ 는 오른다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.6, loc="lower left", frameon=False)
    ax.annotate(f"p=1: {r2_tr[0]:.3f}", xy=(1, r2_tr[0]), xytext=(4.5, 0.42),
                fontsize=9.6, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate(f"p={ps[-1]}: {r2_tr[-1]:.3f}", xy=(ps[-1], r2_tr[-1]),
                xytext=(22, 1.05), fontsize=9.6, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2))
    ax.annotate(f"p={ps[-1]} 에서 {r2_te[-1]:.1f}\n(화면 아래로 벗어난다)",
                xy=(ps[-1] - 0.4, -2.9), xytext=(16, -2.35), fontsize=9.6,
                color=RED, linespacing=1.5, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    clean(ax)

    ax = axes[1]
    f_tr, f_te = fits[kmax]
    lim = 26
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=1.6, ls="--")
    ax.scatter(f_te, y_te, s=30, color=RED, alpha=0.8, edgecolor="none",
               label="검정자료")
    ax.scatter(f_tr, y_tr, s=30, color=BLUE, alpha=0.9, edgecolor="none",
               label="훈련자료")
    ax.set_xlabel(r"예측값 $\hat y$", fontsize=10.5, color=INK)
    ax.set_ylabel(r"실제값 $y$", fontsize=10.5, color=INK)
    ax.set_title(f"p = {1+kmax} 인 모형이 실제로 하는 일", fontsize=11.5,
                 color=INK, pad=6)
    out = int((np.abs(f_te) > lim).sum())
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=9.6, loc="upper left", frameon=False)
    ax.text(0.97, 0.06,
            "훈련점 50개는 대각선 위에 정확히 얹히고\n"
            f"검정점은 사방으로 흩어진다 ({out}개는 화면 밖)",
            transform=ax.transAxes, fontsize=9.6, color=INK, ha="right",
            va="bottom", linespacing=1.6)
    clean(ax)

    fig.tight_layout()
    save(fig, "r2_inflation.png")
    for k, a, b, c in zip(ks, r2_tr, r2_adj, r2_te):
        if k % 8 == 0 or k == kmax:
            print(f"  p={1+k:3d}  훈련R2={a:.4f}  수정R2={b:+.4f}  검정R2={c:+.4f}")


# === 2. 이상점 하나가 MSE 를 지배한다 ===
def fig_mae_rmse():
    e = np.array([-1.0, 1.0, 2.0, -1.0, 6.0])
    abs_e, sq_e = np.abs(e), e ** 2
    mae, mse = abs_e.mean(), sq_e.mean()
    rmse = np.sqrt(mse)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    cols = [MUTED] * 4 + [RED]
    labels = [f"{i+1}번" for i in range(5)]
    left = 0.0
    for i in range(5):
        ax.barh(1, abs_e[i] / abs_e.sum(), left=left, height=0.5,
                color=cols[i], edgecolor="white", lw=1.4)
        ax.text(left + abs_e[i] / abs_e.sum() / 2, 1, labels[i],
                ha="center", va="center", fontsize=9.4, color="white")
        left += abs_e[i] / abs_e.sum()
    left = 0.0
    for i in range(5):
        ax.barh(0, sq_e[i] / sq_e.sum(), left=left, height=0.5,
                color=cols[i], edgecolor="white", lw=1.4)
        if sq_e[i] / sq_e.sum() > 0.05:
            ax.text(left + sq_e[i] / sq_e.sum() / 2, 0, labels[i],
                    ha="center", va="center", fontsize=9.4, color="white")
        left += sq_e[i] / sq_e.sum()
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"MSE 에서 차지하는 몫  $e_i^2$",
                        r"MAE 에서 차지하는 몫  $|e_i|$"], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9)
    ax.text(1 - 6 / 11 / 2, 1.42, f"{6/11:.0%}", ha="center", fontsize=11,
            color=RED)
    ax.text(1 - 36 / 43 / 2, -0.58, f"{36/43:.0%}", ha="center", fontsize=11,
            color=RED)
    ax.set_ylim(-0.85, 1.75)
    ax.set_title(r"잔차 $e=(-1,1,2,-1,6)$ 에서 5번 점의 몫", fontsize=11.5,
                 color=INK, pad=6)
    ax.tick_params(labelsize=9, colors=INK)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)

    # 오른쪽: 오차 분포 모양별 RMSE/MAE 비
    ax = axes[1]
    rng = np.random.default_rng(1)
    m = 400000
    shapes = [
        ("모든 잔차가 같은 크기", np.ones(m)),
        ("균등분포", rng.uniform(-1, 1, m)),
        ("정규분포", rng.normal(0, 1, m)),
        ("위 보기 (이상점 하나)", e),
        ("라플라스분포", rng.laplace(0, 1, m)),
        (r"$t_3$ 분포", rng.standard_t(3, m)),
    ]
    names = [s[0] for s in shapes]
    ratios = [np.sqrt((v ** 2).mean()) / np.abs(v).mean() for s, v in shapes]
    ycs = np.arange(len(names))
    bcols = [MUTED, GREEN, BLUE, RED, ORANGE, PURPLE]
    bars = ax.barh(ycs, ratios, color=bcols, height=0.6)
    for yc, r in zip(ycs, ratios):
        ax.text(r + 0.015, yc, f"{r:.3f}", va="center", fontsize=10,
                color=INK)
    ax.axvline(1.0, color=MUTED, lw=1.4, ls="--")
    ax.set_yticks(ycs)
    ax.set_yticklabels(names, fontsize=9.8)
    ax.set_xlim(0, max(ratios) * 1.22)
    ax.set_xlabel("RMSE / MAE", fontsize=10.5, color=INK)
    ax.set_title("이 비 자체가 진단 정보다", fontsize=11.5, color=INK, pad=6)
    ax.text(1.02, len(names) - 0.55, "1 보다 작을 수 없다", fontsize=9.4,
            color=INK)
    clean(ax)

    fig.tight_layout()
    save(fig, "mae_vs_rmse.png")
    print(f"  MAE={mae:.2f} MSE={mse:.2f} RMSE={rmse:.4f} 비={rmse/mae:.4f}")
    print(f"  5번 점: MAE 의 {6/11:.1%}, MSE 의 {36/43:.1%}")
    for nm, r in zip(names, ratios):
        print(f"  {nm}: {r:.4f}")
    print(f"  정규의 이론값 sqrt(pi/2) = {np.sqrt(np.pi/2):.4f}")


# === 3. MAE 와 RMSE 가 다른 모형을 고른다 ===
def fig_disagreement():
    rng = np.random.default_rng(20)
    m = 200000
    # 모형 A: 정규 오차 — 가끔 큰 오차가 난다
    eA = rng.normal(0, 1, m)
    eA = eA / np.sqrt((eA ** 2).mean()) * 4.0
    # 모형 B: 오차 크기가 늘 3 언저리로 고르다
    eB = rng.uniform(0.867, 6.133, m) * rng.choice([-1, 1], m)

    def met(v):
        return np.abs(v).mean(), np.sqrt((v ** 2).mean()), (v ** 2).mean()

    maeA, rmseA, mseA = met(eA)
    maeB, rmseB, mseB = met(eB)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    bins = np.linspace(-14, 14, 90)
    ax.hist(eA, bins=bins, density=True, histtype="stepfilled",
            color=BLUE_F, edgecolor=BLUE, lw=1.6, label="모형 A")
    ax.hist(eB, bins=bins, density=True, histtype="step",
            color=ORANGE, lw=2.0, label="모형 B")
    ax.axvspan(-14, -8, color=RED, alpha=0.07)
    ax.axvspan(8, 14, color=RED, alpha=0.07)
    ax.text(11, 0.072, "모형 A 만\n여기에 산다", fontsize=9.6, color=RED,
            ha="center", linespacing=1.5)
    ax.set_xlabel("예측오차", fontsize=10.5, color=INK)
    ax.set_ylabel("밀도", fontsize=10.5, color=INK)
    ax.set_title("같은 자료를 맞힌 두 모형의 오차 분포", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.8, loc="upper left", frameon=False)
    clean(ax)

    ax = axes[1]
    idx = np.array([0, 1])
    w = 0.34
    ax.bar(idx - w / 2, [maeA, rmseA], w, color=BLUE, label="모형 A")
    ax.bar(idx + w / 2, [maeB, rmseB], w, color=ORANGE, label="모형 B")
    for i, (a, b) in enumerate([(maeA, maeB), (rmseA, rmseB)]):
        ax.text(i - w / 2, a + 0.08, f"{a:.2f}", ha="center", fontsize=10.5,
                color=INK)
        ax.text(i + w / 2, b + 0.08, f"{b:.2f}", ha="center", fontsize=10.5,
                color=INK)
        win = "A" if a < b else "B"
        ax.text(i, 4.95, f"{win} 승", ha="center", fontsize=12, color=GREEN)
    ax.set_xticks(idx)
    ax.set_xticklabels(["MAE", "RMSE"], fontsize=12)
    ax.set_ylim(0, 5.6)
    ax.set_ylabel("오차 크기", fontsize=10.5, color=INK)
    ax.set_title("어느 자로 재느냐에 따라 순위가 뒤집힌다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=9.8, loc="center", frameon=False, ncol=1)
    clean(ax)

    fig.tight_layout()
    save(fig, "metric_disagreement.png")
    print(f"  모형 A: MAE={maeA:.3f} MSE={mseA:.2f} RMSE={rmseA:.3f} "
          f"비={rmseA/maeA:.3f}")
    print(f"  모형 B: MAE={maeB:.3f} MSE={mseB:.2f} RMSE={rmseB:.3f} "
          f"비={rmseB/maeB:.3f}")


# === 4. MAPE 는 과소예측을 편든다 ===
def fig_mape():
    y = 100.0
    yh = np.linspace(0, 400, 800)
    mape = np.abs(y - yh) / y * 100
    smape = 2 * np.abs(y - yh) / (np.abs(y) + np.abs(yh) + 1e-12) * 100

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))

    ax = axes[0]
    ax.plot(yh, mape, color=ORANGE, lw=2.6, label="MAPE")
    ax.plot(yh, smape, color=BLUE, lw=2.6, label="sMAPE")
    ax.axvline(y, color=MUTED, lw=1.4, ls=":")
    ax.axhline(100, color=MUTED, lw=1.2, ls="--")
    ax.text(y + 6, 290, r"참값 $y=100$", fontsize=9.8, color=INK)
    ax.scatter([0, 200], [100, 100], s=45, color=ORANGE, zorder=6)
    ax.annotate("과소예측은 아무리 심해도 100%",
                xy=(2, 100), xytext=(38, 172), fontsize=9.6, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.annotate("과대예측은 천장이 없다",
                xy=(370, 270), xytext=(150, 232), fontsize=9.6, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_ylim(0, 310)
    ax.set_xlim(0, 400)
    ax.set_xlabel(r"예측값 $\hat y$", fontsize=10.5, color=INK)
    ax.set_ylabel("백분율 오차 (%)", fontsize=10.5, color=INK)
    ax.set_title("같은 크기의 빗나감인데 벌점이 다르다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=10, loc="upper left", frameon=False)
    clean(ax)

    # 오른쪽: 치우친 반응에서 각 척도가 고르는 상수 예측
    rng = np.random.default_rng(7)
    Y = np.exp(rng.normal(0, 1, 400000))       # 로그정규
    cs = np.linspace(0.05, 2.6, 500)
    f_mape = np.array([np.mean(np.abs(Y - c) / Y) for c in cs])
    f_mae = np.array([np.mean(np.abs(Y - c)) for c in cs])
    f_mse = np.array([np.mean((Y - c) ** 2) for c in cs])
    c_mape, c_mae, c_mse = cs[f_mape.argmin()], cs[f_mae.argmin()], cs[f_mse.argmin()]

    ax = axes[1]
    ax.plot(cs, f_mape / f_mape.min(), color=ORANGE, lw=2.6, label="MAPE")
    ax.plot(cs, f_mae / f_mae.min(), color=GREEN, lw=2.6, label="MAE")
    ax.plot(cs, f_mse / f_mse.min(), color=BLUE, lw=2.6, label="MSE")
    for c, col, lab in [(c_mape, ORANGE, "MAPE"), (c_mae, GREEN, "MAE"),
                        (c_mse, BLUE, "MSE")]:
        ax.axvline(c, color=col, lw=1.4, ls=":")
    ax.scatter([c_mape, c_mae, c_mse], [1, 1, 1], s=55,
               color=[ORANGE, GREEN, BLUE], zorder=6)
    ax.text(c_mape, 0.875, f"{c_mape:.2f}", color=ORANGE, ha="center",
            fontsize=10.5)
    ax.text(c_mae, 0.875, f"{c_mae:.2f}", color=GREEN, ha="center",
            fontsize=10.5)
    ax.text(c_mse, 0.875, f"{c_mse:.2f}", color=BLUE, ha="center",
            fontsize=10.5)
    ax.text(0.12, 2.70, "점선 = 각 척도가 고르는 최적 상수 예측", fontsize=9.8,
            color=INK, ha="left")
    ax.set_ylim(0.84, 2.9)
    ax.set_xlim(0, 2.6)
    ax.set_xlabel(r"상수 예측값 $c$   (반응은 로그정규, 중앙값 1, 평균 1.65)",
                  fontsize=9.8, color=INK)
    ax.set_ylabel("최솟값 대비 손실", fontsize=10.5, color=INK)
    ax.set_title("그래서 MAPE 는 낮게 예측하는 쪽으로 끌린다", fontsize=11.5,
                 color=INK, pad=6)
    ax.legend(fontsize=10, loc="upper right", frameon=False)
    clean(ax)

    fig.tight_layout()
    save(fig, "mape_asymmetry.png")
    print(f"  y=100 에서 MAPE: yhat=0 -> {100.0:.0f}%, yhat=50 -> 50%, "
          f"yhat=150 -> 50%, yhat=300 -> 200%")
    print(f"  최적 상수: MAPE c={c_mape:.4f}, MAE c={c_mae:.4f} (중앙값 1), "
          f"MSE c={c_mse:.4f} (평균 {Y.mean():.4f})")
    print(f"  이론값: MAPE 최적 = exp(-1) = {np.exp(-1):.4f}")


if __name__ == "__main__":
    fig_r2_inflation()
    fig_mae_rmse()
    fig_disagreement()
    fig_mape()
