r"""11.6절 「그림으로 보는 통계학」 — 가로축을 하나 잘못 고르면.

이 쪽의 잔차그림은 가로축이 **적합값** $\hat y$ 다. 가로축에 **관측값** $y$ 를 놓으면
어떻게 되는가. 절편이 있는 최소제곱에서는 두 경우가 각각 식으로 적힌다.

  Cov(e, y_hat) = 0                     -> 잔차 대 적합값의 기울기는 정확히 0
  Cov(e, y) = Var(e)                    -> 잔차 대 관측값의 기울기는 정확히 1 - R^2
  corr(e, y) = SD(e)/SD(y) = sqrt(1-R^2)

곧 **관측값을 가로축에 놓으면 언제나 오른쪽으로 올라가는 띠가 나온다.** 모형이
무엇이든, 자료가 무엇이든, 잘못된 것이 하나도 없어도 그렇다. 그 기울기는 모형의
결함이 아니라 $1 - R^2$ 을 다시 적은 것이다.

이 스크립트가 재는 값:

  n = 200, R^2 = 0.7432 인 한 적합에서
    잔차 대 적합값  기울기 -3.4e-16   상관 -5.5e-16
    잔차 대 관측값  기울기 +0.2568    상관 +0.5067
    1 - R^2 = 0.2568,  sqrt(1 - R^2) = 0.5067

  R^2 를 바꿔 가며 (되풀이 400 번씩, 평균)
    잰 R^2   잔차 대 관측값의 기울기   1 - R^2
    0.1031          0.8969            0.8969
    0.2999          0.7001            0.7001
    0.5040          0.4960            0.4960
    0.6999          0.3001            0.3001
    0.9003          0.0997            0.0997

만드는 파일:

  ch11/diagnostics/img/residual_vs_observed.png

실행:  python3 scripts/make_ch13_residual_vs_observed.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/diagnostics/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 200
TARGETS = (0.10, 0.30, 0.50, 0.70, 0.90)
REPS = 400
SEED = 17


def fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ b
    e = y - yhat
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return yhat, e, float(r2)


def slope(u, v):
    uc = u - u.mean()
    return float((uc * (v - v.mean())).sum() / (uc ** 2).sum())


def corr(u, v):
    return float(np.corrcoef(u, v)[0, 1])


def make(rng, r2_target, n=N):
    """결정계수가 대략 r2_target 이 되도록 잡음을 맞춘 자료."""
    x = rng.normal(0, 1, n)
    s = np.sqrt((1 - r2_target) / r2_target)
    return x, 2.0 + 1.0 * x + rng.normal(0, s, n)


def main():
    rng = np.random.default_rng(SEED)
    x, y = make(rng, 0.70)
    yhat, e, r2 = fit(x, y)

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"n = {N}, R^2 = {r2:.4f} 인 한 적합에서")
    print(f"  잔차 대 적합값  기울기 {slope(yhat, e):+.1e}   "
          f"상관 {corr(yhat, e):+.1e}")
    print(f"  잔차 대 관측값  기울기 {slope(y, e):+.4f}   상관 {corr(y, e):+.4f}")
    print(f"  1 - R^2 = {1 - r2:.4f},  sqrt(1 - R^2) = {np.sqrt(1 - r2):.4f}")

    print(f"\nR^2 를 바꿔 가며 (되풀이 {REPS} 번씩, 평균)")
    print("   목표 R^2   잰 R^2   잔차 대 관측값의 기울기   1 - R^2")
    rows = []
    for t in TARGETS:
        ss, rr = [], []
        for k in range(REPS):
            r2rng = np.random.default_rng([SEED, int(t * 100), k])
            xx, yy = make(r2rng, t)
            _, ee, rr2 = fit(xx, yy)
            ss.append(slope(yy, ee))
            rr.append(rr2)
        rows.append((t, float(np.mean(rr)), float(np.mean(ss))))
        print(f"  {t:8.2f} {np.mean(rr):9.4f} {np.mean(ss):22.4f} "
              f"{1 - np.mean(rr):10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 같은 적합, 두 가로축 ──
    ax = axes[0]
    ax.plot(yhat, e, "o", ms=6, color=TEAL, alpha=0.7, label="가로축이 적합값")
    ax.plot(y, e, "o", ms=6, color=RED, alpha=0.55, label="가로축이 관측값")
    g = np.linspace(min(y.min(), yhat.min()) - 0.3,
                    max(y.max(), yhat.max()) + 0.3, 50)
    ax.plot(g, slope(yhat, e) * (g - yhat.mean()), lw=2.6, color=TEAL)
    ax.plot(g, slope(y, e) * (g - y.mean()), lw=2.6, color=RED)
    ax.axhline(0, color=MUTED, lw=1.0, ls=":")
    ax.annotate(f"기울기 ${slope(y, e):.4f} = 1 - R^2$",
                xy=(g[-12], slope(y, e) * (g[-12] - y.mean())),
                xytext=(0.97, 0.10), textcoords="axes fraction",
                ha="right", fontsize=11.4, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    # mathtext 가 1e-16 의 e 를 기울여 찍으므로 10 의 거듭제곱으로 적는다
    sl = slope(yhat, e)
    ex = int(np.floor(np.log10(abs(sl))))
    ax.text(0.03, 0.95, f"기울기 ${sl / 10 ** ex:+.1f} \\times 10^{{{ex}}}$",
            transform=ax.transAxes, va="top", fontsize=11.4, color=TEAL,
            fontweight="bold")
    ax.set_xlabel("가로축에 놓은 것", fontsize=11.5)
    ax.set_ylabel("잔차 $e$", fontsize=11.5)
    ax.set_title(f"① 같은 적합 ($R^2 = {r2:.3f}$), 다른 가로축",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.4, framealpha=0.95, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 기울기는 언제나 1 - R^2 ──
    ax = axes[1]
    gg = np.linspace(0.03, 0.97, 200)
    ax.plot(gg, 1 - gg, lw=2.6, color=INK, label="$1 - R^2$")
    ax.plot([r[1] for r in rows], [r[2] for r in rows], "o", ms=13,
            color=RED, mec="white", mew=1.5,
            label="모의실험으로 잰 기울기")
    ax.plot(gg, np.sqrt(1 - gg), "--", lw=2.2, color=AMBER,
            label="$\\sqrt{1 - R^2}$ — 그때의 상관")
    for t, rr2, s in rows:
        ax.text(rr2, s - 0.075, f"${s:.3f}$", ha="center", fontsize=10.4,
                color=RED, fontweight="bold")
    ax.set_xlabel("모형의 $R^2$", fontsize=11.5)
    ax.set_ylabel("잔차 대 관측값 그림에서 보이는 양", fontsize=11.5)
    ax.set_ylim(0, 1.05)
    ax.set_title("② 모형이 좋을수록 가짜 기울기가 작아진다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("잘못된 가로축은 언제나 같은 무늬를 만든다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"① 은 관측 ${N}$ 개짜리 한 적합이고, ② 는 목표 $R^2$ 마다 "
             f"${REPS}$ 번씩 자료를 새로 만들어 평균한 것이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "residual_vs_observed.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
