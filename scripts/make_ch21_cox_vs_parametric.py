r"""21.4절 「그림으로 보는 통계학」 — 콕스가 「다시 한다면 모수적으로」라고 한 말의 값.

콕스 모형은 기저위험 $h_0(t)$ 를 적지 않는다. 그 대신 부분가능도로 $\beta$ 만 뽑는다.
$1994$ 년 대담에서 콕스 자신은 「나라면 보통은 모수적으로 다루고, 기저위험을 베이불
같은 것으로 두겠다」고 말했다. 그 말의 값을 수로 재어 본다.

자료는 베이불 비례위험에서 만든다 — 모양모수 $k$, 처리효과 $\beta = \ln 2$,
집단당 $100$ 명, 절단 $30\%$.

  ① $k = 2$ 에서 세 추정량의 분포
       콕스(부분가능도) · 맞는 베이불 PH · 모양을 1 로 고정한 지수 PH
  ② 기저위험의 모양 $k$ 를 바꾸며 본 평균 추정값

맞게 적으면 분산이 $2$–$5\%$ 작아지고, 틀리게 적으면 $k = 3$ 에서 $\beta$ 가 절반
아래로 내려간다. 콕스는 어느 $k$ 에서도 참값에 붙는다.

콕스 부분가능도는 직접 뉴턴법으로 풀고 `lifelines` 의 `CoxPHFitter` 와 소수 여섯째
자리까지 같은 것을 확인했다.

만드는 파일:

  ch21/cox/img/cox_vs_parametric.png

실행:  python3 scripts/make_ch21_cox_vs_parametric.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다. 1 분쯤 걸린다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import warnings

import numpy as np
from scipy import optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/cox/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1972
BETA = np.log(2.0)
N = 200          # 집단당 100 명
CENS = 0.30
KS = [0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
K_MAIN = 2.0
REPS_HIST = 2_000
REPS_CURVE = 800


def gen(k, rng):
    """베이불 비례위험 자료 — S(t|x) = exp(-t^k e^{beta x})."""
    x = np.repeat([0.0, 1.0], N // 2)
    t = (-np.log(rng.uniform(size=N)) / np.exp(BETA * x)) ** (1 / k)
    c = rng.exponential(np.quantile(t, 1 - CENS) * 2.0, size=N)
    return np.minimum(t, c), (t <= c).astype(int), x


def cox_beta(y, d, x):
    """이진 공변량의 콕스 부분가능도 — 뉴턴법."""
    o = np.argsort(y)
    y, d, x = y[o], d[o], x[o]
    b = 0.0
    for _ in range(60):
        w = np.exp(b * x)
        s0 = np.cumsum(w[::-1])[::-1]
        s1 = np.cumsum((w * x)[::-1])[::-1]
        s2 = np.cumsum((w * x * x)[::-1])[::-1]
        u = np.sum(d * (x - s1 / s0))
        i = np.sum(d * (s2 / s0 - (s1 / s0) ** 2))
        step = u / i
        b += step
        if abs(step) < 1e-11:
            break
    return b


def weibull_ph(y, d, x, fix_k=None):
    """베이불 비례위험의 최대가능도. fix_k=1 이면 지수 모형이다."""
    def nll(p):
        k = np.exp(p[0]) if fix_k is None else fix_k
        lam, b = np.exp(p[-2]), p[-1]
        return -np.sum(d * (np.log(k) + np.log(lam) + (k - 1) * np.log(y) + b * x)
                       - lam * y ** k * np.exp(b * x))
    p0 = [0.0, 0.0, 0.0] if fix_k is None else [0.0, 0.0]
    return optimize.minimize(nll, p0, method="Nelder-Mead",
                             options=dict(xatol=1e-9, fatol=1e-9,
                                          maxiter=5000)).x[-1]


def run(k, reps, rng):
    out = np.empty((reps, 3))
    for i in range(reps):
        y, d, x = gen(k, rng)
        out[i] = (cox_beta(y, d, x), weibull_ph(y, d, x),
                  weibull_ph(y, d, x, fix_k=1.0))
    return out


def main():
    rng = np.random.default_rng(SEED)

    main_run = run(K_MAIN, REPS_HIST, rng)
    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print(f"k = {K_MAIN} · {REPS_HIST} 번")
    for nm, col in zip(["콕스", "베이불", "지수"], range(3)):
        v = main_run[:, col]
        print(f"  {nm:<5} 평균 {v.mean():+.4f}  표준편차 {v.std(ddof=1):.4f}  "
              f"위험비 {np.exp(v.mean()):.3f}")
    eff = main_run[:, 0].var(ddof=1) / main_run[:, 1].var(ddof=1)
    print(f"  분산비 (콕스/베이불) = {eff:.3f}")

    print(f"\n   k     콕스     베이불       지수    분산비")
    curve = []
    for k in KS:
        r = run(k, REPS_CURVE, rng)
        curve.append([r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(),
                      r[:, 0].var(ddof=1) / r[:, 1].var(ddof=1)])
        print(f"{k:>4} {curve[-1][0]:>+9.4f} {curve[-1][1]:>+9.4f} "
              f"{curve[-1][2]:>+9.4f} {curve[-1][3]:>8.3f}")
    curve = np.array(curve)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── ① 분포 ──
    ax = axes[0]
    bins = np.linspace(0.0, 1.3, 60)
    for col, c, lab in [(2, RED, "지수 PH — 모양을 $1$ 로 고정(틀린 모형)"),
                        (1, TEAL, "베이불 PH — 맞는 모형"),
                        (0, AMBER, "콕스 — 기저위험을 적지 않는다")]:
        ax.hist(main_run[:, col], bins=bins, alpha=0.55, color=c,
                edgecolor="white", label=lab)
    ax.axvline(BETA, color=INK, lw=2.0, ls="--")
    ax.text(BETA + 0.015, ax.get_ylim()[1] * 0.92,
            f"참값 $\\ln 2 = {BETA:.3f}$", fontsize=10.8, color=INK)
    ax.set_xlabel(f"추정된 $\\hat\\beta$  (기저위험 모양 $k = {K_MAIN:.0f}$)",
                  fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title("① 맞는 모형과 틀린 모형", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 모양을 바꿔 가며 ──
    ax = axes[1]
    ax.plot(KS, curve[:, 0], marker="o", ms=11, lw=6, color=AMBER, alpha=0.45,
            label="콕스")
    ax.plot(KS, curve[:, 1], marker="D", ms=6.5, lw=2.2, ls="--", color=TEAL,
            label="베이불 PH (맞는 모형) — 콕스와 겹친다")
    ax.plot(KS, curve[:, 2], marker="s", ms=7, lw=2.6, color=RED,
            label="지수 PH (틀린 모형)")
    ax.axhline(BETA, color=INK, lw=1.8, ls="--")
    ax.text(0.62, BETA + 0.03, f"참값 $\\ln 2$", fontsize=10.8, color=INK)
    ax.annotate(f"$k = 3$ 에서 ${curve[-1,2]:.3f}$\n(위험비 ${np.exp(curve[-1,2]):.2f}$ — "
                f"참은 ${np.exp(BETA):.1f}$)",
                xy=(3.0, curve[-1, 2]), xytext=(1.95, 0.15),
                fontsize=10.6, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("기저위험의 모양 $k$  ($k=1$ 이면 지수가 맞는 모형이다)",
                  fontsize=11.5)
    ax.set_ylabel("$\\hat\\beta$ 의 평균", fontsize=11.5)
    ax.set_title("② 모양을 잘못 적으면", fontsize=12.8, color=INK)
    ax.set_ylim(0.05, 0.95)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("기저위험을 적지 않는 값과 잘못 적는 값", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"집단당 ${N//2}$ 명 · 절단 ${CENS*100:.0f}\\%$ · 참 효과는 위험비 $2$ 배다. "
             f"왼쪽은 ${REPS_HIST//1000}$ 천 번, 오른쪽은 각 점이 ${REPS_CURVE}$ 번이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "cox_vs_parametric.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
