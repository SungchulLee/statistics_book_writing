r"""15.5절 「그림으로 보는 통계학」 — 르빈이 눈감아 준 기댐.

르빈(1960)은 절대편차 $d_{ij} = |x_{ij} - \bar x_i|$ 에 **보통의 일원분산분석**을
돌리자고 했다. 그런데 한 집단 안의 $d$ 들은 같은 $\bar x_i$ 를 빼고 만든 값이라
**독립이 아니다.** 분산분석의 F 는 독립을 가정한다.

그가 괜찮다고 본 근거가 왼쪽 칸이다. 같은 집단 안 두 편차 $x_j - \bar x$ 와
$x_k - \bar x$ 는 상관이 $\rho = -1/(n-1)$ 인 이변량 정규이고, 절댓값을 씌우면
상관은

    Corr(d_j, d_k) = (2/pi)[sqrt(1-rho^2) + rho*arcsin(rho) - 1] / (1 - 2/pi)

가 된다. $\rho$ 가 작을 때 분자가 $\rho^2/\pi$ 로 줄어들므로 이 값은
$1/[(\pi-2)(n-1)^2]$ 로, 곧 $1/n^2$ 차수로 사라진다. 부호도 바뀐다 — 편차끼리는
음의 상관인데 절댓값끼리는 **양의 상관**이다.

오른쪽 칸은 그 「차수로 사라진다」가 실제 크기에서 어떻게 보이는지다. 작은 $n$
에서는 분명히 어긋나고($n = 5$ 에서 명목 $5\%$ 가 $8\%$), $n$ 이 커지면 $5\%$
언저리로 내려오지만 정확히 $5\%$ 로 가지는 않는다.

만드는 파일:

  ch15/robust_tests/img/levene_dependence.png

실행:  python3 scripts/make_ch15_levene_dependence.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/robust_tests/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1960
K = 3                      # 집단 수
REPS = 200_000             # 크기 모의실험 되풀이
CORR_REPS = 8_000_000      # 상관 모의실험 되풀이 — 재는 값이 작아 많이 든다
CHUNK = 25_000
NS_SIZE = [3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
NS_CORR = [3, 4, 5, 10]    # 모의로 상관을 확인할 n — 이보다 크면 모의오차에 묻힌다


def corr_abs(n):
    """같은 집단 안 두 절대편차의 상관 — 닫힌 꼴."""
    rho = -1.0 / (n - 1)
    cov = (2 / np.pi) * (np.sqrt(1 - rho ** 2) + rho * np.arcsin(rho) - 1)
    return cov / (1 - 2 / np.pi)


def levene_size(n, rng, alphas=(0.05, 0.01)):
    """정규자료 · 집단 K 개 · 집단당 n 개에서 평균 중심 르빈의 실제 크기."""
    N = K * n
    hit = np.zeros(len(alphas))
    done = 0
    while done < REPS:
        m = min(CHUNK, REPS - done)
        x = rng.normal(size=(m, K, n))
        z = np.abs(x - x.mean(axis=2, keepdims=True))
        gi = z.mean(axis=2, keepdims=True)
        gm = z.mean(axis=(1, 2), keepdims=True)
        ssb = (n * ((gi - gm) ** 2).sum(axis=1)).ravel()
        ssw = ((z - gi) ** 2).sum(axis=(1, 2))
        p = stats.f.sf((ssb / (K - 1)) / (ssw / (N - K)), K - 1, N - K)
        hit += [(p < a).sum() for a in alphas]
        done += m
    return hit / REPS


def main():
    rng = np.random.default_rng(SEED)

    # ── 상관 — 닫힌 꼴과 모의 ──
    ns = np.arange(3, 201)
    exact = np.array([corr_abs(n) for n in ns])
    asym = 1.0 / ((np.pi - 2) * (ns - 1.0) ** 2)

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("   n     닫힌 꼴        모의    1/((π-2)(n-1)²)")
    sim = []
    for n in NS_CORR:
        s1 = s2 = s11 = s22 = s12 = 0.0
        for _ in range(CORR_REPS // CHUNK):
            x = rng.normal(size=(CHUNK, n))
            d = np.abs(x - x.mean(axis=1, keepdims=True))
            a, b = d[:, 0], d[:, 1]
            s1 += a.sum(); s2 += b.sum()
            s11 += (a * a).sum(); s22 += (b * b).sum(); s12 += (a * b).sum()
        m = CORR_REPS
        cov = s12 / m - (s1 / m) * (s2 / m)
        r = float(cov / np.sqrt((s11 / m - (s1 / m) ** 2)
                                * (s22 / m - (s2 / m) ** 2)))
        sim.append(r)
        print(f"{n:4d} {corr_abs(n):11.6f} {r:11.6f} "
              f"{1/((np.pi-2)*(n-1)**2):14.6f}")

    # ── 실제 크기 ──
    print("\n   n   명목 5%   명목 1%")
    sizes = []
    for n in NS_SIZE:
        s = levene_size(n, rng)
        sizes.append(s)
        print(f"{n:4d} {s[0]:9.4f} {s[1]:9.4f}")
    sizes = np.array(sizes)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── ① 상관의 소멸 ──
    ax = axes[0]
    ax.loglog(ns, exact, lw=5.0, color=TEAL, alpha=0.4, label="닫힌 꼴 — 위의 식")
    ax.loglog(ns, asym, lw=1.8, ls="--", color=INK, zorder=4,
              label=r"$1/[(\pi-2)(n-1)^2]$")
    ax.loglog(NS_CORR, sim, "o", ms=9, color=AMBER, zorder=5,
              label=f"모의 — {CORR_REPS//10_000:,} 만 번")
    # 로그 눈금의 마이너스는 mathtext 라 한글 글꼴에 글리프가 없다 — 두부가 된다
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    for n, v in zip(NS_CORR, [corr_abs(m) for m in NS_CORR]):
        ax.annotate(f"$n={n}$ 에서 ${v:.3f}$", xy=(n, v),
                    xytext=(n * 2.4, v * 0.55), fontsize=10.6, color=INK,
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.0))
    ax.set_xlabel("집단당 관측값 수 $n$", fontsize=11.5)
    ax.set_ylabel("같은 집단 안 두 절대편차의 상관", fontsize=11.5)
    ax.set_title("① 기댐은 $1/n^2$ 의 속도로 사라진다", fontsize=12.8, color=INK, pad=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 그래서 실제 크기는 ──
    ax = axes[1]
    ax.plot(NS_SIZE, sizes[:, 0] * 100, marker="o", ms=7.5, lw=2.6,
            color=AMBER, label="명목 $5\\%$")
    ax.plot(NS_SIZE, sizes[:, 1] * 100, marker="s", ms=7.0, lw=2.6,
            color=BLUE, label="명목 $1\\%$")
    ax.axhline(5, color=MUTED, lw=1.6, ls="--")
    ax.axhline(1, color=MUTED, lw=1.6, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(NS_SIZE)
    ax.set_xticklabels([str(v) for v in NS_SIZE], fontsize=10)
    ax.annotate(f"${sizes[2,0]*100:.1f}\\%$", xy=(5, sizes[2, 0] * 100),
                xytext=(5.9, sizes[2, 0] * 100 + 0.35), fontsize=11.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.annotate(f"${sizes[-1,0]*100:.1f}\\%$", xy=(100, sizes[-1, 0] * 100),
                xytext=(52, sizes[-1, 0] * 100 + 1.0), fontsize=11.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("집단당 관측값 수 $n$  (집단 $3$ 개 · 정규)", fontsize=11.5)
    ax.set_ylabel("실제 거짓 양성률 (%)", fontsize=11.5)
    ax.set_title("② 그가 모의로 확인한 두 자리에서", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("독립이 아닌 값들에 독립을 가정한 검정을 돌린다",
                 fontsize=14.5)
    fig.text(0.5, -0.045,
             f"분산은 세 집단 모두 같게 두었으므로 $H_0$ 가 참이다 "
             f"(각 $n$ 마다 ${REPS//10000}$ 만 번). 왼쪽 곡선은 모의가 아니라 "
             "이변량 정규에서 유도한 식이고, 점만 모의로 맞춰 본 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "levene_dependence.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
