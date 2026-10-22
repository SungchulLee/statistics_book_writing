r"""15.6절 「그림으로 보는 통계학」 — 모수가 경계에 있으면 자유도 셈이 깨진다.

절 5 의 셈은 「자유모수의 수를 빼면 자유도」다. 그 셈은 참값이 모수공간의 **안쪽**에
있을 때의 이야기다. 분산은 음수가 될 수 없으므로 「이 분산이 $0$ 인가」를 묻는 순간
귀무가설이 **경계 위**에 놓이고, 그때 $-2\ln\Lambda$ 의 극한분포는 $\chi^2_1$ 이
아니라

    1/2 * (0 에서의 점질량)  +  1/2 * chi^2_1

이 된다(체르노프 1954). 통계량이 절반쯤은 정확히 $0$ 이기 때문이다 — 제약 없는
최대가 이미 경계 안쪽을 가리키면 제약을 걸어도 최대가 그대로다.

여기서는 균형 일원 변량효과 모형 y_ij = mu + b_i + e_ij 에서 H0: sigma_b^2 = 0 을
검정한다. 균형 자료에서는 ML 추정과 통계량이 닫힌 꼴이다.

    lambda1 = SSB*/(m-1),  lambda2 = SSW/(m(n-1)),  제약: lambda1 >= lambda2

  왼쪽  — 귀무분포. 0 에서 점질량 53%, 나머지가 반쪽 카이제곱
  오른쪽 — 그래서 chi^2_1 표를 그대로 쓰면 크기가 2.2%, 검정력도 그만큼 잃는다

만드는 파일:

  ch15/advanced_methods/img/boundary_lrt.png

실행:  python3 scripts/make_ch15_boundary_lrt.py   (저장소 최상위에서)
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

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/advanced_methods/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1954
REPS = 200_000
CHUNK = 50_000
M, N = 20, 5                      # 집단 20 개 · 집단당 5 개
RATIOS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
C_NAIVE = stats.chi2.ppf(0.95, 1)     # 3.841 — 자유도 1 의 표
C_MIX = stats.chi2.ppf(0.90, 1)       # 2.706 — 혼합분포의 95% 분위수


def lrt(y):
    """H0: sigma_b^2 = 0 의 -2 ln Lambda (균형 자료 · ML)."""
    m, n = y.shape[1], y.shape[2]
    gi = y.mean(axis=2, keepdims=True)
    gm = y.mean(axis=(1, 2), keepdims=True)
    ssb = (n * ((gi - gm) ** 2).sum(axis=1)).ravel()
    ssw = ((y - gi) ** 2).sum(axis=(1, 2))
    l1, l2 = ssb / (m - 1), ssw / (m * (n - 1))
    s2 = (ssb + ssw) / (m * n - 1)
    stat = ((m * n - 1) * np.log(s2)
            - ((m - 1) * np.log(l1) + m * (n - 1) * np.log(l2)))
    return np.where(l1 > l2, stat, 0.0)


def draw(ratio, rng, reps=REPS):
    """sigma_e = 1, sigma_b = ratio 인 자료에서 통계량을 모은다."""
    out, done = [], 0
    while done < reps:
        k = min(CHUNK, reps - done)
        y = rng.normal(size=(k, M, N))
        if ratio > 0:
            y = y + ratio * rng.normal(size=(k, M, 1))
        out.append(lrt(y))
        done += k
    return np.concatenate(out)


def main():
    rng = np.random.default_rng(SEED)

    L0 = draw(0.0, rng)
    p0 = float((L0 <= 0).mean())
    size_naive = float((L0 > C_NAIVE).mean())
    size_mix = float((L0 > C_MIX).mean())
    print(f"집단 {M} 개 · 집단당 {N} 개 · {REPS//10000} 만 번")
    print(f"  P(통계량 = 0) = {p0:.4f}")
    print(f"  문턱 {C_NAIVE:.3f}(자유도 1 의 표)에서 크기 {size_naive:.4f}")
    print(f"  문턱 {C_MIX:.3f}(혼합분포)에서 크기 {size_mix:.4f}")

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("\n 시그마_b/시그마_e   자유도 1 의 표   혼합분포")
    pw = []
    for r in RATIOS:
        L = L0 if r == 0 else draw(r, rng, reps=REPS // 2)
        pw.append([(L > C_NAIVE).mean(), (L > C_MIX).mean()])
        print(f"{r:>12.1f}{pw[-1][0]:>15.3f}{pw[-1][1]:>11.3f}")
    pw = np.array(pw)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── ① 귀무분포 ──
    ax = axes[0]
    pos = L0[L0 > 0]
    bins = np.linspace(0, 9, 46)
    w = np.full(len(pos), 1.0 / len(L0))      # 전체에 대한 비율로 그린다
    ax.hist(pos, bins=bins, weights=w / np.diff(bins)[0], color=AMBER,
            alpha=0.55, edgecolor="white", label="모의한 통계량 ($>0$ 인 것)")
    xs = np.linspace(0.02, 9, 400)
    ax.plot(xs, 0.5 * stats.chi2.pdf(xs, 1), lw=2.8, color=TEAL,
            label=r"$\frac{1}{2}\chi^2_1$ — 경계를 셈에 넣은 것")
    ax.plot(xs, stats.chi2.pdf(xs, 1), lw=2.2, ls="--", color=RED,
            label=r"$\chi^2_1$ — 절 $5$ 의 셈")
    # 0 에서의 점질량은 밀도가 아니므로 막대 하나로 세워 표시한다
    ax.plot([0, 0], [0, 0.90], lw=6, color=TEAL, solid_capstyle="butt",
            alpha=0.9, zorder=5)
    ax.plot([0], [0.90], marker="^", ms=11, color=TEAL, zorder=6)
    ax.annotate(f"$0$ 에서의 점질량 ${p0*100:.0f}\\%$\n(밀도가 아니라 확률이다)",
                xy=(0.06, 0.78), xytext=(1.35, 0.72), fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.axvline(C_NAIVE, color=RED, lw=1.5, ls=":")
    ax.axvline(C_MIX, color=TEAL, lw=1.5, ls=":")
    ax.text(C_MIX + 0.08, 0.40, f"${C_MIX:.3f}$", color=TEAL, fontsize=10.6)
    ax.text(C_NAIVE + 0.08, 0.30, f"${C_NAIVE:.3f}$", color=RED, fontsize=10.6)
    ax.set_ylim(0, 0.92)
    ax.set_xlabel("$-2\\ln\\Lambda$", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title("① 절반이 $0$ 에 쌓인다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 크기와 검정력 ──
    ax = axes[1]
    ax.plot(RATIOS, pw[:, 0] * 100, marker="s", ms=7, lw=2.6, color=RED,
            label=f"자유도 $1$ 의 표 — 문턱 ${C_NAIVE:.3f}$")
    ax.plot(RATIOS, pw[:, 1] * 100, marker="D", ms=7, lw=2.6, color=TEAL,
            label=f"혼합분포 — 문턱 ${C_MIX:.3f}$")
    ax.axhline(5, color=MUTED, lw=1.6, ls="--")
    ax.annotate(f"$H_0$ 에서 ${pw[0,0]*100:.1f}\\%$ 대 ${pw[0,1]*100:.1f}\\%$",
                xy=(0.02, pw[0, 1] * 100), xytext=(0.07, 24),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate(f"${pw[3,0]*100:.0f}\\%$ 대 ${pw[3,1]*100:.0f}\\%$",
                xy=(0.3, pw[3, 1] * 100), xytext=(0.19, 52),
                fontsize=11, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel("$\\sigma_b / \\sigma_e$  "
                  f"(집단 ${M}$ 개 · 집단당 ${N}$ 개)", fontsize=11.5)
    ax.set_ylabel("기각률 (%)", fontsize=11.5)
    ax.set_title("② 반만 쓰면 검정력도 반만 쓴다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("분산이 $0$ 인지 물으면 자유도 표가 맞지 않는다", fontsize=14.5)
    fig.text(0.5, -0.045,
             f"$H_0$ 에서 {REPS//10000} 만 번, 나머지 점은 {REPS//20000} 만 번이다. "
             "왼쪽의 두 곡선은 모의가 아니라 밀도함수다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "boundary_lrt.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
