r"""15.5절 「그림으로 보는 통계학」 — 표준이 1941 년의 비를 아직 쓰는 까닭.

실험실 사이의 정밀도를 재는 공동연구(ring trial)에서는 실험실 $p$ 곳이 같은 시료를
$n$ 번씩 측정한다. $n$ 은 보통 둘이나 셋이다. 어느 실험실의 분산이 유난히 큰지를
가리는 데 국제표준(ISO 5725-2)이 지정하는 것은 **코크란의 $C$** 다.

  C = max_i s_i^2 / sum_i s_i^2

이 비는 최대 분산 하나만 본다. 그래서 「한 곳이 엉망이다」라는 대안에는 강하지만
「여럿이 조금씩 다르다」에는 약하다. 왼쪽 칸이 그 교환이다.

오른쪽 칸이 더 중요하다. 이 책이 기본값으로 권하는 **브라운–포사이드**는 반복수가
적을 때 쓸 수가 없다.

  n = 2 : 집단 안 두 편차가 항상 같아 분모가 0 이다 — 언제나 기각한다 (크기 1.000)
  n = 3 : 중앙값에서 잰 편차 하나가 반드시 0 이다 — 거의 기각하지 못한다 (0.000)
  홀수 n 이 짝수 n 보다 나쁘고, 그 들쭉날쭉이 n = 12 까지 남는다

바틀릿은 1937 년의 보정 덕분에 n = 3 부터 0.05 를 지킨다. 현장의 표준이 로버스트
검정을 쓰지 않는 데에는 까닭이 있다.

만드는 파일:

  ch15/robust_tests/img/cochran_ring_trial.png

실행:  python3 scripts/make_ch15_cochran_ring_trial.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import warnings

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/robust_tests/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

SEED = 1941
REPS = 40_000
CRIT_REPS = 200_000   # 임계값은 꼬리 분위수라 더 많이 든다
# ISO 5725-2:2019 의 표 5 — 실험실 8 곳 · 반복 3 회의 1% 와 5% 임계값
ISO_TABLE5_P8_N3 = (0.615, 0.516)
P = 8          # 실험실 수
N = 3          # 반복 측정 수
RATIO = 3.0    # 「엉망인」 실험실의 표준편차 배수
NS = list(range(2, 13))


def stats_all(x, c_crit=None):
    """코크란 C · 바틀릿(보정) p · 브라운–포사이드 p."""
    k, n = x.shape[1], x.shape[2]
    s2 = x.var(axis=2, ddof=1)
    C = s2.max(axis=1) / s2.sum(axis=1)
    d, N_k = n - 1, k * (n - 1)
    sp2 = (d * s2).sum(axis=1) / N_k
    T = ((N_k * np.log(sp2) - (d * np.log(s2)).sum(axis=1))
         / (1 + (k / d - 1 / N_k) / (3 * (k - 1))))
    p_bart = stats.chi2.sf(T, k - 1)
    z = np.abs(x - np.median(x, axis=2, keepdims=True))
    gm = z.mean(axis=(1, 2), keepdims=True)
    gi = z.mean(axis=2, keepdims=True)
    ssb = (n * ((gi - gm) ** 2).sum(axis=1)).ravel()
    ssw = ((z - gi) ** 2).sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = (ssb / (k - 1)) / (ssw / (k * n - k))
    p_bf = stats.f.sf(F, k - 1, k * n - k)
    return C, p_bart, p_bf


def main():
    rng = np.random.default_rng(SEED)

    # ── 코크란 임계값 — ISO 표 5 가 싣는 그 수 ──
    c0 = []
    for _ in range(CRIT_REPS // REPS):
        x = rng.normal(size=(REPS, P, N))
        c0.append(stats_all(x)[0])
    C0 = np.concatenate(c0)
    m5, m1 = np.quantile(C0, [0.95, 0.99])
    f5 = stats.f.ppf(1 - 0.05 / P, N - 1, (P - 1) * (N - 1))
    f1 = stats.f.ppf(1 - 0.01 / P, N - 1, (P - 1) * (N - 1))
    c5, c1 = 1 / (1 + (P - 1) / f5), 1 / (1 + (P - 1) / f1)
    print(f"실험실 {P} 곳 · 반복 {N} 회에서 코크란 C 의 임계값")
    print(f"  ISO 표 5   5% {ISO_TABLE5_P8_N3[1]:.4f}   1% {ISO_TABLE5_P8_N3[0]:.4f}")
    print(f"  F 근사식   5% {c5:.4f}   1% {c1:.4f}")
    print(f"  모의       5% {m5:.4f}   1% {m1:.4f}   ({CRIT_REPS//10000} 만 번)")

    # ── ① 대안의 모양에 따른 검정력 ──
    mcounts = list(range(0, 7))
    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("\n분산이 큰 실험실 수   코크란    바틀릿    브라운-포사이드")
    pw = []
    for m in mcounts:
        sig = np.array([RATIO] * m + [1.0] * (P - m))
        x = rng.normal(size=(REPS, P, N)) * sig[None, :, None]
        C, pb, pbf = stats_all(x)
        row = [(C > c5).mean(), (pb < 0.05).mean(), (pbf < 0.05).mean()]
        pw.append(row)
        print(f"{m:>12}{row[0]:>16.3f}{row[1]:>9.3f}{row[2]:>12.3f}")
    pw = np.array(pw)

    # ── ② 반복수에 따른 귀무 크기 ──
    print("\n   n   바틀릿   브라운-포사이드   (H0 크기)")
    size = []
    for n in NS:
        x = rng.normal(size=(REPS, P, n))
        _, pb, pbf = stats_all(x)
        size.append([(pb < 0.05).mean(), (pbf < 0.05).mean()])
        print(f"{n:>4}{size[-1][0]:>9.3f}{size[-1][1]:>13.3f}")
    size = np.array(size)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), constrained_layout=True)

    # ── ① ──
    ax = axes[0]
    for col, mk, lab, j in [(AMBER, "o", "코크란 $C$ — 최대 분산 하나", 0),
                            (RED, "s", "바틀릿", 1),
                            (TEAL, "D", "브라운–포사이드", 2)]:
        ax.plot(mcounts, pw[:, j] * 100, marker=mk, ms=7.5, lw=2.6,
                color=col, label=lab)
    ax.axhline(5, color=MUTED, lw=1.6, ls="--")
    ax.set_ylim(-3, 62)
    ax.annotate(f"한 곳만 튈 때 ${pw[1,0]*100:.0f}\\%$ 대 ${pw[1,1]*100:.0f}\\%$",
                xy=(1.05, pw[1, 0] * 100), xytext=(2.7, 57),
                fontsize=10.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.annotate("넷이 튀면 뒤집힌다", xy=(4, pw[4, 1] * 100 + 1),
                xytext=(4.1, 51), fontsize=10.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.set_xlabel(f"분산이 ${RATIO:.0f}$ 배인 실험실의 수  "
                  f"(모두 {P} 곳 · 반복 {N} 회)", fontsize=11.5)
    ax.set_ylabel("기각률 (%)", fontsize=11.5)
    ax.set_title("① 무엇을 겨누는 검정인가", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② ──
    ax = axes[1]
    ax.plot(NS[1:], size[1:, 0] * 100, marker="s", ms=7, lw=2.6, color=RED,
            label="바틀릿 — $1937$ 년의 보정이 붙은 것")
    ax.plot(NS[1:], size[1:, 1] * 100, marker="D", ms=7, lw=2.6, color=TEAL,
            label="브라운–포사이드")
    ax.axhline(5, color=MUTED, lw=1.6, ls="--")
    ax.set_ylim(-0.4, 8.2)
    ax.set_xticks(NS[1:])
    ax.annotate(f"$n = 3$ 에서 ${size[1,1]*100:.1f}\\%$",
                xy=(3, size[1, 1] * 100), xytext=(3.2, 1.6),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(3.2, 7.4, f"$n = 2$ 는 ${size[0,1]*100:.0f}\\%$ — 축 밖이다",
            fontsize=10.8, color=INK)
    ax.set_xlabel(f"실험실당 반복 측정 수 $n$  (실험실 {P} 곳 · 정규 · 분산은 모두 같게)",
                  fontsize=11.2)
    ax.set_ylabel("명목 $5\\%$ 의 실제 거짓 양성률 (%)", fontsize=11.5)
    ax.set_title("② 반복이 둘셋일 때 권장 기본값이 하는 일",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.2, framealpha=0.95, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("현장이 오래된 검정을 버리지 못하는 까닭", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"각 점이 {REPS//10000} 만 번이다. 코크란의 문턱으로는 "
             f"ISO 표 5 의 ${ISO_TABLE5_P8_N3[1]:.3f}$ 과 같은 값을 썼다 "
             f"($F$ 근사식 ${c5:.4f}$ · 모의 ${m5:.4f}$).",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "cochran_ring_trial.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
