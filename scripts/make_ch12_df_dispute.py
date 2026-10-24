r"""12.2절 「그림으로 보는 통계학」 — 자유도 하나를 두고 이십 년.

칼 피어슨이 1900 년에 카이제곱 적합도 검정을 내놓았다. 현대 통계학의
출발점으로 꼽히는 논문이다. 그는 범주가 k 개면 자유도가 언제나 k-1
이라고 보았다 — 자료에서 모수를 추정했더라도.

피셔가 1922 년과 1924 년 논문에서 그것이 틀렸음을 보였다. 자료에서
모수를 하나 추정할 때마다 자유도를 하나씩 더 빼야 한다.

  df = k - 1 - p

피어슨은 받아들이지 않았다. 그는 『바이오메트리카』의 편집자였고 당시
통계학계의 중심이었으며, 피셔보다 서른세 살 위였다. 논쟁은 공개적이었고
끝내 화해하지 못했다.

피셔가 옳았다. 그리고 틀린 자유도를 쓰면 어떻게 되는지는 잴 수 있다.
포아송 자료를 포아송에 적합하고(귀무가설이 참) 5% 검정을 하면

      n     칸 k   df=k-1 로 쓰면   df=k-2 (옳음)
    200      6        0.0263          0.0509
    500      6        0.0262          0.0510
    500      8        0.0296          0.0506
   2000      8        0.0288          0.0494

피어슨의 자유도로는 「5% 검정」이 실제로 2.6% 검정이 된다. 틀린 쪽이
안전한 쪽이라 위험하지는 않지만, 값은 검정력으로 치른다.

만드는 파일:

  ch12/distribution/img/df_dispute.png

실행:  python3 scripts/make_ch10_df_dispute.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch12/distribution/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, LAM, K, R, SEED = 150, 4.0, 8, 60_000, 1922
ALTS = (0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.22)   # 과산포의 세기


def stat_batch(n, lam, k, rng, over=0.0, reps=R):
    """포아송에 적합한 피어슨 X^2. over>0 이면 자료가 과산포다."""
    out = np.empty(reps)
    edges = list(range(k - 1))
    for r in range(reps):
        if over == 0:
            x = rng.poisson(lam, n)
        else:                                    # 감마 혼합 = 음이항
            g = rng.gamma(1 / over, over, n)
            x = rng.poisson(lam * g)
        lh = x.mean()
        obs = np.array([(x == j).sum() for j in edges] + [(x >= k - 1).sum()])
        p = np.array([stats.poisson.pmf(j, lh) for j in edges]
                     + [stats.poisson.sf(k - 2, lh)])
        exp = n * p
        exp = np.maximum(exp, 1e-9)
        out[r] = ((obs - exp) ** 2 / exp).sum()
    return out


def main():
    rng = np.random.default_rng(SEED)
    print(f"{'n':>7}{'칸 k':>7}{'df=k-1':>12}{'df=k-2':>12}")
    for n, lam, k in ((200, 3.0, 6), (500, 3.0, 6), (500, 4.0, 8),
                      (2000, 4.0, 8)):
        s = stat_batch(n, lam, k, rng, reps=20_000)
        print(f"{n:>7}{k:>7}{(s > stats.chi2.ppf(0.95, k-1)).mean():>12.4f}"
              f"{(s > stats.chi2.ppf(0.95, k-2)).mean():>12.4f}")

    s0 = stat_batch(N, LAM, K, rng)
    c_p = stats.chi2.ppf(0.95, K - 1)      # 피어슨의 자유도
    c_f = stats.chi2.ppf(0.95, K - 2)      # 피셔의 자유도
    print(f"\n  n={N}, k={K}:  피어슨 기각률 {(s0 > c_p).mean():.4f}, "
          f"피셔 {(s0 > c_f).mean():.4f}")

    print(f"\n{'과산포':>8}{'피어슨 df 검정력':>18}{'피셔 df 검정력':>16}")
    pw_p, pw_f = [], []
    for a in ALTS:
        s = stat_batch(N, LAM, K, rng, over=a, reps=20_000)
        pw_p.append((s > c_p).mean()); pw_f.append((s > c_f).mean())
        print(f"{a:>8.2f}{pw_p[-1]:>18.4f}{pw_f[-1]:>16.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 통계량이 따르는 분포 ──
    ax = axes[0]
    xs = np.linspace(0, 24, 500)
    ax.hist(s0, bins=90, range=(0, 24), density=True, color=MUTED, alpha=0.85,
            label=f"실제 $X^2$ 의 분포 ($n = {N}$, 모수 하나 추정)")
    ax.plot(xs, stats.chi2.pdf(xs, K - 2), lw=3.0, color=TEAL,
            label=f"$\\chi^2_{{{K-2}}}$ — 피셔의 자유도")
    ax.plot(xs, stats.chi2.pdf(xs, K - 1), lw=2.6, color=RED, ls="--",
            label=f"$\\chi^2_{{{K-1}}}$ — 피어슨의 자유도")
    ax.axvline(c_f, color=TEAL, lw=1.8, ls=":")
    ax.axvline(c_p, color=RED, lw=1.8, ls=":")
    ax.annotate(f"임계값 {c_f:.2f}", xy=(c_f, 0.012), xytext=(c_f - 1.2, 0.072),
                fontsize=10.6, color=TEAL, ha="right",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
    ax.annotate(f"임계값 {c_p:.2f}", xy=(c_p, 0.008), xytext=(c_p + 1.2, 0.050),
                fontsize=10.6, color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3))
    ax.set_xlabel("피어슨 $X^2$ 통계량", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title("통계량은 어느 쪽을 따르는가", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.135)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 치르는 값 ──
    ax = axes[1]
    xa = np.array(ALTS)
    ax.axhline(0.05, color=GREY, lw=1.6, ls=":")
    ax.text(0.215, 0.085, "약속한 $0.05$", fontsize=10.8, color=GREY, ha="right")
    ax.plot(xa, pw_f, lw=3.2, color=TEAL, marker="o", ms=8,
            label=f"$df = k-2$ — 피셔")
    ax.plot(xa, pw_p, lw=2.8, color=RED, marker="s", ms=7,
            label=f"$df = k-1$ — 피어슨")
    ax.fill_between(xa, pw_p, pw_f, color=TEAL, alpha=0.14)
    ax.annotate(f"귀무가설이 참일 때 — 피셔 {pw_f[0]:.3f},\n"
                f"피어슨 {pw_p[0]:.3f} 로 약속의 절반이다",
                xy=(0.004, pw_p[0]), xytext=(0.082, 0.17), fontsize=10.6,
                color=RED, ha="left",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    i = len(ALTS) - 3
    ax.annotate(f"과산포 {xa[i]:.2f} 에서 검정력이\n"
                f"{pw_f[i]:.3f} 대 {pw_p[i]:.3f} —\n이 띠가 버리는 검정력이다",
                xy=(xa[i], (pw_f[i] + pw_p[i]) / 2), xytext=(0.013, 0.88),
                fontsize=10.6, color=TEAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("자료의 과산포 세기 (0 이면 귀무가설이 참)", fontsize=11.5)
    ax.set_ylabel("기각할 확률", fontsize=11.5)
    ax.set_title("틀린 자유도가 치르는 값", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("자유도 하나를 두고 이십 년", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"포아송 자료 ${N}$ 개를 칸 ${K}$ 개로 묶고 평균 하나를 자료에서 "
             "추정했다. 과산포는 감마 혼합으로 만들었으며 $0$ 이면 자료가 "
             "정확히 포아송이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "df_dispute.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
