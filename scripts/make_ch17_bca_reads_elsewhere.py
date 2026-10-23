r"""17.4절 「그림으로 보는 통계학」 — 백분위 구간은 한쪽으로만 빗나간다.

$95\%$ 구간이 $95\%$ 를 덮지 못하는 것보다 더 성가신 일이 있다. **빗나가는 방향이
한쪽으로 쏠리는 것**이다. 양쪽으로 $2.5\%$ 씩 빗나가야 할 것이 한쪽에서 $8\%$,
다른 쪽에서 $1.5\%$ 가 되면, 구간이 통째로 한 방향으로 밀려 있다는 뜻이다.

이 스크립트는 치우친 자료(지수분포)의 평균에 대해 백분위 구간과 BCa 구간을 나란히
재어 그 쏠림을 보인다. 그리고 BCa 가 그것을 어떻게 고치는지 — **붓스트랩 분포에서
읽는 자리를 옮겨서** — 를 수로 적는다.

  ① $n = 15, 40, 100$ 에서 참값이 구간의 **왼쪽**에 있던 비율과 **오른쪽**에
     있던 비율. 둘 다 $2.5\%$ 여야 한다
  ② BCa 가 실제로 읽는 자리. $n = 15$ 에서 아래쪽은 $2.5\%$ 가 아니라 중앙값
     $4.5\%$, 위쪽은 $97.5\%$ 가 아니라 $98.9\%$ 다

①의 두 구간은 `scipy.stats.bootstrap` 으로 구한다. 그 함수의 **기본 방법이 BCa**
라는 것도 함께 찍는다. ②의 보정계수는 직접 계산한다 — 편향보정 $z_0$ 는 붓스트랩
복제값 가운데 관측값보다 작은 것의 비율에서, 가속 $a$ 는 잭나이프 값의 왜도에서.

만드는 파일:

  ch17/bootstrap_ci/img/bca_reads_elsewhere.png

실행:  python3 scripts/make_ch17_bca_reads_elsewhere.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다. 1 분쯤 걸린다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import inspect

import numpy as np
from scipy import stats
from scipy.stats import bootstrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/bootstrap_ci/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

TRUE_MU = 1.0
NS = (15, 40, 100)
REPS = 3000
NB = 999
METHODS = ("percentile", "BCa")


def miss_directions(rng):
    """참값이 구간의 왼쪽에 있던 비율과 오른쪽에 있던 비율."""
    out = {}
    for n in NS:
        cnt = {m: [0, 0] for m in METHODS}
        for _ in range(REPS):
            x = rng.exponential(TRUE_MU, n)
            for m in METHODS:
                r = bootstrap((x,), np.mean, n_resamples=NB, method=m,
                              confidence_level=0.95, random_state=rng)
                lo, hi = r.confidence_interval
                if TRUE_MU < lo:
                    cnt[m][0] += 1          # 구간이 참값보다 위에 있다
                elif TRUE_MU > hi:
                    cnt[m][1] += 1          # 구간이 참값보다 아래에 있다
        out[n] = {m: (c[0] / REPS, c[1] / REPS) for m, c in cnt.items()}
    return out


def bca_levels(rng, n=15, reps=2000):
    """BCa 가 붓스트랩 분포에서 실제로 읽는 두 자리(백분율)."""
    z = stats.norm.ppf
    lo_lv, hi_lv = [], []
    for _ in range(reps):
        x = rng.exponential(TRUE_MU, n)
        th = x.mean()
        bs = x[rng.integers(0, n, (NB, n))].mean(1)
        z0 = z(np.clip((bs < th).mean(), 1 / (NB + 1), 1 - 1 / (NB + 1)))
        jk = np.array([np.delete(x, i).mean() for i in range(n)])
        d = jk.mean() - jk
        a = (d ** 3).sum() / (6 * ((d ** 2).sum()) ** 1.5)
        for lvl, store in ((0.025, lo_lv), (0.975, hi_lv)):
            zz = z0 + (z0 + z(lvl)) / (1 - a * (z0 + z(lvl)))
            store.append(stats.norm.cdf(zz) * 100)
    return np.array(lo_lv), np.array(hi_lv)


def main():
    default_method = inspect.signature(bootstrap).parameters["method"].default
    print(f"scipy.stats.bootstrap 의 기본 방법 = {default_method}")

    miss = miss_directions(np.random.default_rng(17))
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"\n참값을 덮지 못한 방향 ({REPS} 번, 약속은 양쪽 0.025 씩)")
    print("   n   방법          왼쪽   오른쪽    덮음")
    for n in NS:
        for m in METHODS:
            l, r = miss[n][m]
            print(f"  {n:4d}   {m:11s} {l:6.3f} {r:8.3f} {1-l-r:7.3f}")

    lo_lv, hi_lv = bca_levels(np.random.default_rng(71))
    print(f"\nBCa 가 읽는 자리 (n = 15)")
    print(f"  아래쪽 — 중앙값 {np.median(lo_lv):5.2f}%  "
          f"사분위 ({np.percentile(lo_lv, 25):.2f}, {np.percentile(lo_lv, 75):.2f})")
    print(f"  위쪽   — 중앙값 {np.median(hi_lv):5.2f}%  "
          f"사분위 ({np.percentile(hi_lv, 25):.2f}, {np.percentile(hi_lv, 75):.2f})")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 빗나간 방향 ──
    ax = axes[0]
    xs = np.arange(len(NS))
    w = 0.2
    for k, (m, col) in enumerate(zip(METHODS, (AMBER, TEAL))):
        left = [miss[n][m][0] for n in NS]
        right = [miss[n][m][1] for n in NS]
        ax.bar(xs + (2 * k - 1.5) * w, left, width=w, color=col, alpha=0.45,
               label=f"{m} — 구간이 참값 위에 있다")
        ax.bar(xs + (2 * k - 0.5) * w, right, width=w, color=col, alpha=0.95,
               label=f"{m} — 구간이 참값 아래에 있다")
        for x, v in zip(xs + (2 * k - 0.5) * w, right):
            ax.text(x, v + 0.002, f"${v:.3f}$", ha="center", fontsize=9.6,
                    color=col, fontweight="bold")
    ax.axhline(0.025, color=INK, lw=1.6, ls="--")
    ax.text(-0.45, 0.0265, "약속한 $0.025$", ha="left", fontsize=10.6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$n = {n}$" for n in NS], fontsize=11.5)
    ax.set_ylim(0, 0.105)
    ax.set_ylabel("참값을 놓친 비율", fontsize=11.5)
    ax.set_title("① 빗나가는 방향이 한쪽으로 쏠린다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=8.8, framealpha=0.93, loc="upper right", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② BCa 가 읽는 자리 ──
    ax = axes[1]
    ax.hist(lo_lv, bins=40, color=TEAL, alpha=0.8, label="아래쪽 끝을 읽는 자리")
    ax.hist(hi_lv, bins=40, color=AMBER, alpha=0.8, label="위쪽 끝을 읽는 자리")
    for v, col in ((2.5, TEAL), (97.5, AMBER)):
        ax.axvline(v, color=col, lw=2.0, ls="--")
    ax.text(1.4, ax.get_ylim()[1] * 0.93, "백분위법이 읽는 $2.5\\%$", rotation=90,
            ha="right", va="top", fontsize=10.4, color=TEAL)
    ax.text(97.5, ax.get_ylim()[1] * 0.93, "백분위법이 읽는 $97.5\\%$", rotation=90,
            ha="right", va="top", fontsize=10.4, color=AMBER)
    ax.text(np.median(lo_lv), ax.get_ylim()[1] * 0.45,
            f"중앙값 ${np.median(lo_lv):.1f}\\%$", ha="left", fontsize=11,
            color=TEAL, fontweight="bold")
    ax.text(np.median(hi_lv), ax.get_ylim()[1] * 0.45,
            f"중앙값 ${np.median(hi_lv):.1f}\\%$", ha="right", fontsize=11,
            color=AMBER, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_xlabel("붓스트랩 분포에서 읽은 백분위 자리 ($n = 15$)", fontsize=11.5)
    ax.set_ylabel("자료집합의 수", fontsize=11.5)
    ax.set_title("② BCa 는 읽는 자리를 옮긴다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10, framealpha=0.93, loc="upper center")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("덮지 못하는 것보다 한쪽으로 빗나가는 것이 문제다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"자료는 지수분포의 평균이다. ①은 scipy.stats.bootstrap 으로 "
             f"${REPS}$ 번, ②는 보정계수를 직접 계산해 $2000$ 번 되풀이했고 "
             f"붓스트랩은 매번 ${NB}$ 번이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "bca_reads_elsewhere.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
