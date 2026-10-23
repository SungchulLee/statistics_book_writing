r"""17.3절 「그림으로 보는 통계학」 — 홀과 윌슨이 적은 두 지침의 값.

$1991$ 년에 피터 홀과 수전 윌슨이 『Biometrics』에 「붓스트랩 가설검정의 두 지침」을
실었다. 지침은 둘이다.

  ① 귀무가설을 반영하도록 재표집하라 (이 쪽의 「중심이동」)
  ② 신뢰구간에서 좋다고 알려진 것을 쓰라 — 곧 **스튜던트화한** 통계량

이 스크립트는 두 지침이 각각 무엇을 지켜 주는지 수로 잰다.

  ① 지침 1 을 어기면 **검정력이 사라진다.** 중심이동 없이 원자료에서 재표집하면
     참 평균이 얼마든 기각률이 $0$ 이다
  ② 지침 2 는 **크기를 바로잡는다.** 왜도가 큰 자료에서 비스튜던트화 검정은
     $n = 10$ 에서 참인 귀무가설을 $14$–$15\%$ 기각한다

자료는 지수분포다(왜도 $2$). 귀무가설은 $H_0: \mu = 1$ 이고, 붓스트랩은 매번
$B = 499$ 번 다시 뽑는다.

만드는 파일:

  ch17/bootstrap_testing/img/two_guidelines.png

실행:  python3 scripts/make_ch17_two_guidelines.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다. 1 분쯤 걸린다.
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

OUT = "docs/ch17/bootstrap_testing/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

MU0 = 1.0
ALPHA = 0.05
B = 499
REPS = 4000
N_POWER = 20
MUS = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8)
NS = (10, 20, 50, 100)


def power_curves(rng):
    """지침 1 — 중심이동을 넣은 검정과 빼먹은 검정의 기각률."""
    ok, bad, tails = [], [], []
    for mu in MUS:
        o = b = 0
        left = right = 0
        for _ in range(REPS):
            x = rng.exponential(mu, N_POWER)
            xb, s = x.mean(), x.std(ddof=1)
            idx = rng.integers(0, N_POWER, (B, N_POWER))
            xs = (x - xb + MU0)[idx]                   # 귀무가설 쪽으로 옮긴다
            tobs = (xb - MU0) / (s / np.sqrt(N_POWER))
            tstar = (xs.mean(1) - MU0) / (xs.std(1, ddof=1) / np.sqrt(N_POWER))
            hit = np.mean(np.abs(tstar) >= abs(tobs)) < ALPHA
            o += hit
            if hit:                      # 어느 꼬리로 기각했는가
                left += tobs < 0
                right += tobs >= 0
            ys = x[idx]                                 # 옮기지 않고 그대로
            b += np.mean(np.abs(ys.mean(1) - MU0) >= abs(xb - MU0)) < ALPHA
        ok.append(o / REPS)
        bad.append(b / REPS)
        tails.append((left / REPS, right / REPS))
    return np.array(ok), np.array(bad), np.array(tails)


def size_by_n(rng):
    """지침 2 — 스튜던트화한 검정과 하지 않은 검정의 실제 크기."""
    plain, stud = [], []
    for n in NS:
        p = s_ = 0
        for _ in range(REPS):
            x = rng.exponential(MU0, n)
            xb, sd = x.mean(), x.std(ddof=1)
            idx = rng.integers(0, n, (B, n))
            xs = (x - xb + MU0)[idx]
            mb = xs.mean(1)
            p += np.mean(np.abs(mb - MU0) >= abs(xb - MU0)) < ALPHA
            tobs = (xb - MU0) / (sd / np.sqrt(n))
            tstar = (mb - MU0) / (xs.std(1, ddof=1) / np.sqrt(n))
            s_ += np.mean(np.abs(tstar) >= abs(tobs)) < ALPHA
        plain.append(p / REPS)
        stud.append(s_ / REPS)
    return np.array(plain), np.array(stud)


def main():
    ok, bad, tails = power_curves(np.random.default_rng(1991))
    plain, stud = size_by_n(np.random.default_rng(47))

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"지침 1 — n = {N_POWER}, 되풀이 {REPS} 번")
    print("  참 평균   올바른 검정   중심이동을 빼먹은 검정")
    for mu, a, b in zip(MUS, ok, bad):
        print(f"  {mu:7.1f} {a:13.3f} {b:24.3f}")
    print("\n  같은 기각을 두 꼬리로 나누면 — 치우친 자료에서는 한쪽이 먼저 산다")
    print("  참 평균   왼쪽 꼬리   오른쪽 꼬리")
    for mu, (l, r) in zip(MUS, tails):
        print(f"  {mu:7.1f} {l:11.4f} {r:12.4f}")

    print(f"\n지침 2 — 참인 귀무가설을 기각한 비율 (약속은 {ALPHA:.2f})")
    print("   n   비스튜던트화   스튜던트화")
    for n, p, s in zip(NS, plain, stud):
        print(f"  {n:4d} {p:13.4f} {s:12.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7), constrained_layout=True)

    # ── ① 지침 1 ──
    ax = axes[0]
    ax.plot(MUS, ok, "o-", ms=8, lw=2.6, color=TEAL,
            label="지침대로 — 옮기고 스튜던트화")
    ax.plot(MUS, bad, "s--", ms=7, lw=2.4, color=RED,
            label="중심이동을 빼먹었다")
    ax.axhline(ALPHA, color=INK, lw=1.4, ls=":")
    ax.text(2.8, ALPHA + 0.02, f"약속한 ${ALPHA:.2f}$", ha="right", fontsize=10.6,
            color=INK)
    ax.annotate("참 평균이 얼마든 $0$ 이다", xy=(2.0, 0.0), xytext=(1.55, 0.21),
                fontsize=11.4, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel(f"참 평균 $\\mu$   ($H_0: \\mu = {MU0:.0f}$, $n = {N_POWER}$)",
                  fontsize=11.5)
    ax.set_ylabel("기각률", fontsize=11.5)
    ax.set_ylim(-0.03, 0.85)
    ax.set_title("① 지침 1 을 어기면 검정력이 사라진다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 지침 2 ──
    ax = axes[1]
    xs = np.arange(len(NS))
    w = 0.36
    ax.bar(xs - w / 2, plain, width=w, color=AMBER, alpha=0.88,
           label="스튜던트화하지 않음")
    ax.bar(xs + w / 2, stud, width=w, color=TEAL, alpha=0.88, label="스튜던트화")
    for x, v in zip(xs - w / 2, plain):
        ax.text(x, v + 0.004, f"${v:.3f}$", ha="center", fontsize=10.8,
                color=AMBER, fontweight="bold")
    for x, v in zip(xs + w / 2, stud):
        ax.text(x, v + 0.004, f"${v:.3f}$", ha="center", fontsize=10.8,
                color=TEAL, fontweight="bold")
    ax.axhline(ALPHA, color=INK, lw=1.6, ls="--")
    ax.text(-0.42, ALPHA + 0.004, f"약속한 ${ALPHA:.2f}$", ha="left",
            fontsize=10.6, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"$n = {n}$" for n in NS], fontsize=11.5)
    ax.set_ylim(0, 0.175)
    ax.set_ylabel("참인 귀무가설을 기각한 비율", fontsize=11.5)
    ax.set_title("② 지침 2 는 크기를 바로잡는다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("두 줄짜리 지침이 각각 지켜 주는 것", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"자료는 지수분포(왜도 $2$)이고 붓스트랩은 매번 ${B}$ 번 다시 뽑는다. "
             f"두 판 모두 ${REPS}$ 번 되풀이해 센 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "two_guidelines.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
