r"""7.1절 「그림으로 보는 통계학」 — 손이 뜨거웠던 적이 없다는 계산.

길로비치와 발론과 트버스키는 1985 년에 농구의 「뜨거운 손」을 쟀다.
직전 몇 개를 연달아 넣은 뒤의 성공률이 평소 성공률보다 높은가.
답은 「높지 않다」였고, 이 결과는 사람이 무작위에서 무늬를 본다는
이야기의 대표 보기가 되었다.

2018 년에 밀러와 산후르호가 『이코노메트리카』에 그 계산에 **고르기의
편향**이 들어 있다고 적었다.

동전을 n 번 던진다. 앞면이 나온 바로 다음 던지기들만 모아 앞면 비율을
내면, 그 값의 기댓값이 1/2 보다 **작다.** n=4 에서 정확히 0.4048 이다.
동전은 멀쩡하다. 「앞면 다음」이라는 조건으로 자리를 고르는 행위가
편향을 만든다.

  n=4 에서 정확한 열거:  0.4048   (16 가지 가운데 14 가지가 정의된다)

연속 k 개를 조건으로 걸수록 편향이 커진다. 모의로 재면

       n      k=1      k=2      k=3
      10   0.4453   0.3725   0.3525
      50   0.4900   0.4665   0.4173
     100   0.4948   0.4845   0.4600
     200   0.4976   0.4922   0.4821

n=100, k=3 에서 「뜨거운 손이 전혀 없는」 선수의 관측값이 0.461 로
평소보다 0.039 낮다. 뒤집어 말하면, 관측값이 평소 성공률과 같아 보이려면
선수가 진짜로 0.041 만큼 올라야 한다. 그러니 「평소와 같다」는 관측은
뜨거운 손이 없다는 증거가 아니라 **있다는 증거**다.

만드는 파일:

  ch05/mean/img/hot_hand.png

실행:  python3 scripts/make_ch07_hot_hand.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import itertools

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/mean/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

P = 0.5
NS = (5, 8, 12, 20, 35, 50, 75, 100, 150, 200)
KS = ((1, TEAL, "직전 $1$ 개"), (2, AMBER, "직전 $2$ 개"),
      (3, RED, "직전 $3$ 개"))
NSHOW, KSHOW, REPS, SEED = 100, 3, 60000, 1985
HOTS = np.round(np.linspace(0.0, 0.20, 11), 3)


def exact(n, k, p=P):
    """길이 n 의 모든 수열을 열거해 기댓값을 정확히 구한다."""
    tot = w = 0.0
    for s in itertools.product((0, 1), repeat=n):
        a = np.array(s)
        num = den = 0
        for i in range(k, n):
            if a[i - k:i].all():
                den += 1
                num += a[i]
        if den == 0:
            continue
        pr = p ** a.sum() * (1 - p) ** (n - a.sum())
        tot += pr * num / den
        w += pr
    return tot / w


def simulate(n, k, hot, rng, reps=REPS):
    """직전 k 개가 모두 성공이면 확률이 P+hot 로 오르는 선수를 reps 명 돌린다."""
    a = np.empty((reps, n), bool)
    a[:, :k] = rng.random((reps, k)) < P
    for i in range(k, n):
        run = a[:, i - k:i].all(1)
        a[:, i] = rng.random(reps) < np.where(run, P + hot, P)
    num = np.zeros(reps)
    den = np.zeros(reps)
    for i in range(k, n):
        run = a[:, i - k:i].all(1)
        den += run
        num += run & a[:, i]
    ok = den > 0
    return num[ok] / den[ok]


def main():
    rng = np.random.default_rng(SEED)

    print(f"  정확한 열거 (p = {P}, 직전 1 개 조건)")
    for n in (3, 4, 5, 6):
        print(f"    n = {n}   {exact(n, 1):.4f}")

    print(f"\n  모의 — 뜨거운 손이 전혀 없는 선수의 관측값")
    print(f"{'n':>7}" + "".join(f"{'k='+str(k):>10}" for k, _, _ in KS))
    curves = {k: [] for k, _, _ in KS}
    for n in NS:
        row = []
        for k, _, _ in KS:
            v = simulate(n, k, 0.0, rng).mean() if n > k else np.nan
            curves[k].append(v)
            row.append(v)
        print(f"{n:>7}" + "".join(f"{v:>10.4f}" for v in row))

    print(f"\n  n = {NSHOW}, 직전 {KSHOW} 개 조건")
    obs = []
    for hot in HOTS:
        v = simulate(NSHOW, KSHOW, hot, rng)
        obs.append(v.mean())
        print(f"    진짜 상승 +{hot:.3f}   관측 평균 {v.mean():.4f}   "
              f"관측 $-$ 평소 {v.mean()-P:+.4f}   표준편차 {v.std():.4f}")
    obs = np.array(obs)
    zero = float(np.interp(0.0, obs - P, HOTS))
    print(f"\n    관측값이 평소와 같아 보이려면 진짜 상승이 "
          f"+{zero:.3f} 이어야 한다")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 동전은 멀쩡한데 값이 내려앉는다 ──
    ax = axes[0]
    ax.axhline(P, color=INK, lw=2.2, ls="--", label="동전의 참값 $0.5$")
    for k, col, lab in KS:
        ax.plot(NS, curves[k], lw=3.2, color=col, marker="o", ms=6, label=lab)
    ax.axvline(NSHOW, color=GREY, lw=1.8, ls=":")
    e4 = exact(4, 1)
    ax.annotate(f"작은 $n$ 은 손으로 열거할 수 있다\n"
                f"— $n = 4$ 면 정확히 ${e4:.4f}$",
                xy=(5, curves[1][0]), xytext=(17, 0.332), fontsize=10.6,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate(f"${NSHOW}$ 번 던져도\n${curves[3][NS.index(NSHOW)]:.3f}$ 다",
                xy=(NSHOW, curves[3][NS.index(NSHOW)]), xytext=(168, 0.385),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100, 200])
    ax.set_xticklabels(["5", "10", "20", "50", "100", "200"])
    ax.minorticks_off()
    ax.set_xlabel("던진 횟수 $n$", fontsize=11.5)
    ax.set_ylabel("「연속 성공 바로 다음」의 성공률", fontsize=11.5)
    ax.set_title("동전은 멀쩡한데 값이 내려앉는다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.31, 0.545)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 「평소와 같다」가 뜻하는 것 ──
    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.6)
    ax.fill_between(HOTS, obs - P, 0, where=(obs - P) < 0, color=RED,
                    alpha=0.12, zorder=1)
    ax.plot(HOTS, HOTS, lw=3.0, color=GREY, ls="--",
            label="편향이 없다면 이래야 한다")
    ax.plot(HOTS, obs - P, lw=3.6, color=RED, marker="o", ms=7,
            label=f"${NSHOW}$ 번 던져 실제로 재면", zorder=3)
    ax.plot([zero], [0.0], marker="o", ms=14, color=RED, zorder=5,
            mec="white", mew=1.8)
    ax.annotate(f"진짜로 ${zero:.3f}$ 만큼 올라야\n「평소와 같다」가 나온다",
                xy=(zero, 0.0), xytext=(0.122, -0.034), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"두 선 사이의 틈이 편향이다\n— 어디서나 약 ${P-obs[0]:.3f}$",
                xy=(0.168, (HOTS[-2] + obs[-2] - P) / 2), xytext=(0.075, 0.138),
                fontsize=10.8, color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.text(0.010, -0.0575, "이 구간에서는 뜨거운 손이 있는데도\n"
            "「평소보다 낮다」고 읽힌다", fontsize=10.4, color=RED)
    ax.set_xlabel("선수의 진짜 상승폭 (연속 $3$ 개 성공 뒤)", fontsize=11.5)
    ax.set_ylabel("관측한 성공률 $-$ 평소 성공률", fontsize=11.5)
    ax.set_title(f"던지기 ${NSHOW}$ 번으로 재면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(-0.004, 0.204)
    ax.set_ylim(-0.078, 0.205)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("손이 뜨거웠던 적이 없다는 계산", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"모든 경우에 선수의 평소 성공률은 ${P}$ 이다. 오른쪽은 "
             f"상승폭마다 ${NSHOW}$ 번 던지기를 "
             f"${REPS//1000}{{,}}000$ 번 되풀이해 평균 낸 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = "docs/" + OUT + "hot_hand.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
