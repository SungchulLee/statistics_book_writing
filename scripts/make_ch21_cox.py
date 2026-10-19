r"""21.5절 「그림으로 보는 통계학」 — 모르겠다고 적어 두고 답을 얻는 법.

콕스(1972)의 비례위험모형은 위험함수를 이렇게 쓴다.

  h(t | x) = h0(t) exp(beta x)

그리고 **h0(t) 를 끝까지 추정하지 않는다.** 부분가능도가 기저위험을
약분해 버리기 때문이다. 처음 발표되었을 때 이것은 낯선 물건이었다.
가능도가 아닌 것을 가능도처럼 쓰는 일이 정당한가 — 논문 토론에서
바로 그 질문이 나왔고, 콕스는 1975 년에 부분가능도를 따로 정의하는
논문을 다시 써야 했다. 점근이론은 그보다 더 뒤에 완성되었다.

그 논문이 지금 과학 전체에서 가장 많이 인용된 축에 든다.

버리는 값은 얼마인가. 기저위험이 **정말로** 지수일 때, 그것을 알고
지수모형을 적합한 사람과 모른다고 적어 둔 콕스를 견주면

  절단  0%   상대효율 0.67       절단 30%   상대효율 0.84

모수모형이 정확히 맞을 때 콕스는 효율을 15–33% 잃는다. 적지 않다.

그런데 그 모형이 틀렸다면. 진짜 기저위험이 와이블(모양 k)인데 분석자가
지수로 적합하면 이렇게 된다 (참 beta = 0.693).

    k        콕스      지수모형
   0.5      0.698      1.029
   1.0      0.705      0.700
   2.0      0.698      0.400
   3.0      0.698      0.279

콕스는 어느 모양에서나 참값을 맞힌다. 지수모형은 k = 3 에서 0.279 —
위험비로 옮기면 2.00 이어야 할 것을 1.32 로 보고한다.

**모르겠다고 적어 두는 값이 효율 조금이고, 안다고 잘못 적는 값이
답 전체다.**

만드는 파일:

  ch21/cox/img/cox_partial.png

실행:  python3 scripts/make_ch21_cox.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/cox/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

BETA = np.log(2.0)
N, REPS, SEED = 200, 1500, 1972
CENS = (0.0, 0.15, 0.30, 0.50)
KS = (0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0)


def cox_beta(t, d, x):
    o = np.argsort(t)
    t, d, x = t[o], d[o], x[o]
    b = 0.0
    for _ in range(60):
        e = np.exp(b * x)
        s0 = np.cumsum(e[::-1])[::-1]
        s1 = np.cumsum((e * x)[::-1])[::-1]
        s2 = np.cumsum((e * x * x)[::-1])[::-1]
        u = np.sum(d * (x - s1 / s0))
        i = np.sum(d * (s2 / s0 - (s1 / s0) ** 2))
        if i <= 0:
            break
        step = u / i
        b += step
        if abs(step) < 1e-10:
            break
    return b


def exp_beta(t, d, x):
    """기저위험을 지수로 가정한 모수적 MLE."""
    b = 0.0
    for _ in range(100):
        w = np.exp(b * x) * t
        lam = d.sum() / w.sum()
        u = np.sum(d * x) - lam * np.sum(w * x)
        i = lam * np.sum(w * x * x)
        if i <= 0:
            break
        step = u / i
        b += step
        if abs(step) < 1e-10:
            break
    return b


def main():
    rng = np.random.default_rng(SEED)

    print("    기저위험이 정말 지수일 때 — 콕스가 치르는 값")
    print(f"{'절단':>7}{'콕스 sd':>11}{'지수 sd':>11}{'상대효율':>11}")
    eff = []
    for c in CENS:
        bc, bp = [], []
        for _ in range(REPS):
            x = rng.normal(size=N)
            T = rng.exponential(1 / np.exp(BETA * x))
            C = (np.full(N, np.inf) if c == 0
                 else rng.exponential((1 - c) / c, N))
            t = np.minimum(T, C)
            d = (T <= C).astype(float)
            bc.append(cox_beta(t, d, x))
            bp.append(exp_beta(t, d, x))
        sc, sp = np.std(bc), np.std(bp)
        eff.append((sp / sc) ** 2)
        print(f"{c:>7.0%}{sc:>11.4f}{sp:>11.4f}{eff[-1]:>11.3f}")

    print("\n    기저위험이 와이블인데 지수로 적합하면 — 치우침")
    print(f"{'k':>6}{'콕스':>11}{'지수모형':>12}{'지수의 치우침':>15}")
    cm, pm, cs, ps = [], [], [], []
    for k in KS:
        bc, bp = [], []
        for _ in range(REPS):
            x = rng.normal(size=N)
            u = rng.random(N)
            T = (-np.log(u) / np.exp(BETA * x)) ** (1 / k)
            C = rng.exponential(2.0, N)
            t = np.minimum(T, C)
            d = (T <= C).astype(float)
            bc.append(cox_beta(t, d, x))
            bp.append(exp_beta(t, d, x))
        cm.append(np.mean(bc)); pm.append(np.mean(bp))
        cs.append(np.std(bc)); ps.append(np.std(bp))
        print(f"{k:>6.1f}{cm[-1]:>11.4f}{pm[-1]:>12.4f}{pm[-1]-BETA:>+15.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 모형이 맞을 때 치르는 값 ──
    ax = axes[0]
    xs = np.arange(len(CENS))
    ax.bar(xs, eff, width=0.5, color=TEAL, alpha=0.85)
    ax.axhline(1.0, color=INK, lw=2.2, ls="--")
    ax.text(len(CENS) - 0.55, 1.016, "모수모형과 같은 정밀도", fontsize=11,
            color=INK, ha="right")
    for i, v in enumerate(eff):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=11.5,
                color=INK)
        ax.text(i, v / 2, f"{100*(1-v):.0f}%\n더 넓다", ha="center",
                va="center", fontsize=10.6, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"절단 {c:.0%}" for c in CENS], fontsize=11)
    ax.set_ylabel("콕스의 상대효율 (모수모형 $=1$)", fontsize=11.5)
    ax.set_title("기저위험이 **정말로** 지수일 때".replace("**", ""),
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 1.12)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 모형이 틀렸을 때 ──
    ax = axes[1]
    cm, pm, cs, ps = map(np.array, (cm, pm, cs, ps))
    ax.axhline(BETA, color=INK, lw=2.4, ls="--")
    ax.text(0.52, BETA + 0.028, f"참값 $\\beta = \\log 2 = {BETA:.3f}$",
            fontsize=11, color=INK)
    ax.fill_between(KS, cm - cs, cm + cs, color=TEAL, alpha=0.18)
    ax.plot(KS, cm, lw=3.0, color=TEAL, marker="o", ms=7,
            label="콕스 — 기저위험을 추정하지 않는다")
    ax.fill_between(KS, pm - ps, pm + ps, color=RED, alpha=0.14)
    ax.plot(KS, pm, lw=2.8, color=RED, marker="s", ms=6,
            label="지수모형 — 기저위험을 잘못 지정했다")
    ax.axvline(1.0, color=GREY, lw=1.6, ls=":")
    ax.text(1.04, 0.06, "$k=1$ 에서만\n지수가 옳다", fontsize=10.6, color=GREY)
    i3 = KS.index(3.0)
    ax.annotate(f"$k = 3$ 에서 {pm[i3]:.3f} —\n"
                f"위험비 $2.00$ 을 ${np.exp(pm[i3]):.2f}$ 로 보고한다",
                xy=(3.0, pm[i3]), xytext=(2.05, 0.14), fontsize=10.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("진짜 기저위험의 와이블 모양 $k$", fontsize=11.5)
    ax.set_ylabel("추정한 $\\beta$", fontsize=11.5)
    ax.set_title("기저위험의 모양을 틀리면", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0.0, 1.22)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("모르겠다고 적어 두는 값과, 안다고 잘못 적는 값",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             f"표본 ${N}$ 명, 각 조건 {REPS//100}00 번. 오른쪽의 띠는 "
             "추정값의 표준편차 범위다. 와이블 기저위험에서도 비례위험은 "
             "성립하므로 콕스가 겨누는 $\\beta$ 는 어느 $k$ 에서나 같다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "cox_partial.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
