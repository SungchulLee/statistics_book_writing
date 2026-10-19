r"""1.2절 「그림으로 보는 통계학」 — 빈칸을 평균으로 메우면.

설문에서 답을 안 한 사람이 생기면 흔히 두 가지 가운데 하나를 한다.
그 사람을 빼거나, 빈칸을 그 문항의 평균으로 메운다. 둘 다 점추정이
똑같고, 둘 다 틀린다.

도널드 루빈이 1987 년에 『Multiple Imputation for Nonresponse in
Surveys』에서 다른 길을 적었다. 빈칸을 **한 번이 아니라 여러 번** 메우되
매번 다르게 메우고, 그렇게 얻은 여러 답을 합치라는 것이다. 합치는 규칙도
함께 주었다.

  T = U-bar + (1 + 1/m) B

U-bar 는 대입 안쪽의 분산이고 B 는 대입 사이의 분산이다. 둘째 항이
「빈칸을 어떻게 메우느냐에 따라 답이 달라진다」는 사실을 값으로 적는
자리이고, 평균으로 한 번 메우는 방법에는 그 자리가 없다.

n = 300, y = 1 + 0.8x + e, x 가 클수록 더 많이 빠지게(MAR) 두고 재면

  결측비율    있는 것만             평균 대입              다중대입(m=20)
    0.1    -0.116  포함 0.691   -0.116  0.569    +0.002  0.958
    0.2    -0.192       0.361   -0.192  0.195    +0.003  0.948
    0.3    -0.256       0.172   -0.256  0.054    +0.002  0.946
    0.5    -0.374       0.044   -0.374  0.004    -0.000  0.946
    0.7    -0.502       0.016   -0.502  0.000    -0.001  0.952

앞의 두 줄은 편향이 **글자 하나까지 같다.** 점추정이 같은 식이기
때문이다. 다른 것은 구간의 너비뿐이고, 평균 대입 쪽이 더 좁다.
같은 오답에 더 큰 자신감을 붙이는 셈이다.

만드는 파일:

  ch01/classical/img/multiple_imputation.png

실행:  python3 scripts/make_ch01_multiple_imputation.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import t as tdist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch01/classical/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, B1, SIG, M, REPS, SEED = 300, 0.8, 1.0, 20, 2500, 1987
TRUE, SLOPE_MIS = 1.0, 1.2
FRACS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
#        열쇠     색     선     표식   이름
WAYS = (("cc", RED, "-", "o", "있는 것만 쓴다"),
        ("mean", AMBER, "--", "s", "빈칸을 평균으로 메운다"),
        ("mi", TEAL, "-", "D", f"다중대입 — 서로 다르게 ${M}$ 번 메운다"))


def once(frac, rng):
    """자료 한 벌에서 세 방법의 (점추정, 구간 반너비) 를 돌려준다."""
    x = rng.normal(size=N)
    y = TRUE + B1 * x + rng.normal(0, SIG, N)
    a = np.log(frac / (1 - frac))
    mis = rng.random(N) < 1 / (1 + np.exp(-(a + SLOPE_MIS * x)))
    if (~mis).sum() < 12:
        return None
    yo, xo = y[~mis], x[~mis]
    n_o = len(yo)

    res = {}
    se = yo.std(ddof=1) / np.sqrt(n_o)                      # 있는 것만
    res["cc"] = (yo.mean(), 1.96 * se)

    f = y.copy()
    f[mis] = yo.mean()                                      # 평균으로 메우기
    res["mean"] = (f.mean(), 1.96 * f.std(ddof=1) / np.sqrt(N))

    X = np.c_[np.ones(n_o), xo]                             # 다중대입
    XtXi = np.linalg.inv(X.T @ X)
    bh = XtXi @ X.T @ yo
    r = yo - X @ bh
    s2 = r @ r / (n_o - 2)
    L = np.linalg.cholesky(XtXi)
    qs, us = np.empty(M), np.empty(M)
    for i in range(M):
        s2s = s2 * (n_o - 2) / rng.chisquare(n_o - 2)       # 사후에서 뽑는다
        bs = bh + np.sqrt(s2s) * (L @ rng.normal(size=2))
        ym = y.copy()
        ym[mis] = bs[0] + bs[1] * x[mis] + rng.normal(0, np.sqrt(s2s),
                                                      mis.sum())
        qs[i], us[i] = ym.mean(), ym.var(ddof=1) / N
    Q, U, Bv = qs.mean(), us.mean(), qs.var(ddof=1)
    T = U + (1 + 1 / M) * Bv                                # 루빈의 규칙
    df = (M - 1) * (1 + U / ((1 + 1 / M) * Bv)) ** 2
    res["mi"] = (Q, float(tdist.ppf(0.975, df) * np.sqrt(T)))
    return res


def main():
    rng = np.random.default_rng(SEED)
    est = {k: [] for k, *_ in WAYS}
    half = {k: [] for k, *_ in WAYS}
    cov = {k: [] for k, *_ in WAYS}

    print(f"{'결측비율':>8}" + "".join(f"{w[-1].split(' —')[0]:>24}"
                                   for w in WAYS))
    for frac in FRACS:
        box = {k: ([], [], []) for k, *_ in WAYS}
        for _ in range(REPS):
            r = once(frac, rng)
            if r is None:
                continue
            for k, (e, h) in r.items():
                box[k][0].append(e)
                box[k][1].append(h)
                box[k][2].append(abs(e - TRUE) <= h)
        line = ""
        for k, *_ in WAYS:
            e, h, c = (np.mean(box[k][0]), np.mean(box[k][1]),
                       np.mean(box[k][2]))
            est[k].append(e); half[k].append(h); cov[k].append(c)
            line += f"   편향{e-TRUE:+.3f} 포함{c:.3f}"
        print(f"{frac:>8.2f}{line}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 구간이 참값을 떠난다 ──
    ax = axes[0]
    ax.axhline(TRUE, color=INK, lw=2.4, ls="--")
    ax.text(0.052, TRUE + 0.035, f"참값 ${TRUE:.0f}$", fontsize=10.8,
            color=INK)
    for k, col, ls, mk, lab in WAYS:
        e, h = np.array(est[k]), np.array(half[k])
        ax.fill_between(FRACS, e - h, e + h, color=col, alpha=0.18, zorder=2)
        ax.plot(FRACS, e, lw=3.4, color=col, ls=ls, marker=mk, ms=7,
                label=lab, zorder=3)
    ax.annotate("두 방법의 점추정이\n글자 하나까지 같다",
                xy=(0.52, est["cc"][FRACS.index(0.5)]), xytext=(0.30, -0.02),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("답하지 않은 사람의 비율", fontsize=11.5)
    ax.set_ylabel("추정값과 $95\\%$ 구간", fontsize=11.5)
    ax.set_title("구간이 참값을 떠난다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0.03, 0.72)
    ax.set_ylim(-0.1, 1.45)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 95% 라고 적어 놓은 구간이 ──
    ax = axes[1]
    ax.axhline(0.95, color=INK, lw=2.4, ls="--")
    ax.text(0.052, 0.90, "약속한 $0.95$", fontsize=10.8, color=INK)
    for k, col, ls, mk, lab in WAYS:
        ax.plot(FRACS, cov[k], lw=3.4, color=col, ls=ls, marker=mk, ms=7,
                label=lab)
    i3 = FRACS.index(0.3)
    ax.annotate(f"결측이 ${FRACS[i3]*100:.0f}\\%$ 면\n"
                f"평균 대입의 포함률이 ${cov['mean'][i3]:.3f}$ 다",
                xy=(FRACS[i3], cov["mean"][i3]), xytext=(0.47, 0.30),
                fontsize=10.8, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.set_xlabel("답하지 않은 사람의 비율", fontsize=11.5)
    ax.set_ylabel("$95\\%$ 구간이 참값을 담은 비율", fontsize=11.5)
    ax.set_title("$95\\%$ 라고 적어 놓은 구간이", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(0.03, 0.72)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="center left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("빈칸을 평균으로 메우면", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"관측 ${N}$ 명, $y = 1 + {B1}x + \\varepsilon$ 이고 $x$ 가 "
             "클수록 답하지 않을 확률이 높다. 재려는 것은 $y$ 의 평균이고 "
             f"참값은 ${TRUE:.0f}$ 이다. 각 점은 ${REPS//1000}{{,}}500$ 번의 "
             "결과다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "multiple_imputation.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
