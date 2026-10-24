r"""11.5절 「그림으로 보는 통계학」 — 잔차는 오차가 아니다.

이 쪽의 검정 넷은 모두 **잔차**에 걸린다. 그런데 가정이 말하는 것은 **오차**의
정규성이다. 둘은 같지 않다 — 잔차는 $\mathbf e = (\mathbf I - \mathbf P)\boldsymbol\varepsilon$
이므로 오차들의 **일차결합**이고, 섞을수록 정규 쪽으로 끌려간다.

얼마나 끌려가는지 잰다. 관측 $100$ 개에 설명변수 수 $p$ 를 바꿔 가며, 같은 자료에서
**오차 자체**와 **잔차**에 각각 샤피로–윌크 검정을 걸어 기각률을 비교한다.

이 스크립트가 재는 값(되풀이 1000 번, 유의수준 0.05):

  오차가 균등분포일 때 (초과첨도 -1.2)
     p    오차에서 기각   잔차에서 기각   잔차의 평균 초과첨도
     2        0.998          0.988            -1.146
    10        0.993          0.781            -0.991
    25        0.996          0.242            -0.692
    50        0.995          0.051            -0.334
    70        0.997          0.037            -0.156

  오차가 지수분포일 때 (초과첨도 6.0)
     2        1.000          1.000             4.064
    10        1.000          1.000             3.327
    25        1.000          0.978             2.359
    50        1.000          0.598             1.042
    70        1.000          0.199             0.310

곧 $p = 50$ 에서 **오차는 $99.5\%$ 기각되는데 잔차는 $5.1\%$ 기각된다** — 유의수준
그대로다. 검정이 눈을 감은 것이다. 스튜던트화해도 달라지지 않는다($0.051 \to 0.055$).

만드는 파일:

  ch11/assumptions/img/residuals_are_not_errors.png

실행:  python3 scripts/make_ch13_residuals_are_not_errors.py   (저장소 최상위에서)
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

OUT = "docs/ch11/assumptions/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 100
PS = (2, 10, 25, 50, 70)
REPS = 1000
ALPHA = 0.05
SEED = 23
KINDS = (("균등", -1.2, TEAL), ("지수", 6.0, RED))


def draw(rr, kind):
    if kind == "균등":
        return rr.uniform(-np.sqrt(3), np.sqrt(3), N)
    return rr.exponential(1.0, N) - 1.0


def main():
    res = {}
    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"n = {N}, 되풀이 {REPS} 번, 유의수준 {ALPHA}")
    for kind, truek, _ in KINDS:
        rows = []
        print(f"\n  오차가 {kind}분포일 때 (참 초과첨도 {truek})")
        print("     p    오차에서 기각   잔차에서 기각   잔차의 평균 초과첨도")
        for p in PS:
            re_, rr_, ku = [], [], []
            for r in range(REPS):
                rng = np.random.default_rng([SEED, p, r])
                eps = draw(rng, kind)
                X = np.column_stack([np.ones(N),
                                     rng.normal(size=(N, p - 1))])
                y = X @ np.ones(p) + eps
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                e = y - X @ b
                re_.append(stats.shapiro(eps).pvalue < ALPHA)
                rr_.append(stats.shapiro(e).pvalue < ALPHA)
                ku.append(stats.kurtosis(e))
            rows.append((p, float(np.mean(re_)), float(np.mean(rr_)),
                         float(np.mean(ku))))
            print(f"  {p:4d} {rows[-1][1]:14.3f} {rows[-1][2]:15.3f} "
                  f"{rows[-1][3]:19.3f}")
        res[kind] = rows

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 기각률 ──
    ax = axes[0]
    for kind, truek, col in KINDS:
        rows = res[kind]
        ax.plot(PS, [r[1] for r in rows], "s--", ms=8, lw=2.0, color=col,
                alpha=0.55, label=f"{kind} 오차 — **오차**에 건 검정")
        ax.plot(PS, [r[2] for r in rows], "o-", ms=10, lw=2.8, color=col,
                label=f"{kind} 오차 — **잔차**에 건 검정")
    ax.axhline(ALPHA, color=INK, lw=1.6, ls=":")
    ax.text(68, ALPHA + 0.035, f"유의수준 ${ALPHA}$", ha="right", fontsize=10.8,
            color=INK)
    r50 = [r for r in res["균등"] if r[0] == 50][0]
    ax.annotate(f"$p = 50$ 에서\n오차 ${r50[1]:.3f}$ 대 잔차 ${r50[2]:.3f}$",
                xy=(50, r50[2]), xytext=(23, 0.33), fontsize=11.2, color=TEAL,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("설명변수의 수 $p$  (관측은 $100$ 개로 고정)", fontsize=11.5)
    ax.set_ylabel(f"${ALPHA}$ 수준에서 기각한 비율", fontsize=11.5)
    ax.set_ylim(-0.03, 1.12)
    ax.set_title("① 같은 자료인데 오차는 잡히고 잔차는 안 잡힌다",
                 fontsize=12.6, color=INK)
    ax.grid(alpha=0.22)
    leg = ax.legend(fontsize=9.4, framealpha=0.93, loc="center left")
    for t in leg.get_texts():                       # 마크다운은 쓰지 못한다
        t.set_text(t.get_text().replace("**", ""))
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 초과첨도가 0 쪽으로 ──
    ax = axes[1]
    for kind, truek, col in KINDS:
        rows = res[kind]
        ax.plot(PS, [r[3] for r in rows], "o-", ms=10, lw=2.8, color=col,
                label=f"{kind} 오차에서 나온 잔차")
        ax.axhline(truek, color=col, lw=1.6, ls="--", alpha=0.6)
        # 위쪽 선의 꼬리표는 왼쪽에 둔다 — 오른쪽 위는 범례 자리다
        ax.text(3 if truek > 0 else 70, truek + (0.28 if truek > 0 else -0.45),
                f"오차의 참값 ${truek}$",
                ha="left" if truek > 0 else "right", fontsize=10.6, color=col)
    ax.axhline(0, color=INK, lw=1.8)
    ax.text(3, 0.3, "정규분포의 $0$", fontsize=11, color=INK)
    ax.set_xlabel("설명변수의 수 $p$", fontsize=11.5)
    ax.set_ylabel("잔차의 평균 초과첨도", fontsize=11.5)
    ax.set_ylim(-2.1, 7.0)
    ax.set_title("② 섞을수록 정규 쪽으로 끌려간다", fontsize=12.6, color=INK)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("가정은 오차를 말하는데 검정은 잔차를 본다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개짜리 자료를 ${REPS}$ 번 만들어 같은 자료의 오차와 "
             f"잔차에 각각 샤피로–윌크 검정을 걸었다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "residuals_are_not_errors.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
