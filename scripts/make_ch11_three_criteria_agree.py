r"""11.8절 「그림으로 보는 통계학」 — 세 기준이 같은 자리를 가리킬 확률.

이 쪽의 「결과」 절은 AIC·BIC·$5$ 겹 교차검증이 모두 크기 $3$ 에서 최소가 된 것을 두고
**「서로 전혀 다른 출발점에서 온 세 기준이 같은 자리를 가리킨다는 사실이 이 선택에
대한 가장 강한 근거다」**로 닫는다. 그 자료에서는 맞는 말이다. 자료를 다시 뽑으면
얼마나 자주 그렇게 되는가.

같은 설정으로 $1000$ 번 되풀이한다 — 관측 $200$ 개, 설명변수 여덟 개,
참 계수 $(3,\,-2,\,1.5,\,0,\,0,\,0,\,0,\,0)$, 잡음 표준편차 $2$. 전진선택은 AIC 로
변수를 하나씩 들이고, 세 기준이 각각 어느 크기에서 멈추는지 센다.

이 스크립트가 재는 값:

  먼저 들어온 셋이 참 변수 집합인 비율       1.000
  순서까지 x1, x2, x3 인 비율                0.978

  크기 3 을 고른 비율 — AIC 0.408   BIC 0.880   5겹 CV 0.422
  고른 크기의 평균    — AIC 3.84    BIC 3.13    5겹 CV 3.90

  둘씩 같은 크기를 고른 비율
    AIC = BIC   0.474     AIC = CV   0.568     BIC = CV   0.474
  셋이 모두 같은 크기를 고른 비율  0.333

곧 **고르는 변수는 흔들리지 않는데 어디서 멈추는지는 흔들린다.** 이 쪽이 보여 준
「세 기준의 일치」는 세 번에 한 번 오는 일이다.

만드는 파일:

  ch11/model_selection/img/three_criteria_agree.png

실행:  python3 scripts/make_ch13_three_criteria_agree.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
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

OUT = "docs/ch11/model_selection/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

N = 200
P = 8
SIGMA = 2.0
BETA = np.array([3.0, -2.0, 1.5, 0, 0, 0, 0, 0])
TRUE = {0, 1, 2}
FOLDS = 5
REPS = 1000
SEED = 7


def rss(X, y):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(((y - X @ b) ** 2).sum())


def ic(r, n, k, log_n=False):
    return n * np.log(r / n) + (np.log(n) if log_n else 2.0) * k


def cv_mse(X, y):
    n = len(y)
    fs = n // FOLDS
    tot, cnt = 0.0, 0
    for f in range(FOLDS):
        te = np.zeros(n, bool)
        te[f * fs:(f + 1) * fs] = True
        b = np.linalg.lstsq(X[~te], y[~te], rcond=None)[0]
        e = y[te] - X[te] @ b
        tot += (e ** 2).sum()
        cnt += int(te.sum())
    return tot / cnt


def one(r):
    """전진선택을 한 번 돌리고 세 기준이 멈추는 크기를 돌려준다."""
    rr = np.random.default_rng([SEED, r])
    X = rr.normal(size=(N, P))
    y = X @ BETA + rr.normal(0, SIGMA, N)
    rem, cur, order = list(range(P)), [], []
    aics, bics, cvs = [], [], []
    for _ in range(P):
        best, bj = None, None
        for j in rem:                                  # AIC 를 가장 낮추는 변수
            D = np.column_stack([np.ones(N)] + [X[:, c] for c in cur + [j]])
            v = ic(rss(D, y), N, D.shape[1])
            if best is None or v < best:
                best, bj = v, j
        cur = cur + [bj]
        rem.remove(bj)
        order.append(bj)
        D = np.column_stack([np.ones(N)] + [X[:, c] for c in cur])
        aics.append(ic(rss(D, y), N, D.shape[1]))
        bics.append(ic(rss(D, y), N, D.shape[1], log_n=True))
        cvs.append(cv_mse(D, y))
    return (order, int(np.argmin(aics)) + 1, int(np.argmin(bics)) + 1,
            int(np.argmin(cvs)) + 1)


def main():
    res = [one(r) for r in range(REPS)]
    A = np.array([a for _, a, _, _ in res])
    B = np.array([b for _, _, b, _ in res])
    C = np.array([c for _, _, _, c in res])
    first3 = float(np.mean([set(o[:3]) == TRUE for o, _, _, _ in res]))
    exact = float(np.mean([o[:3] == [0, 1, 2] for o, _, _, _ in res]))

    # 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤은 공백을 직접 적는다
    print(f"되풀이 {REPS} 번 — n = {N}, 설명변수 {P} 개, 참 계수 셋, "
          f"잡음 표준편차 {SIGMA:g}")
    print(f"  먼저 들어온 셋이 참 변수 집합인 비율  {first3:.3f}")
    print(f"  순서까지 x1, x2, x3 인 비율           {exact:.3f}")
    print("\n  기준     크기 3 을 고른 비율   평균 크기   크기별 분포(2~8)")
    dist = {}
    for nm, Z in (("AIC", A), ("BIC", B), ("5겹 CV", C)):
        d = np.bincount(Z, minlength=P + 1)[2:P + 1] / REPS
        dist[nm] = d
        print(f"  {nm:7s} {np.mean(Z == 3):16.3f} {Z.mean():12.2f}   "
              f"{np.round(d, 3)}")

    pairs = (("AIC = BIC", np.mean(A == B)), ("AIC = CV", np.mean(A == C)),
             ("BIC = CV", np.mean(B == C)),
             ("셋이 모두", np.mean((A == B) & (B == C))))
    print("\n  같은 크기를 고른 비율")
    for nm, v in pairs:
        print(f"    {nm:10s} {v:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), constrained_layout=True)

    # ── ① 고른 크기의 분포 ──
    ax = axes[0]
    sizes = np.arange(2, P + 1)
    w = 0.26
    for i, (nm, col) in enumerate((("AIC", RED), ("BIC", TEAL),
                                   ("5겹 CV", AMBER))):
        ax.bar(sizes + (i - 1) * w, dist[nm], width=w * 0.9, color=col,
               alpha=0.9, label=nm)
    ax.axvline(3, color=INK, lw=1.2, ls=":", zorder=0, alpha=0.45)
    ax.text(3.62, 0.90, "참 크기 $3$", fontsize=11.2, color=INK)
    # 크기 3 의 세 값은 상자의 표에 있으므로, 붙어 버리는 둘은 적지 않는다
    ax.text(3, dist["BIC"][1] + 0.018, f"${dist['BIC'][1]:.2f}$", ha="center",
            fontsize=10.6, color=TEAL, fontweight="bold")
    ax.set_xticks(sizes)
    ax.set_xlabel("각 기준이 멈춘 모형 크기", fontsize=11.5)
    ax.set_ylabel("비율", fontsize=11.5)
    ax.set_ylim(0, 1.0)
    ax.set_title("① 고른 변수는 같고, 멈춘 자리는 다르다", fontsize=12.8,
                 color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 일치 비율 ──
    ax = axes[1]
    names = [p[0] for p in pairs]
    vals = [float(p[1]) for p in pairs]
    cols = [MUTED, MUTED, MUTED, RED]
    ax.bar(np.arange(4), vals, width=0.52, color=cols, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.022, f"${v:.3f}$", ha="center", fontsize=12.4,
                color=cols[i], fontweight="bold")
    ax.axhline(1.0, color=INK, lw=1.2, ls="--")
    ax.text(3.45, 1.02, "늘 일치한다면", ha="right", fontsize=10.8, color=INK)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(names, fontsize=11.4)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("같은 크기를 고른 비율", fontsize=11.5)
    ax.set_title("② 「세 기준이 같은 자리를 가리켰다」가 오는 빈도",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("한 번의 일치는 얼마나 흔한 일인가", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"관측 ${N}$ 개, 설명변수 ${P}$ 개 가운데 셋만 계수가 $0$ 이 아닌 "
             f"자료를 ${REPS}$ 번 만들어 그때마다 전진선택을 돌린 것이다. "
             f"모든 수는 이 스크립트가 계산한 값이다.",
             ha="center", fontsize=10.4, color=INK)

    os.makedirs(OUT, exist_ok=True)
    path = OUT + "three_criteria_agree.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
