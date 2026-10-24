r"""11.2절 「그림으로 보는 통계학」 — 남는 식을 버릴 것인가 쓸 것인가.

미지수가 3 개인데 관측이 41 개면 식이 38 개 남는다. 최소제곱 이전에
쓰이던 두 가지 길을 나란히 놓고, 짧은 호에서 멀리 외삽했을 때의 오차를
잰다.

  세 점만      딱 필요한 만큼만 골라 정확히 푼다. 나머지 38 개는 버린다.
  평균법       마이어(1750)의 것. 41 개를 셋으로 갈라 묶음마다 더해
               3x3 으로 만든다. 잡음이 묶음 안에서 상쇄된다.
  최소제곱     41 개를 다 쓰고 잔차제곱합을 최소로 한다.

관측은 t in [0, 1] 의 짧은 구간이고 예측은 t = 8 에서 한다 — 41 일을
보고 열 달 뒤를 맞히던 사정을 흉내 낸 것이다.

만드는 파일:

  ch11/estimation_inference/img/least_squares_vs_rivals.png

실행:  python3 scripts/make_ch13_least_squares_vs_rivals.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/estimation_inference/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N, REPS, SD, SEED = 41, 4000, 0.02, 1801
TRUE = np.array([1.0, 0.6, -0.25])
TPRED = 8.0

t = np.linspace(0, 1, N)
V = np.column_stack([np.ones(N), t, t ** 2])
vp = np.array([1.0, TPRED, TPRED ** 2])
TRUE_PRED = float(vp @ TRUE)
GROUPS = np.array_split(np.arange(N), 3)
PICK = np.array([0, N // 2, N - 1])


def avg_method(y):
    """마이어의 평균법 — 셋으로 갈라 묶음마다 더해 3x3 을 푼다."""
    M = np.array([V[g].sum(axis=0) for g in GROUPS])
    r = np.array([y[g].sum() for g in GROUPS])
    return np.linalg.solve(M, r)


def three_only(y):
    return np.linalg.solve(V[PICK], y[PICK])


METHODS = [("세 점만 골라", three_only, RED),
           ("마이어의 평균법", avg_method, AMBER),
           ("최소제곱", lambda y: np.linalg.lstsq(V, y, rcond=None)[0], TEAL)]


def main():
    rng = np.random.default_rng(SEED)
    errs = {nm: [] for nm, _, _ in METHODS}
    for _ in range(REPS):
        y = V @ TRUE + rng.normal(0, SD, N)
        for nm, f, _ in METHODS:
            errs[nm].append(abs(float(vp @ f(y)) - TRUE_PRED))
    med = {nm: float(np.median(v)) for nm, v in errs.items()}
    for nm, _, _ in METHODS:
        print(f"    {nm:10} 예측오차 중앙값 {med[nm]:7.3f}"
              f"   (세 점 대비 {med['세 점만 골라']/med[nm]:.2f} 배)")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 한 표본에서 세 길이 갈라지는 모습 ──
    ax = axes[0]
    rng2 = np.random.default_rng(11)
    y = V @ TRUE + rng2.normal(0, SD, N)
    g = np.linspace(0, TPRED + 0.6, 300)
    G = np.column_stack([np.ones_like(g), g, g ** 2])
    ax.plot(g, G @ TRUE, lw=3.4, color=MUTED, alpha=0.8, label="참 궤적")
    for nm, f, col in METHODS:
        ax.plot(g, G @ f(y), lw=2.2, color=col, label=nm)
    ax.scatter(t, y, s=20, color=INK, zorder=5, label=f"관측 {N} 개")
    ax.axvspan(0, 1, color=INK, alpha=0.07)
    ax.axvline(TPRED, color=INK, lw=1.4, ls=":")
    ax.text(1.1, (G @ TRUE).max() * 0.98, "여기까지만 보았다", fontsize=10.4, color=INK)
    ax.text(TPRED - 0.2, (G @ TRUE).min() + 0.4, "여기를 맞혀야 한다",
            fontsize=10.4, color=INK, ha="right")
    ax.set_xlabel("$t$", fontsize=12)
    ax.set_ylabel("$y$", fontsize=12)
    ax.set_title("짧게 보고 멀리 내다본다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=9.8, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 예측오차의 분포 ──
    ax = axes[1]
    data = [errs[nm] for nm, _, _ in METHODS]
    bp = ax.boxplot(data, vert=False, widths=0.55, showfliers=False,
                    patch_artist=True, medianprops=dict(color=INK, lw=2.2))
    for patch, (_, _, col) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    ax.set_yticks(range(1, len(METHODS) + 1))
    ax.set_yticklabels([nm for nm, _, _ in METHODS], fontsize=11.5)
    for i, (nm, _, _) in enumerate(METHODS, start=1):
        ax.text(med[nm], i + 0.33, f"중앙값 {med[nm]:.2f}",
                fontsize=10.4, color=INK, ha="center")
    ax.set_xlabel(f"$t = {TPRED:g}$ 에서의 예측오차", fontsize=11.5)
    ax.set_title(f"같은 자료를 어떻게 쓰느냐만 다르다  ({REPS//1000} 천 번)",
                 fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25, axis="x")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("남는 식을 버리지 않는 것만으로 오차가 절반 아래로", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"미지수 셋에 관측 {N} 개라 식이 {N-3} 개 남는다. 세 길 모두 치우침이 "
             f"없는데 흩어짐만 다르고, 그 차이가 멀리 내다볼수록 커진다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "least_squares_vs_rivals.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
