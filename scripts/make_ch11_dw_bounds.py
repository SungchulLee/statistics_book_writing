r"""11.3절 「그림으로 보는 통계학」 — 답을 못 적은 칸이 있는 표.

Durbin-Watson 통계량의 귀무분포는 **설명변수 행렬 X 에 따라 달라진다.**
그래서 n 과 k 만으로는 임계값을 하나로 적을 수가 없다. 더빈과 왓슨이
표에 d_L 과 d_U 두 수를 싣고 그 사이를 「판정 불가」로 남긴 까닭이다.

이 그림은 그 X 의존을 직접 보인다. n = 30, 설명변수 2 개로 고정하고
X 만 세 가지로 바꾸어(무작위 · 추세 · 계절) 귀무분포를 각각 그린다.
오차는 세 경우 모두 독립인 정규다 — 곧 H0 가 참이다.

세 분포의 5% 분위수가 서로 다르다. 어느 X 를 쓰느냐에 따라 같은 d 값이
기각도 되고 아니기도 한다는 뜻이다.

(공식 표의 d_L·d_U 는 **가능한 모든 X** 를 가두는 값이므로 여기서 잰
폭보다 넓다. 이 그림은 그 폭이 왜 생기는지를 보일 뿐 표를 다시 만들지
않는다.)

만드는 파일:

  ch11/assumptions/img/dw_depends_on_x.png

실행:  python3 scripts/make_ch13_dw_bounds.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

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
MUTED = "#90A4AE"

N = 30
REPS = 200_000
SEED = 7


def designs():
    """설명변수 2 개 + 절편. n 과 k 는 같고 X 만 다르다."""
    t = np.arange(N)
    rng = np.random.default_rng(1)
    return [
        ("무작위", rng.normal(size=(N, 2)), MUTED),
        ("추세", np.column_stack([t, t ** 2]), TEAL),
        ("계절", np.column_stack([np.sin(2 * np.pi * t / 12),
                                  np.cos(2 * np.pi * t / 12)]), AMBER),
    ]


def dw_null(X, rng):
    """H0(오차가 독립) 아래에서 DW 통계량을 REPS 개 뽑는다."""
    A = np.column_stack([np.ones(N), X])
    M = np.eye(N) - A @ np.linalg.pinv(A)        # 잔차 만드는 사영
    e = rng.normal(size=(REPS, N)) @ M.T
    return (np.diff(e, axis=1) ** 2).sum(axis=1) / (e ** 2).sum(axis=1)


def main():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(10.6, 5.4), constrained_layout=True)

    grid = np.linspace(0.6, 3.4, 600)
    q05 = []
    for name, X, col in designs():
        d = dw_null(X, rng)
        q = np.quantile(d, 0.05)
        q05.append((name, q, col))
        hist, edges = np.histogram(d, bins=220, range=(0.6, 3.4), density=True)
        ax.plot((edges[:-1] + edges[1:]) / 2, hist, lw=2.4, color=col,
                label=f"{name} X   5% 분위수 {q:.3f}")
        ax.axvline(q, color=col, lw=1.8, ls=":")
        print(f"    {name:4} X   평균 {d.mean():.4f}   5% 분위수 {q:.4f}")

    lo = min(q for _, q, _ in q05)
    hi = max(q for _, q, _ in q05)
    ax.axvspan(lo, hi, color=RED, alpha=0.14, zorder=0)
    ax.annotate(f"여기 떨어지면 X 를 봐야\n알 수 있다 (폭 {hi-lo:.3f})",
                xy=((lo + hi) / 2, 0.22), xytext=((lo + hi) / 2 - 0.62, 0.72),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax.set_xlabel("Durbin-Watson 통계량 $d$", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title(f"오차는 셋 다 독립인데 귀무분포가 다르다  "
                 f"(n = {N}, 설명변수 2 개, {REPS//1000} 천 번)",
                 fontsize=13.2, color=INK, pad=10)
    ax.set_xlim(0.6, 3.4)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.5, -0.045,
             "같은 n, 같은 설명변수 개수인데 5% 임계값이 세 자리로 흩어진다. "
             "표에 수 하나를 적을 수 없는 까닭이 이것이고, 공식 표의 $d_L$·$d_U$ 는 "
             "가능한 모든 $X$ 를 가두므로 이 폭보다 넓다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "dw_depends_on_x.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    5% 임계값의 폭 {lo:.4f} ~ {hi:.4f}  (차 {hi-lo:.4f})")


if __name__ == "__main__":
    main()
