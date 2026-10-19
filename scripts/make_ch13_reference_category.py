r"""13.1절 「그림으로 보는 통계학」 — 기준범주는 OLS 에서만 무해하다.

더미 부호화에서 어느 범주를 기준으로 삼을지는 자유다. 최소제곱에서는
그 선택이 **계수의 뜻만 바꾸고 적합값은 그대로** 둔다 — 이 쪽 §2 와
정리하며가 적은 그대로다.

그런데 벌점을 붙이는 순간 그 말이 깨진다. 능형·라쏘는 **계수벡터의
크기**를 벌하는데, 기준을 바꾸면 같은 모형이 다른 크기의 계수벡터로
적히기 때문이다. 벌점이 보는 것이 달라지니 축소도 달라진다.

네 집단의 참 평균을 10, 12, 11, 20 으로 두고 재면

  lambda = 0    네 기준이 모두 같은 적합값을 낸다
  lambda = 30   기준에 따라 집단 D 의 적합값이 13.95 에서 15.48 까지 벌어진다

만드는 파일:

  ch13/linear_regression/img/reference_category.png

실행:  python3 scripts/make_ch13_reference_category.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch13/linear_regression/img/"

INK = "#37474F"
MUTED = "#90A4AE"
COLS = ["#37474F", "#1565C0", "#B07C2C", "#00838F"]
RED = "#D32F2F"

MU = np.array([10.0, 12.0, 11.0, 20.0])
NAMES = list("ABCD")
M, SEED = 20, 3
LAMS = np.concatenate([[0.0], np.geomspace(0.3, 60, 40)])
SHOW = (0.0, 30.0)


def main():
    rng = np.random.default_rng(SEED)
    g = np.repeat(np.arange(4), M)
    n = len(g)
    y = MU[g] + rng.normal(0, 2.0, n)
    obs = np.array([y[g == j].mean() for j in range(4)])
    print("    집단별 표본평균", np.round(obs, 4))

    def fitted(ref, lam):
        cols = [j for j in range(4) if j != ref]
        A = np.column_stack([np.ones(n)] + [(g == j).astype(float) for j in cols])
        P = np.eye(A.shape[1])
        P[0, 0] = 0.0                       # 절편은 벌하지 않는다
        b = np.linalg.solve(A.T @ A + lam * P, A.T @ y)
        return np.array([A[g == j][0] @ b for j in range(4)]), float(b[1:] @ b[1:])

    for lam in SHOW:
        print(f"    lambda = {lam:g}")
        for ref in range(4):
            f, pen = fitted(ref, lam)
            print(f"      기준 {NAMES[ref]}   적합값 {np.round(f,4)}   벌점항 {pen:7.2f}")
    spread = {lam: max(fitted(r, lam)[0][3] for r in range(4))
                   - min(fitted(r, lam)[0][3] for r in range(4)) for lam in SHOW}
    print(f"    집단 D 적합값의 폭 — lambda 0: {spread[0.0]:.4f}, "
          f"lambda 30: {spread[30.0]:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.6), constrained_layout=True)

    for ax, lam in zip(axes[:2], SHOW):
        w = 0.2
        for k, ref in enumerate(range(4)):
            f, _ = fitted(ref, lam)
            ax.bar(np.arange(4) + (k - 1.5) * w, f, width=w, color=COLS[k],
                   alpha=0.9, label=f"기준 {NAMES[ref]}")
        ax.plot(np.arange(4), obs, "k_", ms=34, mew=2.4, zorder=6)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"집단 {c}" for c in NAMES], fontsize=11)
        ax.set_ylim(0, 22)
        ax.set_ylabel("적합한 집단 평균", fontsize=11.5)
        ax.set_title(f"$\\lambda = {lam:g}$ " +
                     ("— 넷이 똑같다" if lam == 0 else "— 넷이 갈라진다"),
                     fontsize=12.8, color=(INK if lam == 0 else RED), pad=8)
        ax.grid(alpha=0.22, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=9.6, framealpha=0.93, ncol=2, loc="upper left")
    axes[0].text(0.02, 0.63, "검은 막대는 표본평균", transform=axes[0].transAxes,
                 fontsize=10, color=INK)
    axes[1].annotate(f"폭 {spread[30.0]:.2f}", xy=(3, 15.4), xytext=(2.1, 19.2),
                     fontsize=11, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))

    ax = axes[2]
    for k, ref in enumerate(range(4)):
        vals = [fitted(ref, l)[0][3] for l in LAMS]
        ax.plot(LAMS, vals, lw=2.4, color=COLS[k], label=f"기준 {NAMES[ref]}")
    ax.axhline(obs[3], color=MUTED, lw=1.8, ls="--")
    ax.text(30, obs[3] - 0.30, "벌점이 없을 때의 값 (표본평균)",
            ha="center", fontsize=10.2, color=MUTED)
    ax.set_xlabel("벌점의 세기 $\\lambda$", fontsize=11.5)
    ax.set_ylabel("집단 D 의 적합값", fontsize=11.5)
    ax.set_title("$\\lambda$ 가 커질수록 벌어진다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=9.6, framealpha=0.93)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("기준범주를 바꾸면 벌점이 보는 것이 달라진다", fontsize=14.5)
    fig.text(0.5, -0.055,
             "네 집단의 참 평균은 $10, 12, 11, 20$ 이다. 왼쪽에서 네 기준이 "
             "하나로 겹치는 것이 최소제곱의 성질이고, 가운데·오른쪽이 그 성질이 "
             "벌점 아래에서 사라지는 모습이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "reference_category.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
