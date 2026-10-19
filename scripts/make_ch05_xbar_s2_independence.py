r"""5.2절 「그림으로 보는 통계학」 — 독립은 정규에만 있다.

같은 자료에서 계산한 두 값 X-bar 와 S^2 을 n = 8 에서 2 만 번 뽑아 흩어
그린다. 세 모집단이 세 가지 다른 모습을 보인다.

  정규    구름이 가로로 고르다 — 어느 X-bar 에서 잘라도 S^2 의 퍼짐이 같다.
          무상관일 뿐 아니라 독립이다.
  균등    상관계수는 0 이다(대칭이라 3 차 중심적률이 0 이므로). 그런데
          구름이 아치 모양이다 — X-bar 가 끝으로 가면 S^2 이 커질 수가 없다.
          **무상관인데 독립이 아닌** 본보기다.
  지수    오른쪽으로 치우쳐 3 차 중심적률이 양수라 상관계수부터 양수다.

Cov(X-bar, S^2) = mu_3 / n 이므로 상관의 부호는 모집단의 비대칭이 정한다.
독립은 그보다 훨씬 강한 요구이고, 정규분포만 그것을 만족한다.

만드는 파일:

  ch05/distributions/img/xbar_s2_independence.png

실행:  python3 scripts/make_ch05_xbar_s2_independence.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/distributions/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"

N = 8
REPS = 20_000
SEED = 3


def sample(kind, rng):
    if kind == "normal":
        return rng.normal(0.0, 1.0, (REPS, N))
    if kind == "uniform":
        return rng.uniform(0.0, 1.0, (REPS, N))
    return rng.exponential(1.0, (REPS, N))


PANELS = [("normal", "정규", TEAL,
           "가로로 고르다 — 독립이다"),
          ("uniform", "균등", AMBER,
           "상관은 0 인데 아치가 보인다"),
          ("exponential", "지수", RED,
           "치우쳐 있어 상관부터 생긴다")]


def main():
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.7), constrained_layout=True)

    for ax, (kind, nm, col, note) in zip(axes, PANELS):
        x = sample(kind, rng)
        xb, s2 = x.mean(axis=1), x.var(axis=1, ddof=1)
        r = np.corrcoef(xb, s2)[0, 1]

        ax.scatter(xb, s2, s=3, alpha=0.10, color=col, edgecolors="none")
        ax.set_title(f"{nm}모집단   상관계수 {r:+.3f}\n{note}",
                     fontsize=12.2, color=col, pad=8)
        ax.set_xlabel("표본평균", fontsize=11)
        ax.set_ylim(0, np.quantile(s2, 0.995))
        ax.set_xlim(*np.quantile(xb, [0.002, 0.998]))
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)

        # 세로로 잘라 S^2 의 퍼짐이 자리마다 달라지는지 본다
        edges = np.quantile(xb, np.linspace(0.02, 0.98, 7))
        mids, spread = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (xb >= lo) & (xb < hi)
            mids.append((lo + hi) / 2)
            spread.append(np.quantile(s2[m], [0.25, 0.75]))
        spread = np.array(spread)
        ax.plot(mids, spread[:, 0], lw=2.2, color=INK, zorder=5)
        ax.plot(mids, spread[:, 1], lw=2.2, color=INK, zorder=5)

        band = spread.mean(axis=1)            # 세로 자름마다 사분위 띠의 한가운데
        print(f"    {nm}  상관 {r:+.4f}   띠의 높이 "
              f"왼끝 {band[0]:.4f} · 가운데 {band[len(band)//2]:.4f} · 오른끝 {band[-1]:.4f}")

    axes[0].set_ylabel("표본분산", fontsize=11)
    fig.suptitle(f"같은 자료에서 잰 두 값을 흩어 그렸다  (n = {N}, {REPS//1000} 천 번)",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             "검은 두 선은 세로로 잘랐을 때 표본분산의 사분위 구간이다. "
             "정규에서만 그 띠가 어디서나 같은 높이에 같은 두께로 놓인다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "xbar_s2_independence.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
