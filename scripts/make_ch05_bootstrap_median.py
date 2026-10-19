r"""5.3절 「그림으로 보는 통계학」 — 공식이 없는 자리에서 표준오차를 얻는 법.

평균의 표준오차에는 sigma/sqrt(n) 이라는 공식이 있지만, 중앙값에는 그렇게
간단한 공식이 없다. 붓스트랩은 손에 쥔 표본 하나를 모집단인 셈 치고 다시
뽑아 그 분포를 만들어 낸다.

왼쪽  표본 하나에서 만든 붓스트랩 분포 (실제로 할 수 있는 일)
오른쪽 모집단에서 새 표본을 2만 번 뽑아 만든 참 표집분포 (실제로는 못 하는 일)

둘의 폭이 비슷하다는 것이 붓스트랩이 듣는다는 뜻이다.

만드는 파일:

  ch05/applications/img/bootstrap_median.png

실행:  python3 scripts/make_ch05_bootstrap_median.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
TEAL, TEAL_F = "#00838F", "#D5EBED"
AMBER, AMBER_F = "#B07C2C", "#F3E7D2"
MUTED = "#90A4AE"

N, B, REP = 40, 4000, 20000
SEED = 4


def main():
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1.0, N)                       # 손에 쥔 표본 하나
    boot = np.median(rng.choice(x, (B, N), replace=True), axis=1)
    true = np.median(rng.exponential(1.0, (REP, N)), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.7), sharex=True,
                             constrained_layout=True)

    for ax, (d, col, fill, title, sub) in zip(axes, [
            (boot, TEAL, TEAL_F, "붓스트랩 분포",
             f"표본 {N} 개에서 다시 뽑기 {B:,} 번 — 실제로 할 수 있는 일"),
            (true, AMBER, AMBER_F, "참 표집분포",
             f"모집단에서 새 표본 {REP:,} 번 — 실제로는 못 하는 일")]):
        ax.hist(d, bins=50, density=True, color=fill, ec="white", lw=0.3)
        ax.axvline(d.mean(), color=col, lw=2.4)
        ax.set_title(f"{title}\n{sub}", fontsize=12, color=col, pad=8)
        ax.text(0.97, 0.90, f"표준편차 {d.std(ddof=1):.4f}", transform=ax.transAxes,
                ha="right", fontsize=13, color=col,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=col, lw=1.3))
        ax.set_xlabel("중앙값", fontsize=11.5)
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("중앙값의 표준오차 — 공식 없이 얻는다", fontsize=14.5)
    fig.text(0.5, -0.045,
             "왼쪽은 표본 하나만 가지고 만든 것이고 오른쪽은 모집단을 알아야 만들 수 있다. "
             "그런데 폭이 비슷하다 — 그래서 왼쪽을 오른쪽 대신 쓴다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "bootstrap_median.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    붓스트랩 SE {boot.std(ddof=1):.4f}   참 SE {true.std(ddof=1):.4f}")
    print(f"    참고 — 평균이라면 공식이 있다: 1/sqrt({N}) = {1/np.sqrt(N):.4f}")


if __name__ == "__main__":
    main()
