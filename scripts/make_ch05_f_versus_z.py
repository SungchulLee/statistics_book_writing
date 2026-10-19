r"""5.2절 「그림으로 보는 통계학」 — 피셔가 F 대신 z 를 쓴 까닭.

F 는 비(比)라서 오른쪽으로 길게 늘어진다. 로그를 씌워 반으로 나눈
z = (1/2) ln F 는 거의 대칭이 된다. 손으로 표를 만들고 정규근사에 기대던
시절에는 그 차이가 결정적이었다 — 피셔가 z 로 적고, 스네데커가 F 로
표를 만들어 퍼뜨렸다.

F(8,12) 에서 왜도가 +3.0 에서 -0.15 로 떨어진다.

만드는 파일:

  ch05/distributions/img/f_versus_z.png

실행:  python3 scripts/make_ch05_f_versus_z.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "ch05/distributions/img/"
OUT = "docs/" + OUT

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
MUTED = "#90A4AE"

D1, D2 = 8, 12
SEED = 7


def main():
    f = stats.f.rvs(D1, D2, size=400_000, random_state=SEED)
    z = 0.5 * np.log(f)

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)

    # ── 왼쪽: F 그대로 ──
    ax = axes[0]
    xs = np.linspace(0.01, 6, 600)
    ax.plot(xs, stats.f.pdf(xs, D1, D2), lw=3.0, color=AMBER)
    ax.axvline(f.mean(), color=MUTED, lw=2.0, ls="--")
    ax.axvline(np.median(f), color=INK, lw=2.0, ls=":")
    ax.text(f.mean() + 0.12, ax.get_ylim()[1] * 0.80,
            f"평균 {f.mean():.2f}", fontsize=10.5, color=MUTED)
    ax.text(np.median(f) - 0.12, ax.get_ylim()[1] * 0.93,
            f"중앙값 {np.median(f):.2f}", fontsize=10.5, color=INK, ha="right")
    ax.set_title(f"F({D1}, {D2}) — 비(比)라서 오른쪽으로 늘어진다\n왜도 {stats.skew(f):+.2f}",
                 fontsize=12.5, color=AMBER, pad=8)
    ax.set_xlabel("F", fontsize=11.5)
    ax.set_xlim(0, 6)

    # ── 오른쪽: z = (1/2) ln F ──
    ax = axes[1]
    zs = np.linspace(-2.2, 2.2, 600)
    ax.hist(z, bins=140, range=(-2.2, 2.2), density=True,
            color="#DCEBFB", ec="white", lw=0.3)
    ax.plot(zs, stats.norm.pdf(zs, z.mean(), z.std()), lw=2.6, color=TEAL)
    ax.axvline(z.mean(), color=MUTED, lw=2.0, ls="--")
    ax.axvline(np.median(z), color=INK, lw=2.0, ls=":")
    ax.text(0.03, 0.80, f"평균 {z.mean():+.3f}\n중앙값 {np.median(z):+.3f}\n거의 겹친다",
            transform=ax.transAxes, fontsize=10.5, color=INK)
    ax.set_title(f"z = ½ ln F — 거의 대칭이다\n왜도 {stats.skew(z):+.2f}",
                 fontsize=12.5, color=TEAL, pad=8)
    ax.set_xlabel("z", fontsize=11.5)
    ax.set_xlim(-2.2, 2.2)

    for ax in axes:
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle("같은 분포를 두 가지로 적으면 — 한쪽은 정규가 듣고 한쪽은 안 듣는다",
                 fontsize=14)
    fig.text(0.5, -0.04,
             "청록 곡선은 같은 평균·표준편차의 정규분포다. 오른쪽에서는 겹치고 왼쪽에서는 겹칠 수가 없다. "
             "표를 손으로 만들던 시절에 피셔가 z 로 적은 까닭이 이것이다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "f_versus_z.png"
    fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    F 왜도 {stats.skew(f):+.3f} · 평균 {f.mean():.3f} · 중앙값 {np.median(f):.3f}")
    print(f"    z 왜도 {stats.skew(z):+.3f} · 평균 {z.mean():+.4f} · 중앙값 {np.median(z):+.4f}")


if __name__ == "__main__":
    main()
