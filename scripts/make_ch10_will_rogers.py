r"""10.5절 「그림으로 보는 통계학」 — 아무도 낫지 않았는데 모두 좋아졌다.

윌 로저스가 했다고 전해지는 말이 있다. 「오클라호마 사람들이
캘리포니아로 옮겨 가면서 두 주의 평균 지능이 모두 올랐다.」

1985 년에 파인스타인과 소신과 웰스가 『뉴잉글랜드 의학저널』에 폐암
자료에서 꼭 그 일이 일어났다고 보고했다. 그들은 이것을 병기 이동
(stage migration), 또는 윌 로저스 현상이라 불렀다.

까닭은 영상 장비가 좋아진 것이다. 전산화단층촬영과 골스캔이 들어오면서
예전에는 보이지 않던 작은 전이가 보이기 시작했다. 그 환자들은 어제까지
초기 병기였고 오늘부터 진행 병기다.

  초기 병기는 가장 나쁜 환자들을 잃었다  -> 초기의 생존율이 오른다
  진행 병기는 기존 환자들보다 나은 사람들을 받았다 -> 진행의 생존율도 오른다
  아무도 치료를 더 받지 않았다 -> 전체 생존율은 그대로다

모의로 재면 (문턱 0.70 -> 0.45)

                      예전 장비   새 장비     변화
  초기 병기의 생존기간    49.2      55.9      +6.6 개월
  진행 병기의 생존기간    10.5      15.2      +4.8 개월
  전체                  30.0      30.0       0.0

  병기가 바뀐 환자가 전체의 14.1% 다.

만드는 파일:

  ch10/ecological_correlation/img/will_rogers.png

실행:  python3 scripts/make_ch12_will_rogers.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch10/ecological_correlation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, SEED = 300_000, 1985
T_OLD, T_NEW = 0.70, 0.45
NSHOW = 260


def cohort(rng, n):
    """숨은 중증도 z 와 생존기간(개월). 중증도가 클수록 짧다."""
    z = rng.exponential(1.0, n)
    return z, 72 * np.exp(-1.5 * z) * rng.lognormal(0, 0.28, n)


def means(z, s, t):
    """문턱 t 로 나눈 두 병기의 평균 생존기간."""
    return s[z <= t].mean(), s[z > t].mean()


def main():
    rng = np.random.default_rng(SEED)
    z, s = cohort(rng, N)
    e_old, a_old = means(z, s, T_OLD)
    e_new, a_new = means(z, s, T_NEW)

    print(f"  환자 {N:,} 명,  전체 평균 생존 {s.mean():.1f} 개월")
    print(f"{'':>14}{'예전 장비':>12}{'새 장비':>12}{'변화':>10}")
    print(f"{'초기 병기':>14}{e_old:>12.1f}{e_new:>12.1f}{e_new-e_old:>+10.1f}")
    print(f"{'진행 병기':>14}{a_old:>12.1f}{a_new:>12.1f}{a_new-a_old:>+10.1f}")
    print(f"{'전체':>14}{s.mean():>12.1f}{s.mean():>12.1f}{0.0:>+10.1f}")
    moved = ((z > T_NEW) & (z <= T_OLD)).mean()
    print(f"\n  병기가 바뀐 환자 {moved*100:.1f}%  "
          f"(그들의 평균 생존 {s[(z > T_NEW) & (z <= T_OLD)].mean():.1f} 개월)")

    ts = np.linspace(0.08, 1.4, 60)
    curves = np.array([means(z, s, t) for t in ts])

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 옮겨 간 사람들 ──
    ax = axes[0]
    zz, ss = cohort(np.random.default_rng(7), NSHOW)
    mid = (zz > T_NEW) & (zz <= T_OLD)
    ax.axvspan(T_NEW, T_OLD, color=AMBER, alpha=0.16, zorder=1)
    ax.plot(zz[~mid], ss[~mid], lw=0, marker="o", ms=5.5, color=MUTED,
            zorder=2)
    ax.plot(zz[mid], ss[mid], lw=0, marker="o", ms=7, color=AMBER, zorder=3)
    for t, ls, col in ((T_OLD, "--", GREY), (T_NEW, "-", INK)):
        ax.axvline(t, color=col, lw=2.2, ls=ls, zorder=4)
    for (lo, hi), v, ls, col in (((0, T_OLD), e_old, "--", TEAL),
                                 ((0, T_NEW), e_new, "-", TEAL),
                                 ((T_OLD, 3.2), a_old, "--", RED),
                                 ((T_NEW, 3.2), a_new, "-", RED)):
        ax.plot([lo, hi], [v, v], lw=3.0, ls=ls, color=col, zorder=5)
    ax.annotate(f"이 사람들이 초기에서 진행으로 옮겨 갔다\n"
                f"전체의 ${moved*100:.0f}\\%$ 다",
                xy=(0.60, 36), xytext=(1.78, 45), fontsize=10.8, color=AMBER,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    ax.set_xlabel("숨은 중증도 — 오른쪽일수록 나쁘다", fontsize=11.5)
    ax.set_ylabel("생존기간 (개월)", fontsize=11.5)
    ax.set_title("옮겨 간 사람들", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 78)
    ax.legend(handles=[
        Line2D([], [], color=GREY, lw=2.2, ls="--", label="예전 장비의 문턱"),
        Line2D([], [], color=INK, lw=2.2, label="새 장비의 문턱"),
        Line2D([], [], color=TEAL, lw=3.0, label="초기 병기의 평균"),
        Line2D([], [], color=RED, lw=3.0, label="진행 병기의 평균")],
        fontsize=10.0, framealpha=0.93, loc="upper right", ncol=2)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 둘 다 올라가는데 전체는 그대로다 ──
    ax = axes[1]
    ax.plot(ts, curves[:, 0], lw=3.4, color=TEAL, label="초기 병기")
    ax.plot(ts, curves[:, 1], lw=3.4, color=RED, label="진행 병기")
    ax.axhline(s.mean(), lw=3.0, color=INK, ls="--",
               label=f"전체 — 언제나 ${s.mean():.1f}$ 개월")
    for t, mk in ((T_OLD, "D"), (T_NEW, "o")):
        a, b = means(z, s, t)
        ax.plot([t, t], [a, b], lw=0, marker=mk, ms=11, color=INK, zorder=5,
                mec="white", mew=1.6)
        ax.axvline(t, color=GREY, lw=1.6, ls=":", zorder=1)
    ax.annotate("장비가 좋아지는 쪽", xy=(0.30, 6.0), xytext=(0.86, 6.0),
                fontsize=11, color=GREY, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=2.0))
    ax.annotate(f"${e_new-e_old:+.1f}$ 개월",
                xy=(T_NEW, e_new), xytext=(0.79, 52), fontsize=11.4,
                color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate(f"${a_new-a_old:+.1f}$ 개월",
                xy=(T_NEW, a_new), xytext=(0.84, 23.5), fontsize=11.4,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("병기를 가르는 문턱 — 오른쪽일수록 잘 찾아낸다", fontsize=11.5)
    ax.set_ylabel("평균 생존기간 (개월)", fontsize=11.5)
    ax.set_title("둘 다 올라가는데 전체는 그대로다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(1.42, 0.06)
    ax.set_ylim(0, 62)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("아무도 낫지 않았는데 모두 좋아졌다", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"환자 ${N//1000}{{,}}000$ 명을 모의했다. 치료는 처음부터 끝까지 "
             "한 번도 바뀌지 않았고, 바뀐 것은 누구를 어느 병기로 부르는가 "
             "뿐이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "will_rogers.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
