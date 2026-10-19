r"""21.3절 「그림으로 보는 통계학」 — 두 사람이 같은 것을 따로 보냈다.

이 쪽은 곱-극한 추정량을 유도한다. 그 식이 어떻게 세상에 나왔는지와,
그것이 없으면 무엇이 틀어지는지를 함께 본다.

1952 년에 에드워드 카플란(벨 연구소)과 폴 마이어(존스홉킨스)가 사실상
같은 추정량을 담은 원고를 『미국통계학회지』에 **따로** 보냈다. 편집자
존 투키가 겹침을 알아보고 둘에게 합치라고 권했다. 편지가 오가는 데
네 해가 걸렸고, 공저 논문은 1958 년에야 실렸다.

그 논문이 지금 과학 전체에서 가장 많이 인용된 축에 든다.

없으면 무엇이 틀어지는가. 중도절단된 관측을 다루는 소박한 두 방법과
견주어 보자.

  (가) 절단된 사람을 빼고 사건이 관측된 사람만 쓴다
  (나) 절단 시점을 사건 시점으로 친다

참 중앙생존시간이 0.693 인 지수 모형에서 재면 이렇다.

  절단 비율    카플란-마이어   (가) 버림   (나) 사건으로
      0%        0.691        0.696       0.696
     20%        0.697        0.557       0.557
     50%        0.699        0.349       0.347
     70%        0.716        0.210       0.208

절단이 절반이면 소박한 방법은 중앙생존시간을 **절반으로** 줄여 보고한다.
까닭은 같다 — 절단된 사람은 「그 시점까지 살아 있었다」는 정보를 가지고
있는데, 두 방법 모두 그 정보를 버리거나 뒤집어 쓴다.

만드는 파일:

  ch21/nonparametric/img/kaplan_meier.png

실행:  python3 scripts/make_ch21_kaplan_meier.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "ch21/nonparametric/img/"
OUT = "docs/" + OUT

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

LAM, N, REPS, SEED = 1.0, 200, 4000, 1958
MED = np.log(2) / LAM
CS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
SHOW_C = 0.5


def censor(rng, n, c):
    T = rng.exponential(1 / LAM, n)
    C = np.full(n, np.inf) if c == 0 else rng.exponential((1 - c) / (LAM * c), n)
    return np.minimum(T, C), (T <= C).astype(int)


def km_curve(t, d):
    """곱-극한 추정량의 계단 (시각, 생존확률)."""
    o = np.argsort(t)
    t, d = t[o], d[o]
    n = len(t)
    xs, ys, S, at = [0.0], [1.0], 1.0, n
    for i in range(n):
        if d[i]:
            S *= (at - 1) / at
            xs.append(t[i]); ys.append(S)
        at -= 1
    return np.array(xs), np.array(ys)


def ecdf_curve(v):
    v = np.sort(v)
    n = len(v)
    return np.concatenate([[0.0], v]), np.concatenate([[1.0], 1 - np.arange(1, n + 1) / n])


def median_from(xs, ys):
    k = np.where(ys <= 0.5)[0]
    return xs[k[0]] if len(k) else np.nan


def main():
    rng = np.random.default_rng(SEED)
    print(f"    참 중앙생존시간 = {MED:.4f}")
    print(f"{'절단 비율':>10}{'카플란-마이어':>14}{'절단을 버림':>13}{'절단을 사건으로':>16}")
    km_m, drop_m, ev_m = [], [], []
    for c in CS:
        a, b, e = [], [], []
        for _ in range(REPS):
            t, d = censor(rng, N, c)
            a.append(median_from(*km_curve(t, d)))
            u = t[d == 1]
            b.append(np.median(u) if len(u) else np.nan)
            e.append(np.median(t))
        km_m.append(np.nanmean(a)); drop_m.append(np.nanmean(b))
        ev_m.append(np.nanmean(e))
        print(f"{c:>10.0%}{km_m[-1]:>14.4f}{drop_m[-1]:>13.4f}{ev_m[-1]:>16.4f}")

    t, d = censor(np.random.default_rng(7), N, SHOW_C)
    kx, ky = km_curve(t, d)
    dx, dy = ecdf_curve(t[d == 1])
    ex, ey = ecdf_curve(t)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 한 자료에서의 곡선 넷 ──
    ax = axes[0]
    xs = np.linspace(0, 3.2, 400)
    ax.plot(xs, np.exp(-LAM * xs), lw=3.0, color=INK, ls="--",
            label="참 생존함수")
    ax.step(kx, ky, where="post", lw=3.0, color=TEAL,
            label="카플란–마이어")
    ax.step(dx, dy, where="post", lw=2.4, color=RED,
            label="절단된 사람을 빼고 계산")
    ax.step(ex, ey, where="post", lw=2.4, color=AMBER, ls=":",
            label="절단 시점을 사건으로 침")
    ax.axhline(0.5, color=GREY, lw=1.4, ls=":")
    ax.text(3.1, 0.52, "생존확률 $0.5$", fontsize=10.6, color=GREY, ha="right")
    ax.plot([MED], [0.5], marker="o", ms=10, color=INK, zorder=5)
    ax.annotate(f"참 중앙생존시간 {MED:.2f}", xy=(MED, 0.5), xytext=(1.55, 0.70),
                fontsize=10.8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("시간", fontsize=11.5)
    ax.set_ylabel("생존확률", fontsize=11.5)
    ax.set_title(f"같은 자료, 네 개의 곡선 (절단 {SHOW_C:.0%})",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 3.2)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 절단이 늘어날수록 ──
    ax = axes[1]
    cs = np.array(CS) * 100
    ax.axhline(MED, color=INK, lw=2.4, ls="--")
    ax.text(2, MED + 0.016, f"참값 {MED:.3f}", fontsize=11, color=INK)
    ax.plot(cs, km_m, lw=3.0, color=TEAL, marker="o", ms=7,
            label="카플란–마이어")
    ax.plot(cs, drop_m, lw=2.6, color=RED, marker="s", ms=6,
            label="절단된 사람을 빼고 계산")
    ax.plot(cs, ev_m, lw=2.6, color=AMBER, marker="^", ms=6, ls=":",
            label="절단 시점을 사건으로 침")
    ax.fill_between(cs, drop_m, MED, color=RED, alpha=0.12)
    i = CS.index(0.5)
    ax.annotate(f"절단이 절반이면\n중앙생존시간을 {drop_m[i]:.2f} 로 —\n"
                f"참값의 {drop_m[i]/MED*100:.0f}% 로 보고한다",
                xy=(50, drop_m[i]), xytext=(27, 0.21), fontsize=10.8,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("절단이 늘어도 흔들리지 않는다",
                xy=(60, km_m[CS.index(0.6)]), xytext=(42, 0.88),
                fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("중도절단된 비율 (%)", fontsize=11.5)
    ax.set_ylabel("추정한 중앙생존시간", fontsize=11.5)
    ax.set_title("정보를 버리면 수명이 짧아 보인다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0.10, 1.0)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("중도절단된 사람은 「모른다」가 아니라 「그때까지 살았다」이다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             f"지수 생존시간, 표본 {N} 명, 각 절단 비율마다 {REPS//1000} 천 번. "
             "두 소박한 방법은 방향도 크기도 거의 같게 빗나간다 — "
             "둘 다 「그 시점까지 살아 있었다」는 정보를 쓰지 않기 때문이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "kaplan_meier.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
