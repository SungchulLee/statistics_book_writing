r"""2.6절 「그림으로 보는 통계학」 — 기계가 틀렸다고 생각했다.

1985 년 5 월, 파먼과 가디너와 섕클린이 『네이처』에 남극 핼리만의
오존이 1960 년대보다 크게 줄었다고 보고했다. 지상에 세워 둔 도브슨
분광계로 1957 년부터 재어 온 기록이었다.

그 무렵 미국 항공우주국의 님버스-7 위성은 1978 년부터 같은 하늘의
오존을 재고 있었다. 그 자료에도 남극의 낮은 값이 들어 있었다.

항공우주국이 그 값을 버린 것은 아니다. 확인했다. 다른 자료와 맞대어
보았고, 그들이 가진 다른 자료는 오존이 300 도브슨 단위쯤이라고 했다.
그래서 **위성 쪽이 고장 났다고 판단했다.** 핼리만의 지상 기록은
그들의 손에 없었다. 파먼의 논문이 나온 뒤에야 위성 값이 맞았다는 것을
알았고, 그해 8 월 프라하에서 바르티아 등이 위성이 그린 첫 오존 구멍
지도를 내놓았다.

파먼 쪽도 똑같이 의심했다. 값이 너무 낮아 자기 장비를 먼저 의심했고,
새 분광계를 들여왔다. 새 장비가 같은 답을 냈다.

두 팀 다 「기계가 틀렸다」고 생각했다. 한쪽에는 「네가 틀렸다」고
말해 주는 대조 자료가 있었고, 다른 쪽에는 「네가 맞다」고 말해 주는
것이 있었다.

이 쪽 정의 1 의 1.5 IQR 울타리를 1957-1975 년 자료로 세우고, 걸린
값을 자동으로 버리면 어떻게 되는지 잰다.

  아래쪽 울타리 258.7 DU
  1981 년부터는 모든 관측이 걸린다
  1994 년까지 와도 걸러낸 쪽의 추세는 10 년에 -0.4 DU 로 평평하다
  같은 자료 전부를 쓰면 10 년에 -51.6 DU

만드는 파일:

  ch02/shape/img/ozone_outlier.png

실행:  python3 scripts/make_ch02_ozone_outlier.py   (저장소 최상위에서)
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

OUT = "docs/ch02/shape/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

Y0, Y1, BASE_END = 1957, 1994, 1975
SEED = 8                    # 기준 구간에 군더더기 표시가 없는 판


def series(rng):
    """핼리만 10 월 평균 오존을 본떠 만든 자료. 실제 기록이 아니다."""
    yr = np.arange(Y0, Y1 + 1)
    mu = np.where(yr <= 1976, 305.0,
                  np.where(yr <= 1984, 305.0 - 12.0 * (yr - 1976),
                           209.0 - 9.5 * (yr - 1984)))
    return yr, mu + rng.normal(0, 13.0, len(yr))


def slope(x, y):
    """10 년당 기울기. 점이 셋 미만이면 재지 않는다."""
    if len(x) < 3:
        return np.nan
    return float(np.polyfit(x, y, 1)[0] * 10)


def main():
    rng = np.random.default_rng(SEED)
    yr, du = series(rng)

    base = du[yr <= BASE_END]
    q1, q3 = np.percentile(base, [25, 75])
    fence = q1 - 1.5 * (q3 - q1)
    flag = du < fence
    first = int(yr[next(i for i in range(len(yr)) if flag[i:].all())])

    print(f"  {Y0}-{BASE_END} 로 세운 울타리: Q1 {q1:.1f}  Q3 {q3:.1f}  "
          f"아래쪽 울타리 {fence:.1f} DU")
    print(f"  처음 걸리는 해 {first},  {Y1} 까지 걸린 관측 {flag.sum()} 개")

    # 해마다 그해까지의 자료로 추세를 다시 잰다
    ys = np.arange(1970, Y1 + 1)
    s_all = np.array([slope(yr[yr <= y], du[yr <= y]) for y in ys])
    s_cut = np.array([slope(yr[(yr <= y) & ~flag], du[(yr <= y) & ~flag])
                      for y in ys])
    print(f"\n  {Y1} 년까지의 자료로 잰 추세 (10 년당)")
    print(f"    전부 쓰면        {s_all[-1]:+.1f} DU")
    print(f"    걸린 것을 버리면  {s_cut[-1]:+.1f} DU")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 울타리에 걸린 것이 곧 발견이었다 ──
    ax = axes[0]
    ax.axvspan(Y0 - 0.6, BASE_END + 0.5, color=MUTED, alpha=0.14)
    ax.axhline(fence, color=RED, lw=2.2, ls="--")
    ax.plot(yr, du, lw=1.8, color=MUTED, zorder=2)
    ax.plot(yr[~flag], du[~flag], lw=0, marker="o", ms=8, color=INK, zorder=3)
    ax.plot(yr[flag], du[flag], lw=0, marker="X", ms=11, color=RED, zorder=4)
    ax.text(Y0 + 0.4, 334, "울타리를 세운 구간", fontsize=10.6, color=GREY)
    ax.annotate(f"아래쪽 울타리 ${fence:.0f}$ DU\n"
                f"$1.5 \\times$ IQR 로 세웠다",
                xy=(1968.5, fence), xytext=(1966.5, 192), fontsize=10.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(f"${first}$ 년부터는 모든 관측이\n「이상치」로 걸린다",
                xy=(1986, du[yr == 1986][0]), xytext=(1979.5, 118),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("연도", fontsize=11.5)
    ax.set_ylabel("$10$ 월 평균 오존 전량 (DU)", fontsize=11.5)
    ax.set_title("울타리에 걸린 것이 곧 발견이었다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(Y0 - 0.6, Y1 + 0.6)
    ax.set_ylim(100, 352)
    ax.legend(handles=[Line2D([], [], lw=0, marker="o", ms=8, color=INK,
                              label="울타리 안 — 그대로 쓴다"),
                       Line2D([], [], lw=0, marker="X", ms=11, color=RED,
                              label="울타리 밖 — 버린다")],
              fontsize=10.4, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 버리고 나면 아무 일도 없다 ──
    ax = axes[1]
    ax.axhline(0, color=INK, lw=1.6)
    ax.plot(ys, s_all, lw=3.4, color=INK, marker="o", ms=5,
            label="자료를 전부 쓴다")
    ax.plot(ys, s_cut, lw=3.2, color=RED, marker="D", ms=5, ls="--",
            label="울타리에 걸린 것을 버린다")
    ax.axvline(first, color=GREY, lw=1.8, ls=":")
    ax.annotate(f"${Y1}$ 년에 와서도\n${s_cut[-1]:+.1f}$ DU — 평평하다",
                xy=(Y1 - 0.4, s_cut[-1]), xytext=(1986.8, -13.5),
                fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"${s_all[-1]:+.1f}$ DU",
                xy=(Y1 - 0.4, s_all[-1]), xytext=(1986.6, -55.0),
                fontsize=11.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.5))
    ax.text(first + 0.5, 19, f"${first}$ 년\n첫 「이상치」", fontsize=10.4,
            color=GREY)
    ax.set_xlabel("이 해까지의 자료로 추세를 잰다", fontsize=11.5)
    ax.set_ylabel("추정한 추세 ($10$ 년당 DU)", fontsize=11.5)
    ax.set_title("버리고 나면 아무 일도 없다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(1969.5, Y1 + 0.6)
    ax.set_ylim(-58, 28)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("기계가 틀렸다고 생각했다", fontsize=14.5)
    fig.text(0.5, -0.06,
             "핼리만 $10$ 월 평균 오존 전량의 모양을 본떠 만든 자료이며 "
             "실제 기록이 아니다. 울타리는 정의 $1$ 의 $1.5 \\times$ IQR 을 "
             f"${Y0}$–${BASE_END}$ 년 자료에 적용해 세웠다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "ozone_outlier.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
