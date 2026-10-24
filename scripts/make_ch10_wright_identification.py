r"""10.4절 「그림으로 보는 통계학」 — 버터값에서 태어난 방법.

1928 년, 필립 라이트가 『동식물성 유지에 대한 관세』라는 책을 냈다.
버터와 식용유에 매기는 관세를 올리면 값이 얼마나 오르는지가 물음이었고,
그러려면 수요곡선의 기울기를 알아야 했다.

그런데 자료에는 값과 거래량의 짝뿐이다. 그 짝 하나하나는 **그날의
수요곡선과 그날의 공급곡선이 만난 자리**다. 두 곡선이 모두 움직이므로,
점구름에 직선을 맞추면 수요의 기울기도 공급의 기울기도 아닌 엉뚱한
수가 나온다.

  참 수요 기울기  -1.00
  참 공급 기울기  +0.80
  점구름에 최소제곱  -0.48  (이론값 -0.4767)

라이트가 부록 B 에 적은 해법이 이것이다. **공급만 움직이고 수요는
건드리지 않는 것**을 하나 찾아 쓰면 수요곡선을 되살릴 수 있다.
날씨 같은 것이다. 오늘날 도구변수라 부르는 방법의 첫 등장이다.

  IV 추정  -1.00  (10,000 번 평균 -1.0026)

누가 썼는지에 대해 오래 다툼이 있었다. 필립의 아들이 경로분석을 만든
유전학자 시월 라이트이고, 부록 B 가 그 경로분석에 기대고 있기 때문이다.
스톡과 트레비가 2003 년에 문체를 통계적으로 분석해 아버지 쪽이라고
결론지었다 — 아들이 조금 거들었을 수는 있다는 단서와 함께.

만드는 파일:

  ch10/causation/img/wright_identification.png

실행:  python3 scripts/make_ch12_wright_identification.py   (저장소 최상위에서)
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

OUT = "docs/ch10/causation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

BD, BS = 1.0, 0.8            # 수요·공급의 기울기 크기
AD, AS = 10.0, 2.0           # 절편
SD_D, SD_S, CW = 1.0, 1.0, 1.2
N, REPS, SEED = 400, 10_000, 1928


def market(rng, n):
    """수요와 공급이 매일 조금씩 움직이고, 공급만 날씨 W 에 반응한다."""
    ud = rng.normal(0, SD_D, n)
    us = rng.normal(0, SD_S, n)
    w = rng.normal(0, 1, n)
    p = (AD - AS + ud - us - CW * w) / (BD + BS)
    q = AD - BD * p + ud
    return p, q, w, ud, us


def slopes(p, q, w):
    ols = np.cov(p, q, ddof=1)[0, 1] / np.var(p, ddof=1)
    iv = np.cov(w, q, ddof=1)[0, 1] / np.cov(w, p, ddof=1)[0, 1]
    return ols, iv


def main():
    rng = np.random.default_rng(SEED)
    o = np.empty(REPS)
    v = np.empty(REPS)
    for i in range(REPS):
        p, q, w, _, _ = market(rng, N)
        o[i], v[i] = slopes(p, q, w)
    print(f"  참 수요 기울기 {-BD:+.2f},  참 공급 기울기 {+BS:+.2f}")
    print(f"  최소제곱 {o.mean():+.4f}  (표준편차 {o.std():.4f})")
    print(f"  도구변수 {v.mean():+.4f}  (표준편차 {v.std():.4f})")
    vs = SD_S ** 2 + CW ** 2                      # 공급 쪽 흔들림 전체
    theo = (BS * SD_D ** 2 - BD * vs) / (SD_D ** 2 + vs)
    print(f"  최소제곱의 이론값 {theo:+.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 점 하나하나가 서로 다른 두 곡선의 교점이다 ──
    ax = axes[0]
    rs = np.random.default_rng(11)
    p, q, w, ud, us = market(rs, N)
    ax.plot(p, q, lw=0, marker="o", ms=4.0, color=MUTED, alpha=0.75, zorder=2)
    gp = np.linspace(2.6, 6.4, 10)
    for k in range(4):
        i = int(k * N / 4) + 7
        ax.plot(gp, AD - BD * gp + ud[i], lw=1.6, color=TEAL, alpha=0.55,
                zorder=3)
        ax.plot(gp, AS + BS * gp + CW * w[i] + us[i], lw=1.6, color=AMBER,
                alpha=0.55, zorder=3)
        ax.plot([p[i]], [q[i]], lw=0, marker="o", ms=9, color=INK, zorder=5,
                mec="white", mew=1.4)
    b = np.cov(p, q, ddof=1)[0, 1] / np.var(p, ddof=1)
    a = q.mean() - b * p.mean()
    ax.plot(gp, a + b * gp, lw=3.6, color=RED, zorder=4)
    ax.annotate(f"점구름에 그은 직선 — 기울기 ${b:+.2f}$\n"
                f"수요도 공급도 아니다",
                xy=(5.6, a + b * 5.6), xytext=(5.1, 9.6), fontsize=10.6,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xlabel("값", fontsize=11.5)
    ax.set_ylabel("거래량", fontsize=11.5)
    ax.set_title("점 하나가 그날의 교점이다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(2.6, 6.4)
    ax.set_ylim(3.2, 10.4)
    ax.legend(handles=[
        Line2D([], [], color=TEAL, lw=2.2, label=f"수요 (기울기 ${-BD:+.1f}$)"),
        Line2D([], [], color=AMBER, lw=2.2, label=f"공급 (기울기 ${BS:+.1f}$)"),
        Line2D([], [], lw=0, marker="o", ms=7, color=INK, label="그날의 교점")],
        fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 공급만 미는 것을 하나 쥐면 ──
    ax = axes[1]
    edges = np.linspace(-1.35, 0.42, 120)
    ax.hist(o, bins=edges, density=True, histtype="step", lw=3.2, color=RED,
            label=f"최소제곱 — 평균 ${o.mean():+.3f}$")
    ax.hist(v, bins=edges, density=True, histtype="step", lw=3.2, color=TEAL,
            label=f"도구변수 — 평균 ${v.mean():+.3f}$")
    ax.axvline(-BD, color=INK, lw=2.4, ls="--")
    ax.text(-1.33, 10.5, f"참 수요 기울기 ${-BD:.2f}$ (점선)", fontsize=10.8,
            color=INK, ha="left")
    ax.annotate("날씨처럼 공급만 미는 것을\n하나 쥐면 되살아난다",
                xy=(-1.03, 5.3), xytext=(-1.32, 7.9), fontsize=10.6,
                color=TEAL, ha="left",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xlabel("추정한 기울기", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title(f"같은 자료, 다른 셈법 (${REPS//1000}{{,}}000$ 번)",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-1.35, 0.42)
    ax.set_ylim(0, 11.6)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("버터값에서 태어난 방법", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"장마다 ${N}$ 일을 관측했다. 수요와 공급이 매일 조금씩 "
             "움직이고, 공급만 날씨에 반응한다. 자료에 들어 있는 것은 값과 "
             "거래량뿐이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "wright_identification.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
