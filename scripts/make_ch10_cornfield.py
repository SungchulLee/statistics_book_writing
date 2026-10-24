r"""10.3절 「그림으로 보는 통계학」 — 「그럴 수도 있다」에 수를 붙이기.

이 쪽의 첫 기준은 「연관의 강도」이고, 상대위험 10 이 1.2 보다 교란으로
설명하기 어렵다고 적는다. 맞는 말인데 수가 없다. 그 수가 1959 년에
나왔고, 나오게 만든 사람이 피셔다.

1950 년대에 돌과 브래드퍼드 힐이 영국 의사 수만 명을 좇아 흡연자의 폐암
사망률이 비흡연자의 여러 배임을 보였다. 피셔는 반대했다. 무작위화와
분산분석과 유의성검정을 만든 그 사람이, 관찰자료의 상관은 인과가 아니라고
했다 — 담배를 피우고 싶게 만들면서 동시에 암에 약하게 만드는 유전자가
있다면 자료가 똑같이 나온다는 것이다(「체질 가설」).

형식논리로는 반박할 수 없는 말이다. 콘필드와 동료들(1959)은 반박하는 대신
**조건을 계산했다.**

  관찰된 상대위험이 R 이고 그것을 교란변수 하나가 통째로 설명하려면,
  그 교란변수는 적어도 노출군에서 비노출군보다 R 배 흔해야 한다.

흡연의 R 이 대략 9 였다. 그러니 「흡연 유전자」는 흡연자에게서 비흡연자보다
아홉 배 넘게 흔해야 한다.

오른쪽 그림은 그것을 더 세게 조인다. 그 유전자를 **흡연자 전원이** 가지고
있고 그 효력이 **무한히** 세다고 하자. 그래도 만들 수 있는 가짜 상대위험의
상한은 1/p0 이다(p0 = 비흡연자 중 보유율). R = 9 를 만들려면 p0 <= 11.1%
여야 한다. 비흡연자의 12% 가 그 유전자를 가지고 있으면, 아무리 센 유전자를
가정해도 9 배가 나오지 않는다.

E값(밴더윌리·딩 2017)은 같은 물음의 현대판으로, 교란변수가 노출과 결과
**양쪽**에 얼마나 세게 붙어야 하는지를 준다.

  E = R + sqrt(R(R-1))     R = 9 이면 17.5,  R = 1.2 이면 1.69

만드는 파일:

  ch10/causation/img/cornfield.png

실행:  python3 scripts/make_ch12_cornfield.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
PURPLE = "#6A1B9A"

RR_SMOKE = 9.0          # 흡연자 대 비흡연자, 폐암 (대략)
P0S = (0.05, 1 / 9, 0.20, 1 / 3)


def evalue(r):
    return r + np.sqrt(r * (r - 1))


def main():
    print("    관찰된 상대위험 R 에 맞서려면 교란변수가 얼마나 세야 하는가")
    print(f"{'R':>6}{'콘필드 하한':>12}{'E값':>10}")
    for r in (1.05, 1.2, 1.5, 2, 4, 9, 20):
        print(f"{r:>6}{r:>12.2f}{evalue(r):>10.2f}")
    print(f"\n    흡연자 전원이 가지고 효력이 무한해도 만들 수 있는 가짜 R 의 상한")
    for p0 in P0S:
        print(f"      비흡연자 보유율 {p0*100:5.1f}%  ->  상한 {1/p0:5.2f}")
    print(f"    R = {RR_SMOKE:.0f} 을 만들려면 보유율이 {1/RR_SMOKE*100:.1f}% 이하여야 한다")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)

    # ── 왼쪽: 관찰된 R 과 그것을 지우는 데 드는 값 ──
    ax = axes[0]
    rs = np.linspace(1, 21, 400)
    ax.plot(rs, rs, lw=2.8, color=TEAL, label="콘필드 하한 — 노출군에서 몇 배 흔해야 하나")
    ax.plot(rs, evalue(rs), lw=3.0, color=RED,
            label="E값 — 노출과 결과 양쪽에 얼마나 세게 붙어야 하나")
    ax.fill_between(rs, rs, evalue(rs), color=RED, alpha=0.10)
    ax.plot([1, 21], [1, 1], lw=1.6, color=GREY, ls=":")
    for r, lab, col, dy in (
            (1.2, "약한 연관 $R = 1.2$ — E값 $1.69$\n이 정도는 쉽게 지워진다",
             AMBER, (3.4, 2.0)),
            (RR_SMOKE, f"흡연 $R = {RR_SMOKE:.0f}$ — E값 ${evalue(RR_SMOKE):.1f}$\n알려진 어떤 유전 요인도 이만큼 세지 않다", RED, (11.2, 8.0))):
        ax.plot([r], [evalue(r)], marker="o", ms=11, color=col, zorder=5)
        ax.annotate(lab, xy=(r, evalue(r)), xytext=dy,
                    fontsize=10.6, color=col, ha="left",
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.set_xlabel("관찰된 상대위험 $R$", fontsize=11.5)
    ax.set_ylabel("교란변수에게 요구되는 세기", fontsize=11.5)
    ax.set_title("연관이 셀수록 지우기가 어렵다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(1, 21)
    ax.set_ylim(0, 44)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 유전자를 가장 유리하게 가정해도 ──
    ax = axes[1]
    gam = np.logspace(0, 3, 400)        # 유전자가 암 위험을 몇 배 올리는가
    for p0, col in zip(P0S, (BLUE, TEAL, AMBER, PURPLE)):
        spur = gam / (p0 * gam + 1 - p0)
        ax.plot(gam, spur, lw=2.8, color=col,
                label=f"비흡연자의 {p0*100:.1f}% 가 보유 — 상한 {1/p0:.1f}")
        ax.axhline(1 / p0, color=col, lw=1.2, ls=":")
    ax.axhline(RR_SMOKE, color=RED, lw=2.4, ls="--")
    ax.text(1.25, RR_SMOKE * 1.12, f"관찰된 흡연의 $R = {RR_SMOKE:.0f}$",
            fontsize=11.2, color=RED)
    ax.text(950, 6.3, "$11.1\\%$ 면 꼭 $9$ 에서 멈춘다 — 넘지 못한다",
            ha="right", fontsize=10.6, color=TEAL)
    ax.text(950, 3.9, "$20\\%$ 면 $5$ 가 천장이다", ha="right",
            fontsize=10.6, color=AMBER)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1 배", "10 배", "100 배", "1,000 배"])
    ax.minorticks_off()
    ax.set_xlabel("유전자가 폐암 위험을 올리는 배수 (흡연자는 전원 보유로 가정)",
                  fontsize=11.5)
    ax.set_ylabel("그 유전자로 만들 수 있는 가짜 상대위험", fontsize=11.5)
    ax.set_title("가장 유리하게 가정해도 넘지 못하는 천장",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 22)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("「교란일 수도 있다」는 반론에는 수를 붙일 수 있다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "오른쪽은 흡연에 인과효과가 전혀 없다고 두고, 유전자 하나만으로 "
             "자료를 설명하려 한 것이다. 유전자를 흡연자 전원이 가졌다고 "
             "가정했으니 교란에 가장 유리한 경우다. 그래도 가짜 상대위험은 "
             "비흡연자 보유율의 역수를 넘지 못한다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "cornfield.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
