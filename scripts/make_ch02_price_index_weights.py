r"""2.1절 「그림으로 보는 통계학」 — 가중치를 누가 정하는가.

가중평균의 식에는 가중치가 어디서 오는지가 적혀 있지 않다. 그런데
가중치를 누가 어떻게 정하느냐로 답이 통째로 달라지는 수가 있다.
물가지수다.

같은 품목, 같은 가격 변화를 놓고 가중치만 바꾸면

   가중치                      물가상승률
   지출 비중 (쓴 돈의 비율)      1.42%
   균등 (품목마다 똑같이)        3.60%
   구입 빈도 (사는 횟수의 비율)  6.14%

네 배가 넘게 벌어진다. 사람들이 「체감물가가 발표보다 높다」고 말하는
사정의 한 조각이 여기 있다 — 자주 사는 것은 많이 올랐고, 덜 오른
품목은 가끔 사거나 아예 안 산다.

가중치를 언제 재느냐도 문제다. 값이 오른 품목은 덜 사게 되므로,

  라스파이레스  옛 바구니로 잰다  ->  바꾸기를 모르니 높게 나온다
  파셰          새 바구니로 잰다  ->  바꾼 뒤만 보니 낮게 나온다
  피셔          둘의 기하평균      ->  가운데

한쪽 품목이 30% 오를 때 대체 탄력성을 0 에서 2 까지 올리며 재면

   탄력성   라스파이레스    파셰    피셔    참값
     0.0        1.150      1.150   1.150   1.150
     0.5        1.150      1.140   1.145   1.145
     1.0        1.150      1.130   1.140   1.140
     2.0        1.150      1.112   1.131   1.131

라스파이레스는 꼼짝하지 않고 나머지는 내려간다. 나라의 물가지수는
대개 라스파이레스 꼴이라 이 편향을 안고 간다.

만드는 파일:

  ch02/center/img/price_index_weights.png

실행:  python3 scripts/make_ch02_price_index_weights.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch02/center/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

ITEMS = ("식료품", "교통", "외식", "통신", "주거·가전")
CHANGE = np.array([8.0, 6.0, 5.0, 1.0, -2.0])          # 품목별 가격 변화 (%)
WEIGHTS = (("지출 비중", np.array([.14, .12, .13, .05, .56]), TEAL),
           ("균등", np.full(5, 0.2), GREY),
           ("구입 빈도", np.array([.45, .25, .20, .08, .02]), RED))
SHOCK = 0.30                                            # 한 품목의 가격 상승


def indices(sigma, shock=SHOCK):
    """두 품목, 처음 지출이 반반일 때의 네 가지 지수."""
    p0 = np.array([1.0, 1.0])
    p1 = np.array([1.0 + shock, 1.0])
    q0 = np.array([1.0, 1.0])
    if abs(sigma - 1.0) < 1e-9:                         # 콥-더글러스
        true = np.sqrt(p1[0] * p1[1])
    else:
        true = (0.5 * p1[0] ** (1 - sigma)
                + 0.5 * p1[1] ** (1 - sigma)) ** (1 / (1 - sigma))
    ratio = (p1[0] / p1[1]) ** (-sigma)                 # 새 수량의 비
    q1 = np.array([ratio, 1.0])
    q1 = q1 * (true * (p0 @ q0) / (p1 @ q1))            # 같은 효용에 맞춘다
    lasp = (p1 @ q0) / (p0 @ q0)
    paas = (p1 @ q1) / (p0 @ q1)
    return lasp, paas, np.sqrt(lasp * paas), true


def main():
    print(f"  품목별 가격 변화 (%): "
          f"{', '.join(f'{n} {c:+.0f}' for n, c in zip(ITEMS, CHANGE))}")
    vals = []
    for nm, w, _ in WEIGHTS:
        v = float(w @ CHANGE)
        vals.append(v)
        print(f"    {nm:>10} 로 가중하면 {v:+.2f}%")
    print(f"    가장 높은 쪽이 가장 낮은 쪽의 "
          f"{max(vals)/min(vals):.1f} 배")

    print(f"\n  한 품목이 {SHOCK:.0%} 오를 때")
    print(f"{'탄력성':>8}{'라스파이레스':>14}{'파셰':>10}{'피셔':>10}{'참값':>10}")
    for s in (0.0, 0.5, 1.0, 2.0):
        l, p, f, t = indices(s)
        print(f"{s:>8.1f}{l:>14.3f}{p:>10.3f}{f:>10.3f}{t:>10.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 같은 가격, 다른 가중치 ──
    ax = axes[0]
    xs = np.arange(len(ITEMS))
    ax.bar(xs, CHANGE, width=0.52, color=MUTED, alpha=0.85, zorder=3)
    for x, c in zip(xs, CHANGE):
        ax.text(x, c + (0.28 if c > 0 else -0.65), f"${c:+.0f}\\%$",
                ha="center", fontsize=10.8, color=INK)
    for (nm, w, col), v in zip(WEIGHTS, vals):
        ax.axhline(v, color=col, lw=2.8, ls="--", zorder=4)
        ax.text(4.55, v + 0.16, f"{nm}  ${v:+.2f}\\%$", fontsize=10.6,
                color=col, va="bottom", ha="left")
    ax.axhline(0, color=INK, lw=1.4)
    ax.set_xticks(xs)
    ax.set_xticklabels(ITEMS, fontsize=11)
    ax.set_ylabel("가격 변화", fontsize=11.5)
    ax.set_title("같은 가격 자료, 세 가지 「물가상승률」", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(-0.6, 7.4)
    ax.set_ylim(-3.4, 9.6)
    ax.set_yticks([-2, 0, 2, 4, 6, 8])
    ax.set_yticklabels(["$-2\\%$", "$0$", "$2\\%$", "$4\\%$", "$6\\%$",
                        "$8\\%$"])
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 가중치를 언제 재는가 ──
    ax = axes[1]
    ss = np.linspace(0, 2.2, 300)
    g = np.array([indices(s) for s in ss])
    for j, (col, lab, ls) in enumerate((
            (RED, "라스파이레스 — 옛 바구니", "-"),
            (TEAL, "파셰 — 새 바구니", "-"),
            (AMBER, "피셔 — 둘의 기하평균", "--"),
            (INK, "참값 (생계비지수)", (0, (1, 2.2))))):
        ax.plot(ss, g[:, j], lw=3.4 if j < 2 else 3.0, color=col, ls=ls,
                label=lab)
    ax.annotate("값이 오르면 덜 사는데\n옛 바구니는 그것을 모른다",
                xy=(1.75, indices(1.75)[0]), xytext=(1.30, 1.1432),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate("피셔가 참값과 거의 겹친다",
                xy=(0.62, indices(0.62)[2]), xytext=(0.30, 1.1225),
                fontsize=10.6, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xlabel("대체 탄력성 — 오른쪽일수록 잘 바꾼다", fontsize=11.5)
    ax.set_ylabel("물가지수", fontsize=11.5)
    ax.set_title(f"한 품목이 ${SHOCK*100:.0f}\\%$ 올랐을 때", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(0, 2.2)
    ax.set_ylim(1.104, 1.156)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("가중치를 누가 정하는가", fontsize=14.5)
    fig.text(0.5, -0.06,
             "왼쪽 품목과 가격 변화는 설명을 위해 지어낸 보기다. 오른쪽은 "
             "두 품목의 지출이 처음에 반반일 때 식으로 구한 값이며, "
             "「참값」은 같은 효용을 누리는 데 드는 돈의 비다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "price_index_weights.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
