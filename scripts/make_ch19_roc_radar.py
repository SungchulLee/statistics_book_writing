r"""19.4절 「그림으로 보는 통계학」 — 레이더 앞에 앉은 사람.

ROC 는 Receiver Operating Characteristic 의 줄임말이다. 「수신기의
작동 특성」이라는 뜻인데, 분류기 이야기에 왜 수신기가 나오는가.

제2차 세계대전 때 영국과 미국의 레이더 기지에서 화면의 흐릿한 점이
적기인지 잡음인지 판정하는 일을 사람이 했다. 같은 장비 앞에서도
판정은 사람마다 달랐다. 어떤 사람은 의심스러우면 무조건 경보를
울렸고(놓치는 적기는 적지만 헛경보가 잦다), 어떤 사람은 확실할 때만
울렸다.

여기서 핵심 통찰이 나왔다. **한 사람의 성적은 수 하나로 적을 수 없다.**
적중률과 헛경보율은 함께 움직이고, 그 맞교환 전체가 그 사람의 성적이다.
그리고 그 곡선을 정하는 것은 수신기의 성능이지 사람의 조심성이 아니다.

  능력(d')  — 신호와 잡음이 얼마나 떨어져 있는가. 곡선을 정한다.
  문턱(c)   — 어디서 경보를 울릴 것인가. 곡선 위의 점을 정한다.

전쟁 뒤 이것이 신호탐지이론으로 정리되었고(1950 년대), 심리학으로 갔다가
1970 년대에 의학 진단으로, 다시 기계학습으로 왔다.

왜 이 구별이 중요한가. **정확도 하나로는 능력과 문턱을 가를 수 없다.**

  대원 A:  d' = 1.0,  문턱 c =  0.50   정확도 0.691   AUC 0.760
  대원 B:  d' = 2.0,  문턱 c = -0.27   정확도 0.691   AUC 0.921

두 사람의 정확도가 **같다.** 그런데 A 는 좋은 수신기를 가진 사람이
아니라 그냥 조심성이 적당한 사람이고, B 는 훨씬 좋은 수신기를 가지고도
헛경보를 많이 내는 자리에 문턱을 두었을 뿐이다. 문턱을 옮기면 B 는
A 가 결코 닿을 수 없는 곳으로 간다.

만드는 파일:

  ch19/evaluation/img/roc_radar.png

실행:  python3 scripts/make_ch19_roc_radar.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats, optimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/evaluation/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

DA, CA = 1.0, 0.50          # 대원 A
DB = 2.0                    # 대원 B — 문턱은 정확도를 맞추어 정한다


def acc(d, c):
    return 0.5 * (stats.norm.cdf(d - c) + stats.norm.cdf(c))


def roc(d, cs):
    return stats.norm.sf(cs), stats.norm.sf(cs - d)   # (FPR, TPR)


def main():
    target = acc(DA, CA)
    cb = optimize.brentq(lambda c: acc(DB, c) - target, -3, 0.5)
    auc_a = stats.norm.cdf(DA / np.sqrt(2))
    auc_b = stats.norm.cdf(DB / np.sqrt(2))
    fa_a, tp_a = stats.norm.sf(CA), stats.norm.sf(CA - DA)
    fa_b, tp_b = stats.norm.sf(cb), stats.norm.sf(cb - DB)
    print(f"    대원 A  d'={DA:.1f}  c={CA:+.2f}   적중 {tp_a:.3f}  "
          f"헛경보 {fa_a:.3f}  정확도 {acc(DA, CA):.3f}  AUC {auc_a:.3f}")
    print(f"    대원 B  d'={DB:.1f}  c={cb:+.2f}   적중 {tp_b:.3f}  "
          f"헛경보 {fa_b:.3f}  정확도 {acc(DB, cb):.3f}  AUC {auc_b:.3f}")
    cstar = DB / 2
    print(f"    대원 B 가 문턱을 {cstar:+.2f} 로 옮기면  정확도 "
          f"{acc(DB, cstar):.3f}  (적중 {stats.norm.sf(cstar-DB):.3f}, "
          f"헛경보 {stats.norm.sf(cstar):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), constrained_layout=True)

    # ── 왼쪽: 화면 위의 두 분포 ──
    ax = axes[0]
    xs = np.linspace(-3.6, 5.6, 700)
    for d, c, col, nm, dy in ((DA, CA, AMBER, "대원 A", 0.0),
                              (DB, cb, TEAL, "대원 B", -0.62)):
        ax.plot(xs, stats.norm.pdf(xs) + dy, lw=2.2, color=GREY)
        ax.plot(xs, stats.norm.pdf(xs, d) + dy, lw=2.6, color=col)
        ax.fill_between(xs, dy, stats.norm.pdf(xs, d) + dy,
                        where=xs >= c, color=col, alpha=0.30)
        ax.fill_between(xs, dy, stats.norm.pdf(xs) + dy,
                        where=xs >= c, color=RED, alpha=0.26)
        ax.plot([c, c], [dy, dy + 0.45], color=INK, lw=2.0, ls="--")
        ax.text(c, dy + 0.47, f"문턱 {c:+.2f}", fontsize=10.4, color=INK,
                ha="center")
        ax.text(-3.4, dy + 0.34, f"{nm} — 능력 $d' = {d:.1f}$", fontsize=11.2,
                color=col)
    ax.text(-0.1, 0.055, "잡음", fontsize=10.6, color=GREY, ha="right")
    ax.text(DA + 0.1, 0.055, "적기", fontsize=10.6, color=AMBER, ha="left")
    ax.set_yticks([])
    ax.set_xlabel("화면에 나타난 신호의 세기", fontsize=11.5)
    ax.set_title("능력은 두 산이 떨어진 거리다", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(-3.6, 5.6)
    ax.set_ylim(-0.66, 0.70)
    ax.spines[["top", "right", "left"]].set_visible(False)

    # ── 오른쪽: ROC 평면 ──
    ax = axes[1]
    cs = np.linspace(-5, 6, 600)
    for d, col, lab, a in ((DA, AMBER, "대원 A", auc_a), (DB, TEAL, "대원 B", auc_b)):
        f, t_ = roc(d, cs)
        ax.plot(f, t_, lw=3.0, color=col, label=f"{lab} — AUC {a:.3f}")
    ax.plot([0, 1], [0, 1], lw=1.6, color=GREY, ls=":")
    b = 2 * target - 1
    ax.plot([0, 1], [b, b + 1], lw=2.0, color=RED, ls="--")
    ax.text(0.035, b + 0.035 - 0.085, f"정확도 {target:.3f} 인 자리",
            fontsize=10.6, color=RED, rotation=38, rotation_mode="anchor")
    ax.plot([fa_a], [tp_a], marker="o", ms=13, color=AMBER, zorder=5,
            mec="white", mew=1.6)
    ax.plot([fa_b], [tp_b], marker="o", ms=13, color=TEAL, zorder=5,
            mec="white", mew=1.6)
    ax.plot([stats.norm.sf(cstar)], [stats.norm.sf(cstar - DB)], marker="*",
            ms=20, color=TEAL, zorder=6, mec="white", mew=1.2)
    ax.annotate("두 사람이 같은 정확도에 서 있다",
                xy=((fa_a + fa_b) / 2, (tp_a + tp_b) / 2 - 0.03),
                xytext=(0.60, 0.26), fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.annotate(f"B 가 문턱만 옮기면\n정확도 {acc(DB, cstar):.3f} —\n"
                "A 는 어떤 문턱에서도 닿지 못한다",
                xy=(stats.norm.sf(cstar), stats.norm.sf(cstar - DB)),
                xytext=(0.56, 0.56), fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("헛경보율 (위양성률)", fontsize=11.5)
    ax.set_ylabel("적중률 (참양성률)", fontsize=11.5)
    ax.set_title("문턱은 곡선 위를, 능력은 곡선을 바꾼다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("ROC — 수신기의 작동 특성", fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽 회색 산이 잡음, 색 있는 산이 적기다. 두 산이 겹친 정도가 "
             "능력이고, 세로 파선이 그 사람이 정한 문턱이다. "
             "붉게 칠한 부분이 헛경보, 색으로 칠한 부분이 적중이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "roc_radar.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
