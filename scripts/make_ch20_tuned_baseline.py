r"""20.1절 「그림으로 보는 통계학」 — 기준선을 얼마나 성의 있게 맞췄는가.

1990 년대 후반부터 십여 년 동안, 다범주 분류 논문들은 정교한 새 방법이
단순한 OvR 을 이긴다고 거듭 보고했다. 리프킨과 클라우타우는 2004 년에
「일대다 분류를 옹호함」이라는 논문으로 그 보고들을 다시 재어 보고,
이긴 폭의 상당 부분이 **OvR 쪽 기준선을 제대로 맞추지 않은 데서** 나온
것이라고 결론지었다.

여기서 그 구조를 직접 재 본다. 인공자료 15 벌(n=400, 특징 40, 범주 5)에
OvR 과 소프트맥스를 각각 벌점 세기 C 를 바꿔 가며 5-겹 교차검증한다.
벌점을 고르는 절차만 바꾸고 나머지는 전부 같게 둔다.

  둘 다 기본값 C=1 로        소프트맥스 -0.0105   (OvR 이 오히려 낫다)
  소프트맥스만 맞춤           소프트맥스 +0.0267   (2.7 포인트 이긴다)
  둘 다 맞춤                  소프트맥스 -0.0022   (차이가 없다)

가운데 줄이 발표할 만한 수다. 그런데 같은 OvR 을 C 로 움직이기만 해도
정확도가 11.4 포인트 폭으로 오르내린다 — 「이긴 폭」의 네 배가 넘는다.

만드는 파일:

  ch20/strategies/img/tuned_baseline.png

실행:  python3 scripts/make_ch20_tuned_baseline.py   (저장소 최상위에서)
       15 벌 × 13 가지 C × 2 방법 × 5 겹이라 몇 분 걸린다.
필요:  numpy, matplotlib, scikit-learn — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import warnings

import numpy as np

warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.multiclass import OneVsRestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch20/strategies/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

CS = np.logspace(-4, 2, 13)
SEEDS = range(15)
DEFAULT = 1.0                      # scikit-learn 의 기본 벌점 세기


def measure():
    """자료 15 벌 각각에서 두 방법의 교차검증 정확도를 C 별로 잰다."""
    O = np.zeros((len(SEEDS), len(CS)))
    S = np.zeros_like(O)
    for i, sd in enumerate(SEEDS):
        X, y = make_classification(
            n_samples=400, n_features=40, n_informative=14, n_redundant=6,
            n_classes=5, n_clusters_per_class=1, class_sep=1.8, flip_y=0.02,
            random_state=sd)
        for j, C in enumerate(CS):
            O[i, j] = cross_val_score(
                OneVsRestClassifier(LogisticRegression(C=C, max_iter=4000)),
                X, y, cv=5).mean()
            S[i, j] = cross_val_score(
                LogisticRegression(C=C, max_iter=4000,
                                   multi_class="multinomial"),
                X, y, cv=5).mean()
    return O, S


def main():
    O, S = measure()
    o, s = O.mean(0), S.mean(0)
    d = int(np.argmin(abs(CS - DEFAULT)))

    # 자료 한 벌 안에서 고르고 나서 평균을 낸다 — 이것이 정직한 순서다
    gaps = ((S[:, d] - O[:, d]).mean(),
            (S.max(1) - O[:, d]).mean(),
            (S.max(1) - O.max(1)).mean())
    own = (O.max(1) - O.min(1)).mean()

    print(f"{'C':>10}{'OvR':>10}{'소프트맥스':>12}")
    for C, a, b in zip(CS, o, s):
        print(f"{C:>10.3g}{a:>10.4f}{b:>12.4f}")
    print(f"\n  둘 다 기본값 C=1 로     {gaps[0]:+.4f}")
    print(f"  소프트맥스만 맞춤       {gaps[1]:+.4f}")
    print(f"  둘 다 맞춤              {gaps[2]:+.4f}")
    print(f"\n  같은 OvR 이 C 로 움직인 폭 {own:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 벌점 하나로 11 포인트가 움직인다 ──
    ax = axes[0]
    ax.axvline(DEFAULT, color=GREY, lw=1.8, ls=":")
    ax.plot(CS, o * 100, lw=3.4, color=TEAL, marker="o", ms=6,
            label="OvR — 옛 기준선")
    ax.plot(CS, s * 100, lw=3.2, color=RED, marker="D", ms=6, ls="--",
            label="소프트맥스 — 새 방법")
    ax.plot([CS[o.argmax()]], [o.max() * 100], marker="o", ms=14, color=TEAL,
            zorder=5, mec="white", mew=1.8)
    ax.plot([DEFAULT], [o[d] * 100], marker="o", ms=12, color=GREY, zorder=5,
            mec="white", mew=1.8)
    ax.annotate("기본값을 그대로 쓰면\n여기서 잰다",
                xy=(DEFAULT, o[d] * 100), xytext=(6.0, 95.6), fontsize=10.6,
                color=GREY, ha="center",
                arrowprops=dict(arrowstyle="->", color=GREY, lw=1.4))
    ax.annotate(f"성의껏 맞추면 여기다\n같은 OvR 인데 ${own*100:.1f}$ 포인트 위",
                xy=(CS[o.argmax()], o.max() * 100), xytext=(0.0022, 95.6),
                fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xscale("log")
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    ax.set_xticklabels(["$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$",
                        "$1$", "$10$", "$100$"])
    ax.minorticks_off()
    ax.set_xlabel("벌점을 푸는 정도 $C$ — 오른쪽일수록 덜 벌한다", fontsize=11.5)
    ax.set_ylabel("교차검증 정확도", fontsize=11.5)
    ax.set_title("손잡이 하나로 $11$ 포인트가 움직인다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(79, 99)
    ax.set_yticks([80, 85, 90, 95])
    ax.set_yticklabels(["$80\\%$", "$85\\%$", "$90\\%$", "$95\\%$"])
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 고르는 절차만 바꾸면 결론이 바뀐다 ──
    ax = axes[1]
    labs = ("둘 다\n기본값", "소프트맥스만\n맞춤", "둘 다\n맞춤",
            "같은 OvR 안에서\n$C$ 로 생긴 폭")
    vals = (gaps[0] * 100, gaps[1] * 100, gaps[2] * 100, own * 100)
    cols = (INK, RED, TEAL, GREY)
    for x, (v, c) in enumerate(zip(vals, cols)):
        ax.bar(x, v, width=0.58, color=c, alpha=0.9, zorder=3)
        ax.text(x, v + (0.38 if v > 0 else -0.95), f"${v:+.2f}$", ha="center",
                fontsize=12.4, color=c, fontweight="bold")
    ax.axhline(0, color=INK, lw=1.6)
    ax.annotate("발표할 만한 수가 여기서 나온다",
                xy=(1.28, 2.62), xytext=(1.95, 8.6), fontsize=10.8, color=RED,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("맞추고 나면 사라진다", xy=(2.29, -0.30), xytext=(2.78, -3.6),
                fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=10.6)
    ax.set_ylabel("소프트맥스 $-$ OvR (정확도, 포인트)", fontsize=11.2)
    ax.set_title("고르는 절차만 바꾸면 결론이 뒤집힌다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(-5.0, 13.5)
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("기준선을 얼마나 성의 있게 맞췄는가", fontsize=14.5)
    fig.text(0.5, -0.07,
             f"인공자료 {len(SEEDS)} 벌의 평균이다. 벌점을 고르는 절차만 "
             "바꾸고 자료도 모형도 겹 나누기도 모두 같게 두었다. 오른쪽 "
             "마지막 막대는 두 방법의 비교가 아니라 $\\;$OvR 하나를 $C$ 로 "
             "움직였을 때의 폭이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "tuned_baseline.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
