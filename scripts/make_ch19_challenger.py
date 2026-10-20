r"""19.2절 「그림으로 보는 통계학」 — 손상이 없었던 비행은 그림에 없었다.

1986 년 1 월 27 일 밤, 우주왕복선 챌린저의 발사를 하루 앞두고 나사와
고체로켓 제작사 기술진이 회의를 했다. 다음 날 예보 기온이 섭씨 영하
0.6 도(화씨 31 도)로, 그때까지의 어떤 발사보다도 추웠다.

기술진 일부가 연기를 주장했다. 추우면 O링의 탄성이 떨어져 이음매가
새기 쉽다는 것이었다. 그 주장을 뒷받침하려고 자료를 모아 표로 만들었다.

사고 뒤의 조사와 재분석이 지적한 것이 그 표의 모양이다. **O링 손상이
있었던 비행만 모여 있었다.** 손상이 없었던 열여섯 번은 거기 없었다.

손상이 있었던 일곱 번만 놓고 보면 온도와의 관계가 흐릿하다 — 기울기
-0.027, 상관 -0.58, 그리고 점이 일곱 개다. 설득력 있는 그림이 되지
못한다.

스물세 번을 모두 놓으면 달라진다. 손상이 없었던 비행은 거의 모두
따뜻한 날이었다.

  손상 있음 (7 회):  53, 57, 58, 63, 70, 70, 75 도
  손상 없음 (16 회): 66, 67, 67, 67, 68, 69, 70, 70, 72, 73, 75, 76, 76,
                     78, 79, 81 도

이항 로지스틱 회귀를 적합하면

  로짓(p) = 6.653 - 0.1423 * 온도      (온도 1 도 내려갈 때 오즈 1.15 배)

  81 도  O링 하나가 손상될 확률 0.008
  70 도                        0.035
  53 도                        0.292
  31 도                        0.904

발사는 진행되었다. 챌린저는 발사 73 초 만에 공중에서 분해되었고
탑승자 일곱 명이 사망했다.

만드는 파일:

  ch19/logistic_regression/img/challenger.png

실행:  python3 scripts/make_ch19_challenger.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch19/logistic_regression/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

# 널리 공개되어 있는 챌린저 이전 23 회 비행 — (발사 온도 °F, 손상된 O링 수 / 6)
DATA = [(53, 3), (57, 1), (58, 1), (63, 1), (66, 0), (67, 0), (67, 0),
        (67, 0), (68, 0), (69, 0), (70, 0), (70, 0), (70, 1), (70, 1),
        (72, 0), (73, 0), (75, 0), (75, 2), (76, 0), (76, 0), (78, 0),
        (79, 0), (81, 0)]
M = 6.0
T_LAUNCH = 31.0


def fit(t, y):
    X = np.column_stack([np.ones(len(t)), t])
    b = np.zeros(2)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ b))
        W = M * p * (1 - p)
        s = np.linalg.solve((X * W[:, None]).T @ X, X.T @ (y - M * p))
        b += s
        if np.abs(s).max() < 1e-12:
            break
    return b


def main():
    t = np.array([d[0] for d in DATA], float)
    y = np.array([d[1] for d in DATA], float)
    hit = y > 0
    b = fit(t, y)
    print(f"    비행 {len(DATA)} 회 — 손상 있음 {int(hit.sum())} 회, "
          f"없음 {int((~hit).sum())} 회")
    print(f"    로짓(p) = {b[0]:+.4f} {b[1]:+.4f} * 온도   "
          f"(1 도 낮아질 때 오즈 {np.exp(-b[1]):.3f} 배)")
    for T in (81, 70, 60, 53, 40, T_LAUNCH):
        p = 1 / (1 + np.exp(-(b[0] + b[1] * T)))
        print(f"      {T:>4.0f} °F   O링 하나 {p:.4f}   "
              f"여섯 중 하나라도 {1-(1-p)**6:.4f}")
    c = np.polyfit(t[hit], y[hit], 1)
    r = np.corrcoef(t[hit], y[hit])[0, 1]
    print(f"\n    손상 있었던 7 회만: 기울기 {c[0]:+.4f}, 상관 {r:+.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 손상이 있었던 비행만 ──
    ax = axes[0]
    ax.scatter(t[hit], y[hit], s=190, color=RED, zorder=5, edgecolor="white",
               linewidth=1.8)
    xs = np.linspace(50, 84, 50)
    ax.plot(xs, np.polyval(c, xs), lw=2.2, color=RED, ls="--", alpha=0.8)
    ax.set_xlabel("발사 때의 기온 (°F)", fontsize=11.5)
    ax.set_ylabel("손상된 O링의 수", fontsize=11.5)
    ax.set_title(f"손상이 **있었던** {int(hit.sum())} 번만 놓으면".replace("**", ""),
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(50, 84)
    ax.set_ylim(0.5, 2.6)
    ax.set_yticks([1, 2])
    ax.text(0.03, 0.96, f"기울기 ${c[0]:+.3f}$,  상관 ${r:+.2f}$,  점 "
            f"${int(hit.sum())}$ 개\n흐릿하다. 설득할 수 있는 그림이 아니다.",
            transform=ax.transAxes, va="top", fontsize=11, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GREY,
                      alpha=0.95))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 스물세 번 모두 ──
    ax = axes[1]
    xs = np.linspace(28, 85, 400)
    pr = 1 / (1 + np.exp(-(b[0] + b[1] * xs)))
    ax.plot(xs, pr, lw=3.2, color=TEAL)
    ax.scatter(t[hit], y[hit] / M, s=160, color=RED, zorder=5,
               edgecolor="white", linewidth=1.6, label="손상이 있었던 비행")
    ax.scatter(t[~hit], np.zeros((~hit).sum()), s=120, color=BLUE, zorder=5,
               edgecolor="white", linewidth=1.4, alpha=0.9,
               label="손상이 없었던 비행")
    ax.axvline(T_LAUNCH, color=INK, lw=2.2, ls="--")
    p31 = 1 / (1 + np.exp(-(b[0] + b[1] * T_LAUNCH)))
    ax.plot([T_LAUNCH], [p31], marker="o", ms=14, color=INK, zorder=6)
    ax.annotate(f"챌린저 발사 예보 기온 ${T_LAUNCH:.0f}$ °F\n"
                f"추정 확률 ${p31:.2f}$",
                xy=(T_LAUNCH, p31), xytext=(46, 0.80), fontsize=11,
                color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.6))
    ax.annotate("손상이 없었던 비행은 거의 모두\n따뜻한 날이었다 —\n"
                "이 점들이 그림에서 빠져 있었다",
                xy=(74, 0.02), xytext=(56, 0.40), fontsize=10.6, color=BLUE,
                ha="left", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.axvspan(28, 53, color=GREY, alpha=0.12)
    ax.text(40.5, 0.035, "이 구간에는\n자료가 없다", fontsize=10.4,
            color=GREY, ha="center")
    ax.set_xlabel("발사 때의 기온 (°F)", fontsize=11.5)
    ax.set_ylabel("O링 하나가 손상될 확률", fontsize=11.5)
    ax.set_title(f"{len(DATA)} 번을 모두 놓으면", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_xlim(28, 85)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("손상이 없었던 비행은 그림에 없었다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "오른쪽 곡선은 비행 한 번에 O링 여섯 개를 둔 이항 로지스틱 "
             "회귀다. 회색 구간은 그때까지 아무도 발사해 본 적 없는 "
             "온도이고, 그래서 거기서의 값은 외삽이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "challenger.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
