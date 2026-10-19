r"""6.5절 「그림으로 보는 통계학」 — 내일 해가 뜰 확률.

라플라스는 1814 년 『확률에 관한 철학적 시론』에서 이렇게 물었다.
해가 n 번 떴다. 내일도 뜰 확률은 얼마인가.

균등 사전분포를 둔 베타-이항 갱신에서 답이 나온다.

  P(다음에도 성공 | n 번 모두 성공) = (n+1)/(n+2)

기록된 역사를 5000 년으로 잡으면 n = 1,826,213 일이고, 확률이
1,826,214 / 1,826,215 다. 실패할 확률이 1,826,215 분의 1.

이 계산은 이백 년 동안 조롱거리였다. 「라플라스는 해가 안 뜰 수도
있다고 생각했다」는 식이다. 그런데 **라플라스 자신이 그 다음 문장에서
이 수를 쓰면 안 되는 경우를 적어 두었다** — 낮과 계절을 지배하는
원리를 아는 사람에게는 이 수가 턱없이 작다는 것이다.

요점은 이것이다. 이 답은 해에 대해 아무것도 모른다. **오직 n 과
사전분포만 보고 나온 수**다. 그리고 k = n 인 구석에서는 사전분포의
영향이 자료가 아무리 쌓여도 사라지지 않는다.

  P(다음에 실패)   균등 1/(n+2)   제프리스 0.5/(n+1)   홀데인 0

균등과 제프리스의 비가 n 이 커져도 2 에 머문다. 홀데인은 영원히 0 이라
한다. 보통 자리(k/n = 0.5)에서는 같은 비가 1 로 수렴한다.

만드는 파일:

  ch06/bayesian/img/succession.png

실행:  python3 scripts/make_ch06_succession.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/bayesian/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N_LAPLACE = 1_826_213           # 5000 년 × 365.2425 일


def main():
    ns = np.logspace(0, 7.2, 400)
    fail_u = 1 / (ns + 2)                 # 균등  Beta(1,1)
    fail_j = 0.5 / (ns + 1)               # 제프리스 Beta(.5,.5)

    print("  n 번 모두 성공했을 때 「다음에 실패할 확률」")
    print(f"{'n':>12}{'균등':>14}{'제프리스':>14}{'홀데인':>9}{'균등/제프리스':>14}")
    for n in (1, 5, 50, 1000, 10**6, N_LAPLACE):
        u, j = 1 / (n + 2), 0.5 / (n + 1)
        print(f"{n:>12,}{u:>14.3e}{j:>14.3e}{0.0:>9.1f}{u/j:>14.4f}")
    print(f"\n  라플라스의 수: 내일 해가 뜰 확률 "
          f"{(N_LAPLACE+1)/(N_LAPLACE+2):.9f}")
    print(f"    = {N_LAPLACE+1:,} / {N_LAPLACE+2:,}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 답은 n 만 보고 나온다 ──
    ax = axes[0]
    ax.plot(ns, fail_u, lw=3.2, color=TEAL,
            label="균등 사전분포 — $1/(n+2)$  (라플라스)")
    ax.plot(ns, fail_j, lw=3.0, color=AMBER, ls="--",
            label="제프리스 — $0.5/(n+1)$")
    ax.axhline(1e-8, color=RED, lw=2.6)
    ax.text(1.2e7, 1.4e-8, "홀데인 사전분포는 $0$ 이라고 한다", fontsize=10.8,
            color=RED, ha="right")
    ax.axvline(N_LAPLACE, color=INK, lw=1.8, ls=":")
    ax.plot([N_LAPLACE], [1 / (N_LAPLACE + 2)], marker="o", ms=13, color=INK,
            zorder=5)
    ax.annotate(f"기록된 역사 $5{{,}}000$ 년\n$= {N_LAPLACE:,}$ 일"
                .replace(",", "{,}") + f"\n실패 확률 $1/{N_LAPLACE+2:,}$"
                .replace(",", "{,}"),
                xy=(N_LAPLACE, 1 / (N_LAPLACE + 2)), xytext=(700, 2.0e-3),
                fontsize=10.6, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 1e2, 1e4, 1e6])
    ax.set_xticklabels(["1", "100", "1 만", "100 만"])
    ax.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    ax.set_yticklabels(["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1"])
    ax.minorticks_off()
    ax.set_xlabel("연달아 성공한 횟수 $n$", fontsize=11.5)
    ax.set_ylabel("「다음에 실패할 확률」", fontsize=11.5)
    ax.set_title("이 답은 해에 대해 아무것도 모른다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(5e-9, 2)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 구석에서는 사전분포가 안 지워진다 ──
    ax = axes[1]
    ratio_corner = (1 / (ns + 2)) / (0.5 / (ns + 1))
    k_half = ns / 2
    ratio_mid = ((ns - k_half + 1) / (ns + 2)) / ((ns - k_half + 0.5) / (ns + 1))
    ax.axhline(1.0, color=INK, lw=2.2, ls="--")
    ax.text(1.3, 1.035, "사전분포가 지워진 상태", fontsize=11, color=INK)
    ax.plot(ns, ratio_corner, lw=3.4, color=RED,
            label="$k = n$ — 한 번도 실패를 못 본 경우")
    ax.plot(ns, ratio_mid, lw=3.0, color=TEAL,
            label="$k/n = 0.5$ — 보통 자리")
    ax.axhline(2.0, color=RED, lw=1.4, ls=":")
    ax.annotate("자료가 아무리 쌓여도\n두 답의 비가 $2$ 에 머문다",
                xy=(3e4, ratio_corner[np.argmin(abs(ns - 3e4))]),
                xytext=(900, 1.62), fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("보통 자리에서는 금세 $1$ 이 된다",
                xy=(300, ratio_mid[np.argmin(abs(ns - 300))]),
                xytext=(1.2e4, 1.22), fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks([1, 1e2, 1e4, 1e6])
    ax.set_xticklabels(["1", "100", "1 만", "100 만"])
    ax.minorticks_off()
    ax.set_xlabel("표본크기 $n$", fontsize=11.5)
    ax.set_ylabel("균등 $\\div$ 제프리스 (같은 물음에 대한 두 답의 비)",
                  fontsize=11.0)
    ax.set_title("구석에서는 사전분포가 지워지지 않는다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(0.9, 2.15)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="center right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("내일 해가 뜰 확률", fontsize=14.5)
    fig.text(0.5, -0.05,
             "물음은 「다음 시행이 실패할 확률」이다. 오른쪽은 같은 물음에 "
             "두 사전분포가 주는 답의 비이며, $1$ 이면 사전분포의 영향이 "
             "지워진 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "succession.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
