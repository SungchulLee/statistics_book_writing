r"""6.3절 「그림으로 보는 통계학」 — 「아무것도 모른다」를 어떻게 적는가.

이 쪽은 피셔 정보량이 모수화에 의존한다는 것과, 그래서 제프리스
사전분포 pi(theta) ∝ sqrt(I(theta)) 가 매력적이라는 것을 풀이에 적어
둔다. 왜 매력적인지는 보이지 않는다.

문제는 이렇게 시작한다. 모르는 비율 p 에 대해 「아무 의견이 없다」를
어떻게 적을 것인가. 가장 자연스러운 답이 p 에 평평한 분포다.

그런데 **p 에 평평하면 로그오즈에는 평평하지 않다.** 같은 모르는 것을
다른 눈금으로 적었을 뿐인데 「의견 없음」이 「의견 있음」이 된다. 오즈로
적으면 또 다르고, sqrt(p) 로 적으면 또 다르다.

모두 Beta(a,b) 꼴로 적힌다.

  p 에 평평            a=1,   b=1
  제프리스              a=0.5, b=0.5   (= arcsin sqrt(p) 에 평평)
  로그오즈에 평평        a=0,   b=0     (홀데인)
  sqrt(p) 에 평평       a=0.5, b=1

n=10 일 때 사후평균이 이렇게 갈린다.

   k    MLE    홀데인   sqrt p   제프리스   p 에 평평
   0   0.000   0.000   0.043    0.045     0.083
   2   0.200   0.200   0.217    0.227     0.250
  10   1.000   1.000   0.913    0.955     0.917

한 번도 보지 못한 사건(k=0)에서 0 과 0.083 사이가 갈린다.

제프리스(1939)의 답은 이것이다. 피셔 정보량의 제곱근을 사전분포로
쓰면, 어떤 눈금으로 모형을 적어도 **같은 답이 나온다.** 모수화에 불변인
유일한 규칙이다.

만드는 파일:

  ch07/theory/img/jeffreys.png

실행:  python3 scripts/make_ch06_jeffreys.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch07/theory/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

N = 10
PRIORS = ((0.0, 0.0, PURPLE, "로그오즈에 평평 — Beta$(0,0)$"),
          (0.5, 1.0, AMBER, "$\\sqrt{p}$ 에 평평 — Beta$(0.5,1)$"),
          (0.5, 0.5, TEAL, "제프리스 — Beta$(0.5,0.5)$"),
          (1.0, 1.0, RED, "$p$ 에 평평 — Beta$(1,1)$"))


def main():
    print(f"  n={N} 일 때 사후평균")
    head = "".join(f"{lab.split(' —')[0]:>14}" for *_, lab in PRIORS)
    print(f"{'k':>4}{'MLE':>8}" + head)
    for k in (0, 1, 2, 3, 5, 8, 10):
        row = ""
        for a, b, _, _ in PRIORS:
            A, B = a + k, b + N - k
            row += f"{A/(A+B):>14.3f}"
        print(f"{k:>4}{k/N:>8.3f}{row}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 「정보 없음」을 자처하는 넷 ──
    ax = axes[0]
    p = np.linspace(0.004, 0.996, 600)
    for a, b, col, lab in PRIORS:
        d = p ** (a - 1) * (1 - p) ** (b - 1)
        d = d / np.trapz(d, p) if a > 0 and b > 0 else d / d[len(d) // 2] * 1.0
        ax.plot(p, d, lw=3.0, color=col, label=lab)
    ax.set_xlabel("$p$", fontsize=11.5)
    ax.set_ylabel("사전분포 밀도", fontsize=11.5)
    ax.set_title("넷 다 「아무 의견이 없다」고 말한다", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_ylim(0, 4.4)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper center")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 눈금을 바꾸면 ──
    ax = axes[1]
    phi = np.linspace(-6, 6, 700)
    pp = 1 / (1 + np.exp(-phi))
    jac = pp * (1 - pp)                       # |dp/dphi|
    flat_push = 1.0 * jac                     # p 에 평평 → phi 척도
    jeff_p_push = pp ** -0.5 * (1 - pp) ** -0.5 * jac   # 제프리스(p) → phi 척도
    jeff_phi = np.sqrt(pp * (1 - pp))         # phi 로 적은 모형의 sqrt(I(phi))
    for v, col, lw, ls, lab in (
            (flat_push, RED, 3.0, "-", "「$p$ 에 평평」을 로그오즈 눈금으로 옮긴 것"),
            (jeff_p_push, TEAL, 3.4, "-", "제프리스$(p)$ 를 옮긴 것"),
            (jeff_phi, INK, 2.2, ":", "로그오즈로 적은 모형의 $\\sqrt{I(\\varphi)}$")):
        ax.plot(phi, v / np.trapz(v, phi), lw=lw, color=col, ls=ls,
                label=lab)
    jn = jeff_phi / np.trapz(jeff_phi, phi)
    ax.annotate("두 선이 겹친다 — 제프리스는\n눈금을 바꾸어도 제프리스다",
                xy=(2.4, jn[np.argmin(abs(phi - 2.4))]),
                xytext=(3.5, 0.205), fontsize=10.6, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.annotate("평평했던 것이 평평하지 않다 —\n"
                "$p$ 에 대한 「의견 없음」이\n로그오즈에 대해서는 의견이다",
                xy=(-2.6, flat_push[np.argmin(abs(phi + 2.6))]
                    / np.trapz(flat_push, phi)),
                xytext=(-3.6, 0.235), fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    ax.set_xlabel("로그오즈 $\\varphi = \\log\\dfrac{p}{1-p}$", fontsize=11.5)
    ax.set_ylabel("밀도", fontsize=11.5)
    ax.set_title("같은 모름을 다른 눈금으로 적으면", fontsize=12.8,
                 color=INK, pad=8)
    ax.set_xlim(-6, 6)
    ax.set_ylim(0, 0.34)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("「아무것도 모른다」를 어떻게 적는가", fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽 네 곡선은 모두 「정보 없음」을 자처하는 사전분포다. "
             "오른쪽은 그 가운데 둘을 로그오즈 눈금으로 옮긴 것이며, "
             "검은 점선은 모형을 처음부터 로그오즈로 적고 구한 제프리스 "
             "사전분포다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "jeffreys.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
