r"""6.1절 「AI로 보는 통계학」 — 한 번도 못 본 것의 확률.

말뭉치에서 단어를 N 개 보았다. 다음에 나올 단어가 **지금까지 한 번도
보지 못한 단어**일 확률은 얼마인가?

MLE 는 0 이라고 답한다. 보지 못한 것의 상대도수가 0 이기 때문이다. 그런데
말뭉치를 천 개만 본 상태라면 참값은 0.40 쯤이다. 열 번에 네 번이 처음 보는
단어인데 MLE 는 그런 일이 일어날 수 없다고 말한다.

라플라스 평활(가상의 관측을 한 개씩 더하기)은 반대로 넘친다. 어휘가
50,000 종이면 보지 못한 칸들에 통째로 0.97 을 준다.

튜링이 블레츨리 파크에서 쓴 답은 이것이었다.

  빠진 질량의 추정값 = (딱 한 번 나온 단어의 수) / N

딱 한 번 나온 단어의 개수만 세면 된다. 왜 그것으로 되는가 — 말뭉치를 한
개 더 늘렸을 때 「처음 보는 단어」가 될 뻔했던 것들이 바로 지금 한 번만
나온 단어들이기 때문이다.

이 방법은 전시에 묶여 있다가 튜링의 조수였던 I. J. 굿이 1953 년에
『바이오메트리카』에 실으며 공으로 돌렸다. 통계적 언어모형에서 평활의
바탕이 된 것이 이것이고, 신경망이 오기 전까지 n-그램 모형은 모두 이
계열을 썼다.

재 보면 이렇다 (지프 꼴 어휘 50,000 종).

   말뭉치     참값     굿-튜링     MLE     라플라스
    1,000   0.4034   0.4290     0.0     0.9701
   20,000   0.1947   0.1920     0.0     0.6387
  500,000   0.0269   0.0268     0.0     0.0254

만드는 파일:

  ch06/bayesian/img/good_turing.png

실행:  python3 scripts/make_ch06_good_turing.py   (저장소 최상위에서)
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

V, ZIPF, SEED = 50_000, 1.07, 1943
SIZES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000)
SHOW = 20_000          # 오른쪽 그림에 쓸 말뭉치 크기


def main():
    rng = np.random.default_rng(SEED)
    p = 1.0 / np.arange(1, V + 1) ** ZIPF
    p /= p.sum()

    true, gt, lap = [], [], []
    detail = None
    print(f"{'말뭉치':>9}{'참 빠진 질량':>13}{'굿-튜링':>10}{'MLE':>7}{'라플라스':>11}"
          f"{'1회 단어':>10}{'본 종수':>9}")
    for N in SIZES:
        x = rng.choice(V, size=N, p=p)
        cnt = np.bincount(x, minlength=V)
        seen = cnt > 0
        t = float(p[~seen].sum())
        n1 = int((cnt == 1).sum())
        g = n1 / N
        l = float((V - seen.sum()) / (N + V))
        true.append(t); gt.append(g); lap.append(l)
        print(f"{N:>9,}{t:>13.4f}{g:>10.4f}{0.0:>7.1f}{l:>11.4f}"
              f"{n1:>10,}{int(seen.sum()):>9,}")
        if N == SHOW:
            detail = (cnt, n1, t, g)

    cnt, n1, t_show, g_show = detail
    nr = np.bincount(cnt[cnt > 0], minlength=12)[1:11]   # 딱 r 번 나온 단어의 수

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 세 추정값과 참값 ──
    ax = axes[0]
    ax.plot(SIZES, true, lw=3.4, color=INK, marker="o", ms=8,
            label="참값 — 보지 못한 단어들의 실제 확률")
    ax.plot(SIZES, gt, lw=2.8, color=TEAL, marker="D", ms=7, ls="--",
            label="굿–튜링 — $(1$ 번 나온 단어 수$)\\,/\\,N$")
    ax.plot(SIZES, lap, lw=2.6, color=RED, marker="s", ms=6,
            label="라플라스 평활 — 가상 관측 $1$ 개씩")
    ax.plot(SIZES, np.zeros(len(SIZES)), lw=2.6, color=BLUE, marker="v", ms=6,
            label="MLE — 「그런 일은 일어나지 않는다」")
    ax.set_xscale("log")
    ax.set_xticks([1e3, 1e4, 1e5, 5e5])
    ax.set_xticklabels(["1 천", "1 만", "10 만", "50 만"])
    ax.minorticks_off()
    ax.annotate(f"말뭉치 {SIZES[0]:,} 개에서\n"
                f"참값 {true[0]:.2f}\n굿–튜링 {gt[0]:.2f}\n"
                f"라플라스 {lap[0]:.2f}\nMLE $0$",
                xy=(SIZES[0] * 1.06, true[0] + 0.02), xytext=(1.25e3, 0.80),
                fontsize=10.4, color=INK, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.set_xlabel("본 단어 수 $N$", fontsize=11.5)
    ax.set_ylabel("다음 단어가 처음 보는 단어일 확률", fontsize=11.5)
    ax.set_title("세 답 가운데 하나만 맞는다", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(-0.04, 1.04)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 굿-튜링이 쓰는 것은 첫 막대 하나뿐이다 ──
    ax = axes[1]
    cols = [TEAL] + [MUTED] * 9
    ax.bar(range(1, 11), nr, color=cols, width=0.68)
    for r, v in zip(range(1, 11), nr):
        ax.text(r, v + max(nr) * 0.018, f"{v:,}", ha="center", fontsize=9.8,
                color=INK)
    ax.annotate(f"굿–튜링이 쓰는 것은\n이 막대 하나뿐이다\n"
                f"${n1:,} / {SHOW:,} = {g_show:.4f}$",
                xy=(1.42, nr[0] * 0.72), xytext=(4.6, nr[0] * 0.80),
                fontsize=11.0, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.6))
    ax.text(4.6, nr[0] * 0.42,
            f"참값은 ${t_show:.4f}$ 였다.\n한 번만 나온 단어를 세는 것으로\n"
            "보지 못한 단어 전체의 몫을 맞힌다.",
            fontsize=10.6, color=INK, ha="center",
            bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=GREY, alpha=0.95))
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("말뭉치에 나온 횟수 $r$", fontsize=11.5)
    ax.set_ylabel("딱 $r$ 번 나온 단어의 종 수", fontsize=11.5)
    ax.set_title(f"말뭉치 {SHOW:,} 개에서 센 것", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("한 번도 보지 못한 것의 확률을 센다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"어휘 ${V:,}$ 종이 지프 꼴로 퍼져 있다고 두고 말뭉치를 뽑았다. "
             "「참값」은 뽑히지 않은 단어들의 확률을 실제로 더한 것이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "good_turing.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
