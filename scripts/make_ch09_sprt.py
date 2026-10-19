r"""9.3절 「그림으로 보는 통계학」 — 전쟁이 끝날 때까지 묶여 있던 검정.

이 책은 순차확률비검정(SPRT)을 한 줄로만 적어 두었다. 「고정 표본 검정
보다 평균 표본크기가 작다는 것이 증명되어 있다.」 그 말이 얼마나 작다는
뜻인지는 어디에도 없다. 여기서 잰다.

1943 년 컬럼비아 대학의 통계연구단(SRG)은 탄약 검사 문제를 받았다. 고정
표본 계획은 「181 발을 쏘아 보고 불량이 4 발 이하면 합격」 같은 모양이다.
그런데 첫 스무 발에서 불량이 다섯이면 왜 계속 쏘는가. 첫 백 발이 모두
멀쩡하면 왜 계속 쏘는가.

왈드의 답은 한 발마다 로그우도비를 더해 가며 위아래 두 문턱과 견주는
것이었다. 위를 넘으면 기각, 아래를 넘으면 합격, 사이면 한 발 더.

  A = log((1-beta)/alpha),   B = log(beta/(1-alpha))

이 방법은 군사적 가치가 커서 전쟁이 끝날 때까지 공개가 묶였다. 왈드의
책 『순차분석』은 1947 년에야 나왔다. 영국에서는 바너드가 따로 같은 것을
얻었고 그쪽도 묶였다.

1948 년에 왈드와 울포위츠가 더 센 것을 증명했다. 오류확률이 alpha, beta
이하인 **모든** 검정 가운데 — 순차든 고정이든 — SPRT 가 두 가설 아래의
기대 표본크기를 **동시에 최소화한다.**

재 보면 이렇다 (불량률 1% 대 5%, alpha = beta = 0.05).

  고정표본 계획            181 발, 불량 4 발 이하면 합격
  SPRT  좋은 묶음에서      평균 111.9 발   38.2% 절감
  SPRT  나쁜 묶음에서      평균  78.3 발   56.7% 절감

다만 공짜가 아니다. 좋은 묶음에서 SPRT 가 쓰는 발수의 90 분위는 192 로
고정 계획의 181 을 **넘는다.** 평균으로 아끼고 가끔 더 치르는 것이다.

만드는 파일:

  ch09/errors_and_power/img/sprt.png

실행:  python3 scripts/make_ch09_sprt.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch09/errors_and_power/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

P0, P1 = 0.01, 0.05        # 합격시킬 불량률 / 되돌려보낼 불량률
ALPHA = BETA = 0.05
REPS, CAP, SEED = 40_000, 20_000, 1943


def fixed_plan(p0, p1, al, be):
    """두 오류확률을 모두 지키는 가장 작은 고정표본 계획."""
    for n in range(5, 4001):
        for c in range(n + 1):
            if stats.binom.sf(c, n, p0) <= al and stats.binom.cdf(c, n, p1) <= be:
                return n, c, float(stats.binom.sf(c, n, p0)), \
                    float(stats.binom.cdf(c, n, p1))
    raise RuntimeError("계획을 찾지 못했다")


def run(p, rng, reps=REPS, keep_paths=0):
    A = np.log((1 - BETA) / ALPHA)
    B = np.log(BETA / (1 - ALPHA))
    up = np.log(P1 / P0)
    dn = np.log((1 - P1) / (1 - P0))
    N = np.empty(reps)
    rej = np.zeros(reps, bool)
    paths = []
    for r in range(reps):
        s, n = 0.0, 0
        trail = [0.0] if r < keep_paths else None
        while B < s < A and n < CAP:
            n += 1
            s += up if rng.random() < p else dn
            if trail is not None:
                trail.append(s)
        N[r], rej[r] = n, s >= A
        if trail is not None:
            paths.append(np.array(trail))
    return N, rej, paths, A, B


def main():
    rng = np.random.default_rng(SEED)
    n_fix, c_fix, a_act, b_act = fixed_plan(P0, P1, ALPHA, BETA)
    print(f"    고정표본 계획: {n_fix} 발을 보고 불량 {c_fix} 발 이하면 합격")
    print(f"      실제 alpha {a_act:.4f},  실제 beta {b_act:.4f}")

    N0, rej0, path0, A, B = run(P0, rng, keep_paths=6)
    N1, rej1, path1, _, _ = run(P1, rng, keep_paths=6)
    print(f"    문턱  A = {A:+.3f},  B = {B:+.3f}")
    for lab, N, rej in (("좋은 묶음 (불량 1%)", N0, rej0),
                        ("나쁜 묶음 (불량 5%)", N1, rej1)):
        print(f"    {lab:<20} 평균 {N.mean():6.1f}   중앙값 {np.median(N):5.0f}   "
              f"90분위 {np.quantile(N,0.9):5.0f}   되돌려보낼 확률 {rej.mean():.4f}   "
              f"절감 {100*(1-N.mean()/n_fix):4.1f}%")
    over = (N0 > n_fix).mean()
    print(f"    좋은 묶음에서 고정 계획보다 더 많이 본 비율 = {over:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 두 문턱 사이를 걷는다 ──
    ax = axes[0]
    ax.axhspan(B, A, color=MUTED, alpha=0.15, zorder=0)
    ax.axhline(A, color=RED, lw=2.2)
    ax.axhline(B, color=TEAL, lw=2.2)
    ax.text(4, A + 0.22, f"$A = {A:.2f}$ — 넘으면 묶음을 되돌려보낸다",
            fontsize=10.8, color=RED)
    ax.text(4, B - 0.55, f"$B = {B:.2f}$ — 넘으면 합격시킨다",
            fontsize=10.8, color=TEAL)
    for p, col in ((path0, TEAL), (path1, RED)):
        for tr in p:
            ax.plot(np.arange(len(tr)), tr, lw=1.5, color=col, alpha=0.75)
    ax.axvline(n_fix, color=INK, lw=2.0, ls="--")
    ax.annotate(f"고정표본 계획은\n여기까지 본다 — {n_fix} 발",
                xy=(n_fix, -2.0), xytext=(46, -4.7), fontsize=10.8,
                color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.4))
    ax.legend(handles=[
        Line2D([], [], color=TEAL, lw=2.0, label="좋은 묶음에서 걸은 길"),
        Line2D([], [], color=RED, lw=2.0, label="나쁜 묶음에서 걸은 길")],
        fontsize=10.2, framealpha=0.93, loc="lower right")
    ax.set_xlabel("살펴본 발수", fontsize=11.5)
    ax.set_ylabel("쌓인 로그우도비", fontsize=11.5)
    ax.set_title("한 발마다 더하고, 문턱에 닿으면 멈춘다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 230)
    ax.set_ylim(-5.4, 4.2)
    ax.grid(alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 몇 발이면 끝나는가 ──
    ax = axes[1]
    grid = np.linspace(0, 320, 640)
    for N, col, lab in ((N0, TEAL, "좋은 묶음 (불량 $1\\%$)"),
                        (N1, RED, "나쁜 묶음 (불량 $5\\%$)")):
        ecdf = (N[:, None] <= grid[None, :]).mean(0)
        ax.plot(grid, ecdf, lw=3.0, color=col,
                label=f"{lab} — 평균 {N.mean():.0f} 발")
        ax.plot([N.mean()], [(N <= N.mean()).mean()], marker="o", ms=10,
                color=col, zorder=5)
    ax.axvline(n_fix, color=INK, lw=2.6, ls="--")
    ax.text(n_fix - 7, 0.17, f"고정표본 계획 {n_fix} 발", fontsize=11.4,
            color=INK, ha="right", rotation=90, va="bottom")
    done0 = (N0 <= n_fix).mean()
    ax.plot([n_fix], [done0], marker="s", ms=9, color=AMBER, zorder=6)
    ax.annotate(f"좋은 묶음도 {n_fix} 발 안에\n"
                f"{done0*100:.1f}% 만 끝난다 —\n"
                f"나머지 {over*100:.0f}% 는 더 본다",
                xy=(n_fix, done0), xytext=(243, 0.52), fontsize=10.6,
                color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))
    for N, col, dx in ((N0, TEAL, 10), (N1, RED, -10)):
        ax.annotate(f"{100*(1-N.mean()/n_fix):.0f}% 절감",
                    xy=(N.mean(), (N <= N.mean()).mean()),
                    xytext=(N.mean() + dx, (N <= N.mean()).mean() - 0.16),
                    fontsize=11, color=col,
                    ha="left" if dx > 0 else "right")
    ax.set_xlabel("묶음 하나를 판정하는 데 쓴 발수", fontsize=11.5)
    ax.set_ylabel("그때까지 판정이 끝났을 확률", fontsize=11.5)
    ax.set_title("평균으로 아끼고 가끔 더 치른다",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(0, 320)
    ax.set_ylim(0, 1.04)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("같은 오류확률을 지키면서 표본을 절반으로 줄이다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"불량률 ${P0*100:.0f}\\%$ 짜리 묶음은 합격시키고 "
             f"${P1*100:.0f}\\%$ 짜리는 되돌려보내되, 두 오류확률을 모두 "
             f"${ALPHA}$ 이하로 지킨다는 조건이다. 모의실험 {REPS//10000} 만 번.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "sprt.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
