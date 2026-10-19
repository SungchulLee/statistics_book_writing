r"""15.7절 「그림으로 보는 통계학」 — 통과했다는 것이 무슨 뜻인가.

이 쪽 §3 은 두 단계 절차의 크기가 0.057 로 어정쩡한 까닭을 「사전검정이
자주 놓치기 때문」이라고 적는다. 절반만 맞는 설명이다. 나머지 절반이
이 그림이다.

사전검정을 통과했다는 것은 **분산이 같다는 뜻이 아니라, 큰 분산을 가진
집단이 마침 작게 나왔다는 뜻**이다. 그러면

  MSW (분모)  통과분에서 38.90 — 참 평균분산 43.8 보다 11% 작다
  MSB (분자)  통과분에서 43.66 — 거의 그대로다

분모만 작아지므로 F 가 더 부푼다. 그래서 **통과분에서의 1 종 오류율이
전체에서의 것보다 높다** — 0.084 대 0.067. 사전검정이 분산분석에
넘겨주는 자료는 무작위 표본이 아니라 **골라낸 표본**이다.

sd = (5, 5, 5, 10), 집단 4 개에 각 15 개.

만드는 파일:

  ch15/applications/img/pretest_conditional.png

실행:  python3 scripts/make_ch15_pretest_conditional.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/applications/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

G, M, REPS, SEED = 4, 15, 60_000, 1953
SD = np.array([5.0, 5.0, 5.0, 10.0])
N = G * M


def parts(z):
    gm = z.mean(axis=(1, 2), keepdims=True)
    gi = z.mean(axis=2, keepdims=True)
    msb = (M * ((gi - gm) ** 2).sum(axis=1)).ravel() / (G - 1)
    msw = ((z - gi) ** 2).sum(axis=(1, 2)) / (N - G)
    return msb, msw


def pval(z):
    msb, msw = parts(z)
    return stats.f.sf(msb / msw, G - 1, N - G)


def main():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, (REPS, G, M)) * SD[None, :, None]   # 평균이 모두 같다

    p_bf = pval(np.abs(x - np.median(x, axis=2, keepdims=True)))   # 사전검정
    p_aov = pval(x)
    ok = p_bf >= 0.05                      # 사전검정을 통과한 표본
    msb, msw = parts(x)
    sd_big = x[:, 3, :].std(axis=1, ddof=1)

    e_all = float((p_aov < 0.05).mean())
    e_ok = float((p_aov[ok] < 0.05).mean())
    e_no = float((p_aov[~ok] < 0.05).mean())

    print(f"    사전검정 통과 비율 {ok.mean():.4f}")
    print(f"    표준 분산분석의 1 종 오류율   전체 {e_all:.4f}   "
          f"통과분 {e_ok:.4f}   기각분 {e_no:.4f}")
    print(f"    MSW  전체 {msw.mean():.2f}  통과분 {msw[ok].mean():.2f}  "
          f"(참 평균분산 {(SD**2).mean():.1f})")
    print(f"    MSB  전체 {msb.mean():.2f}  통과분 {msb[ok].mean():.2f}")
    print(f"    참 sd=10 집단의 관측 sd   전체 {sd_big.mean():.3f}  "
          f"통과분 {sd_big[ok].mean():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), constrained_layout=True)

    # ── 왼쪽: 통과한 표본은 어떤 표본인가 ──
    ax = axes[0]
    bins = np.linspace(4, 17, 70)
    ax.hist(sd_big[~ok], bins=bins, color=MUTED, alpha=0.75,
            label=f"사전검정이 기각  ({(~ok).mean()*100:.0f}%)")
    ax.hist(sd_big[ok], bins=bins, color=TEAL, alpha=0.85,
            label=f"사전검정을 통과  ({ok.mean()*100:.0f}%)")
    ax.axvline(10, color=RED, lw=2.4)
    ax.text(10.2, ax.get_ylim()[1] * 0.93, "참값 $\\sigma = 10$",
            fontsize=10.8, color=RED)
    ax.annotate(f"통과분의 평균 {sd_big[ok].mean():.2f}",
                xy=(sd_big[ok].mean(), ax.get_ylim()[1] * 0.35),
                xytext=(4.6, ax.get_ylim()[1] * 0.62), fontsize=10.6, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xlabel("표준편차가 가장 큰 집단의 **관측** 표준편차".replace("**", ""),
                  fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title("통과한 표본은 무작위 표본이 아니다",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 그래서 뒤이은 분산분석이 더 나빠진다 ──
    ax = axes[1]
    labels = ["모든 표본에\n표준 분산분석", "사전검정을 통과한\n표본에만",
              "사전검정이 기각한\n표본에만"]
    vals = [e_all, e_ok, e_no]
    cols = [MUTED, RED, TEAL]
    ax.bar(range(3), np.array(vals) * 100, color=cols, alpha=0.88, width=0.6)
    ax.axhline(5, color=INK, lw=2.0, ls="--")
    ax.text(2.45, 5.15, "약속한 5%", ha="right", fontsize=10.4, color=INK)
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 0.12, f"{v*100:.2f}%", ha="center",
                fontsize=12, color=INK)
    ax.annotate("통과분이 더 나쁘다", xy=(1, e_ok * 100),
                xytext=(1.5, e_ok * 100 + 0.9), fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.text(0.5, 1.2, f"MSW 가 통과분에서 {msw[ok].mean():.1f} 로\n"
                      f"참값 {(SD**2).mean():.1f} 보다 "
                      f"{(1-msw[ok].mean()/(SD**2).mean())*100:.0f}% 작다\n"
                      f"(MSB 는 {msb.mean():.1f} → {msb[ok].mean():.1f} 로 그대로)",
            fontsize=10.4, color=INK)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=10.4)
    ax.set_ylabel("표준 분산분석의 1 종 오류율 (%)", fontsize=11.5)
    ax.set_title(f"평균은 네 집단 모두 같게 두었다  ({REPS//1000} 천 번)",
                 fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0, 9.8)
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("사전검정은 자료를 걸러서 넘긴다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"표준편차를 $(5,5,5,10)$ 으로 두고 집단 {G} 개에 각 {M} 개씩 뽑았다. "
             "브라운포사이드를 통과한 표본만 모으면, 큰 분산 집단이 마침 작게 나온 "
             "표본들이 모인다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "pretest_conditional.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
