r"""11.3절 「그림으로 보는 통계학」 — 블록화가 나온 자리.

이 쪽은 블록화를 「둘째 요인이 설명하는 변동을 오차에서 빼낸다」로
적는다. 맞는 말인데, 그 생각이 **왜** 필요했는지는 밭에서 나왔다.

밭은 고르지 않다. 한쪽이 기름지고 다른 쪽이 메마른데 품종 A 를 메마른
절반에 B 를 기름진 절반에 심고 수확을 견주면, **품종의 차이와 흙의 차이가
섞인다.**

세 가지 배치를 같은 밭에서 재어 본다.

  반 가르기     메마른 절반 A, 기름진 절반 B -> 추정값이 참값의 세 배
  완전 무작위    64 칸에 아무렇게나 32:32    -> 치우침은 없고 흔들림이 크다
  난괴법        행마다 블록, 블록 안에서 4:4 -> 치우침도 없고 흔들림도 작다

피셔가 로담스테드에서 한 일이 이 둘이다. **무작위화**가 치우침을 없애고
**블록화**가 흔들림을 줄인다.

만드는 파일:

  ch11/anova_two_way/img/blocking_origin.png

실행:  python3 scripts/make_ch11_blocking_origin.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch11/anova_two_way/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

R, C, REPS, SEED = 8, 8, 20_000, 1919
DELTA = 2.0
GRAD = np.linspace(0, 7, R)[:, None] * 1.0        # 위에서 아래로 커지는 비옥도


def layouts(rng, reps):
    """세 배치의 할당을 만든다. 1 이면 품종 B."""
    half = np.zeros((R, C))
    half[R // 2:, :] = 1
    A = np.broadcast_to(half, (reps, R, C))

    i = np.argsort(rng.random((reps, R * C)), axis=1)
    B = np.zeros((reps, R * C))
    np.put_along_axis(B, i[:, : R * C // 2], 1, axis=1)
    B = B.reshape(reps, R, C)

    i = np.argsort(rng.random((reps, R, C)), axis=2)
    D = np.zeros((reps, R, C))
    np.put_along_axis(D, i[:, :, : C // 2], 1, axis=2)
    return A, B, D


def main():
    rng = np.random.default_rng(SEED)
    A, B, D = layouts(rng, REPS)

    def harvest(assign):
        return GRAD + rng.normal(0, 1.0, (REPS, R, C)) + DELTA * assign

    y = harvest(A)
    e_split = (y[A == 1].reshape(REPS, -1).mean(1)
               - y[A == 0].reshape(REPS, -1).mean(1))
    y = harvest(B)
    e_cr = ((y * B).sum((1, 2)) / B.sum((1, 2))
            - (y * (1 - B)).sum((1, 2)) / (1 - B).sum((1, 2)))
    y = harvest(D)
    e_rb = (((y * D).sum(2) / D.sum(2))
            - ((y * (1 - D)).sum(2) / (1 - D).sum(2))).mean(1)

    rows = [("반 가르기", e_split, RED), ("완전 무작위", e_cr, AMBER),
            ("난괴법 (블록화)", e_rb, TEAL)]
    for nm, e, _ in rows:
        print(f"    {nm:14} 평균 {e.mean():+7.3f}  (참값 {DELTA:+.1f})   "
              f"표준편차 {e.std():.4f}")
    print(f"    난괴법이 완전 무작위보다 표준편차 {e_cr.std()/e_rb.std():.2f} 배 작다")

    fig = plt.figure(figsize=(12.6, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1])

    one = layouts(np.random.default_rng(3), 1)
    titles = ["① 반 가르기", "② 완전 무작위", "③ 난괴법 — 행마다 블록"]
    for j, (asg, ttl, col) in enumerate(zip(one, titles, [RED, AMBER, TEAL])):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(np.broadcast_to(GRAD, (R, C)), cmap="YlOrBr_r",
                  interpolation="nearest")
        a = asg[0]
        for r in range(R):
            for c in range(C):
                ax.text(c, r, "B" if a[r, c] else "A", ha="center", va="center",
                        fontsize=9.5, fontweight="bold",
                        color=("#1A237E" if a[r, c] else "#B71C1C"))
        if j == 2:
            for r in range(1, R):
                ax.axhline(r - 0.5, color="white", lw=1.6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(ttl, fontsize=12.4, color=col, pad=7)
        if j == 0:
            # GRAD 는 위에서 아래로 커진다 — 맨 윗줄이 가장 메마르다
            ax.set_ylabel("↑ 메마름        기름짐 ↓", fontsize=10.6, color=INK)

    ax = fig.add_subplot(gs[1, :])
    bins = np.linspace(0.5, 7.0, 180)
    for nm, e, col in rows:
        ax.hist(e, bins=bins, color=col, alpha=0.75,
                label=f"{nm} — 평균 {e.mean():+.2f}, 표준편차 {e.std():.2f}")
    ax.axvline(DELTA, color=INK, lw=2.4, ls="--")
    ax.text(DELTA + 0.07, ax.get_ylim()[1] * 0.92,
            f"참값 $\\delta = {DELTA:g}$", fontsize=11.5, color=INK)
    ax.annotate("흙의 기울기가 그대로 더해진다",
                xy=(e_split.mean(), ax.get_ylim()[1] * 0.45),
                xytext=(4.6, ax.get_ylim()[1] * 0.72), fontsize=11, color=RED,
                ha="center", arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate(f"②보다 {e_cr.std()/e_rb.std():.1f} 배 좁다",
                xy=(2.0, ax.get_ylim()[1] * 0.55),
                xytext=(0.75, ax.get_ylim()[1] * 0.80), fontsize=11, color=TEAL)
    ax.set_xlabel("품종 효과의 추정값", fontsize=11.5)
    ax.set_ylabel("횟수", fontsize=11.5)
    ax.set_title(f"같은 밭에 배치만 바꾸어 {REPS//1000} 천 번 심어 보았다",
                 fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.22, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무작위화가 치우침을 없애고 블록화가 흔들림을 줄인다",
                 fontsize=14.5)
    fig.text(0.5, -0.035,
             "칸의 색이 흙의 기름짐이다 — 밝을수록 기름지다(아래쪽). 품종 B 가 A 보다 "
             f"참으로 ${DELTA:g}$ 만큼 낫게 두었다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "blocking_origin.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
