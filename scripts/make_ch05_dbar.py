r"""5.9절 D-bar 의 표집분포 — 그림 둘.

(1) dbar_rho_shape_width.png
    짝 안의 상관 rho 가 D-bar 의 표집분포에서 무엇을 바꾸고 무엇을 안
    바꾸는지 나란히 둔다.

      왼쪽  t = D-bar / (S_D/sqrt n) 의 분포. rho 를 -0.9 에서 +0.9 까지
            옮겨도 셋이 t_{n-1} 위에 그대로 포개진다. 모양은 안 바뀐다.
      오른쪽 평균 구간 폭. sqrt(2(1-rho)) 에 정확히 올라앉는다. 바뀌는
            것은 폭뿐이다.

(2) dbar_personal_equation.png
    상자에 쓸 그림. X_i = mu_X + b_i + e_i, Y_i = mu_Y + b_i + f_i 에서
    차를 잡으면 b_i 가 통째로 사라진다는 것을 천문 관측의 꼴로 보인다.
    원자료의 흔들림은 b_i 가 쥐고 있고, 차의 흔들림은 e 와 f 만 남는다.

만드는 파일:

  ch05/applications/img/dbar_rho_shape_width.png
  ch05/applications/img/dbar_personal_equation.png

실행:  python3 scripts/make_ch05_dbar.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch05/applications/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
MUTED = "#90A4AE"

N = 10
B = 40_000
SEED = 5
SHOW = (-0.9, 0.0, 0.9)
GRID = np.linspace(-0.95, 0.95, 25)


def diffs(n, rho, reps, rng):
    """평균이 같은 이변량 정규 쌍 n 개를 reps 벌 만들어 차를 돌려준다."""
    z1 = rng.normal(size=(reps, n))
    z2 = rng.normal(size=(reps, n))
    return z1 - (rho * z1 + np.sqrt(1 - rho ** 2) * z2)


def fig_shape_width():
    rng = np.random.default_rng(SEED)
    tc = stats.t.ppf(0.975, N - 1)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), constrained_layout=True)

    ax = axes[0]
    for rho, col in zip(SHOW, (MUTED, TEAL, AMBER)):
        d = diffs(N, rho, B, rng)
        t = d.mean(axis=1) / (d.std(axis=1, ddof=1) / np.sqrt(N))
        ax.hist(t, bins=160, range=(-5, 5), density=True, histtype="step",
                lw=2.2, color=col, label=f"ρ = {rho:+.1f}")
    xs = np.linspace(-5, 5, 400)
    ax.plot(xs, stats.t.pdf(xs, N - 1), lw=3.4, color=RED, alpha=0.55,
            zorder=0, label=f"t({N-1}) 밀도")
    ax.set_xlim(-5, 5)
    ax.set_yticks([])
    ax.set_xlabel("t = (차의 평균) / (S_D/√n)", fontsize=11.5)
    ax.set_title("상관을 바꿔도 모양은 그대로다", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.4, framealpha=0.93)
    ax.spines[["top", "right", "left"]].set_visible(False)

    ax = axes[1]
    widths = []
    for rho in GRID:
        d = diffs(N, rho, 8000, rng)
        widths.append((2 * tc * d.std(axis=1, ddof=1) / np.sqrt(N)).mean())
    widths = np.array(widths)

    # E[S_D] = c4(n) * sigma_D 이므로 폭의 기댓값은 정확히 적을 수 있다.
    c4 = np.sqrt(2 / (N - 1)) * gamma(N / 2) / gamma((N - 1) / 2)
    theory = 2 * tc * c4 * np.sqrt(2 * (1 - GRID)) / np.sqrt(N)

    ax.plot(GRID, theory, lw=3.0, color=RED, alpha=0.55, zorder=0,
            label="2 t c4 sqrt(2(1-ρ)/n) — 정확한 기댓값")
    ax.scatter(GRID, widths, s=38, color=TEAL, zorder=4, label="모의에서 잰 평균 폭")
    for rho in SHOW:
        j = int(np.argmin(np.abs(GRID - rho)))
        w = widths[j]
        right = rho > 0.5                       # 오른쪽 끝은 글자가 눈금 밖으로 나간다
        ax.annotate(f"ρ = {rho:+.1f}\n폭 {w:.2f}", xy=(GRID[j], w),
                    xytext=(GRID[j] + (-0.07 if right else 0.07), w + 0.30),
                    ha="right" if right else "left", fontsize=10.2, color=INK,
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    ax.set_xlabel("짝 안의 상관 ρ", fontsize=11.5)
    ax.set_ylabel("명목 95% 구간의 평균 폭", fontsize=11.5)
    ax.set_title("바뀌는 것은 폭뿐이다", fontsize=12.8, color=INK, pad=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 3.2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.4, framealpha=0.93)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"짝 안의 상관은 자를 바꿀 뿐 분포를 바꾸지 않는다  (n = {N})",
                 fontsize=14.5)
    path = OUT + "dbar_rho_shape_width.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    c4({N}) = {c4:.6f}")
    for j in (0, len(GRID) // 2, len(GRID) - 1):
        print(f"    ρ={GRID[j]:+.2f}  모의 {widths[j]:.4f}  기댓값 {theory[j]:.4f}")


def fig_personal_equation():
    rng = np.random.default_rng(11)
    nights = 20
    b = rng.normal(0, 1.0, nights)          # 그날 그 별이 지나간 실제 시각의 흔들림
    bias = 0.80                             # 두 관측자 사이의 「개인 방정식」
    x = b + rng.normal(0, 0.15, nights)
    y = b + bias + rng.normal(0, 0.15, nights)
    d = y - x
    k = np.arange(1, nights + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), sharey=True,
                             constrained_layout=True)

    ax = axes[0]
    ax.plot(k, x, marker="o", ms=7, lw=1.6, color=TEAL, label="관측자 가")
    ax.plot(k, y, marker="s", ms=7, lw=1.6, color=AMBER, label="관측자 나")
    ax.set_title(f"적어 둔 시각 그대로 — 표준편차 {x.std(ddof=1):.2f} 초",
                 fontsize=12.4, color=INK, pad=8)
    ax.set_xlabel("관측한 밤", fontsize=11.5)
    ax.set_ylabel("기준시각과의 차 (초)", fontsize=11.5)
    ax.legend(fontsize=10.4, framealpha=0.93)

    ax = axes[1]
    ax.plot(k, d, marker="D", ms=7, lw=1.6, color=RED)
    ax.axhline(bias, color=MUTED, lw=2.0, ls="--")
    ax.text(1.0, bias + 0.30, f"두 사람 사이의 참 치우침 {bias:.2f} 초",
            fontsize=10.4, color=MUTED)
    ax.set_title(f"둘의 차만 보면 — 표준편차 {d.std(ddof=1):.2f} 초",
                 fontsize=12.4, color=RED, pad=8)
    ax.set_xlabel("관측한 밤", fontsize=11.5)

    for ax in axes:
        ax.set_xlim(0, nights + 1)
        ax.set_xticks(range(0, nights + 1, 5))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("그날그날의 흔들림은 두 사람에게 똑같이 얹힌다 — 빼면 사라진다",
                 fontsize=14.5)
    fig.text(0.5, -0.05,
             "왼쪽 두 줄이 나란히 오르내리는 것은 그날 별이 지나간 시각이 둘 모두에게 "
             "같이 얹혀 있기 때문이다. 그 공통분이 상관이고, 차를 잡으면 통째로 빠진다.",
             ha="center", fontsize=10.5, color=INK)

    path = OUT + "dbar_personal_equation.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    print(f"    원자료 표준편차 {x.std(ddof=1):.4f} · 차의 표준편차 {d.std(ddof=1):.4f}"
          f"  (좁아진 배수 {x.std(ddof=1)/d.std(ddof=1):.1f})")
    print(f"    짝 안의 상관 {np.corrcoef(x, y)[0,1]:.4f} · 차의 평균 {d.mean():.4f}")


if __name__ == "__main__":
    fig_shape_width()
    fig_personal_equation()
