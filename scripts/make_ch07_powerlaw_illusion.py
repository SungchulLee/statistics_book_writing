r"""7.4절 「그림으로 보는 통계학」 — 직선은 증거가 아니다.

앞 상자에서 파레토는 로그-로그 종이 위의 직선이었다. 그 관찰을 거꾸로
읽으면 곤란하다. 로그-로그에서 직선으로 보인다고 멱법칙인 것은 아니다.

로그정규분포에는 멱법칙이 한 조각도 들어 있지 않다. 그런데 윗꼬리를
흔히 그리는 방식으로 그려 보면

  로그정규(sigma=2) 의 윗 5%
     기울기 1.333,  R^2 0.993,  범위가 2.9 자릿수

논문에 실려도 아무도 의심하지 않을 그림이다.

문제는 그 「기울기」가 어디서 자르느냐에 따라 움직인다는 것이다.
같은 자료 20 만 개에서

  자르는 자리   50%    30%    20%    10%     5%     3%     2%     1%
  로그정규     0.811  0.960  1.059  1.209  1.333  1.412  1.473  1.565
  파레토       1.313  1.316  1.317  1.320  1.328  1.333  1.334  1.332

진짜 멱법칙은 **평평하다.** 꼬리의 어디를 보든 같은 지수가 나오는 것이
멱법칙의 뜻이기 때문이다. 로그정규는 미끄러진다 — 두 배가 된다.

그러니 직선인지가 아니라 **지수가 자리를 옮겨도 그대로인지**를 봐야
한다. 클로짓과 샬리지와 뉴먼이 2009 년에 발표된 멱법칙 주장들을 이
기준으로 다시 재어 보았고, 살아남은 것이 많지 않았다.

만드는 파일:

  ch07/non_normal/img/powerlaw_illusion.png

실행:  python3 scripts/make_ch07_powerlaw_illusion.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch07/non_normal/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

N, SIG, ALPHA, SEED = 200_000, 2.0, 1.3, 2009
SHOW = 0.05
FRACS = np.array([0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.035,
                  0.025, 0.015, 0.01])


def tail(x, frac):
    """윗 frac 비율만 남겨 (값, 꼬리확률) 을 돌려준다."""
    xs = np.sort(x)
    n = len(xs)
    k = int(n * (1 - frac))
    p = 1 - np.arange(k + 1, n + 1) / (n + 1)
    g = p > 0
    return xs[k:][g], p[g]


def fit(x, frac):
    xs, p = tail(x, frac)
    sl, ic = np.polyfit(np.log(xs), np.log(p), 1)
    r = np.corrcoef(np.log(xs), np.log(p))[0, 1]
    return -sl, ic, r ** 2, xs


def main():
    rng = np.random.default_rng(SEED)
    data = {"로그정규": rng.lognormal(0, SIG, N),
            "파레토": rng.pareto(ALPHA, N) + 1}
    cols = {"로그정규": RED, "파레토": TEAL}

    print(f"  n = {N}")
    print(f"{'자르는 자리':>12}" + "".join(f"{f*100:>8.1f}%" for f in FRACS))
    curves = {}
    for nm, x in data.items():
        curves[nm] = np.array([fit(x, f)[0] for f in FRACS])
        print(f"{nm:>12}" + "".join(f"{v:>9.3f}" for v in curves[nm]))
    for nm, x in data.items():
        a, ic, r2, xs = fit(x, SHOW)
        print(f"\n  {nm} 의 윗 {SHOW*100:.0f}%: 기울기 {a:.3f}, R^2 {r2:.5f}, "
              f"범위 {np.log10(xs[-1]/xs[0]):.1f} 자릿수")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1), constrained_layout=True)

    # ── 왼쪽: 둘 다 직선이다 ──
    ax = axes[0]
    for nm, x in data.items():
        a, ic, r2, _ = fit(x, SHOW)
        xs, p = tail(x, SHOW)
        s = max(1, len(xs) // 400)
        ax.plot(xs[::s], p[::s], lw=0, marker="o", ms=4.5, color=cols[nm],
                alpha=0.8, zorder=2)
        gx = np.array([xs[0], xs[-1]])
        ax.plot(gx, np.exp(ic) * gx ** (-a), lw=2.6, color=INK, ls="--",
                zorder=3)
        ax.plot([], [], lw=0, marker="o", ms=7, color=cols[nm],
                label=f"{nm} — 기울기 ${a:.3f}$, $R^2 = {r2:.3f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks([10, 100, 1000, 10000])
    ax.set_xticklabels(["$10$", "$100$", "$10^3$", "$10^4$"])
    ax.set_yticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    ax.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$",
                        "$10^{-1}$"])
    ax.set_ylim(3e-6, 0.09)
    ax.set_xlabel("$x$", fontsize=11.5)
    ax.set_ylabel("$P(X > x)$", fontsize=11.5)
    ax.set_title(f"윗 ${SHOW*100:.0f}\\%$ 만 그리면 둘 다 직선이다", fontsize=12.8,
                 color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 자르는 자리를 옮겨 보면 ──
    ax = axes[1]
    for nm in data:
        ax.plot(FRACS, curves[nm], lw=3.6, color=cols[nm], marker="o", ms=7,
                label=nm)
    ax.axhline(ALPHA, color=TEAL, lw=1.8, ls=":")
    ax.text(0.47, ALPHA + 0.045, f"파레토의 참값 ${ALPHA}$", fontsize=10.6,
            color=TEAL)
    ax.annotate(f"꼬리의 어디를 보든 같다\n"
                f"${curves['파레토'].min():.3f}$ ~ "
                f"${curves['파레토'].max():.3f}$",
                xy=(0.055, curves["파레토"][FRACS.tolist().index(0.05)]),
                xytext=(0.085, 0.84), fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.annotate(f"자르는 자리만 옮겼는데\n${curves['로그정규'].min():.3f}$ 에서 "
                f"${curves['로그정규'].max():.3f}$ 로 두 배가 된다",
                xy=(0.02, curves["로그정규"][FRACS.tolist().index(0.025)]),
                xytext=(0.105, 1.82), fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.set_xscale("log")
    ax.set_xticks([0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    ax.set_xticklabels(["$50\\%$", "$20\\%$", "$10\\%$", "$5\\%$",
                        "$2\\%$", "$1\\%$"])
    ax.minorticks_off()
    ax.invert_xaxis()
    ax.set_xlabel("직선을 맞추는 데 쓴 윗꼬리의 비율", fontsize=11.5)
    ax.set_ylabel("읽어 낸 기울기", fontsize=11.5)
    ax.set_title("자르는 자리를 옮겨 보면", fontsize=12.8, color=INK, pad=8)
    ax.set_ylim(0.70, 1.98)
    ax.legend(fontsize=10.6, framealpha=0.93, loc="upper left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("직선은 증거가 아니다", fontsize=14.5)
    fig.text(0.5, -0.06,
             f"로그정규는 $\\sigma = {SIG:.0f}$, 파레토는 $\\alpha = {ALPHA}$ "
             f"이고 각각 ${N//10000}$ 만 개를 뽑았다. 로그정규에는 멱법칙이 "
             "한 조각도 들어 있지 않다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "powerlaw_illusion.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
