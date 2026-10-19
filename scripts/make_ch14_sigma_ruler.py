r"""14.6절 「그림으로 보는 통계학」 — 25 표준편차짜리 날.

이 쪽은 1987 년 폭락이 「대략 20 표준편차에 해당하며 사실상 확률이 0인
사건」이라고 적고 멈춘다. 그 다음 물음이 이 그림이다 — **그 표준편차는
어디서 왔는가?**

2007 년 8 월, 골드만삭스의 재무책임자 데이비드 비니아는 파이낸셜 타임스에
이렇게 말했다. "우리는 25 표준편차짜리 움직임을 며칠 연속으로 보고
있었습니다."

정규분포에서 25 표준편차를 기다리는 시간은 6.5e134 년이다. 우주의 나이가
1.38e10 년이다. 그러니 저 문장은 두 가지로 읽힌다.

  (가) 우주의 나이보다 10^124 배 긴 시간에 한 번 올 일이 사흘 연속 일어났다
  (나) 25 라는 수를 만든 자가 틀렸다

(나)다. 그 σ 는 **조용하던 기간의 자료로 추정한 값**이다. 큰 사건이 없던
창에서 잰 자를 들고 큰 사건을 재면, 눈금이 터무니없이 커진다.

이 그림은 그것을 잰다. 꼬리가 두꺼운 평범한 모형(표준화한 t 분포)에서
자료를 만들고, 실무자가 하는 그대로 **앞선 250 거래일**로 σ 를 추정해
그날의 움직임을 「몇 σ」로 환산한다. 정규분포에서는 20,000 년을 돌려도
5.3σ 가 최대인데, t(3) 에서는 한 해에 25σ 넘는 날이 생길 확률이 0.91% —
110 년에 한 번이다. 모형이 특별해서가 아니라 자가 틀려서다.

만드는 파일:

  ch14/applications/img/sigma_ruler.png

실행:  python3 scripts/make_ch14_sigma_ruler.py   (저장소 최상위에서)
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

OUT = "docs/ch14/applications/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

TRADING = 252          # 한 해의 거래일
WIN = 250              # 변동성을 재는 창
YEARS, SEED = 20_000, 2007
UNIVERSE = 1.38e10     # 우주의 나이 (년)

MODELS = [
    ("정규 — 교과서의 가정", lambda rng, sz: rng.standard_normal(sz), BLUE),
    ("$t_6$", lambda rng, sz: rng.standard_t(6, sz) / np.sqrt(6 / 4), TEAL),
    ("$t_4$", lambda rng, sz: rng.standard_t(4, sz) / np.sqrt(4 / 2), AMBER),
    ("$t_3$ — 일간 수익률에 흔한 두께",
     lambda rng, sz: rng.standard_t(3, sz) / np.sqrt(3), RED),
]


def annual_max_sigma(sampler, rng):
    """앞선 250 일로 추정한 sigma 로 잰, 한 해 안의 최대 「몇 σ」."""
    n = WIN + TRADING
    x = sampler(rng, (YEARS, n))
    out = np.zeros(YEARS)
    for t in range(WIN, n):
        s = x[:, t - WIN:t].std(1, ddof=1)
        out = np.maximum(out, np.abs(x[:, t]) / s)
    return out


def main():
    rng = np.random.default_rng(SEED)

    ks = np.linspace(3, 26, 300)
    wait = 1.0 / (2 * stats.norm.sf(ks)) / TRADING      # 정규에서 기다리는 햇수
    print("    정규분포에서 kσ 인 날을 기다리는 시간")
    for k in (4, 5, 10, 20, 25):
        w = 1.0 / (2 * stats.norm.sf(k)) / TRADING
        print(f"      {k:2}σ   {w:.3e} 년   (우주 나이의 {w/UNIVERSE:.2e} 배)")

    curves = {}
    print(f"\n    앞선 {WIN} 일로 σ 를 추정해 잰 「한 해의 최대 σ」 ({YEARS:,} 해)")
    print(f"{'모형':>26}{'중앙값':>9}{'99분위':>9}{'최댓값':>9}"
          f"{'P(≥5σ)':>10}{'P(≥10σ)':>10}{'P(≥25σ)':>10}")
    for name, f, col in MODELS:
        m = annual_max_sigma(f, rng)
        curves[name] = (m, col)
        plain = name.replace("$", "").replace("_", "")
        print(f"{plain:>26}{np.median(m):>9.2f}{np.quantile(m,0.99):>9.2f}"
              f"{m.max():>9.1f}{(m>=5).mean():>10.4f}{(m>=10).mean():>10.4f}"
              f"{(m>=25).mean():>10.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), constrained_layout=True)

    # ── 왼쪽: 정규분포에서 kσ 를 기다리는 시간 ──
    ax = axes[0]
    ax.plot(ks, wait, lw=3.0, color=BLUE)
    ax.set_yscale("log")
    ax.axhspan(1, UNIVERSE, color=GREY, alpha=0.16, zorder=0)
    ax.axhline(UNIVERSE, color=GREY, lw=1.6, ls=":")
    ax.text(25.6, 3e4, "우주가 생긴 뒤로 흐른 시간 전부 — 138 억 년",
            fontsize=10.6, color=INK, ha="right", va="center")
    for k, col, lab, dy, ha, dx in (
            (4, TEAL, "$4\\sigma$ — 63 년에 한 번", 1.6e28, "left", -0.8),
            (20, AMBER, "1987 년 폭락 — 약 $20\\sigma$", 1e22, "right", -0.7),
            (25, RED, "비니아의 $25\\sigma$ (2007)\n우주 나이의 $10^{124}$ 배", 1.6e5,
             "right", -0.7)):
        w = 1.0 / (2 * stats.norm.sf(k)) / TRADING
        ax.plot([k], [w], marker="o", ms=10, color=col, zorder=5)
        ax.annotate(lab, xy=(k, w), xytext=(k + dx, w * dy),
                    fontsize=10.6, color=col, ha=ha,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
    ax.set_xlabel("움직임의 크기 (표준편차 단위)", fontsize=11.5)
    ax.set_ylabel("정규분포라면 기다려야 하는 햇수", fontsize=11.5)
    ax.set_yticks([1e2, 1e10, 1e20, 1e40, 1e60, 1e80, 1e100, 1e120, 1e140])
    ax.set_yticklabels(["100", "$10^{10}$", "$10^{20}$", "$10^{40}$",
                        "$10^{60}$", "$10^{80}$", "$10^{100}$",
                        "$10^{120}$", "$10^{140}$"])
    ax.minorticks_off()
    ax.set_ylim(1, 1e150)
    ax.set_xlim(3, 26)
    ax.set_title("$25\\sigma$ 를 기다리는 시간", fontsize=12.8, color=INK, pad=8)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 자를 바꾸면 25σ 가 흔해진다 ──
    ax = axes[1]
    grid = np.linspace(2, 32, 400)
    for name, _, _ in MODELS:
        m, col = curves[name]
        surv = (m[:, None] >= grid[None, :]).mean(0)
        ax.plot(grid, np.where(surv > 0, surv, np.nan), lw=2.8, color=col,
                label=name)
    ax.axvline(25, color=RED, lw=2.0, ls="--")
    ax.text(24.4, 1.25, "비니아가 말한 $25\\sigma$", fontsize=11, color=RED,
            ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("한 해의 최대 움직임 (추정한 $\\sigma$ 로 잰 것)", fontsize=11.5)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["1 만 해에 한 번", "천 해에 한 번", "백 해에 한 번",
                        "열 해에 한 번", "해마다"])
    ax.minorticks_off()
    ax.set_ylim(5e-5, 1.6)
    ax.set_xlim(2, 32)
    m3 = curves["$t_3$ — 일간 수익률에 흔한 두께"][0]
    ax.annotate(f"$t_3$ 에서는 {(m3>=25).mean()*100:.2f}% 의 해에 일어난다\n"
                f"— {1/(m3>=25).mean():.0f} 년에 한 번",
                xy=(25, (m3 >= 25).mean()), xytext=(20.0, 1.7e-4),
                fontsize=10.6, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.4))
    mn = curves["정규 — 교과서의 가정"][0]
    ax.set_title("같은 사건, 다른 자", fontsize=12.8, color=INK, pad=8)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="lower left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("25 표준편차짜리 날은 자가 틀렸다는 뜻이다", fontsize=14.5)
    fig.text(0.5, -0.05,
             f"오른쪽은 네 모형에서 각각 {YEARS:,} 해를 돌린 것이다. 어느 모형이든 "
             f"분산을 $1$ 로 맞추었고, 실무에서 하는 그대로 앞선 {WIN} 거래일로 "
             "$\\sigma$ 를 추정해 그날의 움직임을 환산했다. 꼬리가 두꺼워지면 "
             "사건이 커지는 것이 아니라 자가 짧아진다.\n"
             "정규분포(파랑)는 20,000 해를 돌려도 최대가 $5.3\\sigma$ 였다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "sigma_ruler.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
