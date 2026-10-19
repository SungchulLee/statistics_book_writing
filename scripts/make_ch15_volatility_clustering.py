r"""15.7절 「그림으로 보는 통계학」 — 수익률은 못 맞혀도 변동성은 맞힌다.

이 쪽 §3 의 그림은 **변동성이 일정한데도** 이동 추정값이 출렁이는 모습,
곧 「출렁임이 잡음일 수 있다」를 보였다. 이 그림은 그 반대쪽이다 —
출렁임이 **신호**인 경우가 어떻게 생겼고, 둘을 어떻게 가르는가.

  위     무조건분산이 똑같은 두 계열. 독립인 t(5) 와 GARCH(1,1).
         눈으로 보면 전혀 다르다 — 뒤엣것은 조용한 때와 요란한 때가
         뭉쳐 있다.
  아래 왼 수익률 자체의 자기상관. 둘 다 0 언저리다. 내일 오를지 내릴지는
         못 맞힌다.
  아래 오 제곱수익률의 자기상관. GARCH 만 또렷이 양수다. 내일이 요란할지
         조용할지는 맞힐 수 있다.

이 비대칭이 엥글이 1982 년에 붙잡은 것이다.

GARCH(1,1) 은 alpha = 0.1, beta = 0.85 로 이 쪽 §5 가 쓰는 값과 맞췄다.
지속성 alpha + beta = 0.95 이므로 충격이 절반으로 사그라드는 데
ln(0.5)/ln(0.95) = 13.5 일 걸린다.

만드는 파일:

  ch15/applications/img/volatility_clustering.png

실행:  python3 scripts/make_ch15_volatility_clustering.py   (저장소 최상위에서)
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

ALPHA, BETA = 0.10, 0.85
T, SEED = 900, 1982
LAGS = 20
LONG = 400_000          # 자기상관을 정밀하게 재려고 길게 한 번 더 돌린다


def garch(T, rng):
    """GARCH(1,1) — 무조건분산이 1 이 되도록 omega 를 잡는다."""
    omega = 1.0 - ALPHA - BETA
    s2 = np.empty(T)
    r = np.empty(T)
    s2[0] = 1.0
    z = rng.standard_normal(T)
    for t in range(T):
        if t:
            s2[t] = omega + ALPHA * r[t - 1] ** 2 + BETA * s2[t - 1]
        r[t] = np.sqrt(s2[t]) * z[t]
    return r, np.sqrt(s2)


def acf(x, lags):
    x = x - x.mean()
    d = float(x @ x)
    return np.array([float(x[k:] @ x[:-k]) / d for k in range(1, lags + 1)])


def main():
    rng = np.random.default_rng(SEED)
    r_g, sig = garch(T, rng)
    r_i = rng.standard_t(5, T) / np.sqrt(5 / 3)        # 분산 1 로 맞춘 t(5)

    rl_g, _ = garch(LONG, rng)
    rl_i = rng.standard_t(5, LONG) / np.sqrt(5 / 3)
    a_r_g, a_r_i = acf(rl_g, LAGS), acf(rl_i, LAGS)
    a_s_g, a_s_i = acf(rl_g ** 2, LAGS), acf(rl_i ** 2, LAGS)

    half = np.log(0.5) / np.log(ALPHA + BETA)
    print(f"    지속성 alpha+beta = {ALPHA+BETA:.2f}   반감기 {half:.1f} 일")
    print(f"    무조건 표준편차  GARCH {rl_g.std():.4f}   t(5) {rl_i.std():.4f}")
    print(f"    수익률 자기상관 (시차 1)     GARCH {a_r_g[0]:+.4f}   t(5) {a_r_i[0]:+.4f}")
    print(f"    제곱수익률 자기상관 (시차 1)  GARCH {a_s_g[0]:+.4f}   t(5) {a_s_i[0]:+.4f}")
    print(f"    제곱수익률 자기상관 (시차 5)  GARCH {a_s_g[4]:+.4f}   t(5) {a_s_i[4]:+.4f}")
    print(f"    초과첨도  GARCH {stats.kurtosis(rl_g):.2f}   t(5) {stats.kurtosis(rl_i):.2f}")

    fig = plt.figure(figsize=(12.6, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(r_i + 9, lw=0.8, color=MUTED)
    ax.plot(r_g - 9, lw=0.8, color=TEAL)
    ax.plot(2 * sig - 9, lw=1.8, color=RED)
    ax.plot(-2 * sig - 9, lw=1.8, color=RED)
    ax.text(5, 14.2, f"독립인 $t(5)$ — 표준편차 {r_i.std():.2f}",
            fontsize=11, color=INK)
    ax.text(5, -3.2, f"GARCH(1,1) — 표준편차 {r_g.std():.2f}  "
                     "(붉은 선은 그날의 참 $\\pm 2\\sigma_t$)",
            fontsize=11, color=TEAL)
    ax.set_yticks([])
    ax.set_xlabel("거래일", fontsize=11.5)
    ax.set_title("무조건분산이 똑같은 두 계열", fontsize=13, color=INK, pad=8)
    ax.set_xlim(0, T)
    ax.spines[["top", "right", "left"]].set_visible(False)

    k = np.arange(1, LAGS + 1)
    band = 1.96 / np.sqrt(LONG)
    for ax, (ag, ai, ttl) in zip(
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
            [(a_r_g, a_r_i, "수익률 $r_t$ 의 자기상관"),
             (a_s_g, a_s_i, "제곱수익률 $r_t^2$ 의 자기상관")]):
        ax.bar(k - 0.19, ag, width=0.38, color=TEAL, label="GARCH(1,1)")
        ax.bar(k + 0.19, ai, width=0.38, color=MUTED, label="독립인 $t(5)$")
        ax.axhspan(-band, band, color=RED, alpha=0.12)
        ax.axhline(0, color=INK, lw=0.9)
        ax.set_xlabel("시차 (일)", fontsize=11.5)
        ax.set_title(ttl, fontsize=12.6, color=INK, pad=8)
        ax.set_ylim(-0.05, 0.30)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.22, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=10, framealpha=0.93, loc="upper right")
    fig.axes[-1].annotate(f"시차 $1$ 에서 {a_s_g[0]:+.3f}",
                          xy=(1, a_s_g[0]), xytext=(4.5, 0.245),
                          fontsize=11, color=TEAL,
                          arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    fig.axes[-2].text(10.5, 0.24, "둘 다 붉은 띠 안 — 못 맞힌다",
                      fontsize=10.6, color=RED, ha="center")

    fig.suptitle("내일 오를지는 못 맞혀도, 내일이 요란할지는 맞힌다", fontsize=14.5)
    fig.text(0.5, -0.035,
             f"붉은 띠는 자기상관이 $0$ 일 때의 $95\\%$ 범위다. 자기상관은 "
             f"${LONG//1000}$ 천 일을 따로 돌려 쟀고, 위 그림은 그 가운데 "
             f"${T}$ 일이다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "volatility_clustering.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
