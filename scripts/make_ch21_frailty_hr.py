r"""21.5절 「그림으로 보는 통계학」 — 무작위배정은 첫날에만 유효하다.

이 쪽 5 절은 위험비의 오해를 여럿 모아 두었다. 하나가 더 있다. 그리고
그것은 모형을 잘못 쓴 탓이 아니라 **위험비라는 양 자체의 성질**이다.

완벽하게 무작위배정된 시험을 생각하자. 교란은 없다. 그리고 치료가
**모든 사람의** 위험을 어느 순간에나 정확히 절반으로 낮춘다고 하자.
개인 수준의 위험비는 0.5 로 못 박혀 있다.

그런데 사람마다 타고난 위험이 다르다. 그 보이지 않는 차이를 허약성
(frailty) Z 라 하고 평균 1 인 감마분포로 두자.

  h(t | 치료, Z) = Z * lambda * 0.5,   h(t | 대조, Z) = Z * lambda

시간이 흐르면 양쪽 집단에서 허약한 사람부터 먼저 죽는다. 그런데
치료군에서는 그 속도가 절반이다. 그래서 **치료군에는 허약한 사람이
더 많이 남는다.** 시점 t 에서 비교되는 두 집단은 더 이상 같은 사람들이
아니다 — 무작위배정이 보장한 것은 첫날의 동질성뿐이다.

관측되는 주변 위험비는 이렇게 끌려간다 (감마 허약성, 분산 theta).

      t    theta=0   theta=0.5   theta=1   theta=2
      0      0.500      0.500     0.500     0.500
     10      0.500      0.600     0.667     0.750
     40      0.500      0.750     0.833     0.900
    160      0.500      0.900     0.944     0.971

개인 수준 효과는 조금도 변하지 않았는데 관측값이 0.50 에서 0.94 로
간다. 모의실험으로도 확인했다(2 백만 명, theta=1): t=0~5 에서 0.601,
t=35~45 에서 0.845, t=150~170 에서 0.958.

만드는 파일:

  ch21/cox/img/frailty_hr.png

실행:  python3 scripts/make_ch21_frailty_hr.py   (저장소 최상위에서)
필요:  numpy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch21/cox/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"
PURPLE = "#6A1B9A"

LAM, HR0, SEED, NSIM = 0.10, 0.50, 2010, 2_000_000
THETAS = ((0.0, BLUE, "$\\theta = 0$ — 모두가 똑같이 타고났다"),
          (0.5, TEAL, "$\\theta = 0.5$"),
          (1.0, AMBER, "$\\theta = 1$"),
          (2.0, PURPLE, "$\\theta = 2$ — 타고난 차이가 크다"))
TH_SHOW = 1.0
BANDS = ((0, 5), (8, 13), (18, 23), (35, 45), (75, 85), (150, 170))


def hr_marginal(t, th):
    if th == 0:
        return np.full_like(np.asarray(t, float), HR0)
    return HR0 * (1 + th * LAM * t) / (1 + th * LAM * HR0 * t)


def mean_frailty(t, th, treated):
    h = LAM * (HR0 if treated else 1.0)
    return 1.0 / (1 + th * h * t)


def main():
    print("  관측되는 주변 위험비 (개인 수준은 어느 시점에서나 0.5)")
    print(f"{'t':>6}" + "".join(f"{f'theta={th}':>12}" for th, _, _ in THETAS))
    for t in (0, 5, 10, 20, 40, 80, 160):
        print(f"{t:>6}" + "".join(f"{float(hr_marginal(t, th)):>12.3f}"
                                  for th, _, _ in THETAS))

    rng = np.random.default_rng(SEED)
    z = rng.gamma(1 / TH_SHOW, TH_SHOW, NSIM)
    x = rng.integers(0, 2, NSIM)
    T = rng.exponential(1 / (z * LAM * HR0 ** x))
    print(f"\n  모의실험 {NSIM//10**6} 백만 명 (theta={TH_SHOW})")
    mids, sim = [], []
    for a, b in BANDS:
        def hz(g):
            at = ((T >= a) & (x == g)).sum()
            ev = ((T >= a) & (T < b) & (x == g)).sum()
            return ev / at / (b - a)
        r = hz(1) / hz(0)
        mids.append((a + b) / 2); sim.append(r)
        print(f"    t = {a:>3}~{b:<3}  비 {r:.3f}   (식 {float(hr_marginal((a+b)/2, TH_SHOW)):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1), constrained_layout=True)

    # ── 왼쪽: 관측되는 위험비가 1 로 끌려간다 ──
    ax = axes[0]
    ts = np.linspace(0, 170, 400)
    for th, col, lab in THETAS:
        ax.plot(ts, hr_marginal(ts, th), lw=2.9, color=col, label=lab)
    ax.axhline(1.0, color=GREY, lw=1.6, ls=":")
    ax.text(166, 1.012, "효과 없음", fontsize=10.8, color=GREY, ha="right")
    ax.plot(mids, sim, lw=0, marker="o", ms=10, mfc="white", mec=AMBER,
            mew=2.2, zorder=5, label=f"모의실험 ($\\theta = {TH_SHOW:g}$)")
    ax.annotate("개인 수준 위험비는\n어느 시점에서나 $0.5$ 다",
                xy=(55, HR0), xytext=(8, 0.40), fontsize=10.8, color=BLUE,
                ha="left", arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.4))
    ax.annotate(f"$t = 160$ 에서 관측값이 "
                f"{float(hr_marginal(160, 1.0)):.2f} —\n효과가 거의 사라진 것처럼 보인다",
                xy=(160, float(hr_marginal(160, 1.0))), xytext=(92, 0.70),
                fontsize=10.6, color=AMBER, ha="center",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.4))
    ax.set_xlabel("배정 뒤 지난 시간", fontsize=11.5)
    ax.set_ylabel("관측되는 위험비", fontsize=11.5)
    ax.set_title("아무것도 변하지 않았는데 위험비는 변한다",
                 fontsize=12.6, color=INK, pad=8)
    ax.set_xlim(0, 170)
    ax.set_ylim(0.33, 1.07)
    ax.legend(fontsize=10.0, framealpha=0.93, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 살아남은 사람이 달라진다 ──
    ax = axes[1]
    ft = mean_frailty(ts, TH_SHOW, True)
    fc = mean_frailty(ts, TH_SHOW, False)
    ax.plot(ts, ft, lw=3.0, color=TEAL, label="치료군에 남은 사람의 평균 허약도")
    ax.plot(ts, fc, lw=3.0, color=RED, label="대조군에 남은 사람의 평균 허약도")
    ax.fill_between(ts, fc, ft, color=AMBER, alpha=0.18)
    ax.axhline(1.0, color=INK, lw=1.8, ls="--")
    ax.text(166, 1.02, "배정하던 날 — 두 집단이 같다", fontsize=10.8,
            color=INK, ha="right")
    for t in (20, 80):
        ax.annotate("", xy=(t, mean_frailty(t, TH_SHOW, True)),
                    xytext=(t, mean_frailty(t, TH_SHOW, False)),
                    arrowprops=dict(arrowstyle="<->", color=AMBER, lw=1.8))
        ax.text(t + 6, mean_frailty(t, TH_SHOW, True) + 0.14,
                f"$t={t}$ 에서\n{mean_frailty(t, TH_SHOW, True):.2f} 대 "
                f"{mean_frailty(t, TH_SHOW, False):.2f}",
                fontsize=10.4, color=AMBER, va="bottom")
    ax.set_xlabel("배정 뒤 지난 시간", fontsize=11.5)
    ax.set_ylabel("살아남은 사람의 평균 허약도", fontsize=11.5)
    ax.set_title("치료가 허약한 사람을 살려 둔다", fontsize=12.6, color=INK,
                 pad=8)
    ax.set_xlim(0, 170)
    ax.set_ylim(0, 1.14)
    ax.legend(fontsize=10.4, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("무작위배정이 보장하는 것은 첫날의 동질성뿐이다", fontsize=14.5)
    fig.text(0.5, -0.05,
             "교란도 없고 비례위험 위배도 없다. 모든 사람의 위험이 어느 "
             "순간에나 정확히 절반으로 낮아진다. 그런데도 시점마다 비교되는 "
             "두 집단은 서로 다른 사람들이 되어 간다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "frailty_hr.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
