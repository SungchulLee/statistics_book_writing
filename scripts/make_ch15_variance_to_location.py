r"""15.5절 「그림으로 보는 통계학」 — 산포를 묻는 일을 위치를 묻는 일로.

로버스트 분산 검정 셋(르빈·브라운포사이드·플리그너킬린)은 겉보기에 서로
다른 검정 같지만 **한 가지 수법**의 변주다.

  1 단계  관측값을 「제 집단의 중심에서 얼마나 떨어졌나」로 바꾼다
  2 단계  그렇게 만든 값에 보통의 일원분산분석을 돌린다

2 단계가 **위치 검정**이라는 것이 핵심이다. 분산 검정이 비정규성에
무너지던 까닭은 카이제곱·F 가 4 차 적률에 매여 있기 때문인데, 위치
문제로 바꿔 놓으면 중심극한정리의 보호 아래로 돌아온다.

세 검정은 1 단계만 다르다.

  르빈           |x - 집단평균|
  브라운포사이드   |x - 집단중앙값|      <- 평균을 중앙값으로 바꾼 것뿐
  플리그너킬린    |x - 집단중앙값| 의 순위를 정규점수로

네 검정의 구현은 scipy 의 bartlett · levene(center=...) · fligner 와
p 값이 소수점 아래 끝자리까지 같은 것을 확인했다.

만드는 파일:

  ch15/robust_tests/img/variance_to_location.png

실행:  python3 scripts/make_ch15_variance_to_location.py   (저장소 최상위에서)
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

OUT = "docs/ch15/robust_tests/img/"

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

G, M, REPS, SEED = 3, 25, 20_000, 1960
N = G * M
KINDS = ["균등", "정규", "t(4)", "지수", "로그정규"]


def draw(kind, size, rng):
    """평균 0 · 분산 1 로 맞춘 표본 — 집단 사이에 분산 차이가 없게 한다."""
    if kind == "정규":
        return rng.normal(0, 1, size)
    if kind == "균등":
        return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
    if kind == "t(4)":
        return rng.standard_t(4, size) / np.sqrt(2)
    if kind == "지수":
        return rng.exponential(1, size) - 1
    return (np.exp(rng.normal(0, 1, size)) - np.exp(0.5)) / np.sqrt((np.e - 1) * np.e)


def anova_p(z):
    """z : (REPS, G, M) — 일원분산분석의 p 값."""
    gm = z.mean(axis=(1, 2), keepdims=True)
    gi = z.mean(axis=2, keepdims=True)
    ssb = (M * ((gi - gm) ** 2).sum(axis=1)).ravel()
    ssw = ((z - gi) ** 2).sum(axis=(1, 2))
    return stats.f.sf((ssb / (G - 1)) / (ssw / (N - G)), G - 1, N - G)


def all_tests(x):
    """네 검정의 p 값. scipy 와 끝자리까지 같은 것을 확인했다."""
    s2 = x.var(axis=2, ddof=1)
    d = M - 1
    sp2 = (d * s2).sum(axis=1) / (N - G)
    T = (N - G) * np.log(sp2) - (d * np.log(s2)).sum(axis=1)
    C = 1 + (G / d - 1 / (N - G)) / (3 * (G - 1))
    p_bart = stats.chi2.sf(T / C, G - 1)

    p_lev = anova_p(np.abs(x - x.mean(axis=2, keepdims=True)))
    p_bf = anova_p(np.abs(x - np.median(x, axis=2, keepdims=True)))

    a = np.abs(x - np.median(x, axis=2, keepdims=True)).reshape(len(x), -1)
    r = np.argsort(np.argsort(a, axis=1), axis=1) + 1.0
    sc = stats.norm.ppf(0.5 + r / (2 * (N + 1)))
    g = sc.reshape(len(x), G, M)
    X2 = (M * ((g.mean(axis=2) - sc.mean(axis=1, keepdims=True)) ** 2)).sum(axis=1) \
        / sc.var(axis=1, ddof=1)
    p_fk = stats.chi2.sf(X2, G - 1)
    return p_bart, p_lev, p_bf, p_fk


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'모집단':9}{'바틀릿':>9}{'르빈(평균)':>11}{'BF(중앙값)':>11}{'FK(순위)':>10}")
    for kind in KINDS:
        x = draw(kind, (REPS, G, M), rng)
        ps = all_tests(x)
        rows.append([float((p < 0.05).mean()) for p in ps])
        print(f"{kind:9}" + "".join(f"{v:>11.4f}" for v in rows[-1]))
    rows = np.array(rows)

    fig, ax = plt.subplots(figsize=(9.8, 5.2), constrained_layout=True)

    # ── 실제 1 종 오류율 ──
    xs_ = np.arange(len(KINDS))
    for j, (nm, col, mk) in enumerate([("바틀릿", RED, "o"),
                                       ("르빈 — 평균", AMBER, "s"),
                                       ("브라운포사이드 — 중앙값", TEAL, "D"),
                                       ("플리그너킬린 — 순위", BLUE, "^")]):
        ax.plot(xs_, rows[:, j] * 100, marker=mk, ms=8, lw=2.6, color=col, label=nm)
    ax.axhline(5, color=MUTED, lw=2.0, ls="--")
    ax.annotate(f"{rows[-1,0]*100:.1f}%", xy=(4, rows[-1, 0] * 100),
                xytext=(3.1, rows[-1, 0] * 100 - 4), fontsize=11.5, color=RED)
    ax.annotate(f"{rows[-1,2]*100:.1f}%", xy=(4, rows[-1, 2] * 100),
                xytext=(3.3, 14), fontsize=11.5, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.4))
    ax.set_xticks(xs_)
    ax.set_xticklabels(KINDS, fontsize=11)
    ax.set_xlabel("모집단 — 오른쪽으로 갈수록 정규에서 멀다", fontsize=11.5)
    ax.set_ylabel("명목 5% 의 실제 1 종 오류율 (%)", fontsize=11.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10, framealpha=0.93, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(f"한 글자씩 바꾸었을 뿐인데 ${rows[-1,0]*100:.1f}\\%$ 가 "
                 f"${rows[-1,2]*100:.1f}\\%$ 가 된다", fontsize=14.5)
    ax.set_title(f"분산은 세 집단 모두 같게 두었다  ({REPS//1000} 천 번)",
                 fontsize=12.8, color=INK, pad=8)
    fig.text(0.5, -0.05,
             f"집단 {G} 개에 각 {M} 개씩, 세 집단의 분산을 모두 같게 두었으므로 "
             "$H_0$ 가 참이다. 네 검정 모두 scipy 의 것과 $p$ 값이 끝자리까지 "
             "같은 것을 확인했다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "variance_to_location.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
