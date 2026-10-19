r"""10.3절 「그림으로 보는 통계학」 — 닳은 쪽은 앞쪽이었다.

1881 년에 천문학자 사이먼 뉴컴이 도서관의 로그표를 보다가 이상한 것을
알아챘다. 앞쪽 책장이 뒤쪽보다 훨씬 더 닳아 있었다. 1 로 시작하는 수를
찾아보는 일이 9 로 시작하는 수를 찾아보는 일보다 많다는 뜻이다.
그는 첫 자리가 d 일 확률을

  P(첫 자리 = d) = log10(1 + 1/d)

이라고 적었다. 1 이 30.1%, 9 가 4.6% 다.

이 글은 잊혔다. 1938 년에 제너럴 일렉트릭의 물리학자 프랭크 벤포드가
같은 것을 다시 찾아냈고, 이번에는 스무 가지 출처에서 2 만 개가 넘는
수를 모아 맞춰 보았다 — 강의 유역 면적, 인구, 물리상수, 신문에 실린
수, 사망률, 집 주소.

언제 맞는가. 자료가 여러 자릿수에 걸쳐 퍼져 있을 때다. 로그정규의
log10 표준편차 s 로 재면

  s = 0.12  벤포드와의 거리 0.44   (전혀 아니다)
  s = 0.2                 0.277
  s = 0.3                 0.103
  s = 0.5                 0.005  (사실상 맞는다)
  s = 1.0                 0.002

n = 1000 으로 적합도검정을 하면 s = 0.3 짜리 멀쩡한 자료의
카이제곱이 50.2 (df=8, p = 3.7e-08) 로 보란 듯이 기각된다.

반 자릿수만 퍼져도 맞고, 오분의 일 자릿수면 안 맞는다. 그래서 「벤포드로
부정을 잡는다」는 말은 자료가 어떻게 퍼져 있는지를 먼저 보지 않으면
쓸 수 없다. 개표소별 득표수처럼 좁게 모인 수에 들이대면 멀쩡한 자료가
걸린다.

만드는 파일:

  ch10/test/img/benford.png

실행:  python3 scripts/make_ch10_benford.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import numpy as np
from scipy.stats import chi2 as chi2dist

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch10/test/img/"

INK = "#37474F"
MUTED = "#90A4AE"
GREY = "#78909C"
RED = "#D32F2F"
TEAL = "#00838F"
AMBER = "#B07C2C"
BLUE = "#1565C0"

D = np.arange(1, 10)
BEN = np.log10(1 + 1 / D)
N, SEED = 400_000, 1938
SS = np.array([0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8,
               1.0, 1.4, 2.0, 3.0])
NGOF = 1000                      # 적합도검정에 쓸 표본 크기


def first(x):
    x = np.abs(x)
    x = x[x > 0]
    d = (x / 10.0 ** np.floor(np.log10(x))).astype(int)
    return np.bincount(d, minlength=10)[1:10] / len(x)


def main():
    rng = np.random.default_rng(SEED)
    cases = (("로그정규 — $s = 1.0$ (한 자릿수)", 10 ** rng.normal(3, 1.0, N),
              TEAL),
             ("로그정규 — $s = 0.3$ (삼분의 일)", 10 ** rng.normal(3, 0.3, N),
              AMBER),
             ("균등 $100$~$999$", rng.uniform(100, 999, N), RED))
    props = [first(v) for _, v, _ in cases]

    print("  첫 자리의 비율")
    print("        " + "".join(f"{d:>8}" for d in D))
    print(f"{'벤포드':>22}" + "".join(f"{p:>8.3f}" for p in BEN))
    for (nm, _, _), p in zip(cases, props):
        tv = 0.5 * np.abs(p - BEN).sum()
        g = NGOF * ((p - BEN) ** 2 / BEN).sum()
        print(f"{nm.replace('$','').replace('~','-'):>22}"
              + "".join(f"{q:>8.3f}" for q in p)
              + f"   거리 {tv:.4f}   n={NGOF} 의 카이제곱 {g:>8.1f}"
                f"  p={chi2dist.sf(g, 8):.3g}")

    tvs = np.array([0.5 * np.abs(first(10 ** rng.normal(3, s, N)) - BEN).sum()
                    for s in SS])
    print(f"\n  퍼진 폭 s 에 따른 벤포드와의 거리")
    for s, t in zip(SS, tvs):
        print(f"    s = {s:>5}   {t:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), constrained_layout=True)

    # ── 왼쪽: 1 이 30%, 9 가 5% ──
    ax = axes[0]
    w = 0.26
    for k, ((nm, _, col), p) in enumerate(zip(cases, props)):
        ax.bar(D + (k - 1) * w, p, width=w, color=col, alpha=0.88, label=nm,
               zorder=3)
    ax.plot(D, BEN, lw=0, marker="_", ms=34, mew=3.0, color=INK, zorder=5,
            label="벤포드 $\\log_{10}(1 + 1/d)$")
    ax.set_xticks(D)
    ax.set_xlabel("첫 자리 $d$", fontsize=11.5)
    ax.set_ylabel("그 자리로 시작할 비율", fontsize=11.5)
    ax.set_title("$1$ 이 $30\\%$, $9$ 가 $5\\%$", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.set_yticklabels(["$0$", "$0.1$", "$0.2$", "$0.3$", "$0.4$"])
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.grid(alpha=0.25, axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    # ── 오른쪽: 얼마나 퍼져 있어야 쓸 수 있는가 ──
    ax = axes[1]
    ax.axvspan(SS[0], 0.33, color=RED, alpha=0.09)
    ax.plot(SS, tvs, lw=3.6, color=INK, marker="o", ms=7)
    for s, col, lab in ((1.0, TEAL, None), (0.3, AMBER, None)):
        i = int(np.argmin(abs(SS - s)))
        ax.plot([SS[i]], [tvs[i]], marker="o", ms=14, color=col, zorder=5,
                mec="white", mew=1.8)
    ax.annotate("개표소별 득표수처럼\n좁게 모인 수가 여기 있다",
                xy=(0.19, tvs[int(np.argmin(abs(SS - 0.19)))]),
                xytext=(0.52, 0.47), fontsize=10.8, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("반 자릿수만 퍼져도\n사실상 벤포드다",
                xy=(0.65, tvs[int(np.argmin(abs(SS - 0.65)))]),
                xytext=(1.55, 0.0135), fontsize=10.8, color=TEAL, ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1, 2, 3])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1", "2", "3"])
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["$0.001$", "$0.01$", "$0.1$", "$1$"])
    ax.minorticks_off()
    ax.set_xlabel("자료가 퍼진 폭 — 자릿수로 센 표준편차 $s$", fontsize=11.5)
    ax.set_ylabel("벤포드와의 거리", fontsize=11.5)
    ax.set_title("얼마나 퍼져 있어야 쓸 수 있는가", fontsize=12.8, color=INK,
                 pad=8)
    ax.set_ylim(5e-4, 1.3)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("닳은 쪽은 앞쪽이었다", fontsize=14.5)
    fig.text(0.5, -0.06,
             "거리는 두 분포의 총변동거리이며 $0$ 이면 완전히 같다. "
             f"각 경우에 수를 ${N//1000}{{,}}000$ 개씩 뽑았다. 붉은 구간은 "
             "자료가 너무 좁게 모여 벤포드를 기준으로 삼을 수 없는 자리다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "benford.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
