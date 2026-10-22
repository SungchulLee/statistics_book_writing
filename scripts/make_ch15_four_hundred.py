r"""15.3절 「그림으로 보는 통계학」 — 4 할 타자가 사라진 자리.

$1941$ 년 테드 윌리엄스의 $.406$ 뒤로 아무도 $4$ 할을 치지 못했다. 스티븐 제이
굴드는 『Full House』($1996$)에서 그것이 타자가 나빠진 증거가 아니라 **흩어짐이
줄어든 증거**라고 했다. 리그 평균 타율은 백 년 동안 $.27$–$.29$ 에 머물렀다.

이 스크립트는 레이먼 야구자료(1871-2016)에서 **한 해 400 타수 이상**을 친 타자만
골라 시즌마다

  평균 타율 · 최고 타율 · 타율의 표준편차

를 세고, 그 표준편차를 둘로 가른다. 타율은 타석 500 번의 비율이므로 선수의 실력이
완전히 같아도 이항 요동이 남는다.

  Var(관측 타율) = Var(실력) + E[p(1-p)/AB]
                   \_______/   \___________/
                     실력          운

오른쪽 항은 시대가 달라도 거의 변하지 않는다(표준편차로 .0196-.0204). 줄어든 것은
왼쪽 항이다. 그래서 **두 시대의 관측 분산비(F)는 실력 분산비를 1 쪽으로 끌어내린
값**이다 — 회귀의 희석과 같은 구조다.

만드는 파일:

  ch15/f_test/img/four_hundred.png

실행:  python3 scripts/make_ch15_four_hundred.py   (저장소 최상위에서)
필요:  pandas, numpy, scipy, matplotlib + 인터넷(자료를 내려받는다).
       문서 빌드에는 필요하지 않다. PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch15/f_test/img/"

# 레이먼 야구자료(Lahman's Baseball Database)의 타격 기록 — 1871-2016
URL = ("https://raw.githubusercontent.com/ohumbe/analyzing_baseball_data/"
       "master/Batting.csv")

INK = "#37474F"
AMBER = "#B07C2C"
TEAL = "#00838F"
RED = "#D32F2F"
BLUE = "#1565C0"
MUTED = "#90A4AE"

MIN_AB = 400       # 자격 기준 — 한 해 400 타수
MIN_N = 30         # 자격 타자가 이보다 적은 해는 버린다(초기 리그·1994 파업)
SMOOTH = 9         # 이동평균 창
ERA_A = (1901, 1920)
ERA_B = (1997, 2016)


def load():
    """시즌 × 선수 타율 — 한 해에 여러 팀을 뛰었으면 합산한다."""
    b = pd.read_csv(URL)
    b = b[b["lgID"].isin(["AL", "NL"])]
    g = b.groupby(["yearID", "playerID"])[["AB", "H"]].sum().reset_index()
    g = g[g["AB"] >= MIN_AB].copy()
    g["BA"] = g["H"] / g["AB"]
    g["luck"] = g["BA"] * (1 - g["BA"]) / g["AB"]     # 이항 분산
    keep = g.groupby("yearID")["BA"].size()
    return g[g["yearID"].isin(keep[keep >= MIN_N].index)]


def smooth(v, k=SMOOTH):
    return pd.Series(v).rolling(k, center=True, min_periods=max(3, k // 2)).mean()


def main():
    g = load()
    s = g.groupby("yearID").agg(n=("BA", "size"), mean=("BA", "mean"),
                                obs=("BA", lambda v: v.std(ddof=1)),
                                mx=("BA", "max"), luck=("luck", "mean"))
    s["bino"] = np.sqrt(s["luck"])
    s["skill"] = np.sqrt(np.maximum(s["obs"] ** 2 - s["luck"], 0))
    yr = s.index.values
    print(f"시즌 {yr[0]}-{yr[-1]} · 자격 타자 {int(s['n'].sum()):,} 명")

    # 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
    print("\n     연대      선수    평균    관측      운    실력")
    for d0 in range(1880, 2017, 20):
        w = g[(g["yearID"] >= d0) & (g["yearID"] < d0 + 20)]
        if len(w) < 100:
            continue
        r = w["BA"] - w.groupby("yearID")["BA"].transform("mean")
        obs, bino = r.std(ddof=1), np.sqrt(w["luck"].mean())
        # 자료가 없는 해가 있으므로 칸 이름에 실제 연도 범위를 적는다
        lab = f"{int(w['yearID'].min())}-{int(w['yearID'].max())}"
        print(f"{lab:>9} {len(w):>8,} {w['BA'].mean():7.4f} {obs:7.4f} "
              f"{bino:7.4f} {np.sqrt(max(obs**2 - bino**2, 0)):7.4f}")

    # 두 시대의 분산 비교 — 시즌평균을 뺀 잔차를 모은다
    def era(lo, hi):
        w = g[(g["yearID"] >= lo) & (g["yearID"] <= hi)]
        r = (w["BA"] - w.groupby("yearID")["BA"].transform("mean")).values
        return r, float(w["luck"].mean())

    ra, la = era(*ERA_A)
    rb, lb = era(*ERA_B)
    va, vb = ra.var(ddof=1), rb.var(ddof=1)
    F = va / vb
    da, db = len(ra) - 1, len(rb) - 1
    p = 2 * min(stats.f.sf(F, da, db), stats.f.cdf(F, da, db))
    print(f"\n{ERA_A}  n = {len(ra):,}  sd = {np.sqrt(va):.4f}  "
          f"운 = {np.sqrt(la):.4f}  실력 = {np.sqrt(va-la):.4f}")
    print(f"{ERA_B}  n = {len(rb):,}  sd = {np.sqrt(vb):.4f}  "
          f"운 = {np.sqrt(lb):.4f}  실력 = {np.sqrt(vb-lb):.4f}")
    print(f"관측 F = {F:.3f}  (자유도 {da}, {db})   p = {p:.2e}")
    print(f"실력만의 분산비 = {(va-la)/(vb-lb):.3f}")
    print(f"첨도 = {stats.kurtosis(ra, fisher=False):.2f}, "
          f"{stats.kurtosis(rb, fisher=False):.2f}   "
          f"브라운-포사이드 p = {stats.levene(ra, rb, center='median').pvalue:.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), constrained_layout=True)

    # ── ① 평균은 그대로, 꼭대기는 내려온다 ──
    ax = axes[0]
    ax.plot(yr, s["mx"], lw=0.9, color=MUTED, alpha=0.8)
    ax.plot(yr, smooth(s["mx"].values), lw=2.8, color=AMBER, label="시즌 최고 타율")
    ax.plot(yr, s["mean"], lw=0.9, color=MUTED, alpha=0.8)
    ax.plot(yr, smooth(s["mean"].values), lw=2.8, color=TEAL, label="자격 타자 평균")
    ax.axhline(0.400, color=RED, lw=1.6, ls="--")
    ax.text(2000, 0.403, "$.400$", color=RED, fontsize=11.5)
    for y, nm in [(1941, "윌리엄스 $.406$"), (1980, "브렛 $.390$"),
                  (1994, "그윈 $.394$")]:
        if y in s.index:
            ax.plot([y], [s.loc[y, "mx"]], "o", ms=7, color=RED, zorder=5)
    ax.annotate("$1941$ 윌리엄스 $.406$ — 마지막 $4$ 할",
                xy=(1941, s.loc[1941, "mx"]), xytext=(1944, 0.437),
                fontsize=10.8, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
    ax.annotate("그 뒤 가장 가까웠던 둘 —\n$1980$ 브렛 $.390$ · $1994$ 그윈 $.394$",
                xy=(1980, 0.3880), xytext=(1934, 0.3125), fontsize=10.2, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.set_xlabel("시즌  (굵은 선은 $9$ 년 이동평균)", fontsize=11.5)
    ax.set_ylabel("타율", fontsize=11.5)
    ax.set_title("① 평균은 그대로인데 꼭대기가 내려온다", fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.5, framealpha=0.93, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    # ── ② 줄어든 것은 실력의 흩어짐 ──
    ax = axes[1]
    ax.plot(yr, s["obs"], lw=0.8, color=MUTED, alpha=0.7)
    ax.plot(yr, smooth(s["obs"].values), lw=2.8, color=AMBER,
            label="관측 타율의 표준편차")
    ax.plot(yr, smooth(s["bino"].values), lw=2.6, color=BLUE,
            label="그 가운데 운 — $\\sqrt{E[p(1-p)/AB]}$")
    ax.plot(yr, smooth(s["skill"].values), lw=2.8, color=TEAL,
            label="남는 것 — 실력의 표준편차")
    ax.fill_between(yr, smooth(s["skill"].values), smooth(s["obs"].values),
                    color=AMBER, alpha=0.10)
    ax.annotate(f"$.0370 \\to .0270$", xy=(1995, 0.0272), xytext=(1958, 0.0335),
                fontsize=11, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))
    ax.annotate(f"$.0309 \\to .0185$", xy=(1995, 0.0187), xytext=(1955, 0.0135),
                fontsize=11, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.3))
    ax.set_ylim(0.010, 0.046)
    ax.set_xlabel("시즌  (굵은 선은 $9$ 년 이동평균)", fontsize=11.5)
    ax.set_ylabel("타율의 표준편차", fontsize=11.5)
    ax.set_title("② 내려온 것은 흩어짐이고, 운의 몫은 그대로다",
                 fontsize=12.8, color=INK)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=10.2, framealpha=0.93, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("$4$ 할이 사라진 것은 평균이 아니라 폭의 일이다", fontsize=14.5)
    fig.text(0.5, -0.045,
             f"한 해 {MIN_AB} 타수 이상을 친 타자만 센다. 자격 타자가 "
             f"{MIN_N} 명에 못 미치는 네 해(초기 리그 셋과 $1981$ 년 파업 "
             "시즌)는 뺐다. $1994$ 년도 파업으로 짧았다.",
             ha="center", fontsize=10.4, color=INK)

    path = OUT + "four_hundred.png"
    fig.savefig(path, dpi=165, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
