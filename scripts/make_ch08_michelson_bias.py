r"""구간은 흔들림만 말하고 치우침은 말하지 않는다 — 마이컬슨의 1879 년 광속.

이 쪽은 $\bar X \pm t_{n-1,\,0.975}\, S/\sqrt{n}$ 을 세우고 그 포함확률을 모의로 잰다.
모의에서는 참값을 우리가 정해 주므로 구간이 약속대로 $95\%$ 를 담는다. 현실의 측정에서는
**같은 장치가 모든 관측에 같은 치우침을 싣는다.** 그러면 구간은 좁아지면서 과녁을 비껴간다.

자료: 마이컬슨이 1879 년 6 월 5 일부터 7 월 2 일까지 공기 중 광속을 잰 100 회.
실험 다섯 번 × 각 20 회. 값은 km/s 에서 299000 을 뺀 것이다.
출처는 Stigler (1977) *Do robust estimators work with real data?* Annals of Statistics 5,
1055–1098 의 Table 6 이고, R 의 `datasets::morley` 와 `MASS::michelson` 이 같은 자료다.
MASS 문서가 **「이 측정 척도에서 현재 받아들여지는 값은 734.5」**라 적어 둔다 — 공기 중
광속이므로 진공값 299792.458 이 아니다.

① 100 회의 $95\%$ 구간은 반폭 15.7 인데 참값은 117.9 (= 14.9 SE) 아래에 있다.
② 실험 다섯의 구간 가운데 참값을 담는 것이 **하나도 없다.** 치우침이 실험마다 다르다.
③ $n$ 을 키우면 반폭만 $1/\sqrt{n}$ 로 줄고 치우침은 꼼짝하지 않는다.

만드는 파일: docs/ch08/one_sample_intervals/img/michelson_bias.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

# === 자료 (Stigler 1977 Table 6; km/s - 299000) =======================
RUNS = [
    # 실험 1
    [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880,
     1000, 980, 930, 650, 760, 810, 1000, 1000, 960, 960],
    # 실험 2
    [960, 940, 960, 940, 880, 800, 850, 880, 900, 840,
     830, 790, 810, 880, 880, 830, 800, 790, 760, 800],
    # 실험 3
    [880, 880, 880, 860, 720, 720, 620, 860, 970, 950,
     880, 910, 850, 870, 840, 840, 850, 840, 840, 840],
    # 실험 4
    [890, 810, 810, 820, 800, 770, 760, 740, 750, 760,
     910, 920, 890, 860, 880, 720, 840, 850, 850, 780],
    # 실험 5
    [890, 840, 780, 810, 760, 810, 790, 810, 820, 850,
     870, 870, 810, 740, 810, 940, 950, 800, 810, 870],
]
BASE = 299000.0
TRUE = 734.5        # MASS 문서: 이 척도에서 현재 받아들여지는 공기 중 광속
x = np.array([v for g in RUNS for v in g], float)


def ci(v, conf=0.95):
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * se
    return v.mean(), h, se


# === ① 전체 100 회 =================================================
m, hw, se = ci(x)
print("① 100 회 전체")
print(f"   n = {len(x)}, 평균 = {BASE + m:.1f}, 표준편차 = {x.std(ddof=1):.1f}, "
      f"SE = {se:.2f}")
print(f"   95% 구간 = [{BASE + m - hw:.1f}, {BASE + m + hw:.1f}]  반폭 {hw:.1f}")
print(f"   받아들여지는 값 {BASE + TRUE:.1f} 을 담는가: "
      f"{'예' if abs(m - TRUE) <= hw else '아니다'}")
print(f"   치우침 = {m - TRUE:+.1f} km/s = {(m - TRUE) / se:.1f} SE "
      f"= 반폭의 {(m - TRUE) / hw:.1f} 배")

# === ② 실험 다섯 =====================================================
print("\n② 실험 다섯 번을 따로 보면")
# 한글은 고정폭에서 두 칸을 차지하므로 f-string 자리맞춤이 보이는 열과 어긋난다
print("   실험     평균        95% 구간          담는가")
MS, HS = [], []
for i, g in enumerate(RUNS, 1):
    v = np.array(g, float)
    mi, hi_, _ = ci(v)
    MS.append(mi)
    HS.append(hi_)
    ok = "예" if abs(mi - TRUE) <= hi_ else "아니다"
    print(f"   {i:>2}    {BASE + mi:>9.1f}   "
          f"[{BASE + mi - hi_:>9.1f}, {BASE + mi + hi_:>9.1f}]   {ok}")
cover = sum(1 for mi, hi_ in zip(MS, HS) if abs(mi - TRUE) <= hi_)
F = stats.f_oneway(*[np.array(g, float) for g in RUNS])
print(f"   참값을 담는 구간: {cover} / 5")
print(f"   다섯 평균의 폭 = {max(MS) - min(MS):.1f} km/s")
print(f"   실험 간 일원분산분석 F = {F.statistic:.3f}, p = {F.pvalue:.6f}")
print("   **치우침이 실험마다 다르다는 것이 그의 자료 안에서 이미 보인다.**")

# === ③ n 을 키우면 ==================================================
NS = np.array([10, 30, 100, 300, 1000, 3000, 10000], float)
s_all = x.std(ddof=1)
HWN = stats.t.ppf(0.975, NS - 1) * s_all / np.sqrt(NS)
print("\n③ 같은 장치로 n 만 키우면")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("        n     반폭    치우침")
for nn, hh in zip(NS, HWN):
    print(f"   {int(nn):>6}   {hh:>6.1f}   {m - TRUE:>6.1f}")
print("   반폭은 1/sqrt(n) 로 줄고 치우침은 상수다. 둘이 만나는 자리는 없다.")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_M, C_T, C_B = "#1f77b4", "#2ca02c", "#d62728"

# ① 100 회와 구간
ax1.plot(np.arange(1, 101), BASE + x, "o", ms=4.5, color="#9ecae1",
         label="100 회의 측정값")
ax1.axhspan(BASE + m - hw, BASE + m + hw, color=C_M, alpha=0.25,
            label="평균의 $95\\%$ 구간")
ax1.axhline(BASE + m, color=C_M, lw=2.2, label=f"평균 ${BASE + m:,.1f}$"
            .replace(",", "{,}"))
ax1.axhline(BASE + TRUE, color=C_T, lw=2.4, ls="--",
            label=f"받아들여지는 값 ${BASE + TRUE:,.1f}$".replace(",", "{,}"))
ax1.annotate("", xy=(52, BASE + TRUE), xytext=(52, BASE + m),
             arrowprops=dict(arrowstyle="<->", color=C_B, lw=2.0))
ax1.text(55, BASE + (m + TRUE) / 2, f"치우침 ${m - TRUE:.1f}$\n$= 14.9$ SE",
         color=C_B, fontsize=11, va="center")
ax1.set_xlabel("측정 순서")
ax1.set_ylabel("공기 중 광속 (km/s)")
ax1.set_title("① 좁은데 비껴갔다", fontsize=12)
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(alpha=0.25)

# ② 실험 다섯
ys = np.arange(5)
for i in range(5):
    ax2.plot([BASE + MS[i] - HS[i], BASE + MS[i] + HS[i]], [ys[i], ys[i]],
             color=C_B, lw=4, alpha=0.75, solid_capstyle="butt")
    ax2.plot([BASE + MS[i]], [ys[i]], "o", color="#333333", ms=8, zorder=5)
ax2.axvline(BASE + TRUE, color=C_T, lw=2.4, ls="--")
ax2.text(BASE + TRUE - 6, 4.42, "받아들여지는 값", color=C_T, fontsize=10.5,
         ha="right", va="top")
ax2.set_yticks(ys)
ax2.set_yticklabels([f"실험 {i}" for i in range(1, 6)], fontsize=11)
ax2.set_xlabel("공기 중 광속 (km/s)")
ax2.set_ylim(-0.7, 4.7)
ax2.invert_yaxis()
ax2.set_title(f"② 다섯 구간이 모두 빗나간다 ($F = {F.statistic:.2f}$, "
              f"$p = {F.pvalue:.3f}$)", fontsize=12)
ax2.grid(alpha=0.25, axis="x")

# ③ n 을 키우면
ax3.loglog(NS, HWN, "o-", color=C_M, lw=2.6, ms=9,
           label="구간 반폭 ($\\propto 1/\\sqrt{n}$)")
ax3.axhline(m - TRUE, color=C_B, lw=2.6, ls="--",
            label=f"치우침 ${m - TRUE:.1f}$ (상수)")
ax3.plot([100], [hw], "s", color="#333333", ms=12, zorder=5)
ax3.annotate("마이컬슨이 선 자리\n$n = 100$", (100, hw),
             xytext=(0.9, 1.1), textcoords="offset fontsize",
             fontsize=10, color="#333333")
ax3.set_xlabel("측정 횟수 $n$")
ax3.set_ylabel("km/s")
ax3.set_xticks([10, 100, 1000, 10000])
ax3.set_xticklabels(["$10$", "$10^{2}$", "$10^{3}$", "$10^{4}$"])
ax3.set_yticks([1, 10, 100])
ax3.set_yticklabels(["$1$", "$10$", "$100$"])
ax3.set_ylim(0.9, 330)   # 치우침 선이 위 테두리에 붙지 않게 여유를 둔다
ax3.minorticks_off()
ax3.set_title("③ 더 재도 가까워지지 않는다", fontsize=12)
ax3.legend(loc="lower left", fontsize=10.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/one_sample_intervals/img/michelson_bias.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
