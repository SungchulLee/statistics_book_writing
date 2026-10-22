r"""「히스토그램」이라는 말이 처음 찍힌 자리를 그 수로 다시 그린다.

칼 피어슨, 「진화의 수학적 이론에 관하여 II — 균질한 재료에서의 치우친 변이」,
『Philosophical Transactions of the Royal Society of London A』 $186$ 권 $343$–$414$ 쪽
($1895$). $399$ 쪽 각주에 「histogram」이 처음 찍히고, 본문의 그 문장은
「그러나 히스토그램은 곡선의 양 끝에서 어긋나는 **정도**를 보여 준다」이다.

자료는 같은 논문 보기 $10$ (제 $30$ 절) — 고셴의 $1887$ 년 왕립통계학회 회장 연설에서
가져온 「잉글랜드·웨일스 주택 평가액, $1885$–$1886$」이다.

① 그 자료를 밀도 히스토그램으로 그리고 적률로 맞춘 정규곡선을 겹친다.
② 피어슨이 실은 「이론」과 「통계」 두 열을 나란히 세운다 — 어긋남은 양 끝에 있다.

만드는 파일: docs/ch14/graphical_methods/img/pearson_1895_histogram.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

# --- 피어슨이 실은 자료 (논문 396 쪽) -------------------------------
# 평가액 구간의 경계(파운드)와 집의 수. 첫 구간의 아래끝은 피어슨이 적은
# 「최저 연평가액 4파운드 10실링」을 따른다.
EDGES = np.array([4.5, 10, 20, 30, 40, 50, 60, 80, 100, 150, 300, 500, 1000, 1500])
COUNTS = np.array([3_174_806, 1_450_781, 441_595, 259_756, 150_968, 90_432,
                   104_128, 47_326, 58_871, 37_988, 8_781, 3_002, 1_036],
                  dtype=float)

# --- 피어슨이 실은 「이론」과 「통계」 두 열 (천 집 단위, 논문 398 쪽) ---
GROUPS = ["£20 미만", "20–30", "30–40", "40–50", "50–60", "60–80", "80–100", "£100 초과"]
THEORY = np.array([4625, 452, 253, 153, 97, 102, 46, 66], dtype=float)
OBSERV = np.array([4626, 442, 260, 151, 90, 104, 47, 110], dtype=float)

print("① 피어슨의 자료 — 집 수 합계", f"{COUNTS.sum():,.0f}")
print("② 두 열의 합 (천 집)   이론", f"{THEORY.sum():,.0f}",
      "  통계", f"{OBSERV.sum():,.0f}")
print("   £100 아래  이론", f"{THEORY[:-1].sum():,.0f}",
      "  통계", f"{OBSERV[:-1].sum():,.0f}",
      "   (피어슨이 적은 값 5729 · 5720)")

# 구간 폭이 같지 않으므로 높이는 도수를 폭으로 나눈 값이다
width = np.diff(EDGES)
dens = COUNTS / width / COUNTS.sum()
mid = (EDGES[:-1] + EDGES[1:]) / 2
mu = float((COUNTS * mid).sum() / COUNTS.sum())
sd = float(np.sqrt((COUNTS * (mid - mu) ** 2).sum() / COUNTS.sum()))
print(f"\n   구간 중점으로 적률을 재면 평균 £{mu:.2f}, 표준편차 £{sd:.2f}")
print(f"   그 정규곡선이 0 아래에 두는 확률  {stats.norm.cdf(0, mu, sd):.4f}")
print(f"   같은 곡선이 £{EDGES[1]:.0f} 아래에 두는 확률 {stats.norm.cdf(10, mu, sd):.4f}"
      f"   (실제 비율 {COUNTS[0] / COUNTS.sum():.4f})")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2))

# === ① 밀도 히스토그램 + 적률로 맞춘 정규곡선 =========================
XMAX = 160
ax1.bar(EDGES[:-1], dens, width=width, align="edge", color="#d9a441",
        edgecolor="#6b4e16", lw=1.0, label="피어슨의 히스토그램")
xs = np.linspace(-20, XMAX, 600)
ax1.plot(xs, stats.norm.pdf(xs, mu, sd), color="#1f77b4", lw=2.4,
         label=f"적률로 맞춘 정규곡선\n(평균 £{mu:.0f}, 표준편차 £{sd:.0f})")
ax1.axvline(0, color="#777777", lw=1.0, ls=":")
ax1.set_xlim(-20, XMAX)
ax1.set_ylim(0, dens.max() * 1.08)
ax1.set_xlabel("연간 평가액 (파운드) — 꼬리는 £1,500 까지 이어진다")
ax1.set_ylabel("밀도 (도수를 구간 폭으로 나눈 값)")
ax1.set_title("① 정규곡선을 겹쳐 본다면", fontsize=12)
ax1.legend(loc="upper right", fontsize=10)
ax1.grid(alpha=0.25, axis="y")
ax1.text(
    0.52, 0.52,
    f"이 정규곡선은 집의 ${stats.norm.cdf(0, mu, sd) * 100:.0f}\\%$ 를\n"
    "평가액 $0$ 아래에 둔다",
    transform=ax1.transAxes, ha="left", va="top", fontsize=10.5,
    bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e7", ec="#cc9900", lw=1.0),
)

# === ② 피어슨의 두 열 =================================================
idx = np.arange(len(GROUPS))
ax2.bar(idx - 0.2, OBSERV, 0.4, color="#d9a441", edgecolor="#6b4e16", lw=0.8,
        label="통계 (고셴의 보고)")
ax2.bar(idx + 0.2, THEORY, 0.4, color="#1f77b4", edgecolor="#13405f", lw=0.8,
        label="이론 (피어슨이 맞춘 곡선)")
ax2.set_yscale("log")
ax2.set_xticks(idx)
ax2.set_xticklabels(GROUPS, rotation=30, ha="right", fontsize=10)
ax2.set_ylabel("집의 수 (천 집, 로그 눈금)")
ax2.set_title("② 어긋남은 양 끝에 있다", fontsize=12)
ax2.legend(loc="upper right", fontsize=10)
ax2.grid(alpha=0.25, axis="y", which="both")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax2.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: ""))
for k in (0, len(GROUPS) - 1):
    ax2.annotate("", xy=(k, OBSERV[k] * 1.9), xytext=(k, OBSERV[k] * 1.15),
                 arrowprops=dict(arrowstyle="-", color="#d62728", lw=0))
ax2.add_patch(plt.Rectangle((len(GROUPS) - 1.5, 40), 1.0, 200, fill=False,
                            edgecolor="#d62728", lw=1.8, ls="--"))
ax2.annotate(
    f"$66$ 대 ${OBSERV[-1]:.0f}$",
    xy=(len(GROUPS) - 1, 110), xytext=(len(GROUPS) - 3.3, 330),
    fontsize=11.5, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)

fig.tight_layout()
out = "docs/ch14/graphical_methods/img/pearson_1895_histogram.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

print("\n③ 구간별로 본 두 열의 차 (천 집)")
print(f"{'구간':<12}{'통계':>8}{'이론':>8}{'차':>8}{'비':>9}")
for g, o, t in zip(GROUPS, OBSERV, THEORY):
    print(f"{g:<12}{o:>8.0f}{t:>8.0f}{o - t:>8.0f}{o / t:>9.3f}")
