"""16.5절 프리드먼 검정 §5 가 설명하는데 없던 그림.

쪽의 §5 본문이 「왼쪽 (가)가 원점수이고 … 오른쪽 (나)가 블록 내 순위이다」
라고 두 칸짜리 그림을 설명하는데, 그 자리의 참조가 friedman_blocks.png(지금의 friedman_blocks_1.png)
(블록 사이 변동과 네 검정의 검정력) 를 가리키고 있었다. 한 PNG 에 서로
다른 설명 둘이 붙어 있던 셈이고, --strict 도 lint_figures 도 둘 다
존재하는 파일을 가리키므로 잡지 못한다.

본문이 그림을 정확히 적어 두어서 그대로 그린다. 자료는 쪽의 보기 1 표와
같고, 순위합 5 · 12 · 20 · 13 도 본문과 맞는지 확인한다.

만드는 파일: docs/ch16/multi_group_nonparametric/img/friedman_raw_vs_within_ranks.png
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch16/multi_group_nonparametric/img/"
D = np.array([[30, 45, 60, 50],
              [25, 40, 55, 35],
              [35, 50, 65, 45],
              [20, 30, 50, 40],
              [40, 55, 70, 60]], float)
DRUGS = ["약 A", "약 B", "약 C", "약 D"]
R = np.vstack([stats.rankdata(row) for row in D])
Rsum = R.sum(0)
print("[friedman §5] 쪽의 보기 1 자료")
for i, row in enumerate(D, 1):
    print(f"  환자 {i}: {row.astype(int)}  평균 {row.mean():6.2f}  "
          f"순위 {R[i-1].astype(int)}")
print(f"  순위합 {Rsum.astype(int)}  (본문은 5 · 12 · 20 · 13)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), sharex=True)
x = np.arange(4)
COLS = plt.cm.viridis(np.linspace(0.08, 0.82, 5))

for i in range(5):
    ax1.plot(x, D[i], "o-", color=COLS[i], lw=2.4, ms=10,
             label=f"환자 {i + 1}")
ax1.set_xticks(x, DRUGS, fontsize=12.5)
ax1.set_ylabel("통증 완화 평가 (1~100)", fontsize=11.5)
ax1.set_ylim(10, 80)
ax1.set_title("(가) 원점수 — 선이 통째로 위아래로 밀려 있다", fontsize=13)
ax1.grid(alpha=0.25)
ax1.legend(fontsize=10.2, loc="upper left", ncol=2)
ax1.annotate(f"환자 5 의 평균 {D[4].mean():.2f}", xy=(0.12, D[4, 0]),
             xytext=(0.35, 24), fontsize=11, color=COLS[4],
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color=COLS[4], lw=1.8))
ax1.annotate(f"환자 4 의 평균 {D[3].mean():.2f}", xy=(0.12, D[3, 0]),
             xytext=(0.35, 15.5), fontsize=11, color=COLS[3],
             fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color=COLS[3], lw=1.8))
ax1.plot([2], [D[3, 2]], "o", ms=20, mfc="none", mec="tab:red", mew=2.6)
ax1.plot([1], [D[4, 1]], "o", ms=20, mfc="none", mec="tab:red", mew=2.6)
ax1.annotate(f"환자 4 가 가장 좋다고 꼽은 {D[3, 2]:.0f} 이\n"
             f"환자 5 가 둘째로 꼽은 {D[4, 1]:.0f} 보다 낮다",
             xy=(1.95, D[3, 2]), xytext=(2.1, 20), fontsize=10.8,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))

for i in range(5):
    jitter = (i - 2) * 0.035
    ax2.plot(x, R[i] + jitter, "o-", color=COLS[i], lw=2.4, ms=10)
ax2.set_xticks(x, DRUGS, fontsize=12.5)
ax2.set_xlim(-0.45, 3.45)
ax2.set_yticks([1, 2, 3, 4], ["1 위", "2 위", "3 위", "4 위"], fontsize=12)
ax2.set_ylim(0.5, 5.15)
ax2.set_ylabel("환자 안에서 매긴 순위", fontsize=11.5)
ax2.set_title("(나) 블록 내 순위 — 기준선이 통째로 지워진다", fontsize=13)
ax2.grid(alpha=0.25)
for j in range(4):
    ax2.text(j, 4.72, f"순위합 {Rsum[j]:.0f}", ha="center", va="center",
             fontsize=12.5, fontweight="bold", color="0.15")
ax2.annotate("다섯 환자가 한목소리를 낸다 —\n"
             "약 A 는 다섯 번 모두 1 위, 약 C 는 다섯 번 모두 4 위",
             xy=(0.08, 1.05), xytext=(1.05, 1.55), fontsize=11.3,
             color="0.15", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="0.15", lw=1.8))

fig.suptitle("블록 안에서 순위를 매긴다는 한 줄이 하는 일", fontsize=15,
             y=0.98)
plt.tight_layout()
fig.savefig(OUT + "friedman_raw_vs_within_ranks.png", dpi=170,
            facecolor="white", bbox_inches="tight")
print("saved", OUT + "friedman_raw_vs_within_ranks.png")
