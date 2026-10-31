"""9.3절 두 평균의 t 검정 — 그 자료의 척도가 정규가 아니다.

테네시 학급규모 실험의 점수는 **보통 학급 학생들의 순위**로 매긴
백분위라서, 대조군의 분포가 0 에서 100 까지 고르게 퍼진 꼴이다. 균등분포의
표준편차는 100/sqrt(12) = 28.87 이고 초과첨도는 -1.2 다(정규는 0).
소규모 학급은 그 척도 위에서 위로 밀려 올라갔으니 산포가 같을 이유도 없다.

그 두 가지 --- 정규가 아니라는 것과 산포가 같을 이유가 없다는 것 --- 이
equal_var 플래그를 고르게 만드는 조건이다.

쪽의 pooled_vs_welch_alpha 는 합동 t 가 깨지는 자리를, allocation_power 는
배분에 따른 검정력을, t_vs_mannwhitney_power 는 다른 검정과의 견줌을
그린다. 여기서는 플래그를 고르기 전의 자리 --- 자료의 모양 --- 만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/two_sample_tests/img/"
M_CTRL, M_SMALL = 49.9, 54.7          # 쪽이 인용한 유치원 백분위 평균
N_CTRL, N_SMALL = 4000, 1900
rng = np.random.default_rng(1999)

ctrl = rng.uniform(0, 100, N_CTRL)                      # 순위로 만든 척도다
ctrl = np.clip(ctrl - ctrl.mean() + M_CTRL, 0, 100)     # 보고된 평균에 맞춘다
shift = (M_SMALL - M_CTRL) / 100
u = rng.uniform(0, 1, N_SMALL) ** (1 / (1 + 4 * shift))  # 위로 민다
small = 100 * u
small = small - small.mean() + M_SMALL
small = np.clip(small, 0, 100)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
bins = np.linspace(0, 100, 26)
ax1.hist(ctrl, bins=bins, density=True, color="0.6", alpha=0.75,
         label=f"보통 학급 — 평균 {ctrl.mean():.1f}, 표준편차 "
               f"{ctrl.std(ddof=1):.1f}")
ax1.hist(small, bins=bins, density=True, color="tab:blue", alpha=0.6,
         label=f"소규모 학급 — 평균 {small.mean():.1f}, 표준편차 "
               f"{small.std(ddof=1):.1f}")
ax1.axvline(ctrl.mean(), color="0.3", lw=2.0)
ax1.axvline(small.mean(), color="tab:blue", lw=2.0)
ax1.annotate("", xy=(small.mean(), 0.0205), xytext=(ctrl.mean(), 0.0205),
             arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.8))
ax1.text((ctrl.mean() + small.mean()) / 2, 0.0212,
         f"차 {small.mean() - ctrl.mean():.2f} 점", ha="center", va="bottom",
         fontsize=11.5, color="tab:red", fontweight="bold")
ax1.set_ylim(0, 0.024)
ax1.set_xlabel("백분위 점수", fontsize=11)
ax1.set_ylabel("밀도", fontsize=11)
ax1.set_title("순위로 만든 척도라 평평하다", fontsize=12.5)
ax1.legend(fontsize=9.8, loc="lower center", framealpha=0.95)
ax1.grid(axis="y", alpha=0.2)

g = np.linspace(-30, 130, 900)
sd = 100 / np.sqrt(12)
ax2.plot(g, stats.norm.pdf(g, 50, sd), color="tab:red", lw=2.4,
         label=f"같은 평균·표준편차의 정규 (초과첨도 0)")
ax2.plot([0, 0, 100, 100], [0, 0.01, 0.01, 0], color="0.3", lw=2.6,
         label=f"균등 (표준편차 {sd:.2f}, 초과첨도 $-1.2$)")
ax2.fill_between([0, 100], [0.01, 0.01], color="0.3", alpha=0.18)
ax2.set_xlim(-30, 130)
ax2.set_ylim(0, 0.016)
ax2.set_xlabel("백분위 점수", fontsize=11)
ax2.set_ylabel("밀도", fontsize=11)
ax2.set_title("꼬리가 아예 없다 — 정규와 다른 모양", fontsize=12.5)
ax2.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax2.grid(alpha=0.2)

fig.suptitle("정규도 아니고 두 산포가 같을 이유도 없다 — 그래서 플래그가 "
             "물어진다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "percentile_scale.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  보통 학급 평균 {ctrl.mean():.2f} 표준편차 {ctrl.std(ddof=1):.2f} "
      f"초과첨도 {stats.kurtosis(ctrl):.3f}")
print(f"  소규모 학급 평균 {small.mean():.2f} 표준편차 "
      f"{small.std(ddof=1):.2f} 초과첨도 {stats.kurtosis(small):.3f}")
print(f"  차 {small.mean() - ctrl.mean():.2f} · 표준편차의 비 "
      f"{small.std(ddof=1)/ctrl.std(ddof=1):.4f}")
print(f"  균등분포의 이론값: 표준편차 {sd:.4f}, 초과첨도 -1.2")
