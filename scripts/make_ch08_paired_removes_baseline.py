"""8.4절 μ_D 의 신뢰구간 — 처음 몸무게는 뺄셈에서 사라진다.

들머리의 기제를 그린다. 사람마다 처음 몸무게가 다르므로 끝에 잰 몸무게만
모아 평균을 내면 그 개인차가 그대로 분모에 남는다. 같은 사람의 두 수를
빼면 개인차가 지워지고 약이 한 일만 남는다.

자료는 이 책이 만든 것이고 시험의 측정값이 아니다. 시험이 보고한 평균
변화율(-20.9%)만 가져다 쓰고, 처음 몸무게의 흩어짐과 사람마다의 반응
차이는 그럴듯한 값으로 두었다.

쪽의 first_t_data 와 pairing_gain 은 사람마다의 선을 잇는 그림이고
arena_pairing 은 설계 둘의 반폭을 견준다. 여기서는 선을 하나도 긋지 않고
**분포 둘**로 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/paired_sample_intervals/img/"

N = 600            # 한 군의 크기쯤
W0_MEAN, W0_SD = 105.0, 20.0      # 처음 몸무게 (kg)
PCT_MEAN, PCT_SD = -20.9, 7.0     # 사람마다의 변화율 (%)
Z = stats.norm.ppf(0.975)

rng = np.random.default_rng(72)
w0 = rng.normal(W0_MEAN, W0_SD, N)
pct = rng.normal(PCT_MEAN, PCT_SD, N)
w1 = w0 * (1 + pct / 100)
d = w1 - w0                                   # 사람마다의 차이 (kg)

se_paired = d.std(ddof=1) / np.sqrt(N)
se_unpaired = np.sqrt(w0.var(ddof=1) / N + w1.var(ddof=1) / N)
lo, hi = d.mean() - Z * se_paired, d.mean() + Z * se_paired

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.7))

# ── (좌) 끝 몸무게만 모으면 ────────────────────────────────────────
bins = np.linspace(40, 180, 60)
ax1.hist(w0, bins=bins, color="0.55", alpha=0.65, label="시작할 때")
ax1.hist(w1, bins=bins, color="tab:blue", alpha=0.60, label="72 주 뒤")
for v, c in ((w0.mean(), "0.3"), (w1.mean(), "tab:blue")):
    ax1.axvline(v, color=c, lw=2.2)
ax1.annotate("", xy=(w1.mean(), 44), xytext=(w0.mean(), 44),
             arrowprops=dict(arrowstyle="<->", color="tab:red", lw=1.8))
ax1.text((w0.mean() + w1.mean()) / 2, 46,
         f"평균이 움직인 거리 {w0.mean() - w1.mean():.1f} kg", ha="center",
         va="bottom", fontsize=11, color="tab:red", fontweight="bold")
ax1.set_xlabel("몸무게 (kg)", fontsize=11)
ax1.set_ylabel("사람 수", fontsize=11)
ax1.set_ylim(0, 56)
ax1.set_title("몸무게를 그냥 모으면 — 두 무리가 크게 겹친다", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax1.grid(axis="y", alpha=0.2)

# ── (우) 같은 사람의 두 수를 빼면 ──────────────────────────────────
ax2.hist(d, bins=40, color="tab:green", alpha=0.70,
         label="사람마다의 차이")
ax2.axvline(0, color="0.15", lw=2.2)
ax2.text(0.6, ax2.get_ylim()[1] * 0.93, "변화 없음", fontsize=10.5,
         color="0.15")
ax2.axvspan(lo, hi, color="tab:red", alpha=0.85)
ax2.annotate(f"차이들의 평균에 걸린 95% 구간\n"
             f"({lo:.2f}, {hi:.2f}) — 폭 {hi - lo:.2f} kg",
             xy=(d.mean(), ax2.get_ylim()[1] * 0.78),
             xytext=(0.04, 0.80), textcoords="axes fraction",
             fontsize=11, color="tab:red", fontweight="bold", va="center",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="tab:red", lw=1.0),
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax2.set_xlabel("72 주 뒤 몸무게 빼기 시작 몸무게 (kg)", fontsize=11)
ax2.set_ylabel("사람 수", fontsize=11)
ax2.set_title("같은 사람의 두 수를 빼면 — 개인차가 지워진다", fontsize=12.5)
ax2.legend(fontsize=10, loc="upper right", framealpha=0.95,
           bbox_to_anchor=(0.80, 1.0))
ax2.grid(axis="y", alpha=0.2)
ax2.text(0.985, 0.035,
         f"짝을 지었을 때 표준오차 {se_paired:.3f} kg\n"
         f"짝을 풀었을 때 {se_unpaired:.3f} kg — {se_unpaired/se_paired:.1f} 배",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=10.5,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("사람마다 달랐던 처음 몸무게는 뺄셈에서 사라진다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "paired_removes_baseline.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n이 책이 만든 자료: n = {N}, 처음 몸무게 N({W0_MEAN}, {W0_SD}^2) kg, "
      f"변화율 N({PCT_MEAN}, {PCT_SD}^2) %")
print(f"  시작 평균 {w0.mean():.2f} (표준편차 {w0.std(ddof=1):.2f}) · "
      f"72 주 뒤 평균 {w1.mean():.2f} (표준편차 {w1.std(ddof=1):.2f})")
print(f"  차이의 평균 {d.mean():.3f} kg · 표준편차 {d.std(ddof=1):.3f} kg")
print(f"  짝을 지었을 때 표준오차 {se_paired:.4f} · 95% 구간 ({lo:.3f}, {hi:.3f})")
print(f"  짝을 풀었을 때 표준오차 {se_unpaired:.4f}  →  "
      f"{se_unpaired/se_paired:.2f} 배")
