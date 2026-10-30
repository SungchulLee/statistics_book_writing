"""9.3절 두 평균의 z 검정 — p 값에는 바닥이 있다.

보기 1 의 자료에서 A 학군은 더 조사할 수 없고 B 학군만 늘린다.
n2 를 아무리 키워도 표준오차가 sigma_1/sqrt(n1) 아래로 내려가지 못하므로
p 값이 한 값에서 멈춘다. 쪽의 두 수(0.120 과 0.018422)를 다시 계산해 확인한다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

OUT = "docs/ch09/two_sample_tests/img/"

DIFF = 4.0                       # 78 - 74
S1, S2 = 12.0, 15.0              # 알려진 두 모표준편차
N1 = 50                          # A 학군 — 더 조사할 수 없다
ALPHA = 0.05


def pvalue(n2):
    se = np.sqrt(S1**2 / N1 + S2**2 / n2)
    return 2 * (1 - stats.norm.cdf(DIFF / se))


n2 = np.unique(np.round(np.logspace(np.log10(60), 6, 500)).astype(int))
p = np.array([pvalue(m) for m in n2])

se_floor = S1 / np.sqrt(N1)                       # n2 -> 무한대일 때의 표준오차
p_floor = 2 * (1 - stats.norm.cdf(DIFF / se_floor))
p_start = pvalue(60)

fig, ax = plt.subplots(figsize=(10.6, 4.4))

ax.plot(n2, p, "-", color="tab:blue", lw=2.4, zorder=3)
ax.axhline(p_floor, color="tab:red", ls="--", lw=1.6)
ax.axhline(ALPHA, color="0.45", ls=":", lw=1.4)

ax.fill_between(n2, p_floor, p, color="tab:blue", alpha=0.10)
ax.fill_between(n2, 0, p_floor, color="tab:red", alpha=0.07)

ax.scatter([60], [p_start], s=110, c="tab:blue", zorder=5,
           edgecolor="white", lw=1.5)
ax.annotate(f"B 학군 {60} 명에서  p = {p_start:.3f}", (60, p_start),
            textcoords="offset points", xytext=(16, 8), ha="left",
            fontsize=11, color="tab:blue")

ax.text(6e5, p_floor + 0.004, f"바닥  p = {p_floor:.6f}", ha="right",
        va="bottom", fontsize=11, color="tab:red", fontweight="bold")
ax.text(6e5, p_floor - 0.004, "백만 명을 조사해도 이 아래로 내려가지 않는다",
        ha="right", va="top", fontsize=9.5, color="tab:red")
# 곡선과 겹치지 않도록 곡선이 평평해진 뒤에 적는다
ax.text(3000, ALPHA + 0.003, "유의수준 0.05", ha="left", va="bottom",
        fontsize=10, color="0.35")

ax.set_xscale("log")
ticks = [60, 100, 1000, 10_000, 100_000, 1_000_000]
ax.xaxis.set_major_locator(FixedLocator(ticks))
ax.xaxis.set_major_formatter(FixedFormatter([f"{t:,}" for t in ticks]))
ax.xaxis.set_minor_locator(FixedLocator([]))
ax.set_xlim(55, 1.2e6)
ax.set_ylim(0, 0.135)
ax.set_xlabel("B 학군의 표본크기 (A 학군은 50 명으로 묶여 있다, 로그 눈금)")
ax.set_ylabel("양측 p 값")
ax.set_title("한쪽 집단만 늘리면 p 값은 어디까지 내려가는가", fontsize=12)
ax.grid(alpha=0.25)

plt.tight_layout()
path = OUT + "pvalue_floor.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"n2 = 60      p = {p_start:.4f}   (쪽: 0.120)")
print(f"n2 -> 무한대  p = {p_floor:.6f} (쪽: 0.018422)")
print(f"표준오차의 바닥 = {S1}/sqrt({N1}) = {se_floor:.6f}")
for m in (100, 1000, 10_000, 1_000_000):
    print(f"  n2 = {m:>9,}  p = {pvalue(m):.6f}")
