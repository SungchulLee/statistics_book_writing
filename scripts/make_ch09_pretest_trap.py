"""9.3 이표본 t — 「먼저 등분산 검정」 두 단계 절차가 어디서 무너지는가.

왜 이 그림인가. 책은 「사전검정으로 고르지 말고 웰치를 쓰라」를 마흔 곳 넘게
적어 두었는데, 두 표본에서 그 권고를 **수로** 받치는 자리가 5 장 연습문제
하나뿐이고 그마저 σ 비 3 에서만 재었다. σ 비 3 은 사전검정이 97 퍼센트
잡아내는 자리라 두 단계 절차가 멀쩡해 보인다 — 권고의 근거가 가장 약하게
보이는 자리에서만 재어 둔 셈이다.

이 그림은 σ 비를 1 에서 4 까지 쓸어 본다. 그러면 두 단계 절차의 실제
크기가 **봉우리**를 그린다 — 양 끝에서는 괜찮고 가운데(σ 비 1.5 쯤)에서
명목의 1.6 배가 된다. 까닭은 오른쪽 칸에 있다. 거기서 F 검정이 이분산을
셋에 하나밖에 잡지 못하므로, 못 잡은 세 번 가운데 두 번은 그대로 합동
t 로 넘어가 합동의 팽창을 물려받는다.

**사전검정은 차이가 클 때만 일하고, 판정이 필요한 것은 차이가 작을 때다.**

실행:  python3 scripts/make_ch09_pretest_trap.py   (저장소 최상위에서)
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch09/two_sample_tests/img/"
N1, N2 = 10, 30          # 작은 집단에 큰 분산이 붙는 배치
REP = 40_000
ALPHA = 0.05

ratios = np.arange(1.0, 4.01, 0.125)
lo = stats.f.ppf(ALPHA / 2, N1 - 1, N2 - 1)
hi = stats.f.ppf(1 - ALPHA / 2, N1 - 1, N2 - 1)

rng = np.random.default_rng(20261105)
pooled, welch, twostep, caught = [], [], [], []
for r in ratios:
    a = rng.normal(0, r, (REP, N1))
    b = rng.normal(0, 1, (REP, N2))
    pp = stats.ttest_ind(a, b, axis=1, equal_var=True).pvalue
    pw = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
    F = a.var(1, ddof=1) / b.var(1, ddof=1)
    c = ~((F >= lo) & (F <= hi))          # 사전검정이 등분산을 기각한 경우
    two = np.where(c, pw, pp)             # 기각하면 웰치, 못 하면 합동
    pooled.append((pp < ALPHA).mean())
    welch.append((pw < ALPHA).mean())
    twostep.append((two < ALPHA).mean())
    caught.append(c.mean())

pooled, welch = np.array(pooled), np.array(welch)
twostep, caught = np.array(twostep), np.array(caught)
k = int(np.argmax(twostep))

print(f"n = ({N1}, {N2}) · 반복 {REP:,} · 명목 α = {ALPHA}")
print(f"{'σ비':>6}{'합동':>9}{'웰치':>9}{'두 단계':>9}{'F 가 잡음':>10}")
for i, r in enumerate(ratios):
    if abs(r - round(r * 4) / 4) < 1e-9 and (r * 4) % 1 == 0 and r in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        print(f"{r:>6.2f}{pooled[i]:>9.4f}{welch[i]:>9.4f}{twostep[i]:>9.4f}{caught[i]:>10.4f}")
print(f"\n두 단계 절차의 최악은 σ비 {ratios[k]:.3f} 에서 {twostep[k]:.4f} "
      f"(그때 F 가 잡은 비율 {caught[k]:.4f})")

# ════════════════════════════════════════════════════════════════════
RED, ORANGE, GREEN, GREY = "#C62828", "#EF6C00", "#2E7D32", "#78909C"
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)

ax = axes[0]
ax.axhline(ALPHA, color="0.35", lw=1.4, ls=(0, (5, 3)), zorder=2)
ax.text(3.55, ALPHA + 0.008, "명목 0.05", color="0.35", fontsize=10.5,
        ha="center", va="bottom")
ax.plot(ratios, pooled, color=RED, lw=2.6, label="언제나 합동 (스튜던트 t)")
ax.plot(ratios, twostep, color=ORANGE, lw=2.6, label="두 단계 (F 검정으로 고름)")
ax.plot(ratios, welch, color=GREEN, lw=2.6, label="언제나 웰치")
ax.plot([ratios[k]], [twostep[k]], "o", color=ORANGE, ms=8, zorder=5)
ax.annotate(f"최악 {twostep[k]:.3f}\n(명목의 {twostep[k]/ALPHA:.1f} 배)",
            xy=(ratios[k], twostep[k]), xytext=(ratios[k] + 0.55, twostep[k] + 0.035),
            color=ORANGE, fontsize=11,
            arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4))
ax.set_xlabel("작은 집단의 표준편차 배수  $\\sigma_1/\\sigma_2$", fontsize=11.5)
ax.set_ylabel("실제 제1종 오류율", fontsize=11.5)
ax.set_title("(가) 세 전략의 실제 크기", fontsize=12.8)
ax.set_ylim(0, 0.27)
ax.legend(fontsize=10.4, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(ratios, caught, color=GREY, lw=2.6)
ax.fill_between(ratios, 0, caught, color=GREY, alpha=0.18)
ax.axvline(ratios[k], color=ORANGE, lw=1.6, ls=(0, (4, 3)))
ax.plot([ratios[k]], [caught[k]], "o", color=GREY, ms=8, zorder=5)
ax.annotate(f"여기서 {caught[k]:.0%} 만 잡는다",
            xy=(ratios[k], caught[k]), xytext=(ratios[k] + 0.5, caught[k] + 0.22),
            color="0.2", fontsize=11,
            arrowprops=dict(arrowstyle="-|>", color="0.35", lw=1.4))
ax.set_xlabel("작은 집단의 표준편차 배수  $\\sigma_1/\\sigma_2$", fontsize=11.5)
ax.set_ylabel("F 사전검정이 이분산을 기각한 비율", fontsize=11.5)
ax.set_title("(나) 사전검정은 무엇을 잡아내는가", fontsize=12.8)
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.25)

fig.suptitle("두 단계 절차의 최악은 양 끝이 아니라 가운데다", fontsize=14.4, y=0.99)
plt.tight_layout(rect=(0, 0, 1, 0.95))
path = OUT + "pretest_trap.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
