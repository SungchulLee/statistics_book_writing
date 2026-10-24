r"""이분산이 「가정 위반」이 아니라 **찾던 것**인 자리.

제어인자 $C$(석회 함량)와 통제할 수 없는 잡음인자 $N$(가마 안의 온도 편차)이 있고,
출력(타일 크기)이

$$y = \mu + \beta(C)\,N + \varepsilon$$

이라 하자. $C$ 는 평균을 바꾸지 않고 **$N$ 에 대한 민감도** $\beta(C)$ 만 바꾼다.
이것이 제어인자와 잡음인자의 상호작용이고, 다구치가 쓴 길이다.

① 수준마다 평균은 같고 퍼짐만 다르다.
② 평균에 대한 $F$ 는 아무것도 못 보고, 분산에 대한 검정만 본다.
③ 민감도를 낮추면 온도가 흔들려도 불량률이 거의 안 오른다.

만드는 파일: docs/ch12/assumptions/img/robust_design.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(20261021)

LIME = np.array([1, 2, 3, 4, 5])          # 석회 함량 (%)
BETA = np.array([1.00, 0.80, 0.60, 0.40, 0.25])   # 온도 편차에 대한 민감도
SIG_E = 0.30          # 석회로 줄일 수 없는 몫
SIG_N0 = 1.00         # 현장의 온도 편차
SPEC = 1.00           # 규격 한계 (목표치에서 ±SPEC)
NPER, REPS = 30, 4000

sd = np.sqrt((BETA * SIG_N0) ** 2 + SIG_E ** 2)
scrap = 2 * stats.norm.sf(SPEC / sd)
print(f"잡음 표준편차 {SIG_N0}, 규격 ±{SPEC}, 줄일 수 없는 몫 {SIG_E}")
print(f"{'석회 %':>7}{'민감도 β':>10}{'출력 SD':>10}{'불량률':>10}")
for l, b, s, p in zip(LIME, BETA, sd, scrap):
    print(f"{l:>7}{b:>10.2f}{s:>10.3f}{p:>10.4f}")
print(f"분산비 (1% 대 5%) = {(sd[0] / sd[-1]) ** 2:.1f} 배")

# 같은 결과를 「가마를 고쳐서」 얻으려면 온도 편차를 얼마나 줄여야 하는가
need = np.sqrt(max(sd[-1] ** 2 - SIG_E ** 2, 0)) / BETA[0]
print(f"석회 1% 를 그대로 두고 같은 불량률을 내려면 온도 편차를 "
      f"{SIG_N0:.2f} -> {need:.3f} 로, {SIG_N0 / need:.1f} 배 줄여야 한다")

# === ② 두 검정이 보는 것 ============================================
rej_f = rej_l = rej_b = 0
for _ in range(REPS):
    g = [RNG.normal(0.0, s, NPER) for s in sd]
    rej_f += stats.f_oneway(*g).pvalue < 0.05
    rej_l += stats.levene(*g, center="median").pvalue < 0.05
    rej_b += stats.bartlett(*g).pvalue < 0.05
print(f"\n② 수준마다 {NPER} 장씩, {REPS} 회")
print(f"   평균에 대한 일원분산분석 F : 기각률 {rej_f / REPS:.3f}  (참 평균은 모두 같다)")
print(f"   레빈 검정                 : 기각률 {rej_l / REPS:.3f}")
print(f"   바틀릿 검정               : 기각률 {rej_b / REPS:.3f}")

# === ③ 온도 편차에 대한 민감도 =======================================
SN = np.linspace(0, 2.0, 120)
curves = {int(l): 2 * stats.norm.sf(SPEC / np.sqrt((b * SN) ** 2 + SIG_E ** 2))
          for l, b in zip(LIME, BETA)}
print(f"\n③ 온도 편차가 두 배({2 * SIG_N0:.1f})로 나빠질 때의 불량률")
for l in LIME:
    i0 = int(np.argmin(np.abs(SN - SIG_N0)))
    i1 = int(np.argmin(np.abs(SN - 2 * SIG_N0)))
    print(f"   석회 {l}% : {curves[int(l)][i0]:.4f} -> {curves[int(l)][i1]:.4f}  "
          f"({curves[int(l)][i1] - curves[int(l)][i0]:+.4f})")

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))
COL = plt.cm.viridis(np.linspace(0.08, 0.86, len(LIME)))

xs = np.linspace(-3.2, 3.2, 500)
for l, s, c in zip(LIME, sd, COL):
    ax1.plot(xs, stats.norm.pdf(xs, 0, s), color=c, lw=2.5,
             label=f"석회 ${l}\\%$")
ax1.axvline(-SPEC, color="#d62728", lw=1.8, ls="--")
ax1.axvline(SPEC, color="#d62728", lw=1.8, ls="--")
ax1.text(SPEC + 0.08, ax1.get_ylim()[1] * 0.93, "규격", color="#d62728",
         fontsize=10.5)
ax1.axvline(0, color="#555555", lw=1.1, ls=":")
ax1.text(1.25, ax1.get_ylim()[1] * 0.72, "평균은 다섯 수준이\n모두 같다",
         fontsize=10, va="center", color="#555555")
ax1.set_xlabel("타일 크기 (목표치로부터)")
ax1.set_ylabel("밀도")
ax1.set_title("① 평균은 그대로, 퍼짐만 바뀐다", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25)

names = ["일원분산분석\n(평균)", "레빈 검정\n(분산)", "바틀릿 검정\n(분산)"]
vals = [rej_f / REPS, rej_l / REPS, rej_b / REPS]
cols = ["#999999", "#1f77b4", "#1f77b4"]
ax2.bar(np.arange(3), vals, 0.6, color=cols)
for i, v in enumerate(vals):
    ax2.text(i, v + 0.025, f"${v:.3f}$", ha="center", fontsize=12)
ax2.axhline(0.05, color="black", lw=1.4, ls=":")
ax2.text(2.42, 0.075, "$0.05$", fontsize=10, ha="right")
ax2.set_xticks(np.arange(3))
ax2.set_xticklabels(names, fontsize=10)
ax2.set_ylim(0, 1.12)
ax2.set_ylabel("기각률")
ax2.set_title("② 평균을 보는 검정은 못 본다", fontsize=12)
ax2.grid(alpha=0.25, axis="y")

for l, c in zip(LIME, COL):
    ax3.plot(SN, curves[int(l)], color=c, lw=2.6, label=f"석회 ${l}\\%$")
ax3.axvline(SIG_N0, color="#333333", lw=1.5, ls="-.")
ax3.text(SIG_N0 + 0.04, 0.52, "현장의\n온도 편차", fontsize=10)
ax3.annotate("", xy=(need, scrap[-1]), xytext=(SIG_N0, scrap[-1]),
             arrowprops=dict(arrowstyle="<->", color="#d62728", lw=1.8))
ax3.text(need - 0.03, scrap[-1] + 0.10,
         f"가마로 같은 데 가려면\n${SIG_N0 / need:.1f}$ 배 줄여야 한다",
         fontsize=10, color="#d62728", ha="left")
ax3.set_xlabel("가마 안 온도 편차 (잡음인자)")
ax3.set_ylabel("규격을 벗어날 확률")
ax3.set_ylim(0, 0.62)
ax3.set_title("③ 둔감하게 만드는 쪽이 싸다", fontsize=12)
ax3.legend(loc="upper left", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/assumptions/img/robust_design.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
