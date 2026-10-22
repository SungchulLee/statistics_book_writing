r"""9.3 「두 평균의 z 검정」 쪽의 상자에 들어가는 그림을 만든다.

Gelman 과 Stern(2006)의 보기 — 추정값 25 +- 10 과 10 +- 10 — 을 그대로 그리고,
한쪽이 유의하고 다른 쪽이 유의하지 않다는 것이 둘의 차이에 대해 무엇을 말하는지
(거의 아무것도 말하지 않는다) 보인다.

만드는 파일:
  ch09/two_sample_tests/img/significant_difference.png

실행:  python3 scripts/make_ch09_significant_difference.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

INK = "#37474F"
BLUE = "#1565C0"
ORANGE = "#E65100"
GREEN = "#33691E"
MUTED = "#90A4AE"
RED = "#D32F2F"

OUT = "docs/ch09/two_sample_tests/img/"
os.makedirs(OUT, exist_ok=True)

ZC = stats.norm.ppf(0.975)

# Gelman 과 Stern 이 든 보기 그대로
EST = [25.0, 10.0]
SE = [10.0, 10.0]
D = EST[0] - EST[1]
SED = np.hypot(SE[0], SE[1])

print("Gelman-Stern 보기")
for nm, e, s in zip(["연구 A", "연구 B"], EST, SE):
    z = e / s
    print(f"  {nm}: {e:.0f} +- {s:.0f}   z = {z:.2f},  p = {2*stats.norm.sf(abs(z)):.4f}")
print(f"  차이: {D:.0f} +- {SED:.2f}   z = {D/SED:.2f},  "
      f"p = {2*stats.norm.sf(abs(D/SED)):.4f}")
print(f"  차이의 95% 구간 = [{D - ZC*SED:.1f}, {D + ZC*SED:.1f}]")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.4))

# -------------------------------------------------- 왼쪽: 두 추정값과 그 차이
labels = ["연구 A", "연구 B", "차이 A $-$ B"]
vals = [EST[0], EST[1], D]
ses = [SE[0], SE[1], SED]
cols = [BLUE, MUTED, ORANGE]
ys = [2, 1, 0]
for y, lab, v, s, c in zip(ys, labels, vals, ses, cols):
    axL.plot([v - ZC*s, v + ZC*s], [y, y], color=c, lw=3.4,
             solid_capstyle="round", zorder=3)
    axL.plot([v], [y], "o", color=c, ms=8, zorder=4)
    z = v / s
    p = 2 * stats.norm.sf(abs(z))
    axL.text(52, y, f"$z = {z:.2f}$,  $p = {p:.3f}$", fontsize=9.5,
             color=c, va="center")
axL.axvline(0, color=RED, lw=1.6, zorder=2)
axL.set_yticks(ys)
axL.set_yticklabels(labels, fontsize=10.5)
axL.set_xlim(-15, 92)
axL.set_ylim(-0.6, 2.6)
axL.set_xlabel("효과 추정값과 $95\\%$ 구간")
axL.set_title("$25 \\pm 10$ 과 $10 \\pm 10$ — 그리고 그 차이",
              color=INK, fontsize=12, pad=9)
for s_ in ("top", "right"):
    axL.spines[s_].set_visible(False)

# -------------------------------------------------- 오른쪽: z1 을 고정하고 z2 를 훑는다
Z1 = 1.96
z2 = np.linspace(-4, 6, 1200)
pdiff = 2 * stats.norm.sf(np.abs(Z1 - z2) / np.sqrt(2))
axR.plot(z2, pdiff, color=ORANGE, lw=2.4, zorder=4,
         label="두 연구의 차이에 대한 $p$ 값")
axR.axhline(0.05, color=RED, lw=1.3, ls="--", zorder=3)
axR.axvspan(-ZC, ZC, color=MUTED, alpha=0.18, zorder=1)
axR.text(0, 0.92, "이 띠 안에서는\n연구 B 가 유의하지 않다",
         ha="center", fontsize=9, color=INK, va="top")
lo = Z1 - np.sqrt(2) * ZC
axR.axvspan(-ZC, lo, color=GREEN, alpha=0.22, zorder=2)
axR.annotate(f"둘 다 참이려면 $z_2 < {lo:.2f}$ —\n"
             "연구 B 가 반대 방향을 가리켜야 한다",
             xy=((-ZC + lo) / 2, 0.035), xytext=(1.6, 0.28), fontsize=9,
             color=GREEN, arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.0))
axR.text(5.9, 0.075, "$0.05$", color=RED, fontsize=9.5, ha="right")
axR.set_xlim(-4, 6)
axR.set_ylim(0, 1.03)
axR.set_xlabel("연구 B 의 $z_2$  (연구 A 는 $z_1 = 1.96$ 으로 고정)")
axR.set_ylabel("차이에 대한 $p$ 값")
axR.set_title("「한쪽만 유의하다」가 차이에 대해 말해 주는 것",
              color=INK, fontsize=12, pad=9)
axR.legend(loc="upper right", fontsize=9, framealpha=0.95)
for s_ in ("top", "right"):
    axR.spines[s_].set_visible(False)

print(f"\nz1 = {Z1} 일 때 차이가 유의하려면 z2 < {lo:.3f} 또는 "
      f"z2 > {Z1 + np.sqrt(2)*ZC:.3f}")

fig.tight_layout()
path = OUT + "significant_difference.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
