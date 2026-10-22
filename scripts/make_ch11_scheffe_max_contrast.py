r"""11.3 「Bonferroni와 Scheffé 방법」 쪽의 상자에 들어가는 그림을 만든다.

전역 F 가 기각했는데 어느 쌍도 유의하지 않은 자료를 만들고, Scheffé 가 반드시
찾아 주는 대비 — c_i 가 (ybar_i - ybar) 에 비례하는 그것 — 를 보인다.
그 대비의 F_L 은 언제나 전역 F 의 (k-1) 배다.

만드는 파일:
  ch11/post_hoc/img/scheffe_max_contrast.png

실행:  python3 scripts/make_ch11_scheffe_max_contrast.py   (저장소 최상위에서)
필요:  numpy, scipy, matplotlib — 문서 빌드에는 필요하지 않다.
       PNG는 커밋되므로 CI에서 다시 그리지 않는다.
"""
import os
from itertools import combinations

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

OUT = "docs/ch11/post_hoc/img/"
os.makedirs(OUT, exist_ok=True)

K, N_I, MSE = 4, 6, 1.60
MEANS = np.array([10.0, 10.2, 11.8, 12.0])
DF = K * (N_I - 1)
ALPHA = 0.05

F = N_I * ((MEANS - MEANS.mean()) ** 2).sum() / (K - 1) / MSE
P_F = stats.f.sf(F, K - 1, DF)
Q = stats.studentized_range.ppf(1 - ALPHA, K, DF)
HSD = Q * np.sqrt(MSE / N_I)
SCH = (K - 1) * stats.f.ppf(1 - ALPHA, K - 1, DF)


def contrast_F(c):
    L = (np.asarray(c) * MEANS).sum()
    return L ** 2 / (MSE * (np.asarray(c) ** 2 / N_I).sum()), L


print(f"집단 평균 {MEANS},  MSE = {MSE},  집단당 n = {N_I}")
print(f"전역 F = {F:.3f},  p = {P_F:.4f}")
print(f"Tukey HSD 문턱 = {HSD:.3f} (q = {Q:.3f})")
print(f"Scheffé 문턱 (k-1)F = {SCH:.3f}\n")

pairs = []
for i, j in combinations(range(K), 2):
    c = np.zeros(K)
    c[j], c[i] = 1.0, -1.0
    FL, L = contrast_F(c)
    p_t = stats.studentized_range.sf(abs(L) / np.sqrt(MSE / N_I), K, DF)
    pairs.append((f"$\\mu_{j+1} - \\mu_{i+1}$", L, FL, p_t))
    print(f"  쌍 {i+1}-{j+1}: 차이 {L:+.2f}  Tukey p = {p_t:.4f}  F_L = {FL:5.2f}")

cmax = MEANS - MEANS.mean()
FLmax, Lmax = contrast_F(cmax)
ctwo = np.array([-0.5, -0.5, 0.5, 0.5])
FLtwo, Ltwo = contrast_F(ctwo)
print(f"\n최대 대비 c = {np.round(cmax, 3)}:  F_L = {FLmax:.4f}")
print(f"  확인  (k-1) x 전역 F = {(K - 1) * F:.4f}   차이 {abs(FLmax - (K-1)*F):.2e}")
print(f"보기 좋은 대비 (-1,-1,1,1)/2:  L = {Ltwo:.2f}, F_L = {FLtwo:.4f}, "
      f"Scheffé p = {stats.f.sf(FLtwo/(K-1), K-1, DF):.4f}")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.5))

# ------------------------------------------------- 왼쪽: 어느 쌍도 유의하지 않다
ys = np.arange(len(pairs))[::-1]
for y, (lab, L, FL, p_t) in zip(ys, pairs):
    half = HSD
    axL.plot([L - half, L + half], [y, y], color=MUTED, lw=3.0,
             solid_capstyle="round", zorder=2)
    axL.plot([L], [y], "o", color=BLUE, ms=7, zorder=4)
    axL.text(4.6, y, f"$p = {p_t:.3f}$", fontsize=9, color=INK, va="center")
axL.axvline(0, color=RED, lw=1.5, zorder=3)
axL.set_yticks(ys)
axL.set_yticklabels([p[0] for p in pairs], fontsize=10)
axL.set_xlim(-3.0, 6.4)
axL.set_ylim(-0.8, len(pairs) - 0.2)
axL.set_xlabel("평균 차이와 Tukey 구간")
axL.set_title(f"전역 $F = {F:.2f}$ ($p = {P_F:.4f}$) 인데 "
              "여섯 쌍 모두 $0$ 을 품는다", color=INK, fontsize=11.5, pad=9)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

# ------------------------------------------------- 오른쪽: 대비들의 F_L
labels = [p[0] for p in pairs] + ["$\\frac{\\mu_3+\\mu_4}{2}-\\frac{\\mu_1+\\mu_2}{2}$",
                                  "최대 대비"]
vals = [p[2] for p in pairs] + [FLtwo, FLmax]
cols = [MUTED] * len(pairs) + [GREEN, BLUE]
x = np.arange(len(vals))
axR.bar(x, vals, width=0.62, color=cols, edgecolor=INK, linewidth=0.5, zorder=3)
axR.axhline(SCH, color=RED, lw=1.6, ls="--", zorder=4,
            label=f"셰페 문턱 $(k-1)F_{{0.05}} = {SCH:.2f}$")
for xi, v in zip(x, vals):
    axR.text(xi, v + 0.35, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
axR.set_xticks(x)
axR.set_xticklabels(labels, fontsize=8.5, rotation=32, ha="right")
axR.set_ylim(0, FLmax * 1.30)
axR.set_ylabel("대비의 $F_L$")
axR.set_title("쌍만 보면 못 찾는다 — 쪼갠 방향을 바꾸면 보인다",
              color=INK, fontsize=11.5, pad=9)
axR.legend(loc="upper left", fontsize=9, framealpha=0.95)
axR.text(len(vals) - 1, FLmax + 1.35,
         f"$= (k-1)\\times F = 3 \\times {F:.2f}$", ha="center", fontsize=9,
         color=BLUE)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
path = OUT + "scheffe_max_contrast.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
plt.close(fig)
print("saved", path)
