r"""6.3절 「그림으로 보는 통계학」 — 기다린 사람만 남는다.

기하분포의 가장 오래된 응용 하나는 **임신까지 걸린 주기 수**다. 한 주기에 임신할
확률 $p$ 가 모두 같다면 주기 수는 기하분포를 따르고, MLE 는 $\hat p = 1/\bar X$ 다.
그런데 사람마다 $p$ 가 다르면 **주기가 갈수록 남아 있는 사람들의 평균 $p$ 가 낮아진다** —
위험률이 떨어지고, 하나의 $p$ 로 적합한 값은 평균보다 작아진다.

① 주기별 임신 비율 — 평평한 기하 대 떨어지는 베타–기하.
② 남은 사람들의 평균 가임력이 주기마다 내려간다.
③ 하나의 $p$ 로 적합하면 평균이 아니라 더 작은 수가 나온다.

만드는 파일: docs/ch06/mle/img/fecundability.png
실행:  python3 scripts/make_ch06_fecundability.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch06/mle/img/fecundability.png"
RNG = np.random.default_rng(20261022)

MEAN_P = 0.30            # 평균 가임력
A = 2.0                  # 베타분포의 모양 — 평균을 0.30 에 맞춘다
B = A * (1 - MEAN_P) / MEAN_P
N = 200_000
CYC = 12

print(f"베타({A:.1f}, {B:.2f}) — 평균 {A/(A+B):.3f}, "
      f"표준편차 {np.sqrt(A*B/((A+B)**2*(A+B+1))):.3f}")

p_ind = RNG.beta(A, B, N)                      # 사람마다 다른 가임력
x_het = RNG.geometric(p_ind)                   # 이질적인 세상
x_hom = RNG.geometric(MEAN_P, N)               # 모두 같은 세상

haz_het, haz_hom, mean_left = [], [], []
for c in range(1, CYC + 1):
    left_h = x_het >= c
    left_o = x_hom >= c
    haz_het.append(np.mean(x_het[left_h] == c))
    haz_hom.append(np.mean(x_hom[left_o] == c))
    mean_left.append(p_ind[left_h].mean())
    if c in (1, 2, 3, 6, 12):
        print(f"주기 {c:>2}:  그 주기에 임신한 비율 {haz_het[-1]:.3f} "
              f"(모두 같다면 {haz_hom[-1]:.3f}),  남은 사람들의 평균 가임력 "
              f"{mean_left[-1]:.3f}")

p_mle = 1 / x_het.mean()
print(f"\n하나의 p 로 적합한 MLE = 1/x̄ = {p_mle:.4f}  (참 평균 {MEAN_P:.3f})")
print(f"   이론값 1/E[1/P] = {(A - 1) / (A + B - 1):.4f}")
print(f"   {CYC} 주기에서 끊고(중도절단) 센 평균 가임력 = "
      f"{1/np.minimum(x_het, CYC).mean():.4f}")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
C1, C2 = "#1f6f8b", "#c0532b"
cyc = np.arange(1, CYC + 1)

ax = axes[0]
ax.plot(cyc, haz_hom, "s--", color="0.45", lw=1.8,
        label=f"모두 p = {MEAN_P:.2f} 이라면")
ax.plot(cyc, haz_het, "o-", color=C1, lw=2.2, label="사람마다 다르다면")
ax.set_xlabel("주기")
ax.set_ylabel("그 주기에 임신할 비율 (아직 안 된 사람 가운데)")
ax.set_ylim(0, 0.38)
ax.set_title("① 위험률이 평평하지 않다", fontsize=11)
ax.legend(fontsize=9)

ax = axes[1]
ax.plot(cyc, mean_left, "o-", color=C2, lw=2.2)
ax.axhline(MEAN_P, color="0.4", ls="--", lw=1.3)
ax.text(CYC, MEAN_P - 0.006, f"처음의 평균 {MEAN_P:.2f}", color="0.35",
        fontsize=9, ha="right", va="top")
ax.set_xlabel("주기")
ax.set_ylabel("아직 임신하지 않은 사람들의 평균 가임력")
ax.set_title("② 남은 사람들이 달라진다", fontsize=11)
ax.text(0.05, 0.12, f"주기 {CYC} 에 남은 사람들의\n평균 {mean_left[-1]:.3f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[2]
bins = np.arange(0.5, 25.5)
ax.hist(np.minimum(x_het, 24), bins=bins, density=True, color="0.78",
        edgecolor="white", label="관측된 주기 수")
g = np.arange(1, 25)
ax.plot(g, p_mle * (1 - p_mle) ** (g - 1), "o-", color=C2, lw=2,
        label=f"하나의 p 로 적합  ($\\hat p$ = {p_mle:.3f})")
ax.plot(g, MEAN_P * (1 - MEAN_P) ** (g - 1), "s--", color=C1, lw=1.8,
        label=f"평균 가임력을 쓴 기하분포 (p = {MEAN_P:.2f})")
ax.set_xlabel("임신까지 걸린 주기 수  (24 이상은 한 칸에 모았다)")
ax.set_ylabel("비율")
ax.set_title("③ 하나의 p 로 적합하면", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
