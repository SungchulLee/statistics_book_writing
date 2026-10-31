"""9.2절 뮤에 대한 t 검정 — 교과서의 37.0 을 반박하는 수.

8.2절이 체온 자료에 폭을 붙였다면 이 쪽은 판정을 붙인다. 성인 148 명의
평균 36.8, 되짚은 산포 0.46 으로 t = -5.2894 이고 양측 p = 4.4e-7 이다.
귀무분포 t(147) 위에 그 수를 얹으면 꼬리가 보이지도 않는 자리다.

쪽의 body_temperature 는 평균의 구간과 한 사람의 범위와 하루 안의 변동을
다루고 t_vs_z_critical 은 t 와 z 의 임계값을 견준다. 여기서는 귀무분포
위에서 관측값이 선 자리만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/one_sample_tests/img/"
N, XBAR, S, MU0 = 148, 36.8, 0.46, 37.0
DF = N - 1
SE = S / np.sqrt(N)
T = (XBAR - MU0) / SE
P = 2 * stats.t.sf(abs(T), DF)
TC = stats.t.ppf(0.975, DF)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1.3, 1]})

g = np.linspace(-6.4, 6.4, 1600)
d = stats.t.pdf(g, DF)
ax1.plot(g, d, color="0.25", lw=2.2)
ax1.fill_between(g[g <= -TC], d[g <= -TC], color="tab:red", alpha=0.55)
ax1.fill_between(g[g >= TC], d[g >= TC], color="tab:red", alpha=0.55)
ax1.axvline(T, color="tab:blue", lw=2.6)
ax1.plot([T], [0], "v", ms=14, color="tab:blue", clip_on=False, zorder=5)
ax1.annotate(f"관측된 $t$ = {T:.4f}", xy=(T, 0.045), xytext=(-5.9, 0.27),
             fontsize=12.5, color="tab:blue", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:blue", lw=1.8))
for s_ in (-1, 1):
    ax1.annotate(f"{s_*TC:+.4f}", xy=(s_ * TC, 0.012),
                 xytext=(s_ * TC + s_ * 0.9, 0.085), fontsize=10.5,
                 color="tab:red", ha="center",
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.2))
ax1.text(0.0, 0.20, f"귀무가설이 참이라면\n$t$ 는 여기 떨어진다",
         ha="center", fontsize=11.5, color="0.3")
ax1.set_xlim(-6.6, 6.6)
ax1.set_ylim(0, 0.42)
ax1.set_xlabel(f"$t$ — 자유도 {DF}", fontsize=11)
ax1.set_ylabel("밀도", fontsize=11)
ax1.set_title(f"$H_0$: 평균이 {MU0} 라면", fontsize=12.5)
ax1.grid(alpha=0.2)

mags = [1, 2, 3, 4, 5, abs(T)]
ps = [2 * stats.t.sf(m, DF) for m in mags]
y = np.arange(len(mags))[::-1]
cols = ["0.65"] * 5 + ["tab:blue"]
ax2.barh(y, ps, height=0.56, color=cols, alpha=0.88)
for yy, m, pv in zip(y, mags, ps):
    ax2.text(max(pv, 2e-7) * 1.5, yy, f"{pv:.3g}", va="center", fontsize=11,
             fontweight="bold", color="0.2")
ax2.axvline(0.05, color="tab:red", ls="--", lw=1.8)
ax2.text(0.058, len(mags) - 1.35, "0.05", ha="left", fontsize=11,
         color="tab:red", fontweight="bold")
ax2.set_yticks(y, [f"$|t|$ = {m:.4f}" if m > 5.1 else f"$|t|$ = {m}"
                   for m in mags], fontsize=11)
ax2.set_xscale("log")
ax2.set_xlim(1e-7, 2.0)
TK = [1e-7, 1e-5, 1e-3, 1e-1, 1]
ax2.set_xticks(TK, ["0.0000001", "0.00001", "0.001", "0.1", "1"], fontsize=9)
ax2.set_xticks([], minor=True)
ax2.set_xlabel("양측 $p$ 값 (로그 눈금)", fontsize=11)
ax2.set_title("$|t|$ 가 조금만 커져도 $p$ 는 자릿수로 떨어진다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle(f"성인 {N} 명의 평균 {XBAR} 로 교과서의 {MU0} 를 반박한다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "body_temp_t.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  n={N}, xbar={XBAR}, s={S}, mu0={MU0}")
print(f"  표준오차 {SE:.4f} · t = {T:.4f} · 양측 p = {P:.3g} · "
      f"임계값 ±{TC:.4f}")
