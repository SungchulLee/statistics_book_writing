"""11.2절 심슨의 역설 — 합친 수는 「어느 학과에 지원했는지」까지 함께 잰다.

쪽의 표에 있는 1973 년 가을 버클리 여섯 학과 자료를 그대로 쓴다.
학과마다 합격률이 크게 다른데(A 0.643 에서 F 0.065) 여성 지원자는
합격률이 낮은 학과에 몰려 있다. 그래서 학과별로는 여성이 같거나 더
높은데 합치면 뒤집힌다.

쪽의 berkeley_levels 는 학과별 합격률과 합친 점을 견준다. 여기서는 그
뒤집힘을 만든 것 --- 지원이 어디로 갔는가 --- 만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

OUT = "docs/ch11/ecological_correlation/img/"
DEPT = list("ABCDEF")
nm = np.array([825, 560, 325, 417, 191, 373], float)
rm = np.array([0.62, 0.63, 0.37, 0.33, 0.28, 0.06])
nw = np.array([108, 25, 593, 375, 393, 341], float)
rw = np.array([0.82, 0.68, 0.34, 0.35, 0.24, 0.07])

tot = nm + nw
adm = nm * rm + nw * rw
rate = adm / tot                      # 학과의 전체 합격률
share_w = nw / tot                    # 그 학과 지원자 중 여성 비율
corr = float(np.corrcoef(rate, share_w)[0, 1])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.7),
                               gridspec_kw={"width_ratios": [1, 1.15]})

ax1.plot(100 * rate, 100 * share_w, "o", ms=13, color="tab:purple")
for d, x, y in zip(DEPT, 100 * rate, 100 * share_w):
    ax1.text(x, y + 3.4, d, ha="center", fontsize=14, fontweight="bold",
             color="0.15")
z = np.polyfit(100 * rate, 100 * share_w, 1)
g = np.linspace(2, 70, 50)
ax1.plot(g, np.polyval(z, g), "--", color="0.5", lw=1.6)
ax1.set_xlim(0, 72); ax1.set_ylim(0, 82)
ax1.set_xlabel("학과의 전체 합격률 (%)", fontsize=11)
ax1.set_ylabel("그 학과 지원자 중 여성 비율 (%)", fontsize=11)
ax1.set_title(f"들어가기 쉬운 학과일수록 여성이 적다 ($r$ = {corr:.3f})",
              fontsize=12.5)
ax1.grid(alpha=0.25)

x = np.arange(6); w = 0.38
ax2.bar(x - w / 2, 100 * nm / nm.sum(), w, color="tab:blue", alpha=0.85,
        label=f"남성 지원 {nm.sum():.0f} 명이 간 곳")
ax2.bar(x + w / 2, 100 * nw / nw.sum(), w, color="tab:orange", alpha=0.85,
        label=f"여성 지원 {nw.sum():.0f} 명이 간 곳")
for xi, (a, b) in enumerate(zip(100 * nm / nm.sum(), 100 * nw / nw.sum())):
    ax2.text(xi - w / 2, a + 0.9, f"{a:.0f}", ha="center", fontsize=10,
             color="tab:blue", fontweight="bold")
    ax2.text(xi + w / 2, b + 0.9, f"{b:.0f}", ha="center", fontsize=10,
             color="tab:orange", fontweight="bold")
ax2.set_xticks(x, [f"{d}\n({100*r:.0f}%)" for d, r in zip(DEPT, rate)],
               fontsize=11)
ax2.set_xlabel("학과 (괄호는 그 학과의 전체 합격률)", fontsize=11)
ax2.set_ylabel("자기 성별 지원자 가운데 비율 (%)", fontsize=11)
ax2.set_ylim(0, 42)
ax2.set_title("지원이 어디로 갔는가", fontsize=12.5)
ax2.legend(fontsize=10.5, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

fig.suptitle("합친 수는 합격률만 재는 것이 아니라 "
             "어느 학과에 지원했는지까지 함께 잰다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "where_they_applied.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  전체 남성 {nm.sum():.0f} 명 중 합격 {(nm*rm).sum():.0f} "
      f"({(nm*rm).sum()/nm.sum():.4%})")
print(f"  전체 여성 {nw.sum():.0f} 명 중 합격 {(nw*rw).sum():.0f} "
      f"({(nw*rw).sum()/nw.sum():.4%})")
for d, t, r, s in zip(DEPT, tot, rate, share_w):
    print(f"  {d}: 지원 {t:.0f} · 전체 합격률 {r:.4f} · 여성 비율 {s:.4f}")
print(f"  합격률과 여성 비율의 상관 {corr:.4f}")
