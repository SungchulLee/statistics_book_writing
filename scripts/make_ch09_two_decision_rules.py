"""9.1절 유의수준과 판정 규칙 §1 — p 값 접근과 임계값 접근은 같은 사건을 본다.

§1 의 풀이(σ = 10, n = 25, μ0 = 50, 양측 α = 0.05)를 그대로 쓴다.
왼쪽은 x̄ = 53.5 (z = 1.75, 기각 못함), 오른쪽은 같은 설정에서 x̄ = 54.6
이었다면 (z = 2.30, 기각). 붉게 칠한 것이 기각역(넓이 α), 빗금이 관측값
바깥의 넓이(p 값)다. 관측값이 임계값 바깥에 있는 것과 빗금 넓이가 붉은
넓이보다 작은 것이 같은 사건임을 보인다.

쪽의 rejection_region_demo_51 은 기각역과 통계량만 그리고 p 값의 넓이는
그리지 않는다. 여기서는 두 넓이를 한 그림에 겹친다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/foundations/img/"
MU0, SIGMA, N, ALPHA = 50, 10, 25, 0.05
SE = SIGMA / np.sqrt(N)
ZC = stats.norm.ppf(1 - ALPHA / 2)
CASES = [(53.5, "기각하지 못한다"), (54.6, "기각한다")]

x = np.linspace(-3.6, 3.6, 900)
y = stats.norm.pdf(x)

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6), sharey=True)
for ax, (xbar, verdict) in zip(axes, CASES):
    z = (xbar - MU0) / SE
    p = 2 * stats.norm.sf(abs(z))
    print(f"x-bar = {xbar}: z = {z:.4f}, p = {p:.6f}, z_crit = {ZC:.6f}")
    ax.plot(x, y, color="0.2", lw=1.8)
    for sgn in (-1, 1):
        m = x * sgn >= ZC
        ax.fill_between(x[m], y[m], color="tab:red", alpha=0.35, lw=0,
                        label="기각역 (넓이 α = 0.05)" if sgn == 1 else None)
        m = x * sgn >= abs(z)
        ax.fill_between(x[m], y[m], facecolor="none", edgecolor="tab:blue",
                        hatch="////", lw=0,
                        label=f"관측값 바깥 (넓이 p = {p:.4f})" if sgn == 1 else None)
        ax.plot([sgn * ZC] * 2, [0, 0.30], color="tab:red", lw=1.6, ls="--")
        ax.plot([sgn * abs(z)] * 2, [0, 0.22], color="tab:blue", lw=2.2)
    ax.text(ZC, 0.31, f"임계값 {ZC:.2f}", ha="center", va="bottom",
            fontsize=11, color="tab:red", fontweight="bold")
    ax.text(-ZC, 0.31, f"-{ZC:.2f}", ha="center", va="bottom",
            fontsize=11, color="tab:red", fontweight="bold")
    inside = abs(z) < ZC
    # 글자가 임계값 점선과 겹치지 않도록 점선 반대쪽으로 붙인다
    ax.text(abs(z) + (-0.06 if inside else 0.06), 0.23, f"z = {z:.2f}",
            ha="right" if inside else "left", va="bottom",
            fontsize=11, color="tab:blue", fontweight="bold")
    ax.set_title(
        f"표본평균 {xbar}:  |z| = {abs(z):.2f} {'<' if inside else '>'} {ZC:.2f}"
        f"  그리고  p = {p:.4f} {'>' if inside else '<'} 0.05  →  {verdict}",
        fontsize=11.5)
    ax.set_xlabel("검정통계량 z", fontsize=11)
    ax.set_ylim(0, 0.43)
    ax.set_xlim(-3.6, 3.6)
    ax.legend(loc="upper left", fontsize=9.5, frameon=False)
    ax.grid(axis="x", alpha=0.25)
axes[0].set_ylabel("$H_0$ 아래의 밀도", fontsize=11)

fig.tight_layout()
fig.savefig(OUT + "two_decision_rules.png", dpi=150)
print("saved", OUT + "two_decision_rules.png")
