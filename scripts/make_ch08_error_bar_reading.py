r"""「간신히 유의한」 자리는 정확히 어디인가 — 눈대중이 아니라 셈으로.

표준오차가 같은 두 평균 $\bar x_A, \bar x_B$ 를 생각한다. 차이를 $d$, 각 평균의
표준오차를 $s$ 라 하면 독립일 때 $\mathrm{SE}_d = s\sqrt{2}$ 이고 상관 $\rho$ 로 짝지어져
있으면 $\mathrm{SE}_d = s\sqrt{2(1-\rho)}$ 이다. 유의성은 $|d| / \mathrm{SE}_d$ 로 정해지는데,
그림에 그려진 막대는 $s$ 만 보여 준다.

① 겹침과 $p$ 값의 관계 — $95\%$ CI 막대와 SE 막대.
② 독립일 때 「간신히 유의한」 두 쌍의 모습.
③ 짝지어져 있으면 같은 그림이 전혀 다른 뜻이 된다.

만드는 파일: docs/ch08/foundations/img/error_bar_reading.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

Z95 = stats.norm.isf(0.025)          # 1.959964
S = 1.0                              # 평균 하나의 표준오차를 1 로 둔다


def pval(d, rho=0.0):
    """표준오차가 같은 두 평균의 차이 d 에 대한 양측 p 값."""
    se_d = S * np.sqrt(2 * (1 - rho))
    return 2 * stats.norm.sf(np.abs(d) / se_d)


def d_crit(rho=0.0, alpha=0.05):
    """간신히 유의해지는 차이."""
    return stats.norm.isf(alpha / 2) * S * np.sqrt(2 * (1 - rho))


def overlap_per_arm(d, arm):
    """두 막대가 겹친 길이를 팔 하나의 길이로 나눈 값 (음수면 떨어져 있다)."""
    return (2 * arm - d) / arm


ARM_CI, ARM_SE = Z95 * S, S

# === 셈 ============================================================
d05 = d_crit()                                   # 독립, p = .05
ov_ci_05 = overlap_per_arm(d05, ARM_CI)
ov_se_05 = overlap_per_arm(d05, ARM_SE)
p_touch_ci = pval(2 * ARM_CI)                    # CI 막대가 딱 붙을 때
p_touch_se = pval(2 * ARM_SE)                    # SE 막대가 딱 붙을 때

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ov = np.linspace(-1.6, 2.0, 400)
for arm, c, lab in ((ARM_CI, "#1f77b4", "$95\\%$ CI 막대"),
                    (ARM_SE, "#d62728", "SE 막대")):
    d = (2 - ov) * arm
    ax1.semilogy(ov, pval(d), color=c, lw=2.6, label=lab)
ax1.axhline(0.05, color="black", lw=1.5, ls="--")
ax1.text(-1.55, 0.057, "$p = 0.05$", fontsize=10)
ax1.axvline(0, color="#999999", lw=1.2, ls=":")
ax1.text(0.04, 3e-4, "막대가 딱 붙는 자리", rotation=90, fontsize=9.5,
         color="#666666", va="bottom")
for x, c in ((ov_ci_05, "#1f77b4"), (ov_se_05, "#d62728")):
    ax1.plot([x], [0.05], "o", color=c, ms=8, zorder=5)
ax1.annotate(f"CI 막대는 팔 길이의\n${ov_ci_05:.2f}$ 배까지 겹쳐도 유의",
             xy=(ov_ci_05, 0.05), xytext=(0.78, 0.0035), fontsize=9.5,
             color="#1f77b4", ha="left",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
ax1.annotate(f"SE 막대는 ${-ov_se_05:.2f}$ 배\n**떨어져야** 유의".replace("**", ""),
             xy=(ov_se_05, 0.05), xytext=(-1.55, 0.0012), fontsize=9.5,
             color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax1.set_xlabel("겹친 길이 $\\div$ 팔 하나의 길이  (음수면 떨어져 있다)")
ax1.set_ylabel("두 평균 차이의 $p$ 값")
ax1.set_title("① 같은 겹침, 전혀 다른 $p$", fontsize=12)
ax1.set_ylim(1e-4, 1.3)
ax1.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
ax1.set_yticklabels(["$10^{-4}$", "$0.001$", "$0.01$", "$0.1$", "$1$"])
ax1.minorticks_off()
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(alpha=0.25, which="major")


def draw_pair(ax, y, d, arm, c, lab):
    """평균 두 개와 오차막대를 가로로 그린다."""
    for x, m in ((0.0, "o"), (d, "s")):
        ax.plot([x - arm, x + arm], [y, y], color=c, lw=2.6,
                solid_capstyle="butt")
        for e in (x - arm, x + arm):
            ax.plot([e, e], [y - 0.13, y + 0.13], color=c, lw=2.6)
        ax.plot([x], [y], m, color=c, ms=8)
    ax.text(-1.05 * ARM_CI - 0.5, y, lab, ha="right", va="center", fontsize=10)


for arm, c, lab, y in ((ARM_CI, "#1f77b4", "$95\\%$ CI 막대", 1.0),
                       (ARM_SE, "#d62728", "SE 막대", 0.0)):
    draw_pair(ax2, y, d05, arm, c, lab)
    o = overlap_per_arm(d05, arm)
    txt = (f"팔 길이의 ${o:.2f}$ 배 겹친다" if o > 0
           else f"팔 길이의 ${-o:.2f}$ 배 떨어진다")
    ax2.text(d05 / 2, y + 0.36, txt, ha="center", fontsize=10, color=c)
ax2.set_xlim(-6.2, 7.0)
ax2.set_ylim(-0.75, 1.75)
ax2.set_yticks([])
ax2.set_xlabel("평균 (표준오차를 $1$ 로 둔 눈금)")
ax2.set_title(f"② 둘 다 $p = 0.05$ 다 ($d = {d05:.2f}$)", fontsize=12)
ax2.grid(alpha=0.25, axis="x")

RHOS = (0.0, 0.5, 0.8, 0.95)
for i, rho in enumerate(RHOS):
    y = len(RHOS) - 1 - i
    dc = d_crit(rho)
    draw_pair(ax3, y, dc, ARM_CI, "#1f77b4", f"$\\rho = {rho}$")
    o = overlap_per_arm(dc, ARM_CI)
    ax3.text(dc / 2, y + 0.33, f"${o:.2f}$ 배 겹침", ha="center", fontsize=10,
             color="#1f77b4")
ax3.set_xlim(-6.2, 7.0)
ax3.set_ylim(-0.75, len(RHOS) - 0.25)
ax3.set_yticks([])
ax3.set_xlabel("평균 (표준오차를 $1$ 로 둔 눈금)")
ax3.set_title("③ 짝지어져 있으면 — 전부 $p = 0.05$ 다", fontsize=12)
ax3.grid(alpha=0.25, axis="x")

fig.tight_layout()
out = "docs/ch08/foundations/img/error_bar_reading.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n표준오차 s = 1, 독립 두 평균 — 간신히 유의한 차이 d = {d05:.4f}")
print(f"  $95\\%$ CI 막대 (팔 {ARM_CI:.3f}): 겹침 {ov_ci_05:+.3f} 배")
print(f"  SE 막대       (팔 {ARM_SE:.3f}): 겹침 {ov_se_05:+.3f} 배")
print(f"\n막대가 딱 붙을 때의 p 값")
print(f"  95% CI 막대: p = {p_touch_ci:.4f}")
print(f"  SE 막대    : p = {p_touch_se:.4f}")
print(f"\nSE 막대가 팔 길이만큼(=1) 떨어질 때: p = {pval(3 * ARM_SE):.4f}")

print(f"\n③ 상관에 따라 — 간신히 유의한 d 와 그때 95% CI 막대의 겹침")
print(f"{'rho':>6}{'SE_d':>9}{'d (p=.05)':>12}{'겹침/팔':>10}")
for rho in (0.0, 0.3, 0.5, 0.8, 0.95):
    dc = d_crit(rho)
    print(f"{rho:>6.2f}{np.sqrt(2 * (1 - rho)):>9.3f}{dc:>12.3f}"
          f"{overlap_per_arm(dc, ARM_CI):>10.3f}")

# --- 셈이 맞는지 모의로 확인 ------------------------------------------
rng = np.random.default_rng(20261020)
n, R = 400, 200_000
for rho in (0.0, 0.8):
    # 두 평균의 표준오차가 각각 1 이 되도록 맞춘 정규 자료
    sd = np.sqrt(n)
    z1 = rng.standard_normal((R,))
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal((R,))
    d_obs = z1 - z2 + d_crit(rho)      # 참 차이를 임계값에 둔다
    se_d = np.sqrt(2 * (1 - rho))
    rej = np.mean(2 * stats.norm.sf(np.abs(d_obs) / se_d) < 0.05)
    print(f"  점검: rho = {rho}, 참 차이를 임계값에 두면 기각률 {rej * 100:.1f}% "
          f"(이론 50%)")
