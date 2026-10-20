r"""$0.05$ 를 $0.005$ 로 낮추자는 제안 — 그 수들을 직접 재 본다.

① $p$ 값 하나가 담을 수 있는 **증거의 상한**(보브크–셀케 경계).
② 사전 승산에 따른 「유의하면 참일 확률」 — $0.05$ 와 $0.005$.
③ 검정력을 지키려면 표본을 얼마나 더 모아야 하는가.

만드는 파일: docs/ch09/foundations/img/alpha_debate.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

E = np.e


# === ① 증거의 상한 ==================================================
# p < 1/e 에서, H0 에 맞서는 베이즈 인수는 어떤 대립가설을 고르더라도
#   BF <= -1 / (e p ln p)
# 를 넘지 못한다. p 값이 담을 수 있는 증거의 천장이다.
def bf_bound(p):
    p = np.asarray(p, dtype=float)
    return -1.0 / (E * p * np.log(p))


# === ② 유의하면 참일 확률 ===========================================
def post(prior_odds, alpha, power):
    return prior_odds * power / (prior_odds * power + alpha)


# === ③ 표본을 얼마나 더 ==============================================
def n_ratio(a_new, a_old, power=0.8):
    zb = stats.norm.isf(1 - power)
    return (
        (stats.norm.isf(a_new / 2) + zb) / (stats.norm.isf(a_old / 2) + zb)
    ) ** 2


# 표본을 그대로 두고 문턱만 낮추면 검정력이 얼마가 되는가
DELTA = stats.norm.isf(0.025) + stats.norm.isf(0.2)  # 검정력 0.8 을 주는 효과크기
PW_SAME_N = float(stats.norm.sf(stats.norm.isf(0.0025) - DELTA))

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.9))

ps = np.logspace(np.log10(2e-4), np.log10(0.2), 400)
ax1.loglog(ps, bf_bound(ps), color="#1f77b4", lw=2.6)
for p0, c, fx, fy in [(0.05, "#d62728", 0.60, 2.3), (0.005, "#2ca02c", 0.33, 0.42)]:
    b = float(bf_bound(p0))
    ax1.plot([p0], [b], "o", color=c, ms=9, zorder=5)
    ax1.annotate(
        f"$p = {p0}$\n$\\mathrm{{BF}} \\leq {b:.1f}$",
        xy=(p0, b),
        xytext=(p0 * fx, b * fy),
        ha="center",
        fontsize=10,
        color=c,
    )
for lev, lab in [(3, "「거의 증거 없음」"), (10, "「상당한 증거」"), (30, "「강한 증거」")]:
    ax1.axhline(lev, color="#999999", lw=0.9, ls=":")
    ax1.text(1.7e-4, lev * 1.12, lab, fontsize=9, color="#777777")
ax1.set_xlabel("$p$ 값")
ax1.set_ylabel("베이즈 인수의 상한")
ax1.set_title("① $p$ 값이 담을 수 있는 증거의 천장", fontsize=12)
ax1.set_xticks([1e-3, 1e-2, 1e-1])
ax1.set_xticklabels(["$0.001$", "$0.01$", "$0.1$"])
ax1.set_yticks([1, 10, 100])
ax1.set_yticklabels(["$1$", "$10$", "$100$"])
ax1.minorticks_off()
ax1.set_xlim(1.6e-4, 0.25)
ax1.set_ylim(0.8, 400)
ax1.grid(alpha=0.25, which="major")

odds = np.logspace(np.log10(1 / 40), np.log10(1.2), 300)
for a, pw, c, lab in [
    (0.05, 0.80, "#d62728", "$\\alpha = 0.05$, 검정력 $0.80$"),
    (0.005, 0.80, "#2ca02c", "$\\alpha = 0.005$, 검정력 $0.80$"),
    (0.005, PW_SAME_N, "#ff7f0e", f"$\\alpha = 0.005$, 표본 그대로 (검정력 ${PW_SAME_N:.2f}$)"),
]:
    ax2.semilogx(odds, post(odds, a, pw) * 100, color=c, lw=2.5, label=lab)
ax2.axhline(50, color="#555555", lw=1.0, ls="--")
for o, lab in [(1 / 10, "$1:10$"), (1 / 5, "$1:5$")]:
    ax2.axvline(o, color="#999999", lw=0.9, ls=":")
    ax2.text(o * 1.06, 4, lab, fontsize=9.5, color="#777777")
ax2.set_xlabel("사전 승산 (참인 가설 : 거짓인 가설)")
ax2.set_ylabel("유의한 결과가 참일 확률 ($\\%$)")
ax2.set_title("② 문턱을 낮추면 믿을 만해진다", fontsize=12)
ax2.set_xticks([0.025, 0.1, 0.5, 1.0])
ax2.set_xticklabels(["$1\\!:\\!40$", "$1\\!:\\!10$", "$1\\!:\\!2$", "$1\\!:\\!1$"])
ax2.minorticks_off()
ax2.set_ylim(0, 100)
ax2.legend(loc="upper left", fontsize=9)
ax2.grid(alpha=0.25)

alphas = np.logspace(np.log10(0.05), np.log10(0.0005), 200)
for pw, c in [(0.80, "#1f77b4"), (0.90, "#9467bd")]:
    ax3.semilogx(alphas, n_ratio(alphas, 0.05, pw), color=c, lw=2.5, label=f"검정력 ${pw}$")
r = float(n_ratio(0.005, 0.05, 0.8))
ax3.plot([0.005], [r], "o", color="#2ca02c", ms=9, zorder=5)
ax3.annotate(
    f"$0.005$ 로 가려면\n표본을 ${(r - 1) * 100:.0f}\\%$ 더",
    xy=(0.005, r),
    xytext=(0.0011, 1.06),
    ha="left",
    fontsize=10,
    color="#2ca02c",
    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.1),
)
ax3.axhline(1, color="#555555", lw=1.0, ls="--")
ax3.set_xlabel("새 유의수준 $\\alpha$")
ax3.set_ylabel("필요한 표본의 배수  ($\\alpha = 0.05$ 기준)")
ax3.set_title("③ 값은 표본으로 치른다", fontsize=12)
ax3.set_xticks([0.05, 0.01, 0.005, 0.001])
ax3.set_xticklabels(["$0.05$", "$0.01$", "$0.005$", "$0.001$"])
ax3.minorticks_off()
ax3.set_ylim(0.95, 2.6)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch09/foundations/img/alpha_debate.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① 증거의 상한  BF <= -1/(e p ln p)")
for p0 in (0.05, 0.01, 0.005, 0.001, 0.0001):
    print(f"  p = {p0:<8}:  BF <= {float(bf_bound(p0)):8.2f}")

print("\n③ 표본의 배수 (검정력 0.8 유지)")
for a in (0.01, 0.005, 0.001):
    print(f"  0.05 → {a:<7}:  {float(n_ratio(a, 0.05)):.4f} 배  ({(float(n_ratio(a, 0.05)) - 1) * 100:+.0f}%)")

print("\n②  표본을 늘리지 않고 문턱만 0.005 로 낮추면 검정력이 얼마가 되는가")
print(f"  검정력 0.80 → {PW_SAME_N:.4f}")
pw_new = PW_SAME_N

print("\n② 유의하면 참일 확률")
print(f"{'사전 승산':>10}{'α=0.05 (0.80)':>16}{'α=0.005 (0.80)':>17}{'α=0.005 (그대로)':>19}")
for o, lab in [(1 / 40, "1:40"), (1 / 10, "1:10"), (1 / 5, "1:5"), (1 / 2, "1:2")]:
    print(
        f"{lab:>10}{post(o, .05, .8) * 100:>15.1f}%{post(o, .005, .8) * 100:>16.1f}%"
        f"{post(o, .005, pw_new) * 100:>18.1f}%"
    )
