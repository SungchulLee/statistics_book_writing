r"""A/B 검정의 세 가지 산술 — 얼마나 커야 하고, 여럿을 재면 어떻게 되고, 이긴 것을 믿어도 되는가.

① 전환율 $5\%$ 에서 $0.1\%$p 를 잡으려면 팔당 몇 명이 필요한가.
② 똑같은 변형 $k$ 개를 한꺼번에 재면 「승자」가 나올 확률과 그 승자의 부풀림.
③ 아이디어가 참일 사전확률이 낮으면 유의한 결과도 못 믿는다.

만드는 파일: docs/ch12/practical_applications/img/ab_scale.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats, integrate  # noqa: E402

ZA = stats.norm.isf(0.025)  # 1.96
ZB = stats.norm.isf(0.20)  # 0.84
BASE = 0.05  # 기준 전환율


# === ① 필요한 표본 ==================================================
def n_per_arm(delta, p0=BASE, za=ZA, zb=ZB):
    p1 = p0 + delta
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / delta**2


# === ② k 개를 한꺼번에 재면 =========================================
def fwer(k, alpha=0.05):
    return 1 - (1 - alpha) ** k


def e_max(k):
    """표준정규 k 개의 최댓값의 기댓값 — 수치적분으로."""
    f = lambda z: z * k * stats.norm.pdf(z) * stats.norm.cdf(z) ** (k - 1)
    return integrate.quad(f, -10, 12)[0]


# === ③ 이긴 것을 믿어도 되는가 ======================================
def post(prior, power, alpha=0.05):
    return prior * power / (prior * power + (1 - prior) * alpha)


# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.0, 4.8))

ds = np.logspace(np.log10(0.0005), np.log10(0.03), 300)
ax1.loglog(ds * 100, n_per_arm(ds), color="#1f77b4", lw=2.4)
for d, c in [(0.001, "#d62728"), (0.005, "#ff7f0e"), (0.02, "#2ca02c")]:
    ax1.plot([d * 100], [n_per_arm(d)], "o", color=c, ms=8, zorder=5)
    ax1.annotate(
        f"$+{d * 100:g}\\%$p\n{n_per_arm(d):,.0f} 명",
        xy=(d * 100, n_per_arm(d)),
        xytext=(d * 100 * 1.3, n_per_arm(d) * 1.5),
        fontsize=9.5,
        color=c,
    )
ax1.set_xlabel("잡으려는 전환율 상승 ($\\%$p)   기준 $5\\%$")
ax1.set_ylabel("팔당 필요한 표본")
ax1.set_title("① 작은 개선일수록 제곱으로 비싸진다", fontsize=12)
ax1.set_xticks([0.1, 1.0])
ax1.set_xticklabels(["$0.1$", "$1$"])
ax1.set_yticks([1e3, 1e5, 1e7])
ax1.set_yticklabels(["$10^3$", "$10^5$", "$10^7$"])
ax1.minorticks_off()
ax1.grid(alpha=0.25, which="both")

ks = np.arange(1, 61)
ax2.plot(ks, [fwer(k) for k in ks], color="#d62728", lw=2.4, label="승자가 나올 확률")
ax2.set_xlabel("한꺼번에 견주는 변형의 수 $k$")
ax2.set_ylabel("확률", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax2.set_ylim(0, 1.05)
ax2.set_xlim(0, 61)
ax2.grid(alpha=0.25)
ax2b = ax2.twinx()
ax2b.plot(ks, [e_max(k) for k in ks], color="#1f77b4", lw=2.4, label="승자의 부풀림")
ax2b.set_ylabel("$E[\\max Z]$ (표준오차 단위)", color="#1f77b4")
ax2b.tick_params(axis="y", labelcolor="#1f77b4")
ax2b.set_ylim(0, 3.0)
for a, y, c, txt, tx, ty, ha in [
    (ax2, fwer(41), "#d62728", f"${fwer(41):.3f}$", 38.5, 0.93, "right"),
    (ax2b, e_max(41), "#1f77b4", f"${e_max(41):.2f}\\,$SE", 43.5, 1.93, "left"),
]:
    a.plot([41], [y], "o", color=c, ms=8, zorder=5)
    a.annotate(txt, xy=(41, y), xytext=(tx, ty), ha=ha, fontsize=10.5, color=c)
ax2.axvline(41, color="#777777", lw=1.0, ls=":")
ax2.text(39, 0.06, "파란색 $41$ 가지", ha="right", fontsize=10, color="#555555")
ax2.set_title("② 모두 똑같아도 승자는 나온다", fontsize=12)

pri = np.linspace(0.01, 0.6, 300)
for pw, c in [(0.8, "#1f77b4"), (0.5, "#ff7f0e"), (0.3, "#d62728")]:
    ax3.plot(pri * 100, post(pri, pw) * 100, color=c, lw=2.4, label=f"검정력 ${pw}$")
for x, lab in [(1 / 3, "마이크로소프트 $1/3$"), (0.15, "빙 $15\\%$")]:
    ax3.axvline(x * 100, color="#777777", lw=1.0, ls=":")
    ax3.text(x * 100 + 1.2, 16, lab, rotation=90, fontsize=9.5, color="#555555")
ax3.axhline(50, color="#555555", lw=1.0, ls="--")
ax3.set_xlabel("아이디어가 참으로 효과 있을 비율 ($\\%$)")
ax3.set_ylabel("유의한 결과가 참일 확률 ($\\%$)")
ax3.set_title("③ 이긴 것을 믿어도 되는가", fontsize=12)
ax3.set_xlim(0, 60)
ax3.set_ylim(0, 100)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch12/practical_applications/img/ab_scale.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n① 팔당 필요한 표본 (기준 5%, 양측 0.05, 검정력 0.8)")
for d in (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
    print(
        f"  +{d * 100:>4.2f}%p (상대 {d / BASE * 100:>4.1f}%):  "
        f"{n_per_arm(d):>12,.0f} 명/팔"
    )

print("\n② 변형 k 개가 전부 대조군과 같을 때")
for k in (1, 5, 10, 20, 41, 60):
    print(f"  k = {k:>3}:  승자가 나올 확률 {fwer(k):.4f}   E[max Z] = {e_max(k):.3f}")

print("\n③ 유의한 결과가 참일 확률")
print(f"{'사전확률':>10}{'검정력 0.8':>12}{'0.5':>10}{'0.3':>10}")
for pr in (0.05, 0.10, 0.15, 1 / 3, 0.5):
    print(
        f"{pr * 100:>9.1f}%{post(pr, .8) * 100:>11.1f}%"
        f"{post(pr, .5) * 100:>9.1f}%{post(pr, .3) * 100:>9.1f}%"
    )
