r"""식스시그마의 $3.4$ 는 $\Phi(-6)$ 이 아니라 $\Phi(-4.5)$ 다.

① 중심이 맞은 정규와 $1.5\sigma$ 밀린 정규의 꼬리.
② 시그마 수준별 백만 개당 불량 — 유명한 표가 어디서 나오는가.
③ 그 $\sigma$ 는 추정값이다. $n$ 이 작으면 시그마 수준 자체가 흔들린다.

만드는 파일: docs/ch15/introduction/img/six_sigma.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

SHIFT = 1.5


def dpmo_centered(k):
    return 2 * stats.norm.sf(k) * 1e6


def dpmo_shifted(k, s=SHIFT):
    """평균이 s 만큼 밀렸을 때 양쪽 규격을 벗어나는 비율."""
    return (stats.norm.sf(k - s) + stats.norm.sf(k + s)) * 1e6


# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.2, 4.8))

# ① 두 정규와 ±6 규격선
x = np.linspace(-7.5, 7.5, 1200)
ax1.semilogy(x, stats.norm.pdf(x), color="#1f77b4", lw=2.2, label="중심이 맞았을 때")
ax1.semilogy(
    x, stats.norm.pdf(x, SHIFT), color="#d62728", lw=2.2, label="$1.5\\sigma$ 밀렸을 때"
)
xt = x[x >= 6]
ax1.fill_between(xt, 1e-11, stats.norm.pdf(xt), color="#1f77b4", alpha=0.45)
ax1.fill_between(xt, 1e-11, stats.norm.pdf(xt, SHIFT), color="#d62728", alpha=0.35)
for s in (-6, 6):
    ax1.axvline(s, color="#555555", lw=1.2, ls="--")
ax1.text(5.7, 1e-2, "규격선 $\\pm 6\\sigma$", ha="right", fontsize=10, color="#555555")
ax1.set_xlim(-7.5, 7.5)
ax1.set_ylim(1e-11, 1.2)
ax1.set_yticks([1e-9, 1e-6, 1e-3, 1])
ax1.set_yticklabels(["$10^{-9}$", "$10^{-6}$", "$10^{-3}$", "$1$"])
ax1.minorticks_off()
ax1.set_xlabel("표준편차 단위")
ax1.set_ylabel("밀도 (로그 눈금)")
ax1.set_title("① $1.5$ 를 밀면 꼬리가 천 배 두꺼워진다", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25, which="major")
ax1.annotate(
    f"${dpmo_centered(6):.3f}$ ppm",
    xy=(6.45, stats.norm.pdf(6.45) / 3),
    xytext=(1.2, 3e-11),
    fontsize=10,
    color="#1f77b4",
    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0),
)
ax1.annotate(
    f"${dpmo_shifted(6):.1f}$ ppm",
    xy=(6.45, stats.norm.pdf(6.45, SHIFT) / 3),
    xytext=(-1.2, 1e-7),
    fontsize=10,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
)

# ② 시그마 수준별 백만 개당 불량
ks = np.linspace(1, 6.5, 300)
ax2.semilogy(ks, dpmo_centered(ks), color="#1f77b4", lw=2.4, label="중심이 맞았을 때")
ax2.semilogy(
    ks, dpmo_shifted(ks), color="#d62728", lw=2.4, label="$1.5\\sigma$ 밀렸을 때"
)
for k in (3, 4, 5, 6):
    ax2.plot([k], [dpmo_shifted(k)], "o", color="#d62728", ms=7, zorder=5)
ax2.annotate(
    "$3.4$ ppm — 교과서의 그 수",
    xy=(6, dpmo_shifted(6)),
    xytext=(4.1, 0.2),
    fontsize=10,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0),
)
ax2.set_xlabel("시그마 수준 $k$ (규격선이 $\\pm k\\sigma$)")
ax2.set_ylabel("백만 개당 불량 (ppm)")
ax2.set_title("② 유명한 표가 나오는 자리", fontsize=12)
ax2.set_ylim(1e-3, 1e6)
ax2.set_yticks([1e-3, 1, 1e3, 1e6])
ax2.set_yticklabels(["$0.001$", "$1$", "$10^3$", "$10^6$"])
ax2.minorticks_off()
ax2.legend(loc="lower left", fontsize=9.5)
ax2.grid(alpha=0.25, which="major")

# ③ sigma 는 추정값이다 — 시그마 수준의 신뢰구간
ns = np.arange(5, 201)
df = ns - 1
lo = np.sqrt(df / stats.chi2.isf(0.025, df))  # sigma_hat 대비 sigma 의 하한 배수
hi = np.sqrt(df / stats.chi2.isf(0.975, df))
# 참 시그마 수준이 6 이라고 보고했을 때, 실제 수준 6*sigma_hat/sigma 의 구간
ax3.fill_between(ns, 6 / hi, 6 / lo, color="#ff7f0e", alpha=0.25)
ax3.plot(ns, 6 / hi, color="#ff7f0e", lw=2.0)
ax3.plot(ns, 6 / lo, color="#ff7f0e", lw=2.0)
ax3.axhline(6, color="#555555", lw=1.4, ls="--", label="보고된 $6\\sigma$")
ax3.set_xlabel("$\\sigma$ 를 추정하는 데 쓴 표본 크기 $n$")
ax3.set_ylabel("참 시그마 수준의 $95\\%$ 구간")
ax3.set_title("③ 그 $\\sigma$ 는 추정값이다", fontsize=12)
ax3.set_xlim(5, 200)
ax3.set_ylim(2.5, 11)
ax3.legend(loc="upper right", fontsize=10)
ax3.grid(alpha=0.25)
for n0 in (30, 100):
    i = n0 - 5
    ax3.annotate(
        f"$n = {n0}$\n$[{6 / hi[i]:.2f},\\ {6 / lo[i]:.2f}]$",
        xy=(n0, 6 / hi[i]),
        xytext=(n0 + 6, 6 / hi[i] - 1.5),
        fontsize=9.5,
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
    )

fig.tight_layout()
out = "docs/ch15/introduction/img/six_sigma.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n② 시그마 수준별 백만 개당 불량")
print(f"{'k':>4}{'중심이 맞았을 때':>18}{'1.5σ 밀렸을 때':>18}")
for k in (1, 2, 3, 4, 5, 6):
    print(f"{k:>4}{dpmo_centered(k):>18,.1f}{dpmo_shifted(k):>18,.1f}")
print(f"\n  Φ(-6)*2 = {dpmo_centered(6):.6f} ppm   (십억 개에 두 개)")
print(f"  Φ(-4.5) = {stats.norm.sf(4.5) * 1e6:.4f} ppm  ← 「3.4」 의 정체")
print(f"  두 수의 비 = {dpmo_shifted(6) / dpmo_centered(6):,.0f} 배")

print("\n③ 시그마 수준의 95% 구간 (참 수준이 6 일 때)")
for n0 in (10, 30, 50, 100, 200):
    i = n0 - 5
    print(f"  n = {n0:>4}:  [{6 / hi[i]:.2f}, {6 / lo[i]:.2f}]")
    print(
        f"            ppm 으로 [{dpmo_shifted(6 / lo[i]):.1f}, "
        f"{dpmo_shifted(6 / hi[i]):,.0f}]"
    )
