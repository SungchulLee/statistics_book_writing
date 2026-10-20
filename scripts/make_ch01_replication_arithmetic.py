r"""재현성 프로젝트의 세 수를 식으로 따져 본다.

재현 연구들은 **원래 효과의 $75\%$ 를 $90\%$ 검정력으로** 잡도록 설계되었다. 그 설계에서
참 효과가 보고된 것의 몇 배냐에 따라 재현 성공 확률이 어떻게 달라지는가가 왼쪽이고,
승자의 저주가 주는 부풀림이 원래 연구의 검정력과 어떻게 묶이는가가 오른쪽이다.

만드는 파일: docs/ch01/classical/img/replication_arithmetic.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats, optimize  # noqa: E402

Z = stats.norm.isf(0.025)  # 1.96
# 설계: 원래 효과의 0.75 배에서 검정력 0.90
LAM75 = Z + stats.norm.isf(0.10)
K = LAM75 / 0.75  # 참/보고 비 r 당 비중심 모수

OBS_RATIO = 0.197 / 0.403  # 재현 효과 / 원래 효과 (상관계수 단위)
OBS_RATE = 0.36  # 실제로 유의했던 비율


def rep_power(r):
    lam = K * np.asarray(r, dtype=float)
    return stats.norm.sf(Z - lam) + stats.norm.cdf(-Z - lam)


def inflation(d):
    """참 효과 d 인 연구가 유의했을 때 보고되는 효과의 부풀림 배수."""
    c = Z - d
    return 1 + stats.norm.pdf(c) / stats.norm.sf(c) / d


def orig_power(d):
    return stats.norm.sf(Z - d) + stats.norm.cdf(-Z - d)


R_AT_OBS = optimize.brentq(lambda r: rep_power(r) - OBS_RATE, 0.05, 1.5)
D_STAR = optimize.brentq(lambda d: inflation(d) - 1 / OBS_RATIO, 0.05, 4.0)
P_STAR = orig_power(D_STAR)

# === 그림 ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.0))

rs = np.linspace(0.05, 1.2, 400)
ax1.plot(rs, rep_power(rs) * 100, color="#1f77b4", lw=2.6)
pts = [
    (1.0, "#2ca02c", "보고된 효과가\n참값이라면"),
    (0.75, "#9467bd", "설계가 겨냥한 자리\n($90\\%$)"),
    (OBS_RATIO, "#ff7f0e", f"관측된 효과크기 비\n$({OBS_RATIO:.2f})$ 를 넣으면"),
]
for r, c, lab in pts:
    ax1.plot([r], [rep_power(r) * 100], "o", color=c, ms=9, zorder=5)
ax1.annotate(pts[0][2], xy=(1.0, 99.1), xytext=(0.78, 76),
             fontsize=9.5, color=pts[0][1],
             arrowprops=dict(arrowstyle="->", color=pts[0][1], lw=1.0))
ax1.annotate(pts[1][2], xy=(0.75, 90.0), xytext=(0.93, 56),
             ha="center", fontsize=9.5, color=pts[1][1],
             arrowprops=dict(arrowstyle="->", color=pts[1][1], lw=1.0))
ax1.annotate(pts[2][2], xy=(OBS_RATIO, rep_power(OBS_RATIO) * 100),
             xytext=(0.30, 76), ha="center", fontsize=9.5, color=pts[2][1],
             arrowprops=dict(arrowstyle="->", color=pts[2][1], lw=1.0))
ax1.axhline(OBS_RATE * 100, color="#d62728", lw=1.6, ls="--")
ax1.plot([R_AT_OBS], [OBS_RATE * 100], "o", color="#d62728", ms=9, zorder=5)
ax1.annotate(
    f"실제로 유의했던 비율 ${OBS_RATE * 100:.0f}\\%$\n"
    f"→ 참 효과가 보고된 것의 ${R_AT_OBS:.2f}$ 배여야 한다",
    xy=(R_AT_OBS, OBS_RATE * 100),
    xytext=(0.62, 18),
    ha="center",
    fontsize=9.5,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
)
ax1.set_xlabel("참 효과 $\\div$ 보고된 효과")
ax1.set_ylabel("재현에서 유의할 확률 ($\\%$)")
ax1.set_title("① 설계대로라면 몇 퍼센트가 나와야 하는가", fontsize=12)
ax1.set_xlim(0.05, 1.2)
ax1.set_ylim(0, 105)
ax1.grid(alpha=0.25)

ds = np.linspace(0.35, 3.6, 400)
ax2.plot(orig_power(ds) * 100, inflation(ds), color="#1f77b4", lw=2.6)
ax2.axhline(1 / OBS_RATIO, color="#d62728", lw=1.5, ls="--")
ax2.plot([P_STAR * 100], [1 / OBS_RATIO], "o", color="#d62728", ms=9, zorder=5)
ax2.annotate(
    f"관측된 부풀림 ${1 / OBS_RATIO:.2f}$ 배\n→ 원래 연구의 검정력 ${P_STAR:.2f}$",
    xy=(P_STAR * 100, 1 / OBS_RATIO),
    xytext=(55, 2.4),
    ha="center",
    fontsize=10,
    color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.1),
)
ax2.axhline(1.0, color="#555555", lw=1.0, ls=":")
ax2.text(42, 1.05, "부풀림 없음", ha="left", va="bottom", fontsize=9.5, color="#555555")
ax2.set_xlabel("원래 연구의 검정력 ($\\%$)")
ax2.set_ylabel("유의한 것만 추렸을 때의 부풀림 배수")
ax2.set_title("② 부풀림은 검정력이 정한다 — 정리 $2$", fontsize=12)
ax2.set_xlim(5, 100)
ax2.set_ylim(0.9, 3.2)
ax2.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch01/classical/img/replication_arithmetic.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n설계: 원래 효과의 0.75 배를 검정력 0.90 으로 → 비중심 모수 {LAM75:.4f}")
print(f"  r 한 단위당 {K:.4f}\n")
print(f"{'참/보고':>10}{'재현에서 유의할 확률':>22}")
for r in (1.0, 0.75, OBS_RATIO, 0.5, R_AT_OBS):
    print(f"{r:>10.3f}{rep_power(r) * 100:>20.1f}%")
print(f"\n  관측된 효과크기 비 {OBS_RATIO:.4f} 를 넣으면 {rep_power(OBS_RATIO) * 100:.1f}%")
print(f"  그런데 실제로 유의했던 것은 {OBS_RATE * 100:.0f}%  → 차이 {(rep_power(OBS_RATIO) - OBS_RATE) * 100:.1f}%p")
print(f"  {OBS_RATE * 100:.0f}% 가 나오려면 참/보고 가 {R_AT_OBS:.4f} 여야 한다")

print("\n정리 2 쪽 (이 계산은 재현 설계와 무관하다)")
print(f"  관측된 부풀림 {1 / OBS_RATIO:.4f} 배 → 원래 연구의 검정력 {P_STAR:.4f}")
for p in (0.20, 0.24, 0.35, 0.50, 0.80):
    dd = optimize.brentq(lambda d: orig_power(d) - p, 0.01, 8)
    print(f"  검정력 {p:.2f} → 부풀림 {inflation(dd):.3f} 배 (참/보고 {1 / inflation(dd):.3f})")
