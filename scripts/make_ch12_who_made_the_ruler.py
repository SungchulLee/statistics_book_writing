"""12.5절 잔차 분석 — 재는 자를 그 이상점이 함께 만든다.

균형 일원배치에서 sigma^2 의 추정값은 세 집단분산의 단순평균이다. 표본크기가
똑같은데도 집단 C 혼자 70.27% 를 내는데, 거기 이상점이 심어져 있기 때문이다.

C 의 흩어짐을 심기 전 값으로 되돌리면 sigma_hat 이 1.430551 에서 1.022494 로
내려간다. 곧 지금 쓰고 있는 자가 40% 늘어난 자다. 방향은 한쪽이다 — 분모가
커졌으므로 표준화 잔차는 모두 작아지고, 정작 그 이상점이 덜 튀어 보인다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch12/diagnostics/img/"

SDS = np.array([0.870178, 1.033538, 2.077021])      # 쪽의 보기 자료
SD_C_CLEAN = 1.145019                                # 이상점을 심기 전의 C
NAMES = ["A", "B", "C"]

var = SDS ** 2
mse = var.mean()
sigma = np.sqrt(mse)

var_clean = var.copy(); var_clean[2] = SD_C_CLEAN ** 2
mse_clean = var_clean.mean()
sigma_clean = np.sqrt(mse_clean)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4),
                               gridspec_kw={"width_ratios": [1, 1.1]})

# ── (좌) 합동분산을 누가 냈나 ───────────────────────────────────────
left = 0.0
cols = ["tab:blue", "tab:green", "tab:red"]
for nm, v, c in zip(NAMES, var, cols):
    ax1.barh([0], [v], left=left, height=0.46, color=c, alpha=0.85,
             label=f"집단 {nm}")
    ax1.text(left + v / 2, 0, f"{v / var.sum() * 100:.2f}%", ha="center",
             va="center", fontsize=11.5, color="white", fontweight="bold")
    left += v

left = 0.0
for nm, v, c in zip(NAMES, var_clean, cols):
    ax1.barh([-0.75], [v], left=left, height=0.46, color=c, alpha=0.5)
    ax1.text(left + v / 2, -0.75, f"{v / var_clean.sum() * 100:.2f}%",
             ha="center", va="center", fontsize=10.5, color="0.2",
             fontweight="bold")
    left += v
ax1.text(var_clean.sum() + 0.12, -0.75, "← C 를 심기 전으로 되돌리면",
         ha="left", va="center", fontsize=10.5, color="0.25")

ax1.set_yticks([0, -0.75], [f"지금\n합 {var.sum():.4f}",
                            f"되돌리면\n합 {var_clean.sum():.4f}"], fontsize=10)
ax1.set_xlim(0, var.sum() * 1.04)
ax1.set_ylim(-1.3, 0.55)
ax1.set_xlabel("집단분산의 합 (셋으로 나누면 MSE 다)")
ax1.set_title("표본크기가 똑같은데 한 집단이 70% 를 낸다", fontsize=11.5)
ax1.legend(fontsize=9.5, loc="upper center", ncol=3, framealpha=0.95)
ax1.grid(axis="x", alpha=0.25)

# ── (우) 늘어난 자로 재면 모두 작아진다 ─────────────────────────────
pts = np.array([3.5, 2.5, 1.8, 1.0])      # 보이기 위한 잔차 몇 개
x = np.arange(len(pts)); w = 0.34
ax2.bar(x - w / 2, pts / sigma, w, color="tab:red", alpha=0.85,
        label=f"지금의 자 — 흩어짐 추정값 {sigma:.6f}")
ax2.bar(x + w / 2, pts / sigma_clean, w, color="tab:blue", alpha=0.85,
        label=f"되돌린 자 — {sigma_clean:.6f}")
ax2.axhline(2, color="0.35", ls="--", lw=1.5)
ax2.text(-0.45, 2.06, "흔히 쓰는 문턱 2", ha="left", va="bottom", fontsize=10,
         color="0.35")
for xi, v in zip(x, pts):
    ax2.text(xi - w / 2, v / sigma + 0.05, f"{v / sigma:.2f}", ha="center",
             va="bottom", fontsize=9.5, color="tab:red")
    ax2.text(xi + w / 2, v / sigma_clean + 0.05, f"{v / sigma_clean:.2f}",
             ha="center", va="bottom", fontsize=9.5, color="tab:blue")

ax2.set_xticks(x, [f"잔차 {v:.1f}" for v in pts], fontsize=10.5)
ax2.set_ylabel("표준화 잔차")
ax2.set_ylim(0, 4.3)
ax2.set_title(f"자가 {sigma / sigma_clean - 1:.1%} 늘어나면 모두 그만큼 작아진다",
              fontsize=11.5)
ax2.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "who_made_the_ruler.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for nm, v in zip(NAMES, var):
    print(f"  집단 {nm}: 분산 {v:.4f}  몫 {v / var.sum() * 100:.2f}%")
print(f"  MSE = {mse:.6f}  sigma_hat = {sigma:.6f}   (쪽: 2.046476 · 1.430551)")
print(f"  C 를 되돌리면 sigma_hat = {sigma_clean:.6f}   (쪽: 1.022494)")
print(f"  {sigma:.6f}/{sigma_clean:.6f} = {sigma / sigma_clean:.3f}   (쪽: 1.399)")
