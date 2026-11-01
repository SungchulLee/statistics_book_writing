"""10.1절 상관의 이해 — 상관계수의 눈금은 고르지 않다.

점구름을 등밀도 타원으로 보고 장축 대 단축의 비로 재면
sqrt((1+rho)/(1-rho)) 다. 그 자로 보면 같은 0.05 가 rho = 0 근처에서는
거의 아무것도 바꾸지 못하고 0.90 근처에서는 축비를 1.886 이나 올린다.

쪽의 galton_arithmetic 은 0.46 의 산술을, understanding_113 은 상관의
스펙트럼을 그린다. 여기서는 그 둘이 건드리지 않는 것 --- 눈금 자체가
고르지 않다는 것 --- 만 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

OUT = "docs/ch10/correlation/img/"


def axis_ratio(r):
    return np.sqrt((1 + r) / (1 - r))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6),
                               gridspec_kw={"width_ratios": [1.25, 1]})

g = np.linspace(0, 0.985, 600)
ax1.plot(g, axis_ratio(g), color="tab:blue", lw=2.6)
MARK = [(0.40, "tab:orange"), (0.60, "tab:orange"),
        (0.90, "tab:red"), (0.95, "tab:red")]
for r, c in MARK:
    ax1.plot([r], [axis_ratio(r)], "o", ms=9, color=c)
    ax1.annotate(f"$\\rho$ = {r}\n{axis_ratio(r):.3f}", xy=(r, axis_ratio(r)),
                 xytext=(r - 0.16, axis_ratio(r) + 0.9), fontsize=10.5,
                 color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.3))
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 11)
ax1.set_xlabel("모상관 $\\rho$", fontsize=11)
ax1.set_ylabel("점구름 타원의 장축 ÷ 단축", fontsize=11)
ax1.set_title("눈금이 고르지 않다", fontsize=12.5)
ax1.grid(alpha=0.25)

STEPS = [(0.00, 0.05), (0.40, 0.45), (0.90, 0.95)]
names = [f"{a} → {b}" for a, b in STEPS]
vals = [axis_ratio(b) - axis_ratio(a) for a, b in STEPS]
y = np.arange(len(names))[::-1]
ax2.barh(y, vals, height=0.52, color=["0.65", "tab:orange", "tab:red"],
         alpha=0.88)
for yy, v in zip(y, vals):
    ax2.text(v + 0.045, yy, f"+{v:.4f}", va="center", fontsize=12.5,
             fontweight="bold", color="0.2")
ax2.set_yticks(y, names, fontsize=12)
ax2.set_xlim(0, 2.35)
ax2.set_xlabel("축비가 오른 폭", fontsize=11)
ax2.set_title("같은 $0.05$ 를 올려도 값이 다르다", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.text(0.97, 0.06, f"위아래가 {vals[2] / vals[0]:.0f} 배 차이다",
         transform=ax2.transAxes, ha="right", fontsize=12,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("$0.40$ 과 $0.45$ 의 거리는 $0.90$ 과 $0.95$ 의 거리와 같지 않다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "uneven_ruler.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
for r in (0.0, 0.05, 0.40, 0.45, 0.46, 0.60, 0.90, 0.95):
    print(f"  rho {r:.2f} → 축비 {axis_ratio(r):.4f}")
for a, b in STEPS:
    print(f"  {a} → {b}: 축비가 {axis_ratio(b) - axis_ratio(a):+.4f}")
