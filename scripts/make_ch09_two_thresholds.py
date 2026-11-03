"""9장 장 개요 들머리 — 같은 증거, 두 문턱, 갈린 평결.

왜 이 그림인가. 장 개요가 백신 시험으로 열고 있었는데, 그 들머리가
스스로 「이들은 이 장에서 배울 p 값으로 판정하지 **않았다**」고 적어
두었다. 장을 여는 보기가 그 장의 주된 도구를 쓰지 않고 왜 그래도
여기 있는지를 세 문단에 걸쳐 변명한다. 들머리가 할 일이 아니다.

대신 심슨 재판으로 연다 — 한 몸의 증거에 두 문턱을 대어 평결이 갈린
일이다. 다만 **그림은 법정을 그리지 않는다.** 「민사 = 알파 0.5」 같은
등치는 느슨하다(「증거의 우위」는 가능도비가 1 을 넘는다는 뜻이지
제1종 오류율 50 퍼센트가 아니다). 그래서 그림은 그 이야기의 통계
쪽만 정직하게 그린다 — **문턱을 옮기면 두 오류가 맞바뀐다.**

왼쪽은 문턱을 멀리 둔 경우(형사), 오른쪽은 두 밀도가 만나는 자리에
둔 경우(민사에 대응한다 — 가능도비가 1 이 되는 곳이다). 관측값은
하나인데 그 사이에 떨어져 판정이 갈린다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch09/img/"
MU1 = 2.2            # 「유죄」 쪽 세상의 중심
ZOBS = 1.8           # 관측된 증거의 세기
Z_CRIM = stats.norm.ppf(0.99)      # 합리적 의심을 넘어 — 문턱을 멀리 둔다
Z_CIVIL = MU1 / 2                  # 두 밀도가 만나는 자리 = 가능도비 1

x = np.linspace(-3.2, 5.6, 1400)
h0 = stats.norm.pdf(x)
h1 = stats.norm.pdf(x, MU1)

print("같은 자료에 문턱을 달리 두면")
print(f"  관측된 증거의 세기 z = {ZOBS}")
print(f"  두 세상의 중심 0 과 {MU1}")
rows = []
for name, c in (("문턱을 멀리 둔다", Z_CRIM), ("두 밀도가 만나는 자리", Z_CIVIL)):
    a = stats.norm.sf(c)                  # 제1종 — 무고한 쪽을 유죄로
    b = stats.norm.cdf(c, MU1)            # 제2종 — 유죄인 쪽을 놓침
    verdict = "기각" if ZOBS > c else "기각 못함"
    rows.append((name, c, a, b, verdict))
    print(f"  {name:<14s} 문턱 {c:5.2f} · 제1종 {a:.4f} · 제2종 {b:.4f} · {verdict}")

# ════════════════════════════════════════════════════════════════════
C0, C1 = "#2c7fb8", "#c0392b"
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.2), sharey=True)

for ax, (name, c, a, b, verdict) in zip(axes, rows):
    ax.plot(x, h0, color=C0, lw=2.6,
            label="귀무가설 $H_0$ — 아무 일도 없는 세상  $N(0,\\,1)$")
    ax.plot(x, h1, color=C1, lw=2.6,
            label="대립가설 $H_1$ — 정말 그런 세상  $N(2.2,\\,1)$")
    ax.fill_between(x, 0, h0, where=x >= c, color=C0, alpha=0.38)
    ax.fill_between(x, 0, h1, where=x <= c, color=C1, alpha=0.26)
    ax.axvline(c, color="0.15", lw=2.4)
    ax.text(c, 0.505, f"문턱 {c:.2f}", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color="0.15")
    ax.axvline(ZOBS, color="black", lw=2.2, ls=(0, (5, 3)))
    ax.text(ZOBS, 0.462, f"관측 {ZOBS}", ha="center", va="bottom",
            fontsize=11.5, fontweight="bold")
    ax.set_ylim(0, 0.60)
    ax.set_xlim(-3.2, 5.6)
    ax.set_xlabel("증거의 세기", fontsize=11.5)
    ax.set_title(f"{name}  →  {verdict}", fontsize=13.2)
    ax.text(0.02, 0.975,
            f"$H_0$ 인데 기각 — 파란 넓이  {a:.3f}",
            transform=ax.transAxes, va="top", fontsize=11.0, color=C0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.78"))
    ax.text(0.02, 0.885,
            f"$H_1$ 인데 기각 못함 — 붉은 넓이  {b:.3f}",
            transform=ax.transAxes, va="top", fontsize=11.0, color=C1,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.78"))
    ax.grid(alpha=0.2)

axes[0].set_ylabel("밀도", fontsize=11.5)
axes[0].legend(fontsize=10.0, loc="upper right", framealpha=0.95)

fig.suptitle("증거는 하나인데 문턱을 옮기면 두 잘못이 맞바뀐다",
             fontsize=14.6, y=0.985)
plt.tight_layout(rect=(0, 0, 1, 0.97))
path = OUT + "two_thresholds_one_evidence.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("\nsaved", path)
