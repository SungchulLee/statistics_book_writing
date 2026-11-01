"""10.4절 허위상관 — 같은 높은 r 을 만드는 네 가지 길.

들머리의 마지막 문장을 그린다. 「상관은 분명히 거기 있는데 인과는 없다 ---
그 둘을 무엇으로 가리는가. 이 쪽은 허위상관을 만드는 기제를 넷으로 갈라
하나씩 다룬다.」 그 넷을 한 자리에 모아, **인과가 하나도 없는 자료에서**
각 기제가 얼마나 높은 r 을 만드는지 보인다.

자료는 모두 이 책이 만든 것이다. 네 판 모두 X 와 Y 사이에 화살표가 없다.

① 공통원인 --- 기온이 아이스크림과 익사를 함께 올린다.
② 계절성 --- 쪽의 trend_spurious 가 장기 추세를 다루므로 여기서는 공통
   추세의 다른 얼굴인 계절성을 쓴다.
③ 우연과 자료 채굴 --- 독립인 변수 100 개를 n = 30 으로 재어 4,950 쌍을
   모두 보고 가장 센 쌍 하나만 골라 낸다 (쪽의 보기 3 과 같은 판이다).
④ 충돌자 --- X 와 Y 는 독립인데 둘의 합이 큰 것만 골라 보면 음의 상관이
   생긴다 (쪽의 5 절).

쪽의 trend_spurious 는 기제 ②의 장기 추세 하나를, spurious_regression 은
무작위 걷기의 기각률을 다룬다. 여기서는 넷을 **같은 눈금에 나란히** 놓는
일만 한다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/confounding/img/"

rng = np.random.default_rng(2024)

# ① 공통원인 --- 기온이 둘을 함께 올린다
n1 = 400
temp = rng.normal(22, 7, n1)                       # 기온 (℃)
ice = 120 + 9.0 * temp + rng.normal(0, 38, n1)     # 아이스크림 판매 (만원)
drown = 0.6 + 0.09 * temp + rng.normal(0, 0.45, n1)   # 익사 사고 (건)
r1 = stats.pearsonr(ice, drown).statistic
# 기온을 통제한 편상관
r_id, r_it, r_dt = r1, stats.pearsonr(ice, temp).statistic, \
    stats.pearsonr(drown, temp).statistic
r1_part = (r_id - r_it * r_dt) / np.sqrt((1 - r_it ** 2) * (1 - r_dt ** 2))

# ② 계절성 --- 서로 아무 연결 없는 두 달별 계열
n2 = 72
month = np.arange(n2)
season = np.sin(2 * np.pi * month / 12)
a2 = 50 + 12 * season + rng.normal(0, 3.0, n2)
b2 = 30 + 8 * season + rng.normal(0, 2.2, n2)
r2 = stats.pearsonr(a2, b2).statistic
# 계절을 빼고(같은 달끼리의 평균을 뺀 뒤) 다시 재면
a2d = a2 - np.array([a2[month % 12 == m].mean() for m in month % 12])
b2d = b2 - np.array([b2[month % 12 == m].mean() for m in month % 12])
r2_adj = stats.pearsonr(a2d, b2d).statistic

# ③ 우연과 자료 채굴 --- 4,950 쌍 가운데 가장 센 하나
n3, p3 = 30, 100
M = rng.normal(0, 1, (n3, p3))
R = np.corrcoef(M, rowvar=False)
iu = np.triu_indices(p3, 1)
k = int(np.argmax(np.abs(R[iu])))
i3, j3 = iu[0][k], iu[1][k]
r3 = R[i3, j3]
n_pairs = p3 * (p3 - 1) // 2
p_each = stats.pearsonr(M[:, i3], M[:, j3]).pvalue

# ④ 충돌자 --- 독립인 둘에서 합이 큰 것만 고른다
n4 = 600
x4 = rng.normal(0, 1, n4)
y4 = rng.normal(0, 1, n4)
r4_all = stats.pearsonr(x4, y4).statistic
pick = x4 + y4 > 1.1
r4 = stats.pearsonr(x4[pick], y4[pick]).statistic

fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))
(axA, axB), (axC, axD) = axes

# ① ───────────────────────────────────────────────────────────────
sc = axA.scatter(ice, drown, c=temp, cmap="coolwarm", s=34, alpha=0.9,
                 edgecolors="white", linewidths=0.4)
cb = fig.colorbar(sc, ax=axA)
cb.set_label("기온 (℃)", fontsize=10)
axA.set_title(f"① 공통원인 — 기온이 둘을 함께 올린다\n"
              f"$r = {r1:.4f}$   ·   기온을 통제하면 ${r1_part:+.4f}$",
              fontsize=12)
axA.set_xlabel("아이스크림 판매", fontsize=10.5)
axA.set_ylabel("익사 사고", fontsize=10.5)
axA.grid(alpha=0.25)

# ② ───────────────────────────────────────────────────────────────
axB.scatter(a2, b2, s=34, color="tab:green", alpha=0.85,
            edgecolors="white", linewidths=0.4)
axB.set_title(f"② 계절성 — 둘 다 같은 달에 오르내린다\n"
              f"$r = {r2:.4f}$   ·   계절을 빼면 ${r2_adj:+.4f}$",
              fontsize=12)
axB.set_xlabel("계열 A (달마다)", fontsize=10.5)
axB.set_ylabel("계열 B (달마다)", fontsize=10.5)
axB.grid(alpha=0.25)

# ③ ───────────────────────────────────────────────────────────────
axC.scatter(M[:, i3], M[:, j3], s=42, color="tab:purple", alpha=0.85,
            edgecolors="white", linewidths=0.4)
sl, ic = np.polyfit(M[:, i3], M[:, j3], 1)
xx = np.linspace(M[:, i3].min(), M[:, i3].max(), 2)
axC.plot(xx, sl * xx + ic, "-", lw=2.0, color="tab:purple")
axC.set_title(f"③ 우연과 자료 채굴 — 독립인 변수 {p3} 개의 "
              f"{n_pairs:,} 쌍에서\n가장 센 하나를 골랐다   "
              f"$r = {r3:.4f}$ ($p = {p_each:.4f}$)", fontsize=12)
axC.set_xlabel(f"변수 {i3 + 1} 번", fontsize=10.5)
axC.set_ylabel(f"변수 {j3 + 1} 번", fontsize=10.5)
axC.grid(alpha=0.25)

# ④ ───────────────────────────────────────────────────────────────
axD.scatter(x4[~pick], y4[~pick], s=20, color="0.78", alpha=0.8,
            label=f"뽑히지 않은 것 (모두 쓰면 $r = {r4_all:+.4f}$)")
axD.scatter(x4[pick], y4[pick], s=34, color="tab:red", alpha=0.9,
            edgecolors="white", linewidths=0.4,
            label=f"합이 큰 것만 골랐다 ($r = {r4:+.4f}$)")
gx = np.linspace(-3.4, 3.4, 2)
axD.plot(gx, 1.1 - gx, "--", lw=1.8, color="0.25")
axD.set_title(f"④ 충돌자 — 독립인 둘에서 합이 큰 것만 보면\n"
              f"$r = {r4_all:+.4f}$ 가 ${r4:+.4f}$ 가 된다", fontsize=12)
axD.set_xlabel("X", fontsize=10.5)
axD.set_ylabel("Y", fontsize=10.5)
axD.set_xlim(-3.4, 3.4)
axD.set_ylim(-3.4, 3.4)
axD.legend(fontsize=9.5, loc="lower left", framealpha=0.95)
axD.grid(alpha=0.25)

fig.suptitle("네 판 모두 X 와 Y 사이에 화살표가 없다 — 그래도 상관은 거기 있다",
             fontsize=14, y=0.995)
plt.tight_layout()
path = OUT + "four_mechanisms.png"
fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
print("saved", path)

print("\n네 기제 (자료는 모두 이 책이 만든 것이다. 참 관계는 넷 다 없다)")
print(f"  ① 공통원인   n = {n1}   r = {r1:+.4f}  →  기온 통제 "
      f"{r1_part:+.4f}")
print(f"  ② 계절성     n = {n2}   r = {r2:+.4f}  →  계절 제거 "
      f"{r2_adj:+.4f}")
print(f"  ③ 자료 채굴  변수 {p3} 개 · n = {n3} · {n_pairs:,} 쌍 중 "
      f"가장 센 쌍 ({i3 + 1}, {j3 + 1})  r = {r3:+.4f}  p = {p_each:.4f}")
print(f"  ④ 충돌자     n = {n4}   모두 쓰면 {r4_all:+.4f}  →  "
      f"합 > 1.1 인 {pick.sum()} 개만 보면 {r4:+.4f}")
