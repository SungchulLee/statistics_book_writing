"""10.6절 상관 시각화 — 섞는 비율은 상관이 아니다.

들머리의 마지막 요점을 그린다. 이 쪽은 답을 손에 들고 시작하려고 네
변수를 직접 지어낸다.

    x1 = z1,  x2 = 0.7 z1 + 0.3 z2,  x3 = -0.5 z1 + 0.8 eps,  x4 = eps'

그런데 **섞는 비율이 곧 상관은 아니다.** x1 과 x2 의 상관은 0.7 이 아니라
0.9191 이다. 나누는 표준편차 sqrt(0.58) = 0.7616 이 1 보다 작아 비율을
끌어올리기 때문이다. 계수를 상관으로 읽는 것이 이런 자료에서 가장 흔한
실수이고, 그 실수는 상관행렬을 **지어낼 때**에만 생기므로 열지도를 아무리
들여다보아도 드러나지 않는다.

왼쪽은 섞는 비율 a 를 0 에서 1 까지 움직인 곡선이다. a = 0.5 에서 이미
상관이 0.7071 이고, 곡선이 대각선 위로 크게 부풀어 있다. 오른쪽은 쪽이
지어낸 네 변수의 세 쌍에서 「적어 넣은 수」와 「나온 상관」을 맞댄 것이다.

쪽의 corr_viz_58 과 corr_viz_89 는 그 자료의 열지도와 산점도 행렬을,
corr_feature_filter 는 주변 상관으로 특징을 거르는 일의 손실을 다룬다.
여기서는 그 셋이 건드리지 않는 것 --- 자료를 지어낼 때 적은 수와 실제로
나오는 상관이 다르다는 것 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch10/visualization/img/"

A_PAGE = 0.7                      # 쪽이 적어 넣은 섞는 비율
V2 = 0.7 ** 2 + 0.3 ** 2          # Var(x2) = 0.58
V3 = 0.5 ** 2 + 0.8 ** 2          # Var(x3) = 0.89
R12 = 0.7 / np.sqrt(V2)
R13 = -0.5 / np.sqrt(V3)
R23 = (0.7 * -0.5) / np.sqrt(V2 * V3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 4.9),
                               gridspec_kw={"width_ratios": [1.1, 1]})

# ── (좌) 섞는 비율을 움직이면 ──────────────────────────────────────
a = np.linspace(0, 1, 400)
rho = np.divide(a, np.sqrt(a ** 2 + (1 - a) ** 2))
ax1.plot(a, rho, "-", lw=2.8, color="tab:red",
         label="실제로 나오는 상관")
ax1.plot(a, a, "--", lw=2.0, color="0.45",
         label="적어 넣은 섞는 비율 (대각선)")
ax1.fill_between(a, a, rho, color="tab:red", alpha=0.10)
for aa, col, tx, ty in ((0.5, "0.3", 0.53, 0.50),
                        (A_PAGE, "tab:red", 0.72, 0.845)):
    rr = aa / np.sqrt(aa ** 2 + (1 - aa) ** 2)
    ax1.plot([aa, aa], [aa, rr], "-", lw=1.6, color=col)
    ax1.plot([aa], [rr], "o", ms=11, color=col, zorder=5)
    ax1.plot([aa], [aa], "o", ms=7, mfc="white", mec=col, mew=2.0, zorder=5)
    ax1.text(tx, ty, f"{aa:.1f} 을 적었는데\n{rr:.4f} 이 나온다",
             ha="left", va="top", fontsize=11, color=col,
             fontweight="bold")
ax1.set_xlim(0, 1.0)
ax1.set_ylim(0, 1.05)
ax1.set_xlabel("$x_2 = a\\,z_1 + (1-a)\\,z_2$ 의 섞는 비율 a", fontsize=11)
ax1.set_ylabel("$x_1$ 과 $x_2$ 의 상관", fontsize=11)
ax1.set_title("나누는 표준편차가 1 보다 작아 비율을 끌어올린다",
              fontsize=12.5)
ax1.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 쪽이 지어낸 네 변수의 세 쌍 ───────────────────────────────
PAIRS = [("$x_1$ 과 $x_2$", 0.7, R12, "tab:red"),
         ("$x_1$ 과 $x_3$", -0.5, R13, "tab:blue"),
         ("$x_2$ 와 $x_3$", 0.7 * -0.5, R23, "tab:purple")]
y = np.arange(len(PAIRS))[::-1]
ax2.barh(y + 0.17, [p[1] for p in PAIRS], height=0.30, color="0.62",
         label="적어 넣은 수 (공분산)")
ax2.barh(y - 0.17, [p[2] for p in PAIRS], height=0.30,
         color=[p[3] for p in PAIRS], label="나온 상관")
for yy, (name, coef, r, col) in zip(y, PAIRS):
    ax2.text(coef + np.sign(coef) * 0.02, yy + 0.17, f"{coef:+.4f}",
             va="center", ha="left" if coef > 0 else "right", fontsize=11,
             color="0.3", fontweight="bold")
    ax2.text(r + np.sign(r) * 0.02, yy - 0.17, f"{r:+.4f}", va="center",
             ha="left" if r > 0 else "right", fontsize=11.5, color=col,
             fontweight="bold")
ax2.axvline(0, color="0.15", lw=1.6)
ax2.set_yticks(y, [p[0] for p in PAIRS], fontsize=12)
ax2.set_xlim(-0.95, 1.18)
ax2.set_ylim(-0.6, len(PAIRS) - 0.35)
ax2.set_xlabel("값", fontsize=11)
ax2.set_title("쪽이 지어낸 네 변수 — 셋 다 어긋난다", fontsize=12.5)
ax2.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax2.grid(axis="x", alpha=0.25)

fig.suptitle("계수를 상관으로 읽는 것이 지어낸 자료에서 가장 흔한 실수다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "coefficient_is_not_correlation.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nVar(x2) = {V2:.4f} (표준편차 {np.sqrt(V2):.4f}) · "
      f"Var(x3) = {V3:.4f} (표준편차 {np.sqrt(V3):.4f})")
print("  쌍            적어 넣은 수     나온 상관")
for name, coef, r, _ in PAIRS:
    plain = name.replace("$", "")
    print(f"  {plain:<14s} {coef:+.4f}        {r:+.4f}")
print("\n섞는 비율 a 에 따른 x1 과 x2 의 상관")
for aa in (0.1, 0.3, 0.5, 0.7, 0.9):
    print(f"  a = {aa:.1f}  →  {aa / np.sqrt(aa**2 + (1-aa)**2):.4f}")
