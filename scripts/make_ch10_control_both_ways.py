"""10.5절 인과 모의실험 — 같은 한 줄이 상관을 지우기도 하고 만들기도 한다.

「Z 를 통제한다」는 한 줄을 두 구조에 똑같이 건다.
  교란  X <- Z -> Y : 상관 0.6515 가 0 으로 지워진다
  충돌부 X -> Z <- Y : 0 이던 상관이 -0.5 로 만들어진다

코드는 글자 하나 다르지 않고, 상관행렬만 보아서는 어느 쪽인지 가려지지 않는다.
답은 자료 밖에 있다 — 그것이 이 그림이 보이려는 전부다.

교란: Z ~ N(0,1), X = 0.6Z + N(0, 0.25), Y = 0.8Z + N(0, 0.25)
      Var(X) = 0.61, Var(Y) = 0.89, rho = 0.48/sqrt(0.61*0.89) = 0.651450
충돌부: X, Y, e ~ N(0,1) 독립, Z = X + Y + e
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/causation/img/"
# 표본을 3,000 으로 둔 것은 뽑은 값이 닫힌 꼴에 가깝게 앉게 하려는 것이다.
# 그림이 말하는 수는 닫힌 꼴(0.651450 → 0 과 0 → -0.5)이고 뽑은 값은 그 확인이다.
N = 3000
rng = np.random.default_rng(12)

# ── 두 구조 ──────────────────────────────────────────────────────────
Z = rng.normal(0, 1, N)
Xc = 0.6 * Z + rng.normal(0, np.sqrt(0.25), N)
Yc = 0.8 * Z + rng.normal(0, np.sqrt(0.25), N)

Xk, Yk = rng.normal(0, 1, N), rng.normal(0, 1, N)
Zk = Xk + Yk + rng.normal(0, 1, N)


def partial(x, y, z):
    """z 를 통제했을 때의 편상관."""
    rxy, rxz, ryz = (stats.pearsonr(x, y)[0], stats.pearsonr(x, z)[0],
                     stats.pearsonr(y, z)[0])
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2))


# 이름 · X · Y · Z · 설명 · 닫힌 꼴의 통제 전 · 닫힌 꼴의 통제 후
CASES = [("교란   X ← Z → Y", Xc, Yc, Z, "통제하면 상관이 지워진다",
          0.651450, 0.0),
         ("충돌부 X → Z ← Y", Xk, Yk, Zk, "통제하면 없던 상관이 생긴다",
          0.0, -0.5)]

def resid(v, z):
    """v 에서 z 로 설명되는 몫을 뺀 나머지. 두 잔차의 상관이 곧 편상관이다."""
    b, a = np.polyfit(z, v, 1)
    return v - (a + b * z)


fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.4))

for row, (name, x, y, z, note, th_all, th_par) in zip(axes, CASES):
    rx, ry = resid(x, z), resid(y, z)
    panels = [("통제 전 — 원래의 X 와 Y", x, y,
               stats.pearsonr(x, y)[0], th_all, "0.35"),
              ("통제 후 — Z 를 뺀 나머지끼리", rx, ry,
               stats.pearsonr(rx, ry)[0], th_par, "tab:red")]

    for ax, (title, a_, b_, r, th, col) in zip(row, panels):
        ax.scatter(a_, b_, s=7, c=col, alpha=0.28, zorder=2)
        xs = np.linspace(a_.min(), a_.max(), 10)
        m, c0 = np.polyfit(a_, b_, 1)
        ax.plot(xs, c0 + m * xs, "-", color=col, lw=2.8, zorder=4)
        ax.axhline(0, color="0.8", lw=0.9, zorder=1)
        ax.axvline(0, color="0.8", lw=0.9, zorder=1)
        ax.set_title(f"{title}\nr = {r:+.4f}   (닫힌 꼴 {th:+.4f})",
                     fontsize=10.5, color=col)
        ax.set_xlabel("X" if "통제 전" in title else "X 의 잔차", fontsize=9.5)
        ax.set_ylabel("Y" if "통제 전" in title else "Y 의 잔차", fontsize=9.5)
        ax.tick_params(labelsize=8.5)
        ax.grid(alpha=0.18)

    row[0].text(-0.26, 0.5, f"{name}\n{note}", transform=row[0].transAxes,
                rotation=90, ha="center", va="center", fontsize=12,
                fontweight="bold")

fig.suptitle("같은 한 줄 「Z 를 통제한다」를 두 구조에 똑같이 걸면",
             fontsize=13.5, y=0.985)

plt.tight_layout()
path = OUT + "control_both_ways.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for name, x, y, z, _, th_all, th_par in CASES:
    print(f"{name:20s} 통제 전 {stats.pearsonr(x, y)[0]:+.4f} "
          f"(닫힌 꼴 {th_all:+.6f})   "
          f"통제 후 {partial(x, y, z):+.4f} (닫힌 꼴 {th_par:+.4f})")
print("쪽의 모의는 교란 0.6515 → 0.052, 충돌부 0 → -0.4996 이다 "
      "— 난수씨가 달라 뽑은 값은 다르고 닫힌 꼴이 같다")
