"""11.4절 교란변수 — 어떤 변수가 부호를 뒤집는지는 미리 셀 수 있다.

들머리의 물음에 답하는 그림이다. 「한 변수를 넣고 빼는 일이 추정값의
부호까지 바꾼다면, 어떤 변수가 그런 힘을 가지는지 미리 알아낼 수 있는가.」
알아낼 수 있다 --- 빼고 돌린 기울기는 쪽의 자료에서

    참 효과 β + a·c/(c² + 1)

이고, a 는 Z 가 Y 에 미치는 효과, c 는 Z 가 X 에 들어가는 세기다. 쪽의
세 판은 c = 1 로 못 박고 a 만 +2.0, -1.5, -4.0 으로 바꾼 것이라 기울기가
1 + a/2 라는 **직선 위의 세 점**이다. 그 직선이 0 을 지나는 자리가
a = -2 이고, 쪽의 연습문제 5 가 네 번째 판으로 그 자리를 실제로 찍는다.

오른쪽은 c 를 함께 풀어 둔 평면이다. 편향의 부호가 sign(ac) 이므로
가로축이나 세로축에 닿으면 편향이 사라지고, |c| 가 커지면 ac/(c²+1) 가
오히려 0 으로 돌아간다 --- Z 가 X 를 거의 다 설명해 버리면 X 에 남은
변동이 Z 와 무관해지기 때문이다.

쪽의 confounding_directions 는 세 판의 산점도와 기울기를, healthy_user 는
관측되지 않는 교란에 대한 민감도를 다룬다. 여기서는 그 둘이 건드리지
않는 것 --- 불일치를 미리 세는 식 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np

OUT = "docs/ch11/confounding/img/"

BETA, N = 1.0, 700
C_PAGE = 1.0                      # 쪽의 세 판은 X = Z + 잡음 이라 c = 1
# a, 이름, 이름표를 놓을 자리 (겹치지 않게 손으로 떼어 놓는다)
PANELS = [(2.0, "부풀린다", 0.0, 0.42, "center", "bottom"),
          (-1.5, "가린다", 0.75, -0.18, "left", "top"),
          (-4.0, "뒤집는다", 0.0, 0.42, "center", "bottom"),
          (-2.0, "지운다", -0.30, -0.62, "right", "top")]


def crude_and_adjusted(a, rng):
    """쪽의 연습문제 5 와 같은 자료 --- Z 를 빼고 넣은 두 기울기."""
    Z = rng.normal(0, 1, N)
    X = C_PAGE * Z + rng.normal(0, 1, N)
    Y = BETA * X + a * Z + rng.normal(0, 0.7, N)
    M = np.column_stack([np.ones(N), X, Z])
    return (np.polyfit(X, Y, 1)[0],
            np.linalg.lstsq(M, Y, rcond=None)[0][1])


rng = np.random.default_rng(11)   # 쪽의 그림과 같은 씨앗
obs = [crude_and_adjusted(row[0], rng) for row in PANELS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 4.9),
                               gridspec_kw={"width_ratios": [1.15, 1]})

# ── (좌) 세 판은 한 직선 위의 세 점이다 ────────────────────────────
aa = np.linspace(-6, 4, 400)
crude_line = BETA + aa * C_PAGE / (C_PAGE ** 2 + 1)
ax1.axhspan(BETA, 3.5, color="tab:red", alpha=0.08)
ax1.axhspan(0, BETA, color="tab:orange", alpha=0.10)
ax1.axhspan(-3.0, 0, color="tab:purple", alpha=0.10)
ax1.plot(aa, crude_line, "-", lw=2.6, color="tab:red",
         label="$Z$ 를 빼고 돌린 기울기")
ax1.plot(aa, np.full_like(aa, BETA), "-", lw=2.6, color="tab:blue",
         label="$Z$ 를 넣고 돌린 기울기")
for (a, word, dx, dy, ha, va), (cr, adj) in zip(PANELS, obs):
    ax1.plot([a], [cr], "o", ms=11, color="tab:red", zorder=5)
    ax1.plot([a], [adj], "s", ms=9, color="tab:blue", zorder=5)
    ax1.text(a + dx, cr + dy, f"{word}\n{cr:+.2f}", ha=ha, va=va,
             fontsize=10.5, color="tab:red", fontweight="bold")
A_FLIP = -BETA * (C_PAGE ** 2 + 1) / C_PAGE      # 1 + a·c/(c²+1) = 0
ax1.axvline(A_FLIP, color="0.3", ls="--", lw=1.6)
ax1.text(-2.08, -2.65, "부호가 뒤집히기 시작하는 자리  $a = -2$", ha="right",
         fontsize=11, color="0.2", fontweight="bold")
ax1.axhline(0, color="0.15", lw=1.6)
ax1.text(3.9, 3.05, "참값보다 크다", ha="right", fontsize=10.5,
         color="tab:red", fontweight="bold")
ax1.text(3.9, 0.42, "참값보다 작다", ha="right", fontsize=10.5,
         color="tab:orange", fontweight="bold")
ax1.text(3.9, -1.2, "부호가 뒤집혔다", ha="right", fontsize=10.5,
         color="tab:purple", fontweight="bold")
ax1.set_xlim(-6.2, 4.2)
ax1.set_ylim(-3.0, 3.5)
ax1.set_xlabel("Z 가 Y 에 미치는 효과 a", fontsize=11)
ax1.set_ylabel("X 의 기울기 추정값", fontsize=11)
ax1.set_title(f"참 효과는 줄곧 $+{BETA:.0f}$ 이다 ($n = {N}$)", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 두 화살표를 함께 풀어 둔 평면 ──────────────────────────────
cs = np.linspace(-4, 4, 400)
A, C = np.meshgrid(aa, cs)
crude = BETA + A * C / (C ** 2 + 1)
m = ax2.pcolormesh(aa, cs, crude, cmap="RdBu_r", vmin=-1.5, vmax=3.5,
                   shading="auto")
cb = fig.colorbar(m, ax=ax2)
cb.set_label("빼고 돌린 기울기", fontsize=10.5)
ax2.contour(aa, cs, crude, levels=[0.0], colors="k", linewidths=2.4)
ax2.contour(aa, cs, crude, levels=[BETA], colors="w", linewidths=2.0,
            linestyles="--")
for row, _ in zip(PANELS, obs):
    ax2.plot([row[0]], [C_PAGE], "o", ms=9, color="k", mfc="yellow", mew=1.6)
ax2.text(-5.8, C_PAGE + 0.25, "쪽의 네 판 (c = 1)", fontsize=10.5,
         color="k", fontweight="bold")
ax2.text(0.03, 0.03, "검은 선 — 여기서 부호가 바뀐다\n"
         "흰 점선 — 편향이 없다 (두 축이 그것이다)",
         transform=ax2.transAxes, ha="left", va="bottom", fontsize=10,
         fontweight="bold", color="0.1",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92,
                   edgecolor="0.4", lw=1.0))
ax2.set_xlim(-6, 4)
ax2.set_ylim(-4, 4)
ax2.set_xlabel("Z 가 Y 에 미치는 효과 a", fontsize=11)
ax2.set_ylabel("Z 가 X 에 들어가는 세기 c", fontsize=11)
ax2.set_title("어느 축에 닿아도 편향은 사라진다", fontsize=12.5)

fig.suptitle("어떤 변수가 부호를 뒤집는지는 자료를 보기 전에 셀 수 있다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "tipping_point.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N} · 참 효과 {BETA:+.1f} · c = {C_PAGE:.0f} (쪽과 같은 씨앗)")
print("    a      빼고       예측 1+a/2      넣고")
for (a, word, *_), (cr, adj) in zip(PANELS, obs):
    print(f"  {a:+.1f}   {cr:+.4f}      {BETA + a / 2:+.4f}      {adj:+.4f}"
          f"   ({word})")
print(f"\n부호가 뒤집히는 자리: a = {A_FLIP:+.1f}")
print("c 를 풀면 편향 = a·c/(c²+1) 이고, c 가 커지면 다시 0 으로 간다")
for c in (0.0, 0.5, 1.0, 2.0, 4.0, 10.0):
    print(f"  a = +2 일 때 c = {c:>4.1f}  →  편향 "
          f"{2 * c / (c * c + 1):+.4f}")
