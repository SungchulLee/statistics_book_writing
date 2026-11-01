"""10.5절 DAG — 자료가 가를 수 있는 것과 가를 수 없는 것.

들머리가 「자료를 보고 정할 수 없다」고 한 것이 정확히 어디까지인지
그린다. 세 구조에서 자료가 내놓는 수는 둘뿐이다 --- 주변 상관과
가운데 변수를 통제한 편상관.

충돌자는 그 둘로 갈린다(주변이 0, 편상관이 큰 음수). 그런데 **사슬과
갈래는 같은 점에 떨어진다.** 우연이 아니라 대수다 --- 표준화하면 두
구조 모두 r(A,B) = p, r(B,C) = q, r(A,C) = pq 를 내므로 공분산행렬이
글자 그대로 같다. 자료로 가를 수 없는 짝은 이 둘이고, 쪽의 2 절도
그렇게 적는다.

문제는 그 둘에서 해야 할 일이 **정반대**라는 것이다. 사슬에서 B 를
통제하면 진짜 효과를 지우고, 갈래에서 B 를 통제하면 가짜 효과를
지운다. 같은 자료, 같은 코드 한 줄, 반대의 결과다.

쪽의 three_structures_conditioning 은 세 구조에서 조건화가 하는 일을,
four_structures_same_r 은 네 구조가 같은 r 을 내는 것을 그린다. 여기서는
그 둘이 건드리지 않는 것 --- 자료가 가를 수 있는 선이 정확히 어디인지와
그 선의 양쪽에서 판단이 어떻게 갈리는지 --- 만 본다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/causation/img/"

N = 900
P = Q = np.sqrt(0.70)             # 사슬·갈래의 두 변 --- 곱이 주변 상관 0.70
R_COL = np.sqrt(0.81 / 1.81)      # 충돌자: 편상관이 -0.81 이 되는 변


def partial(a, b, c):
    """c 를 통제한 a 와 b 의 상관."""
    rab = stats.pearsonr(a, b).statistic
    rac = stats.pearsonr(a, c).statistic
    rbc = stats.pearsonr(b, c).statistic
    return (rab - rac * rbc) / np.sqrt((1 - rac ** 2) * (1 - rbc ** 2))


def chain(rng):
    """A → B → C."""
    a = rng.normal(0, 1, N)
    b = P * a + np.sqrt(1 - P ** 2) * rng.normal(0, 1, N)
    c = Q * b + np.sqrt(1 - Q ** 2) * rng.normal(0, 1, N)
    return a, b, c


def fork(rng):
    """A ← B → C."""
    b = rng.normal(0, 1, N)
    a = P * b + np.sqrt(1 - P ** 2) * rng.normal(0, 1, N)
    c = Q * b + np.sqrt(1 - Q ** 2) * rng.normal(0, 1, N)
    return a, b, c


def collider(rng):
    """A → B ← C."""
    a = rng.normal(0, 1, N)
    c = rng.normal(0, 1, N)
    b = R_COL * (a + c) + np.sqrt(1 - 2 * R_COL ** 2) * rng.normal(0, 1, N)
    return a, b, c


STRUCTS = [("사슬  A → B → C", chain, "tab:blue", "o"),
           ("갈래  A ← B → C", fork, "tab:green", "s"),
           ("충돌자  A → B ← C", collider, "tab:red", "^")]

rng = np.random.default_rng(1005)
pts = []
for name, gen, col, mk in STRUCTS:
    a, b, c = gen(rng)
    pts.append((name, col, mk, stats.pearsonr(a, c).statistic,
                partial(a, c, b)))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# ── (좌) 자료가 내놓는 두 수의 평면 ────────────────────────────────
for name, col, mk, rm, rp in pts:
    ax1.plot([rm], [rp], mk, ms=16, color=col, alpha=0.85, zorder=4,
             label=f"{name}   ({rm:+.2f}, {rp:+.2f})")
ax1.annotate("사슬과 갈래가 같은 점에 떨어진다\n— 공분산행렬이 글자 그대로 같다",
             xy=(pts[0][3], pts[0][4]), xytext=(0.26, 0.42), fontsize=11,
             color="0.15", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       alpha=0.94, edgecolor="0.4", lw=1.0),
             arrowprops=dict(arrowstyle="->", color="0.3", lw=1.8))
ax1.annotate("충돌자는 갈린다", xy=(pts[2][3], pts[2][4]),
             xytext=(0.30, -0.60), fontsize=11, color="tab:red",
             fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.8))
ax1.axhline(0, color="0.7", lw=1.2)
ax1.axvline(0, color="0.7", lw=1.2)
ax1.set_xlim(-0.25, 0.95)
ax1.set_ylim(-1.0, 0.75)
ax1.set_xlabel("A 와 C 의 상관 (그대로 잰 것)", fontsize=11)
ax1.set_ylabel("B 를 통제한 A 와 C 의 상관", fontsize=11)
ax1.set_title(f"자료가 내놓는 수는 이 둘뿐이다 ($n = {N}$)", fontsize=12.5)
ax1.legend(fontsize=10, loc="upper left", framealpha=0.95)
ax1.grid(alpha=0.25)

# ── (우) 같은 코드 한 줄, 반대의 결과 ──────────────────────────────
ROWS = [("사슬 — B 는 매개자다", "tab:blue",
         [("통제하지 않으면", 0.70, "맞다"), ("B 를 통제하면", 0.00, "틀리다")],
         0.70, "참 효과 0.70"),
        ("갈래 — B 는 교란변수다", "tab:green",
         [("통제하지 않으면", 0.70, "틀리다"), ("B 를 통제하면", 0.00, "맞다")],
         0.00, "참 효과 0")]
y = 0
yt, ylab = [], []
for name, col, bars, truth, note in ROWS:
    ax2.text(1.0, y + 0.52, name, ha="right", va="bottom", fontsize=11.5,
             fontweight="bold", color=col)
    for lab, val, verdict in bars:
        good = verdict == "맞다"
        ax2.barh(y, max(val, 0.004), height=0.56, color=col,
                 alpha=0.85 if good else 0.28, edgecolor=col, lw=2.0)
        ax2.text(val + 0.03, y, f"{val:.2f}   {verdict}", va="center",
                 fontsize=11.5, fontweight="bold",
                 color=col if good else "tab:red")
        yt.append(y)
        ylab.append(lab)
        y -= 1
    ax2.plot([truth, truth], [y + 1.62, y + 0.38], "--", lw=2.2, color="0.2")
    ax2.text(truth + 0.012, y + 0.44, note, fontsize=10.5, color="0.2",
             fontweight="bold")
    y -= 1.0
ax2.set_yticks(yt, ylab, fontsize=11)
ax2.set_xlim(-0.03, 1.02)
ax2.set_ylim(y + 0.45, 1.25)
ax2.set_xlabel("A 와 C 사이에서 읽어 낸 값", fontsize=11)
ax2.set_title("같은 자료 · 같은 코드 한 줄 · 반대의 결과", fontsize=12.5)
ax2.grid(axis="x", alpha=0.25)
ax2.text(0.99, y + 0.62, "어느 쪽인지는 자료가 아니라\n화살표가 정한다",
         ha="right", va="bottom", fontsize=11.5, fontweight="bold",
         color="0.15",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("자료는 충돌자를 가려 주지만 사슬과 갈래는 가려 주지 못한다",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "same_numbers_two_pictures.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N} · 사슬·갈래의 두 변 p = q = {P:.6f} (곱 {P*Q:.4f})")
print(f"충돌자의 두 변 {R_COL:.6f} (편상관 "
      f"{-R_COL**2 / (1 - R_COL**2):.4f} 가 되는 값)")
print("\n     구조                주변 상관    B 를 통제한 상관")
for name, col, mk, rm, rp in pts:
    print(f"  {name:<20s} {rm:+.4f}       {rp:+.4f}")
print("\n표준화한 세 변수의 모집단 상관")
print(f"  사슬:   r(A,B) = {P:.4f}, r(B,C) = {Q:.4f}, r(A,C) = {P*Q:.4f}")
print(f"  갈래:   r(A,B) = {P:.4f}, r(B,C) = {Q:.4f}, r(A,C) = {P*Q:.4f}"
      "   <- 사슬과 같다")
print(f"  충돌자: r(A,B) = r(B,C) = {R_COL:.4f}, r(A,C) = 0")
