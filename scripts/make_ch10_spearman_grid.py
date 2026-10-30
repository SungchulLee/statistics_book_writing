"""10.4절 스피어만 검정 — 눈금에 그 칸이 없다.

순위의 순열이 n! 가지뿐이므로 r_s 도, 따라서 양측 p 값도 유한하다.
n = 4 면 가장 작은 양측 p 가 2/24 = 0.0833 이라 어떤 자료로도 기각되지 않는다.

쪽의 표를 베끼지 않고 순열을 전부 세어 다시 만든다 — 9! = 362,880 이라 셀 수 있다.
(이 쪽에 이미 있는 그림은 n = 8 의 귀무분포와 검정력이라 겹치지 않는다.)
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from itertools import permutations
from scipy import stats

OUT = "docs/ch10/correlation_test/img/"
ALPHA = 0.05
NS = [4, 5, 6, 7, 8, 9]


def exact_table(n):
    """순열을 전부 세어 양측 p 의 눈금과 두 규칙의 실제 크기를 구한다."""
    base = np.arange(n)
    perms = np.array(list(permutations(base)))
    # r_s 는 순위의 피어슨 상관이고, 순위가 고정이면 d^2 합만으로 정해진다
    d2 = ((perms - base) ** 2).sum(axis=1)
    rs = 1 - 6 * d2 / (n * (n**2 - 1))

    uniq, cnt = np.unique(np.round(rs, 12), return_counts=True)
    total = len(perms)
    # 양측 p = P(|r_s| >= |관측값|)
    pv = np.array([cnt[np.abs(uniq) >= abs(v) - 1e-12].sum() / total
                   for v in uniq])

    size_exact = cnt[pv <= ALPHA + 1e-12].sum() / total
    # t 근사를 같은 순열 전부에 걸어 본다
    with np.errstate(divide="ignore", invalid="ignore"):
        t = uniq * np.sqrt((n - 2) / np.maximum(1 - uniq**2, 1e-300))
    p_t = 2 * stats.t.sf(np.abs(t), n - 2)
    size_t = cnt[p_t <= ALPHA + 1e-12].sum() / total

    return dict(n=n, perms=total, distinct=len(np.unique(np.round(pv, 12))),
                pmin=pv.min(), size_exact=size_exact, size_t=size_t,
                grid=np.unique(np.round(pv, 12)))


rows = [exact_table(n) for n in NS]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.4),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) p 값이 가질 수 있는 값의 눈금 ───────────────────────────────
y = np.arange(len(rows))[::-1]
for yy, r in zip(y, rows):
    g = r["grid"][r["grid"] <= 0.45]
    below = g[g <= ALPHA]
    ax1.scatter(g, [yy] * len(g), marker="|", s=320,
                color="0.62", lw=1.6, zorder=3)
    if len(below):
        ax1.scatter(below, [yy] * len(below), marker="|", s=320,
                    color="tab:blue", lw=2.4, zorder=4)
    note = (f"가장 작은 p = {r['pmin']:.4f}" if r["n"] <= 5
            else f"0.05 아래 칸 {len(below)} 개")
    ax1.text(0.455, yy, note, va="center", ha="left", fontsize=10,
             color="tab:red" if r["n"] == 4 else "0.4")

ax1.axvline(ALPHA, color="tab:red", ls="--", lw=1.6)
# 위쪽은 화살표 설명이 쓰므로 아래에 둔다
ax1.text(ALPHA + 0.006, -0.52, "0.05", ha="left", va="bottom",
         fontsize=10.5, color="tab:red")
ax1.annotate("n = 4 는 0.05 아래에 칸이 하나도 없다",
             (rows[0]["pmin"], y[0]), textcoords="offset points",
             xytext=(30, 26), ha="left", fontsize=10.5, color="tab:red",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.3))

ax1.set_yticks(y, [f"n = {r['n']}\n순열 {r['perms']:,}" for r in rows],
               fontsize=10)
ax1.set_xlim(-0.02, 0.68)
ax1.set_ylim(-0.6, len(rows) - 0.15)
ax1.set_xlabel("양측 p 값이 가질 수 있는 값")
ax1.set_title("눈금은 띄엄띄엄하고 n 마다 다르다", fontsize=11.5)

# ── (우) 두 규칙의 실제 크기 ─────────────────────────────────────────
x = np.arange(len(rows))
w = 0.36
ax2.bar(x - w / 2, [r["size_exact"] for r in rows], w, color="tab:blue",
        alpha=0.85, label="정확검정")
ax2.bar(x + w / 2, [r["size_t"] for r in rows], w, color="tab:red",
        alpha=0.85, label="t 근사")
ax2.axhline(ALPHA, color="black", ls="--", lw=1.4)
# 오른쪽 끝은 막대가 쓰므로 왼쪽에 둔다
ax2.text(-0.45, ALPHA + 0.002, "명목 0.05", ha="left", va="bottom",
         fontsize=10)

for xi, r in zip(x, rows):
    ax2.text(xi - w / 2, r["size_exact"] + 0.002, f"{r['size_exact']:.4f}",
             ha="center", va="bottom", fontsize=8.5, rotation=90,
             color="tab:blue")
    ax2.text(xi + w / 2, r["size_t"] + 0.002, f"{r['size_t']:.4f}",
             ha="center", va="bottom", fontsize=8.5, rotation=90,
             color="tab:red")

ax2.set_xticks(x, [f"{r['n']}" for r in rows], fontsize=11)
ax2.set_ylim(0, 0.118)
ax2.set_xlabel("표본크기 n")
ax2.set_ylabel("「p ≤ 0.05」 규칙의 실제 크기")
ax2.set_title("정확검정은 못 미치고 t 근사는 넘는다", fontsize=11.5)
ax2.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "spearman_grid.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print("  n   순열 수   서로 다른 양측 p   가장 작은 p   p <= 0.05 의 실제 크기   t 근사의 크기")
for r in rows:
    print(f"{r['n']:>3} {r['perms']:>9,} {r['distinct']:>17} "
          f"{r['pmin']:>13.4f} {r['size_exact']:>23.4f} {r['size_t']:>15.4f}")
