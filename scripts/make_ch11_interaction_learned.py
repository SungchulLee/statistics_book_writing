r"""교호작용은 **모형에 적어 넣어야** 보이는 것이다 — 요인이 둘일 때까지는.

① 순수 교호작용은 주효과에 아무 자취도 남기지 않는다. 항을 빼면 그 변동이
   통째로 오차로 간다.
② 요인이 늘면 적어 넣을 항의 수가 터진다. $p$ 개면 $2$ 차만 $\binom p2$,
   전부면 $2^p - p - 1$ 이다.
③ 그래서 요즘 모형은 적어 넣는 대신 **배운다.** 대신 자료를 더 요구한다.

만드는 파일: docs/ch11/anova_two_way/img/interaction_learned.png
"""

import warnings

import numpy as np
import matplotlib

warnings.filterwarnings("ignore")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from itertools import combinations  # noqa: E402
from math import comb  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402

RNG = np.random.default_rng(20261020)

# === ① 순수 교호작용의 제곱합 ========================================
# 2x2 요인설계. 칸 평균을 순수 교호작용으로 둔다 (주효과가 정확히 0).
CELL = {(0, 0): 1.0, (0, 1): -1.0, (1, 0): -1.0, (1, 1): 1.0}
NREP, SD1 = 40, 0.8
rows = []
for (a, b), mu in CELL.items():
    rows.append((a, b, mu + SD1 * RNG.standard_normal(NREP)))

y_all = np.concatenate([r[2] for r in rows])
gm = y_all.mean()
mA = {a: np.concatenate([r[2] for r in rows if r[0] == a]).mean() for a in (0, 1)}
mB = {b: np.concatenate([r[2] for r in rows if r[1] == b]).mean() for b in (0, 1)}
n_cell = NREP
SS_A = n_cell * 2 * sum((mA[a] - gm) ** 2 for a in (0, 1))
SS_B = n_cell * 2 * sum((mB[b] - gm) ** 2 for b in (0, 1))
SS_T = ((y_all - gm) ** 2).sum()
SS_cells = n_cell * sum((r[2].mean() - gm) ** 2 for r in rows)
SS_AB = SS_cells - SS_A - SS_B
SS_E = SS_T - SS_cells
print(f"① SS_A {SS_A:.2f}, SS_B {SS_B:.2f}, SS_AB {SS_AB:.2f}, SS_E {SS_E:.2f}")
F_with = (SS_AB / 1) / (SS_E / (4 * n_cell - 4))
F_without = None  # 교호작용을 빼면 그 제곱합이 오차로 들어간다
SS_E2 = SS_E + SS_AB
print(f"   교호작용을 넣으면 F = {F_with:.1f}")
print(f"   빼면 그 {SS_AB:.1f} 이 오차로 가서 MSE 가 "
      f"{SS_E / (4 * n_cell - 4):.3f} -> {SS_E2 / (4 * n_cell - 3):.3f} 로 커진다")

# === ② 항의 수 ======================================================
PS = np.arange(2, 31)
n2 = np.array([comb(p, 2) for p in PS])
n3 = np.array([comb(p, 3) for p in PS])
nall = np.array([2**p - p - 1 for p in PS], dtype=float)

# === ③ 적어 넣기 대 배우기 ===========================================
P = 20
NS = (200, 1000, 5000, 20000)
REPS = 3


def truth(X):
    """주효과 셋, 2 차 교호작용 하나, 3 차 교호작용 하나."""
    return (X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2]
            + 1.5 * X[:, 3] * X[:, 4]
            + 2.0 * X[:, 5] * X[:, 6] * X[:, 7])


PAIRS = list(combinations(range(P), 2))


def design(X, kind):
    if kind == "main":
        return np.column_stack([np.ones(len(X)), X])
    if kind == "two":
        inter = np.column_stack([X[:, i] * X[:, j] for i, j in PAIRS])
        return np.column_stack([np.ones(len(X)), X, inter])
    if kind == "oracle":
        return np.column_stack([np.ones(len(X)), X[:, :3],
                                X[:, 3] * X[:, 4],
                                X[:, 5] * X[:, 6] * X[:, 7]])
    raise ValueError(kind)


Xte = RNG.standard_normal((20_000, P))
fte = truth(Xte)


def r2(pred):
    return 1 - np.mean((pred - fte) ** 2) / np.var(fte)


res = {k: [] for k in ("main", "two", "oracle", "mlp")}
for n in NS:
    cur = {k: [] for k in res}
    for _ in range(REPS):
        X = RNG.standard_normal((n, P))
        y = truth(X) + RNG.standard_normal(n)
        for kind in ("main", "two", "oracle"):
            D = design(X, kind)
            b = np.linalg.lstsq(D, y, rcond=None)[0]
            cur[kind].append(r2(design(Xte, kind) @ b))
        m = MLPRegressor((128, 128), activation="relu", max_iter=600,
                         early_stopping=True, n_iter_no_change=20,
                         random_state=0).fit(X, y)
        cur["mlp"].append(r2(m.predict(Xte)))
    for k in res:
        res[k].append(np.mean(cur[k]))
    print(f"  n = {n:>6}  " + "  ".join(f"{k} {res[k][-1]:+.3f}" for k in res))

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

labs = ["$A$ 주효과", "$B$ 주효과", "$A\\times B$ 교호작용", "오차"]
vals = [SS_A, SS_B, SS_AB, SS_E]
cols = ["#9e9e9e", "#9e9e9e", "#d62728", "#cccccc"]
ax1.bar(range(4), vals, 0.6, color=cols)
for i, v in enumerate(vals):
    ax1.text(i, v + SS_T * 0.015, f"${v:.1f}$", ha="center", fontsize=10.5)
ax1.set_xticks(range(4))
ax1.set_xticklabels(labs, fontsize=10)
ax1.set_ylabel("제곱합")
ax1.set_title("① 주효과에는 아무 자취도 없다", fontsize=12)
ax1.set_ylim(0, max(vals) * 1.28)
ax1.text(0.5, max(vals) * 1.12,
         "칸 평균을 순수 교호작용으로 두었다\n"
         f"항을 빼면 ${SS_AB:.0f}$ 이 통째로 오차로 간다",
         ha="center", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax1.grid(alpha=0.25, axis="y")

ax2.semilogy(PS, n2, "o-", color="#1f77b4", lw=2.2, ms=4, label="$2$ 차 항")
ax2.semilogy(PS, n3, "s-", color="#2ca02c", lw=2.2, ms=4, label="$3$ 차 항")
ax2.semilogy(PS, nall, "^-", color="#d62728", lw=2.2, ms=4,
             label="모든 차수의 항")
ax2.axvline(2, color="#999999", lw=1.2, ls=":")
ax2.text(2.4, 3e6, "이 쪽이 다루는 자리\n($p = 2$, 항 하나)", fontsize=9.5,
         color="#555555", va="top")
for p_ in (20,):
    ax2.plot([p_], [2**p_ - p_ - 1], "o", color="#d62728", ms=9, zorder=5)
    ax2.annotate(f"$p = {p_}$ 이면\n${2 ** p_ - p_ - 1:,}$ 개".replace(",", "{,}"),
                 xy=(p_, 2**p_ - p_ - 1), xytext=(22.5, 8e3),
                 ha="left", fontsize=10, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax2.set_xlabel("요인 수 $p$")
ax2.set_ylabel("적어 넣어야 할 교호작용 항의 수")
ax2.set_title("② 적어 넣기는 금방 불가능해진다", fontsize=12)
ax2.set_ylim(0.5, 1e9)
ax2.set_yticks([1, 1e2, 1e4, 1e6, 1e8])
ax2.set_yticklabels(["$1$", "$10^2$", "$10^4$", "$10^6$", "$10^8$"])
ax2.minorticks_off()
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, which="major")

for k, c, lab in (("main", "#9e9e9e", "주효과만"),
                  ("two", "#1f77b4", f"$2$ 차 교호작용 모두 (${len(PAIRS)}$ 항)"),
                  ("mlp", "#d62728", "신경망 — 적어 주지 않음"),
                  ("oracle", "#2ca02c", "참 항만 (알고 있다면)")):
    ax3.semilogx(NS, np.array(res[k]) * 100, "o-", color=c, lw=2.3, ms=6,
                 label=lab, ls="--" if k == "oracle" else "-")
ax3.set_xlabel("표본 수 $n$")
ax3.set_ylabel("시험집합 $R^2$ ($\\%$, 잡음을 뺀 참 함수 대비)")
ax3.set_title(f"③ 요인 ${P}$ 개 — 적어 넣기 대 배우기", fontsize=12)
ax3.set_xticks(NS)
ax3.set_xticklabels([f"${v:,}$".replace(",", "{,}") for v in NS])
ax3.minorticks_off()
ax3.set_ylim(-5, 108)
ax3.annotate(f"항 ${len(PAIRS) + P + 1}$ 개에 자료 ${NS[0]}$ 개 —\n"
             f"$R^2 = {res['two'][0] * 100:,.0f}\\%$ 로 축 밖이다".replace(",", "{,}"),
             xy=(NS[0] * 1.06, -3), xytext=(NS[0] * 1.45, 58),
             fontsize=9.5, color="#1f77b4", ha="left",
             arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
ax3.legend(loc="lower right", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch11/anova_two_way/img/interaction_learned.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n② 항의 수")
print(f"{'p':>4}{'2 차':>10}{'3 차':>10}{'전부':>14}")
for p_ in (2, 5, 10, 20, 30):
    print(f"{p_:>4}{comb(p_, 2):>10,}{comb(p_, 3):>10,}{2 ** p_ - p_ - 1:>14,}")

print(f"\n③ 시험집합 R^2 (요인 {P} 개, 반복 {REPS} 회)")
print(f"{'n':>8}{'주효과만':>11}{'2 차 모두':>11}{'신경망':>10}{'참 항만':>10}")
for i, n in enumerate(NS):
    print(f"{n:>8}" + "".join(f"{res[k][i] * 100:>10.1f}%"
                              for k in ("main", "two", "mlp", "oracle")))
