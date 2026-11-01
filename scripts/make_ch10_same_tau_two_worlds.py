"""10.3절 Kendall의 타우 검정 — 요약값으로서의 τ 와 검정으로서의 τ.

들머리가 갈라 두라고 한 둘을 그린다. 모형 스무 개의 참 순위를 두 가지로
어긋나게 만들되 **어긋난 쌍의 수를 똑같이 열둘로** 맞춘다. 하나는 아래쪽이
흔들린 것이고 하나는 그것의 거울상이라 위쪽이 흔들린 것이다.

쌍의 수가 같으므로 S 도 τ 도 p 값도 **글자 그대로 같다.** 검정은 둘을
가르지 못한다 --- 가를 까닭도 없다. 이 쪽의 검정이 묻는 것은 「연관이 0
인가」 하나이고 두 자료 모두 그 물음에는 같은 답을 주기 때문이다. 그런데
리더보드를 읽는 사람이 알고 싶은 것은 그 물음이 아니다. 상위 k 개만 떼어
재면 한쪽은 1 에 머물고 다른 쪽은 -1 에서 시작한다.

쪽의 kendall_exact_vs_normal 은 작은 n 에서 정확분포와 정규근사의 어긋남을,
tau_easy_pairs 는 잡음 모형에서 쌍의 난이도를 다룬다. 여기서는 모의실험을
하지 않고 **순열 둘을 손으로 만들어** 같은 τ 가 다른 것을 뜻할 수 있음만
보인다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/correlation_test/img/"

N = 20
TRUE = np.arange(1, N + 1)

# 아래쪽이 흔들린 자료 --- 어긋난 쌍 열둘이 모두 11 위 아래에 있다.
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              12, 11, 14, 13, 15,
              20, 19, 18, 17, 16])
# 그 거울상 --- 같은 열둘이 모두 10 위 위에 있다.
B = np.array([21 - A[N - 1 - i] for i in range(N)])


def discordant(m):
    """참 순위와 어긋난 쌍의 수."""
    return sum(1 for i in range(N) for j in range(i + 1, N) if m[i] > m[j])


def tau_top(m, k):
    """참 순위로 위 k 개만 떼어 잰 τ."""
    return stats.kendalltau(TRUE[:k], m[:k]).statistic


WORLDS = [("아래쪽이 흔들린 자료", A, "tab:blue"),
          ("위쪽이 흔들린 자료", B, "tab:red")]

fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.8),
                         gridspec_kw={"width_ratios": [1, 1, 1.3]})

# ── (좌·중) 같은 수의 쌍이 어긋났는데 자리가 다르다 ──────────────────
for ax, (name, m, col) in zip(axes[:2], WORLDS):
    ok = m == TRUE
    ax.plot([0.5, N + 0.5], [0.5, N + 0.5], "-", color="0.80", lw=1.4)
    ax.plot(TRUE[ok], m[ok], "o", ms=7, color="0.55")
    ax.plot(TRUE[~ok], m[~ok], "o", ms=10, color=col)
    lo, hi = TRUE[~ok].min(), TRUE[~ok].max()
    ax.axvspan(lo - 0.5, hi + 0.5, color=col, alpha=0.10)
    tau = stats.kendalltau(TRUE, m)
    ax.set_title(f"{name}\n어긋난 쌍 {discordant(m)} 개 · "
                 f"$\\tau = {tau.statistic:.6f}$", fontsize=12.5)
    ax.set_xlabel("참 순위 (1 등이 왼쪽)", fontsize=11)
    ax.set_ylabel("지표가 매긴 순위", fontsize=11)
    ax.set_xlim(0.2, N + 0.8)
    ax.set_ylim(0.2, N + 0.8)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25)
    ax.text(0.5, 0.035, f"위 다섯만 떼면 $\\tau = {tau_top(m, 5):+.2f}$",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=col,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      alpha=0.94, edgecolor=col, lw=1.0))

# ── (우) 위에서 k 개만 떼어 재면 ───────────────────────────────────
ks = np.arange(2, N + 1)
ax = axes[2]
for name, m, col in WORLDS:
    ax.plot(ks, [tau_top(m, k) for k in ks], "o-", ms=6, lw=2.2, color=col,
            label=name)
tau_all = stats.kendalltau(TRUE, A)
ax.axhline(tau_all.statistic, color="0.3", ls="--", lw=1.6)
ax.text(2.0, tau_all.statistic - 0.06,
        f"스물을 다 쓴 값 {tau_all.statistic:.4f}\n— 두 자료가 같다",
        ha="left", va="top", fontsize=10.5, color="0.2", fontweight="bold")
ax.axhline(0, color="0.75", lw=1.2)
ax.set_xlim(1.4, N + 0.6)
ax.set_ylim(-1.15, 1.25)
ax.set_xticks([2, 5, 10, 15, 20])
ax.set_xlabel("위에서 k 개만 떼어 쟀다", fontsize=11)
ax.set_ylabel("그 k 개에서 잰 τ", fontsize=11)
ax.set_title("검정은 가르지 못하고, 리더보드는 가른다", fontsize=12.5)
ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
ax.grid(alpha=0.25)

fig.suptitle("같은 τ, 같은 p 값 — 그런데 흔들린 자리가 다르다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "same_tau_two_worlds.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

for name, m, _ in WORLDS:
    t = stats.kendalltau(TRUE, m)
    d = discordant(m)
    print(f"\n{name}")
    print(f"  지표가 매긴 순위 {m.tolist()}")
    print(f"  어긋난 쌍 {d} 개 · S = {N*(N-1)//2 - 2*d} · "
          f"tau = {t.statistic:.6f} · p = {t.pvalue:.3e}")
    print(f"  위 k 개만 떼어 잰 tau: " +
          " ".join(f"k={k}:{tau_top(m, k):+.3f}" for k in (2, 3, 5, 10)))
tA, tB = stats.kendalltau(TRUE, A), stats.kendalltau(TRUE, B)
print(f"\n두 자료의 tau 가 같은가: {tA.statistic == tB.statistic}")
print(f"두 자료의 p 값이 같은가: {tA.pvalue == tB.pvalue}")
print(f"위 다섯에서 잰 tau: {tau_top(A, 5):+.4f} 대 {tau_top(B, 5):+.4f}")
