"""11.1절 스피어만 순위상관 — 순위만 보면 잃는 것.

쪽의 rank_invariance 는 「값을 바꾸어도 r_s 는 그대로」라는 얻는 쪽을
그린다. 여기서는 그 뒷면을 본다. 순위가 **똑같은** 두 자료를 만들면
r_s 가 소수점 아래까지 같은데, 값 사이의 간격이 전혀 달라서 피어슨의
r 은 크게 갈린다. 간격이 뜻을 갖는 자료에서는 그 뜻을 버리는 셈이다.

자료는 이 책이 만든 것이다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch11/correlation/img/"
N = 12
x = np.arange(1, N + 1, dtype=float)
order = np.array([2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11], dtype=float)

yA = order                              # 간격이 고르다
yB = 1.6 ** order                       # 순위는 같고 간격만 벌어진다

rA, rB = np.corrcoef(x, yA)[0, 1], np.corrcoef(x, yB)[0, 1]
sA = stats.spearmanr(x, yA).statistic
sB = stats.spearmanr(x, yB).statistic

fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
for ax, (y, name, col) in zip(axes[:2],
                              [(yA, "자료 ㄱ — 간격이 고르다", "tab:blue"),
                               (yB, "자료 ㄴ — 위쪽이 멀리 벌어졌다",
                                "tab:orange")]):
    ax.plot(x, y, "o", ms=10, color=col)
    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.set_title(name, fontsize=12.5, color=col)
    ax.grid(alpha=0.25)
axes[0].text(0.04, 0.95,
             f"$r$ = {rA:.4f}\n$r_s$ = {sA:.4f}", transform=axes[0].transAxes,
             va="top", fontsize=13, fontweight="bold", color="0.15")
axes[1].text(0.04, 0.95,
             f"$r$ = {rB:.4f}\n$r_s$ = {sB:.4f}", transform=axes[1].transAxes,
             va="top", fontsize=13, fontweight="bold", color="0.15")

ax = axes[2]
rk = stats.rankdata(yA)
ax.plot(x, rk, "o", ms=16, mfc="none", mec="tab:blue", mew=2.6,
        label="자료 ㄱ 의 순위")
ax.plot(x, stats.rankdata(yB), "x", ms=11, mew=3.0, color="tab:orange",
        label="자료 ㄴ 의 순위")
ax.set_xlabel("$x$", fontsize=11)
ax.set_ylabel("$y$ 의 순위", fontsize=11)
ax.set_title("순위로 바꾸면 두 자료가 포개진다", fontsize=12.5)
ax.legend(fontsize=10.5, loc="upper left", framealpha=0.95)
ax.grid(alpha=0.25)
ax.text(0.5, 0.06, f"그래서 $r_s$ 가 둘 다 {sA:.4f} 다",
        transform=ax.transAxes, ha="center", fontsize=12,
        fontweight="bold", color="0.15",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                  edgecolor="0.5", lw=1.0))

fig.suptitle("순위가 같으면 $r_s$ 도 같다 — 간격이 아무리 달라도",
             fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "ranks_lose_spacing.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  자료 ㄱ: r {rA:.6f}  r_s {sA:.6f}")
print(f"  자료 ㄴ: r {rB:.6f}  r_s {sB:.6f}")
print(f"  두 자료의 순위가 같은가: "
      f"{np.array_equal(stats.rankdata(yA), stats.rankdata(yB))}")
print(f"  r 의 차 {abs(rA - rB):.6f} · r_s 의 차 {abs(sA - sB):.6f}")
