"""9.6절 절단점 고르기 — 그렇게 고른 수에는 뜻이 없다.

나이와 생존이 **아무 관계가 없는** 자료를 만들고, 가능한 절단점을 모두
시도해 p 값이 가장 작은 것을 고른다. 한 번 해 보면 그럴듯한 「최적
절단점」이 하나 나오지만, 자료를 다시 뽑을 때마다 그 수가 사방으로
옮겨 다닌다 --- 그 표본에서 p 를 가장 작게 만든 수일 뿐이기 때문이다.

쪽의 cutoff_search_inflation 은 절단점 후보 수에 따른 오류율을 그린다.
여기서는 고른 절단점 **자체**가 얼마나 흔들리는지를 본다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import stats

OUT = "docs/ch09/multiple_testing/img/"
N, P, R, A = 400, 0.40, 3000, 0.05
CUTS = np.arange(15, 66, 1.0)          # 나이 절단점 후보
rng = np.random.default_rng(1994)


def one(rg):
    age = np.clip(rg.normal(38, 14, N), 1, 85)
    surv = rg.random(N) < P            # 나이와 아무 관계가 없다
    ps = []
    for c in CUTS:
        lo, hi = surv[age < c], surv[age >= c]
        if len(lo) < 15 or len(hi) < 15:
            ps.append(1.0); continue
        a, b = lo.sum(), hi.sum()
        n1, n2 = len(lo), len(hi)
        ph = (a + b) / (n1 + n2)
        se = np.sqrt(ph * (1 - ph) * (1 / n1 + 1 / n2))
        z = (a / n1 - b / n2) / se if se > 0 else 0.0
        ps.append(2 * stats.norm.sf(abs(z)))
    return np.array(ps)


ps0 = one(np.random.default_rng(12))   # 가운데쯤에서 「유의한」 절단점이 나온 자료
best0 = CUTS[int(np.argmin(ps0))]

picks, minps = np.empty(R), np.empty(R)
for r in range(R):
    ps = one(rng)
    i = int(np.argmin(ps))
    picks[r], minps[r] = CUTS[i], ps[i]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 4.6))
ax1.plot(CUTS, ps0, "-", color="tab:blue", lw=2.0)
ax1.axhline(A, color="tab:red", ls="--", lw=1.8)
ax1.text(65, A + 0.02, f"$\\alpha$ = {A}", ha="right", fontsize=11,
         color="tab:red", fontweight="bold")
ax1.plot([best0], [ps0.min()], "o", ms=11, color="tab:red")
ax1.annotate(f"「최적」 절단점 {best0:.0f} 세\n$p$ = {ps0.min():.4f}",
             xy=(best0, ps0.min()), xytext=(best0 + 4, 0.42), fontsize=11.5,
             color="tab:red", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax1.set_xlabel("절단점 (나이)", fontsize=11)
ax1.set_ylabel("그 절단점에서의 $p$ 값", fontsize=11)
ax1.set_ylim(0, 1.05)
ax1.set_title("자료 한 벌에서 절단점을 모두 시도하면", fontsize=12.5)
ax1.grid(alpha=0.25)

ax2.hist(picks, bins=np.arange(14.5, 67, 2), color="tab:purple", alpha=0.8)
ax2.axvline(best0, color="tab:red", lw=2.2)
ax2.text(best0 + 1, ax2.get_ylim()[1] * 0.93, f"왼쪽에서 고른 {best0:.0f} 세",
         fontsize=11, color="tab:red", fontweight="bold")
ax2.set_xlabel("「최적」이라 고른 절단점 (나이)", fontsize=11)
ax2.set_ylabel(f"자료를 다시 뽑은 횟수 (전체 {R:,})", fontsize=11)
ax2.set_title("자료를 다시 뽑을 때마다 그 수가 옮겨 다닌다", fontsize=12.5)
ax2.grid(axis="y", alpha=0.25)
ax2.text(0.97, 0.76, f"최소 $p$ 가 {A} 아래였던 자료\n"
         f"{(minps < A).mean():.4f}", transform=ax2.transAxes, ha="right",
         fontsize=11.5, fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

fig.suptitle("나이와 생존이 아무 관계가 없는 자료인데도 이렇다", fontsize=14,
             y=1.0)
plt.tight_layout()
path = OUT + "cutoff_wanders.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  후보 절단점 {len(CUTS)} 개 · 되풀이 {R:,} · n = {N}")
print(f"  보기 자료의 「최적」 {best0:.0f} 세, 최소 p {ps0.min():.4f}")
print(f"  고른 절단점의 사분위 "
      f"{np.percentile(picks, [25, 50, 75]).round(1).tolist()} · "
      f"범위 {picks.min():.0f}~{picks.max():.0f}")
print(f"  최소 p 가 {A} 아래였던 비율 {(minps < A).mean():.4f}")
