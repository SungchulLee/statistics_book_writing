r"""어텐션 열지도를 「모형이 여기를 봤다」로 읽으면 안 되는 이유를 수로 센다.

토큰 $T$ 개에 임베딩 $h_t \in \mathbb R^d$ 가 붙어 있고, 어텐션 $a$ 가 단체(simplex)
위의 분포라 하자. 아래 층으로 넘어가는 것은 문맥벡터

$$c = \sum_t a_t\, h_t \in \mathbb R^d$$

하나뿐이다. 제약이 $d$ 개인데 자유도는 $T-1$ 개이므로, **같은 $c$ 를 내는 다른
$a$ 가 아주 많다.**

① 같은 문맥벡터를 내면서 질량을 거의 전부 옮긴 어텐션을 선형계획으로 찾는다.
② 그 거리가 $T$ 와 $d$ 에 따라 어떻게 되는가.
③ 설령 $a$ 가 유일하더라도, 열지도가 보여 주는 것은 $a_t$ 이고 실제 기여는
   $a_t\,(w\cdot h_t)$ 다.

만드는 파일: docs/ch12/visualization/img/attention_heatmap.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats                     # noqa: E402
from scipy.optimize import linprog          # noqa: E402

RNG = np.random.default_rng(20261022)

T0, D0 = 50, 8          # 보기로 쓸 토큰 수와 임베딩 차원
TS = (10, 20, 50)
DS = np.array([1, 2, 4, 8, 16])
REPS = 300
CONC = 0.7              # 어텐션을 뽑는 디리클레 모수 (작을수록 뾰족하다)


def far_attention(H, a):
    """같은 문맥벡터를 내면서 a 의 질량을 최대한 덜어내는 어텐션을 찾는다.

    ``max`` 거리를 직접 푸는 것은 볼록최대화라 어려우므로, $\\sum_t a_t x_t$ 를
    최소화하는 선형계획을 푼다. 이렇게 얻은 거리는 **하한**이다.
    """
    T = len(a)
    c = H.T @ a
    res = linprog(c=a, A_eq=np.vstack([H.T, np.ones(T)]),
                  b_eq=np.append(c, 1.0), bounds=[(0.0, 1.0)] * T,
                  method="highs")
    if not res.success:
        return None, np.nan
    x = np.clip(res.x, 0, None)
    x = x / x.sum()
    return x, float(0.5 * np.abs(a - x).sum())


# === ① 보기 하나 =====================================================
H0 = RNG.standard_normal((T0, D0))
a0 = RNG.dirichlet(np.full(T0, CONC))
x0, tv0 = far_attention(H0, a0)
c_a, c_x = H0.T @ a0, H0.T @ x0
print(f"토큰 {T0} 개, 임베딩 차원 {D0}")
print(f"  두 어텐션의 TV 거리 {tv0:.4f}")
print(f"  문맥벡터의 차이 (최대절대) {np.abs(c_a - c_x).max():.2e}")
print(f"  원래 어텐션의 질량이 큰 토큰 상위 5: {np.round(np.sort(a0)[::-1][:5], 3)}")
print(f"  적대적 어텐션이 쓰는 토큰 수 {int((x0 > 1e-6).sum())} 개")

# === ② T 와 d 에 따라 ================================================
print(f"\n② 찾아낸 TV 거리의 평균 ({REPS} 회)")
print(f"{'d':>4}" + "".join(f"{f'T = {t}':>12}" for t in TS))
grid = np.full((len(DS), len(TS)), np.nan)
for i, d in enumerate(DS):
    row = f"{int(d):>4}"
    for j, T in enumerate(TS):
        if d >= T:
            row += f"{'—':>12}"
            continue
        vals = []
        for _ in range(REPS):
            H = RNG.standard_normal((T, int(d)))
            a = RNG.dirichlet(np.full(T, CONC))
            vals.append(far_attention(H, a)[1])
        grid[i, j] = float(np.nanmean(vals))
        row += f"{grid[i, j]:>12.4f}"
    print(row)

# === ③ 가중치와 기여 =================================================
R3 = 20000
hit, rho = [], []
for _ in range(R3):
    H = RNG.standard_normal((T0, D0))
    w = RNG.standard_normal(D0)
    a = RNG.dirichlet(np.full(T0, CONC))
    contrib = np.abs(a * (H @ w))
    hit.append(np.argmax(a) == np.argmax(contrib))
    rho.append(stats.spearmanr(a, contrib).statistic)
hit_rate, rho_med = float(np.mean(hit)), float(np.median(rho))
print(f"\n③ 토큰 {T0} 개, {R3} 회")
print(f"   열지도에서 가장 진한 토큰이 실제 기여 1 위일 확률 {hit_rate:.3f} "
      f"(우연이면 {1 / T0:.3f})")
print(f"   어텐션과 |기여| 의 순위상관 중앙값 {rho_med:.3f}")

Hs = RNG.standard_normal((T0, D0))
ws = RNG.standard_normal(D0)
as_ = RNG.dirichlet(np.full(T0, CONC))
cs = np.abs(as_ * (Hs @ ws))

# === 그림 ============================================================
fig = plt.figure(figsize=(15.4, 5.1))
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=0.62, wspace=0.28)
axA = fig.add_subplot(gs[0, 0])
axB = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])

vmax = max(a0.max(), x0.max())
for ax, vec, name in ((axA, a0, "① 원래 어텐션 $a$"),
                      (axB, x0, "적대적 어텐션 $\\tilde a$ — 같은 문맥벡터")):
    ax.imshow(vec[None, :], aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_yticks([])
    ax.set_xticks([0, T0 // 2, T0 - 1])
    ax.set_xticklabels(["$1$", f"${T0 // 2 + 1}$", f"${T0}$"], fontsize=9.5)
    ax.set_title(name, fontsize=11.5, pad=5)

axB.set_xlabel(f"토큰  —  TV 거리 ${tv0:.3f}$, 문맥벡터의 차이 "
               f"$< 10^{{-14}}$", fontsize=10.5, color="#d62728")

MK = ["o-", "s-", "^-"]
COL = ["#1f77b4", "#2ca02c", "#d62728"]
for j, T in enumerate(TS):
    ok = ~np.isnan(grid[:, j])
    ax2.plot(DS[ok], grid[ok, j], MK[j], color=COL[j], lw=2.6, ms=8,
             label=f"토큰 ${T}$ 개")
ax2.axhline(1.0, color="black", lw=1.2, ls=":")
ax2.text(4.0, 1.035, "질량을 전부 옮긴 것", fontsize=10, ha="center")
ax2.set_xscale("log", base=2)
ax2.set_xticks(DS)
ax2.set_xticklabels([f"${int(v)}$" for v in DS])
ax2.minorticks_off()
ax2.set_xlabel("임베딩 차원 $d$ (제약의 수)")
ax2.set_ylabel("같은 문맥벡터를 내는 어텐션까지의 TV 거리")
ax2.set_ylim(0, 1.08)
ax2.set_title("② 토큰이 많을수록 자유롭다", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25)

ax3.scatter(as_, cs, s=32, alpha=0.75, color="#1f77b4", edgecolors="none")
i_a, i_c = int(np.argmax(as_)), int(np.argmax(cs))
ax3.plot(as_[i_a], cs[i_a], "s", color="#d62728", ms=13)
ax3.plot(as_[i_c], cs[i_c], "*", color="#2ca02c", ms=22)
ax3.set_ylim(-0.012, cs.max() * 1.32)
ax3.annotate("열지도에서\n가장 진한 토큰", xy=(as_[i_a], cs[i_a]),
             xytext=(as_[i_a] * 0.62, cs[i_a] * 1.45), fontsize=10.5,
             color="#d62728", ha="center",
             arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0))
ax3.annotate("실제로 가장 많이\n기여한 토큰", xy=(as_[i_c], cs[i_c]),
             xytext=(as_[i_c] - 0.006, cs[i_c] * 0.60), fontsize=10.5,
             color="#2ca02c", ha="center",
             arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.0))
ax3.text(0.97, 0.05,
         f"$1$ 위가 일치할 확률 ${hit_rate:.3f}$\n순위상관 중앙값 ${rho_med:.3f}$",
         transform=ax3.transAxes, ha="right", fontsize=11,
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc"))
ax3.set_xlabel("어텐션 가중치 $a_t$ (열지도가 보여 주는 것)")
ax3.set_ylabel("실제 기여 $|a_t\\,(w \\cdot h_t)|$")
ax3.set_title("③ 진한 것과 중요한 것은 다르다", fontsize=12)
ax3.grid(alpha=0.25)

out = "docs/ch12/visualization/img/attention_heatmap.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
