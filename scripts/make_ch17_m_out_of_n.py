r"""17.5절 「그림으로 보는 통계학」 — n 개를 다 뽑지 않는 편이 나을 때.

균등분포의 오른쪽 끝 $\theta$ 를 재는 문제는 붓스트랩이 무너지는 고전적인 자리다.
$X^*_{(n)}$ 은 $X_{(n)}$ 을 넘지 못하고 $1 - (1-1/n)^n \to 0.632$ 의 확률로 **그 자리에
붙어** 있다. 비켈·괴체·판즈벗($1997$)이 정리한 처방은 **덜 뽑는 것**이다.

① 한 자료에서 — 참 분포와 두 붓스트랩 분포.
② 포함확률 — 전부 뽑으면 $n$ 을 키워도 약속에 닿지 못한다.
③ 그러면 m 을 얼마로 — n 을 묶어 두고 m 만 바꾼다. 양쪽 끝이 다 나쁘다.

최댓값 쪽 계산은 재표집을 **순위로** 한다. $m$ 개를 복원으로 뽑았을 때 최대 순위가
$k$ 이하일 확률이 $(k/n)^m$ 이므로 $K = \lceil n U^{1/m} \rceil$ 로 바로 뽑을 수 있다.

만드는 파일: docs/ch17/comparison/img/m_out_of_n.png
실행:  python3 scripts/make_ch17_m_out_of_n.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

OUT = "docs/ch17/comparison/img/m_out_of_n.png"
RNG = np.random.default_rng(20261022)
ALPHA = 0.05
TARGET = -np.log(ALPHA)        # Exp(1) 의 95% 분위수 = 2.996


def boot_max_rank(rng, n, m, B):
    """F̂ₙ 에서 m 개를 복원으로 뽑았을 때 최대 순위 K (1..n)."""
    return np.ceil(n * rng.random(B) ** (1.0 / m)).astype(int)


def quantile_c(x_sorted, rng, n, m, B=4000):
    """m 개만 뽑는 붓스트랩으로 m(X₍ₙ₎ - X*₍ₘ₎) 의 95% 분위수를 구한다."""
    k = boot_max_rank(rng, n, m, B)
    return np.quantile(m * (x_sorted[-1] - x_sorted[k - 1]), 1 - ALPHA)


# --- ① 한 자료에서 -----------------------------------------------------
N1, M1, B1 = 200, 20, 40_000
x1 = np.sort(RNG.random(N1))
rep_full = N1 * (x1[-1] - x1[boot_max_rank(RNG, N1, N1, B1) - 1])
rep_m = M1 * (x1[-1] - x1[boot_max_rank(RNG, N1, M1, B1) - 1])
true_draw = N1 * (1.0 - np.sort(RNG.random((20_000, N1)), axis=1)[:, -1])
atom = float(np.mean(rep_full == 0))
print(f"① n = {N1}:  전부 뽑은 붓스트랩이 0 에 붙어 있는 비율 = {atom:.3f} "
      f"(이론값 {1 - (1 - 1/N1)**N1:.3f})")
print(f"   95% 분위수 — 참 {np.quantile(true_draw, 0.95):.3f}, "
      f"전부 뽑기 {np.quantile(rep_full, 0.95):.3f}, "
      f"m = {M1} {np.quantile(rep_m, 0.95):.3f}  (Exp(1) 이면 {TARGET:.3f})")

# --- ② 포함확률 --------------------------------------------------------
NS = (50, 200, 800, 3200)
REPS = 500
cover = {k: [] for k in ("full", "sqrt", "sub")}
for n in NS:
    m = int(round(np.sqrt(n)))
    c_full, c_m, c_sub = [], [], []
    for _ in range(REPS):
        xs = np.sort(RNG.random(n))
        t_true = n * (1.0 - xs[-1])                 # θ = 1 이므로 이것이 참 통계량
        c1 = quantile_c(xs, RNG, n, n)              # 전부 뽑기
        c2 = quantile_c(xs, RNG, n, m)              # m 개만 복원으로
        sub = np.array([xs[RNG.choice(n, m, replace=False)].max()
                        for _ in range(400)])
        c3 = np.quantile(m * (xs[-1] - sub), 1 - ALPHA)   # m 개만 비복원으로
        c_full.append(t_true <= c1)
        c_m.append(t_true <= c2)
        c_sub.append(t_true <= c3)
    cover["full"].append(np.mean(c_full))
    cover["sqrt"].append(np.mean(c_m))
    cover["sub"].append(np.mean(c_sub))
    print(f"② n = {n:>5} (m = {m:>3}):  전부 뽑기 {cover['full'][-1]:.3f}   "
          f"복원 m {cover['sqrt'][-1]:.3f}   비복원 m {cover['sub'][-1]:.3f}")

# --- ③ 그러면 m 을 얼마로 할 것인가 ------------------------------------
# n 을 800 으로 묶어 두고 m 만 바꾼다. 논문은 m 의 선택을 「다른 곳에서 다루겠다」고
# 적고 끝나는데, 그 자리가 바로 여기다.
N3, REPS3 = 800, 800
MS = (4, 8, 16, 32, 64, 128, 256, 800)
cov3 = []
for m in MS:
    c = np.empty(REPS3, bool)
    for r in range(REPS3):
        xs = np.sort(RNG.random(N3))
        c[r] = N3 * (1.0 - xs[-1]) <= quantile_c(xs, RNG, N3, m)
    cov3.append(c.mean())
    print(f"③ n = {N3}, m = {m:>3}:  포함확률 {c.mean():.3f}")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"

ax = axes[0]
grid = np.linspace(0, 6, 400)
ax.plot(grid, 1 - np.exp(-grid), color="0.35", lw=2, ls="--",
        label="참 분포 — Exp(1)")
for rep, col, lab in ((rep_full, C2, f"전부 뽑기 (m = n = {N1})"),
                      (rep_m, C1, f"덜 뽑기 (m = {M1})")):
    ax.step(np.sort(rep), np.arange(1, len(rep) + 1) / len(rep),
            color=col, lw=1.8, where="post", label=lab)
ax.set_xlim(-0.15, 6)
ax.set_ylim(0, 1.02)
ax.set_xlabel(r"$n(X_{(n)} - X^{*})$ 또는 $m(X_{(n)} - X^{*})$")
ax.set_ylabel("누적확률")
ax.set_title(f"① 0 에 {atom*100:.0f}% 가 붙어 있다", fontsize=11)
ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
ax.plot(NS, cover["full"], "o-", color=C2, lw=2, label="전부 뽑기 (m = n)")
ax.plot(NS, cover["sqrt"], "s-", color=C1, lw=2, label=r"복원으로 $m = \sqrt{n}$")
ax.plot(NS, cover["sub"], "^-", color=C3, lw=2, label=r"비복원으로 $m = \sqrt{n}$")
ax.axhline(1 - ALPHA, color="0.35", ls="--", lw=1.2)
ax.text(NS[-1], 1 - ALPHA + 0.012, "약속한 0.95", color="0.3", fontsize=9, ha="right")
ax.set_xscale("log")
ax.set_xticks(NS)
ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel("표본크기 n")
ax.set_ylabel("포함확률")
ax.set_ylim(0.5, 1.02)
ax.set_title("② 끝점을 덮는 비율", fontsize=11)
ax.legend(fontsize=9, loc="center right")

ax = axes[2]
ax.plot(MS, cov3, "o-", color=C1, lw=2)
ax.axhline(1 - ALPHA, color="0.35", ls="--", lw=1.2)
ax.text(MS[-1], 1 - ALPHA + 0.008, "약속한 0.95", color="0.3", fontsize=9, ha="right")
ax.set_xscale("log", base=2)
ax.set_xticks(MS)
ax.set_xticklabels([str(m) for m in MS])
ax.set_xlabel(f"재표집 크기 m  (n = {N3} 으로 묶어 둔다)")
ax.set_ylabel("포함확률")
ax.set_ylim(0.5, 1.02)
ax.set_title("③ 그러면 m 을 얼마로 할 것인가", fontsize=11)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
