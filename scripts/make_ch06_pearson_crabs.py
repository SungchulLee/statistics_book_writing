r"""6.2절 「그림으로 보는 통계학」 — 손으로 푼 9 차 방정식.

$1894$ 년 피어슨은 웰던이 나폴리에서 잰 게 $1{,}000$ 마리의 이마너비÷몸길이 비가
한쪽으로 치우친 것을 보고, 그것을 **정규분포 둘의 섞임**으로 풀었다. 방법이 적률법이다 —
모수 다섯 개를 표본적률 다섯 개에 맞추는 것. 그 연립방정식은 $9$ 차 방정식으로 줄고,
뿌리가 여럿이라 **여섯째 적률**로 하나를 골랐다.

① 섞인 것이 한 봉우리로 보인다.
② 적률 다섯 개를 맞추는 해가 하나가 아니다 — 여섯째 적률이 고른다.
③ 값: 같은 자료에서 적률법은 최대가능도보다 훨씬 흔들린다.

자료는 그 이야기의 모양을 본떠 **직접 만든 것**이다(실제 게 자료가 아니다).

만드는 파일: docs/ch06/estimation_methods/img/pearson_crabs.png
실행:  python3 scripts/make_ch06_pearson_crabs.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from math import factorial

from scipy import optimize, stats

OUT = "docs/ch06/estimation_methods/img/pearson_crabs.png"
RNG = np.random.default_rng(20261022)

P, M1, S1, M2, S2 = 0.45, 0.634, 0.018, 0.657, 0.013
N = 1000


def draw(rng, n):
    z = rng.random(n) < P
    return np.where(z, rng.normal(M1, S1, n), rng.normal(M2, S2, n))


def mix_moments(par, k=6):
    """섞인 분포의 원적률 k 개 — 정규분포의 적률을 가중평균한다."""
    p, m1, s1, m2, s2 = par
    return np.array([p * _norm_raw(m1, s1, r) + (1 - p) * _norm_raw(m2, s2, r)
                     for r in range(1, k + 1)])


def _norm_raw(mu, sd, r):
    """정규분포의 r 차 원적률 (닫힌 꼴)."""
    tot = 0.0
    for j in range(0, r // 2 + 1):
        c = factorial(r) / (factorial(r - 2 * j) * factorial(j) * 2 ** j)
        tot += c * mu ** (r - 2 * j) * sd ** (2 * j)
    return tot


def unpack(u):
    """제약 없는 벡터를 (p, m1, s1, m2, s2) 로."""
    p = 1 / (1 + np.exp(-u[0]))
    return np.array([p, u[1], np.exp(u[2]), u[3], np.exp(u[4])])


def solve_mom(x, rng, tries=400):
    """표본적률 다섯 개를 맞추는 해들을 여러 출발점에서 찾는다."""
    target = np.array([np.mean(x ** r) for r in range(1, 6)])
    sols = []
    for _ in range(tries):
        u0 = np.array([rng.normal(0, 1),
                       np.mean(x) + rng.normal(0, 2 * np.std(x)),
                       np.log(np.std(x) * np.exp(rng.normal(0, 0.5))),
                       np.mean(x) + rng.normal(0, 2 * np.std(x)),
                       np.log(np.std(x) * np.exp(rng.normal(0, 0.5)))])
        try:
            r = optimize.root(lambda u: mix_moments(unpack(u), 5) - target,
                              u0, method="hybr", tol=1e-14)
        except Exception:
            continue
        if not r.success:
            continue
        par = unpack(r.x)
        if max(abs(mix_moments(par, 5) - target) / np.abs(target)) > 1e-8:
            continue
        if par[1] > par[3]:                       # 이름표 바꾸기를 정리한다
            par = np.array([1 - par[0], par[3], par[4], par[1], par[2]])
        if any(np.allclose(par, s, rtol=1e-4, atol=1e-7) for s in sols):
            continue
        sols.append(par)
    return sols, target


def em(x, rng, iters=400):
    """두 정규의 섞임에 대한 EM — 최대가능도."""
    p, m1, s1, m2, s2 = 0.5, np.quantile(x, 0.3), np.std(x), np.quantile(x, 0.7), np.std(x)
    for _ in range(iters):
        a = p * stats.norm.pdf(x, m1, s1)
        b = (1 - p) * stats.norm.pdf(x, m2, s2)
        w = a / np.maximum(a + b, 1e-300)
        p = w.mean()
        m1 = (w * x).sum() / w.sum()
        m2 = ((1 - w) * x).sum() / (1 - w).sum()
        s1 = np.sqrt(max((w * (x - m1) ** 2).sum() / w.sum(), 1e-10))
        s2 = np.sqrt(max(((1 - w) * (x - m2) ** 2).sum() / (1 - w).sum(), 1e-10))
    if m1 > m2:
        p, m1, s1, m2, s2 = 1 - p, m2, s2, m1, s1
    return np.array([p, m1, s1, m2, s2])


# --- ①② 자료 하나 -----------------------------------------------------
x = draw(RNG, N)
print(f"표본 {N} 개 — 평균 {x.mean():.5f}, 표준편차 {x.std(ddof=1):.5f}, "
      f"왜도 {stats.skew(x):+.3f}")
sols, target = solve_mom(x, RNG)
m6 = np.mean(x ** 6)
print(f"적률 다섯 개를 맞추는 해를 {len(sols)} 개 찾았다 "
      f"(표본의 여섯째 원적률 {m6:.6e})")
best, bestd = None, np.inf
for s in sols:
    s6 = mix_moments(s, 6)[5]
    d = abs(s6 - m6) / m6
    print(f"   p={s[0]:.3f}  μ=({s[1]:.4f}, {s[3]:.4f})  "
          f"σ=({s[2]:.4f}, {s[4]:.4f})   여섯째 적률 어긋남 {d:.2e}")
    if d < bestd:
        best, bestd = s, d
print(f"→ 여섯째 적률이 고른 해: p={best[0]:.3f}, μ=({best[1]:.4f}, {best[3]:.4f}), "
      f"σ=({best[2]:.4f}, {best[4]:.4f})")
print(f"   참값:            p={P:.3f}, μ=({M1:.4f}, {M2:.4f}), σ=({S1:.4f}, {S2:.4f})")

# --- ③ 적률법과 최대가능도를 견준다 ------------------------------------
REPS = 200
gap_mom, gap_em, fail = [], [], 0
for r in range(REPS):
    xx = draw(RNG, N)
    ss, _ = solve_mom(xx, RNG, tries=120)
    if not ss:
        fail += 1
    else:
        m6x = np.mean(xx ** 6)
        pick = min(ss, key=lambda s: abs(mix_moments(s, 6)[5] - m6x))
        gap_mom.append(pick[3] - pick[1])
    gap_em.append(em(xx, RNG)[3] - em(xx, RNG)[1])
gap_mom, gap_em = np.array(gap_mom), np.array(gap_em)
true_gap = M2 - M1
print(f"\n③ {REPS} 판 — 적률법이 해를 못 찾은 판 {fail} 번")
print(f"   두 평균의 차 (참값 {true_gap:.4f})")
print(f"   적률법: 중앙값 {np.median(gap_mom):.4f}, "
      f"가운데 80% [{np.quantile(gap_mom,0.1):.4f}, {np.quantile(gap_mom,0.9):.4f}]")
print(f"   최대가능도: 중앙값 {np.median(gap_em):.4f}, "
      f"가운데 80% [{np.quantile(gap_em,0.1):.4f}, {np.quantile(gap_em,0.9):.4f}]")
iqr_ratio = (np.quantile(gap_mom, 0.9) - np.quantile(gap_mom, 0.1)) / \
            (np.quantile(gap_em, 0.9) - np.quantile(gap_em, 0.1))
print(f"   가운데 80% 폭의 비 = {iqr_ratio:.1f} 배")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"
grid = np.linspace(x.min() - 0.01, x.max() + 0.01, 600)


def dens(par, g):
    p, m1, s1, m2, s2 = par
    return p * stats.norm.pdf(g, m1, s1) + (1 - p) * stats.norm.pdf(g, m2, s2)


ax = axes[0]
ax.hist(x, bins=36, density=True, color="0.78", edgecolor="white")
ax.plot(grid, dens([P, M1, S1, M2, S2], grid), color=C1, lw=2.2, label="섞인 분포")
ax.plot(grid, P * stats.norm.pdf(grid, M1, S1), color=C2, lw=1.5, ls="--",
        label="성분 1")
ax.plot(grid, (1 - P) * stats.norm.pdf(grid, M2, S2), color=C3, lw=1.5, ls="--",
        label="성분 2")
ax.set_xlabel("이마너비 ÷ 몸길이")
ax.set_ylabel("밀도")
ax.set_title("① 둘인데 봉우리는 하나다", fontsize=11)
ax.legend(fontsize=9)
ax.text(0.03, 0.70, f"표본 왜도 {stats.skew(x):+.2f}", transform=ax.transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6",
                              alpha=0.93))

ax = axes[1]
ax.hist(x, bins=36, density=True, color="0.85", edgecolor="white")
for i, sol in enumerate(sols):
    d6 = abs(mix_moments(sol, 6)[5] - m6) / m6
    ax.plot(grid, dens(sol, grid), lw=2.0, ls=("-" if i == 0 else "--"),
            color=(C1 if i == 0 else C2),
            label=(f"해 {i+1}: p={sol[0]:.2f}, "
                   f"$\\mu$=({sol[1]:.3f}, {sol[3]:.3f})"))
ax.set_xlabel("이마너비 ÷ 몸길이")
ax.set_ylabel("밀도")
ax.set_title("② 밀도는 겹치는데 모수가 다르다", fontsize=11)
ax.legend(fontsize=8.5, loc="upper left")
ax.text(0.03, 0.62,
        "다섯 적률을 똑같이 맞춘다\n"
        f"여섯째 적률 어긋남 "
        f"{abs(mix_moments(sols[0],6)[5]-m6)/m6:.0e} 대 "
        f"{abs(mix_moments(sols[1],6)[5]-m6)/m6:.0e}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.93))

ax = axes[2]
bins = np.linspace(min(gap_mom.min(), gap_em.min()),
                   max(gap_mom.max(), gap_em.max()), 40)
ax.hist(gap_mom, bins=bins, color=C2, alpha=0.75, label="적률법")
ax.hist(gap_em, bins=bins, color=C1, alpha=0.75, label="최대가능도 (EM)")
ax.axvline(true_gap, color="0.3", ls="--", lw=1.4)
ax.text(true_gap, ax.get_ylim()[1] * 0.95, f" 참값 {true_gap:.3f}", color="0.3",
        fontsize=9, va="top")
ax.set_xlabel("추정한 두 평균의 차")
ax.set_ylabel("판 수")
ax.set_title(f"③ 같은 자료, 다른 흔들림 ({REPS} 판)", fontsize=11)
ax.legend(fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
