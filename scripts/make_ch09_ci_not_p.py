r"""9.3절 「그림으로 보는 통계학」 — 구간은 검정 하나가 아니다.

$1986$ 년 가드너와 올트먼이 『BMJ』에 「$P$ 값이 아니라 신뢰구간을: 가설검정이 아니라
추정을」을 실었다. 쌍대성은 그 글의 뼈대이기도 하다 — 구간이 $\theta_0$ 을 품지 않는
것과 $p < \alpha$ 는 같은 말이고, **구간은 모든 $\theta_0$ 에 대한 검정을 한 그림에
모아 둔 것**이다.

① 같은 효과, 다른 크기의 연구 넷 — $p$ 값만 보면 둘로 갈린다.
② 구간으로 그리면 갈리지 않는다 — 작은 연구의 구간이 무엇을 품고 있는가.
③ $p$ 값 함수 — 수평으로 자르면 구간이 나온다.

만드는 파일: docs/ch09/foundations/img/ci_not_p.png
실행:  python3 scripts/make_ch09_ci_not_p.py   (저장소 최상위에서)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats

OUT = "docs/ch09/foundations/img/ci_not_p.png"
RNG = np.random.default_rng(10)

DELTA, SIGMA = 5.0, 10.0        # 참 효과와 표준편차 (같은 단위)
IMPORTANT = 2.0                 # 임상적으로 의미 있다고 보는 크기
NS = (15, 30, 80, 300)

rows = []
for n in NS:
    a = RNG.normal(0.0, SIGMA, n)
    b = RNG.normal(DELTA, SIGMA, n)
    d = b.mean() - a.mean()
    se = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
    df = (se ** 4) / ((a.var(ddof=1) / n) ** 2 / (n - 1)
                      + (b.var(ddof=1) / n) ** 2 / (n - 1))
    t = d / se
    p = 2 * stats.t.sf(abs(t), df)
    crit = stats.t.ppf(0.975, df)
    rows.append(dict(n=n, d=d, se=se, df=df, p=p,
                     lo=d - crit * se, hi=d + crit * se))
    print(f"군당 {n:>3} 명:  차이 {d:+6.2f}  (SE {se:4.2f})  p = {p:.4f}  "
          f"95% 구간 [{d - crit*se:+6.2f}, {d + crit*se:+6.2f}]  "
          f"{'유의' if p < 0.05 else '유의하지 않음'}")

print(f"\n참 효과는 넷 다 {DELTA:+.1f} 이다. "
      f"의미 있다고 본 크기 {IMPORTANT:+.1f} 를 품은 구간: "
      f"{sum(r['lo'] <= IMPORTANT <= r['hi'] for r in rows)} / {len(rows)}")

# --- ③ p 값 함수 --------------------------------------------------------
pick = rows[1]
grid = np.linspace(pick["d"] - 4 * pick["se"], pick["d"] + 4 * pick["se"], 600)
pfun = 2 * stats.t.sf(np.abs((pick["d"] - grid) / pick["se"]), pick["df"])
c95 = stats.t.ppf(0.975, pick["df"]) * pick["se"]
c99 = stats.t.ppf(0.995, pick["df"]) * pick["se"]
print(f"\n군당 {pick['n']} 명 연구의 p 값 함수에서")
print(f"   0.05 로 자르면 [{pick['d']-c95:+.2f}, {pick['d']+c95:+.2f}]")
print(f"   0.01 로 자르면 [{pick['d']-c99:+.2f}, {pick['d']+c99:+.2f}]")
print(f"   θ₀ = 0 에서 p = {2*stats.t.sf(abs(pick['d']/pick['se']), pick['df']):.4f}")

# --- 그림 ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.6))
C1, C2, C3 = "#1f6f8b", "#c0532b", "#6a8a3a"

ax = axes[0]
xs = np.arange(len(rows))
cols = [C2 if r["p"] < 0.05 else "0.6" for r in rows]
ax.bar(xs, [r["p"] for r in rows], 0.55, color=cols)
ax.axhline(0.05, color="0.3", ls="--", lw=1.3)
ax.text(len(rows) - 0.5, 0.055, "0.05", color="0.3", fontsize=9, ha="right")
for i, r in enumerate(rows):
    ax.text(i, r["p"] + 0.012, f"{r['p']:.3f}", ha="center", fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels([f"군당 {r['n']}" for r in rows], fontsize=9)
ax.set_ylabel("$p$ 값")
ax.set_ylim(0, max(r["p"] for r in rows) * 1.22)
ax.set_title("① 같은 효과를 잰 연구 넷", fontsize=11)

ax = axes[1]
for i, r in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([r["lo"], r["hi"]], [y, y], color=C1, lw=3, solid_capstyle="round")
    ax.plot([r["d"]], [y], "o", color=C1, ms=8)
    ax.text(r["hi"] + 0.4, y, f"p = {r['p']:.3f}", va="center", fontsize=9,
            color=("0.25" if r["p"] < 0.05 else "0.55"))
ax.axvline(0, color="0.35", lw=1.3)
ax.axvline(IMPORTANT, color=C3, ls="--", lw=1.4)
ax.text(IMPORTANT + 0.25, len(rows) - 1.35, f"의미 있다고 본 크기 {IMPORTANT:.0f}",
        color=C3, fontsize=9, rotation=90, va="center")
ax.axvline(DELTA, color=C2, ls=":", lw=1.4)
ax.text(DELTA + 0.25, 0.35, f"참값 {DELTA:.0f}", color=C2, fontsize=9,
        rotation=90, va="center")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"군당 {r['n']}" for r in reversed(rows)], fontsize=9)
ax.set_xlabel("두 군의 차이")
ax.set_title("② 구간으로 그리면", fontsize=11)
ax.set_xlim(min(r["lo"] for r in rows) - 1.5, max(r["hi"] for r in rows) + 4.5)

ax = axes[2]
ax.plot(grid, pfun, color=C1, lw=2.2)
for c, lv, col, dy in ((c95, 0.05, C2, 0.10), (c99, 0.01, C3, 0.24)):
    ax.axhline(lv, color=col, ls="--", lw=1.3)
    ax.plot([pick["d"] - c, pick["d"] + c], [lv, lv], "o", color=col, ms=7)
    ax.text(grid[0] + 0.2, lv + dy, f"{lv:.2f} 로 자르면  "
            f"[{pick['d']-c:+.1f}, {pick['d']+c:+.1f}]", color=col, fontsize=9)
ax.axvline(0, color="0.35", lw=1.2)
ax.text(0.2, 0.9, "$\\theta_0 = 0$", color="0.35", fontsize=9)
ax.set_xlabel(r"검정하려는 값 $\theta_0$")
ax.set_ylabel("$p$ 값")
ax.set_title(f"③ 군당 {pick['n']} 명 연구의 $p$ 값 함수", fontsize=11)
ax.set_ylim(0, 1.05)

fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"\n저장: {OUT}")
