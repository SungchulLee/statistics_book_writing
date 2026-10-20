r"""잭나이프는 붓스트랩의 **일차근사**다 — 그래서 일차근사가 안 통하는 자리에서 무너진다.

홀수 $n = 2m+1$ 에서 중앙값의 잭나이프 복제값은 **세 값**밖에 못 가진다.
아래쪽 점 하나를 빼면 $(x_{(m+1)} + x_{(m+2)})/2$, 위쪽 점 하나를 빼면
$(x_{(m)} + x_{(m+1)})/2$, 중앙값 자신을 빼면 $(x_{(m)} + x_{(m+2)})/2$ 이다.
그러니 잭나이프 표준오차는 **가운데 세 순서통계량의 간격**만으로 정해지고,
그 간격은 $n$ 이 커져도 안정되지 않는다.

① 한 자료에서 — 잭나이프 복제값 세 개와 붓스트랩 복제값의 분포.
② 중앙값 — $n$ 을 키워도 잭나이프 비는 모이지 않는다.
③ 평균 — 매끄러운 통계량에서는 잭나이프가 흠잡을 데 없다.

만드는 파일: docs/ch17/bootstrap/img/jackknife_median.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)
NS = (15, 31, 61, 121, 241, 481)
REPS, B = 300, 800
TRUE_REPS = 40_000


def se_jack(x, stat):
    """하나씩 빼며 만든 복제값들로 표준오차를 추정한다."""
    n = len(x)
    rep = np.array([stat(np.delete(x, i)) for i in range(n)])
    return np.sqrt((n - 1) / n * ((rep - rep.mean()) ** 2).sum()), rep


def se_boot(x, stat_axis, rng, B=B):
    """복원추출한 표본들의 통계량 흩어짐으로 표준오차를 추정한다."""
    rep = stat_axis(x[rng.integers(0, len(x), (B, len(x)))], axis=1)
    return rep.std(ddof=1), rep


STATS = {
    "중앙값": (np.median, np.median),
    "평균": (np.mean, np.mean),
}

# --- 참 표준오차는 모의로 직접 잰다 (점근식을 쓰지 않는다) -------------
true_se = {}
for n in NS:
    z = RNG.standard_normal((TRUE_REPS, n))
    true_se[("중앙값", n)] = np.median(z, axis=1).std(ddof=1)
    true_se[("평균", n)] = z.mean(axis=1).std(ddof=1)
    print(f"  n = {n:>3}  참 SE: 중앙값 {true_se[('중앙값', n)]:.4f}, "
          f"평균 {true_se[('평균', n)]:.4f}")

# --- 두 추정량의 비를 모은다 ------------------------------------------
ratio = {(s, m, n): [] for s in STATS for m in ("jack", "boot") for n in NS}
for n in NS:
    for _ in range(REPS):
        x = RNG.standard_normal(n)
        for s, (f1, fa) in STATS.items():
            ratio[(s, "jack", n)].append(se_jack(x, f1)[0] / true_se[(s, n)])
            ratio[(s, "boot", n)].append(se_boot(x, fa, RNG)[0] / true_se[(s, n)])
    print(f"  n = {n} 끝")

# --- ① 한 자료 --------------------------------------------------------
N1 = 31
x1 = RNG.standard_normal(N1)
_, jrep = se_jack(x1, np.median)
_, brep = se_boot(x1, np.median, RNG, B=20000)
jvals, jcnt = np.unique(np.round(jrep, 12), return_counts=True)

# === 그림 ==========================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.4, 4.9))

ax1.hist(brep, bins=45, density=True, color="#9e9e9e", alpha=0.75,
         edgecolor="none", label=f"붓스트랩 복제값 ($B = 20{{,}}000$)".replace(",", "{,}"))
top = ax1.get_ylim()[1] * 1.45
ax1.set_ylim(0, top)
# 세 값 가운데 둘은 거의 붙어 있으므로 이름표 높이를 엇갈리게 둔다
for k, (v, c) in enumerate(zip(jvals, jcnt)):
    h = top * (0.70 if c == 1 else 0.58)
    ax1.vlines(v, 0, h, color="#d62728", lw=2.4)
    ax1.text(v, h + top * 0.012, f"${c}$ 번", ha="center", fontsize=10,
             color="#d62728")
ax1.vlines([], [], [], color="#d62728", lw=2.4,
           label=f"잭나이프 복제값 — 서로 다른 값 ${len(jvals)}$ 개")
ax1.text(0.97, 0.52,
         f"잭나이프 SE  ${se_jack(x1, np.median)[0]:.3f}$\n"
         f"붓스트랩 SE  ${brep.std(ddof=1):.3f}$\n"
         f"참 SE          ${true_se[('중앙값', N1)]:.3f}$",
         transform=ax1.transAxes, ha="right", va="top", fontsize=10.5,
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc"))
ax1.set_xlabel("중앙값 복제값")
ax1.set_ylabel("밀도")
ax1.set_title(f"① 같은 자료 ($n = {N1}$), 두 가지 복제", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5)
ax1.grid(alpha=0.25)

for ax, s, ttl in ((ax2, "중앙값", "② 중앙값 — 잭나이프는 모이지 않는다"),
                   (ax3, "평균", "③ 평균 — 일차근사가 통하는 자리")):
    pos = np.arange(len(NS))
    for m, c, off, lab in (("jack", "#d62728", -0.18, "잭나이프"),
                           ("boot", "#1f77b4", 0.18, "붓스트랩")):
        data = [ratio[(s, m, n)] for n in NS]
        bp = ax.boxplot(data, positions=pos + off, widths=0.3, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black", lw=1.2))
        for box in bp["boxes"]:
            box.set(facecolor=c, alpha=0.65, edgecolor=c)
        ax.plot([], [], color=c, lw=6, alpha=0.65, label=lab)
    ax.axhline(1.0, color="black", lw=1.4, ls="--")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"${n}$" for n in NS])
    ax.set_xlabel("표본크기 $n$")
    ax.set_ylabel("추정한 SE $\\div$ 참 SE")
    ax.set_title(ttl, fontsize=12)
    ax.set_ylim(0.3, 2.1)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(alpha=0.25, axis="y")

fig.tight_layout()
out = "docs/ch17/bootstrap/img/jackknife_median.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print(f"\n① n = {N1} 한 자료")
print(f"   잭나이프 복제값의 서로 다른 값 {len(jvals)} 개 — "
      + ", ".join(f"{v:.4f}({c} 번)" for v, c in zip(jvals, jcnt)))
xs = np.sort(x1)
m = N1 // 2
print(f"   가운데 순서통계량 x({m}) = {xs[m - 1]:.4f}, x({m + 1}) = {xs[m]:.4f}, "
      f"x({m + 2}) = {xs[m + 1]:.4f}")
print(f"   잭나이프 SE {se_jack(x1, np.median)[0]:.4f}, "
      f"붓스트랩 SE {brep.std(ddof=1):.4f}, "
      f"참 SE {true_se[('중앙값', 31)]:.4f}")

for s in STATS:
    print(f"\n{s} — 추정 SE ÷ 참 SE (자료 {REPS} 벌)")
    print(f"{'n':>5}{'잭 평균':>10}{'잭 표준편차':>12}{'잭 95% 폭':>22}"
          f"{'붓 평균':>10}{'붓 표준편차':>12}")
    for n in NS:
        j = np.array(ratio[(s, "jack", n)])
        b = np.array(ratio[(s, "boot", n)])
        lo, hi = np.quantile(j, [0.025, 0.975])
        print(f"{n:>5}{j.mean():>10.3f}{j.std(ddof=1):>12.3f}"
              f"{f'[{lo:.2f}, {hi:.2f}]':>22}{b.mean():>10.3f}{b.std(ddof=1):>12.3f}")
    j0 = np.array(ratio[(s, "jack", NS[0])]).std(ddof=1)
    j1 = np.array(ratio[(s, "jack", NS[-1])]).std(ddof=1)
    b0 = np.array(ratio[(s, "boot", NS[0])]).std(ddof=1)
    b1 = np.array(ratio[(s, "boot", NS[-1])]).std(ddof=1)
    print(f"   n 을 {NS[0]} 에서 {NS[-1]} 로 ({NS[-1] // NS[0]} 배) 키울 때 비의 표준편차: "
          f"잭나이프 {j0:.3f} → {j1:.3f} ({j0 / j1:.2f} 배), "
          f"붓스트랩 {b0:.3f} → {b1:.3f} ({b0 / b1:.2f} 배)")
