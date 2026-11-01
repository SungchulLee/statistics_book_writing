"""10.4절 교란과 인과 시연 — 화살표 둘을 모두 가진 변수만 교란변수다.

들머리가 세운 두 문장을 그린다.

(1) 편향은 두 수의 곱이다 --- 빠뜨린 변수가 결과에 들어가는 세기 δ 와,
    그 변수가 처치와 얽힌 정도 γ. 둘 중 하나라도 0 이면 곱이 0 이다.
    화살표를 하나만 가진 변수는 아무리 세게 걸어도 기울기를 만들지 못한다.
(2) 그 기울기가 전부 편향인데 p 값은 그것을 알아채지 못한다. 표본을
    키우면 추정값은 참값이 아니라 0.8 로 내려앉고 p 값만 더 작아진다.

쪽의 ovb_and_simpson 은 한 자료의 층계와 심슨의 역설을 그린다. 여기서는
자료를 셋 만들어 **어느 변수가 편향을 만드는가**만 가른다.

자료생성과정은 쪽의 것을 그대로 쓴다 --- (T, C) 가 상관 0.8 인 이변량정규,
Y = C + ε. n = 500 에서 np.random.seed(42) 로 쪽의 0.7772 와 2.8e-37 이
그대로 나온다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch10/confounding/img/"

RHO = 0.8
N_PAGE = 500
REPS = 5000


def page_data(n, rng):
    """쪽의 자료생성과정 --- 화살표 둘을 모두 가진 변수."""
    tc = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], n)
    t, c = tc[:, 0], tc[:, 1]
    return t, c, c + rng.normal(0, 1, n)


def arrow_to_t_only(n, rng):
    """Z 는 T 와 얽혔지만 Y 에는 들어가지 않는다 (δ = 0)."""
    tz = rng.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], n)
    t, z = tz[:, 0], tz[:, 1]
    return t, z, rng.normal(0, np.sqrt(2), n)


def arrow_to_y_only(n, rng):
    """Z 는 Y 에 들어가지만 T 와 얽히지 않았다 (γ = 0)."""
    t = rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    return t, z, z + rng.normal(0, 1, n)


CASES = [("처치로만 가는 화살표", arrow_to_t_only, "tab:green"),
         ("결과로만 가는 화살표", arrow_to_y_only, "tab:orange"),
         ("화살표 둘 — 교란변수", page_data, "tab:red")]

rng = np.random.default_rng(404)
summary = []
for name, gen, col in CASES:
    short = np.empty(REPS)
    deltas = np.empty(REPS)
    gammas = np.empty(REPS)
    for k in range(REPS):
        t, z, y = gen(N_PAGE, rng)
        short[k] = stats.linregress(t, y).slope
        X = np.column_stack([np.ones(N_PAGE), t, z])
        deltas[k] = np.linalg.lstsq(X, y, rcond=None)[0][2]
        gammas[k] = stats.linregress(t, z).slope
    summary.append((name, col, short, deltas, gammas))

def fmt(v):
    """0 에 붙은 값을 -0.00 으로 적지 않는다."""
    return "0.00" if abs(v) < 0.005 else f"{v:+.2f}"


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.0),
                               gridspec_kw={"width_ratios": [1.1, 1]})

# ── (좌) 하나만 가진 변수는 기울기를 만들지 못한다 ──────────────────
xs = np.arange(3)
means = [s.mean() for _, _, s, _, _ in summary]
sds = [s.std(ddof=1) for _, _, s, _, _ in summary]
cols = [c for _, c, _, _, _ in summary]
ax1.bar(xs, means, yerr=sds, width=0.52, color=cols, alpha=0.85,
        capsize=7, error_kw=dict(lw=1.8, ecolor="0.3"))
ax1.axhline(0, color="0.15", lw=1.8)
for x, (name, col, s, d, g) in zip(xs, summary):
    ax1.text(x, s.mean() + np.sign(s.mean() or 1) * (s.std(ddof=1) + 0.035),
             f"{s.mean():+.4f}", ha="center",
             va="bottom" if s.mean() >= 0 else "top", fontsize=12,
             fontweight="bold", color=col)
    ax1.text(x, -0.185, f"결과로 가는 세기 δ = {fmt(d.mean())}\n"
             f"처치와의 얽힘 γ = {fmt(g.mean())}\n"
             f"곱 = {fmt(d.mean() * g.mean())}", ha="center", va="top",
             fontsize=10.5, color="0.25")
ax1.set_xticks(xs, [n for n, _, _, _, _ in summary], fontsize=11.5)
ax1.set_ylim(-0.62, 1.02)
ax1.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax1.set_ylabel("빠뜨리고 돌린 회귀의 기울기", fontsize=11)
ax1.set_title(f"참 효과는 셋 다 $0$ 이다 (각 {REPS:,} 번 · $n = {N_PAGE}$)",
              fontsize=12.5)
ax1.grid(axis="y", alpha=0.25)
ax1.text(0.5, 0.965, "둘 중 하나라도 $0$ 이면 곱이 $0$ 이다",
         transform=ax1.transAxes, ha="center", va="top", fontsize=12,
         fontweight="bold", color="0.2",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.94,
                   edgecolor="0.5", lw=1.0))

# ── (우) p 값은 편향을 알아채지 못한다 ─────────────────────────────
ns = np.unique(np.concatenate([
    np.round(np.logspace(np.log10(50), np.log10(20000), 24)),
    [50, 100, 500, 1000, 5000, 20000]])).astype(int)
slopes, logp = [], []
for n in ns:
    np.random.seed(42)                      # 쪽과 같은 씨앗
    tc = np.random.multivariate_normal([0, 0], [[1, RHO], [RHO, 1]], n)
    t, c = tc[:, 0], tc[:, 1]
    y = c + np.random.normal(0, 1, n)
    res = stats.linregress(t, y)
    slopes.append(res.slope)
    tstat = res.slope / res.stderr
    # 쪽이 쓰는 linregress 와 같게 t 분포로 잰다. 아주 작은 p 는
    # 밑흐름이 나므로 로그로 받는다.
    ls = stats.t.logsf(abs(tstat), n - 2)
    if not np.isfinite(ls):      # 큰 n 에서는 t 의 로그 꼬리도 밑흐른다.
        ls = stats.norm.logsf(abs(tstat))   # 자유도가 수천이라 차이가 없다
    logp.append(-(np.log10(2) + ls / np.log(10)))
slopes, logp = np.array(slopes), np.array(logp)

ax2.plot(ns, logp, "o-", ms=6, lw=2.2, color="tab:purple")
ax2.set_yscale("log")
ax2.set_ylim(0.6, 1.2e4)
ax2.set_yticks([1, 10, 100, 1000, 10000], ["1", "10", "100", "1,000",
                                           "10,000"])
ax2.set_yticks([], minor=True)
ax2.axhline(-np.log10(0.05), color="0.4", ls="--", lw=1.6)
ax2.text(52, 1.55, "$p = 0.05$ 인 자리", fontsize=10.5,
         color="0.4", fontweight="bold")
i500 = int(np.argmin(np.abs(ns - N_PAGE)))
ax2.plot([ns[i500]], [logp[i500]], "o", ms=13, mfc="none", mec="tab:purple",
         mew=2.4)
ax2.annotate(f"쪽의 보기 ($n = {ns[i500]}$)\n기울기 {slopes[i500]:.4f}",
             xy=(ns[i500], logp[i500]), xytext=(1100, 4.2), fontsize=10.5,
             color="tab:purple", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color="tab:purple", lw=1.6))
ax2.set_xscale("log")
ax2.set_xlim(45, 24000)
ax2.set_xticks([50, 100, 500, 1000, 5000, 20000],
               ["50", "100", "500", "1,000", "5,000", "20,000"])
ax2.set_xticks([], minor=True)
ax2.set_xlabel("표본 크기 n (로그 눈금)", fontsize=11)
ax2.set_ylabel("log10(1/p) — 클수록 유의하다", fontsize=11,
               color="tab:purple")
ax2.tick_params(axis="y", colors="tab:purple")
ax2.grid(alpha=0.25)

ax3 = ax2.twinx()
ax3.plot(ns, slopes, "s-", ms=6, lw=2.2, color="tab:blue")
ax3.axhline(RHO, color="tab:blue", ls=":", lw=1.8)
ax3.axhline(0, color="tab:red", lw=2.0)
ax3.text(23000, RHO + 0.05, f"추정값이 가는 곳 {RHO}", ha="right",
         fontsize=10.5, color="tab:blue", fontweight="bold")
ax3.text(23000, 0.04, "참 효과 0", ha="right", fontsize=10.5,
         color="tab:red", fontweight="bold")
ax3.set_ylim(-0.30, 1.55)
ax3.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax3.set_ylabel("빠뜨리고 돌린 회귀의 기울기", fontsize=11, color="tab:blue")
ax3.tick_params(axis="y", colors="tab:blue")
ax2.set_title("표본을 키우면 p 값만 작아진다", fontsize=12.5)

fig.suptitle("화살표 둘을 모두 가진 변수만 기울기를 만든다 — 그리고 "
             "p 값은 그것을 알아채지 못한다", fontsize=13.5, y=1.0)
plt.tight_layout()
path = OUT + "two_arrows.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\n각 {REPS:,} 번 · n = {N_PAGE} · 참 효과는 셋 다 0")
for name, _, s, d, g in summary:
    print(f"  {name:<22s} 기울기 {s.mean():+.4f} (sd {s.std(ddof=1):.4f})  "
          f"δ = {d.mean():+.4f}  γ = {g.mean():+.4f}  "
          f"곱 = {d.mean()*g.mean():+.4f}")
print("\n표본을 키우면 (쪽과 같은 씨앗)")
for n, sl, lp in zip(ns, slopes, logp):
    if n in (50, 101, 500, 1000, 5000, 20000) or n == ns[i500]:
        print(f"  n = {n:>6,}  기울기 {sl:+.4f}  log10(1/p) = {lp:8.2f}")
print(f"  쪽이 적은 자리: n = {ns[i500]} 에서 기울기 {slopes[i500]:.4f} · "
      f"p 는 10 의 {logp[i500]:.1f} 제곱분의 1")
