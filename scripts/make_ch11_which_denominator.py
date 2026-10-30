"""11.1절 OLS 모의실험 — 분모를 n 으로 두면 체계적으로 모자란다.

같은 잔차를 n-k 로 나누면 평균이 참값 sigma^2 에 붙고, n 으로 나누면
(n-k)/n 배만큼 낮다. n = 200, k = 3 이면 그 모자람이 k/n = 1.5% 라 작아
보이는데, 같은 셈이 n = 20 이면 15%, n = 10 이면 30% 다.

왼쪽은 n = 200 에서 두 분포를, 오른쪽은 그 치우침이 n 을 따라 어떻게
커지는지를 보인다. 닫힌 꼴은 E[RSS/(n-k)] = sigma^2 과
E[RSS/n] = sigma^2 (n-k)/n 이고 그림의 수는 뽑아서 확인한 것이다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

OUT = "docs/ch11/estimation_inference/img/"
N, K, SIGMA, REP = 200, 3, 2.0, 5000
BETA = np.array([2.0, 3.0, -1.0])
rng = np.random.default_rng(11)


def rss_draws(n, k, rep):
    """회귀를 rep 번 되풀이해 잔차제곱합을 모은다."""
    out = np.empty(rep)
    for i in range(rep):
        X = np.hstack([np.ones((n, 1)), rng.standard_normal((n, k - 1))])
        y = X @ BETA[:k] + rng.normal(0, SIGMA, n)
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ beta_hat
        out[i] = e @ e
    return out


rss = rss_draws(N, K, REP)
s2 = rss / (N - K)
s2_n = rss / N

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 같은 잔차, 두 분모 ─────────────────────────────────────────
bins = np.linspace(min(s2_n.min(), s2.min()), max(s2.max(), s2_n.max()), 60)
ax1.hist(s2, bins=bins, color="tab:blue", alpha=0.6, label="RSS / (n - k)")
ax1.hist(s2_n, bins=bins, color="tab:red", alpha=0.6, label="RSS / n")
ax1.axvline(SIGMA**2, color="black", ls="--", lw=1.8, zorder=5)
for v, c in [(s2.mean(), "tab:blue"), (s2_n.mean(), "tab:red")]:
    ax1.axvline(v, color=c, lw=2.0, zorder=4)

# 세 선이 거의 포개지므로 — 그것이 요점이다 — 수는 상자에 모아 적는다.
# monospace 에는 한글 글리프가 없어 두부가 되므로 글꼴을 바꾸지 않고 줄을 나눈다.
ax1.text(0.025, 0.97,
         f"참값 σ² = {SIGMA**2:.3f}\n"
         f"RSS/(n-k) 의 평균 = {s2.mean():.3f}  →  참값에 붙는다\n"
         f"RSS/n 의 평균 = {s2_n.mean():.3f}  →  "
         f"{(1 - s2_n.mean() / s2.mean()) * 100:.1f}% 낮다\n\n"
         f"n 이 {N} 이라 눈으로는 거의 겹친다",
         transform=ax1.transAxes, va="top", ha="left", fontsize=10.5,
         bbox=dict(fc="white", ec="0.75", alpha=0.95))

ax1.set_xlabel("분산 추정값")
ax1.set_ylabel(f"되풀이 {REP:,} 회의 빈도")
ax1.set_title(f"n = {N}, k = {K} — 잔차는 하나도 다르지 않고 분모만 다르다",
              fontsize=11)
ax1.legend(fontsize=10, loc="upper right", framealpha=0.95)
ax1.grid(alpha=0.2)

# ── (우) 치우침은 k/n 이다 ──────────────────────────────────────────
NS = [10, 20, 50, 200, 1000]
bias = [K / n * 100 for n in NS]
x = np.arange(len(NS))
cols = ["tab:red" if b > 5 else "tab:orange" if b > 1 else "tab:green" for b in bias]
ax2.bar(x, bias, width=0.55, color=cols, alpha=0.85)
for xi, (n, b) in zip(x, zip(NS, bias)):
    ax2.text(xi, b + 2.4, f"{b:.1f}%", ha="center", va="bottom", fontsize=11.5,
             fontweight="bold")
    # 짧은 막대 안에는 글자가 들어가지 않으므로 늘 막대 위에 적는다
    ax2.text(xi, b + 0.7, f"E[RSS/n] = {SIGMA**2 * (n - K) / n:.3f}", ha="center",
             va="bottom", fontsize=8.5, color="0.4")

ax2.set_xticks(x, [f"n = {n}" for n in NS], fontsize=10.5)
ax2.set_ylim(0, 37)
ax2.set_ylabel("RSS/n 이 참값보다 모자라는 몫")
ax2.set_title(f"그 몫은 정확히 k/n 이다   (k = {K})", fontsize=11)
ax2.grid(axis="y", alpha=0.25)

plt.tight_layout()
path = OUT + "which_denominator.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"되풀이 {REP:,} 회, n = {N}, k = {K}, sigma = {SIGMA}")
print(f"  RSS/(n-k) 평균 {s2.mean():.4f}   (닫힌 꼴 {SIGMA**2:.4f})")
print(f"  RSS/n     평균 {s2_n.mean():.4f}   (닫힌 꼴 {SIGMA**2 * (N - K) / N:.4f})")
print(f"  비 {s2_n.mean() / s2.mean():.6f}  (닫힌 꼴 {(N - K) / N:.6f} = 1 - k/n)")
print(f"  쪽의 모의는 3.996 과 3.936 이다 — 난수씨가 달라 뽑은 값은 다르고 닫힌 꼴이 같다")
for n in NS:
    print(f"    n = {n:>5}: 모자라는 몫 {K / n * 100:>5.1f}%  →  E[RSS/n] = {SIGMA**2 * (n - K) / n:.4f}")
