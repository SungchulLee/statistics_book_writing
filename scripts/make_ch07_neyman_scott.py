r"""네이만과 스콧의 이름이 붙은 두 자리.

① **네이만–스콧 문제** — 짝 자료 $X_{i1}, X_{i2} \sim \mathcal{N}(\mu_i, \sigma^2)$.
   관측은 $2m$ 개인데 모수가 $m+1$ 개라서, $m \to \infty$ 여도 MLE 가
   $\sigma^2/2$ 에 머문다. 일치성이 깨진다.
② 견줌 — 고른 포아송 점들.
③ **네이만–스콧 과정** — 보이지 않는 어미점 둘레에 새끼점이 흩어진다.
   은하의 뭉침을 적으려고 만든 모형이다.

①과 ③ 모두 **자료와 함께 늘어나는, 보이지 않는 양**이 한가운데 있다.

만드는 파일: docs/ch07/theory/img/neyman_scott.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402

RNG = np.random.default_rng(19480101)
SIG2 = 4.0  # 참 분산


def mle_sigma2(m, rep=400):
    """짝 자료 m 쌍에서 (mu_i, sigma^2) 의 MLE. sigma^2 쪽만 돌려준다."""
    x = RNG.normal(0.0, np.sqrt(SIG2), (rep, m, 2))  # mu_i 는 0 으로 두어도 된다
    d = x - x.mean(axis=2, keepdims=True)            # mu_i 의 MLE 로 중심화
    return (d**2).sum(axis=(1, 2)) / (2 * m)         # 관측 2m 개로 나눈다


MS = np.unique(np.round(np.logspace(np.log10(5), 5, 22)).astype(int))
mean_mle, lo, hi = [], [], []
for m in MS:
    rep = 400 if m <= 10_000 else 120
    v = mle_sigma2(m, rep)
    mean_mle.append(v.mean())
    lo.append(np.quantile(v, 0.05))
    hi.append(np.quantile(v, 0.95))
mean_mle, lo, hi = map(np.array, (mean_mle, lo, hi))

print(f"① 짝 자료의 MLE — 참 분산 {SIG2}, 그 절반은 {SIG2/2}")
print("        m    관측 수   모수 수    MLE 평균   보정하면")
for target in [5, 20, 100, 1_000, 10_000, 100_000]:
    i = int(np.argmin(np.abs(np.log(MS) - np.log(target))))
    m, v = int(MS[i]), mean_mle[i]
    print(f"  {m:>7,}   {2*m:>8,}  {m+1:>8,}    {v:8.5f}   {2*v:8.5f}")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.6, 4.8),
                                    gridspec_kw={"width_ratios": [1.65, 1, 1]})

# === ① 일치성이 깨지는 모습 ==========================================
ax1.fill_between(MS, lo, hi, color="#1f77b4", alpha=0.18, label="$90\\%$ 범위")
ax1.semilogx(MS, mean_mle, "o-", color="#1f77b4", lw=2.2, ms=5,
             label="$\\hat\\sigma^2$ 의 MLE")
ax1.semilogx(MS, 2 * mean_mle, "s-", color="#2ca02c", lw=2.0, ms=4.5,
             label="두 배로 보정한 것")
ax1.axhline(SIG2, color="#333333", lw=1.8, ls="--")
ax1.axhline(SIG2 / 2, color="#d62728", lw=1.6, ls=":")
ax1.text(6, SIG2 + 0.13, f"참값 $\\sigma^2 = {SIG2:.0f}$", fontsize=10.5)
ax1.text(6, SIG2 / 2 + 0.13, f"$\\sigma^2/2 = {SIG2/2:.0f}$", fontsize=10.5,
         color="#d62728")
ax1.set_xlabel("짝의 수 $m$ — 관측은 $2m$ 개, 모수는 $m+1$ 개")
ax1.set_ylabel("$\\hat\\sigma^2$")
ax1.set_ylim(1.2, 5.2)
ax1.set_title("① 자료를 아무리 모아도 절반에 머문다", fontsize=12)
ax1.legend(loc="center right", fontsize=9.5)
ax1.grid(alpha=0.25, which="both")
ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

# === ②③ 고른 점과 뭉친 점 ===========================================
N_POINTS = 700
pois = RNG.random((N_POINTS, 2))

LAM_PARENT, SPREAD = 28, 0.035
parents = RNG.random((LAM_PARENT, 2))
counts = RNG.poisson(N_POINTS / LAM_PARENT, LAM_PARENT)
kids = np.concatenate([
    p + RNG.normal(0, SPREAD, (c, 2)) for p, c in zip(parents, counts) if c > 0
])
kids = kids[(kids >= 0).all(1) & (kids <= 1).all(1)]

for ax, pts, title, col in [
    (ax2, pois, "② 견줌 — 고르게 흩뿌린 점", "#777777"),
    (ax3, kids, "③ 네이만–스콧 과정", "#d9a441"),
]:
    ax.plot(pts[:, 0], pts[:, 1], ".", color=col, ms=4.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
ax3.plot(parents[:, 0], parents[:, 1], "x", color="#1f77b4", ms=7, mew=1.8)
ax3.text(0.5, -0.055, "파란 $\\times$ 가 어미점 — 자료에는 보이지 않는다",
         transform=ax3.transAxes, ha="center", fontsize=10, color="#1f77b4")
ax2.text(0.5, -0.055, f"점 {len(pois)} 개", transform=ax2.transAxes, ha="center",
         fontsize=10, color="#555555")

fig.tight_layout()
out = "docs/ch07/theory/img/neyman_scott.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

print(f"\n③ 점의 수 — 고른 쪽 {len(pois)}, 뭉친 쪽 {len(kids)} "
      f"(어미점 {LAM_PARENT} 개, 흩어짐 {SPREAD})")
print("   뭉침을 수로 — 가장 가까운 이웃까지의 거리 중앙값")
for name, pts in [("고른 점", pois), ("뭉친 점", kids)]:
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    print(f"     {name}  {np.median(d.min(1)):.5f}")
