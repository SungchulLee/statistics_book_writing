r"""Cochran 의 $Q$ 는 피험자가 몇 명부터 믿을 만한가.

귀무가설($k$ 과제의 성공확률이 모두 같다) 아래에서 $X_{ij} \sim \text{Bern}(p)$ 를
만들고 $\chi^2_{k-1}$ 임계값으로 판정했을 때의 **실제 기각률**을 잰다.

① 피험자 수 $n$ 을 가로축에 두면 $(k, p)$ 마다 곡선이 흩어진다.
② 같은 점들을 **정보를 주는 피험자의 기대 수**
   $n\,[1 - p^k - (1-p)^k]$ 에 대해 그리면 한 곡선으로 모인다.

만드는 파일: docs/ch13/test/img/cochran_q_size.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from matplotlib.ticker import FuncFormatter  # noqa: E402
from scipy import stats  # noqa: E402

RNG = np.random.default_rng(19500101)
REP = 60_000
ALPHA = 0.05


def size(n, k, p, rep=REP):
    """Q 를 chi2_{k-1} 로 판정했을 때의 실제 기각률."""
    x = (RNG.random((rep, n, k)) < p).astype(np.float64)
    T = x.sum(1)                       # 과제별 성공 수 (rep, k)
    L = x.sum(2)                       # 사람별 성공 수 (rep, n)
    num = k * (k - 1) * ((T - T.mean(1, keepdims=True)) ** 2).sum(1)
    den = k * L.sum(1) - (L**2).sum(1)
    q = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return float((q > stats.chi2.ppf(1 - ALPHA, k - 1)).mean())


SETTINGS = [(3, 0.5), (3, 0.3), (3, 0.1), (5, 0.5), (5, 0.2), (4, 0.4)]
NS = np.array([4, 6, 8, 10, 13, 16, 20, 26, 34, 45, 60, 80, 110, 150, 200])

res = {}
print(f"① 실제 기각률 (약속한 {ALPHA}, 반복 {REP:,})")
for k, p in SETTINGS:
    res[(k, p)] = np.array([size(n, k, p) for n in NS])
    info = 1 - p**k - (1 - p) ** k
    print(f"\n  k = {k}, p = {p}   (정보를 주는 비율 {info:.4f})")
    print("      n  " + "".join(f"{n:>8}" for n in NS[:8]))
    print("    크기 " + "".join(f"{v:>8.4f}" for v in res[(k, p)][:8]))
    print("      n  " + "".join(f"{n:>8}" for n in NS[8:]))
    print("    크기 " + "".join(f"{v:>8.4f}" for v in res[(k, p)][8:]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.4, 5.0))

COL = {3: "#1f77b4", 4: "#2ca02c", 5: "#d9a441"}
MRK = {0.5: "o", 0.4: "^", 0.3: "s", 0.2: "D", 0.1: "v"}

for (k, p), v in res.items():
    info = 1 - p**k - (1 - p) ** k
    lab = f"$k = {k}$,  $p = {p}$"
    ax1.semilogx(NS, v, MRK[p] + "-", color=COL[k], lw=1.6, ms=5, label=lab)
    ax2.semilogx(NS * info, v, MRK[p] + "-", color=COL[k], lw=1.6, ms=5, label=lab)

for ax, xlab, title in [
    (ax1, "피험자 수 $n$", "① 사람 수로 보면 여섯 곡선이 흩어진다"),
    (ax2, "정보를 주는 피험자의 기대 수  $n\\,[1 - p^k - (1-p)^k]$",
     "② 그 수로 보면 $p$ 가 사라지고 $k$ 만 남는다"),
]:
    ax.axhline(ALPHA, color="#333333", lw=1.8, ls="--")
    ax.axhspan(0.04, 0.06, color="#2ca02c", alpha=0.10)
    ax.set_xlabel(xlab)
    ax.set_ylabel("실제 기각률")
    ax.set_ylim(0, 0.085)
    ax.set_title(title, fontsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax1.legend(loc="lower right", fontsize=9)
ax2.text(0.97, 0.06, "초록 띠는 $0.04$–$0.06$", transform=ax2.transAxes,
         ha="right", fontsize=10, color="#2a7a2a")

fig.tight_layout()
out = "docs/ch13/test/img/cochran_q_size.png"
fig.savefig(out, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out)

# === 띠 안으로 들어오는 자리 =========================================
print("\n② 실제 기각률이 0.04–0.06 안에 들어와 머무는 첫 자리")
print("   k    p    n     정보를 주는 사람 수")
for (k, p), v in res.items():
    info = 1 - p**k - (1 - p) ** k
    idx = None
    for i in range(len(NS)):
        if np.all((v[i:] >= 0.04) & (v[i:] <= 0.06)):
            idx = i
            break
    if idx is None:
        print(f"  {k}  {p:.1f}   눈금 안에서 들어오지 않는다")
    else:
        print(f"  {k}  {p:.1f}  {NS[idx]:>4}   {NS[idx] * info:>8.1f}")
