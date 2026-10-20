r"""1933 년 콜모고로프 분포의 두 얼굴 — 분포무관성과 수렴 속도.

왼쪽: 서로 전혀 다른 연속분포 넷에서 뽑아도 $\sqrt{n} D_n$ 의 분포가 하나로 겹친다.
오른쪽: 글리벤코-칸텔리가 "0 으로 간다"고 말한 그 속도가 $1/\sqrt n$ 임을 로그-로그로 본다.

만드는 파일: docs/ch14/formal_tests/img/kolmogorov_1933.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

RNG = np.random.default_rng(20261020)
REP = 20000
N_LEFT = 200
NS = np.array([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])

# === 콜모고로프의 극한분포 =========================================
# K(t) = 1 - 2 sum_{k>=1} (-1)^{k-1} exp(-2 k^2 t^2)


def K(t, terms=100):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    pos = t > 0
    k = np.arange(1, terms + 1)[:, None]
    s = ((-1) ** (k - 1) * np.exp(-2 * k**2 * t[pos] ** 2)).sum(axis=0)
    out[pos] = 1 - 2 * s
    return np.clip(out, 0, 1)


# === D_n 을 균등난수 하나로 다 잰다 ================================
# 확률적분변환: F 가 연속이면 F(X) ~ U(0,1) 이고 D_n 은 F 에 무관하다.
# 그래도 "무관하다"를 보이는 것이 목적이니, 실제로 네 분포에서 뽑아
# 각자의 F 로 변환해 D_n 을 잰다.


def dn(u):
    """u: (rep, n) 각 행이 F(x_i) 값. 정렬해 sup 편차를 낸다."""
    rep, n = u.shape
    v = np.sort(u, axis=1)
    i = np.arange(1, n + 1) / n
    return np.maximum(v - (i - 1 / n), i - v).max(axis=1)


from scipy import stats  # noqa: E402

FAMILY = [
    ("정규 $N(0,1)$", stats.norm()),
    ("지수 $\\mathrm{Exp}(1)$", stats.expon()),
    ("코시", stats.cauchy()),
    ("로그정규", stats.lognorm(s=1.0)),
]

left = []
for name, dist in FAMILY:
    x = dist.rvs(size=(REP, N_LEFT), random_state=RNG)
    left.append((name, np.sqrt(N_LEFT) * dn(dist.cdf(x))))

# === 오른쪽: D_n 의 평균과 95 백분위가 1/sqrt(n) 으로 줄어든다 =====
mean_d, q95_d = [], []
for n in NS:
    u = RNG.random((REP, int(n)))
    d = dn(u)
    mean_d.append(d.mean())
    q95_d.append(np.quantile(d, 0.95))
mean_d = np.array(mean_d)
q95_d = np.array(q95_d)

# === 그림 ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.0))

grid = np.linspace(0, 2.6, 600)
COL = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
for (name, s), c, w in zip(left, COL, [6.5, 4.6, 2.9, 1.5]):
    xs = np.sort(s)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax1.plot(xs, ys, color=c, lw=w, alpha=0.75, label=name, solid_capstyle="butt")
ax1.plot(grid, K(grid), color="black", lw=1.4, ls="--", label="콜모고로프 $K(t)$")
ax1.set_xlim(0.2, 2.2)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel("$t = \\sqrt{n}\\,D_n$   ($n = 200$, 각 $20{,}000$ 회)")
ax1.set_ylabel("누적확률")
ax1.set_title("① 어느 분포에서 뽑아도 같은 자 하나", fontsize=12)
ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax1.grid(alpha=0.25)
ax1.axvline(1.358, color="#555555", lw=1.0, ls=":")
ax1.annotate(
    "$K(1.358) = 0.95$\n→ 표의 $5\\%$ 임계값",
    xy=(1.358, 0.95),
    xytext=(1.44, 0.58),
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

ax2.loglog(NS, mean_d, "o-", color="#1f77b4", lw=2.0, ms=6, label="$E[D_n]$")
ax2.loglog(NS, q95_d, "s-", color="#d62728", lw=2.0, ms=6, label="$D_n$ 의 $95$ 백분위")
ax2.loglog(
    NS,
    1.358 / np.sqrt(NS.astype(float)),
    "--",
    color="black",
    lw=1.3,
    label="$1.358/\\sqrt{n}$",
)
ax2.set_xlabel("표본 크기 $n$")
ax2.set_ylabel("$\\sup_x |F_n(x) - F(x)|$")
ax2.set_title("② 0 으로 가는 속도는 $1/\\sqrt{n}$", fontsize=12)
ax2.legend(loc="lower left", fontsize=10)
ax2.grid(alpha=0.25, which="both")
ax2.set_xticks([10, 100, 1000])
ax2.set_xticklabels(["$10$", "$100$", "$1000$"])
ax2.set_yticks([0.01, 0.1, 0.5])
ax2.set_yticklabels(["$0.01$", "$0.1$", "$0.5$"])
ax2.minorticks_off()
ax2.annotate(
    "글리벤코·칸텔리: 0 으로 간다\n콜모고로프: 이 속도로, 이 분포로",
    xy=(200, 0.0955),
    xytext=(300, 0.30),
    ha="center",
    fontsize=9.5,
    arrowprops=dict(arrowstyle="->", color="#555555", lw=1.0),
)

fig.tight_layout()
out = "docs/ch14/formal_tests/img/kolmogorov_1933.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)

# === 본문에 적을 수 =================================================
print("\n왼쪽 — 네 분포에서 잰 sqrt(n) D_n 의 분위수 (n=200)")
print(f"{'분포':<16}{'중앙값':>9}{'95%':>9}{'99%':>9}")
for name, s in left:
    print(
        f"{name:<16}{np.quantile(s, .5):>9.4f}{np.quantile(s, .95):>9.4f}"
        f"{np.quantile(s, .99):>9.4f}"
    )
from scipy.optimize import brentq  # noqa: E402

print(
    f"{'콜모고로프 극한':<16}"
    f"{brentq(lambda t: K(t) - .50, .3, 3):>9.4f}"
    f"{brentq(lambda t: K(t) - .95, .3, 3):>9.4f}"
    f"{brentq(lambda t: K(t) - .99, .3, 3):>9.4f}"
)

print("\n급수가 몇 항에서 멎는가 (t = 1.358)")
t = 1.358
for m in (1, 2, 3):
    k = np.arange(1, m + 1)
    v = 1 - 2 * ((-1) ** (k - 1) * np.exp(-2 * k**2 * t**2)).sum()
    print(f"  {m} 항까지: {v:.9f}")
print(f"  극한값   : {K(np.array([t]))[0]:.9f}")

print("\n오른쪽 — 기울기 (로그-로그 최소제곱)")
for lab, y in (("평균", mean_d), ("95 백분위", q95_d)):
    sl = np.polyfit(np.log(NS.astype(float)), np.log(y), 1)[0]
    print(f"  {lab}: {sl:.4f}")
print(f"  n=5 의 95 백분위 {q95_d[0]:.4f} 대 1.358/sqrt(5) = {1.358 / np.sqrt(5):.4f}")
print(
    f"  n=5000 의 95 백분위 {q95_d[-1]:.5f} 대 "
    f"1.358/sqrt(5000) = {1.358 / np.sqrt(5000):.5f}"
)
