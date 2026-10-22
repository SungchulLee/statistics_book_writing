r"""튜키의 **경첩**은 사분위수가 아니다.

경첩 — 자료를 반으로 가르고 각 반의 중앙값. $n$ 이 홀수면 중앙값을 양쪽 반에
모두 넣는다. 보간이 아니라 **또 하나의 순서통계량**이므로, 보간법을 어떻게 바꾸어도
경첩이 나오지 않는 $n$ 이 많다.

① $\mathcal{N}(0,1)$ 에서 경첩과 보간 사분위수의 기댓값 차이를 $n$ 을 따라 잰다.
② 울타리 판정이 갈리는 자료 하나 — 같은 $12$ 개인데 한쪽에서는 이상점이고
   다른 쪽에서는 아니다.

만드는 파일: docs/ch14/graphical_methods/img/tukey_hinge.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats  # noqa: E402

RNG = np.random.default_rng(19770101)
REP = 200_000


def hinges(x):
    """튜키의 경첩. x 는 정렬된 1 차원 배열."""
    n = len(x)
    half = (n + 1) // 2  # n 이 홀수면 중앙값이 양쪽에 모두 들어간다
    return np.median(x[:half]), np.median(x[n - half:])


def hinges_rows(xs):
    """(rep, n) 의 각 행에 대한 경첩. 행은 정렬되어 있다고 본다."""
    n = xs.shape[1]
    half = (n + 1) // 2
    return np.median(xs[:, :half], axis=1), np.median(xs[:, n - half:], axis=1)


# === ① n 마다 경첩과 보간 사분위수는 얼마나 다른가 ====================
NS = np.arange(5, 61)
d_lo, h_lo, q_lo = [], [], []
for n in NS:
    xs = np.sort(RNG.standard_normal((REP, n)), axis=1)
    lo, _ = hinges_rows(xs)
    q = np.quantile(xs, 0.25, axis=1, method="linear")
    h_lo.append(lo.mean())
    q_lo.append(q.mean())
    d_lo.append((lo - q).mean())
d_lo = np.array(d_lo)
h_lo = np.array(h_lo)
q_lo = np.array(q_lo)

IQR0 = 2 * stats.norm.ppf(0.75)  # 참 사분위범위 1.3490
same = np.abs(d_lo) < 1e-3
print(f"① N(0,1), 반복 {REP:,} — 아래쪽 경첩과 보간 Q1 의 기댓값 차 (참 IQR = {IQR0:.4f})")
print("   n    경첩     보간 Q1        차    IQR 대비")
for n, h, q, d in zip(NS, h_lo, q_lo, d_lo):
    if n <= 16 or n in (20, 24, 30, 40, 60):
        mark = "  ← 같다" if abs(d) < 1e-3 else ""
        print(f"{n:>4}  {h:+.4f}   {q:+.4f}  {d:+.5f}   {d / IQR0:+.4f}{mark}")
print(f"   둘이 일치하는 n: {list(NS[same])}")
print(f"   가장 크게 벌어지는 n: {NS[np.argmax(np.abs(d_lo))]} "
      f"(차 {d_lo[np.argmax(np.abs(d_lo))]:+.4f}, IQR 의 "
      f"{abs(d_lo[np.argmax(np.abs(d_lo))]) / IQR0 * 100:.1f}%)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.6, 5.2),
                               gridspec_kw={"width_ratios": [1.05, 1.0]})

ax1.axhline(0, color="#777777", lw=1.0)
ax1.plot(NS, d_lo / IQR0 * 100, "o-", color="#1f77b4", lw=1.8, ms=4.5)
ax1.plot(NS[same], d_lo[same] / IQR0 * 100, "o", color="#2ca02c", ms=8,
         label="둘이 꼭 같아지는 $n$")
ax1.set_xlabel("표본 크기 $n$")
ax1.set_ylabel("(경첩 $-$ 보간 $Q_1$) $\\div$ 참 IQR, 백분율")
ax1.set_title("① 같은 자료에서 상자의 모서리가 어디에 놓이는가", fontsize=12)
ax1.legend(loc="lower right", fontsize=10)
ax1.grid(alpha=0.25)
ax1.set_xlim(4, 61)

# === ② 울타리 판정이 갈리는 자료 =====================================
DATA = np.array([5, 8, 10, 20, 35, 45, 55, 70, 80, 90, 120, 185], dtype=float)
lo_h, hi_h = hinges(DATA)
lo_q = np.quantile(DATA, 0.25, method="linear")
hi_q = np.quantile(DATA, 0.75, method="linear")
rules = [
    ("튜키의 경첩", lo_h, hi_h, "#d9a441", "#6b4e16"),
    ("보간 사분위수", lo_q, hi_q, "#1f77b4", "#13405f"),
]
print(f"\n② 자료 {DATA.astype(int).tolist()}")
for name, lo, hi, _, _ in rules:
    spread = hi - lo
    fence = (lo - 1.5 * spread, hi + 1.5 * spread)
    out = DATA[(DATA < fence[0]) | (DATA > fence[1])]
    print(f"   {name:<8} 모서리 {lo:7.3f} {hi:7.3f}   폭 {spread:7.3f}   "
          f"울타리 {fence[0]:8.3f} {fence[1]:8.3f}   이상점 {out.astype(int).tolist()}")

for k, (name, lo, hi, fc, ec) in enumerate(rules):
    y = 1 - k
    spread = hi - lo
    f_lo, f_hi = lo - 1.5 * spread, hi + 1.5 * spread
    inside = DATA[(DATA >= f_lo) & (DATA <= f_hi)]
    out = DATA[(DATA < f_lo) | (DATA > f_hi)]
    med = float(np.median(DATA))
    ax2.add_patch(plt.Rectangle((lo, y - 0.17), spread, 0.34, fc=fc, ec=ec, lw=1.6,
                                alpha=0.85))
    ax2.plot([med, med], [y - 0.17, y + 0.17], color=ec, lw=2.6)
    ax2.plot([inside.min(), lo], [y, y], color=ec, lw=1.4)
    ax2.plot([hi, inside.max()], [y, y], color=ec, lw=1.4)
    for v in (inside.min(), inside.max()):
        ax2.plot([v, v], [y - 0.09, y + 0.09], color=ec, lw=1.6)
    ax2.plot(out, [y] * len(out), "o", mfc="none", mec="#d62728", mew=2.0, ms=10)
    ax2.plot([f_hi, f_hi], [y - 0.26, y + 0.26], color="#d62728", lw=1.3, ls="--")
    ax2.text(f_hi, y + 0.30, f"위 울타리 {f_hi:.1f}", color="#d62728", fontsize=10,
             ha="center")
    ax2.text(-2, y, name, ha="right", va="center", fontsize=11.5)

ax2.plot(DATA, [-0.62] * len(DATA), "|", color="#555555", ms=12, mew=1.4)
ax2.text(-2, -0.62, "자료 $12$ 개", ha="right", va="center", fontsize=10.5,
         color="#555555")
ax2.set_xlim(-55, 215)
ax2.set_ylim(-0.95, 1.65)
ax2.set_yticks([])
ax2.set_xlabel("값")
ax2.set_title("② 같은 $12$ 개, 갈리는 판정", fontsize=12)
ax2.grid(alpha=0.25, axis="x")
ax2.annotate(
    "$185$ — 한쪽에서만 이상점이다",
    xy=(185, 1.0), xytext=(60, 1.42), fontsize=11.5, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)

fig.tight_layout()
out_path = "docs/ch14/graphical_methods/img/tukey_hinge.png"
fig.savefig(out_path, dpi=150, facecolor="white", bbox_inches="tight")
print("\nsaved", out_path)

# === ③ 어떤 보간법도 경첩을 모든 n 에서 재현하지 못한다 ==============
print("\n③ 보간법 다섯 가지가 경첩과 일치하는 n (5 에서 40 까지)")
METHODS = ["linear", "lower", "higher", "midpoint", "nearest"]
agree = {m: [] for m in METHODS}
for n in range(5, 41):
    x = np.sort(RNG.standard_normal(n))      # 아무 자료나 쓰면 된다 — 동점만 없으면
    lo, hi = hinges(x)
    for m in METHODS:
        if (abs(np.quantile(x, 0.25, method=m) - lo) < 1e-12
                and abs(np.quantile(x, 0.75, method=m) - hi) < 1e-12):
            agree[m].append(n)
for m in METHODS:
    odd = [n for n in agree[m] if n % 2]
    even = [n for n in agree[m] if n % 2 == 0]
    print(f"   {m:<10} 홀수 {len(odd):>2} 개 · 짝수 {len(even):>2} 개"
          f"   짝수에서 맞는 n: {even}")
print("   어느 한 가지도 36 개 n 을 다 맞추지 못한다.")
