"""11.2절 생태학적 오류 — 집계된 수는 묶는 방식이 정한다.

로빈슨의 두 수(주 수준 +0.53, 개인 수준 -0.11)가 나오도록 5,760 명을
만든다. 그런 다음 **같은 사람들**을 세 가지로 묶어 평균을 낸다. 주로
묶으면 +0.53 이고, 무작위로 48 덩어리를 지으면 개인 수준 값으로
돌아오며, x 순으로 묶으면 또 다른 수가 나온다.

곧 집계 상관은 사람들의 성질이 아니라 **묶는 방식의 성질**이다.
쪽의 robinson_reversal 은 주로 묶은 한 가지를 개인 수준과 맞댄다.
"""
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
from scipy import optimize

OUT = "docs/ch11/ecological_correlation/img/"
G, M = 48, 120
N = G * M
R_IND, R_STATE, RB = -0.11, 0.53, 0.90
rng = np.random.default_rng(1950)


def solve():
    """집단 간 분산 몫 vb 와 집단 내 상관 rw 를 목표 두 수에서 되짚는다."""
    def eq(p):
        vb, rw = p
        vw = 1 - vb
        return [RB * vb + rw * vw - R_IND,
                (RB * vb + rw * vw / M) / (vb + vw / M) - R_STATE]
    return optimize.fsolve(eq, [0.02, -0.15])


def pair(seed, n, rho, var):
    g = np.random.default_rng(seed)
    a = g.standard_normal(n); b = g.standard_normal(n)
    a = (a - a.mean()) / a.std()
    b = b - b.mean() - (b @ a) / (a @ a) * a
    b = b / b.std()
    return np.sqrt(var) * a, np.sqrt(var) * (rho * a +
                                             np.sqrt(1 - rho ** 2) * b)


def build(vb, rw):
    bx, by = pair(11, G, RB, vb)
    wx, wy = pair(22, N, rw, 1 - vb)
    return np.repeat(bx, M) + wx, np.repeat(by, M) + wy


# 기댓값으로 푼 값은 뽑은 자료에서 조금 빗나간다. 쪽이 적은 두 수(+0.53,
# -0.11)가 **이 자료에서** 그대로 나오도록 vb 를 훑어 맞춘다.
VB0, _ = solve()
best = None
for vb in np.linspace(0.004, 0.08, 400):
    rw = (R_IND - RB * vb) / (1 - vb)
    if abs(rw) >= 1:
        continue
    Xt, Yt = build(vb, rw)
    st = np.array([Xt[np.repeat(np.arange(G), M) == g].mean()
                   for g in range(G)])
    sy = np.array([Yt[np.repeat(np.arange(G), M) == g].mean()
                   for g in range(G)])
    r = float(np.corrcoef(st, sy)[0, 1])
    d = abs(r - R_STATE)
    if best is None or d < best[0]:
        best = (d, vb, rw)
VB, RW = best[1], best[2]
X, Y = build(VB, RW)


def agg_corr(labels):
    mx = np.array([X[labels == g].mean() for g in np.unique(labels)])
    my = np.array([Y[labels == g].mean() for g in np.unique(labels)])
    return float(np.corrcoef(mx, my)[0, 1]), mx, my


state = np.repeat(np.arange(G), M)
rand = np.random.default_rng(33).permutation(
    np.repeat(np.arange(G), M))
bysize = np.repeat(np.arange(G), M)[np.argsort(np.argsort(X))]

WAYS = [("주 48 개로 묶기", state, "tab:blue"),
        ("무작위로 48 덩어리", rand, "0.55"),
        ("$x$ 순으로 48 덩어리", bysize, "tab:green")]
r_ind = float(np.corrcoef(X, Y)[0, 1])

fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharex=True, sharey=True)
for ax, (name, lab, col) in zip(axes, WAYS):
    r, mx, my = agg_corr(lab)
    ax.plot(X, Y, ".", ms=1.6, color="0.88")
    ax.plot(mx, my, "o", ms=8, color=col)
    a, b = np.polyfit(mx, my, 1)
    xs = np.linspace(mx.min(), mx.max(), 2)
    ax.plot(xs, a * xs + b, "-", lw=3.0, color=col)
    ax.set_title(f"{name}\n집계 상관 {r:+.4f}", fontsize=12.5, color=col)
    ax.set_xlabel("$x$", fontsize=11)
    ax.grid(alpha=0.2)
    # 평균 48 개는 가운데 작게 몰리므로 확대해 함께 보인다
    ins = ax.inset_axes([0.56, 0.58, 0.42, 0.40])
    ins.plot(mx, my, "o", ms=4.5, color=col)
    xs2 = np.linspace(mx.min(), mx.max(), 2)
    ins.plot(xs2, a * xs2 + b, "-", lw=2.2, color=col)
    pad = 0.12 * max(mx.ptp(), my.ptp())
    ins.set_xlim(mx.min() - pad, mx.max() + pad)
    ins.set_ylim(my.min() - pad, my.max() + pad)
    ins.set_xticks([]); ins.set_yticks([])
    ins.set_title("평균 48 개만 확대", fontsize=9, color="0.35", pad=3)
    for sp in ins.spines.values():
        sp.set_edgecolor("0.6")
axes[0].set_ylabel("$y$", fontsize=11)
axes[0].set_xlim(-3.6, 3.6); axes[0].set_ylim(-3.6, 3.6)
# 무작위로 묶으면 집계 상관이 개인 수준 값 언저리에서 흔들린다.
# 한 번 뽑은 것만 보이면 오해를 부르므로 500 번 뽑아 평균을 함께 적는다.
gr = np.random.default_rng(77)
many = []
for _ in range(500):
    lab = gr.permutation(np.repeat(np.arange(G), M))
    many.append(agg_corr(lab)[0])
many = np.array(many)
axes[1].text(0.5, 0.045,
             f"500 번 다시 묶으면 평균 {many.mean():+.4f}\n"
             f"(개인 수준 {r_ind:+.4f} 언저리에서 흔들린다)",
             transform=axes[1].transAxes, ha="center", fontsize=10.5,
             color="0.3", fontweight="bold")

fig.suptitle(f"같은 {N:,} 명이다 — 달라진 것은 누구를 한 덩어리로 "
             f"묶었는가뿐이다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "grouping_makes_the_number.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)
print(f"  되짚은 값: 집단 간 분산 몫 {VB:.4f}, 집단 내 상관 {RW:.4f}")
print(f"  개인 수준 상관 {r_ind:+.4f} (목표 {R_IND})")
for name, lab, _ in WAYS:
    print(f"  {name:<22s} 집계 상관 {agg_corr(lab)[0]:+.4f}")
print(f"  무작위 500 번: 평균 {many.mean():+.4f} · 표준편차 "
      f"{many.std(ddof=1):.4f}")
