"""8.2절 p의 신뢰구간 — 같은 95% 가 방법마다 다른 수다.

들머리의 주장은 **방법마다 실제 포함확률이 다르고, 비율이 가장자리로 갈수록
그 차이가 벌어진다**는 것이다. 들머리가 든 조사(전국 1,000 명)가 44% 와 3%
를 한 표에 담고 있어 그 두 자리를 그대로 쓴다.

왼쪽은 들머리로 들어가는 자리다. 발표문 끝줄의 오차한계는 지지율이 50% 일
때의 값이고 오차는 바로 거기서 가장 크다. 그래서 그 한 줄을 표의 모든 수에
갖다 대면 3% 짜리 후보에서 아래끝이 0 아래로 넘어간다 --- 오차를 작게 잡아서가
아니라 세 배 가까이 크게 잡은 탓이다.

오른쪽이 주장이다. 처음에 여기에 **네 구간의 끝점**을 막대로 그렸는데 그것은
들머리의 표가 이미 숫자로 적고 있어 겹쳤고, 더 나쁘게는 끝점의 차이가 0.2%p
남짓이라 **「넷이 거의 같다」로 읽혔다.** 차이가 어디서 드러나는지를 그려야
한다 --- 이항분포를 끝까지 세어 얻은 **실제 포함확률**이다.

중첩을 보았다. 4 절의 wald_wilson_coverage 는 왈드와 윌슨 **둘**을 참 비율
p 전체에 걸쳐 그리고, 5 절의 four_methods_tradeoff 는 n = 50, k = 12 에서
최소 포함확률과 평균 너비의 트레이드오프를 그린다. **이 조사의 두 후보 자리에서
네 방법을 맞대는 것**은 어느 쪽도 하지 않는다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/one_sample_intervals/img/"

N = 1000
K = 30             # 3% 짜리 후보 — 1,000 명 가운데 서른
ANN = 3.1          # 발표문 끝줄의 오차한계 (퍼센트포인트)
Z = stats.norm.ppf(0.975)


def moe(p, n=N):
    return 100 * Z * np.sqrt(p * (1 - p) / n)


def four(k, n=N):
    """같은 자료에 네 방법. 모두 퍼센트로 돌려준다."""
    ph = k / n
    h = Z * np.sqrt(ph * (1 - ph) / n)
    wald = (ph - h, ph + h)
    d = Z * Z / n
    c = (ph + d / 2) / (1 + d)
    hw = (Z / (1 + d)) * np.sqrt(ph * (1 - ph) / n + d / (4 * n))
    wilson = (c - hw, c + hw)
    nt = n + Z * Z
    pt = (k + Z * Z / 2) / nt
    ht = Z * np.sqrt(pt * (1 - pt) / nt)
    ac = (max(pt - ht, 0.0), pt + ht)
    cp = (stats.beta.ppf(0.025, k, n - k + 1),
          stats.beta.ppf(0.975, k + 1, n - k))
    return [("왈드 z-구간", wald), ("윌슨", wilson),
            ("아그레스티–코울", ac), ("클로퍼–피어슨", cp)]


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.8),
                               gridspec_kw={"width_ratios": [1.2, 1]})

# ── (좌) 오차한계는 50% 에서 가장 크다 ──────────────────────────────
g = np.linspace(0.001, 0.999, 999)
ax1.plot(100 * g, moe(g), color="tab:blue", lw=2.4)
ax1.axhline(ANN, color="tab:red", ls="--", lw=1.8)
ax1.text(99, ANN + 0.07, f"발표문의 ±{ANN} 퍼센트포인트", ha="right",
         fontsize=11, color="tab:red", fontweight="bold")
SPOT = ((0.50, "50% 자리", "0.3", (16, -0.45), "left"),
        (0.44, "44% 후보", "tab:green", (-17, -0.75), "right"),
        (0.03, "3% 후보", "tab:orange", (7, -0.55), "left"))
for p, name, c, (dx, dy), ha in SPOT:
    m = moe(p)
    ax1.plot([100 * p], [m], "o", ms=9, color=c)
    ax1.annotate(f"{name}\n±{m:.2f}", xy=(100 * p, m),
                 xytext=(100 * p + dx, m + dy),
                 ha=ha, fontsize=10.5, color=c, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=c, lw=1.4))
ax1.annotate(f"이 수를 3% 에 그대로 쓰면\n아래끝이 "
             f"{3 - ANN:.2f}% — 음수다",
             xy=(3.2, ANN - 0.02), xytext=(34, 0.88), fontsize=10.5,
             color="tab:red", fontweight="bold", ha="left",
             arrowprops=dict(arrowstyle="->", color="tab:red", lw=1.6))
ax1.set_xlim(0, 100)
ax1.set_ylim(0, 3.75)
ax1.set_xlabel("지지율 (%)", fontsize=11)
ax1.set_ylabel("오차한계 (퍼센트포인트)", fontsize=11)
ax1.set_title(f"오차한계는 50% 에서 가장 크다 (n = {N:,})", fontsize=12.5)
ax1.grid(alpha=0.25)

# ── (우) 같은 95% 가 실제로는 몇 퍼센트인가 ────────────────────────
# 이항분포를 끝까지 세므로 모의실험 오차가 없다
def cover(fn, n, p):
    ks = np.arange(n + 1)
    w = stats.binom.pmf(ks, n, p)
    hit = np.array([1.0 if lo <= p <= hi else 0.0
                    for lo, hi in (fn(k, n) for k in ks)])
    return float((w * hit).sum())


def by_name(name):
    return lambda k, n: dict(four_raw(k, n))[name]


def four_raw(k, n):
    """k = 0 과 k = n 까지 받는 판본 — 포함확률을 셀 때 끝을 지나간다."""
    ph = k / n
    h = Z * np.sqrt(ph * (1 - ph) / n)
    d = Z * Z / n
    c = (ph + d / 2) / (1 + d)
    hw = (Z / (1 + d)) * np.sqrt(ph * (1 - ph) / n + d / (4 * n))
    nt = n + Z * Z
    pt = (k + Z * Z / 2) / nt
    ht = Z * np.sqrt(pt * (1 - pt) / nt)
    lo_cp = 0.0 if k == 0 else stats.beta.ppf(0.025, k, n - k + 1)
    hi_cp = 1.0 if k == n else stats.beta.ppf(0.975, k + 1, n - k)
    return [("왈드 z-구간", (ph - h, ph + h)),
            ("윌슨", (c - hw, c + hw)),
            ("아그레스티–코울", (max(pt - ht, 0.0), pt + ht)),
            ("클로퍼–피어슨", (lo_cp, hi_cp))]


NAMES = [nm for nm, _ in four_raw(K, N)]
COLS = ["tab:red", "tab:blue", "tab:green", "tab:purple"]
CANDS = [(0.44, "44% 후보 — 꼭대기 옆"), (0.03, "3% 후보 — 비탈 아래")]
cov = {p: [cover(by_name(nm), N, p) for nm in NAMES] for p, _ in CANDS}

xx = np.arange(len(CANDS))
w_ = 0.19
for j, nm in enumerate(NAMES):
    vals = [cov[p][j] for p, _ in CANDS]
    ax2.bar(xx + (j - 1.5) * w_, vals, width=w_, color=COLS[j], label=nm)
    for i, v in enumerate(vals):
        ax2.text(xx[i] + (j - 1.5) * w_, v + 0.0012, f"{v:.4f}",
                 ha="center", va="bottom", fontsize=9.6, color=COLS[j],
                 fontweight="bold", rotation=90)
ax2.axhline(0.95, color="0.15", lw=2.4, ls="--")
ax2.text(len(CANDS) - 0.52, 0.9508, "약속한 0.95", ha="right", va="bottom",
         fontsize=11, color="0.15", fontweight="bold")
ax2.set_xticks(xx, [lab for _, lab in CANDS], fontsize=11.5)
ax2.set_ylabel("실제 포함확률", fontsize=11)
ax2.set_ylim(0.9245, 0.9685)
ax2.legend(fontsize=9.8, ncols=2, loc="upper left", framealpha=0.95)
ax2.grid(axis="y", alpha=0.25)
ax2.set_title(f"그 「95%」가 실제로 몇 퍼센트인가 (n = {N:,})\n"
              f"3% 자리에서 왈드만 {cov[0.03][0]:.4f} — 스물에 하나라 "
              f"적고 {1 / (1 - cov[0.03][0]):.0f} 에 하나를 놓친다",
              fontsize=12.5)
fig.suptitle("같은 95% 라고 적는 네 방법이 같은 수를 주지 않는다 — 갈리는 자리는 가장자리다", fontsize=14, y=1.0)
plt.tight_layout()
path = OUT + "moe_curve_small_p.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N}, 95% 오차한계")
for p in (0.50, 0.44, 0.03):
    print(f"  p = {p:.2f} 에서 ±{moe(p):.4f} 퍼센트포인트")
print(f"  발표문의 ±{ANN} 을 3% 에 그대로 쓰면 아래끝 {3 - ANN:.2f}% (음수)")
print(f"\nk = {K} (3%) 에 네 구간")
for name, (lo, hi) in four(K):
    print(f"  {name:<22s} ({100*lo:7.4f}, {100*hi:7.4f}) %")
print(f"\n명목 0.95 의 실제 포함확률 (이항분포를 끝까지 세었다)")
for pp, lab in CANDS:
    print(f"  {lab}")
    for nm, v in zip(NAMES, cov[pp]):
        print(f"    {nm:<18s} {v:.4f}   (스무 번에 "
              f"{20 * (1 - v):.2f} 번 놓친다)")
