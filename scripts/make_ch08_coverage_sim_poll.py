"""8.1절 신뢰수준과 포함확률 — 발표문의 95 를 되풀이로 채점한다.

들머리가 든 조사(전국 성인 1,000 명, 대통령 직무 수행 긍정 39%)를 그대로
되풀이한다. 참 지지율이 꼭 0.39 라 가정하고 같은 크기의 조사를 100 번 하면
구간이 100 개 나오고, 그 가운데 몇은 참값을 놓친다. 놓친 구간에는 아무 표시가
없다는 것이 요점이다.

오른쪽은 되풀이를 늘려 가며 담은 비율을 누적으로 센 것이다. 왈드 구간의
실제 포함확률은 이항이 유한하므로 모의 없이 정확히 계산할 수 있고,
n = 1000, p = 0.39 에서 0.9482 다 --- 명목 0.95 가 아니다.
"""

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
from scipy import stats

OUT = "docs/ch08/foundations/img/"

N = 1000          # 발표문의 표본 크기
P = 0.39          # 참 지지율이 꼭 이 값이라 가정한다
SHOW = 100        # 왼쪽에 그릴 조사 횟수
LONG = 20_000     # 오른쪽에서 누적으로 셀 횟수
Z = stats.norm.ppf(0.975)

# ── 왈드 구간의 실제 포함확률 (이항이 유한하므로 정확히 센다) ──────────
k = np.arange(N + 1)
pmf = stats.binom.pmf(k, N, P)
ph_all = k / N
se_all = np.sqrt(ph_all * (1 - ph_all) / N)
exact = pmf[(ph_all - Z * se_all <= P) & (P <= ph_all + Z * se_all)].sum()

rng = np.random.default_rng(39)
ks = rng.binomial(N, P, LONG)
ph = ks / N
se = np.sqrt(ph * (1 - ph) / N)
lo, hi = ph - Z * se, ph + Z * se
hit = (lo <= P) & (P <= hi)

n_miss = int((~hit[:SHOW]).sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.2),
                               gridspec_kw={"width_ratios": [1.25, 1]})

# ── (좌) 조사 100 번, 구간 100 개 ──────────────────────────────────
y = np.arange(SHOW)
for i in y:
    c = "tab:blue" if hit[i] else "tab:red"
    ax1.plot([lo[i] * 100, hi[i] * 100], [i, i], "-", color=c,
             lw=1.6 if hit[i] else 2.6, alpha=0.85 if hit[i] else 1.0)
    ax1.plot(ph[i] * 100, i, "o", color=c, ms=2.8)
ax1.axvline(P * 100, color="0.15", lw=2.0)
ax1.text(P * 100 + 0.12, SHOW + 1.5, f"참 지지율 {P * 100:.0f}%",
         fontsize=11, color="0.15", fontweight="bold")
ax1.set_xlabel("지지율 (%)", fontsize=11)
ax1.set_ylabel("조사 번호", fontsize=11)
ax1.set_ylim(-2, SHOW + 6)
ax1.set_title(f"같은 조사를 {SHOW} 번 되풀이하면 구간이 {SHOW} 개 나온다",
              fontsize=12.5)
ax1.grid(axis="x", alpha=0.2)
ax1.text(0.015, 0.015,
         f"참값을 놓친 구간 {n_miss} 개 (붉은 선)\n"
         "— 겉모습으로는 가려낼 수 없다",
         transform=ax1.transAxes, fontsize=10.8, color="tab:red",
         fontweight="bold", va="bottom",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.92,
                   edgecolor="tab:red", lw=1.0))

# ── (우) 되풀이를 늘려 가며 센 포함 비율 ────────────────────────────
run = np.cumsum(hit) / np.arange(1, LONG + 1)
START = 20          # 한 자릿수 되풀이는 0 과 1 사이를 튀어 읽을 것이 없다
ax2.plot(np.arange(START, LONG + 1), run[START - 1:], color="tab:blue", lw=1.3)
ax2.axhline(0.95, color="0.35", ls="--", lw=1.5)
ax2.axhline(exact, color="tab:red", lw=1.8)
ax2.text(START * 1.1, 0.9525, "명목 0.95", fontsize=10.5, color="0.35", va="bottom")
ax2.text(START * 1.1, exact - 0.0025, f"실제 포함확률 {exact:.4f}", fontsize=10.5,
         color="tab:red", va="top", fontweight="bold")
ax2.axvline(SHOW, color="0.6", ls=":", lw=1.4)
ax2.text(SHOW * 1.3, 0.898, f"왼쪽 그림이 선 자리\n({SHOW} 번)", fontsize=10,
         color="0.4")
ax2.set_xscale("log")
ax2.set_xlim(START, LONG)
ax2.set_ylim(0.885, 1.005)
ax2.set_xlabel("되풀이한 조사 횟수", fontsize=11)
ax2.set_ylabel("그때까지 참값을 담은 비율", fontsize=11)
ax2.set_title("채점은 되풀이로만 된다", fontsize=12.5)
ax2.grid(alpha=0.25)

fig.suptitle("「95%」는 구간 하나가 아니라 구간을 만드는 절차에 붙는 수다",
             fontsize=14, y=0.995)
plt.tight_layout()
path = OUT + "coverage_sim_poll.png"
fig.savefig(path, dpi=170, facecolor="white", bbox_inches="tight")
print("saved", path)

print(f"\nn = {N}, 참 p = {P}, 왈드 95% 구간")
print(f"  실제 포함확률 (이항으로 정확히) = {exact:.4f}   놓치는 비율 {1 - exact:.4f}")
print(f"  {SHOW} 번 되풀이하면 기대되는 놓침 = {SHOW * (1 - exact):.1f} 개")
print(f"  이 모의의 처음 {SHOW} 번에서 실제로 놓친 것 = {n_miss} 개")
print(f"  {LONG:,} 번까지 가면 담은 비율 = {run[-1]:.4f}")
