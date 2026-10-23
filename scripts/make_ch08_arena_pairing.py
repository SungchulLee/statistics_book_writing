r"""심판마다 자가 다를 때 — 점수를 매기지 않고 둘을 나란히 놓는 까닭.

이 쪽은 대응 설계와 독립 설계 가운데 무엇을 고를지 묻는다. 챗봇 아레나는 사람마다
채점 기준이 다르다는 이유로 **절대 점수를 받지 않고 둘을 견주게** 한다. 그 선택이
구간의 너비에서 얼마만큼인지를 잰다.

모형 둘의 참 실력 차는 delta 로 고정하고, 심판마다 제 나름의 치우침을 얹는다.
- 독립 설계: 심판 하나가 모형 하나만 보고 점수를 매긴다. 심판의 치우침이 그대로 남는다.
- 대응 설계: 심판 하나가 둘을 나란히 보고 견준다. 치우침이 차에서 지워진다.

① 심판 치우침이 커질수록 두 설계의 구간 너비가 갈린다.
② 같은 수의 심판으로 얻는 구간 — 대응 쪽이 몇 배 좁은가.
③ 참 차이를 가려내는 데 드는 심판 수.

만드는 파일: docs/ch08/paired_sample_intervals/img/arena_pairing.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Apple SD Gothic Neo"
plt.rcParams["axes.unicode_minus"] = False

from scipy import stats   # noqa: E402

RNG = np.random.default_rng(0)
ALPHA = 0.05
SIG_E = 1.0          # 같은 심판이 같은 모형을 봐도 남는 흔들림
REP = 40_000


def widths(n, sig_j, delta=0.0):
    """두 설계의 95% 구간 반폭. **읽는 답변 수를 2n 으로 맞춘다.**

    대응   : 심판 n 명이 각자 답변 둘을 읽고 견준다      → 2n 개를 읽는다
    독립   : 심판 2n 명이 각자 답변 하나를 읽고 점수를 준다 → 2n 개를 읽는다
    이렇게 맞추지 않으면 대응 쪽이 사람 품을 두 배 쓰고 이긴 것이 된다.
    """
    # --- 대응: 심판의 치우침 b_j 가 차에서 지워진다. Var(D) = 2 sigma_e^2
    d = RNG.normal(delta, SIG_E * np.sqrt(2), (REP, n))
    hw_p = stats.t.ppf(0.975, n - 1) * d.std(1, ddof=1) / np.sqrt(n)

    # --- 독립: 모형마다 심판 n 명. 각 점수에 치우침이 그대로 남는다.
    sd = np.sqrt(SIG_E ** 2 + sig_j ** 2)
    a = RNG.normal(delta, sd, (REP, n))
    b = RNG.normal(0.0, sd, (REP, n))
    sp2 = ((n - 1) * a.var(1, ddof=1) + (n - 1) * b.var(1, ddof=1)) / (2 * n - 2)
    hw_i = stats.t.ppf(0.975, 2 * n - 2) * np.sqrt(sp2 * (2 / n))
    return hw_p.mean(), hw_i.mean()


# === ① 심판 치우침을 키우며 ==========================================
SIGJ = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
N1 = 200
print(f"① 심판 {N1} 명, 같은 모형에 남는 흔들림 sigma_e = {SIG_E}")
# 한글은 고정폭에서 두 칸을 차지하므로 자리맞춤을 직접 적는다
print("   심판 치우침   대응 반폭   독립 반폭      비")
WP, WI = [], []
for sj in SIGJ:
    p, i = widths(N1, sj)
    WP.append(p)
    WI.append(i)
    print(f"   {sj:>9.1f}   {p:>9.4f}   {i:>9.4f}   {i / p:>6.2f}")
print(f"   심판 치우침이 0 이면 비가 {WI[0] / WP[0]:.2f} 로 거의 같다 "
      f"(대응이 쓰는 자유도가 두 배인 만큼만 이득이다).")
print(f"   치우침이 흔들림의 세 배면 독립 쪽 구간이 {WI[-1] / WP[-1]:.1f} 배 넓다.")

# === ② 같은 심판 수에서 ==============================================
NS = [20, 50, 100, 200, 500, 1000]
SJ2 = 1.5
print(f"\n② 심판 치우침 {SJ2} 일 때 심판 수를 키우며")
print("     심판 수   대응 반폭   독립 반폭      비")
P2, I2 = [], []
for n in NS:
    p, i = widths(n, SJ2)
    P2.append(p)
    I2.append(i)
    print(f"   {n:>9}   {p:>9.4f}   {i:>9.4f}   {i / p:>6.2f}")
print(f"   비가 {np.mean([i / p for p, i in zip(P2, I2)]):.2f} 로 n 과 거의 무관하다 "
      f"— **설계가 주는 이득은 표본을 늘려 따라잡는 것이 아니다.**")

# === ③ 참 차이를 가려내려면 ==========================================
DELTAS = [0.1, 0.2, 0.3]
NS3 = [20, 50, 100, 200, 500, 1000, 2000]
print(f"\n③ 참 차이를 95% 구간이 0 에서 떼어 놓을 확률 (심판 치우침 {SJ2})")
print("     심판 수" + "".join(f"   delta={d}" for d in DELTAS))
POW_P = {d: [] for d in DELTAS}
POW_I = {d: [] for d in DELTAS}
for n in NS3:
    row = []
    for d in DELTAS:
        dd = RNG.normal(d, SIG_E * np.sqrt(2), (REP, n))
        hw = stats.t.ppf(0.975, n - 1) * dd.std(1, ddof=1) / np.sqrt(n)
        POW_P[d].append(float(np.mean(np.abs(dd.mean(1)) > hw)))
        sd = np.sqrt(SIG_E ** 2 + SJ2 ** 2)
        a = RNG.normal(d, sd, (REP, n))
        b = RNG.normal(0.0, sd, (REP, n))
        sp2 = ((n - 1) * a.var(1, ddof=1) + (n - 1) * b.var(1, ddof=1)) / (2 * n - 2)
        hwi = stats.t.ppf(0.975, 2 * n - 2) * np.sqrt(sp2 * (2 / n))
        POW_I[d].append(float(np.mean(np.abs(a.mean(1) - b.mean(1)) > hwi)))
        row.append((POW_P[d][-1], POW_I[d][-1]))
    print(f"   {n:>9}" + "".join(f"   대응 {p:.2f} 독립 {i:.2f}" for p, i in row))

# === 그림 ============================================================
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.6, 5.2))
C_P, C_I = "#1f77b4", "#d62728"

ax1.plot(SIGJ, WP, "o-", color=C_P, lw=2.8, ms=10, label="대응 — 한 심판이 둘을 견준다")
ax1.plot(SIGJ, WI, "s-", color=C_I, lw=2.8, ms=10, label="독립 — 심판마다 하나씩 본다")
ax1.set_xlabel("심판마다 다른 치우침의 표준편차 $\\sigma_j$")
ax1.set_ylabel("$95\\%$ 구간의 반폭")
ax1.set_title(f"① 심판이 제각각일수록 갈린다 (답변 ${2 * N1}$ 개)", fontsize=12)
ax1.legend(loc="upper left", fontsize=10.5)
ax1.grid(alpha=0.25)

ax2.loglog([2 * v for v in NS], P2, "o-", color=C_P, lw=2.8, ms=10, label="대응")
ax2.loglog([2 * v for v in NS], I2, "s-", color=C_I, lw=2.8, ms=10, label="독립")
ax2.set_xlabel("읽는 답변 수 (두 설계가 같다)")
ax2.set_ylabel("$95\\%$ 구간의 반폭")
ax2.set_xticks([40, 100, 1000])
ax2.set_xticklabels(["$40$", "$10^{2}$", "$10^{3}$"])
ax2.set_yticks([0.1, 0.3, 1.0])
ax2.set_yticklabels(["$0.1$", "$0.3$", "$1$"])
ax2.minorticks_off()
ax2.set_title(f"② 간격이 줄지 않는다 (치우침 ${SJ2}$)", fontsize=12)
ax2.legend(loc="upper right", fontsize=10.5)
ax2.grid(alpha=0.25)

for i, d in enumerate(DELTAS):
    c = ["#9ecae1", "#4292c6", "#08519c"][i]
    ax3.semilogx([2 * v for v in NS3], POW_P[d], "o-", color=c, lw=2.6, ms=9,
                 label=f"대응, 참 차이 ${d}$")
    ax3.semilogx([2 * v for v in NS3], POW_I[d], "s--", color=c, lw=2.0, ms=7, alpha=0.75)
ax3.axhline(0.8, color=C_I, lw=1.8, ls="--")
ax3.text(42, 0.83, "검정력 $0.8$", fontsize=10.5, color=C_I)
ax3.set_xlabel("읽는 답변 수 (두 설계가 같다)")
ax3.set_ylabel("구간이 $0$ 에서 떨어질 확률")
ax3.set_xticks([40, 100, 1000])
ax3.set_xticklabels(["$40$", "$10^{2}$", "$10^{3}$"])
ax3.minorticks_off()
ax3.set_ylim(0, 1.03)
ax3.set_title("③ 실선이 대응, 점선이 독립", fontsize=12)
ax3.legend(loc="lower right", fontsize=9.5)
ax3.grid(alpha=0.25)

fig.tight_layout()
out = "docs/ch08/paired_sample_intervals/img/arena_pairing.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("\nsaved", out)
